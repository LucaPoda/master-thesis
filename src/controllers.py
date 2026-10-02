import json
import os
import numpy as np
from datetime import datetime
import math

from interfaces import BaseController
from core_types import MovementCommand, PerceptionInput

class RecordingKeyboardController(BaseController):
    def __init__(self, base_controller, log_dir):
        self.base_controller = base_controller
        self.log_file = os.path.join(log_dir, f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.history = []

    def get_command(self, inputs: PerceptionInput, dt: float, **kwargs) -> MovementCommand:
        cmd = self.base_controller.get_command(inputs, dt, **kwargs)
        
        cmd_dict = {
            "velocity": cmd.velocity.tolist(),
            "yaw_delta": cmd.yaw_delta,
            "pitch_delta": cmd.pitch_delta
        }
        
        self.history.append({
            "timestamp": inputs.get("timestamp"), 
            "cmd": cmd_dict, 
            "state": inputs.get("agent_state").to_dict()
        })
        return cmd

    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f)

class ReplayController(BaseController):
    def __init__(self, log_file):
        with open(log_file, 'r') as f:
            self.history = json.load(f)
        self.frame = 0

    def get_command(self, inputs: PerceptionInput, dt: float, **kwargs) -> MovementCommand:
        if self.frame >= len(self.history):
            return MovementCommand(np.zeros(3), 0.0, 0.0)
            
        cmd_data = self.history[self.frame]["cmd"]
        cmd = MovementCommand(
            velocity=np.array(cmd_data["velocity"]),
            yaw_delta=cmd_data["yaw_delta"],
            pitch_delta=cmd_data["pitch_delta"]
        )
        self.frame += 1
        return cmd

class KeyboardController(BaseController):
    def __init__(self, config):
        self.speed = config["max_speed"]
        self.mouse_sens = config["mouse_sensitivity"]

    def get_command(self, inputs: PerceptionInput, dt: float, **kwargs) -> MovementCommand:
        agent_state = inputs.get("agent_state")
        raw_inputs = inputs.get("raw_inputs")  
        
        keys = raw_inputs.get("keys", {})
        yaw_delta = -raw_inputs.get("mouse_dx", 0) * self.mouse_sens
        pitch_delta = -raw_inputs.get("mouse_dy", 0) * self.mouse_sens

        fwd = agent_state.get_forward_vector()
        right = np.array([fwd[1], -fwd[0], 0.0])
        up = np.array([0.0, 0.0, 1.0])

        vel = np.zeros(3)
        if keys.get("w"): vel += fwd
        if keys.get("s"): vel -= fwd
        if keys.get("a"): vel -= right
        if keys.get("d"): vel += right
        if keys.get("space"): vel += up
        if keys.get("lshift"): vel -= up

        if np.linalg.norm(vel) > 0:
            vel = (vel / np.linalg.norm(vel)) * self.speed

        return MovementCommand(velocity=vel, yaw_delta=yaw_delta, pitch_delta=pitch_delta)

class PositionController(BaseController):
    def __init__(self, config):
        self.speed = config["max_speed"]
        self.tolerance = config["position_tolerance"]
        self.target_position = None

    def set_target(self, pos: np.ndarray):
        self.target_position = pos

    def get_command(self, inputs: PerceptionInput, dt: float, **kwargs) -> MovementCommand:
        agent_state = inputs.get("agent_state")
        
        if self.target_position is None:
            return MovementCommand(np.zeros(3), 0.0, 0.0)

        vec_to_target = self.target_position - agent_state.position
        distance = np.linalg.norm(vec_to_target)

        if distance < self.tolerance:
            self.target_position = None
            return MovementCommand(np.zeros(3), 0.0, 0.0)

        vel = (vec_to_target / distance) * self.speed
        return MovementCommand(velocity=vel, yaw_delta=0.0, pitch_delta=0.0)

class TrajectoryController(BaseController):
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.current_wp_idx = 0
        self.move_speed = 2.0  
        self.turn_speed = 90.0

    def get_command(self, inputs: PerceptionInput, dt: float, **kwargs) -> MovementCommand:
        gt_state = inputs.get("agent_state")

        if self.current_wp_idx >= len(self.waypoints):
            return MovementCommand(velocity=np.zeros(3), yaw_delta=0.0, pitch_delta=0.0)

        target = self.waypoints[self.current_wp_idx]
        
        dx = target['x'] - gt_state.position[0]
        dy = target['y'] - gt_state.position[1]
        dz = target['z'] - gt_state.position[2]
        
        dyaw = target['yaw'] - gt_state.yaw
        dyaw = (dyaw + 180) % 360 - 180
        dpitch = target['pitch'] - gt_state.pitch
        
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        
        cmd_vel = np.zeros(3)
        cmd_yaw = 0.0
        cmd_pitch = 0.0
        
        if dist > 0.1:
            cmd_vel[0] = (dx / dist) * self.move_speed
            cmd_vel[1] = (dy / dist) * self.move_speed
            cmd_vel[2] = (dz / dist) * self.move_speed
        
        if abs(dyaw) > 1.0:
            cmd_yaw = max(min(dyaw, self.turn_speed * dt), -self.turn_speed * dt)
            
        if abs(dpitch) > 1.0:
            cmd_pitch = max(min(dpitch, self.turn_speed * dt), -self.turn_speed * dt)
            
        if dist <= 0.1 and abs(dyaw) <= 1.0 and abs(dpitch) <= 1.0:
            self.current_wp_idx += 1
            
        return MovementCommand(velocity=cmd_vel, yaw_delta=cmd_yaw, pitch_delta=cmd_pitch)