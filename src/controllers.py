import numpy as np
from interfaces import BaseController
from core_types import MovementCommand, PerceptionInput

class KeyboardController(BaseController):
    def __init__(self, config):
        self.speed = config["max_speed"]
        self.mouse_sens = config["mouse_sensitivity"]

    def get_command(self, inputs: PerceptionInput, dt: float, **kwargs) -> MovementCommand:
        # Extract the required components from the unconstrained ROS-like buffer
        agent_state = inputs.get("agent_state")
        raw_inputs = inputs.get("raw_inputs")  
        
        # raw_inputs is a standard dictionary coming from GraphicEngine.poll_inputs()
        keys = raw_inputs.get("keys", {})
        
        # Apply dictionary get() with default values on raw_inputs, not on PerceptionInput
        yaw_delta = -raw_inputs.get("mouse_dx", 0) * self.mouse_sens
        pitch_delta = -raw_inputs.get("mouse_dy", 0) * self.mouse_sens

        fwd = agent_state.get_forward_vector()
        right = np.array([fwd[1], -fwd[0], 0.0]) # Rotated -90 degrees on Z
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
            self.target_position = None # Raggiunto
            return MovementCommand(np.zeros(3), 0.0, 0.0)

        vel = (vec_to_target / distance) * self.speed
        return MovementCommand(velocity=vel, yaw_delta=0.0, pitch_delta=0.0)