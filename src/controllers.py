import numpy as np
from core_types import AgentState, MovementCommand

class BaseController:
    def get_command(self, agent_state: AgentState, dt: float, **kwargs) -> MovementCommand:
        raise NotImplementedError

class KeyboardController(BaseController):
    def __init__(self, config):
        self.speed = config["max_speed"]
        self.mouse_sens = config["mouse_sensitivity"]

    def get_command(self, agent_state: AgentState, dt: float, inputs=None) -> MovementCommand:
        """
        Inputs is a dictionary coming from the graphics engine (e.g., keys pressed and mouse delta).
        To decouple, the controller does not read directly from Panda3D, but from the raw data passed in the loop.
        """
        inputs = inputs or {"keys": {}, "mouse_dx": 0, "mouse_dy": 0}
        keys = inputs.get("keys", {})
        
        yaw_delta = -inputs.get("mouse_dx", 0) * self.mouse_sens
        pitch_delta = -inputs.get("mouse_dy", 0) * self.mouse_sens

        fwd = agent_state.get_forward_vector()
        right = np.array([fwd[1], -fwd[0], 0.0]) # Ruotato di -90 gradi su Z
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

    def get_command(self, agent_state: AgentState, dt: float, **kwargs) -> MovementCommand:
        if self.target_position is None:
            return MovementCommand(np.zeros(3), 0.0, 0.0)

        vec_to_target = self.target_position - agent_state.position
        distance = np.linalg.norm(vec_to_target)

        if distance < self.tolerance:
            self.target_position = None # Raggiunto
            return MovementCommand(np.zeros(3), 0.0, 0.0)

        vel = (vec_to_target / distance) * self.speed
        return MovementCommand(velocity=vel, yaw_delta=0.0, pitch_delta=0.0)