from dataclasses import dataclass
import numpy as np

@dataclass
class AgentState:
    position: np.ndarray  # [x, y, z]
    yaw: float            # Rotation around Z axis (in degrees)
    pitch: float          # Rotation around Y axis (in degrees, for the camera)

    def get_forward_vector(self) -> np.ndarray:
        # Calculate the direction vector based on yaw
        rad = np.radians(self.yaw)
        # In Panda3D standard, forward is on the Y axis.
        return np.array([-np.sin(rad), np.cos(rad), 0.0])

@dataclass
class MovementCommand:
    velocity: np.ndarray  # [vx, vy, vz]
    yaw_delta: float      # Variation in rotation
    pitch_delta: float    # Variation in visual height