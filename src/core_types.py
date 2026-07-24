import numpy as np
from dataclasses import dataclass, field
from typing import Any, Set

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

@dataclass(frozen=True)
class SpatialRelation:
    subject: str
    relation: str
    target: str

@dataclass
class SemanticState:
    visible_objects: Set[str] = field(default_factory=set)
    relations: Set[SpatialRelation] = field(default_factory=set)

    def to_frozen_key(self) -> frozenset:
        """Combines objects and relations into a single frozenset for hashing."""
        return frozenset(list(self.visible_objects) + list(self.relations))

class PerceptionInput:
    """
    This payload acts as an untyped/flexible subscriber buffer (similar to ROS 2 topics).
    Each concrete perception or reasoning component attempts to access its expected fields dynamically.
    If an expected field is None or missing, it raises an error as if a required ROS topic was unpopulated.
    """
    def __init__(self, **kwargs):
        self._data = kwargs

    def get(self, key: str) -> Any:
        if key not in self._data or self._data[key] is None:
            raise ValueError(f"Missing required input field/topic: '{key}'")
        return self._data[key]