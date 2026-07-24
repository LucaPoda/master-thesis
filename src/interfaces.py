from abc import ABC, abstractmethod
from typing import List, Any
from core_types import PerceptionInput, SemanticState, MovementCommand

class BasePerceptionSystem(ABC):
    @abstractmethod
    def scan_environment(self, inputs: PerceptionInput) -> List[str]:
        pass

class BaseSpatialReasoner(ABC):
    @abstractmethod
    def compute_relations(self, visible_objects: List[str], inputs: PerceptionInput) -> SemanticState:
        pass

class BaseGraphVisualizer(ABC):
    @abstractmethod
    def start(self) -> None:
        pass
        
    @abstractmethod
    def publish_update(self, payload: str) -> None:
        pass

class BaseController(ABC):
    @abstractmethod
    def get_command(self, inputs: PerceptionInput, dt: float, **kwargs) -> MovementCommand:
        pass