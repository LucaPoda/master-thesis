import numpy as np
from typing import List
from interfaces import BaseSpatialReasoner
from core_types import PerceptionInput, SemanticState, SpatialRelation

class GroundTruthSpatialReasoner(BaseSpatialReasoner):
    def compute_relations(self, visible_objects: List[str], inputs: PerceptionInput) -> SemanticState:
        # Fails gracefully if the required "topic" is missing
        world_objects = inputs.get("world_objects") 
        relations = set()
        
        # O(N^2) naive bounding-box distance check
        for i in range(len(visible_objects)):
            for j in range(i + 1, len(visible_objects)):
                obj_a = visible_objects[i]
                obj_b = visible_objects[j]
                
                pos_a = np.array(world_objects[obj_a]["center"])
                pos_b = np.array(world_objects[obj_b]["center"])
                
                dist = np.linalg.norm(pos_a - pos_b)
                if dist < 4.0:  # Proximity threshold
                    relations.add(SpatialRelation(subject=obj_a, relation="near", target=obj_b))
                    relations.add(SpatialRelation(subject=obj_b, relation="near", target=obj_a))
                    
        return SemanticState(visible_objects=set(visible_objects), relations=relations)

class CVSpatialReasoner(BaseSpatialReasoner):
    def compute_relations(self, visible_objects: List[str], inputs: PerceptionInput) -> SemanticState:
        # TODO: Implement vision-based spatial inferencing (e.g., bounding boxes, depth maps).
        # Expected inputs: inputs.get("rgb_frame"), inputs.get("depth_map")
        return SemanticState(visible_objects=set(visible_objects), relations=set())