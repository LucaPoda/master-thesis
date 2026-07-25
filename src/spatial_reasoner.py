import numpy as np
from typing import List
from interfaces import BaseSpatialReasoner
from core_types import PerceptionInput, SemanticState, SpatialRelation, AgentState

class GroundTruthSpatialReasoner(BaseSpatialReasoner):
    def compute_relations(self, visible_objects: List[str], inputs: PerceptionInput) -> SemanticState:
        world_objects = inputs.get("world_objects") 
        agent_state = inputs.get("agent_state")  # Extract Agent State
        
        relations = set()
        
        # Evaluate pairwise relations bi-directionally
        for i in range(len(visible_objects)):
            for j in range(i + 1, len(visible_objects)):
                obj_a = visible_objects[i]
                obj_b = visible_objects[j]
                
                self._extract_relations(obj_a, obj_b, world_objects[obj_a], world_objects[obj_b], agent_state, relations)
                self._extract_relations(obj_b, obj_a, world_objects[obj_b], world_objects[obj_a], agent_state, relations)
                    
        return SemanticState(visible_objects=set(visible_objects), relations=relations)

    def _extract_relations(self, a_id: str, b_id: str, data_a: dict, data_b: dict, agent: AgentState, relations: set):
        c_a = np.array(data_a["center"], dtype=float)
        e_a = np.array(data_a["extensions"], dtype=float)
        
        c_b = np.array(data_b["center"], dtype=float)
        e_b = np.array(data_b["extensions"], dtype=float)
        
        a_min, a_max = c_a - e_a, c_a + e_a
        b_min, b_max = c_b - e_b, c_b + e_b
        
        # 1. Proximity: Near (Distance is invariant, uses world coordinates)
        dist = np.linalg.norm(c_a - c_b)
        is_near = dist < 8.0
        
        if is_near:
            relations.add(SpatialRelation(a_id, "near", b_id))
            
        # 2. Vertical Relations (Z is invariant, overlap uses world X/Y)
        overlap_x = bool((a_min[0] < b_max[0]) and (a_max[0] > b_min[0]))
        overlap_y = bool((a_min[1] < b_max[1]) and (a_max[1] > b_min[1]))
        
        z_tolerance = 0.1
        if overlap_x and overlap_y:
            if abs(a_min[2] - b_max[2]) <= z_tolerance:
                relations.add(SpatialRelation(a_id, "on-top", b_id))
            elif a_min[2] > b_max[2] + z_tolerance:
                relations.add(SpatialRelation(a_id, "over", b_id))
                
            if abs(a_max[2] - b_min[2]) <= z_tolerance:
                relations.add(SpatialRelation(a_id, "on-bottom", b_id))
            elif a_max[2] < b_min[2] - z_tolerance:
                relations.add(SpatialRelation(a_id, "under", b_id))
                
        # 3. Horizontal Relations (Egocentric Frame Transformation)
        if is_near:
            rad = np.radians(agent.yaw)
            fwd = np.array([-np.sin(rad), np.cos(rad), 0.0])
            right = np.array([np.cos(rad), np.sin(rad), 0.0])
            
            # Vectors from agent to objects
            v_a = c_a - agent.position
            v_b = c_b - agent.position
            
            # Project onto local axes
            local_x_a = np.dot(v_a, right)
            local_y_a = np.dot(v_a, fwd)
            
            local_x_b = np.dot(v_b, right)
            local_y_b = np.dot(v_b, fwd)
            
            # Evaluate Local Left/Right
            if local_x_a < local_x_b:
                relations.add(SpatialRelation(a_id, "left", b_id))
            elif local_x_a > local_x_b:
                relations.add(SpatialRelation(a_id, "right", b_id))
                
            # Evaluate Local Front/Back (Smaller Y means it is closer to the agent)
            if local_y_a < local_y_b:
                relations.add(SpatialRelation(a_id, "front", b_id))
            elif local_y_a > local_y_b:
                relations.add(SpatialRelation(a_id, "back", b_id))

class CVSpatialReasoner(BaseSpatialReasoner):
    def compute_relations(self, visible_objects: List[str], inputs: PerceptionInput) -> SemanticState:
        return SemanticState(visible_objects=set(visible_objects), relations=set())