import numpy as np
from core_types import AgentState

class PerceptionSystem:
    def __init__(self, sensor_config):
        fov_h_rad = np.radians(sensor_config.get("fov_horizontal", 90.0))
        fov_v_rad = np.radians(sensor_config.get("fov_vertical", 60.0))
        
        self.tan_half_fov_h = np.tan(fov_h_rad / 2.0)
        self.tan_half_fov_v = np.tan(fov_v_rad / 2.0)
        
        # Read the value of max_range (default to 0.0 if omitted)
        self.max_range = sensor_config.get("max_range", 0.0)

    def _is_point_in_frustum(self, point: np.ndarray, agent_pos: np.ndarray, 
                             agent_fwd: np.ndarray, agent_right: np.ndarray, agent_up: np.ndarray) -> bool:
        vec_to_vertex = point - agent_pos
        y_local = np.dot(vec_to_vertex, agent_fwd)
        
        # The object must always be found rigorously in front
        if y_local <= 0: 
            return False
            
        # If max_range is set (> 0), we discard what is too far away
        if self.max_range > 0 and y_local > self.max_range:
            return False
            
        x_local = np.dot(vec_to_vertex, agent_right)
        z_local = np.dot(vec_to_vertex, agent_up)
        
        if abs(x_local / y_local) > self.tan_half_fov_h: return False
        if abs(z_local / y_local) > self.tan_half_fov_v: return False
        return True

    def scan_environment(self, collision_queue, agent: AgentState, world_objects: dict) -> list:
        # 1. BROAD-PHASE (Physical collision detection using Panda3D)
        nearby_objects = set()
        for i in range(collision_queue.getNumEntries()):
            hit_node = collision_queue.getEntry(i).getIntoNodePath()
            if hit_node.hasTag("obj_id"):
                nearby_objects.add(hit_node.getTag("obj_id"))

        # 2. NARROW-PHASE (Math)
        visible_objects = []
        if not nearby_objects:
            return visible_objects
            
        agent_pos = agent.position + np.array([0, 0, 2])
        agent_fwd = agent.get_forward_vector()
        
        global_up = np.array([0.0, 0.0, 1.0])
        agent_right = np.cross(agent_fwd, global_up)
        if np.linalg.norm(agent_right) > 0:
            agent_right = agent_right / np.linalg.norm(agent_right)
        agent_up = np.cross(agent_right, agent_fwd)

        for obj_id in nearby_objects:
            obj_data = world_objects[obj_id]
            # Directly use the vertices calculated at the construction of the WorldState
            vertices = obj_data["vertices"]
            
            for vertex in vertices:
                if self._is_point_in_frustum(vertex, agent_pos, agent_fwd, agent_right, agent_up):
                    visible_objects.append(obj_id)
                    break

        return visible_objects