import numpy as np
from interfaces import BasePerceptionSystem
from core_types import PerceptionInput

class FrustumPerceptionSystem(BasePerceptionSystem):
    def __init__(self, sensor_config):
        fov_h_rad = np.radians(sensor_config.get("fov_horizontal", 90.0))
        fov_v_rad = np.radians(sensor_config.get("fov_vertical", 60.0))
        
        self.tan_half_fov_h = np.tan(fov_h_rad / 2.0)
        self.tan_half_fov_v = np.tan(fov_v_rad / 2.0)
        
        self.max_range = sensor_config.get("max_range", 0.0)

    def _is_aabb_in_frustum(self, vertices: list, agent_pos: np.ndarray, 
                            agent_fwd: np.ndarray, agent_right: np.ndarray, agent_up: np.ndarray) -> bool:
        """
        Runs a robust Frustum Culling by evaluating the object against the planes of the frustum.
        If all 8 vertices of the AABB are on the "wrong" side of a single plane,
        the object is mathematically outside the field of view.
        """
        # Convert all vertices in the local space of the camera
        local_vertices = []
        for v in vertices:
            vec = v - agent_pos
            y = np.dot(vec, agent_fwd)   # Depth (Front/Back)
            x = np.dot(vec, agent_right) # Horizontal Axis (Right/Left)
            z = np.dot(vec, agent_up)    # Vertical Axis (Up/Down)
            local_vertices.append((x, y, z))

        # Test 1: Near Plane (All vertices are behind the camera?)
        if all(y <= 0 for x, y, z in local_vertices): return False
        
        # Test 2: Far Plane (All vertices are beyond the maximum range?)
        if self.max_range > 0 and all(y > self.max_range for x, y, z in local_vertices): return False

        # Test 3: Right Plane (All vertices are beyond the right edge?)
        if all(x > y * self.tan_half_fov_h for x, y, z in local_vertices): return False
        
        # Test 4: Left Plane (All vertices are beyond the left edge?)
        if all(x < -y * self.tan_half_fov_h for x, y, z in local_vertices): return False

        # Test 5: Top Plane (All vertices are beyond the upper limit?)
        if all(z > y * self.tan_half_fov_v for x, y, z in local_vertices): return False
        
        # Test 6: Bottom Plane (All vertices are below the lower limit?)
        if all(z < -y * self.tan_half_fov_v for x, y, z in local_vertices): return False

        # If the object has not been discarded by any of the planes, then it crosses the frustum!
        return True

    def scan_environment(self, inputs: PerceptionInput) -> list[str]:
        # Read from generic payload
        collision_queue = inputs.get("collision_queue")
        agent = inputs.get("agent_state")
        world_objects = inputs.get("world_objects")

        # 1. BROAD-PHASE (Physics)[cite: 7]
        nearby_objects = set()
        for i in range(collision_queue.getNumEntries()):
            hit_node = collision_queue.getEntry(i).getIntoNodePath()
            if hit_node.hasTag("obj_id"):
                nearby_objects.add(hit_node.getTag("obj_id"))

        # 2. NARROW-PHASE (Mathematics)
        visible_objects = []
        if not nearby_objects:
            return visible_objects
            
        agent_pos = agent.position + np.array([0, 0, 2])
        
        # Compute the forward vector based on yaw and pitch, including the camera's pitch for vertical orientation.
        pitch_rad = np.radians(agent.pitch)
        yaw_rad = np.radians(agent.yaw)
        
        fwd_x = -np.sin(yaw_rad) * np.cos(pitch_rad)
        fwd_y = np.cos(yaw_rad) * np.cos(pitch_rad)
        fwd_z = np.sin(pitch_rad)
        
        agent_fwd = np.array([fwd_x, fwd_y, fwd_z])
        agent_fwd = agent_fwd / np.linalg.norm(agent_fwd)
        
        # Restore Right and Up axes based on the new forward vector
        global_up = np.array([0.0, 0.0, 1.0])
        agent_right = np.cross(agent_fwd, global_up)
        
        # Mathematical protection in the case we are looking perfectly up/down
        if np.linalg.norm(agent_right) > 0.001:
            agent_right = agent_right / np.linalg.norm(agent_right)
        else:
            agent_right = np.array([1.0, 0.0, 0.0]) 
            
        agent_up = np.cross(agent_right, agent_fwd)
        agent_up = agent_up / np.linalg.norm(agent_up)

        # Object scanning: For each nearby object, check if its AABB is within the frustum defined by the agent's position and orientation.
        for obj_id in nearby_objects:
            obj_data = world_objects[obj_id]
            vertices = obj_data["vertices"]
            
            if self._is_aabb_in_frustum(vertices, agent_pos, agent_fwd, agent_right, agent_up):
                visible_objects.append(obj_id)

        return visible_objects