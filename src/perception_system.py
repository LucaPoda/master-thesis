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

class NoisyPerceptionSystem(FrustumPerceptionSystem):
    def __init__(self, sensor_config):
        super().__init__(sensor_config)
        
        # Load probabilistic settings directly from the configuration file[cite: 10]
        self.pos_std = sensor_config.get("noise_pos_std_dev", 0.05)
        self.ext_std = sensor_config.get("noise_ext_std_dev", 0.02)
        self.min_conf = sensor_config.get("min_confidence_threshold", 0.40)
        self.base_pos_std = sensor_config.get("noise_pos_base_std", 0.005)
        self.max_pos_std = sensor_config.get("noise_pos_max_std", 0.30)

        # Set the sensor's update frequency[cite: 10]
        update_hz = sensor_config.get("update_frequency_hz", 10.0)
        self.update_interval = 1.0 / update_hz
        self.last_update_time = -self.update_interval 
        self.cached_perception = {}

    def scan_environment(self, inputs):
        # Use cache to respect the sensor update frequency[cite: 9, 10]
        current_time = inputs.get("timestamp")
        if current_time - self.last_update_time < self.update_interval:
            return self.cached_perception

        self.last_update_time = current_time
        visible_ids = super().scan_environment(inputs)
        world_objects = inputs.get("world_objects")
        
        perceived_data = {}
        
        for obj_id in visible_ids:
            gt_data = world_objects[obj_id]

            # Compute confidence score (using a fixed 0.05 deviation since it is absent from the config)[cite: 9, 10]
            confidence = 0.90 + np.random.normal(0, 0.05)
            
            # Handle false negatives by discarding objects below the confidence threshold[cite: 9, 10]
            if confidence < self.min_conf:
                continue

            # Calculate true distance between agent and object[cite: 9]
            agent_pos = inputs.get("agent_state").position
            dist = np.linalg.norm(np.array(gt_data["center"]) - agent_pos)
                        
            # Apply quadratic model for position noise growth, capped at max_std[cite: 9, 10]
            dynamic_pos_std = min(self.max_pos_std, self.base_pos_std * (dist ** 2))
            
            # Apply random effects simulating measurement fluctuations (zero-mean Gaussian noise)[cite: 6, 9]
            noisy_center = np.array(gt_data["center"]) + np.random.normal(0, dynamic_pos_std, 3)
            noisy_ext = np.array(gt_data["extensions"]) + np.random.normal(0, self.ext_std, 3)
            
            perceived_data[obj_id] = {
                "center": noisy_center.tolist(),
                "extensions": noisy_ext.tolist(),
                "confidence": min(1.0, confidence)
            }
            
        self.cached_perception = perceived_data
        return perceived_data