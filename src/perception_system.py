import numpy as np
from interfaces import BasePerceptionSystem
from core_types import PerceptionInput

class OdometrySystem:
    def __init__(self, noise_config):
        self.noise_config = noise_config
        self.believed_state = None
        self.last_gt_pos = None
        self.last_gt_yaw = None
        self.last_gt_pitch = None

    def estimate(self, gt_state, dt: float):
        import copy, random
        
        if self.believed_state is None:
            self.believed_state = copy.deepcopy(gt_state)
            self.last_gt_pos = gt_state.position.copy()
            self.last_gt_yaw = gt_state.yaw
            self.last_gt_pitch = gt_state.pitch
            return self.believed_state
            
        dx = gt_state.position[0] - self.last_gt_pos[0]
        dy = gt_state.position[1] - self.last_gt_pos[1]
        dz = gt_state.position[2] - self.last_gt_pos[2]
        
        dyaw = gt_state.yaw - self.last_gt_yaw
        dyaw = (dyaw + 180) % 360 - 180
        dpitch = gt_state.pitch - self.last_gt_pitch
        
        drift_xy = self.noise_config.get('drift_rate_xy', 0.05)
        drift_z = self.noise_config.get('drift_rate_z', 0.01)
        imu_noise = self.noise_config.get('imu_noise_std_dev', 0.02)
        
        is_moving = abs(dx) > 0.0001 or abs(dy) > 0.0001 or abs(dyaw) > 0.0001 or abs(dpitch) > 0.0001
        
        if is_moving:
            self.believed_state.position[0] += dx + random.gauss(0, drift_xy * dt)
            self.believed_state.position[1] += dy + random.gauss(0, drift_xy * dt)
            self.believed_state.position[2] += dz + random.gauss(0, drift_z * dt)
            self.believed_state.yaw += dyaw + random.gauss(0, imu_noise * 10 * dt)
            self.believed_state.pitch += dpitch + random.gauss(0, imu_noise * 10 * dt)
        
        self.last_gt_pos = gt_state.position.copy()
        self.last_gt_yaw = gt_state.yaw
        self.last_gt_pitch = gt_state.pitch
        
        return self.believed_state

class FrustumPerceptionSystem(BasePerceptionSystem):
    def __init__(self, sensor_config):
        fov_h_rad = np.radians(sensor_config.get("fov_horizontal", 90.0))
        fov_v_rad = np.radians(sensor_config.get("fov_vertical", 60.0))
        
        self.tan_half_fov_h = np.tan(fov_h_rad / 2.0)
        self.tan_half_fov_v = np.tan(fov_v_rad / 2.0)
        self.max_range = sensor_config.get("max_range", 0.0)

    def _is_aabb_in_frustum(self, vertices: list, agent_pos: np.ndarray, 
                            agent_fwd: np.ndarray, agent_right: np.ndarray, agent_up: np.ndarray) -> bool:
        local_vertices = []
        for v in vertices:
            vec = v - agent_pos
            y = np.dot(vec, agent_fwd)
            x = np.dot(vec, agent_right)
            z = np.dot(vec, agent_up)
            local_vertices.append((x, y, z))

        if all(y <= 0 for x, y, z in local_vertices): return False
        if self.max_range > 0 and all(y > self.max_range for x, y, z in local_vertices): return False
        if all(x > y * self.tan_half_fov_h for x, y, z in local_vertices): return False
        if all(x < -y * self.tan_half_fov_h for x, y, z in local_vertices): return False
        if all(z > y * self.tan_half_fov_v for x, y, z in local_vertices): return False
        if all(z < -y * self.tan_half_fov_v for x, y, z in local_vertices): return False

        return True

    def scan_environment(self, inputs: PerceptionInput) -> dict:
        collision_queue = inputs.get("collision_queue")
        agent = inputs.get("agent_state")
        world_objects = inputs.get("world_objects")

        nearby_objects = set()
        for i in range(collision_queue.getNumEntries()):
            hit_node = collision_queue.getEntry(i).getIntoNodePath()
            if hit_node.hasTag("obj_id"):
                nearby_objects.add(hit_node.getTag("obj_id"))

        visible_objects = {}
        if not nearby_objects:
            return visible_objects
            
        agent_pos = agent.position + np.array([0, 0, 2])
        
        pitch_rad = np.radians(agent.pitch)
        yaw_rad = np.radians(agent.yaw)
        
        fwd_x = -np.sin(yaw_rad) * np.cos(pitch_rad)
        fwd_y = np.cos(yaw_rad) * np.cos(pitch_rad)
        fwd_z = np.sin(pitch_rad)
        
        agent_fwd = np.array([fwd_x, fwd_y, fwd_z])
        agent_fwd = agent_fwd / np.linalg.norm(agent_fwd)
        
        global_up = np.array([0.0, 0.0, 1.0])
        agent_right = np.cross(agent_fwd, global_up)
        
        if np.linalg.norm(agent_right) > 0.001:
            agent_right = agent_right / np.linalg.norm(agent_right)
        else:
            agent_right = np.array([1.0, 0.0, 0.0]) 
            
        agent_up = np.cross(agent_right, agent_fwd)
        agent_up = agent_up / np.linalg.norm(agent_up)

        for obj_id in nearby_objects:
            obj_data = world_objects[obj_id]
            vertices = obj_data["vertices"]
            if self._is_aabb_in_frustum(vertices, agent_pos, agent_fwd, agent_right, agent_up):
                visible_objects[obj_id] = {
                    "center": obj_data["center"],
                    "extensions": obj_data["extensions"],
                    "confidence": 1.0
                }

        return visible_objects

class NoisyPerceptionSystem(FrustumPerceptionSystem):
    def __init__(self, sensor_config):
        super().__init__(sensor_config)
        self.pos_std = sensor_config.get("noise_pos_std_dev", 0.05)
        self.ext_std = sensor_config.get("noise_ext_std_dev", 0.02)
        self.min_conf = sensor_config.get("min_confidence_threshold", 0.40)
        self.base_pos_std = sensor_config.get("noise_pos_base_std", 0.005)
        self.max_pos_std = sensor_config.get("noise_pos_max_std", 0.30)

        update_hz = sensor_config.get("update_frequency_hz", 10.0)
        self.update_interval = 1.0 / update_hz
        self.last_update_time = -self.update_interval 
        self.cached_perception = {}

    def scan_environment(self, inputs):
        current_time = inputs.get("timestamp")
        if current_time - self.last_update_time < self.update_interval:
            return self.cached_perception

        self.last_update_time = current_time
        visible_dict = super().scan_environment(inputs)
        
        perceived_data = {}
        
        for obj_id, gt_data in visible_dict.items():
            confidence = 0.90 + np.random.normal(0, 0.05)
            
            if confidence < self.min_conf:
                continue

            agent_pos = inputs.get("agent_state").position
            dist = np.linalg.norm(np.array(gt_data["center"]) - agent_pos)
                        
            dynamic_pos_std = min(self.max_pos_std, self.base_pos_std * (dist ** 2))
            noisy_center = np.array(gt_data["center"]) + np.random.normal(0, dynamic_pos_std, 3)
            noisy_ext = np.array(gt_data["extensions"]) + np.random.normal(0, self.ext_std, 3)
            
            perceived_data[obj_id] = {
                "center": noisy_center.tolist(),
                "extensions": noisy_ext.tolist(),
                "confidence": min(1.0, confidence)
            }
            
        self.cached_perception = perceived_data
        return perceived_data