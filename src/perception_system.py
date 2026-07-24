# perception_system.py
import numpy as np
from core_types import AgentState

class PerceptionSystem:
    def __init__(self, sensor_config):
        fov_h_rad = np.radians(sensor_config.get("fov_horizontal", 90.0))
        fov_v_rad = np.radians(sensor_config.get("fov_vertical", 60.0))
        
        self.tan_half_fov_h = np.tan(fov_h_rad / 2.0)
        self.tan_half_fov_v = np.tan(fov_v_rad / 2.0)
        
        self.max_range = sensor_config.get("max_range", 0.0)

    def _is_aabb_in_frustum(self, vertices: list, agent_pos: np.ndarray, 
                            agent_fwd: np.ndarray, agent_right: np.ndarray, agent_up: np.ndarray) -> bool:
        """
        Esegue un Frustum Culling robusto valutando l'oggetto contro i piani della piramide.
        Se tutti gli 8 vertici dell'AABB sono sul lato "sbagliato" di un singolo piano,
        l'oggetto è matematicamente fuori dal campo visivo.
        """
        # Convertiamo tutti i vertici nello spazio locale della telecamera
        local_vertices = []
        for v in vertices:
            vec = v - agent_pos
            y = np.dot(vec, agent_fwd)   # Profondità (Avanti)
            x = np.dot(vec, agent_right) # Asse Orizzontale (Destra/Sinistra)
            z = np.dot(vec, agent_up)    # Asse Verticale (Alto/Basso)
            local_vertices.append((x, y, z))

        # Test 1: Piano Near (Tutti i vertici sono dietro la telecamera?)
        if all(y <= 0 for x, y, z in local_vertices): return False
        
        # Test 2: Piano Far (Tutti i vertici superano la distanza massima?)
        if self.max_range > 0 and all(y > self.max_range for x, y, z in local_vertices): return False

        # Test 3: Piano Destro (Tutti i vertici sono oltre il bordo destro?)
        if all(x > y * self.tan_half_fov_h for x, y, z in local_vertices): return False
        
        # Test 4: Piano Sinistro (Tutti i vertici sono oltre il bordo sinistro?)
        if all(x < -y * self.tan_half_fov_h for x, y, z in local_vertices): return False

        # Test 5: Piano Superiore (Tutti i vertici sono oltre il limite alto?)
        if all(z > y * self.tan_half_fov_v for x, y, z in local_vertices): return False
        
        # Test 6: Piano Inferiore (Tutti i vertici sono sotto il limite basso?)
        if all(z < -y * self.tan_half_fov_v for x, y, z in local_vertices): return False

        # Se l'oggetto non è stato scartato da nessuno dei piani, allora attraversa l'inquadratura!
        return True

    def scan_environment(self, collision_queue, agent: AgentState, world_objects: dict) -> list:
        # 1. BROAD-PHASE (Fisica)
        nearby_objects = set()
        for i in range(collision_queue.getNumEntries()):
            hit_node = collision_queue.getEntry(i).getIntoNodePath()
            if hit_node.hasTag("obj_id"):
                nearby_objects.add(hit_node.getTag("obj_id"))

        # 2. NARROW-PHASE (Matematica)
        visible_objects = []
        if not nearby_objects:
            return visible_objects
            
        agent_pos = agent.position + np.array([0, 0, 2])
        
        # CORREZIONE PITCH: Calcoliamo il vero vettore tridimensionale 
        # includendo l'inclinazione della telecamera (Pitch)
        pitch_rad = np.radians(agent.pitch)
        yaw_rad = np.radians(agent.yaw)
        
        fwd_x = -np.sin(yaw_rad) * np.cos(pitch_rad)
        fwd_y = np.cos(yaw_rad) * np.cos(pitch_rad)
        fwd_z = np.sin(pitch_rad)
        
        agent_fwd = np.array([fwd_x, fwd_y, fwd_z])
        agent_fwd = agent_fwd / np.linalg.norm(agent_fwd)
        
        # Rigeneriamo gli assi Right e Up basandoci sul nuovo Forward inclinato
        global_up = np.array([0.0, 0.0, 1.0])
        agent_right = np.cross(agent_fwd, global_up)
        
        # Protezione matematica nel caso in cui stiamo guardando perfettamente verso l'alto/basso
        if np.linalg.norm(agent_right) > 0.001:
            agent_right = agent_right / np.linalg.norm(agent_right)
        else:
            agent_right = np.array([1.0, 0.0, 0.0]) 
            
        agent_up = np.cross(agent_right, agent_fwd)
        agent_up = agent_up / np.linalg.norm(agent_up)

        # Scansione Oggetti
        for obj_id in nearby_objects:
            obj_data = world_objects[obj_id]
            vertices = obj_data["vertices"]
            
            if self._is_aabb_in_frustum(vertices, agent_pos, agent_fwd, agent_right, agent_up):
                visible_objects.append(obj_id)

        return visible_objects