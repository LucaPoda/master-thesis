import numpy as np
from core_types import AgentState, MovementCommand

class WorldState:
    def __init__(self, map_config):
        self.objects = map_config["objects"]
        self.agent = AgentState(
            position=np.array(map_config["agent_start_pos"], dtype=float),
            yaw=map_config["agent_start_yaw"],
            pitch=0.0
        )
        
        # Pre-compute the 8 vertices for the optical perception system
        for obj_id, data in self.objects.items():
            cx, cy, cz = data["center"]
            hx, hy, hz = data["estensioni"]
            data["vertices"] = [
                np.array([cx + dx, cy + dy, cz + dz])
                for dx in (-hx, hx)
                for dy in (-hy, hy)
                for dz in (-hz, hz)
            ]

    def get_agent_state(self) -> AgentState:
        return self.agent

    def get_objects(self) -> dict:
        return self.objects

    def apply_command(self, cmd: MovementCommand, dt: float):
        """Applica la cinematica lineare (Velocità * Tempo)"""
        self.agent.position += cmd.velocity * dt
        self.agent.yaw += cmd.yaw_delta
        self.agent.pitch = np.clip(self.agent.pitch + cmd.pitch_delta, -80.0, 80.0)