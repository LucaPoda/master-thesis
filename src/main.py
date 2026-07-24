import yaml
from pathlib import Path
from direct.task import Task

# Import the classes from their respective modules in src/
from world_state import WorldState
from perception_system import PerceptionSystem
from graphic_engine import GraphicEngine
from controllers import KeyboardController
from agent_state import GraphTracker
from visualizer_server import GraphVisualizerBridge

def load_yaml_config(file_name: str) -> dict:
    """Helper function to load yaml files from the config folder"""
    base_path = Path(__file__).parent.parent / "config"
    file_path = base_path / file_name
    
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

class SimulationCoordinator:
    def __init__(self):
        # 0. Load Dynamic Configurations
        map_config = load_yaml_config("config_map.yaml")
        sensor_config = load_yaml_config("config_sensors.yaml")
        view_config = load_yaml_config("config_view.yaml")
        controller_config = load_yaml_config("config_controllers.yaml")

        # 0. Load Dynamic Configurations
        self.world = WorldState(map_config)
        self.perception = PerceptionSystem(sensor_config)
        self.graphics = GraphicEngine(view_config, sensor_config)
        self.controller = KeyboardController(controller_config)
        
        # Setup Graph Tracker e Web Server ---
        self.tracker = GraphTracker()
        
        # Extract dynamically the colors from the WorldState converting them for the Web
        dynamic_colors = {}
        for obj_id, obj_data in self.world.get_objects().items():
            # obj_data["color"] is an [r, g, b, a] list in a [0.0 - 1.0] range
            r, g, b, a = obj_data["color"]
            dynamic_colors[obj_id] = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"
            
        self.bridge = GraphVisualizerBridge(
            tracker=self.tracker, 
            config_colors=dynamic_colors, 
            host="127.0.0.1", 
            port=8000
        )
        self.bridge.start() # Start the web server in a daemon thread
        
        # 2. Setup Grafico e Fisico
        self.graphics.initialize_world_graphics(self.world.get_objects())

        # 3. Register the main simulation loop
        self.graphics.taskMgr.add(self.tick, "main_simulation_loop")

    def tick(self, task):
        dt = self.graphics.taskMgr.globalClock.getDt()
        
        # --- PHASE 1: INPUT ---
        raw_inputs = self.graphics.poll_inputs()
        
        # --- PHASE 2: DECISIONE / CONTROLLO ---
        agent_state = self.world.get_agent_state()
        command = self.controller.get_command(agent_state, dt, inputs=raw_inputs)
        
        # --- PHASE 3: AGGIORNAMENTO FISICO (Ground Truth) ---
        self.world.apply_command(command, dt)
        
        # --- PHASE 4: PERCEZIONE SIMULATA ---
        perceived_ids = self.perception.scan_environment(
            self.graphics.cHandler,
            self.world.get_agent_state(),
            self.world.get_objects()
        )
        
        self.tracker.update_state(perceived_ids)
        
        # --- PHASE 5: RENDERING ---
        self.graphics.update_render(
            self.world.get_agent_state(),
            self.world.get_objects(),
            perceived_ids
        )
        
        return Task.cont

    def run(self):
        self.graphics.run()

if __name__ == "__main__":
    sim = SimulationCoordinator()
    sim.run()