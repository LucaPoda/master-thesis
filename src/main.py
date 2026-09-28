import yaml
import argparse
from pathlib import Path
from direct.task import Task

# Importa i tuoi moduli originali
from core_types import PerceptionInput
from world_state import WorldState
from perception_system import NoisyPerceptionSystem
from spatial_reasoner import GroundTruthSpatialReasoner
from graphic_engine import GraphicEngine
from controllers import KeyboardController
from agent_state import GraphTracker
from visualizer_server import GraphVisualizerBridge
from map_generator import MapGenerator, save_generated_map

def handle_map_selection():
    parser = argparse.ArgumentParser(description="Semantic SLAM Map Loader")
    parser.add_argument("--random", action="store_true", help="Generate a new procedural map")
    parser.add_argument("--level", type=int, default=1, help=" (0-4+)")
    parser.add_argument("--rooms", type=int, default=1, help="Number of rooms to generate")
    parser.add_argument("--run", type=str, help="Load a specific run (e.g., run_20260928_231215)")
    parser.add_argument("--default", action="store_true", help="Force loading of config_map.yaml")
    args = parser.parse_args()

    root_path = Path(__file__).parent.parent
    config_path = root_path / "config"
    log_path = root_path / "log"
    log_path.mkdir(parents=True, exist_ok=True)
    last_run_file = log_path / "last_run.txt"
    
    map_file_path = config_path / "config_map.yaml" # Fallback
    current_run = "default"

    if args.default:
        map_file_path = config_path / "config_map.yaml"
        current_run = "default"
    elif args.run:
        current_run = args.run
        map_file_path = log_path / current_run / "map.yaml"
    elif args.random:
        print(f"Generating procedural map (Level {args.level}, Rooms {args.rooms})...")
        
        gen_config_path = config_path / "config_generator.yaml"
        generator_config = {}
        if gen_config_path.exists():
            with open(gen_config_path, "r") as fc:
                generator_config = yaml.safe_load(fc)

        generator = MapGenerator(level=args.level, num_rooms=args.rooms, config=generator_config)
        new_map_data = generator.generate()
        
        current_run = save_generated_map(new_map_data, root_path)
        map_file_path = log_path / current_run / "map.yaml"
        print(f"New run saved in: log/{current_run}")
    else:
        # Recupera l'ultima run eseguita
        if last_run_file.exists():
            with open(last_run_file, "r") as f:
                saved_run = f.read().strip()
                
            if saved_run == "default":
                map_file_path = config_path / "config_map.yaml"
                current_run = "default"
            else:
                test_path = log_path / saved_run / "map.yaml"
                if test_path.exists():
                    map_file_path = test_path
                    current_run = saved_run
                else:
                    print(f"The previous run ({saved_run}) no longer exists, using default.")

    # Registra la run corrente per la prossima esecuzione senza argomenti
    with open(last_run_file, "w") as f:
        f.write(current_run)
        
    print(f"Loading run: {current_run} (File: {map_file_path})")
    
    # Ritorna l'oggetto Path assoluto
    return map_file_path

def load_yaml_config(file_name_or_path) -> dict:
    """Helper function to load yaml files. Supporta Path assoluti e nomi file (in config)."""
    path = Path(file_name_or_path)
    
    # Se è un nome file semplice (es. 'config_sensors.yaml'), cerca in config/
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).parent.parent / "config" / file_name_or_path
        
    with open(path, "r") as f:
        return yaml.safe_load(f)

class SimulationCoordinator:
    def __init__(self, map_file: str = None):
        # 0. Load configs
        map_config = load_yaml_config(map_file)
        sensor_config = load_yaml_config("config_sensors.yaml")
        view_config = load_yaml_config("config_view.yaml")
        controller_config = load_yaml_config("config_controllers.yaml")

        # 1. Instantiate State
        self.world = WorldState(map_config)
        self.tracker = GraphTracker()
        
        # 2. Dependency Injection / Concrete Strategies
        self.perception = NoisyPerceptionSystem(sensor_config) # Perception system with noise simulation
        self.spatial_reasoner = GroundTruthSpatialReasoner()
        self.controller = KeyboardController(controller_config)
        self.graphics = GraphicEngine(view_config, sensor_config)
        
        # Setup Web Bridge
        dynamic_colors = {}
        for obj_id, obj_data in self.world.get_objects().items():
            r, g, b, a = obj_data["color"]
            dynamic_colors[obj_id] = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"
            
        self.bridge = GraphVisualizerBridge(
            tracker=self.tracker, 
            config_colors=dynamic_colors
        )
        self.bridge.start()
        
        # Initialize rendering
        self.graphics.initialize_world_graphics(self.world.get_objects())
        self.graphics.taskMgr.add(self.tick, "main_simulation_loop")

    def tick(self, task):
        dt = self.graphics.taskMgr.globalClock.getDt()
        current_time = self.graphics.taskMgr.globalClock.getFrameTime()

        # 1. Populate untyped input buffer (ROS Topic equivalent)
        inputs = PerceptionInput(
            raw_inputs=self.graphics.poll_inputs(),
            agent_state=self.world.get_agent_state(),
            world_objects=self.world.get_objects(),
            collision_queue=self.graphics.cHandler,
            timestamp=current_time
        )
        
        # 2. Controller
        command = self.controller.get_command(inputs, dt)
        
        # 3. Physics Ground Truth
        self.world.apply_command(command, dt)
        
        # 4. Perception & Reasoning
        perceived_objects = self.perception.scan_environment(inputs)
        
        perceived_ids = list(perceived_objects.keys())
        semantic_state = self.spatial_reasoner.compute_relations(perceived_ids, inputs)
        
        # 5. Graph Topology 
        self.tracker.update_state(semantic_state)
        
        # 6. Render
        self.graphics.update_render(
            self.world.get_agent_state(),
            perceived_objects
        )
        
        return Task.cont

    def run(self):
        self.graphics.run()

if __name__ == "__main__":
    selected_map = handle_map_selection()
    sim = SimulationCoordinator(selected_map)
    sim.run()