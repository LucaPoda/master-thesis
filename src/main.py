import yaml
from pathlib import Path
from direct.task import Task

from core_types import PerceptionInput
from world_state import WorldState
from perception_system import FrustumPerceptionSystem
from spatial_reasoner import GroundTruthSpatialReasoner
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
        # 0. Load configs
        map_config = load_yaml_config("config_map.yaml")
        sensor_config = load_yaml_config("config_sensors.yaml")
        view_config = load_yaml_config("config_view.yaml")
        controller_config = load_yaml_config("config_controllers.yaml")

        # 1. Instantiate State
        self.world = WorldState(map_config)
        self.tracker = GraphTracker()
        
        # 2. Dependency Injection / Concrete Strategies
        self.perception = FrustumPerceptionSystem(sensor_config)
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
        
        # 1. Populate untyped input buffer (ROS Topic equivalent)
        inputs = PerceptionInput(
            raw_inputs=self.graphics.poll_inputs(),
            agent_state=self.world.get_agent_state(),
            world_objects=self.world.get_objects(),
            collision_queue=self.graphics.cHandler
        )
        
        # 2. Controller
        command = self.controller.get_command(inputs, dt)
        
        # 3. Physics Ground Truth
        self.world.apply_command(command, dt)
        
        # 4. Perception & Reasoning
        perceived_ids = self.perception.scan_environment(inputs)
        semantic_state = self.spatial_reasoner.compute_relations(perceived_ids, inputs)
        
        # 5. Graph Topology 
        self.tracker.update_state(semantic_state)
        
        # 6. Render
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