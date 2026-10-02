# src/main.py
import yaml
import argparse
import time
import jsonlines
from pathlib import Path
from direct.task import Task
import sys

# Core modules
from core_types import PerceptionInput, RoomBounds
from world_state import WorldState
from perception_system import NoisyPerceptionSystem, OdometrySystem
from spatial_reasoner import GroundTruthSpatialReasoner
from graphic_engine import GraphicEngine
from agent_state import GraphTracker
from visualizer_server import GraphVisualizerBridge
from map_generator import MapGenerator, save_generated_map, generate_room_coverage

# Controllers
from controllers import (
    KeyboardController, 
    RecordingKeyboardController, 
    ReplayController, 
    TrajectoryController
)

def parse_arguments_and_setup():
    parser = argparse.ArgumentParser(description="Semantic SLAM Map Loader")
    parser.add_argument("--random", action="store_true", help="Generate a new procedural map")
    parser.add_argument("--level", type=int, default=1, help=" (0-4+)")
    parser.add_argument("--rooms", type=int, default=1, help="Number of rooms to generate")
    parser.add_argument("--run", type=str, help="Load a specific run (e.g., run_20260928_231215)")
    parser.add_argument("--default", action="store_true", help="Force loading of config_map.yaml")
    
    parser.add_argument("--record", action="store_true", help="Record manual trajectory to JSON")
    parser.add_argument("--replay", type=str, help="Path to trajectory JSON for deterministic playback")
    parser.add_argument("--auto", action="store_true", help="Automated room coverage via TrajectoryController")
    
    args = parser.parse_args()

    root_path = Path(__file__).parent.parent
    config_path = root_path / "config"
    log_path = root_path / "log"
    log_path.mkdir(parents=True, exist_ok=True)
    last_run_file = log_path / "last_run.txt"
    
    map_file_path = config_path / "config_map.yaml"
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

    with open(last_run_file, "w") as f:
        f.write(current_run)
        
    print(f"Loading run: {current_run} (File: {map_file_path})")
    
    run_log_dir = log_path / current_run if current_run != "default" else log_path / "default"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    
    return args, map_file_path, run_log_dir

def load_yaml_config(file_name_or_path) -> dict:
    path = Path(file_name_or_path)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).parent.parent / "config" / file_name_or_path
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
class TelemetryLogger:
    def __init__(self, log_dir, rate_hz=10.0):
        self.interval = 1.0 / rate_hz
        self.last_log_time = time.time()
        self.file_path = Path(log_dir) / "telemetry.jsonl"
        self.file = jsonlines.open(self.file_path, mode='w')

    def tick(self, gt_state, believed_state, raw_perception, filtered_state=None):
        now = time.time()
        if now - self.last_log_time >= self.interval:
            self.file.write({
                "timestamp": now,
                "ground_truth": gt_state.to_dict() if hasattr(gt_state, 'to_dict') else gt_state.__dict__,
                "believed_pre_filter": believed_state.to_dict() if hasattr(believed_state, 'to_dict') else believed_state.__dict__,
                "raw_perception": raw_perception,
                "believed_post_filter": filtered_state.to_dict() if filtered_state and hasattr(filtered_state, 'to_dict') else None
            })
            self.last_log_time = now

class SimulationCoordinator:
    def __init__(self, args, map_file: str, log_dir: Path):
        map_config = load_yaml_config(map_file)
        sensor_config = load_yaml_config("config_sensors.yaml")
        view_config = load_yaml_config("config_view.yaml")
        controller_config = load_yaml_config("config_controllers.yaml")

        self.world = WorldState(map_config)
        self.tracker = GraphTracker()
        
        telemetry_rate = sensor_config.get('telemetry', {}).get('logging_rate_hz', 10.0)
        self.logger = TelemetryLogger(log_dir, rate_hz=telemetry_rate)
        
        self.perception = NoisyPerceptionSystem(sensor_config)
        self.spatial_reasoner = GroundTruthSpatialReasoner()
        self.graphics = GraphicEngine(view_config, sensor_config)
        
        noise_config = sensor_config.get('odometry', {})
        self.odometry = OdometrySystem(noise_config)
        
        if args.replay:
            print(f"Initializing ReplayController with: {args.replay}")
            self.controller = ReplayController(args.replay)
        elif args.auto:
            print("Initializing TrajectoryController for auto-exploration...")
            if 'rooms' in map_config and len(map_config['rooms']) > 0:
                raw_bounds = map_config['rooms'][0]['bounds']
            else:
                objects = map_config.get("objects", {})
                if objects:
                    min_x = min([obj["center"][0] for obj in objects.values()])
                    max_x = max([obj["center"][0] for obj in objects.values()])
                    min_y = min([obj["center"][1] for obj in objects.values()])
                    max_y = max([obj["center"][1] for obj in objects.values()])
                else:
                    world_size = map_config.get('world_size', [50, 50, 1])
                    min_x, max_x = -world_size[0]/2, world_size[0]/2
                    min_y, max_y = -world_size[1]/2, world_size[1]/2

                # Compute bounded perimeter padding
                raw_bounds = {'min_x': min_x + 2, 'max_x': max_x - 2, 'min_y': min_y + 2, 'max_y': max_y - 2}

            room_bounds = RoomBounds(
                min_x=int(raw_bounds['min_x']), max_x=int(raw_bounds['max_x']), 
                min_y=int(raw_bounds['min_y']), max_y=int(raw_bounds['max_y'])
            )
            waypoints = generate_room_coverage(room_bounds)
            print(f"Generated {len(waypoints)} scanning waypoints.")
            self.controller = TrajectoryController(waypoints)
        else:
            base_ctrl = KeyboardController(controller_config)
            if args.record:
                print(f"Initializing RecordingKeyboardController. Logs will save to {log_dir}")
                self.controller = RecordingKeyboardController(base_ctrl, log_dir)
            else:
                self.controller = base_ctrl
        
        dynamic_colors = {}
        for obj_id, obj_data in self.world.get_objects().items():
            r, g, b, a = obj_data["color"]
            dynamic_colors[obj_id] = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"
            
        self.bridge = GraphVisualizerBridge(
            tracker=self.tracker, 
            config_colors=dynamic_colors
        )
        self.bridge.start()
        
        self.graphics.initialize_world_graphics(self.world.get_objects())
        self.graphics.taskMgr.add(self.tick, "main_simulation_loop")

    def tick(self, task):
        dt = self.graphics.taskMgr.globalClock.getDt()
        current_time = self.graphics.taskMgr.globalClock.getFrameTime()

        inputs = PerceptionInput(
            raw_inputs=self.graphics.poll_inputs(),
            agent_state=self.world.get_agent_state(),
            world_objects=self.world.get_objects(),
            collision_queue=self.graphics.cHandler,
            timestamp=current_time
        )
        
        command = self.controller.get_command(inputs, dt)
        self.world.apply_command(command, dt)
        gt_state = self.world.get_agent_state()
        believed_state = self.odometry.estimate(gt_state, dt)
        
        perceived_objects = self.perception.scan_environment(inputs)
        perceived_ids = list(perceived_objects.keys())
        semantic_state = self.spatial_reasoner.compute_relations(perceived_ids, inputs)
        
        self.tracker.update_state(semantic_state)
        
        self.logger.tick(
            gt_state=gt_state, 
            believed_state=believed_state, 
            raw_perception=perceived_objects,
            filtered_state=None
        )
        
        self.graphics.update_render(gt_state, perceived_objects)
        return Task.cont

    def run(self):
        try:
            self.graphics.run()
        finally:
            if hasattr(self.controller, 'save'):
                print("Saving recorded trajectory...")
                self.controller.save()
            if hasattr(self.logger, 'file'):
                self.logger.file.close()

if __name__ == "__main__":
    args, selected_map, log_dir = parse_arguments_and_setup()
    sim = SimulationCoordinator(args, selected_map, log_dir)
    sim.run()