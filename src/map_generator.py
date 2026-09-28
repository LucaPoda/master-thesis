import yaml
import random
import numpy as np
from pathlib import Path
from datetime import datetime

class MapGenerator:
    def __init__(self, level: int, num_rooms: int, config: dict):
        self.level = level
        self.num_rooms = num_rooms
        
        # Parameter extraction from the configuration dictionary
        gen_cfg = config.get("generator", {})
        self.room_size = gen_cfg.get("room_size", 20.0)
        self.wall_thick = gen_cfg.get("wall_thickness", 0.5)
        self.wall_height = gen_cfg.get("wall_height", 3.0)
        self.density = gen_cfg.get("density", 1.0)
        self.margin = gen_cfg.get("margin", 0.2)
        
        self.objects = {}
        self.obj_counter = 0

    def _check_overlap(self, center, extents):
        c1 = np.array(center)
        e1 = np.array(extents)
        for obj_data in self.objects.values():
            c2 = np.array(obj_data["center"])
            e2 = np.array(obj_data["extensions"])
            
            overlap = True
            for i in range(3):
                if abs(c1[i] - c2[i]) >= (e1[i] + e2[i] + self.margin):
                    overlap = False
                    break
            if overlap:
                return True
        return False

    def _add_object(self, prefix, center, extents, color):
        name = f"{prefix}_{self.obj_counter}"
        self.objects[name] = {
            "center": [round(float(c), 2) for c in center],
            "extensions": [round(float(e), 2) for e in extents],
            "color": [round(float(c), 2) for c in color]
        }
        self.obj_counter += 1
        return name

    def generate(self) -> dict:
        self.objects = {}
        self.obj_counter = 0
        
        # Adapts dynamically the world dimensions to the number of rooms
        grid_cols = int(np.ceil(np.sqrt(self.num_rooms)))
        self.world_dim = grid_cols * self.room_size + 50
        world_size = [self.world_dim, self.world_dim, 1]
        
        if self.level == 0:
            self._generate_level_0()
        else:
            self._generate_rooms()
            if self.level >= 1:
                self._generate_level_1()
            if self.level >= 2:
                self._generate_level_2()
            if self.level >= 3:
                self._generate_level_3()

        return {
            "world_size": world_size,
            "ground_color": [0.2, 0.2, 0.2, 1.0],
            "agent_start_pos": [self.room_size/2, self.room_size/2, 1.0],
            "agent_start_yaw": 0.0,
            "objects": self.objects
        }

    def _generate_level_0(self):
        num_objects = int(15 * self.num_rooms * self.density)
        for _ in range(num_objects):
            extents = [random.uniform(0.5, 2.0) for _ in range(3)]
            for _ in range(50): 
                center = [
                    random.uniform(-self.world_dim/2, self.world_dim/2), 
                    random.uniform(-self.world_dim/2, self.world_dim/2), 
                    extents[2] + random.uniform(0, 5)
                ]
                if not self._check_overlap(center, extents):
                    color = [random.random(), random.random(), random.random(), 1.0]
                    self._add_object("RND", center, extents, color)
                    break

    def _create_wall_with_door(self, prefix, center, extents, is_horizontal, color):
        door_width = 3.0
        door_height = 2.0
        
        axis_idx = 0 if is_horizontal else 1
        wall_length = extents[axis_idx] * 2
        start = center[axis_idx] - extents[axis_idx]
        
        door_center = random.uniform(
            start + door_width/2 + 1.0, 
            start + wall_length - door_width/2 - 1.0
        )
        
        # Block A
        len_1 = door_center - (door_width / 2) - start
        if len_1 > 0.1:
            c1 = list(center)
            c1[axis_idx] = start + len_1 / 2
            e1 = list(extents)
            e1[axis_idx] = len_1 / 2
            self._add_object(f"{prefix}_A", c1, e1, color)
            
        # Block B
        end = start + wall_length
        len_2 = end - (door_center + door_width / 2)
        if len_2 > 0.1:
            c2 = list(center)
            c2[axis_idx] = end - len_2 / 2
            e2 = list(extents)
            e2[axis_idx] = len_2 / 2
            self._add_object(f"{prefix}_B", c2, e2, color)

    def _generate_rooms(self):
        grid_cols = int(np.ceil(np.sqrt(self.num_rooms)))
        built_walls = set()
        
        for i in range(self.num_rooms):
            col = i % grid_cols
            row = i // grid_cols
            color = [0.85, 0.85, 0.85, 1.0]
            
            walls_to_build = [
                ("H", col, row),     
                ("H", col, row - 1), 
                ("V", col, row),     
                ("V", col - 1, row)  
            ]
            
            for wall_id in walls_to_build:
                if wall_id in built_walls:
                    continue
                
                orientation, c, r = wall_id
                prefix = f"WALL_{orientation}_{c}_{r}"
                
                if orientation == "H":
                    center = [c * self.room_size + self.room_size/2, r * self.room_size + self.room_size, self.wall_height/2]
                    extents = [self.room_size/2, self.wall_thick/2, self.wall_height/2]
                    self._create_wall_with_door(prefix, center, extents, True, color)
                else:
                    center = [c * self.room_size + self.room_size, r * self.room_size + self.room_size/2, self.wall_height/2]
                    extents = [self.wall_thick/2, self.room_size/2, self.wall_height/2]
                    self._create_wall_with_door(prefix, center, extents, False, color)
                
                built_walls.add(wall_id)

    def _generate_level_1(self):
        inner_size = self.room_size - (self.wall_thick * 2) - 1.0
        grid_cols = int(np.ceil(np.sqrt(self.num_rooms)))
        num_items = int(5 * self.density)
        
        for i in range(self.num_rooms):
            col = i % grid_cols
            row = i // grid_cols
            offset_x = col * self.room_size + self.room_size/2
            offset_y = row * self.room_size + self.room_size/2
            
            for _ in range(num_items):
                ext = [random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)]
                for _ in range(20):
                    center = [
                        offset_x + random.uniform(-inner_size/2, inner_size/2),
                        offset_y + random.uniform(-inner_size/2, inner_size/2),
                        ext[2]
                    ]
                    if not self._check_overlap(center, ext):
                        self._add_object("FLOOR", center, ext, [0.3, 0.5, 0.8, 1.0])
                        break

    def _generate_level_2(self):
        inner_size = self.room_size - (self.wall_thick * 2) - 1.0
        grid_cols = int(np.ceil(np.sqrt(self.num_rooms)))
        num_tables = max(1, int(3 * self.density))
        
        for i in range(self.num_rooms):
            col = i % grid_cols
            row = i // grid_cols
            offset_x = col * self.room_size + self.room_size/2
            offset_y = row * self.room_size + self.room_size/2
            
            for _ in range(num_tables):
                t_ext = [random.uniform(1.0, 2.0), random.uniform(1.0, 2.0), 0.5]
                t_center = [0, 0, 0]
                placed = False
                
                for _ in range(20):
                    t_center = [
                        offset_x + random.uniform(-inner_size/2, inner_size/2),
                        offset_y + random.uniform(-inner_size/2, inner_size/2),
                        t_ext[2]
                    ]
                    if not self._check_overlap(t_center, t_ext):
                        self._add_object("TABLE", t_center, t_ext, [0.6, 0.3, 0.1, 1.0])
                        placed = True
                        break
                
                if placed:
                    # Adds on average one object per unit of density on the table
                    for _ in range(max(1, int(1 * self.density))):
                        o_ext = [random.uniform(0.2, 0.4), random.uniform(0.2, 0.4), random.uniform(0.2, 0.5)]
                        o_center = [
                            t_center[0] + random.uniform(-t_ext[0]*0.5, t_ext[0]*0.5),
                            t_center[1] + random.uniform(-t_ext[1]*0.5, t_ext[1]*0.5),
                            (t_center[2] + t_ext[2]) + o_ext[2] 
                        ]
                        self._add_object("ONTOP", o_center, o_ext, [0.9, 0.1, 0.1, 1.0])

    def _generate_level_3(self):
        """Objects hung on the walls (paintings, whiteboards, screens)."""
        wall_pieces = [(k, v) for k, v in self.objects.items() if "WALL_" in k and not k.endswith("_TOP")]
        
        num_wall_objs = int(len(wall_pieces) * 0.4 * self.density)

        print(f"Generating {num_wall_objs} objects hung on the walls...")
        
        for _ in range(num_wall_objs):
            w_name, w_data = random.choice(wall_pieces)
            c_w = w_data["center"]
            e_w = w_data["extensions"]
            
            is_horizontal = e_w[0] > e_w[1]
            p_ext = [random.uniform(0.4, 1.2), random.uniform(0.4, 1.2), random.uniform(0.4, 1.0)]
            side = random.choice([1, -1]) 
            
            # Applies an offset that explicitly includes the margin to overcome the overlap check
            offset_from_wall = self.margin + 0.02
            
            if is_horizontal:
                p_ext[1] = 0.05 
                c_p = [
                    c_w[0] + random.uniform(-e_w[0]*0.4, e_w[0]*0.4),
                    c_w[1] + side * (e_w[1] + p_ext[1] + offset_from_wall), # We add the offset_from_wall here to ensure the object is placed outside the wall's bounding box and pass the overlap check
                    random.uniform(1.5, self.wall_height - p_ext[2] - 0.2)
                ]
            else:
                p_ext[0] = 0.05 
                c_p = [
                    c_w[0] + side * (e_w[0] + p_ext[0] + offset_from_wall), # We add the offset_from_wall here to ensure the object is placed outside the wall's bounding box and pass the overlap check
                    c_w[1] + random.uniform(-e_w[1]*0.4, e_w[1]*0.4),
                    random.uniform(1.5, self.wall_height - p_ext[2] - 0.2)
                ]
                
            if not self._check_overlap(c_p, p_ext):
                print(f"Object placed successfully on wall {w_name}.")
                self._add_object("WALL_OBJ", c_p, p_ext, [0.1, 0.1, 0.8, 1.0])
            else:
                print(f"Object not placed on wall {w_name} due to overlap.")

def save_generated_map(config_dict: dict, root_path: Path) -> str:
    # Generate an unique ID for the run based on the timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    
    # Create the run directory inside log/
    run_folder = root_path / "log" / run_id
    run_folder.mkdir(parents=True, exist_ok=True)
    
    # Save the map with a static name inside the run folder
    file_path = run_folder / "map.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False)
        
    return run_id