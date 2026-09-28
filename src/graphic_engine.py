# graphic_engine.py
from panda3d.core import loadPrcFileData
loadPrcFileData("", "audio-library-name null")

import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    WindowProperties, LColor, NodePath, Point3, Vec3,
    CollisionTraverser, CollisionHandlerQueue, CollisionNode, 
    CollisionSphere, CollisionBox, BitMask32,
    TransparencyAttrib, GeomVertexFormat, GeomVertexData,
    GeomVertexWriter, GeomTriangles, Geom, GeomNode
)

class GraphicEngine(ShowBase):
    def __init__(self, view_config, sensor_config):
        super().__init__()
        
        # Window and Input Setup
        props = WindowProperties()
        props.setTitle(view_config["window_title"])
        self.win.requestProperties(props)
        self.disableMouse()
        self.camLens.setFov(view_config["global_fov"])
        self.mouse_locked = False
        self.accept("mouse1", self.lock_mouse)
        self.accept("escape", self.unlock_mouse)
        self.raw_inputs = {"keys": {k: False for k in ["w", "a", "s", "d", "space", "lshift"]}, "mouse_dx": 0, "mouse_dy": 0}
        self.setup_input_listeners()
        
        # Hierarchy and Rendering Setup
        self.render_nodes = {}
        self.agent_node = NodePath("AgentVisual")
        self.agent_node.reparentTo(self.render)
        self.camera.reparentTo(self.agent_node)

        # Physical System and Collisions
        self.cTrav = CollisionTraverser()
        self.cHandler = CollisionHandlerQueue()
        # self.cTrav.showCollisions(self.render)
        
        self.setup_fov_sensor(sensor_config)
        self.setup_fov_visual(sensor_config)

    def setup_fov_sensor(self, sensor_config):
        """Spherical sensor for physical broad-phase collision detection. The radius is based on the max_range parameter."""
        max_range = sensor_config.get("max_range", 0.0)
        
        # If 0, we use an enormous radius to cover the entire world.
        # Otherwise, we use the specified radius.
        collider_radius = max_range if max_range > 0 else 100000.0
        
        fov_solid = CollisionSphere(0, 0, 0, collider_radius)
        fov_node = CollisionNode('agent_fov_sensor')
        fov_node.addSolid(fov_solid)
        
        fov_node.setFromCollideMask(BitMask32.bit(1))
        fov_node.setIntoCollideMask(BitMask32.allOff())
        
        self.fov_path = self.camera.attachNewNode(fov_node)
        self.cTrav.addCollider(self.fov_path, self.cHandler)

    def setup_fov_visual(self, sensor_config):
        """Draw the visual cone (Frustum) based on the range"""
        fov_h_rad = np.radians(sensor_config.get("fov_horizontal", 90.0))
        fov_v_rad = np.radians(sensor_config.get("fov_vertical", 60.0))
        max_range = sensor_config.get("max_range", 0.0)
        
        # For screen rendering, if it's "infinite", we stop at 1000 units
        # per evitare glitch legati alla profondità della telecamera (Z-fighting).
        visual_range = max_range if max_range > 0 else 1000.0 

        y_far = visual_range
        x_far = visual_range * np.tan(fov_h_rad / 2.0)
        z_far = visual_range * np.tan(fov_v_rad / 2.0)

        # 1. Prepare the data for the vertices
        format = GeomVertexFormat.getV3()
        vdata = GeomVertexData('frustum', format, GeomVertexData.UHStatic)
        vertex = GeomVertexWriter(vdata, 'vertex')

        # Vertex 0: Origin (camera position)
        vertex.addData3f(0, 0, 0)
        # Vertices 1-4: far plane corners (in camera space)
        vertex.addData3f(-x_far, y_far, z_far)   # Top Left
        vertex.addData3f(x_far, y_far, z_far)    # Top Right
        vertex.addData3f(x_far, y_far, -z_far)   # Bottom Right
        vertex.addData3f(-x_far, y_far, -z_far)  # Bottom Left

        # 2. Create the triangles for the 4 side faces and the front face
        tris = GeomTriangles(GeomVertexData.UHStatic)
        tris.addVertices(0, 1, 2) # Top
        tris.addVertices(0, 2, 3) # Right
        tris.addVertices(0, 3, 4) # Bottom
        tris.addVertices(0, 4, 1) # Left
        tris.addVertices(1, 4, 3) # Front (Distant Face 1)
        tris.addVertices(1, 3, 2) # Front (Distant Face 2)

        # 3. Assemble the node
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        visual_node = GeomNode('fov_visual')
        visual_node.addGeom(geom)

        self.fov_visual_path = self.camera.attachNewNode(visual_node)
        
        # 4. Apply Color and Transparency (RGBA: Yellow at 15% opacity)
        self.fov_visual_path.setColor(1.0, 1.0, 0.0, 0.15)
        self.fov_visual_path.setTransparency(TransparencyAttrib.MAlpha)
        
        # Make the pyramid visible from both sides in the case we are in third person
        self.fov_visual_path.setTwoSided(True) 
        self.fov_visual_path.setLightOff(True) # Prevents lights from altering the color of the FOV

    def initialize_world_graphics(self, world_objects: dict):
        for obj_id, data in world_objects.items():
            center = data["center"]
            hx, hy, hz = data["extensions"]
            color = data["color"]
            
            # 1. Root Node for the object, positioned at the center of the AABB
            obj_root = NodePath(f"root_{obj_id}")
            obj_root.reparentTo(self.render)
            obj_root.setPos(*center)
            
            # 2. Ghost Model (Transparent, locked on the Ground Truth)
            ghost_model = self.loader.loadModel("models/box")
            ghost_model.reparentTo(obj_root)
            ghost_model.setScale(hx * 2, hy * 2, hz * 2)
            ghost_model.setPos(-hx, -hy, -hz) # Centers the ghost model on the root
            ghost_model.setColor(LColor(0.5, 0.5, 0.5, 0.3)) # Semi-transparent gray
            ghost_model.setTransparency(TransparencyAttrib.MAlpha)
            ghost_model.setTextureOff(1)
            
            # 3. Solid Model (Noisy Perception)
            # Locked to self.render for using absolute coordinates without going crazy with deltas
            solid_model = self.loader.loadModel("models/box")
            solid_model.reparentTo(self.render)
            solid_model.setColor(LColor(*color))
            solid_model.setTextureOff(1)
            solid_model.hide() # Hidden until the camera sees it
            
            # 4. Collision Model (Based on the AABB, used for broad-phase detection)
            c_box = CollisionBox(Point3(0, 0, 0), hx, hy, hz)
            c_node = CollisionNode(f"physics_{obj_id}")
            c_node.addSolid(c_box)
            c_node.setIntoCollideMask(BitMask32.bit(1))
            c_node.setFromCollideMask(BitMask32.allOff())
            
            c_path = obj_root.attachNewNode(c_node)
            c_path.setTag("obj_id", obj_id)
            
            # Saving references to the models for later updates
            self.render_nodes[obj_id] = {
                "ghost": ghost_model,
                "solid": solid_model
            }

    def update_render(self, agent_state, perceived_objects):
        self.cTrav.traverse(self.render)
        self.agent_node.setPos(*agent_state.position)
        self.agent_node.setH(agent_state.yaw)
        self.camera.setP(agent_state.pitch)
        self.camera.setPos(0, 0, 2)

        for obj_id, nodes in self.render_nodes.items():
            solid_node = nodes["solid"]

            if obj_id in perceived_objects:
                solid_node.show()
                
                # Extract the perceived data for the object (noisy perception)
                p_data = perceived_objects[obj_id]
                cx, cy, cz = p_data["center"]
                px, py, pz = p_data["extensions"]
                
                # Application of the distorted parameters
                solid_node.setScale(px * 2, py * 2, pz * 2) 
                # We report the correct center by compensating for the origin of the Panda3D model
                solid_node.setPos(cx - px, cy - py, cz - pz)
            else:
                solid_node.hide()

    def setup_input_listeners(self):
        for key in self.raw_inputs["keys"].keys():
            self.accept(key, self.raw_inputs["keys"].update, [{key: True}])
            self.accept(f"{key}-up", self.raw_inputs["keys"].update, [{key: False}])

    def lock_mouse(self):
        props = WindowProperties()
        props.setCursorHidden(True)
        props.setMouseMode(WindowProperties.M_absolute)
        self.win.requestProperties(props)
        self.mouse_locked = True
        cx, cy = self.win.getXSize() // 2, self.win.getYSize() // 2
        self.win.movePointer(0, cx, cy)

    def unlock_mouse(self):
        props = WindowProperties()
        props.setCursorHidden(False)
        props.setMouseMode(WindowProperties.M_absolute)
        self.win.requestProperties(props)
        self.mouse_locked = False

    def poll_inputs(self) -> dict:
        dx, dy = 0, 0
        if self.mouse_locked and self.mouseWatcherNode.hasMouse():
            md = self.win.getPointer(0)
            x, y = md.getX(), md.getY()
            cx = self.win.getXSize() // 2
            cy = self.win.getYSize() // 2
            dx = x - cx
            dy = y - cy
            self.win.movePointer(0, cx, cy)
        if abs(dx) > 150 or abs(dy) > 150: 
            dx, dy = 0, 0
        self.raw_inputs["mouse_dx"] = dx
        self.raw_inputs["mouse_dy"] = dy
        return self.raw_inputs