# Semantic Scene Graph - Simulator & Visualizer

This project was developed as a **preliminary study for a Master's thesis**. The main objective is to build a simulated system capable of converting an agent's dynamic visual perception into a **Semantic Spatial Graph (Semantic Scene Graph)**.

Each node in the graph represents a **unique visual state** (the set of objects perceived by the agent at a specific moment), while the edges represent topological transitions between these states during environmental exploration.

---

## 🛠️ Architecture and Design Principles

The system was designed following the **Single Responsibility Principle (SRP)** to ensure complete decoupling between components. This modular structure will allow replacing the pure Python simulation with a **ROS / Real Robot / Gazebo** environment in the future without modifying the graph logic or user interface.

### 1. `WorldState` & Ground-Truth (`world_state.py`)
* **Role:** Represents the absolute state of the environment and the agent in 3D space.
* **Logic:** Reads the map configuration and pre-computes the 8 vertices of each object's Axis-Aligned Bounding Box (AABB) mathematically to enable fast geometric calculations.
* **Parameters:** Objects are defined purely by geometric parameters (`center` and `extensions`), eliminating dependencies on external 3D models or `.gltf`/`.egg` files.

### 2. `GraphicEngine` (`graphic_engine.py`)
* **Role:** Handles 3D rendering (via Panda3D) and basic collision detection.
* **Logic:**
  * Generates visual geometry by scaling Panda3D's default cube (`models/box`) around the true geometric center.
  * Associates each model with a mathematical `CollisionBox` (`BitMask32` configured as an *Into* target only to conserve CPU resources).
  * Renders a **translucent view frustum pyramid** attached to the agent's camera to visualize the active visual field.
  * Manages visual highlighting: currently perceived objects retain their original color, while objects outside the FOV are converted to grayscale.

### 3. `PerceptionSystem` (`perception_system.py`)
* **Role:** Idealized optical sensor and camera model.
* **Logic (Broad-Phase vs Narrow-Phase):**
  * **Broad-Phase (Physics):** Uses a `CollisionSphere` (with customizable radius, or set to infinite by assigning `max_range: 0` in configuration) to intercept nearby objects.
  * **Narrow-Phase (Frustum Culling):** Implements a 6-plane geometric algorithm (AABB vs Frustum Planes) based on the agent's 3D orientation (including *Yaw* and *Pitch*).
  * **Perception Notes:** If *all* vertices of an object fall outside any single frustum plane, the object is culled. Otherwise, it is considered perceived (even if the agent is very close and sees only the center of a surface).

### 4. `GraphTracker` (`agent_state.py`)
* **Role:** Pure semantic graph management (backend logic, zero web or graphics dependencies).
* **Logic:**
  * Each node is identified by a `frozenset` of object IDs. The order of perception does not matter, ensuring that the same set of perceived objects always maps to the same node.
  * At each simulator tick, it compares the set of visible objects against the current state.
  * If the state changes, it checks whether the node already exists in the graph: if yes, it transitions to that state and creates an edge with the previous node; if no, a new node is instantiated.
  * Implements the *Observer/Callback* pattern to notify external modules of graph updates.

### 5. `GraphVisualizerBridge` & Web UI (`visualizer_server.py` & `index.html`)
* **Role:** Communication micro-service and graphical web interface for the graph.
* **Logic:**
  * Launches a **FastAPI + WebSockets** server in a separate *Daemon Thread* so as not to block Panda3D's 60 FPS execution loop.
  * Receives updates from `GraphTracker`, serializes the graph, and streams it to the frontend in real time.
  * Maintains a synchronized color palette by reading `RGBA` colors from the map configuration file.
  * **Frontend (`index.html`):** Uses the **Vis.js** library to render the graph topology. Supports interactive inspection via *hover* (to view objects present in a node) and *clicking on edges* (to view the state delta, i.e., objects entering or leaving the field of view).

---

## 🧠 Design Notes & Thesis Considerations

1. **Future Evolution of Spatial Relations (Semantic Graph):**
   * *Current behavior:* Equality between two graph nodes is based strictly on the set of visible objects (e.g., `['B1', 'M1']`).
   * *Future work:* Relational logic based on relative spatial positions will be introduced (e.g., `{'B1', 'M1', ('B1', 'LEFT_OF', 'M1')}`). Two states containing the same objects but arranged in different spatial configurations will yield distinct nodes, reflecting different agent poses in the physical world.

2. **Why NO continuous spatial mapping inside graph nodes?**
   * A deliberate choice was made **not** to store exact physical coordinates or spatial footprint polygons within the graph nodes. This avoids managing complex, discontinuous geometric regions, keeping the graph lightweight and focused strictly on the level of semantic and topological abstraction.

3. **ROS and Real Hardware Integration:**
   * Thanks to the `GraphTracker <-> Bridge <-> Frontend` decoupling, integrating with a real system simply requires creating a ROS node that receives object detections from a Computer Vision pipeline and passes them to `GraphTracker.update_state()`. The entire tracking logic and web dashboard will function without modification.

---

## 📁 Project Structure

```text
.
├── config/
│   ├── config_controllers.yaml  # Mouse sensitivity and movement speeds
│   ├── config_map.yaml          # Ground-truth: Objects, colors, positions, and extents
│   ├── config_sensors.yaml      # FOV aperture angles (horizontal/vertical) and max range
│   └── config_view.yaml         # Window settings and camera parameters
├── src/
│   ├── agent_state.py           # Pure Graph logic (GraphTracker)
│   ├── controllers.py           # Keyboard and mouse control handlers
│   ├── core_types.py            # Dataclasses (AgentState, MovementCommand)
│   ├── graphic_engine.py        # Panda3D rendering and procedural mesh generation
│   ├── index.html               # Web Graph Visualizer (Vis.js)
│   ├── main.py                  # Main simulation orchestrator entry point
│   ├── perception_system.py     # 6-plane Frustum Culling calculation
│   ├── visualizer_server.py     # FastAPI & WebSockets background server
│   └── world_state.py           # World state management
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🚀 Getting Started

### 1. Requirements & Installation
Ensure you have **Python >= 3.10** installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Simulationx
Execute the main entry point from the project root directory:

```bash
python src/main.py
```

### 3. Opening the Web Visualizer
Upon starting the simulation, the integrated web server listens on [http://127.0.0.1:8000](http://127.0.0.1:8000).

Open your browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) to view the graph topology expand in real time as you navigate the camera through the 3D window.

## 🎮 Simulation Controls

| Key / Input | Action |
| :--- | :--- |
| **Left Mouse Click** | Lock mouse cursor and enable 3D camera look (Look Around) |
| **ESC** | Unlock mouse cursor |
| **W, A, S, D** | Planar movement (Forward, Left, Backward, Right) |
| **SPACE / LSHIFT** | Vertical elevation (Up / Down) |