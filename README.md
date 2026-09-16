# Semantic Scene Graph - Simulator & Visualizer

This project is a simulated system capable of converting an agent's dynamic visual perception into a **Semantic Spatial Graph (Semantic Scene Graph)**. It serves as a study in translating pure geometric 3D vision into topological and relational semantic data.

## 🧪 Local Unit & Integration Test Suite

The project includes an exhaustive, fast, and 100% headless test suite executing via `pytest` without invoking any 3D graphics or Web Servers.

**Test Coverage Areas:**

1. **Core Data Structures:** Hashability, Immutability, `SemanticState` frozen keys.
2. **Graph Topology:** Node creation, idempotency (no duplicate states), cyclic traversal (reusing undirected edges).
3. **Ego-Centric Reasoning:** Negative coordinate boundaries, intersection limits, and perspective shifts (verifying that 180-degree Yaw inversions successfully alter horizontal relations).
4. **Edge Cases:** Headless pipeline integration, falsy inputs, empty map arrays, missing topics.

To execute tests instantly:

```bash
pytest tests/ -v
```

## 📁 Project Structure

The project structure is organized as follows:

```plaintext
.
├── config/
│   ├── config_map.yaml          # Ground-truth: Stacked zones, alignment grids, clusters
│   └── ...                      # Dynamic configurations (controllers, view, sensors)
├── src/
│   ├── agent_state.py           # Pure Graph logic & Undirected edges
│   ├── controllers.py
│   ├── core_types.py            # Interfaces, PerceptionInput, SpatialRelation, SemanticState
│   ├── graphic_engine.py
│   ├── index.html               # Vis.js Web UI with Edge diffs & Node pinning
│   ├── interfaces.py            # Base abstract classes for ROS-like nodes
│   ├── main.py                  # Simulation orchestration and Pipeline IO
│   ├── perception_system.py
│   ├── spatial_reasoner.py      # Ego-centric Transformation Logic
│   ├── visualizer_server.py
│   └── world_state.py
├── tests/                       # Headless Pytest suite
│   ├── test_agent_state.py
│   ├── test_core_types.py
│   ├── test_edge_cases.py
│   ├── test_perception_input.py
│   └── test_spatial_reasoner.py
├── pytest.ini                   # Pytest configuration
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Requirements & Installation

Ensure you have **Python >= 3.10** installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Simulation

Execute the main entry point from the project root directory:

```bash
python src/main.py
```

### 3. Opening the Web Visualizer

Upon starting the simulation, the integrated web server listens on http://127.0.0.1:8000.

Open your browser and navigate to http://127.0.0.1:8000 to view the graph topology expand in real time as you navigate the camera through the 3D window.

## 🎮 Simulation Controls

| **Key / Input** | **Action** |
| --- | --- |
| **Left Mouse Click** | Lock mouse cursor and enable 3D camera look (Look Around) |
| **ESC** | Unlock mouse cursor |
| **W, A, S, D** | Planar movement (Forward, Left, Backward, Right) |
| **SPACE / LSHIFT** | Vertical elevation (Up / Down) |