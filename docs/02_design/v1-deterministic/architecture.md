# Architecture (v1 Deterministic)

The system is decoupled using the **Single Responsibility Principle (SRP)** and **Dependency Inversion**. Components communicate via an untyped generic data buffer, mimicking ROS publisher/subscriber mechanics.

## `PerceptionInput` (Topic Buffer)
Acts as a generic subscriber payload. Concrete perception or reasoning components dynamically pull "topics" (e.g., `agent_state`, `world_objects`) from this buffer. If an expected data field is missing, components fail gracefully with a `ValueError`.

## Core Pipelines
1. **`WorldState`**: Ground-truth definition of the 3D map environment.
2. **`GraphicEngine`**: Panda3D visualizer and physical engine running broad-phase collision detection.
3. **`PerceptionSystem`**: Simulates optical camera vision using 6-plane Frustum Culling algorithms mathematically driven by the agent's Yaw and Pitch.

## Ego-Centric Spatial Reasoning (`spatial_reasoner.py`)
The `GroundTruthSpatialReasoner` performs relational inference in the **Agent's Camera Frame**. Objects are mathematically projected into relative coordinates based on the agent's Yaw. 
*   **Proximity:** Euclidean distance `< 8.0` evaluates to `near`.
*   **Vertical:** `on-top`, `on-bottom`, `over`, `under`.
*   **Directional:** `left`, `right`, `front`, `back`.

## Semantic Graph Tracker (`agent_state.py`)
Graph logic handles topological state transitions entirely independent of rendering. 
*   **Identity Hashing:** State uniqueness is dictated by `SemanticState.to_frozen_key()`.
*   **Undirected Edges:** Graph edges represent contiguous physical exploration.
