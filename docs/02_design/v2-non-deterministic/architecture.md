# Architecture (v2 Procedural & Probabilistic)

The system is decoupled using the **Single Responsibility Principle (SRP)** and **Dependency Inversion**. Components communicate via an untyped generic data buffer, mimicking ROS publisher/subscriber mechanics.

## `SimulationCoordinator`
The centralized root composer. It handles dependency injection, loading YAML configurations dynamically, and orchestrating the sequential execution of physics, perception, and mapping tasks during each simulation tick.

## `PerceptionInput` (Topic Buffer)
Acts as a generic subscriber payload populated by the coordinator. Concrete perception or reasoning components dynamically pull "topics" (e.g., `agent_state`, `world_objects`) from this buffer. 

## Core Pipelines
1. **`MapGenerator`**: Procedurally generates environments from random scatterings (Level 0) up to complex, multi-room structures with hung objects and furniture (Level 3).
2. **`WorldState`**: Ground-truth definition of the 3D map environment and physics tracking.
3. **`Controllers`**: Handles execution of movement commands through decoupled modules: `KeyboardController` (manual), `RecordingKeyboardController` (manual + serialization), `ReplayController` (deterministic playback), and `TrajectoryController` (automatic waypoints).
4. **`GraphicEngine`**: Panda3D visualizer running broad-phase collision detection.
5. **`PerceptionSystem`**: 
   *   *Frustum Base*: Simulates optical camera vision using 6-plane Frustum Culling.
   *   *Noisy Implementation*: Mimics real-world hardware by caching data to simulate lower update frequencies, injecting distance-scaled quadratic Gaussian noise, and deliberately dropping objects below confidence thresholds.
   *   *OdometrySystem*: Simulates internal IMU/encoder drift to generate the agent's believed state independently from the kinematic controllers.
6. **`TelemetryLogger`**: Asynchronously logs internal state vs. ground truth to disk at a fixed frequency for post-run analysis.

## Ego-Centric Spatial Reasoning (`spatial_reasoner.py`)
Currently performs relational inference in the **Agent's Camera Frame**. Objects are projected into relative coordinates based on the agent's Yaw.
*   **Proximity:** Evaluates euclidean distance.
*   **Vertical:** `on-top`, `on-bottom`, `over`, `under`.
*   **Directional:** `left`, `right`, `front`, `back`.

## Semantic Graph Tracker (`agent_state.py`)
Graph logic handles topological state transitions. 
*   **Identity Hashing:** State uniqueness is dictated by `SemanticState.to_frozen_key()`.
*   **Undirected Edges:** Graph edges represent contiguous physical exploration.