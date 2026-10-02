# Glossary

*   **Semantic Scene Graph**: A topological map where nodes represent distinct semantic states (what is visible and how objects relate to each other) rather than specific metric coordinates.
*   **SemanticState**: A hashable, immutable snapshot combining a set of currently visible object IDs and their `SpatialRelation`s.
*   **Ego-Centric Frame**: A coordinate system relative to the agent's current position and orientation (Yaw/Pitch), as opposed to an Allocentric (world-fixed) frame.
*   **Frustum Culling**: A mathematical 3D rendering and perception technique. It uses 6 bounding planes (Near, Far, Left, Right, Top, Bottom) extending from the camera to determine exactly which objects intersect the agent's field of view.
*   **PerceptionInput**: An untyped, flexible data buffer used in this architecture to mimic ROS (Robot Operating System) publisher/subscriber mechanics.
*   **Procedural Map Generation**: The algorithmic creation of structured 3D environments (rooms, walls, furniture) with varying density and complexity to stress-test spatial reasoning algorithms without relying on hardcoded layouts.
*   **SimulationCoordinator**: The root application composer responsible for dependency injection, configuration loading, and orchestrating the execution order of the simulation loop.
*   **Noisy Perception**: Simulated visual data that mimics real-world sensor imperfections, incorporating update frequency caps, distance-based quadratic spatial deviations (Gaussian noise), and false negatives based on confidence thresholds.