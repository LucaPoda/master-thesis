# Roadmap & Architectural Reflections
We have just finalized the first version of the architecture (`v1_deterministic`). This initial phase relies heavily on idealized assumptions: perfect geometric perception ("ground truth") and strictly deterministic state transitions.

However, the entire component-based system—particularly the use of the flexible `PerceptionInput` buffer—has been designed and structured to accommodate future versions that will progressively relax these initial assumptions.

## Towards a Realistic Use Case

The goal of upcoming iterations is to bring the simulator closer to the real-world operating conditions of autonomous robotics and computer vision. The main areas of focus will be:

1. **Moving Beyond Pure Determinism:** We will replace rigid geometric thresholds (e.g., "A is exactly to the left of B") with probabilistic models and confidence intervals. This will allow for the handling of visual noise, camera micro-movements (chattering), and spatial ambiguity.
2. **Perception Uncertainty:** The `PerceptionSystem` module—currently based on infallible frustum culling against map data—will evolve to simulate noisy sensors, false positives/negatives, or integration with real visual inference models (e.g., YOLO/VLM).
3. **Active Exploration:** We will shift from manually guided exploration to Active SLAM logic, where the agent must plan its movements to maximize information gain and resolve ambiguities within the semantic graph.