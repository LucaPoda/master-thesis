# Roadmap & Architectural Reflections

We have successfully transitioned the architecture from its initial deterministic state (`v1`) to a more robust, probabilistic setup (`v2`). 

## Recent Achievements
1. **Procedural Generation:** We moved away from static configurations. The `MapGenerator` allows us to instantly create and persist dynamic environments, ensuring our topological reasoning handles edge cases and unpredictable clutter.
2. **Perception Uncertainty:** The `NoisyPerceptionSystem` successfully mimics real-world hardware limitations. By introducing distance-scaled Gaussian noise, artificial delays (update frequencies), and false negatives via confidence dropouts, we've broken the "perfect vision" assumption.
3. **Centralized Coordination:** The `SimulationCoordinator` handles dependency injection, making it trivial to swap between idealized components and realistic, noisy equivalents for A/B testing.

## Next Steps: Handling Ambiguity

With noisy perception actively causing visual chattering and spontaneous object disappearance, our semantic mapping logic will break if it expects perfect continuity. The main areas of focus moving forward are:

1. **Robust Temporal Tracking:** The `GraphTracker` must be updated to handle false negatives. If an object drops below the confidence threshold for a split second, the graph should not instantly snap to an entirely new semantic node. We need to introduce memory buffers or hysteresis logic.
2. **Bayesian State Updates:** Implement a Kalman filter or similar statistical framework. The spatial reasoner needs to update state probabilities by combining the agent's known relative movement (odometry) with the newly fluctuating visual coordinates.
3. **Active Exploration (SLAM):** Shift from manually guided (keyboard) exploration to autonomous logic. The agent should evaluate the semantic graph for uncertainties and actively plan paths to maximize visual information gain.