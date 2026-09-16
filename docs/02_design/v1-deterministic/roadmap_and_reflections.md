# Roadmap & Architectural Reflections

We have just finalized the first version of the architecture (`v1_deterministic`). This initial phase relies heavily on idealized assumptions: perfect geometric perception ("ground truth") and strictly deterministic state transitions.

However, the entire component-based system—particularly the use of the flexible `PerceptionInput` buffer—has been designed and structured to accommodate future versions that will progressively relax these initial assumptions.

## Towards a Realistic Use Case (study/non-deterministic-demo)

The goal of the upcoming iterations is to bring the simulator closer to the real-world operating conditions of autonomous robotics and computer vision. The main areas of focus will be:

1. **Bayesian Inference & Kalman Filters:** We will replace rigid geometric thresholds with probabilistic models. The system will implement a Bayesian update rule to combine movement knowledge (which has a certain tolerance/error) with visual perception data (which includes the probability of hallucinations).
2. relative Odometry over Absolute Spatial Knowledge: The system does not possess an absolute spatial notion. Instead, we know the 6DOF displacement between one node and another. The reliability of the visual perception must be evaluated against this relative displacement.
3. **Perception Uncertainty:** The `PerceptionSystem` module will evolve into a noisy node to simulate real-world inaccuracies, false positives, and false negatives. 
4. **Active Exploration:** We will shift from manually guided exploration to Active SLAM logic, where the agent must plan its movements to maximize information gain.

### Literature Review Targets
To properly configure the probabilities and errors in the system, a thorough literature review is necessary. 
*   **Keywords:** `semantic slam`, `semantic graph`, `probabilistic semantic slam`, `bayesian state estimation`, `neuro-symbolic perception`.
*   **Target Venues:** ICRA, TRO, RAL.