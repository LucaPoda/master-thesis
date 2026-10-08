# TODO / Next Steps

## 1. Architecture Preparation (Testing & Baselines)
- [x] **Record & Replay Trajectories:** Implement a system to record keyboard movements during a live run and replay them automatically. This avoids the overhead of building a motion planner while ensuring reproducible trajectories for comparative testing.
- [ ] **Fix Automatic Mode:** Fix the automatic mode (`TrajectoryController`) which is currently not working (deferred for now).
- [ ] **Edge Odometry Tracking:** Update the graph edge creation to record the relative 6-axis displacement (position and orientation) between one node and the next. 
- [x] **Noisy Perception Node:** Create a new perception component that introduces artificial error/noise for both object perception and movement tracking, leaving the current ground truth pipeline intact for comparison.
- [x] **Ghost Ground Truth Visualization:** Update the Panda3D engine to render perceived objects as solid blocks, while rendering the actual ground truth objects as transparent "ghosts" (similar to racing game ghosts) for visual debugging.
- [ ] **Diff. Analysis:** When we generate a node in the graph with the noisy perception we check how it should have been using the ground truth, we say the differences and we publish them to the visualizer for showing them in the sidebar.
- [x] **Extract Baseline Metrics:** Run the recorded trajectories with the noisy perception node and extract baseline performance metrics *before* introducing any Bayesian inference.

## 2. Probabilistic Reasoning & Literature
- [x] **Literature Review for Realistic Probabilities:** Search ICRA, TRO, RAL for papers on `semantic slam` and `probabilistic semantic slam`. Extract realistic confidence boundaries for modern CV models and typical drift/error rates for odometry/IMU sensors.
- [x] **Bayesian Update Implementation:** Implement a Kalman filter (or similar Bayesian inference algorithm) to update the state probability by combining the known relative movement with the noisy visual features.

## 3. Egocentric Shift & SLAM Maturation (Current Focus)
- [ ] **Egocentric Perception Refactoring:** Modify the `NoisyPerceptionSystem` to output egocentric coordinates (range and bearing/angles) instead of injecting noise directly onto absolute allocentric coordinates.
- [ ] **EKF Jacobian Implementation:** Upgrade the Extended Kalman Filter to handle the non-linear trigonometric transformations required to map egocentric range/bearing data back into the allocentric world frame.
- [ ] **Process Noise ($Q$) Tuning:** Split the process noise matrix $Q$. Odometry (ego-state) must have a dynamic $Q$ reflecting cumulative drift, while static landmarks must have a near-zero $Q$ to ensure stable convergence.

## 4. Future / Conceptual (#ToReview)
- [ ] **#ToReview Semantic Inference Engine & Global State:** Evaluate wrapping the `SemanticState` in a logic solver to derive implicit relations (transitivity) and maintain a global map of out-of-sight objects based on the initial spawn frame.