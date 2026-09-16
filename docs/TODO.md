# TODO / Next Steps

## 1. Architecture Preparation (Testing & Baselines)
- [ ] **Record & Replay Trajectories:** Implement a system to record keyboard movements during a live run and replay them automatically. This avoids the overhead of building a motion planner while ensuring reproducible trajectories for comparative testing.
- [ ] **Edge Odometry Tracking:** Update the graph edge creation to record the relative 6-axis displacement (position and orientation) between one node and the next. 
- [ ] **Noisy Perception Node:** Create a new perception component that introduces artificial error/noise for both object perception and movement tracking, leaving the current ground truth pipeline intact for comparison.
- [ ] **Ghost Ground Truth Visualization:** Update the Panda3D engine to render perceived objects as solid blocks, while rendering the actual ground truth objects as transparent "ghosts" (similar to racing game ghosts) for visual debugging.
- [ ] **Extract Baseline Metrics:** Run the recorded trajectories with the noisy perception node and extract baseline performance metrics *before* introducing any Bayesian inference.

## 2. Probabilistic Reasoning & Literature
- [ ] **Literature Review for Realistic Probabilities:** Search ICRA, TRO, RAL for papers on `semantic slam` and `probabilistic semantic slam`. Extract realistic confidence boundaries for modern CV models and typical drift/error rates for odometry/IMU sensors.
- [ ] **Bayesian Update Implementation:** Implement a Kalman filter (or similar Bayesian inference algorithm) to update the state probability by combining the known relative movement with the noisy visual features.

## 3. Future / Conceptual (#ToReview)
- [ ] **#ToReview Semantic Inference Engine & Global State:** Evaluate wrapping the `SemanticState` in a logic solver to derive implicit relations (transitivity) and maintain a global map of out-of-sight objects based on the initial spawn frame.