# ADR 003: Dual Rendering & Noisy Perception Node

**Date:** 2026-09-16
**Status:** #Accepted

## Context
Moving to the `study/non-deterministic-demo` branch requires testing how noise and hallucinations affect the semantic graph. We need a way to easily compare what the robot "thinks" it sees versus what is actually there, both mathematically and visually.

## Decision
1. Retain the current deterministic ground truth pipeline as a baseline.
2. Implement a new noisy perception node that injects error into both odometry (movement) and bounding box detection (vision).
3. Modify the rendering engine to display perceived objects as solid meshes, and the ground truth objects as transparent "ghosts".

## Consequences
*   **Pros:** Instant visual debugging of perception drift; allows for easy extraction of baseline metrics comparing the noisy output directly against the perfect ground truth.
*   **Cons:** Slight increase in rendering overhead due to drawing objects twice (solid + transparent), though negligible in Panda3D for our current scale.
