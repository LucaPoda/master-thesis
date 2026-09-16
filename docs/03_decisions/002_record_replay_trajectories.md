# ADR 002: Record and Replay for Automated Trajectories

**Date:** 2026-09-16
**Status:** #Accepted

## Context
To test the upcoming probabilistic perception models, the system requires an agent to follow precise, repeatable trajectories. Implementing a full motion planning and navigation stack right now would be counterproductive and shift focus away from the core goal of semantic inference.

## Decision
Implement a "Record & Replay" controller. The system will record keyboard-driven movements during an initial run and save them. Subsequent runs can use this saved data to drive the agent automatically along the exact same path.

## Consequences
*   **Pros:** Extremely fast to implement; guarantees 100% reproducible test scenarios for comparing ground truth against noisy perception; keeps the focus on perception rather than control.
*   **Cons:** Trajectories are hardcoded to the specific map layout and cannot react dynamically to new obstacles.
