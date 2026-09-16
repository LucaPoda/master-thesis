# ADR 001: Initial Python Simulation Setup & Custom ROS-like Buffer

**Date:** 2026-09-16
**Status:** Accepted

## Context
We need to rapidly prototype and prove the mathematical foundations of ego-centric spatial reasoning and semantic state hashing. Building this directly in C++ with native ROS 2 would introduce massive overhead regarding message compilation, hardware integration, and visualization boilerplate, slowing down the theoretical validation.

## Decision
We decided to build the `v1_deterministic` prototype entirely in Python using Panda3D for rendering and ground-truth physics. 
Furthermore, instead of using a rigid pipeline, we implemented `PerceptionInput`, an untyped flexible buffer that mimics ROS publisher/subscriber mechanics. Components pull data dynamically as if subscribing to topics.

## Consequences
*   **Pros:** Immediate visual feedback via Panda3D; fast iteration on the `SemanticState` hashing logic; allows easy bridging to a WebSocket FastAPI server for real-time Vis.js graph rendering.
*   **Cons:** Python's Global Interpreter Lock (GIL) and untyped buffers will inevitably become a performance bottleneck. This architecture is disposable: once the logic is proven, it will necessitate a full rewrite in C++ for actual ROS 2 deployment.
