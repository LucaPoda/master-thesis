# Project Goals & Context

The primary goal of this project is to bridge the gap between pure geometric 3D vision and topological/relational semantic data. 

Traditional spatial mapping (like metric SLAM) excels at creating coordinate grids, but lacks human-like conceptual understanding of space. This project aims to convert an agent's dynamic visual perception into a **Semantic Spatial Graph (Semantic Scene Graph)**.

By evaluating the environment ego-centrically (from the agent's perspective), we can map objects and their spatial relations, effectively translating raw pixels/bounding boxes into logical predicates (e.g., "Object A is left of Object B").
