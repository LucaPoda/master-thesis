# agent_state.py
from typing import Callable, Set, Dict, Tuple, FrozenSet

class GraphTracker:
    def __init__(self):
        # Maps the unique ID of a node to its set of objects (frozenset for hashing/comparison)
        self.nodes: Dict[int, FrozenSet[str]] = {}
        # Set of undirected edges, represented as sorted tuples (id1, id2)
        self.edges: Set[Tuple[int, int]] = set()
        
        self.current_node_id: int | None = None
        self.node_counter: int = 0
        self.callbacks: list[Callable[[], None]] = []

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Register a function to be called when the graph changes."""
        self.callbacks.append(callback)

    def _notify(self) -> None:
        for cb in self.callbacks:
            cb()

    def update_state(self, visible_objects: list[str]) -> bool:
        """
        Update the graph based on the currently visible objects.
        Returns True if the graph/state has changed, False otherwise.
        """
        state_set = frozenset(visible_objects)
        
        # 1. If the state is identical to the current node, ignore
        if self.current_node_id is not None and self.nodes[self.current_node_id] == state_set:
            return False
            
        # 2. Search for an existing node with this exact set
        target_id = None
        for nid, objs in self.nodes.items():
            if objs == state_set:
                target_id = nid
                break
                
        # 3. If it doesn't exist, create a new node
        if target_id is None:
            self.node_counter += 1
            target_id = self.node_counter
            self.nodes[target_id] = state_set
            
        # 4. Management of edges (create link with previous node)
        if self.current_node_id is not None and self.current_node_id != target_id:
            # Sort the tuple to ensure that the edge (A, B) is equal to (B, A)
            edge = tuple(sorted((self.current_node_id, target_id)))
            self.edges.add(edge)
            
        # 5. Update the current node and notify
        self.current_node_id = target_id
        self._notify()
        
        return True

    def get_graph_data(self) -> dict:
        """Export the graph in a dictionary format for serialization."""
        return {
            "nodes": [
                {
                    "id": nid,
                    "label": f"State {nid}",
                    "objects": list(objs),
                    "is_current": (nid == self.current_node_id)
                }
                for nid, objs in self.nodes.items()
            ],
            "edges": [
                {"from": e[0], "to": e[1], "id": f"{e[0]}-{e[1]}"} 
                for e in self.edges
            ]
        }