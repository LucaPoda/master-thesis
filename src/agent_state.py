from typing import Callable, Dict, Tuple, FrozenSet, Any
from core_types import SemanticState

class GraphTracker:
    def __init__(self):
        # Maps unique node IDs to their frozen semantic key
        self.nodes: Dict[int, frozenset] = {}
        # Stores the raw SemanticState for JSON serialization
        self.node_states: Dict[int, SemanticState] = {}
        self.edges: set[Tuple[int, int]] = set()
        
        self.current_node_id: int | None = None
        self.node_counter: int = 0
        self.callbacks: list[Callable[[], None]] = []

    def add_callback(self, callback: Callable[[], None]) -> None:
        self.callbacks.append(callback)

    def _notify(self) -> None:
        for cb in self.callbacks:
            cb()

    def update_state(self, state: SemanticState) -> bool:
        """
        Update the graph based on the currently visible objects.
        Returns True if the graph/state has changed, False otherwise.
        """
        state_key = state.to_frozen_key()
        
        # 1. If the state is identical to the current node, ignore
        if self.current_node_id is not None and self.nodes[self.current_node_id] == state_key:
            return False

        # 2. Search for an existing node with this exact set
        target_id = None
        for nid, key in self.nodes.items():
            if key == state_key:
                target_id = nid
                break

        # 3. If it doesn't exist, create a new node
        if target_id is None:
            self.node_counter += 1
            target_id = self.node_counter
            self.nodes[target_id] = state_key
            self.node_states[target_id] = state
            
        # 4. Management of edges (create link with previous node)
        if self.current_node_id is not None and self.current_node_id != target_id:
            # Sort the tuple to ensure that the edge (A, B) is equal to (B, A) (Undirected)
            edge = tuple(sorted((self.current_node_id, target_id)))
            self.edges.add(edge)
            
        # 5. Update the current node and notify
        self.current_node_id = target_id
        self._notify()

        return True

    def get_graph_data(self) -> dict:
        return {
            "nodes": [
                {
                    "id": nid,
                    "label": f"State {nid}",
                    "objects": list(self.node_states[nid].visible_objects),
                    "relations": [f"{r.subject}_{r.relation}_{r.target}" for r in self.node_states[nid].relations],
                    "is_current": (nid == self.current_node_id)
                }
                for nid in self.nodes.keys()
            ],
            "edges": [
                {"from": e[0], "to": e[1], "id": f"{e[0]}-{e[1]}"} 
                for e in self.edges
            ]
        }