import pytest
from agent_state import GraphTracker
from core_types import AgentState, SemanticState, PerceptionInput, SpatialRelation
from spatial_reasoner import GroundTruthSpatialReasoner
import numpy as np

def test_empty_semantic_state():
    """Verify that an empty state does not crash the tracker and serializes correctly."""
    tracker = GraphTracker()
    
    # 1. Start with an empty state (e.g., looking at a blank wall)
    empty_state = SemanticState(visible_objects=set(), relations=set())
    changed = tracker.update_state(empty_state)
    
    assert changed is True
    assert tracker.current_node_id == 1
    
    # Verify serialization format for empty state
    data = tracker.get_graph_data()
    assert len(data["nodes"]) == 1
    assert data["nodes"][0]["objects"] == []
    assert data["nodes"][0]["relations"] == []
    
    # 2. Transition to a populated state
    populated_state = SemanticState(visible_objects={"A"})
    tracker.update_state(populated_state)
    
    assert tracker.current_node_id == 2
    assert len(tracker.edges) == 1
    assert (1, 2) in tracker.edges or (2, 1) in tracker.edges
    
    # 3. Transition back to the empty state
    changed_back = tracker.update_state(empty_state)
    
    assert changed_back is True
    assert tracker.current_node_id == 1
    assert len(tracker.nodes) == 2  # No new nodes created
    assert len(tracker.edges) == 1  # Reused existing edge

def test_single_object_no_relations():
    """Verify the reasoner gracefully handles a single object without computing pairwise relations."""
    reasoner = GroundTruthSpatialReasoner()
    
    # Provide world context including extensions
    mock_world_objects = {
        "A": {"center": [0, 0, 0], "extensions": [1, 1, 1]},
        "B": {"center": [1, 1, 0], "extensions": [1, 1, 1]}
    }
    dummy_agent = AgentState(position=np.array([0, 0, 0]), yaw=0.0, pitch=0.0)
    inputs = PerceptionInput(world_objects=mock_world_objects, agent_state=dummy_agent)
    
    # Compute relations for only 1 visible object
    state = reasoner.compute_relations(["A"], inputs)
    
    assert state.visible_objects == {"A"}
    assert len(state.relations) == 0  # Cannot have spatial relations with oneself

def test_pipeline_integration_headless():
    """Headless integration test connecting Reasoner -> Tracker -> Serialization."""
    reasoner = GroundTruthSpatialReasoner()
    tracker = GraphTracker()
    
    # 1. Setup mock environment (Ensure extensions are present for AABB calculations)
    mock_world_objects = {
        "Obj_1": {"center": [0, 0, 0], "extensions": [1, 1, 1]},
        "Obj_2": {"center": [2, 0, 0], "extensions": [1, 1, 1]},  # Near Obj_1 (dist = 2.0 < 8.0 threshold)
        "Obj_3": {"center": [50, 50, 50], "extensions": [1, 1, 1]} # Far away
    }
    dummy_agent = AgentState(position=np.array([0, 0, 0]), yaw=0.0, pitch=0.0)
    inputs = PerceptionInput(world_objects=mock_world_objects, agent_state=dummy_agent)
    
    # 2. Execute Reasoner (Mocking perception output: we see Obj_1 and Obj_2)
    perceived_ids = ["Obj_1", "Obj_2"]
    semantic_state = reasoner.compute_relations(perceived_ids, inputs)
    
    # 3. Execute Tracker
    tracker.update_state(semantic_state)
    
    # 4. Verify Pipeline Results via Serialization
    data = tracker.get_graph_data()
    
    assert len(data["nodes"]) == 1
    node = data["nodes"][0]
    
    # Verify objects passed through
    assert set(node["objects"]) == {"Obj_1", "Obj_2"}
    
    # Verify relations were calculated and stored in the graph
    assert "Obj_1_near_Obj_2" in node["relations"]
    assert "Obj_2_near_Obj_1" in node["relations"]
    
def test_topology_divergence_on_relations():
    """Verify nodes correctly diverge if relations change but objects remain identical."""
    tracker = GraphTracker()
    
    # State 1: A and B are near
    state_1 = SemanticState(
        visible_objects={"A", "B"},
        relations={SpatialRelation("A", "near", "B")}
    )
    
    # State 2: A and B are visible, but A is now ON-TOP of B
    state_2 = SemanticState(
        visible_objects={"A", "B"},
        relations={SpatialRelation("A", "on-top", "B")}
    )
    
    tracker.update_state(state_1)
    tracker.update_state(state_2)
    
    # Assert they created two distinct nodes because the relations mutated the frozen key
    assert tracker.current_node_id == 2
    assert len(tracker.nodes) == 2