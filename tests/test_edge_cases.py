import pytest
from agent_state import GraphTracker
from core_types import SemanticState, PerceptionInput
from spatial_reasoner import GroundTruthSpatialReasoner

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
    
    # Provide world context, but we will only pass one object as "visible"
    mock_world_objects = {
        "A": {"center": [0, 0, 0]},
        "B": {"center": [1, 1, 0]}
    }
    inputs = PerceptionInput(world_objects=mock_world_objects)
    
    # Compute relations for only 1 visible object
    state = reasoner.compute_relations(["A"], inputs)
    
    assert state.visible_objects == {"A"}
    assert len(state.relations) == 0  # Cannot have spatial relations with oneself

def test_pipeline_integration_headless():
    """Headless integration test connecting Reasoner -> Tracker -> Serialization."""
    reasoner = GroundTruthSpatialReasoner()
    tracker = GraphTracker()
    
    # 1. Setup mock environment
    mock_world_objects = {
        "Obj_1": {"center": [0, 0, 0]},
        "Obj_2": {"center": [2, 0, 0]},  # Near Obj_1 (dist = 2.0 < 4.0 threshold)
        "Obj_3": {"center": [50, 50, 50]} # Far away
    }
    inputs = PerceptionInput(world_objects=mock_world_objects)
    
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