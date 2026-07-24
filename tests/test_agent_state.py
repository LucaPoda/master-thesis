import pytest
import json
from agent_state import GraphTracker
from core_types import SemanticState, SpatialRelation

@pytest.fixture
def tracker():
    return GraphTracker()

def test_initial_state(tracker):
    assert tracker.current_node_id is None
    assert len(tracker.nodes) == 0
    assert len(tracker.edges) == 0

def test_node_creation(tracker):
    state = SemanticState(visible_objects={"A", "B"})
    changed = tracker.update_state(state)
    
    assert changed is True
    assert tracker.current_node_id == 1
    assert len(tracker.nodes) == 1

def test_state_idempotency(tracker):
    # First state
    state = SemanticState(visible_objects={"A"})
    tracker.update_state(state)
    
    # Identical consecutive state update
    changed = tracker.update_state(SemanticState(visible_objects={"A"}))
    
    assert changed is False
    assert tracker.current_node_id == 1
    assert len(tracker.nodes) == 1
    assert len(tracker.edges) == 0

def test_transition_and_loops(tracker):
    state_A = SemanticState(visible_objects={"A"})
    state_B = SemanticState(visible_objects={"A", "B"})
    
    tracker.update_state(state_A) # Enters Node 1
    tracker.update_state(state_B) # Transitions to Node 2
    
    assert tracker.current_node_id == 2
    assert len(tracker.nodes) == 2
    assert len(tracker.edges) == 1
    assert (1, 2) in tracker.edges or (2, 1) in tracker.edges
    
    # Loop back to State A
    tracker.update_state(SemanticState(visible_objects={"A"}))
    
    assert tracker.current_node_id == 1 # Re-visits Node 1
    assert len(tracker.nodes) == 2      # No new node created
    assert len(tracker.edges) == 1      # No new edge created (1, 2) already exists

def test_observer_callbacks(tracker):
    callback_calls = []
    def dummy_callback():
        callback_calls.append("called")
        
    tracker.add_callback(dummy_callback)
    
    tracker.update_state(SemanticState(visible_objects={"A"}))
    assert len(callback_calls) == 1
    
    tracker.update_state(SemanticState(visible_objects={"A"})) # Idempotent update
    assert len(callback_calls) == 1 # Should not fire callback
    
    tracker.update_state(SemanticState(visible_objects={"B"})) # Transition
    assert len(callback_calls) == 2

def test_serialization(tracker):
    rel = SpatialRelation("A", "near", "B")
    state = SemanticState(visible_objects={"A", "B"}, relations={rel})
    tracker.update_state(state)
    
    data = tracker.get_graph_data()
    
    # Validate structure
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 1
    
    node = data["nodes"][0]
    assert node["id"] == 1
    assert set(node["objects"]) == {"A", "B"}
    assert "A_near_B" in node["relations"]
    
    # Ensure it is purely JSON serializable
    json_str = json.dumps(data)
    assert "A_near_B" in json_str