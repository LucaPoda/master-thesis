import pytest
import numpy as np
from spatial_reasoner import GroundTruthSpatialReasoner
from core_types import PerceptionInput, SpatialRelation, AgentState
from agent_state import GraphTracker

@pytest.fixture
def reasoner():
    return GroundTruthSpatialReasoner()

@pytest.fixture
def dummy_agent():
    # Agent sitting at origin, facing North (+Y)
    return AgentState(position=np.array([0.0, 0.0, 0.0]), yaw=0.0, pitch=0.0)

def test_vertical_relations(reasoner, dummy_agent):
    mock_world = {
        "P1": {"center": [0, 0, 1.0], "extensions": [2, 2, 1.0]}, 
        "B1": {"center": [0, 0, 3.0], "extensions": [1, 1, 1.0]}, 
        "A1": {"center": [0, 0, 6.0], "extensions": [1, 1, 1.0]}, 
    }
    inputs = PerceptionInput(world_objects=mock_world, agent_state=dummy_agent)
    state = reasoner.compute_relations(["P1", "B1", "A1"], inputs)
    
    assert SpatialRelation("B1", "on-top", "P1") in state.relations
    assert SpatialRelation("A1", "over", "P1") in state.relations

def test_horizontal_relations(reasoner, dummy_agent):
    mock_world = {
        "C1": {"center": [0, 0, 1], "extensions": [1, 1, 1]},
        "C2": {"center": [5, 0, 1], "extensions": [1, 1, 1]}, # Right of C1 (local)
        "C3": {"center": [0, 5, 1], "extensions": [1, 1, 1]}, # Further away (+Y) -> Behind C1
    }
    inputs = PerceptionInput(world_objects=mock_world, agent_state=dummy_agent)
    state = reasoner.compute_relations(["C1", "C2", "C3"], inputs)
    
    assert SpatialRelation("C1", "left", "C2") in state.relations
    assert SpatialRelation("C2", "right", "C1") in state.relations
    # C1 is at y=0, C3 is at y=5. C1 is closer, so C1 is in front of C3
    assert SpatialRelation("C1", "front", "C3") in state.relations
    assert SpatialRelation("C3", "back", "C1") in state.relations

def test_near_relation(reasoner, dummy_agent):
    mock_world = {
        "N1": {"center": [0, 0, 0], "extensions": [1, 1, 1]},
        "N2": {"center": [4, 0, 0], "extensions": [1, 1, 1]}, 
        "F1": {"center": [50, 0, 0], "extensions": [1, 1, 1]},
    }
    inputs = PerceptionInput(world_objects=mock_world, agent_state=dummy_agent)
    state = reasoner.compute_relations(["N1", "N2", "F1"], inputs)
    
    assert SpatialRelation("N1", "near", "N2") in state.relations
    assert SpatialRelation("N1", "near", "F1") not in state.relations

def test_horizontal_relations_ignored_if_far(reasoner, dummy_agent):
    mock_world = {
        "C1": {"center": [0, 0, 1], "extensions": [1, 1, 1]},
        "C2": {"center": [15, 0, 1], "extensions": [1, 1, 1]}, 
        "C3": {"center": [0, 15, 1], "extensions": [1, 1, 1]}, 
    }
    inputs = PerceptionInput(world_objects=mock_world, agent_state=dummy_agent)
    state = reasoner.compute_relations(["C1", "C2", "C3"], inputs)
    
    assert SpatialRelation("C1", "left", "C2") not in state.relations
    assert SpatialRelation("C3", "front", "C1") not in state.relations

def test_vertical_relations_no_horizontal_overlap(reasoner, dummy_agent):
    mock_world = {
        "P1": {"center": [0, 0, 1.0], "extensions": [1, 1, 1.0]}, 
        "B1": {"center": [5, 5, 3.0], "extensions": [1, 1, 1.0]}  
    }
    inputs = PerceptionInput(world_objects=mock_world, agent_state=dummy_agent)
    state = reasoner.compute_relations(["P1", "B1"], inputs)
    
    assert SpatialRelation("P1", "near", "B1") in state.relations
    assert SpatialRelation("B1", "on-top", "P1") not in state.relations

def test_intersecting_objects(reasoner, dummy_agent):
    mock_world = {
        "Big": {"center": [0, 0, 0], "extensions": [5, 5, 5]},
        "Small": {"center": [0, 0, 0], "extensions": [1, 1, 1]}
    }
    inputs = PerceptionInput(world_objects=mock_world, agent_state=dummy_agent)
    state = reasoner.compute_relations(["Big", "Small"], inputs)
    
    assert SpatialRelation("Big", "near", "Small") in state.relations
    assert SpatialRelation("Small", "on-top", "Big") not in state.relations

def test_negative_coordinates(reasoner, dummy_agent):
    mock_world = {
        "C1": {"center": [-10, -10, 1], "extensions": [1, 1, 1]},
        "C2": {"center": [-15, -10, 1], "extensions": [1, 1, 1]}, 
    }
    inputs = PerceptionInput(world_objects=mock_world, agent_state=dummy_agent)
    state = reasoner.compute_relations(["C1", "C2"], inputs)
    
    assert SpatialRelation("C2", "left", "C1") in state.relations

def test_egocentric_relations_change_with_agent_orientation(reasoner):
    """Verify looking at the same objects from opposite angles generates different nodes."""
    tracker = GraphTracker()
    
    mock_world = {
        "A": {"center": [0, 5, 0], "extensions": [1, 1, 1]},
        "B": {"center": [2, 5, 0], "extensions": [1, 1, 1]},
    }
    
    # Perspective 1: Agent at Origin, Facing North (Y+)
    agent_north = AgentState(position=np.array([0.0, 0.0, 0.0]), yaw=0.0, pitch=0.0)
    inputs_north = PerceptionInput(world_objects=mock_world, agent_state=agent_north)
    
    state_north = reasoner.compute_relations(["A", "B"], inputs_north)
    assert SpatialRelation("A", "left", "B") in state_north.relations
    
    # Perspective 2: Agent moves past objects, Facing South (Y-)
    agent_south = AgentState(position=np.array([0.0, 10.0, 0.0]), yaw=180.0, pitch=0.0)
    inputs_south = PerceptionInput(world_objects=mock_world, agent_state=agent_south)
    
    state_south = reasoner.compute_relations(["A", "B"], inputs_south)
    # What was on the left is now on the right when looking backwards!
    assert SpatialRelation("A", "right", "B") in state_south.relations
    
    # Ensure they map to distinct topological nodes
    tracker.update_state(state_north)
    tracker.update_state(state_south)
    
    assert tracker.current_node_id == 2
    assert len(tracker.nodes) == 2