import pytest
from spatial_reasoner import GroundTruthSpatialReasoner, CVSpatialReasoner
from core_types import PerceptionInput, SpatialRelation

def test_ground_truth_reasoner():
    reasoner = GroundTruthSpatialReasoner()
    
    # Mock mathematical map context
    mock_world_objects = {
        "A": {"center": [0, 0, 0]},
        "B": {"center": [1, 1, 0]},   # Near A (distance ~1.4)
        "C": {"center": [10, 10, 0]}  # Far away from A and B
    }
    
    inputs = PerceptionInput(world_objects=mock_world_objects)
    
    # Run reasoning
    semantic_state = reasoner.compute_relations(["A", "B", "C"], inputs)
    
    assert "A" in semantic_state.visible_objects
    
    # Verify proximity relations
    assert SpatialRelation("A", "near", "B") in semantic_state.relations
    assert SpatialRelation("B", "near", "A") in semantic_state.relations
    
    # Verify "C" is correctly excluded from relations
    assert SpatialRelation("A", "near", "C") not in semantic_state.relations

def test_cv_reasoner_missing_input():
    reasoner = CVSpatialReasoner()
    
    # Empty mock buffer (should not crash right now as it's a skeleton, 
    # but tests that it safely returns the objects at a minimum)
    inputs = PerceptionInput()
    semantic_state = reasoner.compute_relations(["A"], inputs)
    
    assert "A" in semantic_state.visible_objects
    assert len(semantic_state.relations) == 0