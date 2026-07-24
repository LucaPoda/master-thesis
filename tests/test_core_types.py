from core_types import SpatialRelation, SemanticState

def test_spatial_relation_hashability():
    rel1 = SpatialRelation("obj_A", "near", "obj_B")
    rel2 = SpatialRelation("obj_A", "near", "obj_B")
    rel3 = SpatialRelation("obj_B", "near", "obj_A")

    # Test equality
    assert rel1 == rel2
    assert rel1 != rel3

    # Test set hashability
    relation_set = {rel1, rel2, rel3}
    assert len(relation_set) == 2
    assert rel1 in relation_set
    assert rel3 in relation_set

def test_semantic_state_frozen_key():
    rel_A_B = SpatialRelation("obj_A", "near", "obj_B")
    
    # State 1
    state1 = SemanticState(visible_objects={"obj_A", "obj_B"}, relations={rel_A_B})
    
    # State 2 (Same logical state, defined in a different order)
    state2 = SemanticState(visible_objects={"obj_B", "obj_A"}, relations={SpatialRelation("obj_A", "near", "obj_B")})
    
    # The frozen keys must be completely identical
    assert state1.to_frozen_key() == state2.to_frozen_key()