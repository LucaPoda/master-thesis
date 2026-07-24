import pytest
from core_types import PerceptionInput

def test_perception_input_valid():
    # Test valid retrieval
    inputs = PerceptionInput(rgb_frame="fake_image_data", collision_queue=[])
    assert inputs.get("rgb_frame") == "fake_image_data"
    assert inputs.get("collision_queue") == []

def test_perception_input_missing():
    # Test graceful failure matching ROS unpopulated topics
    inputs = PerceptionInput(agent_state="state")
    
    with pytest.raises(ValueError, match="Missing required input field/topic: 'world_objects'"):
        inputs.get("world_objects")
        
def test_perception_input_none():
    # Test when a topic is explicitly None
    inputs = PerceptionInput(world_objects=None)
    
    with pytest.raises(ValueError, match="Missing required input field/topic: 'world_objects'"):
        inputs.get("world_objects")