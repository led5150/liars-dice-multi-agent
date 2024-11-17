"""
Test script for metrics functionality.
"""
from agent import RandomAgent, InformedAgent, AdaptiveAgent, LLMAgent
from metrics import SimulationMetrics, GameMetrics
from environment import Environment

def test_agent_type_naming():
    """Test agent type naming convention"""
    metrics = SimulationMetrics(verbose=2)
    
    # Test each agent type
    agents = [
        RandomAgent("Random1"),
        InformedAgent("Informed1"),
        AdaptiveAgent("Adaptive1"),
        LLMAgent("LLM1")
    ]
    
    # Record agent types
    for agent in agents:
        agent_type = metrics._get_agent_type(agent.__class__.__name__)
        metrics.agent_types[agent.name] = agent_type
        print(f"Agent {agent.name}: class_name={agent.__class__.__name__}, type={agent_type}")
    
    # Verify expected types
    expected_types = {
        "Random1": "Random",
        "Informed1": "Informed",
        "Adaptive1": "Adaptive",
        "LLM1": "LLM"
    }
    
    for name, expected_type in expected_types.items():
        actual_type = metrics.agent_types[name]
        assert actual_type == expected_type, f"Expected {expected_type}, got {actual_type}"
    print("✓ Agent type naming test passed")

def test_metric_collection():
    """Test collection of list-type metrics"""
    metrics = SimulationMetrics(verbose=2)
    
    # Create a test game with some metrics
    game = GameMetrics(
        winner="Test1",
        rounds=5,
        final_lives={"Test1": 2, "Test2": 1},
        elimination_order=["Test2", "Test1"]
    )
    
    # Add some list metrics
    game.dice_remaining_successful_bluff["Test1"] = [5, 4, 3]
    game.predicted_probability["Test1"] = [0.8, 0.6, 0.7]
    game.actual_outcome["Test1"] = [1, 0, 1]
    
    # Record agent types
    metrics.agent_types["Test1"] = "Test"
    metrics.agent_types["Test2"] = "Test"
    
    # Add game and convert to DataFrame
    metrics.add_game(game)
    df = metrics.to_dataframe()
    
    # Verify metrics are present
    test1_data = df[df['agent_name'] == 'Test1'].iloc[0]
    assert len(test1_data['dice_remaining_successful_bluff']) == 3, "List metric not preserved"
    assert len(test1_data['predicted_probability']) == 3, "List metric not preserved"
    assert len(test1_data['actual_outcome']) == 3, "List metric not preserved"
    print("✓ Metric collection test passed")

def main():
    """Run all tests"""
    print("\nRunning metrics tests...")
    test_agent_type_naming()
    test_metric_collection()
    print("\nAll tests passed!")

if __name__ == "__main__":
    main()
