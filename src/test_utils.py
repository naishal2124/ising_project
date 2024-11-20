import numpy as np
from utils import *
import os
import tempfile
import shutil

def test_temperature_range():
    """Detailed test of temperature range generation."""
    print("\nTest 4: Temperature Range Generation")
    T_range = create_temperature_range()
    Tc = 2.269185
    
    # Basic tests
    assert len(T_range) > 0, "Temperature range should not be empty"
    assert np.all(np.diff(T_range) > 0), "Temperatures should be strictly increasing"
    assert abs(T_range[0] - 2.0) < 1e-10, "Should start at T_min"
    assert abs(T_range[-1] - 2.5) < 1e-10, "Should end at T_max"
    
    # Test critical region
    critical_window = 0.01
    critical_points = T_range[np.abs(T_range - Tc) < critical_window]
    n_critical = len(critical_points)
    
    # Count points in regions of equal size away from Tc
    def count_points_in_window(center: float, width: float) -> int:
        return np.sum((T_range >= center - width/2) & (T_range <= center + width/2))
    
    n_below = count_points_in_window(Tc - 0.1, critical_window)
    n_above = count_points_in_window(Tc + 0.1, critical_window)
    
    # Print detailed information
    print(f"\nTemperature range analysis:")
    print(f"Total points: {len(T_range)}")
    print(f"Points within {critical_window} of Tc: {n_critical}")
    print(f"Points in equal window below Tc-0.1: {n_below}")
    print(f"Points in equal window above Tc+0.1: {n_above}")
    print(f"Temperature points near Tc: {critical_points}")
    
    # Assertions
    assert n_critical >= 3, f"Should have at least 3 points near Tc, got {n_critical}"
    assert n_critical > n_below, "Should have more points near Tc than away from it"
    assert n_critical > n_above, "Should have more points near Tc than away from it"
    assert Tc in T_range, "Tc should be explicitly included"
    
    print("Temperature range test passed")
    
def test_utils():
    """Run validation tests for utility functions."""
    print("Testing utility functions...")
    
    # Test 1: SimulationLogger
    print("\nTest 1: SimulationLogger")
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SimulationLogger(tmpdir)
        logger.log("Test message")
        logger.log("Another test")
        
        with open(logger.log_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert "Test message" in lines[0]
            assert "Another test" in lines[1]
    print("Logger test passed")
    
    # Test 2: Save and Load Results
    print("\nTest 2: Save/Load Results")
    test_data = {
        'array': np.array([1, 2, 3]),
        'nested': {
            'array': np.array([4, 5, 6]),
            'value': 42
        },
        'value': 3.14
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'data'))
        save_path = os.path.join(tmpdir, 'data', 'test.json')
        
        save_results(test_data, 'test.json')
        loaded_data = load_results('test.json')
        
        np.testing.assert_array_equal(test_data['array'], loaded_data['array'])
        np.testing.assert_array_equal(test_data['nested']['array'], 
                                    loaded_data['nested']['array'])
        assert test_data['nested']['value'] == loaded_data['nested']['value']
        assert test_data['value'] == loaded_data['value']
    print("Save/Load test passed")
    
    # Test 3: Error Bar Calculation
    print("\nTest 3: Error Bar Calculation")
    true_mean = 1.0
    true_std = 0.1
    n_samples = 1000
    data = np.random.normal(true_mean, true_std, n_samples)
    
    lower, upper = calculate_error_bars(data)
    error_range = upper - lower
    expected_range = 2 * true_std / np.sqrt(n_samples)
    assert abs(error_range - expected_range) < 0.1 * expected_range
    print("Error bar test passed")
    
    # Test 4: Temperature Range
    test_temperature_range()

if __name__ == "__main__":
    test_utils()