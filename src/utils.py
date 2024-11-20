import numpy as np
from typing import List, Tuple, Dict
import json
import time
from pathlib import Path
import os

class SimulationLogger:
    """Logger for tracking simulation progress and results."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.log_file = self.output_dir / "simulation.log"
        self.start_time = time.time()
    
    def log(self, message: str):
        """Log a message with timestamp."""
        elapsed = time.time() - self.start_time
        with open(self.log_file, 'a') as f:
            f.write(f"[{elapsed:.1f}s] {message}\n")

def ensure_dir(path: Path):
    """Ensure directory exists, creating if necessary."""
    path.mkdir(parents=True, exist_ok=True)

def save_results(results: Dict, filename: str):
    """Save simulation results to JSON."""
    output_path = Path("results") / "data" / filename
    ensure_dir(output_path.parent)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                       for k, v in value.items()}
        else:
            serializable_results[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def load_results(filename: str) -> Dict:
    """Load simulation results from JSON."""
    input_path = Path("results") / "data" / filename
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    # Convert lists back to numpy arrays
    for key, value in results.items():
        if isinstance(value, list):
            results[key] = np.array(value)
        elif isinstance(value, dict):
            results[key] = {k: np.array(v) if isinstance(v, list) else v 
                          for k, v in value.items()}
    
    return results

def create_temperature_range(T_min: float = 2.0, T_max: float = 2.5, 
                           n_points: int = 101) -> np.ndarray:  # Increased points
    """
    Create temperature range focused around Tc.
    Returns temperatures with higher density near Tc = 2.269185.
    """
    Tc = 2.269185  # Exact critical temperature
    
    # Create a non-uniform temperature grid
    # More points near Tc, fewer points far from Tc
    
    # Define regions
    critical_window = 0.01  # Smaller window around Tc
    
    # Create temperature points in different regions
    T_below_far = np.linspace(T_min, Tc - 5*critical_window, n_points//6)
    T_below_near = np.linspace(Tc - 5*critical_window, Tc - critical_window, n_points//4)
    T_critical = np.linspace(Tc - critical_window, Tc + critical_window, n_points//3)
    T_above_near = np.linspace(Tc + critical_window, Tc + 5*critical_window, n_points//4)
    T_above_far = np.linspace(Tc + 5*critical_window, T_max, n_points//6)
    
    # Combine all regions and remove duplicates
    T_all = np.concatenate([
        T_below_far[:-1],
        T_below_near[:-1],
        T_critical,
        T_above_near[1:],
        T_above_far[1:]
    ])
    
    # Add Tc explicitly
    T_all = np.sort(np.unique(np.append(T_all, Tc)))
    
    return T_all

def calculate_error_bars(data: np.ndarray, n_bootstrap: int = 1000) -> Tuple[float, float]:
    """Calculate error bars using bootstrap method."""
    bootstrap_samples = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    return np.percentile(bootstrap_means, [16, 84])