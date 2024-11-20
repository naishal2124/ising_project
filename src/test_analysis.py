import numpy as np
from ising import IsingModel
from cluster import ClusterUpdater
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from typing import Dict, List

def test_binder_cumulant():
    """Test Binder cumulant calculation near Tc."""
    print("\nTest 1: Binder Cumulant Analysis")
    L_values = [16, 32]
    T_values = np.linspace(2.0, 2.5, 10)
    
    plt.figure(figsize=(10, 6))
    for L in L_values:
        U4_values = []
        for T in tqdm(T_values, desc=f"L={L}"):
            model = IsingModel(L, T)
            updater = ClusterUpdater(model)
            
            # Equilibrate
            for _ in range(100):
                updater.wolff_update()
            
            # Measure
            m2_sum = 0
            m4_sum = 0
            n_measurements = 1000
            
            for _ in range(n_measurements):
                updater.wolff_update()
                m = model.magnetization / model.N
                m2 = m * m
                m2_sum += m2
                m4_sum += m2 * m2
            
            m2_avg = m2_sum / n_measurements
            m4_avg = m4_sum / n_measurements
            U4 = m4_avg / (m2_avg * m2_avg)
            U4_values.append(U4)
        
        plt.plot(T_values, U4_values, 'o-', label=f'L={L}')
    
    plt.axvline(2.269, color='r', linestyle='--', label='Exact Tc')
    plt.xlabel('Temperature T/J')
    plt.ylabel('Binder Cumulant U₄')
    plt.title('Binder Cumulant vs Temperature')
    plt.legend()
    plt.show()

def test_algorithm_comparison():
    """Compare Wolff and Swendsen-Wang algorithms."""
    print("\nTest 2: Algorithm Comparison")
    L = 32
    T = 2.27
    model = IsingModel(L, T)
    updater = ClusterUpdater(model)
    
    # Test both algorithms
    n_updates = 1000
    
    # Wolff
    print("\nTesting Wolff algorithm:")
    t0 = time.time()
    cluster_sizes_wolff = []
    for _ in tqdm(range(n_updates)):
        size = updater.wolff_update()
        cluster_sizes_wolff.append(size)
    time_wolff = time.time() - t0
    
    # Reset system
    model = IsingModel(L, T)
    updater = ClusterUpdater(model)
    
    # Swendsen-Wang
    print("\nTesting Swendsen-Wang algorithm:")
    t0 = time.time()
    cluster_sizes_sw = []
    for _ in tqdm(range(n_updates)):
        sizes = updater.swendsen_wang_update()
        cluster_sizes_sw.extend(sizes)
    time_sw = time.time() - t0
    
    # Plot cluster size distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.hist(cluster_sizes_wolff, bins=50, density=True, alpha=0.7, label='Wolff')
    plt.xlabel('Cluster Size')
    plt.ylabel('Probability Density')
    plt.title('Wolff Cluster Size Distribution')
    plt.legend()
    
    plt.subplot(122)
    plt.hist(cluster_sizes_sw, bins=50, density=True, alpha=0.7, label='SW')
    plt.xlabel('Cluster Size')
    plt.ylabel('Probability Density')
    plt.title('Swendsen-Wang Cluster Size Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nWolff time: {time_wolff:.2f}s")
    print(f"SW time: {time_sw:.2f}s")
    print(f"\nAverage cluster sizes:")
    print(f"Wolff: {np.mean(cluster_sizes_wolff):.1f}")
    print(f"SW: {np.mean(cluster_sizes_sw):.1f}")

def test_error_analysis():
    """Test error estimation methods."""
    print("\nTest 3: Error Analysis")
    
    # Generate synthetic data with known properties
    true_mean = 1.0
    true_std = 0.1
    n_samples = 1000
    data = np.random.normal(true_mean, true_std, n_samples)
    
    # Bootstrap error estimation
    n_bootstrap = 1000
    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(sample)
    
    bootstrap_error = np.std(bootstrap_means)
    
    print(f"True standard error: {true_std/np.sqrt(n_samples):.6f}")
    print(f"Bootstrap error estimate: {bootstrap_error:.6f}")

if __name__ == "__main__":
    test_binder_cumulant()
    test_algorithm_comparison()
    test_error_analysis()