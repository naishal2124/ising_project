from src.ising import IsingModel
import numpy as np
import time
from typing import List, Dict, Tuple, Set
from collections import deque

class ClusterUpdater:
    """
    Implements both Wolff and Swendsen-Wang cluster updates for the 2D Ising model.
    """
    
    def __init__(self, model: IsingModel):
        self.model = model
    
    def wolff_update(self) -> int:
        """
        Perform one Wolff cluster update.
        Returns: Size of flipped cluster
        """
        L = self.model.L
        # Pick random starting site
        i, j = np.random.randint(0, L, 2)
        initial_spin = self.model.spins[i, j]
        
        # Use queue for stack-free implementation (avoids recursion limits)
        to_check = deque([(i, j)])
        cluster = {(i, j)}
        
        while to_check:
            current_i, current_j = to_check.popleft()
            
            # Check all neighbors
            for ni, nj in self.model.get_neighbor_spins(current_i, current_j):
                if ((ni, nj) not in cluster and 
                    self.model.spins[ni, nj] == initial_spin and
                    np.random.random() < self.model.p_add):
                    
                    cluster.add((ni, nj))
                    to_check.append((ni, nj))
        
        # Flip cluster and update observables
        dM = 0  # Change in magnetization
        dE = 0  # Change in energy
        
        for i, j in cluster:
            # Update energy: count bonds between cluster and non-cluster sites
            neighbors = self.model.get_neighbor_spins(i, j)
            for ni, nj in neighbors:
                if (ni, nj) not in cluster:
                    dE += 2 * self.model.spins[i, j] * self.model.spins[ni, nj]
            
            # Flip spin
            self.model.spins[i, j] *= -1
            dM -= 2 * initial_spin
        
        # Update model observables
        self.model.energy += dE
        self.model.magnetization += dM
        
        return len(cluster)
    
    def swendsen_wang_update(self) -> List[int]:
        """
        Perform one Swendsen-Wang update.
        Returns: List of cluster sizes
        """
        L = self.model.L
        labels = np.arange(L * L).reshape(L, L)  # Initial labels
        clusters = {}  # cluster_label -> set of sites
        
        # First pass: create bonds with probability p_add
        for i in range(L):
            for j in range(L):
                current_spin = self.model.spins[i, j]
                current_label = labels[i, j]
                
                # Check right neighbor
                ni, nj = self.model.get_neighbor_spins(i, j)[0]
                if (self.model.spins[ni, nj] == current_spin and 
                    np.random.random() < self.model.p_add):
                    # Merge clusters
                    new_label = min(current_label, labels[ni, nj])
                    old_label = max(current_label, labels[ni, nj])
                    labels[labels == old_label] = new_label
                
                # Check down neighbor
                ni, nj = self.model.get_neighbor_spins(i, j)[2]
                if (self.model.spins[ni, nj] == current_spin and 
                    np.random.random() < self.model.p_add):
                    # Merge clusters
                    new_label = min(current_label, labels[ni, nj])
                    old_label = max(current_label, labels[ni, nj])
                    labels[labels == old_label] = new_label
        
        # Identify clusters
        for i in range(L):
            for j in range(L):
                label = labels[i, j]
                if label not in clusters:
                    clusters[label] = set()
                clusters[label].add((i, j))
        
        # Flip clusters randomly
        dM = 0  # Change in magnetization
        dE = 0  # Change in energy
        cluster_sizes = []
        
        for sites in clusters.values():
            cluster_sizes.append(len(sites))
            if np.random.random() < 0.5:  # Flip with 50% probability
                # Calculate energy change
                for i, j in sites:
                    neighbors = self.model.get_neighbor_spins(i, j)
                    for ni, nj in neighbors:
                        if (ni, nj) not in sites:
                            dE += 2 * self.model.spins[i, j] * self.model.spins[ni, nj]
                    
                    # Flip spin
                    dM -= 2 * self.model.spins[i, j]
                    self.model.spins[i, j] *= -1
        
        # Update model observables
        self.model.energy += dE
        self.model.magnetization += dM
        
        return cluster_sizes

def compare_algorithms(L: int, T: float, n_updates: int = 1000, 
                      equilibration: int = 100) -> Dict:
    """
    Compare Wolff and Swendsen-Wang algorithms.
    """
    # Initialize model
    model = IsingModel(L, T)
    updater = ClusterUpdater(model)
    
    results = {
        'wolff': {'time': 0, 'cluster_sizes': []},
        'sw': {'time': 0, 'cluster_sizes': []}
    }
    
    # Test Wolff algorithm
    model_wolff = IsingModel(L, T)
    updater_wolff = ClusterUpdater(model_wolff)
    
    # Equilibration
    for _ in range(equilibration):
        updater_wolff.wolff_update()
    
    # Measurements
    t0 = time.time()
    for _ in range(n_updates):
        size = updater_wolff.wolff_update()
        results['wolff']['cluster_sizes'].append(size)
    results['wolff']['time'] = time.time() - t0
    results['wolff']['final_energy'] = model_wolff.energy/model_wolff.N
    results['wolff']['final_mag'] = abs(model_wolff.magnetization)/model_wolff.N
    
    # Test Swendsen-Wang algorithm
    model_sw = IsingModel(L, T)
    updater_sw = ClusterUpdater(model_sw)
    
    # Equilibration
    for _ in range(equilibration):
        updater_sw.swendsen_wang_update()
    
    # Measurements
    t0 = time.time()
    for _ in range(n_updates):
        sizes = updater_sw.swendsen_wang_update()
        results['sw']['cluster_sizes'].extend(sizes)
    results['sw']['time'] = time.time() - t0
    results['sw']['final_energy'] = model_sw.energy/model_sw.N
    results['sw']['final_mag'] = abs(model_sw.magnetization)/model_sw.N
    
    return results

if __name__ == "__main__":
    # Test both algorithms
    L = 32  # Larger system for meaningful comparison
    T = 2.27  # Near critical temperature
    
    results = compare_algorithms(L, T, n_updates=1000, equilibration=100)
    
    print(f"\nResults for {L}x{L} lattice at T={T}:")
    print("\nWolff algorithm:")
    print(f"Time: {results['wolff']['time']:.3f}s")
    print(f"Average cluster size: {np.mean(results['wolff']['cluster_sizes']):.1f}")
    print(f"Final energy per spin: {results['wolff']['final_energy']:.6f}")
    print(f"Final |m| per spin: {results['wolff']['final_mag']:.6f}")
    
    print("\nSwendsen-Wang algorithm:")
    print(f"Time: {results['sw']['time']:.3f}s")
    print(f"Average cluster size: {np.mean(results['sw']['cluster_sizes']):.1f}")
    print(f"Final energy per spin: {results['sw']['final_energy']:.6f}")
    print(f"Final |m| per spin: {results['sw']['final_mag']:.6f}")