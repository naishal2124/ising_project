import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from ising import IsingModel
from cluster import ClusterUpdater
import time
from typing import Tuple, List, Dict, Set
from tqdm import tqdm
from collections import deque

class IsingVisualizer:
    def __init__(self, model: IsingModel):
        self.model = model
        self.cluster_updater = ClusterUpdater(model)
    
    def plot_configuration(self, title: str = None, save_path: str = None):
        """Plot current spin configuration with domain highlighting."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Raw spin configuration
        im1 = ax1.imshow(self.model.spins, cmap='coolwarm', 
                        vmin=-1, vmax=1)
        ax1.set_title('Spin Configuration')
        plt.colorbar(im1, ax=ax1, label='Spin')
        
        # Domain visualization
        domains = self._identify_domains_iterative()  # Using iterative version
        im2 = ax2.imshow(domains, cmap='tab20')
        ax2.set_title('Domain Structure')
        plt.colorbar(im2, ax=ax2, label='Domain ID')
        
        if title:
            fig.suptitle(title)
            
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print domain statistics
        domain_stats = self._analyze_domains(domains)
        print("\nDomain Statistics:")
        print(f"Number of domains: {domain_stats['n_domains']}")
        print(f"Average domain size: {domain_stats['avg_size']:.1f}")
        print(f"Largest domain size: {domain_stats['max_size']}")
        print(f"Total interface length: {domain_stats['interface_length']}")
    
    def _identify_domains_iterative(self) -> np.ndarray:
        """Identify connected domains using iterative approach."""
        L = self.model.L
        domains = np.zeros((L, L), dtype=int)
        domain_id = 1
        
        def get_domain(i: int, j: int, spin: int) -> Set[Tuple[int, int]]:
            """Get all points in a domain using BFS."""
            domain = set()
            queue = deque([(i, j)])
            
            while queue:
                current_i, current_j = queue.popleft()
                if (current_i < 0 or current_i >= L or 
                    current_j < 0 or current_j >= L or
                    domains[current_i, current_j] != 0 or
                    self.model.spins[current_i, current_j] != spin):
                    continue
                
                domains[current_i, current_j] = domain_id
                domain.add((current_i, current_j))
                
                for ni, nj in self.model.get_neighbor_spins(current_i, current_j):
                    queue.append((ni, nj))
            
            return domain
        
        for i in range(L):
            for j in range(L):
                if domains[i, j] == 0:
                    domain = get_domain(i, j, self.model.spins[i, j])
                    if domain:  # Only increment domain_id if we found a domain
                        domain_id += 1
        
        return domains
    
    def _analyze_domains(self, domains: np.ndarray) -> Dict:
        """Analyze domain structure and statistics."""
        unique_domains = np.unique(domains[domains > 0])  # Exclude 0s
        domain_sizes = [np.sum(domains == d) for d in unique_domains]
        
        # Calculate interface length
        L = self.model.L
        interface_length = 0
        for i in range(L):
            for j in range(L):
                current = domains[i,j]
                for ni, nj in self.model.get_neighbor_spins(i, j):
                    if domains[ni,nj] != current:
                        interface_length += 1
        
        return {
            'n_domains': len(unique_domains),
            'avg_size': np.mean(domain_sizes) if domain_sizes else 0,
            'max_size': max(domain_sizes) if domain_sizes else 0,
            'size_distribution': domain_sizes,
            'interface_length': interface_length // 2  # Divide by 2 as each interface is counted twice
        }
    
    def create_evolution_animation(self, n_frames: int, 
                                 algorithm: str = 'wolff',
                                 save_path: str = None):
        """Create animation of system evolution."""
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(self.model.spins, cmap='coolwarm', 
                      animated=True, vmin=-1, vmax=1)
        plt.colorbar(im)
        
        def update(frame):
            if algorithm == 'wolff':
                self.cluster_updater.wolff_update()
            else:
                self.cluster_updater.swendsen_wang_update()
            
            im.set_array(self.model.spins)
            ax.set_title(f'T={self.model.T:.2f}, Frame {frame}')
            return [im]
        
        anim = FuncAnimation(fig, update, frames=n_frames, 
                           interval=100, blit=True)
        
        if save_path:
            writer = PillowWriter(fps=10)
            anim.save(save_path, writer=writer)
        
        plt.show()

if __name__ == "__main__":
    # Test visualization
    print("Initializing model...")
    L = 50
    T = 2.27  # Critical temperature
    model = IsingModel(L, T)
    viz = IsingVisualizer(model)
    
    # Plot initial configuration
    print("\nPlotting initial configuration...")
    viz.plot_configuration(title="Initial Configuration")
    
    # Run some updates
    print("\nPerforming updates...")
    for _ in tqdm(range(1000), desc="Equilibrating"):
        viz.cluster_updater.wolff_update()
    
    # Plot final configuration
    print("\nPlotting final configuration...")
    viz.plot_configuration(title="After 1000 Wolff Updates")
    
    # Create animation
    print("\nCreating evolution animation...")
    viz.create_evolution_animation(n_frames=50, algorithm='wolff')
    
    print("\nComparing algorithms...")
    # Wolff
    model_wolff = IsingModel(L, T)
    viz_wolff = IsingVisualizer(model_wolff)
    for _ in tqdm(range(1000), desc="Wolff updates"):
        viz_wolff.cluster_updater.wolff_update()
    viz_wolff.plot_configuration(title="After 1000 Wolff Updates")
    
    # Swendsen-Wang
    model_sw = IsingModel(L, T)
    viz_sw = IsingVisualizer(model_sw)
    for _ in tqdm(range(1000), desc="Swendsen-Wang updates"):
        viz_sw.cluster_updater.swendsen_wang_update()
    viz_sw.plot_configuration(title="After 1000 Swendsen-Wang Updates")

