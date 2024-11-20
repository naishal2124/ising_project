import numpy as np
from typing import Tuple, List, Dict
import time

class IsingModel:
    """
    2D Ising model implementation optimized for cluster updates.
    Key features:
    - Periodic boundary conditions
    - Temperature in units of J/kb (J=1)
    - Support for both Wolff and Swendsen-Wang
    """
    
    def __init__(self, L: int, T: float):
        """
        Initialize Ising model with given size and temperature.
        
        Args:
            L: Linear system size
            T: Temperature in units of J/kb (J=1)
        """
        self.L = L
        self.N = L * L
        self.T = T
        self.beta = 1.0 / self.T
        
        # Initialize random spin configuration
        self.spins = np.random.choice([-1, 1], size=(L, L))
        
        # Cluster update probability
        self.p_add = 1.0 - np.exp(-2.0 * self.beta)
        
        # For measurements
        self.energy = self._compute_energy()
        self.magnetization = np.sum(self.spins)
    
    def _periodic_index(self, i: int) -> int:
        """Handle periodic boundary conditions."""
        return i % self.L
    
    def _compute_energy(self) -> float:
        """
        Compute total energy of the configuration.
        Optimized using numpy operations.
        """
        # Compute neighbor sum using roll operations
        right = np.roll(self.spins, 1, axis=1)
        left = np.roll(self.spins, -1, axis=1)
        up = np.roll(self.spins, 1, axis=0)
        down = np.roll(self.spins, -1, axis=0)
        
        neighbor_sum = right + left + up + down
        return -0.5 * np.sum(self.spins * neighbor_sum)
    
    def get_neighbor_spins(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Get coordinates of neighboring spins with periodic boundaries."""
        return [
            (self._periodic_index(i+1), j),
            (self._periodic_index(i-1), j),
            (i, self._periodic_index(j+1)),
            (i, self._periodic_index(j-1))
        ]
    
    def measure_observables(self) -> Dict[str, float]:
        """
        Measure key observables: energy, magnetization, specific heat susceptibility.
        Returns normalized (per-spin) quantities.
        """
        E = self.energy / self.N
        M = self.magnetization / self.N
        
        return {
            'energy': E,
            'magnetization': M,
            'abs_magnetization': abs(M)
        }
    
    def measure_binder_cumulant(self, measurements: int = 1000) -> float:
        """
        Compute Binder cumulant U₄ = <m⁴>/<m²>².
        Args:
            measurements: Number of measurements to average over
        """
        m2_sum = 0
        m4_sum = 0
        
        for _ in range(measurements):
            m = self.magnetization / self.N
            m2 = m * m
            m2_sum += m2
            m4_sum += m2 * m2
        
        m2_avg = m2_sum / measurements
        m4_avg = m4_sum / measurements
        
        return m4_avg / (m2_avg * m2_avg)
    
    def single_flip_sweep(self) -> None:
        """
        Perform one sweep of single-spin Metropolis updates.
        Mainly used for testing and equilibration.
        """
        for i in range(self.L):
            for j in range(self.L):
                # Compute local field
                neighbors = self.get_neighbor_spins(i, j)
                local_field = sum(self.spins[x,y] for x,y in neighbors)
                
                # Compute energy change
                dE = 2.0 * self.spins[i,j] * local_field
                
                # Metropolis acceptance
                if dE <= 0 or np.random.random() < np.exp(-self.beta * dE):
                    self.spins[i,j] *= -1
                    self.energy += dE
                    self.magnetization += 2.0 * self.spins[i,j]

    def validate_state(self) -> bool:
        """
        Validate internal state consistency.
        Returns True if state is consistent.
        """
        # Check energy
        computed_energy = self._compute_energy()
        energy_match = abs(computed_energy - self.energy) < 1e-10
        
        # Check magnetization
        computed_mag = np.sum(self.spins)
        mag_match = computed_mag == self.magnetization
        
        return energy_match and mag_match

# Test the implementation
if __name__ == "__main__":
    # Create small system near critical temperature
    L = 16
    T = 2.27  # Near Tc
    model = IsingModel(L, T)
    
    # Test measurements
    print("Initial measurements:")
    obs = model.measure_observables()
    for key, value in obs.items():
        print(f"{key}: {value:.6f}")
    
    # Test state validation
    print("\nState validation:", model.validate_state())
    
    # Test single flip update
    t0 = time.time()
    for _ in range(100):
        model.single_flip_sweep()
    print(f"\nTime for 100 sweeps: {time.time() - t0:.3f}s")
    
    # Verify state remains consistent
    print("State validation after updates:", model.validate_state())
    
    # Test Binder cumulant
    U4 = model.measure_binder_cumulant(measurements=100)
    print(f"\nBinder cumulant: {U4:.6f}")