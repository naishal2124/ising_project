import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
from scipy import stats
import seaborn as sns
from pathlib import Path
import json

from ising import IsingModel
from cluster import ClusterUpdater

class PaperReplication:
    """
    Class to replicate results from arXiv:1401.2000.
    Focus on efficient computation while maintaining accuracy.
    """
    
    def __init__(self, 
                 L_values: List[int] = [8, 12, 16, 24, 32, 48, 64],
                 n_measurements: int = 100000,  # Reduced from paper's 1,280,000
                 thermalization: int = 10000,   # Reduced from paper's ~128,000
                 save_dir: str = 'results/data'):
        """
        Initialize replication parameters.
        
        Args:
            L_values: System sizes (modified from paper for efficiency)
            n_measurements: Measurements per point
            thermalization: Equilibration steps
            save_dir: Directory to save results
        """
        self.L_values = L_values
        self.n_measurements = n_measurements
        self.thermalization = thermalization
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Critical parameters
        self.Tc = 2.269185
        self.beta_critical = 0.125
        self.nu_critical = 1.0
        self.gamma_critical = 1.75
        
        # Temperature ranges
        self.T_range_wide = np.linspace(2.15, 2.40, 25)  # For susceptibility
        self.T_range_critical = np.linspace(2.26, 2.28, 20)  # For Binder cumulant
        
    def measure_susceptibility(self) -> Dict:
        """Replicate Figure 1: Susceptibility vs Temperature."""
        results = {}
        start_time = time.time()
        
        for L in self.L_values:
            print(f"\nStarting L={L}")
            results[L] = {'T': [], 'chi': [], 'chi_err': []}
            
            for T in tqdm(self.T_range_wide, desc=f"Size L={L}"):
                model = IsingModel(L, T)
                updater = ClusterUpdater(model)
                
                # Thermalization
                for _ in range(self.thermalization):
                    updater.wolff_update()
                
                # Measurements with binning
                m_bins = []
                bin_size = 1000
                current_bin = []
                
                for _ in range(self.n_measurements):
                    updater.wolff_update()
                    m = model.magnetization/model.N
                    current_bin.append(m)
                    
                    if len(current_bin) == bin_size:
                        m_bins.append(np.mean(current_bin))
                        current_bin = []
                
                # Calculate susceptibility and error
                m_bins = np.array(m_bins)
                chi = model.beta * model.N * (np.mean(m_bins**2) - np.mean(np.abs(m_bins))**2)
                chi_err = self._jackknife_error(m_bins)
                
                results[L]['T'].append(T)
                results[L]['chi'].append(chi)
                results[L]['chi_err'].append(chi_err)
            
            # Save results after each L
            self._save_results(results, 'susceptibility.json')
            
        print(f"\nTotal time: {(time.time() - start_time)/3600:.2f} hours")
        return results
    
    def measure_binder_cumulant(self) -> Dict:
        """Replicate Figure 2: Binder Cumulant Analysis."""
        results = {}
        start_time = time.time()
        
        for L in self.L_values:
            print(f"\nStarting L={L}")
            results[L] = {'T': [], 'U4': [], 'U4_err': []}
            
            for T in tqdm(self.T_range_critical, desc=f"Size L={L}"):
                model = IsingModel(L, T)
                updater = ClusterUpdater(model)
                
                # Thermalization
                for _ in range(self.thermalization):
                    updater.wolff_update()
                
                # Measurements with binning
                m2_bins = []
                m4_bins = []
                bin_size = 1000
                current_m2 = []
                current_m4 = []
                
                for _ in range(self.n_measurements):
                    updater.wolff_update()
                    m = model.magnetization/model.N
                    m2 = m*m
                    m4 = m2*m2
                    current_m2.append(m2)
                    current_m4.append(m4)
                    
                    if len(current_m2) == bin_size:
                        m2_bins.append(np.mean(current_m2))
                        m4_bins.append(np.mean(current_m4))
                        current_m2 = []
                        current_m4 = []
                
                # Calculate U4 and error
                m2_bins = np.array(m2_bins)
                m4_bins = np.array(m4_bins)
                U4 = np.mean(m4_bins)/(np.mean(m2_bins)**2)
                U4_err = self._jackknife_error_u4(m2_bins, m4_bins)
                
                results[L]['T'].append(T)
                results[L]['U4'].append(U4)
                results[L]['U4_err'].append(U4_err)
            
            # Save results after each L
            self._save_results(results, 'binder.json')
            
        print(f"\nTotal time: {(time.time() - start_time)/3600:.2f} hours")
        return results
    
    def _jackknife_error(self, data: np.ndarray) -> float:
        """Calculate error using jackknife resampling."""
        n = len(data)
        jackknife_estimates = np.zeros(n)
        
        for i in range(n):
            resampled = np.delete(data, i)
            jackknife_estimates[i] = np.var(resampled)
        
        return np.sqrt((n-1) * np.var(jackknife_estimates))
    
    def _jackknife_error_u4(self, m2: np.ndarray, m4: np.ndarray) -> float:
        """Calculate Binder cumulant error using jackknife."""
        n = len(m2)
        jackknife_estimates = np.zeros(n)
        
        for i in range(n):
            m2_resampled = np.delete(m2, i)
            m4_resampled = np.delete(m4, i)
            jackknife_estimates[i] = np.mean(m4_resampled)/(np.mean(m2_resampled)**2)
        
        return np.sqrt((n-1) * np.var(jackknife_estimates))
    
    def _save_results(self, results: Dict, filename: str):
        """Save results to JSON file."""
        save_path = self.save_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for L in results:
            serializable_results[str(L)] = {
                key: np.array(value).tolist() 
                for key, value in results[L].items()
            }
            
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f)
    
    def plot_susceptibility(self, results: Dict):
        """Create publication-quality susceptibility plot."""
        plt.figure(figsize=(10, 6))
        
        for L in self.L_values:
            T = results[L]['T']
            chi = results[L]['chi']
            err = results[L]['chi_err']
            
            plt.errorbar(T, chi, yerr=err, fmt='o-', 
                        label=f'L={L}', capsize=3)
        
        plt.xlabel('Temperature T/J')
        plt.ylabel('Susceptibility χ')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.title("Temperature Dependence of Susceptibility")
        
        # Save plot
        plt.savefig(self.save_dir / 'susceptibility.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.show()
    
    def plot_binder(self, results: Dict):
        """Create publication-quality Binder cumulant plot."""
        plt.figure(figsize=(10, 6))
        
        for L in self.L_values:
            T = results[L]['T']
            U4 = results[L]['U4']
            err = results[L]['U4_err']
            
            plt.errorbar(T, U4, yerr=err, fmt='o-', 
                        label=f'L={L}', capsize=3)
        
        plt.axvline(self.Tc, color='r', linestyle='--', 
                   label='Exact Tc')
        plt.xlabel('Temperature T/J')
        plt.ylabel('Binder Cumulant U₄')
        plt.grid(True)
        plt.legend()
        plt.title("Binder Cumulant Analysis")
        
        # Save plot
        plt.savefig(self.save_dir / 'binder.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.show()

if __name__ == "__main__":
    replicator = PaperReplication()
    
    # Run susceptibility analysis
    chi_results = replicator.measure_susceptibility()
    replicator.plot_susceptibility(chi_results)
    
    # Run Binder cumulant analysis
    binder_results = replicator.measure_binder_cumulant()
    replicator.plot_binder(binder_results)