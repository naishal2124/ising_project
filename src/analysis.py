import numpy as np
from typing import Dict, List, Tuple
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import psutil
import warnings

@dataclass
class SimulationParameters:
    """Encapsulates simulation parameters for reproducibility."""
    L: int
    T: float
    n_sweeps: int
    equilibration: int
    n_bins: int = 20
    n_bootstrap: int = 1000
    
class AdvancedAnalysis:
    """Comprehensive analysis framework for cluster algorithm comparison."""
    
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.measurements = {
            'wolff': {'energies': [], 'magnetizations': [], 'cluster_sizes': [],
                     'times': [], 'memory': []},
            'sw': {'energies': [], 'magnetizations': [], 'cluster_sizes': [],
                  'times': [], 'memory': []}
        }
        
    def autocorrelation_time(self, data: np.ndarray, max_time: int = None) -> float:
        """Calculate integrated autocorrelation time."""
        if max_time is None:
            max_time = len(data) // 4
            
        mean = np.mean(data)
        var = np.var(data)
        
        acf = np.correlate(data - mean, data - mean, mode='full')
        acf = acf[len(data)-1:] / (var * np.arange(len(data), 0, -1))
        
        # Find where correlation drops below e^-2
        cutoff = np.where(acf < np.exp(-2))[0]
        if len(cutoff) > 0:
            return cutoff[0]
        return max_time
    
    def run_simulation(self, algorithm: str = 'both') -> Dict:
        """
        Run full simulation with progress tracking and advanced measurements.
        """
        from cluster import compare_algorithms  # Import here to avoid circular import
        
        results = {}
        L_values = [8, 16, 32, 64, 128]
        T_values = np.linspace(2.0, 2.5, 20)
        
        # Progress bars for nested loops
        L_pbar = tqdm(L_values, desc='System sizes')
        for L in L_pbar:
            results[L] = {}
            T_pbar = tqdm(T_values, desc=f'Temperatures (L={L})', leave=False)
            for T in T_pbar:
                # Run simulation with progress tracking
                params = SimulationParameters(L=L, T=T, 
                                           n_sweeps=self.params.n_sweeps,
                                           equilibration=self.params.equilibration)
                
                # Track memory usage
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                sim_results = compare_algorithms(L, T, 
                                              n_updates=params.n_sweeps,
                                              equilibration=params.equilibration)
                
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = final_memory - initial_memory
                
                # Enhanced measurements
                for alg in ['wolff', 'sw']:
                    if algorithm != 'both' and alg != algorithm:
                        continue
                        
                    self.measurements[alg]['memory'].append(memory_used)
                    tau = self.autocorrelation_time(
                        np.array(sim_results[alg]['magnetizations']))
                    
                    results[L][T] = {
                        'energy': sim_results[alg]['final_energy'],
                        'magnetization': sim_results[alg]['final_mag'],
                        'autocorr_time': tau,
                        'cluster_sizes': sim_results[alg]['cluster_sizes'],
                        'execution_time': sim_results[alg]['time'],
                        'memory_usage': memory_used
                    }
                    
                # Update progress description
                T_pbar.set_description(
                    f'T={T:.2f}, E={results[L][T]["energy"]:.3f}, '
                    f'|m|={results[L][T]["magnetization"]:.3f}')
                    
        return results
    
    def error_analysis(self, data: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive error analysis using multiple methods.
        """
        # Jackknife error
        n = len(data)
        jackknife_estimates = np.zeros(n)
        for i in range(n):
            jackknife_estimates[i] = np.mean(np.delete(data, i))
        jackknife_error = np.sqrt((n-1) * np.var(jackknife_estimates))
        
        # Bootstrap error
        bootstrap_estimates = np.zeros(self.params.n_bootstrap)
        for i in range(self.params.n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_estimates[i] = np.mean(bootstrap_sample)
        bootstrap_error = np.std(bootstrap_estimates)
        
        # Binning analysis
        bin_errors = []
        for bin_size in [1, 2, 4, 8, 16]:
            if len(data) < bin_size:
                continue
            n_bins = len(data) // bin_size
            binned_data = np.mean(data[:n_bins*bin_size].reshape(-1, bin_size), axis=1)
            bin_errors.append(np.std(binned_data) / np.sqrt(n_bins))
        
        return {
            'jackknife': jackknife_error,
            'bootstrap': bootstrap_error,
            'binning': np.max(bin_errors) if bin_errors else None
        }
    
    def analyze_critical_behavior(self, results: Dict) -> Dict:
        """
        Analyze critical behavior and extract critical exponents.
        """
        L_values = sorted(results.keys())
        T_values = sorted(results[L_values[0]].keys())
        
        # Find critical temperature using Binder cumulant crossings
        binder_crossings = []
        for i in range(len(L_values)-1):
            L1, L2 = L_values[i], L_values[i+1]
            for j in range(len(T_values)-1):
                T1, T2 = T_values[j], T_values[j+1]
                U1 = results[L1][T1]['binder']
                U2 = results[L2][T1]['binder']
                U3 = results[L1][T2]['binder']
                U4 = results[L2][T2]['binder']
                
                if (U1-U2)*(U3-U4) < 0:  # Crossing detected
                    Tc = T1 + (T2-T1)*(U1-U2)/(U1-U2-U3+U4)
                    binder_crossings.append(Tc)
        
        Tc_estimate = np.mean(binder_crossings)
        Tc_error = np.std(binder_crossings) / np.sqrt(len(binder_crossings))
        
        # Extract critical exponents
        exponents = {}
        for L in L_values:
            m = [results[L][T]['magnetization'] for T in T_values]
            chi = [results[L][T]['susceptibility'] for T in T_values]
            
            # Finite-size scaling analysis
            t = (np.array(T_values) - Tc_estimate) / Tc_estimate
            m = np.array(m)
            chi = np.array(chi)
            
            # Extract β/ν from magnetization scaling
            mask = np.abs(t) < 0.1
            beta_nu, _ = np.polyfit(np.log(L), np.log(m[mask].mean()), 1)
            exponents[f'beta_nu_L{L}'] = -beta_nu
            
            # Extract γ/ν from susceptibility scaling
            gamma_nu, _ = np.polyfit(np.log(L), np.log(chi[mask].mean()), 1)
            exponents[f'gamma_nu_L{L}'] = gamma_nu
        
        return {
            'Tc': Tc_estimate,
            'Tc_error': Tc_error,
            'critical_exponents': exponents
        }
    
    def plot_analysis(self, results: Dict, save_path: str = None):
        """
        Create publication-quality plots of the analysis.
        """
        plt.style.use('seaborn')
        
        # Create figure grid
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 3)
        
        # Plot 1: Magnetization vs Temperature
        ax1 = fig.add_subplot(gs[0, 0])
        for L in sorted(results.keys()):
            T = sorted(results[L].keys())
            m = [results[L][t]['magnetization'] for t in T]
            ax1.plot(T, m, 'o-', label=f'L={L}')
        ax1.set_xlabel('Temperature T/J')
        ax1.set_ylabel('|m|')
        ax1.legend()
        
        # Plot 2: Autocorrelation times
        ax2 = fig.add_subplot(gs[0, 1])
        L_values = sorted(results.keys())
        tau_wolff = [np.mean([results[L][T]['wolff']['autocorr_time'] 
                            for T in results[L]]) for L in L_values]
        tau_sw = [np.mean([results[L][T]['sw']['autocorr_time'] 
                          for T in results[L]]) for L in L_values]
        ax2.loglog(L_values, tau_wolff, 'o-', label='Wolff')
        ax2.loglog(L_values, tau_sw, 's-', label='SW')
        ax2.set_xlabel('System size L')
        ax2.set_ylabel('τ')
        ax2.legend()
        
        # Add more plots...
        

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()