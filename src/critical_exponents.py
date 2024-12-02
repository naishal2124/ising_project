import numpy as np
from scipy import stats, optimize
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings


__all__ = ['CriticalExponentsAnalyzer', 'ExponentResult', 'calculate_error_bars']

class ExponentResult(NamedTuple):
    value: float
    error: float
    bootstrap_error: Dict[str, float]
    jackknife_error: Dict[str, float]
    r_squared: float
    intercept: float
    data: Dict

class CriticalExponentsAnalyzer:
    def __init__(self, Tc: float = 2.269185, n_bootstrap: int = 1000):
        self.Tc = Tc
        self.n_bootstrap = n_bootstrap
    
    def _bootstrap_error(self, data: np.ndarray) -> Dict[str, float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bootstrap_samples = np.random.choice(data, size=(self.n_bootstrap, len(data)), replace=True)
            bootstrap_means = np.mean(bootstrap_samples, axis=1)
            
            return {
                'mean': np.mean(bootstrap_means),
                'std': np.std(bootstrap_means),
                'confidence_95': np.percentile(bootstrap_means, [2.5, 97.5])
            }
    
    def _jackknife_error(self, data: np.ndarray) -> Dict[str, float]:
        n = len(data)
        jackknife_estimates = np.zeros(n)
        
        for i in range(n):
            jackknife_estimates[i] = np.mean(np.delete(data, i))
        
        return {
            'mean': np.mean(jackknife_estimates),
            'std': np.sqrt((n-1) * np.var(jackknife_estimates)),
            'bias': (n-1) * (np.mean(jackknife_estimates) - np.mean(data))
        }

    def extract_beta(self, T_values: np.ndarray, m_values: np.ndarray) -> ExponentResult:
        mask = (T_values < self.Tc) & (T_values > self.Tc - 0.1)
        t = (T_values[mask] - self.Tc) / self.Tc
        m = m_values[mask]
        
        log_t = np.log10(np.abs(t))
        log_m = np.log10(m)
        
        slope, intercept, r_value, p_value, stderr = stats.linregress(log_t, log_m)
        
        bootstrap_results = []
        for _ in range(self.n_bootstrap):
            idx = np.random.randint(0, len(t), len(t))
            s, i, _, _, _ = stats.linregress(log_t[idx], log_m[idx])
            bootstrap_results.append(s)
        
        bootstrap_error = self._bootstrap_error(np.array(bootstrap_results))
        jackknife_error = self._jackknife_error(m)
        
        return ExponentResult(
            value=np.abs(slope),
            error=stderr,
            bootstrap_error=bootstrap_error,
            jackknife_error=jackknife_error,
            r_squared=r_value**2,
            intercept=intercept,
            data={'t': t, 'm': m, 'bootstrap_samples': bootstrap_results}
        )

    def extract_nu(self, L_values: np.ndarray, binder_derivatives: np.ndarray) -> ExponentResult:
        if len(L_values) < 2 or len(binder_derivatives) < 2:
            raise ValueError("Insufficient data points for nu regression")
            
        log_L = np.log10(L_values)
        log_deriv = np.log10(np.abs(binder_derivatives))
        
        if np.any(np.isnan(log_deriv)) or np.any(np.isinf(log_deriv)):
            raise ValueError("Invalid values in Binder derivatives")
        
        slope, intercept, r_value, p_value, stderr = stats.linregress(log_L, log_deriv)
        
        bootstrap_results = []
        for _ in range(self.n_bootstrap):
            idx = np.random.randint(0, len(L_values), len(L_values))
            s, i, _, _, _ = stats.linregress(log_L[idx], log_deriv[idx])
            bootstrap_results.append(1.0/s)
        
        bootstrap_error = self._bootstrap_error(np.array(bootstrap_results))
        jackknife_error = self._jackknife_error(binder_derivatives)
        
        return ExponentResult(
            value=1.0/slope,
            error=stderr/(slope**2),
            bootstrap_error=bootstrap_error,
            jackknife_error=jackknife_error,
            r_squared=r_value**2,
            intercept=intercept,
            data={'L': L_values, 'derivatives': binder_derivatives, 
                  'bootstrap_samples': bootstrap_results}
        )

    def extract_gamma(self, L_values: np.ndarray, chi_max: np.ndarray) -> ExponentResult:
        if len(L_values) < 2 or len(chi_max) < 2:
            raise ValueError("Insufficient data points for gamma regression")
            
        log_L = np.log10(L_values)
        log_chi = np.log10(chi_max)
        
        slope, intercept, r_value, p_value, stderr = stats.linregress(log_L, log_chi)
        
        bootstrap_results = []
        for _ in range(self.n_bootstrap):
            idx = np.random.randint(0, len(L_values), len(L_values))
            s, i, _, _, _ = stats.linregress(log_L[idx], log_chi[idx])
            bootstrap_results.append(s)
        
        bootstrap_error = self._bootstrap_error(np.array(bootstrap_results))
        jackknife_error = self._jackknife_error(chi_max)
        
        return ExponentResult(
            value=slope,
            error=stderr,
            bootstrap_error=bootstrap_error,
            jackknife_error=jackknife_error,
            r_squared=r_value**2,
            intercept=intercept,
            data={'L': L_values, 'chi_max': chi_max, 
                  'bootstrap_samples': bootstrap_results}
        )

    def analyze_all(self, data: Dict) -> Dict[str, ExponentResult]:
        results = {}
        
        T = np.array(data['T'])
        L = np.array(data['L'])
        m = np.array(data['magnetization'])
        chi = np.array(data['susceptibility'])
        binder = np.array(data['binder'])
        
        m_largest = m[-1, :]
        
        dU_dT = []
        for l_idx in range(len(L)):
            mask = np.abs(T - self.Tc) < 0.05
            T_near = T[mask]
            U_near = binder[l_idx, mask]
            dU_dT.append(np.gradient(U_near, T_near)[len(T_near)//2])
        
        chi_peaks = []
        for l_idx in range(len(L)):
            chi_peaks.append(np.max(chi[l_idx]))
        
        try:
            results['beta'] = self.extract_beta(T, m_largest)
        except Exception as e:
            print(f"Warning: Could not extract β: {e}")
            results['beta'] = None
        
        try:
            results['nu'] = self.extract_nu(L, np.array(dU_dT))
        except Exception as e:
            print(f"Warning: Could not extract ν: {e}")
            results['nu'] = None
        
        try:
            results['gamma'] = self.extract_gamma(L, np.array(chi_peaks))
        except Exception as e:
            print(f"Warning: Could not extract γ: {e}")
            results['gamma'] = None
        
        return results
    
    def plot_comprehensive_analysis(self, results: Dict[str, ExponentResult], save_path: str = None):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Beta
        if results['beta'] is not None:
            t = results['beta'].data['t']
            m = results['beta'].data['m']
            ax1.loglog(np.abs(t), m, 'o', label='Data')
            
            t_fit = np.linspace(np.min(t), np.max(t), 100)
            m_fit = 10**results['beta'].intercept * np.abs(t_fit)**results['beta'].value
            ax1.loglog(np.abs(t_fit), m_fit, label=f"Fit: β={results['beta'].value:.3f}±{results['beta'].error:.3f}")
            
            ax1.set_xlabel('|t|')
            ax1.set_ylabel('m')
            ax1.legend()
            ax1.set_title('Magnetization Scaling')
        
        # Nu
        if results['nu'] is not None:
            L = results['nu'].data['L']
            dU_dT = results['nu'].data['derivatives']
            ax2.loglog(L, np.abs(dU_dT), 'o', label='Data')
            
            L_fit = np.linspace(np.min(L), np.max(L), 100)
            dU_dT_fit = 10**results['nu'].intercept * L_fit**(1/results['nu'].value)
            ax2.loglog(L_fit, dU_dT_fit, label=f"Fit: ν={results['nu'].value:.3f}±{results['nu'].error:.3f}")
            
            ax2.set_xlabel('L')
            ax2.set_ylabel('|dU/dT|')
            ax2.legend()
            ax2.set_title('Correlation Length Scaling')
        
        # Gamma
        if results['gamma'] is not None:
            L = results['gamma'].data['L']
            chi_max = results['gamma'].data['chi_max']
            ax3.loglog(L, chi_max, 'o', label='Data')
            
            L_fit = np.linspace(np.min(L), np.max(L), 100)
            chi_max_fit = 10**results['gamma'].intercept * L_fit**results['gamma'].value
            ax3.loglog(L_fit, chi_max_fit, label=f"Fit: γ={results['gamma'].value:.3f}±{results['gamma'].error:.3f}")
            
            ax3.set_xlabel('L')
            ax3.set_ylabel('χ_max')
            ax3.legend()
            ax3.set_title('Susceptibility Scaling')
        
        # Data collapse
        beta = 0.125 if results['beta'] is None else results['beta'].value
        nu = 1.0 if results['nu'] is None else results['nu'].value
        
        if results['beta'] is not None:
            T_values = results['beta'].data['t'] * self.Tc + self.Tc
            m_values = results['beta'].data['m']
            L_values = results['beta'].data.get('L', np.array([8, 16, 32, 48]))  # Default L values if not available
            
            for L_idx, L in enumerate(L_values):
                scaled_t = (T_values - self.Tc) / self.Tc * L**(1/nu)
                scaled_m = m_values[L_idx] * L**(beta/nu)
                ax4.plot(scaled_t, scaled_m, 'o', label=f'L={L}')
            
            ax4.set_xlabel(r'$L^{1/\nu} t$')
            ax4.set_ylabel(r'$L^{\beta/\nu} m$')
            ax4.legend()
            ax4.set_title('Magnetization Data Collapse')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
def calculate_error_bars(data: np.ndarray, n_bootstrap: int = 1000) -> Tuple[float, float]:
    """Calculate error bars using bootstrap method."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bootstrap_samples = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        return np.percentile(bootstrap_means, [16, 84])