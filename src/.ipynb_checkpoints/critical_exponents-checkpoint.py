import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
from dataclasses import dataclass

@dataclass
class CriticalExponent:
    value: float
    error: float
    r_squared: float
    fit_range: Tuple[float, float]
    
class CriticalExponentsAnalyzer:
    """Extracts critical exponents from Ising model data."""
    
    def __init__(self, Tc: float = 2.269185):
        self.Tc = Tc
        
    def _compute_reduced_temperature(self, T: float) -> float:
        """Calculate reduced temperature t = (T-Tc)/Tc."""
        return (T - self.Tc) / self.Tc
    
    def extract_beta(self, T_values: np.ndarray, m_values: np.ndarray, 
                    T_range: Tuple[float, float] = None) -> CriticalExponent:
        """Extract β from magnetization data below Tc."""
        # Select data below Tc
        mask = T_values < self.Tc
        if T_range:
            mask &= (T_values > T_range[0]) & (T_values < T_range[1])
            
        t = self._compute_reduced_temperature(T_values[mask])
        m = m_values[mask]
        
        # Log-log fit
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log(-t), np.log(m)
            )
        
        return CriticalExponent(
            value=slope,
            error=std_err,
            r_squared=r_value**2,
            fit_range=(T_values[mask][0], T_values[mask][-1])
        )
    
    def extract_gamma(self, L_values: np.ndarray, chi_max: np.ndarray) -> CriticalExponent:
        """Extract γ/ν from finite-size scaling of susceptibility."""
        # Log-log fit of χmax vs L
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log(L_values), np.log(chi_max)
            )
        
        return CriticalExponent(
            value=slope,
            error=std_err,
            r_squared=r_value**2,
            fit_range=(L_values[0], L_values[-1])
        )
    
    def extract_nu(self, L_values: np.ndarray, binder_derivatives: np.ndarray) -> CriticalExponent:
        """Extract ν from Binder cumulant derivatives at Tc."""
        # Log-log fit of dU/dT|_Tc vs L
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log(L_values), np.log(np.abs(binder_derivatives))
            )
        
        return CriticalExponent(
            value=1/slope,  # Note: slope = 1/ν
            error=std_err/(slope**2),
            r_squared=r_value**2,
            fit_range=(L_values[0], L_values[-1])
        )
    
    def analyze_all_exponents(self, results: Dict) -> Dict:
        """Complete analysis of all critical exponents."""
        exponents = {}
        
        for alg in ['wolff', 'sw']:
            L_values = sorted(list(results[alg].keys()))
            T_values = sorted(list(results[alg][L_values[0]].keys()))
            
            # Prepare data for β extraction
            T_arr = np.array(T_values)
            m_arr = np.array([results[alg][L_values[-1]][T]['mag_mean'] 
                            for T in T_values])
            
            # Prepare data for γ/ν extraction
            chi_max = np.array([max(results[alg][L][T]['susceptibility'] 
                                  for T in T_values)
                              for L in L_values])
            
            # Prepare data for ν extraction
            # Compute numerical derivatives of Binder cumulant at Tc
            dT = 0.01
            binder_derivs = []
            for L in L_values:
                U_vals = []
                for T in [self.Tc - dT, self.Tc + dT]:
                    if T in results[alg][L]:
                        U_vals.append(results[alg][L][T]['binder'])
                    else:
                        # Interpolate if necessary
                        T_idx = np.searchsorted(T_values, T)
                        if T_idx > 0 and T_idx < len(T_values):
                            T1, T2 = T_values[T_idx-1], T_values[T_idx]
                            U1 = results[alg][L][T1]['binder']
                            U2 = results[alg][L][T2]['binder']
                            U = U1 + (U2-U1)*(T-T1)/(T2-T1)
                            U_vals.append(U)
                
                if len(U_vals) == 2:
                    deriv = (U_vals[1] - U_vals[0])/(2*dT)
                    binder_derivs.append(deriv)
                else:
                    binder_derivs.append(np.nan)
            
            binder_derivs = np.array(binder_derivs)
            valid_mask = ~np.isnan(binder_derivs)
            
            # Extract exponents
            exponents[alg] = {
                'beta': self.extract_beta(T_arr, m_arr),
                'gamma_over_nu': self.extract_gamma(
                    np.array(L_values), chi_max
                ),
                'nu': self.extract_nu(
                    np.array(L_values)[valid_mask],
                    binder_derivs[valid_mask]
                )
            }
        
        return exponents

def analyze_critical_scaling(results: Dict) -> Dict:
    """Wrapper function for critical exponents analysis."""
    analyzer = CriticalExponentsAnalyzer()
    return analyzer.analyze_all_exponents(results)