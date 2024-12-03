import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict

class CriticalExponentsAnalyzer:
    def __init__(self, Tc: float = 2.269185):
        self.Tc = Tc
    
    def analyze_magnetization(self, T: np.ndarray, m: np.ndarray, L: np.ndarray) -> Dict[str, float]:
        """
        Estimate the magnetization exponent (β) using finite-size scaling.
        """
        # Find the index of the temperature closest to Tc
        Tc_index = np.argmin(np.abs(T - self.Tc))
        
        # Extract magnetization values at Tc for each system size
        m_Tc = m[:, Tc_index]
        
        # Perform linear regression of log(m) vs. log(L)
        log_m = np.log(m_Tc)
        log_L = np.log(L)
        slope, intercept, r_value, p_value, stderr = stats.linregress(log_L, log_m)
        
        # Calculate the error in the slope (β/ν) using the standard error of the regression
        beta_over_nu_error = stderr
        
        return {
            'beta_over_nu': -slope,
            'beta_over_nu_error': beta_over_nu_error,
            'r_squared': r_value**2,
            'log_m': log_m,
            'log_L': log_L
        }
    
    def analyze_susceptibility(self, T: np.ndarray, chi: np.ndarray, L: np.ndarray) -> Dict[str, float]:
        """
        Estimate the susceptibility exponent (γ) using finite-size scaling.
        """
        # Find the index of the temperature closest to Tc
        Tc_index = np.argmin(np.abs(T - self.Tc))
        
        # Extract susceptibility values at Tc for each system size
        chi_Tc = chi[:, Tc_index]
        
        # Perform linear regression of log(χ) vs. log(L)
        log_chi = np.log(chi_Tc)
        log_L = np.log(L)
        slope, intercept, r_value, p_value, stderr = stats.linregress(log_L, log_chi)
        
        # Calculate the error in the slope (γ/ν) using the standard error of the regression
        gamma_over_nu_error = stderr
        
        return {
            'gamma_over_nu': slope,
            'gamma_over_nu_error': gamma_over_nu_error,
            'r_squared': r_value**2,
            'log_chi': log_chi,
            'log_L': log_L
        }
    
    def analyze(self, data: Dict) -> Dict[str, float]:
        """
        Analyze the critical exponents using the provided data.
        """
        results = {}
        
        for alg in ['wolff', 'sw']:
            T = data[alg]['T']
            L = np.array(data[alg]['L'])
            m = np.array(data[alg]['magnetization'])
            chi = np.array(data[alg]['susceptibility'])
            
            # Analyze magnetization exponent (β)
            mag_results = self.analyze_magnetization(T, m, L)
            
            # Analyze susceptibility exponent (γ)
            sus_results = self.analyze_susceptibility(T, chi, L)
            
            # Assuming ν = 1, calculate β and γ
            beta = mag_results['beta_over_nu']
            beta_error = mag_results['beta_over_nu_error']
            gamma = sus_results['gamma_over_nu']
            gamma_error = sus_results['gamma_over_nu_error']
            
            results[alg] = {
                'beta': beta,
                'beta_error': beta_error,
                'gamma': gamma,
                'gamma_error': gamma_error,
                'beta_r_squared': mag_results['r_squared'],
                'gamma_r_squared': sus_results['r_squared'],
                'log_m': mag_results['log_m'],
                'log_chi': sus_results['log_chi'],
                'log_L': mag_results['log_L']
            }
        
        return results
    
    def plot_magnetization_scaling(self, results: Dict, alg: str):
        """
        Plot the magnetization scaling relation.
        """
        log_m = results[alg]['log_m']
        log_L = results[alg]['log_L']
        beta = results[alg]['beta']
        
        plt.figure(figsize=(6, 4))
        plt.scatter(log_L, log_m, label='Data')
        plt.plot(log_L, -beta * log_L, label='Fit', color='r')
        plt.xlabel(r'$\log(L)$')
        plt.ylabel(r'$\log(|m|)$')
        plt.legend()
        plt.title(f'Magnetization Scaling - {alg.upper()} Algorithm')
        plt.show()
    
    def plot_susceptibility_scaling(self, results: Dict, alg: str):
        """
        Plot the susceptibility scaling relation.
        """
        log_chi = results[alg]['log_chi']
        log_L = results[alg]['log_L']
        gamma = results[alg]['gamma']
        
        plt.figure(figsize=(6, 4))
        plt.scatter(log_L, log_chi, label='Data')
        plt.plot(log_L, gamma * log_L, label='Fit', color='r')
        plt.xlabel(r'$\log(L)$')
        plt.ylabel(r'$\log(\chi)$')
        plt.legend()
        plt.title(f'Susceptibility Scaling - {alg.upper()} Algorithm')
        plt.show()