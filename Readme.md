# Critical Phenomena and Cluster Algorithms: A Comparative Analysis of the 2D Ising Model

This project focuses on the comparative analysis of the Wolff single-cluster and Swendsen-Wang multi-cluster algorithms in the study of critical phenomena in the 2D Ising model. The project includes a comprehensive implementation of the 2D Ising model, along with advanced analysis techniques, visualization tools, and a detailed study of the performance and behavior of the cluster algorithms near the critical temperature.

## Features

- Implementation of the 2D Ising model with periodic boundary conditions
- Wolff single-cluster and Swendsen-Wang multi-cluster algorithms for efficient Monte Carlo sampling
- Advanced analysis techniques, including finite-size scaling, Binder cumulant analysis, and critical exponent extraction
- Comprehensive error estimation using jackknife resampling, bootstrap resampling, and binning analysis
- Interactive visualization of the 2D Ising model, including spin configuration plots, domain structure analysis, and real-time evolution animation
- Comparative analysis of the Wolff and Swendsen-Wang algorithms in terms of performance, autocorrelation times, and critical behavior
- Replication of key results from a reference paper and comparison with the project's findings
- Detailed documentation, including a comprehensive lab report and presentation slides

## Repository Structure

- `src/`: Contains the source code for the 2D Ising model implementation, cluster algorithms, analysis tools, and visualization functions
- `notebooks/`: Includes Jupyter notebooks for running simulations, performing analysis, and generating figures
- `data/`: Stores the raw simulation data and processed results
- `figures/`: Contains high-resolution versions of the figures generated during the analysis
- `presentation/`: Includes the final presentation slides (`2D_Ising_Model_Presentation.pptx`)
- `report/`: Contains the comprehensive lab report (`Final_Project_Naishal_Patel_Physics_514.pdf`)
- `README.md`: Provides an overview of the project, installation instructions, and usage guidelines
- `requirements.txt`: Lists the required Python packages and their versions for running the code

## Installation and Usage

1. Clone the repository:
git clone https://github.com/your-username/ising-project.git
cd ising-project
Copy
2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # For Unix/Linux
venv\Scripts\activate  # For Windows
Copy
3. Install the required packages:
pip install -r requirements.txt
Copy
4. Run the Jupyter notebooks in the `notebooks/` directory to perform simulations, analysis, and generate figures.

5. Refer to the lab report (`report/Final_Project_Naishal_Patel_Physics_514.pdf`) for a detailed description of the project, methods, results, and conclusions.

6. The final presentation slides (`presentation/2D_Ising_Model_Presentation.pptx`) provide an overview of the project and its key findings.

## References

- Wolff Algorithm: U. Wolff, "Collective Monte Carlo updating for spin systems," Physical Review Letters, vol. 62, no. 4, pp. 361-364, 1989.
- Swendsen-Wang Algorithm: R. H. Swendsen and J.-S. Wang, "Nonuniversal critical dynamics in Monte Carlo simulations," Physical Review Letters, vol. 58, no. 2, pp. 86-88, 1987.
- Reference Paper: A. Author et al., "A model project for reproducible papers," arXiv:1234.5678, 2023.

Please refer to the lab report for a complete list of references.