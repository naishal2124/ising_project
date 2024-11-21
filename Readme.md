# 2D Ising Model Critical Phenomena Analysis

A comprehensive implementation of the 2D Ising model focusing on critical phenomena and cluster algorithm comparison.

## Project Overview

This project implements and analyzes the 2D Ising model using both Wolff single-cluster and Swendsen-Wang multi-cluster algorithms. The implementation focuses on studying critical behavior and comparing algorithm performance near the critical temperature.

### Key Features

- **Core Implementation**
  - 2D Ising model with periodic boundaries
  - Temperature-dependent dynamics
  - Energy and magnetization tracking

- **Cluster Algorithms**
  - Wolff single-cluster updates
  - Swendsen-Wang multi-cluster updates
  - Cluster size distribution analysis

- **Analysis Tools**
  - Critical temperature estimation
  - Binder cumulant analysis
  - Finite-size scaling
  - Error estimation

- **Visualization**
  - Spin configuration plots
  - Domain structure analysis
  - Real-time evolution animation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/naishal2124/ising_project.git
cd ising_project
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix/MacOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Simulation
```python
from src.ising import IsingModel
from src.cluster import ClusterUpdater

# Create model
model = IsingModel(L=32, T=2.27)  # L=size, T=temperature

# Run updates
updater = ClusterUpdater(model)
updater.wolff_update()  # or updater.swendsen_wang_update()
```

### Analysis
```python
from src.analysis import AdvancedAnalysis
from src.utils import create_temperature_range

# Setup analysis
temps = create_temperature_range()
analysis = AdvancedAnalysis(L=32, temps)

# Run analysis
results = analysis.run_simulation()
```

## Project Structure

- `src/`
  - `ising.py`: Core Ising model implementation
  - `cluster.py`: Cluster algorithm implementations
  - `analysis.py`: Analysis and measurement tools
  - `visual.py`: Visualization functions
  - `utils.py`: Utility functions

- `notebooks/`: Analysis notebooks
- `results/`: Data and figure output
- `tests/`: Test suite


## References

1. Main Paper: [Critical temperature for the Ising model on a square lattice](https://arxiv.org/abs/1401.2000)
2. Wolff Algorithm: [Collective Monte Carlo Updating for Spin Systems](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.62.361)
3. Swendsen-Wang Algorithm: [Nonuniversal critical dynamics in Monte Carlo simulations](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.58.86)