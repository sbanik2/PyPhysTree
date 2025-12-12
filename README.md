# PyPhysTree: Physics-Informed Tree Search Framework for High-Dimensional Computational Design


**PyPhysTree** is a Python framework for high-dimensional, continuous Global Optimization. It adapts **Monte Carlo Tree Search (MCTS)**—traditionally used in discrete applications for scientific discovery tasks in continuous spaces.

PyPhysTree is designed for **"black-box" design problems** where gradients are unavailable, evaluations are expensive (e.g., DFT, FEM), and the landscape is rugged or multimodal.

-----

## Key Implementation

This repository implements the algorithms described in *"Physics-Informed Tree Search for High-Dimensional Computational Design"*. It introduces three major deviations from standard MCTS to handle continuous physics problems:

1.  **Continuous Action Space** 
2.  **Directional Learning** 
3.  **Hierarchical Search**


-----

## 📦 Installation

The code is tested  with **Python 3.10+** or above.

While the core logic runs with standard pip packages, the **Crystal Structure (02)** and **Potential Fitting (03)** examples require **LAMMPS** and **MPI**. We strongly recommend using **Anaconda/Miniconda** to manage these non-Python dependencies.

```bash
git clone https://github.com/sbanik2/PyPhysTree.git
cd PyPhysTree

# 1. Create and Activate Conda Environment
# Note: 'lammps' and 'openmpi' are required for the Crystal Structure 
# and Potential Fitting.
conda create -n lmp -c conda-forge python=3.12 lammps "openmpi<5.0"
conda activate lmp

# 2. Install Python dependencies and the package
pip install -r requirements.txt
pip install .
````

-----

## ⚡ Quick Start

PyPhysTree can be used to optimize any scalar objective function. The main entry point is `run_optimisation_mcts`.

```python
import numpy as np
from lgtree import run_optimisation_mcts

# 1. Define your Objective Function
# Returns: (relaxed_parameters, score)
def my_objective(x):
    # Example: Simple Sphere function (Minimize x^2)
    score = np.sum(np.array(x)**2)
    return x, score 

# 2. Set Bounds (e.g., 5 dimensions)
lb = [-5.0] * 5
ub = [5.0] * 5

# 3. Run Physics-Informed MCTS
min_score, best_params = run_optimisation_mcts(
    objfunc=my_objective,
    logger=None,        # Optional logging
    lb=lb, ub=ub,       # Search bounds
    ntrees=5,           # Number of global trees
    top_k=2,            # Number of trees to keep for local refinement
    n_total_evals=1000, # Budget
    sampling_mode="logistic" # Options: "hypersphere" or "logistic"
)

print(f" Optimization Complete. Best Score: {min_score}")
```

-----

## 📂 Repository Structure & Applications

This repository is organized into the core source code and four distinct application examples demonstrated in the paper.

```text
PyPhysTree/
├── lgtree/                  <-- Core Source Code
│   ├── lgtree.py            # Main driver (Global/Local Batch logic)
│   ├── MCTS.py              # The Continuous MCTS Engine
│   └── utils.py             # Logistic Surrogate & Perturbation logic
├── examples/                <-- Paper Reproducibility & Demos
│   ├── 01_High_Dim_Benchmarks/
│   ├── 02_Crystal_Structure/
│   ├── 03_Potential_Fitting/
│   └── 04_Continuum_Design/
└── setup.py
```

### Tutorials & Paper Reproduction

Each folder below contains its own `README` and specific code to reproduce the results from the publication.

| Directory | Application Area | Paper Context |
| :--- | :--- | :--- |
| [**01\_High\_Dim\_Benchmarks**](./examples/01_High_Dim_Benchmarks) | **Math Optimization** | Reproduces **Table 1** & **Fig 3**. Benchmarks on F1-F23 (Rastrigin, Ackley, etc.) comparing MCTS vs. PSO/WOA. |
| [**02\_Crystal\_Structure**](./examples/02_Crystal_Structure) | **Materials Science** | Reproduces **Fig 4**. Includes $Au_{35}$ cluster optimization, Silicene polymorphism search, and Bulk Si lattice optimization. (Requires Conda env) |
| [**03\_Potential\_Fitting**](./examples/03_Potential_Fitting) | **Inverse Design** | Reproduces **Fig 5**. Fitting Tersoff potential parameters for Aluminum nanoclusters against DFT data. (Requires Conda env) |
| [**04\_Continuum\_Design**](./examples/04_Continuum_Design) | **Engineering** | Reproduces **Fig 6**. Constrained design of Pressure Vessels and Welded Beams. |


-----

## 📄 Citation

If you utilize **PyPhysTree** or the **Logistic Surrogate MCTS** strategy in your research, please cite:

```bibtex
@article{banik2025physics,
  title={Physics-Informed Tree Search for High-Dimensional Computational Design},
  author={Banik, Suvo and Loeffler, Troy D. and Chan, Henry and Manna, Sukriti and Yildiz, Orcun and Peterka, Tom and Sankaranarayanan, Subramanian},
  journal={arXiv preprint},
  year={2025}
}
```

```
```
