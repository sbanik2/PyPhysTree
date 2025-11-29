# Crystal Structure Prediction (CSP)

This repository contains a demonstration version of **Crystal Structure Prediction (CSP)** using a **Monte Carlo Tree Search (MCTS)**-based decision tree optimizer.

> **Note:** The full implementation of this method is available within the **CASTING framework**:
> [Banik et al., *npj Comput. Mater.* **9**, 177 (2023)](https://doi.org/10.1038/s41524-023-01128-y).

---

## Codebase Overview

* **`Crystal_search.py`**: The main driver script. It configures the material system (defining **composition**, **atomic count**, and **lattice bounds**) alongside the **MCTS hyperparameters** (tree population and evaluation budget) to guide the search through the physical energy landscape.
* **`Evaluator.py`**: Acts as the interface between the search algorithm and the physics engine (**LAMMPS**). It converts parameter vectors into atomic structures, runs energy minimizations, and returns scores.
* **`Perturb.py`**: Handles geometric logic, including enforcing **periodic boundary conditions**, verifying physical constraints (e.g., minimum interatomic distances), and generating valid random starting configurations.
* **`Analysis.ipynb`**: An interactive notebook used to visualize **optimization convergence history**, compare **lattice parameters** against reference data, and view the resulting atomic structures.

---

## Usage

1.  **Configure Parameters:**
    Open `Crystal_search.py` to set the material constraints and algorithm parameters.

2.  **Run the Optimization:**

```bash
python Crystal_search.py
````

-----

## Demonstration

The demo directory contains a demonstration example of **bulk silicon (Si)** structure optimization using **10% bounds** from the article.

#### Results for other applications of structure prediction demonstrated in this article can also be downloaded from the Zenodo repository [link].

