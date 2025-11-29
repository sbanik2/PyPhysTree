# Interatomic Potential Fitting (Force Field Design)

This repository contains a demonstration version of **Interatomic Potential Fitting** using a **Physics-Informed Monte Carlo Tree Search (MCTS)** optimizer. The workflow automates the design of empirical force fields (e.g., Tersoff) by matching Density Functional Theory (DFT) data.


-----

## Codebase Overview

  * **`settings.yaml`**: The central configuration file. It controls **physical bounds**, **file paths**, **MCTS hyperparameters** (e.g., tree count, subsampling size), and the **selection strategy** (Energy vs. Force vs. Balanced).
  * **`phys_utils.py`**: Acts as the interface between the search algorithm and the physics engine (**LAMMPS**). It creates potential files dynamically, handles atomic data evaluation, and calculates error metrics (MSE/MAE).
  * **`fit.py`**: The **Step 1 (Global Search)** driver. It performs MCTS optimization using histogram-based subsampling to explore the high-dimensional parameter space efficiently.
  * **`2_filter.py`**: The **Step 2 (Validation)** script. It parses the global search logs, re-evaluates top candidates on the **test dataset**, and selects the best parameters based on a multi-objective strategy (Energy, Force, or Balanced).
  * **`3_simplex.py`**: The **Step 3 (Local Refinement)** driver. It takes the best candidate from the filtering stage and performs a **Nelder-Mead Simplex** optimization to fine-tune the parameters.
  * **`Analysis.ipynb`**: An interactive notebook used to visualize **parity plots** (DFT vs. Predicted), calculate final **RMSE/MAE metrics**, and analyze the performance of the developed potential.

-----

## Usage

The optimization is designed as a sequential pipeline. Ensure **LAMMPS** is installed and accessible via the ASE interface.

1.  **Configure Parameters:**
    Edit `settings.yaml` to define your element, parameter bounds, and dataset paths.

2.  **Step 1: Global Search (MCTS)**
    Run the tree search to explore the landscape and generate a pool of candidates.

    ```bash
    python fit.py
    ```

3.  **Step 2: Filter & Select**
    Sort the MCTS results, validate on the test set, and extract the best starting point.

    ```bash
    python 2_filter.py
    ```

4.  **Step 3: Local Refinement (Simplex)**
    Refine the selected candidate to reach the local optimum.

    ```bash
    python 3_simplex.py
    ```

5.  **Step 4: Analysis**
    Open `Analysis.ipynb` to view the final energy and force parity plots.

-----

## Demonstration

This workflow demonstrates the fitting of a **13-parameter Tersoff potential** for **Aluminum (Al)**. It utilizes a **histogram-subsampling** technique to handle large training datasets efficiently during the global search phase, followed by full-dataset validation during the refinement phase.



#### Results for the original application demonstrated in this article can also be downloaded from the Zenodo repository [link].





###