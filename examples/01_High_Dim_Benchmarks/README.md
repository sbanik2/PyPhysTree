# High-Dimensional Benchmark Functions

This demonstration evaluates the performance of `PyPhysTree`'s **MCTS optimization framework** on a comprehensive suite of 23 canonical test functions (F1–F23). These functions range from simple unimodal to highly complex, multimodal landscapes designed to trap standard optimizers.

---

## Overview

* **Objective**: Find the global minimum of various mathematical landscapes, demonstrating **convergence speed**, **precision**, and **robustness**.
* **Search Space**: Scalable dimensions (up to 30D) and fixed low-dimensional (2D-6D) deceptive landscapes.
* **Evaluator**: Analytical functions (e.g., Rosenbrock, Rastrigin, Ackley) defined in Python.
* **Optimizer**: Physics-Informed **MCTS** with adaptive window scaling and population-based hierarchical batching.

---

## Code Structure

The demonstration logic is split into three files:

1.  **`Benchmark.ipynb`**: The interactive Jupyter Notebook driver.
    * Loads configurations.
    * Initializes the MCTS optimizer.
    * Visualizes 2D landscapes using 3D surface plots.
    * Reports the best scores and parameters found.
2.  **`Trial_functions.py`**: Contains the definitions for all 23 benchmark functions (F1–F23).
    * Includes a `@log_execution` decorator that automatically saves every evaluation to `dumpfile.dat`.
3.  **`mcts_benchmarks.yaml`**: The configuration file defining the physics/search bounds, dimensionality, and MCTS hyperparameters for each specific function.

## Demonstration
The demo directory contains a simple run example: Running the benchmark for **F2 (Schwefel 2.22) function** on 30 dimensions.

#### The performance results for each individual function used in the article can be downloaded from the Zenodo repository [link].