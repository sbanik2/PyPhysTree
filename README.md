# PyPhysTree: Physics-Informed Tree Search for High-Dimensional Computational Design

[cite_start]**PyPhysTree** is a Python framework implementing the algorithms described in the paper *"Physics-Informed Tree Search for High-Dimensional Computational Design"*[cite: 1, 3]. [cite_start]It extends Monte Carlo Tree Search (MCTS)—traditionally used for discrete decision-making in games—to continuous, high-dimensional scientific optimization tasks where gradients are unavailable or unreliable[cite: 14, 28].

[cite_start]This framework integrates population-level decision trees with surrogate-guided directional sampling, reward shaping, and hierarchical switching between global exploration and local exploitation to traverse complex non-convex landscapes[cite: 15].

---

## 📥 Data & Results Availability

The raw result data, optimized structures, logs, and plotting scripts used to generate the figures in the paper are hosted externally.

| Dataset | Format | Source |
| :--- | :--- | :--- |
| **Paper Results & Figures** | `.zip` | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://zenodo.org/record/XXXXXX) |

[cite_start]*> **Note:** To reproduce the exact results and figures found in the manuscript (e.g., Figures 3–6), please download the results zip file from Zenodo[cite: 819].*

---

## 🔧 Installation

To set up the environment, clone the repository and install the required dependencies.

```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/PyPhysTree.git](https://github.com/yourusername/PyPhysTree.git)
cd PyPhysTree

# 2. Install dependencies (numpy, scipy, scikit-learn, pyDOE)
pip install -r requirements.txt

# 3. (Optional) Install in editable mode if developing
pip install -e .
````

-----

## ⚙️ The Mechanism: Continuous Physics-Informed MCTS

Standard MCTS operates on discrete actions. **PyPhysTree** modifies this for continuous scientific design through three key mechanistic components:

1.  **Continuous Action Sampling**: Instead of discrete branches, we sample child nodes from a continuous design space.

      * [cite\_start]*Hypersphere Sampling*: Samples isotropically around a node[cite: 199].
      * [cite\_start]*Directional Logistic Surrogate*: Learns a "pseudo-gradient" from the search history to bias sampling toward promising regions[cite: 208, 211].

2.  **Adaptive Window Scaling**: As the tree deepens, the search radius ($r_{max}$) shrinks exponentially. [cite\_start]This forces the algorithm to transition from **Global Exploration** (shallow nodes) to **Local Exploitation** (deep nodes)[cite: 241, 250].

    $$s(depth) = b \cdot \exp(-a \cdot depth^2)$$

3.  **Hierarchical Global-Local Batching**:

      * [cite\_start]**Global Batch**: A population of trees explores the landscape to identify promising basins[cite: 262].
      * [cite\_start]**Local Batch**: The best candidates spawn "local trees" focused on fine-tuning, with aggressive window shrinking[cite: 268, 384].

[cite\_start]*(See Figure 2 in the paper for the visual schematic of the tree growth and depth scaling [cite: 164])*

-----

## 📂 Repository Structure

The repository is organized into the core package (`lgtree`) and distinct folders for each demonstration case discussed in the paper:

  * [cite\_start]**`lgtree/`**: The core source code[cite: 819]:

      * `MCTS.py`: The tree search logic (Selection, Expansion, Simulation, Backpropagation).
      * `utils.py`: Contains the `DirectionalLogisticSurrogate` and `Perturbate` classes.
      * `lgtree.py`: The `MCTSBatch` manager for Global/Local tree populations.

  * **`examples/`**:

      * [cite\_start]**`01_High_Dim_Benchmarks/`**: (Section 3.1) [cite: 392]
          * Reproduces results for the 23 mathematical benchmark functions (e.g., Rosenbrock, Ackley).
          * Contains `Benchmark_Demo.ipynb`.
      * [cite\_start]**`02_Crystal_Structure/`**: (Section 3.2) [cite: 433]
          * Demonstrations for Au nanoclusters (0D), Silicene polymorphs (2D), and Bulk Silicon (3D).
      * [cite\_start]**`03_Potential_Fitting/`**: (Section 3.3) [cite: 601]
          * Workflows for inverse design of Tersoff potential parameters.
      * [cite\_start]**`04_Continuum_Design/`**: (Section 3.4) [cite: 754]
          * Engineering design optimization for Welded Beams and Pressure Vessels.

-----

## 💻 Usage & Demonstration

To run an optimization, you need to define an objective function and bounds. [cite\_start]Below is a minimal example using the high-dimensional **Rosenbrock function** (F5)[cite: 404].

You can run the interactive notebook at `examples/01_High_Dim_Benchmarks/Benchmark_Demo.ipynb` or use the script below:

```python
import numpy as np
import logging
from lgtree.lgtree import run_optimisation_mcts

# 1. Define Objective Function
def rosenbrock(x):
    # [cite_start]F5 from the paper [cite: 404]
    x = np.asarray(x)
    value = np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    return x, value

# 2. Setup Configuration
dim = 30
lb = np.full(dim, -30.0)
ub = np.full(dim, 30.0)

# 3. Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCTS_Demo")

# 4. Run Optimization
# [cite_start]- 'logistic' mode enables the learned directional surrogate [cite: 211]
# [cite_start]- 'ntrees' sets the population size for the batch [cite: 262]
min_score, best_params = run_optimisation_mcts(
    objfunc=rosenbrock,
    logger=logger,
    lb=lb,
    ub=ub,
    ntrees=20,
    top_k=1,
    niterations_global=10,
    niterations_local=20,
    sampling_mode="logistic",
    verbose=True
)

print(f"Final Score: {min_score}")
```

-----

## 📜 Citation

If you use this code or data in your research, please cite the original paper:

```bibtex
@article{banik2025physics,
  title={Physics-Informed Tree Search for High-Dimensional Computational Design},
  author={Banik, Suvo and Loeffler, Troy D. and Chan, Henry and Manna, Sukriti and Yildiz, Orcun and Peterka, Tom and Sankaranarayanan, Subramanian},
  journal={arXiv preprint},
  year={2025},
  note={Argonne National Laboratory}
}
```

-----

## 📝 Acknowledgments

[cite\_start]This work was supported by the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences, Data, Artificial Intelligence, and Machine Learning at DOE Scientific User Facilities program under Award Number 34532 (Digital Twins)[cite: 808].

```
```
