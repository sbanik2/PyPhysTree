import numpy as np
import random
import logging
import math
# Import the main driver from your current lgtree repository
from lgtree.lgtree import run_optimisation_mcts 
from itertools import product

# -------------------- Constants --------------------
EVAL_CUTOFF = 15000  # Max allowed evaluations (total across all trees)
DIM = 4
# [T_s, T_h, R, L] (Shell Thickness, Head Thickness, Inner Radius, Cylinder Length)
# Note: Bounds from the paper (Figure 6) often treat these as discrete or continuous.
# We use continuous float bounds here.
LB = np.array([0.0, 0.0, 10.0, 10.0], dtype=float)    # Lower bounds
UB = np.array([99.0, 99.0, 200.0, 200.0], dtype=float)  # Upper bounds

# Optimization Hyperparameters for MCTS
NTREES = 20          # Number of global search trees
ALPHA = 0.08         # Exponent for adaptive window scaling in local search
DROP_FACTOR = 0.98   # Aggressive decay factor for non-improving steps
A_MIN, A_MAX = 0.008, 0.08 # Range for depth-scaling parameter 'a'
EXPLORE_CONSTANT_MIN = 5e-1 # Minimum UCB constant for exploitation
TOP_K = 4            # Number of best global trees to promote to local search

# -------------------- Objective Function --------------------

# Constants
pi = np.pi


def pressure_vessel_constraints(x):
    """Constraints: g_i(x) <= 0"""
    x1, x2, x3, x4 = x  # [T_s, T_h, R, L]

    g = np.zeros(4)
    g[0] = -x1 + 0.0193 * x3                                  # g1(x)
    g[1] = -x2 + 0.00954 * x3                                 # g2(x)
    g[2] = -pi * x3**2 * x4 - (4.0 / 3.0) * pi * x3**3 + 1296000  # g3(x)
    g[3] = x4 - 240                                           # g4(x)

    return g

def pressure_vessel_objective(x):
    """Objective: Minimize cost if feasible, else apply large penalty"""
    x1, x2, x3, x4 = x  # [T_s, T_h, R, L]
    constraints = pressure_vessel_constraints(x)

    # Penalize if any constraint or bound is violated
    if np.any(constraints > 0) or np.any(x < LB) or np.any(x > UB):
        val = 1e10
    else:
        val = (
            0.6224 * x1 * x3 * x4 +
            1.7781 * x2 * x3**2 +
            3.1661 * x1**2 * x4 +
            19.84 * x1**2 * x3
        )

    # Logging
    with open("dumpfile.dat", "a") as outfile:
        outfile.write(f"{x1:.6f} {x2:.6f} {x3:.6f} {x4:.6f} | {val:.6f} | " +
                      " ".join(f"{g:.2e}" for g in constraints) + "\n")

    return val


def pressure_vessel_wrapper(x):
    """
    Adapter function for the MCTS framework.
    MCTS requires objective functions to return (relaxed_parameters, score).
    Since this problem is analytic and doesn't involve local relaxation (like LAMMPS),
    we return the input parameters 'x' unchanged.
    """
    score = pressure_vessel_objective(x)
    return list(x), score

# -------------------- Seeding --------------------
seed_value = random.randint(1, 100000)
random.seed(seed_value)
np.random.seed(seed_value)

with open("used_seeds.txt", "a") as seedfile:
    seedfile.write(f"{seed_value}\n")

# -------------------- Logging --------------------
logging.basicConfig(filename='mcts_tuning_log.txt', level=logging.INFO, filemode='w')
logger = logging.getLogger('MCTSLogger')

# -------------------- Run Optimization --------------------
if __name__ == "__main__":
    
    print("Starting Pressure Vessel Optimization (4D Constrained)...")
    
    # Configure Logistic Regression settings for the surrogate.
    # We use "logistic" mode which employs the Directional Logistic Surrogate
    # to bias sampling toward promising directions.
    logistic_settings = {
        "rbf_count": 10,
        "history_window": 100
    }

    minscore, best_parameter = run_optimisation_mcts(
        objfunc=pressure_vessel_wrapper,
        logger=logger,
        lb=LB,
        ub=UB,
        ntrees=NTREES,
        top_k=TOP_K,
        a_min=A_MIN,
        a_max=A_MAX,
        b0=0.5, # Initial window size factor
        explore_constant_min=EXPLORE_CONSTANT_MIN,
        
        # Iteration controls
        n_total_evals=EVAL_CUTOFF,
        
        # Adaptive Parameters
        alpha=ALPHA,
        target=5000.0, # Approximate expected cost (used for adaptive shrinking)
        aggressive_drop_factor=DROP_FACTOR,
        
        # Sampling Strategy
        sampling_mode="logistic", # Use the physics-informed directional sampler
        logistic_kwargs=logistic_settings,
        
        verbose=True
    )

    print("\n" + "="*35)
    print(f"Optimization Finished.")
    print(f"Best Cost: {minscore:.4f}")
    print(f"Best Parameters [T_s, T_h, R, L]: {best_parameter}")
    print("="*35)