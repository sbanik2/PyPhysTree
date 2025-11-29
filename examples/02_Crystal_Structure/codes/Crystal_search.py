import numpy as np
import logging
import pandas as pd

# New library imports
from lgtree.lgtree import MCTSBatch, run_global_tree_batch, run_local_tree_batch
from Perturb import createRandomData, check_constrains, impose_pbc_and_bounds
from Evaluator import LammpsEvaluator 

# -------------------- Configuration --------------------
r = pd.DataFrame(np.array([1.5]), columns=["Si"], index=["Si"])

constrains = {     
    "composition": {"Si": 1},
    "nAtomRange": [8, 8],
    "latVecRange": [3.7, 7],     
    "latAngRange": [63, 117],     
    "r": r,
    "shape": "bulk"
}

# Hyperparameters
NTREES = 10
EVAL_CUTOFF = 20000
TOP_K = 5
A_MIN, A_MAX = 0.005, 0.1
EXPLORE_CONST = 5e-1

# Logging
logging.basicConfig(filename='mcts_tuning_log.txt', level=logging.INFO, filemode='w')
logger = logging.getLogger('MCTSLogger')

# -------------------- Evaluator Setup --------------------
# Generate one instance to get dimensionality
temp_params, STATIC_SPECIES = createRandomData(constrains, trials=1000)
DIM = len(temp_params)

lammps_runner = LammpsEvaluator(
    constrains,
    "pair_style sw",
    "pair_coeff * * Si.sw Si",
    shape='bulk',
    minimize=True
).Runlammps

def objective_wrapper(flat_parameters):
    """
    Evaluates a candidate vector while enforcing separate physical actions.
    """
    
    # 1. SEPARATE ACTION: Apply different boundary rules
    # Lattice -> Clipped, Atoms -> Wrapped (PBC)
    # This step modifies the vector 'physically' before we even check constraints.
    physical_parameters = impose_pbc_and_bounds(flat_parameters)

    struct_data = {
        "parameters": physical_parameters,
        "species": STATIC_SPECIES
    }

    # 2. CONSTRAINT CHECK
    if not check_constrains(struct_data, constrains):
        # Return the corrected parameters even if invalid, so MCTS learns boundaries
        return physical_parameters, 1e9

    # 3. RELAXATION (LAMMPS)
    energy, relaxed_data, _ = lammps_runner(struct_data)

    # 4. FEEDBACK LOOP
    # We return 'relaxed_data["parameters"]' which contains the relaxed position.
    # This tells MCTS that the particle moved (relaxed) or wrapped (PBC).
    return relaxed_data["parameters"], energy

# -------------------- Initialization --------------------
LB = np.zeros(DIM)
UB = np.ones(DIM)

head_points = []
print("Generating initial seed structures...")
for _ in range(NTREES):
    p, _ = createRandomData(constrains)
    head_points.append(p)

# Global Batch Config
a_values = np.linspace(A_MIN, A_MAX, NTREES)
config_dict = {}

for i in range(NTREES):
    config_dict[i] = {
        "a": float(a_values[i]),
        "b": 0.5,
        "root_data": head_points[i],
        "head_data": None,
        "lower_bounds": LB,
        "upper_bounds": UB
    }

# -------------------- Execution --------------------

# 1. Global Search
global_batch = MCTSBatch(
    objfunc=objective_wrapper,
    config_dict=config_dict,
    logger=logger,
    nplayouts=5,
    explore_constant_max=1e10, 
    explore_constant_min=EXPLORE_CONST,
    max_depth=12,
    niterations=20,
    sampling_mode="hypersphere"
)

print("Starting Global Search...")
top_stats, total_evals = run_global_tree_batch(global_batch, top_k=TOP_K)

# 2. Local Refinement
local_config_dict = {}
for idx, res in top_stats.items():
    local_config_dict[idx] = {
        "a": res["a"],
        "b": res["b"],
        "root_data": res["parameter"],
        "head_data": None,
        "lower_bounds": res["lb"],
        "upper_bounds": res["ub"],
    }

local_batch = MCTSBatch(
    objfunc=objective_wrapper,
    config_dict=local_config_dict,
    logger=logger,
    nplayouts=5,
    max_depth=100,
    niterations=10, 
    explore_constant_max=0.0,
    explore_constant_min=0.0
)


print("Starting Local Refinement...")
final_stats, final_evals, minscore, best_param = run_local_tree_batch(
    local_batch=local_batch,
    top_stats=top_stats,
    objfunc=objective_wrapper,
    logger=logger,
    n_iterations=EVAL_CUTOFF,
    total_evals_sofar=total_evals,
    target = -5,
)

print(f"Optimization Finished. Best Energy: {minscore}")