import numpy as np
import random
import logging
from collections import defaultdict
from ase.io import read
from lgtree.lgtree import run_optimisation_mcts 
import phys_utils as phys

# Load Settings
CFG = phys.CFG
MCTS_CFG = CFG['mcts']

TRAIN_FILE = CFG['files']['train']
DUMP_FILE = CFG['files']['dump']
SUBSAMPLE_N = MCTS_CFG['subsample_n']
NBINS = MCTS_CFG['nbins']

# --- PRE-LOAD DATA & SETUP SUBSAMPLER ---
print(f"Loading {TRAIN_FILE}...")
atoms_list = read(TRAIN_FILE, index=":")

# Pre-calculate DFT energies once
print("Pre-calculating DFT reference energies...")
for atoms in atoms_list:
    e_val = atoms.get_potential_energy() / len(atoms)
    f_val = atoms.get_forces().flatten()
    atoms.info['energy_dft'] = e_val
    atoms.info['forces_dft'] = f_val

energies = np.array([a.info['energy_dft'] for a in atoms_list])

def subsample_atoms():
    """Histogram-based subsampling logic"""
    hist, bin_edges = np.histogram(energies, bins=NBINS)
    bin_indices = np.digitize(energies, bin_edges[:-1], right=False)
    
    bins = defaultdict(list)
    for idx, bin_id in enumerate(bin_indices):
        bins[bin_id].append(atoms_list[idx])

    subsampled = []
    total = sum(len(v) for v in bins.values())

    for bin_id, grp in bins.items():
        n_bin = int(round(len(grp) / total * SUBSAMPLE_N))
        if len(grp) < n_bin: n_bin = len(grp)
        subsampled.extend(random.sample(grp, n_bin))

    # Adjust exact count
    if len(subsampled) > SUBSAMPLE_N:
        subsampled = random.sample(subsampled, SUBSAMPLE_N)
    elif len(subsampled) < SUBSAMPLE_N:
        deficit = SUBSAMPLE_N - len(subsampled)
        subsampled.extend(random.sample(atoms_list, deficit))

    return subsampled

# --- OBJECTIVE FUNCTION ---
def obj_function(x):
    # 1. Get Subsample
    sublist = subsample_atoms()
    
    # 2. Evaluate
    mse_e, mse_f = phys.evaluate_on_atoms(x, sublist)
    
    # 3. Weighted Score
    total_score = (mse_e * 1000 + mse_f * 1) * 10
    
    # 4. Dump detailed metrics
    params_str = ",".join(map(str, x))
    with open(DUMP_FILE, "a") as f:
        f.write(f"{params_str} | {total_score} | {mse_e} | {mse_f}\n")
        
    return list(x), total_score

if __name__ == "__main__":
    open(DUMP_FILE, 'w').close() # Clear dump

    logging.basicConfig(filename='mcts.log', level=logging.INFO, filemode='w')
    logger = logging.getLogger('MCTS')
    logistic_settings = {"rbf_count": 10, "history_window": 100}

    print(f"Starting MCTS (Subsampling N={SUBSAMPLE_N})...")
    print(f"Hyperparameters: Trees={MCTS_CFG['ntrees']}, Alpha={MCTS_CFG['alpha']}, Drop={MCTS_CFG['drop_factor']}")
    
    run_optimisation_mcts(
        objfunc=obj_function,
        logger=logger,
        lb=phys.LB,
        ub=phys.UB,
        n_total_evals=MCTS_CFG['eval_cutoff'],
        
        # --- Tuned Hyperparameters from YAML ---
        ntrees=MCTS_CFG['ntrees'],
        top_k=MCTS_CFG['top_k'],
        a_min=MCTS_CFG['a_min'],
        a_max=MCTS_CFG['a_max'],
        b0=MCTS_CFG['b_null'],
        explore_constant_min=MCTS_CFG['explore_min'],
        alpha=MCTS_CFG['alpha'],
        aggressive_drop_factor=MCTS_CFG['drop_factor'],
        
        target=-4.0,
        sampling_mode="logistic",
        logistic_kwargs=logistic_settings,
        verbose=True
    )