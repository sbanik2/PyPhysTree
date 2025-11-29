import numpy as np
from scipy.optimize import minimize
from ase.io import read
import phys_utils as phys

CFG = phys.CFG
STRATEGY = CFG['simplex']['strategy']
DUMP_SIMPLEX = CFG['files']['dump_simplex'] 
MAX_EVALS = CFG['simplex']['max_evals'] # Controlled by YAML

# Select Parameter File for Initial Guess
if STRATEGY == "energy": PARAM_FILE = CFG['files']['params_energy']
elif STRATEGY == "force": PARAM_FILE = CFG['files']['params_force']
else: PARAM_FILE = CFG['files']['params_balanced']

TRAIN_FILE = CFG['files']['train']
VAR_PERCENT = CFG['simplex']['variation_percent']

if __name__ == "__main__":
    # Clear previous simplex dump
    open(DUMP_SIMPLEX, 'w').close()

    print(f"--- STEP 3: Simplex ({STRATEGY.upper()}) ---")
    print(f"Max Evaluations: {MAX_EVALS}")
    
    try: x0 = np.loadtxt(PARAM_FILE)
    except: exit(f"Error loading {PARAM_FILE}")

    print("Loading Training Data...")
    train_atoms = read(TRAIN_FILE, index=":")
    
    # Pre-calc DFT Refs
    for atoms in train_atoms:
        atoms.info['energy_dft'] = atoms.get_potential_energy()/len(atoms)
        atoms.info['forces_dft'] = atoms.get_forces().flatten()

    # Setup Bounds
    lb_local = np.maximum(x0 * (1 - VAR_PERCENT/100.0), phys.LB)
    ub_local = np.minimum(x0 * (1 + VAR_PERCENT/100.0), phys.UB)
    for i in range(len(x0)):
        if lb_local[i] > ub_local[i]: lb_local[i], ub_local[i] = ub_local[i], lb_local[i]

    # Objective Function with Logging
    def simplex_obj(x):
        # Bounds Check
        if np.any(x < lb_local) or np.any(x > ub_local): return 1e9
        
        # Evaluate
        mse_e, mse_f = phys.evaluate_on_atoms(x, train_atoms)
        
        # Formula: (MSE_E*1000 + MSE_F*1) * 10
        total_score = (mse_e * 1000 + mse_f * 1) * 10
        
        # Log to simplex dump file
        with open(DUMP_SIMPLEX, "a") as f:
            f.write(",".join(map(str, x)) + f" | {total_score}\n")
            
        return total_score

    print(f"Running Simplex (logging to {DUMP_SIMPLEX})...")
    
    # Run Minimize with maxfev control
    res = minimize(
        simplex_obj, 
        x0, 
        method="Nelder-Mead", 
        options={"maxfev": MAX_EVALS, "disp": True}
    )
    
    print(f"Simplex Done. Final Score: {res.fun:.4f}")
    
    # Save Final Parameters
    output_filename = f"final_params_{STRATEGY}.txt"
    np.savetxt(output_filename, res.x)
    print(f"Parameters saved to {output_filename}")