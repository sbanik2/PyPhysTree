import yaml
import numpy as np
from ase.calculators.lammpslib import LAMMPSlib

# --- LOAD CONFIGURATION ---
with open("settings.yaml", "r") as f:
    CFG = yaml.safe_load(f)

# Unpack convenient variables
ELEMENT = CFG['element']
LB = np.array(CFG['bounds']['lb'])
UB = np.array(CFG['bounds']['ub'])

def get_calculator(x):
    """Creates the LAMMPS calculator with specific parameters x."""
    # Construct Tersoff line dynamically
    params_str = " ".join(map(str, x))
    tersoff_line = f"{ELEMENT} {ELEMENT} {ELEMENT} 1.0 {params_str}"
    
    with open("generated.tersoff", "w") as f:
        f.write(tersoff_line + "\n")

    lmpcmds = [
        "pair_style tersoff",
        f"pair_coeff * * generated.tersoff {ELEMENT}",
        "neighbor 2.0 nsq",
        "neigh_modify delay 0 every 1 check yes",
    ]
    
    # keep_alive=True speeds up re-runs significantly
    return LAMMPSlib(lmpcmds=lmpcmds, log_file=None, keep_alive=True)

def evaluate_on_atoms(x, atoms_sublist):
    """
    Evaluates parameter set x on a specific list of atoms.
    Expects atoms to have .info['energy_dft'] and .info['forces_dft'] pre-calculated.
    """
    calc = get_calculator(x)
    
    pred_E, pred_F = [], []
    target_E, target_F = [], []
    
    for atoms in atoms_sublist:
        atoms.calc = calc
        try:
            p_e = atoms.get_potential_energy() / len(atoms)
            p_f = atoms.get_forces().flatten()
        except Exception:
            # Return high error if LAMMPS crashes (e.g., unstable parameters)
            return 1e9, 1e9 

        pred_E.append(p_e)
        pred_F.append(p_f)
        # Fetch pre-calculated DFT refs
        target_E.append(atoms.info['energy_dft'])
        target_F.append(atoms.info['forces_dft'])

    pred_E = np.array(pred_E)
    target_E = np.array(target_E)
    
    mse_e = np.mean((target_E - pred_E) ** 2)
    
    if len(pred_F) > 0:
        pred_F_flat = np.concatenate(pred_F)
        target_F_flat = np.concatenate(target_F)
        mse_f = np.mean((target_F_flat - pred_F_flat) ** 2)
    else:
        mse_f = 0.0

    return mse_e, mse_f