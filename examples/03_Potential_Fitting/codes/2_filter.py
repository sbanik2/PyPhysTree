import numpy as np
import logging
from ase.io import read
import phys_utils as phys

CFG = phys.CFG
DUMP_FILE = CFG['files']['dump']
TEST_FILE = CFG['files']['test']
POOL_SIZE = CFG['filter']['pool_size']

# Weights for "Balanced" calculation (MAE space)
W_E = 1.0 
W_F = 1.0 

def load_shortlist_from_dump():
    """Reads dumpfile (MSE scores) and returns top N candidates."""
    candidates = []
    try:
        with open(DUMP_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 2: continue
                
                params = list(map(float, parts[0].split(",")))
                obj_score = float(parts[1]) 
                candidates.append({'params': params, 'train_obj': obj_score})
    except FileNotFoundError:
        return []

    # Sort by Training Objective (MSE based) and take top N
    candidates.sort(key=lambda x: x['train_obj'])
    return candidates[:POOL_SIZE]

def calculate_mae_on_test(x, test_atoms):
    """
    Custom evaluation function for the Filter step.
    Calculates Mean Absolute Error (MAE) in meV.
    """
    calc = phys.get_calculator(x)
    
    pred_E, pred_F = [], []
    target_E, target_F = [], []
    
    for atoms in test_atoms:
        atoms.calc = calc
        try:
            p_e = atoms.get_potential_energy() / len(atoms)
            p_f = atoms.get_forces().flatten()
        except Exception:
            return 1e9, 1e9 # Penalty

        pred_E.append(p_e)
        pred_F.append(p_f)
        target_E.append(atoms.info['energy_dft'])
        target_F.append(atoms.info['forces_dft'])

    # Vectorize
    pred_E = np.array(pred_E)
    target_E = np.array(target_E)
    
    # --- MAE Calculation (eV) ---
    mae_e_ev = np.mean(np.abs(target_E - pred_E))
    
    if len(pred_F) > 0:
        pred_F_flat = np.concatenate(pred_F)
        target_F_flat = np.concatenate(target_F)
        mae_f_ev = np.mean(np.abs(target_F_flat - pred_F_flat))
    else:
        mae_f_ev = 0.0

    # --- Convert to meV ---
    return mae_e_ev * 1000.0, mae_f_ev * 1000.0

if __name__ == "__main__":
    print(f"--- STEP 2: Multi-Objective Selection (Metric: MAE in meV) ---")
    
    # 1. Get Shortlist (based on Fit MSE)
    print(f"Reading dump and selecting top {POOL_SIZE} candidates...")
    shortlist = load_shortlist_from_dump()
    
    if not shortlist:
        print("Error: No candidates found.")
        exit()

    # 2. Load Test Data
    print(f"Loading Test Data: {TEST_FILE}")
    test_atoms = read(TEST_FILE, index=":")
    for atoms in test_atoms:
        atoms.info['energy_dft'] = atoms.get_potential_energy() / len(atoms)
        atoms.info['forces_dft'] = atoms.get_forces().flatten()

    # 3. Re-Evaluate Shortlist using MAE
    print(f"{'ID':<4} | {'MAE_E (meV/atom)':<18} | {'MAE_F (meV/A)':<18} | {'Balanced Score':<15}")
    print("-" * 65)
    
    evaluated_pool = []

    for i, cand in enumerate(shortlist):
        # Calculate MAE in meV
        mae_e, mae_f = calculate_mae_on_test(cand['params'], test_atoms)
        
        # Balanced Score in MAE space
        balanced_score = (mae_e * W_E) + (mae_f * W_F)
        
        entry = {
            'params': cand['params'],
            'mae_e': mae_e,
            'mae_f': mae_f,
            'score': balanced_score
        }
        evaluated_pool.append(entry)
        
        print(f"{i:<4} | {mae_e:.4f}             | {mae_f:.4f}             | {balanced_score:.4f}")

    # 4. Select The 3 Champions based on MAE metrics
    best_energy = min(evaluated_pool, key=lambda x: x['mae_e'])
    best_force = min(evaluated_pool, key=lambda x: x['mae_f'])
    best_balanced = min(evaluated_pool, key=lambda x: x['score'])

    print("\n--- SELECTION RESULTS (MAE in meV) ---")
    print(f"[Best Energy]   MAE_E: {best_energy['mae_e']:.4f} (Force: {best_energy['mae_f']:.4f})")
    print(f"[Best Force]    MAE_F: {best_force['mae_f']:.4f} (Energy: {best_force['mae_e']:.4f})")
    print(f"[Best Balanced] Score: {best_balanced['score']:.4f}")

    # 5. Save Files
    np.savetxt(CFG['files']['params_energy'], best_energy['params'])
    np.savetxt(CFG['files']['params_force'], best_force['params'])
    np.savetxt(CFG['files']['params_balanced'], best_balanced['params'])
    
    print("\nSaved 3 parameter files.")