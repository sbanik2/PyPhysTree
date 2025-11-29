import numpy as np
import pandas as pd
from pymatgen.core import Lattice
from collections import Counter
from random import random, shuffle, choice

def gel_latt_coords(parameters):
    """Splits flat vector into lattice (6) and coords (3*N)."""
    lattice = parameters[:6]
    coords = parameters[6:]
    coords = coords.reshape(int(coords.shape[0]/3), 3)
    return lattice, coords

def impose_pbc_and_bounds(parameters):
    """
    Enforces physical boundary conditions separately for Lattice and Atoms.
    - Lattice: Clipped to [0, 1] (Hard bounds)
    - Atoms: Wrapped to [0, 1] (Periodic Boundary Conditions)
    """
    # 1. Split
    lattice, coords = gel_latt_coords(parameters)

    # 2. Lattice Action: Rigid Clipping
    # Lattice parameters cannot "wrap around"; they must stay within defined bounds.
    lattice = np.clip(lattice, 0.0, 1.0)

    # 3. Atom Action: Periodic Wrapping (PBC)
    # Atoms leaving the box at 1.1 should enter at 0.1.
    coords = coords - np.floor(coords)
    
    # 4. Recombine
    return np.concatenate((lattice, coords.flatten()))

def createRandomData(constrains, trials=1000):
    """Generates a valid starting point satisfying all constraints."""
    factor = sum(list(constrains["composition"].values()))
    if constrains["nAtomRange"][0] == constrains["nAtomRange"][1]:
        natoms = constrains["nAtomRange"][0]
    else:
        natoms = choice(np.arange(constrains["nAtomRange"][0], constrains["nAtomRange"][1], factor))
    
    species = []
    for key in constrains["composition"].keys():
        nkey = int((constrains["composition"][key]/sum(list(constrains["composition"].values())))*natoms)
        for _ in range(nkey):
            species.append(key)
    
    count = 0
    while count < trials:
        shuffle(species)
        # Generate random vector [0, 1]
        parameters = np.array([random() for _ in range(3*natoms+6)])
        
        structData = {"parameters": parameters, "species": species}
        
        # Check if the random structure is physically valid
        if check_constrains(structData, constrains):
            return parameters, species
        count += 1
    
    raise ValueError("Could not generate valid initial structure within trial limit.")

def check_constrains(structData, constrains, verbose=False):
    """
    Checks constraints. Handles Lattice and Atoms SEPARATELY.
    """
    parameters = structData["parameters"].copy()
    species = structData["species"].copy()
    lattice, coords = gel_latt_coords(parameters)

    composition = constrains["composition"]
    latVecRange = constrains["latVecRange"]
    latAngRange = constrains["latAngRange"]
    r = constrains["r"]

    # --- 1. Lattice Constraints ---
    # Map [0,1] to real units
    ub_lat = np.array([latVecRange[1]] * 3 + [latAngRange[1]] * 3)
    lb_lat = np.array([latVecRange[0]] * 3 + [latAngRange[0]] * 3)
    real_lattice = lb_lat + (ub_lat - lb_lat) * lattice

    try:
        latt = Lattice.from_parameters(
            a=real_lattice[0], b=real_lattice[1], c=real_lattice[2],
            alpha=real_lattice[3], beta=real_lattice[4], gamma=real_lattice[5]
        )
    except ValueError:
        if verbose: print("Failed: Invalid Lattice construction")
        return False

    # Volume Check
    LatticeVolume = latt.volume
    specieCount = dict(Counter(species))
    mintargetVolume = 0.0
    for key in specieCount:
        rad = r.loc[key, key] * 0.5
        mintargetVolume += (4 / 3) * np.pi * (rad ** 3) * specieCount[key]

    if LatticeVolume < mintargetVolume:
        return False

    # --- 2. Atom Constraints ---
    # Distance Matrix Calculation
    M = np.array(latt.matrix)
    D = DistanceMatrix(coords, M) 
    np.fill_diagonal(D, 1e300)
    
    DF = pd.DataFrame(D, columns=species, index=species)
    elements = r.columns.tolist()

    # Pair interaction check
    for s1 in elements:
        for s2 in elements:
            if s1 in DF.index and s2 in DF.index:
                sub_df = DF.loc[s1, s2]
                vals = sub_df.values if hasattr(sub_df, "values") else np.array([sub_df])
                cutoff = r.loc[s1, s2]
                if (vals < cutoff).any():
                    if verbose: print(f"Failed: too short {s1}-{s2}")
                    return False

    return True

def DistanceMatrix(frac_coordinates, M):
    """Calculates distance matrix with Minimum Image Convention (PBC)."""
    a, b, c = frac_coordinates[:,0], frac_coordinates[:,1], frac_coordinates[:,2]

    def getDist(mat):
        n, m = np.meshgrid(mat, mat)
        dist = m - n
        dist -= np.rint(dist) # Enforce PBC on distance calculation
        return dist

    da, db, dc = getDist(a), getDist(b), getDist(c)
    
    DX = M[0][0]*da + M[1][0]*db + M[2][0]*dc
    DY = M[0][1]*da + M[1][1]*db + M[2][1]*dc
    DZ = M[0][2]*da + M[1][2]*db + M[2][2]*dc

    return np.sqrt(np.square(DX) + np.square(DY) + np.square(DZ))