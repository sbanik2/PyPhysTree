#!/usr/bin/env python
# coding: utf-8

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

from lammps import lammps
from ase.io import read
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData

from Perturb import gel_latt_coords

import warnings


# ------------------- LAMMPS Evaluator Class -------------------

class LammpsEvaluator:
    """
    Performs energy evaluations using LAMMPS for a given structure.
    """

    def __init__(self, constrains, pair_style, pair_coeff,
                 shape="cluster", minimize=True, pad=20, P=None, calcfp=False):
        self.constrains = constrains
        self.pair_style = pair_style
        self.pair_coeff = pair_coeff
        self.shape = shape
        self.minimize = minimize
        self.pad = pad
        self.P = P
        self.calcfp = calcfp

        # Initialize LAMMPS instance
        self.lmp = lammps(cmdargs=["-screen", "log.screen"])

    def Runlammps(self, structData):
        """
        Evaluates the given structure using LAMMPS and returns energy,
        optimized parameters, and fingerprint descriptor (placeholder).
        """
        struct = StructureFrmParams(
            structData, self.constrains, shape=self.shape, pad=self.pad
        )

        # Write LAMMPS data file
        LammpsData.from_structure(struct, atom_style="atomic").write_file("in.data")

        lmp = self.lmp
        lmp.command("clear")
        lmp.command("dimension 3")
        lmp.command("box tilt large")
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command("neighbor 2.0 bin")
        lmp.command("atom_modify map array sort 0 0")
        lmp.command("boundary p p p")
        lmp.command("read_data in.data")
        lmp.command(self.pair_style)
        lmp.command(self.pair_coeff)
        lmp.command("thermo 1000")
        lmp.command("thermo_style custom step etotal atoms vol")
        lmp.command("thermo_modify format float %5.14g")
        lmp.command("variable potential equal pe/atoms")
        lmp.command("neigh_modify one 5000 delay 0 every 1 check yes")

        # Apply box relaxation if needed
        if self.P is not None:
            lmp.command(f"fix 1 all box/relax iso {self.P} vmax 0.1")

        # Energy minimization
        if self.minimize:
            lmp.command("run 0 pre no")
            tmp_eng = lmp.extract_variable("potential", None, 0)

            if math.isnan(tmp_eng) or math.isinf(tmp_eng):
                lmp.command("write_data min.geo")
            else:
                lmp.command("minimize 1.0e-8 1.0e-8 10000 10000")
                lmp.command("write_data min.geo")
        else:
            lmp.command("run 0 pre no")

        # Extract energy
        energy = lmp.extract_variable("potential", None, 0)
        if math.isnan(energy) or math.isinf(energy):
            energy = 1e300

        # Extract structure from minimized geometry
        minStruct = LammpsData.from_file("min.geo", atom_style="atomic").structure
        outData = ParamsfromStruct(
            minStruct,
            self.constrains,
            energy=energy,
            write_file="dumpfile.dat",
            shape=self.shape,
            pad=self.pad
        )

        return energy, outData, [0]  # [0] placeholder for fingerprint




def StructureFrmParams(structData, constrains, shape='bulk', pad=20):
    """
    Reconstruct a pymatgen Structure from normalized parameters.

    Args:
        structData (dict): Contains "parameters" (np.array) and "species" (list of str).
        constrains (dict): Dictionary with "latVecRange" and "latAngRange" ranges.
        shape (str): Structure type, e.g., 'bulk', 'sheet', or 'cluster'.
        pad (float): Padding in Angstroms for non-periodic directions.

    Returns:
        pymatgen Structure object
    """
    parameters = structData["parameters"].copy()
    species = structData["species"].copy()

    lattice, coords = gel_latt_coords(parameters)

    latVecRange = constrains["latVecRange"]
    latAngRange = constrains["latAngRange"]

    lattice[:3] = lattice[:3] * (latVecRange[1] - latVecRange[0]) + latVecRange[0]
    lattice[3:] = lattice[3:] * (latAngRange[1] - latAngRange[0]) + latAngRange[0]

    if shape in ['sheet', 'cluster']:
        lattice[3] = 90
        lattice[4] = 90
        if shape == 'cluster':
            lattice[5] = 90

    # Create lattice and structure
    lattice = Lattice.from_parameters(*lattice[:6])
    struct = Structure(lattice, species, coords, to_unit_cell=True)

    # Add vacuum padding if needed
    struct = add_padding(struct, shape=shape, pad=pad)

    return struct




def get_string(struct,energy):
    
    lattice = struct.lattice
    frac = np.array([list(site.frac_coords) for site in struct.sites]).flatten().tolist()
    species = [site.specie.symbol for site in struct.sites]
    latt = [lattice.a,lattice.b,lattice.c,lattice.alpha,lattice.beta,lattice.gamma]
    
    dataString = " ".join(map(str,latt+frac)) + "|" +  " ".join(species)+ "|" + "{}".format(energy)
    
    return dataString





def ParamsfromStruct(struct, constrains, energy=1e300, write_file="dumpfile.dat", shape='bulk', pad=20):
    """
    Converts a pymatgen Structure object into a normalized parameter vector.

    Args:
        struct (Structure): A pymatgen Structure object.
        constrains (dict): Contains keys "latVecRange" and "latAngRange".
        energy (float): Energy to include in the written log.
        write_file (str): Output file to write structure+energy string.
        shape (str): Structure shape ('bulk', 'sheet', or 'cluster').
        pad (float): Padding to remove from lattice vectors before normalization.

    Returns:
        dict: {"parameters": np.array, "species": list of element symbols}
    """
    # Write structure string with energy
    data_string = get_string(struct, energy)
    with open(write_file, "a") as f:
        f.write(f"{data_string}\n")

    # Remove artificial padding
    struct = unpad(struct, shape=shape, pad=pad)

    lattice = struct.lattice
    latt = [lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma]
    frac_coords = np.array([site.frac_coords for site in struct.sites]).flatten()
    species = [site.specie.symbol for site in struct.sites]


    # Normalize lattice lengths and angles
    latt = np.array(latt)
    latVecRange = constrains["latVecRange"]
    latAngRange = constrains["latAngRange"]


    if latVecRange[1] != latVecRange[0]:
        latt[:3] = (latt[:3] - latVecRange[0]) / (latVecRange[1] - latVecRange[0])

        
    else:
        warnings.warn("latVecRange bounds are equal. Assigning 0.0 to lattice lengths.")
        latt[:3] = 0.0

    if latAngRange[1] != latAngRange[0]:
        latt[3:] = (latt[3:] - latAngRange[0]) / (latAngRange[1] - latAngRange[0])
    else:
        warnings.warn("latAngRange bounds are equal. Assigning 0.0 to lattice angles.")
        latt[3:] = 0.0

    # Construct final parameter vector
    parameters = np.concatenate((latt, frac_coords))
    return {"parameters": parameters, "species": species}

    


def center(structure,shape="bulk"):
    if shape=="bulk":
        return structure
    
    center = np.average([s.frac_coords[2] for s in structure.sites])
    if shape=="sheet":
        translation = (0, 0, 0.5 - center)
    
    if shape=="cluster":
        translation = (0.5 - center, 0.5 - center, 0.5 - center)

    structure.translate_sites(range(len(structure.sites)), translation)
    
    return structure


#--------------------------------

def add_padding(structure,shape="bulk",pad=20):
    if shape=="bulk":
        return structure
    
    pos = [list(cord.frac_coords) for cord in structure.sites]
    natoms = len(pos)
    species = [cord.specie.symbol for cord in structure.sites]
    latt = np.array(structure.lattice.matrix)
    
    if shape=="sheet":
        multiplier = (pad+structure.lattice.c)/structure.lattice.c
        latt[2] = latt[2]*multiplier
        for i,site in enumerate(pos):
            pos[i][2] = site[2]/multiplier
    
    if shape=="cluster":
        
        multiplier_a = (pad+structure.lattice.a)/structure.lattice.a
        latt[0] = latt[0]*multiplier_a
        multiplier_b = (pad+structure.lattice.b)/structure.lattice.b
        latt[1] = latt[1]*multiplier_b
        multiplier_c = (pad+structure.lattice.c)/structure.lattice.c
        latt[2] = latt[2]*multiplier_c
        
        for i,site in enumerate(pos):
            pos[i][0] = site[0]/multiplier_a
            pos[i][1] = site[1]/multiplier_b
            pos[i][2] = site[2]/multiplier_c
            

    newstructure = Structure(latt,species,pos,to_unit_cell=True)
    return center(newstructure,shape=shape) 



def unpad(structure, shape="bulk", pad=20):
    """
    Removes added padding from sheet or cluster structures.

    Args:
        structure (Structure): Pymatgen Structure object.
        shape (str): 'bulk', 'sheet', or 'cluster'.
        pad (float): Amount of padding to remove in Å.

    Returns:
        Structure: Modified structure with padding removed.
    """
    if shape == "bulk":
        return structure

    positions = [site.frac_coords.tolist() for site in structure.sites]
    species = [site.specie.symbol for site in structure.sites]
    lattice = np.array(structure.lattice.matrix)

    if shape == "sheet":
        a, b, c = structure.lattice.a, structure.lattice.b, structure.lattice.c
        multiplier_c = (c - pad) / c
        lattice[2] *= multiplier_c
        for i, pos in enumerate(positions):
            positions[i][2] /= multiplier_c

    elif shape == "cluster":
        a, b, c = structure.lattice.a, structure.lattice.b, structure.lattice.c
        multiplier_a = (a - pad) / a
        multiplier_b = (b - pad) / b
        multiplier_c = (c - pad) / c
        lattice[0] *= multiplier_a
        lattice[1] *= multiplier_b
        lattice[2] *= multiplier_c
        for i, pos in enumerate(positions):
            positions[i][0] /= multiplier_a
            positions[i][1] /= multiplier_b
            positions[i][2] /= multiplier_c

    new_struct = Structure(lattice, species, positions, to_unit_cell=True)
    return center(new_struct, shape=shape)

