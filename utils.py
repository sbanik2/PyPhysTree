import numpy as np
from pyDOE import lhs
from typing import Callable, List, Optional, Tuple, Deque,Dict
from random import choice,random







# Define the function to generate initial points using Latin Hypercube Sampling (LHS)
def _latin(npoints: int, lb: List[float], ub: List[float]) -> List[List[float]]:
    """
    Use Latin Hypercube Sampling to generate initial points for MCTS.
    
    Args:
    - npoints: Number of initial points to sample.
    - lb: Lower bounds for each dimension.
    - ub: Upper bounds for each dimension.
    
    Returns:
    - seedpoints: List of initial points generated using LHS within the bounds.
    """
    dim = len(ub)
    #print(f"Lower bound: {lb}, Upper bound: {ub}, Number of points: {npoints}")
    
    # Use lhs function from pyDOE to generate samples
    samples_LH = lhs(dim, samples=npoints)  # Latin Hypercube Sampling
    
    # Scale the samples to the given bounds
    seedpoints = [np.array(lb) + samp * (np.array(ub) - np.array(lb)) for samp in samples_LH]
    
    return seedpoints




def adaptive_bounds(ub: np.ndarray, lb: np.ndarray, scale: float, params: np.ndarray):
    reduced_coords = (params - lb) / (ub - lb)
    delta = scale
    newub = np.clip(reduced_coords + delta * 0.5, 0, 1) * (ub - lb) + lb
    newlb = np.clip(reduced_coords - delta * 0.5, 0, 1) * (ub - lb) + lb
    return newub, newlb





class Perturbate:
    """
    A class to perturb the parameter within the given bounds and dimension.
    Uses depth scaling to adjust the magnitude of the perturbation based on current depth.
    """
    def __init__(self, lower_bound, upper_bound, a=4e-2, b=1/2**0.5):
        """
        Initialize the Perturbate class.
        
        Args:
        - lower_bound: The lower bound for each dimension.
        - upper_bound: The upper bound for each dimension.
        - a: A scaling parameter (default: 4e-2).
        - b: A scaling parameter (default: 1/2**0.5).
        """
        # Ensure that lower and upper bounds have the same dimension
        assert len(lower_bound) == len(upper_bound)
        
        # Dimension of the parameter (number of elements)
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.dimension = len(upper_bound)
        
        # Scaling parameters
        self.a = a
        self.b = b

    def DepthScale(self, depth):
        """
        Compute a scaling factor based on depth. This factor will adjust the magnitude of perturbation.
        
        Args:
        - depth: The current depth at which scaling should be applied.
        
        Returns:
        - depthscale: The scaling factor for the given depth.
        """
        # Depth scaling based on the exponential function
        
        depthscale = self.b * 0.5 * (self.dimension)**(0.5) * np.exp(-self.a * (depth)**2)
        return depthscale

    def generate_new_parameter(self, parameter, depth):
        """
        Generates a new parameter by sampling a point from a hypersphere around the
        normalized parameter vector and maps it back to real space.
    
        Args:
        - parameter: np.ndarray, the current parameter vector
        - depth: int, the depth of the current MCTS node
    
        Returns:
        - new_parameter: np.ndarray, the perturbed parameter within bounds
        """
        assert len(parameter) == self.dimension, "Parameter dimension mismatch"
    
        # Convert to reduced space [0, 1]
        x_r = (parameter - self.lower_bound) / (self.upper_bound - self.lower_bound + 1e-12)
    
        # Sample direction on hypersphere surface using standard normal distribution
        u = np.random.normal(0.0, 1.0, self.dimension)
        norm = np.linalg.norm(u)
        if norm == 0:
            u = np.ones(self.dimension)
            norm = np.sqrt(self.dimension)
    
        rmax = self.DepthScale(depth)
    
        # Generate random r in [0, rmax]
        r = random() * rmax
        x_step = (r / norm) * u
        x_new = x_r + x_step
    
        # Clip to [0, 1]
        x_new = np.clip(x_new, 0.0, 1.0)
    
        # Map back to real space
        new_parameter = x_new * (self.upper_bound - self.lower_bound) + self.lower_bound
    
        return new_parameter



