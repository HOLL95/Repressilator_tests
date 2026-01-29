"""
ODE model and parameter inference for the Repressilator system.

This module implements the Repressilator ODE model and uses PINTS
for Bayesian parameter inference.
"""

import numpy as np
import pints
from scipy.integrate import solve_ivp
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from pints.plot import trace
def d_mrna_dt(mrna, a, a0, protein, hill):
    return -mrna+(a/(1+protein**hill))+a0
def d_protein_dt(protein, mrna, beta):
    return -beta*(protein-mrna)
class RepressilatorModel(pints.ForwardModel):
    """
    ODE model for the Repressilator genetic circuit.

    The Repressilator consists of three repressors (LacI, TetR, CI) that
    form a cyclic negative feedback loop. Each repressor inhibits the
    transcription of the next repressor in the cycle.

    State variables:
    - m1, m2, m3: mRNA concentrations for repressors 1, 2, 3
    - p1, p2, p3: Protein concentrations for repressors 1, 2, 3

    Parameters:
    - alpha: Transcription rate
    - alpha0: Basal transcription rate
    - beta: Translation rate
    - n: Hill coefficient (cooperativity)
    - gamma_m: mRNA degradation rate
    - gamma_p: Protein degradation rate
    """
    _param_names=[
                    "hill", 
                    "mrna_half_life",
                    "p_half_life", 
                    "K_m", 
                    "T_e", 
                    "initial_c_p", 
                    "initial_n_p_1", 
                    "initial_n_p_2", 
                    "initial_c_m", 
                    "initial_n_m_1",
                    "initial_n_m_2",
                    "alpha", 
                    "alpha0"]
    def __init__(self, times: np.ndarray):
        """
        Initialize the Repressilator model.

        Args:
            times: Array of time points for simulation
        """
        self.times = times
        
   

    def n_parameters(self) -> int:
        """Return the number of model parameters."""
        return len(self._param_names)

    def n_outputs(self) -> int:
        """Return the number of observable outputs."""
        # We observe 2 proteins (nuclear and cytoplasmic)
        return 2

    def simulate(self, parameters: List[float], times_in_seconds: np.ndarray) -> np.ndarray:
        """
        Simulate the Repressilator ODE system.

        Args:
            parameters: Model parameters [alpha, alpha0, beta, n, gamma_m, gamma_p]
            times: Time points for simulation

        Returns:
            Array of shape (n_times, n_outputs) with protein concentrations
        """
        hill, mrna_half_life, p_half_life, K_m, T_e, initial_c_p, initial_n_p_1, initial_n_p_2, initial_c_m, initial_n_m_1, initial_n_m_2, alpha, alpha0=parameters
        
        beta=p_half_life/mrna_half_life
        p_decay=p_half_life/np.log(2)
        m_decay=mrna_half_life/np.log(2)
        nd_time=times_in_seconds/(m_decay*60)
        alpha=alpha*60*p_decay*T_e/K_m
        alpha0*=alpha
        y0 = [  initial_n_p_1, initial_n_p_2, initial_c_p, initial_n_m_1, initial_n_m_2,initial_c_m]  # [m1, m2, m3, p1, p2, p3]
        # Solve ODE system
        solution = solve_ivp(self.repressilator_odes, (0, nd_time[-1]), y0,  args=(alpha, alpha0,hill, beta, K_m), t_eval=nd_time)
        #nuclear protein 1, cytosolic 1,
        output=solution.y[[0,2],:].T
        return output
    def repressilator_odes(self, t,y, *p):
        """ODE system for the Repressilator."""
        n_p_1, n_p_2, c_p,  n_m_1, n_m_2,c_m,=y
        alpha, alpha0,hill_coeff, beta, K_m=p
        p_i=[x/K_m for x in [n_p_1, n_p_2, c_p]]
        m_i=[n_m_1, n_m_2, c_m]
        dps=[0 for _ in range(0, 3)]
        dms=[0 for _ in range(0, 3)]
        for i in range(0, 3):
            dps[i]=d_protein_dt(p_i[i], m_i[i], beta)
            protein_idx=(i+2)%3
            dms[i]=d_mrna_dt(m_i[i], alpha, alpha0, p_i[protein_idx], hill_coeff)
        return dps+dms


def infer_parameters(
    times: np.ndarray,
    observations: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer Repressilator parameters using CMAES optimization followed by adaptive covariance MCMC.

    Args:
        times: Time points (in minutes)
        observations: Observed protein concentrations of shape (n_times, 2)
        n_iterations: Number of optimization iterations
        sigma: Noise standard deviation for likelihood calculation
        n_mcmc_iterations: Number of MCMC iterations

    Returns:
        Tuple of (mcmc_samples, best_parameters_from_optimization)
    """
    # Create model
    model = RepressilatorModel(times)

    # Define parameter bounds based on test data analysis
    # Format: {param_name: (lower_bound, upper_bound)}
    param_bounds = {
        "hill": (1.0, 3.0),
        "mrna_half_life": (1.0, 3.0),
        "p_half_life": (5.0, 15.0),
        "K_m": (20.0, 50.0),
        "T_e": (20.0, 40.0),
        "initial_c_p": (1.0, 50.0),
        "initial_n_p_1": (1.0, 50.0),
        "initial_n_p_2": (1.0, 50.0),
        "initial_c_m": (5.0, 150.0),
        "initial_n_m_1": (5.0, 150.0),
        "initial_n_m_2": (5.0, 150.0),
        "alpha": (0,1),
        "alpha0": (0.0005, 0.002)
    }

    # Extract bounds in the correct parameter order
    lower_bounds = [param_bounds[name][0] for name in model._param_names]+[0, 0]
    upper_bounds = [param_bounds[name][1] for name in model._param_names]+[1e3, 1e3]

    # Create initial guess (midpoint of bounds)
    score=-1e23
    for i in range(0,5):
        x0 = [np.random.uniform(low=lower, high=upper, size=1)for lower, upper in zip(lower_bounds, upper_bounds)]

        # Create error measure

        problem=pints.MultiOutputProblem(model,times, observations)
        
        error = pints.GaussianLogLikelihood(problem )
        
        # Create boundaries
        boundaries = pints.RectangularBoundaries(lower_bounds, upper_bounds)

        # Run CMAES optimization
        opt = pints.OptimisationController(
            error,
            x0,
            boundaries=boundaries,
            method=pints.CMAES
        )
        opt.set_max_unchanged_iterations(200, threshold=1e-1)
        opt.set_log_to_screen(True)
        opt.set_parallel(True)
        # Run optimization
        best_params, best_error = opt.run()
        if best_error>score:
            score=best_error
            params=best_params
    
    return params[:-2]


