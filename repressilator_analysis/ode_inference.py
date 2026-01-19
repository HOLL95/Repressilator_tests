"""
ODE model and parameter inference for the Repressilator system.

This module implements the Repressilator ODE model and uses PINTS
for Bayesian parameter inference.
"""

import numpy as np
import pints
from scipy.integrate import odeint
from typing import List, Tuple, Dict, Optional


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

    def __init__(self, times: np.ndarray):
        """
        Initialize the Repressilator model.

        Args:
            times: Array of time points for simulation
        """
        self.times = times

    def n_parameters(self) -> int:
        """Return the number of model parameters."""
        return 6  # [alpha, alpha0, beta, n, gamma_m, gamma_p]

    def n_outputs(self) -> int:
        """Return the number of observable outputs."""
        # We observe 2 proteins (nuclear and cytoplasmic)
        return 2

    def simulate(self, parameters: List[float], times: np.ndarray) -> np.ndarray:
        """
        Simulate the Repressilator ODE system.

        Args:
            parameters: Model parameters [alpha, alpha0, beta, n, gamma_m, gamma_p]
            times: Time points for simulation

        Returns:
            Array of shape (n_times, n_outputs) with protein concentrations
        """
        alpha, alpha0, beta, n, gamma_m, gamma_p = parameters

        # Initial conditions (start at equilibrium estimate)
        y0 = [1.0, 1.0, 1.0, 10.0, 10.0, 10.0]  # [m1, m2, m3, p1, p2, p3]

        def repressilator_odes(y, t):
            """ODE system for the Repressilator."""
            m1, m2, m3, p1, p2, p3 = y

            # Hill function for repression
            def hill_repression(repressor_conc):
                return alpha / (1 + (repressor_conc ** n)) + alpha0

            # mRNA dynamics
            dm1_dt = hill_repression(p3) - gamma_m * m1
            dm2_dt = hill_repression(p1) - gamma_m * m2
            dm3_dt = hill_repression(p2) - gamma_m * m3

            # Protein dynamics
            dp1_dt = beta * m1 - gamma_p * p1
            dp2_dt = beta * m2 - gamma_p * p2
            dp3_dt = beta * m3 - gamma_p * p3

            return [dm1_dt, dm2_dt, dm3_dt, dp1_dt, dp2_dt, dp3_dt]

        # Solve ODE system
        solution = odeint(repressilator_odes, y0, times)

        # Extract observable proteins (p1 and p2 as nuclear and cytoplasmic)
        # Note: p3 is the unobserved protein without fluorescence
        output = solution[:, [3, 4]]  # [p1, p2]

        return output


class RepressilatorLogLikelihood(pints.ProblemLogLikelihood):
    """
    Log-likelihood for Repressilator parameter inference.

    Assumes Gaussian noise on observations.
    """

    def __init__(
        self,
        model: RepressilatorModel,
        times: np.ndarray,
        observations: np.ndarray,
    ):
        """
        Initialize log-likelihood.

        Args:
            model: RepressilatorModel instance
            times: Time points for observations
            observations: Observed data of shape (n_times, n_proteins)
        """
        super().__init__(pints.SingleOutputProblem(model, times, observations))
        self.times = times
        self.observations = observations
        self.n_times = len(times)

    def __call__(self, parameters: List[float]) -> float:
        """
        Calculate log-likelihood for given parameters.

        Args:
            parameters: Model parameters + noise parameter

        Returns:
            Log-likelihood value
        """
        # Last parameter is noise standard deviation
        model_params = parameters[:-1]
        sigma = parameters[-1]

        if sigma <= 0:
            return -np.inf

        # Simulate model
        try:
            predictions = self._model.simulate(model_params, self.times)
        except Exception:
            return -np.inf

        # Calculate log-likelihood (assuming Gaussian noise)
        residuals = self.observations - predictions
        log_likelihood = -0.5 * np.sum((residuals / sigma) ** 2)
        log_likelihood -= self.n_times * np.log(sigma)
        log_likelihood -= 0.5 * self.n_times * np.log(2 * np.pi)

        return log_likelihood


def create_prior(parameter_names: List[str]) -> pints.LogPrior:
    """
    Create prior distributions for model parameters.

    Args:
        parameter_names: List of parameter names

    Returns:
        PINTS prior object
    """
    # Define prior bounds (uniform priors)
    priors = []

    for name in parameter_names:
        if name == 'alpha':
            priors.append(pints.UniformLogPrior(0, 1000))
        elif name == 'alpha0':
            priors.append(pints.UniformLogPrior(0, 10))
        elif name == 'beta':
            priors.append(pints.UniformLogPrior(0, 100))
        elif name == 'n':
            priors.append(pints.UniformLogPrior(1, 5))
        elif name == 'gamma_m':
            priors.append(pints.UniformLogPrior(0.01, 1))
        elif name == 'gamma_p':
            priors.append(pints.UniformLogPrior(0.001, 0.1))
        elif name == 'sigma':
            priors.append(pints.UniformLogPrior(0.1, 100))
        else:
            priors.append(pints.UniformLogPrior(0, 100))

    return pints.ComposedLogPrior(*priors)


def infer_parameters(
    times: np.ndarray,
    observations: np.ndarray,
    n_iterations: int = 1000,
    n_chains: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer Repressilator parameters using Bayesian inference (MCMC).

    Args:
        times: Time points (in minutes)
        observations: Observed protein concentrations of shape (n_times, 2)
        n_iterations: Number of MCMC iterations
        n_chains: Number of parallel MCMC chains

    Returns:
        Tuple of (parameter_samples, parameter_means)
    """
    # Create model
    model = RepressilatorModel(times)

    # Create log-likelihood
    log_likelihood = RepressilatorLogLikelihood(model, times, observations)

    # Create prior
    parameter_names = ['alpha', 'alpha0', 'beta', 'n', 'gamma_m', 'gamma_p', 'sigma']
    log_prior = create_prior(parameter_names)

    # Create log-posterior
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    # Initial parameter guesses
    initial_params = [
        [100, 1, 10, 2, 0.1, 0.01, 10],  # Chain 1
        [200, 0.5, 20, 2.5, 0.2, 0.02, 5],  # Chain 2
        [150, 2, 15, 1.5, 0.15, 0.015, 8],  # Chain 3
    ][:n_chains]

    # Run MCMC
    mcmc = pints.MCMCController(
        log_posterior,
        n_chains,
        initial_params,
        method=pints.HaarioBardenetACMC
    )
    mcmc.set_max_iterations(n_iterations)
    mcmc.set_log_to_screen(True)

    print(f"Running MCMC with {n_chains} chains for {n_iterations} iterations...")
    chains = mcmc.run()

    # Extract samples (discard burn-in)
    burn_in = n_iterations // 2
    samples = np.vstack([chain[burn_in:] for chain in chains])

    # Calculate parameter means
    param_means = np.mean(samples, axis=0)

    print("\nInferred parameters:")
    for name, value in zip(parameter_names, param_means):
        print(f"  {name}: {value:.4f}")

    return samples, param_means


def run_inference_for_cell(
    times: np.ndarray,
    cell_data: Dict[str, List[float]],
    n_iterations: int = 1000,
) -> Dict[str, any]:
    """
    Run parameter inference for a single cell's time-series data.

    Args:
        times: Time points in minutes
        cell_data: Dictionary with 'nuclear' and 'cytoplasmic' protein concentrations
        n_iterations: Number of MCMC iterations

    Returns:
        Dictionary with inference results
    """
    # Prepare observations
    nuclear = np.array(cell_data.get('nuclear', []))
    cytoplasmic = np.array(cell_data.get('cytoplasmic', []))

    if len(nuclear) == 0 or len(cytoplasmic) == 0:
        raise ValueError("Cell data must contain 'nuclear' and 'cytoplasmic' measurements")

    observations = np.column_stack([nuclear, cytoplasmic])

    # Run inference
    samples, param_means = infer_parameters(times, observations, n_iterations)

    results = {
        'times': times,
        'observations': observations,
        'parameter_samples': samples,
        'parameter_means': param_means,
        'parameter_names': ['alpha', 'alpha0', 'beta', 'n', 'gamma_m', 'gamma_p', 'sigma'],
    }

    return results
