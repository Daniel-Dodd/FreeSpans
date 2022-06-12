from typing import Optional, Tuple
import jax.random as jr
import jax.numpy as jnp
from jax import vmap

from gpjax.likelihoods import Bernoulli, Gaussian
from gpjax.kernels import Kernel
from gpjax import Prior

from .utils import Scaler
from .types import Array, SimulatedSpanData


def simulate_bernoulli(kernel: Kernel,
            start_time: float, 
            end_time: float, 
            start_pipe: float, 
            end_pipe: float,
            time_width: Optional[float] = 1.,
            location_width: Optional[float] = 1.,
            seed: Optional[int] = 42,
            scaler: Optional[Scaler] = None,
            ) -> Tuple[Array]:
    """Simulate Bernoulli span data.
    Args:
        kernel (gpjax.Kernel): GPJax Kernel function.
        start_time (float): The start time for the simulating the data.
        end_time (float): The end time for the simulating the data.
        start_pipe (float): The start of the pipe for the simulating the data.
        end_pipe (float): The end of the pipe for the simulating the data.
        time_width (float, optional): The temporal distance between samples, e.g.,
                                1. is once per time unit, .5 is twice per time unit.
        location_width (float, optional): The spatial distance between samples.
        seed (int, optional): The random seed for simulating the data.
        scaler: (Scaler, optional): A scalar for the covariates.
    Returns:
        SimulatedSpanData: A simulated dataset from a Bernoulli GP model.
    """
    
    key_f, key_y = jr.split(jr.PRNGKey(seed))

    # Create pipe locations and time indicies:
    L = jnp.arange(start_pipe, end_pipe, location_width) + location_width/2.
    T = jnp.arange(start_time, end_time + time_width/2, time_width)
    
    # Create covariates:
    X = vmap(lambda t: vmap(lambda l: jnp.array([t,l]))(L))(T).reshape(-1, 2)

    if scaler is not None:
        X = scaler(X)

    # Simulate f:
    prior = Prior(kernel = kernel)
    fx = prior.predict(prior.params)(X)
    f_sample = fx.sample(seed = key_f)

    # Simulate y:
    n = X.shape[0]
    likelihood = Bernoulli(num_datapoints=n)
    y = likelihood.link_function(f_sample, likelihood.params)
    y_sample = y.sample(seed = key_y).reshape(-1, 1)
    
    return SimulatedSpanData(X=X, y=y_sample, L=L, T=T, f=f_sample) 


def simulate_indicator(kernel: Kernel,
            start_time: float, 
            end_time: float, 
            start_pipe: float, 
            end_pipe: float,
            time_width: Optional[float] = 1.,
            location_width: Optional[float] = 1.,
            seed: Optional[int] = 42,
            scaler: Optional[Scaler] = None,
            ) -> Tuple[Array]:
    """Simulate indicator span data.
    Args:
        kernel (gpjax.Kernel): GPJax Kernel function.
        start_time (float): The start time for the simulating the data.
        end_time (float): The end time for the simulating the data.
        start_pipe (float): The start of the pipe for the simulating the data.
        end_pipe (float): The end of the pipe for the simulating the data.
        time_width (float, optional): The temporal distance between samples, e.g.,
                                1. is once per time unit, .5 is twice per time unit.
        location_width (float, optional): The spatial distance between samples.
        seed (int, optional): The random seed for simulating the data.
        scaler: (Scaler, optional): A scalar for the covariates.
    Returns:
        SimulatedSpanData: A simulated dataset from an indicator GP model.
    """
    
    key_f, key_y = jr.split(jr.PRNGKey(seed))

    # Create pipe locations and time indicies:
    L = jnp.arange(start_pipe, end_pipe, location_width) + location_width/2.
    T = jnp.arange(start_time, end_time + time_width/2, time_width)
    
    # Create covariates:
    X = vmap(lambda t: vmap(lambda l: jnp.array([t,l]))(L))(T).reshape(-1, 2)

    if scaler is not None:
        X = scaler(X)

    # Simulate f:
    prior = Prior(kernel = kernel)
    fx = prior.predict(prior.params)(X)
    f_sample = fx.sample(seed = key_f)

    # Determine y:
    y_sample = (f_sample < 0.).astype(jnp.float64).reshape(-1, 1)

    return SimulatedSpanData(X=X, y=y_sample, L=L, T=T, f=f_sample) 


def simulate_gaussian(kernel: Kernel,
            start_time: int, 
            end_time: int, 
            start_pipe: float, 
            end_pipe: float,
            time_width: Optional[float] = 1.,
            location_width: Optional[float] = 1.,
            seed: Optional[int] = 42,
            scaler: Optional[Scaler] = None,
            obs_noise: Optional[float] = 1.
            ) -> Tuple[Array]:
    """Simulate Gaussian span data.
    Args:
        kernel (gpjax.Kernel): GPJax Kernel function.
        start_time (float): The start time for the simulating the data.
        end_time (float): The end time for the simulating the data.
        start_pipe (float): The start of the pipe for the simulating the data.
        end_pipe (float): The end of the pipe for the simulating the data.
        time_width (float, optional): The temporal distance between samples, e.g.,
                                1. is once per time unit, .5 is twice per time unit.
        location_width (float, optional): The spatial distance between samples.
        seed (int, optional): The random seed for simulating the data.
        scaler (Scaler, optional): A scalar for the covariates.
        obs_noise (float, optional): The observation noise in the Gaussian likelihood.
    Returns:
        SimulatedSpanData: A simulated dataset from a Gaussian GP model.
    """
    key_f, key_y = jr.split(jr.PRNGKey(seed))

    # Create pipe locations and time indicies:
    L = jnp.arange(start_pipe, end_pipe, location_width) + location_width/2.
    T = jnp.arange(start_time, end_time + time_width/2, time_width)
    
    # Create covariates:
    X = vmap(lambda t: vmap(lambda l: jnp.array([t,l]))(L))(T).reshape(-1, 2)

    if scaler is not None:
        X = scaler(X)

    # Simulate f:
    prior = Prior(kernel = kernel)
    fx = prior.predict(prior.params)(X)
    f_sample = fx.sample(seed = key_f)

    # Simulate y:
    n = X.shape[0]
    likelihood = Gaussian(num_datapoints=n)
    likelihood.params["obs_noise"] = obs_noise
    y = likelihood.link_function(f_sample, likelihood.params)
    y_sample = y.sample(seed = key_y).reshape(-1, 1)
    
    return SimulatedSpanData(X=X, y=y_sample, L=L, T=T, f=f_sample) 

