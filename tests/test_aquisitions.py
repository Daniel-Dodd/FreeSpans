import jax.numpy as jnp

import pytest

from jax import vmap
import gpjax as gpx
from gpjax import Dataset
import freespans

def dataset():
    # Create pipe locations and time indicies:
    L = jnp.linspace(-1, 1, 10)
    T = jnp.linspace(-5, 5, 20)
    
    # Create covariates:
    X = vmap(lambda t: vmap(lambda l: jnp.array([t,l]))(L))(T).reshape(-1, 2)

    # Create labels
    y = (jnp.sin(jnp.linspace(-jnp.pi/2, jnp.pi/2, 10*20)) < 0).astype(jnp.float32).reshape(-1, 1)
    D = gpx.Dataset(X=X, y=y)

    return D

def variational_family_and_posterior():
    D = dataset()
    kernel = gpx.RBF()
    prior = gpx.Prior(kernel=kernel)
    likelihood = gpx.Bernoulli(num_datapoints = D.n)

    p = prior * likelihood

    Lz = jnp.linspace(-1, 1, 4)
    Tz = jnp.linspace(-5, 5, 5)

    z = vmap(lambda t: vmap(lambda l: jnp.array([t,l]))(Lz))(Tz).reshape(-1, 2)
    q = gpx.VariationalGaussian(prior=prior, inducing_inputs=z)

    return  p, q

def test_pred_entropy():
    p, q =  variational_family_and_posterior()

    # Create pipe locations and time indicies:
    Ld = jnp.linspace(-2, 2, 10)
    Td = jnp.linspace(-1, 1, 5)
    
    # Create design locations and time indicies:
    d = vmap(lambda t: vmap(lambda l: jnp.array([t,l]))(Ld))(Td).reshape(-1, 2)

    svgp = gpx.StochasticVI(posterior=p, variational_family = q)
    params =  svgp.params

  


    entropy_fn = freespans.aquisitions.PredictiveEntropy(model=q, 
                                                    design_likelihood = p.likelihood, 
                                                    inner_samples=32, 
                                                    outer_samples=16, 
                                                    seed=42,
                                                    )

    entropy = entropy_fn(params, d)

    assert entropy.shape == ()
    assert entropy.dtype == jnp.float32 or jnp.float64 or float
 
def test_pred_information():
    p, q =  variational_family_and_posterior()

    

    # Create design locations and time indicies:
    Ld = jnp.linspace(-2, 2, 10)
    Td = jnp.linspace(-1, 1, 5)
    d = vmap(lambda t: vmap(lambda l: jnp.array([t,l]))(Ld))(Td).reshape(-1, 2)

    # Create test locations and time indicies:
    Lt = jnp.linspace(1, 2, 3)
    Tt = jnp.linspace(4, 5, 2)
    t = vmap(lambda t: vmap(lambda l: jnp.array([t,l]))(Lt))(Tt).reshape(-1, 2)

    svgp = gpx.StochasticVI(posterior=p, variational_family = q)

    params =  svgp.params

    information_fn = freespans.aquisitions.PredictiveInformation(model=q, 
                                                    design_likelihood= p.likelihood, 
                                                    inner_samples=32, 
                                                    outer_samples=16, 
                                                    seed=42,
                                                    )

    information = information_fn(params=params, design=d, test=t)

    assert information.shape == ()
    assert information.dtype == jnp.float32 or jnp.float64 or float