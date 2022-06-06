import jax.numpy as jnp


import gpjax as gpx

import pytest
import spans
from jax import vmap

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

@pytest.mark.parametrize("num_inducing", [1, 2, 10])
def test_kmeans(num_inducing):
    D = dataset()
    z = spans.models.kmeans_init_inducing(D, num_inducing)

    assert z.shape == (num_inducing, 2)

@pytest.mark.parametrize("num_inducing", [1, 2, 10])
@pytest.mark.parametrize("type", ["bernoulli", "gaussian"])
def test_svgp(num_inducing, type):
    D = dataset()

    if type == "bernoulli":
        posterior, variational_family, svgp = spans.models.bernoulli_svgp(
            gpx.kernels.RBF(), D, num_inducing
        )

        assert isinstance(posterior.likelihood, gpx.Bernoulli)
        assert isinstance(svgp.posterior, gpx.gps.NonConjugatePosterior)
    elif type == "gaussian":
        posterior, variational_family, svgp = spans.models.gaussian_svgp(
            gpx.kernels.RBF(), D, num_inducing
        )
        assert isinstance(posterior.likelihood, gpx.Gaussian)
        assert isinstance(svgp.posterior, gpx.gps.ConjugatePosterior)

    params = variational_family.params

    assert params["variational_family"]["inducing_inputs"].shape == (num_inducing, 2)
    assert isinstance(variational_family, gpx.WhitenedVariationalGaussian)
    assert isinstance(svgp, gpx.StochasticVI)
    
