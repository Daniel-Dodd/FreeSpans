import jax.numpy as jnp


import gpjax as gpx

import pytest
import spans
from jax import vmap

@pytest.mark.parametrize("num_inducing", [1, 2, 10])
def test_kmeans(num_inducing):
    # Create pipe locations and time indicies:
    L = jnp.linspace(-1, 1, 10)
    T = jnp.linspace(-5, 5, 20)
    
    # Create covariates:
    X = vmap(lambda t: vmap(lambda l: jnp.array([t,l]))(L))(T).reshape(-1, 2)

    # Create labels
    y = (jnp.sin(jnp.linspace(-jnp.pi/2, jnp.pi/2, 10*20)) < 0).astype(jnp.float32).reshape(-1, 1)
    D = gpx.Dataset(X=X, y=y)

    z = spans.models.kmeans_init_inducing(D, num_inducing)

    assert z.shape == (num_inducing, 2)