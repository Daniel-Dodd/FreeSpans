import jax.numpy as jnp
import gpjax as gpx
from gpjax.parameters import initialise
from gpjax.kernels import Kernel
import freespans


import pytest


@pytest.mark.parametrize("base", [gpx.RBF(active_dims=[0, 1]), gpx.Matern12(active_dims=[0, 1]), gpx.Matern32(active_dims=[0, 1]), gpx.Matern52(active_dims=[0, 1])])
def test_kernel_call(base):

    base_params = base.params
    drift_params = {"theta": jnp.array([jnp.pi/2.]), "scale": jnp.array([1.])}
  
    kernel = freespans.kernels.DriftKernel(base_kernel = base)

    assert isinstance(kernel, Kernel)
    

    params = gpx.config.get_defaults()
    assert "theta" in params["transformations"].keys()
    assert "scale" in params["transformations"].keys()
    assert kernel.params == {**drift_params, **base_params}

    params, _, _, _ = initialise(kernel)
    x, y = jnp.array([[1.0, 2.0]]), jnp.array([[0.5, 1.0]])
    point_corr = kernel(x, y, params)
    assert isinstance(point_corr, jnp.DeviceArray)
    assert point_corr.shape == ()