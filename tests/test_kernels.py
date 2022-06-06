import jax.numpy as jnp
import gpjax as gpx

from gpjax.kernels import Kernel
import spans


import pytest


@pytest.mark.parametrize("base", [gpx.RBF(active_dims=[0, 1]), gpx.Matern12(active_dims=[0, 1]), gpx.Matern32(active_dims=[0, 1]), gpx.Matern52(active_dims=[0, 1])])
def test_kernel_call(base):

    base_params = base.params
    drift_params = {"theta": jnp.array([jnp.pi/2.]), "scale": jnp.array([1.])}
  
    kernel = spans.kernels.DriftKernel(base_kernel = base)

    assert isinstance(kernel, Kernel)
    

    params = gpx.config.get_defaults()
    assert "theta" in params["transformations"].keys()
    assert "scale" in params["transformations"].keys()
    assert kernel.params == {**drift_params, **base_params}
