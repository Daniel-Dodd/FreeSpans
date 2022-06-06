import jax.numpy as jnp
from typing import Optional

from gpjax.types import Array
from gpjax.kernels import Kernel
from gpjax.config import Softplus, add_parameter

from chex import dataclass
import distrax as dx

import pytest

from jax import vmap
import gpjax as gpx
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


def test_compute_percentages():
    D = dataset()
    regions = jnp.array([[-.5, 0.], [.5, 1.5]])
    percentage = freespans.optimal_design.compute_percentages(regions, D)
    assert percentage.shape == ()
    assert percentage.dtype == jnp.float32 or jnp.float64

def test_pred_entropy():
    pass
 
def test_pred_information():
    pass

def test_box_design():
    #spans.optimal_design.box_design
    pass

def test_inspection_region_design():
    pass

def test_at_reveal():
    pass

def test_before_reveal():
    pass

def test_after_reveal():
    pass

def test_box_reveal():
    pass

def test_region_reveal():
    pass

def test_naive_predictor():
    pass

def test_make_naive_predictor():
    pass

def test_inspection_region_reveal():
    pass

def test_optimal_design():
    pass