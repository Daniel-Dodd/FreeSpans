import jax.numpy as jnp

import pytest

from gpjax import Dataset
import freespans

import gpjax as gpx

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

@pytest.mark.parametrize("start_time, end_time, start_pipe, end_pipe", [(0, 1, 0, 1), (-1, 1, 0, 1)])
def test_box_design(start_time, end_time, start_pipe, end_pipe):

    X = freespans.optimal_design.box_design(start_time, end_time, start_pipe, end_pipe)
    assert X.shape[1] == 2
    assert (X[:,0] >= start_time - 1e-8).all() 
    assert (X[:,0] <= end_time + 1e-8).all() 
    assert (X[:,1] >= start_pipe - 1e-8).all() 
    assert (X[:,1] <= end_pipe + 1e-8).all()


@pytest.mark.parametrize("regions", [jnp.array([[1., 2.]]), jnp.array([[1., 2.], [3., 4.]])])
def test_inspection_region_design(regions):
    inspection_time = 0.
    location_width = 1.

    d = freespans.inspection_region_design(inspection_time, regions, location_width)
    assert d.shape[1] == 2
    assert (d[:,0] == inspection_time).all()
    assert d[:,1].min() >= regions.flatten().min()
    assert d[:,1].max() <= regions.flatten().max()

    

def test_at_reveal():
    D = dataset()
    time = 5
    D_reveal = freespans.optimal_design.at_reveal(time, D)
    assert D_reveal.X.shape[1] == 2
    assert (D_reveal.X[:,0] == time).all()
    assert isinstance(D, Dataset)


def test_before_reveal():
    D = dataset()
    time = 3
    D_reveal = freespans.optimal_design.before_reveal(time, D)
    assert D_reveal.X.shape[1] == 2
    assert (D_reveal.X[:,0] <= time).all()
    assert isinstance(D, Dataset)

def test_after_reveal():
    D = dataset()
    time = 3
    D_reveal = freespans.optimal_design.after_reveal(time, D)
    assert D_reveal.X.shape[1] == 2
    assert (D_reveal.X[:,0] >= time).all()
    assert isinstance(D, Dataset)

@pytest.mark.parametrize("start_time, end_time, start_pipe, end_pipe", [(0, 1, 0, 1), (-1, 1, 0, 1)])
def test_box_reveal(start_time, end_time, start_pipe, end_pipe):
    D = dataset()

    D_reveal = freespans.optimal_design.box_reveal(start_time, end_time, start_pipe, end_pipe, D)
    assert D_reveal.X.shape[1] == 2
    assert (D_reveal.X[:,0] >= start_time - 1e-8).all() 
    assert (D_reveal.X[:,0] <= end_time + 1e-8).all() 
    assert (D_reveal.X[:,1] >= start_pipe - 1e-8).all() 
    assert (D_reveal.X[:,1] <= end_pipe + 1e-8).all()


@pytest.mark.parametrize("regions", [jnp.array([[0., .5]]), jnp.array([[-1., 0.], [.5, 1.]])])
def test_region_reveal_and_inspection_region_reveal(regions):
    D = dataset()

    D_reveal = freespans.optimal_design.region_reveal(regions, D)
    assert D_reveal.X.shape[1] == 2
    assert (D_reveal.X[:,1] >= regions.flatten().min() - 1e-8).all()
    assert (D_reveal.X[:,1] <= regions.flatten().max() + 1e-8).all()
 

    time = 5.
    D_reveal = freespans.optimal_design.inspection_region_reveal(time, regions, D)
    assert D_reveal.X.shape[1] == 2
    assert (D_reveal.X[:,1] >= regions.flatten().min() - 1e-8).all()
    assert (D_reveal.X[:,1] <= regions.flatten().max() + 1e-8).all()
    assert (D_reveal.X[:,0] == time).all()


def test_optimal_design():
    pass