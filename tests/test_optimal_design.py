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

    entropy = freespans.pred_entropy(posterior=p,
	                variational_family=q,
	                 params = params,
	                 design = d, 
	                 inner_samples=32, 
	                 outer_samples=16,
	                 seed=42,
	                )

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

    information = freespans.pred_information(posterior=p,
	                variational_family=q,
	                 params = params,
	                 design = d, 
                     test = t,
	                 inner_samples=32, 
	                 outer_samples=16,
	                 seed=42,
	                )

    assert information.shape == ()
    assert information.dtype == jnp.float32 or jnp.float64 or float

@pytest.mark.parametrize("start_pipe, end_pipe", [(0., 1.), (2., 4.7)])
@pytest.mark.parametrize("start_time, end_time", [(0., 1.), (2., 4.7)])
@pytest.mark.parametrize("location_width, time_width", [(.1, .1), (.2, .2)])
def test_box_design(start_time, end_time, start_pipe, end_pipe, time_width, location_width):
    L = jnp.arange(start_pipe, end_pipe, location_width) + location_width/2.
    T = jnp.arange(start_time, end_time + time_width/2, time_width)

    d = freespans.box_design(start_time, end_time, start_pipe, end_pipe, time_width, location_width)

    assert d.shape == (len(T)*len(L), 2)
    assert d.dtype == jnp.float32 or jnp.float64
    assert d[:,0].min() >= start_time
    assert d[:,0].max() <= end_time + 1e-8
    assert d[:,1].min() >= start_pipe
    assert d[:,1].max() <= end_pipe + 1e-8


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

def test_box_reveal():
    pass

#@pytest.mark.parametrize("regions", [jnp.array([[0., .5]]), jnp.array([[-1., 0.], [.5, 1.]])])
def test_region_reveal():
    D = dataset()
    pass

#@pytest.mark.parametrize("regions", [jnp.array([[0., .5]]), jnp.array([[-1., 0.], [.5, 1.]])])
def test_inspection_region_reveal():
    D = dataset()
    pass

def test_optimal_design():
    pass