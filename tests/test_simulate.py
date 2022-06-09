import jax.numpy as jnp

import gpjax as gpx

import pytest

from freespans import Scaler
from freespans.types import SimulatedSpanData

from freespans.simulate import simulate_bernoulli, simulate_gaussian, simulate_indicator

@pytest.mark.parametrize("start_pipe, end_pipe", [(0., 2.), (1., 4.)])
@pytest.mark.parametrize("start_time, end_time", [(0., 3.), (1., 5.)])
@pytest.mark.parametrize("location_width, time_width" , [(.5, 1.), (1., .5)])
@pytest.mark.parametrize("simulator", [simulate_bernoulli, simulate_gaussian, simulate_indicator])
def test_simulate_bernoulli(start_pipe, end_pipe, start_time, end_time, location_width, time_width, simulator):

    kernel = gpx.RBF()

    data = simulator(kernel = kernel,
	            start_time = start_time,
	            end_time = end_time,
	            start_pipe = start_pipe,
	            end_pipe = end_pipe,
	            time_width = time_width,
	            location_width = location_width,
	            seed = 42,
	            scaler = None,
	            )
    
    assert isinstance(data, SimulatedSpanData)
    assert data.X.shape == (data.n, 2)
    assert data.y.shape == (data.n, 1)
    assert data.X.dtype == jnp.float32 or jnp.float64
    assert data.y.dtype == jnp.float32 or jnp.float64

    # test scaler.
    scaler = Scaler()

    scaled_data  =  simulator(kernel = kernel,
	            start_time = start_time,
	            end_time = end_time,
	            start_pipe = start_pipe,
	            end_pipe = end_pipe,
	            time_width = time_width,
	            location_width = location_width,
	            seed = 42,
	            scaler = scaler,
	            )

    assert isinstance(scaled_data, SimulatedSpanData)
    assert scaled_data.X.shape == (data.n, 2)
    assert scaled_data.y.shape == (data.n, 1)
    assert scaled_data.X.dtype == jnp.float32 or jnp.float64
    assert scaled_data.y.dtype == jnp.float32 or jnp.float64 
    assert (scaled_data.X == (data.X - data.X.mean(0))/ data.X.std(0)).all()