import jax.numpy as jnp

from gpjax import Dataset

import spans

import pytest

@pytest.mark.parametrize("start_pipe", [0., 1.])
@pytest.mark.parametrize("end_pipe", [10., 20.])
@pytest.mark.parametrize("start_time", [0., -10.])
@pytest.mark.parametrize("end_time", [1., 10.])
@pytest.mark.parametrize("location_width", [.5, 1.])
@pytest.mark.parametrize("time_width", [.5, 1.])
def test_scaler_and_scaler_dataset(start_pipe, end_pipe, start_time, end_time, location_width, time_width):
    L1 = jnp.arange(start_pipe, end_pipe, location_width) + location_width/2.
    T1 = jnp.arange(start_time, end_time + 1, time_width)
    
    x1 = jnp.array([[t, l] for t in T1 for l in L1])
    x1_mean = x1.mean(axis=0)
    x1_std = x1.std(axis=0)

    L2 = jnp.arange(start_pipe+10, end_pipe+10, location_width/3) + location_width/6.
    T2 = jnp.arange(start_time+10, end_time + 11, time_width/3)

    x2 = jnp.array([[t, l] for t in T2 for l in L2])

    y1 = jnp.ones((T1.shape[0]*L1.shape[0], 1))
    y2 = jnp.ones((T2.shape[0]*L2.shape[0], 1))

    # Test scaler:
    scaler = spans.Scaler()

    x1_scaled = scaler(x1)
    x2_scaled = scaler(x2)

    assert x1_scaled.shape == x1.shape
    assert x2_scaled.shape == x2.shape
    assert (x1_scaled == ((x1-x1_mean)/x1_std)).all()
    assert (x2_scaled == ((x2-x1_mean)/x1_std)).all()

    # Test scaler dataset:
    D1 = Dataset(X=x1, y=y1)
    D2 = Dataset(X=x2, y=y2)

    scaler = spans.Scaler()
    D1_scaled = scaler(D1)
    D2_scaled = scaler(D2)

    assert D1_scaled.X.shape == D1.X.shape
    assert D2_scaled.X.shape == D2.X.shape
    assert (D1_scaled.X == ((D1.X-x1_mean)/x1_std)).all()
    assert (D2_scaled.X == ((D2.X-x1_mean)/x1_std)).all()
    assert isinstance(D1_scaled, Dataset)
    assert isinstance(D2_scaled, Dataset)

def test_confusion_matrix():
    # test 1:
    pred = jnp.array([1.,1., 0., 1., 1., 0., 1.])
    true = jnp.array([1.,0., 0., 0., 1., 1., 1.])
    cm = spans.confusion_matrix(pred, true)

    assert cm.dtype == "int32"
    assert cm[0,0] == 1
    assert cm[0,1] == 1
    assert cm[1,0] == 2
    assert cm[1,1] == 3

    # test 2:
    pred = jnp.array([1, 1, 0, 1, 1, 1, 1, 1])
    true = jnp.array([1, 0, 1, 1, 0, 1, 1, 1])
    cm = spans.confusion_matrix(pred, true)

    assert cm.dtype == "int32"
    assert cm[0,0] == 0
    assert cm[0,1] == 1
    assert cm[1,0] == 2
    assert cm[1,1] == 5


