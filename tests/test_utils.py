import jax.numpy as jnp
from jax import vmap
import gpjax as gpx
from gpjax import Dataset

import freespans

import pytest

@pytest.mark.parametrize("start_pipe, end_pipe", [(0., 10.), (1., 20.)])
@pytest.mark.parametrize("start_time, end_time", [(0., 10.), (1., 20.)])
@pytest.mark.parametrize("location_width, time_width" , [(.5, 1.), (1., .5)])
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
    scaler = freespans.Scaler()

    x1_scaled = scaler(x1)
    x2_scaled = scaler(x2)

    assert x1_scaled.shape == x1.shape
    assert x2_scaled.shape == x2.shape
    assert (x1_scaled == ((x1-x1_mean)/x1_std)).all()
    assert (x2_scaled == ((x2-x1_mean)/x1_std)).all()

    # Test scaler dataset:
    D1 = Dataset(X=x1, y=y1)
    D2 = Dataset(X=x2, y=y2)

    scaler = freespans.Scaler()
    D1_scaled = scaler(D1)
    D2_scaled = scaler(D2)

    assert D1_scaled.X.shape == D1.X.shape
    assert D2_scaled.X.shape == D2.X.shape
    assert (D1_scaled.X == ((D1.X-x1_mean)/x1_std)).all()
    assert (D2_scaled.X == ((D2.X-x1_mean)/x1_std)).all()
    assert isinstance(D1_scaled, Dataset)
    assert isinstance(D2_scaled, Dataset)

def test_confusion_matrix_and_metrics():
    # test 1:
    pred = jnp.array([1.,1., 0., 1., 1., 0., 1.])
    true = jnp.array([1.,0., 0., 0., 1., 1., 1.])
    cm = freespans.confusion_matrix(true, pred)
    reg_met = freespans.RegressorMetrics(pred_labels=pred, true_labels=true)
    class_met = freespans.ClassifierMetrics(pred_labels=pred, true_labels=true)

    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]

    assert cm.dtype == "int32"
    assert cm[0,0] == 1
    assert cm[0,1] == 2
    assert cm[1,0] == 1
    assert cm[1,1] == 3

    assert reg_met.MAE() == jnp.mean((true - pred) ** 2)
    assert reg_met.RMSE() == jnp.sqrt(jnp.mean((true - pred) ** 2))
    assert reg_met.MSE() == jnp.mean((true - pred) ** 2)

    assert class_met.tp == tp
    assert class_met.tn == tn
    assert class_met.fn == fn
    assert class_met.fp == fp
    assert (class_met.cm == cm).all()

    assert class_met.tpr() == tp/(tp + fn)
    assert class_met.fpr() == fp/(tn + tp)
    assert class_met.recall() == tp/(tp + fn)
    assert class_met.precision() == tp/(tp + fp)

    # test 2:
    pred = jnp.array([1, 1, 0, 1, 1, 1, 1, 1])
    true = jnp.array([1, 0, 1, 1, 0, 1, 1, 1])
    cm = freespans.confusion_matrix(true, pred)
    reg_met = freespans.RegressorMetrics(pred_labels=pred, true_labels=true)
    class_met = freespans.ClassifierMetrics(pred_labels=pred, true_labels=true)

    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]

    assert cm.dtype == "int32"
    assert cm[0,0] == 0
    assert cm[0,1] == 2
    assert cm[1,0] == 1
    assert cm[1,1] == 5

    assert reg_met.MAE() == jnp.mean((true - pred) ** 2)
    assert reg_met.RMSE() == jnp.sqrt(jnp.mean((true - pred) ** 2))
    assert reg_met.MSE() == jnp.mean((true - pred) ** 2)

    assert class_met.tp == tp
    assert class_met.tn == tn
    assert class_met.fn == fn
    assert class_met.fp == fp
    assert (class_met.cm == cm).all()

    assert class_met.tpr() == tp/(tp + fn)
    assert class_met.fpr() == fp/(tn + tp)
    assert class_met.recall() == tp/(tp + fn)
    assert class_met.precision() == tp/(tp + fp)



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


def dataset1():
    # Create pipe locations and time indicies:
    L = jnp.arange(0, 20, 1.)
    t = 1.
    
    # Create covariates:
    X = vmap(lambda l: jnp.array([t,l]))(L).reshape(-1, 2)

    # Create labels
    y = jnp.zeros_like(X[:,0]).reshape(-1, 1)
    D = gpx.Dataset(X=X, y=y)

    return D

def dataset2():
    # Create pipe locations and time indicies:
    L = jnp.arange(5, 10, 1.)
    t = 2.
    
    # Create covariates:
    X = vmap(lambda l: jnp.array([t,l]))(L).reshape(-1, 2)

    # Create labels
    y = jnp.ones_like(X[:,0]).reshape(-1, 1)
    D = gpx.Dataset(X=X, y=y)

    return D

def test_compute_percentages():
    D = dataset()
    regions = jnp.array([[-.5, 0.], [.5, 1.5]])
    percentage = freespans.compute_percentages(regions, D)
    assert percentage.shape == ()
    assert percentage.dtype == jnp.float32 or jnp.float64

def test_naive_predictor():
    D1 = dataset1()
    D2 = dataset2()

    # combine datasets:
    D = D1 + D2

    # test naive predictor:
    naive_predictor = freespans.naive_predictor(D)

    assert naive_predictor.shape == (20, 1)
    assert naive_predictor.dtype == jnp.float32 or jnp.float64
    assert (naive_predictor[:5] == jnp.array([0., 0., 0., 0., 0.])).all()
    assert (naive_predictor[5:10] == jnp.array([1., 1., 1., 1., 1.])).all()
    assert (naive_predictor[10:] == jnp.array([0., 0., 0., 0., 0.])).all()