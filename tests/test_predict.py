import jax.numpy as jnp
from jax import vmap
import gpjax as gpx
from gpjax import Dataset

import freespans
from freespans.types import SpanData

import pytest

def Spandataset1():
    # Create pipe locations and time indicies:
    L = jnp.arange(0, 20, 1.)
    t = 1.
    
    # Create covariates:
    X = vmap(lambda l: jnp.array([t,l]))(L).reshape(-1, 2)

    # Create labels
    y = jnp.zeros_like(X[:,0]).reshape(-1, 1)
    D = SpanData(X=X, y=y, L=L, T=jnp.array([t]))

    return D

def Spandataset2():
    # Create pipe locations and time indicies:
    L = jnp.arange(5, 10, 1.)
    t = 2.
    
    # Create covariates:
    X = vmap(lambda l: jnp.array([t,l]))(L).reshape(-1, 2)

    # Create labels
    y = jnp.ones_like(X[:,0]).reshape(-1, 1)
    D = SpanData(X=X, y=y, L=L, T=jnp.array([t]))

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


def test_get_naive_predictor():
    D1 = dataset1()
    D2 = dataset2()

    # combine datasets:
    D = D1 + D2

    # test naive predictor:
    naive_predictor = freespans.predict.get_naive_predictor(D)

    assert isinstance(naive_predictor, Dataset)
    assert naive_predictor.y.shape == (20, 1)
    assert naive_predictor.y.dtype == jnp.float32 or jnp.float64
    assert (naive_predictor.y[:5] == jnp.array([0., 0., 0., 0., 0.])).all()
    assert (naive_predictor.y[5:10] == jnp.array([1., 1., 1., 1., 1.])).all()
    assert (naive_predictor.y[10:] == jnp.array([0., 0., 0., 0., 0.])).all()

    D1 = Spandataset1()
    D2 = Spandataset2()

    # combine datasets:
    D = D1 + D2

    # test naive predictor:
    naive_predictor = freespans.predict.get_naive_predictor(D)

    assert isinstance(naive_predictor, SpanData)
    assert naive_predictor.y.shape == (20, 1)
    assert naive_predictor.y.dtype == jnp.float32 or jnp.float64
    assert (naive_predictor.y[:5] == jnp.array([0., 0., 0., 0., 0.])).all()
    assert (naive_predictor.y[5:10] == jnp.array([1., 1., 1., 1., 1.])).all()
    assert (naive_predictor.y[10:] == jnp.array([0., 0., 0., 0., 0.])).all()


def test_naive_predictor():
    pass