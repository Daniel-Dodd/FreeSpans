import jax.numpy as jnp
from jax import vmap
import gpjax as gpx
from gpjax import Dataset

import freespans
from freespans.types import SpanData
import distrax as dx
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

def Spandataset3():
    # Create pipe locations and time indicies:
    L = jnp.arange(10, 30, 1.)
    T = jnp.arange(2, 4, 1.)
    
    # Create covariates:
    X = vmap(lambda t: vmap(lambda l: jnp.array([t,l]))(L))(T).reshape(-1, 2)

    # Create labels
    y = jnp.ones_like(X[:,0]).reshape(-1, 1)
    D = SpanData(X=X, y=y, L=L, T=T)

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


def dataset3():
    # Create pipe locations and time indicies:
    L = jnp.arange(10, 30, 1.)
    T = jnp.arange(2, 4, 1.)
    
    # Create covariates:
    X = vmap(lambda t: vmap(lambda l: jnp.array([t,l]))(L))(T).reshape(-1, 2)

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
    D = freespans.predict.naive_predictor(Spandataset1(), Spandataset3())

    assert isinstance(D, SpanData)
    assert (D.X == Spandataset3().X).all()
    assert (D.y[:10] == Spandataset1().y[:10]).all()
    assert (D.y[20:30] == Spandataset1().y[:10]).all()
    assert jnp.isnan(D.y[10:20]).all()
    assert jnp.isnan(D.y[30:40]).all()

    D = freespans.predict.naive_predictor(dataset1(), dataset3())

    assert isinstance(D, Dataset)
    assert (D.X == Spandataset3().X).all()
    assert (D.y[:10] == Spandataset1().y[:10]).all()
    assert (D.y[20:30] == Spandataset1().y[:10]).all()
    assert jnp.isnan(D.y[10:20]).all()
    assert jnp.isnan(D.y[30:40]).all()


@pytest.mark.parametrize("D", [Spandataset1(), dataset1()])
def test_variational_predict_stuff(D):

    # Model:
    likelihood = gpx.Bernoulli(num_datapoints=D.n)
    prior = gpx.Prior(kernel=gpx.RBF())
    posterior =  prior * likelihood
    
    # Variational family:
    z = freespans.kmeans_init_inducing(D, 10)
    q = gpx.WhitenedVariationalGaussian(prior=prior, inducing_inputs=z)

    variational_predict = freespans.predict.variational_predict(q, q.params, D.X, full_cov=True)
    variational_predict = freespans.predict.variational_predict(q, q.params, D.X, full_cov=False)


    assert isinstance(variational_predict, dx.Distribution)

    mean, std = freespans.predict.predictive_mean_and_std(posterior, q, q.params, D.X)

    assert isinstance(mean, jnp.ndarray)
    assert isinstance(std, jnp.ndarray)



    pred_data = freespans.predict.predicted_data(posterior, q, q.params, D)

    assert (pred_data.X == D.X).all()
    assert pred_data.y.shape == D.y.shape