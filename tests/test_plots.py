from tkinter import N
import jax.numpy as jnp
from jax import vmap
import gpjax as gpx
from gpjax import Dataset

import freespans
from freespans.types import SpanData
from freespans.utils import Scaler

import matplotlib.pyplot as plt

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


@pytest.mark.parametrize("n", [1, 2, 10])
def test_elbo(n):
    hist = jnp.arange(n)

    freespans.plots.plot_elbo(hist)


def test_make_grid():
    D = Spandataset1()
    val = jnp.ones_like(D.y)
    grid = freespans.plots._make_grid(D, val)

    assert grid.shape == (D.nl, D.nt)

    # test missing values
    D.y = D.y[:5]
    D.X = D.X[:5]
    grid = freespans.plots._make_grid(D, D.y)
    assert grid.shape == (D.nl, D.nt)


def test_make_mesh():
    D = Spandataset1()
    XX, YY = freespans.plots._make_mesh(D)
    assert XX.shape == (D.nl, D.nt)
    assert YY.shape == (D.nl, D.nt)


def test_title_labels_and_ticks():
    plt.plot()
    freespans.plots._title_labels_and_ticks


@pytest.mark.parametrize("D, plot_binary", 
                        [(freespans.simulate.simulate_gaussian(gpx.RBF(), 0, 3, 0, 3), False), 
                        (freespans.simulate.simulate_bernoulli(gpx.RBF(), 0, 3, 0, 3), True)])
@pytest.mark.parametrize("latent", [True, False])
@pytest.mark.parametrize("drift", [None, 1.])
@pytest.mark.parametrize("scale", [True, False])
def test_plot_truth_and_visualise(D, plot_binary, latent, drift, scale):
    freespans.plots._plot_latent(D)
    freespans.plots._plot_truth(D, plot_binary=plot_binary)

    if scale:
        scaler = Scaler()
        D = scaler(D)
    else:
        scaler = None
    freespans.plots.visualise(D, plot_binary = plot_binary, latent = latent, drift_angle = drift, drift_scaler = scaler)

def test_visualise_latent_spandata():
    D =  Spandataset1()
    with pytest.raises(TypeError):
        freespans.plots.visualise(D, latent = True)

def test_plot_roc():
    # Model:
    D =  Spandataset1()
    likelihood = gpx.Bernoulli(num_datapoints=D.n)
    prior = gpx.Prior(kernel=gpx.RBF())
    posterior =  prior * likelihood
    
    # Variational family:
    z = freespans.kmeans_init_inducing(D, 10)
    q = gpx.WhitenedVariationalGaussian(prior=prior, inducing_inputs=z)

    x = D.X
    y = q(q.params)(x).mean()
    pred_data = gpx.Dataset(X=x, y=y)

    fig, ax = plt.subplots()
    freespans.plots.plot_roc(pred_data, D, ax=None)
    freespans.plots.plot_roc(pred_data, D, ax=ax)


def test_plot_pr():
    # Model:
    D =  Spandataset1()
    likelihood = gpx.Bernoulli(num_datapoints=D.n)
    prior = gpx.Prior(kernel=gpx.RBF())
    posterior =  prior * likelihood
    
    # Variational family:
    z = freespans.kmeans_init_inducing(D, 10)
    q = gpx.WhitenedVariationalGaussian(prior=prior, inducing_inputs=z)

    x = D.X
    y = q(q.params)(x).mean()
    pred_data = gpx.Dataset(X=x, y=y)

    fig, ax = plt.subplots()
    freespans.plots.plot_pr(pred_data, D, ax=None)
    freespans.plots.plot_pr(pred_data, D, ax=ax)

def test_plot_naive_roc():
    D1 = Spandataset1()
    D2 = Spandataset2()

    fig, ax = plt.subplots()
    freespans.plots.plot_naive_roc(D1, D2, ax=None)
    freespans.plots.plot_naive_roc(D1, D2, ax=ax)



def test_plot_naive_pr():
    D1 = Spandataset1()
    D2 = Spandataset2()

    freespans.plots.plot_naive_pr(D1, D2)

    fig, ax = plt.subplots()
    freespans.plots.plot_naive_pr(D1, D2, ax=None)
    freespans.plots.plot_naive_pr(D1, D2, ax=ax)


def test_plot_rocpr():
    D1 = Spandataset1()
    D2 = Spandataset2()

    # Model:
    likelihood = gpx.Bernoulli(num_datapoints=D1.n)
    prior = gpx.Prior(kernel=gpx.RBF())
    posterior =  prior * likelihood
    
    # Variational family:
    z = freespans.kmeans_init_inducing(D1, 10)
    q = gpx.WhitenedVariationalGaussian(prior=prior, inducing_inputs=z)

    # Create predicted datasets:
    x = D1.X
    y = q(q.params)(x).mean()
    pred_data_1 = gpx.Dataset(X=x, y=y)
    pred_data_2 = gpx.Dataset(X=x, y=y/2)
    pred_datasets = {"A": pred_data_1, "B": pred_data_2}


    freespans.plots.plot_rocpr(pred_datasets = pred_datasets, test_data = D1, naive_data = None)
    freespans.plots.plot_rocpr(pred_datasets = pred_datasets, test_data = D1, naive_data = D2)





