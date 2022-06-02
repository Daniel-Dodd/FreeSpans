from typing import Optional
from sklearn.cluster import KMeans

import jax.numpy as jnp
import matplotlib.pyplot as plt

import gpjax as gpx
from gpjax.kernels import Kernel

from .types import Array, SpanData

def kmeans_init_inducing(train_data: SpanData, num_inducing: int) -> Array:
    """ Initialise inducing inputs via Kmeans.
    Args:
        train_data (SpanData): The training span dataset.
        num_inducing (int): The number of inducing inputs.
    Returns:
        Array: An set of inducing points.
    """
    kmeans = KMeans(n_clusters = num_inducing, random_state = 0).fit(train_data.X)
    z = kmeans.cluster_centers_.copy()
    return z

def bernoulli_svgp(
        kernel: Kernel, train_data: SpanData, num_inducing: int
    ):
    """ Initialise inducing inputs via Kmeans.
    Args:
        kernel (Kernel): The GP prior kernel.
        train_data (SpanData): The training span dataset.
        num_inducing (int): The number of inducing inputs.
    Returns:
        Tuple[NonConjugatePosterior, VariationalFamily, VariationalPosterior]: An tuple for SVGP training and prediction.
    """

    # Initialise inducing inputs
    z = kmeans_init_inducing(train_data, num_inducing)
    
    # Model
    prior = gpx.Prior(kernel=kernel)
    likelihood = gpx.Bernoulli(num_datapoints=train_data.n)
    posterior = prior * likelihood

    # Variational family:
    variational_family = gpx.WhitenedVariationalGaussian(prior=prior, inducing_inputs=z)

    # SVGP object:
    svgp = gpx.SVGP(posterior=posterior, variational_family=variational_family)
    
    return posterior, variational_family, svgp

def gaussian_svgp(
    kernel: Kernel, train_data: SpanData, num_inducing: int
):
    """ Initialise inducing inputs via Kmeans.
    Args:
        kernel (Kernel): The GP prior kernel.
        train_data (SpanData): The training span dataset.
        num_inducing (int): The number of inducing inputs.
    Returns:
        Tuple[NonConjugatePosterior, VariationalFamily, VariationalPosterior]: An tuple for SVGP training and prediction.
    """
    # Initialise inducing inputs
    z = kmeans_init_inducing(train_data, num_inducing)
    
    # Model
    prior = gpx.Prior(kernel=kernel)
    likelihood = gpx.Gaussian(num_datapoints=train_data.n)
    posterior = prior * likelihood

    # Variational family:
    variational_family = gpx.WhitenedVariationalGaussian(prior=prior, inducing_inputs=z)

    # SVGP object:
    svgp = gpx.SVGP(posterior=posterior, variational_family=variational_family)
    
    return posterior, variational_family, svgp