from typing import Optional, Tuple
from sklearn.cluster import KMeans

import jax.numpy as jnp
import matplotlib.pyplot as plt

import gpjax as gpx
from gpjax.kernels import Kernel
from gpjax.gps import NonConjugatePosterior
from gpjax.variational import VariationalFamily
from gpjax.sparse_gps import VariationalPosterior

from .types import Array, SpanData

def Kmeans_initalise_inducing(train_data: SpanData, num_inducing: int) -> Array:
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

def BernoulliSVGP(
        kernel: Kernel, train_data: SpanData, num_inducing: int
    ) -> Tuple[NonConjugatePosterior, VariationalFamily, VariationalPosterior]:
    """ Initialise inducing inputs via Kmeans.
    Args:
        kernel (Kernel): The GP prior kernel.
        train_data (SpanData): The training span dataset.
        num_inducing (int): The number of inducing inputs.
    Returns:
        Tuple[NonConjugatePosterior, VariationalFamily, VariationalPosterior]: An tuple for SVGP training and prediction.
    """

    # Initialise inducing inputs
    z = Kmeans_initalise_inducing(train_data, num_inducing)
    
    # Model
    prior = gpx.Prior(kernel=kernel)
    likelihood = gpx.Bernoulli(num_datapoints=train_data.n)
    posterior = prior * likelihood

    # Variational family:
    variational_family = gpx.WhitenedVariationalGaussian(prior=prior, inducing_inputs=z)

    # SVGP object:
    svgp = gpx.SVGP(posterior=posterior, variational_family=variational_family)
    
    return posterior, variational_family, svgp

def GaussianSVGP(
    kernel: Kernel, train_data: SpanData, num_inducing: int
) -> Tuple[NonConjugatePosterior, VariationalFamily]:
    """ Initialise inducing inputs via Kmeans.
    Args:
        kernel (Kernel): The GP prior kernel.
        train_data (SpanData): The training span dataset.
        num_inducing (int): The number of inducing inputs.
    Returns:
        Tuple[NonConjugatePosterior, VariationalFamily, VariationalPosterior]: An tuple for SVGP training and prediction.
    """
    # Initialise inducing inputs
    z = Kmeans_initalise_inducing(train_data, num_inducing)
    
    # Model
    prior = gpx.Prior(kernel=kernel)
    likelihood = gpx.Gaussian(num_datapoints=train_data.n)
    posterior = prior * likelihood

    # Variational family:
    variational_family = gpx.WhitenedVariationalGaussian(prior=prior, inducing_inputs=z)

    # SVGP object:
    svgp = gpx.SVGP(posterior=posterior, variational_family=variational_family)
    
    return posterior, variational_family, svgp

def plot_elbo(history: Array, plot_step: Optional[int] = 10):
    """Plot ELBO training history.
    Args:
        history (Array): Training history values.
        plot_step (int, optional): Thins training history for a clearer plot. Defaults to 10.
    Returns:
        Plot
    """
    elbo_history = -jnp.array(history)
    total_iterations = elbo_history.shape[0]
    iterations = jnp.arange(1, total_iterations + 1)

    plt.figure(constrained_layout = True)
    plt.plot(iterations[::plot_step], elbo_history[::plot_step])
    plt.ylabel("ELBO")
    plt.xlabel("Iterations")

    return plt