import jax.numpy as jnp
import pytest

from freespans.plots import plot_elbo

@pytest.mark.parametrize("length, plot_step", [(1, 1), (20, 2), (100, 10)])
def test_plot_elbo(length, plot_step):
    history =  jnp.arange(length)
    plot_elbo(history=history, plot_step=plot_step)
