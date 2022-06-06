import jax.numpy as jnp
import pytest

from spans.types import SpanData

from gpjax.types import Dataset, verify_dataset

def test_dataset():

    # Create artificial x and y:
    start_pipe = -5.
    end_pipe = 5.
    start_time = 0.
    end_time = 5.
    location_width = .5
    time_width = 1.
    L = jnp.arange(start_pipe, end_pipe, location_width) + location_width/2.
    T = jnp.arange(start_time, end_time + 1, time_width)

    nt = T.shape[0]
    nl = L.shape[0]
    

    x = jnp.array([[t, l] for t in T for l in L])
    y = jnp.ones((nt*nl, 1))

    # Test span dataset speciying L and T:
    D = SpanData(X=x, y=y, L=L, T=T)
    verify_dataset(D)
    assert D.n == nt*nl
    assert D.nt == nt
    assert D.nl == nl
    assert D.in_dim == 2
    assert D.out_dim == 1
    assert D.X.shape == (nt*nl, 2)
    assert D.y.shape == (nt*nl, 1)
    assert isinstance(D, Dataset)
    assert D.L.shape == (nl, )
    assert D.T.shape == (nt, )
    assert (D.L == L).all()
    assert (D.T == T).all()

    D.visualise()

    # Test span dataset without specifying L and T:
    D = SpanData(X=x, y=y)
    verify_dataset(D)
    assert D.n == nt*nl
    assert D.nt == nt
    assert D.nl == nl
    assert D.in_dim == 2
    assert D.out_dim == 1
    assert D.X.shape == (nt*nl, 2)
    assert D.y.shape == (nt*nl, 1)
    assert isinstance(D, Dataset)
    assert D.L.shape == (nl, )
    assert D.T.shape == (nt, )
    assert (D.L == L).all()
    assert (D.T == T).all()

    
    D.visualise()