import jax.numpy as jnp
import pytest

from spans.types import SpanData

from gpjax.types import Dataset, NoneType, verify_dataset


def test_nonetype():
    assert isinstance(None, NoneType)


@pytest.mark.parametrize("start_pipe", [0., 1.])
@pytest.mark.parametrize("end_pipe", [10., 20.])
@pytest.mark.parametrize("start_time", [0., -10.])
@pytest.mark.parametrize("end_time", [1., 10.])
@pytest.mark.parametrize("location_width", [.5, 1.])
@pytest.mark.parametrize("time_width", [.5, 1.])
def test_dataset(start_pipe, end_pipe, start_time, end_time, location_width, time_width):

    # Create artificial x and y:
    L = jnp.arange(start_pipe, end_pipe, location_width) + location_width/2.
    T = jnp.arange(start_time, end_time + 1, time_width)

    nt = T.shape[0]
    nl = L.shape[0]

    x = jnp.ones((nt*nl, 2))
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