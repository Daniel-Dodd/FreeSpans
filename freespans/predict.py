import jax.numpy as jnp
from typing import Union

from gpjax import Dataset
from .types import SpanData
from jax import vmap

def get_naive_predictor(train_data: Union[Dataset, SpanData]) -> Union[Dataset, SpanData]:
    """
    Get a naive predictor for the data.
    Args:
        train_data (Dataset): The training data.
    Returns:
        Array: The naive predictor.
    """
    x, y = train_data.X, train_data.y

    indicies = x[:,0].argsort()

    x = x[indicies]
    y = y[indicies]

    locations = jnp.unique(x[:,1])

    y = vmap(lambda loc: y[::-1][(x[::-1,1] == loc).argmax()])(locations).reshape(-1,1)
    x = vmap(lambda loc: x[::-1][(x[::-1,1] == loc).argmax()])(locations).reshape(-1,2)
    
    if isinstance(train_data, SpanData):
        return SpanData(X=x, y=y, L=train_data.L, T=train_data.T)

    else:
        return Dataset(X=x, y=y)


def naive_predictor(train_data: Union[Dataset, SpanData], test_data: Union[Dataset, SpanData]) -> Union[Dataset, SpanData]:
    naive_predictor = get_naive_predictor(train_data)


    x_test = test_data.X
    x_naive = naive_predictor.X
    y_naive = naive_predictor.y

    nt = jnp.unique(x_test[:,0]).shape[0]
    nl = jnp.unique(x_test[:,1]).shape[0]


    _, naive_indicies, test_indicies = jnp.intersect1d(x_naive[:,1], x_test[:,1], return_indices=True)


    vals = jnp.nan * jnp.ones((nt, nl, 1))
    vals =  vals.at[:, test_indicies].set(y_naive[naive_indicies][None, :]).reshape(-1, 1)
    
    if isinstance(test_data, SpanData):
        L = test_data.L
        T = test_data.T
        return SpanData(X=x_test, y=vals, L=L, T=T)

    else:
        return Dataset(X=x_test, y=vals)