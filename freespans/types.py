from typing import Optional, Union
from chex import dataclass
from gpjax import Dataset
import numpy as np
import jax.numpy as jnp
from jax import vmap

Array = Union[np.ndarray, jnp.ndarray]

@dataclass(repr=False)
class SpanData(Dataset):
    """Span dataset class."""

    L: Optional[Array] = None
    T: Optional[Array] = None

    def __add__(self, other: 'SpanData') -> 'SpanData':
        """Add two datasets together."""
        x = jnp.concatenate((self.X, other.X))
        y = jnp.concatenate((self.y, other.y))

        if (self.L is not None) & (other.L is not None) & (self.T is not None) & (other.T is not None):
            L = jnp.unique(jnp.concatenate((self.L, other.L)))
            T = jnp.unique(jnp.concatenate((self.T, other.T)))
            return SpanData(X=x, y=y, L=L, T=T)

        else:
            return SpanData(X=x, y=y)

    def __post_init__(self):
        if self.L is None and self.T is None:
            self.L = jnp.unique(self.X[:,1])
            self.T = jnp.unique(self.X[:,0])
        
    def __repr__(self) -> str:
        string = f"- Number of datapoints: {self.X.shape[0]}\n- Dimension:" f" {self.X.shape[1]}"
        if self.T is not None:
            string += "\n- Years:" f" {self.T.min()}-{self.T.max()}"
        if self.L is not None:
             string += "\n- Pipe KP (km):" f" {round(self.L.min(),ndigits=10)}-{round(self.L.max(),ndigits=10)}"
        return string

    @property
    def nt(self) -> int:
        """Number of temporal points."""
        return self.T.shape[0]
    
    @property
    def nl(self) -> int:
        """Number of spatial points."""
        return self.L.shape[0]

@dataclass(repr=False)
class SimulatedSpanData(SpanData):
    """Span dataset class for artificial data."""
    f: Optional[Array] = None

    def __add__(self, other: 'SpanData') -> 'SpanData':
        """Add two datasets together."""
        x = jnp.concatenate((self.X, other.X))
        y = jnp.concatenate((self.y, other.y))
        f = jnp.concatenate((self.f, other.f))

        if (self.L is not None) & (other.L is not None) & (self.T is not None) & (other.T is not None):
            L = jnp.unique(jnp.concatenate((self.L, other.L)))
            T = jnp.unique(jnp.concatenate((self.T, other.T)))
            return SimulatedSpanData(X=x, y=y, L=L, T=T, f=f)

        else:
            return SimulatedSpanData(X=x, y=y, f=f)