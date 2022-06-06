from typing import Optional, Union
from chex import dataclass
from gpjax import Dataset
import numpy as np
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

Array = Union[np.ndarray, jnp.ndarray]

@dataclass(repr=False)
class SpanData(Dataset):
    """Span dataset class."""

    L: Optional[Array] = None
    T: Optional[Array] = None

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