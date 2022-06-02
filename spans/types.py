from typing import Optional, Union
from chex import dataclass
from gpjax import Dataset
import numpy as np
import jax.numpy as jnp
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

    def visualise(self):
        """Visualise span dataset."""
        plt.figure(figsize=(6, 3), constrained_layout=True)
        XX, YY = jnp.meshgrid(self.T, self.L)
        plt.ylabel("Year")
        plt.xlabel("Pipe KP (km)")

        if all((self.y == 1.) + (self.y == 0.)):
            plt.contourf(YY, XX, self.y_as_ts.T, levels=1, colors=['none', 'black'])
        else:
            plt.contourf(YY, XX, self.y_as_ts.T, levels=10)
            plt.colorbar()
    
        plt.yticks(jnp.arange(int(self.T.min()), int(self.T.max()) + 1, step=1))

    @property
    def nt(self) -> int:
        """Number of temporal points."""
        return self.T.shape[0]
    
    @property
    def nl(self) -> int:
        """Number of spatial points."""
        return self.L.shape[0]

    @property
    def y_as_ts(self) -> Array:
        """Matrix where rows comprise spatial series corresponding to each time point."""
        return self.y.reshape(self.nt, self.nl)

    #def drift(self, units: int) -> "SpanData":
    #    """Data drifter (shifts data in positive direction)."""
    #    if units<1:
    #        return self
    #    else:
    #        y = self.y_as_ts
    #    
    #    for i in range(1, self.nt):
    #        y = y.at[i, :].set(list([0.0] * units * i + list(y[i][: - i * jnp.abs(units)])))
    #    
    #    return SpanData(X=self.X, y = y.reshape(-1,1), L=self.L, T=self.T)

@dataclass(repr=False)
class SimulatedSpanData(SpanData):
    """Span dataset class for artificial data."""

    f: Optional[Array] = None

    def visualise(self):
        """Visualise span dataset."""
        plt.figure(figsize=(12,4))
        XX, YY = jnp.meshgrid(self.T, self.L)

        plt.subplot(1,2,1)
        plt.title('Observations')
        plt.ylabel("Time")
        plt.xlabel("Pipe KP (km)")
        plt.yticks(jnp.arange(int(self.T.min()), int(self.T.max()) + 1, step=1))

        if all((self.y == 1.) + (self.y == 0.)):
            plt.contourf(YY, XX, self.y_as_ts.T, levels=1, colors=['none', 'black'])
        else:
            plt.contourf(YY, XX, self.y_as_ts.T, levels=10)
            plt.colorbar()

        plt.subplot(1,2,2)
        plt.title('Seabed')
        plt.ylabel("Time")
        plt.xlabel("Pipe KP")
        plt.contourf(YY,XX, self.f.reshape(len(self.T),len(self.L)).T, levels=10)
        plt.colorbar()
        plt.yticks(jnp.arange(int(self.T.min()), int(self.T.max()) + 1, step=1))

        plt.tight_layout()

