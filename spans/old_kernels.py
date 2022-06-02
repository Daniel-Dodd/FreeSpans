import abc
import tensorflow_probability.substrates.jax.bijectors as tfb
import jax.numpy as jnp
from typing import Optional

from gpjax.kernels import Kernel, euclidean_distance
from gpjax.types import Array
from gpjax.config import add_parameter

from chex import dataclass

@dataclass(repr=False)
class OldDriftKernel(Kernel):
    """Base constructor for drift kernels."""

    name: Optional[str] = "DriftKernel"
    
    def __post_init__(self):
        self.active_dims = [0,1]
        self.stationary = True
        self.ndims = 2
        self._params = {
            "diag": jnp.array([1.0, 1.0]),
            "rho": jnp.array([0.0]),
            "variance": jnp.array([1.0])}
        
        add_parameter("diag", tfb.Softplus())
        add_parameter("rho", tfb.Tanh())

    @abc.abstractmethod
    def __call__(self, x: Array, y: Array, params: dict) -> Array:
        raise NotImplementedError
    
    def quad(self, x, y, params):
        """Method to compute the quadratic form of drift kernels."""

        # Compute Eigen decomposition of drift kernel covariance:
        diag = params["diag"]
        rho = params["rho"]

        cov = jnp.array([[diag[0]**2 + 1e-6, diag[0]*diag[1]*rho[0]],
               [diag[0]*diag[1]*rho[0], diag[1]**2 + 1e-6]])
               
        eigen_values, eigen_basis = jnp.linalg.eigh(cov)

        # Transform data through othorgonal projection:
        x = jnp.matmul(x, eigen_basis)
        y = jnp.matmul(y, eigen_basis)

        x = self.slice_input(x) / eigen_values
        y = self.slice_input(y) / eigen_values

        return euclidean_distance(x, y)
    
    
@dataclass(repr=False)
class OldDriftMatern12(OldDriftKernel):
    """Drift Matérn kernel with smoothness parameter fixed at 0.5."""
    name: Optional[str] = "Matern 1/2 Drift"

    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> Array:
        tau = self.quad(x, y, params)
        K = params["variance"] * jnp.exp(-0.5 * tau)
        return K.squeeze()
    
@dataclass(repr=False)
class OldDriftMatern32(OldDriftKernel):
    """Drift Matérn kernel with smoothness parameter fixed at 1.5."""
    name: Optional[str] = "Matern 3/2 Drift"

    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> Array:
        tau = self.quad(x, y, params)
        K = params["variance"] * (1.0 + jnp.sqrt(3.0) * tau) * jnp.exp(-jnp.sqrt(3.0) * tau)
        return K.squeeze()
    
@dataclass(repr=False)
class OldDriftMatern52(OldDriftKernel):
    """Drift Matérn kernel with smoothness parameter fixed at 2.5."""
    name: Optional[str] = "Matern 5/2 Drift"

    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> Array:
        tau = self.quad(x, y, params)
        K = (
            params["variance"]
            * (1.0 + jnp.sqrt(5.0) * tau + 5.0 / 3.0 * jnp.square(tau))
            * jnp.exp(-jnp.sqrt(5.0) * tau)
        )
        return K.squeeze()