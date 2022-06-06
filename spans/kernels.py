import jax.numpy as jnp
from typing import Optional

from gpjax.types import Array
from gpjax.kernels import Kernel
from gpjax.config import Softplus, add_parameter

from chex import dataclass
import distrax as dx


@dataclass(repr=False)
class _DriftKernel:
    base_kernel: Kernel


@dataclass(repr=False)
class DriftKernel(Kernel, _DriftKernel):
    """Base constructor for drift kernels."""
    
    name: Optional[str] = "DriftKernel"
    
    def __post_init__(self):
        self.stationary = True
        self.ndims = 2
        drift_params = {
            "theta": jnp.array([jnp.pi/2.]),
            "scale": jnp.array([1.])
        }
        
        assert self.base_kernel.ndims == 2
        
        add_parameter("scale", Softplus)
        add_parameter("theta", dx.Lambda(lambda x: (1. + jnp.tanh(x))*jnp.pi/2.))
        
        base_kernel_params = self.base_kernel.params
        self._params = {**drift_params, **base_kernel_params}
    
    @property
    def params(self) -> dict:
        return self._params

    def __call__(self, x: Array, y: Array, params: dict) -> Array:
        projection_matrix = self.projection(params=params)     
        return self.base_kernel(jnp.matmul(x, projection_matrix), 
                                jnp.matmul(y, projection_matrix), params=params)
    
    def projection(self, params) -> Array:
        """Method to compute data transformation for drift kernels."""
        
        scale = params["scale"].squeeze()
        cos_theta = jnp.cos(jnp.pi/2. - params["theta"]).squeeze()
        sin_theta = jnp.sin(jnp.pi/2. - params["theta"]).squeeze()
        
        projection_matrix = jnp.array([[cos_theta, -scale * sin_theta],
                                     [sin_theta, scale * cos_theta]])

        return projection_matrix