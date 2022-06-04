import jax.numpy as jnp
from typing import Optional

from gpjax.types import Array
from gpjax.kernels import Kernel
from gpjax.config import Softplus, add_parameter

from chex import dataclass
import distrax as dx

import pytest


def test_kernel_init():
    assert 1 == 1