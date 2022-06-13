from pyexpat import model
from typing import Optional, Union
import jax.random as jr
import jax.numpy as jnp

from .types import Array
import abc

import abc
from typing import Any, Optional


from gpjax.gps import AbstractPosterior
from gpjax.likelihoods import AbstractLikelihood
from gpjax.variational_families import AbstractVariationalFamily

from chex import dataclass

@dataclass
class Aquisition:
    """Abstract predictive aquisition function class."""
    model: Union[AbstractPosterior, AbstractVariationalFamily]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate aquisition function."""
        return self.evaluate(*args, **kwargs)

    @abc.abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate aquisition function."""
        raise NotImplementedError

@dataclass
class PredictiveAquisition:
    """Abstract predictive aquisition function class."""

    design_likelihood: AbstractLikelihood


@dataclass
class NestedMonteCarlo:
    """Nested Monte Carlo.
    Args:
        inner_samples (int, optional): The number of inner NMC samples to be used in the approximation.
        outer_samples (int, optional): The number of outer NMC samples to be used in the approximation.
        seed (int, optional): The random seed for simulating the data.
    """

    inner_samples: Optional[int] = 32
    outer_samples: Optional[int] = 16
    seed: Optional[int] = 42


@dataclass
class PredictiveEntropy(NestedMonteCarlo, Aquisition, PredictiveAquisition):
    """Compute predictive entropy of labels under the design likelihood via nested Monte Carlo (NMC).

        i.e., we compute the integral - ∫ [∫p(y|f, d) q(f|d) df] log([∫p(y|f, d) q(f|d) df]) dy
    """

    def evaluate(self, params: dict, design: Array, *args: Any, **kwargs: Any) -> float:
        """Evaluate predictive entropy.
        Args:
            params (dict): The parameters of the model and likelihood function.
            design (Array): The design.
        Returns:
        float: The estimated entropy.
        """

        n = self.outer_samples
        m = self.inner_samples
        d = design

        key = jr.PRNGKey(self.seed)

        key1, key2, key3 = jr.split(key, num = 3)

        fd = self.model(*args, **kwargs, params=params)(d)

        # Outer samples.
        fn_samples = fd.sample(seed=key1, sample_shape=(n,))
        yn_samples = self.design_likelihood.link_function(fn_samples, params).sample(seed=key2)

        # Inner samples.
        fnm_samples = fd.sample(seed=key3, sample_shape=(n,m))

        # Entropy calculation H[y].
        pnm_dist = self.design_likelihood.link_function(fnm_samples, params)
        pnm_prob = pnm_dist.prob(yn_samples[:, None, :])

        prob_sum = jnp.sum(jnp.log(pnm_prob), axis=2)
        prob_max = prob_sum.max(axis=1)
        prob_sum_minus_max = (prob_sum - prob_max[:, None])
        log_exp_sum = jnp.log(jnp.mean(jnp.exp(prob_sum_minus_max), axis=1))

        return jnp.mean(jnp.log(m) - prob_max - log_exp_sum)


@dataclass
class PredictiveInformation(NestedMonteCarlo, Aquisition, PredictiveAquisition):
    """Compute predictive mutual information of labels under the likelihood via nested Monte Carlo (NMC).

        i.e., we compute the integral ∫ [∫p([y_d, y_t]|f, [d, t]) q(f|[d, t]) df] log([∫p([y_d, y_t]|f, [d, t]) q(f|[d, t]) df]/
                                                                [∫p(y_d|f, d) q(f|d) df][∫p(y_t|f, t) q(f|t) df]) d[y_d, y_t]
    """
    test_likelihood: Optional[AbstractLikelihood] = None
    joint_likelihood: Optional[AbstractLikelihood] = None

    def __post_init__(self):
        if self.test_likelihood is None:
            self.test_likelihood = self.design_likelihood

        if self.joint_likelihood is None:
            self.joint_likelihood = self.design_likelihood

    def evaluate(self, params: dict, design: Array, test: Array,  *args: Any, **kwargs: Any) -> float:
        """Evaluate mutual information.
            Args:
                params (dict): The parameters of the model and likelihood function.
                design (Array): The design.
            Returns:
            float: The estimated entropy.
        """
    
        n = self.outer_samples
        m = self.inner_samples
        d = design
        t = test
        dt = jnp.concatenate([d, t])
        nd = d.shape[0]

        key = jr.PRNGKey(self.seed)

        key1, key2, key3 = jr.split(key, num = 3)

        fdt = self.model(*args, **kwargs, params=params)(dt)
        fd = self.model(*args, **kwargs, params=params)(d)
        ft = self.model(*args, **kwargs, params=params)(t)

        # Outer samples.
        fdt_n_samples = fdt.sample(seed=key1, sample_shape=(n,))
        ydt_n_samples = self.joint_likelihood.link_function(fdt_n_samples, params).sample(seed=key2)
        yd_n_samples = ydt_n_samples[:,:d.shape[0]]
        yt_n_samples = ydt_n_samples[:,d.shape[0]:]

        # Inner samples.
        fdt_nm_samples = fdt.sample(seed=key3, sample_shape=(n,m))
        fd_nm_samples = fd.sample(seed=key3, sample_shape=(n,m))
        ft_nm_samples = ft.sample(seed=key3, sample_shape=(n,m))

        # Entropy calculation H[y_d, y_t].
        pdt_nm_dist = self.design_likelihood.link_function(fdt_nm_samples, params)
        pdt_nm_prob = pdt_nm_dist.prob(ydt_n_samples[:, None, :])

        prob_sum = jnp.sum(jnp.log(pdt_nm_prob), axis=2)
        prob_max = prob_sum.max(axis=1)
        prob_sum_minus_max = (prob_sum - prob_max[:, None])
        log_exp_sum = jnp.log(jnp.mean(jnp.exp(prob_sum_minus_max), axis=1))

        Hdt = jnp.mean(jnp.log(m) - prob_max - log_exp_sum)

        # Entropy calculation H[y_d].
        pd_nm_dist = self.design_likelihood.link_function(fd_nm_samples, params)
        pd_nm_prob = pd_nm_dist.prob(yd_n_samples[:, None, :])

        prob_sum = jnp.sum(jnp.log(pd_nm_prob), axis=2)
        prob_max = prob_sum.max(axis=1)
        prob_sum_minus_max = (prob_sum - prob_max[:, None])
        log_exp_sum = jnp.log(jnp.mean(jnp.exp(prob_sum_minus_max), axis=1))

        Hd = jnp.mean(jnp.log(m) - prob_max - log_exp_sum)

        # Entropy calculation H[y_t].
        pt_nm_dist = self.test_likelihood.link_function(ft_nm_samples, params)
        pt_nm_prob = pt_nm_dist.prob(yt_n_samples[:, None, :])

        prob_sum = jnp.sum(jnp.log(pt_nm_prob), axis=2)
        prob_max = prob_sum.max(axis=1)
        prob_sum_minus_max = (prob_sum - prob_max[:, None])
        log_exp_sum = jnp.log(jnp.mean(jnp.exp(prob_sum_minus_max), axis=1))

        Ht = jnp.mean(jnp.log(m) - prob_max - log_exp_sum)

        return Ht + Hd - Hdt