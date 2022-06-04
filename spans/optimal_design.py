from typing import Optional
import jax.random as jr
import jax.numpy as jnp

from .types import Array
from .utils import Scaler


from gpjax import Dataset

def compute_percentages(regions: Array, data: Dataset) -> float:
    """Compute the percentage of spans in each region.
    Args:
        regions (Array): A batch of regions to inspect.
        dataset (Dataset): The dataset to compute the percentages for.
    Returns:
        float: The percentage of spans in each region.
    """
    return jnp.mean(inpsection_region_reveal(regions, data))

def pred_entropy(model,
                 params: dict,
                 design: Array, 
                 inner_samples: Optional[int]=32, 
                 outer_samples: Optional[int]=16,
                 seed: Optional[int]=42,
                ) -> float:
    """Compute predictive entropy of labels under the likelihood via nested Monte Carlo (NMC).

        i.e., we compute the integral - ∫ [∫p(y|f, d) q(f|d) df] log([∫p(y|f, d) q(f|d) df]) dy

    Args:
        model (gpjax.VariationalFamily): GPJax variational model.
        params (dict): The model's parameters.
        end_time (float): The end year for the simulating the data.
        inner_samples (int, optional): The number of inner NMC samples to be used in the approximation.
        outer_samples (int, optional): The number of outer NMC samples to be used in the approximation.
        seed (int, optional): The random seed for simulating the data.
    Returns:
        float: The estimated entropy.
    """
  
    n = outer_samples
    m = inner_samples
    d = design
    
    key = jr.PRNGKey(seed)
    
    key1, key2, key3 = jr.split(key, num = 3)

    fd = model.predict(params)(d)

    # Outer samples.
    fn_samples = fd.sample(seed=key1, sample_shape=(n,))
    yn_samples = model.likelihood.link_function(fn_samples, params).sample(seed=key2)

    # Inner samples.
    fnm_samples = fd.sample(seed=key3, sample_shape=(n,m))

    # Entropy calculation H[y].
    pnm_dist = model.likelihood.link_function(fnm_samples, params)
    pnm_prob = pnm_dist.prob(yn_samples[:, None, :])
    
    prob_sum = jnp.sum(jnp.log(pnm_prob), axis=2)
    prob_max = prob_sum.max(axis=1)
    prob_sum_minus_max = (prob_sum - prob_max[:, None])
    log_exp_sum = jnp.log(jnp.mean(jnp.exp(prob_sum_minus_max), axis=1))

    return jnp.mean(jnp.log(m) - prob_max - log_exp_sum)


def pred_information(model,
                 params: dict,
                 design: Array,
                 test: Array,
                 inner_samples: Optional[int]=128, 
                 outer_samples: Optional[int]=32,
                 seed: Optional[int]=42,
                ) -> float:
    """Compute predictive mutual information of labels under the likelihood via nested Monte Carlo (NMC).

        i.e., we compute the integral ∫ [∫p([y_d, y_t]|f, [d, t]) q(f|[d, t]) df] log([∫p([y_d, y_t]|f, [d, t]) q(f|[d, t]) df]/
                                                                [∫p(y_d|f, d) q(f|d) df][∫p(y_t|f, t) q(f|t) df]) d[y_d, y_t]

    Args:
        model (gpjax.VariationalFamily): GPJax variational model.
        params (dict): The model's parameters.
        end_time (float): The end year for the simulating the data.
        inner_samples (int, optional): The number of inner NMC samples to be used in the approximation.
        outer_samples (int, optional): The number of outer NMC samples to be used in the approximation.
        seed (int, optional): The random seed for simulating the data.
    Returns:
        float: The estimated mutual information.
    """
    
    n = outer_samples
    m = inner_samples
    d = design
    t = test
    dt = jnp.concatenate([d,t])
    
    key = jr.PRNGKey(seed)

    key1, key2, key3 = jr.split(key, num = 3)

    fdt = model.predict(params)(dt)
    fd = model.predict(params)(d)
    ft = model.predict(params)(t)

    # Outer samples.
    fdt_n_samples = fdt.sample(seed=key1, sample_shape=(n,))
    ydt_n_samples = model.likelihood.link_function(fdt_n_samples, params).sample(seed=key2)
    yd_n_samples = ydt_n_samples[:,:d.shape[0]]
    yt_n_samples = ydt_n_samples[:,d.shape[0]:]

    # Inner samples.
    fdt_nm_samples = fdt.sample(seed=key3, sample_shape=(n,m))
    fd_nm_samples = fd.sample(seed=key3, sample_shape=(n,m))
    ft_nm_samples = ft.sample(seed=key3, sample_shape=(n,m))

    # Entropy calculation H[y_d, y_t].
    pdt_nm_dist = model.likelihood.link_function(fdt_nm_samples, params)
    pdt_nm_prob = pdt_nm_dist.prob(ydt_n_samples[:, None, :])

    prob_sum = jnp.sum(jnp.log(pdt_nm_prob), axis=2)
    prob_max = prob_sum.max(axis=1)
    prob_sum_minus_max = (prob_sum - prob_max[:, None])
    log_exp_sum = jnp.log(jnp.mean(jnp.exp(prob_sum_minus_max), axis=1))

    Hdt = jnp.mean(jnp.log(m) - prob_max - log_exp_sum)

    # Entropy calculation H[y_d].
    pd_nm_dist = model.likelihood.link_function(fd_nm_samples, params)
    pd_nm_prob = pd_nm_dist.prob(yd_n_samples[:, None, :])

    prob_sum = jnp.sum(jnp.log(pd_nm_prob), axis=2)
    prob_max = prob_sum.max(axis=1)
    prob_sum_minus_max = (prob_sum - prob_max[:, None])
    log_exp_sum = jnp.log(jnp.mean(jnp.exp(prob_sum_minus_max), axis=1))

    Hd = jnp.mean(jnp.log(m) - prob_max - log_exp_sum)

    # Entropy calculation H[y_t].
    pt_nm_dist = model.likelihood.link_function(ft_nm_samples, params)
    pt_nm_prob = pt_nm_dist.prob(yt_n_samples[:, None, :])

    prob_sum = jnp.sum(jnp.log(pt_nm_prob), axis=2)
    prob_max = prob_sum.max(axis=1)
    prob_sum_minus_max = (prob_sum - prob_max[:, None])
    log_exp_sum = jnp.log(jnp.mean(jnp.exp(prob_sum_minus_max), axis=1))

    Ht = jnp.mean(jnp.log(m) - prob_max - log_exp_sum)

    return Ht + Hd - Hdt

def box_design(start_time: int, 
    end_time: int, 
    start_pipe: float, 
    end_pipe: float,
    time_width: Optional[float] = 1.,
    location_width: Optional[float] = 1.,
    scaler: Optional[Scaler] = None,
    ) -> Array:
    """Create discrete box-shaped design space.
    Args:
        start_time (float): The start time for the design space.
        end_time (float): The end time for the design space.
        start_pipe (float): The start of the pipe for the design space.
        end_pipe (float): The end of the pipe for the design space.
        time_width (float, optional): The temporal distance between design points e.g.,
                                1. is once per time unit, .5 is twice per time unit.
        location_width (float, optional): The spatial distance between design points.
        scaler (Scaler, optional): A scalar for the covariates.
    Returns:
        Array: Batch of design points.
    """

    # Create pipe locations and time indicies:
    L = jnp.arange(start_pipe, end_pipe, location_width) + location_width/2.
    T = jnp.arange(start_time, end_time + 1, time_width)
    
    # Create covariates:
    X = jnp.array([[t, l] for t in T for l in L])

    if scaler is not None:
        X = scaler(X)
    
    return X

def inspection_region_design(inspection_time: float, 
    regions: Array, 
    location_width: Optional[float] = 1., 
    scaler: Optional[Scaler] = None,
    ) -> Array:
    """ 
    Create discrete design space for inspection regions.
    Args:
        inspection_time (float): The time at which to inspect the region.
        regions (Array): A batch of regions to inspect.
        location_width (float, optional): The spatial distance between design points.
        scaler (Scaler, optional): A scalar for the covariates.
    Returns:
        Array: A batch of design points.
    """

    # Create pipe locations and time indicies:
    d = {(float(region[0]), float(region[1])): jnp.arange(region[0], region[1], location_width) + location_width/2. for region in regions}
    L = jnp.concatenate(list(d.values()))

    # Create covariates:
    X = jnp.array([[inspection_time, l] for l in L])

    if scaler is not None:
        X = scaler(X)
    
    return X

from .types import SpanData

def at_reveal(at_time, data:Dataset) -> Dataset:
    """
    Filter data to only include points at the given time.
    Args:
        at_time (float): The time at which to filter the data.
        data (Dataset): The data to filter.
    Returns:
        Dataset: The filtered data.
    """
    x, y = data.X, data.y

    indicies = (x[:,0] == at_time)

    return Dataset(X=x[indicies], y=y[indicies])

def before_reveal(before_time, data: Dataset) -> Dataset:
    """
    Filter data to only include data before the reveal.
    Args:
        before_time (float): The time before the reveal.
        data (Dataset): The dataset to filter.
    Returns:
        Dataset: The filtered dataset.
    """
    x, y = data.X, data.y

    indicies = (x[:, 0] <= before_time)

    return Dataset(X=x[indicies], y=y[indicies])

def after_reveal(after_time, data: Dataset) -> Dataset:
    """
    Filter data to only include data after the reveal.
    Args:
        after_time (float): The time after the reveal.
        data (Dataset): The dataset to filter.
    Returns:
        Dataset: The filtered dataset.
    """
    x, y = data.X, data.y

    indicies = (x[:, 0] >= after_time)

    return Dataset(X=x[indicies], y=y[indicies])

def box_reveal(start_time: int, 
    end_time: int, 
    start_pipe: float, 
    end_pipe: float,
    data: Dataset,
    ) -> Dataset:
    """Reveal data from existing dataset.
    Args:
        start_time (float): The start time for the design space.
        end_time (float): The end time for the design space.
        start_pipe (float): The start of the pipe for the design space.
        end_pipe (float): The end of the pipe for the design space.
        data (gpjax.Dataset): The dataset to reveal.
    Returns:
        gpjax.Dataset: Revealed data.
    """
    x, y = data.X, data.y

    indicies = (x[:,0] <= end_time) & (x[:,0] >= start_time) & (x[:,1] <= end_pipe) & (x[:,1] >= start_pipe)
    
    return SpanData(X=x[indicies], y=y[indicies])

def naive_predictor(train_data: Dataset) -> Array:
    """
    Get a naive predictor for the data.
    Args:
        train_data (Dataset): The training data.
    Returns:
        Array: The naive predictor.
    """
    return train_data.y_as_ts[-1]

def make_naive_predictor(train_data: Dataset, test_data: Dataset) -> Array:
    T = jnp.unique(test_data.X[:,0])
    return jnp.array([naive_predictor(train_data) for time in T]).reshape(-1, 1)


def inpsection_region_reveal(inspection_time: float, 
                            regions: Array, 
                            data: Dataset,
                            ) -> Dataset:
    """Reveals data from an existing dataset.
    Args:
        inspection_time (float): The time at which to inspect the region.
        regions (Array): A batch of regions to inspect.
        data (gpjax.Dataset): The dataset to reveal.
    Returns:
        gpjax.Dataset: Revealed data.

    Example:
    
        R1 = jnp.array([1.5, 2.7.])
        R2 = jnp.array([21., 22.])

        regions = jnp.array([R1, R2])
        
        reveal(0, regions, x, y) would return points 
        for the first 1.5 to 2.7 units then from 21 to 22 units.
    
    """
    x, y = data.X, data.y
    
    time_indicies = x[:,0] == inspection_time
    x_time = x[time_indicies]
    y_time = y[time_indicies]
    
    assert len(x_time)!=0, "inspection_time not in dataset covariates"
    
    region_indicies = jnp.array([bool((point < regions.reshape(-1)).argmax() % 2) for point in x_time[:,1]])
    
    return Dataset(X=x_time[region_indicies], y=y_time[region_indicies])