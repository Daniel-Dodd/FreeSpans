from typing import Optional
import jax.random as jr
import jax.numpy as jnp

from .types import Array, SpanData
from jax import vmap


from gpjax import Dataset

def box_design(start_time: int, 
    end_time: int, 
    start_pipe: float, 
    end_pipe: float,
    time_width: Optional[float] = 1.,
    location_width: Optional[float] = 1.
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
    Returns:
        Array: Batch of design points.
    """

    # Create pipe locations and time indicies:
    L = jnp.arange(start_pipe, end_pipe, location_width) + location_width/2.
    T = jnp.arange(start_time, end_time + time_width/2, time_width)
    
    return vmap(lambda t: vmap(lambda l: jnp.array([t,l]))(L))(T).reshape(-1, 2)


def inspection_region_design(inspection_time: float, 
    regions: Array, 
    location_width: Optional[float] = 1.
    ) -> Array:
    """ 
    Create discrete design space for inspection regions.
    Args:
        inspection_time (float): The time at which to inspect the region.
        regions (Array): A batch of regions to inspect.
        location_width (float, optional): The spatial distance between design points.
    Returns:
        Array: A batch of design points.
    """
    L = jnp.arange(regions.min(), regions.max(), location_width) + location_width/2.
    x = vmap(lambda l: jnp.array([inspection_time, l]))(L)

    indicies = vmap(lambda point: ((point < regions.reshape(-1)).argmax() % 2).astype(bool))(x[:,1])
    
    return x[indicies]


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


def region_reveal(
    regions: Array, 
    data: Dataset,
    ) -> Dataset:
    """Filter data to only include points in the given regions.
    Args:
        regions (Array): A batch of regions to inspect.
        data (gpjax.Dataset): The dataset to reveal.
    Returns:
        gpjax.Dataset: Revealed data.
    """
    x, y = data.X, data.y

    indicies = (x[:,1] <= regions.max()) & (x[:,1] >= regions.min())

    return SpanData(X=x[indicies], y=y[indicies])

def inspection_region_reveal(inspection_time: float, 
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
    
    region_indicies = vmap(lambda point: ((point < regions.reshape(-1)).argmax() % 2).astype(bool))(x_time[:,1])
    
    return Dataset(X=x_time[region_indicies], y=y_time[region_indicies])


# NEED TO REWRITE AND TEST:
# def optimal_design(aquisition, posterior, variational_family, params , inspection_time, regions):
    
#     utility = lambda region: aquisition(posterior=posterior,
#                    variational_family=variational_family, 
#                    params=params,            
#                    design = inpsection_region_reveal(inspection_time, jnp.atleast_2d(region)),
#                   )

#     utilities = jnp.array([utility(region) for region in regions])
        
#     return utilities 