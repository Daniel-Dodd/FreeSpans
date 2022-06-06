from copy import deepcopy
from typing import Optional, Union
from .types import Array
import jax.numpy as jnp
from jax import lax

from gpjax import Dataset

from chex import dataclass

@dataclass(repr=False)
class Scaler:
    """Covariate scaler."""
    mu: Optional[Array] = None
    sigma: Optional[Array] = None
     
    def __call__(self, data: Union[Array, Dataset]) -> Array:
        """Fit scaler to first observations passed."""
        if isinstance(data, Dataset):
            scaled_data = deepcopy(data)
            if self.mu is None or self.sigma is None:
                self.mu = data.X.mean(0)
                self.sigma = jnp.sqrt(data.X.var(0))

            scaled_data .X = (data.X - self.mu)/ self.sigma
              
            return scaled_data 

        else:
            if self.mu is None or self.sigma is None:
                self.mu = data.mean(0)
                self.sigma = jnp.sqrt(data.var(0))
            
            return (data - self.mu)/ self.sigma

def combine(a: Dataset, b: Dataset) -> Dataset:
    """Combine two datasets.
    Args:
        a (gpjax.Dataset): GPJax dataset.
        b (gpjax.Dataset): GPJax dataset.
    Returns:
        gpjax.Dataset: Combined data.
    """

    x = jnp.concatenate((a.X, b.X))  
    y = jnp.concatenate((a.y, b.y))  

    return Dataset(X=x, y=y)
        

@dataclass(repr=False)
class ClassifierMetrics:
    """Compute classifier metrics."""
    true_labels: Array
    pred_labels: Array
    
    def __post_init__(self):
        self.cm = confusion_matrix(self.true_labels, self.pred_labels)
        self.tn = self.cm[0][0]
        self.fn = self.cm[1][0]
        self.tp = self.cm[1][1]
        self.fp = self.cm[0][1]

    def tpr(self) -> float:
        """True positive rate."""
        return self.tp / (self.tp + self.fn)

    def fpr(self) -> float:
        """False positive rate."""
        return self.fp/(self.tn + self.tp)
    
    def recall(self) -> float:
        """Recall."""
        return self.tp / (self.tp + self.fn)
    
    def precision(self) -> float:
        """Precision."""
        return self.tp / (self.tp + self.fp)

@dataclass(repr=False)
class RegressorMetrics:
    """Compute regression metrics."""
    true_labels: Array
    pred_labels: Array

    def MSE(self) -> float:
        """Mean square error."""
        return jnp.mean((self.true_labels - self.pred_labels) ** 2)

    def RMSE(self) -> float:
        """Root mean square error."""
        return jnp.sqrt(jnp.mean((self.true_labels - self.pred_labels) ** 2))
    
    def MAE(self) -> float:
        """Mean absolute error."""
        return jnp.mean(jnp.abs(self.true_labels - self.pred_labels))


def confusion_matrix(true_labels: Array, pred_labels: Array) -> Array:
    """Creates binary confusion matrix.
    Args:
        true_labels (Array): True binary labels.
        pred_labels (Array): Predicted binary labels.
    Returns:
        Array: A 2x2 binary confusion matrix.
    """
    true_labels = jnp.array(true_labels, dtype=jnp.int32)
    pred_labels = jnp.array(pred_labels, dtype=jnp.int32)
    
    cm, _ = lax.scan(
        lambda carry, pair: (carry.at[pair].add(1), None), 
        jnp.zeros((2,2), dtype=jnp.int32), 
        (true_labels, pred_labels)
        )
    return cm

# TO DO: sort this function out.
# def drift(self, units: int) -> "SpanData":
#     """Data drifter shifts data in positive direction (TO DO: negative direction)."""
#     if units<1:
#         return self
#     else:
#         y = self.y_as_ts
  
#     for i in range(1, self.nt):
#         y = y.at[i, :].set(list([0.0] * units * i + list(y[i][: - i * jnp.abs(units)])))
    
#     return SpanData(X=self.X, y = y.reshape(-1,1), L=self.L, T=self.T)