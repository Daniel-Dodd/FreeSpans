import matplotlib.pyplot as plt
from typing import List, Optional
from sklearn.metrics import roc_curve, precision_recall_curve

from .utils import ClassifierMetrics, Scaler
from .types import SpanData, SimulatedSpanData, Array
from gpjax import Dataset
import jax.numpy as jnp
from jax import vmap

# THIS FILE IS CURRENTLY A MESS.

def visualise(data: SpanData):
    """
    Visualise span datasets.
    Args:
        data (SpanData): The data to visualise.
    Returns:
        Plot.
    """
    if isinstance(data, SimulatedSpanData):
        return _visualise_simulated_data(data)
    else:
        return _visualise_real_data(data)

def _visualise_real_data(data: SpanData):
    """Visualise span dataset."""
    plt.figure(figsize=(6, 3), constrained_layout=True)
    XX, YY = jnp.meshgrid(data.T, data.L)
    plt.ylabel("Year")
    plt.xlabel("Pipe KP (km)")

    if all((data.y == 1.) + (data.y == 0.)):
        if data.n < data.nt * data.nl:
            print("Missing data!")
            Xplot = vmap(lambda t: vmap(lambda l: jnp.array([t,l]))(data.L))(data.T).reshape(-1, 2)
            indicies = vmap(lambda x: (Xplot == x).all(axis=1).argmax(), in_axes=0)(data.X)
            yplot = -1. * jnp.ones((data.nt * data.nl, 1)) # -1. is for missing data
            yplot = yplot.at[indicies].set(data.y)
            plt.contourf(YY, XX, yplot.reshape(data.nt, data.nl).T, levels=1, colors=['red','none', 'black'])
        else:
            plt.contourf(YY, XX, data.y.reshape(data.nt, data.nl).T, levels=1, colors=['none', 'black'])
    else:
        plt.contourf(YY, XX, data.y.reshape(data.nt, data.nl).T, levels=10)
        plt.colorbar()

    plt.yticks(jnp.arange(int(data.T.min()), int(data.T.max()) + 1, step=1))

def _visualise_simulated_data(data: SpanData):
    """Visualise span dataset."""
    plt.figure(figsize=(12,4))
    XX, YY = jnp.meshgrid(data.T, data.L)

    plt.subplot(1,2,1)
    plt.title('Observations')
    plt.ylabel("Time")
    plt.xlabel("Pipe")
    plt.yticks(jnp.arange(int(data.T.min()), int(data.T.max()) + 1, step=1))

    if all((data.y == 1.) + (data.y == 0.)):
        plt.contourf(YY, XX, data.y.reshape(data.nt, data.nl).T, levels=1, colors=['none', 'black'])
    else:
        plt.contourf(YY, XX, data.y.reshape(data.nt, data.nl).T, levels=10)
        plt.colorbar()

    plt.subplot(1,2,2)
    plt.title('Seabed')
    plt.ylabel("Time")
    plt.xlabel("Pipe")
    plt.contourf(YY,XX, data.f.reshape(data.nt, data.nl).T, levels=10)
    plt.colorbar()
    plt.yticks(jnp.arange(int(data.T.min()), int(data.T.max()) + 1, step=1))

    plt.tight_layout()



def plot_rocpr(truth: SpanData, 
                 preds: List[Array], 
                 names: List, 
                 year: Array = None,
                 naive: bool = None,
                ) -> plt:
    """
    Plot the reciever oper ROC and precision recall (PR) curves.
    Args:
        truth (SpanData): The true data.
        preds (List[Array]): The predictions of the model.
        names (List): The names of the predictions.
        year (Array): The year to plot.
        naive (bool): Whether to plot the naive curve.
    Returns:    
        Plot
    """
    
    y = truth.y
    
    plt.figure(figsize=(10, 4))
    
    if year is not None:
        if naive is not None:
            truth_y_as_ts = truth.y.reshape(truth.nt, truth.nl)
            naive = truth_y_as_ts[year - 1 - truth.T.min()]
        
        def get_year(vals: Array) -> Array:
            return vals.reshape(truth.nt, truth.nl)[year - truth.T.min()]
        
        y = get_year(y)
        preds = list(map(get_year, preds))
        

    for pred, name in zip(preds, names):
        plt.subplot(1,2,1)
        fpr, tpr, _ = roc_curve(y, pred)
        plt.plot(fpr,tpr, label = name)

        plt.subplot(1,2,2)
        prec, rec, _ = precision_recall_curve(y, pred)
        plt.plot(prec, rec, label = name)

        
    if naive is not None:
        naive_metr = ClassifierMetrics(true_labels = y, 
                                           pred_labels = naive)
        plt.subplot(1,2,1)
        fpr, tpr = naive_metr.fpr(), naive_metr.tpr()
        plt.plot(fpr,tpr,"*", markersize=8 ,label = "Naive")
        
        plt.subplot(1,2,2)
        precision, recall = naive_metr.precision(), naive_metr.recall()
        plt.plot(precision, recall,"*", markersize=8 ,label = "Naive")
    
    
    plt.subplot(1,2,1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()

    plt.subplot(1,2,2)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend()

    plt.tight_layout()



def plot_mse(pred_labels: Array, true_data: SpanData, names: List[str], year: int, naive: Optional[bool] = True):
    """
    Plot the mean squared error of the prediction.
    Args:
        pred_labels (Array): The predicted labels.
        true_data (SpanData): The true data.
        names (List[str]): The names of the predictions.
        year (int): The year to plot.
        naive (bool): Whether to plot the naive prediction.
    Returns:
        Plot
    """

    L = true_data.L
    T = true_data.T
    y = true_data.y.reshape(len(T),len(L))
    
    def f(thresh, pred):
        pred = pred.reshape(len(T),len(L))
        return sum((y[year - T.min()] - (pred[year - T.min()]>thresh))**2)/len(L)

    thresh = jnp.linspace(1e-6, 1 - 1e-6, 100)

    plt.figure(figsize=(6, 3), constrained_layout=True)
    for pred, name in zip(pred_labels, names):
        values = jnp.array([f(thr, pred) for thr in thresh])
        plt.plot(thresh, values, label = name)
    
    if naive:
        naive = sum((y[year - T.min()]-y[year - T.min()-1])**2)/len(L)
        plt.hlines(naive, xmin = thresh[0], xmax = thresh[-1], label = "Naive", color= "purple")
        
    plt.legend()
    


def plot_pred(pred_labels: Array, true_data: SpanData):
    """
    Plot the predictions of the model.
    Args:
        pred_labels (Array): The predictions of the model.
        true_data (SpanData): The true data.
    Returns:
        Plot
    """
    plt.figure(figsize=(6, 3), constrained_layout=True)
    XX, YY = jnp.meshgrid(true_data.T, true_data.L)
    plt.contourf(YY,XX, pred_labels.reshape(len(true_data.T),len(true_data.L)).T)
    plt.yticks(jnp.arange(int(true_data.T.min()),int(true_data.T.max())+1, step=1))
    plt.ylabel("Year")
    plt.xlabel("Pipe KP (km)")
    plt.colorbar()

def add_drift(test_data: Dataset, theta: Array, scaler: Scaler = None):
    """
    Add drift to the prediction plot.
    Args:
        test_data (Dataset): The test data.
        theta (Array): The drift angle parameter of the model.
        scaler (Scaler): The scaler used to scale the data.
    Returns:
        Plot
    """

    x = jnp.linspace(test_data.L.min(), test_data.L.max(), 200)
    
    if scaler is not None:
        x_scaled = (x - scaler.mu[1])/scaler.sigma[1]
        y_scaled = jnp.tan(theta) * x_scaled
        y = scaler.mu[0] + y_scaled * scaler.sigma[0]
    else:
        y = jnp.tan(theta) * x

    plt.plot(x, y, color="red")
    plt.ylim(test_data.T.min(), test_data.T.max())



def plot_elbo(history: Array, plot_step: Optional[int] = 10):
    """Plot ELBO training history.
    Args:
        history (Array): Training history values.
        plot_step (int, optional): Thins training history for a clearer plot. Defaults to 10.
    Returns:
        Plot
    """
    elbo_history = -jnp.array(history)
    total_iterations = elbo_history.shape[0]
    iterations = jnp.arange(1, total_iterations + 1)

    plt.figure(constrained_layout = True)
    plt.plot(iterations[::plot_step], elbo_history[::plot_step])
    plt.ylabel("ELBO")
    plt.xlabel("Iterations")