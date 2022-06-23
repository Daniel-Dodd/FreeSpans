import matplotlib.pyplot as plt
from typing import Dict, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

from .utils import ClassifierMetrics, Scaler #, naive_predictor
from .types import SpanData, SimulatedSpanData, Array
from .predict import naive_predictor
from gpjax import Dataset
import jax.numpy as jnp
from jax import vmap, lax


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

    fig, ax = plt.subplots()
    ax.plot(iterations[::plot_step], elbo_history[::plot_step])
    ax.set_ylabel("ELBO")
    ax.set_xlabel("Iterations")


########################################################################################################################
#   Need to add docstrings and write tests for below.                                                                  #
########################################################################################################################

def _make_grid(data: SpanData, val: Array) -> Array:
    """
    Make a grid of values.
    Args:
        data (SpanData): The data to make the grid for.
        val (Array): The values to make the grid for.
    Returns:
        Array: The grid.
    """
    x, L, T, n, nt, nl = data.X, data.L, data.T, data.n, data.nt, data.nl

    if n < nl*nt:
        xpl = vmap(lambda t: vmap(lambda l: jnp.array([t, l]))(L))(T).reshape(-1, 2)
        
        def f(_, xi): 
            return None, (xpl == xi).all(axis=1)
    
        _, res = lax.scan(f, None, x) 
        
        indicies = res.any(axis=0)
    
        val_plot = jnp.nan * jnp.ones((nt * nl, 1))
        val_plot =  val_plot.at[indicies].set(val.reshape(-1,1))
    
    else:
        val_plot = val

    return val_plot.reshape(data.nt, data.nl).T

def _make_mesh(data: SpanData):
    """
    Make a mesh of values.
    Args:
        data (SpanData): The data to make the mesh for.
    Returns:
        Array: The mesh.
    """
    XX, YY = jnp.meshgrid(data.T, data.L)
    return XX, YY

def _title_labels_and_ticks(data: SpanData):
    """
    Add titles and labels to the plot.
    Args:
        data (SpanData): The data to plot.
    """
    plt.title('Seabed')
    plt.ylabel("Time")
    plt.xlabel("Pipe")
    plt.yticks(jnp.arange(int(data.T.min()), int(data.T.max()) + 1, step=1))
    
def _plot_latent(data: SimulatedSpanData):
    """
    Plot the latent variables.
    Args:
        data (SimulatedSpanData): The data to plot.
    """
    grid = _make_grid(data, data.f)
    XX, YY = _make_mesh(data)
    plt.contourf(YY, XX, grid, levels=10)
    _title_labels_and_ticks(data)
    plt.colorbar()
    plt.tight_layout()
    
def _plot_continuous(data: SpanData):
    """
    Plot continuous labels.
    Args:
        data (SpanData): The data to plot.
    Returns:
        Plot.
    """
    XX, YY = _make_mesh(data)
    grid = _make_grid(data, data.y)
    plt.contourf(YY, XX, grid, levels=10)
    _title_labels_and_ticks(data)
    plt.colorbar()
    plt.tight_layout()

def _plot_binary(data: SpanData):
    """
    Plot binary labels.
    """
    XX, YY = _make_mesh(data)
    grid = _make_grid(data, data.y)
    plt.contourf(YY, XX, grid, levels=1, colors=['none', 'black'])
    _title_labels_and_ticks(data)
    plt.colorbar()
    plt.tight_layout()


def _plot_truth(data: SpanData, plot_binary: bool):
    """
    Plot the true data.
    Args:
        data (SpanData): The data to plot.
    Returns:
        Plot.
    """
    if plot_binary == True:
        _plot_binary(data)
    else:
        _plot_continuous(data)

def visualise(data: SpanData, plot_binary: Optional[int] = True, latent: Optional[bool] = False, drift_angle: Optional[Array] = None, drift_scaler: Optional[Scaler] = None):
    """
    Visualise the data.
    Args:
        data (SpanData): The data to visualise.
        latent (bool, optional): Whether to plot the latent variables. Defaults to False.
        drift_angle(Array): The drift angle parameter of the model in π-radians. Defaults to None.
        drift_scaler (Scaler): If scaling has been used (e.g., model training) the scaler be passed here to convert the drift angle back to the original scale. Defaults to None.
    """

    if latent is True:
        if isinstance(data, SimulatedSpanData):
            plt.subplot(1, 2, 1)
            _plot_truth(data, plot_binary)
            plt.subplot(1, 2, 2)
            _plot_latent(data)
        else:
            raise TypeError("No latent variables to visualise.")
    else:
        _plot_truth(data, plot_binary)

    # Add drift angle if provided.
    if drift_angle is not None:
        Lmax = data.L.max()
        Lmin = data.L.min()

        L = jnp.linspace(Lmin, Lmax, 200)
        
        if drift_scaler is not None:
            L_scaled = (L - drift_scaler.mu[1])/drift_scaler.sigma[1]
            T_scaled = jnp.tan(drift_angle) * L_scaled
            T = drift_scaler.mu[0] + T_scaled * drift_scaler.sigma[0]
        else:
            T = jnp.tan(drift_angle) * L

        plt.plot(L, T, color="red")
        plt.ylim(Lmin, Lmax)


def plot_roc(pred_data: Dataset, test_data: Dataset, ax = None, name: str = None):
    """
    Plot the ROC curve.
    Args:
        pred_data (Dataset): The data to plot the ROC curve for.
        test_data (Dataset): The data to plot the ROC curve for.
    """
    fpr, tpr, _ = roc_curve(test_data.y, pred_data.y)
    roc_auc = auc(fpr, tpr)

    if name is None:
        name = 'ROC curve'

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue',
             lw=2, label='{} (area = %0.2f)'.format(name) % roc_auc)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")


def plot_pr(pred_data: Dataset, test_data: Dataset, ax = None, name:str = None):
    """
    Plot the precision recall (PR) curve.
    Args:
        pred_data (Dataset): The data to plot the PR curve for.
        test_data (Dataset): The data to plot the PR curve for.
        ax (Axes): Optional matplotlib axes argument.
    """
    precision, recall, _ = precision_recall_curve(test_data.y, pred_data.y)
    average_precision = average_precision_score(test_data.y, pred_data.y)

    if name is None:
        name = 'Pecision-recall curve'

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(recall, precision, color='blue',
                lw=2, label='{} (area = %0.2f)'.format(name) % average_precision)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")


def plot_naive_roc(naive_data: Dataset, test_data: Dataset, ax=None):
    """
    Plot the ROC for the naive model.
    Args:
        pred_data (Dataset): The data to plot the ROC curve for.
        test_data (Dataset): The data to plot the ROC curve for.
        ax (Axes): Optional matplotlib axes argument.
    """
    if ax is None:
        fig, ax = plt.subplots()

    naive_pred =  naive_predictor(naive_data, test_data)
    naive_metr = ClassifierMetrics(true_labels = test_data.y, pred_labels = naive_pred.y)
    fpr, tpr = naive_metr.fpr(), naive_metr.tpr()
    plt.plot(fpr, tpr, "*", markersize=8, label = "Naive")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")


def plot_naive_pr(naive_data: Dataset, test_data: Dataset, ax=None):
    """
    Plot the PR for the naive model.
    Args:
        pred_data (Dataset): The data to plot the ROC curve for.
        test_data (Dataset): The data to plot the ROC curve for.
        ax (Axes): Optional matplotlib axes argument.
    """
    if ax is None:
        fig, ax = plt.subplots()

    naive_pred =  naive_predictor(naive_data, test_data)
    naive_metr = ClassifierMetrics(true_labels = test_data.y, pred_labels = naive_pred.y)
    precision, recall = naive_metr.precision(), naive_metr.recall()
    plt.plot(precision, recall, "*", markersize=8, label = "Naive")
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")


def plot_rocpr(pred_datasets: Dict[str, Dataset], test_data: Dataset, naive_data: Dataset = None):
    """
    Plot the ROC and PR curves.
    Args:
        pred_datasets (Dict[str, Dataset]): Predicted datasets to plot the ROC and PR curves for.
        test_data (Dataset): The data to plot the ROC and PR curves for.
        naive_data (Dataset): The data to train the naive predictor on.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)
    for name, pred_data in pred_datasets.items():
        plot_roc(pred_data, test_data, ax=ax1, name=name)
        plot_pr(pred_data, test_data, ax=ax2, name=name)

    if naive_data is not None:
        plot_naive_roc(naive_data, test_data, ax=ax1)
        plot_naive_pr(naive_data, test_data, ax=ax2)

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
