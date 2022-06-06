from .types import SpanData
from .utils import Scaler, RegressorMetrics, ClassifierMetrics, confusion_matrix
from .kernels import DriftKernel
from .simulate import simulate_bernoulli, simulate_gaussian, simulate_indicator
from .optimal_design import pred_entropy, pred_information, box_design, box_reveal, inpsection_region_reveal, inspection_region_design
from .models import bernoulli_svgp, gaussian_svgp, kmeans_init_inducing
from .plots import plot_rocpr, plot_mse, plot_pred, add_drift


__version__ = "0.1"