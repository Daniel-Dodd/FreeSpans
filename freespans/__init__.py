from .types import SpanData
from .utils import Scaler, RegressorMetrics, ClassifierMetrics, confusion_matrix, compute_percentages
from .kernels import DriftKernel
from .simulate import simulate_bernoulli, simulate_gaussian, simulate_indicator
from .optimal_design import box_design, box_reveal, at_reveal, before_reveal, region_reveal, inspection_region_reveal, inspection_region_design
from .models import bernoulli_svgp, gaussian_svgp, kmeans_init_inducing
from .aquisitions import PredictiveEntropy, PredictiveInformation
from .predict import *
from .plots import *

__version__ = "0.1.1"