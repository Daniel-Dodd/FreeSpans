from .types import SpanData
from .utils import Scaler, RegressorMetrics, ClassifierMetrics
from .kernels import DriftKernel
from .simulate import GaussianSim, BernoulliSim
from .optimal_design import PredEntropy, PredMutualInf
from .models import GaussianSVGP, BernoulliSVGP
from .plots import plot_rocpr, plot_mse, plot_pred, add_drift
from .old_kernels import OldDriftKernel, OldDriftMatern12, OldDriftMatern32, OldDriftMatern52


__version__ = "0.1"