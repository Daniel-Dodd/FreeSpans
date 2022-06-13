.. role:: hidden
    :class: hidden-section

**********************
API
**********************

Aquisitions
#################################
.. automodule:: freespans.aquisitions
.. currentmodule:: freespans.aquisitions

Abstract aquisitions
*********************************

.. autoclass:: Aquisition
   :members:

.. autoclass:: PredictiveAquisition
   :members:

Approximation strategies
*********************************
.. autoclass:: NestedMonteCarlo
   :members:

Predictive entropy
*********************************
.. autoclass:: PredictiveEntropy
    :members:

Predictive information
*********************************
.. autoclass:: PredictiveInformation
    :members:


Kernels
#################################

.. automodule:: freespans.kernels
.. currentmodule:: freespans.kernels


Drift Kernel
*********************************

.. autoclass:: DriftKernel
   :members:

Models
#################################

.. automodule:: freespans.models
.. currentmodule:: freespans.models

.. autofunction:: kmeans_init_inducing
.. autofunction:: bernoulli_svgp
.. autofunction:: gaussian_svgp


Optimal design
#################################
.. automodule:: freespans.optimal_design
.. currentmodule:: freespans.optimal_design


Designs
*********************************
.. autofunction:: box_design
.. autofunction:: inspection_region_design

Revealers
*********************************
.. autofunction:: box_reveal
.. autofunction:: at_reveal
.. autofunction:: before_reveal
.. autofunction:: region_reveal
.. autofunction:: inspection_region_reveal


Simulate
#################################

.. automodule:: freespans.simulate
.. currentmodule:: freespans.simulate


.. autofunction:: simulate_bernoulli
.. autofunction:: simulate_indicator
.. autofunction:: simulate_gaussian

Types
#################################
.. automodule:: freespans.types
.. currentmodule:: freespans.types

.. autoclass:: SpanData
   :members:

.. autoclass:: SimulatedSpanData
   :members:

Utils
#################################
.. automodule:: freespans.utils
.. currentmodule:: freespans.utils

.. autoclass:: Scalar
   :members:

.. autoclass:: ClassifierMetrics
   :members:

.. autoclass:: RegressorMetrics
   :members:

.. autofunction:: confusion_matrix
.. autofunction:: compute_percentages
.. autofunction:: naive_predictor

