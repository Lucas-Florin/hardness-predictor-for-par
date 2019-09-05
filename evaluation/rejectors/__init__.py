"""
Some strategies for rejecting samples with given hardness scores.
Partly based on:

@InProceedings{wang2018towards,
  author    = {Wang, Pei and Vasconcelos, Nuno},
  title     = {Towards realistic predictors},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2018},
  pages     = {36--51},
  file      = {:2018_Wang_Towards_Realistic_Predictors_ECCV_2018_paper.pdf:PDF},
}

@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

from .none_rejector import NoneRejector
from .mean_accuracy_rejector import MeanAccuracyRejector
from .median_rejector import MedianRejector
from .threshold_rejector import ThresholdRejector
from .quantile_rejector import QuantileRejector

# TODO: Test rejection strategies.