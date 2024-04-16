"""
This file contains the code for the wrapper class SpacePartitioning
that calculates the space partitions and the statistics per region.
The statistics are calculated using the distance_measures.py file.
The space partitions are calculated using the kd_tree_partioning.py file.
"""

import copy
import typing
from dataclasses import dataclass

import numpy as np
import scipy
from scipy.spatial import KDTree

import distance_measures
import kd_tree_partioning


@dataclass
class SpacePartitioning:
    space_partitions: kd_tree_partioning.KDTreePartitions
    kdtree: KDTree

    def __init__(self,
                 input_test_data: np.ndarray,
                 leafsize: int,
                 balanced_tree: bool = True,
                 compact_nodes: bool = True,
                 copy_data: bool = False):

        # input data and cooresponding predictions
        self.input_test_data = input_test_data

        # tree parameters
        self.leafsize = leafsize
        self.balanced_tree = balanced_tree
        self.compact_nodes = compact_nodes
        self.copy_data = copy_data

    def calc_space_partitions(self, ):
        self.kdtree = KDTree(
            self.input_test_data,
            leafsize=self.leafsize,
            compact_nodes=self.compact_nodes,
            copy_data=self.copy_data,
            balanced_tree=self.balanced_tree,
            boxsize=None)

        self.space_partitions = kd_tree_partioning.get_space_partitions(
            self.kdtree,
            mins=np.min(self.input_test_data, axis=0),
            maxes=np.max(self.input_test_data, axis=0))

    def calc_stats_per_region(self,
                              predictions: typing.Union[np.ndarray, distance_measures.Gaussian],
                              output_data: np.ndarray,
                              method: str = 'anees',
                              alpha: float = 0.01,  # significance level; only used for anees
                              m_bins: int = 1,  # number of bins for ece
                              ):

        return distance_calculation(predictions,
                                    output_data,
                                    self.space_partitions,
                                    method=method,
                                    alpha=alpha,
                                    m_bins=m_bins,)


def distance_calculation(prediction: typing.Union[np.ndarray, distance_measures.Gaussian],
                         output_data: np.ndarray,
                         space_partitions: kd_tree_partioning.KDTreePartitions,
                         method: str = 'anees',
                         alpha: float = 0.01,  # significance level; only used for anees
                         m_bins: int = 1,  # number of bins for ece
                         ):

    stat_per_region = np.zeros(len(space_partitions.mins))
    accept_stat_per_region = None

    if method == 'anees':
        accept_stat_per_region = np.zeros(len(space_partitions.mins), dtype=bool)
        nees = distance_measures.squared_mahalanobis_distance(
            prediction, output_data)

        for idx_partition, indices_per_partition in enumerate(space_partitions.indices):
            # Get the data points in the partition
            nees_in_partition = nees[indices_per_partition]
            # mean of the nees in the partition
            stat_per_region[idx_partition] = np.mean(nees_in_partition)
            # get the number of data points in the partition
            n_points = len(nees_in_partition)
            # get critical values for the chi squared distribution
            # with n_points degrees of freedom and alpha confidence
            # (two sided test)
            level = np.array([alpha/2, 1-alpha/2])
            chi2_critical = 1/n_points * scipy.stats.chi2.ppf(level, n_points)
            # check if the mean of the nees in the partition is within the confidence interval
            accept_stat_per_region[idx_partition] =\
                chi2_critical[0] <= stat_per_region[idx_partition] <= chi2_critical[1]

    elif method == 'mse':
        squared_error = distance_measures.squared_error(prediction, output_data).T
        for idx_partition, indices_per_partition in enumerate(space_partitions.indices):
            stat_per_region[idx_partition] = np.mean(squared_error[indices_per_partition])

    elif method == 'ece':
        # binary classification
        # prediction is a numpy array of probabilities, each row sums to 1
        # dimensions are (n_samples, n_classes)
        # we need to convert to (n_samples, 2)
        # where the first column is the probability of the negative class
        # and the second column is the probability of the positive class

        if isinstance(prediction, distance_measures.Gaussian):
            if prediction.mean.ndim == 1:
                pred_mean = copy.deepcopy(prediction.mean.reshape(-1, 1))
            classification_result = np.concatenate((1-pred_mean, pred_mean), axis=1)
        else:
            raise ValueError('Unknown prediction type')

        for idx_partition, indices_per_partition in enumerate(space_partitions.indices):
            # Get the data points in the partition
            stat_per_region[idx_partition] = distance_measures.expected_calibration_error(
                classification_result[indices_per_partition],
                output_data[indices_per_partition], m_bins=m_bins)

    else:
        raise ValueError('Unknown method')

    return stat_per_region, accept_stat_per_region
