"""
This file contains the code for the wrapper class SpacePartitioning
that calculates the space partitions and the statistics per region.
The statistics are calculated using the distance_measures.py file.
The space partitions are calculated using the kd_tree_partioning.py file.
"""

import typing
from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

import distance_measures
import distance_stat_wrapper
import kd_tree_partioning


@dataclass
class SpacePartitioning:
    """ Wrapper class to calculate the space partitions and the statistics per region."""
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
        """ Calculate the space partitions using the KDTree."""
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
                              ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """ Calculate the distance measure and statistical tests per region. 

        :param predictions: Predictions of the model. 
        Given as Gaussian with mean and variance or as samples (dirac mixture).
        :type predictions: typing.Union[np.ndarray, distance_measures.Gaussian]
        :param output_data: output test data
        :type output_data: np.ndarray
        :param method: Method to calculate the distance measure. Options are 'anees', 'mse', 'ece'.
        :type method: str
        :param alpha: Significance level; only used for anees.
        :type alpha: float
        :param m_bins: Number of bins for ece.
        :type m_bins: int
        :return: test statistic and acceptance per region. The acceptance is given as a boolean array.
        :rtype: tuple[np.ndarray[float], np.ndarray[bool]]

        """

        return distance_stat_wrapper.distance_calculation(
            predictions,
            output_data,
            self.space_partitions.indices,
            method=method,
            alpha=alpha,
            m_bins=m_bins,)
