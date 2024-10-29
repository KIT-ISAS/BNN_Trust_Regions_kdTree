""" Wrapper for distance calculation and statistical tests per region. """
import copy
import typing

import numpy as np
from netcal.metrics import UCE
import scipy.stats

import distance_measures


def distance_calculation(prediction: typing.Union[np.ndarray, distance_measures.Gaussian],
                         output_data: np.ndarray,
                         indices: typing.List[int],
                         method: str = 'anees',
                         alpha: float = 0.01,  # significance level; only used for anees
                         m_bins: int = 1,  # number of bins for ece
                         ):
    """ Calculate the distance measure and statistical tests per region.

    :param prediction: Predictions of the model. Given as Gaussian with mean and variance or as samples (dirac mixture).
    :type prediction: typing.Union[np.ndarray, distance_measures.Gaussian]
    :param output_data: output test data
    :type output_data: np.ndarray
    :param space_partitions: Space partitions of the input data.
    Indices are given for each region as a list of integers.
    :type space_partitions: typing.List[int]
    :param method: Method to calculate the distance measure. Options are 'anees', 'mse', 'ece'.
    :type method: str
    :param alpha: Significance level; only used for anees.
    :type alpha: float
    :param m_bins: Number of bins for ece.
    :type m_bins: int
    :return: Distance measure and statistical tests per region.
    :rtype: tuple[np.ndarray, np.ndarray]

    """

    stat_per_region = np.zeros(len(indices))
    accept_stat_per_region = None

    if method == 'anees':
        accept_stat_per_region = np.zeros(len(indices), dtype=bool)
        nees = distance_measures.squared_mahalanobis_distance(
            prediction, output_data)

        for idx_partition, indices_per_partition in enumerate(indices):
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
        for idx_partition, indices_per_partition in enumerate(indices):
            stat_per_region[idx_partition] = np.mean(squared_error[indices_per_partition])

    elif method == 'uce':
        # calibration metric for regression
        # see Laves, M.H., Ihler,
        # S., Fast, J.F., Kahrs, L.A., Ortmaier, T.:
        # Well-calibrated regression uncertainty in medical imaging with deep learning.
        # In: Medical Imaging with Deep Learning. pp. 393–412. PMLR (2020)

        if isinstance(prediction, distance_measures.Gaussian):
            if prediction.mean.ndim == 1:
                pred_mean = copy.deepcopy(prediction.mean.reshape(-1, 1))
                pred_cov = copy.deepcopy(prediction.cov.reshape(-1, 1))

        uce = UCE(bins=m_bins)

        for idx_partition, indices_per_partition in enumerate(indices):
            # Get the data points in the partition
            if isinstance(prediction, distance_measures.Gaussian):
                stat_per_region[idx_partition] = uce.measure(
                    (pred_mean[indices_per_partition], pred_cov[indices_per_partition]),
                    output_data[indices_per_partition], kind='meanstd')
            else:
                stat_per_region[idx_partition] = uce.measure(
                    prediction,
                    output_data[indices_per_partition], kind='meanstd')

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

        for idx_partition, indices_per_partition in enumerate(indices):
            # Get the data points in the partition
            stat_per_region[idx_partition] = distance_measures.expected_calibration_error(
                classification_result[indices_per_partition],
                output_data[indices_per_partition], m_bins=m_bins)

    else:
        raise ValueError('Unknown method')

    return stat_per_region, accept_stat_per_region
