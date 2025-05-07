""" distance functions for probability distributions"""

from dataclasses import dataclass
import typing

import numpy as np
import scipy.stats

from tqdm import tqdm

# import data_processing.data_structures as dstruc


@dataclass(frozen=False)
class Gaussian:
    """
    Class representing the result of Gaussian predictions.

    :ivar mean: The means of the Gaussian distribution.
    :vartype mean: numpy.ndarray
    :ivar cov: The covariances of the Gaussian distribution.
    :vartype cov: numpy.ndarray
    """
    mean: np.ndarray = None
    cov: np.ndarray = None


def squared_error(x_pred: np.ndarray, x_observed: np.ndarray):
    """
    Calculates the squared error between predicted and observed values.

    :param x_pred: A numpy array of predicted values.
    :type x_pred: numpy.ndarray
    :param x_observed: A numpy array of observed values.
    :type x_observed: numpy.ndarray
    :return: The squared error between the predicted and observed values.
    :rtype: float
    """
    if isinstance(x_pred, Gaussian):
        x_pred = x_pred.mean
    return np.square(x_pred - x_observed.T)

def mean_square_error(x_pred, x_observed):
    """
    Calculates the mean squared error between predicted and observed values.

    :param x_pred: A numpy array of predicted values.
    :type x_pred: numpy.ndarray
    :param x_observed: A numpy array of observed values.
    :type x_observed: numpy.ndarray
    :return: The mean squared error between the predicted and observed values.
    :rtype: float
    """
    if isinstance(x_pred, Gaussian):
        x_pred = x_pred.mean
    return np.square(x_pred - x_observed.T).mean(axis=None)


def squared_mahalanobis_distance(prediction: typing.Union[np.ndarray, Gaussian],
                                 output_data: np.ndarray,):
    """
    The function calculates the
    squared Mahalanobis distance between a prediction and output data.

    :param prediction: The `prediction`
    parameter can be either a numpy array or an instance of the
    `Gaussian` class. If it is a numpy array, it represents the predicted values. If it is
    an instance of `Gaussian`, it contains the mean and covariance of
    :type prediction: typing.Union[np.ndarray, Gaussian]
    :param output_data: The `output_data`
    parameter is a numpy array that contains the actual output
    values
    :type output_data: np.ndarray
    :return: the squared Mahalanobis distance
    between the prediction and the output data.
    """
    if isinstance(prediction, Gaussian):
        mean_prediction = prediction.mean
        var_prediction = prediction.cov
    else:
        mean_prediction = np.mean(prediction, axis=0)
        var_prediction = np.var(prediction, axis=0)
    output_data = np.squeeze(output_data)

    inverse_var = 1 / var_prediction
    return np.square(output_data - mean_prediction) * inverse_var



def expected_calibration_error(samples, true_labels, m_bins=1):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, m_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece


# wasserstein distances
def wasserstein_p1_univariate_gaussian_dirac_multiple_distances(samples: np.ndarray,
                                                                mean1: np.ndarray, var1: np.ndarray,
                                                                verbose: bool = False):
    """
    The function calculates the Wasserstein distance 
    between n=`num_distributions` univariate Gaussian and Dirac mixture
    distributions.

    :param samples: The `samples` parameter is a numpy array
    containing the samples from a univariate
    distribution. Each row of the array represents a different sample, and each column
    represents a different distribution.
    The shape of the array should be (n_samples, num_distributions),
    :type samples: np.ndarray
    :param mean1: The parameter `mean1` represents
    the mean of a univariate Gaussian distribution
    :type mean1: np.ndarray
    :param var1: The parameter `var1` represents
    the variance of a univariate Gaussian distribution
    :type var1: np.ndarray
    :param verbose: The `verbose` parameter is a
    boolean flag that determines whether or not to display
    progress information during the computation.
    If `verbose` is set to `True`, progress information
    will be displayed.
    If `verbose` is set to `False`, progress information will be hidden, defaults to
    False
    :type verbose: bool (optional)
    :return: an array of Wasserstein distances between the given samples and the ground truth
    distribution.
    """
    _, num_distributions = samples.shape
    wd_1 = np.empty((num_distributions,))
    for i in tqdm(range(0, num_distributions),
                  desc="W1(Gaussian, Dirac mixture)", disable=not verbose):
        wd_1[i] = gaussian_dirac_mixture_wasserstein(
            samples=samples[:, i], mean=mean1[i], var=var1[i], p_norm=1)
    return wd_1


def gaussian_dirac_mixture_wasserstein(samples: np.ndarray,
                                       mean: float,
                                       var: float,
                                       p_norm: int = 1):
    """
    Calculate the Wasserstein distance between
    a Gaussian distribution and a Dirac mixture.
    Use only to calculate between
    one Gaussian and one Dirac mixture.
    To calculate multiple distances, use
    `wasserstein_p1_univariate_gaussian_dirac_multiple_distances`.

    :param samples: The samples from the Dirac mixture.
    :type samples: numpy.ndarray (n_samples, 1)
    :param mean: The mean of the Gaussian distribution.
    :type mean: float
    :param var: The variance of the Gaussian distribution.
    :type var: float
    :param p_norm: The p-norm used in the
    Wasserstein distance calculation. Default is 1.
    :type p_norm: int, optional

    :return: The Wasserstein distance between
    the Gaussian distribution and the Dirac mixture.
    :rtype: float
    """
    sorted_samples = np.sort(samples)
    num_samples = sorted_samples.shape[0]
    empirical_cdf = (np.arange(num_samples)+1) / (num_samples)
    mid_points_ecdf = empirical_cdf - 1/(2 * num_samples)
    gaussian_sample_positions = scipy.stats.norm.ppf(mid_points_ecdf, loc=mean, scale=np.sqrt(var))
    norm = np.power(np.abs(gaussian_sample_positions - sorted_samples), p_norm)
    wasserstein_dist = np.mean(norm)
    return wasserstein_dist


def wasserstein_p1_univariate_gaussian_gaussian(mean1: typing.Union[float, np.ndarray],
                                                var1: typing.Union[float, np.ndarray],

                                                mean2: typing.Union[float, np.ndarray] = None,
                                                var2: typing.Union[float, np.ndarray] = None,):
    """
    The function calculates the 1-Wasserstein distance
    between two univariate Gaussian distributions.

    :param mean1: The mean of the first Gaussian distribution
    :type mean1: typing.Union[float, np.ndarray]
    :param var1: The variable `var1`
    represents the variance of the first Gaussian distribution
    :type var1: typing.Union[float, np.ndarray]
    :param mean2: The parameter `mean2`
    represents the mean of the second Gaussian distribution
    :type mean2: typing.Union[float, np.ndarray]
    :param var2: The parameter `var2`
    represents the variance of the second Gaussian distribution
    :type var2: typing.Union[float, np.ndarray]
    :return: the 1-Wasserstein distance between two univariate Gaussian distributions.
    """

    # calculate 1-wasserstein distance between two gaussians
    # On the 1-Wasserstein Distance between
    # Location-Scale Distributions and the Effect of Differential Privacy
    # Saurab Chhachhi, Fei Teng
    # https://doi.org/10.48550/arXiv.2304.14869 equation (34)

    if mean2 is None and var2 is None:
        mean_y = mean1
        cov_y = var1
    elif mean2 is not None and var2 is not None:
        mean_y = mean1-mean2
        cov_y = np.square(np.sqrt(var1) - np.sqrt(var2))
    else:
        raise ValueError("mean2 and var2 must be both None or both not None")

    mean_y_abs = np.abs(mean_y)
    std_y_abs = np.abs(np.sqrt(cov_y))
    factor1 = mean_y_abs

    factor2 = np.empty_like(mean_y_abs)
    # factor3 = np.empty_like(mean_y_abs)
    factor4 = np.empty_like(mean_y_abs)

    where_std_y_zero = std_y_abs == 0
    non_zero = np.logical_not(where_std_y_zero)

    if np.sum(non_zero) > 0:
        factor2[non_zero] = 1 - 2*scipy.stats.norm.cdf(- mean_y_abs[non_zero] / std_y_abs[non_zero])
        factor4[non_zero] = np.exp(-np.square(mean_y_abs[non_zero])/(2*cov_y[non_zero]))

    if np.sum(where_std_y_zero) > 0:
        # limt of factor1 for cov_y -> 0 is 1; see paper
        factor2[where_std_y_zero] = 1

        # limt of factor4 for cov_y -> 0 is 1; see paper
        factor4[where_std_y_zero] = np.ones_like(factor4[where_std_y_zero])

    factor3 = std_y_abs * np.sqrt(2/np.pi)

    term1 = factor1 * factor2
    term2 = factor3 * factor4

    # # debug only
    # ref = term1 + term2
    # limit = np.sqrt(2/np.pi) * np.abs(np.sqrt(cov_y))
    # error = (ref - limit) > 0.

    return term1 + term2
