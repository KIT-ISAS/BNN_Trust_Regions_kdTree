"""
This module contains functions to calculate the Wasserstein distance between two distributions.
"""
# disable pylint warning Line too long
# pylint: disable=C0301
import typing
from dataclasses import dataclass

import joblib
import numpy as np
import ot
import scipy.stats

from tqdm import tqdm


@dataclass(frozen=False)
class UnivariateGaussian:
    """
    Class representing the result of Gaussian predictions.

    :ivar mean: The means of the Gaussian distribution.
    :vartype mean: numpy.ndarray
    :ivar cov: The covariances of the Gaussian distribution.
    :vartype cov: numpy.ndarray
    """
    mean: np.ndarray = None
    var: np.ndarray = None


class WassersteinDistance:
    """
    # The class WassersteinDistance calculates 
    the Wasserstein distance between two sets of predictions
    # using the specified p-norm and parallel computing options.
    """

    def __init__(self,
                 p_norm: int = 1,
                 parallel_computing: bool = False,
                 verbose: bool = False):
        """
        The function initializes an object with two sets of predictions, a p-norm value, a flag for
        parallel computing, and a flag for verbosity.

        :param predictions_a: The parameter `predictions_a`
        is used to pass the predictions for dataset
        A. It can be either a numpy array or an instance of the `UnivariateGaussian` class
        :type predictions_a: typing.Union[np.ndarray, UnivariateGaussian]
        :param predictions_b: The `predictions_b` parameter
        is used to pass the predictions for the
        second set of data. It can be either a numpy array
        or an instance of the `UnivariateGaussian`
        class
        :type predictions_b: typing.Union[np.ndarray, UnivariateGaussian]
        :param p_norm: The p_norm parameter
        is an integer that specifies the norm to be used for
        calculating the distance between the predictions.
        It is used in the calculation of the distance
        between two predictions, where the distance
        is calculated as the p-norm of the difference
        between the two predictions. The p-norm is a general, defaults to 1
        :type p_norm: int (optional)
        :param parallel_computing: The `parallel_computing` parameter
        is a boolean flag that indicates
        whether or not to use parallel computing for the calculations.
        If set to `True`, the
        calculations will be performed in parallel,
        which can potentially speed up the computation time.
        If set to `False`, the calculations will be performed sequentially, defaults to False
        :type parallel_computing: bool (optional)
        :param verbose: The `verbose` parameter
        is a boolean flag that determines whether or not to
        print additional information during the execution of the code.
        If `verbose` is set to `True`,
        then additional information will be printed.
        If `verbose` is set to `False`, then no additional
        information will be printed, defaults to False
        :type verbose: bool (optional)
        """
        self.p_norm = p_norm
        self.parallel_computing = parallel_computing
        self.verbose = verbose

    def calc_wasserstein_distance(self, predictions_a: typing.Union[np.ndarray, UnivariateGaussian],
                                  predictions_b: typing.Union[np.ndarray, UnivariateGaussian],):
        """
        The function calculates the Wasserstein distance between two sets of predictions.
        :return: the result of the `wasserstein_univariate_wrapper` function, which calculates the
        Wasserstein distance between `self.predictions_a` and `self.predictions_b`.
        """
        return wasserstein_univariate_wrapper(predictions_a, predictions_b,
                                              p_norm=self.p_norm,
                                              parallel_computing=self.parallel_computing,
                                              verbose=self.verbose)


def wasserstein_univariate_wrapper(pred_a: typing.Union[np.ndarray, UnivariateGaussian],
                                   pred_b: typing.Union[np.ndarray, UnivariateGaussian],
                                   p_norm: int = 1,
                                   parallel_computing: bool = False,
                                   verbose: bool = False):
    """
    The function calculates the Wasserstein distance between two univariate distributions.

    """

    # distance between two univariate gaussians
    if isinstance(pred_a, UnivariateGaussian) and isinstance(pred_b, UnivariateGaussian):
        # this is the analytical solution for the 1-Wasserstein distance
        # between two univariate Gaussians
        if p_norm == 1:
            return wasserstein_p1_univariate_gaussian_gaussian(
                pred_a.mean,
                pred_a.var,
                pred_b.mean,
                pred_b.var)

        if p_norm == 2:
            raise NotImplementedError(
                "The 2-Wasserstein distance betweentwo univariate Gaussians is not implemented.")

        raise ValueError(
            "The p-norm must be either 1 or 2 for the Wasserstein distance "+
            "between two univariate Gaussians.")

    # distance between a univariate gaussian and a dirac mixture
    if isinstance(pred_a, UnivariateGaussian) and isinstance(pred_b, np.ndarray):
        return wasserstein_univariate_gaussian_dirac_multiple_distances(
            pred_b, pred_a.mean, pred_a.var,
            p_norm=p_norm, parallel_computing=parallel_computing, verbose=verbose)

    # distance between a dirac mixture and a univariate gaussian
    if isinstance(pred_a, np.ndarray) and isinstance(pred_b, UnivariateGaussian):
        return wasserstein_univariate_gaussian_dirac_multiple_distances(
            pred_a, pred_b.mean, pred_b.var,
            p_norm=p_norm, parallel_computing=parallel_computing, verbose=verbose)

    # distance between two dirac mixtures
    if isinstance(pred_a, np.ndarray) and isinstance(pred_b, np.ndarray):
        return wasserstein_optimal_transport(pred_a, pred_b,
                                             p_norm=p_norm, parallel_computing=parallel_computing, verbose=verbose)

    raise ValueError(
        "The input parameters must be either a numpy array or an instance of the Gaussian class.")


def wasserstein_optimal_transport(samples1: np.ndarray,  # n_samples x n_distributions

                                  samples2: np.ndarray,  # m_samples x n_distributions
                                  weights1: np.ndarray = None,  # weight per sample
                                  weights2: np.ndarray = None,  # weight per sample
                                  p_norm: int = 1,  # p-norm
                                  parallel_computing: bool = False,  # parallel computation
                                  verbose: bool = False):  # verbose
    """
    The function `wasserstein_optimal_transport` computes the Wasserstein Optimal Transport distance
    between two sets of samples.

    :param samples1: An array of shape (n_samples, n_distributions) representing the first set of
    samples. Each row corresponds to a sample, and each column corresponds to a distribution
    :type samples1: np.ndarray
    :param samples2: Array of shape (m_samples, n_distributions) representing the second set of samples
    :type samples2: np.ndarray
    :param weights1: Array of shape (n_samples,) representing the weights for each sample in samples1.
    These weights determine the importance of each sample in the computation of the Wasserstein Optimal
    Transport distance. If not provided, equal weights are assigned to each sample
    :type weights1: np.ndarray
    :param weights2: The `weights2` parameter is an optional array of shape (m_samples,) representing
    the weights for each sample in `samples2`. These weights are used to compute the Wasserstein Optimal
    Transport distance between the two sets of samples. If `weights2` is not provided, it defaults to an
    :type weights2: np.ndarray
    :param p_norm: The parameter `p_norm` is the p-norm to be used in the computation of the Wasserstein
    Optimal Transport distance. It determines the distance metric used to measure the discrepancy
    between the distributions. The default value is 1, which corresponds to the Earth Mover's distance.
    Other common choices, defaults to 1
    :type p_norm: int (optional)
    :param parallel_computing: The `parallel_computing` parameter is a boolean flag indicating whether
    to use parallel computation or not. If set to `True`, the function will use parallel computation to
    speed up the computation of the Wasserstein Optimal Transport distance. If set to `False`, the
    function will use a single thread, defaults to False
    :type parallel_computing: bool (optional)
    :param verbose: The `verbose` parameter is a boolean flag indicating whether to display progress
    information during the computation. If set to `True`, it will display progress information; if set
    to `False`, it will not display any progress information, defaults to False
    :type verbose: bool (optional)
    :return: an array of shape (n_distributions,) containing the Wasserstein Optimal Transport distance
    for each distribution.
    """

    _, num_distributions = samples1.shape
    # parallel execution of statistical test
    if weights1 is None:
        weights1 = np.ones(samples1.shape[0]) / samples1.shape[0]
    if weights2 is None:
        weights2 = np.ones(samples2.shape[0]) / samples2.shape[0]

    if parallel_computing:
        n_jobs = -2
    else:
        n_jobs = 1

    wd = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
        joblib.delayed(earth_mover_distance)(samples1[:, i], samples2[:, i],
                                             weights1, weights2, p_norm) for i in tqdm(range(0, num_distributions), disable=not verbose))

    return np.asarray(wd)


def earth_mover_distance(sample1: np.ndarray, sample2: np.ndarray,
                         weights_sample1: np.ndarray, weights_sample2: np.ndarray,
                         p_norm: int = 1):
    """
    Compute the Earth Mover's Distance (EMD) between two samples.

    :param sample1: The first sample as a numpy array.
    :param sample2: The second sample as a numpy array.
    :param weights_sample1: The weights associated with the elements in sample1 as a numpy array.
    :param weights_sample2: The weights associated with the elements in sample2 as a numpy array.
    :param p_norm: The p-norm to be used in the computation of the EMD. Default is 1.

    :return: The Earth Mover's Distance between the two samples.
    """
    return ot.emd2_1d(sample1, sample2, a=weights_sample1, b=weights_sample2,
                      metric="minkowski", p=p_norm)  # direct computation of OT loss


def wasserstein_univariate_gaussian_dirac_multiple_distances(samples: np.ndarray,
                                                             mean1: np.ndarray, var1: np.ndarray,
                                                             p_norm: int = 1,
                                                             parallel_computing: bool = False,
                                                             verbose: bool = False):
    """
    The function calculates the Wasserstein distance between n=`num_distributions` univariate Gaussian and Dirac mixture
    distributions.

    :param samples: The `samples` parameter is a numpy array containing the samples from a univariate
    distribution. Each row of the array represents a different sample, and each column
    represents a different distribution. The shape of the array should be (n_samples, num_distributions),
    :type samples: np.ndarray
    :param mean1: The parameter `mean1` represents the mean of a univariate Gaussian distribution
    :type mean1: np.ndarray
    :param var1: The parameter `var1` represents the variance of a univariate Gaussian distribution
    :type var1: np.ndarray
    :param verbose: The `verbose` parameter is a boolean flag that determines whether or not to display
    progress information during the computation. If `verbose` is set to `True`, progress information
    will be displayed. If `verbose` is set to `False`, progress information will be hidden, defaults to
    False
    :type verbose: bool (optional)
    :return: an array of Wasserstein distances between the given samples and the ground truth
    distribution.
    """
    _, num_distributions = samples.shape
    wd_1 = np.empty((num_distributions,))
    # for i in tqdm(range(0, num_distributions), desc="W1(Gaussian, Dirac mixture)", disable=not verbose):
    #     wd_1[i] = gaussian_dirac_mixture_wasserstein(
    #         samples=samples[:, i], mean=mean1[i], var=var1[i], p_norm=p_norm)

    if parallel_computing:
        n_jobs = -2
    else:
        n_jobs = 1

    wd_1_batch = wasserstein_univariate_gaussian_dirac_mixture(
        samples, mean1, var1, p_norm=p_norm).reshape(-1, 1)

    # wd_1 = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
    #     joblib.delayed(
    #         wasserstein_univariate_gaussian_dirac_mixture)(
    #             # ignore nan values in samples
    #             # samples[:, i],
    #             samples[~np.isnan(samples[:, i]), i],
    #             mean1[i],
    #             var1[i],
    #             p_norm) for i in tqdm(range(0, num_distributions), disable=not verbose))
    # wd_1 = np.asarray(wd_1)
    # are loop and batch results the same?
    # assert np.allclose(wd_1, wd_1_batch)
    return wd_1_batch


def wasserstein_univariate_gaussian_dirac_mixture(samples: np.ndarray, mean: float, var: float, p_norm: int = 1):
    """
    Calculate the Wasserstein distance between a Gaussian distribution and a Dirac mixture.
    Use only to calculate between one Gaussian and one Dirac mixture.
    To calculate multiple distances, use `wasserstein_p1_univariate_gaussian_dirac_multiple_distances`.

    :param samples: The samples from the Dirac mixture.
    :type samples: numpy.ndarray (n_samples, 1)
    :param mean: The mean of the Gaussian distribution.
    :type mean: float
    :param var: The variance of the Gaussian distribution.
    :type var: float
    :param p_norm: The p-norm used in the Wasserstein distance calculation. Default is 1.
    :type p_norm: int, optional

    :return: The Wasserstein distance between the Gaussian distribution and the Dirac mixture.
    :rtype: float
    """
    # count non nan elements per column
    num_samples = np.sum(~np.isnan(samples), axis=0).reshape(-1,)
    num_distances = len(num_samples)

    # sort samples
    sorted_samples = np.sort(samples, axis=0).reshape(-1, num_distances)

    # copy samples to calculate empirical cdf for multiple distances
    stacked_num_samples = np.stack([np.arange(np.max(num_samples))] * num_distances, axis=1)
    # set to nan values where there are no samples
    stacked_num_samples = np.where(stacked_num_samples < num_samples, stacked_num_samples, np.nan)
    # calculate empirical cdf
    empirical_cdf = (stacked_num_samples + 1) / (num_samples)
    mid_points_ecdf = empirical_cdf - 1/(2 * num_samples)
    gaussian_sample_positions = scipy.stats.norm.ppf(mid_points_ecdf,
                                                     loc=mean.reshape(1, -1), scale=np.sqrt(var.reshape(1, -1)))
    norm = np.power(np.abs(gaussian_sample_positions - sorted_samples), p_norm)
    # ignore nan values for averaging

    # wasserstein_dist = np.mean(norm, axis=0)
    wasserstein_dist = np.nanmean(norm, axis=0)
    return wasserstein_dist


def wasserstein_p1_univariate_gaussian_gaussian(mean1: typing.Union[float, np.ndarray], var1: typing.Union[float, np.ndarray],
                                                mean2: typing.Union[float, np.ndarray], var2: typing.Union[float, np.ndarray],):
    """
    The function calculates the 1-Wasserstein distance between two univariate Gaussian distributions.

    :param mean1: The mean of the first Gaussian distribution
    :type mean1: typing.Union[float, np.ndarray]
    :param var1: The variable `var1` represents the variance of the first Gaussian distribution
    :type var1: typing.Union[float, np.ndarray]
    :param mean2: The parameter `mean2` represents the mean of the second Gaussian distribution
    :type mean2: typing.Union[float, np.ndarray]
    :param var2: The parameter `var2` represents the variance of the second Gaussian distribution
    :type var2: typing.Union[float, np.ndarray]
    :return: the 1-Wasserstein distance between two univariate Gaussian distributions.
    :rtype: np.ndarray
    """
    # calculate 1-wasserstein distance between two gaussians
    # On the 1-Wasserstein Distance between Location-Scale Distributions and the Effect of Differential Privacy
    # Saurab Chhachhi, Fei Teng
    # https://doi.org/10.48550/arXiv.2304.14869 equation (34)

    # flatten arrays
    mean1 = np.ravel(mean1)
    mean2 = np.ravel(mean2)
    var1 = np.ravel(var1)
    var2 = np.ravel(var2)

    mean_y = mean1 - mean2
    cov_y = np.square(np.sqrt(var1) - np.sqrt(var2))
    mean_y_abs = np.abs(mean_y)
    std_y_abs = np.abs(np.sqrt(cov_y))
    factor1 = mean_y_abs

    factor2 = np.empty_like(mean_y_abs)
    factor4 = np.empty_like(mean_y_abs)

    where_std_y_zero = std_y_abs == 0
    non_zero = np.logical_not(where_std_y_zero)

    if np.sum(non_zero) > 0:
        factor2[non_zero] = 1 - 2 * \
            scipy.stats.norm.cdf(-mean_y_abs[non_zero] / std_y_abs[non_zero])
        factor4[non_zero] = np.exp(-np.square(mean_y_abs) / (2 * cov_y[non_zero]))

    if np.sum(where_std_y_zero) > 0:
        # limit of factor1 for cov_y -> 0 is 1; see paper
        factor2[where_std_y_zero] = 1

        # limit of factor4 for cov_y -> 0 is 1; see paper
        factor4[where_std_y_zero] = np.ones_like(factor4[where_std_y_zero])

    factor3 = std_y_abs * np.sqrt(2 / np.pi)

    term1 = factor1 * factor2
    term2 = factor3 * factor4

    return term1 + term2
