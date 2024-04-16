"""
This script is used to test the kd tree based space partitioning
and the distance calculation between prediction and output data.
"""
# disable pylint warning Constant name doesn't conform to UPPER_CASE naming style
# pylint: disable=C0103
import copy
import os
import pickle
import typing

from matplotlib import axis, pyplot as plt
import matplotlib.colors


import numpy as np
import torch

import distance_measures
import wasserstein_dist

import cmap_norm_plot_settings
import create_plots
import kd_tree_partioning
import kd_tree_testing
import matplotlib_settings

from kd_tree_graph import KDTreeGraph


def load_data(file: str, folder: str):
    """ Loads the data from the given file."""
    if folder is not None:
        file = os.path.join(folder, file)

    with open(file, 'rb') as f:
        prediction_results = pickle.load(f)
    return prediction_results


def convert_to_numpy(data_array: typing.Union[torch.Tensor, np.ndarray]):
    """
    Converts a torch tensor to a numpy array.

    :param data_array: The data array that should be converted.
    :type data_array: typing.Union[torch.Tensor, np.ndarray]
    :return: The converted data array.
    :rtype: np.ndarray
    """
    if isinstance(data_array, np.ndarray):
        return data_array

    if isinstance(data_array, torch.Tensor):
        return data_array.numpy()

    raise ValueError('Unknown data type')


if __name__ == '__main__':

    ########################################################
    # hyper parameter
    ########################################################

    # leafsize parameter for the kd tree
    # if actual node size is greater than the given leafsize,
    # the algorithm will attempt to break the node into smaller nodes
    # a balanced tree has leafnodes of size: leafsize/2 <  size <= leafsize
    LEAFSIZE = 40
    BALENCED_TREE = True

    # plot input data points in the partitioning plot
    plot_input_data_in_part_plot = False
    # plot input data points in the test plot
    plot_input_data_in_test_plot = False

    np.random.seed(61)
    use_subsampling = True  # subsampling of test data (assume that test data is more limited)
    num_test_data = 1000  # 250 is good with LEAFSIZE = 40, random seed 61

    mean_over_runs = False  # if true, the mean and variance over all runs is used

    ########################################################
    # laod data
    ########################################################
    # DATA_ID = 0     # data: 'data_30.01.2024' linear regression
    # DATA_ID = 1     # data: 'data_03.02.2024' linear regression
    # # (used in FUSION paper) # 250 test data
    # DATA_ID = 11    # data: 'data_03.02.2024' linear regression
    # # (used in FUSION paper) # 1000 test data

    # DATA_ID = 2     # same scenario as data_03.02.2024 but with 500 training instances:
    # # Folder: 'data_06.02.2024_500_Tr_instances' linear regression
    # # (used in FUSION paper)

    # DATA_ID = 3     # same scenario as data_03.02.2024 but with 5000 training instances:
    # # Folder: 'data_06.02.2024_5000_Tr_instances' linear regression
    # # (used in FUSION paper)

    # DATA_ID = x   # data: 'data_30.01.2024' logistic regression
    # DATA_ID = 100   # data: 'data_03.02.2024' logistic regression
    # # (used in FUSION paper)

    # DATA_ID = 1000  # data: 'mackey_glass_5000_Tr_instances/mcmc' regression

    DATA_ID = 1000

    is_logistic_regression = False  # if true, the data is a binary classification problem
    if DATA_ID == 0:
        is_logistic_regression = False
        data_file_name = 'data_30.01.2024'

        gamma = np.array([1, 2])  # is this correct?
        delta = 2

    elif DATA_ID == 1:
        is_logistic_regression = False
        data_file_name = 'data_03.02.2024'
        file_names = ['y_pred_custom_lin_reg.pkl',
                      'y_cov_custom_lin_reg.pkl',
                      'X_test_custom_lin_reg.pkl',
                      'y_test_custom_lin_reg.pkl',
                      ]

        num_test_data = 250  # 250 is good with LEAFSIZE = 40, random seed 61
        plot_input_data_in_part_plot = True

        def gt_mean(x: np.ndarray, gamma: np.ndarray, delta: float):
            """Ground truth mean function for the regression scenario."""
            return np.log(1+np.exp(x @ gamma + delta))

        gamma = np.array([1, 1])
        delta = 2
        noise_var = 0.01

        min_data = np.array([-2, -2])
        max_data = np.array([2, 2])

    elif DATA_ID == 11:
        is_logistic_regression = False
        data_file_name = 'data_03.02.2024'
        file_names = ['y_pred_custom_lin_reg.pkl',
                      'y_cov_custom_lin_reg.pkl',
                      'X_test_custom_lin_reg.pkl',
                      'y_test_custom_lin_reg.pkl',
                      ]

        num_test_data = 1000  # 250 is good with LEAFSIZE = 40, random seed 61
        # or 1000

        def gt_mean(x: np.ndarray, gamma: np.ndarray, delta: float):
            """Ground truth mean function for the regression scenario."""
            return np.log(1+np.exp(x @ gamma + delta))

        gamma = np.array([1, 1])
        delta = 2
        noise_var = 0.01

        min_data = np.array([-2, -2])
        max_data = np.array([2, 2])

    elif DATA_ID == 2:
        is_logistic_regression = False
        data_file_name = 'data_06.02.2024_500_Tr_instances'
        file_names = ['y_pred_custom_lin_reg.pkl',
                      'y_cov_custom_lin_reg.pkl',
                      'X_test_custom_lin_reg.pkl',
                      'y_test_custom_lin_reg.pkl',
                      ]

        num_test_data = 500

        def gt_mean(x: np.ndarray, gamma: np.ndarray, delta: float):
            """Ground truth mean function for the regression scenario."""
            return np.log(1+np.exp(x @ gamma + delta))

        gamma = np.array([1, 1])
        delta = 2
        noise_var = 0.01
        min_data = np.array([-2, -2])
        max_data = np.array([2, 2])

    elif DATA_ID == 3:
        is_logistic_regression = False
        data_file_name = 'data_06.02.2024_5000_Tr_instances'
        file_names = ['y_pred_custom_lin_reg.pkl',
                      'y_cov_custom_lin_reg.pkl',
                      'X_test_custom_lin_reg.pkl',
                      'y_test_custom_lin_reg.pkl',
                      ]

        def gt_mean(x: np.ndarray, gamma: np.ndarray, delta: float):
            """Ground truth mean function for the regression scenario."""
            return np.log(1+np.exp(x @ gamma + delta))

        num_test_data = 5000

        gamma = np.array([1, 1])
        delta = 2
        noise_var = 0.01

        min_data = np.array([-2, -2])
        max_data = np.array([2, 2])

    elif DATA_ID == 100:
        data_file_name = 'data_03.02.2024'
        min_data = np.array([-1.5, -1.5])
        max_data = np.array([1.5, 1.5])
        is_logistic_regression = True
        file_names = ['y_pred_custom_log_reg.pkl',
                      'y_cov_custom_log_reg.pkl',
                      'X_test_custom_log_reg.pkl',
                      'y_test_custom_log_reg.pkl',
                      ]

    elif DATA_ID == 1000:
        data_file_name = 'mackey_glass_5000_Tr_instances/'
        # min_data = np.array([0.4, 0.4])
        # max_data = np.array([1.4, 1.4])
        min_data = np.array([-0, -0])
        max_data = np.array([2, 2])
        is_logistic_regression = False
        train_method = 'mcmc'
        LEAFSIZE = 20  # The code is attempting to assign a value to the variable `train_method`, but
        # there is a typo in the value being assigned. The intended value seems to be
        # `'meanfield'`, but it is cut off with an underscore. The code should be
        # corrected to `train_method = 'meanfield'` to remove the typo.

        # train_method = 'meanfield_svi'
        # train_method = 'pbp'
        # train_method = 'ukf'
        # train_method = 'kbnn'
        file_names = [f'{train_method}_test.pkl',
                      f'{train_method}_test.pkl',
                      'nn_test.pkl',
                      'nn_test.pkl',
                      ]

        ### not used for the mackey glass data; THIS IS NOT THE GT FOR THIS SCENARIO ###
        def gt_mean(x: np.ndarray, gamma: np.ndarray, delta: float):
            """Ground truth mean function for the regression scenario."""
            return np.log(1+np.exp(x @ gamma + delta))
        gamma = np.array([1, 1])
        delta = 0.5
        noise_var = 0.01**2

    else:
        raise ValueError('Unknown data id')

    data_folder = 'data/' + data_file_name  # data is stored in the data folder
    data = []
    if DATA_ID == 1000:
        with open(os.path.join(data_folder+f'{train_method}_test.pkl'), 'rb') as f:
            test_predictions = pickle.load(f)
        y_test_pred_mean = np.mean(test_predictions, axis=0)
        y_pred_test_cov = np.var(test_predictions, axis=0)

        with open(data_folder+'nn_test.pkl', 'rb') as f:
            x_test, y_test = pickle.load(f)

        # rotate every data point in x_test by 45 degrees
        rotate_by = np.pi/4
        x_test = x_test @ np.array([[np.cos(rotate_by), -np.sin(rotate_by)],
                                    [np.sin(rotate_by), np.cos(rotate_by)]])

    else:
        for file_name in file_names:
            data.append(convert_to_numpy(load_data(file_name, data_folder)))
        y_test_pred_mean, y_pred_test_cov, x_test, y_test = data

    if x_test.ndim == 3:
        n_runs = x_test.shape[0]

        # use only the first run as the test data
        use_run_idx = 0

        x_test = x_test[use_run_idx, :, :]
        y_test = y_test[use_run_idx, :, :]

        # use mean and variance over all runs (for nicer/smoother plots?)
        if mean_over_runs:
            y_test_pred_mean = np.mean(y_test_pred_mean, axis=0)
            y_pred_test_cov = np.mean(y_pred_test_cov, axis=0)
        else:
            y_test_pred_mean = y_test_pred_mean[use_run_idx, :, :]
            y_pred_test_cov = y_pred_test_cov[use_run_idx, :, :]

    # if DATA_ID == 1:
    #     min_data = np.array([-2, -2])
    #     max_data = np.array([2, 2])

    #     # get indices of data points that are within the data range
    #     # idx_in_range = np.logical_and(min_data <= x_test, x_test <= max_data)
    #     # idx_in_range = np.all(idx_in_range, axis=1)
    #     # x_test = x_test[idx_in_range]
    #     # y_test = y_test[idx_in_range]
    #     # y_test_pred_mean = y_test_pred_mean.squeeze()[idx_in_range]
    #     # y_pred_test_cov = y_pred_test_cov.squeeze()[idx_in_range]
    # elif DATA_ID == 100:
    #     min_data = np.array([-1.5, -1.5])
    #     max_data = np.array([1.5, 1.5])

    # for comparison with ground truth
    x_gt = copy.deepcopy(x_test)
    predictions_comp_gt = distance_measures.Gaussian(copy.deepcopy(y_test_pred_mean.squeeze()),
                                                     copy.deepcopy(y_pred_test_cov.squeeze()))

    # subsampling of test data (assume that test data is more limited)
    if use_subsampling:

        idx = np.random.choice(len(x_test), size=num_test_data, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]
        y_test_pred_mean = y_test_pred_mean[idx]
        y_pred_test_cov = y_pred_test_cov[idx]

    # wrap the predictions in a Gaussian object
    predictions = distance_measures.Gaussian(y_test_pred_mean.squeeze(), y_pred_test_cov.squeeze())

    if is_logistic_regression:
        used_dist = 'ece'
    else:
        # used_dist = 'mse'
        used_dist = 'anees'

        y_gt = wasserstein_dist.UnivariateGaussian(mean=gt_mean(x_gt, gamma, delta),
                                                   var=noise_var * np.ones(shape=(len(x_gt), )))
        # TODO. ANEES plot settings

    ########################################################
    # deviding the input space and calculating the statistics
    ########################################################

    space_partitioning = kd_tree_testing.SpacePartitioning(input_test_data=x_test,
                                                           leafsize=LEAFSIZE,
                                                           balanced_tree=BALENCED_TREE,)
    space_partitioning.calc_space_partitions()
    space_partitions = space_partitioning.space_partitions
    stat_per_region, accept_stat_per_region = space_partitioning.calc_stats_per_region(
        predictions=predictions,
        output_data=y_test,
        method=used_dist,
        alpha=0.01,  # significance level; only used for anees
        m_bins=5,  # number of bins for ece
    )

    # stat_per_region = distance_calculation(test_predictions, output_test_data,
    #                                        space_partitions, method='anees')
    kd_tree_partioning.print_distances_per_partition(space_partitions, stat_per_region, accept_stat_per_region)

    ########################################################
    # distance between prediction and ground truth
    ########################################################
    if not is_logistic_regression:
        w_dist_settings = wasserstein_dist.WassersteinDistance(p_norm=1, )
        w_dist = w_dist_settings.calc_wasserstein_distance(
            predictions_a=wasserstein_dist.UnivariateGaussian(predictions_comp_gt.mean, predictions_comp_gt.cov),
            predictions_b=y_gt,)

    ########################################################
    # Plotting
    ########################################################
    plot_folder = 'plots/' + data_file_name  # plots are stored in the plot folder

    matplotlib_settings.init_settings(use_tex=True,
                                      scaling_factor=6,
                                      unscaled_fontsize=10,
                                      fig_width=3.5/2,
                                      height_to_width_ratio=1)

    # set color map for ANEES

    if not is_logistic_regression:
        # add string to the plot folder
        plot_folder = os.path.join(plot_folder, 'non_lin_reg')
        os.makedirs(plot_folder, exist_ok=True)
        # plot the distance between prediction and ground truth
        create_plots.wasserstein_plot_2d_input(x_gt, w_dist, plot_folder=plot_folder,
                                               min_data=min_data, max_data=max_data,)

        contour_lvls_mean = np.arange(start=0., stop=6.5, step=0.1)
        contour_lvls_var = np.arange(start=0.006, stop=0.021, step=0.001)
        # show the mean and variance of the ground truth
        create_plots.regression_plot_gaussian(x_gt, (y_gt.mean, y_gt.var),
                                              plot_folder=plot_folder,
                                              min_data=min_data, max_data=max_data,
                                              contour_lvls_mean=contour_lvls_mean,
                                              contour_lvls_var=contour_lvls_var,
                                              file_prefix='gt',
                                              cmap_name='plasma',
                                              plot_every_nth_contline=6,)

        # show the mean and variance of the predictions
        mask_almost_zero = np.where(np.isclose(predictions_comp_gt.mean, 0.))
        predictions_comp_gt.mean[mask_almost_zero] = 0.
        create_plots.regression_plot_gaussian(x_gt, (predictions_comp_gt.mean, predictions_comp_gt.cov),
                                              plot_folder=plot_folder,
                                              contour_lvls_mean=contour_lvls_mean,
                                              contour_lvls_var=contour_lvls_var,
                                              min_data=min_data, max_data=max_data,
                                              file_prefix='pred',
                                              cmap_name='plasma',
                                              plot_every_nth_contline=6,)
    else:
        plot_folder = os.path.join(plot_folder, 'log_reg')
        os.makedirs(plot_folder, exist_ok=True)
        # create_plots.varianz_plot_2d_input(x_gt, predictions_comp_gt.cov,
        #                                    plot_folder=plot_folder,
        #                                    min_data=min_data, max_data=max_data,
        #                                    y_label=r'$\sigma^2$', )

        # Plot true decision boundary using polynomial coefficients
        polynomial_coeffs = [0., 0.5, -1, -.5]
        true_boundary_x1 = np.linspace(min_data[0], max_data[0], 100)
        true_boundary_x2 = np.polyval(polynomial_coeffs, true_boundary_x1)
        true_boundary = np.stack((true_boundary_x1, true_boundary_x2), axis=1)

        create_plots.logistic_regression_plot_2d_input(x_gt, (predictions_comp_gt.mean, predictions_comp_gt.cov),
                                                       plot_folder=plot_folder,
                                                       min_data=min_data, max_data=max_data,
                                                       decision_boundary=true_boundary,
                                                       cmap_name='plasma',
                                                       file_prefix='pred_with_decision_bound',
                                                       plot_every_nth_contline=20,)

        create_plots.regression_plot_gaussian(x_gt, (predictions_comp_gt.mean, predictions_comp_gt.cov),
                                              plot_folder=plot_folder,
                                              min_data=min_data, max_data=max_data,
                                              cmap_name='plasma',
                                              file_prefix='pred',
                                              plot_every_nth_contline=20,)

    if used_dist == 'anees':
        vmin = 0.  # minimum value for the color map
        vmax = 2.  # maximum value for the color map
        cmap, norm = cmap_norm_plot_settings.get_anees_plot_settings(vmin=0.,
                                                                     vmax=2.,
                                                                     set_under_color=None,
                                                                     set_over_color=None,)

        if vmin > 0.:
            extend = 'both'
        else:
            extend = 'max'
        hatch_type = '/'
    elif used_dist == 'ece':
        # vmax = stat_per_region.max()
        vmax = 0.6
        cmap, norm = create_plots.get_wasserstein_plot_settings(vmin=0.,
                                                                vmax=vmax,
                                                                cmap_name='viridis')
        extend = 'max'
        hatch_type = None

    else:
        cmap = 'viridis'
        norm = None
        extend = 'neither'
        hatch_type = None

    ########################################################
    # plot the graph of the kd tree
    ########################################################
    kd_graph = KDTreeGraph(space_partitioning.kdtree,
                           use_tex=True,

                           graph_attr={'ranksep': '0.2',  # default 0.5 # min = 0.02
                                       'nodesep': '0.1',  # default 0.25 min = 0.02
                                       'bgcolor': 'transparent',  # default white
                                       #    'fillcolor': 'purple',
                                       #    'style': 'filled',
                                       })
    kd_graph.kdtree_to_graphviz()
    kd_graph.save_graph(f'kd_tree_graph', directory=plot_folder, format='svg')
    kd_graph.save_graph_as_tex(directory=plot_folder, )
    region_labels = kd_graph.get_leaf_labels()

    split_planes = kd_graph.get_split_plains()
    split_points, split_labels = split_planes.get_split_points_labels()

    ########################################################
    # create gif of the kd tree graph
    ########################################################
    gif_folder = os.path.join(plot_folder, 'kd_tree_graph_gif')
    # create folder if it does not exist
    os.makedirs(gif_folder, exist_ok=True)

    if num_test_data < 500:
        split_planes.create_gif_of_2d_input_graph_creation(
            mins=min_data,
            maxes=max_data,
            duration=1000,
            input_data=x_test,
            plot_dir=gif_folder,
            image_format='png',
            dpi=300,
            axis_labels=[r'$x_1$', r'$x_2$'],
        )
    else:
        warn_txt = 'Skipping gif creation due to large number of test data points'
        # print(warn_txt)

        print('\x1b[33;37;41m' + warn_txt + '\x1b[0m')

    # space partitioning plot
    fig, axes = plt.subplots(constrained_layout=True)
    # equal axis scaling
    axes.set_aspect('equal', 'box')
    if plot_input_data_in_part_plot:
        axes.plot(x_test[:, 0], x_test[:, 1], 'o', label='Data Points', alpha=0.25, )

    # space_partitions.mins[np.argmin(space_partitions.mins, axis=0)] = min_data
    # space_partitions.maxes[np.argmax(space_partitions.maxes, axis=0)] = max_data

    kd_tree_partioning.plot_partitions(axes, space_partitions, facecolor='white', region_labels=region_labels)

    if split_points is not None:
        # split_points has dimension (n_splits, 3*dim_data)
        # see kd_tree_graph.py for the structure of split_points
        axes.plot(split_points[:, 0], split_points[:, 1], 'o', label='Split Points',)

        for i, txt in enumerate(split_labels):
            axes.annotate(txt, (split_points[i, 0], split_points[i, 1]),)

    # set axis limits according to (artificial) data mins and maxes
    axes.set_xlim(min_data[0], max_data[0])
    axes.set_ylim(min_data[1], max_data[1])

    axes.set_xlabel(r'$x_1$')
    axes.set_ylabel(r'$x_2$')
    create_plots.set_major_ticks(axes)

    fig.savefig(plot_folder + '/kd_regions_raw.svg', transparent=True)
    plt.close()

    # space partitioning plot with distance
    fig, axes = plt.subplots()

    # equal axis scaling
    axes.set_aspect('equal', 'box')
    kd_tree_partioning.plot_kd_space(
        space_partitions, stat_per_region, axes=axes,
        accept_stat_per_region=accept_stat_per_region,
        distance_label=used_dist.upper(),
        cmap=cmap, norm=norm, extend=extend,
        hatch_type=hatch_type)

    if plot_input_data_in_test_plot:
        axes.plot(x_test[:, 0], x_test[:, 1], 'o', label='Data Points',
                  markerfacecolor='None', markeredgecolor='white', alpha=0.25)

    # set axis limits according to (artificial) data mins and maxes
    axes.set_xlim(min_data[0], max_data[0])
    axes.set_ylim(min_data[1], max_data[1])

    create_plots.set_major_ticks(axes)

    # set axis labels
    axes.set_xlabel(r'$x_1$')
    axes.set_ylabel(r'$x_2$')
    # fig.savefig(plot_folder + '/kd_regions_dist.svg')

    create_plots.custom_safefig(fig, plot_folder + '/kd_regions_dist.svg')
