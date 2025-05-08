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

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import balltree
import cmap_norm_plot_settings
import create_plots
import distance_measures
import distance_stat_wrapper
import kd_tree_partioning
import kd_tree_testing
import matplotlib_settings
import train_moon
import wasserstein_dist
from kd_tree_graph import KDTreeGraph
from matplotlib_settings import mark_inset


def load_data(file: str, folder: str):
    """Loads the data from the given file."""
    if folder is not None:
        file = os.path.join(folder, file)

    with open(file, "rb") as f:
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

    raise ValueError("Unknown data type")


def main():
    """load predictions and data. partition the input space by a kd tree and calculate the statistic per region."""

    ########################################################
    # hyper parameter
    ########################################################

    gif_limit = 5000  # limit for the number of test data points for the gif creation

    # leafsize parameter for the kd tree
    # if actual node size is greater than the given leafsize,
    # the algorithm will attempt to break the node into smaller nodes
    # a balanced tree has leafnodes of size: leafsize/2 <  size <= leafsize
    LEAFSIZE_KDTREE = 40
    # scikit learn ball tree definition is different from scipy (scikitlearn has leafsize = 2*leafsize_scipy)
    LEAFSIZE_BALLTREE = (
        20  # 40  # 800 # 800 is used to simplify tree depth for schematic plots
    )
    SIGNIFICANCE_LEVEL = 0.01  # significance level for the ANEES test
    BALENCED_KDTREE = True

    show_split_points = True  # show the split points in the partitioning plot

    # plot input data points in the partitioning plot
    plot_input_data_in_part_plot = False
    # plot input data points in the test plot
    plot_input_data_in_test_plot = False

    np.random.seed(61)
    use_subsampling = (
        False  # subsampling of test data (assume that test data is more limited)
    )
    num_test_data = 0  # 250 is good with LEAFSIZE = 40, random seed 61

    mean_over_runs = False  # if true, the mean and variance over all runs is used
    moon_data = False
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
    # 5000 training instances
    # DATA_ID = 1001  # data: y=sin(x^T x)+epsilon regression (d2sin_squared), epsilon ~ N(0, 0.1^2)
    # ca. 2250 training instances

    # DATA_ID = 1010  # moon data set
    # 2250 training instances. unbalanced data set

    # DATA_ID = 2000  # data: 'd2sin_squared_more_points/prior_var_1000' regression
    # FUSION 2025 paper

    DATA_ID = 2000

    x_viz, y_viz_pred_mean, y_viz_pred_cov = None, None, None
    is_logistic_regression = (
        False  # if true, the data is a binary classification problem
    )
    if DATA_ID == 0:
        is_logistic_regression = False
        data_file_name = "data_30.01.2024"

        gamma = np.array([1, 2])  # is this correct?
        delta = 2

    elif DATA_ID == 1:
        is_logistic_regression = False
        data_file_name = "data_03.02.2024"
        file_names = [
            "y_pred_custom_lin_reg.pkl",
            "y_cov_custom_lin_reg.pkl",
            "X_test_custom_lin_reg.pkl",
            "y_test_custom_lin_reg.pkl",
        ]

        use_subsampling = True
        num_test_data = 250  # 250 is good with LEAFSIZE = 40, random seed 61
        plot_input_data_in_part_plot = True

        def gt_mean(x: np.ndarray, gamma: np.ndarray, delta: float):
            """Ground truth mean function for the regression scenario."""
            return np.log(1 + np.exp(x @ gamma + delta))

        gamma = np.array([1, 1])
        delta = 2
        noise_var = 0.01

        min_data = np.array([-2, -2])
        max_data = np.array([2, 2])

    elif DATA_ID == 11:
        is_logistic_regression = False
        data_file_name = "data_03.02.2024"
        file_names = [
            "y_pred_custom_lin_reg.pkl",
            "y_cov_custom_lin_reg.pkl",
            "X_test_custom_lin_reg.pkl",
            "y_test_custom_lin_reg.pkl",
        ]
        use_subsampling = True
        num_test_data = 1000  # 250 is good with LEAFSIZE = 40, random seed 61
        # or 1000

        def gt_mean(x: np.ndarray, gamma: np.ndarray, delta: float):
            """Ground truth mean function for the regression scenario."""
            return np.log(1 + np.exp(x @ gamma + delta))

        gamma = np.array([1, 1])
        delta = 2
        noise_var = 0.01

        min_data = np.array([-2, -2])
        max_data = np.array([2, 2])

    elif DATA_ID == 2:
        is_logistic_regression = False
        data_file_name = "data_06.02.2024_500_Tr_instances"
        file_names = [
            "y_pred_custom_lin_reg.pkl",
            "y_cov_custom_lin_reg.pkl",
            "X_test_custom_lin_reg.pkl",
            "y_test_custom_lin_reg.pkl",
        ]
        use_subsampling = True
        num_test_data = 500

        def gt_mean(x: np.ndarray, gamma: np.ndarray, delta: float):
            """Ground truth mean function for the regression scenario."""
            return np.log(1 + np.exp(x @ gamma + delta))

        gamma = np.array([1, 1])
        delta = 2
        noise_var = 0.01
        min_data = np.array([-2, -2])
        max_data = np.array([2, 2])

    elif DATA_ID == 3:
        is_logistic_regression = False
        data_file_name = "data_06.02.2024_5000_Tr_instances"
        file_names = [
            "y_pred_custom_lin_reg.pkl",
            "y_cov_custom_lin_reg.pkl",
            "X_test_custom_lin_reg.pkl",
            "y_test_custom_lin_reg.pkl",
        ]

        def gt_mean(x: np.ndarray, gamma: np.ndarray, delta: float):
            """Ground truth mean function for the regression scenario."""
            return np.log(1 + np.exp(x @ gamma + delta))

        use_subsampling = True
        num_test_data = 5000

        gamma = np.array([1, 1])
        delta = 2
        noise_var = 0.01

        min_data = np.array([-2, -2])
        max_data = np.array([2, 2])

    elif DATA_ID == 100:
        data_file_name = "data_03.02.2024"
        min_data = np.array([-1.5, -1.5])
        max_data = np.array([1.5, 1.5])
        is_logistic_regression = True
        file_names = [
            "y_pred_custom_log_reg.pkl",
            "y_cov_custom_log_reg.pkl",
            "X_test_custom_log_reg.pkl",
            "y_test_custom_log_reg.pkl",
        ]

    elif DATA_ID == 1000:
        data_file_name = "mackey_glass_5000_Tr_instances/"
        min_data = np.array([0.4, 0.4])
        max_data = np.array([1.4, 1.4])
        # min_data = np.array([-0, -0])
        # max_data = np.array([2, 2])
        show_split_points = False
        is_logistic_regression = False

        LEAFSIZE_BALLTREE = 800
        LEAFSIZE_KDTREE = 1600

        # LEAFSIZE_KDTREE = 40  # The code is attempting to assign a value to the variable `train_method`, but
        # there is a typo in the value being assigned. The intended value seems to be
        # `'meanfield'`, but it is cut off with an underscore. The code should be
        # corrected to `train_method = 'meanfield'` to remove the typo.

        # train_method = 'mcmc'
        # train_method = 'meanfield_svi'
        train_method = "pbp"
        # train_method = 'ukf'
        # train_method = 'kbnn'
        file_names = [
            f"{train_method}_test.pkl",
            f"{train_method}_test.pkl",
            "nn_test.pkl",
            "nn_test.pkl",
        ]

        ### not used for the mackey glass data; THIS IS NOT THE GT FOR THIS SCENARIO ###
        def gt_mean(x: np.ndarray, gamma: np.ndarray, delta: float):
            """Ground truth mean function for the regression scenario."""
            return np.log(1 + np.exp(x @ gamma + delta))

        gamma = np.array([1, 1])
        delta = 0.5
        noise_var = 0.01**2

    elif DATA_ID == 1001:
        data_file_name = "d2sin_squared/"
        min_data = np.array([-2.5, -2.5])
        max_data = np.array([2.5, 2.5])

        show_split_points = False
        is_logistic_regression = False

        # LEAFSIZE_KDTREE = 40  # The code is attempting to assign a value to the variable `train_method`, but
        # there is a typo in the value being assigned. The intended value seems to be
        # `'meanfield'`, but it is cut off with an underscore. The code should be
        # corrected to `train_method = 'meanfield'` to remove the typo.

        train_method = "mcmc"
        # train_method = 'meanfield_svi'
        # train_method = 'pbp'
        # train_method = 'ukf'
        # train_method = 'kbnn'
        file_names = [
            f"{train_method}_test.pkl",
            f"{train_method}_test.pkl",
            "nn_test.pkl",
            "nn_train.pkl",
        ]

        ### not used for the mackey glass data; THIS IS NOT THE GT FOR THIS SCENARIO ###
        def gt_mean(x: np.ndarray, gamma: np.ndarray, delta: float):
            """Ground truth mean function for the regression scenario."""
            return np.sin(np.square(x[:, 0]) + np.square(x[:, 1]))

        gamma = np.array([1, 1])
        delta = 0.5
        noise_var = 0.1**2

    elif DATA_ID == 1010:
        gif_limit = 500
        # moon data set
        data_file_name = "moon"
        moon_data = True
        min_data = np.array([-1.6, -1.1])
        max_data = np.array([2.5, 2.0])
        is_logistic_regression = True
        file_names = [
            "y_test_pred_moon.pkl",
            "y_test_cov_moon.pkl",
            "X_test_data_moon.pkl",
            "y_test_data_moon.pkl",
        ]

        # background visualiziation data and predictions
        viz_files = [
            "y_visualize_pred_moon.pkl",
            "y_visualize_cov_moon.pkl",
            "x_visualize_data_moon.pkl",
        ]

        gamma = None
        delta = None
        noise_var = None

    elif DATA_ID == 2000:
        # data: same as in SDF24 Voronoi paper: used for comparison in FUSION25 paper
        gif_limit = 500
        data_file_name = "d2sin_squared_more_points/prior_var_1000/"

        # train_method = "mcmc"
        train_method = "meanfield_svi"
        min_data = np.array([-2.2, -2.2])  # same limits as in FUSION25 paper
        max_data = np.array([2.2, 2.2])
        # train_method = 'pbp'
        # train_method = 'ukf'
        # train_method = 'kbnn'
        file_names = [
            f"{train_method}_test.pkl",
            f"{train_method}_test.pkl",
            "nn_test.pkl",
            "nn_train.pkl",
        ]

        ### not used for the mackey glass data; THIS IS NOT THE GT FOR THIS SCENARIO ###
        def gt_mean(x: np.ndarray, gamma: np.ndarray, delta: float):
            """Ground truth mean function for the regression scenario."""
            return np.sin(np.square(x[:, 0]) + np.square(x[:, 1]))

        gamma = None
        delta = None
        noise_var = 0.1**2

    else:
        raise ValueError("Unknown data id")

    # data_folder = 'data/' + data_file_name  # data is stored in the data folder
    data_folder = os.path.join("data", data_file_name)
    data = []
    if DATA_ID == 1000 or DATA_ID == 1001 or DATA_ID == 2000:
        with open(os.path.join(data_folder + f"{train_method}_test.pkl"), "rb") as f:
            test_predictions = pickle.load(f)
        y_test_pred_mean = np.mean(test_predictions, axis=0)
        y_pred_test_cov = np.var(test_predictions, axis=0)

        with open(data_folder + "nn_test.pkl", "rb") as f:
            x_test, y_test = pickle.load(f)

        with open(data_folder + "nn_train.pkl", "rb") as f:
            x_train, y_train = pickle.load(f)

    else:
        for file_name in file_names:
            data.append(convert_to_numpy(load_data(file_name, data_folder)))

        assert len(data) == 4, "Data has to contain 4 elements"
        y_test_pred_mean, y_pred_test_cov, x_test, y_test = data  # pylint: disable=W0632:unbalanced-tuple-unpacking

    if DATA_ID == 1010:  # moon data set
        # load visualization data
        data_viz = []
        for file_name in viz_files:
            data_viz.append(convert_to_numpy(load_data(file_name, data_folder)))

        assert len(data_viz) == 3, "Data has to contain 3 elements"
        (
            y_viz_pred_mean,
            y_viz_pred_cov,
            x_viz,
        ) = data_viz  # pylint: disable=W0632:unbalanced-tuple-unpacking

    if x_test.ndim == 3:
        n_runs = x_test.shape[0]

        _ = n_runs

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

    # for comparison with ground truth
    x_gt = copy.deepcopy(x_test)
    predictions_comp_gt = distance_measures.Gaussian(
        copy.deepcopy(y_test_pred_mean.squeeze()),
        copy.deepcopy(y_pred_test_cov.squeeze()),
    )

    # subsampling of test data (assume that test data is more limited)
    if use_subsampling:
        idx = np.random.choice(len(x_test), size=num_test_data, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]
        y_test_pred_mean = y_test_pred_mean[idx]
        y_pred_test_cov = y_pred_test_cov[idx]
    else:
        num_test_data = len(x_test)

    # wrap the predictions in a Gaussian object
    predictions = distance_measures.Gaussian(
        y_test_pred_mean.squeeze(), y_pred_test_cov.squeeze()
    )

    if is_logistic_regression:
        used_dist = "ece"
    else:
        # used_dist = 'mse'
        # used_dist = 'anees'
        used_dist = "uce"

        y_gt = wasserstein_dist.UnivariateGaussian(
            mean=gt_mean(x_gt, gamma, delta),
            var=noise_var * np.ones(shape=(len(x_gt),)),
        )
        # TODO. ANEES plot settings

    ########################################################
    # deviding the input space and calculating the statistics
    ########################################################

    space_partitioning = kd_tree_testing.SpacePartitioning(
        input_test_data=x_test,
        leafsize=LEAFSIZE_KDTREE,
        balanced_tree=BALENCED_KDTREE,
    )
    space_partitioning.calc_space_partitions()
    space_partitions = space_partitioning.space_partitions
    stat_per_region, accept_stat_per_region = space_partitioning.calc_stats_per_region(
        predictions=predictions,
        output_data=y_test,
        method=used_dist,
        alpha=SIGNIFICANCE_LEVEL,  # significance level; only used for anees
        m_bins=1,  # number of bins for ece
    )

    # stat_per_region = distance_calculation(test_predictions, output_test_data,
    #                                        space_partitions, method='anees')
    kd_tree_partioning.print_distances_per_partition(
        space_partitions, stat_per_region, accept_stat_per_region
    )

    ########################################################
    # distance between prediction and ground truth
    ########################################################
    if not is_logistic_regression:
        w_dist_settings = wasserstein_dist.WassersteinDistance(
            p_norm=1,
        )
        w_dist = w_dist_settings.calc_wasserstein_distance(
            predictions_a=wasserstein_dist.UnivariateGaussian(
                predictions_comp_gt.mean, predictions_comp_gt.cov
            ),
            predictions_b=y_gt,
        )

    ########################################################
    # Plotting
    ########################################################
    plot_folder = "plots/" + data_file_name  # plots are stored in the plot folder

    matplotlib_settings.init_settings(
        use_tex=True,
        scaling_factor=6,
        unscaled_fontsize=10,
        fig_width=3.5 / 2,
        height_to_width_ratio=1,
    )

    # plot train data
    if DATA_ID == 1000 or DATA_ID == 1001:
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.scatter(
            x_train[:, 0],
            x_train[:, 1],
            label="training data",
            alpha=1,
            s=40,
            zorder=2.0,
        )
        ax.scatter(
            x_test[:, 0], x_test[:, 1], label="test data", alpha=0.5, s=40, zorder=1.9
        )
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        # set min and max of ax
        ax.set_xlim(min_data[0], max_data[0])
        ax.set_ylim(min_data[1], max_data[1])
        create_plots.set_major_ticks(ax)
        # show legend
        ax.legend(loc="upper right", markerscale=5, prop={"size": 35})
        fig.savefig(os.path.join(plot_folder, "train_data.svg"), bbox_inches="tight")

    # set color map for ANEES

    if DATA_ID < 1000:
        vmax = 0.6
    else:
        vmax = None

    cmap, norm, extend, hatch_type = get_color_style(
        used_dist, vmax=vmax, stat_per_region=stat_per_region
    )

    if not is_logistic_regression:
        # add string to the plot folder
        plot_folder = os.path.join(plot_folder, "non_lin_reg")
        os.makedirs(plot_folder, exist_ok=True)
        # plot the distance between prediction and ground truth
        create_plots.wasserstein_plot_2d_input(
            x_gt,
            w_dist,
            plot_folder=plot_folder,
            min_data=min_data,
            max_data=max_data,
        )

        if DATA_ID < 1000:
            # plots for bayesian perceptron
            contour_lvls_mean = np.arange(start=0.0, stop=6.5, step=0.1)
            contour_lvls_var = np.arange(start=0.006, stop=0.021, step=0.001)
            plot_every_nth_contline = 6
        else:
            contour_lvls_mean = np.arange(start=-1.8, stop=1.8, step=0.1)
            contour_lvls_var = np.arange(start=0.0, stop=0.5, step=0.01)
            plot_every_nth_contline = -1
        # show the mean and variance of the ground truth
        create_plots.regression_plot_gaussian(
            x_gt,
            (y_gt.mean, y_gt.var),
            plot_folder=plot_folder,
            min_data=min_data,
            max_data=max_data,
            contour_lvls_mean=contour_lvls_mean,
            contour_lvls_var=contour_lvls_var,
            file_prefix="gt",
            cmap_name="plasma",
            plot_every_nth_contline=plot_every_nth_contline,
        )

        # show the mean and variance of the predictions
        mask_almost_zero = np.where(np.isclose(predictions_comp_gt.mean, 0.0))
        predictions_comp_gt.mean[mask_almost_zero] = 0.0
        create_plots.regression_plot_gaussian(
            x_gt,
            (predictions_comp_gt.mean, predictions_comp_gt.cov),
            plot_folder=plot_folder,
            contour_lvls_mean=contour_lvls_mean,
            contour_lvls_var=contour_lvls_var,
            min_data=min_data,
            max_data=max_data,
            file_prefix="pred",
            cmap_name="plasma",
            plot_every_nth_contline=plot_every_nth_contline,
        )
    else:
        plot_folder = os.path.join(plot_folder, "log_reg")
        os.makedirs(plot_folder, exist_ok=True)

        if moon_data:
            zoom_range_x = (-0.15, 0.5)
            zoom_range_y = (0.3, 0.75)
            zoom_range = (zoom_range_x, zoom_range_y)

            height_to_width_ratio_zoom = (zoom_range_y[1] - zoom_range_y[0]) / (
                zoom_range_x[1] - zoom_range_x[0]
            )

            relative_width = 0.95  # relative to parent axes

            # relative height of the zoomed area relative to the parent axes
            # we want to keep the aspect ratio of the zoomed area
            relative_height = height_to_width_ratio_zoom * relative_width

            zoom_settings = {
                "zoom_range": zoom_range,
                "zoom_factor": 2.5,
                "width": f"{int(relative_width * 100)}%",  # width of the zoomed area relative to the parent axes
                # height of the zoomed area relative to the width
                # ratio is stored as float (ax aspect ratio for individual axes are considered later)
                "relative_height": relative_height,
                "bbox_to_anchor": (0.025, 0.75, 1.0, 1.0),
                "bbox_transform": "transAxes",  # bbox_transform=ax.transAxes,
                "borderpad": 0,  # padding between the inset and the surrounding axes
                "loc": "lower left",  # locates the zoomed area
                "loc1": 3,  # connecting corner for of the zoomed area with loc11 of the original axes
                "loc11": 2,  # corner number of the highlighted bix in the original axes
                "loc2": 4,  # connecting corner for of the zoomed area with loc22 of the original axes
                "loc22": 1,  # corner number of the highlighted bix in the original axes
                "facecolor": "k",
                "edgecolor": "k",
                "fill": False,
                "linewidth": mpl.rcParams["axes.linewidth"],
                "zoomed_scatter_size": 300,
                "zoomed_scatter_edge_width": 3,
                "subplots_adjust": {
                    "left": 0.05,
                    "bottom": 0.2,
                    "right": 1.0,
                    "top": 0.69,
                },
                "save_transparent": False,
                "zoom_axex_facecolor": "w",
            }

            # plot the moon data
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            ax.scatter(
                x_test[:, 0],
                x_test[:, 1],
                c=y_test,
                cmap="coolwarm",
                edgecolors="w",
                s=40,
            )
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            ax.set_xlim(min_data[0], max_data[0])
            ax.set_ylim(min_data[1], max_data[1])
            create_plots.set_major_ticks(ax, y_axis=True, x_axis=True)

            fig.savefig(os.path.join(plot_folder, "moon_data.svg"), bbox_inches="tight")
            fig.savefig(os.path.join(plot_folder, "moon_data.png"), bbox_inches="tight")

            # axins = zoomed_inset_axes(ax,
            #                           zoom=zoom_settings['zoom_factor'],
            #                           # locates the zoomed area
            #                           loc=zoom_settings['loc'],
            #                           #   bbox_to_anchor=(1.02, 10000, ),  # bbox_transform=ax.transAxes,
            #                           bbox_to_anchor=(0, 0, 1, 1),
            #                           )  # zoom = 2

            # ratio of the parent axes: height/width eg. height in inch / width in inch
            # axes_ratio = ax.get_position().width / ax.get_position().height
            axins = inset_axes(
                ax,
                width=zoom_settings["width"],
                height=f"{zoom_settings['relative_height'] * 100 * ax.get_position().width / ax.get_position().height}%",
                # locates the zoomed area
                loc=zoom_settings["loc"],
                borderpad=zoom_settings["borderpad"],
                bbox_to_anchor=zoom_settings["bbox_to_anchor"],
                bbox_transform=ax.transAxes
                if zoom_settings["bbox_transform"] == "transAxes"
                else None,
            )
            axins.patch.set_facecolor(zoom_settings["zoom_axex_facecolor"])

            axins.scatter(
                x_test[:, 0],
                x_test[:, 1],
                c=y_test,
                cmap="coolwarm",
                edgecolors="k",
                s=zoom_settings["zoomed_scatter_size"],
            )
            axins.set_xlim(zoom_range[0])
            axins.set_ylim(zoom_range[1])
            plt.xticks(visible=False)
            plt.yticks(visible=False)

            # mark_inset(ax, axins, loc1=zoom_settings['loc1'], loc2=zoom_settings['loc2'],
            mark_inset(
                ax,
                axins,
                loc1=zoom_settings["loc1"],
                loc2=zoom_settings["loc2"],
                loc11=zoom_settings["loc11"],
                loc22=zoom_settings["loc22"],
                facecolor=zoom_settings["facecolor"],
                edgecolor=zoom_settings["edgecolor"],
                fill=zoom_settings["fill"],
                linewidth=zoom_settings["linewidth"],
            )

            fig.subplots_adjust(**zoom_settings["subplots_adjust"])
            fig.savefig(
                os.path.join(plot_folder, "moon_data_zoom.svg"),
                transparent=zoom_settings["save_transparent"],
            )
            fig.savefig(
                os.path.join(plot_folder, "moon_data_zoom.svg"),
                transparent=zoom_settings["save_transparent"],
            )

            plt.close()

            # plot the moon data mean predictions and errors
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            moon_norm = mpl.colors.Normalize(
                vmin=0, vmax=1
            )  # classification mean is between 0 and 1
            train_moon.plot_pred_mean_and_errors(
                ax,
                x_in_test=x_test,
                predicted_mean_test=y_test_pred_mean,
                true_class=y_test,
                x_visualize=x_viz,
                predicted_visualize_mean=y_viz_pred_mean,
                edgecolors="w",
                cmap="coolwarm",
                norm=moon_norm,
                scatter_size=40,
                scatter_edge_width=0.75,
                colorbar_label=r"$\mu$",
            )
            # set min and max of ax
            ax.set_xlim(min_data[0], max_data[0])
            # ax.set_ylim(min_data[1], max_data[1])
            # axis label
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            create_plots.set_major_ticks(ax, y_axis=True, x_axis=True)
            fig.savefig(
                os.path.join(plot_folder, "moon_data_pred_mean_errors.svg"),
                bbox_inches="tight",
            )
            fig.savefig(
                os.path.join(plot_folder, "moon_data_pred_mean_errors.png"),
                bbox_inches="tight",
            )

            ################################################################################
            # plot the moon data mean predictions and errors again with zoom
            ################################################################################

            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            moon_norm = mpl.colors.Normalize(
                vmin=0, vmax=1
            )  # classification mean is between 0 and 1
            train_moon.plot_pred_mean_and_errors(
                ax,
                x_in_test=x_test,
                predicted_mean_test=y_test_pred_mean,
                true_class=y_test,
                x_visualize=x_viz,
                predicted_visualize_mean=y_viz_pred_mean,
                edgecolors="w",
                cmap="coolwarm",
                norm=moon_norm,
                scatter_size=40,
                scatter_edge_width=0.75,
                colorbar_label=r"$\mu$",
                add_colorbar=False,
            )
            # set min and max of ax
            ax.set_xlim(min_data[0], max_data[0])
            # ax.set_ylim(min_data[1], max_data[1])
            # axis label
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            create_plots.set_major_ticks(ax, y_axis=True, x_axis=True)

            # axins = zoomed_inset_axes(ax, zoom=zoom_settings['zoom_factor'], loc=zoom_settings['loc'])  # zoom = 2

            axins = inset_axes(
                ax,
                width=zoom_settings["width"],
                height=f"{zoom_settings['relative_height'] * 100 * ax.get_position().width / ax.get_position().height}%",
                # locates the zoomed area
                loc=zoom_settings["loc"],
                borderpad=zoom_settings["borderpad"],
                bbox_to_anchor=zoom_settings["bbox_to_anchor"],
                bbox_transform=ax.transAxes
                if zoom_settings["bbox_transform"] == "transAxes"
                else None,
            )
            axins.patch.set_facecolor(
                zoom_settings["zoom_axex_facecolor"]
            )  # set background color of the zoomed axes

            train_moon.plot_pred_mean_and_errors(
                axins,
                x_in_test=x_test,
                predicted_mean_test=y_test_pred_mean,
                true_class=y_test,
                x_visualize=x_viz,
                predicted_visualize_mean=y_viz_pred_mean,
                edgecolors="w",
                cmap="coolwarm",
                norm=moon_norm,
                scatter_size=zoom_settings["zoomed_scatter_size"],
                scatter_edge_width=zoom_settings["zoomed_scatter_edge_width"],
                colorbar_label=r"$\mu$",
                add_colorbar=False,
            )
            axins.set_xlim(zoom_range[0])
            axins.set_ylim(zoom_range[1])
            plt.xticks(visible=False)
            plt.yticks(visible=False)

            mark_inset(
                ax,
                axins,
                loc1=zoom_settings["loc1"],
                loc2=zoom_settings["loc2"],
                loc11=zoom_settings["loc11"],
                loc22=zoom_settings["loc22"],
                facecolor=zoom_settings["facecolor"],
                edgecolor=zoom_settings["edgecolor"],
                fill=zoom_settings["fill"],
                linewidth=zoom_settings["linewidth"],
            )

            # fig.subplots_adjust(left=0.2, bottom=0.2, right=1., )
            fig.subplots_adjust(**zoom_settings["subplots_adjust"])
            fig.savefig(
                os.path.join(plot_folder, "moon_data_pred_mean_errors_zoom.svg"),
                transparent=zoom_settings["save_transparent"],
            )
            fig.savefig(
                os.path.join(plot_folder, "moon_data_pred_mean_errors_zoom.png"),
            )

            plt.close()

            ############################################################################
            # plot the moon data variance predictions and errors
            ############################################################################
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            moon_norm = mpl.colors.Normalize(
                vmin=0, vmax=1
            )  # variance is also between 0 and 1
            train_moon.plot_pred_mean_and_errors(
                ax,
                x_in_test=x_test,
                predicted_mean_test=y_test_pred_mean,
                true_class=y_test,
                x_visualize=x_viz,
                predicted_visualize_mean=y_viz_pred_cov,
                edgecolors="w",
                cmap="plasma",
                norm=moon_norm,
                scatter_size=40,
                scatter_edge_width=0.75,
                colorbar_label=r"$\sigma^2$",
            )
            # set min and max of ax
            ax.set_xlim(min_data[0], max_data[0])
            # ax.set_ylim(min_data[1], max_data[1])
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            create_plots.set_major_ticks(ax, y_axis=True, x_axis=True)
            fig.savefig(
                os.path.join(plot_folder, "moon_data_pred_var_errors.svg"),
                bbox_inches="tight",
            )
            fig.savefig(
                os.path.join(plot_folder, "moon_data_pred_var_errors.png"),
                bbox_inches="tight",
            )

            # plot the moon data variance predictions and errors again with zoom
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            moon_norm = mpl.colors.Normalize(
                vmin=0, vmax=1
            )  # variance is also between 0 and 1
            train_moon.plot_pred_mean_and_errors(
                ax,
                x_in_test=x_test,
                predicted_mean_test=y_test_pred_mean,
                true_class=y_test,
                x_visualize=x_viz,
                predicted_visualize_mean=y_viz_pred_cov,
                edgecolors="w",
                cmap="plasma",
                norm=moon_norm,
                scatter_size=40,
                scatter_edge_width=0.75,
                colorbar_label=r"$\sigma^2$",
                add_colorbar=False,
            )
            # set min and max of ax
            ax.set_xlim(min_data[0], max_data[0])
            # ax.set_ylim(min_data[1], max_data[1])
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            create_plots.set_major_ticks(ax, y_axis=True, x_axis=True)

            # axins = zoomed_inset_axes(ax, zoom=zoom_settings['zoom_factor'], loc=zoom_settings['loc'])

            axins = inset_axes(
                ax,
                width=zoom_settings["width"],
                height=f"{zoom_settings['relative_height'] * 100 * ax.get_position().width / ax.get_position().height}%",
                # locates the zoomed area
                loc=zoom_settings["loc"],
                borderpad=zoom_settings["borderpad"],
                bbox_to_anchor=zoom_settings["bbox_to_anchor"],
                bbox_transform=ax.transAxes
                if zoom_settings["bbox_transform"] == "transAxes"
                else None,
            )
            axins.patch.set_facecolor(
                zoom_settings["zoom_axex_facecolor"]
            )  # set background color of the zoomed axes

            train_moon.plot_pred_mean_and_errors(
                axins,
                x_in_test=x_test,
                predicted_mean_test=y_test_pred_mean,
                true_class=y_test,
                x_visualize=x_viz,
                predicted_visualize_mean=y_viz_pred_cov,
                edgecolors="w",
                cmap="plasma",
                norm=moon_norm,
                scatter_size=zoom_settings["zoomed_scatter_size"],
                scatter_edge_width=zoom_settings["zoomed_scatter_edge_width"],
                colorbar_label=r"$\sigma^2$",
                add_colorbar=False,
            )
            axins.set_xlim(zoom_range[0])
            axins.set_ylim(zoom_range[1])
            plt.xticks(visible=False)
            plt.yticks(visible=False)

            mark_inset(
                ax,
                axins,
                loc1=zoom_settings["loc1"],
                loc2=zoom_settings["loc2"],
                loc11=zoom_settings["loc11"],
                loc22=zoom_settings["loc22"],
                facecolor=zoom_settings["facecolor"],
                edgecolor=zoom_settings["edgecolor"],
                fill=zoom_settings["fill"],
                linewidth=zoom_settings["linewidth"],
            )

            fig.subplots_adjust(**zoom_settings["subplots_adjust"])
            fig.savefig(
                os.path.join(plot_folder, "moon_data_pred_var_errors_zoom.svg"),
                transparent=zoom_settings["save_transparent"],
            )
            fig.savefig(
                os.path.join(plot_folder, "moon_data_pred_var_errors_zoom.png"),
            )

            plt.close()

        else:
            # create_plots.varianz_plot_2d_input(x_gt, predictions_comp_gt.cov,
            #                                    plot_folder=plot_folder,
            #                                    min_data=min_data, max_data=max_data,
            #                                    y_label=r'$\sigma^2$', )

            # Plot true decision boundary using polynomial coefficients
            polynomial_coeffs = [0.0, 0.5, -1, -0.5]
            true_boundary_x1 = np.linspace(min_data[0], max_data[0], 100)
            true_boundary_x2 = np.polyval(polynomial_coeffs, true_boundary_x1)
            true_boundary = np.stack((true_boundary_x1, true_boundary_x2), axis=1)

            create_plots.logistic_regression_plot_2d_input(
                x_gt,
                (predictions_comp_gt.mean, predictions_comp_gt.cov),
                plot_folder=plot_folder,
                min_data=min_data,
                max_data=max_data,
                decision_boundary=true_boundary,
                cmap_name="plasma",
                file_prefix="pred_with_decision_bound",
                plot_every_nth_contline=20,
            )

            create_plots.regression_plot_gaussian(
                x_gt,
                (predictions_comp_gt.mean, predictions_comp_gt.cov),
                plot_folder=plot_folder,
                min_data=min_data,
                max_data=max_data,
                cmap_name="plasma",
                file_prefix="pred",
                plot_every_nth_contline=20,
            )

    ########################################################
    # plot the graph of the kd tree
    ########################################################
    kd_graph = KDTreeGraph(
        space_partitioning.kdtree,
        use_tex=True,
        graph_attr={
            "ranksep": "0.2",  # default 0.5 # min = 0.02
            "nodesep": "0.1",  # default 0.25 min = 0.02
            "bgcolor": "transparent",  # default white
            #    'fillcolor': 'purple',
            #    'style': 'filled',
        },
    )
    kd_graph.kdtree_to_graphviz()
    kd_graph.save_graph("kd_tree_graph", directory=plot_folder, file_format="svg")
    kd_graph.save_graph_as_tex(
        directory=plot_folder,
    )
    region_labels = kd_graph.get_leaf_labels()

    split_planes = kd_graph.get_split_plains()
    split_points, split_labels = split_planes.get_split_points_labels()

    ########################################################
    # create gif of the kd tree graph
    ########################################################
    gif_folder = os.path.join(plot_folder, "kd_tree_graph_gif")
    # create folder if it does not exist
    os.makedirs(gif_folder, exist_ok=True)

    if num_test_data < gif_limit:
        split_planes.create_gif_of_2d_input_graph_creation(
            mins=min_data,
            maxes=max_data,
            duration=1000,
            input_data=x_test,
            plot_dir=gif_folder,
            image_format="png",
            dpi=300,
            axis_labels=[r"$x_1$", r"$x_2$"],
        )
    else:
        warn_txt = "Skipping gif creation due to large number of test data points"
        # print(warn_txt)

        print("\x1b[33;37;41m" + warn_txt + "\x1b[0m")

    # space partitioning plot
    fig, axes = plt.subplots(constrained_layout=True)
    # equal axis scaling
    axes.set_aspect("equal", "box")
    if plot_input_data_in_part_plot:
        axes.plot(
            x_test[:, 0],
            x_test[:, 1],
            "o",
            label="Data Points",
            alpha=0.25,
        )

    # space_partitions.mins[np.argmin(space_partitions.mins, axis=0)] = min_data
    # space_partitions.maxes[np.argmax(space_partitions.maxes, axis=0)] = max_data

    kd_tree_partioning.plot_partitions(
        axes,
        space_partitions,
        facecolor="white",
        region_labels=region_labels,
        plot_region_labels=show_split_points,
    )

    if split_points is not None and show_split_points:
        # split_points has dimension (n_splits, 3*dim_data)
        # see kd_tree_graph.py for the structure of split_points
        axes.plot(
            split_points[:, 0],
            split_points[:, 1],
            "o",
            label="Split Points",
        )

        for i, txt in enumerate(split_labels):
            axes.annotate(
                txt,
                (split_points[i, 0], split_points[i, 1]),
            )

    # set axis limits according to (artificial) data mins and maxes
    axes.set_xlim(min_data[0], max_data[0])
    axes.set_ylim(min_data[1], max_data[1])

    axes.set_xlabel(r"$x_1$")
    axes.set_ylabel(r"$x_2$")
    create_plots.set_major_ticks(axes)

    fig.savefig(os.path.join(plot_folder, "kd_regions_raw.svg"), transparent=True)
    plt.close()

    # if scenario number is 2000, use matplotlib settiungs as in Fusion 25 Paper
    if DATA_ID == 2000:
        mpl.rcParams.update(mpl.rcParamsDefault)
        USE_TEX = True
        height_to_width_ratio = 0.75
        unscl_fontsize = 12
        fig_width = 3.0
        matplotlib_settings.init_settings(
            use_tex=USE_TEX,
            height_to_width_ratio=height_to_width_ratio,
            fig_width=fig_width,
            unscaled_fontsize=unscl_fontsize,
        )
        cbar_padding = 0.5  # used in FUSION 25 Paper
    else:
        cbar_padding = 0.15  # used in FUSION 24 Paper and MFI 2024

    # space partitioning plot with distance
    fig, axes = plt.subplots()

    # equal axis scaling
    axes.set_aspect("equal", "box")
    _, cbar = kd_tree_partioning.plot_kd_space(
        space_partitions,
        stat_per_region,
        axes=axes,
        accept_stat_per_region=accept_stat_per_region,
        distance_label=used_dist.upper(),
        cmap=cmap,
        norm=norm,
        extend=extend,
        hatch_type=hatch_type,
        cbar_padding=cbar_padding,
    )
    if DATA_ID == 2000:
        max_ticks = 5
        from matplotlib import ticker

        # get color bar instance from figure
        # cbar = fig.get_children()[2].cbar
        cbar.locator = ticker.MaxNLocator(nbins=max_ticks)
        cbar.update_ticks()

    if plot_input_data_in_test_plot:
        axes.plot(
            x_test[:, 0],
            x_test[:, 1],
            "o",
            label="Data Points",
            markerfacecolor="None",
            markeredgecolor="white",
            alpha=0.25,
        )

    # set axis limits according to (artificial) data mins and maxes
    axes.set_xlim(min_data[0], max_data[0])
    axes.set_ylim(min_data[1], max_data[1])

    create_plots.set_major_ticks(axes)

    # set axis labels
    axes.set_xlabel(r"$x_1$")
    axes.set_ylabel(r"$x_2$")
    # fig.savefig(plot_folder + '/kd_regions_dist.svg')

    plot_type = ["pdf", "svg"]
    for p_type in plot_type:
        create_plots.custom_safefig(
            fig, os.path.join(plot_folder, f"kd_regions_dist.{p_type}")
        )

    plt.close()

    ########################################################
    # use ball tree
    ########################################################
    btr = balltree.BallTreeRegions(
        x_test, leaf_size=LEAFSIZE_BALLTREE
    )  # create BallTreeRegions object
    # btr.print_tree_data_and_structure()  # print tree data and structure
    # btr.plot_2d_ball_tree_regions()  # plot 2D ball tree regions
    idx_per_region = btr.get_idx_in_leaf_balls()  # get the indices per leaf ball

    gif_folder = os.path.join(plot_folder, "ball_tree_gif")

    cbt = balltree.CustomBallTree(
        leaf_size=LEAFSIZE_BALLTREE,
    )
    cbt.rebuild_from_sklearn_tree(ball_tree=btr.tree)

    if num_test_data < gif_limit:
        cbt.create_gif(
            data=x_test,
            plot_dir=gif_folder,
            image_format="png",
            dpi=100,
            duration=1000,
            axis_labels=[r"$x_1$", r"$x_2$"],
            scatter_point_size=10,
        )

    cbt.calc_stat_per_node(
        predictions, y_test, method=used_dist, alpha=SIGNIFICANCE_LEVEL
    )
    cbt.calc_combined_stat_per_leaf()  # calculate the combined distance measure per region

    # calc stat for each region
    idx_per_region = btr.get_idx_per_ball()  # get the indices per leaf ball
    stat_per_region, accept_stat_per_region = (
        distance_stat_wrapper.distance_calculation(
            predictions,
            y_test,
            idx_per_region,
            method=used_dist,
            alpha=SIGNIFICANCE_LEVEL,
        )
    )
    btr.set_stat_per_region(stat_per_region)  # set the distance measure per region

    # print results as pretty table
    btr.print_stat_per_region()

    print(f"Tree depth: {cbt.get_tree_depth()}")

    ########################################################
    # ball tree graph
    ########################################################
    cbt.draw_tree_graph(
        filename="balltree_graph",
        use_latex=True,
        directory=plot_folder,
        file_format="svg",
    )
    cbt.draw_latex_tree_graph(
        filename="balltree", directory=plot_folder, compile_pdf=False
    )
    cbt.draw_latex_tree_graph(
        filename="balltree_with_stat",
        directory=plot_folder,
        use_stat=True,
        compile_pdf=False,
    )
    ########################################################
    # ball tree input space plots
    ########################################################
    # get color style for the distance plot
    # vmax = None # maximum value for the color map
    cmap, norm, extend, hatch_type = get_color_style(
        used_dist, vmax=None, stat_per_region=stat_per_region
    )

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)
    btr.plot_2d_ball_tree_regions(
        ax=ax, show_stats=False
    )  # plot 2D ball tree regions with data points
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_facecolor("white")
    # create_plots.set_major_ticks(ax)
    fig.savefig(os.path.join(plot_folder, "balltree_regions.svg"), bbox_inches="tight")
    plt.close()

    # plot 2D ball tree regions with data points
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    # ax = btr.plot_2d_ball_tree_regions(ax=ax, show_stats=True,
    #                                    plot_parents=False, plot_circles=False, point_size=15,
    #                                    cmap=cmap, norm=norm)
    ax = cbt.plot_2d_ball_tree_regions(
        ax=ax,
        show_stats=True,
        show_combined_stat=True,
        plot_parents=False,
        plot_circles=True,
        plot_points=True,
        point_size=15,
        alpha_transparency_circles=0.5,
        edgecolor_points="k",
        edgecolor_circles="k",
        cmap=cmap,
        norm=norm,
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim(min_data[0], max_data[0])
    if DATA_ID < 1010:
        ax.set_ylim(min_data[1], max_data[1])
        create_plots.set_major_ticks(ax, y_axis=True, x_axis=True)
    else:
        ax.set_ylim(min_data[1], max_data[1])
        create_plots.set_major_ticks(ax, y_axis=True, x_axis=True)
        # create_plots.set_colorbar_max_ticks(ax, max_ticks=5)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)

    # add colorbar to the plot
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        label=used_dist.upper(),
        extend=extend,
    )
    create_plots.set_colorbar_max_ticks(cbar, max_ticks=5)

    # set axis labels

    fig.savefig(
        os.path.join(plot_folder, "ball_tree_regions_dist.svg"), bbox_inches="tight"
    )
    plt.close()

    ######################################################################################################################
    # plot again with zoom in area
    ######################################################################################################################

    if not moon_data:
        return

    # plot 2D ball tree regions with data points
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    # ax = btr.plot_2d_ball_tree_regions(ax=ax, show_stats=True,
    #                                    plot_parents=False, plot_circles=False, point_size=15,
    #                                    cmap=cmap, norm=norm)
    ax = cbt.plot_2d_ball_tree_regions(
        ax=ax,
        show_stats=True,
        show_combined_stat=True,
        plot_parents=False,
        plot_circles=True,
        plot_points=True,
        point_size=15,
        alpha_transparency_circles=0.5,
        edgecolor_points="k",
        edgecolor_circles="k",
        cmap=cmap,
        norm=norm,
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim(min_data[0], max_data[0])
    if DATA_ID < 1010:
        ax.set_ylim(min_data[1], max_data[1])
        create_plots.set_major_ticks(ax, y_axis=True, x_axis=True)
    else:
        ax.set_ylim(min_data[1], max_data[1])
        create_plots.set_major_ticks(ax, y_axis=True, x_axis=True)
        # create_plots.set_colorbar_max_ticks(ax, max_ticks=5)

    # axins = zoomed_inset_axes(ax, zoom=zoom_settings['zoom_factor'], loc=zoom_settings['loc'])

    axins = inset_axes(
        ax,
        width=zoom_settings["width"],
        height=f"{zoom_settings['relative_height'] * 100 * ax.get_position().width / ax.get_position().height}%",
        # locates the zoomed area
        loc=zoom_settings["loc"],
        borderpad=zoom_settings["borderpad"],
        bbox_to_anchor=zoom_settings["bbox_to_anchor"],
        bbox_transform=ax.transAxes
        if zoom_settings["bbox_transform"] == "transAxes"
        else None,
    )
    axins.patch.set_facecolor(
        zoom_settings["zoom_axex_facecolor"]
    )  # set background color of the zoomed axes

    cbt.plot_2d_ball_tree_regions(
        ax=axins,
        show_stats=True,
        show_combined_stat=True,
        plot_parents=False,
        plot_circles=True,
        plot_points=True,
        point_size=zoom_settings["zoomed_scatter_size"],
        alpha_transparency_circles=0.5,
        edgecolor_points="k",
        edgecolor_circles="k",
        cmap=cmap,
        norm=norm,
    )

    axins.set_xlim(zoom_range[0])
    axins.set_ylim(zoom_range[1])
    plt.xticks(visible=False)
    plt.yticks(visible=False)

    # mark the region of the zoomed area
    mark_inset(
        ax,
        axins,
        loc1=zoom_settings["loc1"],
        loc2=zoom_settings["loc2"],
        loc11=zoom_settings["loc11"],
        loc22=zoom_settings["loc22"],
        #    facecolor=zoom_settings['facecolor'],
        #    edgecolor=zoom_settings['edgecolor'],
        facecolor="w",
        edgecolor="w",
        fill=zoom_settings["fill"],
        linewidth=zoom_settings["linewidth"],
    )

    fig.subplots_adjust(**zoom_settings["subplots_adjust"])
    fig.savefig(
        os.path.join(plot_folder, "ball_tree_regions_dist_zoom.svg"),
        transparent=zoom_settings["save_transparent"],
    )
    plt.close()


def get_color_style(
    used_dist: str, vmax: float = None, stat_per_region: np.ndarray = None
):
    """Get color style for the distance plot."""

    if vmax is None:
        vmax = stat_per_region.max()
    if used_dist == "anees":
        vmin = 0.0  # minimum value for the color map
        vmax = 2.0  # maximum value for the color map
        cmap, norm = cmap_norm_plot_settings.get_anees_plot_settings(
            vmin=0.0,
            vmax=2.0,
            set_under_color=None,
            set_over_color=None,
        )

        if vmin > 0.0:
            extend = "both"
        else:
            extend = "max"
        hatch_type = "/"
    elif used_dist == "ece":
        # vmax = stat_per_region.max()

        # round to the next 0.01
        vmax = np.ceil(vmax * 1000) / 1000

        # use viridis or gnuplot colormap
        cmap, norm = create_plots.get_wasserstein_plot_settings(
            vmin=0.0, vmax=vmax, cmap_name="viridis"
        )
        extend = "max"
        hatch_type = None
    elif used_dist == "uce":
        # vmax = stat_per_region.max()

        # round to the next 0.01
        vmax = 1.0
        vmax = np.ceil(vmax * 1000) / 1000

        # use viridis or gnuplot colormap
        cmap, norm = create_plots.get_wasserstein_plot_settings(
            vmin=0.0, vmax=vmax, cmap_name="viridis"
        )
        extend = "max"
        hatch_type = None

    else:
        cmap = "viridis"
        norm = None
        extend = "neither"
        hatch_type = None

    return cmap, norm, extend, hatch_type


if __name__ == "__main__":
    main()
