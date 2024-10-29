""" Module to create 2D plots for the Wasserstein distance and the Gaussian regression. """


import os
import typing
from matplotlib import ticker


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import scipy.interpolate


def wasserstein_plot_2d_input(x_test,
                              w_dist,
                              plot_folder: str,
                              min_data: typing.Union[np.ndarray, list],
                              max_data: typing.Union[np.ndarray, list],
                              contour_lvls: np.ndarray = None,
                              ):

    # create figure and axes
    fig, axes = plt.subplots()
    # equal axis scaling
    axes.set_aspect('equal', 'box')
    # 2D data with distance as color
    # plot at triangle mesh with the distance as color

    if contour_lvls is None:
        vmax = np.max(w_dist)
        vmin = 0.
        # get "nice" contour levels for the plot

        step = vmax / 40
        contour_lvls = np.arange(start=vmin, stop=vmax+step, step=step, )

    cmap, norm = get_wasserstein_plot_settings(
        vmin=vmin, vmax=vmax, cmap_name='viridis',)
    if np.max(w_dist) > vmax:
        extend = 'max'
    else:
        extend = 'neither'
    contour_plot = axes.tricontourf(x_test[:, 0], x_test[:, 1], w_dist,
                                    levels=contour_lvls,
                                    cmap=cmap, norm=norm, extend=extend,)

    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.15)

    # axes.scatter(x_test[:, 0], x_test[:, 1], c=w_dist, cmap='viridis', )
    axes.set_xlim(min_data[0], max_data[0])
    axes.set_ylim(min_data[1], max_data[1])
    axes.set_xlabel(r'$x_1$')
    axes.set_ylabel(r'$x_2$')
    # colorbar for distance
    cbar = fig.colorbar(contour_plot, cax=cax,)

    set_colorbar_max_ticks(cbar, max_ticks=5)

    cbar.set_label(r'$1$-Wasserstein distance')
    file_path = os.path.join(plot_folder, 'wasserstein_dist.svg')

    set_major_ticks(axes)
    # fig.savefig(file_path)
    custom_safefig(fig, file_path)


def logistic_regression_plot_2d_input(x_test, gaussian, plot_folder: str, min_data, max_data,
                                      file_prefix: str = '',
                                      contour_lvls_mean: np.ndarray = None,
                                      contour_lvls_var: np.ndarray = None,
                                      decision_boundary: np.ndarray = None,
                                      plot_every_nth_contline: int = 6,
                                      cmap_name='plasma'
                                      ):
    mean, var = gaussian

    save_path_mean = os.path.join(plot_folder, f'{file_prefix}_mean.svg')
    save_path_var = os.path.join(plot_folder, f'{file_prefix}_variance.svg')
    # mean plot
    if contour_lvls_mean is None:
        vmax = np.max(mean)
        vmin = np.min(mean)
        step = (vmax - vmin) / 40
        contour_lvls_mean = np.arange(start=vmin, stop=vmax+step, step=step, )
    ax = varianz_plot_2d_input(x_test, mean, plot_folder, min_data, max_data,
                               y_label=r'$\mu$', return_axes=True, contour_lvls=contour_lvls_mean,
                               cmap_name=cmap_name)

    if decision_boundary is not None:
        ax.plot(decision_boundary[:, 0], decision_boundary[:, 1], label='Decision boundary', color='k', linestyle='--',)

    fig = ax.get_figure()
    plot_2d_contour_lines(ax, x_test, mean,
                          contour_lvls_mean, plot_every_nth_contline, cmap_name=cmap_name,
                          colornorm_truncation_offset=0.01)
    # set major ticks of x and y axis to be equal
    set_major_ticks(ax)
    # save the plot
    custom_safefig(fig, save_path_mean)

    # var plot
    if contour_lvls_var is None:
        vmax = np.max(var)
        vmin = np.min(var)
        step = (vmax - vmin) / 40
        # if the variance is constant, only plot one level
        if np.isclose(vmax, vmin, atol=1e-5):
            contour_lvls_var = [vmin-1e-5, vmin, vmin+1e-5]
        else:
            contour_lvls_var = np.arange(start=vmin, stop=vmax+step, step=step, )
    ax = varianz_plot_2d_input(x_test, var, plot_folder, min_data, max_data,
                               y_label=r'$\sigma^2$', return_axes=True, contour_lvls=contour_lvls_var,
                               cmap_name=cmap_name)
    plot_2d_contour_lines(ax, x_test, var, contour_lvls_mean, plot_every_nth_contline, cmap_name=cmap_name)

    if decision_boundary is not None:
        ax.plot(decision_boundary[:, 0], decision_boundary[:, 1], label='Decision boundary', color='w', linestyle='--',)

    # set major ticks of x and y axis to be equal
    set_major_ticks(ax)

    fig = ax.get_figure()
    custom_safefig(fig, save_path_var)


def regression_plot_gaussian(x_test, gaussian, plot_folder: str, min_data, max_data,
                             file_prefix: str = '',
                             contour_lvls_mean: np.ndarray = None,
                             contour_lvls_var: np.ndarray = None,
                             plot_every_nth_contline: int = 6,
                             cmap_name='plasma'
                             ):
    mean, var = gaussian

    save_path_mean = os.path.join(plot_folder, f'{file_prefix}_mean.svg')
    save_path_var = os.path.join(plot_folder, f'{file_prefix}_variance.svg')
    # mean plot
    if contour_lvls_mean is None:
        vmax = np.max(mean)
        vmin = np.min(mean)
        step = (vmax - vmin) / 40
        contour_lvls_mean = np.arange(start=vmin, stop=vmax+step, step=step, )

    if plot_every_nth_contline < 1:
        plot_every_nth_contline = -1  # plot every given contour line

    ax = varianz_plot_2d_input(x_test, mean, plot_folder, min_data, max_data,
                               y_label=r'$\mu$', return_axes=True, contour_lvls=contour_lvls_mean,
                               cmap_name=cmap_name)
    fig = ax.get_figure()
    # contour line
    # contour_line = ax.tricontour(x_test[:, 0], x_test[:, 1], mean,
    #                              levels=contour_lvls_mean[::plot_every_nth_contline], colors='w',)  # black contour line

    if plot_every_nth_contline > 0:
        plot_2d_contour_lines(ax, x_test, mean, contour_lvls_mean, plot_every_nth_contline, cmap_name=cmap_name)
    # set major ticks of x and y axis to be equal
    set_major_ticks(ax)

    # save the plot
    custom_safefig(fig, save_path_mean)

    # var plot
    if contour_lvls_var is None:
        vmax = np.max(var)
        vmin = np.min(var)
        step = (vmax - vmin) / 40
        # if the variance is constant, only plot one level
        if np.isclose(vmax, vmin, atol=1e-5):
            contour_lvls_var = [vmin-1e-5, vmin, vmin+1e-5]
        else:
            contour_lvls_var = np.arange(start=vmin, stop=vmax+step, step=step, )
    ax = varianz_plot_2d_input(x_test, var, plot_folder, min_data, max_data,
                               y_label=r'$\sigma^2$', return_axes=True, contour_lvls=contour_lvls_var,
                               cmap_name='plasma')
    if plot_every_nth_contline > 0:
        plot_2d_contour_lines(ax, x_test, var, contour_lvls_mean, plot_every_nth_contline, cmap_name=cmap_name)

    # set major ticks of x and y axis to be equal
    set_major_ticks(ax)

    fig = ax.get_figure()
    custom_safefig(fig, save_path_var)


def set_major_ticks(ax: plt.Axes, x_axis: bool = True, y_axis: bool = True,
                    xtol: float = 0.1,
                    ytol: float = 0.1,  max_ticks: int = 5,):
    """ Set major ticks for the x and y axis of the plot.

    :param ax: Axes of the plot
    :type ax: plt.Axes
    :param x_axis: If True, set major ticks for the x axis
    :type x_axis: bool
    :param y_axis: If True, set major ticks for the y axis
    :type y_axis: bool
    :param xtol: Tolerance for the x axis
    :type xtol: float
    :param ytol: Tolerance for the y axis
    :type ytol: float
    """
    # xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # # add tolerance

    # if x_axis:
    #     x_range = np.arange(xlim[0] - xtol, xlim[1] + xtol+1, 1)
    #     # round depanding on the range
    #     if x_range[-1] - x_range[0] < 10:
    #         x_range = np.round(x_range, 0)
    #         x_range = np.arange(x_range[0], x_range[-1]+1, 0.5)

    #     # remove ticks which are outside the range
    #     x_range = x_range[(x_range >= xlim[0]) & (x_range <= xlim[1])]
    #     ax.set_xticks(x_range)

    # if y_axis:
    #     y_range = np.arange(ylim[0]-ytol, ylim[1] + ytol+1, 1)
    #     if y_range[-1] - y_range[0] < 10:
    #         y_range = np.round(y_range, 0)
    #         y_range = np.arange(y_range[0], y_range[-1]+1, 0.5)

    #     # remove ticks which are outside the range
    #     y_range = y_range[(y_range >= ylim[0]) & (y_range <= ylim[1])]

    #     # remove ticks if more
    #     ax.set_yticks(y_range)

    locator = ticker.MaxNLocator(prune='both', nbins=5)
    ax.yaxis.set_major_locator(locator)
    ax.xaxis.set_major_locator(locator)


def set_colorbar_max_ticks(cbar: matplotlib.colorbar.Colorbar, max_ticks: int = 5):
    """ Set the maximum number of ticks for the colorbar.

    :param cbar: Colorbar of the plot
    :type cbar: matplotlib.colorbar.Colorbar
    :param max_ticks: Maximum number of ticks for the colorbar
    :type max_ticks: int
    """
    tick_locator = ticker.MaxNLocator(nbins=max_ticks)
    cbar.locator = tick_locator
    cbar.update_ticks()


def logistic_regression_varianz_plot(x_test, varianz, plot_folder: str, min_data, max_data):

    polynomial_coeffs = [0., 0.5, -1, -.5]

    sample_boundary_x = np.linspace(min_data[0], max_data[0], 100)
    true_boundary_x = np.polyval(polynomial_coeffs, sample_boundary_x)

    ax = varianz_plot_2d_input(x_test, varianz, plot_folder, min_data, max_data, return_axes=True)

    ax.plot(sample_boundary_x, true_boundary_x, label='True boundary')

    file_path = os.path.join(plot_folder, 'predicted_variance.svg')
    ax.get_figure().savefig(file_path)


def varianz_plot_2d_input(x_test, varianz, plot_folder: str,
                          min_data, max_data,
                          y_label: str = r'$y$',
                          contour_lvls: np.ndarray = None,
                          return_axes=False,
                          cmap_name: str = 'viridis'):

    # create figure and axes
    fig, axes = plt.subplots()
    # equal axis scaling
    axes.set_aspect('equal', 'box')
    # 2D data with distance as color
    # plot at triangle mesh with the distance as color

    if contour_lvls is None:
        vmax = np.max(varianz)
        vmin = np.min(varianz)
        step = 0.001
        contour_lvls = np.arange(start=vmin, stop=vmax+step, step=step, )
    else:
        vmax = np.max(contour_lvls)
        vmin = np.min(contour_lvls)

    cmap, norm = get_wasserstein_plot_settings(
        vmin=vmin, vmax=vmax,
        cmap_name=cmap_name,)

    is_lower_than_max = np.max(varianz) < vmax or np.isclose(np.max(varianz), vmax, )
    is_higher_than_min = np.min(varianz) > vmin or np.isclose(np.min(varianz), vmin, )

    # are both bounds respected?
    if is_lower_than_max and is_higher_than_min:
        extend = 'neither'
    # are both bounds violated?
    elif not is_lower_than_max and not is_higher_than_min:
        extend = 'both'
    # only the upper bound is violated
    elif not is_lower_than_max and is_higher_than_min:
        extend = 'max'
    else:
        extend = 'min'

    contour_plot = axes.tricontourf(x_test[:, 0], x_test[:, 1], varianz,
                                    levels=contour_lvls,
                                    cmap=cmap, norm=norm, extend=extend,)

    # axes.scatter(x_test[:, 0], x_test[:, 1], c=w_dist, cmap='viridis', )
    axes.set_xlim(min_data[0], max_data[0])
    axes.set_ylim(min_data[1], max_data[1])
    axes.set_xlabel(r'$x_1$')
    axes.set_ylabel(r'$x_2$')
    # colorbar for distance

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.15)

    cbar = fig.colorbar(contour_plot, cax=cax, extend=extend)
    # set_colorbar_max_ticks(cbar, max_ticks=5)
    cbar.set_label(y_label)
    cbar.ax.locator_params(nbins=5)

    if return_axes:
        return axes
    file_path = os.path.join(plot_folder, 'predicted_variance.svg')
    fig.savefig(file_path)


def get_wasserstein_plot_settings(vmin=None,
                                  vmax=None,
                                  cmap_name='viridis',
                                  set_under_color='steelblue',
                                  set_over_color='gold',):
    cmap = plt.get_cmap(cmap_name)
    # if vmin and vmax are set, normalize the color map
    if vmin is not None and vmax is not None:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    if vmin is not None:
        cmap.set_under(set_under_color)
    if vmax is not None:
        cmap.set_over(set_over_color)

    return cmap, norm


def invert_colormap(cmap):
    # Get the colors from the original colormap
    colors = cmap(np.linspace(0, 1, cmap.N))

    # Invert each RGB color
    inverted_colors = 1.0 - colors[:, :3]  # Invert RGB components

    # Create a new colormap with inverted colors
    inverted_cmap = ListedColormap(inverted_colors)

    return inverted_cmap


class LinearNDInterpolatorExtrapolator:
    """ LinearNDInterpolator used for extrapolation
    if the point is outside the convex hull of the input points, the nearest neighbor is used for extrapolation
    """

    def __init__(self, points: np.ndarray, values: np.ndarray):
        """ Use ND-linear interpolation over the convex hull of points, and nearest neighbor outside (for
            extrapolation)

            Idea taken from https://stackoverflow.com/questions/20516762/extrapolate-with-linearndinterpolator
        """
        self.linear_interpolator = scipy.interpolate.LinearNDInterpolator(points, values)
        self.nearest_neighbor_interpolator = scipy.interpolate.NearestNDInterpolator(points, values)

    def __call__(self, *args) -> typing.Union[float, np.ndarray]:
        t = self.linear_interpolator(*args)
        t[np.isnan(t)] = self.nearest_neighbor_interpolator(*args)[np.isnan(t)]
        if t.size == 1:
            return t.item(0)
        return t


def extrapolate_2D_data(x_test: np.ndarray, z_values: np.ndarray, num_points: int = 10, use_extrapolate: bool = True,
                        ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """ Extrapolate the 2D data to remove artefacts at the border of the plot. 
    Original data is concatenated with the extrapolated data. 
    cut of the data range in plot to remove artefacts at the border of the plot. 

    :param x_test: 2D data
    :type x_test: np.ndarray
    :param z_values: 1D output values of the 2D data
    :type z_values: np.ndarray
    :param num_points: Number of points for the extrapolation
    :type num_points: int
    :param use_extrapolate: If True, use the extrapolation
    :type use_extrapolate: bool
    :return: Extrapolated 2D data and output values
    :rtype: typing.Tuple[np.ndarray, np.ndarray]
    """

    if not use_extrapolate:
        return x_test, z_values
    # extrapolate over bounds of x_test to remove artefacts at the border
    # new points for extrapolation:
    # xxxxxxx
    # x-----x
    # x|   |x
    # x|   |x
    # x-----x
    # xxxxxxx
    # Get the maximum and min values in both dimensions of x_test
    max_x_test = np.max(x_test, axis=0)
    min_x_test = np.min(x_test, axis=0)
    # Generate new points for extrapolation above the maximum values in both dimensions

    x0 = np.linspace(min_x_test[0]-1, max_x_test[0]+1, num_points)
    x1 = np.linspace(min_x_test[1]-1, max_x_test[1]+1, num_points)
    above_below = np.array([[x, y] for x in x0 for y in [min_x_test[1]-1, max_x_test[1]+1]])
    left_right = np.array([[min_x_test[0]-1, y] for y in x1] + [[max_x_test[0]+1, y] for y in x1])
    x_expanded = np.concatenate((above_below, left_right), axis=0)

    extrap = LinearNDInterpolatorExtrapolator(x_test, z_values)
    z2 = extrap(x_expanded)

    x_test = np.concatenate((x_test, x_expanded), axis=0)
    z_values = np.concatenate((z_values, z2), axis=0)
    return x_test, z_values


def plot_2d_contour_lines(ax: plt.Axes, x_test: np.ndarray,
                          z_values: np.ndarray,
                          contour_lvls_z_values: np.ndarray,
                          plot_every_nth_contline: int,
                          cmap_name: str = 'viridis',
                          # truncate colornorm for contour lines to get higher contrast
                          shortend_range_factor: float = 0.499,
                          colornorm_truncation_offset: float = 0.0,
                          use_binary_cmap: bool = True,  # if True, use binary colormap for the contour lines
                          # if False, use inverted (not reversed) colormap for the contour lines
                          use_extrapolate: bool = True,  # if True, use extrapolation for the contour lines
                          # used to remove artefacts at the border of the plot

                          ):
    # get current colormap and norm and use higher contrast colormap for the contour lines

    if use_binary_cmap:
        cmap = plt.get_cmap('binary')
        # cmap = plt.get_cmap(cmap_name).reversed()
        num_range = np.max(z_values) - np.min(z_values)

        shortend_range_factor = shortend_range_factor*num_range
    else:
        cmap = invert_colormap(plt.get_cmap(cmap_name))
        shortend_range_factor = 0

    # truncate the colormap to the range of the data
    truncated_min = np.min(z_values) + colornorm_truncation_offset + shortend_range_factor
    truncated_max = np.max(z_values) + colornorm_truncation_offset - shortend_range_factor
    norm = plt.Normalize(vmin=truncated_min, vmax=truncated_max)

    x_test, z_values = extrapolate_2D_data(x_test, z_values, use_extrapolate=use_extrapolate)

    contour_line_object = ax.tricontour(x_test[:, 0], x_test[:, 1], z_values,
                                        levels=contour_lvls_z_values[::plot_every_nth_contline],
                                        cmap=cmap, norm=norm,
                                        #  colors='w',
                                        )  # black contour line

    change_contour_label_positions(ax, x_test, z_values,
                                   contour_line_object,
                                   contour_lvls_z_values,
                                   cmap, norm,
                                   plot_every_nth_contline=plot_every_nth_contline,)


def change_contour_label_positions(ax: plt.Axes, x_test, z_values,
                                   contour_line_object,
                                   contour_lvls_z_values,
                                   cmap, norm,
                                   threshold: float = 0.05,
                                   plot_every_nth_contline: int = 8,
                                   manual_pos: bool = False,  # if True, use manual positions for the labels
                                   ):

    # contour_lvls_z_values = contour_line_object.levels
    # get limits if they're automatic
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_range = xmax-xmin
    y_range = ymax-ymin

    # manuel positions by setting x values
    if manual_pos:
        num_contour_lines = len(contour_lvls_z_values[::plot_every_nth_contline])+1

        pos = 1/num_contour_lines * np.arange(0, num_contour_lines+1, 1) * x_range + xmin
        pos = pos[1:-1]
        pos = [(x, x) for x in pos]

        cls = ax.clabel(contour_line_object, inline=True,
                        #   inline_spacing=100,
                        #   fontsize=10
                        manual=pos,
                        )
        # get all labels with the same text
        text = [label.get_text() for label in cls]
        unique_text = np.unique(text)

        for unique_label in unique_text:
            # count how often the text occurs
            count = text.count(unique_label)
            if count < 2:
                continue
            # get the indices of the labels with the same text
            idx = [i for i, label in enumerate(cls) if label.get_text() == unique_label]
            # get the positions of the labels
            positions = [label.get_position() for label in cls]
            # get the index of the label which is nearest to the center
            center_x, center_y = (xmin+xmax)/2, (ymin+ymax)/2
            dist = []
            for i in idx:
                x, y = positions[i]
                dist.append((x-center_x)**2 + (y-center_y)**2)

            # get the index of the label which is nearest to the center
            keep_idx = idx[np.argmin(dist)]
            idx.remove(keep_idx)
            # remove the labels which are not nearest to the center
            cls = [v for i, v in enumerate(cls) if i not in idx]

    # initial labels
    cls = ax.clabel(contour_line_object, inline=True,)
    # check which labels are near a border
    keep_labels = []
    for label in cls:
        lx, ly = label.get_position()
        if (xmin+threshold*x_range < lx < xmax-threshold*x_range and
                ymin+threshold*y_range < ly < ymax-threshold*y_range):
            # inlier, redraw it later
            keep_labels.append((lx, ly))

    # delete the original lines, redraw manually the labels we want to keep
    # this will leave unlabelled full contour lines instead of overlapping labels

    for cline in contour_line_object.collections:
        cline.remove()
    for label in cls:
        label.remove()

    contour_line_object = ax.tricontour(x_test[:, 0], x_test[:, 1], z_values,
                                        levels=contour_lvls_z_values[::plot_every_nth_contline],
                                        cmap=cmap, norm=norm,
                                        #  colors='w',
                                        )  # black contour line
    cls = ax.clabel(contour_line_object, inline=True, manual=keep_labels)


def custom_safefig(fig: plt.Figure, file_path: str, dpi: int = 300,
                   transparent: bool = True,
                   export_raw_strings: bool = False,
                   use_latex=True, replace_dict={r'$': r'\$', r'U+2212': r'-'},
                   bbox_inches='tight',):
    fig.savefig(file_path, dpi=dpi, transparent=transparent, bbox_inches=bbox_inches,)

    if export_raw_strings:
        plt.rcParams.update({
            'svg.fonttype': 'none',
            'text.usetex': False,
        })
        file_path = file_path.replace('.svg', '_raw.svg')
        recursive_set_text(fig, replace_dict)
        fig.savefig(file_path,
                    # dpi=dpi,
                    transparent=transparent,
                    bbox_inches=bbox_inches,
                    )

        plt.rcParams.update({
            'text.usetex': True,
        })
    plt.close(fig)


def recursive_set_text(obj, replace_dict):
    # Check if the object has a 'set_text' method
    if hasattr(obj, 'set_text'):
        # Get the current text and replace it if needed
        current_text = obj.get_text()
        # new_text = replace_dict.get(current_text, current_text)
        # replace all occurrences of the replace_char with replace_with
        for replace_char, replace_with in replace_dict.items():
            current_text = current_text.replace(replace_char, replace_with)

        obj.set_text(current_text)

    # Recursively iterate through sub-elements
    if hasattr(obj, 'get_children'):
        for child in obj.get_children():
            recursive_set_text(child, replace_dict)

    return obj


def generate_gif(folder_name,
                 file_name: str = 'animation',
                 separation_str: str = '_',
                 duration: int = 50,  # duration of each frame of the multiframe gif, in milliseconds
                 image_types=('png', 'jpg', 'jpeg', ),
                 save_all: bool = True,
                 loop: int = 0,
                 verbose: bool = False,  # if True, print the file names of the images which are added to the gif
                 ):
    """ 
    Generate a GIF from a folder with images.
    Note that the images should be numerated and sorted like 1{separation_str}name.png, 2{separation_str}name.png...

    :param folder_name: Folder with images

    """
    img_paths = []

    all_files = os.listdir(folder_name)
    # if file ends with some image type, add it to the list
    image_files = [file for file in all_files if file.endswith(image_types)]

    # remove image type and get str before the separation string
    cutted_files = [file.split('.') for file in image_files]

    file_typ = cutted_files[0][1]

    # seperate file name from index with separation string
    cutted_filenames = [file[0].split(separation_str) for file in cutted_files]

    # check which part is the number
    number_idx = [idx for idx, name_part in enumerate(cutted_filenames[0]) if name_part.isdigit()][0]

    # get files which start with a number
    # image_files = [file for file in image_files if file.split(separation_str)[number_idx].isdigit()]

    # sort files according to their index
    sorted_files = sorted(cutted_filenames, key=lambda x: int(x[number_idx]))
    #  sorted_files = sorted(image_files, key=lambda x: int(x.split(separation_str)[0]))

    # get every plot in target folder (files should be numerated and sorted like 1.png, 2.png...)
    for file in sorted_files:
        if file_typ.endswith(image_types):
            file = file[0]+'_' + file[1] + '.' + file_typ
            img_paths.append(os.path.join(folder_name, file,))

    frames = []
    for i in img_paths:
        try:
            img = Image.open(i)
        except FileNotFoundError:
            if verbose:
                print("File '" + i + "' not found.")
            continue
        frames.append(img)
        if verbose:
            print(i + " added.")

    if len(frames) == 0:
        raise ValueError("No images found in folder.")

    # save GIF in folder
    frames[0].save(os.path.join(folder_name, file_name + '.gif'), format='GIF', append_images=frames[1:],
                   save_all=save_all, duration=duration, loop=loop)
