""" Matplotlib settings for the project. """""
import typing

import matplotlib.pyplot as plt


def init_settings(use_tex: bool = False,
                  scaling_factor: float = None,
                  unscaled_fontsize: float = None,
                  fig_width: float = None,
                  height_to_width_ratio: float = None,
                  ) -> None:
    """
    Initialize matplotlib LaTeX settings for better quality text and figures.

    This function updates the default parameters for matplotlib to improve
    the quality of text and figures that use LaTeX for typesetting.

    :param use_tex: Whether to use LaTeX for typesetting.
    :type use_tex: bool

    :return: None
    """

    if use_tex:
        print('\x1b[6;30;42m' + 'Using LaTeX for typesetting in matplotlib.' + '\x1b[0m')
        print('\x1b[1;37;41m' + 'Make sure to have LaTeX installed on your system ' +
              'or set use_tex=False to use matplotlib\'s default typesetting.' + '\x1b[0m')

    # some plot settings
    if scaling_factor is None:
        scaling_factor = 6  # 6
    if unscaled_fontsize is None:
        unscaled_fontsize = 8  # 10 # 8 # font size

    marker_scaling_factor = scaling_factor  # / 2
    axes_border_scaling_factor = scaling_factor / 2

    # a list of all rcParams can be found here: https://matplotlib.org/stable/api/matplotlib_configuration_api.html
    text_and_legend_font_size = scaling_factor*unscaled_fontsize  # font size

    # width, height in inches
    # https://www.ieee-ies.org/images/files/conferences/ieee-pages-and-margins-2016.pdf
    # default column width: 3.5in
    # default page width: 7.25in
    # figsize = (3.5, 3)
    if fig_width is None:
        fig_width = 3.5  # inches

    # fig_width = 1.0*505.89*1/72  # factor_of_textwidth*textwidth_in_pt*pt_to_inches
    if height_to_width_ratio is None:
        height_to_width_ratio = 0.75

    # figsize = (scaling_factor*fig_width, height_to_width_ratio*scaling_factor*fig_width)
    figsize = scale_figsize(fig_width=fig_width, scaling_factor=scaling_factor,
                            height_to_width_ratio=height_to_width_ratio)
    tick_font_size = text_and_legend_font_size

    plt.rcParams.update({
        # general settings
        'text.usetex': use_tex,
        'figure.figsize': figsize,
        'svg.fonttype': 'none',
        # font settings
        'font.size': text_and_legend_font_size,
        'font.family': 'serif',
        # axes settings
        'axes.linewidth': 1 * axes_border_scaling_factor,
        'axes.titlesize': text_and_legend_font_size,
        'axes.labelsize': text_and_legend_font_size,
        'axes.xmargin': 0,
        'axes.ymargin': 0,
        # x ticks
        'xtick.labelsize': tick_font_size,
        'xtick.major.width':   axes_border_scaling_factor,     # major tick width in points, default 0.8
        'xtick.major.size': axes_border_scaling_factor * plt.rcParams['xtick.major.size'],
        'xtick.minor.width':   0.8 * axes_border_scaling_factor,     # minor tick width in points, default 0.6
        'xtick.minor.size': axes_border_scaling_factor * plt.rcParams['xtick.minor.size'],
        # y ticks
        'ytick.labelsize': tick_font_size,
        'ytick.major.width':   axes_border_scaling_factor,     # major tick width in points, default 0.8
        'ytick.major.size': axes_border_scaling_factor * plt.rcParams['ytick.major.size'],
        'ytick.minor.width':   0.8 * axes_border_scaling_factor,     # minor tick width in points, default 0.6
        'ytick.minor.size': axes_border_scaling_factor * plt.rcParams['ytick.minor.size'],
        # lines and markers
        'lines.markersize': 3/4 * marker_scaling_factor * plt.rcParams['lines.markersize'],
        'lines.linewidth': 1 * marker_scaling_factor,
        # legend settings
        'legend.fontsize': text_and_legend_font_size,
        'legend.columnspacing': 0.5,  # relative to fontsize
        'legend.labelspacing': 0.25,  # relative to fontsize
        # error bar settings
        'errorbar.capsize': 4 * marker_scaling_factor,  # error bar cap line in pixels
    })

    # set LaTeX preamble
    if use_tex:
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def scale_figsize(fig_width: float, scaling_factor: float = 1.0, height_to_width_ratio: float = 0.75) -> typing.List[float]:
    """
    Scale the figure size by a given scaling factor and width_to_height_ratio.

    :param fig_width: The original figure width.
    :type fig_width: float
    :param scaling_factor: The scaling factor to use.
    :type scaling_factor: float

    :return: The scaled figure size.
    :rtype: tuple
    """
    return [scaling_factor * fig_width, height_to_width_ratio * scaling_factor * fig_width]
