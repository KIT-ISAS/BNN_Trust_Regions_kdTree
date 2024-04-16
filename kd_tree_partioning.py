# pylint: disable=redefined-outer-name
""" KD Tree partitioning for visualization of regions in input space and their local errors"""

from dataclasses import dataclass
import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from prettytable import PrettyTable
from scipy.spatial import KDTree


@dataclass
class KDTreePartitions:
    """
    Represents the partitions of a KD-tree.

    :ivar mins: The minimum values for each dimension of the partitions.
    :vartype mins: List[np.ndarray]
    :ivar maxes: The maximum values for each dimension of the partitions.
    :vartype maxes: List[np.ndarray]
    :ivar indices: The indices of the data points within each partition.
    :vartype indices: List[np.ndarray]
    """
    mins: typing.List[np.ndarray]
    maxes: typing.List[np.ndarray]
    indices: typing.List[np.ndarray]


def get_nearest_neighbor(new_point, kdtree: KDTree) -> int:
    """
    Get the index of the nearest neighbor to the new point in the given KD tree.

    :param new_point: The new point for which to find the nearest neighbor.
    :type new_point: array-like

    :param kdtree: The KD tree to query.
    :type kdtree: KDTree

    :return: The index of the nearest point (indicating the region).
    :rtype: int
    """
    _, nearest_index = kdtree.query(new_point)
    return nearest_index


def get_partition_index(nearest_index: int, space_partitions: KDTreePartitions) -> int:
    """ Get the index of the partition containing the nearest neighbor.

    :param nearest_index: The index of the nearest neighbor.
    :type nearest_index: int
    :param space_partitions: The partitions of the KD tree.
    :type space_partitions: KDTreePartitions

    :return: The index of the partition containing the nearest neighbor.
    :rtype: int
    """
    for partition_idx, partition in enumerate(space_partitions.indices):
        if nearest_index in partition:
            return partition_idx
    raise ValueError('Nearest index not found in any partition')


def get_partition_vecorized(new_points: np.ndarray,  kdtree: KDTree,
                            space_partitions: KDTreePartitions) -> np.ndarray:
    """ Get the index of the partition containing the nearest neighbor.
    If point is outside of min or max range of the KD tree, partition index is -1.

    :param new_points: The new points for which to find the nearest neighbor. 
    Points have to be in the same dimension as the KD tree
    (N=number of points, D=number of dimensions).
    :type np.ndarray (N, D)
    :param kdtree: The KD tree to query.
    :type kdtree: KDTree
    :param space_partitions: The partitions of the KD tree.
    :type space_partitions: KDTreePartitions

    :return: The index of the partition containing the nearest neighbor.
    if point is outside of min or max range of the KD tree, partition index is -1
    :rtype: np.ndarray (N, ) of int

    """
    # new_points: (N, D)
    # N = number of new points
    # D = number of dimensions

    # Check if the new points are a single point or multiple points
    # If a single point, reshape to (1, D)
    if len(new_points.shape) == 1:
        new_points = new_points.reshape(1, -1)

    # Query the KD tree to find the nearest neighbors to the new points
    _, nearest_indices = kdtree.query(new_points)

    # Return the index of the partition for each new point
    partition_indices = np.zeros(new_points.shape[0], dtype=int)
    for partition_idx, partition in enumerate(space_partitions.indices):

        # Find the indices in nearest_indices that are in the current partition
        indices_in_partition = np.isin(nearest_indices, partition)

        # continue loop if no indices in the current partition
        if not np.any(indices_in_partition):
            continue

        # Set the partition index for the new points that are in the current partition
        partition_indices[indices_in_partition] = partition_idx

    # set partition index to -1 if point is outside of min or max range of the KD tree
    out_of_range = np.logical_or(new_points < space_partitions.mins.min(axis=0),
                                 new_points > space_partitions.maxes.max(axis=0))
    partition_indices[out_of_range.any(axis=1)] = -1
    return partition_indices


def get_space_partitions(kdtree: KDTree, mins, maxes) -> KDTreePartitions:
    """
    Get the space partitions of a KDTree within the specified range.

    :param kdtree: The KDTree object.
    :type kdtree: KDTree
    :param mins: The minimum values of the range for each dimension.
    :type mins: array-like
    :param maxes: The maximum values of the range for each dimension.
    :type maxes: array-like

    :return: An object containing the space partitions, 
    represented by the minimum and maximum values of each partition,
    and the indices of the data points within each partition.
    :rtype: KDTreePartitions
    """
    space_partitions_mins = []
    space_partitions_maxes = []
    data_indices_per_region = []
    stack = [(kdtree.tree, mins, maxes, list(range(len(kdtree.data))))]

    while stack:
        node, node_mins, node_maxes, indices = stack.pop()

        if isinstance(node, KDTree.leafnode):
            space_partitions_mins.append(node_mins)
            space_partitions_maxes.append(node_maxes)
            data_indices_per_region.append(indices)
        elif isinstance(node, KDTree.innernode):
            axis = node.split_dim
            value = node.split

            (left_mins, left_maxes,
             right_mins, right_maxes) = split_space(axis, value, node_mins, node_maxes)
            left_indices = [i for i in indices if kdtree.data[i, axis] <= value]
            right_indices = [i for i in indices if kdtree.data[i, axis] > value]

            stack.append((node.greater, right_mins, right_maxes, right_indices))
            stack.append((node.less, left_mins, left_maxes, left_indices))
        else:
            raise TypeError('Unknown node type')

    return KDTreePartitions(mins=np.array(space_partitions_mins),
                            maxes=np.array(space_partitions_maxes),
                            indices=data_indices_per_region)


def plot_new_points(new_points: np.ndarray, axes: plt.Axes,
                    space_partitions: KDTreePartitions, partition_idx: np.ndarray) -> plt.Axes:
    """
    Plot new point and highlight the region it belongs to.

    :param new_points: Array of new points to be plotted.
    :type new_points: np.ndarray
    :param axes: The axes object to plot on.
    :type axes: plt.Axes
    :param space_partitions: The space partitions object.
    :type space_partitions: KDTreePartitions
    :param partition_idx: Array of indices indicating the region of each new point.
    :type partition_idx: np.ndarray
    :return: The modified axes object.
    :rtype: plt.Axes
    """
    # Check if the new points are a single point or multiple points
    # If a single point, reshape to (1, D)
    if len(new_points.shape) == 1:
        new_points = new_points.reshape(1, -1)

    # Plot the new point
    axes.plot(new_points[:, 0], new_points[:, 1], 'o',
              markerfacecolor='None', markeredgecolor='black', markersize=10)

    # get only the region of the new point
    region_mins = np.array(space_partitions.mins)[partition_idx.tolist()]
    region_maxes = np.array(space_partitions.maxes)[partition_idx.tolist()]
    region_indices = None  # not needed for plotting the region
    single_partition = KDTreePartitions(mins=region_mins,
                                        maxes=region_maxes,
                                        indices=region_indices)

    # Plot the KD tree space partitions
    plot_partitions(axes, space_partitions=single_partition, facecolor='white', edgecolor='gray')

    return axes


def plot_kd_space(space_partitions: KDTreePartitions,
                  distance_per_partition: np.ndarray,
                  axes: plt.Axes,
                  accept_stat_per_region: np.ndarray = None,
                  norm: mpl.colors.Normalize = None,
                  extend: str = 'neither',
                  #   vmax: typing.Union[float, None] = None,
                  #   vmin: typing.Union[float, None] = None,
                  cmap: str = 'viridis',
                  distance_label: str = 'Distance',
                  # hatch pattern for rejected regions
                  hatch_type: str = 'x',
                  #   color_norm: str = 'two_slopes',
                  ) -> plt.Axes:
    """
    Plots the KD tree space partitions with color-coded distances.

    :param space_partitions: The space partitions of the KD tree.
    :type space_partitions: KDTreePartitions
    :param distance_per_partition: The distances associated with each partition.
    :type distance_per_partition: np.ndarray
    :param axes: The matplotlib axes to plot on.
    :type axes: plt.Axes
    :param distance_label: The label for the colorbar. Defaults to 'Distance'.
    :type distance_label: str
    :return: The matplotlib axes with the plot.
    :rtype: plt.Axes
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    else:
        pass
        # cmap = cmap

    fig = axes.get_figure()
    # set axis limits according to space partitions mins and maxes
    axes.set_xlim(space_partitions.mins[:, 0].min(), space_partitions.maxes[:, 0].max())
    axes.set_ylim(space_partitions.mins[:, 1].min(), space_partitions.maxes[:, 1].max())

    if norm is None:
        norm = mpl.colors.Normalize(vmin=np.min(distance_per_partition),
                                    vmax=np.max(distance_per_partition))
        extend = 'neither'
    # if vmin is None:
    #     vmin = np.min(distance_per_partition)
    # if vmax is None:
    #     vmax = np.max(distance_per_partition)

    # if vmax < np.max(distance_per_partition):
    #     extend = 'max'
    # else:
    #     extend = 'neither'

    # if color_norm == 'two_slopes':
    #     norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=1., vmax=vmax, )
    # elif color_norm == 'normal':
    #     norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # else:
    #     raise ValueError('Unknown color norm')

    color = cmap(norm(distance_per_partition))

    # Plot the KD tree space partitions
    plot_partitions(axes, space_partitions, facecolor=color,
                    accept_stat_per_region=accept_stat_per_region, hatch_type=hatch_type)

    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.15)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 ax=axes, label=distance_label, extend=extend)

    return axes


def print_distances_per_partition(space_partitions: KDTreePartitions,
                                  distance_per_partition: np.ndarray,
                                  accept_stat_per_region: np.ndarray = None,
                                  distance_label: str = 'Distance') -> None:
    """
    Prints the distances per partition along with the number of data points in each partition.

    :param space_partitions: The space partitions.
    :type space_partitions: KDTreePartitions
    :param distance_per_partition: The distances per partition.
    :type distance_per_partition: np.ndarray
    :param distance_label: The label for the distance column. Defaults to 'Distance'.
    :type distance_label: str
    :return: None
    """

    table = PrettyTable()
    table.float_format = '.5'
    if accept_stat_per_region is not None:
        table.field_names = ['Region', distance_label, '# Data Points', 'Accept']
    else:
        table.field_names = ['Region', distance_label, '# Data Points']
    for partition_idx, partition in enumerate(space_partitions.indices):
        if accept_stat_per_region is not None:
            table.add_row([partition_idx, distance_per_partition[partition_idx], len(partition),
                           accept_stat_per_region[partition_idx]])
        else:
            table.add_row([partition_idx, distance_per_partition[partition_idx], len(partition)])

    print(table)


def plot_partitions(axes: plt.Axes,
                    space_partitions: KDTreePartitions,
                    edgecolor='gray',
                    facecolor: typing.Union[np.ndarray, str] = 'None',
                    accept_stat_per_region: np.ndarray = None,
                    hatch_type: str = 'x',
                    region_labels: typing.List[str] = None,
                    ) -> plt.Axes:
    """
    Plot the space partitions of a KD tree on the given axes.

    :param axes: The axes on which to plot the partitions.
    :type axes: plt.Axes
    :param space_partitions: The space partitions of the KD tree.
    :type space_partitions: KDTreePartitions
    :param edgecolor: The color of the partition edges. Defaults to 'gray'.
    :type edgecolor: str, optional
    :param facecolor: The color of the partition faces.
    Can be a single color string or a list of color strings. Defaults to 'None'.
    :type facecolor: Union[np.ndarray, str], optional

    :return: The axes with the partitions plotted.
    :rtype: plt.Axes
    """
    num_partitions = space_partitions.mins.shape[0]

    # fill if facecolor is not 'None'
    fill = True

    # make sure facecolor is a list if not string
    if isinstance(facecolor, str):
        fill = facecolor != 'None'
        facecolor = [facecolor] * num_partitions

    if accept_stat_per_region is None:
        accept_stat_per_region = [True] * num_partitions

    hatch = [None] * num_partitions
    for idx, accept in enumerate(accept_stat_per_region):
        # check if accept is numpy bool false
        if accept == 0:
            hatch[idx] = hatch_type

    if region_labels is None:
        region_labels = [None] * num_partitions

    # Plot the KD tree space partitions
    for idx in range(num_partitions):
        rect = plt.Rectangle((space_partitions.mins[idx][0],
                              space_partitions.mins[idx][1]),
                             space_partitions.maxes[idx][0] - space_partitions.mins[idx][0],
                             space_partitions.maxes[idx][1] - space_partitions.mins[idx][1],
                             facecolor=facecolor[idx],
                             fill=fill,
                             # add hatch pattern if accept_stat_per_region is False
                             hatch=hatch[idx],
                             edgecolor=edgecolor, linestyle='-', label='Space Partition')

        axes.add_patch(rect)

        # add region label
        if region_labels[idx] is not None:
            axes.text((space_partitions.mins[idx][0] + space_partitions.maxes[idx][0]) / 2,
                      (space_partitions.mins[idx][1] + space_partitions.maxes[idx][1]) / 2,
                      region_labels[idx], ha='center', va='center')

    return axes


def split_space(axis: int,
                split_at_value: float,
                mins: np.ndarray,
                maxes: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the space along the given axis at the specified value.

    :param axis: The axis along which to split the space.
    :type axis: int
    :param split_at_value: The value at which to split the space.
    :type split_at_value: float
    :param mins: The minimum values of the space.
    :type mins: np.ndarray
    :param maxes: The maximum values of the space.
    :type maxes: np.ndarray

    :return: A tuple containing four arrays representing the split space.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    left_mins = np.copy(mins)
    left_maxes = np.copy(maxes)
    right_mins = np.copy(mins)
    right_maxes = np.copy(maxes)

    # Split the space along the given axis
    # left space is from mins to split_at_value
    # right space is from split_at_value to maxes
    left_maxes[axis] = np.minimum(split_at_value, maxes[axis])
    right_mins[axis] = np.maximum(split_at_value, mins[axis])

    return left_mins, left_maxes, right_mins, right_maxes


if __name__ == '__main__':

    # Example usage:
    np.random.seed(2536)
    N_POINTS = 1000
    LEAFSIZE = 60
    # Because the splits are balanced in size,
    # the previous level must have at more than LEAFSIZE points.
    # So the minimum size is LEAFSIZE/2, if you set the maximum to LEAFSIZE
    # (except if there are less than LEAFSIZE/2 data points total).

    # Example data points in 2D input space
    # min and max values for each dimension
    MIN_DATA = 0
    MAX_DATA = 1

    # create random data points in 2D input space
    input_data_points = np.random.rand(N_POINTS, 2) * (MAX_DATA - MIN_DATA) + MIN_DATA

    # true y = x1 + x2
    output_data = input_data_points[:, 0] + input_data_points[:, 1]
    # simulated predictions from a model
    # y_model = 0.9 * x1 + 1.1 * x2 + noise
    predictions = 0.9 * input_data_points[:, 0] \
        + 1.1 * input_data_points[:, 1] + np.random.randn(N_POINTS) * 0.0001

    # Build the KD tree
    kdtree = KDTree(input_data_points, leafsize=LEAFSIZE, compact_nodes=True,
                    copy_data=False, balanced_tree=True, boxsize=None)

    # Get the min and max values for each dimension
    mins = np.min(input_data_points, axis=0)
    maxes = np.max(input_data_points, axis=0)
    space_partitions = get_space_partitions(kdtree, mins, maxes)

    fig, axes = plt.subplots()
    # equal axis scaling
    axes.set_aspect('equal', 'box')
    axes.plot(input_data_points[:, 0], input_data_points[:, 1], 'o', label='Data Points',)
    plot_partitions(axes, space_partitions, facecolor='none', )
    axes.set_xlabel(r'$x_1$')
    axes.set_ylabel(r'$x_2$')
    fig.savefig('kd_regions_raw.svg')
    plt.close()

    # mse per region
    mse_per_partition = np.zeros(len(space_partitions.mins))
    for partition_idx, partition in enumerate(space_partitions.indices):
        mse_per_partition[partition_idx]\
            = np.mean((predictions[partition] - output_data[partition])**2)

    distance_per_partition = mse_per_partition
    print_distances_per_partition(space_partitions, distance_per_partition)

    fig, axes = plt.subplots()
    # equal axis scaling
    axes.set_aspect('equal', 'box')
    plot_kd_space(space_partitions, distance_per_partition, axes=axes, distance_label='MSE')

    axes.plot(input_data_points[:, 0], input_data_points[:, 1], 'o', label='Data Points',
              markerfacecolor='None', markeredgecolor='white', alpha=0.25)

    # set axis limits according to (artificial) data mins and maxes
    axes.set_xlim(MIN_DATA, MAX_DATA)
    axes.set_ylim(MIN_DATA, MAX_DATA)
    # set axis labels
    axes.set_xlabel(r'$x_1$')
    axes.set_ylabel(r'$x_2$')
    fig.savefig('kd_regions_dist.svg')
    plt.close()

    test_points = -np.random.randn(10, 2)

    belongs_to_partition = get_partition_vecorized(test_points, kdtree, space_partitions)

    # print(f'Test points: {test_points})')
    # print(f'belong to partition: {belongs_to_partition}')
    print('if partition index is -1, point is outside of min or max range of the KD tree')
    # print as pretty table
    table = PrettyTable()
    table.field_names = ['Test Point', 'Partition Index']
    for test_point, partition_idx in zip(test_points, belongs_to_partition):
        table.add_row([test_point, partition_idx])

    print(table)
