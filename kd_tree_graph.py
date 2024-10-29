""" Visualize a KDTree as a binary tree using Graphviz. """

from dataclasses import dataclass, field
# from lib2to3.pytree import convert
import os
import re
import typing

import dot2tex
import graphviz
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import KDTree


import create_plots


@dataclass
class KDTreeSplitPlane:
    """ Dataclass to store information about a split plane in a KDTree.

    split_value: float - the value where the split is done
    split_along_dim: int - the dimension along which the split is done
    mins: np.ndarray - the minimum values of the region
    maxes: np.ndarray - the maximum values of the region
    depth: int - the depth of the split plane in the tree

    split_point: np.ndarray - the split point
    default value is mean of the region except for the split_along_dim
    in split_along_dim the value is set to split_value

    split_label: str - the label of the split plane
    default value is number of created split planes

    """
    split_value: float

    split_along_dim: int

    mins: np.ndarray
    maxes: np.ndarray
    depth: int

    split_point: np.ndarray = None
    split_label: str = None

    # static counter how many split planes are created
    split_plane_count: int = 0

    def __post_init__(self):
        """ Set the split label to number of created split planes if not set."""
        self.split_plane_count += 1

        if self.split_label is None:
            self.split_label = f'split_{self.split_plane_count}'

        # calculate split point if not set
        self.calculate_split_point()

    # get split point
    def calculate_split_point(self):
        """ Return the split point. If not set, calculate it and store it.

        split point is structures as follows:
        [mean per ax, mins per ax, max per ax, splitted dimension]"""

        if self.split_point is None:
            self.split_point = np.empty((1, self.mins.shape[0]))

            means = (self.mins + self.maxes) / 2
            # self.split_point[:, -(2 * self.mins.shape[0] + 1):-1] = np.concatenate(
            #     [self.mins.reshape(1, -1), self.maxes.reshape(1, -1)], axis=1)
            self.split_point[:, :self.mins.shape[0]] = means.reshape(1, -1)
            self.split_point[:, self.split_along_dim] = self.split_value
            # self.split_point[:, -1] = self.split_along_dim

        return self.split_point

    def calculate_split_plane(self):
        """ Calculate the split plane. 
        Use the split value and the split dimension to calculate the split plane.

        The other dimensions are set to their mins and maxes.
        These two points are used to create the split plane.
        """
        # empty array to store two points of the split plane
        # split_plane[0]= mins
        # split_plane[1]= maxes
        # split_plane[:, self.split_along_dim] = self.split_value
        split_plane = np.empty((2, self.mins.shape[0]))
        split_plane[0, :] = self.mins
        split_plane[1, :] = self.maxes
        split_plane[0, self.split_along_dim] = self.split_value
        split_plane[1, self.split_along_dim] = self.split_value
        return split_plane


@dataclass
class KDTreeSplitPlanes:
    """ Class to store multiple KDTreeSplitPlane objects."""
    split_planes: typing.List[KDTreeSplitPlane] = None

    def __post_init__(self):
        self.split_planes = []

    def add_split_plane(self,
                        split: float,
                        split_along_dim: int,
                        mins: np.ndarray,
                        maxes: np.ndarray,
                        depth: int,
                        label: str = None):
        """" Add a split plane to the list of split planes."""

        self.split_planes.append(KDTreeSplitPlane(split, split_along_dim, mins, maxes, depth, split_label=label))

    def get_split_points_labels(self) -> typing.Tuple[np.ndarray, list]:
        """ Return the split points and corrsponding labels.

        """
        split_label_points = np.array([split.split_point for split in self.split_planes]).squeeze()
        split_labels_list = [split.split_label for split in self.split_planes]

        return split_label_points, split_labels_list

    def create_gif_of_2d_input_graph_creation(self, mins: np.ndarray, maxes:
                                              np.ndarray, input_data: np.ndarray,
                                              plot_dir: str = '',
                                              duration: int = 500,
                                              image_format: str = 'png',
                                              axis_labels: tuple = (r'$x_1$', r'$x_2$'),
                                              plot_intermediate_steps: bool = True,
                                              dpi: int = 300,
                                              ):
        """ Create a gif of the graph creation."""

        if image_format in ['svg']:
            raise ValueError('SVG and PDF formats are not supported.')

        # throw error if not 2D data
        if input_data.shape[1] != 2:
            raise ValueError('Only 2D data is supported.')

        for idx, split in enumerate(self.split_planes, ):

            # multiply the index  to save an intermediate step
            save_idx = 2 * idx
            # plot the split planes from 0 to idx
            split_points = np.array([split.split_point for split in self.split_planes[:idx + 1]]
                                    ).squeeze().reshape(-1, 2)
            split_labels = [split.split_label for split in self.split_planes[:idx + 1]]

            # get split planes for the plot
            split_planes = [split.calculate_split_plane() for split in self.split_planes[:idx + 1]]

            fig, axes = plt.subplots(tight_layout=True)
            axes.set_xlim(mins[0], maxes[0])
            axes.set_ylim(mins[1], maxes[1])
            axes.set_xlabel(axis_labels[0])
            axes.set_ylabel(axis_labels[1])

            create_plots.set_major_ticks(axes)

            # color the split points with tab:orange
            # but highlight the newest split point idx with red and an other marker
            colors = ['tab:orange'] * split_points.shape[0]
            colors[idx] = 'tab:red'
            markers = ['o'] * split_points.shape[0]
            markers[idx] = 'X'
            # highlight the newest split line with dashed line
            line_style = ['-', ] * split_points.shape[0]
            line_style[idx] = '--'

            axes.plot(input_data[:, 0], input_data[:, 1], 'o', label='Data Points', alpha=0.25, )

            # get lendths of axes in current region
            x_len = split.maxes[0] - split.mins[0]
            y_len = split.maxes[1] - split.mins[1]

            if x_len > y_len:
                x_color = 'red'
                y_color = 'black'
            else:
                x_color = 'black'
                y_color = 'red'

            # annotate the length of the axes as arrows
            # arrows shift by 1/4 of the length
            shift_factor = 1/4
            shrink_factor = 0.005  # shrink the arrow length from both sides by 0.5%

            if plot_intermediate_steps:
                axes.annotate('', xy=(split.mins[0] + x_len, split.mins[1] + shift_factor * y_len),
                              xytext=(split.mins[0], split.mins[1] + shift_factor * y_len),
                              arrowprops=dict(arrowstyle='<->', facecolor=x_color,
                                              shrinkA=shrink_factor, shrinkB=shrink_factor))
                axes.annotate('', xy=(split.mins[0] + shift_factor * x_len, split.mins[1] + y_len),
                              xytext=(split.mins[0] + shift_factor * x_len, split.mins[1]),
                              arrowprops=dict(arrowstyle='<->', facecolor=y_color,
                                              shrinkA=shrink_factor, shrinkB=shrink_factor))

                # highlight larger length with red color

                axes.annotate(f'{x_len:0.2f}', (split.mins[0] + x_len/2,
                                                split.mins[1] + shift_factor * y_len), color=x_color)
                axes.annotate(f'{y_len:0.2f}', (split.mins[0] + shift_factor *
                                                x_len, split.mins[1] + (1-shift_factor) * y_len), color=y_color)

            # previous split points and split planes
            for i, txt in enumerate(split_labels):
                if i == idx:  # do not plot the newest split point and plane
                    continue
                axes.plot(split_points[i, 0], split_points[i, 1],
                          color=colors[i],
                          marker=markers[i], label='Split Points',)
                axes.plot(split_planes[i][:, 0], split_planes[i][:, 1], color=colors[i], linestyle=line_style[i])
                axes.annotate(txt, (split_points[i, 0], split_points[i, 1]),)

            if plot_intermediate_steps:
                plot_file = os.path.join(plot_dir, f'{save_idx}_split.{image_format}')
                fig.savefig(plot_file, dpi=dpi)
                plt.close(fig)

            axes.plot(split_points[idx, 0], split_points[idx, 1],
                      color=colors[idx], marker=markers[idx], label='Split Points',)
            axes.plot(split_planes[idx][:, 0], split_planes[idx][:, 1],
                      color=colors[idx], linestyle=line_style[idx])

            plot_file = os.path.join(plot_dir, f'{save_idx+1}_split.{image_format}')
            fig.savefig(plot_file, dpi=dpi)
            plt.close(fig)

        create_plots.generate_gif(plot_dir, separation_str='_', duration=duration)


@dataclass
class KDTreeGraph:
    """ Class to convert a KDTree to a Graphviz representation for visualization."""
    kdtree: KDTree

    dot_graph: graphviz.Digraph = None

    data: np.ndarray = None
    dim_data: int = None

    dim_label: str = r'x'
    ax_idx_starts_at: int = 1
    region_label: str = 'r'

    use_tex: bool = False

    split_label_starts_at: int = 65  # char code for 'A'

    split_labels_list: list = field(default_factory=list)
    leaf_labels_list: list = field(default_factory=list)
    split_planes: KDTreeSplitPlanes = None

    # graphviz attributes
    # https://graphviz.org/doc/info/attrs.html
    # node color is only visible if fillcolor is set and stye is set to filled
    split_node_attr: dict = field(default_factory=lambda: {'shape': 'box',
                                                           'newrank': 'true',  # rank and newrank do not work properly
                                                           'rank': 'max',
                                                           'style': 'rounded,filled',
                                                           'width': '0.0',
                                                           'height': '0.5',  # default 0.5 # min = 0.02
                                                           #    'margin': '0.0',
                                                           'color': 'black',  # color of the border
                                                           'fillcolor': 'white',  # color of the node
                                                           'fontsize': '14',  # default 14
                                                           })

    leaf_note_attr: dict = field(default_factory=lambda: {'shape': 'plaintext',
                                                          'rank': 'min',
                                                          'bgcolor': 'transparent',
                                                          #   'fillcolor': 'none',
                                                          'style': '',
                                                          'color': '',
                                                          #   'margin': '0',
                                                          'height': '0.2',
                                                          'width': '0.',
                                                          })
    graph_attr: dict = field(default_factory=lambda: {'rankdir': 'TB',
                                                      'bgcolor': 'transparent',
                                                      'ranksep': '0.2',  # default 0.5 # min = 0.02
                                                      'nodesep': '0.2',  # default 0.25 min = 0.02
                                                      })

    def __post_init__(self):
        self.dim_data = self.kdtree.m
        self.data = self.kdtree.data

    def kdtree_to_graphviz(self,
                           ):
        """
        Converts a SciPy KDTree to a Graphviz representation for visualization.

        Args:
        data: The data points used to build the KDTree.
        kdtree: The SciPy KDTree object.

        Returns:
        A Graphviz dot object representing the KDTree as a binary tree.
        """

        if not isinstance(self.kdtree, KDTree):
            raise ValueError('kdtree must be a scipy.spatial.KDTree object')

        # create a new directed graph
        dot = graphviz.Digraph(node_attr=self.split_node_attr,
                               graph_attr=self.graph_attr)

        # label leaf alphabetically
        # start with A
        split_label = self.split_label_starts_at
        current_label = None
        leaf_count = 0  # count leaf nodes
        split_planes = KDTreeSplitPlanes()  # store split planes in wrapper class

        leaf_node_id_list = []
        leaf_label_list = []
        split_labels_list = []

        # get string of leaf label
        def get_split_label():
            nonlocal split_label
            nonlocal current_label
            # label = chr(split_label)
            # split_label += 1
            # # char code for 'Z' is reached -> start with 'AA', 'AB', ..., 'ZZ'
            # # if ZZ is reached, start with AAA
            # return label

            if current_label is None:
                current_label = chr(self.split_label_starts_at)
            else:
                next_char = chr(ord(current_label[-1]) + 1)

                if next_char > 'Z':
                    next_char = chr(self.split_label_starts_at)

                # Handle rollover to next letter or increment label length
                if split_label > 90:
                    next_char = chr(self.split_label_starts_at)
                    current_label = next_char + str(split_label)
                else:
                    current_label = next_char

            split_label += 1
            return current_label

        def get_leaf_count():
            nonlocal leaf_count
            leaf_count += 1
            return leaf_count

        def recursive_build(node, depth, ):
            if node is None:
                return

            # Handle intermediate and leaf nodes separately
            if isinstance(node, KDTree.leafnode):  # Leaf node
                # Get data points associated with the node
                # data_points = data[node.idx]
                # label = f'Leaf Node: {data_points}'

                if self.use_tex:
                    leaf_label = r'$' + self.region_label + f'_{get_leaf_count()}' + r'$'
                else:
                    leaf_label = self.region_label + convert_to_subscript(get_leaf_count())

                label = leaf_label
                # place all leaf nodes at bottom
                leaf_label_list.append(leaf_label)
                dot.node(f'n{node}', label=f'{label}', _attributes=self.leaf_note_attr)
                leaf_node_id_list.append(f'n{node}')

                return

            if isinstance(node, KDTree.innernode):  # Intermediate node

                # add position to list
                # pos = np.empty((1, self.dim_data*3+2))

                mins, maxes = np.min(self.data[node._node.indices, :], axis=0), np.max(  # pylint: disable=W0212
                    self.data[node._node.indices, :], axis=0)  # pylint: disable=W0212

                # means = (mins + maxes) / 2
                # pos[:, -(2*self.dim_data+1):-1] = np.concatenate([mins.reshape(1, -1), maxes.reshape(1, -1)], axis=1)
                # pos[:, :self.dim_data] = means.reshape(1, -1)
                # pos[:, node.split_dim] = node.split
                # pos[:, -1] = node.split_dim

                # pos contains:
                # [mean per ax, mins per ax, max per ax, splitted dimension]
                # dims:
                # [0:self.dim_data] = mean
                # [self.dim_data:2*self.dim_data] = mins
                # [2*self.dim_data:3*self.dim_data] = maxes
                # than, mean is replaced by split value

                # e.g. for 2D data:
                # [mean_x1, mean_x2, min_x1, min_x2, max_x1, max_x2]

                # split_positions.append(pos)

                split_point_label = get_split_label()
                split_labels_list.append(split_point_label)

                split_planes.add_split_plane(node.split, node.split_dim, mins, maxes, depth, label=split_point_label)

                if self.use_tex:
                    split_dim_label = self.dim_label + f'_{node.split_dim + self.ax_idx_starts_at}'
                    split_dim_label = r'$' + split_dim_label + r'$'
                    label = f'\n{split_point_label}\n\n' + 'split ' + \
                        f'{split_dim_label}\n\nat: ' + r'$\\num{'+f' {node.split:0.2f}' + r'}$'
                else:
                    split_dim_label = self.dim_label + \
                        convert_to_subscript(node.split_dim + self.ax_idx_starts_at)

                    label = f'{split_point_label}\n' + 'split ' + \
                        f'{split_dim_label}\nat: ' + f' {node.split:0.2f}'

            else:
                raise ValueError('Unknown node type')

            # Add node to graph with data point info
            dot.node(f'n{node}', label=f'{label}', _attributes=self.split_node_attr)

            # Recursively build children for intermediate nodes
            left, right = node.less, node.greater
            if left is not None:
                dot.edge(f'n{node}', f'n{left}')
                recursive_build(left, depth + 1, )
            if right is not None:
                dot.edge(f'n{node}', f'n{right}')
                recursive_build(right, depth + 1, )

        # Start from the root node
        root = self.kdtree.tree
        recursive_build(root, 0, )

        self.dot_graph = dot
        self.split_planes = split_planes
        self.split_labels_list = split_labels_list
        self.leaf_labels_list = leaf_label_list

    def save_graph(self, file_name, directory='./', file_format='svg'):
        """ Save the graph to a file."""
        if self.dot_graph is None:
            raise ValueError('No graph to save. Call kdtree_to_graphviz first.')

        self.dot_graph.render(file_name, directory=directory, format=file_format)

    def save_graph_as_tex(self, directory: str,):
        """ not necessary for this project. usepackage{dot2texi} can be used in LaTeX."""

        # https://dot2tex.readthedocs.io/en/latest/module_guide.html#using-dot2tex-as-a-module
        texcode = dot2tex.dot2tex(self.dot_graph.source, format='tikz', codeonly=True, crop=True, pad='0.05')
        with open(os.path.join(directory, 'kdtree.tex'), 'w', encoding='utf-8') as f:
            f.write(texcode)

        # logstream = dot2tex.get_logstream()
        # print(logstream.getvalue())

    def print_graph_source(self):
        """ Print the graph source code."""
        if self.dot_graph is None:
            raise ValueError('No graph to print. Call kdtree_to_graphviz first.')

        print(self.dot_graph.source)

    def get_split_plains(self) -> KDTreeSplitPlanes:
        """ Return the split points and labels.

        split positions is structures as follows:
        [mean per ax, mins per ax, max per ax, splitted dimension]
        dims:
        [0:self.dim_data] = mean
        [self.dim_data:2*self.dim_data] = mins
        [2*self.dim_data:3*self.dim_data] = maxes
        than, mean is replaced by split value

        """
        if self.dot_graph is None:
            raise ValueError('No graph to print. Call kdtree_to_graphviz first.')

        return self.split_planes

    def get_leaf_labels(self) -> list:
        """ Return the leaf labels."""
        if self.dot_graph is None:
            raise ValueError('No graph to print. Call kdtree_to_graphviz first.')

        if self.use_tex:
            return self.leaf_labels_list

        # convert leaf labels to LaTeX format
        for idx, leaf_label in enumerate(self.leaf_labels_list):
            leaf_idx = re.sub(self.region_label, '', leaf_label)
            leaf_idx = convert_to_regular(leaf_idx)
            self.leaf_labels_list[idx] = r'$' + self.region_label + '_' + leaf_idx + r'$'

        return self.leaf_labels_list

        # convert unicode subscript part to _{x}
        # get unicode from string

        # leaf_labels = [convert_to_regular(label) for label in self.leaf_labels_list]


def convert_to_subscript(number: typing.Union[int, str]):
    """
    Converts a numeric string to its subscripted representation.

    Args:
        number: Numeric string or int to convert.

    Returns:
        str: Subscripted version of the input string.
    """
    assert isinstance(number, (int, str)), 'Input must be an integer or string.'
    number_str = str(number)
    # Mapping of digits to subscript equivalents
    subscript_mapping = {
        '0': '₀',
        '1': '₁',
        '2': '₂',
        '3': '₃',
        '4': '₄',
        '5': '₅',
        '6': '₆',
        '7': '₇',
        '8': '₈',
        '9': '₉'
    }

    # Convert each digit to its subscript equivalent
    subscripted_str = ''.join(subscript_mapping.get(char, char) for char in number_str)
    return subscripted_str


def convert_to_regular(subscripted_str):
    """
    Converts a subscripted string to its regular numeric representation.

    Args:
        subscripted_str (str): Subscripted string to convert.

    Returns:
        str: Regular numeric version of the input string.
    """
    # Mapping of subscripted digits to regular equivalents
    regular_mapping = {
        '₀': '0',
        '₁': '1',
        '₂': '2',
        '₃': '3',
        '₄': '4',
        '₅': '5',
        '₆': '6',
        '₇': '7',
        '₈': '8',
        '₉': '9'
    }

    # Convert each subscripted digit to its regular equivalent
    regular_str = ''.join(regular_mapping.get(char, char) for char in subscripted_str)
    return regular_str


def test_kd_graph():
    """ Test the KDTreeGraph class."""
    # Sample data
    np.random.seed(0)
    data = np.random.rand(20, 2)
    data = np.array(data)

    # Build KDTree
    kdtree = KDTree(data, leafsize=2, balanced_tree=True, compact_nodes=False)

    # Convert to Graphviz and render
    file_name = 'kdtree'
    directory = './'

    # dim_label = r'x'
    # region_label = r'r'

    # dot, split_points = kdtree_to_graphviz(data, kdtree,
    #                                        dim_label=dim_label,
    #                                        region_label=region_label,
    #                                        split_node_attr=node_attr,
    #                                        leaf_note_attr=leaf_note_attr,
    #                                        graph_attr=graph_attr)

    kd_graph = KDTreeGraph(kdtree)
    kd_graph.kdtree_to_graphviz()
    kd_graph.save_graph(file_name, directory=directory, file_format='svg')
    # kd_graph.print_graph_source()

    region_labels = kd_graph.get_leaf_labels()
    split_planes = kd_graph.get_split_plains()
    split_points, split_labels = split_planes.get_split_points_labels()

    print(f'Region labels: {region_labels}')
    print(f'Split points: {split_points}')
    print(f'Split labels: {split_labels}')


if __name__ == '__main__':
    test_kd_graph()
