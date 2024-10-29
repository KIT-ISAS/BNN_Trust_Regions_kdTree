""" BallTreeRegions class to store the BallTree and the resulting regions in input space. """


import os
import typing

from dataclasses import dataclass, field
import graphviz
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import pygraphviz as pgv
from sklearn.neighbors import BallTree

import create_plots
import distance_measures
import distance_stat_wrapper
# import kd_tree_partioning


def print_tree_data_and_structure(tree: BallTree, ):
    """ print tree data and structure of Sklearn BallTree object."""

    tree_arrays = tree.get_arrays()
    print('BallTree arrays:')

    print('tree data:')
    print(tree_arrays[0])

    print('tree index data:')
    print(tree_arrays[1])

    print('tree node data and bounds:')
    print(tree_arrays[2])

    print('data structure of tree_arrays[2]')
    print("dtype=[('idx_start', ' < i8'), ('idx_end', ' < i8'), ('is_leaf', ' < i8'), ('radius', ' < f8')])")


@dataclass
class BallTreeRegions:
    """ Class to store the BallTree and the resulting regions in input space. """

    # init empty tree and empty tree structure
    tree: BallTree = None
    tree_idx_data: np.ndarray = None  # indices of the data points in the tree (reordered campared to input data)

    # start index (from reordered tree_idx_data) of the data points in the ball
    idx_in_ball_start: np.ndarray = None  # dimension n_balls x 1,
    # end index of the data points in the ball
    idx_in_ball_end: np.ndarray = None  # dimension n_balls x 1,
    # is leaf node or not (1 if leaf, 0 if not)
    ball_is_leaf: np.ndarray = None  # dimension n_balls x 1
    # radius of the balls
    radius_array: np.ndarray = None  # dimension n_balls x 1
    # center points of the balls are the mean of the data points in the ball
    center_points: np.ndarray = None  # dimensions n_balls x n_features

    # statistics per region (e.g. ANEES, MSE, ECE)
    stat_per_region: np.ndarray = None  # dimension n_balls x 1
    combined_stats_per_leaf: np.ndarray = None  # dimension n_leafs x 1

    def __init__(self, input_data: np.ndarray, leaf_size: int, ) -> None:

        # store input data and leaf size
        self.input_data = input_data
        self.leaf_size = leaf_size

        # build tree
        self._build_tree()
        self._extract_tree_structure()

    def _build_tree(self, ):
        """ build tree from input data using sklearn BallTree."""
        self.tree = BallTree(self.input_data, leaf_size=self.leaf_size)

    def _extract_tree_structure(self, ):
        (_, self.tree_idx_data, self.idx_in_ball_start, self.idx_in_ball_end,
         self.ball_is_leaf, self.radius_array, self.center_points) = extract_nodes(
            self.tree)

    def print_tree_data_and_structure(self,):
        """ print tree data and structure."""
        print_tree_data_and_structure(self.tree)

    def plot_2d_ball_tree_regions(self, ax: plt.Axes = None,
                                  show_stats: bool = False,
                                  plot_parents: bool = False,
                                  plot_circles: bool = False,
                                  point_size: float = 4,
                                  cmap=None, norm=None) -> plt.Axes:
        """ plot 2D ball tree regions."""

        if show_stats and self.stat_per_region is None:
            raise ValueError('No statistics per region available. Please calculate statistics per region first.')

        if show_stats:
            statistics = self.stat_per_region
        else:
            statistics = None

        ax = plot_2d_ball_tree_regions(self.input_data, self.tree_idx_data, self.idx_in_ball_start,
                                       self.idx_in_ball_end,
                                       self.ball_is_leaf, self.radius_array, self.center_points,
                                       ax=ax,
                                       plot_parents=plot_parents,
                                       plot_circles=plot_circles,
                                       point_size=point_size,
                                       statisics=statistics, cmap=cmap, norm=norm)
        return ax

    def get_idx_in_leaf_balls(self, ) -> typing.List:
        """ get the indices per leaf ball."""
        idx_per_ball = []
        for idx, is_leaf in enumerate(self.ball_is_leaf):
            if is_leaf:
                idx_per_ball.append(self.tree_idx_data[self.idx_in_ball_start[idx]:self.idx_in_ball_end[idx]])

        return idx_per_ball

    def get_idx_per_ball(self, ) -> typing.List:
        """ get the indices per ball. (not only leaf balls)"""
        idx_per_ball = []
        for idx in range(len(self.ball_is_leaf)):
            idx_per_ball.append(self.tree_idx_data[self.idx_in_ball_start[idx]:self.idx_in_ball_end[idx]])

        return idx_per_ball

    def get_data_points_per_leaf_ball(self, ) -> typing.List:
        """ get the data points per leaf ball."""
        data_points_per_ball = []
        for idx, is_leaf in enumerate(self.ball_is_leaf):
            if is_leaf:
                data_points_per_ball.append(
                    self.input_data[self.tree_idx_data[self.idx_in_ball_start[idx]:self.idx_in_ball_end[idx]]])

        return data_points_per_ball

    def get_data_points_per_ball(self, ) -> typing.List:
        """ get the data points per ball. (not only leaf balls)"""
        data_points_per_ball = []
        for idx in range(len(self.ball_is_leaf)):
            data_points_per_ball.append(
                self.input_data[self.tree_idx_data[self.idx_in_ball_start[idx]:self.idx_in_ball_end[idx]]])

        return data_points_per_ball

    def set_stat_per_region(self, stat_per_region: np.ndarray):
        """ set the statistics per region."""
        self.stat_per_region = stat_per_region

    # def set_combined_stats_per_leaf(self, combined_stats_per_leaf: np.ndarray):

    def print_stat_per_region(self,):
        """ print the region ID, the number of data points per region,
        and the statistics per region in a pretty table."""

        data_points_per_ball = self.get_data_points_per_leaf_ball()

        if len(self.stat_per_region) != len(data_points_per_ball):
            data_points_per_ball = self.get_data_points_per_ball()

        # print as pretty table
        table = PrettyTable()
        table.field_names = ['region ID', '# Data Points', 'statistic']
        for idx, stat in enumerate(self.stat_per_region):
            table.add_row([idx, len(data_points_per_ball[idx]), stat])

        print(table)


def extract_nodes(tree: BallTree, ):
    """ extract the nodes of the tree.

    :param tree: sklearn BallTree object
    :type tree: BallTree

    :return: tree_data, tree_idx_data, idx_in_ball_start, idx_in_ball_end, ball_is_leaf, radius_array, center_points
    :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """

    tree_arrays = tree.get_arrays()

    tree_data = tree_arrays[0]
    tree_idx_data = tree_arrays[1]
    tree_node_data = tree_arrays[2]

    idx_in_ball_start = tree_node_data['idx_start']
    idx_in_ball_end = tree_node_data['idx_end']
    ball_is_leaf = tree_node_data['is_leaf']
    radius_array = tree_node_data['radius']

    # center points of the balls are the mean of the data points in the ball
    # dimensions n_balls x n_features
    center_points = np.zeros((len(ball_is_leaf), tree_data.shape[1]))
    for idx in range(len(ball_is_leaf)):
        center_points[idx] = np.mean(tree_data[tree_idx_data[idx_in_ball_start[idx]:idx_in_ball_end[idx]]], axis=0)

    return tree_data, tree_idx_data, idx_in_ball_start, idx_in_ball_end, ball_is_leaf, radius_array, center_points


@dataclass
class Node:
    """ Node class which is used to recursively build the tree."""
    start_idx: int  # start index of the data points in the ball
    end_idx: int  # end index of the data points in the ball
    is_leaf: bool  # is leaf node or not (1 if leaf, 0 if not)
    indices: typing.List[int]  # indices of the data points in the ball
    radius: float  # radius of the ball
    centroid: np.ndarray  # center point of the ball
    left: 'Node'  # left child node (if not leaf node) # 'Node' is a forward reference
    right: 'Node'  # right child node (if not leaf node) # 'Node' is a forward reference

    node_id: typing.ClassVar[int] = 0  # default node id

    parent: 'Node' = None  # parent node (if not root node) # 'Node' is a forward reference
    stat: float = None  # statistic per region (e.g. ANEES, MSE, ECE)
    combined_stat: float = None  # combined statistic per leaf node

    def __post_init__(self):
        """ Assign a unique node id by counting the number of instances of the class."""
        Node.node_id += 1
        self.node_id = Node.node_id


@dataclass
class CustomBallTree:
    """ Custom BallTree class to save the ball tree structure.

    The structure can be created from scratch or from a sklearn BallTree object.
    Note that the kd tree construction is used to build the tree (in both cases).

    """

    root: Node = None
    leaf_size: int = 40
    data_points: np.ndarray = None

    split_node_attr: dict = field(default_factory=lambda: {'shape': 'box',
                                                           'newrank': 'true',  # rank and newrank do not work properly
                                                           'rank': 'max',
                                                           'style': 'rounded,filled',
                                                           'width': '0.0',
                                                           'height': '0.5',  # default 0.5 # min = 0.02
                                                           #    'margin': '0.0',
                                                           'color': 'black',  # color of the border
                                                           'fillcolor': 'white',  # color of the node
                                                           'fontsize': '1',  # default 14
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
                                                      'nodesep': '0.02',  # default 0.25 min = 0.02
                                                      })

    def rebuild_from_sklearn_tree(self, ball_tree: BallTree):
        """ Rebuild the tree from a sklearn BallTree object.

        :param ball_tree: sklearn BallTree object
        :type ball_tree: BallTree

        :return: None"""
        tree_data, reordered_data_indices, tree_structure, _ = ball_tree.get_arrays()
        self.data_points = tree_data
        self.root = rebuild_tree(tree_structure, indices=reordered_data_indices, data=tree_data)
        self._add_parent_references()

    def build_tree(self, points, start_idx, end_idx, ) -> Node:
        """ Build the tree recursively using the kd tree construction algorithm.

        :param points: data points
        :type points: np.ndarray with shape (n_data_points, n_features)
        :param start_idx: start index of the data points in the ball
        :type start_idx: int
        :param end_idx: end index of the data points in the ball
        :type end_idx: int
        """

        # Base case: if not enough points, make a leaf node
        if end_idx - start_idx <= 2 * self.leaf_size:
            return Node(
                start_idx=start_idx,
                end_idx=end_idx,
                is_leaf=True,
                indices=points[start_idx:end_idx],
                radius=calculate_radius(points[start_idx:end_idx]),
                centroid=calculate_centroid(points[start_idx:end_idx]),
                left=None,
                right=None,)
            # start_idx, end_idx, True,

            #         self.calculate_radius(points[start_idx:end_idx]))

        # Choose the dimension with the greatest spread
        # dimensions = points.shape[1]
        spread = calculate_spread(points[start_idx:end_idx], use_variance=False)
        split_dim = np.argmax(spread)  # splitting dimension is the one with the greatest spread (as in kd tree)

        # Split the points into two sets
        indices = np.arange(start_idx, end_idx)
        sorted_indices = indices[np.argsort(points[indices, split_dim])]
        points[indices] = points[sorted_indices]
        median_idx = (end_idx - start_idx) // 2

        # Recurse on each subset
        left_child = self.build_tree(points, start_idx, start_idx + median_idx)
        right_child = self.build_tree(points, start_idx + median_idx, end_idx)

        # Create a new internal node and return it
        return Node(
            start_idx=start_idx,
            end_idx=end_idx,
            is_leaf=False,
            indices=points[start_idx:end_idx],
            radius=calculate_radius(points[start_idx:end_idx]),
            centroid=calculate_centroid(points[start_idx:end_idx]),
            left=left_child,
            right=right_child
        )

    def get_all_leaf_nodes(self, node: Node = None, leaf_nodes: list = None) -> typing.List[Node]:
        """ Get all leaf nodes of the tree. """

        if node is None:
            node = self.root
            leaf_nodes = []

        if node.is_leaf:
            leaf_nodes.append(node)
            return leaf_nodes

        leaf_nodes = self.get_all_leaf_nodes(node.left, leaf_nodes)
        leaf_nodes = self.get_all_leaf_nodes(node.right, leaf_nodes)

        return leaf_nodes

    def get_all_nodes(self, node: Node = None, all_nodes: list = None) -> typing.List[Node]:
        """ Get all nodes of the tree. """

        if node is None:
            node = self.root
            all_nodes = []

        all_nodes.append(node)

        if node.is_leaf:
            return all_nodes

        all_nodes = self.get_all_nodes(node.left, all_nodes)
        all_nodes = self.get_all_nodes(node.right, all_nodes)

        return all_nodes

    def get_stat_leaf_to_root_path(self, node: Node = None,
                                   stat_path: typing.List[float] = None,
                                   radius_path: typing.List[float] = None) -> typing.Tuple[typing.List, typing.List]:
        """ Get the statistics per leaf to root path. 

        Start at the leaf node and go up to the root node."""

        assert node is not None, 'Please provide a node to start the path.'

        if node.is_leaf:
            stat_path = [node.stat]
            radius_path = [node.radius]
        else:
            stat_path.append(node.stat)
            radius_path.append(node.radius)

        if node.parent is None:
            return stat_path, radius_path

        self.get_stat_leaf_to_root_path(node.parent, stat_path, radius_path)
        return stat_path, radius_path

    def calc_combined_stat_per_leaf(self, weight_fnc: callable = None):
        """ Calculate the combined statistics per leaf node. """

        if weight_fnc is None:
            def weight_fnc(radius: np.ndarray):
                return 1/radius

        leaf_nodes = self.get_all_leaf_nodes()
        for leaf in leaf_nodes:
            stat_path, radius_path = self.get_stat_leaf_to_root_path(leaf)
            weights = weight_fnc(np.array(radius_path))
            combined_stat = weights @ np.array(stat_path) * 1 / np.sum(weights)

            leaf.combined_stat = combined_stat

    def calc_stat_per_node(self, predictions: np.ndarray, output_data: np.ndarray,
                           method: str = 'anees', alpha: float = 0.01, m_bins: int = 1):
        """ Calculate statistics per region. All regions are used. """

        calc_stat_per_node(predictions, output_data, self.root, method=method, alpha=alpha, m_bins=m_bins)

    def fit(self, data: np.ndarray) -> None:
        """ Fit the tree to the data points.

        :param data: data points
        :type data: np.ndarray with shape (n_data_points, n_features)

        :return: None
        """
        self.data_points = data
        self.root = self.build_tree(data, 0, len(data))
        self._add_parent_references(self.root)

    def get_structure_array(self):
        """ Get the structure of the tree as an array (Same as sklearn BallTree.get_arrays()[2])"""
        return self._get_structure_array(self.root)

    def plot_2d_ball_tree_regions(self, ax: plt.Axes = None,
                                  show_stats: bool = False,  # show statistics per region
                                  show_combined_stat: bool = False,  # show combined statistics per leaf node
                                  plot_parents: bool = False,  # plot parent nodes
                                  plot_circles: bool = False,  # plot areas with filles circles
                                  alpha_transparency_circles: float = 1,  # transparency of the circles
                                  plot_points: bool = False,  # plot data points
                                  edgecolor_points: str = 'k',  # edge color of the data points
                                  alpha_transparency_points: float = 1,  # transparency of the data points
                                  point_size: float = 4,  # scatter point size of the data points
                                  edgecolor_circles: str = 'k',  # edge color of the circles
                                  line_width_circles: float = 0.5,  # line width of the circles
                                  cmap=None, norm=None) -> plt.Axes:
        """ plot 2D ball tree regions."""

        # get leaf nodes
        if plot_parents:
            nodes = self.get_all_nodes()
        else:
            nodes = self.get_all_leaf_nodes()

        for node in nodes:
            if show_stats:
                if show_combined_stat and node.is_leaf:
                    statistics = node.combined_stat
                else:
                    statistics = node.stat
            else:
                statistics = None
            if plot_circles:  # plot areas with filles circles
                # for plots with many circles not recommended, since many circles will overlap
                circle = plt.Circle(node.centroid, node.radius, edgecolor=edgecolor_circles,
                                    facecolor=cmap(norm(statistics)), linewidth=line_width_circles, alpha=alpha_transparency_circles)
                ax.add_artist(circle)
            if plot_points:
                # tree_data[tree_idx_data[idx_in_ball_start[idx_node]:idx_in_ball_end[idx_node]]]
                points = self.data_points[node.indices]
                ax.scatter(points[:, 0], points[:, 1], color=cmap(norm(statistics)),
                           s=point_size, alpha=alpha_transparency_points, edgecolor=edgecolor_points)
        return ax

    def create_gif(self, data: np.ndarray,
                   plot_dir='ball_tree_frames',
                   #    filename='balltree',
                   image_format='png',
                   dpi=300,
                   duration=1000,
                   axis_labels=(r'$x_1$', r'$x_2$'),
                   scatter_point_size: int = 4,
                   ) -> None:
        """ Create a gif of the tree structure."""

        # create folder
        os.makedirs(plot_dir, exist_ok=True)

        # add parent references to make it easier to plot the tree
        self._add_parent_references()

        # create frames
        frame_idx = 0
        create_frames(self.root, data, frame_idx, plot_dir, 'balltree',
                      axis_labels=axis_labels, dpi=dpi, image_format=image_format,
                      scatter_point_size=scatter_point_size)

        # create gif from single frames
        create_plots.generate_gif(plot_dir, separation_str='_', duration=duration)

    def create_tree_graph(self, node: Node = None, graph=None, use_latex: bool = True):
        """Create a graph from the Ball Tree.

        :param node: current node (default is root)
        :type node: Node
        :param graph: graph object (default is None)
        :type graph: networkx.Graph

        :return: graph object
        :rtype: networkx.Graph
        """
        if graph is None:
            # graph = nx.DiGraph()
            # graph = pgv.AGraph(directed=True)
            graph = graphviz.Digraph(node_attr=self.split_node_attr,
                                     graph_attr=self.graph_attr)
        if node is None:
            node = self.root

        def node_str(node):
            return f'{centerpoint_to_str(node.centroid, use_latex)}\n{radius_to_str(node.radius, use_latex)}'

        if isinstance(graph, (pgv.AGraph, nx.DiGraph)):
            graph.add_node(node_str(node))

        elif isinstance(graph, graphviz.Digraph):
            graph.node(f'{node.node_id}', label=f'{node_str(node)}', _attributes=self.split_node_attr)
        else:
            raise ValueError('Unknown graph type')

        if node.is_leaf:
            return graph

        # add edges to child nodes
        if isinstance(graph, (pgv.AGraph, nx.DiGraph)):
            graph.add_edge(node_str(node), node_str(node.left))
            graph.add_edge(node_str(node), node_str(node.right))
        elif isinstance(graph, graphviz.Digraph):
            graph.edge(f'{node.node_id}', f'{node.left.node_id}')
            graph.edge(f'{node.node_id}', f'{node.right.node_id}')
        else:
            raise ValueError('Unknown graph type')

        # create graph for left and right child nodes
        graph = self.create_tree_graph(node.left, graph, use_latex=use_latex)
        graph = self.create_tree_graph(node.right, graph, use_latex=use_latex)

        return graph

    def draw_tree_graph(self, filename: str = 'balltree', directory='./', file_format='svg', use_latex: bool = False):
        """Draw the graph of the Ball Tree.

        :param filename: filename
        :type filename: str
        :param directory: directory
        :type directory: str
        :param file_format: file format
        :type file_format: str
        :param use_latex: use latex for labels
        :type use_latex: bool

        """
        graph = self.create_tree_graph(use_latex=use_latex)

        if isinstance(graph, pgv.AGraph):
            # assert isinstance(graph, pgv.AGraph)
            graph.layout(prog='dot')  # pylint: disable=no-member
            graph.draw(f'{filename}.dot', prog='dot')    # pylint: disable=no-member
            # graph.draw(f'{filename}.svg', prog='dot', format='svg:cairo')    # pylint: disable=no-member

        elif isinstance(graph, nx.DiGraph):
            write_dot(graph, f'{filename}.dot')
            # folder = os.getcwd()
            # render the dot file to svg and safe it in the given folder
            os.system(f'dot -Tsvg {filename}.dot -o {directory}/{filename}.{file_format}')
        elif isinstance(graph, graphviz.Digraph):
            graph.render(filename=filename, format=file_format, directory=directory, )
            # modify svg and replace &amp; with &
            # filename = os.path.join(directory, f'{filename}.{file_format}')
            # with open(filename, 'r') as file:
            #     data = file.read()
            # data = data.replace('&amp;', '&')
            # with open(filename, 'w') as file:
            #     file.write(data)
        else:
            raise ValueError('Unknown graph type')

        print(f'Dot file {filename} created.')

    def create_latex_tree_graph(self, node: Node = None, graph: str = None, depth: int = 1, use_stat: bool = False):
        """ recursivly create LaTeX file with tikz tree structure """

        split_node = ''
        if graph is None:
            graph = ''

        if node is None:
            node = self.root
            depth = 1
        else:
            depth += 1

        def node_str(node: Node):
            if use_stat:
                stat = node.stat
                if isinstance(stat, np.ndarray):
                    stat = float(stat[0])
            else:
                stat = None
            return node_to_aligend_latex_equation(centroid=node.centroid, radius=node.radius, stat=stat)

        def tabs(n_tabs):
            return '\t' * n_tabs

        if node == self.root:
            graph += f'\\node[{split_node}] {{{node_str(node)}}}'
        else:
            graph += f'node[{split_node}] {{{node_str(node)}}}'

        if node.is_leaf:
            if not use_stat:
                return graph

            stat = node.combined_stat
            if isinstance(stat, np.ndarray):
                stat = float(stat[0])

            # add statistics after leaf node in tree with invisible edge
            graph += '\n' + tabs(depth) + 'child {node[draw=none, fill=none] {' + \
                f'{stats_to_str(stat)}' + '} edge from parent[dashed, -latex] }'
            return graph

        # add edges to child nodes
        graph += '\n'+tabs(depth) + 'child {'  # + f'{node_str(node.left)}' + '}\n'
        graph = self.create_latex_tree_graph(node.left, graph, depth=depth, use_stat=use_stat)
        graph += '\n' + tabs(depth) + '}'

        graph += '\n'+tabs(depth) + 'child {'  # + f'{node_str(node.right)}' + '}\n'
        graph = self.create_latex_tree_graph(node.right, graph, depth=depth, use_stat=use_stat)
        graph += '\n'+tabs(depth) + '}'

        return graph

    def get_tree_depth(self, node: Node = None, depth: int = 0) -> int:
        """ Get the depth of the tree."""

        if node is None:
            node = self.root
            depth = 1

        if node.is_leaf:
            return depth

        return max(self.get_tree_depth(node.left, depth=depth+1),
                   self.get_tree_depth(node.right, depth=depth+1))

    def draw_latex_tree_graph(self, filename: str = 'balltree',
                              directory='',
                              use_stat: bool = False,
                              compile_pdf: bool = False):
        """ Draw the LaTeX tree graph """

        graph = self.create_latex_tree_graph(use_stat=use_stat)
        required_packages = self._get_required_packages()

        # create LaTeX file

        file_header_str = r'\documentclass{{standalone}}'+'\n'
        file_header_str += f'{required_packages}'+'\n'
        file_header_str += r'\begin{document}'+'\n'

        file_str = r'\begin{tikzpicture}'+'\n'
        file_str += r'[level distance=20mm, ' + self._get_sibling_distance() + ',' \
            + r'every node/.style={fill=white, draw=black, rectangle, rounded corners=2pt,inner sep=1pt, solid},' \
            + r' edge from parent/.style={solid,draw,-}]'+'\n'
        # file_str += r'\tikzstyle{split_node} = [rectangle, rounded corners=1pt, draw, fill=white, text centered]'+'\n'
        file_str += f'{graph}\n;%\n'
        file_str += r'\end{tikzpicture}'+'\n'

        file_footer_str = r'\end{document}'+'\n'

        # write tikz picture to file
        with open(os.path.join(directory, f'{filename}.tex'), 'w', encoding='utf-8') as file:
            file.write(file_str)

        # write LaTeX standalone file
        stanalone_name = f'{filename}_standalone'
        standalone_path = os.path.join(directory, stanalone_name)
        with open(standalone_path + '.tex', 'w', encoding='utf-8') as file:
            file.write(file_header_str+file_str+file_footer_str)

        # compile LaTeX standalone file to pdf
        if compile_pdf:
            try:
                os.system(f'pdflatex -output-directory={directory} {standalone_path}.tex')
                # delete aux files, log files, etc.
                os.system(f'rm {standalone_path}.aux {standalone_path}.log')  # {standalone_path}.synctex.gz')
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f'LaTeX Error: {e}')

    def get_combined_stat_per_leaf(self, ) -> np.ndarray:
        """ Get the combined statistics per leaf node. """

        leaf_nodes = self.get_all_leaf_nodes()
        return np.array([leaf.combined_stat for leaf in leaf_nodes])

    def _get_required_packages(self, required_packages: tuple = ('tikz',
                                                                 'amsmath',
                                                                 'amssymb',
                                                                 'mathtools',
                                                                 'siunitx',)
                               ) -> str:
        """ Get the required packages for the LaTeX tree graph."""

        return '\n'.join([f'\\usepackage{{{package}}}' for package in required_packages])

    def _get_sibling_distance(self, leaf_sibling_distance: float = 20.0) -> float:
        """ get the sibling distance for the LaTeX tree graph. 

        return string in form of
        level 1/.style={sibling distance=xmm},
        ...
        level {treedepth}}/.style={sibling distance={leaf_sibling_distance}mm},
        """
        # get tree depth
        treedepth = self.get_tree_depth()

        sibling_distance = ''
        for i in range(1, treedepth+1):
            sibling_distance += f'level {i}/.style={{sibling distance={ 2**(treedepth -i)*leaf_sibling_distance}mm}},\n'

        return sibling_distance

    def _get_structure_array(self, node):
        if node.is_leaf:
            return tuple([(node.start_idx, node.end_idx, 1, node.radius)])
        else:
            return np.concatenate([tuple([(node.start_idx, node.end_idx, 0, node.radius)]),
                                   self._get_structure_array(node.left),
                                   self._get_structure_array(node.right)])

    def _add_parent_references(self, node: Node = None) -> None:
        """ Add parent references to the nodes of the tree.

        :param node: node
        :type node: Node

        :return: None"""
        if node is None:
            node = self.root

        if node.is_leaf:
            return

        node.left.parent = node
        node.right.parent = node

        self._add_parent_references(node.left)
        self._add_parent_references(node.right)


def calc_stat_per_node(predictions: np.ndarray, output_data: np.ndarray,
                       node: Node, method: str = 'anees',
                       alpha: float = 0.01, m_bins: int = 1):
    """ Calculate statistics per region. All regions are used. """

    idx_per_region = [node.indices]
    stat, acccept_stat = calc_stats_per_region(predictions=predictions, output_data=output_data,
                                               idx_per_region=idx_per_region,
                                               method=method, alpha=alpha, m_bins=m_bins)
    # use binary decision per test statistic? E.g. Cauchy combination test
    _ = acccept_stat
    node.stat = float(stat)

    if node.is_leaf:
        return

    calc_stat_per_node(predictions, output_data, node.left, method=method, alpha=alpha, m_bins=m_bins)
    calc_stat_per_node(predictions, output_data, node.right, method=method, alpha=alpha, m_bins=m_bins)


def create_frames(node: Node, data: np.ndarray, frame_idx: int, folder_path: str, file_name: str,
                  image_format='png', dpi=200, axis_labels=(r'$x_1$', r'$x_2$'), scatter_point_size: int = 4) -> int:
    """ recursively create frames for the tree structure."""

    if node.is_leaf:
        frame_idx = create_frames_single_node(node, data, frame_idx, folder_path, file_name, dpi=dpi,
                                              image_format=image_format, axis_labels=axis_labels, scatter_point_size=scatter_point_size)
    else:
        frame_idx = create_frames_single_node(node, data, frame_idx, folder_path, file_name, dpi=dpi,
                                              image_format=image_format,  axis_labels=axis_labels, scatter_point_size=scatter_point_size)
        frame_idx = create_frames(node.left, data, frame_idx, folder_path, file_name, dpi=dpi,
                                  image_format=image_format, axis_labels=axis_labels)
        frame_idx = create_frames(node.right, data, frame_idx, folder_path, file_name, dpi=dpi,
                                  image_format=image_format, axis_labels=axis_labels)

    return frame_idx


def create_frames_single_node(node: Node, data: np.ndarray, idx: int, folder_path: str,
                              file_name: str, dpi: int = 200,
                              image_format='png',  axis_labels=(r'$x_1$', r'$x_2$'),
                              scatter_point_size: float = 4,
                              ) -> int:
    """ create frames for a single node in the tree structure.
    First frame is the node itself with data points and parent balls (thin lines),
    second frame is the split line,
    third frame is the splitted points and the resulting balls.

    :param node: node
    :type node: Node
    :param data: data points
    :type data: np.ndarray with shape (n_data_points, n_features)
    :param idx: index of the frame
    :type idx: int
    :param folder_path: folder path to save the frames
    :type folder_path: str
    :param file_name: file name
    :type file_name: str
    :param dpi: dpi of the image
    :type dpi: int
    :param image_format: image format
    :type image_format: str
    :param axis_labels: axis labels
    :type axis_labels: tuple

    :return: index counter +1 (next index)
    :rtype: int
    """

    if node.is_leaf:
        return idx

    # data in node
    data_in_node = data[node.indices]

    # create subplot
    fig, ax = plt.subplots(constrained_layout=False, )
    ax.set_aspect('equal')

    # set axis labels
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    # set limits
    ax.set_xlim([np.min(data[:, 0]), np.max(data[:, 0])])
    ax.set_ylim([np.min(data[:, 1]), np.max(data[:, 1])])

    # plot all data points transparent
    ax.scatter(data[:, 0], data[:, 1], alpha=0.1, s=scatter_point_size)

    # plot the data points in the node
    ax.scatter(data_in_node[:, 0], data_in_node[:, 1], color='tab:blue', s=scatter_point_size)

    # plot the center point of the node
    ax.plot(node.centroid[0], node.centroid[1], 'x', color='black')

    # plot the circle around the center point
    circle = plt.Circle(node.centroid, node.radius, color='black', fill=False,
                        linewidth=mpl.rcParams['lines.linewidth'])  # use default line width from rc settings
    ax.add_artist(circle)

    # plot the circle around all parent nodes (if not root node)
    parent = node.parent
    while parent is not None:
        circle = plt.Circle(parent.centroid, parent.radius, color='black', fill=False,
                            linewidth=1/2*mpl.rcParams['lines.linewidth'])  # use default line width from rc settings
        ax.add_artist(circle)
        parent = parent.parent

    # save the frame as png
    file_path = os.path.join(folder_path, f'{file_name}_{idx}.{image_format}')
    fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
    idx += 1

    # calculate the split of the node, if node is not a leaf node

    # calculate the split of the node
    split_dim = np.argmax(calculate_spread(data_in_node, use_variance=False))
    split_value = np.median(data_in_node[:, split_dim])

    # plot the split line
    if split_dim == 0:
        ax.axvline(split_value, color='black', linestyle='--')
    else:
        ax.axhline(split_value, color='black', linestyle='--')

    # save the frame as png
    file_path = os.path.join(folder_path, f'{file_name}_{idx}.{image_format}')
    fig.savefig(file_path, dpi=dpi, bbox_inches='tight')

    idx += 1

    # color the left and right points in differnt colors
    left_points = data_in_node[data_in_node[:, split_dim] < split_value]
    right_points = data_in_node[data_in_node[:, split_dim] >= split_value]

    # create subplot
    ax.scatter(left_points[:, 0], left_points[:, 1], color='tab:orange', s=scatter_point_size)
    ax.scatter(right_points[:, 0], right_points[:, 1], color='tab:green', s=scatter_point_size)

    # add the balls around the splitted points
    circle = plt.Circle(node.left.centroid, node.left.radius, color='tab:orange', fill=False,
                        linewidth=mpl.rcParams['lines.linewidth'])  # use default line width from rc settings
    ax.add_artist(circle)

    circle = plt.Circle(node.right.centroid, node.right.radius, color='tab:green', fill=False,
                        linewidth=mpl.rcParams['lines.linewidth'])  # use default line width from rc settings
    ax.add_artist(circle)

    # save the frame as png
    file_path = os.path.join(folder_path, f'{file_name}_{idx}.{image_format}')
    fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
    idx += 1

    plt.close(fig)
    return idx


def calculate_spread(points, use_variance=False) -> np.ndarray:
    """ Calculate the spread of the points in each dimension.

    :param points: data points
    :type points: np.ndarray with shape (n_data_points, n_features)
    :param use_variance: use variance instead of max-min spread
    :type use_variance: bool

    :return: spread of the points in each dimension
    :rtype: np.ndarray with shape (n_features,)"""
    # Calculate the spread of the points in each dimension
    if use_variance:
        return np.var(points, axis=0)

    return np.max(points, axis=0) - np.min(points, axis=0)


def calculate_radius(points) -> float:
    """ Calculate the radius of the ball containing all points.

    :param points: data points
    :type points: np.ndarray with shape (n_data_points, n_features)

    :return: radius of the ball containing all points
    :rtype: float
    """

    # radius is the maximum distance from the center to any point
    center = np.mean(points, axis=0)
    return np.max(np.linalg.norm(points - center, axis=1))


def calculate_centroid(points) -> np.ndarray:
    """ Calculate the centroid of the points.

    :param points: data points
    :type points: np.ndarray with shape (n_data_points, n_features)

    :return: centroid of the points
    :rtype: np.ndarray with shape (n_features,)
    """
    return np.mean(points, axis=0)


def compare_two_trees(tree1: CustomBallTree, tree2: CustomBallTree):
    """ check if two trees are equal."""
    return compare_nodes(tree1.root, tree2.root)


def compare_nodes(node1: Node, node2: Node):
    """ check if two nodes are equal."""

    # Check if nodes are both of the same type
    # (both leaf nodes or both internal nodes)
    if node1.is_leaf != node2.is_leaf:
        return False

    # Check if the start and end indices are the same
    if node1.start_idx != node2.start_idx or node1.end_idx != node2.end_idx:
        return False

    # Check if the radius is the same (within some tolerance)
    if not np.isclose(node1.radius, node2.radius, atol=1e-6):
        return False

    # if both nodes are leaf nodes, return True (no need to check the children)
    if node1.is_leaf:
        return True

    # else, recusively check the left and right children
    return compare_nodes(node1.left, node2.left) and compare_nodes(node1.right, node2.right)


def rebuild_tree(tree_structure, indices: typing.List[int], data=np.ndarray, idx=0) -> Node:
    """ Rebuild the tree from the tree structure.

    :param tree_structure: tree structure
    :type tree_structure: np.ndarray
    :param indices: indices of the data points in the tree
    :type indices: np.ndarray with shape (n_data_points,)
    :param data: data points
    :type data: np.ndarray with shape (n_data_points, n_features)
    :param idx: index of the node in the tree structure
    :type idx: int

    :return: node
    :rtype: Node
    """
    start_idx, end_idx, is_leaf, radius = tree_structure[idx]
    node = Node(start_idx=start_idx,
                end_idx=end_idx,
                is_leaf=is_leaf,
                indices=indices[start_idx:end_idx],
                radius=radius,
                centroid=np.mean(data[indices[start_idx:end_idx]], axis=0),
                left=None,
                right=None)
    # start_idx, end_idx, is_leaf, radius)
    if not is_leaf:
        # Find the indices of the left and right children
        left_child_idx = next((i for i, (s, e, _, _) in enumerate(
            tree_structure) if s == start_idx and e < end_idx), None)
        right_child_idx = next((i for i, (s, e, _, _) in enumerate(
            tree_structure) if s > start_idx and e == end_idx), None)
        if left_child_idx is not None:
            node.left = rebuild_tree(tree_structure=tree_structure, indices=indices, data=data, idx=left_child_idx)
        if right_child_idx is not None:
            node.right = rebuild_tree(tree_structure=tree_structure, indices=indices, data=data, idx=right_child_idx)
    return node


def plot_2d_ball_tree_regions(tree_data: np.ndarray,
                              tree_idx_data: np.ndarray,
                              idx_in_ball_start: np.ndarray,
                              idx_in_ball_end: np.ndarray,
                              ball_is_leaf: np.ndarray,
                              radius_array: np.ndarray,
                              center_points: np.ndarray,
                              ax: plt.Axes = None,
                              statisics: np.ndarray = None,
                              plot_parents: bool = False,
                              plot_circles: bool = False,
                              point_size: float = 4,
                              cmap=None,
                              norm=None) -> plt.Axes:
    """ plot 2D ball tree regions.

    :param tree_data: data points in the tree
    :type tree_data: np.ndarray with shape (n_data_points, n_features)
    :param tree_idx_data: indices of the data points in the tree
    :type tree_idx_data: np.ndarray with shape (n_data_points,)
    :param idx_in_ball_start: start index (from reordered tree_idx_data) of the data points in the ball
    :type idx_in_ball_start: np.ndarray with shape (n_balls, )
    :param idx_in_ball_end: end index of the data points in the ball
    :type idx_in_ball_end: np.ndarray with shape (n_balls, )
    :param ball_is_leaf: is leaf node or not (1 if leaf, 0 if not)
    :type ball_is_leaf: np.ndarray with shape (n_balls, )
    :param radius_array: radius of the balls
    :type radius_array: np.ndarray with shape (n_balls, )
    :param center_points: center points of the balls are the mean of the data points in the ball
    :type center_points: np.ndarray with shape (n_balls, n_features)
    :param statisics: statistics per region (e.g. ANEES, MSE, ECE)
    :type statisics: np.ndarray with shape (n_balls, )

    :return: None
    :rtype: None
    """
    # create figure and axis with constraind layout and equal aspect ratio
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True, )
        ax.set_aspect('equal')

    # get some dimensions
    # num_nodes = len(ball_is_leaf)
    dim_balls = center_points.shape[1]

    if statisics is None:
        # plot partitions/regions without color coded statistics
        # one color per leaf node
        num_leaf_nodes = sum(ball_is_leaf)
        cmap = plt.get_cmap('viridis', num_leaf_nodes)
        plot_raw_partitions(ax, tree_data, tree_idx_data, idx_in_ball_start,
                            idx_in_ball_end, ball_is_leaf, radius_array,
                            center_points, cmap=cmap)
    else:
        plot_partitions_with_stats(ax, tree_data, tree_idx_data, idx_in_ball_start,
                                   idx_in_ball_end, ball_is_leaf,
                                   radius_array, center_points, statisics,
                                   plot_parents=plot_parents,
                                   plot_circles=plot_circles,
                                   point_size=point_size,
                                   cmap=cmap, norm=norm)

    # set min and max of axis so that all circles are fully visible
    # broadcast radius array to shape (num_balls, dim_balls)
    mins = center_points - np.stack([radius_array] * dim_balls, axis=1)
    maxs = center_points + np.stack([radius_array] * dim_balls, axis=1)

    # extend scope by
    extend_scope = 0.1
    ax.set_xlim([np.min(mins[:, 0])-extend_scope, np.max(maxs[:, 0]) + extend_scope])
    ax.set_ylim([np.min(mins[:, 1])-extend_scope, np.max(maxs[:, 1]) + extend_scope])

    return ax


def plot_raw_partitions(
        ax: plt.Axes,
        tree_data: np.ndarray,
        tree_idx_data: np.ndarray,
        idx_in_ball_start: np.ndarray,
        idx_in_ball_end: np.ndarray,
        ball_is_leaf: np.ndarray,
        radius_array: np.ndarray,
        center_points: np.ndarray,
        cmap=None,  # color map for the leaf balls and data points
        marker_size_data_points: float = 1.5,
        marker_size_centroids: float = 5,) -> plt.Axes:
    """ plot partitions of the tree.

    :param tree_data: data points in the tree
    :type tree_data: np.ndarray with shape (n_data_points, n_features)
    :param tree_idx_data: indices of the data points in the tree
    :type tree_idx_data: np.ndarray with shape (n_data_points,)
    :param idx_in_ball_start: start index (from reordered tree_idx_data) of the data points in the ball
    :type idx_in_ball_start: np.ndarray with shape (n_balls, )
    :param idx_in_ball_end: end index of the data points in the ball
    :type idx_in_ball_end: np.ndarray with shape (n_balls, )
    :param ball_is_leaf: is leaf node or not (1 if leaf, 0 if not)
    :type ball_is_leaf: np.ndarray with shape (n_balls, )
    :param radius_array: radius of the balls
    :type radius_array: np.ndarray with shape (n_balls, )
    :param center_points: center points of the balls are the mean of the data points in the ball
    :type center_points: np.ndarray with shape (n_balls, n_features)

    :return: axes with partitions
    :rtype: plt.Axes
    """

    if cmap is None:
        cmap = plt.get_cmap('viridis', len(ball_is_leaf))

    num_nodes = len(ball_is_leaf)
    idx_leaf = 0
    for idx_node in range(num_nodes):

        # if ball is leaf, plot the points in the ball with color of the ball
        if ball_is_leaf[idx_node]:
            circle = plt.Circle(center_points[idx_node], radius_array[idx_node],
                                color=cmap(idx_leaf), fill=False,
                                linewidth=2*mpl.rcParams['lines.linewidth'])  # use default line width from rc settings
            ax.add_artist(circle)

            # plot centroids
            ax.plot(center_points[idx_node, 0], center_points[idx_node, 1],
                    'x', color=cmap(idx_leaf), markersize=marker_size_centroids)

            # plot data points
            points = tree_data[tree_idx_data[idx_in_ball_start[idx_node]:idx_in_ball_end[idx_node]]]
            ax.plot(points[:, 0], points[:, 1], 'o', color=cmap(idx_leaf),  markersize=marker_size_data_points)

            idx_leaf += 1

        # non-leaf nodes are plotted as circles with dashed thin line
        else:
            circle = plt.Circle(center_points[idx_node], radius_array[idx_node],
                                color='black', fill=False, linestyle='dashed',
                                linewidth=mpl.rcParams['lines.linewidth'])
            ax.add_artist(circle)

    return ax


def plot_partitions_with_stats(ax: plt.Axes,
                               tree_data: np.ndarray,
                               tree_idx_data: np.ndarray,
                               idx_in_ball_start: np.ndarray,
                               idx_in_ball_end: np.ndarray,
                               ball_is_leaf: np.ndarray,
                               radius_array: np.ndarray,
                               center_points: np.ndarray,
                               statistics: np.ndarray,
                               plot_parents: bool = False,
                               plot_circles: bool = False,
                               point_size: float = 4,
                               cmap=None,
                               norm=None) -> plt.Axes:
    """ plot partitions of the tree with statistics.

    :param tree_data: data points in the tree
    :type tree_data: np.ndarray with shape (n_data_points, n_features)
    :param tree_idx_data: indices of the data points in the tree
    :type tree_idx_data: np.ndarray with shape (n_data_points,)
    :param idx_in_ball_start: start index (from reordered tree_idx_data) of the data points in the ball
    :type idx_in_ball_start: np.ndarray with shape (n_balls, )
    :param idx_in_ball_end: end index of the data points in the ball
    :type idx_in_ball_end: np.ndarray with shape (n_balls, )
    :param ball_is_leaf: is leaf node or not (1 if leaf, 0 if not)
    :type ball_is_leaf: np.ndarray with shape (n_balls, )
    :param radius_array: radius of the balls
    :type radius_array: np.ndarray with shape (n_balls, )
    :param center_points: center points of the balls are the mean of the data points in the ball
    :type center_points: np.ndarray with shape (n_balls, n_features)
    :param statistics: statistics per region (e.g. ANEES, MSE, ECE)
    :type statistics: np.ndarray with shape (n_balls, )
    :param cmap: color map for the regions
    :type cmap: plt.cm

    :return: axes with partitions
    :rtype: plt.Axes
    """
    _ = tree_data
    _ = tree_idx_data
    _ = idx_in_ball_start
    _ = idx_in_ball_end

    if cmap is None:
        cmap = cmap = plt.get_cmap('viridis')

    if norm is None:
        norm = mpl.colors.Normalize(vmin=np.min(statistics),
                                    vmax=np.max(statistics))

    # plot all leafnodes
    # begin with last node (so that z order corresponds to the depth first strategy of the tree)
    idx = sum(ball_is_leaf) - 1
    zorder = 1.
    for idx_node in range(len(ball_is_leaf) - 1, -1, -1):  # args: start, stop, step
        # if ball is leaf, plot the points in the ball with color of the ball
        if ball_is_leaf[idx_node]:
            if plot_circles:  # plot areas with filles circles
                # for plots with many circles not recommended, since many circles will overlap
                circle = plt.Circle(center_points[idx_node], radius_array[idx_node], edgecolor='black',
                                    facecolor=cmap(norm(statistics[idx_node])), linewidth=0.5)
                ax.add_artist(circle)
            else:
                points = tree_data[tree_idx_data[idx_in_ball_start[idx_node]:idx_in_ball_end[idx_node]]]
                ax.scatter(points[:, 0], points[:, 1], color=cmap(norm(statistics[idx_node])), s=point_size, )

            idx -= 1

            #

        # non-leaf nodes are plotted as circles with dashed thin line
        else:
            if not plot_parents:
                continue

            circle = plt.Circle(center_points[idx_node], radius_array[idx_node],
                                edgecolor='black', facecolor=cmap(norm(statistics[idx_node])),
                                linestyle='dashed', linewidth=0.5, zorder=zorder)
            ax.add_artist(circle)

            zorder -= 0.01

    return ax


def calc_stats_per_region(
    predictions: typing.Union[np.ndarray, distance_measures.Gaussian],
    output_data: np.ndarray,
    idx_per_region: typing.List[int],
    method: str = 'anees',
    alpha: float = 0.01,  # significance level; only used for anees
    m_bins: int = 1,  # number of bins for ece
):
    """ calculate statistics per region.

    :param predictions: predictions of the model
    :type predictions: np.ndarray with shape (n_data_points, n_features)
    :param output_data: output data
    :type output_data: np.ndarray with shape (n_data_points, n_features)
    :param idx_per_region: indices per region
    :type idx_per_region: typing.List[int]
    :param method: method to calculate the statistics
    :type method: str
    :param alpha: significance level; only used for anees
    :type alpha: float
    :param m_bins: number of bins for ece
    :type m_bins: int

    :return: statistics per region
    :rtype: np.ndarray with shape (n_balls, )
    """

    return distance_stat_wrapper.distance_calculation(predictions,
                                                      output_data,
                                                      idx_per_region,
                                                      method=method,
                                                      alpha=alpha,
                                                      m_bins=m_bins,)


def centerpoint_to_str(vect: np.ndarray, latex: bool = False):
    """ Convert a numpy vector to a raw latex string."""

    if not latex:
        return f'{vect}'

    vec_length = len(vect)
    # use \bmatrix
    # vector_str = r'$\vx_\mathrm{c} =  \begin{bmatrix}'
    vector_str = r'\n\n\n\n\n\n\n$\\underline{x}_\\mathrm{c} =  \\begin{bsmallmatrix}'

    for idx, val in enumerate(vect):
        vector_str += r'\\num{' + f'{val:0.2f}' + r'}'
        if idx < vec_length - 1:
            vector_str += r'\\\\'

    vector_str += r'\\end{bsmallmatrix}$'
    vector_str += '\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
    return vector_str


def radius_to_str(radius: float, latex: bool = False):
    """ Convert a radius to a raw latex string."""

    if latex:
        return r'$r = \\num{' + f'{radius:0.2f}' + r'}$\n'

    return f'{radius}'


def node_to_aligend_latex_equation(centroid: np.ndarray, radius: float, stat: float = None):
    """ Convert a node to a aligned latex equation."""

    # use \bmatrix
    # vector_str = r'$\vx_\mathrm{c} =  \begin{bmatrix}'
    vector_str = r'$\begin{aligned}'
    vector_str += r'\underline{x}_\mathrm{c} &=  \begin{bsmallmatrix}'

    for idx, val in enumerate(centroid):
        vector_str += r'\num{' + f'{val:0.2f}' + r'}'
        if idx < len(centroid) - 1:
            vector_str += r'\\'

    vector_str += r'\end{bsmallmatrix} \\'
    vector_str += r'r &= \num{' + f'{radius:0.2f}' + r'}'
    if stat is not None:
        vector_str += r'\\'
        vector_str += r'T_\mathrm{N} &= \num{' + f'{stat:0.2f}' + r'}'
    vector_str += r'\end{aligned}$'

    return vector_str


def stats_to_str(stat: float, ):
    """ Convert a statistics to a raw latex string."""
    return r'$T = \num{' + f'{stat:0.2f}' + r'}$'


def _test_ball_tree():
    # Create a 2D numpy array
    np.random.seed(0)
    example_data = np.random.random((100, 2))  # 10 points in 2 dimensions

    # geneerate normal random data
    example_data = np.random.normal(0, 1, (200, 2))

    btr = BallTreeRegions(example_data, leaf_size=40)  # create BallTreeRegions object
    btr.print_tree_data_and_structure()  # print tree data and structure

    ax = btr.plot_2d_ball_tree_regions()  # plot 2D ball tree regions

    tree_structure = CustomBallTree()
    tree_structure.rebuild_from_sklearn_tree(btr.tree)  # rebuild tree from sklearn tree

    idx_per_region = btr.get_idx_in_leaf_balls()  # get the indices per leaf ball

    print(idx_per_region)

    fig, ax = plt.subplots(constrained_layout=True)
    btr.plot_2d_ball_tree_regions(ax=ax, show_stats=False)  # plot 2D ball tree regions with statistics

    fig.savefig('ball_tree_regions_test.svg')


def _test_custom_ball_tree():
    """ test the custom ball tree implementation.

    The custom created tree and the tree created from the sklearn tree are equal: True
    """
    leaf_size = 40

    np.random.seed(0)
    # example_data = np.random.normal(0, 1, (200, 2))
    example_data = np.random.random((1000, 2))

    cbt = CustomBallTree(leaf_size=leaf_size)

    cbt.fit(example_data)

    cbt1_structure = cbt.get_structure_array()

    cbt2 = CustomBallTree(leaf_size=leaf_size)
    sk_ball_tree = BallTree(example_data, leaf_size=leaf_size)
    cbt2.rebuild_from_sklearn_tree(sk_ball_tree)

    # cbt2_structure_sklearn = sk_ball_tree.get_arrays()[1]
    cbt2_structure = cbt2.get_structure_array()

    # print(cbt2_structure_sklearn)
    print(cbt2_structure)
    print(cbt1_structure)

    trees_are_equal = compare_two_trees(cbt, cbt2)
    print(f'The custom created tree and the tree created from the sklearn tree are equal: {trees_are_equal}')


if __name__ == '__main__':
    _test_ball_tree()

    _test_custom_ball_tree()
