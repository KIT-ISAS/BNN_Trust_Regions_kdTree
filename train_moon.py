""" example of training a Bayesian Neural Network (BNN) using Pyro on the moon dataset."""

import os
import pickle
import typing

# from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.auto import trange


import numpy as np
import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
# from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam  # pylint: disable=no-name-in-module
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

import kalman_bnn


def load_data_set():
    """ Load the moon dataset and split it into training and testing sets."""
    # Load the moon dataset
    rand_state = 684
    n_samples = (2000, 1000)
    noise_std = 0.2  # standard deviation of Gaussian noise added to the data.
    test_size = 0.75
    # Note that noise is added to the input data, not to the output labels.
    x_raw, y = datasets.make_moons(noise=noise_std, n_samples=n_samples, random_state=rand_state)
    x_train, x_test, y_train, y_test = train_test_split(x_raw, y, test_size=test_size, random_state=rand_state)

    # Convert to PyTorch tensors
    x_train = torch.Tensor(x_train, )
    y_train = torch.Tensor(y_train, )
    x_test = torch.Tensor(x_test, )
    y_test = torch.Tensor(y_test, )

    return x_train, y_train, x_test, y_test


class BNN(PyroModule):
    """ Bayesian Neural Network"""

    def __init__(self, in_dim=1,
                 out_dim=1,
                 hid_dim=10,
                 n_hid_layers=1,
                 prior_scale=5.,
                 activation=nn.Sigmoid()):
        super().__init__()

        # self.activation = nn.Tanh()  # could also be ReLU or LeakyReLU
        self.activation = activation

        assert in_dim > 0 and out_dim > 0 and hid_dim > 0 and n_hid_layers > 0  # make sure the dimensions are valid

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim

        # Define the layer sizes and the PyroModule layer list
        self.layer_sizes = [in_dim] + n_hid_layers * [hid_dim] + [out_dim]
        layer_list = [PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in
                      range(1, len(self.layer_sizes))]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        for layer_idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(dist.Normal(0., prior_scale * np.sqrt(2 / self.layer_sizes[layer_idx])).expand(
                [self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0., prior_scale).expand([self.layer_sizes[layer_idx + 1]]).to_event(1))

    def forward(self, x, y=None):
        """ Forward pass through the network."""
        x = x.reshape(-1, self.in_dim)
        x = self.activation(self.layers[0](x))  # input --> hidden
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))  # hidden --> hidden
        mu = self.layers[-1](x).squeeze()  # hidden --> output
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1))  # infer the response noise

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)  # pylint: disable=unused-variable
        return mu

    def train_svi(self, x_train, y_train, num_epochs: int, svi: SVI):
        """ Train the BNN model using the Stochastic Variational Inference (SVI) method."""
        # num_epochs = 1000
        progress_bar = trange(num_epochs)
        for _ in progress_bar:
            loss = svi.step(x_train, y_train)
            progress_bar.set_postfix(loss=f"{loss / x_train.shape[0]:.3f}")


def train_svi(x_train: torch.Tensor,
              y_train: torch.Tensor,
              x_test: torch.Tensor,
              n_hidden_units_per_layer: int = 5,
              n_hidden_layers: int = 2,
              #   activation_hidden: str = 'relu',
              #   activation_output: str = 'sigmoid',
              num_svi_samples=500, num_epochs=1000):
    """ Train the BNN model using the Stochastic Variational Inference (SVI) method."""

    # get the output dimension
    out_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    model = BNN(in_dim=x_train.shape[1], out_dim=out_dim, hid_dim=n_hidden_units_per_layer,
                n_hid_layers=n_hidden_layers, prior_scale=1., activation=nn.Sigmoid())
    guide = AutoDiagonalNormal(model)

    retrain_model = True
    if retrain_model:
        # Create the network

        # Set Pyro random seed
        pyro.set_rng_seed(42)

        # Training
        optimizer = Adam({"lr": 0.01})
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        pyro.clear_param_store()
        model.train_svi(x_train, y_train, num_epochs=num_epochs, svi=svi)

        predictive = Predictive(model, guide=guide, num_samples=num_svi_samples)

        # Save the model and guide
        torch.save(predictive, 'svi_predictive_moon.pth')

        pyro.get_param_store().save('svi_params_moon.pth')

    else:
        # view current state
        pyro.clear_param_store()
        predictive = torch.load('svi_predictive_moon.pth')
        pyro.get_param_store().load('svi_params_moon.pth')
        pyro.module('model', model, update_module_params=True)

    # predictions model

    # Predictions
    predicted_train = predictive(x_train)['obs'].numpy()
    predicted_test = predictive(x_test)['obs'].numpy()

    return predicted_train, predicted_test


def main(save_to_dir: str = ''):
    """ Main function to train a BNN model on the moon dataset and plot the results."""
    #  Load the moon dataset
    x_train, y_train, x_test, y_test = load_data_set()

    # grid over input space
    min_data = np.array([-1.6, -1.1])
    max_data = np.array([2.5, 2.0])
    step_size = 0.1
    x_visualize = torch.Tensor(np.mgrid[min_data[0]:max_data[0]+step_size:step_size,
                               min_data[1]:max_data[1]+step_size:step_size].reshape(2, -1).T)

    num_epochs = 1

    use_kbnn = True
    use_vi = not use_kbnn

    if use_vi:
        # Train SVI model
        (predicted_train, predicted_test) = train_svi(x_train, y_train, x_test,
                                                      num_epochs=num_epochs,
                                                      n_hidden_units_per_layer=5,
                                                      n_hidden_layers=2,
                                                      #   activation_hidden='relu',
                                                      #   activation_output='sigmoid'
                                                      )
        predicted_train_mean = predicted_train.mean(axis=0)
        predicted_train_var = predicted_train.var(axis=0)

        predicted_test_mean = predicted_test.mean(axis=0)
        predicted_test_var = predicted_test.var(axis=0)

    if use_kbnn:
        # Train the KBNN model
        (predicted_train_mean, predicted_train_var,
         predicted_test_mean, predicted_test_var,
         predicted_visualize_mean, predicted_visualize_var,
         ) = train_kbnn(x_train, y_train,
                        x_test,
                        x_visualize=x_visualize,
                        num_epochs=num_epochs,
                        n_hidden_units_per_layer=5,
                        n_hidden_layers=2,
                        activation_hidden='relu',
                        activation_output='sigmoid')

    file_names = [
        # test data and predictions
        'y_test_pred_moon.pkl',
        'y_test_cov_moon.pkl',
        'X_test_data_moon.pkl',
        'y_test_data_moon.pkl',
        # train data and predictions
        'y_train_pred_moon.pkl',
        'y_train_cov_moon.pkl',
        'X_train_data_moon.pkl',
        'y_train_data_moon.pkl',
        # data and predicitions for visualization
        'y_visualize_pred_moon.pkl',
        'y_visualize_cov_moon.pkl',
        'x_visualize_data_moon.pkl',
    ]
    data = [
        # test data and predictions
        predicted_test_mean, predicted_test_var, x_test, y_test,
        # train data and predictions
        predicted_train_mean, predicted_train_var, x_train, y_train,
        # data and predicitions for visualization
        predicted_visualize_mean, predicted_visualize_var, x_visualize,
    ]

    # Save the results
    for file_name, array in zip(file_names, data):
        save_results(save_to_dir, array, file_name)

    # Plot the results
    moon_plot(x_train, y_train, x_test, y_test, predicted_train_mean, predicted_test_mean,
              x_visualize, predicted_visualize_mean,
              save_to_dir=save_to_dir)


def save_results(save_to_dir: str, array: typing.Union[np.ndarray, torch.Tensor], file_name: str):
    """ Save the results to a directory. as pickle file."""
    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)

    # if tensor convert to numpy array
    if isinstance(array, torch.Tensor):
        array = array.numpy()

    file_path = os.path.join(save_to_dir, file_name)

    # save to pickle file
    with open(file_path, 'wb', ) as file:
        pickle.dump(array, file)


def moon_plot(x_train, y_train, x_test, y_test, predicted_train_mean, predicted_test_mean,
              x_visualize, predicted_visualize_mean,
              save_to_dir=''):
    """ Plot the moon dataset and the predictions."""

    # plot the results
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(3, 2, 1)
    ax.set_aspect('equal')
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
    plt.title('Train data')

    ax2 = plt.subplot(3, 2, 2)
    ax2.set_aspect('equal')
    ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='k')
    plt.title('Test data')

    # plot predictions on the training data
    ax3 = plt.subplot(3, 2, 3)
    # use triangulated mesh to plot the predictions
    ax3.set_aspect('equal')
    ax3.tricontourf(x_train[:, 0], x_train[:, 1], predicted_train_mean, cmap='coolwarm')
    # plt.colorbar()

    # plot predictions on the test data
    ax4 = plt.subplot(3, 2, 4)
    ax4.set_aspect('equal')
    # use triangulated mesh to plot the predictions
    ax4.tricontourf(x_test[:, 0], x_test[:, 1], predicted_test_mean, cmap='coolwarm')
    # plt.colorbar()

    # highlight the wrong predicted points with different markes
    ax5 = plt.subplot(3, 2, 5)
    ax5.set_aspect('equal')
    ax5 = plot_classification_errors(ax5, predicted_train_mean, y_train, x_train)

    ax6 = plt.subplot(3, 2, 6)
    ax6.set_aspect('equal')
    cmap = 'coolwarm'
    ax6 = plot_mean_prediction(ax6, x_visualize, predicted_visualize_mean, cmap=cmap, )
    ax6 = plot_classification_errors(ax6, predicted_test_mean, y_test, x_test,
                                     edgecolors='w', scatter_edge_width=0.5, scatter_size=10)
    # add a color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    plt.gcf().colorbar(ax6.collections[0], ax=ax, cax=cax, orientation='vertical', cmap=cmap, label='predicted mean')

    file_path = os.path.join(save_to_dir, 'moon.svg')
    plt.savefig(file_path)

    _, ax = plt.subplots()
    plot_pred_mean_and_errors(ax, x_test, predicted_test_mean, y_test, x_visualize, predicted_visualize_mean,
                              edgecolors='w', cmap='coolwarm', scatter_edge_width=0.5, scatter_size=10)
    file_path = os.path.join(save_to_dir, 'moon_pred_mean_and_errors.svg')
    plt.savefig(file_path)


def train_kbnn(x_train: torch.Tensor,
               y_train: torch.Tensor,
               x_test: torch.Tensor,
               x_visualize: torch.Tensor = None,
               num_epochs: int = 1,
               n_hidden_units_per_layer: int = 5,
               n_hidden_layers: int = 2,
               activation_hidden: str = 'relu',
               activation_output: str = 'sigmoid') -> typing.Tuple[torch.Tensor, torch.Tensor,
                                                                   torch.Tensor, torch.Tensor]:
    """wrapper function to train a Kalman Bayesian Neural Network (KBNN),
      and return predictions for train and test data"""
    activation_type = activation_hidden
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    # construct the network
    n_hidden_vector = n_hidden_layers * [n_hidden_units_per_layer]
    hidden_activation = [activation_type for i in range(len(n_hidden_vector))]
    output_activation = [activation_output]
    activations = hidden_activation + output_activation
    layers = [input_dim] + n_hidden_vector + [output_dim]

    bnn = kalman_bnn.Bayesian_Network_torch(layers, activations,
                                            load_from_keras=False,
                                            normalise=False,  # layer normalization True or False
                                            measurement_noise=torch.Tensor([[0.]]),
                                            prior_cov=1.,
                                            noise=0.01,  # stability epsilon
                                            no_bias=False,
                                            rnd_seed=0
                                            )

    for _ in range(num_epochs):
        bnn.train(x_train, y_train)

    predicted_train_mean, predicted_train_var = bnn.predict(x_train)
    predicted_test_mean, predicted_test_var = bnn.predict(x_test)
    predicted_train_mean = predicted_train_mean.squeeze()
    predicted_test_mean = predicted_test_mean.squeeze()

    predicted_visualize_mean = None
    predicted_visualize_var = None
    if x_visualize is not None:
        predicted_visualize_mean, predicted_visualize_var = bnn.predict(x_visualize)
        predicted_visualize_mean = predicted_visualize_mean.squeeze()

    return (predicted_train_mean, predicted_train_var,
            predicted_test_mean, predicted_test_var,
            predicted_visualize_mean, predicted_visualize_var)


def plot_classification_errors(ax: plt.Axes,
                               predicted_class: np.ndarray,
                               true_class: np.ndarray,
                               input_data: np.ndarray,
                               edgecolors='w',
                               scatter_edge_width=0.5,
                               scatter_size=10):
    """ Plot the classification errors. The wrong predictions are marked with an X.
    The correct predictions are marked with circles.

    :param ax: The axes to plot the data on.
    :type ax: plt.Axes
    :param predicted_class: The predicted class labels.
    :type predicted_class: np.ndarray
    :param true_class: The true class labels.
    :type true_class: np.ndarray
    :param input_data: The input data.
    :type input_data: np.ndarray
    :return: The axes with the plotted data.
    """
    if isinstance(predicted_class, torch.Tensor):
        predicted_class = predicted_class.numpy()
    if isinstance(true_class, torch.Tensor):
        true_class = true_class.numpy()
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.numpy()

    wrong_test = np.abs(predicted_class - true_class) > 0.5

    ax.scatter(input_data[~wrong_test, 0], input_data[~wrong_test, 1],
               c=true_class[~wrong_test], cmap='coolwarm',
               edgecolors=edgecolors, s=scatter_size, linewidths=scatter_edge_width)

    ax.scatter(input_data[wrong_test, 0], input_data[wrong_test, 1],
               c=true_class[wrong_test],  cmap='coolwarm', edgecolors=edgecolors,
               marker='X', s=scatter_size, linewidths=scatter_edge_width,)

    return ax


def plot_mean_prediction(ax: plt.Axes,
                         x_in: np.ndarray,
                         predicted_mean: np.ndarray,
                         cmap='coolwarm',
                         norm=None):
    """ Plot the mean predictions. The mean predictions are plotted using a contour plot."""

    if norm is None:
        norm = mpl.colors.Normalize(vmin=predicted_mean.min(), vmax=predicted_mean.max())

    predicted_mean = predicted_mean.squeeze() if len(predicted_mean.shape) > 1 else predicted_mean
    ax.tricontourf(x_in[:, 0], x_in[:, 1], predicted_mean, cmap=cmap, norm=norm)
    return ax


def plot_pred_mean_and_errors(ax: plt.Axes,
                              # The code `x_in_test` is not doing anything as it is just a variable
                              # name. It is not assigned any value or used in any operation.
                              x_in_test: np.ndarray,
                              predicted_mean_test: np.ndarray,
                              true_class: np.ndarray,
                              x_visualize: np.ndarray,
                              predicted_visualize_mean: np.ndarray,

                              edgecolors='w',
                              cmap='coolwarm',
                              scatter_edge_width=0.5,
                              scatter_size=10,
                              colorbar_label=r'$\mu$',
                              norm=None,
                              add_colorbar=True):
    """ Plot the mean predictions and the correct/incorrect classified points."""
    # ax.set_aspect('equal')

    if norm is None:
        norm = mpl.colors.Normalize(vmin=0, vmax=predicted_mean_test.max())

    ax = plot_mean_prediction(ax, x_visualize, predicted_visualize_mean, cmap=cmap, )
    ax = plot_classification_errors(ax, predicted_mean_test, true_class, x_in_test,
                                    edgecolors=edgecolors,
                                    scatter_edge_width=scatter_edge_width,
                                    scatter_size=scatter_size)

    # return the axes if no color bar is needed
    if not add_colorbar:
        return ax

    # add a color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.gcf().colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                              ax=ax, cax=cax, orientation='vertical', label=colorbar_label)

    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    return ax


if __name__ == '__main__':
    main(save_to_dir=os.path.join('data', 'moon'))
