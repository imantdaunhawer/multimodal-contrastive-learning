"""
Definition of encoder architectures.
"""

from torch import nn
from typing import List, Union
from typing_extensions import Literal


def get_mlp(n_in: int, n_out: int,
            layers: List[int],
            layer_normalization: Union[None, Literal["bn"], Literal["gn"]] = None,
            act_inf_param=0.01):
    """
    Creates an MLP.

    This code originates from the following projects:
    - https://github.com/brendel-group/cl-ica
    - https://github.com/ysharma1126/ssl_identifiability

    Args:
        n_in: Dimensionality of the input data
        n_out: Dimensionality of the output data
        layers: Number of neurons for each hidden layer
        layer_normalization: Normalization for each hidden layer.
            Possible values: bn (batch norm), gn (group norm), None
    """
    modules: List[nn.Module] = []

    def add_module(n_layer_in: int, n_layer_out: int, last_layer: bool = False):
        modules.append(nn.Linear(n_layer_in, n_layer_out))
        # perform normalization & activation not in last layer
        if not last_layer:
            if layer_normalization == "bn":
                modules.append(nn.BatchNorm1d(n_layer_out))
            elif layer_normalization == "gn":
                modules.append(nn.GroupNorm(1, n_layer_out))
            modules.append(nn.LeakyReLU(negative_slope=act_inf_param))

        return n_layer_out

    if len(layers) > 0:
        n_out_last_layer = n_in
    else:
        assert n_in == n_out, "Network with no layers must have matching n_in and n_out"
        modules.append(layers.Lambda(lambda x: x))

    layers.append(n_out)

    for i, l in enumerate(layers):
        n_out_last_layer = add_module(n_out_last_layer, l, i == len(layers)-1)

    return nn.Sequential(*modules)


class TextEncoder2D(nn.Module):
    """2D-ConvNet to encode text data."""

    def __init__(self, input_size, output_size, sequence_length,
                 embedding_dim=128, fbase=25):
        super(TextEncoder2D, self).__init__()
        if sequence_length < 24 or sequence_length > 31:
            raise ValueError(
                "TextEncoder2D expects sequence_length between 24 and 31")
        self.fbase = fbase
        self.embedding = nn.Linear(input_size, embedding_dim)
        self.convnet = nn.Sequential(
            # input size: 1 x sequence_length x embedding_dim
            nn.Conv2d(1, fbase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fbase),
            nn.ReLU(True),
            nn.Conv2d(fbase, fbase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fbase * 2),
            nn.ReLU(True),
            nn.Conv2d(fbase * 2, fbase * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fbase * 4),
            nn.ReLU(True),
            # size: (fbase * 4) x 3 x 16
        )
        self.ldim = fbase * 4 * 3 * 16
        self.linear = nn.Linear(self.ldim, output_size)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.convnet(x)
        x = x.view(-1, self.ldim)
        x = self.linear(x)
        return x
