import os
import pathlib
import sys
from itertools import product

import torch
from torch import nn

sys.path.append(os.path.join(os.path.join(pathlib.Path(__file__).parent.absolute()), '..'))

from deep_poly import construct_initial_shape


def test_linear():
    input_dimension_values = [2, 5, 10, 100, 1000]
    output_dimension_values = [3, 5, 10, 100, 1000]
    reps = 10
    for n, m in product(input_dimension_values, output_dimension_values):
        for _ in range(reps):
            x = torch.rand(n)
            dp = construct_initial_shape(torch.ones(n), 1)
            layer = nn.Linear(n, m)
            x = layer(x)
            dp = dp.propagate_linear(layer)
            assert x.shape == dp.lb.shape == dp.ub.shape
            assert all(x.ge(dp.lb)) and all(x.le(dp.ub))


def test_convolutional():
    input_dimension_values = [(1, 5, 5), (1, 20, 20), (3, 20, 20)]
    reps = 10
    for c, m, n in input_dimension_values:
        for _ in range(reps):
            x = torch.rand((c, m, n))
            dp = construct_initial_shape(torch.ones(c, m, n), 1)
            layer = nn.Conv2d(c, 4, 3)
            x = layer(x).flatten()
            dp = dp.propagate_conv2d(layer)
            assert x.shape == dp.lb.shape == dp.ub.shape
            assert all(x.ge(dp.lb)) and all(x.le(dp.ub))


def test_relu():
    input_dimension_values = [2, 5, 10, 100, 1000]
    reps = 10
    for n in input_dimension_values:
        for _ in range(reps):
            x = torch.rand(n)
            dp = construct_initial_shape(torch.ones(n), 1)
            linear_layer = nn.Linear(n, n)
            relu_layer = nn.ReLU()
            x = linear_layer(x)
            x = relu_layer(x)
            dp = dp.propagate_linear(linear_layer)
            dp = dp.propagate_relu(relu_layer)
            assert x.shape == dp.lb.shape == dp.ub.shape
            assert all(x.ge(dp.lb)) and all(x.le(dp.ub))


def test_leakyrelu():
    input_dimension_values = [2, 5, 10, 100, 1000]
    alphas = [0.01, 0.1, 1, 5]
    reps = 10
    for alpha, n in product(alphas, input_dimension_values):
        for _ in range(reps):
            x = torch.rand(n)
            dp = construct_initial_shape(torch.ones(n), 1)
            linear_layer = nn.Linear(n, n)
            leaky_relu_layer = nn.LeakyReLU(negative_slope=alpha)
            x = linear_layer(x)
            x = leaky_relu_layer(x)
            dp = dp.propagate_linear(linear_layer)
            dp = dp.propagate_leakyrelu(leaky_relu_layer)
            assert x.shape == dp.lb.shape == dp.ub.shape
            assert all(x.ge(dp.lb)) and all(x.le(dp.ub))
