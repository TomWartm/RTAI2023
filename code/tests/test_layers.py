import torch
from torch import nn
import sys
import os
import pathlib
from itertools import product

sys.path.append(os.path.join(os.path.join(pathlib.Path(__file__).parent.absolute()), '..'))

from deep_poly import DeepPoly, construct_initial_shape


def test_linear():
    input_dimension_values = [2, 5, 10, 100, 1000]
    output_dimension_values = [3, 5, 10, 100, 1000]
    reps = 10
    for n, m in product(input_dimension_values, output_dimension_values):
        for _ in range(reps):
            x = torch.rand(n)
            dp = construct_initial_shape(torch.ones(n), 1)
            layer = nn.Linear(n, m)
            y = layer(x)
            dp = dp.propagate_linear(layer)
            assert y.shape == dp.lb.shape == dp.ub.shape
            assert all(y.ge(dp.lb)) and all(y.le(dp.ub))


def test_convolutional():
    assert True


def test_relu():
    assert True


def test_leakyrelu():
    assert True
