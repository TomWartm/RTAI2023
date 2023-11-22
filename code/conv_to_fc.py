"""
Adaptet from: https://gist.github.com/vvolhejn/e265665c65d3df37e381316bf57b8421


The function `conv_to_fc` takes a `torch.nn.Conv2d` layer `conv`
and produces an equivalent `torch.nn.Linear` layer `fc`.
Specifically, this means that the following holds for `x` of a valid shape:
    torch.flatten(conv(x)) == fc(torch.flatten(x))
Or equivalently:
    conv(x) == fc(torch.flatten(x)).reshape(conv(x).shape)
allowing of course for some floating-point error.

The fc layer is computational more expensive and might not be suitable for very large matrices.
"""
from typing import Tuple

import itertools
import torch
import torch.nn as nn
import numpy as np


def conv_to_fc(conv: torch.nn.Conv2d, input_size: Tuple[int, int]) -> torch.nn.Linear:
    _, h, w = input_size

    # Formula from the Torch docs:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    output_size = [
        (input_size[i + 1] + 2 * conv.padding[i] - (conv.kernel_size[i] - 1) - 1)
        // conv.stride[i]
        + 1
        for i in [0, 1]
    ]

    in_shape = (conv.in_channels, h, w)
    out_shape = (conv.out_channels, output_size[0], output_size[1])

    fc = nn.Linear(in_features=np.product(in_shape), out_features=np.product(out_shape))
    fc.weight.data.fill_(0.0)
    fc.bias.data.fill_(0.0)

    # Bias 
    # i.e. Convolutional net has 3 output channels and and output size of 2x2 then
    # conv.bias = [a,b,c]  -> fc.bias = [a,a,a,a,b,b,b,b,c,c,c,c]
    fc.bias = nn.Parameter(conv.bias.repeat_interleave(output_size[0]*output_size[1]))


    # Weights 
    # Output coordinates
    for xo, yo in itertools.product(range(output_size[0]), range(output_size[1])):
        with torch.no_grad():
            # The upper-left corner of the weight matrix
            xi0 = -conv.padding[0] + conv.stride[0] * xo
            yi0 = -conv.padding[1] + conv.stride[1] * yo

            # Position within the filter
            for xd, yd in itertools.product(
                range(conv.kernel_size[0]), range(conv.kernel_size[1])
            ):
                # Output channel
                for co in range(conv.out_channels):
                    for ci in range(conv.in_channels):
                        # Make sure we are within the input image (and not in the padding)
                        if 0 <= xi0 + xd < w and 0 <= yi0 + yd < h:
                            cw = conv.weight[co, ci, xd, yd]
                            # Copy the weights from the conv layer to the fc layer
                            # Helpful explanation: https://www.arxiv-vanity.com/papers/1712.01252/

                            fc.weight[co * np.product(out_shape[1:]) +
                                xo * out_shape[2] +
                                yo,
                                ci * np.product(in_shape[1:]) +
                                (xi0 + xd) * in_shape[2] +
                                (yi0 + yd),
                            ] = cw

    return fc




def test_layer_conversion():
    for stride in [1,2,3]:
        for padding in [1,2,3]:
            for filter_size in [2,3,4]:
                img = torch.rand((2, 16, 16))
                conv = nn.Conv2d(2, 16, filter_size, stride=stride, padding=padding)
                fc = conv_to_fc(conv, img.shape)

                # check: FC(flatten(img)) == flatten(Conv(img))

                res1 = fc(img.reshape((-1))).reshape(conv(img).shape)
                res2 = conv(img)
                worst_error = (res1 - res2).max()

                print("Output shape", res2.shape, "Worst error: ", float(worst_error))
                assert worst_error <= 1.0e-6

    print("Layer conversion ok")


if __name__ == "__main__":
    test_layer_conversion()
