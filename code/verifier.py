import argparse
import torch

from networks import get_network
from utils.loading import parse_spec

from deep_poly import construct_initial_shape, check_postcondition, backsubstitute, check_bounds
from torch import nn

DEVICE = "cpu"


def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    """
    Analyze and verify NN with a DeepPoly approach.

    :param net: NN to verify
    :param inputs:  Input Data to NN for which it is to verify
    :param eps: Perturbation for which the NN should be verified
    :true_label:    Correct OutPut label, the NN should generate
    :return:    True if NN can be verified with perpetuation, False if not.
    """

    dp = construct_initial_shape(inputs, eps)
    prev_layer = None
    counter = 0
    for layer in net:
        if isinstance(layer, nn.Linear):
            dp = dp.propagate_linear(layer)
            counter +=1
        elif isinstance(layer, nn.ReLU):
            dp = dp.propagate_relu(layer)
            counter +=1
        elif isinstance(layer, nn.LeakyReLU):
            dp = dp.propagate_leakyrelu(layer)
            counter +=1
        elif isinstance(layer, nn.Conv2d):
            dp = dp.propagate_conv2d(layer)
            counter +=1
        elif isinstance(layer, nn.Flatten):
            dp = dp.propagate_flatten()
        else:
            raise NotImplementedError(f"Unsupported layer type: {type(layer)}")

        if isinstance(prev_layer, nn.ReLU) or isinstance(prev_layer, nn.LeakyReLU):
            backsubstitute(dp, counter)
            counter = 1
            
        prev_layer = layer

    dp = dp.propagate_final(true_label)
    counter += 1
    #backsubstitute(dp, counter)
    return check_postcondition(dp, true_label)


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
            "test_1",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
