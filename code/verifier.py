import argparse
import torch

from networks import get_network
from utils.loading import parse_spec

from deep_poly import (
    construct_initial_shape,
    check_postcondition,
    backsubstitute,
)
from torch import nn
import random
import numpy as np

DEVICE = "cpu"


def create_random_perturbation(temperature: float, n: int) -> torch.Tensor:
    perturbation = (torch.rand(n) * temperature * 2 - temperature)
    mask = torch.zeros(n)
    if random.random() < temperature:
        for _ in range(2):
            mask[random.randint(0, len(mask) - 1)] = 1
    return perturbation * mask


def validate(net, inputs, true_label, eps, temperature=0, old_perturbations=None):
    dp = construct_initial_shape(inputs, eps)

    counter = 0

    perturbations = {}

    for layer in net:

        if isinstance(layer, nn.Linear):
            dp = dp.propagate_linear(layer)
            counter += 1
        elif isinstance(layer, nn.ReLU):
            perturbation = create_random_perturbation(temperature, dp.ub.shape)
            if old_perturbations is not None:
                perturbation += old_perturbations[counter]
            perturbation = torch.clamp(perturbation, 0, 1)
            dp = dp.propagate_relu(layer, perturbation=perturbation)
            perturbations[counter] = perturbation
            counter += 1
        elif isinstance(layer, nn.LeakyReLU):
            perturbation = create_random_perturbation(temperature, dp.ub.shape)
            if old_perturbations is not None:
                perturbation += old_perturbations[counter]
            perturbation = torch.clamp(perturbation, 0, 1)
            dp = dp.propagate_leakyrelu(layer, perturbation=perturbation)
            perturbations[counter] = perturbation
            counter += 1
        elif isinstance(layer, nn.Conv2d):
            dp = dp.propagate_conv2d(layer)
            counter += 1
        elif isinstance(layer, nn.Flatten):
            dp = dp.propagate_flatten()
        else:
            raise NotImplementedError(f"Unsupported layer type: {type(layer)}")

        backsubstitute(dp, counter)  # Or do we need to backsubstitute back to the start??

    dp = dp.propagate_final(true_label)

    return *check_postcondition(dp, true_label), perturbations


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

    ok, loss, perturbations = validate(net, inputs, true_label, eps)

    if ok:
        return True
    else:
        # run simulated annealing
        initial_temperature = 1.0
        cooling_rate = 0.9998
        temperature = initial_temperature
        min_temperature = 0.1
        temp_graph = []
        loss_graph = []
        i = 0
        while True:
            i += 1
            ok, new_loss, new_perturbations = validate(net, inputs, true_label, eps,
                                                       temperature=temperature,
                                                       old_perturbations=perturbations)
            if ok:
                return True
            if (new_loss < loss) or (random.random() < (np.exp(-10 * (new_loss/loss) / temperature))):
                loss = new_loss
                perturbations = new_perturbations

            temp_graph.append(temperature)
            loss_graph.append(loss)

            temperature *= cooling_rate
            temperature = max(min_temperature, temperature)

    return False


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
