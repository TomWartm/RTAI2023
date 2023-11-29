import torch
from typing import Tuple


def parse_spec(path: str) -> Tuple[int, str, torch.Tensor, float]:
    """Returns label, dataset, image and epsilon from a spec file

    Args:
        path (str): Path to spec file

    Returns:
        Tuple[int, str, torch.Tensor, float]: Label, image and epsilon
    """

    # Get epsilon from filename
    eps = float(".".join(path.split("/")[-1].split("_")[-1].split(".")[:2]))
    # Get dataset from filename
    dataset = path.split("/")[-1].split("_")[1]


    if "mnist" in dataset:
        shape = (1, 28, 28)
    elif "cifar" in dataset:
        shape = (3, 32, 32)
    elif "test" in dataset:
        shape = (2, 1, 1)
    else:
        raise NotImplementedError

    with open(path, "r") as f:
        # First line is the label
        label = int(f.readline().strip())
        # Second line is the image
        image = [float(x) for x in f.readline().strip().split(",")]

    return label, dataset, torch.tensor(image).reshape(shape), eps
