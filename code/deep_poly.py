from torch import nn
import torch
from typing import Optional


class DeepPoly:
    def __init__(self,
                 lb: torch.Tensor,
                 ub: torch.Tensor,
                 lc: torch.Tensor,
                 uc: torch.Tensor,
                 parent: Optional['DeepPoly']):
        self.lb = lb
        self.ub = ub
        self.lc = lc
        self.uc = uc
        self.parent = parent

    def propagate_linear(self, linear_layer: nn.Linear) -> 'DeepPoly':
        lc = torch.cat((torch.unsqueeze(linear_layer.bias, 1), linear_layer.weight), 1)
        uc = lc

        lb, ub = get_bounds_from_conditional(self.lb, self.ub, lc, uc)

        return DeepPoly(lb, ub, lc, uc, self)

    def propagate_conv2d(self, conv_layer: nn.Conv2d) -> 'DeepPoly':
        return self

    def propagate_relu(self, relu_layer: nn.ReLU) -> 'DeepPoly':
        return self

    def propagate_leakyrelu(self, leakyrelu_layer: nn.LeakyReLU) -> 'DeepPoly':
        return self

    def propagate_flatten(self) -> 'DeepPoly':
        return self


def check_postcondition(dp: 'DeepPoly', true_label: int) -> bool:
    augment = torch.zeros(dp.uc.shape[0])
    augment = torch.unsqueeze(augment, 1)
    uc = torch.cat((augment, torch.eye(dp.uc.shape[0])), 1)
    lc = torch.cat((augment, torch.eye(dp.lc.shape[0])), 1)

    while dp.parent:
        e = torch.zeros(dp.uc.shape)[1]
        e[0] = 1
        e = torch.unsqueeze(e, 0)
        augmented_uc = torch.cat((e, dp.uc), 0)
        augmented_lc = torch.cat((e, dp.lc), 0)
        uc = torch.matmul(uc, augmented_uc)
        lc = torch.matmul(lc, augmented_lc)

        lb, ub = get_bounds_from_conditional(dp.parent.lb, dp.parent.ub, lc, uc)

        if check_bounds(lb, ub, true_label):
            return True
        dp = dp.parent

    return False


def construct_initial_shape(x: torch.Tensor, eps: float) -> 'DeepPoly':
    lb = x - eps
    lb.clamp_(min=0, max=1)

    ub = x + eps
    ub.clamp_(min=0, max=1)

    ub = torch.flatten(ub)
    lb = torch.flatten(lb)

    lc = torch.cat((torch.unsqueeze(lb, 1), torch.zeros((len(lb), len(lb)))), 1)
    uc = torch.cat((torch.unsqueeze(ub, 1), torch.zeros((len(ub), len(ub)))), 1)

    return DeepPoly(lb, ub, lc, uc, None)


def check_bounds(lb: torch.tensor, ub: torch.tensor, index: int) -> bool:
    assert (lb.ndim == 1) and (ub.ndim == 1)
    assert (index < len(lb)) and (index < len(ub))
    bounds = ub
    bounds[index] = lb[index]
    return torch.argmax(bounds) == index


def get_bounds_from_conditional(
        lb: torch.tensor, ub: torch.tensor, lc: torch.tensor, uc: torch.tensor
) -> (torch.tensor, torch.tensor):
    augmented_lb = torch.cat((torch.tensor([1]), lb), 0)
    augmented_ub = torch.cat((torch.tensor([1]), ub), 0)
    new_lb = []
    new_ub = []
    for row in lc:
        vec = torch.where(row > 0, augmented_lb, augmented_ub)
        new_lb.append(torch.dot(row, vec))
    for row in uc:
        vec = torch.where(row > 0, augmented_ub, augmented_lb)
        new_ub.append(torch.dot(row, vec))

    return torch.tensor(new_lb), torch.tensor(new_ub)
