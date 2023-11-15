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
        lb = []
        ub = []
        augmented_lb = torch.cat((torch.tensor([1]), self.lb), 0)
        augmented_ub = torch.cat((torch.tensor([1]), self.ub), 0)
        for row in lc:
            vec = torch.where(row > 0, augmented_lb, augmented_ub)
            lb.append(torch.dot(row, vec))
        for row in uc:
            vec = torch.where(row > 0, augmented_ub, augmented_lb)
            ub.append(torch.dot(row, vec))
        return DeepPoly(torch.tensor(lb), torch.tensor(ub), lc, uc, self)

    def propagate_conv2d(self, conv_layer: nn.Conv2d) -> 'DeepPoly':
        return self

    def propagate_relu(self, relu_layer: nn.ReLU) -> 'DeepPoly':
        return self

    def propagate_leakyrelu(self, leakyrelu_layer: nn.LeakyReLU) -> 'DeepPoly':
        return self

    def propagate_flatten(self) -> 'DeepPoly':
        lb = torch.flatten(self.lb)
        ub = torch.flatten(self.ub)
        lc = torch.flatten(self.lc)
        uc = torch.flatten(self.uc)
        return DeepPoly(lb, ub, lc, uc, self)

    def check_postcondition(self, true_label: int) -> bool:
        return False


def construct_initial_shape(x: torch.Tensor, eps: float) -> 'DeepPoly':
    lb = x - eps
    lb.clamp_(min=0, max=1)

    ub = x + eps
    ub.clamp_(min=0, max=1)

    return DeepPoly(lb, ub, lb, ub, None)
