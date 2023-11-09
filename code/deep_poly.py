from torch import nn
import torch
from typing import Optional


class DeepPoly:
    def __init__(self,
                 lb: torch.Tensor,
                 ub: torch.Tensor,
                 lc: Optional[torch.Tensor],
                 uc: Optional[torch.Tensor]):
        self.lb = lb
        self.up = ub
        self.lc = lc
        self.uc = uc

    def propagate_linear(self, linear_layer: nn.Linear) -> 'DeepPoly':
        return self

    def propagate_conv2d(self, conv_layer: nn.Conv2d) -> 'DeepPoly':
        return self

    def propagate_relu(self, relu_layer: nn.ReLU) -> 'DeepPoly':
        return self

    def propagate_leakyrelu(self, leakyrelu_layer: nn.LeakyReLU) -> 'DeepPoly':
        return self

    def propagate_flatten(self, flatten_layer: nn.Flatten) -> 'DeepPoly':
        return self

    def check_postcondition(self, true_label: int) -> bool:
        return False


def construct_initial_shape(x: torch.Tensor, eps: float) -> 'DeepPoly':
    lb = x - eps
    lb.clamp_(min=0, max=1)

    ub = x + eps
    ub.clamp_(min=0, max=1)

    return DeepPoly(lb, ub, None, None)
