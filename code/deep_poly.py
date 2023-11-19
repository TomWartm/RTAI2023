from torch import nn
import torch
from typing import Optional


class DeepPoly:
    """
    A DeepPoly representation of a single layer in a NN.
    The whole NN can be represented as a linked list of DeepPoly objects.
    """

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
        """
        Append new DeepPoly for a linear layer

        :param linear_layer:    Specific linear Layer of NN
        :return:    DeepPoly for that Layer
        """
        lc = torch.cat((torch.unsqueeze(linear_layer.bias, 1), linear_layer.weight), 1)
        uc = lc

        lb, ub = get_bounds_from_conditional(self.lb, self.ub, lc, uc)

        return DeepPoly(lb, ub, lc, uc, self)

    def propagate_conv2d(self, conv_layer: nn.Conv2d) -> 'DeepPoly':
        """
        NOT IMPLEMENTED YET
        Generate DeepPoly for a convolutional layer

        :param conv_layer:    Specific convolutional layer of NN
        :return:    DeepPoly for that Layer
        """
        return self

    def propagate_relu(self, relu_layer: nn.ReLU) -> 'DeepPoly':
        """
        NOT IMPLEMENTED YET
        Generate DeepPoly for a ReLU layer

        :param relu_layer:    Specific ReLU layer of NN
        :return:    DeepPoly for that Layer
        """
        # Naive implementation
        # x{i+1} <= \lambda_i * (x_i - l_{x_i}) ==> uc
        # x{i+1] >= 0   ==> lc
        # \lambda_i = u_{x_i} / (u_{x_i} - l_{x_i})

        # Naive 2:
        # x{i+1} <= \lambda_i * (x_i - l_{x_i}) ==> uc
        # x{i+1] >= x   ==> lc
        # \lambda_i = u_{x_i} / (u_{x_i} - l_{x_i})

        # Alpha Relaxation:
        # x{i+1} <= \lambda_i * (x_i - l_{x_i}) ==> uc
        # x{i+1] >= \alpha * x   ==> lc, with alpha learned
        # \lambda_i = u_{x_i} / (u_{x_i} - l_{x_i})

        #maybe need to catch case self.ub == self.lb
        slope = torch.divide(self.ub, (self.ub - self.lb))
        """Naive 1"""
        lc = torch.zeros((self.lb.shape[0], self.lb.shape[0]+1))
        """Naive 2"""
        #lc = torch.cat((torch.zeros((self.lb.shape[0], 1)), torch.eye(self.lb.shape[0])), 1)
        """Alpha Relaxation"""
        # alpha = ..., element in [0, 1] (for leaky ReLU in [negative_slope, 1])
        # lc = torch.cat((torch.zeros((self.lb.shape[0], 1)), torch.multiply(torch.eye(self.lb.shape[0]), alpha)), 1)

        uc = torch.cat((torch.unsqueeze(torch.multiply(self.lb, -slope), 1), torch.multiply(torch.eye(self.lb.shape[0]), slope)), 1)
        lb, ub = get_bounds_from_conditional(self.lb, self.ub, lc, uc)

        return DeepPoly(lb, ub, lc, uc, self)

    def propagate_leakyrelu(self, leakyrelu_layer: nn.LeakyReLU) -> 'DeepPoly':
        """
        NOT IMPLEMENTED YET
        Generate DeepPoly for a leaky ReLU layer

        :param leakyrelu_layer:    Specific leaky ReLU layer of NN
        :return:    DeepPoly for that Layer
        """
        # Naive:
        # x{i+1} <= \lambda_i * (x_i - l_{x_i}) ==> uc
        # x{i+1] >= x   ==> lc
        # \lambda_i = u_{x_i} / (u_{x_i} - l_{x_i})

        # Alpha Relaxation:
        # x{i+1} <= \lambda_i * (x_i - l_{x_i}) ==> uc
        # x{i+1] >= \alpha * x   ==> lc, with alpha learned
        # \lambda_i = u_{x_i} / (u_{x_i} - l_{x_i})

        # maybe need to catch case self.ub == self.lb
        slope = torch.divide(self.ub, (self.ub - self.lb))
        """Naive"""
        lc = torch.cat((torch.zeros((self.lb.shape[0], 1)), torch.multiply(torch.eye(self.lb.shape[0]), leakyrelu_layer.negative_slope)), 1)
        uc = torch.cat((torch.unsqueeze(torch.multiply(self.lb, -slope), 1), torch.multiply(torch.eye(self.lb.shape[0]), slope)), 1)
        lb, ub = get_bounds_from_conditional(self.lb, self.ub, lc, uc)

        return DeepPoly(lb, ub, lc, uc, self)

    def propagate_flatten(self) -> 'DeepPoly':
        """
        NOT IMPLEMENTED YET
        Generate DeepPoly for a flatten call on NN

        :return:    DeepPoly that adapts for topology change
        """
        return self


def check_postcondition(dp: 'DeepPoly', true_label: int) -> bool:
    """
    Do backsubstitution over all DeepPoly layers and verify bounds.

    :param dp:  Final DeepPoly of the NN
    :param true_label: expected output label of the NN
    :return:    True if NN can be verified with perpetuation, False if not.
    """

    # INITIALIZATION
    #
    #               | 0  1  0  0  .  .  .  0  |
    #               | 0  0  1  0  .  .  .  0  |
    #               | 0  0  0  1  .  .  .  0  |
    # uc  =  lc  =  | .  .  .  .  .        .  |
    #               | .  .  .  .     .     .  |
    #               | .  .  .  .        .  .  |
    #               | 0  0  0  0  .  .  .  1  |

    augment = torch.zeros(dp.uc.shape[0])
    augment = torch.unsqueeze(augment, 1)
    uc = torch.cat((augment, torch.eye(dp.uc.shape[0])), 1)
    lc = torch.cat((augment, torch.eye(dp.lc.shape[0])), 1)

    while dp.parent:

        # UPDATE RULE OF CONDITIONAL MATRICES
        #
        #                               |  1  0  0  .  .  .  0  |
        #                               |                       |
        # CONDITIONAL = CONDITIONAL  X  |   PARENT CONDITIONAL  |
        #                               |                       |

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
    """
    Generate DeepPoly to kickstart the calculation

    :param x:   Input Tensor to the NN
    :param eps: Perturbation for which the NN should be verified
    :return:    Initial DeepPoly which holds the min and max input with perturbation
    """
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
    """
    Checks if bounds of classification layer hold with perpetuation,
    Specifically: if lower bound of true label is bigger than upper bound of all others.

    :param lb:  Tensor that holds all lower bounds at final layer
    :param ub:  Tensor that holds all upper bounds at final layer
    :param index:   True label of input.
    :return:    True if verified, false else
    """
    assert (lb.ndim == 1) and (ub.ndim == 1)
    assert (index < len(lb)) and (index < len(ub))
    bounds = ub
    bounds[index] = lb[index] 
    return torch.argmax(bounds) == index


def get_bounds_from_conditional(
        lb: torch.tensor, ub: torch.tensor, lc: torch.tensor, uc: torch.tensor
) -> (torch.tensor, torch.tensor):
    """
    Calculates new bounds, based previous layer and the conditional bounds.

    :param lb:  Lower bounds from previous layer
    :param ub:  Upper bounds from previous layer
    :param lc:  Conditional lower bounds if this layer
    :param uc:  Conditional upper bounds if this layer
    :return:    Lower and upper of this layer
    """
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
