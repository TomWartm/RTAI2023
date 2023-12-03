from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from conv_to_fc import conv_to_fc



def get_2d_mask(x):
    # repeats x len(x)+1 times
    # tensor([0,1]) -> tensor([[0, 0, 0],
    #                          [1, 1, 1]])
    return torch.transpose(x.repeat(x.shape[0] + 1).reshape(x.shape[0] + 1, x.shape[0]), 0, 1)


class DeepPoly:
    """
    A DeepPoly representation of a single layer in a NN.
    The whole NN can be represented as a linked list of DeepPoly objects.
    """

    def __init__(
            self,
            lb: torch.Tensor,
            ub: torch.Tensor,
            lc: torch.Tensor,
            uc: torch.Tensor,
            parent: Optional["DeepPoly"]
    ):
        self.lb = lb
        self.ub = ub
        self.lc = lc
        self.uc = uc
        self.parent = parent

    def propagate_linear(self, linear_layer: nn.Linear) -> "DeepPoly":
        """
        Append new DeepPoly for a linear layer

        :param linear_layer:    Specific linear Layer of NN
        :return:    DeepPoly for that Layer
        """
        lc = torch.cat((torch.unsqueeze(linear_layer.bias, 1), linear_layer.weight), 1)
        uc = lc

        lb, ub = get_bounds_from_conditional(self.lb, self.ub, lc, uc)

        return DeepPoly(lb, ub, lc, uc, self)

    def propagate_conv2d(self, conv_layer: nn.Conv2d) -> "DeepPoly":
        """
        Generate DeepPoly for a convolutional layer

        :param conv_layer:    Specific convolutional layer of NN
        :return:    DeepPoly for that Layer
        """
        # find input dimensions
        assert self.ub.ndim == 1
        parent_size = self.ub.size()[0]
        img_size = int(np.sqrt(parent_size // conv_layer.in_channels))
        assert conv_layer.in_channels * img_size * img_size == parent_size
        input_size = [
            conv_layer.in_channels,
            img_size,
            img_size,
        ]  # e.g. first layer [1,28,28]

        # convert Convolutional into linear layer
        linear_layer = conv_to_fc(conv_layer, input_size)
        
        return self.propagate_linear(linear_layer)

    def propagate_relu(self, relu_layer: nn.ReLU) -> "DeepPoly":
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

        # alpha = 0 -> Naive 1
        # alpha = 1 -> Naive 2
        # alpha \in [0, 1]^n, alpha relaxation
        return self.propagate_leakyrelu(None, alpha=0)

    def propagate_leakyrelu(self, leakyrelu_layer: Optional['nn.LeakyReLU'],
                            alpha: Union[float, 'torch.tensor'] = 0) -> 'DeepPoly':
        """
        NOT IMPLEMENTED YET
        Generate DeepPoly for a leaky ReLU layer

        :param leakyrelu_layer:    Specific leaky ReLU layer of NN
        :param alpha:   Negative Slope for "non-leaky" ReLU layers of same shape as lb
        :return:    DeepPoly for that Layer
        """
        #alpha = torch.rand(1) # uniform random [0,1]
        alpha = 0.2
        slope = torch.divide(self.ub, torch.subtract(self.ub,
                                                     self.lb))  # only used for in between case i.e. sellf. lv < 0 and self.ub > 0 -> slope >= 0

        if leakyrelu_layer:
            negative_slope = leakyrelu_layer.negative_slope  # [0,3]
        else:
            negative_slope = 0

        # upper bounds =< 0
        lc = torch.cat((torch.unsqueeze(torch.zeros(self.lb.shape[0]), 1),
                        torch.multiply(torch.eye(self.lb.shape[0]), negative_slope)), dim=1)
        uc = torch.cat((torch.unsqueeze(torch.zeros(self.lb.shape[0]), 1),
                        torch.multiply(torch.eye(self.lb.shape[0]), negative_slope)), dim=1)

        # lower bound >= 0
        positive = torch.where(self.lb > 0, torch.ones_like(self.lb, dtype=torch.bool),
                               torch.zeros_like(self.lb, dtype=torch.bool))
        positive_mask = get_2d_mask(positive)  # is 1 in columns having a positive lower bound, else 0

        lc = torch.where(positive_mask,
                         torch.cat((torch.unsqueeze(torch.zeros_like(self.lb), 1), torch.eye(self.lb.shape[0])), 1), lc)
        uc = torch.where(positive_mask,
                         torch.cat((torch.unsqueeze(torch.zeros_like(self.lb), 1), torch.eye(self.lb.shape[0])), 1), uc)

        # lower bound < 0 and upper bound > 0
        between = torch.where(torch.logical_and((self.lb < 0), (self.ub > 0)),
                              torch.ones_like(self.lb, dtype=torch.bool), torch.zeros_like(self.lb, dtype=torch.bool))
        between_mask = get_2d_mask(
            between)  # is 1 in columns having negative lower bound and positive upper bound, else 0

        # case1: negative slope <1
        m = torch.divide(torch.subtract(torch.multiply(negative_slope, self.lb), self.ub),
                         torch.subtract(self.lb, self.ub))
        q = torch.neg(torch.divide(torch.multiply(torch.multiply(self.lb, torch.subtract(negative_slope, 1)), self.ub),
                                   torch.subtract(self.lb, self.ub)))
        assert torch.logical_or(torch.logical_not(between),
                                torch.greater_equal(m, 0)).all(), "slope should be positive in between case"
        if negative_slope <= 1:
            uc = torch.where(between_mask,
                             torch.cat((torch.unsqueeze(q, 1), torch.multiply(torch.eye(self.lb.shape[0]), m)), 1), uc)
            lc = torch.where(between_mask, torch.cat((torch.unsqueeze(torch.zeros(self.lb.shape[0]), 1),
                                                      torch.multiply(torch.eye(self.lb.shape[0]),
                                                                     alpha * negative_slope + (1 - alpha) * 1)), 1), lc)

        # case2: negative slope > 1
        else:
            uc = torch.where(between_mask, torch.cat((torch.unsqueeze(torch.zeros(self.lb.shape[0]), 1),
                                                      torch.multiply(torch.eye(self.lb.shape[0]),
                                                                     alpha * negative_slope + (1 - alpha) * 1)), 1), uc)
            lc = torch.where(between_mask,
                             torch.cat((torch.unsqueeze(q, 1), torch.multiply(torch.eye(self.lb.shape[0]), m)), 1), lc)

        lb, ub = get_bounds_from_conditional(self.lb, self.ub, lc, uc)

        assert torch.greater_equal(ub, lb).all()

        return DeepPoly(lb, ub, lc, uc, self)

    def propagate_flatten(self) -> "DeepPoly":
        """
        NOT IMPLEMENTED YET
        Generate DeepPoly for a flatten call on NN

        :return:    DeepPoly that adapts for topology change
        """
        return self

    def propagate_final(self, true_label) -> 'DeepPoly':
        # if 2 is true_label
        #               | 0  1  0  -1  .  .  .  0  |
        #               | 0  0  1  -1  .  .  .  0  |
        #               | 0  0  0  1-1  .  .  .  0  |
        # uc  =  lc  =  | .  .  .  .  .        .  |
        #               | .  .  .  .     .     .  |
        #               | .  .  .  .        .  .  |
        #               | 0  0  0  -1  .  .  .  1  |

        augment = torch.zeros(self.uc.shape[0])
        augment = torch.unsqueeze(augment, 1)
        uc = torch.cat((augment, torch.eye(self.uc.shape[0])), 1)
        lc = torch.cat((augment, torch.eye(self.lc.shape[0])), 1)

        neg_one = torch.zeros(uc.shape)
        neg_one[:, true_label + 1] = -1

        uc = torch.add(uc, neg_one)
        lc = torch.add(lc, neg_one)

        return DeepPoly(self.lb, self.ub, lc, uc, self)


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


        lb, ub = get_bounds_from_conditional(dp.parent.lb, dp.parent.ub, lc,
                                             uc)  # lb, ub are projections of bounds to output dimensions

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

    # special case for test
    if eps == 1:
        lb = x - eps
        ub = x + eps

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

    assert torch.greater_equal(ub, lb).all()
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

    new_lb = torch.tensor(new_lb)
    new_ub = torch.tensor(new_ub)

    # assert torch.greater_equal(new_ub, new_lb).all() #why doesn´t this pass?

    return new_lb, new_ub


def backsubstitute(dp: "DeepPoly", backprop_counter: int):
    """
    Tigthen lb and ub of dp by backsubstituting . This is called after each ReLu layer


    Multiply current ub with previos ub and lastly get_bounds from conditional of last layer
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

    current_dp = dp

    while backprop_counter > 0:
        backprop_counter -= 1
        # UPDATE RULE OF CONDITIONAL MATRICES
        #
        #                               |  1  0  0  .  .  .  0  |
        #                               |                       |
        # CONDITIONAL = CONDITIONAL  X  |   PARENT CONDITIONAL  |
        #                               |                       |

        e = torch.zeros(dp.uc.shape)[1]
        e[0] = 1
        e = torch.unsqueeze(e, 0)
        augmented_uc = torch.cat((e, dp.uc), 0)  # parent
        augmented_lc = torch.cat((e, dp.lc), 0)
        
        # for upper conditional multiply with lower conditional of parent if parameter of this is negative
        """
        new_lc = []
        for row in lc:
            positive_param = row >= 0
            # TODO: check this shape, must be same as augmented_lc
            positive_mask = torch.transpose(
                positive_param.repeat(augmented_lc.shape[1]).reshape(augmented_lc.shape[1], augmented_lc.shape[0]), 0,
                1)  # in all rows true where param in column was positive

            parent_conditional = torch.where(positive_mask, augmented_lc, augmented_uc)
            new_row = torch.matmul(row, parent_conditional)
            new_lc.append(new_row)

        new_uc = []
        for row in uc:
            positive_param = row > 0
            positive_mask = torch.transpose(
                positive_param.repeat(augmented_uc.shape[1]).reshape(augmented_uc.shape[1], augmented_uc.shape[0]), 0,
                1)  # in all rows true where param in column was positive

            parent_conditional = torch.where(positive_mask, augmented_uc, augmented_lc)
            new_row = torch.matmul(row, parent_conditional)
            new_uc.append(new_row)

        lc = torch.stack(new_lc)
        uc = torch.stack(new_uc)
        """
        positive_lc = torch.where(lc >= 0, lc, torch.zeros_like(lc))
        negative_lc = torch.where(lc < 0, lc, torch.zeros_like(lc))
        lc = torch.matmul(positive_lc , augmented_lc) + torch.matmul(negative_lc , augmented_uc)

        positive_uc = torch.where(uc >= 0, uc, torch.zeros_like(uc))
        negative_uc = torch.where(uc < 0, uc, torch.zeros_like(uc))
        uc = torch.matmul(positive_uc , augmented_uc) + torch.matmul(negative_uc , augmented_lc)
        
        dp = dp.parent
        
    lb, ub = get_bounds_from_conditional(dp.lb, dp.ub, lc, uc)

    # update bounds, if tighter then before
    current_dp.lb = torch.where(lb > current_dp.lb, lb, current_dp.lb) 
    current_dp.ub = torch.where(ub < current_dp.ub, ub, current_dp.ub)
    
    
    
