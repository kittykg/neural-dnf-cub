from enum import Enum
from typing import List

import torch
from torch import nn, Tensor


class SemiSymbolicLayerType(Enum):
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"


class SemiSymbolic(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_type: SemiSymbolicLayerType,
        delta: float,
        weight_init_type: str = "normal",
        epsilon: float = 0.001,
    ) -> None:
        super(SemiSymbolic, self).__init__()

        self.layer_type = layer_type

        self.in_features = in_features  # P
        self.out_features = out_features  # Q

        self.weights = nn.Parameter(
            torch.empty((self.out_features, self.in_features))
        )
        if weight_init_type == "normal":
            nn.init.normal_(self.weights, mean=0.0, std=0.1)
        else:
            nn.init.uniform_(self.weights, a=-6, b=6)
        self.delta = delta

        # For DNF min
        self.epsilon = epsilon

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        abs_weight = torch.abs(self.weights)
        # abs_weight: Q x P
        max_abs_w = torch.max(abs_weight, dim=1)[0]
        # max_abs_w: Q

        # nonzero_weight = torch.where(
        #     abs_weight > self.epsilon, abs_weight.double(), 100.0
        # )
        # nonzero_min = torch.min(nonzero_weight, dim=1)[0]

        sum_abs_w = torch.sum(abs_weight, dim=1)
        # sum_abs_w: Q
        if self.layer_type == SemiSymbolicLayerType.CONJUNCTION:
            bias = max_abs_w - sum_abs_w
            # bias = nonzero_min - sum_abs_w
        else:
            bias = sum_abs_w - max_abs_w
            # bias = sum_abs_w - nonzero_min
        # bias: Q

        out = input @ self.weights.T
        # out: N x Q
        out_bias = self.delta * bias
        # out_bias: Q
        sum = out + out_bias
        # sum: N x Q
        return sum  # .float()


class ConstraintLayer(SemiSymbolic):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        delta: float,
        ordered_constraint_list: List[List[int]],
        enable_training: bool = False,
    ):
        super(ConstraintLayer, self).__init__(
            in_features, out_features, SemiSymbolicLayerType.CONJUNCTION, delta
        )
        self.weights.data.fill_(0)
        for class_idx, cl in enumerate(ordered_constraint_list):
            if len(cl) == 0:
                self.weights.data[class_idx, class_idx] = 6
            else:
                for i in cl:
                    self.weights.data[class_idx, i] = -6
            if not enable_training:
                self.requires_grad_(False)


class DNF(nn.Module):
    conjunctions: SemiSymbolic
    disjunctions: SemiSymbolic

    def __init__(
        self,
        num_preds: int,
        num_conjuncts: int,
        n_out: int,
        delta: float,
        weight_init_type: str = "normal",
    ) -> None:
        super(DNF, self).__init__()

        self.conjunctions = SemiSymbolic(
            in_features=num_preds,  # P
            out_features=num_conjuncts,  # Q
            layer_type=SemiSymbolicLayerType.CONJUNCTION,
            delta=delta,
            weight_init_type=weight_init_type,
        )  # weight: Q x P

        self.disjunctions = SemiSymbolic(
            in_features=num_conjuncts,  # Q
            out_features=n_out,  # R
            layer_type=SemiSymbolicLayerType.DISJUNCTION,
            delta=delta,
        )  # weight R x Q

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        conj = self.conjunctions(input)
        # conj: N x Q
        conj = nn.Tanh()(conj)
        # conj: N x Q
        disj = self.disjunctions(conj)
        # disj: N x R

        return disj


class DNFSP(nn.Module):
    num_conj_per_disj: int

    conjunctions: SemiSymbolic
    disjunctions: nn.ModuleList

    def __init__(
        self,
        num_preds: int,  # P
        num_conj_per_disj: int,  # Q
        num_classes: int,  # R
        delta: float = 1.0,
    ) -> None:
        super(DNFSP, self).__init__()

        self.num_conj_per_disj = num_conj_per_disj

        self.conjunctions = SemiSymbolic(
            in_features=num_preds,  # P
            out_features=num_conj_per_disj * num_classes,  # (Q x R)
            layer_type=SemiSymbolicLayerType.CONJUNCTION,
            delta=delta,
        )  # weights: (Q X R) x P

        self.disjunctions = nn.ModuleList(
            [
                SemiSymbolic(
                    in_features=num_conj_per_disj,  # Q
                    out_features=1,
                    layer_type=SemiSymbolicLayerType.DISJUNCTION,
                    delta=delta,
                )
                for _ in range(num_classes)
            ]
        )  # R disjunctive semi-symbolic layers, each with weight 1 x Q

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        conj = self.conjunctions(input)
        # conj: N x (Q x R)
        conj = nn.Tanh()(conj)
        # conj:
        disj = []
        for i, l in enumerate(self.disjunctions):
            start = i * self.num_conj_per_disj  # i * Q
            end = (i + 1) * self.num_conj_per_disj  # (i + 1) * Q
            c = conj[:, start:end]  # N x Q
            d = l(c)  # N x 1
            disj.append(d)
        # disj: [N x 1] with R elements
        disj = torch.cat(disj, dim=1)
        # disj: N x R

        return disj
