from typing import List, OrderedDict

import torch
from torch import nn, Tensor

from dnf_layer import (
    DNF,
    DNFSP,
    SemiSymbolic,
    SemiSymbolicLayerType,
    ConstraintLayer,
)


class CtoYClassifier(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError


class DNFBasedClassifier(CtoYClassifier):
    conj_weight_mask: Tensor
    disj_weight_mask: Tensor
    dnf: DNF

    def __init__(
        self,
        num_preds: int,  # P
        num_conjuncts: int,  # Q
        num_classes: int,  # R
        delta: float = 1.0,
        weight_init_type: str = "normal",
    ) -> None:
        super(DNFBasedClassifier, self).__init__()

        self.dnf = DNF(
            num_preds, num_conjuncts, num_classes, delta, weight_init_type
        )

        self.conj_weight_mask = torch.ones(
            self.dnf.conjunctions.weights.data.shape
        )
        self.disj_weight_mask = torch.ones(
            self.dnf.disjunctions.weights.data.shape
        )

    def set_delta_val(self, delta_val: float) -> None:
        self.dnf.conjunctions.delta = delta_val
        self.dnf.disjunctions.delta = delta_val

    def update_weight_wrt_mask(self) -> None:
        self.dnf.conjunctions.weights.data *= self.conj_weight_mask
        self.dnf.disjunctions.weights.data *= self.disj_weight_mask


class DNFClassifier(DNFBasedClassifier):
    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        out = self.dnf(input)
        # out: N x R
        return out


class DNFClassifierEO(DNFBasedClassifier):
    eo_layer: SemiSymbolic

    def __init__(
        self,
        num_preds: int,  # P
        num_conjuncts: int,  # Q
        num_classes: int,  # R
        delta: float = 1.0,
        weight_init_type: str = "normal",
    ) -> None:
        super(DNFClassifierEO, self).__init__(
            num_preds, num_conjuncts, num_classes, delta, weight_init_type
        )

        # Exactly one constraint
        # class_1 :- not class_2, not class_3, ... , not class_n
        # class_i :- CONJ[not class_j for j in [1, n] and j != i]
        self.eo_layer = SemiSymbolic(
            in_features=num_classes,
            out_features=num_classes,
            layer_type=SemiSymbolicLayerType.CONJUNCTION,
            delta=delta,
        )
        self.eo_layer.weights.data.fill_(-6)
        self.eo_layer.weights.data.fill_diagonal_(0)
        self.eo_layer.requires_grad_(False)

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        dnf_out = self.dnf(input)
        # dnf_out: N x R
        dnf_out = torch.tanh(dnf_out)
        # dnf_out: N x R
        out = self.eo_layer(dnf_out)

        return out

    def set_delta_val(self, delta_val: float) -> None:
        self.dnf.conjunctions.delta = delta_val
        self.dnf.disjunctions.delta = delta_val
        self.eo_layer.delta = delta_val

    def load_dnf_state_dict(self, sd: dict) -> None:
        # Assuming that sd is from a DNF Classifier model
        sd_keys = list(sd.keys())
        for k in sd_keys:
            sd[k[4:]] = sd.pop(k)
        self.dnf.load_state_dict(sd)


class DNFClassifierCS(DNFBasedClassifier):
    cs_layer: ConstraintLayer

    def __init__(
        self,
        num_preds: int,  # P
        num_conjuncts: int,  # Q
        num_classes: int,  # R
        ordered_constraint_list: List[List[int]],
        delta: float = 1.0,
        weight_init_type: str = "normal",
    ) -> None:
        super(DNFClassifierCS, self).__init__(
            num_preds, num_conjuncts, num_classes, delta, weight_init_type
        )

        # Constraints
        self.cs_layer = ConstraintLayer(
            in_features=num_classes,
            out_features=num_classes,
            delta=delta,
            ordered_constraint_list=ordered_constraint_list,
        )

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        dnf_out = self.dnf(input)
        # dnf_out: N x R
        dnf_out = torch.tanh(dnf_out)
        # dnf_out: N x R
        out = self.cs_layer(dnf_out)
        # out: [N x R]

        return out

    def set_delta_val(self, delta_val: float) -> None:
        self.dnf.conjunctions.delta = delta_val
        self.dnf.disjunctions.delta = delta_val
        self.cs_layer.delta = delta_val

    def load_dnf_state_dict(self, sd: OrderedDict) -> None:
        # Assuming that sd is from a DNF Classifier model
        sd_keys = list(sd.keys())
        for k in sd_keys:
            sd[k[4:]] = sd.pop(k)
        self.dnf.load_state_dict(sd)


class DNFClassifierSP(CtoYClassifier):
    conj_weight_mask: Tensor
    disj_weight_mask: List[Tensor]

    dnfsp: DNFSP

    def __init__(
        self,
        num_preds: int,  # P
        num_conj_per_disj: int,  # Q
        num_classes: int,  # R
        delta: float = 1.0,
    ) -> None:
        super(DNFClassifierSP, self).__init__()

        self.dnfsp = DNFSP(num_preds, num_conj_per_disj, num_classes, delta)

        self.conj_weight_mask = torch.ones(
            self.dnfsp.conjunctions.weights.data.shape
        )
        self.disj_weight_mask = [
            torch.ones(l.weights.data.shape) for l in self.dnfsp.disjunctions  # type: ignore
        ]

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        out = self.dnfsp(input)
        # out: N x R
        return out

    def set_delta_val(self, delta_val: float) -> None:
        self.dnfsp.conjunctions.delta = delta_val
        for l in self.dnfsp.disjunctions:
            l.delta = delta_val  # type: ignore

    def update_weight_wrt_mask(self) -> None:
        self.dnfsp.conjunctions.weights.data *= self.conj_weight_mask
        for i, m in enumerate(self.disj_weight_mask):
            self.dnfsp.disjunctions[i].weights.data *= m


class SingleLayerClassifier(CtoYClassifier):
    def __init__(self, num_preds: int, num_classes: int) -> None:
        super(SingleLayerClassifier, self).__init__()

        self.linear = nn.Linear(num_preds, num_classes)

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        out = self.linear(input)
        # out: N x R
        return out
