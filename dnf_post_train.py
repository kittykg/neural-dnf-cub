from typing import OrderedDict, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import tqdm

from dnf_layer import SemiSymbolicLayerType
from eval import dnf_eval
from rule_learner import DNFBasedClassifier


def prune_layer_weight(
    model: DNFBasedClassifier,
    layer_type: SemiSymbolicLayerType,
    epsilon: float,
    data_loader: DataLoader,
    use_cuda: bool,
    use_jaccard_meter: bool = False,
    show_tqdm: bool = False,
) -> int:
    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        curr_weight = model.dnf.conjunctions.weights.data.clone()
    else:
        curr_weight = model.dnf.disjunctions.weights.data.clone()

    og_accuracy = dnf_eval(model, use_cuda, data_loader, use_jaccard_meter)

    prune_count = 0
    weight_device = curr_weight.device

    flatten_weight_len = len(torch.reshape(curr_weight, (-1,)))
    base_iterator = range(flatten_weight_len)
    iterator = tqdm(base_iterator) if show_tqdm else base_iterator

    for i in iterator:
        curr_weight_flatten = torch.reshape(curr_weight, (-1,))

        if curr_weight_flatten[i] == 0:
            continue

        mask = torch.ones(flatten_weight_len, device=weight_device)
        mask[i] = 0
        mask = mask.reshape(curr_weight.shape)

        masked_weight = curr_weight * mask

        if layer_type == SemiSymbolicLayerType.CONJUNCTION:
            model.dnf.conjunctions.weights.data = masked_weight
        else:
            model.dnf.disjunctions.weights.data = masked_weight

        new_perf = dnf_eval(model, use_cuda, data_loader, use_jaccard_meter)
        performance_drop = og_accuracy - new_perf
        if performance_drop < epsilon:
            prune_count += 1
            curr_weight *= mask

    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        model.dnf.conjunctions.weights.data = curr_weight
    else:
        model.dnf.disjunctions.weights.data = curr_weight
    return prune_count


def remove_unused_conjunctions(model: DNFBasedClassifier) -> int:
    disj_w = model.dnf.disjunctions.weights.data.clone()
    unused_count = 0

    for i, w in enumerate(disj_w.T):
        if torch.all(w == 0):
            # The conjunction is not used at all
            model.dnf.conjunctions.weights.data[i, :] = 0
            unused_count += 1

    return unused_count


def apply_threshold(
    model: DNFBasedClassifier,
    og_conj_weight: Tensor,
    og_disj_weight: Tensor,
    t_val: Tensor,
    const: float = 6.0,
) -> None:
    new_conj_weight = (
        (torch.abs(og_conj_weight) > t_val) * torch.sign(og_conj_weight) * const
    )
    model.dnf.conjunctions.weights.data = new_conj_weight

    new_disj_weight = (
        (torch.abs(og_disj_weight) > t_val) * torch.sign(og_disj_weight) * const
    )
    model.dnf.disjunctions.weights.data = new_disj_weight


def extract_asp_rules(sd: OrderedDict, flatten: bool = False) -> List[str]:
    output_rules = []

    # Get all conjunctions
    conj_w = sd["dnf.conjunctions.weights"]
    conjunction_map = dict()
    for i, w in enumerate(conj_w):
        if torch.all(w == 0):
            # No conjunction is applied here
            continue

        conjuncts = []
        for j, v in enumerate(w):
            if v < 0:
                # Negative weight, negate the atom
                conjuncts.append(f"not has_attr_{j}")
            elif v > 0:
                # Positive weight, normal atom
                conjuncts.append(f"has_attr_{j}")

        conjunction_map[i] = conjuncts

    if not flatten:
        # Add conjunctions as auxilary predicates into final rules list
        # if not flatten
        for k, v in conjunction_map.items():
            output_rules.append(f"conj_{k} :- {', '.join(v)}.")

    # Get DNF
    disj_w = sd["dnf.disjunctions.weights"]
    not_covered_classes = []
    for i, w in enumerate(disj_w):
        if torch.all(w == 0):
            # No DNF for class i
            not_covered_classes.append(i)
            continue

        disjuncts = []
        for j, v in enumerate(w):
            if v < 0 and j in conjunction_map:
                # Negative weight, negate the existing conjunction
                if flatten:
                    # Need to add auxilary predicate (conj_X) which is not yet
                    # in the final rules list
                    output_rules.append(
                        f"conj_{j} :- {', '.join(conjunction_map[j])}."
                    )
                    output_rules.append(f"class({i}) :- not conj_{j}.")
                else:
                    disjuncts.append(f"not conj_{j}")
            elif v > 0 and j in conjunction_map:
                # Postivie weight, add normal conjunction
                if flatten:
                    body = ", ".join(conjunction_map[j])
                    output_rules.append(f"class({i}) :- {body}.")
                else:
                    disjuncts.append(f"conj_{j}")

        if not flatten:
            for disjunct in disjuncts:
                output_rules.append(f"class({i}) :- {disjunct}.")

    return output_rules
