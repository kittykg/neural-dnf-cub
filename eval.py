import logging
from typing import List, Tuple

import clingo
import torch
from torch.utils.data import DataLoader

from analysis import MultiClassAccuracyMeter, JaccardScoreMeter
from common import CUBDNDataItem
from rule_learner import DNFBasedClassifier
from utils import get_dnf_classifier_x_and_y


log = logging.getLogger()


def dnf_eval(
    model: DNFBasedClassifier,
    use_cuda: bool,
    data_loader: DataLoader,
    use_jaccard_meter: bool = False,
    jaccard_threshold: float = 0.0,
    do_logging: bool = False,
):
    model.eval()
    performance_meter = (
        JaccardScoreMeter() if use_jaccard_meter else MultiClassAccuracyMeter()
    )

    for i, data in enumerate(data_loader):
        iter_perf_meter = (
            JaccardScoreMeter()
            if use_jaccard_meter
            else MultiClassAccuracyMeter()
        )

        with torch.no_grad():
            x, y = get_dnf_classifier_x_and_y(data, use_cuda)
            y_hat = model(x)
            if use_jaccard_meter:
                y_hat = (torch.tanh(y_hat) > jaccard_threshold).long()

            iter_perf_meter.update(y_hat, y)
            performance_meter.update(y_hat, y)

        if do_logging:
            log.info(
                "[%3d] Test     avg perf: %.3f"
                % (i + 1, iter_perf_meter.get_average())
            )

    if do_logging:
        log.info(
            "Overall Test   avg perf: %.3f" % performance_meter.get_average()
        )

    return performance_meter.get_average()


def asp_eval(
    test_data: List[CUBDNDataItem], rules: List[str], debug: bool = False
) -> Tuple[float, float]:
    total_sample_count = 0
    correct_count = 0
    jaccard_scores = []

    for d in test_data:
        asp_base = []
        for i, a in enumerate(d.attr_present_label):
            if a == 1:
                asp_base.append(f"has_attr_{i}.")

        asp_base += rules
        asp_base.append("#show class/1.")

        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], " ".join(asp_base))
        ctl.ground([("base", [])])

        all_answer_sets = [str(a) for a in ctl.solve(yield_=True)]

        target_class = f"class({d.label - 1})"

        if debug:
            # Print out
            print(f"y: {target_class}  AS: {all_answer_sets}")

        if len(all_answer_sets) != 1:
            jaccard_scores.append(0)
            continue

        output_classes = all_answer_sets[0].split(" ")
        output_classes_set = set(output_classes)

        target_class_set = {target_class}

        jacc = len(output_classes_set & target_class_set) / len(
            output_classes_set | target_class_set
        )
        jaccard_scores.append(jacc)

        if len(output_classes) == 1 and target_class in output_classes:
            correct_count += 1
        total_sample_count += 1

    accuracy = correct_count / total_sample_count
    avg_jacc_score = sum(jaccard_scores) / len(jaccard_scores)

    return accuracy, avg_jacc_score
