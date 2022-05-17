import logging

import torch
from torch.utils.data import DataLoader

from analysis import MultiClassAccuracyMeter, JaccardScoreMeter
from rule_learner import DNFBasedClassifier
from utils import get_dnf_classifier_x_and_y


log = logging.getLogger()


def dnf_eval(
    model: DNFBasedClassifier,
    use_cuda: bool,
    data_loader: DataLoader,
    use_jaccard_meter: bool = False,
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
                y_hat = (torch.tanh(y_hat) > 0).long()

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
