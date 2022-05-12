from typing import Any, Tuple, List, Optional

from sklearn.metrics import jaccard_score
import torch
from torch import Tensor


class Meter:
    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def get_average(self) -> float:
        raise NotImplementedError


class BinaryAccuracyMeter(Meter):
    accuracy_list: List[float]

    def __init__(self):
        super(BinaryAccuracyMeter, self).__init__()
        self.accuracy_list = []

    def update(self, output: Tensor, target: Tensor) -> None:
        """
        Update the accumulated sample counts and correct counts.

        Pre-conditions:
          - `output` and `target` are of the same type of tensor
          - `output` and `target` should both be binary
          - `output` would be in dimensions like: ... 'x 1', or already squeezed
            the last 'x 1' dimension.
        """
        output_shape_tensor: Tensor = torch.Tensor(list(output.shape))
        if output_shape_tensor[-1] == 1:
            output_shape_tensor = output.squeeze(-1)

        n_sample = int(torch.prod(output_shape_tensor).item())
        n_correct = torch.sum(output == target).item()

        self.accuracy_list.append(n_correct / n_sample)

    def get_average(self) -> float:
        return torch.mean(torch.Tensor(self.accuracy_list)).item()


class MultiClassAccuracyMeter(Meter):
    accuracy_list: List[float]

    def __init__(self):
        super(MultiClassAccuracyMeter, self).__init__()
        self.accuracy_list = []

    def update(self, output: Tensor, target: Tensor) -> None:
        """
        Update the accumulated sample counts and correct counts.

        Pre-conditions:
          - `output` and `target` are of the same type of tensor
          - `output` should be softmaxed already
        """

        n_sample = len(target)

        _, y_pred = torch.max(output, 1)
        n_correct = torch.sum(y_pred == target).item()

        self.accuracy_list.append(n_correct / n_sample)

    def get_average(self) -> float:
        return torch.mean(torch.Tensor(self.accuracy_list)).item()


class MetricValueMeter(Meter):
    metric_name: str
    vals: List[float]

    def __init__(self, metric_name: str):
        super(MetricValueMeter, self).__init__()
        self.metric_name = metric_name
        self.vals = []

    def update(self, val: float) -> None:
        self.vals.append(val)

    def get_average(self) -> float:
        return torch.mean(torch.Tensor(self.vals)).item()


class JaccardScoreMeter(Meter):
    jacc_scores: List[float]

    def __init__(self) -> None:
        super(JaccardScoreMeter, self).__init__()
        self.jacc_scores = []

    def update(self, output: Tensor, target: Tensor) -> None:
        y = torch.zeros(output.shape)
        y[range(output.shape[0]), target.long()] = 1
        y_np = y.detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()
        avg = jaccard_score(y_np, output_np, average="samples")
        self.jacc_scores.append(avg)

    def get_average(self) -> float:
        return sum(self.jacc_scores) / len(self.jacc_scores)
