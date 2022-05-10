from typing import Any, Tuple, List, Optional

from sklearn.metrics import jaccard_score
import torch
from torch import Tensor


class Meter:
    acc_n_sample: int

    def __init__(self):
        self.acc_n_sample = 0

    def update(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def get_average(self) -> float:
        raise NotImplementedError


class BinaryAccuracyMeter(Meter):
    """
    Accuracy meter for binary data
    """

    acc_n_correct: int
    accuracy_list: List[float]

    def __init__(self):
        super(BinaryAccuracyMeter, self).__init__()
        self.acc_n_correct = 0
        self.accuracy_list = []

    def update(self, output: Tensor, target: Tensor) -> Tuple[int, int]:
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

        self.acc_n_sample += n_sample
        self.acc_n_correct += n_correct
        self.accuracy_list.append(n_correct / n_sample)

        return n_sample, n_correct

    def get_average(self) -> float:
        return torch.mean(torch.Tensor(self.accuracy_list)).item()


class MultiClassAccuracyMeter(Meter):
    acc_n_correct: int
    accuracy_list: List[float]

    def __init__(self):
        super(MultiClassAccuracyMeter, self).__init__()
        self.acc_n_correct = 0
        self.accuracy_list = []

    def update(self, output: Tensor, target: Tensor) -> Tuple[int, int]:
        """
        Update the accumulated sample counts and correct counts.

        Pre-conditions:
          - `output` and `target` are of the same type of tensor
          - `output` should be softmaxed already
        """

        n_sample = len(target)

        _, y_pred = torch.max(output, 1)
        n_correct = torch.sum(y_pred == target).item()

        self.acc_n_sample += n_sample
        self.acc_n_correct += n_correct
        self.accuracy_list.append(n_correct / n_sample)

        return n_sample, n_correct

    def get_average(self) -> float:
        return torch.mean(torch.Tensor(self.accuracy_list)).item()


class MetricValueMeter(Meter):
    metric_name: str
    acc_val: float
    vals: List[float]

    def __init__(self, metric_name: str):
        super(MetricValueMeter, self).__init__()
        self.metric_name = metric_name
        self.acc_val = 0.0
        self.vals = []

    def update(
        self, val: float, n_sample: Optional[int] = None
    ) -> Tuple[int, float]:
        self.vals.append(val)
        self.acc_val += val

        n_sample = n_sample if n_sample else 1
        self.acc_n_sample += n_sample

        return n_sample, val

    def get_average(self) -> float:
        return self.acc_val / self.acc_n_sample


class JaccardScoreMeter(Meter):
    jacc_scores: List[float]

    def __init__(self) -> None:
        super(JaccardScoreMeter, self).__init__()
        self.jacc_scores = []

    def update(self, output: Tensor, target: Tensor) -> float:
        y = torch.zeros(output.shape)
        y[range(output.shape[0]), target.long()] = 1
        y_np = y.detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()
        avg = jaccard_score(y_np, output_np, average="samples")
        self.jacc_scores.append(avg)
        return avg

    def get_average(self) -> float:
        return sum(self.jacc_scores) / len(self.jacc_scores)
