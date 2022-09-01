from typing import Tuple

import numpy as np
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    nullary_predicates: np.ndarray
    labels: np.ndarray

    def __init__(
        self, nullary_predicates: np.ndarray, labels: np.ndarray
    ) -> None:
        self.nullary_predicates = nullary_predicates
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        return self.nullary_predicates[idx], self.labels[idx]
