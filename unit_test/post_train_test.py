import sys
import unittest

import torch

sys.path.append("../")

from rule_learner import DNFBasedClassifier
from dnf_post_train import (
    remove_unused_conjunctions,
)


class TestPostTrainMethods(unittest.TestCase):
    def test_remove_unused_conjunctions_1(self):
        test_model = DNFBasedClassifier(4, 3, 2)
        test_model.dnf.disjunctions.weights.data = torch.Tensor(
            [[1, 0, -1], [0, 1, 0]]
        )  # 2 x 3 matrix
        og_conj_weight = torch.randint(-1, 2, (3, 4))  # 3 x 4 matrix
        test_model.dnf.conjunctions.weights.data = og_conj_weight

        unused_count = remove_unused_conjunctions(test_model)

        self.assertEqual(unused_count, 0)
        torch.testing.assert_close(
            test_model.dnf.conjunctions.weights.data, og_conj_weight
        )

    def test_remove_unused_conjunctions_2(self):
        test_model = DNFBasedClassifier(4, 3, 2)
        test_model.dnf.disjunctions.weights.data = torch.Tensor(
            [[1, 0, 0], [0, 0, 0]]
        )  # 2 x 3 matrix
        og_conj_weight = torch.randint(-1, 2, (3, 4))  # 3 x 4 matrix
        test_model.dnf.conjunctions.weights.data = og_conj_weight

        unused_count = remove_unused_conjunctions(test_model)

        expected_new_conj_weight = og_conj_weight
        expected_new_conj_weight[-1, :] = 0
        expected_new_conj_weight[1, :] = 0

        self.assertEqual(unused_count, 2)
        torch.testing.assert_close(
            test_model.dnf.conjunctions.weights.data, og_conj_weight
        )


if __name__ == "__main__":
    unittest.main()
