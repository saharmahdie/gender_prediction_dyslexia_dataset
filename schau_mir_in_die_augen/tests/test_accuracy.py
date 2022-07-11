from unittest import TestCase

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

class TestDatasets(TestCase):

    def setUp(self):

        self.y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 15x0
        self.y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 10x0, 5x1

        self.y_true2a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5])  # 10x0, 1,2,3,4,5

        self.y_true2b = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])  # 5x0, 5x1, 5x2

        self.y_pred2c = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # 10x0, 5x1
        self.y_true2c = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])  # 5x0, 5x1, 5x2

    def test_accuracy(self):

        accuracy_basic = accuracy_score(y_true=self.y_true, y_pred=self.y_pred)
        self.assertAlmostEqual(2/3, accuracy_basic)

        accuracy_balanced = balanced_accuracy(y_true=self.y_true, y_pred=self.y_pred)
        accuracy_balanced2 = balanced_accuracy_score(y_true=self.y_true, y_pred=self.y_pred)
        self.assertAlmostEqual(accuracy_balanced, accuracy_balanced2)
        self.assertAlmostEqual(1/2, accuracy_balanced)

    def test_accuracy2a(self):

        accuracy_basic = accuracy_score(y_true=self.y_true2a, y_pred=self.y_pred)
        self.assertAlmostEqual(2 / 3, accuracy_basic)

        accuracy_balanced = balanced_accuracy(y_true=self.y_true2a, y_pred=self.y_pred)
        accuracy_balanced2 = balanced_accuracy_score(y_true=self.y_true2a, y_pred=self.y_pred)
        self.assertAlmostEqual(accuracy_balanced, accuracy_balanced2)
        self.assertAlmostEqual(1 / 6, accuracy_balanced)

    def test_accuracy2b(self):

        accuracy_basic = accuracy_score(y_true=self.y_true2b, y_pred=self.y_pred)
        self.assertAlmostEqual(1 / 3, accuracy_basic)

        accuracy_balanced = balanced_accuracy(y_true=self.y_true2b, y_pred=self.y_pred)
        accuracy_balanced2 = balanced_accuracy_score(y_true=self.y_true2b, y_pred=self.y_pred)
        self.assertAlmostEqual(accuracy_balanced, accuracy_balanced2)
        self.assertAlmostEqual(1 / 3, accuracy_balanced)

    def test_accuracy2c(self):

        accuracy_basic = accuracy_score(y_true=self.y_true2c, y_pred=self.y_pred2c)
        self.assertAlmostEqual(2 / 3, accuracy_basic)

        accuracy_balanced = balanced_accuracy(y_true=self.y_true2c, y_pred=self.y_pred2c)
        accuracy_balanced2 = balanced_accuracy_score(y_true=self.y_true2c, y_pred=self.y_pred2c)
        self.assertAlmostEqual(accuracy_balanced, accuracy_balanced2)
        self.assertAlmostEqual(2 / 3, accuracy_balanced)


def balanced_accuracy(y_true, y_pred):

    weights = y_true.copy() * 0.0

    bin_count = np.bincount(y_true)

    for idx, count in enumerate(bin_count):
        weights[y_true == idx] = 1 / count

    return accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=weights)
