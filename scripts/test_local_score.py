#!/usr/bin/env python3
"""Unit tests for the transparent local scoring proxies."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_score import absolute_relative_error, score_from_error, weighted_daily_score


class LocalScoreTests(unittest.TestCase):
    def test_relative_error_and_zero_target_convention(self) -> None:
        actual = np.array([100.0, 0.0, 0.0])
        predicted = np.array([110.0, 0.0, 1.0])
        result = absolute_relative_error(actual, predicted)
        self.assertAlmostEqual(result[0], 0.1)
        self.assertEqual(result[1], 0.0)
        self.assertTrue(np.isinf(result[2]))

    def test_threshold_and_shape(self) -> None:
        error = np.array([0.0, 0.15, 0.30, 0.31])
        linear = score_from_error(error, power=1)
        quadratic = score_from_error(error, power=2)
        cubic = score_from_error(error, power=3)
        np.testing.assert_allclose(linear, [10.0, 5.0, 0.0, 0.0])
        np.testing.assert_allclose(quadratic, [10.0, 2.5, 0.0, 0.0])
        np.testing.assert_allclose(cubic, [10.0, 1.25, 0.0, 0.0])

    def test_purchase_redeem_weighting(self) -> None:
        result = weighted_daily_score(
            np.array([100.0]),
            np.array([100.0]),
            np.array([100.0]),
            np.array([130.0]),
            power=1,
        )
        self.assertAlmostEqual(result[0], 4.5)


if __name__ == "__main__":
    unittest.main()
