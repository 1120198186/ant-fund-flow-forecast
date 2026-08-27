#!/usr/bin/env python3
"""Transparent local scoring proxies for the competition holdouts."""

from __future__ import annotations

import numpy as np


ZERO_SCORE_THRESHOLD = 0.30
MAX_DAILY_SCORE = 10.0
PURCHASE_WEIGHT = 0.45
REDEEM_WEIGHT = 0.55


def absolute_relative_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Return absolute relative error with an explicit zero-target convention."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted must have the same shape")

    error = np.empty_like(actual_values, dtype=float)
    nonzero = actual_values != 0
    error[nonzero] = (
        np.abs(predicted_values[nonzero] - actual_values[nonzero])
        / np.abs(actual_values[nonzero])
    )
    both_zero = (~nonzero) & (predicted_values == 0)
    error[both_zero] = 0.0
    error[(~nonzero) & (~both_zero)] = np.inf
    return error


def score_from_error(
    error: np.ndarray,
    power: int = 1,
    threshold: float = ZERO_SCORE_THRESHOLD,
) -> np.ndarray:
    """Map error to a 0-10 proxy score using linear/quadratic/cubic decay."""
    if power not in {1, 2, 3}:
        raise ValueError("power must be one of 1, 2, or 3")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    values = np.asarray(error, dtype=float)
    normalized_credit = np.clip(1.0 - values / threshold, 0.0, 1.0)
    return MAX_DAILY_SCORE * np.power(normalized_credit, power)


def weighted_daily_score(
    actual_purchase: np.ndarray,
    predicted_purchase: np.ndarray,
    actual_redeem: np.ndarray,
    predicted_redeem: np.ndarray,
    power: int = 1,
) -> np.ndarray:
    """Return the 45/55 weighted daily proxy score."""
    purchase_score = score_from_error(
        absolute_relative_error(actual_purchase, predicted_purchase), power
    )
    redeem_score = score_from_error(
        absolute_relative_error(actual_redeem, predicted_redeem), power
    )
    return PURCHASE_WEIGHT * purchase_score + REDEEM_WEIGHT * redeem_score
