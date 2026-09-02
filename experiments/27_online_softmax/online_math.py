from __future__ import annotations

import math
from typing import Iterable


NEG_INF = float("-inf")


def combine_states(m_a: float, l_a: float, m_b: float, l_b: float) -> tuple[float, float]:
    if l_a == 0.0:
        return m_b, l_b
    if l_b == 0.0:
        return m_a, l_a
    m = max(m_a, m_b)
    l = l_a * math.exp(m_a - m) + l_b * math.exp(m_b - m)
    return m, l


def update_state(m: float, l: float, x: float) -> tuple[float, float]:
    return combine_states(m, l, x, 1.0)


def online_normalizer(values: Iterable[float]) -> tuple[float, float]:
    m = NEG_INF
    l = 0.0
    for value in values:
        m, l = update_state(m, l, float(value))
    return m, l


def online_softmax(values: Iterable[float]) -> list[float]:
    xs = [float(value) for value in values]
    if not xs:
        return []
    m, l = online_normalizer(xs)
    return [math.exp(value - m) / l for value in xs]


def stable_softmax(values: Iterable[float]) -> list[float]:
    xs = [float(value) for value in values]
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(value - m) for value in xs]
    denom = sum(exps)
    return [value / denom for value in exps]
