"""Performance indicators, computed from the executed log only, and bootstrap intervals."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .scenario import Scenario, Workshop
from .shop import RunLog


def run_metrics(ws: Workshop, scenario: Scenario, log: RunLog) -> dict[str, float]:
    completion: dict[int, int] = {}
    proc_total = setup_total = 0
    for s in log.segments:
        completion[s.job] = max(completion.get(s.job, 0), s.end)
        if s.kind == "proc":
            proc_total += s.end - s.start
        else:
            setup_total += s.end - s.start
    tardiness = np.array([max(0, completion[j.idx] - j.due) for j in scenario.jobs], dtype=float)
    flow = np.array([completion[j.idx] - j.arrival for j in scenario.jobs], dtype=float)
    makespan = max(completion.values())
    empty = sum(g.end - g.start for g in log.legs if g.job is None)
    loaded = sum(g.end - g.start for g in log.legs if g.job is not None)
    return {
        "makespan": float(makespan),
        "mean_tardiness": float(tardiness.mean()),
        "max_tardiness": float(tardiness.max()),
        "tardy_share": float((tardiness > 0).mean()),
        "mean_flow_time": float(flow.mean()),
        # share of the machine-time available until the last job leaves that is spent processing
        "utilisation": proc_total / (ws.n_machines * makespan),
        "setup_time": float(setup_total),
        "agv_empty_share": empty / max(1, empty + loaded),
    }


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(resamples, len(values)), replace=True)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples.mean(axis=1), [alpha, 1.0 - alpha])
    return float(low), float(high)
