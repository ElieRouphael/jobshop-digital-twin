"""Workshop definition and seeded scenario generation. Everything here is synthetic.

A `Workshop` is the fixed part (machines, routings, setup times, layout); it is the same in
every run. A `Scenario` is what changes with the seed: which jobs arrive when, their due
dates, and when each machine breaks down and comes back. Arrivals and breakdowns are drawn
once per seed, so every decision method faces exactly the same disturbances.

All times are integer minutes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOAD_STATION = 0  # jobs enter the shop here; machine m stands at station m + 1


@dataclass(frozen=True)
class Workshop:
    n_machines: int
    n_agvs: int
    # routings[family][op] = {machine: processing time}; an operation has 1 to 3 eligible machines
    routings: tuple[tuple[dict[int, int], ...], ...]
    # setup[machine][previous family][next family]; zero on the diagonal
    setup: tuple[tuple[tuple[int, ...], ...], ...]
    # travel[a][b] between stations, handling included; zero on the diagonal
    travel: tuple[tuple[int, ...], ...]

    @property
    def n_families(self) -> int:
        return len(self.routings)

    @staticmethod
    def station(machine: int) -> int:
        return machine + 1

    def n_ops(self, family: int) -> int:
        return len(self.routings[family])

    def alternatives(self, family: int, op: int) -> dict[int, int]:
        return self.routings[family][op]

    def proc(self, family: int, op: int, machine: int) -> int:
        return self.routings[family][op][machine]

    def setup_time(self, machine: int, previous_family: int | None, family: int) -> int:
        """Changeover time on a machine. A machine that has not run anything yet needs none."""
        if previous_family is None:
            return 0
        return self.setup[machine][previous_family][family]

    def min_remaining_work(self, family: int, op: int) -> int:
        """Sum of the shortest processing times from operation `op` to the end of the routing."""
        return sum(min(alt.values()) for alt in self.routings[family][op:])


def make_workshop(seed: int = 0, n_machines: int = 5, n_agvs: int = 2, n_families: int = 4) -> Workshop:
    rng = np.random.default_rng([seed, 0])

    routings = []
    for _ in range(n_families):
        ops = []
        for _ in range(int(rng.integers(3, 5))):  # 3 or 4 operations
            k = int(rng.choice([1, 2, 2, 3]))
            machines = sorted(int(m) for m in rng.choice(n_machines, size=k, replace=False))
            base = int(rng.integers(8, 25))
            ops.append({m: max(1, int(round(base * rng.uniform(0.8, 1.3)))) for m in machines})
        routings.append(tuple(ops))

    setup = tuple(
        tuple(
            tuple(0 if f == g else int(rng.integers(3, 11)) for g in range(n_families))
            for f in range(n_families)
        )
        for _ in range(n_machines)
    )

    # layout: load station at the origin, machines on distinct cells of a 4 x 3 grid;
    # travel time = 1 min of handling + half the Manhattan distance, rounded up
    cells = [(x, y) for x in range(1, 5) for y in range(3)]
    picked = rng.choice(len(cells), size=n_machines, replace=False)
    pos = [(0, 1)] + [cells[int(i)] for i in picked]
    travel = tuple(
        tuple(0 if a == b else 1 + (abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) + 1) // 2 for b, pb in enumerate(pos))
        for a, pa in enumerate(pos)
    )
    return Workshop(n_machines, n_agvs, tuple(routings), setup, travel)


@dataclass(frozen=True)
class Job:
    idx: int
    family: int
    arrival: int
    due: int


@dataclass(frozen=True)
class Scenario:
    seed: int
    jobs: tuple[Job, ...]
    # downtime[machine] = ((t_down, t_up), ...), sorted, disjoint; hidden from the controllers
    downtime: tuple[tuple[tuple[int, int], ...], ...]
    # mean time to repair: the only thing a controller knows about a repair in progress
    mttr: float


# The two disturbance levels of the study: mean time between failures per machine (minutes).
LEVELS: dict[str, float | None] = {"no breakdowns": None, "frequent breakdowns": 120.0}


def mean_interarrival(ws: Workshop, load: float) -> float:
    """Inter-arrival time that loads the machines at `load` if every job took its fastest route."""
    mean_work = float(np.mean([ws.min_remaining_work(f, 0) for f in range(ws.n_families)]))
    return mean_work / (load * ws.n_machines)


def make_scenario(
    ws: Workshop,
    seed: int,
    n_jobs: int = 60,
    load: float = 0.60,
    mtbf: float | None = None,
    mttr: float = 20.0,
    tightness: tuple[float, float] = (2.0, 4.0),
) -> Scenario:
    """Poisson arrivals, uniform job mix, due date = arrival + k x shortest work (k uniform).

    Breakdowns: per machine, exponential time between a repair and the next failure (mean
    `mtbf`, calendar time), repair time uniform in [0.5, 1.5] x `mttr`. The arrival stream does
    not depend on `mtbf`: the same seed gives the same jobs at every disturbance level.
    """
    rng = np.random.default_rng([seed, 1])
    gaps = rng.exponential(mean_interarrival(ws, load), size=n_jobs)
    arrivals = np.floor(np.cumsum(gaps)).astype(int)
    families = rng.integers(0, ws.n_families, size=n_jobs)
    factors = rng.uniform(tightness[0], tightness[1], size=n_jobs)
    jobs = tuple(
        Job(i, int(f), int(a), int(a + round(k * ws.min_remaining_work(int(f), 0))))
        for i, (a, f, k) in enumerate(zip(arrivals, families, factors))
    )

    downtime: list[tuple[tuple[int, int], ...]] = []
    horizon = 4 * int(arrivals[-1]) + 5000  # far beyond any run
    for m in range(ws.n_machines):
        spans: list[tuple[int, int]] = []
        if mtbf is not None:
            brng = np.random.default_rng([seed, 2, m])
            t = 0
            while t < horizon:
                down = t + max(1, int(round(brng.exponential(mtbf))))
                up = down + max(1, int(round(brng.uniform(0.5, 1.5) * mttr)))
                spans.append((down, up))
                t = up
        downtime.append(tuple(spans))
    return Scenario(seed, jobs, tuple(downtime), mttr)
