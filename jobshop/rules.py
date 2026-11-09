"""Priority dispatching rules: the baseline every other method is compared with.

Routing is the same for all rules (and for the DQN agent): send the job to the eligible
machine where it is expected to finish first. Sequencing is the rule itself. When a
machine breaks down, the jobs waiting in its queue are routed again and move if another
machine now looks better.
"""
from __future__ import annotations

import time

from .shop import JobRT, MachineRT, Shop


# ---------------------------------------------------------------------- sequencing rules
def _fifo(shop: Shop, m: MachineRT, j: JobRT):
    return (j.queued_at, j.idx)


def _spt(shop: Shop, m: MachineRT, j: JobRT):
    return (shop.ws.proc(j.family, j.op, m.idx), j.idx)


def _edd(shop: Shop, m: MachineRT, j: JobRT):
    return (j.due, j.idx)


def _sst(shop: Shop, m: MachineRT, j: JobRT):
    return (shop.ws.setup_time(m.idx, m.last_family, j.family), j.due, j.idx)


# FIFO: first in the queue; SPT: shortest processing time; EDD: earliest due date;
# SST: shortest setup time (stay on the current family if possible), ties by due date.
RULES = {"FIFO": _fifo, "SPT": _spt, "EDD": _edd, "SST": _sst}


def pick(rule: str, shop: Shop, m: MachineRT) -> JobRT:
    key = RULES[rule]
    return min(m.queue, key=lambda j: key(shop, m, j))


# ---------------------------------------------------------------------- routing rule
def workload(shop: Shop, m: MachineRT, skip: JobRT | None = None) -> int:
    """Processing time committed to a machine: queued and inbound jobs, without `skip`."""
    return sum(shop.ws.proc(j.family, j.op, m.idx) for j in m.queue + m.inbound if j is not skip)


def route_earliest_finish(shop: Shop, jr: JobRT) -> int:
    """Eligible machine with the earliest expected finish (travel + work ahead + processing)."""
    best, best_key = None, None
    for machine, proc in shop.ws.alternatives(jr.family, jr.op).items():
        m = shop.machines[machine]
        arrival = shop.now + shop.ws.travel[jr.station][shop.ws.station(machine)]
        free = shop.expected_free(m) + workload(shop, m, skip=jr)
        key = (max(arrival, free) + proc, proc, machine)
        if best_key is None or key < best_key:
            best, best_key = machine, key
    return best


class RuleController:
    """One sequencing rule + earliest-finish routing + rerouting of the queue on a breakdown."""

    def __init__(self, rule: str):
        self.name = rule
        self.rule = rule
        self.decision_time_s = 0.0
        self.n_replans = 0

    def route(self, shop: Shop, jr: JobRT) -> int:
        t0 = time.perf_counter()
        machine = route_earliest_finish(shop, jr)
        self.decision_time_s += time.perf_counter() - t0
        return machine

    def sequence(self, shop: Shop, m: MachineRT) -> JobRT | None:
        t0 = time.perf_counter()
        jr = pick(self.rule, shop, m)
        self.decision_time_s += time.perf_counter() - t0
        return jr

    def on_event(self, shop: Shop, kind: str, **info) -> None:
        if kind != "breakdown":
            return
        t0 = time.perf_counter()
        m = info["machine"]
        for jr in list(m.queue):
            target = route_earliest_finish(shop, jr)
            if target != m.idx:
                shop.reroute(jr, target)
        self.decision_time_s += time.perf_counter() - t0
