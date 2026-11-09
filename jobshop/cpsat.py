"""CP-SAT rescheduling on a rolling horizon (OR-Tools).

At every arrival, breakdown and repair the controller takes a snapshot of the shop (where
each job is, what each machine is doing, when a broken machine is expected back), solves a
flexible job-shop model for the jobs currently in the shop, and executes the plan until the
next event: only the first decisions of each plan are ever applied.

Model (same patterns as the satellite planning model: optional intervals, NoOverlap,
sequence-dependent transition times):

* one start and one end per remaining operation, one optional interval per eligible machine,
  exactly one of them present;
* NoOverlap over the intervals of a machine, and a circuit over them whose arc (a, b) means
  "b directly follows a" and enforces  start_b >= max(end_a, arrival_b) + setup(family_a, family_b):
  as in the shop, a changeover starts only once the machine is free and the job is there;
* precedence inside a job with a transport lag that depends on the two machines chosen;
* objective: 10 x total tardiness + total completion time;
* warm start: a few greedy schedules and the previous plan, re-timed on the new snapshot, are
  built first; the best one is given to the solver as a complete hint, and is the fallback
  if the solver does not improve on it within its budget.

What the model does not see: the AGVs. A move between two stations is a fixed lag (loaded
travel time + an allowance for the AGV to come); the simulated shop moves real AGVs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from .rules import route_earliest_finish
from .scenario import Workshop
from .shop import JobRT, JobState, MachineRT, Shop


@dataclass
class PlanJob:
    idx: int
    family: int
    due: int
    first_op: int  # first operation that has not started
    ready: dict[int, int]  # eligible machine for that operation -> earliest start there


@dataclass
class Snapshot:
    now: int
    machine_free: list[int]  # earliest time each machine can start a new setup
    machine_family: list[int | None]  # family each machine is (or will be) set up for
    jobs: list[PlanJob]


@dataclass
class Plan:
    ops: dict[tuple[int, int], tuple[int, int, int]] = field(default_factory=dict)  # (job, op) -> (machine, start, end)
    objective: int | None = None
    status: str = "EMPTY"
    wall_s: float = 0.0


def transport_lag(ws: Workshop, a: int, b: int, allowance: int) -> int:
    """Lag the planner assumes between two machines (zero if the job stays on the machine)."""
    return 0 if a == b else allowance + ws.travel[ws.station(a)][ws.station(b)]


def plan_cost(snap: Snapshot, ws: Workshop, ops: dict[tuple[int, int], tuple[int, int, int]]) -> int:
    """Objective of the model: 10 x tardiness + completion time, summed over the planned jobs."""
    total = 0
    for job in snap.jobs:
        end = ops[(job.idx, ws.n_ops(job.family) - 1)][2]
        total += 10 * max(0, end - job.due) + end
    return total


def greedy_plan(ws: Workshop, snap: Snapshot, allowance: int, rule: str,
                previous_plan: dict[tuple[int, int], tuple[int, int, int]] | None = None) -> Plan:
    """List scheduling in the terms of the model: operations are appended one by one to machines.

    `rule` chooses the next (operation, machine) among the next operation of every job:
    "finish" = earliest end, "start" = earliest start then due date, "edd" = most urgent job,
    "previous" = same machines and same order as the previous plan, new operations afterwards.
    """
    free = list(snap.machine_free)
    family = list(snap.machine_family)
    next_op = {job.idx: job.first_op for job in snap.jobs}
    last_done: dict[int, tuple[int, int]] = {}  # job -> (machine, end) of its last scheduled operation
    ops: dict[tuple[int, int], tuple[int, int, int]] = {}
    todo = {job.idx: job for job in snap.jobs}
    while todo:
        best = None
        for job in todo.values():
            op = next_op[job.idx]
            for k, p in ws.alternatives(job.family, op).items():
                if op == job.first_op:
                    if k not in job.ready:
                        continue
                    arrive = job.ready[k]
                else:
                    k_prev, end_prev = last_done[job.idx]
                    arrive = end_prev + transport_lag(ws, k_prev, k, allowance)
                start = max(free[k], arrive) + ws.setup_time(k, family[k], job.family)
                end = start + p
                if rule == "finish":
                    key = (end, job.due, job.idx, k)
                elif rule == "start":
                    key = (start, job.due, end, job.idx, k)
                elif rule == "edd":
                    key = (job.due, job.idx, end, k)
                else:
                    old = previous_plan.get((job.idx, op)) if previous_plan else None
                    key = (0, old[1], job.idx, k) if old is not None and old[0] == k else (1, end, job.idx, k)
                if best is None or key < best[0]:
                    best = (key, job, op, k, start, end)
        _, job, op, k, start, end = best
        ops[(job.idx, op)] = (k, start, end)
        free[k], family[k] = end, job.family
        last_done[job.idx] = (k, end)
        next_op[job.idx] += 1
        if next_op[job.idx] == ws.n_ops(job.family):
            del todo[job.idx]
    return Plan(ops, plan_cost(snap, ws, ops), "GREEDY")


def warm_start(ws: Workshop, snap: Snapshot, allowance: int, previous_plan=None) -> Plan:
    candidates = [greedy_plan(ws, snap, allowance, rule) for rule in ("finish", "start", "edd")]
    if previous_plan:
        candidates.append(greedy_plan(ws, snap, allowance, "previous", previous_plan))
    return min(candidates, key=lambda plan: plan.objective)


def solve_snapshot(
    ws: Workshop,
    snap: Snapshot,
    *,
    allowance: int = 2,
    det_time: float = 0.2,
    wall_time_s: float = 20.0,
    workers: int = 8,
    seed: int = 0,
    previous_plan: dict[tuple[int, int], tuple[int, int, int]] | None = None,
    use_warm_start: bool = True,
    use_solver: bool = True,
) -> Plan:
    t0 = time.perf_counter()
    if not snap.jobs:
        return Plan()
    start_plan = warm_start(ws, snap, allowance, previous_plan) if use_warm_start else None
    if not use_solver:  # ablation: replan with the greedy schedules only
        start_plan.wall_s = time.perf_counter() - t0
        return start_plan
    m = cp_model.CpModel()

    def hint(var, value) -> None:
        if start_plan is not None:
            m.add_hint(var, int(value))

    horizon = max(snap.machine_free + [snap.now])
    max_setup = max(max(max(row) for row in mat) for mat in ws.setup)
    max_lag = allowance + max(max(row) for row in ws.travel)
    for job in snap.jobs:
        horizon = max(horizon, max(job.ready.values()))
    for job in snap.jobs:
        for op in range(job.first_op, ws.n_ops(job.family)):
            horizon += max(ws.alternatives(job.family, op).values()) + max_setup + max_lag

    on_machine: dict[int, list[tuple]] = {k: [] for k in range(ws.n_machines)}
    starts: dict[tuple[int, int], cp_model.IntVar] = {}
    ends: dict[tuple[int, int], cp_model.IntVar] = {}
    present: dict[tuple[int, int], dict[int, cp_model.IntVar]] = {}
    objective = []
    for job in snap.jobs:
        previous = None
        for op in range(job.first_op, ws.n_ops(job.family)):
            key = (job.idx, op)
            alternatives = ws.alternatives(job.family, op)
            if op == job.first_op:
                alternatives = {k: p for k, p in alternatives.items() if k in job.ready}
            s = m.new_int_var(snap.now, horizon, f"s{key}")
            e = m.new_int_var(snap.now, horizon, f"e{key}")
            a = m.new_int_var(snap.now, horizon, f"a{key}")  # arrival of the job at the chosen machine
            m.add(s >= a)
            lits = {}
            for k, p in alternatives.items():
                lit = m.new_bool_var(f"x{key}_{k}")
                iv = m.new_optional_interval_var(s, p, e, lit, f"iv{key}_{k}")
                lits[k] = lit
                on_machine[k].append((key, job.family, s, e, lit, iv, a))
            m.add_exactly_one(lits.values())
            if start_plan is not None:
                k0, s0, e0 = start_plan.ops[key]
                if previous is None:
                    a0 = job.ready[k0]
                else:
                    a0 = start_plan.ops[previous][2] + transport_lag(ws, start_plan.ops[previous][0], k0, allowance)
                hint(s, s0), hint(e, e0), hint(a, a0)
                for k, lit in lits.items():
                    hint(lit, k == k0)
            if previous is None:
                for k, lit in lits.items():
                    m.add(a >= job.ready[k]).only_enforce_if(lit)
            else:
                for k_prev, lit_prev in present[previous].items():
                    for k, lit in lits.items():
                        lag = transport_lag(ws, k_prev, k, allowance)
                        m.add(a >= ends[previous] + lag).only_enforce_if([lit_prev, lit])
            starts[key], ends[key], present[key] = s, e, lits
            previous = key
        tard = m.new_int_var(0, horizon, f"t{job.idx}")
        m.add(tard >= ends[previous] - job.due)
        if start_plan is not None:
            hint(tard, max(0, start_plan.ops[previous][2] - job.due))
        objective.append(10 * tard + ends[previous])

    for k, items in on_machine.items():
        if not items:
            continue
        m.add_no_overlap(item[5] for item in items)
        # circuit: node 0 is the machine's current state, node i + 1 the i-th candidate operation
        idle = m.new_bool_var(f"idle{k}")
        arcs = [(0, 0, idle)]
        # order of the warm-start plan on this machine: which arcs of the circuit it uses
        order = []
        if start_plan is not None:
            order = sorted((start_plan.ops[item[0]][1], i) for i, item in enumerate(items)
                           if start_plan.ops[item[0]][0] == k)
            order = [i for _, i in order]
            hint(idle, not order)
        successor = dict(zip(order, order[1:]))
        for i, (_, fam_i, s_i, e_i, lit_i, _, a_i) in enumerate(items):
            first = m.new_bool_var("")
            arcs.append((0, i + 1, first))
            hint(first, bool(order) and order[0] == i)
            setup = ws.setup_time(k, snap.machine_family[k], fam_i)
            m.add(s_i >= snap.machine_free[k] + setup).only_enforce_if(first)
            m.add(s_i >= a_i + setup).only_enforce_if(first)
            last = m.new_bool_var("")
            arcs.append((i + 1, 0, last))
            hint(last, bool(order) and order[-1] == i)
            arcs.append((i + 1, i + 1, lit_i.Not()))
            for j, (_, fam_j, s_j, _, _, _, a_j) in enumerate(items):
                if i == j:
                    continue
                follows = m.new_bool_var("")
                arcs.append((i + 1, j + 1, follows))
                hint(follows, successor.get(i) == j)
                setup = ws.setup_time(k, fam_i, fam_j)
                m.add(s_j >= e_i + setup).only_enforce_if(follows)
                if setup:
                    m.add(s_j >= a_j + setup).only_enforce_if(follows)
        m.add_circuit(arcs)

    m.minimize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.max_deterministic_time = det_time  # reproducible stop, unlike a wall-clock limit
    solver.parameters.max_time_in_seconds = wall_time_s  # safety net only
    solver.parameters.num_workers = workers
    solver.parameters.interleave_search = True  # the workers take turns: same result on every run
    solver.parameters.random_seed = seed
    status = solver.solve(m)
    plan = Plan(status=solver.status_name(status))
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        plan.objective = int(round(solver.objective_value))
        for key, lits in present.items():
            machine = next(k for k, lit in lits.items() if solver.boolean_value(lit))
            plan.ops[key] = (machine, int(solver.value(starts[key])), int(solver.value(ends[key])))
    if start_plan is not None and (plan.objective is None or plan.objective > start_plan.objective):
        plan = start_plan  # the solver did not reach its own starting point within the budget
    plan.wall_s = time.perf_counter() - t0
    return plan


def take_snapshot(shop: Shop, jobs: list[JobRT], allowance: int) -> Snapshot:
    ws = shop.ws
    machine_free = [shop.expected_free(m) for m in shop.machines]
    machine_family = [m.last_family for m in shop.machines]
    plan_jobs = []
    for jr in jobs:
        alternatives = ws.alternatives(jr.family, jr.op)
        if jr.state is JobState.PROCESSING:
            if jr.op + 1 == jr.n_ops:
                continue  # on its last operation: nothing left to decide
            nxt = ws.alternatives(jr.family, jr.op + 1)
            ready = {k: machine_free[jr.dest] + transport_lag(ws, jr.dest, k, allowance) for k in nxt}
            plan_jobs.append(PlanJob(jr.idx, jr.family, jr.due, jr.op + 1, ready))
            continue
        if jr.state is JobState.IN_TRANSIT:
            ready = {jr.dest: jr.eta}  # destination and arrival time are settled
        elif jr.state is JobState.WAIT_TRANSPORT:
            ready = {jr.dest: shop.now + allowance + ws.travel[jr.station][ws.station(jr.dest)]}
        else:  # QUEUED at a machine (it may move to another one) or NEW at its current station
            ready = {
                k: shop.now if ws.station(k) == jr.station else shop.now + allowance + ws.travel[jr.station][ws.station(k)]
                for k in alternatives
            }
        plan_jobs.append(PlanJob(jr.idx, jr.family, jr.due, jr.op, ready))
    return Snapshot(shop.now, machine_free, machine_family, plan_jobs)


class CPSATController:
    """Event-driven rolling-horizon rescheduling with CP-SAT."""

    name = "CP-SAT"

    def __init__(
        self,
        det_time: float = 0.2,
        allowance: int = 2,
        max_jobs: int = 25,
        strict: bool = True,
        workers: int = 8,
        seed: int = 0,
        use_warm_start: bool = True,
        use_solver: bool = True,
    ):
        self.use_warm_start = use_warm_start
        self.use_solver = use_solver
        self.det_time = det_time
        self.allowance = allowance
        self.max_jobs = max_jobs
        self.strict = strict
        self.workers = workers
        self.seed = seed
        self.plan: dict[tuple[int, int], tuple[int, int, int]] = {}
        self.sequence_of: dict[int, list[tuple[int, int, int]]] = {}  # machine -> [(start, job, op)]
        self.decision_time_s = 0.0
        self.n_replans = 0
        self.n_optimal = 0  # replans solved to proven optimality
        self.n_greedy = 0  # replans where the solver did not improve on the warm start
        self.n_capped = 0  # replans where more than `max_jobs` jobs were in the shop

    # ------------------------------------------------------------------ replanning
    def on_event(self, shop: Shop, kind: str, **info) -> None:
        t0 = time.perf_counter()
        jobs = sorted(shop.active_jobs(), key=lambda j: (j.due, j.idx))
        if len(jobs) > self.max_jobs:  # most urgent jobs are planned, the others follow EDD
            jobs = jobs[: self.max_jobs]
            self.n_capped += 1
        snap = take_snapshot(shop, jobs, self.allowance)
        plan = solve_snapshot(
            shop.ws, snap, allowance=self.allowance, det_time=self.det_time,
            workers=self.workers, seed=self.seed, previous_plan=self.plan,
            use_warm_start=self.use_warm_start, use_solver=self.use_solver,
        )
        self.n_replans += 1
        self.n_optimal += plan.status == "OPTIMAL"
        self.n_greedy += plan.status == "GREEDY"
        if plan.ops or not snap.jobs:
            self.plan = plan.ops
            self.sequence_of = {}
            for (job, op), (machine, start, _) in plan.ops.items():
                self.sequence_of.setdefault(machine, []).append((start, job, op))
            for seq in self.sequence_of.values():
                seq.sort()
        self.decision_time_s += time.perf_counter() - t0
        # queued jobs the new plan wants elsewhere are moved, then idle machines look again
        for m in shop.machines:
            for jr in list(m.queue):
                target = self.plan.get((jr.idx, jr.op))
                if target is not None and target[0] != m.idx:
                    shop.reroute(jr, target[0])
        shop.kick()

    # ------------------------------------------------------------------ executing the plan
    def route(self, shop: Shop, jr: JobRT) -> int:
        planned = self.plan.get((jr.idx, jr.op))
        if planned is not None:
            return planned[0]
        return route_earliest_finish(shop, jr)

    def sequence(self, shop: Shop, m: MachineRT) -> JobRT | None:
        seq = self.sequence_of.get(m.idx, [])
        while seq and self._started(shop, seq[0]):
            seq.pop(0)
        in_queue = {(j.idx, j.op): j for j in m.queue}
        if self.strict and seq:
            # follow the planned order on the machine, waiting for the next planned job if needed
            start, job, op = seq[0]
            if (job, op) in in_queue:
                return in_queue[(job, op)]
            unplanned = [j for j in m.queue if (j.idx, j.op) not in self.plan]
            fits = [
                j for j in unplanned
                if shop.now + shop.ws.setup_time(m.idx, m.last_family, j.family) + shop.ws.proc(j.family, j.op, m.idx) <= start
            ]
            return min(fits, key=lambda j: (j.due, j.idx)) if fits else None
        for _, job, op in seq:  # non-delay: earliest planned job among those present
            if (job, op) in in_queue:
                return in_queue[(job, op)]
        return min(m.queue, key=lambda j: (j.due, j.idx))

    @staticmethod
    def _started(shop: Shop, entry: tuple[int, int, int]) -> bool:
        _, job, op = entry
        jr = shop.jobs[job]
        return jr.op > op or (jr.op == op and jr.state in (JobState.PROCESSING, JobState.DONE))
