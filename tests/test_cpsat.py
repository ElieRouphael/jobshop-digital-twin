"""CP-SAT rescheduling: the model against brute force on tiny snapshots, then inside the shop."""
import itertools

import pytest

from jobshop.checker import check_schedule
from jobshop.cpsat import (
    CPSATController,
    PlanJob,
    Snapshot,
    greedy_plan,
    plan_cost,
    solve_snapshot,
    transport_lag,
    warm_start,
)
from jobshop.scenario import Workshop, make_scenario, make_workshop
from jobshop.shop import simulate

WS = make_workshop()


def one_op_workshop(n_machines: int) -> Workshop:
    """Three families with a single operation each, asymmetric changeovers, no transport."""
    procs = (7, 4, 9)
    routings = tuple(({m: p + 2 * m for m in range(n_machines)},) for p in procs)
    setup = tuple(((0, 5, 2), (3, 0, 6), (4, 1, 0)) for _ in range(n_machines))
    n = n_machines + 1
    travel = tuple(tuple(0 if a == b else 2 for b in range(n)) for a in range(n))
    return Workshop(n_machines, 1, routings, setup, travel)


def cost_of_sequence(ws, machine, start, family0, sequence):
    """10 x tardiness + completion time of the jobs of one machine processed in the given order."""
    t, family, cost = start, family0, 0
    for job in sequence:
        # the changeover starts when the machine is free and the job is there
        t = max(t, job.ready[machine]) + ws.setup_time(machine, family, job.family) + ws.proc(job.family, 0, machine)
        family = job.family
        cost += 10 * max(0, t - job.due) + t
    return cost


def brute_force(ws, snap):
    """Best objective over every assignment of jobs to machines and every order on each machine."""
    best = None
    machines = range(ws.n_machines)
    for assignment in itertools.product(machines, repeat=len(snap.jobs)):
        total = 0
        for m in machines:
            mine = [j for j, a in zip(snap.jobs, assignment) if a == m]
            total += min(
                cost_of_sequence(ws, m, snap.machine_free[m], snap.machine_family[m], order)
                for order in itertools.permutations(mine)
            )
        best = total if best is None else min(best, total)
    return best


def snapshot(ws, families, dues, readies, machine_free=None, machine_family=None):
    jobs = [PlanJob(i, f, d, 0, {m: r for m in range(ws.n_machines)})
            for i, (f, d, r) in enumerate(zip(families, dues, readies))]
    return Snapshot(0, machine_free or [0] * ws.n_machines, machine_family or [None] * ws.n_machines, jobs)


def assert_plan_valid(ws, snap, ops, allowance):
    """Check a plan against the constraints of the model, without using the model."""
    assert len(ops) == sum(ws.n_ops(j.family) - j.first_op for j in snap.jobs)
    arrival = {}
    for job in snap.jobs:
        for op in range(job.first_op, ws.n_ops(job.family)):
            machine, start, end = ops[(job.idx, op)]
            assert end - start == ws.proc(job.family, op, machine)  # eligible machine, full duration
            if op == job.first_op:
                arrival[(job.idx, op)] = job.ready[machine]  # KeyError if the destination was not allowed
            else:
                prev_machine, _, prev_end = ops[(job.idx, op - 1)]
                arrival[(job.idx, op)] = prev_end + transport_lag(ws, prev_machine, machine, allowance)
    family = {j.idx: j.family for j in snap.jobs}
    for m in range(ws.n_machines):
        seq = sorted((start, end, job, op) for (job, op), (machine, start, end) in ops.items() if machine == m)
        previous_end, previous_family = snap.machine_free[m], snap.machine_family[m]
        for start, end, job, op in seq:
            assert start >= max(previous_end, arrival[(job, op)]) + ws.setup_time(m, previous_family, family[job])
            previous_end, previous_family = end, family[job]


SINGLE = [
    dict(families=[0, 1, 2, 1], dues=[12, 30, 18, 9], readies=[0, 0, 0, 0]),
    dict(families=[2, 2, 0, 1, 0], dues=[40, 10, 25, 15, 60], readies=[0, 3, 0, 6, 1]),
    dict(families=[1, 0, 1, 2, 0], dues=[8, 8, 20, 20, 35], readies=[2, 0, 0, 0, 10]),
]
DOUBLE = [
    dict(families=[0, 1, 2, 1, 0], dues=[12, 30, 18, 9, 25], readies=[0, 0, 0, 0, 5]),
    dict(families=[2, 2, 0, 1, 0, 1], dues=[40, 10, 25, 15, 30, 12], readies=[0, 3, 0, 6, 1, 0]),
]


@pytest.mark.parametrize("case", SINGLE)
@pytest.mark.parametrize("warm", [True, False])
def test_single_machine_optimum_matches_brute_force(case, warm):
    ws = one_op_workshop(1)
    snap = snapshot(ws, machine_free=[4], machine_family=[1], **case)
    plan = solve_snapshot(ws, snap, det_time=20.0, use_warm_start=warm)
    assert plan.status == "OPTIMAL"
    assert plan.objective == brute_force(ws, snap) == plan_cost(snap, ws, plan.ops)
    assert_plan_valid(ws, snap, plan.ops, allowance=2)


@pytest.mark.parametrize("case", DOUBLE)
@pytest.mark.parametrize("warm", [True, False])
def test_two_machine_optimum_matches_brute_force(case, warm):
    ws = one_op_workshop(2)
    snap = snapshot(ws, machine_free=[0, 6], machine_family=[2, None], **case)
    plan = solve_snapshot(ws, snap, det_time=20.0, use_warm_start=warm)
    assert plan.status == "OPTIMAL"
    assert plan.objective == brute_force(ws, snap) == plan_cost(snap, ws, plan.ops)
    assert_plan_valid(ws, snap, plan.ops, allowance=2)


def real_snapshot():
    jobs = [PlanJob(i, f, due, 0, {m: 3 * i for m in WS.alternatives(f, 0)})
            for i, (f, due) in enumerate([(0, 80), (1, 60), (2, 90), (3, 70), (0, 100), (2, 50), (1, 75), (3, 120)])]
    jobs.append(PlanJob(8, 0, 95, 2, {4: 20}))  # two operations already done, on its way to machine 4
    return Snapshot(0, [0, 5, 0, 12, 0], [None, 0, 3, None, 1], jobs)


def test_plan_respects_machine_setup_and_job_constraints():
    snap = real_snapshot()
    plan = solve_snapshot(WS, snap, allowance=2, det_time=0.5)
    assert_plan_valid(WS, snap, plan.ops, allowance=2)
    assert plan.ops[(8, 2)][0] == 4 and plan.ops[(8, 2)][1] >= 20  # frozen destination and arrival time kept


@pytest.mark.parametrize("rule", ["finish", "start", "edd"])
def test_greedy_plans_are_valid_and_never_better_than_the_solver(rule):
    snap = real_snapshot()
    greedy = greedy_plan(WS, snap, 2, rule)
    assert_plan_valid(WS, snap, greedy.ops, allowance=2)
    solved = solve_snapshot(WS, snap, allowance=2, det_time=0.5)
    assert solved.objective <= warm_start(WS, snap, 2).objective <= greedy.objective


def test_previous_plan_is_reused_as_a_starting_point():
    snap = real_snapshot()
    first = solve_snapshot(WS, snap, allowance=2, det_time=0.5)
    again = greedy_plan(WS, snap, 2, "previous", first.ops)
    assert_plan_valid(WS, snap, again.ops, allowance=2)
    assert again.objective <= first.objective  # same machines and order, re-timed: cannot be worse
    for key, (machine, _, _) in first.ops.items():
        assert again.ops[key][0] == machine


def test_without_budget_the_warm_start_is_returned():
    snap = real_snapshot()
    plan = solve_snapshot(WS, snap, allowance=2, det_time=1e-6)
    assert plan.objective <= warm_start(WS, snap, 2).objective
    assert_plan_valid(WS, snap, plan.ops, allowance=2)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("mtbf", [None, 80.0])
def test_controller_produces_valid_schedules(strict, mtbf):
    scenario = make_scenario(WS, 77, n_jobs=12, mtbf=mtbf)
    controller = CPSATController(det_time=0.02, strict=strict)
    log = simulate(WS, scenario, controller)
    assert check_schedule(WS, scenario, log) == []
    assert controller.n_replans >= len(scenario.jobs)  # one per arrival, plus breakdowns and repairs


def test_replanning_is_triggered_by_breakdowns_and_repairs():
    calm = CPSATController(det_time=0.01)
    rough = CPSATController(det_time=0.01)
    simulate(WS, make_scenario(WS, 78, n_jobs=10), calm)
    simulate(WS, make_scenario(WS, 78, n_jobs=10, mtbf=40.0), rough)
    assert calm.n_replans == 10
    assert rough.n_replans > calm.n_replans


def test_controller_is_deterministic():
    """The solver stops on a deterministic-time limit, so two runs give the same schedule."""
    scenario = make_scenario(WS, 79, n_jobs=10, mtbf=80.0)
    a = simulate(WS, scenario, CPSATController(det_time=0.05))
    b = simulate(WS, scenario, CPSATController(det_time=0.05))
    assert a.segments == b.segments and a.legs == b.legs


def test_job_cap_falls_back_to_a_rule_and_stays_valid():
    scenario = make_scenario(WS, 80, n_jobs=15, load=1.5)  # overloaded on purpose
    controller = CPSATController(det_time=0.01, max_jobs=3)
    log = simulate(WS, scenario, controller)
    assert controller.n_capped > 0
    assert check_schedule(WS, scenario, log) == []


def test_greedy_replanning_without_the_solver_is_valid():
    scenario = make_scenario(WS, 81, n_jobs=30, mtbf=80.0)
    controller = CPSATController(use_solver=False)
    log = simulate(WS, scenario, controller)
    assert check_schedule(WS, scenario, log) == []
    assert 30 < controller.n_greedy <= controller.n_replans  # a replan with nothing left to plan is not counted
    assert controller.n_optimal == 0
