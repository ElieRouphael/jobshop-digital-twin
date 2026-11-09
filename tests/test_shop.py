"""Simulator: hand-made cases with a known answer, determinism, and the checker on real runs."""
import pytest

from jobshop.checker import check_schedule
from jobshop.metrics import bootstrap_ci, run_metrics
from jobshop.rules import RULES, RuleController
from jobshop.scenario import LEVELS, Job, Scenario, Workshop, make_scenario, make_workshop
from jobshop.shop import Leg, Segment, simulate

WS = make_workshop()


def tiny_workshop(n_agvs=1):
    """One machine at 2 min from the load station, two families (10 and 6 min), 4 min changeover."""
    return Workshop(
        n_machines=1, n_agvs=n_agvs,
        routings=(({0: 10},), ({0: 6},)),
        setup=(((0, 4), (4, 0)),),
        travel=((0, 2), (2, 0)),
    )


# ---------------------------------------------------------------- hand-made cases
def test_breakdown_suspends_the_operation_and_repair_resumes_it():
    ws = tiny_workshop()
    scenario = Scenario(0, (Job(0, 0, 0, 30),), (((5, 9),),), mttr=4.0)
    log = simulate(ws, scenario, RuleController("FIFO"))
    # carried from 0 to 2, processed 2-5, machine down 5-9, processed 9-16 (3 + 7 = 10 min)
    assert log.legs == [Leg(0, 0, 1, 0, 2, 0)]
    assert log.segments == [Segment(0, 0, 0, "proc", 2, 5), Segment(0, 0, 0, "proc", 9, 16)]
    assert check_schedule(ws, scenario, log) == []
    metrics = run_metrics(ws, scenario, log)
    assert metrics["makespan"] == 16 and metrics["mean_tardiness"] == 0
    assert metrics["utilisation"] == pytest.approx(10 / 16)


def test_setup_time_and_single_agv_shuttle():
    ws = tiny_workshop()
    scenario = Scenario(0, (Job(0, 0, 0, 12), Job(1, 1, 0, 20)), ((),), mttr=20.0)
    log = simulate(ws, scenario, RuleController("FIFO"))
    # the only AGV carries job 0 (0-2), comes back empty (2-4), carries job 1 (4-6)
    assert log.legs == [Leg(0, 0, 1, 0, 2, 0), Leg(0, 1, 0, 2, 4, None), Leg(0, 0, 1, 4, 6, 1)]
    # job 0: 2-12; changeover to family 1: 12-16; job 1: 16-22
    assert log.segments == [
        Segment(0, 0, 0, "proc", 2, 12),
        Segment(0, 1, 0, "setup", 12, 16),
        Segment(0, 1, 0, "proc", 16, 22),
    ]
    assert check_schedule(ws, scenario, log) == []
    metrics = run_metrics(ws, scenario, log)
    assert metrics["mean_tardiness"] == pytest.approx((0 + 2) / 2)  # job 1 due at 20, done at 22
    assert metrics["max_tardiness"] == 2 and metrics["setup_time"] == 4


def test_breakdown_during_setup_resumes_the_setup():
    ws = tiny_workshop(n_agvs=2)
    scenario = Scenario(0, (Job(0, 0, 0, 99), Job(1, 1, 0, 99)), (((13, 20),),), mttr=7.0)
    log = simulate(ws, scenario, RuleController("FIFO"))
    assert [s for s in log.segments if s.job == 1] == [
        Segment(0, 1, 0, "setup", 12, 13),
        Segment(0, 1, 0, "setup", 20, 23),
        Segment(0, 1, 0, "proc", 23, 29),
    ]
    assert check_schedule(ws, scenario, log) == []


def test_edd_and_spt_order_the_queue_differently():
    ws = tiny_workshop(n_agvs=2)
    # job 0 keeps the machine busy; jobs 1 (long, urgent) and 2 (short, not urgent) wait
    jobs = (Job(0, 0, 0, 99), Job(1, 0, 1, 30), Job(2, 1, 1, 99))
    scenario = Scenario(0, jobs, ((),), mttr=20.0)
    order = {}
    for rule in ("EDD", "SPT"):
        log = simulate(ws, scenario, RuleController(rule))
        assert check_schedule(ws, scenario, log) == []
        order[rule] = [s.job for s in log.segments if s.kind == "proc"]
    assert order["EDD"] == [0, 1, 2]
    assert order["SPT"] == [0, 2, 1]


# ---------------------------------------------------------------- scenarios
def test_scenario_is_deterministic_per_seed():
    assert make_scenario(WS, 3, mtbf=120.0) == make_scenario(WS, 3, mtbf=120.0)
    assert make_scenario(WS, 3).jobs != make_scenario(WS, 4).jobs


def test_same_jobs_at_both_disturbance_levels():
    calm, rough = make_scenario(WS, 7), make_scenario(WS, 7, mtbf=120.0)
    assert calm.jobs == rough.jobs
    assert all(not spans for spans in calm.downtime) and any(rough.downtime)


def test_downtime_spans_are_sorted_and_disjoint():
    for spans in make_scenario(WS, 1, mtbf=120.0).downtime:
        assert all(down < up for down, up in spans)
        assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))


def test_workshop_data_is_consistent():
    assert all(WS.travel[a][a] == 0 for a in range(WS.n_machines + 1))
    assert all(WS.travel[a][b] == WS.travel[b][a] > 0 for a in range(6) for b in range(a))
    assert all(WS.setup[m][f][f] == 0 for m in range(WS.n_machines) for f in range(WS.n_families))
    assert all(1 <= len(alt) <= 3 for routing in WS.routings for alt in routing)


# ---------------------------------------------------------------- full runs
def test_simulation_is_deterministic_per_seed():
    scenario = make_scenario(WS, 11, n_jobs=30, mtbf=120.0)
    a = simulate(WS, scenario, RuleController("EDD"))
    b = simulate(WS, scenario, RuleController("EDD"))
    assert a.segments == b.segments and a.legs == b.legs


@pytest.mark.parametrize("rule", list(RULES))
@pytest.mark.parametrize("level", list(LEVELS))
def test_rules_produce_valid_schedules(rule, level):
    for seed in (21, 22, 23):
        scenario = make_scenario(WS, seed, n_jobs=40, mtbf=LEVELS[level])
        log = simulate(WS, scenario, RuleController(rule))
        assert check_schedule(WS, scenario, log) == []
        done = {(s.job, s.op) for s in log.segments}
        assert len(done) == sum(WS.n_ops(j.family) for j in scenario.jobs)


def test_breakdowns_never_help():
    """Same jobs, same rule: with breakdowns the last job cannot leave earlier on average."""
    calm = rough = 0.0
    for seed in range(30, 36):
        for mtbf, label in ((None, "calm"), (120.0, "rough")):
            scenario = make_scenario(WS, seed, n_jobs=40, mtbf=mtbf)
            value = run_metrics(WS, scenario, simulate(WS, scenario, RuleController("EDD")))["mean_flow_time"]
            if label == "calm":
                calm += value
            else:
                rough += value
    assert rough > calm


# ---------------------------------------------------------------- statistics
def test_bootstrap_interval_brackets_the_mean_and_shrinks_with_more_data():
    import numpy as np

    rng = np.random.default_rng(0)
    small, large = rng.normal(10, 2, 20), rng.normal(10, 2, 2000)
    lo_s, hi_s = bootstrap_ci(small)
    lo_l, hi_l = bootstrap_ci(large)
    assert lo_s < small.mean() < hi_s and lo_l < large.mean() < hi_l
    assert hi_l - lo_l < hi_s - lo_s
    assert bootstrap_ci([1.0, 1.0, 1.0]) == (1.0, 1.0)
