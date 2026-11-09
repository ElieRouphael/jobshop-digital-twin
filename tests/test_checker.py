"""The checker must catch every kind of violation: each test breaks a valid log in one way."""
from dataclasses import replace

import pytest

from jobshop.checker import check_schedule
from jobshop.rules import RuleController
from jobshop.scenario import make_scenario, make_workshop
from jobshop.shop import Leg, RunLog, simulate

WS = make_workshop()
SCENARIO = make_scenario(WS, 42, n_jobs=40, mtbf=120.0)


@pytest.fixture()
def log() -> RunLog:
    valid = simulate(WS, SCENARIO, RuleController("EDD"))
    assert check_schedule(WS, SCENARIO, valid) == []
    return valid


def machine_sequence(log: RunLog, machine: int):
    return sorted((s for s in log.segments if s.machine == machine), key=lambda s: s.start)


def swap(items: list, old, new) -> None:
    items[items.index(old)] = new


def has(errors: list[str], word: str) -> bool:
    return any(word in e for e in errors)


def test_overlap_on_a_machine(log):
    segs = machine_sequence(log, 0)
    a, b = segs[3], segs[4]
    swap(log.segments, b, replace(b, start=a.start, end=a.start + (b.end - b.start)))
    assert has(check_schedule(WS, SCENARIO, log), "overlap")


def test_precedence_inside_a_job(log):
    """Second operation of a job moved before the end of its first operation."""
    job = next(j for j in SCENARIO.jobs)
    first = [s for s in log.segments if s.job == job.idx and s.op == 0]
    for s in [s for s in log.segments if s.job == job.idx and s.op == 1]:
        shift = s.start - min(x.start for x in first)
        swap(log.segments, s, replace(s, start=s.start - shift, end=s.end - shift))
    errors = check_schedule(WS, SCENARIO, log)
    assert has(errors, "before the job is available") or has(errors, "but the job is at station")


def test_operation_starts_before_the_job_is_delivered(log):
    leg = next(g for g in log.legs if g.job is not None)
    first = min((s for s in log.segments if s.job == leg.job and s.op == 0), key=lambda s: s.start)
    swap(log.segments, first, replace(first, start=leg.start))
    assert check_schedule(WS, SCENARIO, log) != []


def test_setup_time_shortened(log):
    s = next(s for s in log.segments if s.kind == "setup" and s.end - s.start >= 2)
    swap(log.segments, s, replace(s, start=s.start + 1))
    assert has(check_schedule(WS, SCENARIO, log), "of setup, needs")


def test_setup_skipped(log):
    s = next(s for s in log.segments if s.kind == "setup")
    log.segments[:] = [x for x in log.segments if not (x.job == s.job and x.op == s.op and x.kind == "setup")]
    assert has(check_schedule(WS, SCENARIO, log), "of setup, needs")


def test_processing_time_shortened(log):
    s = next(s for s in log.segments if s.kind == "proc")
    swap(log.segments, s, replace(s, end=s.end - 1))
    assert has(check_schedule(WS, SCENARIO, log), "processed")


def test_work_during_downtime(log):
    machine = next(m for m in range(WS.n_machines) if SCENARIO.downtime[m])
    down, up = SCENARIO.downtime[machine][0]
    before = max((s for s in log.segments if s.machine == machine and s.end <= down), key=lambda s: s.end)
    swap(log.segments, before, replace(before, end=down + 1))
    assert has(check_schedule(WS, SCENARIO, log), "downtime")


def test_ineligible_machine(log):
    job = next(j for j in SCENARIO.jobs if len(WS.alternatives(j.family, 0)) < WS.n_machines)
    wrong = next(m for m in range(WS.n_machines) if m not in WS.alternatives(job.family, 0))
    for s in [s for s in log.segments if s.job == job.idx and s.op == 0]:
        swap(log.segments, s, replace(s, machine=wrong))
    assert has(check_schedule(WS, SCENARIO, log), "not eligible")


def test_missing_operation(log):
    log.segments[:] = [s for s in log.segments if not (s.job == 5 and s.op == 1)]
    assert has(check_schedule(WS, SCENARIO, log), "never processed")


def test_agv_in_two_places(log):
    """A leg starts while the same AGV is still on its previous leg."""
    legs = sorted((g for g in log.legs if g.agv == 0), key=lambda g: g.start)
    a, b = legs[4], legs[5]
    swap(log.legs, b, replace(b, start=a.start, end=a.start + (b.end - b.start)))
    assert has(check_schedule(WS, SCENARIO, log), "in two places")


def test_agv_teleports(log):
    legs = sorted((g for g in log.legs if g.agv == 1), key=lambda g: g.start)
    b = legs[3]
    other = next(st for st in range(WS.n_machines + 1) if st not in (b.origin, b.dest))
    swap(log.legs, b, replace(b, origin=other, end=b.start + WS.travel[other][b.dest]))
    assert has(check_schedule(WS, SCENARIO, log), "but the AGV is at")


def test_agv_faster_than_the_layout_allows(log):
    b = next(g for g in log.legs if g.end - g.start >= 2)
    swap(log.legs, b, replace(b, end=b.end - 1))
    assert has(check_schedule(WS, SCENARIO, log), "leg duration")


def test_job_moved_without_an_agv(log):
    """Remove a loaded leg: the job is then processed at a station it never reached."""
    leg = next(g for g in log.legs if g.job is not None and g.origin != 0)
    log.legs.remove(leg)
    errors = check_schedule(WS, SCENARIO, log)
    assert has(errors, "but the job is at station")


def test_two_jobs_on_one_agv_leg_is_impossible_by_construction(log):
    """A second job on an overlapping leg of the same AGV is the 'two places' violation."""
    leg = next(g for g in log.legs if g.job is not None)
    log.legs.append(Leg(leg.agv, leg.origin, leg.dest, leg.start, leg.end, leg.job + 1))
    assert check_schedule(WS, SCENARIO, log) != []
