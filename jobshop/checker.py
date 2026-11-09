"""Independent schedule checker. Every run of every method goes through it.

It never looks at the simulator's internal state or at a solver model: it takes the
workshop data, the scenario (arrivals, downtime) and the executed log (machine work
segments, AGV legs) and re-derives every constraint from them.
"""
from __future__ import annotations

from collections import defaultdict

from .scenario import LOAD_STATION, Scenario, Workshop
from .shop import RunLog


def check_schedule(ws: Workshop, scenario: Scenario, log: RunLog) -> list[str]:
    """Return the list of violations (empty list = the executed schedule is valid)."""
    errors: list[str] = []
    jobs = {j.idx: j for j in scenario.jobs}

    by_op: dict[tuple[int, int], list] = defaultdict(list)
    by_machine: dict[int, list] = defaultdict(list)
    for s in log.segments:
        if s.end <= s.start:
            errors.append(f"segment with non-positive length: {s}")
        if s.kind not in ("setup", "proc"):
            errors.append(f"unknown segment kind: {s}")
        by_op[(s.job, s.op)].append(s)
        by_machine[s.machine].append(s)

    # 1. every operation of every job is done once, on one eligible machine, for its full duration
    op_machine: dict[tuple[int, int], int] = {}
    op_first: dict[tuple[int, int], int] = {}
    op_last: dict[tuple[int, int], int] = {}
    for job in jobs.values():
        for op in range(ws.n_ops(job.family)):
            segs = sorted(by_op.get((job.idx, op), []), key=lambda s: s.start)
            if not segs:
                errors.append(f"job {job.idx} op {op}: never processed")
                continue
            machines = {s.machine for s in segs}
            machine = segs[0].machine
            op_machine[(job.idx, op)] = machine
            op_first[(job.idx, op)] = segs[0].start
            op_last[(job.idx, op)] = segs[-1].end
            if len(machines) > 1:
                errors.append(f"job {job.idx} op {op}: split over machines {sorted(machines)}")
                continue
            alternatives = ws.alternatives(job.family, op)
            if machine not in alternatives:
                errors.append(f"job {job.idx} op {op}: machine {machine} is not eligible")
                continue
            done = sum(s.end - s.start for s in segs if s.kind == "proc")
            if done != alternatives[machine]:
                errors.append(f"job {job.idx} op {op}: processed {done} min, needs {alternatives[machine]}")
            kinds = [s.kind for s in segs]
            if "proc" in kinds and "setup" in kinds[kinds.index("proc"):]:
                errors.append(f"job {job.idx} op {op}: setup work after processing started")
    for job_idx, op in by_op:
        if job_idx not in jobs or op >= ws.n_ops(jobs[job_idx].family):
            errors.append(f"work logged for unknown operation ({job_idx}, {op})")

    # 2. one thing at a time on a machine, no work while down, setup times respected
    for machine, segs in by_machine.items():
        segs.sort(key=lambda s: (s.start, s.end))
        for a, b in zip(segs, segs[1:]):
            if b.start < a.end:
                errors.append(f"machine {machine}: overlap between {a} and {b}")
        for s in segs:
            for down, up in scenario.downtime[machine]:
                if s.start < up and down < s.end:
                    errors.append(f"machine {machine}: work during downtime [{down}, {up}): {s}")
        order: list[tuple[int, int]] = []  # operations in the order the machine ran them
        for s in segs:
            if not order or order[-1] != (s.job, s.op):
                order.append((s.job, s.op))
        if len(order) != len(set(order)):
            errors.append(f"machine {machine}: an operation was interleaved with another one")
        previous_family = None
        for job_idx, op in order:
            if job_idx not in jobs:
                continue
            family = jobs[job_idx].family
            needed = ws.setup_time(machine, previous_family, family)
            done = sum(s.end - s.start for s in by_op[(job_idx, op)] if s.kind == "setup" and s.machine == machine)
            if done != needed:
                errors.append(
                    f"machine {machine}: job {job_idx} op {op} got {done} min of setup, needs {needed} "
                    f"(family {previous_family} -> {family})"
                )
            previous_family = family

    # 3. each AGV follows a continuous path in space and time, at the travel times of the layout
    legs_by_job: dict[int, list] = defaultdict(list)
    legs_by_agv: dict[int, list] = defaultdict(list)
    for leg in log.legs:
        legs_by_agv[leg.agv].append(leg)
        if leg.job is not None:
            legs_by_job[leg.job].append(leg)
    for agv, legs in legs_by_agv.items():
        if not 0 <= agv < ws.n_agvs:
            errors.append(f"unknown AGV {agv}")
        legs.sort(key=lambda g: (g.start, g.end))
        position, free_at = LOAD_STATION, 0
        for leg in legs:
            if leg.origin == leg.dest or leg.end - leg.start != ws.travel[leg.origin][leg.dest]:
                errors.append(f"AGV {agv}: leg duration does not match the layout: {leg}")
            if leg.start < free_at:
                errors.append(f"AGV {agv}: in two places at once, leg starts before the previous one ends: {leg}")
            if leg.origin != position:
                errors.append(f"AGV {agv}: leg starts at station {leg.origin} but the AGV is at {position}: {leg}")
            position, free_at = leg.dest, leg.end

    # 4. precedence inside a job, with the job physically carried from station to station
    used_legs = 0
    for job in jobs.values():
        legs = sorted(legs_by_job.get(job.idx, []), key=lambda g: g.start)
        position, available = LOAD_STATION, job.arrival
        for op in range(ws.n_ops(job.family)):
            key = (job.idx, op)
            if key not in op_machine:
                break
            first = op_first[key]
            if first < available:
                errors.append(f"job {job.idx} op {op}: starts at {first}, before the job is available ({available})")
            moves = [g for g in legs if available <= g.start and g.end <= first]
            used_legs += len(moves)
            for leg in moves:
                if leg.origin != position:
                    errors.append(f"job {job.idx}: picked up at station {leg.origin} but it is at {position}: {leg}")
                if leg.start < available:
                    errors.append(f"job {job.idx}: carried before it is available: {leg}")
                position, available = leg.dest, leg.end
            if position != ws.station(op_machine[key]):
                errors.append(
                    f"job {job.idx} op {op}: processed at station {ws.station(op_machine[key])} "
                    f"but the job is at station {position}"
                )
                position = ws.station(op_machine[key])
            available = op_last[key]
    if used_legs != sum(len(v) for v in legs_by_job.values()):
        errors.append("some loaded AGV legs do not fit between two operations of the job they carry")
    return errors
