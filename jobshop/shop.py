"""Discrete-event simulation of the workshop (SimPy). It plays the role of the real shop.

Processes: one for the arrivals, one per machine for breakdowns and repairs, one per
operation being set up and processed, one per AGV mission. The shop takes no scheduling
decision itself: it asks a controller

* `route(shop, job)`      which eligible machine does the job's next operation,
* `sequence(shop, machine)` which queued job an idle machine starts (None = stay idle),

and tells it about `arrival`, `breakdown` and `repair` events through `on_event`, where the
controller may replan and reroute queued jobs with `shop.reroute`.

AGV dispatching is the same for every controller: transport requests are served first come
first served by the nearest idle AGV.

Everything that happens is written to a `RunLog` (machine work segments and AGV legs). The
checker and the metrics are computed from that log only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import simpy

from .scenario import LOAD_STATION, Job, Scenario, Workshop


class JobState(Enum):
    NEW = "new"  # needs a routing decision
    WAIT_TRANSPORT = "wait_transport"  # destination fixed, waiting for an AGV
    IN_TRANSIT = "in_transit"
    QUEUED = "queued"  # in the input buffer of a machine
    PROCESSING = "processing"  # setup or processing, possibly interrupted by a breakdown
    DONE = "done"


@dataclass(frozen=True)
class Segment:
    """Uninterrupted piece of setup or processing work on a machine."""

    machine: int
    job: int
    op: int
    kind: str  # "setup" or "proc"
    start: int
    end: int


@dataclass(frozen=True)
class Leg:
    """One AGV move between two stations; `job` is None when the AGV travels empty."""

    agv: int
    origin: int
    dest: int
    start: int
    end: int
    job: int | None


@dataclass
class RunLog:
    segments: list[Segment] = field(default_factory=list)
    legs: list[Leg] = field(default_factory=list)


class JobRT:
    """Run-time state of a job in the shop."""

    def __init__(self, job: Job, ws: Workshop):
        self.job = job
        self.idx = job.idx
        self.family = job.family
        self.due = job.due
        self.n_ops = ws.n_ops(job.family)
        self.op = 0  # next operation to do, or the one in progress
        self.state = JobState.NEW
        self.station = LOAD_STATION  # where the job physically is (origin while in transit)
        self.dest: int | None = None  # machine it is going to, queued at, or processed on
        self.queued_at = 0  # time it entered its current queue
        self.eta: int | None = None  # arrival time at `dest`, known once on an AGV
        self.completion: int | None = None


class MachineRT:
    def __init__(self, idx: int):
        self.idx = idx
        self.queue: list[JobRT] = []
        self.inbound: list[JobRT] = []  # jobs on their way to this machine
        self.current: JobRT | None = None
        self.last_family: int | None = None  # family the machine is set up for
        self.down = False
        self.down_since = 0
        self.up_event: simpy.Event | None = None
        self.proc: simpy.Process | None = None
        self.rem_setup = 0
        self.rem_proc = 0
        self.seg_kind = ""
        self.seg_start: int | None = None  # start of the running work segment, if any


class AGVRT:
    def __init__(self, idx: int):
        self.idx = idx
        self.station = LOAD_STATION
        self.idle = True


class Shop:
    def __init__(self, ws: Workshop, scenario: Scenario, controller):
        self.ws = ws
        self.scenario = scenario
        self.controller = controller
        self.env = simpy.Environment()
        self.machines = [MachineRT(m) for m in range(ws.n_machines)]
        self.agvs = [AGVRT(a) for a in range(ws.n_agvs)]
        self.jobs: dict[int, JobRT] = {}  # every job that has arrived
        self.requests: list[JobRT] = []  # pending transport requests, oldest first
        self.log = RunLog()
        self.n_done = 0
        self._all_done = self.env.event()

    # ------------------------------------------------------------------ what a controller may read
    @property
    def now(self) -> int:
        return int(self.env.now)

    @property
    def mttr(self) -> float:
        return self.scenario.mttr

    def active_jobs(self) -> list[JobRT]:
        return [j for j in self.jobs.values() if j.state is not JobState.DONE]

    def expected_up(self, m: MachineRT) -> int:
        """When a machine is expected back: breakdown time + MTTR (the real repair time is hidden)."""
        if not m.down:
            return self.now
        return max(int(round(m.down_since + self.mttr)), self.now + 1)

    def remaining_current(self, m: MachineRT) -> int:
        """Setup and processing time still to do on the operation in progress."""
        if m.current is None:
            return 0
        rem = m.rem_setup + m.rem_proc
        if m.seg_start is not None:
            rem -= self.now - m.seg_start
        return max(rem, 0)

    def expected_free(self, m: MachineRT) -> int:
        """Earliest time the machine is expected to be able to start another job."""
        return self.expected_up(m) + self.remaining_current(m)

    # ------------------------------------------------------------------ what a controller may do
    def reroute(self, jr: JobRT, machine: int) -> None:
        """Send a job that is waiting in a machine queue to another eligible machine."""
        assert jr.state is JobState.QUEUED and machine != jr.dest
        assert machine in self.ws.alternatives(jr.family, jr.op)
        self.machines[jr.dest].queue.remove(jr)
        self._send(jr, machine)

    def kick(self) -> None:
        """Let idle machines ask the controller again (after a replan)."""
        for m in self.machines:
            self._try_start(m)

    # ------------------------------------------------------------------ run
    def run(self) -> RunLog:
        self.env.process(self._arrivals())
        for m in self.machines:
            if self.scenario.downtime[m.idx]:
                self.env.process(self._breakdowns(m))
        self.env.run(until=self._all_done)
        return self.log

    # ------------------------------------------------------------------ processes
    def _arrivals(self):
        for job in self.scenario.jobs:
            if job.arrival > self.env.now:
                yield self.env.timeout(job.arrival - self.env.now)
            jr = JobRT(job, self.ws)
            self.jobs[job.idx] = jr
            self.controller.on_event(self, "arrival", job=jr)
            self._route(jr)

    def _breakdowns(self, m: MachineRT):
        for t_down, t_up in self.scenario.downtime[m.idx]:
            yield self.env.timeout(t_down - self.env.now)
            m.down = True
            m.down_since = self.now
            m.up_event = self.env.event()
            if m.current is not None and m.seg_start is not None:
                self._close_segment(m)  # book the work done so far, then stop the operation
                m.proc.interrupt()
            self.controller.on_event(self, "breakdown", machine=m)
            yield self.env.timeout(t_up - self.env.now)
            m.down = False
            m.up_event.succeed()
            self.controller.on_event(self, "repair", machine=m)
            self._try_start(m)

    def _work(self, m: MachineRT, jr: JobRT):
        """Setup then processing of one operation; a breakdown suspends it, the repair resumes it."""
        while m.rem_setup > 0 or m.rem_proc > 0:
            try:
                if m.down:
                    yield m.up_event
                    continue
                m.seg_kind = "setup" if m.rem_setup > 0 else "proc"
                m.seg_start = self.now
                yield self.env.timeout(m.rem_setup if m.rem_setup > 0 else m.rem_proc)
                self._close_segment(m)
            except simpy.Interrupt:
                pass  # the breakdown process has already closed the segment
        m.current = None
        m.proc = None
        jr.op += 1
        jr.dest = None
        if jr.op == jr.n_ops:
            jr.state = JobState.DONE
            jr.completion = self.now
            self.n_done += 1
            if self.n_done == len(self.scenario.jobs):
                self._all_done.succeed()
        else:
            jr.state = JobState.NEW
            self._route(jr)
        self._try_start(m)

    def _transport(self, agv: AGVRT, jr: JobRT):
        origin = jr.station
        if agv.station != origin:
            start = self.now
            yield self.env.timeout(self.ws.travel[agv.station][origin])
            self.log.legs.append(Leg(agv.idx, agv.station, origin, start, self.now, None))
            agv.station = origin
        dest = self.ws.station(jr.dest)
        start = self.now
        jr.state = JobState.IN_TRANSIT
        jr.eta = start + self.ws.travel[origin][dest]
        yield self.env.timeout(self.ws.travel[origin][dest])
        self.log.legs.append(Leg(agv.idx, origin, dest, start, self.now, jr.idx))
        agv.station = dest
        agv.idle = True
        self.machines[jr.dest].inbound.remove(jr)
        self._enqueue(jr, jr.dest)
        self._dispatch_agvs()

    # ------------------------------------------------------------------ mechanics
    def _close_segment(self, m: MachineRT) -> None:
        if m.seg_start is None:
            return
        elapsed = self.now - m.seg_start
        if elapsed > 0:
            jr = m.current
            self.log.segments.append(Segment(m.idx, jr.idx, jr.op, m.seg_kind, m.seg_start, self.now))
            if m.seg_kind == "setup":
                m.rem_setup -= elapsed
            else:
                m.rem_proc -= elapsed
        m.seg_start = None

    def _route(self, jr: JobRT) -> None:
        machine = self.controller.route(self, jr)
        assert machine in self.ws.alternatives(jr.family, jr.op), "controller chose an ineligible machine"
        self._send(jr, machine)

    def _send(self, jr: JobRT, machine: int) -> None:
        jr.dest = machine
        if jr.station == self.ws.station(machine):  # next operation on the same machine: no transport
            self._enqueue(jr, machine)
            return
        jr.state = JobState.WAIT_TRANSPORT
        jr.eta = None
        self.machines[machine].inbound.append(jr)
        self.requests.append(jr)
        self._dispatch_agvs()

    def _enqueue(self, jr: JobRT, machine: int) -> None:
        m = self.machines[machine]
        jr.state = JobState.QUEUED
        jr.station = self.ws.station(machine)
        jr.queued_at = self.now
        m.queue.append(jr)
        self._try_start(m)

    def _try_start(self, m: MachineRT) -> None:
        if m.current is not None or m.down or not m.queue:
            return
        jr = self.controller.sequence(self, m)
        if jr is None:
            return
        m.queue.remove(jr)
        m.current = jr
        jr.state = JobState.PROCESSING
        m.rem_setup = self.ws.setup_time(m.idx, m.last_family, jr.family)
        m.rem_proc = self.ws.proc(jr.family, jr.op, m.idx)
        m.last_family = jr.family
        m.seg_start = None
        m.proc = self.env.process(self._work(m, jr))

    def _dispatch_agvs(self) -> None:
        """Oldest request first, served by the nearest idle AGV."""
        while self.requests:
            idle = [a for a in self.agvs if a.idle]
            if not idle:
                return
            jr = self.requests.pop(0)
            agv = min(idle, key=lambda a: (self.ws.travel[a.station][jr.station], a.idx))
            agv.idle = False
            self.env.process(self._transport(agv, jr))


def simulate(ws: Workshop, scenario: Scenario, controller) -> RunLog:
    return Shop(ws, scenario, controller).run()
