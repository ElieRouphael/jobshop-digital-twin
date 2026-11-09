"""Gantt chart of one run with breakdowns, rescheduled by CP-SAT (machines and AGVs).

    python scripts/plot_gantt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jobshop.checker import check_schedule  # noqa: E402
from jobshop.scenario import LEVELS, make_scenario, make_workshop  # noqa: E402
from jobshop.shop import simulate  # noqa: E402

FAMILY_COLOURS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]


def plot_gantt(path: Path, seed: int = 0, n_jobs: int = 30, window: tuple[int, int] = (0, 480)) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from jobshop.cpsat import CPSATController

    ws = make_workshop()
    scenario = make_scenario(ws, seed, n_jobs=n_jobs, mtbf=LEVELS["frequent breakdowns"])
    log = simulate(ws, scenario, CPSATController())
    assert check_schedule(ws, scenario, log) == []
    family = {j.idx: j.family for j in scenario.jobs}

    rows = [f"machine {m}" for m in range(ws.n_machines)] + [f"AGV {a}" for a in range(ws.n_agvs)]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    for m in range(ws.n_machines):
        for down, up in scenario.downtime[m]:
            if down < window[1]:
                ax.barh(m, up - down, left=down, height=0.8, color="#D62728", alpha=0.25, hatch="///", edgecolor="#D62728")
    for s in log.segments:
        if s.kind == "setup":
            ax.barh(s.machine, s.end - s.start, left=s.start, height=0.6, color="#BBBBBB", edgecolor="white", linewidth=0.3)
        else:
            ax.barh(s.machine, s.end - s.start, left=s.start, height=0.6, color=FAMILY_COLOURS[family[s.job]],
                    edgecolor="white", linewidth=0.3)
            if s.end - s.start >= 8 and window[0] <= s.start and s.end <= window[1]:
                ax.text((s.start + s.end) / 2, s.machine, str(s.job), ha="center", va="center", fontsize=6, color="white")
    for g in log.legs:
        y = ws.n_machines + g.agv
        colour = "#DDDDDD" if g.job is None else FAMILY_COLOURS[family[g.job]]
        ax.barh(y, g.end - g.start, left=g.start, height=0.5, color=colour, edgecolor="white", linewidth=0.2)
    ax.set_yticks(range(len(rows)), rows)
    ax.invert_yaxis()
    ax.set_xlim(*window)
    ax.set_xlabel("time (min)")
    ax.set_title(f"Executed schedule, seed {seed}, first {window[1]} min: CP-SAT rescheduling under frequent breakdowns")
    handles = [Patch(color=c, label=f"family {i}") for i, c in enumerate(FAMILY_COLOURS)]
    handles += [Patch(color="#BBBBBB", label="changeover"), Patch(color="#DDDDDD", label="AGV empty"),
                Patch(facecolor="#D62728", alpha=0.25, hatch="///", edgecolor="#D62728", label="machine down")]
    ax.legend(handles=handles, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = ROOT / "results" / "gantt_breakdowns.png"
    out.parent.mkdir(exist_ok=True)
    plot_gantt(out)
    print(out)
