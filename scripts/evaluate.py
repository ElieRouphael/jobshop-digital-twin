"""Compare the decision methods on the same seeds and write results/results.md.

    python scripts/evaluate.py            # 20 seeds x 2 disturbance levels x 7 methods, about 40 min
    python scripts/evaluate.py --quick    # 3 seeds, for a smoke run
    python scripts/evaluate.py --render   # rebuild results.md from results/runs.csv without simulating

Every run goes through the independent checker; the script stops if one fails.
The test seeds (0, 1, 2, ...) are used here only: the DQN agent was trained and selected,
and the CP-SAT settings chosen, on other seeds (see jobshop/dqn.py and the README).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jobshop.checker import check_schedule  # noqa: E402
from jobshop.metrics import bootstrap_ci, run_metrics  # noqa: E402
from jobshop.rules import RULES, RuleController  # noqa: E402
from jobshop.scenario import LEVELS, make_scenario, make_workshop  # noqa: E402
from jobshop.shop import Shop  # noqa: E402

# CP-SAT settings, fixed on validation seeds before the test (see README, "Choices made before the test")
CPSAT_SETTINGS = {"det_time": 0.2, "allowance": 2, "max_jobs": 25, "strict": True, "workers": 8}
MODEL_DIR = ROOT / "models"
# "Greedy replan" = same rescheduling loop as CP-SAT, but the plan is the greedy warm start, without the solver
METHODS = ["FIFO", "SPT", "EDD", "SST", "DQN", "Greedy replan", "CP-SAT"]


def make_controller(method: str):
    if method in RULES:
        return RuleController(method)
    if method == "CP-SAT":
        from jobshop.cpsat import CPSATController

        return CPSATController(**CPSAT_SETTINGS)
    if method == "Greedy replan":
        from jobshop.cpsat import CPSATController

        return CPSATController(**CPSAT_SETTINGS, use_solver=False)
    if method.startswith("DQN"):
        import torch

        from jobshop.dqn import DQNController, load_policy

        torch.set_num_threads(1)
        name = "dqn.pt" if method == "DQN" else f"dqn_seed{method.split(':')[1]}.pt"
        return DQNController(load_policy(MODEL_DIR / name))
    raise ValueError(method)


def run_one(task: tuple[str, int, str, int]) -> dict:
    level, seed, method, n_jobs = task
    ws = make_workshop()
    scenario = make_scenario(ws, seed, n_jobs=n_jobs, mtbf=LEVELS[level])
    controller = make_controller(method)
    t0 = time.perf_counter()
    log = Shop(ws, scenario, controller).run()
    wall = time.perf_counter() - t0
    errors = check_schedule(ws, scenario, log)
    row = {"level": level, "seed": seed, "method": method, "checker_errors": len(errors),
           "first_error": errors[0] if errors else "", **run_metrics(ws, scenario, log),
           "decision_time_s": controller.decision_time_s, "replans": controller.n_replans,
           "replans_optimal": getattr(controller, "n_optimal", 0),
           "replans_not_improved": getattr(controller, "n_greedy", 0),
           "replans_capped": getattr(controller, "n_capped", 0), "run_wall_s": wall}
    return row


def run_all(tasks: list[tuple], workers: int, cpsat_procs: int) -> list[dict]:
    """Rules and DQN runs are single-threaded; a CP-SAT run uses 8 solver threads, so fewer at a time."""
    heavy = [t for t in tasks if t[2] == "CP-SAT"]
    light = [t for t in tasks if t[2] != "CP-SAT"]
    done: dict[tuple, dict] = {}
    for group, n in ((light, workers), (heavy, cpsat_procs)):
        if not group:
            continue
        if n <= 1:
            results = [run_one(t) for t in group]
        else:
            with ProcessPoolExecutor(max_workers=n) as pool:
                results = list(pool.map(run_one, group))
        done.update(zip(group, results))
    return [done[t] for t in tasks]


def cell(values, fmt: str = "{:.1f}") -> str:
    low, high = bootstrap_ci(values)
    return f"{fmt.format(np.mean(values))} [{fmt.format(low)}, {fmt.format(high)}]"


def level_tables(rows: list[dict], level: str, methods: list[str]) -> list[str]:
    by = {m: sorted((r for r in rows if r["level"] == level and r["method"] == m), key=lambda r: r["seed"])
          for m in methods}
    out = [
        "| method | makespan (min) | mean tardiness (min) | max tardiness (min) | utilisation (%) "
        "| setup time (min) | decision time per run (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in methods:
        r = by[m]
        dec = np.mean([x["decision_time_s"] for x in r])
        dec_text = f"{dec:.3f}" if dec < 1 else f"{dec:.0f}"
        note = ""
        if r[0]["replans"]:
            n = np.mean([x["replans"] for x in r])
            note = f" ({n:.0f} replans"
            if m == "CP-SAT":
                opt = 100 * sum(x["replans_optimal"] for x in r) / sum(x["replans"] for x in r)
                note += f", {dec / n:.2f} s each, {opt:.0f} % proven optimal"
            note += ")"
        out.append(
            f"| {m} | {cell([x['makespan'] for x in r], '{:.0f}')} | {cell([x['mean_tardiness'] for x in r])} "
            f"| {cell([x['max_tardiness'] for x in r], '{:.0f}')} | {cell([100 * x['utilisation'] for x in r])} "
            f"| {cell([x['setup_time'] for x in r], '{:.0f}')} | {dec_text}{note} |"
        )
    rules = [m for m in methods if m in RULES]
    others = [m for m in methods if m not in RULES]
    if rules and others:
        best = min(rules, key=lambda m: np.mean([x["mean_tardiness"] for x in by[m]]))
        ref = np.array([x["mean_tardiness"] for x in by[best]])
        out += [
            "",
            f"Paired comparison with the best single rule on mean tardiness at this level ({best}), seed by seed:",
            "",
            "| method | difference in mean tardiness (min) | difference in max tardiness (min) | seeds better / equal / worse |",
            "| --- | --- | --- | --- |",
        ]
        ref_max = np.array([x["max_tardiness"] for x in by[best]])
        for m in others:
            d = np.array([x["mean_tardiness"] for x in by[m]]) - ref
            dmax = np.array([x["max_tardiness"] for x in by[m]]) - ref_max
            out.append(
                f"| {m} | {cell(d, '{:+.1f}')} | {cell(dmax, '{:+.0f}')} "
                f"| {int((d < 0).sum())} / {int((d == 0).sum())} / {int((d > 0).sum())} |"
            )
    return out


def write_results(rows: list[dict], agent_rows: list[dict], args, methods: list[str], out_dir: Path) -> None:
    import ortools
    import simpy

    seeds = sorted({r["seed"] for r in rows})
    n_checked = len(rows) + len(agent_rows)
    lines = [
        "# Results",
        "",
        f"Generated by `python scripts/evaluate.py{' --quick' if args.quick else ''}`. "
        f"Test seeds {seeds[0]} to {seeds[-1]} ({len(seeds)} seeds), {args.jobs} jobs per run, synthetic workshop "
        "(5 machines, 2 AGVs, 4 product families). Same arrivals and breakdowns for every method on a given seed.",
        "",
        "Each cell is the mean over the seeds with a 95 % bootstrap confidence interval (percentile, 10,000 resamples). "
        f"All {n_checked} runs passed the independent checker (`jobshop/checker.py`).",
        "",
        "Utilisation = processing time / (machines x makespan). Setup time = total changeover time in the run. "
        "Decision time = computing time spent deciding during one run (wall clock; for CP-SAT, 8 solver threads, "
        f"{args.cpsat_procs} runs in parallel on the same machine), so it is indicative only; "
        "the other columns do not depend on the machine. Training time of the DQN agent is not included.",
        "",
    ]
    for level in LEVELS:
        mtbf = LEVELS[level]
        detail = "" if mtbf is None else f" (per machine: mean time between failures {mtbf:.0f} min, mean repair 20 min)"
        lines += [f"## {level.capitalize()}{detail}", ""] + level_tables(rows, level, methods) + [""]

    if agent_rows:
        training = json.loads((MODEL_DIR / "training.json").read_text())
        lines += [
            "## DQN: spread over training seeds",
            "",
            "The agent in the tables above is the one with the best validation score "
            f"(training seed {training['selected_seed']}). The others, evaluated on the same test seeds "
            "(mean tardiness in minutes, mean over the test seeds):",
            "",
            "| training seed | validation score | best checkpoint (episode) | test, no breakdowns | test, frequent breakdowns |",
            "| --- | --- | --- | --- | --- |",
        ]
        for run in training["runs"]:
            tag = f"DQN:{run['seed']}"
            vals = [np.mean([r["mean_tardiness"] for r in agent_rows if r["method"] == tag and r["level"] == lv])
                    for lv in LEVELS]
            lines.append(f"| {run['seed']} | {run['best_validation']:.2f} | {run['best_episode']} "
                         f"| {vals[0]:.1f} | {vals[1]:.1f} |")
        lines.append("")

    lines += [
        "## Environment",
        "",
        f"Python {platform.python_version()}, SimPy {simpy.__version__}, OR-Tools {ortools.__version__}, "
        f"NumPy {np.__version__}" + (_torch_version() if "DQN" in methods else "") + f"; {platform.system()} {platform.machine()}.",
        f"CP-SAT: {CPSAT_SETTINGS['det_time']} deterministic-time units per replan, "
        f"{CPSAT_SETTINGS['workers']} interleaved workers, "
        f"transport allowance {CPSAT_SETTINGS['allowance']} min, at most {CPSAT_SETTINGS['max_jobs']} jobs per plan.",
        "",
    ]
    (out_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


def _torch_version() -> str:
    import torch

    return f", PyTorch {torch.__version__}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=60)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--workers", type=int, default=8, help="parallel runs for rules and DQN")
    parser.add_argument("--cpsat-procs", type=int, default=3, help="parallel CP-SAT runs (8 solver threads each)")
    parser.add_argument("--quick", action="store_true", help="3 seeds only")
    parser.add_argument("--render", action="store_true", help="rebuild results.md from runs.csv")
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()
    if args.quick:
        args.seeds = 3
    methods = args.methods.split(",")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = range(args.first_seed, args.first_seed + args.seeds)

    if args.render:
        with open(out_dir / "runs.csv", newline="", encoding="utf-8") as f:
            stored = [{k: (v if k in ("level", "method", "first_error") else float(v)) for k, v in row.items()}
                      for row in csv.DictReader(f)]
        for row in stored:
            row["seed"] = int(row["seed"])
        rows = [r for r in stored if not r["method"].startswith("DQN:")]
        agent_rows = [r for r in stored if r["method"].startswith("DQN:")]
        write_results(rows, agent_rows, args, list(dict.fromkeys(r["method"] for r in rows)), out_dir)
        print((out_dir / "results.md").read_text(encoding="utf-8"))
        return

    t0 = time.perf_counter()
    tasks = [(level, seed, m, args.jobs) for level in LEVELS for seed in seeds for m in methods]
    rows = run_all(tasks, args.workers, args.cpsat_procs)
    agent_rows: list[dict] = []
    if "DQN" in methods and (MODEL_DIR / "training.json").exists():
        training = json.loads((MODEL_DIR / "training.json").read_text())
        agent_tasks = [(level, seed, f"DQN:{run['seed']}", args.jobs)
                       for run in training["runs"] for level in LEVELS for seed in seeds]
        agent_rows = run_all(agent_tasks, args.workers, args.cpsat_procs)

    failed = [r for r in rows + agent_rows if r["checker_errors"]]
    if failed:
        for r in failed[:10]:
            print("CHECKER FAILED:", r["level"], r["seed"], r["method"], r["first_error"])
        raise SystemExit(1)

    with open(out_dir / "runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows + agent_rows)
    write_results(rows, agent_rows, args, methods, out_dir)
    if not args.quick and "CP-SAT" in methods:
        from plot_gantt import plot_gantt

        plot_gantt(out_dir / "gantt_breakdowns.png")
    print((out_dir / "results.md").read_text(encoding="utf-8"))
    print(f"total time {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
