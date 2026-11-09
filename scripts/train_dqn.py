"""Train the DQN dispatching agent with several training seeds and keep the best on validation.

    python scripts/train_dqn.py            # 5 training seeds, 1500 episodes each, in parallel

Writes models/dqn_seed<k>.pt (best checkpoint of each seed), models/dqn.pt (the seed with the
best validation score: the agent used by scripts/evaluate.py) and models/training.json
(learning history). Training scenarios use seeds from 100000, validation scenarios seeds
5000-5009; the test seeds of scripts/evaluate.py are never seen here.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from jobshop.dqn import DQNConfig, train  # noqa: E402
from jobshop.scenario import make_workshop  # noqa: E402


def train_one(args: tuple[int, DQNConfig]) -> dict:
    seed, config = args
    return train(make_workshop(), config, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=DQNConfig().episodes)
    parser.add_argument("--out", default=str(ROOT / "models"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = replace(DQNConfig(), episodes=args.episodes)

    with ProcessPoolExecutor(max_workers=args.seeds) as pool:
        runs = list(pool.map(train_one, [(seed, config) for seed in range(args.seeds)]))

    for run in runs:
        torch.save({"hidden_sizes": list(config.hidden_sizes), "state_dict": run["state_dict"]},
                   out / f"dqn_seed{run['seed']}.pt")
    best = min(runs, key=lambda r: r["best_validation"])
    torch.save({"hidden_sizes": list(config.hidden_sizes), "state_dict": best["state_dict"]}, out / "dqn.pt")
    summary = {
        "selected_seed": best["seed"],
        "runs": [{k: v for k, v in run.items() if k != "state_dict"} for run in runs],
    }
    (out / "training.json").write_text(json.dumps(summary, indent=1))
    for run in runs:
        print(f"training seed {run['seed']}: best validation mean tardiness {run['best_validation']:.2f} "
              f"at episode {run['best_episode']}, {run['steps']} decisions, {run['wall_time_s']:.0f} s")
    print(f"selected: training seed {best['seed']}")


if __name__ == "__main__":
    main()
