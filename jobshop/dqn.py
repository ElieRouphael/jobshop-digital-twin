"""DQN dispatching agent: at each sequencing decision it picks which priority rule to apply.

Decision point: a machine is free and at least two jobs wait in its queue.
Action: one of the rules in `ACTIONS`; the job that rule selects is started.
State: features of the job each rule would pick, of the queue, and of the shop.
Reward: minus the increase, since the previous decision, of the tardiness the shop is already
committed to (`estimated_tardiness`). The undiscounted return of an episode is exactly minus
the total tardiness of the run, but a decision that makes a job late is penalised when it is
taken, not when the job finally leaves.

Routing and rerouting are the same as for the plain rules (earliest expected finish).
The agent code follows the structure of the CartPole DQN study it is derived from:
frozen config, replay buffer, Double DQN target, soft target update, Huber loss.
"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from .rules import RuleController, pick
from .scenario import LEVELS, Workshop, make_scenario
from .shop import JobRT, JobState, MachineRT, Shop

ACTIONS = ("FIFO", "SPT", "EDD", "SST")
JOB_FEATURES = 5
STATE_DIM = len(ACTIONS) * JOB_FEATURES + 9


@dataclass(frozen=True, slots=True)
class DQNConfig:
    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 5e-4
    grad_clip_norm: float = 10.0
    gamma: float = 0.97
    double_dqn: bool = True
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer_size: int = 1_000
    tau_soft_update: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30_000
    reward_scale: float = 50.0  # minutes of tardiness per unit of reward
    episodes: int = 1500
    validate_every: int = 25
    n_jobs: int = 60


# ---------------------------------------------------------------------- network and memory
class QNetwork(nn.Module):
    """Feed-forward Q-network: maps a state to one Q-value per rule."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_sizes:
            layers += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminal: bool  # True only when the run is over (all jobs finished)


class ReplayBuffer:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng = rng

    def push(self, *transition) -> None:
        self._buffer.append(Transition(*transition))

    def sample(self, batch_size: int) -> Transition:
        return Transition(*zip(*self._rng.sample(self._buffer, batch_size)))

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------- state
def slack(shop: Shop, j: JobRT) -> float:
    """Time left before the due date once the shortest remaining work is subtracted."""
    return j.due - shop.now - remaining_work(shop, j)


def estimated_tardiness(shop: Shop) -> int:
    """Tardiness already certain at this time: finished jobs count in full, an unfinished job
    counts for the lateness it would have if all its remaining work started now, without waiting.
    It never decreases and equals the total tardiness of the run when every job is finished."""
    total = 0
    for j in shop.jobs.values():
        if j.completion is not None:
            total += max(0, j.completion - j.due)
        else:
            total += max(0, shop.now + remaining_work(shop, j) - j.due)
    return total


def remaining_work(shop: Shop, j: JobRT) -> int:
    """Shortest processing time still to do on a job (the operation in progress counts for what is left)."""
    if j.state is JobState.PROCESSING:
        m = shop.machines[j.dest]
        left = m.rem_proc - (shop.now - m.seg_start if m.seg_start is not None and m.seg_kind == "proc" else 0)
        return left + shop.ws.min_remaining_work(j.family, j.op + 1)
    return shop.ws.min_remaining_work(j.family, j.op)


def features(shop: Shop, m: MachineRT) -> np.ndarray:
    ws = shop.ws
    x: list[float] = []
    for rule in ACTIONS:
        j = pick(rule, shop, m)
        x += [
            ws.proc(j.family, j.op, m.idx) / 20.0,
            ws.setup_time(m.idx, m.last_family, j.family) / 10.0,
            float(np.clip(slack(shop, j) / 100.0, -3.0, 3.0)),
            (j.n_ops - j.op) / 4.0,
            min((shop.now - j.queued_at) / 100.0, 3.0),
        ]
    slacks = np.array([slack(shop, j) for j in m.queue])
    active = shop.active_jobs()
    all_slacks = np.array([slack(shop, j) for j in active])
    x += [
        len(m.queue) / 10.0,
        float((slacks < 0).mean()),
        float(np.clip(slacks.min() / 100.0, -3.0, 3.0)),
        float(np.clip(slacks.mean() / 100.0, -3.0, 3.0)),
        float(np.mean([ws.setup_time(m.idx, m.last_family, j.family) == 0 for j in m.queue])),
        len(active) / 20.0,
        float((all_slacks < 0).mean()),
        sum(k.down for k in shop.machines) / ws.n_machines,
        len(shop.requests) / 5.0,
    ]
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------- learning
def epsilon_by_step(steps_done: int, config: DQNConfig) -> float:
    return config.epsilon_end + (config.epsilon_start - config.epsilon_end) * math.exp(
        -steps_done / config.epsilon_decay_steps
    )


def compute_td_targets(rewards, next_states, terminals, policy_net, target_net, gamma: float, double_dqn: bool):
    """y = r + gamma * (1 - terminal) * Q_target(s', a*), a* chosen by the online net in Double DQN."""
    with torch.no_grad():
        next_q_target = target_net(next_states)
        if double_dqn:
            best_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
        else:
            best_actions = next_q_target.argmax(dim=1, keepdim=True)
        return rewards + gamma * next_q_target.gather(1, best_actions) * (1.0 - terminals)


def soft_update(target_net: nn.Module, policy_net: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.lerp_(policy_param, tau)


class Learner:
    """Holds the networks and the replay memory, and turns decisions into transitions."""

    def __init__(self, config: DQNConfig, seed: int):
        torch.manual_seed(seed)
        self.config = config
        self.rng = random.Random(seed)
        self.policy_net = QNetwork(STATE_DIM, len(ACTIONS), config.hidden_sizes)
        self.target_net = QNetwork(STATE_DIM, len(ACTIONS), config.hidden_sizes)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.buffer = ReplayBuffer(config.buffer_size, self.rng)
        self.steps_done = 0
        self._pending: tuple[np.ndarray, int, int] | None = None  # state, action, tardiness at decision

    def act(self, shop: Shop, state: np.ndarray) -> int:
        """Epsilon-greedy action; closes the previous transition with the tardiness added since."""
        accrued = estimated_tardiness(shop)
        if self._pending is not None:
            prev_state, prev_action, prev_accrued = self._pending
            reward = -(accrued - prev_accrued) / self.config.reward_scale
            self.buffer.push(prev_state, prev_action, reward, state, False)
            self.optimize()
        if self.rng.random() < epsilon_by_step(self.steps_done, self.config):
            action = self.rng.randrange(len(ACTIONS))
        else:
            action = greedy_action(self.policy_net, state)
        self.steps_done += 1
        self._pending = (state, action, accrued)
        return action

    def end_episode(self, total_tardiness: int) -> None:
        if self._pending is not None:
            prev_state, prev_action, prev_accrued = self._pending
            reward = -(total_tardiness - prev_accrued) / self.config.reward_scale
            self.buffer.push(prev_state, prev_action, reward, np.zeros_like(prev_state), True)
            self.optimize()
        self._pending = None

    def optimize(self) -> float | None:
        config = self.config
        if len(self.buffer) < max(config.batch_size, config.min_buffer_size):
            return None
        batch = self.buffer.sample(config.batch_size)
        states = torch.as_tensor(np.asarray(batch.state), dtype=torch.float32)
        actions = torch.as_tensor(batch.action, dtype=torch.int64).unsqueeze(1)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
        next_states = torch.as_tensor(np.asarray(batch.next_state), dtype=torch.float32)
        terminals = torch.as_tensor(batch.terminal, dtype=torch.float32).unsqueeze(1)
        current_q = self.policy_net(states).gather(1, actions)
        targets = compute_td_targets(
            rewards, next_states, terminals, self.policy_net, self.target_net, config.gamma, config.double_dqn
        )
        loss = functional.smooth_l1_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), config.grad_clip_norm)
        self.optimizer.step()
        soft_update(self.target_net, self.policy_net, config.tau_soft_update)
        return float(loss.item())


def greedy_action(model: nn.Module, state: np.ndarray) -> int:
    with torch.no_grad():
        return int(model(torch.as_tensor(state).unsqueeze(0)).argmax(dim=1).item())


# ---------------------------------------------------------------------- controller
class DQNController(RuleController):
    """Sequencing by the Q-network (greedy), or by a `Learner` while training."""

    def __init__(self, policy_net: nn.Module | None = None, learner: Learner | None = None):
        super().__init__("EDD")
        self.name = "DQN"
        self.policy_net = policy_net
        self.learner = learner
        self.action_counts = [0] * len(ACTIONS)

    def sequence(self, shop: Shop, m: MachineRT) -> JobRT | None:
        if len(m.queue) == 1:
            return m.queue[0]
        t0 = time.perf_counter()
        state = features(shop, m)
        if self.learner is not None:
            action = self.learner.act(shop, state)
        else:
            action = greedy_action(self.policy_net, state)
        self.action_counts[action] += 1
        jr = pick(ACTIONS[action], shop, m)
        self.decision_time_s += time.perf_counter() - t0
        return jr


# ---------------------------------------------------------------------- training
TRAIN_SEED_BASE = 100_000  # training scenarios: seeds 100000 + ...
VALIDATION_SEEDS = tuple(range(5_000, 5_010))  # validation scenarios, both disturbance levels


def total_tardiness(shop: Shop) -> int:
    return sum(max(0, j.completion - j.due) for j in shop.jobs.values())


def validate(ws: Workshop, policy_net: nn.Module, n_jobs: int) -> float:
    """Mean tardiness per job of the greedy policy on the validation scenarios."""
    values = []
    for mtbf in LEVELS.values():
        for seed in VALIDATION_SEEDS:
            shop = Shop(ws, make_scenario(ws, seed, n_jobs=n_jobs, mtbf=mtbf), DQNController(policy_net))
            shop.run()
            values.append(total_tardiness(shop) / n_jobs)
    return float(np.mean(values))


def train(ws: Workshop, config: DQNConfig, seed: int, log=print) -> dict:
    """Train one agent. Returns the best checkpoint (by validation) and the learning history."""
    torch.set_num_threads(1)
    learner = Learner(config, seed)
    levels = list(LEVELS.values())
    history = []
    best = {"validation": float("inf"), "state_dict": None, "episode": 0}
    t0 = time.perf_counter()
    for episode in range(1, config.episodes + 1):
        scenario = make_scenario(
            ws, TRAIN_SEED_BASE + 1_000 * seed + episode, n_jobs=config.n_jobs, mtbf=levels[episode % len(levels)]
        )
        shop = Shop(ws, scenario, DQNController(learner=learner))
        shop.run()
        tardiness = total_tardiness(shop)
        learner.end_episode(tardiness)
        row = {"episode": episode, "train_mean_tardiness": tardiness / config.n_jobs,
               "epsilon": epsilon_by_step(learner.steps_done, config)}
        if episode % config.validate_every == 0:
            row["validation"] = validate(ws, learner.policy_net, config.n_jobs)
            if row["validation"] < best["validation"]:
                best = {"validation": row["validation"], "episode": episode,
                        "state_dict": {k: v.clone() for k, v in learner.policy_net.state_dict().items()}}
            log(f"seed {seed} episode {episode:4d}  validation mean tardiness {row['validation']:.2f}  "
                f"(best {best['validation']:.2f} at {best['episode']})  eps {row['epsilon']:.2f}")
        history.append(row)
    return {"seed": seed, "config": asdict(config), "history": history, "best_validation": best["validation"],
            "best_episode": best["episode"], "state_dict": best["state_dict"],
            "wall_time_s": time.perf_counter() - t0, "steps": learner.steps_done}


def load_policy(path) -> nn.Module:
    payload = torch.load(path, weights_only=True)
    net = QNetwork(STATE_DIM, len(ACTIONS), tuple(payload["hidden_sizes"]))
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net
