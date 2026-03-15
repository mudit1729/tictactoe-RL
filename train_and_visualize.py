#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tictactoe.dqn_rl import DQNAgent, DQNConfig
from tictactoe.mcts_rl import MCTSConfig, MCTSPlanner
from tictactoe.rl_core import (
    MixedPolicy,
    TicTacToeEnv,
    encode_state,
    legal_actions,
    outcome_to_score,
    render_board_image,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
DOCS_ASSETS = ROOT / "docs" / "assets"
DOCS_MODELS = ROOT / "docs" / "models"
TB_ROOT = ROOT / "logs" / "tensorboard"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_output_dir(name: str) -> Path:
    path = OUTPUT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_episode(
    env: TicTacToeEnv,
    agent: DQNAgent,
    epsilon: float,
    planner: Optional[MCTSPlanner] = None,
    train: bool = True,
) -> tuple[float, str, list[np.ndarray]]:
    state, info = env.reset()
    frames = [info["board"].copy()]
    total_reward = 0.0

    while True:
        valid_actions = legal_actions(env.board)
        if train and planner is not None and random.random() >= epsilon:
            action = planner.select_action(env.board.copy(), agent, env.opponent_policy, env.rng)
        elif planner is not None and not train:
            action = planner.select_action(env.board.copy(), agent, env.opponent_policy, env.rng)
        else:
            action = agent.act(state, valid_actions, epsilon if train else 0.0)

        next_state, reward, done, step_info = env.step(action)
        total_reward += reward
        frames.append(step_info["board"].copy())
        if train:
            agent.remember(state, action, reward, None if done else next_state, done)
            agent.train_step()
        state = next_state
        if done:
            return total_reward, step_info["outcome"], frames


def evaluate_agent(
    agent: DQNAgent,
    opponent_strength: float,
    episodes: int = 200,
    planner: Optional[MCTSPlanner] = None,
    seed: int = 123,
) -> dict[str, float]:
    env = TicTacToeEnv(opponent_policy=MixedPolicy(optimal_probability=opponent_strength), seed=seed)
    wins = draws = losses = 0
    for episode in range(episodes):
        _, outcome, _ = run_episode(env, agent, epsilon=0.0, planner=planner, train=False)
        if outcome == "win":
            wins += 1
        elif outcome == "draw":
            draws += 1
        else:
            losses += 1
    score = (wins + 0.5 * draws) / episodes
    return {
        "score": score,
        "win_rate": wins / episodes,
        "draw_rate": draws / episodes,
        "loss_rate": losses / episodes,
    }


def train_agent(
    name: str,
    total_episodes: int,
    config: DQNConfig,
    opponent_strength: float,
    eval_strength: float,
    checkpoint_episodes: list[int],
    mcts_config: Optional[MCTSConfig] = None,
    seed: int = 42,
) -> tuple[DQNAgent, dict[str, Any], Path]:
    set_seed(seed)
    save_dir = make_output_dir(name)
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = TicTacToeEnv(opponent_policy=MixedPolicy(optimal_probability=opponent_strength), seed=seed)
    agent = DQNAgent(config)
    planner = MCTSPlanner(mcts_config) if mcts_config else None

    metrics: dict[str, Any] = {
        "episode_rewards": [],
        "episode_scores": [],
        "outcomes": [],
        "epsilon": [],
        "evaluation": [],
        "checkpoints": checkpoint_episodes,
        "config": asdict(config),
        "mcts_config": None if mcts_config is None else asdict(mcts_config),
        "opponent_strength": opponent_strength,
        "eval_strength": eval_strength,
    }

    for episode in range(1, total_episodes + 1):
        epsilon = agent.epsilon_for_episode(episode)
        reward, outcome, _ = run_episode(env, agent, epsilon=epsilon, planner=planner, train=True)
        metrics["episode_rewards"].append(reward)
        metrics["episode_scores"].append(outcome_to_score(outcome))
        metrics["outcomes"].append(outcome)
        metrics["epsilon"].append(epsilon)

        if episode in checkpoint_episodes:
            checkpoint_path = checkpoint_dir / f"checkpoint_{episode}.pt"
            agent.save(checkpoint_path, extra={"episode": episode})
            evaluation = evaluate_agent(
                agent,
                opponent_strength=eval_strength,
                episodes=300,
                planner=planner,
                seed=seed + episode,
            )
            evaluation["episode"] = episode
            metrics["evaluation"].append(evaluation)

        if episode % 2000 == 0:
            recent = metrics["episode_scores"][-500:]
            recent_mean = float(np.mean(recent)) if recent else 0.0
            print(
                f"{name}: episode {episode}/{total_episodes} | "
                f"avg500={recent_mean:.3f} | eps={epsilon:.3f}"
            )

    final_path = checkpoint_dir / "final.pt"
    agent.save(final_path, extra={"episode": total_episodes})
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return agent, metrics, save_dir


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.array([])
    result = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        result.append(float(np.mean(values[start : idx + 1])))
    return np.array(result, dtype=np.float32)


def generate_training_plot(
    dqn_metrics: dict[str, Any],
    mcts_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)

    dqn_roll = rolling_mean(dqn_metrics["episode_scores"], 500)
    mcts_roll = rolling_mean(mcts_metrics["episode_scores"], 500)
    axes[0].plot(dqn_roll, label="DQN", color="#1f77b4", linewidth=2.2)
    axes[0].plot(mcts_roll, label="DQN + MCTS", color="#d62728", linewidth=2.2)
    axes[0].set_title("Rolling Training Score (window=500)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Score")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    dqn_eval_x = [entry["episode"] for entry in dqn_metrics["evaluation"]]
    dqn_eval_y = [entry["score"] for entry in dqn_metrics["evaluation"]]
    mcts_eval_x = [entry["episode"] for entry in mcts_metrics["evaluation"]]
    mcts_eval_y = [entry["score"] for entry in mcts_metrics["evaluation"]]
    axes[1].plot(dqn_eval_x, dqn_eval_y, "o-", label="DQN", color="#1f77b4", linewidth=2.2)
    axes[1].plot(mcts_eval_x, mcts_eval_y, "o-", label="DQN + MCTS", color="#d62728", linewidth=2.2)
    axes[1].set_title("Evaluation Score vs Strong Mixed Opponent")
    axes[1].set_xlabel("Checkpoint episode")
    axes[1].set_ylabel("Score = win + 0.5 * draw")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("Tic-Tac-Toe: Small DQN vs Small DQN + MCTS", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_tensorboard_run(
    run_name: str,
    metrics: dict[str, Any],
    rolling_window: int = 500,
) -> Path:
    log_dir = TB_ROOT / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    scores = metrics["episode_scores"]
    rewards = metrics["episode_rewards"]
    epsilons = metrics["epsilon"]

    for idx, (score, reward, epsilon) in enumerate(zip(scores, rewards, epsilons), start=1):
        writer.add_scalar("train/episode_score", score, idx)
        writer.add_scalar("train/episode_reward", reward, idx)
        writer.add_scalar("train/epsilon", epsilon, idx)
        writer.add_scalar("train/avg_score_running", float(np.mean(scores[:idx])), idx)
        window_start = max(0, idx - rolling_window)
        writer.add_scalar("train/avg_score_500", float(np.mean(scores[window_start:idx])), idx)

    for entry in metrics["evaluation"]:
        episode = entry["episode"]
        writer.add_scalar("eval/score", entry["score"], episode)
        writer.add_scalar("eval/win_rate", entry["win_rate"], episode)
        writer.add_scalar("eval/draw_rate", entry["draw_rate"], episode)
        writer.add_scalar("eval/loss_rate", entry["loss_rate"], episode)

    writer.flush()
    writer.close()
    return log_dir


def best_episode_frames(
    agent: DQNAgent,
    checkpoint_path: Path,
    opponent_strength: float,
    planner: Optional[MCTSPlanner],
    seed_base: int = 100,
) -> tuple[list[np.ndarray], str]:
    _ = checkpoint_path
    env = TicTacToeEnv(opponent_policy=MixedPolicy(optimal_probability=opponent_strength), seed=seed_base)
    best_score = -1.0
    best_frames: list[np.ndarray] = []
    best_outcome = "loss"
    for seed in range(seed_base, seed_base + 16):
        env.reset(seed=seed)
        _, outcome, boards = run_episode(env, agent, epsilon=0.0, planner=planner, train=False)
        score = outcome_to_score(outcome)
        if score > best_score or (score == best_score and len(boards) < len(best_frames)):
            best_score = score
            best_frames = boards
            best_outcome = outcome
    return best_frames, best_outcome


def write_gif(
    boards: list[np.ndarray],
    output_path: Path,
    title: str,
    footer_prefix: str,
) -> None:
    frames = []
    for idx, board in enumerate(boards):
        footer = f"{footer_prefix} | ply {idx}"
        frames.append(render_board_image(board, title=title, footer=footer))
    imageio.mimsave(output_path, frames, duration=0.65, loop=0)


def export_models_for_web(dqn_agent: DQNAgent, mcts_agent: DQNAgent) -> None:
    DOCS_MODELS.mkdir(parents=True, exist_ok=True)
    dqn_agent.export_json(DOCS_MODELS / "dqn_model.json", metadata={"label": "DQN"})
    mcts_agent.export_json(DOCS_MODELS / "dqn_mcts_model.json", metadata={"label": "DQN + MCTS-trained"})


def main() -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
    DOCS_MODELS.mkdir(parents=True, exist_ok=True)
    TB_ROOT.mkdir(parents=True, exist_ok=True)

    config = DQNConfig(hidden_dim=32, device="cpu")
    checkpoint_episodes = [1000, 5000, 10000, 15000, 20000]

    dqn_agent, dqn_metrics, dqn_dir = train_agent(
        name="dqn_small",
        total_episodes=20000,
        config=config,
        opponent_strength=0.70,
        eval_strength=0.85,
        checkpoint_episodes=checkpoint_episodes,
        mcts_config=None,
    )
    mcts_agent, mcts_metrics, mcts_dir = train_agent(
        name="dqn_mcts_small",
        total_episodes=20000,
        config=config,
        opponent_strength=0.70,
        eval_strength=0.85,
        checkpoint_episodes=checkpoint_episodes,
        mcts_config=MCTSConfig(num_simulations=32, max_depth=4, exploration_c=1.4),
    )

    generate_training_plot(dqn_metrics, mcts_metrics, DOCS_ASSETS / "dqn_vs_mcts_training.png")
    write_tensorboard_run("dqn_small", dqn_metrics)
    write_tensorboard_run("dqn_mcts_small", mcts_metrics)

    for episode in checkpoint_episodes:
        checkpoint_path = dqn_dir / "checkpoints" / f"checkpoint_{episode}.pt"
        agent = DQNAgent.load(checkpoint_path, device="cpu")
        boards, outcome = best_episode_frames(agent, checkpoint_path, opponent_strength=0.65, planner=None, seed_base=episode)
        write_gif(
            boards,
            DOCS_ASSETS / f"dqn_checkpoint_{episode}.gif",
            title=f"Small DQN | Episode {episode}",
            footer_prefix=outcome.upper(),
        )

    export_models_for_web(dqn_agent, mcts_agent)

    summary = {
        "dqn_final_eval": dqn_metrics["evaluation"][-1],
        "mcts_final_eval": mcts_metrics["evaluation"][-1],
        "dqn_dir": str(dqn_dir),
        "mcts_dir": str(mcts_dir),
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
