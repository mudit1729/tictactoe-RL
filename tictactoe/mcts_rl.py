from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from tictactoe.dqn_rl import DQNAgent
from tictactoe.rl_core import (
    AGENT_MARK,
    DRAW,
    OPPONENT_MARK,
    OpponentPolicy,
    apply_action,
    check_winner,
    encode_state,
    legal_actions,
)


@dataclass
class MCTSConfig:
    num_simulations: int = 32
    max_depth: int = 4
    exploration_c: float = 1.4


@dataclass
class ChildStats:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class SearchNode:
    def __init__(self, board: np.ndarray, priors: Dict[int, float]) -> None:
        self.board = board.copy()
        self.children = {action: ChildStats(prior=prior) for action, prior in priors.items()}
        self.visits = 0


class MCTSPlanner:
    def __init__(self, config: Optional[MCTSConfig] = None) -> None:
        self.config = config or MCTSConfig()
        self.nodes: Dict[tuple[int, ...], SearchNode] = {}

    def select_action(
        self,
        board: np.ndarray,
        agent: DQNAgent,
        opponent_policy: OpponentPolicy,
        rng: Optional[random.Random] = None,
    ) -> int:
        rng = rng or random.Random()
        self.nodes = {}
        board_key = tuple(board.tolist())
        self._ensure_node(board_key, board, agent)

        for _ in range(self.config.num_simulations):
            self._simulate(board.copy(), agent, opponent_policy, rng, depth=0)

        root = self.nodes[board_key]
        best_action = max(
            root.children.items(),
            key=lambda item: (item[1].visits, item[1].mean_value, item[0]),
        )[0]
        return int(best_action)

    def _ensure_node(self, board_key: tuple[int, ...], board: np.ndarray, agent: DQNAgent) -> SearchNode:
        node = self.nodes.get(board_key)
        if node is not None:
            return node
        q_values = agent.q_values(encode_state(board))
        actions = legal_actions(board)
        logits = np.array([q_values[action] for action in actions], dtype=np.float32)
        logits = logits - logits.max(initial=0.0)
        priors = np.exp(logits)
        if priors.sum() <= 0:
            priors = np.ones_like(priors)
        priors = priors / priors.sum()
        node = SearchNode(board, {action: float(prior) for action, prior in zip(actions, priors.tolist())})
        self.nodes[board_key] = node
        return node

    def _terminal_value(self, winner: int) -> float:
        if winner == AGENT_MARK:
            return 1.0
        if winner == DRAW:
            return 0.25
        return -1.0

    def _evaluate_leaf(self, board: np.ndarray, agent: DQNAgent) -> float:
        actions = legal_actions(board)
        if not actions:
            return 0.25
        q_values = agent.q_values(encode_state(board))
        value = max(q_values[action] for action in actions)
        return float(np.tanh(value))

    def _select_ucb(self, node: SearchNode) -> int:
        total = max(1, node.visits)
        best_score = -math.inf
        best_action = 0
        for action, child in node.children.items():
            explore = self.config.exploration_c * child.prior * math.sqrt(total) / (1 + child.visits)
            score = child.mean_value + explore
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _simulate(
        self,
        board: np.ndarray,
        agent: DQNAgent,
        opponent_policy: OpponentPolicy,
        rng: random.Random,
        depth: int,
    ) -> float:
        winner = check_winner(board)
        if winner is not None:
            return self._terminal_value(winner)
        if depth >= self.config.max_depth:
            return self._evaluate_leaf(board, agent)

        board_key = tuple(board.tolist())
        node = self._ensure_node(board_key, board, agent)
        action = self._select_ucb(node)

        next_board = apply_action(board, action, AGENT_MARK)
        winner = check_winner(next_board)
        if winner is None:
            opponent_action = opponent_policy.select_action(next_board, OPPONENT_MARK, rng)
            next_board = apply_action(next_board, opponent_action, OPPONENT_MARK)
            winner = check_winner(next_board)

        if winner is not None:
            value = self._terminal_value(winner)
        else:
            value = self._simulate(next_board, agent, opponent_policy, rng, depth + 1)

        child = node.children[action]
        child.visits += 1
        child.value_sum += value
        node.visits += 1
        return value
