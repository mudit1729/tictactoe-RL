from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EMPTY = 0
AGENT_MARK = 1
OPPONENT_MARK = -1
DRAW = 0

WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def new_board() -> np.ndarray:
    return np.zeros(9, dtype=np.int8)


def legal_actions(board: np.ndarray) -> List[int]:
    return [idx for idx, value in enumerate(board.tolist()) if value == EMPTY]


def apply_action(board: np.ndarray, action: int, mark: int) -> np.ndarray:
    next_board = board.copy()
    next_board[action] = mark
    return next_board


def check_winner(board: np.ndarray) -> Optional[int]:
    for a, b, c in WIN_LINES:
        total = int(board[a] + board[b] + board[c])
        if total == 3:
            return AGENT_MARK
        if total == -3:
            return OPPONENT_MARK
    if not legal_actions(board):
        return DRAW
    return None


def encode_state(board: np.ndarray, mark: int = AGENT_MARK) -> np.ndarray:
    if mark == AGENT_MARK:
        return board.astype(np.float32)
    return (-board).astype(np.float32)


def action_to_coords(action: int) -> tuple[int, int]:
    return divmod(action, 3)


def outcome_to_score(outcome: str) -> float:
    if outcome == "win":
        return 1.0
    if outcome == "draw":
        return 0.5
    return 0.0


@lru_cache(maxsize=None)
def _negamax_value(board_key: tuple[int, ...], current_mark: int) -> float:
    board = np.array(board_key, dtype=np.int8)
    winner = check_winner(board)
    if winner is not None:
        if winner == DRAW:
            return 0.0
        return 1.0 if winner == current_mark else -1.0

    best_value = -math.inf
    for action in legal_actions(board):
        child = apply_action(board, action, current_mark)
        value = -_negamax_value(tuple(child.tolist()), -current_mark)
        if value > best_value:
            best_value = value
    return best_value


class OpponentPolicy:
    def select_action(self, board: np.ndarray, mark: int, rng: random.Random) -> int:
        raise NotImplementedError


class RandomPolicy(OpponentPolicy):
    def select_action(self, board: np.ndarray, mark: int, rng: random.Random) -> int:
        actions = legal_actions(board)
        return actions[rng.randrange(len(actions))]


class MinimaxPolicy(OpponentPolicy):
    def select_action(self, board: np.ndarray, mark: int, rng: random.Random) -> int:
        best_value = -math.inf
        best_actions: List[int] = []
        board_key = tuple(board.tolist())
        for action in legal_actions(board):
            child = apply_action(board, action, mark)
            value = -_negamax_value(tuple(child.tolist()), -mark)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)
        return best_actions[rng.randrange(len(best_actions))]


class MixedPolicy(OpponentPolicy):
    def __init__(self, optimal_probability: float = 0.75):
        self.optimal_probability = optimal_probability
        self.minimax = MinimaxPolicy()
        self.random_policy = RandomPolicy()

    def select_action(self, board: np.ndarray, mark: int, rng: random.Random) -> int:
        if rng.random() < self.optimal_probability:
            return self.minimax.select_action(board, mark, rng)
        return self.random_policy.select_action(board, mark, rng)


class TicTacToeEnv:
    def __init__(
        self,
        opponent_policy: Optional[OpponentPolicy] = None,
        agent_starts_probability: float = 0.5,
        win_reward: float = 1.0,
        draw_reward: float = 0.35,
        loss_reward: float = -1.0,
        invalid_reward: float = -1.0,
        step_reward: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.opponent_policy = opponent_policy or MixedPolicy()
        self.agent_starts_probability = agent_starts_probability
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.loss_reward = loss_reward
        self.invalid_reward = invalid_reward
        self.step_reward = step_reward
        self.rng = random.Random(seed)
        self.board = new_board()

    def reset(
        self,
        seed: Optional[int] = None,
        agent_starts: Optional[bool] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng.seed(seed)
        self.board = new_board()
        if agent_starts is None:
            agent_starts = self.rng.random() < self.agent_starts_probability
        if not agent_starts:
            action = self.opponent_policy.select_action(self.board, OPPONENT_MARK, self.rng)
            self.board[action] = OPPONENT_MARK
        return encode_state(self.board), {
            "board": self.board.copy(),
            "agent_starts": agent_starts,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.board[action] != EMPTY:
            return encode_state(self.board), self.invalid_reward, True, {
                "board": self.board.copy(),
                "outcome": "invalid",
            }

        self.board[action] = AGENT_MARK
        winner = check_winner(self.board)
        if winner is not None:
            if winner == AGENT_MARK:
                return encode_state(self.board), self.win_reward, True, {
                    "board": self.board.copy(),
                    "outcome": "win",
                }
            return encode_state(self.board), self.draw_reward, True, {
                "board": self.board.copy(),
                "outcome": "draw",
            }

        opponent_action = self.opponent_policy.select_action(self.board, OPPONENT_MARK, self.rng)
        self.board[opponent_action] = OPPONENT_MARK
        winner = check_winner(self.board)
        if winner is not None:
            if winner == OPPONENT_MARK:
                return encode_state(self.board), self.loss_reward, True, {
                    "board": self.board.copy(),
                    "outcome": "loss",
                }
            return encode_state(self.board), self.draw_reward, True, {
                "board": self.board.copy(),
                "outcome": "draw",
            }

        return encode_state(self.board), self.step_reward, False, {
            "board": self.board.copy(),
            "outcome": "ongoing",
        }


def render_board_image(
    board: np.ndarray,
    title: Optional[str] = None,
    footer: Optional[str] = None,
    size: int = 360,
) -> np.ndarray:
    bg_color = (248, 242, 229)
    grid_color = (37, 54, 71)
    x_color = (220, 83, 74)
    o_color = (51, 122, 183)
    text_color = (37, 54, 71)
    margin = 24
    cell = (size - margin * 2) // 3
    title_height = 54 if title else 0
    footer_height = 40 if footer else 0
    image = Image.new("RGB", (size, size + title_height + footer_height), bg_color)
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
        text_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except Exception:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    if title:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        draw.text(((size - (bbox[2] - bbox[0])) / 2, 10), title, fill=text_color, font=title_font)

    offset_y = title_height
    for idx in range(1, 3):
        x = margin + idx * cell
        draw.line((x, offset_y + margin, x, offset_y + size - margin), fill=grid_color, width=6)
        y = offset_y + margin + idx * cell
        draw.line((margin, y, size - margin, y), fill=grid_color, width=6)

    for idx, value in enumerate(board.tolist()):
        row, col = action_to_coords(idx)
        left = margin + col * cell
        top = offset_y + margin + row * cell
        if value == AGENT_MARK:
            draw.ellipse((left + 18, top + 18, left + cell - 18, top + cell - 18), outline=o_color, width=8)
        elif value == OPPONENT_MARK:
            draw.line((left + 18, top + 18, left + cell - 18, top + cell - 18), fill=x_color, width=8)
            draw.line((left + cell - 18, top + 18, left + 18, top + cell - 18), fill=x_color, width=8)

    if footer:
        bbox = draw.textbbox((0, 0), footer, font=text_font)
        y = size + title_height + 8
        draw.text(((size - (bbox[2] - bbox[0])) / 2, y), footer, fill=text_color, font=text_font)

    return np.array(image)
