# Tic Tac Toe with Reinforcement Learning GUI

This is a graphical interface for playing Tic-tac-toe against an AI that uses reinforcement learning algorithms (Q-Learning and SARSA). The GUI interface is shown in `tic_tac_gui.png`:

![Tic Tac Toe GUI Screenshot](tic_tac_gui.png)

## Features

- Clean, modern dark-themed interface
- Large, easy-to-click game buttons
- Option to start new games
- Choice of who moves first (player or AI)
- Visual feedback for game progress
- Automatic AI responses
- Win/Draw detection and game-over notifications

## Running the Game

To play against the AI:

```bash
python tic_tac_gui.py
```

The game will automatically load a trained agent if one exists (q_agent.pkl), or create a new one if none is found. The agent continues to learn from each game played.