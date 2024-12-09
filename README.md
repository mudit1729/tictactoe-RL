# Tic-Tac-Toe with Reinforcement Learning
This is a repository for training an AI agent to play Tic-tac-toe using reinforcement learning. Both the SARSA and Q-learning RL algorithms are implemented. A user may teach the agent themself by playing against it or apply an automated teacher agent. 

## Algorithm Explanation

The project implements two reinforcement learning algorithms: Q-Learning and SARSA (State-Action-Reward-State-Action). Both are temporal difference learning methods but differ in how they update their value estimates.

### Q-Learning (Off-policy TD Control)
- The agent maintains a Q-table that stores values for each state-action pair
- In each state, the agent either:
  * Exploits: Chooses the action with highest Q-value (probability 1-ε)
  * Explores: Chooses a random action (probability ε)
- After each action, the Q-value is updated using the formula:
  ```
  Q(s,a) = Q(s,a) + α[R + γ * max(Q(s',a')) - Q(s,a)]
  ```
  where:
  - α (alpha) is the learning rate (0.5 in this implementation)
  - γ (gamma) is the discount factor (0.9 here)
  - R is the reward
  - s is current state, s' is next state
  - a is current action, a' is next action

### SARSA (On-policy TD Control)
- Similar to Q-learning but uses the actual next action instead of the maximum Q-value
- Updates Q-values using:
  ```
  Q(s,a) = Q(s,a) + α[R + γ * Q(s',a') - Q(s,a)]
  ```
  where a' is the actual next action taken, not the maximum possible

### Teaching Strategy
- The project uses a teacher agent that knows optimal moves
- The teacher only plays optimally with a certain probability
- Otherwise chooses random valid moves
- This helps the agent learn by occasionally allowing it to win and learn from those experiences

### State and Reward Structure
- The game board is converted to a string key for the Q-table
- Each cell is represented as 'X', 'O', or '-'
- All possible board configurations are states
- Rewards:
  * Win: +1 reward
  * Loss: -1 reward
  * Draw: 0 reward
  * Intermediate moves: 0 reward

The learning process can be initialized in two ways:
1. Manual training: Human plays against the agent
2. Automated training: Teacher agent plays thousands of games against the agent

## Code Structure

The directory `tictactoe` contains the core source code for this project.
There are 3 main source code files:
1. game.py - Contains the Game class and core game functionality
2. agent.py - Implements Q-learning and SARSA agents
3. teacher.py - Implements the teaching agent for automated training

#### GUI Version

A graphical user interface version of the game is available in the `gui` directory. The GUI is built using Python's Tkinter library and provides an easy-to-use interface for playing against the trained agent. To run the GUI version:

    python gui/game_gui.py

The GUI version will automatically load a trained agent if one exists (q_agent.pkl), or create a new one if none is found. Features include:
- Clean, modern interface
- Option to start new games
- Choice of who moves first (player or AI)
- Visual feedback for game progress and results
- Automatic saving of agent's learning

## Running the Program

#### Using the GUI
To play against the AI agent using the graphical interface:

1. Simply run:
    ```
    python gui/game_gui.py
    ```
2. Click the squares to make your moves
3. Use "New Game" to start a fresh game with you moving first
4. Use "AI First" to start a new game with the AI moving first

The GUI version will automatically save the agent's learning progress.

#### Training a New Agent (Command Line)
To initialize a new agent and begin a game loop, simply run:

    python play.py -a q                (Q-learner)
    python play.py -a s                (Sarsa-learner)

This will initialize the game and allow you to train the agent manually by playing against the agent yourself. In the process of playing, you will be storing the new agent state with each game iteration. Use the argument `-p` to specify a path where the agent pickle should be saved:

    python play.py -a q -p my_agent_path.pkl

When unspecified, the path is set to either "q_agent.pkl" or "sarsa_agent.pkl" depending on agent type. If the file already exists, you'll be asked to overwrite.

#### Train a new agent automatically via teacher
To initialize a new RL agent and train it automatically with a teacher agent, use the flag `-t` followed by the number of game iterations you would like to train for:

    python play.py -a q -t 5000

Again, specify the pickle save path with the `-p` option.

#### Load an existing agent and continue training
To load an existing agent and continue training, use the `-l` flag:

    python play.py -a q -l             (load agent and train manually)
    python play.py -a q -l -t 5000     (load agent and train via teacher)

The agent will continue to learn and its pickle file will be overwritten. 

For this use case, the argument `-a` is only used to define a default agent path (if not specified by `-p`); otherwise, the agent type is determined by the contents of the loaded pickle.

#### Load a trained agent and view reward history plot
Finally, to load a stored agent and view a plot of its cumulative reward history, use the script plot_agent_reward.py:

    python plot_agent_reward.py -p q_agent.pkl