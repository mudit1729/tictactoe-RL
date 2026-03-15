const state = {
  board: Array(9).fill(0),
  modelPath: "dqn_mcts_model.json",
  model: null,
  gameOver: false,
  aiStarts: false,
  humanSymbol: "X",
  aiSymbol: "O",
  scores: { human: 0, draw: 0, ai: 0 },
};

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("statusText");
const modelSelect = document.getElementById("modelSelect");
const modelBadge = document.getElementById("modelBadge");
const humanScoreEl = document.getElementById("humanScore");
const drawScoreEl = document.getElementById("drawScore");
const aiScoreEl = document.getElementById("aiScore");

const WIN_LINES = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
];

function legalActions(board) {
  return board.flatMap((value, index) => (value === 0 ? [index] : []));
}

function checkWinner(board) {
  for (const [a, b, c] of WIN_LINES) {
    const sum = board[a] + board[b] + board[c];
    if (sum === 3) return 1;
    if (sum === -3) return -1;
  }
  return board.includes(0) ? null : 0;
}

function relu(value) {
  return value > 0 ? value : 0;
}

function dense(input, layer) {
  return layer.weight.map((row, rowIndex) => {
    let total = layer.bias[rowIndex];
    for (let i = 0; i < input.length; i += 1) {
      total += row[i] * input[i];
    }
    return total;
  });
}

function qValues(board) {
  const input = board.slice();
  const hidden1 = dense(input, state.model.layers[0]).map(relu);
  const hidden2 = dense(hidden1, state.model.layers[1]).map(relu);
  return dense(hidden2, state.model.layers[2]);
}

function chooseAiAction(board) {
  const q = qValues(board);
  const actions = legalActions(board);
  let bestAction = actions[0];
  let bestValue = -Infinity;
  for (const action of actions) {
    if (q[action] > bestValue) {
      bestValue = q[action];
      bestAction = action;
    }
  }
  return bestAction;
}

function symbolForCell(value) {
  if (value === 1) return state.aiSymbol;
  if (value === -1) return state.humanSymbol;
  return "";
}

function updateScores() {
  humanScoreEl.textContent = state.scores.human;
  drawScoreEl.textContent = state.scores.draw;
  aiScoreEl.textContent = state.scores.ai;
}

function renderBoard() {
  boardEl.innerHTML = "";
  state.board.forEach((value, index) => {
    const button = document.createElement("button");
    button.className = "cell";
    if (value === 1) button.classList.add("ai");
    if (value === -1) button.classList.add("human");
    button.textContent = symbolForCell(value);
    button.disabled = state.gameOver || value !== 0 || !state.model;
    button.addEventListener("click", () => handleHumanMove(index));
    boardEl.appendChild(button);
  });
}

function finishGame(winner) {
  state.gameOver = true;
  if (winner === 1) {
    state.scores.ai += 1;
    statusEl.textContent = `AI wins as ${state.aiSymbol}.`;
  } else if (winner === -1) {
    state.scores.human += 1;
    statusEl.textContent = `You win as ${state.humanSymbol}.`;
  } else {
    state.scores.draw += 1;
    statusEl.textContent = "Draw.";
  }
  updateScores();
  renderBoard();
}

function resolveBoard() {
  const winner = checkWinner(state.board);
  if (winner !== null) {
    finishGame(winner);
    return true;
  }
  return false;
}

function runAiTurn() {
  if (state.gameOver || !state.model) return;
  const action = chooseAiAction(state.board);
  state.board[action] = 1;
  renderBoard();
  if (!resolveBoard()) {
    statusEl.textContent = `Your turn as ${state.humanSymbol}.`;
  }
}

function handleHumanMove(index) {
  if (state.gameOver || state.board[index] !== 0 || !state.model) return;
  state.board[index] = -1;
  renderBoard();
  if (resolveBoard()) return;
  statusEl.textContent = `AI thinking as ${state.aiSymbol}...`;
  window.setTimeout(runAiTurn, 220);
}

function resetBoard(aiStarts) {
  state.board = Array(9).fill(0);
  state.gameOver = false;
  state.aiStarts = aiStarts;
  state.humanSymbol = aiStarts ? "O" : "X";
  state.aiSymbol = aiStarts ? "X" : "O";
  renderBoard();
  statusEl.textContent = aiStarts
    ? `AI starts as ${state.aiSymbol}.`
    : `You start as ${state.humanSymbol}.`;
  if (aiStarts) {
    window.setTimeout(runAiTurn, 260);
  }
}

async function loadModel(fileName) {
  statusEl.textContent = "Loading exported weights...";
  const response = await fetch(`./models/${fileName}`);
  state.model = await response.json();
  modelBadge.textContent = state.model.metadata.label || "Model loaded";
  statusEl.textContent = "Model ready. Start a new game.";
  renderBoard();
}

document.getElementById("humanFirstBtn").addEventListener("click", () => resetBoard(false));
document.getElementById("aiFirstBtn").addEventListener("click", () => resetBoard(true));
modelSelect.addEventListener("change", async (event) => {
  await loadModel(event.target.value);
  resetBoard(false);
});

updateScores();
renderBoard();
loadModel(state.modelPath).then(() => resetBoard(false));
