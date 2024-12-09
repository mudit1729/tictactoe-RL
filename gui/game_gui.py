import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tictactoe.agent import Qlearner, SARSAlearner
from tictactoe.game import Game, getStateKey

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe vs AI")
        self.root.configure(bg='#2D2D2D')  # Dark background
        
        # Initialize or load the agent
        self.initialize_agent()
        
        # Game state
        self.current_game = None
        self.buttons = []
        self.status_label = None
        
        # Create GUI elements
        self.create_gui()
        
        # Center the window
        self.root.update()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 600
        window_height = 800
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
            
    def initialize_agent(self):
        """Initialize or load a pre-trained agent"""
        agent_path = 'q_agent.pkl'
        if os.path.exists(agent_path):
            with open(agent_path, 'rb') as f:
                self.agent = pickle.load(f)
        else:
            self.agent = Qlearner(alpha=0.5, gamma=0.9, epsilon=0.1)
            
    def create_gui(self):
        """Create the main GUI elements"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2D2D2D', padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="Tic Tac Toe vs AI",
            font=('Helvetica', 36, 'bold'),
            fg='black',  # Changed to black
            bg='#2D2D2D'
        )
        title_label.pack(pady=(0, 20))
        
        # Status label
        self.status_label = tk.Label(
            main_frame,
            text="You are 'X' and the AI is 'O'",
            font=('Helvetica', 18),
            fg='black',  # Changed to black
            bg='#2D2D2D'
        )
        self.status_label.pack(pady=(0, 40))
        
        # Game board frame
        game_frame = tk.Frame(main_frame, bg='#2D2D2D')
        game_frame.pack(pady=20)
        
        # Create game board buttons
        BUTTON_SIZE = 150  # Size in pixels
        for i in range(3):
            row_frame = tk.Frame(game_frame, bg='#2D2D2D')
            row_frame.pack()
            for j in range(3):
                btn = tk.Button(
                    row_frame,
                    text='',
                    font=('Helvetica', 48, 'bold'),
                    width=3,
                    height=1,
                    bg='#404040',
                    fg='black',  # Changed to black
                    activebackground='#505050',
                    activeforeground='black',  # Added for consistency
                    relief='flat',
                    command=lambda row=i, col=j: self.make_move(row, col)
                )
                btn.pack(side='left', padx=5, pady=5)
                btn.configure(width=4, height=2)  # Make buttons square
                self.buttons.append(btn)
                
        # Control buttons frame
        control_frame = tk.Frame(main_frame, bg='#2D2D2D')
        control_frame.pack(pady=40)
        
        button_style = {
            'font': ('Helvetica', 14),
            'width': 20,
            'height': 2,
            'bg': '#404040',
            'fg': 'black',  # Changed to black
            'activebackground': '#505050',
            'activeforeground': 'black',  # Added for consistency
            'relief': 'flat'
        }
        
        # New Game buttons
        tk.Button(
            control_frame,
            text="New Game (You First)",
            command=self.new_game,
            **button_style
        ).pack(side='left', padx=10)
        
        tk.Button(
            control_frame,
            text="New Game (AI First)",
            command=lambda: self.new_game(ai_first=True),
            **button_style
        ).pack(side='left', padx=10)
        
        # Start first game
        self.new_game()
        
    def new_game(self, ai_first=False):
        """Start a new game"""
        # Reset buttons
        for btn in self.buttons:
            btn.configure(text='', state='normal', bg='#404040')
            
        # Create new game instance
        self.current_game = Game(self.agent)
        self.current_game.board = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
        
        # Update status
        self.status_label.configure(
            text="AI's turn..." if ai_first else "Your turn!"
        )
        
        if ai_first:
            self.root.after(1000, self.make_ai_move)
            
    def make_move(self, row, col):
        """Handle player's move"""
        if self.current_game.board[row][col] != '-':
            return
            
        # Update game state and GUI
        self.current_game.board[row][col] = 'X'
        button = self.buttons[row * 3 + col]
        button.configure(text='X', bg='#505050')
        
        # Check for game end
        if self.check_game_end('X'):
            return
            
        # Update status and make AI move after a short delay
        self.status_label.configure(text="AI's turn...")
        self.root.after(1000, self.make_ai_move)
        
    def make_ai_move(self):
        """Handle AI's move"""
        # Get current state
        state = getStateKey(self.current_game.board)
        
        # Get AI's action
        action = self.agent.get_action(state)
        row, col = action
        
        # Update game state and GUI
        self.current_game.board[row][col] = 'O'
        button = self.buttons[row * 3 + col]
        button.configure(text='O', bg='#505050')
        
        # Check for game end
        if not self.check_game_end('O'):
            self.status_label.configure(text="Your turn!")
        
    def check_game_end(self, player):
        """Check if game has ended and handle end game state"""
        if self.current_game.checkForWin(player):
            winner = "You won!" if player == 'X' else "AI won!"
            self.status_label.configure(text=winner)
            messagebox.showinfo("Game Over", winner)
            self.disable_all_buttons()
            return True
            
        elif self.current_game.checkForDraw():
            self.status_label.configure(text="It's a draw!")
            messagebox.showinfo("Game Over", "It's a draw!")
            self.disable_all_buttons()
            return True
            
        return False
        
    def disable_all_buttons(self):
        """Disable all game board buttons"""
        for btn in self.buttons:
            btn.configure(state='disabled')

def main():
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()