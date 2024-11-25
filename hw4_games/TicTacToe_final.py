import math

# Global board state
board = [None] * 9

# Winning conditions for Tic-Tac-Toe
WIN_CONDITIONS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
    [0, 4, 8], [2, 4, 6]              # Diagonals
]

def print_board():
    """Prints the current state of the board."""
    for i in range(3):
        row = [" " if board[3 * i + j] is None else board[3 * i + j] for j in range(3)]
        print(" | ".join(row))
        if i < 2:
            print("---------")

def convert_position(pos):
    """Converts a (row, col) position to the corresponding linear index."""
    row, col = pos
    return (row - 1) * 3 + (col - 1)

def player_move():
    """Handles the player's move."""
    while True:
        try:
            move_input = input("Enter your move as row,col (e.g., 1,3): ")
            row, col = map(int, move_input.split(","))
            move = convert_position((row, col))
            if move < 0 or move >= 9 or board[move] is not None:
                print("Invalid move, try again.")
                continue
            board[move] = 'O'
            break
        except ValueError:
            print("Please enter valid row and column numbers (e.g., 1,3).")

def check_result():
    """Determines the result and prints it."""
    result = evaluate_position(board)
    if result == 1:
        print("AI wins!")
    elif result == -1:
        print("You win!")
    else:
        print("It's a draw!")

def is_game_over(board):
    """Checks if the game is over."""
    for condition in WIN_CONDITIONS:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] and board[condition[0]] is not None:
            return True
    return all(cell is not None for cell in board)  # Draw if board is full

def evaluate_position(board):
    """Evaluates the board and returns a score."""
    for condition in WIN_CONDITIONS:
        if board[condition[0]] == board[condition[1]] == board[condition[2]]:
            if board[condition[0]] == 'X':  # AI wins
                return 1
            elif board[condition[0]] == 'O':  # Player wins
                return -1
    return 0  # Draw or ongoing game

def get_children(board, player):
    """Generates all possible moves for the current player and returns them as new board states."""
    children = []
    for i in range(9):
        if board[i] is None:
            new_board = board[:]
            new_board[i] = player
            children.append(new_board)
    return children

def minimax(board, depth, alpha, beta, maximizingPlayer):
    """The minimax function with alpha-beta pruning and corrected bottom cases."""
    if is_game_over(board):
        # Bottom case evaluation with depth adjustment
        if evaluate_position(board) == 1:  # AI wins
            return 10 - depth
        elif evaluate_position(board) == -1:  # Player wins
            return depth - 10
        else:  # Draw
            return 0

    if maximizingPlayer:
        maxEval = -math.inf
        for child in get_children(board, 'X'):
            eval = minimax(child, depth + 1, alpha, beta, False)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = math.inf
        for child in get_children(board, 'O'):
            eval = minimax(child, depth + 1, alpha, beta, True)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval

def best_move(board):
    """Finds the best move for the AI (player 'X') using the minimax function."""
    best_val = -math.inf
    best_move = None
    for i in range(9):
        if board[i] is None:
            board[i] = 'X'
            move_val = minimax(board, 0, -math.inf, math.inf, False)
            board[i] = None
            if move_val > best_val:
                best_val = move_val
                best_move = i
    return best_move

# Main game loop
def main():
    print("Welcome to Tic-Tac-Toe! You are 'O' and the AI is 'X'.")
    first_player = input("Do you want to play first? (yes / anything else): ").strip().lower()
    print_board()
    
    while not is_game_over(board):
        if first_player == "yes":
            player_move()
            print_board()
            if is_game_over(board):
                break
            print("AI is thinking...")
            ai_move = best_move(board)
            if ai_move is not None:
                board[ai_move] = 'X'
            print_board()
        else:
            print("AI is thinking...")
            ai_move = best_move(board)
            if ai_move is not None:
                board[ai_move] = 'X'
            print_board()
            if is_game_over(board):
                break
            player_move()
            print_board()

    print("Game over!")
    check_result()

if __name__ == "__main__":
    main()
