import math

# Define the board as a list of 9 elements
# Empty spaces are represented by None, 'X' is the AI, and 'O' is the player.
board = [None] * 9

# Helper functions
def print_board():
    """Prints the current state of the board."""
    for i in range(3):
        row = [" " if board[3 * i + j] is None else board[3 * i + j] for j in range(3)]
        print(" | ".join(row))
        if i < 2:
            print("---------")

def is_game_over(board):
    """Checks if the game is over and returns True if it is."""
    # Win conditions
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # Columns
        [0, 4, 8], [2, 4, 6]             # Diagonals
    ]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] and board[condition[0]] is not None:
            return True
    # Draw condition
    if all(x is not None for x in board):
        return True
    return False

def evaluate_position(board):
    """Evaluates the board and returns a score."""
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    for condition in win_conditions:
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
    """The minimax function with alpha-beta pruning."""
    if depth == 0 or is_game_over(board):
        return evaluate_position(board)

    if maximizingPlayer:
        maxEval = -math.inf
        for child in get_children(board, 'X'):
            eval = minimax(child, depth - 1, alpha, beta, False)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = math.inf
        for child in get_children(board, 'O'):
            eval = minimax(child, depth - 1, alpha, beta, True)
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
            move_val = minimax(board, 5, -math.inf, math.inf, False) #5 - consider this as Medium mode. Easy = 3, Impossible = 9
            board[i] = None
            if move_val > best_val:
                best_val = move_val
                best_move = i
    return best_move

# Main game loop
def main():
    print("Welcome to Tic-Tac-Toe! You are 'O' and the AI is 'X'.")
    print_board()
    
    while not is_game_over(board):
        # Player move
        move = int(input("Enter your move (1-9): ")) - 1
        if board[move] is not None:
            print("Invalid move, try again.")
            continue
        board[move] = 'O'
        
        # Check if game is over after player's move
        if is_game_over(board):
            print("Game over!")
            print_board()
            break
        
        # AI move
        ai_move = best_move(board)
        if ai_move is not None:
            board[ai_move] = 'X'
            print("AI makes a move:")
            print_board()
        
        # Check if game is over after AI's move
        if is_game_over(board):
            print("Game over!")
            print_board()
            break

    # Determine the result
    result = evaluate_position(board)
    if result == 1:
        print("AI wins!")
    elif result == -1:
        print("You win!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main()
