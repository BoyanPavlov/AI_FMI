import math

board = [[None] * 3 for _ in range(3)] 

# Constant for win conditions
WIN_CONDITIONS = [
    [(0, 0), (0, 1), (0, 2)],  # Rows
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],

    [(0, 0), (1, 0), (2, 0)],  # Columns
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],

    [(0, 0), (1, 1), (2, 2)],  # Diagonals
    [(0, 2), (1, 1), (2, 0)],
]


def print_board():
    """Prints the current state of the board."""
    for row in board:
        row_symbols = [" " if cell is None else cell for cell in row]
        print(" | ".join(row_symbols))
        print("---------")

def is_game_over(board):
    """Checks if the game is over and returns True if it is."""
    for condition in WIN_CONDITIONS:
        if (
            board[condition[0][0]][condition[0][1]] ==
            board[condition[1][0]][condition[1][1]] ==
            board[condition[2][0]][condition[2][1]] and
            board[condition[0][0]][condition[0][1]] is not None
        ):
            return True
    if all(all(cell is not None for cell in row) for row in board):  # Draw condition
        return True
    return False

def evaluate_position(board):
    """Evaluates the board and returns a score."""
    for condition in WIN_CONDITIONS:
        if (
            board[condition[0][0]][condition[0][1]] ==
            board[condition[1][0]][condition[1][1]] ==
            board[condition[2][0]][condition[2][1]]
        ):
            if board[condition[0][0]][condition[0][1]] == 'X':  # AI wins
                return 1
            elif board[condition[0][0]][condition[0][1]] == 'O':  # Player wins
                return -1
    return 0  # Draw or ongoing game

def get_children(board, player):
    """Generates all possible moves for the current player and returns them as new board states."""
    children = []
    for row in range(3):
        for col in range(3):
            if board[row][col] is None:
                new_board = [r[:] for r in board]
                new_board[row][col] = player
                children.append(new_board)
    return children

def minimax(board, depth, alpha, beta, maximizingPlayer):
    """The minimax function with alpha-beta pruning."""
    score = evaluate_position(board)
    if depth == 0 or is_game_over(board):
        return score

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

def best_move(board, depth=5):
    """Finds the best move for the AI (player 'X') using the minimax function."""
    best_val = -math.inf
    best_move = None
    for row in range(3):
        for col in range(3):
            if board[row][col] is None:
                board[row][col] = 'X'
                move_val = minimax(board, depth, -math.inf, math.inf, False)
                board[row][col] = None
                if move_val > best_val:
                    best_val = move_val
                    best_move = (row, col)
    return best_move

def player_move():
    """Handles the player's move."""
    while True:
        try:
            move = input("Enter your move as row,col (e.g., 1,3): ")
            row, col = map(int, move.split(","))
            row -= 1
            col -= 1
            if row < 0 or row >= 3 or col < 0 or col >= 3 or board[row][col] is not None:
                print("Invalid move, try again.")
                continue
            board[row][col] = 'O'
            break
        except ValueError:
            print("Please enter valid row and column numbers (e.g., 1,3).")

def check_result():
    """Determines the result and prints it."""
    result = evaluate_position(board)
    if result == 10:
        print("AI wins!")
    elif result == -10:
        print("You win!")
    else:
        print("It's a draw!")

# Main game loop
#!BAD VERSION - NOT ALL CASES WITH DIAGONALS AND ETC. COVERED, have a look at the 82027 version
def main():
    print("Welcome to Tic-Tac-Toe! You are 'O' and the AI is 'X'.")
    first_player = input("Do you want to play first? (yes/no): ").strip().lower()
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
                board[ai_move[0]][ai_move[1]] = 'X'
            print_board()
        else:
            print("AI is thinking...")
            ai_move = best_move(board)
            if ai_move is not None:
                board[ai_move[0]][ai_move[1]] = 'X'
            print_board()
            if is_game_over(board):
                break
            player_move()
            print_board()
    
    print("Game over!")
    check_result()

if __name__ == "__main__":
    main()
