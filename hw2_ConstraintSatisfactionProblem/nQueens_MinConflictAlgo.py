import random
import time

"""
Information used from:
    https://www.geeksforgeeks.org/constraint-satisfaction-problems-csp-in-artificial-intelligence/
    https://www.geeksforgeeks.org/n-queen-problem-local-search-using-hill-climbing-with-random-neighbour/

    and help from a friend of the code, because I cannot achive input = 10000 in less than a second.
"""

MAX_ITERATIONS = 1000000

def user_input() -> int:
    return int(input("Enter N for N-Queens (or less than 4 to exit): "))

def print_board(board, number_of_queens):
    for row in range(number_of_queens):
        line = ""
        for col in range(number_of_queens):
            if board[col] == row:
                line += '* '
            else:
                line += '_ '
        print(line.strip())

def init_queens(number_of_queens: int):
    """Initialize the board with random positions for queens."""
    return [random.randint(0, number_of_queens - 1) for _ in range(number_of_queens)]

def get_conflicts(board: list[int], number_of_queens: int):
    """
    Get the conflicts by:
        - row
        - left diagonal
        - right diagonal
    then unite those conflicts in on collection and return it.
    """
    row_conflicts = [0] * number_of_queens
    left_diag_conflicts = [0] * (2 * number_of_queens - 1)
    right_diag_conflicts = [0] * (2 * number_of_queens - 1)
    conflicts = [0] * number_of_queens

    for col in range(number_of_queens):
        row = board[col]
        row_conflicts[row] += 1
        left_diag_conflicts[row + col] += 1
        right_diag_conflicts[row - col + number_of_queens - 1] += 1

    for col in range(number_of_queens):
        row = board[col]
        conflicts[col] = (
            row_conflicts[row] +
            left_diag_conflicts[row + col] +
            right_diag_conflicts[row - col + number_of_queens - 1] 
            - 3 #minus 3, because count the intersection 3x
        )
    
    return conflicts

def gen_rand_position(conflicts: list[int]):
    """Choose a random column where a queen has conflicts."""
    conflict_columns = [col for col, conflict in enumerate(conflicts) if conflict > 0]
    return random.choice(conflict_columns)

def find_row_conflicts(board: list[int], column: int, number_of_queens: int):
    """Find row conflicts for a particular column."""
    row_conflicts = [0] * number_of_queens
    for col in range(number_of_queens):
        if col == column:
            continue
        row = board[col]
        row_conflicts[row] += 1
        left_diag = row - (column - col)
        right_diag = row + (column - col)
        if 0 <= left_diag < number_of_queens:
            row_conflicts[left_diag] += 1
        if 0 <= right_diag < number_of_queens:
            row_conflicts[right_diag] += 1
    return row_conflicts

def find_position_with_equal_value(row_conflicts: list[int], min_conflict_value: int) -> int:
    """Choose a row with the minimum conflict value."""
    rows = [row for row, conflict in enumerate(row_conflicts) if conflict == min_conflict_value]
    return random.choice(rows)

def is_valid_solution(queens: list[int]) -> bool:
    """Check if the current configuration has no conflicts."""
    n = len(queens)
    for i in range(n):
        for j in range(i + 1, n):
            if queens[i] == queens[j] or abs(queens[i] - queens[j]) == abs(i - j):
                return False
    return True

def update_conflicts(
    queens: list[int], conflicts: list[int], column: int, row: int, n: int
) -> int:
    """Update conflict values based on the move to a new row in a column."""
    old_row = queens[column]

    # Recalculate conflicts for affected queens
    for col in range(n):
        if col == column:
            continue
        col_diff = abs(column - col)
        if queens[col] == old_row or abs(queens[col] - old_row) == col_diff:
            conflicts[col] -= 1
        if queens[col] == row or abs(queens[col] - row) == col_diff:
            conflicts[col] += 1

    queens[column] = row  # Move the queen
    conflicts[column] = 0  # Reset conflict count for this queen

    # Recalculate the conflicts for the moved queen
    for col in range(n):
        if col == column:
            continue
        col_diff = abs(column - col)
        if queens[col] == row or abs(queens[col] - row) == col_diff:
            conflicts[column] += 1

    return sum(conflicts)

def solve(number_of_queens: int):
    """Solve the N-Queens problem using hill-climbing with random restarts."""
    attempts = 0

    while attempts < number_of_queens**2:
        board = init_queens(number_of_queens)
        conflicts = get_conflicts(board, number_of_queens)
        total_conflicts = sum(conflicts)

        for _ in range(MAX_ITERATIONS):
            if total_conflicts == 0:
                return board  # Solution found

            col = gen_rand_position(conflicts)
            row_conflicts = find_row_conflicts(board, col, number_of_queens)
            min_conflicts = min(row_conflicts)
            row = find_position_with_equal_value(row_conflicts, min_conflicts)
            total_conflicts = update_conflicts(board, conflicts, col, row, number_of_queens)

        if is_valid_solution(board):
            break
        attempts += 1  # Restart the algorithm with a new initial configuration

    return None  # Return None if solution is not found after many restarts

def main():
    while True:
        n = user_input()
        if n < 4:
            break
        start_time = time.time()
        result = solve(n)
        duration = time.time() - start_time
        if result:
            print("Solution found:")
            print(result)
            print_board(result, n)
        else:
            print("Solution not found.")
        print(f"Elapsed time: {duration:.3f}s")

if __name__ == "__main__":
    main()
