import random
import time


def init(N):
    """Initialize queens in a random configuration, one queen per column."""
    return [random.randint(0, N-1) for _ in range(N)]


def get_conflicts(queens, N):
    """Compute conflicts for each queen based on current positions."""
    row_conflicts = [0] * N
    diag1_conflicts = [0] * (2 * N - 1)
    diag2_conflicts = [0] * (2 * N - 1)

    for col in range(N):
        row = queens[col]
        row_conflicts[row] += 1
        diag1_conflicts[row + col] += 1
        diag2_conflicts[row - col + N - 1] += 1

    return row_conflicts, diag1_conflicts, diag2_conflicts


def compute_conflicts(row, col, row_conflicts, diag1_conflicts, diag2_conflicts, N):
    """Calculate the conflict count for placing a queen at (row, col)."""
    return (row_conflicts[row] +
            diag1_conflicts[row + col] +
            diag2_conflicts[row - col + N - 1] - 3)  # subtract 3 to exclude self


def min_conflict_move(queens, row_conflicts, diag1_conflicts, diag2_conflicts, col, N):
    """Move queen in column 'col' to the row with the minimum conflicts."""
    min_conflicts = N
    best_row = queens[col]

    for row in range(N):
        conflicts = compute_conflicts(row, col, row_conflicts, diag1_conflicts, diag2_conflicts, N)
        if conflicts < min_conflicts:
            min_conflicts = conflicts
            best_row = row

    return best_row


def update_conflicts(row, col, row_conflicts, diag1_conflicts, diag2_conflicts, N, increment):
    """Update conflict counts."""
    delta = 1 if increment else -1
    row_conflicts[row] += delta
    diag1_conflicts[row + col] += delta
    diag2_conflicts[row - col + N - 1] += delta


def solve_n_queens(N, max_steps=100000):
    """Main function to solve the N-Queens problem using optimized min-conflict."""
    queens = init(N)
    row_conflicts, diag1_conflicts, diag2_conflicts = get_conflicts(queens, N)

    for step in range(max_steps):
        max_conflict = 0
        col_with_conflict = -1

        # Find the queen with the highest conflicts
        for col in range(N):
            row = queens[col]
            conflicts = compute_conflicts(row, col, row_conflicts, diag1_conflicts, diag2_conflicts, N)
            if conflicts > max_conflict:
                max_conflict = conflicts
                col_with_conflict = col

        # If there are no conflicts, the solution is found
        if max_conflict == 0:
            return queens

        # Move queen in the identified column to minimize conflicts
        col = col_with_conflict
        current_row = queens[col]
        best_row = min_conflict_move(queens, row_conflicts, diag1_conflicts, diag2_conflicts, col, N)

        # Update conflicts only if the queen moves
        if best_row != current_row:
            update_conflicts(current_row, col, row_conflicts, diag1_conflicts, diag2_conflicts, N, increment=False)
            queens[col] = best_row
            update_conflicts(best_row, col, row_conflicts, diag1_conflicts, diag2_conflicts, N, increment=True)

    # If max_steps reached without finding a solution, restart
    return solve_n_queens(N)

def print_board(queens):
    """Print the board with queens represented as '*' and empty spaces as '_'."""
    N = len(queens)
    for row in range(N):
        line = ""
        for col in range(N):
            if queens[col] == row:
                line += "* "
            else:
                line += "_ "
        print(line.strip())  # Print each row on a new line without trailing spaces



def user_input() -> int:
    return int(input("N: "))

def main():
    while True:
        n = user_input()
        if n == -1:
            break
        start_time = time.time()
        result = solve_n_queens(n)
        duration = time.time() - start_time
        
        print(result)
        print_board(result)
        print(f"Elapsed time: {duration:.4f}s")

if __name__ == "__main__":
    main()
