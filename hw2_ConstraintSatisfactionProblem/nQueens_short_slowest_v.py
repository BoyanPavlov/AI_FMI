import random
import time
from typing import List

def initialize_board(n: int) -> List[int]:
    """Initialize queens randomly on the board, one per column."""
    return [random.randint(0, n - 1) for _ in range(n)]

def get_conflict_counts(queens: List[int]) -> List[int]:
    """Calculate and return the total conflicts for each queen on the board."""
    n = len(queens)
    row_conflicts = [0] * n
    diag1_conflicts = [0] * (2 * n - 1)  # primary diagonals
    diag2_conflicts = [0] * (2 * n - 1)  # secondary diagonals

    for col, row in enumerate(queens):
        row_conflicts[row] += 1
        diag1_conflicts[row + col] += 1
        diag2_conflicts[row - col + n - 1] += 1

    # Compute conflicts for each queen's current position
    conflicts = [0] * n
    for col, row in enumerate(queens):
        conflicts[col] = (row_conflicts[row] +
                          diag1_conflicts[row + col] +
                          diag2_conflicts[row - col + n - 1] - 3)  # exclude itself
    return conflicts

def minimize_conflicts(queens: List[int], max_steps: int = 100000) -> List[int]:
    """Attempt to solve the N-Queens problem by minimizing conflicts iteratively."""
    n = len(queens)
    for step in range(max_steps):
        conflicts = get_conflict_counts(queens)
        
        # If there are no conflicts, solution is found
        if max(conflicts) == 0:
            return queens

        # Pick a column with the highest conflict
        max_conflict_cols = [col for col, conflict in enumerate(conflicts) if conflict == max(conflicts)]
        col = random.choice(max_conflict_cols)
        
        # Move queen in the selected column to minimize conflicts
        min_conflicts, best_row = n, queens[col]
        for row in range(n):
            # Temporarily place the queen in this row and calculate conflicts
            queens[col] = row
            new_conflicts = get_conflict_counts(queens)[col]
            if new_conflicts < min_conflicts:
                min_conflicts, best_row = new_conflicts, row
        queens[col] = best_row  # Place queen in the optimal row found

    # If max steps reached without a solution, retry with new random initialization
    return minimize_conflicts(initialize_board(n), max_steps)

def solve_n_queens(n: int) -> List[int]:
    """Solve the N-Queens problem and return the solution."""
    queens = initialize_board(n)
    return minimize_conflicts(queens)

def print_board(queens: List[int]):
    """Display the board with queens represented as '*' and empty spaces as '_'."""
    n = len(queens)
    for row in range(n):
        line = "".join("* " if queens[col] == row else "_ " for col in range(n))
        print(line.strip())

def main():
    while True:
        try:
            n = int(input("Enter N for N-Queens (-1 to quit): "))
            if n == -1:
                break
            start_time = time.time()
            result = solve_n_queens(n)
            duration = time.time() - start_time
            print("\nSolution:")
            print(result)
            print_board(result)
            print(f"Elapsed time: {duration:.4f}s\n")
        except ValueError:
            print("Please enter a valid integer.")

if __name__ == "__main__":
    main()
