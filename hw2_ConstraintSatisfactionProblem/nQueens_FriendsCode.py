import random
import time
from typing import List, Tuple

MAX_ITERATIONS = 1000000

def user_input() -> int:
    return int(input("N: "))

def init_queens(n: int) -> List[int]:
    queens = [0] * n
    row = 1
    for i in range(n):
        queens[i] = row
        row += 2
        if row >= n:
            row = 0
    return queens

def calc_conflicts(queens: List[int], n: int) -> List[int]:
    dN = n * 2
    rows = [0] * n
    left_diagonals = [0] * dN
    right_diagonals = [0] * dN
    conflicts = [0] * n

    for col, row in enumerate(queens):
        rows[row] += 1
        right_diagonals[row - col + (n - 1)] += 1
        left_diagonals[row + col] += 1

    for col, row in enumerate(queens):
        conflicts[col] = (
            rows[row]
            + right_diagonals[row - col + (n - 1)]
            + left_diagonals[row + col]
            - 3
        )

    return conflicts

def gen_rand_position(conflicts: List[int]) -> int:
    rand_pos = [col for col, conflict in enumerate(conflicts) if conflict > 0]
    return random.choice(rand_pos)

def find_row_conflicts(queens: List[int], column: int, n: int) -> List[int]:
    row_conflicts = [0] * n
    for col in range(n):
        if col == column:
            continue
        row_conflicts[queens[col]] += 1
        left_diag = queens[col] - abs(col - column)
        right_diag = queens[col] + abs(col - column)
        if 0 <= left_diag < n:
            row_conflicts[left_diag] += 1
        if 0 <= right_diag < n:
            row_conflicts[right_diag] += 1
    return row_conflicts

def find_position_with_equal_value(row_conflicts: List[int], val: int) -> int:
    choose_from = [row for row, conflict in enumerate(row_conflicts) if conflict == val]
    return random.choice(choose_from)

def update_track_conflicts(
    queens: List[int], num_conflicts: List[int], column: int, row: int, n: int
) -> int:
    current_row = queens[column]
    num_conflicts[column] = 0

    for col in range(n):
        if col == column:
            continue
        row_diff_old = abs(current_row - queens[col])
        row_diff_new = abs(row - queens[col])
        col_diff = abs(column - col)
        if current_row == queens[col] or row_diff_old == col_diff:
            num_conflicts[col] -= 1
        if row == queens[col] or row_diff_new == col_diff:
            num_conflicts[col] += 1

    queens[column] = row
    return sum(num_conflicts)

def repairing(n: int) -> Tuple[List[int], int]:
    queens = init_queens(n)
    conflicts = calc_conflicts(queens, n)
    all_conflicts = sum(conflicts)

    for i in range(MAX_ITERATIONS):
        if all_conflicts == 0:
            return queens, i
        col = gen_rand_position(conflicts)
        row_conflicts = find_row_conflicts(queens, col, n)
        min_conflicts = min(row_conflicts)
        row = find_position_with_equal_value(row_conflicts, min_conflicts)
        all_conflicts = update_track_conflicts(queens, conflicts, col, row, n)


    if (not is_valid_res(queens)):
        repairing(queens)

    return queens, MAX_ITERATIONS

def is_valid_res(queens: List[int]) -> bool:
    n = len(queens)
    for row1 in range(n):
        for row2 in range(row1 + 1, n):
            if queens[row1] == queens[row2] or abs(queens[row1] - queens[row2]) == abs(row1 - row2):
                return False
    return True

def print_result(queens: List[int]):
    print("[", ", ".join(map(str, queens)), "]")

def print_board(board, number_of_queens):
    for row in range(number_of_queens):
        line = ""
        for col in range(number_of_queens):
            if board[col] == row:
                line+='* '
            else:
                line+='_ '
        print(line.strip())  # Print each row on a new line without trailing spaces

def main():
    while True:
        n = user_input()
        if n == -1:
            break
        start_time = time.time()
        result, iterations = repairing(n)
        print(is_valid_res(result))
        duration = time.time() - start_time
        print_result(result)
        print_board(result,n)
        print(f"Elapsed time: {duration:.4f}s")

if __name__ == "__main__":
    main()
