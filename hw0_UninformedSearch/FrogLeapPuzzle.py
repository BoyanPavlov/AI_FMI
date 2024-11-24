import time

visited = []
leaves = []


def starting_position(n_frogs):
    return '>' * n_frogs + '_' + '<' * n_frogs


def resulting_position(n_frogs):
    return '<' * n_frogs + '_' + '>' * n_frogs


def generate_valid_moves(current_state):
    valid_moves = []
    frog_index = 0

    for frog in current_state:
        new_state = current_state 
        new_state = new_state.replace('_', frog)
        new_state = new_state[:frog_index] + '_' + new_state[frog_index+1:]

        valid_move = True

        # Frog moves to the right, but the gap is to the left
        if (frog == '>') and (frog_index > new_state.find('_')):
            valid_move = False

        # Frog moves to the left, but the gap is to the right
        if (frog == '<') and (frog_index < new_state.find('_')):
            valid_move = False

        # Left-moving frog jumps over two right-moving frogs
        if (frog == '>') and (current_state.index('_') < frog_index):
            valid_move = False
        
        # Right-moving frog jumps over two left-moving frogs
        if (frog == '<') and (current_state.index('_') > frog_index):
            valid_move = False


        # Frog jumps more than one step
        gap_index = current_state.find("_")
        new_gap_index = new_state.find("_")
        if abs(new_gap_index - gap_index) >= 3:
            valid_move = False

        # Iinvalid moves
        if current_state.rfind(">") + 2 < new_state.find("_"):
            valid_move = False

        if current_state.find("<") - 2 > new_state.index("_"):
            valid_move = False

        if valid_move:
            valid_moves.append(new_state)
        frog_index += 1

    valid_moves.remove(current_state)

    return valid_moves


def mark_as_leaf(position):
    leaves.append(position)


def dfs(starting_graph, start_position, result_position):
    global visited

    if start_position == result_position:
        visited.append(start_position)
        return visited

    if start_position in visited:
        return visited

    valid_moves = generate_valid_moves(start_position)
    starting_graph[start_position] = valid_moves

    visited.append(start_position)

    for valid_move in valid_moves:
        visited = dfs(starting_graph, valid_move, result_position)

        if result_position in visited:
            visited.append(start_position)
            return visited

    mark_as_leaf(start_position)
    return visited


def main():
    n_frogs = int(input("Enter the number of frogs: "))

    start_position = starting_position(n_frogs)
    starting_graph = {
        start_position: set([])
    }

    result_position = resulting_position(n_frogs)

    start_time = time.time()

    result = dfs(starting_graph, start_position, result_position)

    end_time = time.time()

    dfs_result = []
    for elem in result:
        if elem not in dfs_result:
            dfs_result.append(elem)

    # print(dfs_result)
    string_result = ' \n'.join(dfs_result)
    print(string_result)
    print(f"Elapsed time: {end_time - start_time:.4f} seconds.")
    
    # print(len(dfs_result))


if __name__ == "__main__":
    main()