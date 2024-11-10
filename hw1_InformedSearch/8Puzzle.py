import heapq
import math
import string
import time
from typing import List, Optional, Tuple

"""
Used resourse from:
https://www.cs.princeton.edu/courses/archive/spring18/cos226/assignments/8puzzle/index.html
https://www.w3schools.com/php/default.asp
"""
#NumberOfBlocks + 1
#sizeOfRowCol = int(math.sqrt(numberOfBlocks + 1))

#Input functions
def readNumberOfBlocks()->int:
    while True:
        try:
            N = int(input("Please enter the number of blocks: "))
            sizeOfRowCol = int(math.sqrt(N + 1))

            # Ensure N + 1 is a perfect square
            if sizeOfRowCol * sizeOfRowCol != N + 1:
                print("Error: The number of blocks plus one must form a perfect square, like 8, 15, 24 etc.")
                continue  # Prompt again if it's not a perfect square

            return N
        except ValueError:
            print("Invalid input. Please enter an integer.")

def readEmptyBlockPosition(sizeOfRowCol)->int:
    while True:
        try:
            emptyBlockPos = int(input("Enter the position of the empty block: "))

            # Adjust empty block position if it's out of range
            maxPosition = sizeOfRowCol * sizeOfRowCol - 1
            if emptyBlockPos < 0 or emptyBlockPos > maxPosition:
                emptyBlockPos = (maxPosition-sizeOfRowCol+1)

            return emptyBlockPos
        except ValueError:
            print("Invalid input. Please enter an integer.")

def readMatrix(numberOfBlocks, emptyBlockPos)->List[List[int]]:
    sizeOfRowCol = int(math.sqrt(numberOfBlocks + 1))
    matrix = [[None for _ in range(sizeOfRowCol)] for _ in range(sizeOfRowCol)]

    # Calculate row and column for the empty block based on its linear position
    empty_row, empty_col = divmod(emptyBlockPos, sizeOfRowCol)
    
    for i in range(sizeOfRowCol):
        for j in range(sizeOfRowCol):
            if i == empty_row and j == empty_col:
                matrix[i][j] = 0
                continue
            else:
                inputMsg=f"Enter value for a[{i}][{j}]: "
                matrix[i][j] = int(input(inputMsg)) #No check for the input, yet
    return matrix

#Matrix functions
"""
We've reached our goal state if the matrix is sorted and the empty block is:
    at position 0
    at position numberOfBlocks (the matrix is N+1)
    at the position middle = numberOfBlocks/2 + 1 (in the middle, not zero based)
"""
def isGoalState(matrix:List[List[int]], numberOfBlocks:int) ->bool:
    
    sizeOfRowCol = int(math.sqrt(numberOfBlocks + 1))
    middle_posX,middle_posY = divmod((numberOfBlocks+1)// 2, sizeOfRowCol)
    
    isFirstPosBlanc = matrix[0][0]==0
    isLastPosBlanc = matrix[sizeOfRowCol-1][sizeOfRowCol-1]==0
    isMiddlePosBlanc = matrix[middle_posX][middle_posY]==0

    result = 0
    if isFirstPosBlanc or isLastPosBlanc or isMiddlePosBlanc:
        flattenMatrix  = flatMatrix(matrix)
        flattenMatrix.remove(0)
        result = flattenMatrix == sorted(flattenMatrix)
        
    return result

# Display the matrix
def displayMatrix(matrix: List[List[int]]) -> None:
    for row in matrix:
        print(row)

# Flat matrix
def flatMatrix(matrix):
    return [num for row in matrix for num in row]

# Number of inversions
def getInversions(matrix: List[int], sizeOfMatrix:int)->int:
    inversions = 0
    for i in range(sizeOfMatrix-1):
        for j in range(i+1,sizeOfMatrix):
            if matrix[i] > matrix[j]:
                inversions+=1 
    return inversions

"""
We need to check if a matrix is solvable before starting:

Odd-sized boards.

    First, we'll consider the case when the board size n is an odd integer.
    In this case, each move changes the number of inversions by an even number.
    Thus, if a board has an odd number of inversions, it is unsolvable because the goal board has an even number of inversions (zero).
"""
def isPuzzleSolvable(matrix: List[List[int]], sizeOfMatrix: int) -> bool:
    flattenMatrix = flatMatrix(matrix)
    flattenMatrix.remove(0)
    numberOfInversions = getInversions(flattenMatrix,sizeOfMatrix-1)
    return (numberOfInversions % 2 == 0 or numberOfInversions == 0)

# Manhattan Distance Calculation

"""
Idea: traverse the 2D array, get the value of pos [i][j]
check if dividing with remainder a value by the length of the row/col
 - gives the correct pos of the value,
   example: 5 should be at position (1, 1), so divmod(5 - 1, 3) yields (1, 1).
if it's at the correct pos add 0, else add to the distance var,
return the distance using Manhattan Distance Calculation
"""

def manhattan_distance(matrix: List[List[int]], sizeOfRowCol: int) -> int:
    distance = 0
    for i in range(sizeOfRowCol):
        for j in range(sizeOfRowCol):
            value = matrix[i][j]
            if value != 0:  # Exclude the blank
                target_row, target_col = divmod(value - 1, sizeOfRowCol) # divmode returs a tuple of (quotient, remainder)
                distance += abs(target_row - i) + abs(target_col - j)
    return distance

def getBlankBlockCoordinates(matrix: List[List[int]], sizeOfRowCol:int):
    for x in range(sizeOfRowCol):
        for y in range(sizeOfRowCol):
            if matrix[x][y] == 0:
                blankX, blankY = x, y
                break
    return blankX, blankY


# Iterative Deepening A* Search (IDA*)

""" 
Full explanation for myself and my program's readers of the algorithm

A Search*: A* uses a cost function f(n)=g(n)+h(n), where:
    g(n) is the actual cost from the start node to node n.
    h(n) is a heuristic estimate of the cost from node n to the goal.
    f(n) is the total estimated cost of reaching the goal from the start via node n.
A* searches nodes with the lowest f(n)

Iterative Deepening: In a regular depth-limited search (DFS), you set a fixed depth limit,
exploring all nodes up to that limit. Iterative deepening repeats this process,
increasing the depth limit gradually until the goal is found.

IDA Search*: IDA* combines these two concepts by:
    Using a threshold for f(n) instead of a fixed depth.
    Expanding nodes in a depth-first manner until the cost exceeds the current threshold.
    If no solution is found within the threshold, IDA* increases the threshold to the smallest
    cost that exceeded the threshold during the last search iteration.
    This process repeats until a solution is found.

Example: 
For a starting matrix with Manhattan distance 5:
    IDA* begins with threshold 5, performing depth-limited search to expand paths with f(n)<=5
    If no solution is found, the threshold is increased to the smallest value above 5.
    The search continues, adjusting the threshold iteratively, until it finds the goal.


"""

# Function to mirror moves - I've made a mistake during programming the answears, so I'll just reverse the answers :D
def mirrorAnswer(moves: List[str]) -> List[str]:
    mirror_moves = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left"
    }
    return [mirror_moves[move] for move in moves]

# Define possible moves
MOVES = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

class Node:
        def __init__(self, state: List[List[int]], gCost:int, move:str, parent):
            self.state = state
            self.gCost = gCost
            self.move = move  # Move that led to this state from parent
            self.parent = parent

        def f_cost(self) -> int:
            # Compute fCost as gCost + heuristic (Manhattan distance)
            sizeOfRowCol = int(math.sqrt(len(self.state)))
            return self.gCost + manhattan_distance(self.state, sizeOfRowCol)
        
        def h_cost(self) ->int:
            sizeOfRowCol = int(math.sqrt(len(self.state)))
            return manhattan_distance(self.state, sizeOfRowCol)

        def __lt__(self, other: 'Node') -> bool:
            return self.f_cost() < other.f_cost() and self.h_cost()<=other.h_cost()


def isCurrentStateAlreadyAdded(node:Node, newStateMatrix:List[List[int]])->bool:
    result = False
    currentNode = node

    while currentNode is not None:
        if currentNode.state == newStateMatrix:
            result=True
            break
        currentNode = currentNode.parent

    return result

# def A_star(start_node: Node, threshold: int, sizeOfRowCol: int) -> Optional[Node]:
def A_star(node: Node, threshold: int, sizeOfRowCol: int) -> Tuple[Optional[Node], int]:

    # Priority queue to prioritize nodes with the lowest fCost
    open_set = []

    heapq.heappush(open_set, (node.f_cost(), node))
    min_f_cost_above_threshold = float('inf')  # Track smallest f_cost above threshold

    while open_set:

        """Note to the new in Python:
        This is Python's unpacking syntax. When heappop returns (fCost, node),
         we're interested only in node, not fCost.
        By using _ as a placeholder for fCost, we tell Python to ignore this value."""

        _, node = heapq.heappop(open_set)
        fCost = node.f_cost()

        if fCost > threshold:
            min_f_cost_above_threshold = min(min_f_cost_above_threshold, fCost)
            continue
        if isGoalState(node.state, sizeOfRowCol**2-1):
            return node, threshold  # Goal found

        blankX, blankY = getBlankBlockCoordinates(node.state, sizeOfRowCol)

        for move_name in MOVES:
            dx, dy = MOVES[move_name]

            newX = blankX + dx
            newY = blankY + dy

            if 0 <= newX < sizeOfRowCol and 0 <= newY < sizeOfRowCol:

                # Create a new state by swapping the blank tile
                newStateMatrix = []
                for row in node.state:
                    newStateMatrix.append(row.copy())

                newStateMatrix[blankX][blankY], newStateMatrix[newX][newY] = newStateMatrix[newX][newY], newStateMatrix[blankX][blankY]

                if isCurrentStateAlreadyAdded(node, newStateMatrix):
                    continue

                # Create new node and add to priority queue
                new_node = Node(newStateMatrix, node.gCost + 1, move_name, node)
                heapq.heappush(open_set, (new_node.f_cost(), new_node))

    return None, min_f_cost_above_threshold  # Return min_f_cost_above_threshold if no solution found
    

def IterativeDeepening(matrix: List[List[int]], sizeOfRowCol: int) -> Optional[List[str]]:
    threshold = manhattan_distance(matrix, sizeOfRowCol)
    startNode = Node(matrix, 0, None, None)

    while True:
        result, new_threshold = A_star(startNode, threshold, sizeOfRowCol)
        if result:
            # Retrieve the path to solution
            moves = []
            while result.parent is not None:
                moves.append(result.move)
                result = result.parent
            return moves[::-1]  # Return path in correct order
        if new_threshold == float('inf'):
            return None  # No solution
        threshold = new_threshold  # Update threshold

def ida_star(matrix: List[List[int]], sizeOfRowCol: int) -> Optional[List[str]]:
    return mirrorAnswer(IterativeDeepening(matrix, sizeOfRowCol))

# Main function
def main() -> None:
    N = readNumberOfBlocks()
    sizeOfRowCol = int(math.sqrt(N + 1))
    emptyBlockPos = readEmptyBlockPosition(sizeOfRowCol)
    matrix = readMatrix(N, emptyBlockPos)

    if not isPuzzleSolvable(matrix,N+1):
        print("Puzzle is unsolveble!")
        return


    # Check if the initial matrix is the goal state
    if isGoalState(matrix, N):
        print("The matrix is already in the goal state!")
    else:
        # Start timer here
        start_time = time.time()

        solution = ida_star(matrix, sizeOfRowCol)
        
        # End timer here
        end_time = time.time()

        if solution:
            print("Solution found:")
            print(f"Number of moves: {len(solution)}")
            for move in solution:
                print(move)
        else:
            print("No solution exists for this puzzle configuration.")

        print(f"Time taken: {end_time - start_time:.2f} seconds")
#main()



def test_isGoalState1():
    N = 8
    emptyBlockPos= -1
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[1,2,3],[4,5,6],[0,7,8]]
    print(False == isGoalState(matrix,N))

def test_isGoalState2():
    N = 8
    emptyBlockPos = 8 
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[1,2,3],[4,5,6],[7,8,0]]
    print(True == isGoalState(matrix,N))

def test_isGoalState3():
    N = 8
    emptyBlockPos = 4
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[1,2,3],[4,0,5],[6,7,8]]
    print(True == isGoalState(matrix,N))

def test_isGoalState4():
    N = 8
    emptyBlockPos = 0
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[0,1,2],[3,4,5],[6,7,8]]
    print(True == isGoalState(matrix,N))

def test_Inversions1():
    N = 8
    matrix = [0,1,2,3,4,5,6,7,8]
    print( 0 == getInversions(matrix,N+1))

def test_Inversions2():
    N = 8
    matrix = [0,1,2,3,4,5,8,6,7]
    print( 2 == getInversions(matrix,N+1))

def test_Inversions3():
    N = 8
    matrix = [0,1,2,5,4,5,8,6,7]
    print( 3 == getInversions(matrix,N+1))

def test_isPuzzleSolvable1():
    N = 8
    matrix = [[0,1,2],[3,4,5],[6,7,8]]
    print(True == isPuzzleSolvable(matrix,N+1))

def test_isPuzzleSolvable2():
    N = 8
    matrix = [[1,2,3],[4,5,6],[0,7,8]]
    print(True == isPuzzleSolvable(matrix,N+1))

def test_isPuzzleSolvable3():
    N = 8
    matrix = [[1,2,3],[4,5,6],[8,7,0]]
    print(False == isPuzzleSolvable(matrix,N+1))

def test_isPuzzleSolvable4():
    N = 8
    matrix = [[1,2,3],[0,4,5],[6,7,8]]
    print(True == isPuzzleSolvable(matrix,N+1))

def test_isPuzzleSolvable5():
    N = 8
    matrix = [[0,1,3],[4,2,5],[7,8,6]]
    print(True == isPuzzleSolvable(matrix,N+1))

def test_isPuzzleSolvable6():
    N = 8
    matrix = [[1,2,3],[8,4,6],[0,5,7]]
    print(False == isPuzzleSolvable(matrix,N+1))

def test_manhattan_distance1():
    N = 8
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[1,2,3],[4,5,6],[0,7,8]]
    print(2 == manhattan_distance(matrix,sizeOfRowCol))

def test_manhattan_distance2():
    N = 8
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[8,1,3],[4,0,2],[7,6,5]]
    print(10 == manhattan_distance(matrix,sizeOfRowCol))

def test_getBlancPosCoordinates1():
    N = 8
    emptyBlockPos= 6
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[1,2,3],[4,5,6],[0,7,8]]
    expected_x,expected_y = divmod(emptyBlockPos,sizeOfRowCol)
    result_x,result_y = getBlankBlockCoordinates(matrix,sizeOfRowCol)
    print(expected_x == result_x and expected_y == result_y)

def test_getBlancPosCoordinates2():
    N = 8
    emptyBlockPos= 2
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[1,2,0],[4,5,6],[3,7,8]]
    expected_x,expected_y = divmod(emptyBlockPos,sizeOfRowCol)
    result_x,result_y = getBlankBlockCoordinates(matrix,sizeOfRowCol)
    print(expected_x == result_x and expected_y == result_y)

def test_ida_star1():
    N = 8
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[1,2,3],[4,5,6],[0,7,8]]
    temp = ida_star(matrix,sizeOfRowCol)
    expected = ['left', 'left']
    print(temp == expected)

def test_ida_star2():
    N = 8
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[1,2,3],[4,5,0],[6,7,8]]
    temp = ida_star(matrix,sizeOfRowCol)
    expected = ['right']
    print(temp == expected)

def test_ida_star3():
    N = 8
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[1,2,0],[3,4,5],[6,7,8]]
    temp = ida_star(matrix,sizeOfRowCol)
    expected = ['right', 'right']
    print(temp == expected)

def test_ida_star4():
    N = 8
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[0,1,2],[3,4,5],[6,7,8]]
    temp = ida_star(matrix,sizeOfRowCol)
    expected = []
    print(temp == expected)

def test_ida_star5():
    N = 8
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[1,2,3],[4,5,6],[7,8,0]]
    temp = ida_star(matrix,sizeOfRowCol)
    expected = []
    print(temp == expected)
    
def test_ida_star5():
    N = 8
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[1,2,3],[4,5,6],[7,8,0]]
    temp = ida_star(matrix,sizeOfRowCol)
    expected = []
    print(temp == expected)

def test_ida_star6(): # there is a difference in the pattern which the alorithm find it's solution, but it's stil a proper solution
    N = 8
    sizeOfRowCol = int(math.sqrt(N + 1))
    matrix = [[6,5,3],[2,4,8],[7,0,1]]
    temp = ida_star(matrix,sizeOfRowCol)
    expected = ['left',
                'down',
                'down',
                'right',
                'right',
                'up',
                'left',
                'up',
                'right',
                'down',
                'left',
                'down',
                'left',
                'up',
                'right',
                'down',
                'right',
                'up',
                'up',
                'left',
                'left']
    print(len(temp) == len(expected))

def tests():
    test_isGoalState1()
    test_isGoalState2()
    test_isGoalState3()
    test_isGoalState4()
    
    test_Inversions1()
    test_Inversions2()
    test_Inversions3()

    test_isPuzzleSolvable1()
    test_isPuzzleSolvable2()
    test_isPuzzleSolvable3()
    test_isPuzzleSolvable4()
    test_isPuzzleSolvable5()
    test_isPuzzleSolvable6()

    test_manhattan_distance1()
    test_manhattan_distance2()

    test_getBlancPosCoordinates1()
    test_getBlancPosCoordinates2()

    test_ida_star1()
    test_ida_star2()
    test_ida_star3()
    test_ida_star4()
    test_ida_star5()
    test_ida_star6()
tests()