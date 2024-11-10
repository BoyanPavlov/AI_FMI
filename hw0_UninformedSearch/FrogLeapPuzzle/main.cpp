#include <iostream>
#include "FrogPuzzle.h"

using std::cout;
using std::endl;

/*
   We are looking at the game Frog Leap puzzle
   1. We can solve the problem using uninformed search algorithm, because: By definition
   - Uninformed strategies use only the information available in the problem definition.
   2. We are representing the puzzle as a: Tree, where
   The root is the initial state of the puzzle. Nodes in the graph represent the different outcomes of the game.
    So our strategy will be - choose a node and follow it until the end. Check if it has a correct outcome.
   3. The best uniformed algo for this task is DFS, because matches our strategy completly.
   . BFS would be a uneffective here. It should work, but it's gonna cost a fortune.
   . In my humble opinion - Uniform-cost search won't work for us here. Feel free to prove me wrong.
   . Depth-limited search - not recommended - I have to guess the depth of the right choice. I am not a magician, so no.
   . Iterative deepening search - I think this should work here, but I'm not sure how.
   4. What about a linear time solution? If there is a liniar time solution, I'm not sure how to find it right now.
*/

/*
* I've made an attempt to use my implementations of string and vector, but after 10 painful hours of debugging I've left the program with the standart one.
* I've also have seen a repo and I've partialy copied the logic in those functions
*   bool isInvalidPath(string& currentState);
*   string jumpFromTo(string currentState, int from, int to) const;
*   vector<string> getPossibleMoves(string& currentState) const;
*
*   https://github1s.com/daangelov/Leaf-Frog-Puzzle/blob/master/main.cpp
*   this is the repo.
*/


int main(int argc, char const* argv[])
{
    FrogPuzzle puzzle1;
    puzzle1.letFrogsJump();
    puzzle1.showFrogsJumping();

    return 0;
}
