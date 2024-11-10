#include "FrogPuzzle.h"

FrogPuzzle::FrogPuzzle()
{
    _numberOfFrogs = readInput();
    fillStartingState(_startingState, (_numberOfFrogs * 2) + 1);
}

FrogPuzzle::~FrogPuzzle()
{
}

unsigned FrogPuzzle::getNumberOfFrogs() const
{
    return _numberOfFrogs;
}

void FrogPuzzle::letFrogsJump()
{
    DFS(_startingState);
}

void FrogPuzzle::showFrogsJumping()
{
    for (int i = _path.size()-1; i>=0; --i)
    {
        std::cout << _path[i] << std::endl;
    }
}

string FrogPuzzle::getGoalState()
{
    string result = _startingState;
    size_t size = result.size();

    for (size_t i = 0; i < size; i++)
    {
        result[size - i - 1] = _startingState.at(i);
    }

    return result;
}

const string& FrogPuzzle::getStartingState() const
{
    return _startingState;
}


unsigned FrogPuzzle::readInput()
{
    unsigned input = 0;
    while (input < 1)
    {
        std::cout << "Please enter a number greater than 1\n";
        std::cin >> input;
        if (input < 1)
        {
            std::cout << "Invalid input...\n";
        }
    }

    return input;
}

void FrogPuzzle::fillStartingState(string& startState, unsigned size)
{
    int index = 0;

    for (index = 0; index < (size / 2); index++)
    {
        startState += 'L';
    }

    startState += '_';

    for (; index < size - 1; index++)
    {
        startState += 'R';
    }
}


bool FrogPuzzle::isInvalidPath(string& currentState)
{
    for (size_t i = 0; i < currentState.size() - 3; i++)
    {
        if (currentState[i] == 'L' && currentState[i + 1] == 'L'
            && currentState[i + 2] == 'R' && currentState[i + 3] == 'R')
        {
            return true;
        }
    }
    return false;
}

string FrogPuzzle::jumpFromTo(string currentState, int from, int to) const
{
    char temp = currentState[from];
    currentState[from] = currentState[to];
    currentState[to] = temp;

    return currentState;
}

vector<string> FrogPuzzle::getPossibleMoves(string& currentState) const
{
    vector<string> possibleMoves;

    for (unsigned int i = 0; i < currentState.size(); ++i)
    {
        if (i < currentState.size() - 1 && currentState.at(i) == 'L' && currentState.at(i + 1) == '_')
            possibleMoves.push_back(jumpFromTo(currentState, i, i + 1));

        if (i < currentState.size() - 2 && currentState.at(i) == 'L' && currentState.at(i + 2) == '_')
            possibleMoves.push_back(jumpFromTo(currentState, i, i + 2));

        if (i > 0 && currentState.at(i) == 'R' && currentState.at(i - 1) == '_')
            possibleMoves.push_back(jumpFromTo(currentState, i, i - 1));

        if (i > 1 && currentState.at(i) == 'R' && currentState.at(i - 2) == '_')
            possibleMoves.push_back(jumpFromTo(currentState, i, i - 2));
    }

    return possibleMoves;
}


bool FrogPuzzle::DFS(string&current)
{
    if (current == getGoalState())
    {
        _path.push_back(current);
        return true;
    }

    if (isInvalidPath(current)) return false;

    vector<string> moves = getPossibleMoves(current);

    for (size_t i = 0; i < moves.size(); ++i)
    {
        if (DFS(moves[i]))
        {
            _path.push_back(current);
            return true;
        }
    }
    return false;
}
