#pragma once

#ifndef _FROG_PUZZLE_
#define _FROG_PUZZLE_

#include <iostream>
#include <vector>
#include<cstring>

using std::vector;
using std::string;

class FrogPuzzle
{
private:
    unsigned _numberOfFrogs = 0;
    string _startingState;

    vector<string> _path;

public:
    FrogPuzzle();
    ~FrogPuzzle();

    string getGoalState();
    const string&getStartingState() const;
    unsigned getNumberOfFrogs() const;
    void letFrogsJump();
    void showFrogsJumping();

private:
    unsigned readInput();
    void fillStartingState(string& startState, unsigned size);
    bool isInvalidPath(string& currentState);
    string jumpFromTo(string currentState, int from, int to) const;
    vector<string> getPossibleMoves(string& currentState) const;
    bool DFS(string& currentState);
};



#endif // !_FROG_PUZZLE_
