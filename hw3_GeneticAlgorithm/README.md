Core Concepts of Genetic Algorithms
Population

A population is a group of candidate solutions (often called individuals or chromosomes).
Each individual represents a potential solution to the problem, encoded in a format such as a binary string, a set of numbers, or another suitable representation.
Fitness Function

A fitness function evaluates how good each solution is for the problem at hand.
The fitness value is used to guide the selection process, favoring better solutions.
Selection

Selection determines which individuals are chosen to pass their genes to the next generation.
Common selection methods include:
Roulette Wheel Selection: Probabilistic selection based on fitness.
Tournament Selection: Random groups are formed, and the best individual in each group is selected.
Elitism: The best individuals are guaranteed to survive to the next generation.
Crossover (Recombination)

Crossover combines the genetic material of two parent individuals to create offspring.
Common methods:
Single-Point Crossover: A crossover point is chosen, and parts of the parents are swapped.
Two-Point Crossover: Two points are chosen for swapping segments.
Uniform Crossover: Each gene in the offspring is randomly chosen from one of the parents.
Mutation

Mutation introduces randomness by altering genes in an individual, promoting diversity.
For example, in a binary string representation, mutation might flip a bit from 0 to 1 or vice versa.
Replacement

Replacement determines how the next generation is formed.
It can include the best individuals from the previous generation or only the newly created offspring.
Termination Criteria

The algorithm stops when:
A satisfactory solution is found.
A fixed number of generations are completed.
Improvement over generations becomes negligible.