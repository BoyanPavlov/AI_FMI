How Min-Conflicts Works with Hill Climbing
The Min-Conflicts algorithm is indeed a type of hill climbing because it:

Starts with an initial configuration (a random or partially structured initial placement of queens).
Iteratively makes local adjustments to reduce the number of conflicts.
Searches for a solution by minimizing "bad moves": It moves queens only to positions that minimize the current number of conflicts.
However, unlike basic hill climbing, where the algorithm might get "stuck" in a local minimum, Min-Conflicts is designed to handle constraint satisfaction problems more effectively by allowing small changes that reduce or minimize conflicts without requiring a strict decrease in conflicts every step.

Key Characteristics of the Min-Conflicts Algorithm
Local Optimization:

The algorithm repeatedly identifies problematic queens (queens involved in conflicts).
For each problematic queen, it considers moves to different rows in its column and chooses the row that minimizes conflicts in the current configuration.
Random Selection for Ties and Non-deterministic Choices:

When multiple rows have the same minimum conflict count, the algorithm picks one randomly.
This randomness helps the algorithm avoid getting stuck in simple cyclic patterns or local minima, where every adjustment might appear to increase conflicts.
Adaptability to Large Search Spaces:

Because it only adjusts queens that are currently causing problems, Min-Conflicts is particularly efficient for large N-Queens problems, where brute force or other hill-climbing approaches would struggle due to the massive number of possible configurations.
Why Min-Conflicts is Considered a Hill Climbing Variant
Hill climbing traditionally involves:

Starting at an arbitrary point and moving iteratively to a "better" neighboring state.
Continuing the process until no better neighbors are found.
Min-Conflicts fits this model with a focus on minimizing "conflicts" (the heuristic cost) instead of maximizing a direct measure of "fitness" as in typical hill climbing.

Why Min-Conflicts is Effective for N-Queens
For constraint satisfaction problems like N-Queens:

Min-Conflicts leverages the fact that we can minimize conflict locally with each adjustment of a queen’s position.
The approach is particularly effective because the solution space is densely populated with valid solutions — if we can systematically reduce conflicts, we’re likely to arrive at a valid solution.
Summary
In short, the algorithm is indeed hill climbing with a Min-Conflicts heuristic, where it climbs towards a solution by minimizing the total number of conflicts iteratively. It’s a powerful strategy for problems where constraints can be satisfied incrementally and works well for the N-Queens problem due to its straightforward structure and the density of solutions in the search space.