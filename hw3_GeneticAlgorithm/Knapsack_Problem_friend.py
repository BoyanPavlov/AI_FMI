import random
import time
from collections import Counter

POPULATION = 10000
MAX_GENERATIONS = 50
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.7
INITIAL_MUTATION_RATE = 0.02
FINAL_MUTATION_RATE = 0.001

# Optimization from the Internet:
# The algorithm proceeds until the best solution during the evolution process doesn't change to 
# a better value for a predefined value of generations. This predefined value can be 20% or 30% of 
# the generation number which the best solution has found so far. I.e. the algorithm reaches to a value 
# of 200 at generation 50, then this value doesn't change for 15 generations (30% of 50), so the algorithm stops.


# Name for the predefined number of generations where the best solution remains unchanged
STOPPING_GENERATIONS_THRESHOLD = int(MAX_GENERATIONS * 0.3)

final_arr = []

current_mutation_rate = 0.0

class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

class Individual:
    def __init__(self, num_items, items_list):
        self.chromosome = [random.choice([True, False]) for _ in range(num_items)]
        self.items = items_list
        self.weight = sum(items_list[i].weight * self.chromosome[i] for i in range(num_items))
        self.value = sum(items_list[i].value * self.chromosome[i] for i in range(num_items))

    def crossover(self, other):
        len_chr = len(self.chromosome)

        crossover_point = random.randint(0, len_chr - 1)
        child = Individual(len_chr, self.items)
        child.weight, child.value = 0, 0

        child.chromosome[:crossover_point] = self.chromosome[:crossover_point]
        child.chromosome[crossover_point:] = other.chromosome[crossover_point:]

        for i in range(len_chr):
            child.weight += self.items[i].weight * child.chromosome[i]
            child.value += self.items[i].value * child.chromosome[i]

        return child

    def mutate(self):
        len_chr = len(self.chromosome)

        for i in range(len_chr):
            if random.random() < current_mutation_rate:
                self.flip(i)

    def flip(self, i):
        if self.chromosome[i]:
            self.weight -= self.items[i].weight
            self.value -= self.items[i].value
        else:
            self.weight += self.items[i].weight
            self.value += self.items[i].value
        self.chromosome[i] = not self.chromosome[i]

def getFitness(individual, m):
    while individual.weight > m:
        geneIdx = random.randint(0, len(individual.chromosome) - 1)
        if individual.chromosome[geneIdx]:
            individual.flip(geneIdx)
    return individual.value

def selection_tournament(population_list, m_limit):
    len_population = len(population_list)
    bestParent = random.randint(0, len_population - 1)

    for _ in range(1, TOURNAMENT_SIZE):
        candidate = random.randint(0, len_population - 1)
        if getFitness(population_list[candidate], m_limit) > getFitness(population_list[bestParent], m_limit):
            bestParent = candidate
    return population_list[bestParent]

def more_than_n_elements_same(arr, n):
    counts = Counter(arr)
    for count in counts.values():
        if count > n:
            return True
    return False

def solve(in_stream):
    m, n = map(int, in_stream.readline().split())
    items_list = []

    for _ in range(n):
        weight, value = map(int, in_stream.readline().split())
        items_list.append(Item(weight, value))

    population_list = [Individual(n, items_list) for _ in range(POPULATION)]
    new_population_list = [Individual(n, items_list) for _ in range(POPULATION)]

    for generation in range(MAX_GENERATIONS):
        current_mutation_rate = INITIAL_MUTATION_RATE + (FINAL_MUTATION_RATE - INITIAL_MUTATION_RATE) * (generation / MAX_GENERATIONS)
        
        # Elitism
        elites = sorted(population_list, key=lambda x: getFitness(x, m), reverse=True)[:2]
        new_population_list = elites[:]

        # Crossover
        for i in range(2, POPULATION, 2):
            parent_1 = selection_tournament(population_list, m)
            parent_2 = selection_tournament(population_list, m)
            
            if random.random() < CROSSOVER_RATE:
                child_1 = parent_1.crossover(parent_2)
                child_2 = parent_2.crossover(parent_1)

                child_1.mutate()
                child_2.mutate()

                new_population_list.extend([child_1, child_2])
            else:
                new_population_list.extend([parent_1, parent_2])

        population_list = new_population_list

        # Print best for each generation
        best = max(getFitness(individual, m) for individual in population_list)
        final_arr.append(best)

        if generation % 6 == 0:
            print(f"Generation {generation + 1}: Best value = {best}")
            
        if more_than_n_elements_same(final_arr, STOPPING_GENERATIONS_THRESHOLD):
            return

if __name__ == "__main__":
    start_time = time.time()
    # solve(open("test.txt"))
    # solve(open("short_test.txt"))
    solve(open("hw3_GeneticAlgorithm/KP_input_test/KP_short_test_data.txt"))
    #solve(open("hw3_GeneticAlgorithm/KP_input_test/KP_long_test_data.txt"))
    end_time = time.time()
    print("Full: ", final_arr)
    print(f"Time taken: {end_time - start_time:.2f} seconds")
