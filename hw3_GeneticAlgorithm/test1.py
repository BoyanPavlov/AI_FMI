import random
import time
from collections import Counter

# Constants
POPULATION_SIZE = 10000  # Number of individuals
MAX_GENERATIONS = 50  # Maximum generations
TOURNAMENT_SIZE = 5  # Number of individuals competing in selection
CROSSOVER_RATE = 0.7  # Chance of crossover
INITIAL_MUTATION_RATE = 0.02  # Start mutation rate
FINAL_MUTATION_RATE = 0.001  # End mutation rate
STOP_THRESHOLD = int(MAX_GENERATIONS * 0.3)  # Generations without improvement

# Class for items (weight and value)
class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

# Class for individuals in the population
class Individual:
    def __init__(self, items, chromosome=None):
        self.items = items
        self.chromosome = chromosome or [random.choice([True, False]) for _ in items]
        self.weight = self.calculate_weight()
        self.value = self.calculate_value()

    def calculate_weight(self):
        return sum(item.weight for item, selected in zip(self.items, self.chromosome) if selected)

    def calculate_value(self):
        return sum(item.value for item, selected in zip(self.items, self.chromosome) if selected)

    def mutate(self, mutation_rate):
        for i in range(len(self.chromosome)):
            if random.random() < mutation_rate:
                self.chromosome[i] = not self.chromosome[i]  # Flip gene
        self.weight = self.calculate_weight()
        self.value = self.calculate_value()

    def crossover(self, other):
        point = random.randint(1, len(self.chromosome) - 1)  # Choose crossover point
        child_chromosome = self.chromosome[:point] + other.chromosome[point:]
        return Individual(self.items, child_chromosome)

def fitness(individual, max_weight):
    if individual.weight > max_weight:
        return 0  # Invalid solutions have zero fitness
    return individual.value

def tournament_selection(population, max_weight):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    return max(tournament, key=lambda ind: fitness(ind, max_weight))

def has_converged(values, threshold):
    return Counter(values).most_common(1)[0][1] > threshold

def solve_knapsack(file):
    max_weight, num_items = map(int, file.readline().split())
    items = [Item(*map(int, file.readline().split())) for _ in range(num_items)]

    # Create initial population
    population = [Individual(items) for _ in range(POPULATION_SIZE)]
    best_values = []

    for generation in range(MAX_GENERATIONS):
        mutation_rate = INITIAL_MUTATION_RATE + (FINAL_MUTATION_RATE - INITIAL_MUTATION_RATE) * (generation / MAX_GENERATIONS)

        # Sort population by fitness and keep the best two (elitism)
        population = sorted(population, key=lambda ind: fitness(ind, max_weight), reverse=True)
        new_population = population[:2]  # Elites

        # Generate the rest of the population
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, max_weight)
            parent2 = tournament_selection(population, max_weight)
            if random.random() < CROSSOVER_RATE:
                child = parent1.crossover(parent2)
                child.mutate(mutation_rate)
                new_population.append(child)
            else:
                new_population.append(parent1)
                new_population.append(parent2)

        population = new_population[:POPULATION_SIZE]  # Ensure population size stays constant
        best_value = max(fitness(ind, max_weight) for ind in population)
        best_values.append(best_value)

        # Print progress
        if generation % 5 == 0 or generation == MAX_GENERATIONS - 1:
            print(f"Generation {generation + 1}: Best value = {best_value}")

        # Check for convergence
        if has_converged(best_values, STOP_THRESHOLD):
            break

    print(f"Best value found: {max(best_values)}")

# Example usage
if __name__ == "__main__":
    start_time = time.time()
    #with open("hw3_GeneticAlgorithm/KP/KP_short_test_data.txt") as f:
    with open("hw3_GeneticAlgorithm/KP/KP_long_test_data.txt") as f:
        solve_knapsack(f)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
