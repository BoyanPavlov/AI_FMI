
"""
Used resources in the research and logic implementation:
    https://www.geeksforgeeks.org/genetic-algorithms/
    https://github.com/kiecodes/genetic-algorithms
    https://www.youtube.com/watch?v=uQj5UNhCPuo&ab_channel=KieCodes
"""

import random
import time
from collections import Counter


POPULATION = 10000
SELECTION_SIZE = 5
MAX_GENERATIONS = 50
CROSSOVER_RATE = 0.7
# Name for the predefined number of generations where the best solution remains unchanged
STOPPING_GENERATIONS_THRESHOLD = int(MAX_GENERATIONS * 0.3)
#values from Internet
INITIAL_MUTATION_RATE = 0.02
FINAL_MUTATION_RATE = 0.001

final_arr = []
num_of_items_printed=0

"""
Goal of the program: 
    fill the backpack with the max items, which have the highest value
    use genetic algorithm
Input: backpack capacity in grams, max_items
    item1: weight, value
    ...
    itemn: weight, value
Output:
    Generation i: Best value = X
    ...
    Generation k: Best value = X_k
    Generation last: Best value = X_last
    Full array with best solutions: array
"""
class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

class Backpack:

    """
    - Genome: Array with 0 and 1
    - Items: list of items
    - Weight:Int - we have list of items and if this items is added (check it in the genome): add it's weight to the backpack
    - Value:Int - The same logic here as the one used in weight
    """

    def __init__(self, num_items, items_list):
        self.genome = [random.choice([True, False]) for _ in range(num_items)]
        self.items = items_list
        self.weight = sum(items_list[i].weight * self.genome[i] for i in range(num_items))
        self.value = sum(items_list[i].value * self.genome[i] for i in range(num_items))


def single_point_crossover(backpack1, backpack2):
        
        """
        "Breeding function" used to create new backpacks(individuals) from the given
        """

        num_items = len(backpack1.genome)

        crossover_point = random.randint(0, num_items - 1)
        child = Backpack(num_items, backpack1.items)
        child.weight, child.value = 0, 0

        child.genome[:crossover_point] = backpack1.genome[:crossover_point]
        child.genome[crossover_point:] = backpack2.genome[crossover_point:]

        for i in range(num_items):
            child.weight += backpack1.items[i].weight * child.genome[i]
            child.value += backpack1.items[i].value * child.genome[i]

        return child

def mutate(backpack, current_mutation_rate):
        """
        Mutate function: change given individual(backpack), so we have a different one
        """
        len_chr = len(backpack.genome)

        for i in range(len_chr):
            if random.random() < current_mutation_rate:
                flip_bit(backpack,i)

def flip_bit(backpack, pos):
        """
         - Flip bit in the genome arr
         - add or remove item to the backpack
         - add or remove item's weight and value
        """

        if backpack.genome[pos]:
            backpack.weight -= backpack.items[pos].weight
            backpack.value -= backpack.items[pos].value
        else:
            backpack.weight += backpack.items[pos].weight
            backpack.value += backpack.items[pos].value
        backpack.genome[pos] = not backpack.genome[pos]

#Maybe this function should be called fitness or someting (using the convention on internet, but i'm not sure)
def get_fittest_value(backpack, max_weight):
    """
    Get the configuration(backpack) with best value
    """
    while backpack.weight > max_weight:
        randIdx = random.randint(0, len(backpack.items) - 1)
        if backpack.genome[randIdx]:
            flip_bit(backpack,randIdx)
    return backpack.value


def selection(population_list, m_limit):
    len_population = len(population_list)
    best_parent = random.randint(0, len_population - 1)

    for _ in range(1, SELECTION_SIZE):
        candidate = random.randint(0, len_population - 1)
        if get_fittest_value(population_list[candidate], m_limit) > get_fittest_value(population_list[best_parent], m_limit):
            best_parent = candidate
    return population_list[best_parent]

def more_than_n_elements_same(arr, n):
    return any(count > n for count in Counter(arr).values())

def solve_kp(in_stream):
    max_weight, num_of_items = map(int, in_stream.readline().split())
    items_list = []

    for _ in range(num_of_items):
        weight, value = map(int, in_stream.readline().split())
        items_list.append(Item(weight, value))

    population_list = [Backpack(num_of_items, items_list) for _ in range(POPULATION)]
    new_population_list = [Backpack(num_of_items, items_list) for _ in range(POPULATION)]

    for generation in range(MAX_GENERATIONS):
        current_mutation_rate = INITIAL_MUTATION_RATE + (FINAL_MUTATION_RATE - INITIAL_MUTATION_RATE) * (generation / MAX_GENERATIONS)
        
        # Elitism
        elites = sorted(population_list, key=lambda backpack: get_fittest_value(backpack, max_weight), reverse=True)[:2]
        new_population_list = elites[:]

        # Crossover
        for i in range(2, POPULATION, 2):
            parent_1 = selection(population_list, max_weight)
            parent_2 = selection(population_list, max_weight)
            
            if random.random() < CROSSOVER_RATE:
                child_1 = single_point_crossover(parent_1,parent_2)
                child_2 = single_point_crossover(parent_2,parent_1)

                mutate(child_1,current_mutation_rate)
                mutate(child_2,current_mutation_rate)

                new_population_list.extend([child_1, child_2])
            else:
                new_population_list.extend([parent_1, parent_2])

        population_list = new_population_list

        # Print best for each generation
        best = max(get_fittest_value(individual, max_weight) for individual in population_list)
        final_arr.append(best)

        
        # global num_of_items_printed
        # num_of_items_printed += 1

        if(generation % 4 == 0):
            print(f"Generation {generation + 1}: Best value = {best}")
            
        if more_than_n_elements_same(final_arr, STOPPING_GENERATIONS_THRESHOLD):
            return
        
            
#Not usefull because I want to show the process of getting generations, not just to print the generations
#But it's faster this way
# def print_generations(arr):
#     if not arr:
#         print("The array is empty.")
#         return
#     print(f"Generation 1: Best value = {arr[0]}")
#     print(f"Generation {len(arr)}: Best value = {arr[-1]}")
#     random_indices = random.sample(range(1, len(arr) - 1), min(8, len(arr) - 2))
#     for idx in sorted(random_indices):
#         print(f"Generation {idx + 1}: Best value = {arr[idx]}")

if __name__ == "__main__":

    path = ("hw3_GeneticAlgorithm/KP_input_test/KP_short_test_data.txt")
    path1 = ("hw3_GeneticAlgorithm/KP_input_test/KP_long_test_data.txt")
    start_time = time.time()

    solve_kp(open(path))

    end_time = time.time()
    print(f"Generation Last: Best value = {final_arr[-1]}") 
    print("Full: ", final_arr)
    print(f"Time taken: {end_time - start_time:.2f} seconds")