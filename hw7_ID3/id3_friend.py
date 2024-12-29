import math
import random
from collections import Counter, defaultdict

class DecisionTree:
    def __init__(self, pruning_method):
        self.K = 6
        self.class_index = 9
        self.taken_indexes = {self.class_index}
        self.pruning_method = pruning_method
        self.max_depth = 10
        self.min_examples_in_leaf = 5
        self.min_gain = 0.01

    def calculate_table(self, index, dataset):
        return Counter(row[index] for row in dataset)

    def calculate_table_2d(self, index, class_index, dataset):
        table = defaultdict(lambda: defaultdict(int))
        for row in dataset:
            table[row[index]][row[class_index]] += 1
        return table

    def entropy(self, occurrences, total):
        return -sum((count / total) * math.log2(count / total) for count in occurrences.values() if count > 0)

    def one_attribute_entropy(self, index, dataset):
        occurrences = self.calculate_table(index, dataset)
        return self.entropy(occurrences, len(dataset))

    def two_attribute_entropy(self, index, dataset):
        total_entropy = 0
        attribute_occurrences = self.calculate_table(index, dataset)
        occurrences = self.calculate_table_2d(index, self.class_index, dataset)

        for key, class_occurrences in occurrences.items():
            probability = attribute_occurrences[key] / len(dataset)
            entropy = self.entropy(class_occurrences, attribute_occurrences[key])
            total_entropy += probability * entropy

        return total_entropy

    def information_gain(self, dataset):
        class_entropy = self.one_attribute_entropy(self.class_index, dataset)
        if class_entropy == 0:
            return -1

        max_gain, best_index = 0, -1
        for i in range(len(dataset[0])):
            if i in self.taken_indexes:
                continue

            attribute_entropy = self.two_attribute_entropy(i, dataset)
            gain = class_entropy - attribute_entropy
            if gain > max_gain:
                max_gain, best_index = gain, i

        return best_index

    def build_tree(self, dataset, depth=0):
        if len(dataset) <= self.K or depth >= self.max_depth:
            most_common_class = Counter(row[self.class_index] for row in dataset).most_common(1)[0][0]
            return self.Node(-1, None, True, most_common_class)

        index = self.information_gain(dataset)
        if index == -1:
            return self.Node(-1, None, True, dataset[0][self.class_index])

        self.taken_indexes.add(index)
        children = {}
        groups = defaultdict(list)
        for row in dataset:
            groups[row[index]].append(row)

        for key, subset in groups.items():
            children[key] = self.build_tree(subset, depth + 1)

        self.taken_indexes.remove(index)
        node = self.Node(index, children)

        if self.pruning_method in {1, 2}:
            node = self.post_pruning(node, dataset)

        return node

    def post_pruning(self, node, dataset):
        if node.is_leaf:
            return node

        accuracy_before = self.test_model(node, dataset)
        most_common_class = Counter(row[self.class_index] for row in dataset).most_common(1)[0][0]
        pruned_node = self.Node(-1, None, True, most_common_class)
        accuracy_after = self.test_model(pruned_node, dataset)

        if accuracy_after >= accuracy_before:
            return pruned_node

        for key, child in node.children.items():
            node.children[key] = self.post_pruning(child, dataset)

        return node

    def test_model(self, model, dataset):
        correct = 0
        for row in dataset:
            node = model
            while not node.is_leaf:
                node = node.children.get(row[node.index], next(iter(node.children.values())))
            if node.value == row[self.class_index]:
                correct += 1
        return correct / len(dataset)

    def train_and_test(self, dataset):
        random.shuffle(dataset)
        fold_size = len(dataset) // 10
        accuracies = []

        for i in range(10):
            test_set = dataset[i * fold_size:(i + 1) * fold_size]
            train_set = dataset[:i * fold_size] + dataset[(i + 1) * fold_size:]

            model = self.build_tree(train_set)
            accuracy = self.test_model(model, test_set)
            accuracies.append(accuracy)

        average_accuracy = sum(accuracies) / len(accuracies)
        print(f"Average Accuracy: {average_accuracy:.2%}")

    class Node:
        def __init__(self, index, children, is_leaf=False, value=None):
            self.index = index
            self.children = children
            self.is_leaf = is_leaf
            self.value = value

if __name__ == "__main__":
    pruning_method = int(input("Enter pruning method (0 = Pre-pruning, 1 = Post-pruning, 2 = Both): "))

    with open("hw7_ID3/breast+cancer/breast-cancer.data") as file:
        data = [line.strip().split(',') for line in file]

    model = DecisionTree(pruning_method)
    model.train_and_test(data)
