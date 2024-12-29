"""
Used information from:

https://www.saedsayad.com/decision_tree.htm

and Chat GPT

"""

import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self, pruning_method):
        self.K = 6
        self.class_index = 9
        self.taken_indexes = {self.class_index}
        self.pruning_method = pruning_method
        self.max_depth = 5  # Limit tree depth to prevent overfitting
        self.min_examples_in_leaf = 10  # Prevent leaves with very few examples
        self.min_gain = 0.01  # Require minimum gain for a split

    def preprocess_dataset(self, dataset):
        """
        Handles missing values in the dataset by imputing with the mode.
        """
        dataset = dataset.replace('?', np.nan)

        print("Preprocessing: Imputing missing values with mode.")
        for column in dataset.columns:
            if dataset[column].isnull().any():
                mode = dataset[column].mode()[0]
                dataset[column] = dataset[column].fillna(mode)

        return dataset

    def calculate_table(self, index, dataset):
        return dataset.iloc[:, index].value_counts()

    def calculate_table_2d(self, index, class_index, dataset):
        return pd.crosstab(dataset.iloc[:, index], dataset.iloc[:, class_index])

    def entropy(self, occurrences, total):
        probabilities = occurrences / total
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def one_attribute_entropy(self, index, dataset):
        occurrences = self.calculate_table(index, dataset)
        return self.entropy(occurrences.values, len(dataset))

    def two_attribute_entropy(self, index, dataset):
        total_entropy = 0
        attribute_occurrences = self.calculate_table(index, dataset)
        occurrences = self.calculate_table_2d(index, self.class_index, dataset)

        for key in occurrences.index:
            probability = attribute_occurrences[key] / len(dataset)
            entropy = self.entropy(occurrences.loc[key], attribute_occurrences[key])
            total_entropy += probability * entropy

        return total_entropy

    def information_gain(self, dataset):
        class_entropy = self.one_attribute_entropy(self.class_index, dataset)
        if class_entropy == 0:
            return -1

        max_gain, best_index = 0, -1
        for i in range(dataset.shape[1]):
            if i in self.taken_indexes:
                continue

            attribute_entropy = self.two_attribute_entropy(i, dataset)
            gain = class_entropy - attribute_entropy
            if gain > max_gain:
                max_gain, best_index = gain, i

        return best_index

    def build_tree(self, dataset, depth=0):
        """
        Builds the decision tree recursively, applying pre- and post-pruning as specified.
        """
        if len(dataset) <= self.K or depth >= self.max_depth:
            most_common_class = dataset.iloc[:, self.class_index].mode()[0]
            return self.Node(-1, None, True, most_common_class)

        index = self.information_gain(dataset)
        if index == -1:
            most_common_class = dataset.iloc[:, self.class_index].mode()[0]
            return self.Node(-1, None, True, most_common_class)

        if self.pruning_method == 0 or self.pruning_method == 2:
            max_gain = self.one_attribute_entropy(self.class_index, dataset) - self.two_attribute_entropy(index, dataset)
            if max_gain < self.min_gain:
                most_common_class = dataset.iloc[:, self.class_index].mode()[0]
                return self.Node(-1, None, True, most_common_class)

        self.taken_indexes.add(index)
        children = {}
        groups = dataset.groupby(dataset.iloc[:, index])

        for key, subset in groups:
            children[key] = self.build_tree(subset, depth + 1)

        self.taken_indexes.remove(index)
        node = self.Node(index, children)

        return node

    def post_prune_tree(self, node, dataset):
        """
        Applies post-pruning after the tree is fully built.
        """
        if node.is_leaf:
            return node

        accuracy_before = self.test_model(node, dataset)
        most_common_class = dataset.iloc[:, self.class_index].mode()[0]
        pruned_node = self.Node(-1, None, True, most_common_class)
        accuracy_after = self.test_model(pruned_node, dataset)

        if accuracy_after >= accuracy_before:
            return pruned_node

        for key, child in node.children.items():
            node.children[key] = self.post_prune_tree(child, dataset)

        return node

    def build_and_prune_tree(self, dataset):
        """
        Build the tree and optionally apply post-pruning based on the selected method.
        """
        tree = self.build_tree(dataset)
        if self.pruning_method in {1, 2}:
            tree = self.post_prune_tree(tree, dataset)
        return tree

    def test_model(self, model, dataset):
        correct = 0
        for _, row in dataset.iterrows():
            node = model
            while not node.is_leaf:
                node = node.children.get(row[node.index], next(iter(node.children.values())))
            if node.value == row[self.class_index]:
                correct += 1
        return correct / len(dataset)

    def compute_train_accuracy(self, dataset):
        model = self.build_and_prune_tree(dataset)
        accuracy = self.test_model(model, dataset)
        print("1. Train Set Accuracy:")
        print(f"    Accuracy: {accuracy * 100:.2f}%\n")
        return model

    def cross_validation(self, dataset):
        fold_size = len(dataset) // 10
        accuracies = []
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        print("10-Fold Cross-Validation Results:")

        for i in range(10):
            fold_start = i * fold_size
            fold_end = (i + 1) * fold_size if i < 9 else len(dataset)

            test_fold = dataset.iloc[fold_start:fold_end]
            train_fold = pd.concat([dataset.iloc[:fold_start], dataset.iloc[fold_end:]])

            model = self.build_and_prune_tree(train_fold)
            fold_accuracy = self.test_model(model, test_fold)
            accuracies.append(fold_accuracy)

            print(f"    Accuracy Fold {i + 1}: {fold_accuracy * 100:.2f}%")

        average_accuracy = np.mean(accuracies)
        std_deviation = np.std(accuracies)
        print(f"\n    Average Accuracy: {average_accuracy * 100:.2f}%")
        print(f"    Standard Deviation: {std_deviation * 100:.2f}%\n")

    def compute_test_accuracy(self, model, dataset):
        accuracy = self.test_model(model, dataset)
        print("2. Test Set Accuracy:")
        print(f"    Accuracy: {accuracy * 100:.2f}%")

    def train_and_test(self, dataset):
        dataset = pd.DataFrame(dataset)
        dataset = dataset.drop_duplicates()

        dataset = self.preprocess_dataset(dataset)

        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        split_index = int(0.8 * len(dataset))
        train_set = dataset.iloc[:split_index]
        test_set = dataset.iloc[split_index:]

        model = self.compute_train_accuracy(train_set)
        self.cross_validation(train_set)
        self.compute_test_accuracy(model, test_set)

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

