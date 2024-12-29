
import random
import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self, pruning_method, missing_data_handling="mode"):
        self.K = 6  # Minimum examples in leaf for pre-pruning
        self.class_index = 9  # Index of the class label
        self.taken_indexes = {self.class_index}
        self.pruning_method = pruning_method
        self.max_depth = 5  # Maximum tree depth for pre-pruning
        self.min_examples_in_leaf = 10  # Minimum examples in leaf for pre-pruning
        self.min_gain = 0.01  # Minimum information gain for splitting
        self.missing_data_handling = missing_data_handling

    def preprocess(self, dataset):
        dataset = pd.DataFrame(dataset)
        for col in dataset.columns:
            if dataset[col].dtype == 'object':
                if self.missing_data_handling == "mode":
                    mode = dataset[col].mode()[0]
                    dataset[col] = dataset[col].replace('?', mode)
                elif self.missing_data_handling == "drop":
                    dataset = dataset[dataset[col] != '?']
            else:
                mean = dataset[col].astype(float).mean()
                dataset[col] = dataset[col].replace('?', mean).astype(float)
        return dataset

    def stratified_split(self, dataset, test_size=0.2):
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset
        class_counts = dataset.iloc[:, self.class_index].value_counts()
        train_indices, test_indices = [], []

        for class_value, count in class_counts.items():
            indices = dataset[dataset.iloc[:, self.class_index] == class_value].index.tolist()
            split_point = int(len(indices) * (1 - test_size))
            train_indices.extend(indices[:split_point])
            test_indices.extend(indices[split_point:])

        train_set = dataset.loc[train_indices].reset_index(drop=True)
        test_set = dataset.loc[test_indices].reset_index(drop=True)
        return train_set, test_set

    def cross_validation_split(self, dataset, k_folds=10):
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        class_counts = dataset.iloc[:, self.class_index].value_counts()
        folds = [[] for _ in range(k_folds)]

        for class_value, count in class_counts.items():
            indices = dataset[dataset.iloc[:, self.class_index] == class_value].index.tolist()
            random.shuffle(indices)

            for i, index in enumerate(indices):
                folds[i % k_folds].append(index)

        return [dataset.loc[fold].reset_index(drop=True) for fold in folds]

    def calculate_table(self, index, dataset):
        return dataset.iloc[:, index].value_counts()

    def calculate_table_2d(self, index, class_index, dataset):
        return pd.crosstab(dataset.iloc[:, index], dataset.iloc[:, class_index])

    def entropy(self, occurrences, total):
        probabilities = occurrences / total
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Add small value to avoid log(0)

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
        if self.pruning_method in {0, 2}:  # Apply pre-pruning
            if len(dataset) <= self.min_examples_in_leaf or depth >= self.max_depth:
                most_common_class = dataset.iloc[:, self.class_index].mode()[0]
                return self.Node(-1, None, True, most_common_class)

        index = self.information_gain(dataset)
        if index == -1 or (self.pruning_method in {0, 2} and self.min_gain > 0):
            most_common_class = dataset.iloc[:, self.class_index].mode()[0]
            return self.Node(-1, None, True, most_common_class)

        self.taken_indexes.add(index)
        children = {}
        groups = dataset.groupby(dataset.iloc[:, index])

        for key, subset in groups:
            children[key] = self.build_tree(subset, depth + 1)

        self.taken_indexes.remove(index)
        node = self.Node(index, children)

        if self.pruning_method in {1, 2}:  # Apply post-pruning
            node = self.post_pruning(node, dataset)

        return node

    def post_pruning(self, node, dataset):
        if node.is_leaf:
            return node

        accuracy_before = self.test_model(node, dataset)
        most_common_class = dataset.iloc[:, self.class_index].mode()[0]
        pruned_node = self.Node(-1, None, True, most_common_class)
        accuracy_after = self.test_model(pruned_node, dataset)

        if accuracy_after >= accuracy_before:
            return pruned_node

        for key, child in node.children.items():
            node.children[key] = self.post_pruning(child, dataset)

        return node

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
        model = self.build_tree(dataset)
        accuracy = self.test_model(model, dataset)
        print("1. Train Set Accuracy:")
        print(f"    Accuracy: {accuracy * 100:.2f}%\n")
        return model

    def cross_validation(self, dataset):
        folds = self.cross_validation_split(dataset)
        accuracies = []
        for i in range(len(folds)):
            test_fold = folds[i]
            train_folds = pd.concat(folds[:i] + folds[i + 1:]).reset_index(drop=True)
            model = self.build_tree(train_folds)
            fold_accuracy = self.test_model(model, test_fold)
            accuracies.append(fold_accuracy)

        average_accuracy = np.mean(accuracies)
        std_deviation = np.std(accuracies)
        print("10-Fold Cross-Validation Results:")
        for i, accuracy in enumerate(accuracies):
            print(f"    Accuracy Fold {i + 1}: {accuracy * 100:.2f}%")
        print(f"\n    Average Accuracy: {average_accuracy * 100:.2f}%")
        print(f"    Standard Deviation: {std_deviation * 100:.2f}%\n")

    def compute_test_accuracy(self, model, dataset):
        accuracy = self.test_model(model, dataset)
        print("2. Test Set Accuracy:")
        print(f"    Accuracy: {accuracy * 100:.2f}%")

    def train_and_test(self, dataset):
        dataset = self.preprocess(dataset)
        train_set, test_set = self.stratified_split(dataset, test_size=0.2)

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
    missing_data_handling = input("Handle missing data? (mode/drop): ").strip()

    with open("hw7_ID3/breast+cancer/breast-cancer.data") as file:
        data = [line.strip().split(',') for line in file]

    model = DecisionTree(pruning_method, missing_data_handling)
    model.train_and_test(data)
