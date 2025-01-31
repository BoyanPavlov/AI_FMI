import numpy as np
import math
import random
from collections import Counter, defaultdict


"""
Used information from:

    https://scikit-learn.org/stable/modules/naive_bayes.html

    https://www.geeksforgeeks.org/naive-bayes-classifiers/

    https://www.geeksforgeeks.org/cross-validation-machine-learning/


"""

def load_data(file_path):
    with open(file_path, 'r') as file:
        return [line.strip().split(',') for line in file.readlines()]
    

def preprocess_data(raw_data : list[list[str]], choice: int):
    """
    Function about "?" treatment:
     - treat it as a third value "abstain"
     - Or replace it with the most common value in the column
    """
    # Split data into labels and features and handle missing values
    labels = [0 if row[0] == "democrat" else 1 for row in raw_data]
    features = [row[1:] for row in raw_data]

    if choice == 0:
        features = [[val if val != '?' else '2' for val in row] for row in features]
    else: 
        for j in range(len(features[0])):
            column_values = [row[j] for row in features if row[j] != '?']

            #Notes to myself for code specifications:
            # This method returns the most common elements in the Counter.
            # The argument 1 specifies that we want the single most common element.
            most_common = Counter(column_values).most_common(1)[0][0]
            for i in range(len(features)):
                if features[i][j] == '?':
                    features[i][j] = most_common

    return labels, features

def split_data(features: list[list[str]], labels: list[int], train_ratio: float):
    """
    Split data into training and test sets
    """
    data = list(zip(features, labels))
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    #Notes to myself for code specifications:
    # The * operator unpacks the train_data list of tuples.
    # Example: train_data = [(['yes', 'no', 'yes'], 0), (['no', 'yes', 'no'], 1)]
    # Output: (['yes', 'no', 'yes'], ['no', 'yes', 'no']), (0, 1)
    train_features, train_labels = zip(*train_data)
    test_features, test_labels = zip(*test_data)

    return list(train_features), list(train_labels), list(test_features), list(test_labels)


def train_naive_bayes(train_features: list[list[str]], train_labels: list[int]):
    """
    Train a Naive Bayes classifier with Laplace smoothing.
    """
    unique_values = list(set(val for row in train_features for val in row))
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}

    class_counts = Counter(train_labels)
    feature_counts = {label: defaultdict(lambda: defaultdict(int)) for label in class_counts}

    for features_row, label in zip(train_features, train_labels):
        for j, value in enumerate(features_row):
            value_index = value_to_index[value]
            feature_counts[label][j][value_index] += 1

    # Apply Laplace smoothing
    for label in feature_counts:
        for j in feature_counts[label]:
            total_count = sum(feature_counts[label][j].values())
            for value_index in value_to_index.values():
                # Laplace smoothing: Add 1 to each count
                feature_counts[label][j][value_index] += 1
            # Adjust the total count to include the smoothing
            total_count += len(value_to_index)
            # Normalize counts
            for value_index in feature_counts[label][j]:
                feature_counts[label][j][value_index] /= total_count

    return feature_counts, class_counts, value_to_index

def calculate_probability(
    label: int, 
    feature_row: list[str], 
    model: dict[int, dict[int, dict[int, float]]], 
    class_counts: Counter, 
    value_to_index: dict[str, int]
):
    """
    Calculate the log-probability for a given class label.
     - Uses the logarithm of probabilities to avoid underflow when multiplying many small probabilities.
    """
    prob = math.log(class_counts[label] / sum(class_counts.values()))
    
    for j, value in enumerate(feature_row):
        value_index = value_to_index[value]
        prob += math.log(model[label][j][value_index])
    
    return prob

def predict(
    feature_row: list[str], 
    model: dict[int, dict[int, dict[int, float]]], 
    class_counts: Counter, 
    value_to_index: dict[str, int]):
    """
    Predict the class for a given row of features.
    """
    probabilities = [
        calculate_probability(label, feature_row, model, class_counts, value_to_index)
        for label in class_counts
    ]
    return 0 if probabilities[0] > probabilities[1] else 1

def evaluate(
    features_set: list[list[str]], 
    labels_set: list[int], 
    model: dict[int, dict[int, dict[int, float]]], 
    class_counts: Counter, 
    value_to_index: dict[str, int]
):
    # Calculate classifier accuracy
    correct = 0
    for feature_row, label in zip(features_set, labels_set):
        if predict(feature_row, model, class_counts, value_to_index) == label:
            correct += 1
    return correct / len(features_set)


def k_fold_cross_validate(features: list[list[str]], labels: list[int], folds: int):
    fold_size = len(features) // folds
    data = list(zip(features, labels))
    random.shuffle(data)

    accuracies = []

    for fold in range(folds):
        test_data = data[fold * fold_size:(fold + 1) * fold_size]
        train_data = data[:fold * fold_size] + data[(fold + 1) * fold_size:]

        train_features, train_labels = zip(*train_data)
        test_features, test_labels = zip(*test_data)

        model, class_counts, value_to_index = train_naive_bayes(list(train_features), list(train_labels))
        accuracy = evaluate(list(test_features), list(test_labels), model, class_counts, value_to_index)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    std_dev_accuracy = np.std(accuracies)
    return accuracies, mean_accuracy, std_dev_accuracy



def main():
    file_path = "hw6_NaiveBayesClassifier/congressional+voting+records/house-votes-84.data"
    choice = int(input("Enter 0 for 'abstain' or 1 to replace '?' with the most common element: "))
    
    raw_data = load_data(file_path)
    labels, features = preprocess_data(raw_data, choice)
    
    train_features, train_labels, test_features, test_labels = split_data(features, labels, 0.8)
    
    model, class_counts, value_to_index = train_naive_bayes(train_features, train_labels)
    
    train_accuracy = evaluate(train_features, train_labels, model, class_counts, value_to_index)
    print(f"Train Set Accuracy: {train_accuracy * 100:.2f}%")
    
    accuracies, mean_accuracy, std_dev_accuracy = k_fold_cross_validate(features, labels, 10)
    print("\n10-Fold Cross-Validation Results:")
    for i, acc in enumerate(accuracies):
        print(f"\tAccuracy Fold {i + 1}: {acc * 100:.2f}%")
    print(f"\n\tAverage Accuracy: {mean_accuracy * 100:.2f}%")
    print(f"\tStandard Deviation: {std_dev_accuracy * 100:.2f}%")
    
    test_accuracy = evaluate(test_features, test_labels, model, class_counts, value_to_index)
    print(f"\nTest Set Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()