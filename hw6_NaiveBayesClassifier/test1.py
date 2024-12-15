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

def preprocess_data(raw_data, choice):
    # Split data into labels and features and handle missing values
    labels = [0 if row[0] == "democrat" else 1 for row in raw_data]
    features = [row[1:] for row in raw_data]

    if choice == 0:
        # Treat '?' as a third value "abstain"
        features = [[val if val != '?' else '2' for val in row] for row in features]
    else:
        # Replace '?' with the most common value in the column
        for j in range(len(features[0])):
            column_values = [row[j] for row in features if row[j] != '?']

            #This method returns the most common elements in the Counter. The argument 1 specifies that we want the single most common element.
            most_common = Counter(column_values).most_common(1)[0][0]
            for i in range(len(features)):
                if features[i][j] == '?':
                    features[i][j] = most_common

    return labels, features

def split_data(features, labels, train_ratio):
    # Split data into training and test sets
    data = list(zip(features, labels))
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    #The * operator unpacks the train_data list of tuples.
    # Example: train_data = [(['yes', 'no', 'yes'], 0), (['no', 'yes', 'no'], 1)]
    # Output: (['yes', 'no', 'yes'], ['no', 'yes', 'no']), (0, 1)
    train_features, train_labels = zip(*train_data)
    test_features, test_labels = zip(*test_data)

    return list(train_features), list(train_labels), list(test_features), list(test_labels)

# def train_naive_bayes(train_features, train_labels):
#     # Train a Naive Bayes classifier
#     unique_values = list(set(val for row in train_features for val in row))
#     value_to_index = {val: idx for idx, val in enumerate(unique_values)}

#     class_counts = Counter(train_labels)
#     feature_counts = {label: defaultdict(lambda: defaultdict(int)) for label in class_counts}

#     for features_row, label in zip(train_features, train_labels):
#         for j, value in enumerate(features_row):
#             feature_counts[label][j][value_to_index[value]] += 1

#     return feature_counts, class_counts, value_to_index

# def train_naive_bayes(train_features, train_labels):
#     # Get unique values across all features
#     unique_values = list(set(val for row in train_features for val in row))
#     value_to_index = {val: idx for idx, val in enumerate(unique_values)}

#     # Count the occurrences of each class
#     class_counts = Counter(train_labels)

#     # Initialize nested dictionaries to store feature counts per class
#     feature_counts = {label: defaultdict(lambda: defaultdict(int)) for label in class_counts}

#     # Populate the feature counts based on the training data
#     for features_row, label in zip(train_features, train_labels):
#         for j, value in enumerate(features_row):
#             value_index = value_to_index[value]
#             feature_counts[label][j][value_index] += 1

#     # Apply Laplace Smoothing to feature counts
#     smoothed_model = {label: {} for label in class_counts}
#     for label in feature_counts:
#         smoothed_model[label] = {}
#         for j in feature_counts[label]:
#             smoothed_model[label][j] = {}
#             total_count = sum(feature_counts[label][j].values()) + len(value_to_index)  # Total with smoothing
#             for value_index in range(len(value_to_index)):
#                 # Add 1 to numerator for Laplace smoothing and normalize
#                 smoothed_model[label][j][value_index] = (feature_counts[label][j].get(value_index, 0) + 1) / total_count

#     return smoothed_model, class_counts, value_to_index

def train_naive_bayes(train_features, train_labels):
    """
    Train a Naive Bayes classifier using Laplace smoothing.
    
    Args:
        train_features: List of feature rows (categorical features).
        train_labels: List of class labels corresponding to each feature row.
    
    Returns:
        smoothed_model: Dictionary with smoothed probabilities for each feature value per class.
        class_priors: Dictionary with prior probabilities for each class.
        value_to_index: Mapping of unique feature values to indices.
    """
    
    # Extract unique values across all features
    unique_values = list(set(val for row in train_features for val in row))
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}

    # Count occurrences of each class
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)

    # Calculate class priors
    class_priors = {label: count / total_samples for label, count in class_counts.items()}

    # Initialize nested dictionaries to store feature counts per class
    feature_counts = {label: defaultdict(lambda: defaultdict(int)) for label in class_counts}

    # Populate the feature counts based on the training data
    for features_row, label in zip(train_features, train_labels):
        for j, value in enumerate(features_row):
            value_index = value_to_index[value]
            feature_counts[label][j][value_index] += 1

    # Apply Laplace smoothing to feature counts
    smoothed_model = {label: {} for label in class_counts}
    num_unique_values = len(value_to_index)
    for label in feature_counts:
        smoothed_model[label] = {}
        for j in feature_counts[label]:
            smoothed_model[label][j] = {}
            total_count = sum(feature_counts[label][j].values()) + num_unique_values  # Total with smoothing
            for value_index in range(num_unique_values):
                # Add 1 to numerator for Laplace smoothing and normalize
                smoothed_model[label][j][value_index] = (
                    feature_counts[label][j].get(value_index, 0) + 1
                ) / total_count

    return smoothed_model, class_priors, value_to_index


def predict(feature_row, model, class_counts, value_to_index):
    # Predict the class for a given row of features
    probabilities = []
    for label in class_counts:
        #Uses the logarithm of probabilities to avoid underflow when multiplying many small probabilities.
        prob = math.log(class_counts[label] / sum(class_counts.values()))
        for j, value in enumerate(feature_row):
            value_index = value_to_index[value]
            prob += math.log(model[label][j][value_index] + 1)
            prob -= math.log(sum(model[label][j].values()))
        probabilities.append(prob)
    return 0 if probabilities[0] > probabilities[1] else 1

def evaluate(features_set, labels_set, model, class_counts, value_to_index):
    # Calculate classifier accuracy
    correct = 0
    for feature_row, label in zip(features_set, labels_set):
        if predict(feature_row, model, class_counts, value_to_index) == label:
            correct += 1
    return correct / len(features_set)

def cross_validate(features, labels, folds):
    # Perform k-fold cross-validation
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

if __name__ == "__main__":
    file_path = "hw6_NaiveBayesClassifier/congressional+voting+records/house-votes-84.data"
    choice = int(input("Enter 0 for 'abstain' or 1 to replace '?': "))
    
    raw_data = load_data(file_path)
    labels, features = preprocess_data(raw_data, choice)
    
    train_features, train_labels, test_features, test_labels = split_data(features, labels, 0.8)
    
    model, class_counts, value_to_index = train_naive_bayes(train_features, train_labels)
    
    train_accuracy = evaluate(train_features, train_labels, model, class_counts, value_to_index)
    print(f"Train Set Accuracy: {train_accuracy * 100:.2f}%")
    
    accuracies, mean_accuracy, std_dev_accuracy = cross_validate(features, labels, 10)
    print("\n10-Fold Cross-Validation Results:")
    for i, acc in enumerate(accuracies):
        print(f"    Accuracy Fold {i + 1}: {acc * 100:.2f}%")
    print(f"\n    Average Accuracy: {mean_accuracy * 100:.2f}%")
    print(f"    Standard Deviation: {std_dev_accuracy * 100:.2f}%")
    
    test_accuracy = evaluate(test_features, test_labels, model, class_counts, value_to_index)
    print(f"\nTest Set Accuracy: {test_accuracy * 100:.2f}%")
