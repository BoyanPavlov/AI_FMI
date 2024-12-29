import math
import pandas as pd
import numpy as np

# Constants
FEATURE_NAMES = [
    'Class',
    'age',
    'menopause',
    'tumor-size',
    'inv-nodes',
    'node-caps',
    'deg-malig',
    'breast',
    'breast-quad',
    'irradiat'
]

CLASSES = ['no-recurrence-events', 'recurrence-events']
ATTRIBUTES = dict()
K_THRESHOLD = 30

def read_dataset_csv():
    """Reads the dataset and returns a Pandas DataFrame."""
    file_path = "hw7_ID3/breast+cancer/breast-cancer.data"
    dataframe = pd.read_csv(file_path, names=FEATURE_NAMES)
    dataframe = dataframe.replace('?', np.nan)
    dataframe.dropna(inplace=True)
    return dataframe

def set_attribute_values(dataframe):
    """Sets the unique attribute values for each feature."""
    for column in FEATURE_NAMES:
        ATTRIBUTES[column] = dataframe[column].unique()

def generate_folds(dataframe):
    """Generates 10 folds for cross-validation."""
    fold_size = int(dataframe.shape[0] / 10)
    return [dataframe.sample(n=fold_size) for _ in range(10)]

class Node:
    """Represents a decision tree node."""
    def __init__(self, attribute=None, branches=None, classification=None):
        self.attribute = attribute
        self.branches = {} if branches is None else branches
        self.classification = classification

def check_same_class(dataframe):
    """Checks if all instances belong to the same class."""
    return len(dataframe['Class'].unique()) == 1

def identify_majority_class(dataframe):
    """Identifies the majority class in the dataset."""
    class_counts = dataframe['Class'].value_counts()
    return class_counts.idxmax()

def calculate_entropy(probabilities):
    """Calculates entropy given probabilities."""
    return -sum(prob * math.log(prob, 2) if prob > 0 else 0 for prob in probabilities)

def calculate_class_entropy(dataframe):
    """Calculates entropy of the 'Class' attribute."""
    class_counts = dataframe['Class'].value_counts(normalize=True)
    return calculate_entropy(class_counts)

def calculate_information_gain(dataframe, attribute):
    """Calculates information gain for a specific attribute."""
    gain = calculate_class_entropy(dataframe)
    for value in dataframe[attribute].unique():
        attribute_rows = dataframe[dataframe[attribute] == value]
        class_counts = attribute_rows['Class'].value_counts(normalize=True)
        probabilities = [class_counts.get(cls, 0) for cls in CLASSES]
        gain -= len(attribute_rows) / len(dataframe) * calculate_entropy(probabilities)
    return gain

def construct_decision_tree(dataframe, allowed_attributes):
    """Constructs a decision tree using ID3 algorithm."""
    root = Node()
    if check_same_class(dataframe) or len(allowed_attributes) == 0 or dataframe.shape[0] <= K_THRESHOLD:
        root.classification = identify_majority_class(dataframe)
        return root
    else:
        best_attribute = max(allowed_attributes, key=lambda attr: calculate_information_gain(dataframe, attr))
        root.attribute = best_attribute
        child_attributes = [attr for attr in allowed_attributes if attr != best_attribute]
        grouped_data = dataframe.groupby(best_attribute)
        for value, group in grouped_data:
            child_node = Node()
            if check_same_class(group) or group.shape[0] <= K_THRESHOLD:
                child_node.classification = identify_majority_class(group)
            else:
                child_node = construct_decision_tree(group, child_attributes)
            root.branches[value] = child_node
    return root

def make_prediction(row, root):
    """Makes a prediction for a single row using the decision tree."""
    node = root
    while node and node.branches:
        attribute_value = row[node.attribute]
        node = node.branches.get(attribute_value, None)
    return node.classification if node else root.classification

def evaluate_accuracy(dataframe, root):
    """Evaluates the accuracy of the decision tree."""
    correct_predictions = sum(
        1 for _, row in dataframe.iterrows() if row['Class'] == make_prediction(row, root)
    )
    return correct_predictions / float(dataframe.shape[0])

def perform_training(folds):
    """Performs cross-validation training and computes accuracy."""
    accuracies = []
    for index, fold in enumerate(folds):
        train_folds = folds[:index] + folds[index+1:]
        test = fold.copy()
        train = pd.concat(train_folds)
        decision_tree_root = construct_decision_tree(train, FEATURE_NAMES[1:])
        accuracy = evaluate_accuracy(test, decision_tree_root)
        print(f"Fold {index+1} Accuracy: {accuracy}")
        accuracies.append(accuracy)
    print('Mean accuracy:', np.mean(accuracies))

def main():
    print("Loading dataset...")
    dataframe = read_dataset_csv()
    set_attribute_values(dataframe)
    folds = generate_folds(dataframe)
    perform_training(folds)

if __name__ == "__main__":
    main()
