import random
from collections import defaultdict
from math import log

FOLDS_NUMBER = 10
ATTRIBUTES_NUMBER = 17
ALPHA = 1
CLASSES = 2

class Observation(list):
    pass


def file_load(file_name):
    with open(file_name, 'r') as file:
        return [Observation(line.strip().split(',')) for line in file]
    
file_data = file_load("house-votes-84.data")
print(file_data)
# missing something here
random.shuffle(file_data)
print(file_data)


# missing something here

def calculate_probabilities(observ, class_type, counts, class_counts, training_size):
    probabilities = {class_type: log(class_counts[class_type] / training_size)}

    for j in range(1, ATTRIBUTES_NUMBER):
        if observ[j] != "?":
            enum = counts[class_type][j][observ[j]] + ALPHA
            denom = len(counts[class_type][j]) + CLASSES * ALPHA
            probabilities[class_type] += log(enum / denom)

    return probabilities

def make_predictions(fold_data, counts, class_counts, training_size, types):
    predictions = []
    actual = [observ[0] for observ in fold_data]

    for observ in fold_data:
        probabilities = {}

        for class_type in types:
            probabilities.update(calculate_probabilities(observ, class_type, counts, class_counts, training_size))

        predicted_class = max(probabilities, key=probabilities.get)
        predictions.append(predicted_class)

    return predictions, actual

def calc_accuracy(predicted_data, actual_data):
    return sum(1 for pred, act in zip(predicted_data, actual_data) if pred == act) / len(predicted_data)

def naive_bayes_classification(file_data):
    types = ["democrat", "republican"]
    data_size = len(file_data)
    fold_size = data_size // FOLDS_NUMBER

    sets = [file_data[i * fold_size : (i + 1) * fold_size] for i in range(FOLDS_NUMBER)]

    total_accuracy = 0.0
    folds_info = ""

    for fold, fold_data in enumerate(sets):
        class_counts = defaultdict(int)
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for i, f_data in enumerate(sets):
            if i == fold:
                continue

            for observ in f_data:
                class_type = observ[0]
                class_counts[class_type] += 1

                for j in range(1, ATTRIBUTES_NUMBER):
                    if observ[j] != "?":
                        counts[class_type][j][observ[j]] += 1
        
        predictions, actual = make_predictions(fold_data, counts, class_counts, sum(class_counts.values()), types)

        accuracy = calc_accuracy(predictions, actual)
        total_accuracy += accuracy
        folds_info += f"Fold {fold+1} Accuracy: {accuracy}\n"

    avg_accuracy = total_accuracy / FOLDS_NUMBER
    folds_info += 'Average accuracy: ' + str(avg_accuracy)

    return folds_info


folds = naive_bayes_classification(file_data)
print(folds)


# Fold 1 Accuracy: 0.9069767441860465
# Fold 2 Accuracy: 0.9069767441860465
# Fold 3 Accuracy: 1.0
# Fold 4 Accuracy: 0.8837209302325582
# Fold 5 Accuracy: 0.8837209302325582
# Fold 6 Accuracy: 0.8372093023255814
# Fold 7 Accuracy: 0.8604651162790697
# Fold 8 Accuracy: 0.9767441860465116
# Fold 9 Accuracy: 0.8372093023255814
# Fold 10 Accuracy: 0.9302325581395349
# Average accuracy: 0.9023255813953487