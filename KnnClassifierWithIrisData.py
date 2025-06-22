import pandas as pd
import numpy as np
from collections import Counter


# KNN Classifier
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def knn_classifier(train_data, test_data, k=3):
    distances = []
    for i in range(len(train_data)):
        train_row = train_data.iloc[i, :-1].values  # features (exclude 'Name')
        label = train_data.iloc[i, -1]  # label (the iris type)
        dist = euclidean_distance(train_row, test_data)
        distances.append((dist, label))

    # Sort distances by the first value (distance)
    distances.sort(key=lambda x: x[0])

    # Get the k nearest neighbors
    neighbors = distances[:k]

    # Get the most common class among the neighbors
    labels = [neighbor[1] for neighbor in neighbors]
    most_common_label = Counter(labels).most_common(1)[0][0]

    return most_common_label


# Read data from CSV file
df = pd.read_csv('data.csv')


# Function to prompt for user input and classify
def predict_iris(k=3):
    try:
        # Prompt the user for input
        sepal_length = float(input("Enter sepal length (in cm): "))
        sepal_width = float(input("Enter sepal width (in cm): "))
        petal_length = float(input("Enter petal length (in cm): "))
        petal_width = float(input("Enter petal width (in cm): "))

        # Combine user inputs into a test data array
        test_data = np.array([sepal_length, sepal_width, petal_length, petal_width])

        # Get the prediction using KNN
        prediction = knn_classifier(df, test_data, k)

        # Output the result
        print(f"The predicted iris species is: {prediction}")
    except ValueError:
        print("Invalid input. Please enter numeric values for all measurements.")


# Example Usage
predict_iris(k=3)
