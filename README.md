
# KNN Classifier for Iris Species Prediction

This Python script implements a basic K-Nearest Neighbors (KNN) algorithm to classify the species of the Iris flower based on its physical measurements: sepal length, sepal width, petal length, and petal width. The classification is based on the Euclidean distance between the test sample and the training data.

## Requirements

To run this code, you will need:

* Python 3.x
* pandas
* numpy

You can install the required libraries using pip:

```bash
pip install pandas numpy
```

## Files

* `data.csv`: A CSV file containing the training data. This file should have columns representing the physical measurements of the Iris flowers (sepal length, sepal width, petal length, petal width), along with the corresponding Iris species label.

## Code Overview

### 1. **`euclidean_distance(a, b)`**

* Calculates the Euclidean distance between two points `a` and `b`.
* This function is used to measure the "closeness" between test data and training data.

### 2. **`knn_classifier(train_data, test_data, k=3)`**

* Takes the training data (`train_data`), the test data (`test_data`), and the value of `k` as inputs.
* Computes the Euclidean distance from each row of the training data to the test data.
* Sorts the distances and selects the `k` nearest neighbors.
* The most common class (label) among the neighbors is returned as the predicted species.

### 3. **`predict_iris(k=3)`**

* Prompts the user to input the physical measurements of an Iris flower.
* Uses the KNN classifier to predict the species of the flower based on the provided measurements.
* Outputs the predicted species.

## Example Usage

1. Make sure the `data.csv` file is present in the same directory as the script, containing the Iris dataset.
2. Run the script:

```bash
python knn_classifier.py
```

3. When prompted, enter the sepal and petal measurements of the Iris flower. For example:

   * Sepal length (in cm): `5.1`
   * Sepal width (in cm): `3.5`
   * Petal length (in cm): `1.4`
   * Petal width (in cm): `0.2`

4. The script will output the predicted Iris species, such as `Iris-setosa`, `Iris-versicolor`, or `Iris-virginica`.

## Example Output

```bash
Enter sepal length (in cm): 5.1
Enter sepal width (in cm): 3.5
Enter petal length (in cm): 1.4
Enter petal width (in cm): 0.2
The predicted iris species is: Iris-setosa
```

## Notes

* The dataset used in this script is assumed to be in the form of a CSV file (`data.csv`) containing measurements of Iris flowers with columns for sepal length, sepal width, petal length, petal width, and species label.
* The KNN algorithm used here is a simple implementation based on the Euclidean distance and the majority vote among the `k` nearest neighbors. It is recommended to use more sophisticated libraries like `scikit-learn` for larger datasets or more advanced functionality.

## Customization

* You can change the value of `k` by modifying the function call `predict_iris(k=3)` to use a different number of neighbors for classification.

## License

This code is provided under the MIT License. Feel free to use and modify it as needed.
