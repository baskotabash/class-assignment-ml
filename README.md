
# Iris Flower Species Prediction with K-Nearest Neighbors (KNN)

This Python program uses the K-Nearest Neighbors (KNN) algorithm to predict the species of an Iris flower based on its sepal and petal measurements. The program is trained on the Iris dataset and accepts user input to make real-time predictions.

## Prerequisites

Before running the program, ensure you have the following Python packages installed:

* pandas
* scikit-learn

You can install them using pip if they are not already installed:
pip install pandas scikit-learn

## Dataset

The program uses the famous [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris), where each row represents a flower with four features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The target variable is the Iris flower species (`Setosa`, `Versicolor`, `Virginica`).

### Data Format

The dataset should be stored in a `data.csv` file in the same directory as this script, with the following columns:

| SepalLength | SepalWidth | PetalLength | PetalWidth | Name   |
| ----------- | ---------- | ----------- | ---------- | ------ |
| 5.1         | 3.5        | 1.4         | 0.2        | Setosa |
| 4.9         | 3.0        | 1.4         | 0.2        | Setosa |
| ...         | ...        | ...         | ...        | ...    |

## How to Run

1. Ensure your Python environment has the required packages (pandas, scikit-learn).
2. Download the 'data.csv' dataset and place it in the same directory as the script.
3. Run the Python script:


python iris_prediction.py

## User Input

Once the script is running, it will prompt the user to enter the following flower measurements (in cm):

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

Example:

Enter flower measurements:
Sepal Length (cm): 5.1
Sepal Width (cm): 3.5
Petal Length (cm): 1.4
Petal Width (cm): 0.2

## Output

After entering the measurements, the script will output the predicted Iris species.

Example output:
The predicted Iris species is: Setosa


## Model Details

* The script uses the **K-Nearest Neighbors (KNN)** algorithm with `n_neighbors=3` to classify the flower species.
* The target labels are encoded using `LabelEncoder` to handle the categorical output of species.
* The dataset is split into training and testing sets (80% training, 20% testing) to evaluate the model's performance.

## License

This project is open-source and available under the [MIT License](LICENSE).
