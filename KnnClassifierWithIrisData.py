import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
# data is loaded on data.csv file on same directory
df = pd.read_csv("data.csv")

# Features and labels
X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = df['Name']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Get user input
print("Enter flower measurements:")
sepal_length = float(input("Sepal Length (cm): "))
sepal_width = float(input("Sepal Width (cm): "))
petal_length = float(input("Petal Length (cm): "))
petal_width = float(input("Petal Width (cm): "))

# Make prediction
input_data = pd.DataFrame([{
    'SepalLength': sepal_length,
    'SepalWidth': sepal_width,
    'PetalLength': petal_length,
    'PetalWidth': petal_width
}])

prediction = knn.predict(input_data)
predicted_label = le.inverse_transform(prediction)

# Output result
print(f"\nThe predicted Iris species is: {predicted_label[0]}")
