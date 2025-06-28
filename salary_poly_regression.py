
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# --- 1. Load and shuffle dataset ---
data = pd.read_csv("Salary_Data.csv")
data = data.sample(frac=1).reset_index(drop=True)  # shuffle data

print("\nData Preview:\n", data.head())

# --- 2. Visualize data ---
plt.figure(figsize=(8, 5))
sns.scatterplot(x='YearsExperience', y='Salary', data=data)
plt.title("Years of Experience vs. Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.grid(True)
plt.show()

# --- 3. Prepare features and labels ---
X = data[['YearsExperience']].values
y = data['Salary'].values

# --- 4. Cross-validation evaluation ---
degree = 3  # You can adjust this
pipeline = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

print("\nCross-Validation (5-Fold) R² Scores:")
for i, score in enumerate(cv_scores, start=1):
    print(f" Fold {i}: {score:.4f}")
print(f"Average R²: {cv_scores.mean():.4f}")

# --- 5. Split into train/test sets (randomized) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # random split each run

# --- 6. Train the model on training data ---
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

# --- 7. Evaluate on test data ---
y_predic_test = model.predict(X_test_poly)
print(f"\nTest Set R²:  {r2_score(y_test, y_predic_test):.4f}")
print(f"Test RMSE:    {mean_squared_error(y_test, y_predic_test, squared=False):.2f}")

# --- 8. Visualize the fitted curve ---
X_curve = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_curve_poly = poly.transform(X_curve)
y_curve = model.predict(X_curve_poly)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Actual data")
plt.plot(X_curve, y_curve, color='red', label=f"Polynomial Fit (degree={degree})")
plt.title("Polynomial Regression Fit")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.show()

# --- 9. Predict for custom input ---
new_input = [[6.5]]  # Example: 6.5 years
new_input_poly = poly.transform(new_input)
predicted_salary = model.predict(new_input_poly)
print(f"\nPredicted salary for {new_input[0][0]} years experience: ${predicted_salary[0]:,.2f}")

