# 📈 Salary Prediction Using Polynomial Regression

This project uses a **Polynomial Regression** model to predict an employee's **salary** based on their **years of experience**. It includes data visualization, model evaluation, and prediction capabilities using a dataset loaded from a CSV file.

---

## 🧠 Objective

To build a regression model that captures the non-linear relationship between years of experience and salary, and to demonstrate how polynomial features can improve prediction accuracy.

---

## 📂 Files

- `salary_poly_regression.py` — Main Python script that:
  - Loads and shuffles salary data from `Salary_Data.csv`
  - Visualizes the dataset
  - Builds and evaluates a polynomial regression model
  - Predicts salary for a given input (e.g., 6.5 years of experience)

- `Salary_Data.csv` — CSV file containing the dataset with two columns:
  - `YearsExperience`
  - `Salary`

---

## 📊 Dataset Format

Ensure your `Salary_Data.csv` file looks like this:

```csv
YearsExperience,Salary
1.1,39343.00
1.3,46205.00
1.5,37731.00
...
