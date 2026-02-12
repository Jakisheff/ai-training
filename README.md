# Machine Learning: Model Evaluation and Metrics

This repository contains exercises and solutions for evaluating Machine Learning models using Scikit-Learn. The goal is to understand how to choose and compute the right metrics for both regression and classification problems, handle imbalanced datasets, compare different ML algorithms, and optimize hyperparameters using Grid Search.

## Project Structure

- `exercises/`
  - `check_env.py`: Script to verify the Python environment and required libraries.
  - `ex01_mse.py`: Calculates Mean Squared Error (MSE).
  - `ex02_accuracy.py`: Calculates Accuracy score.
  - `ex03_regression.py`: Evaluates Linear Regression on California Housing data (R2, MSE, MAE).
  - `ex04_classification.py`: Evaluates Logistic Regression on Breast Cancer data (F1, Accuracy, Precision, Recall, ROC AUC) and plots ROC curve.
  - `ex05_models.py`: Compares Linear Regression, SVM, Decision Tree, Random Forest, and Gradient Boosting models.
  - `ex06_grid_search.py`: Performs Grid Search hyperparameter tuning on a Random Forest Regressor.
  - `ML_Model_Evaluation.ipynb`: Jupyter Notebook containing all exercises in an interactive format.
  - `roc_curve.png`: Generated ROC curve plot from Exercise 4.

## Prerequisites

- Python 3.9+
- Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `jupyter`

You can verify your environment by running:
```bash
python exercises/check_env.py
```

## How to Run

### Run Individual Scripts
You can run each exercise script directly from the terminal:

```bash
# Exercise 1: MSE
python exercises/ex01_mse.py

# Exercise 2: Accuracy
python exercises/ex02_accuracy.py

# Exercise 3: Regression Metrics
python exercises/ex03_regression.py

# Exercise 4: Classification Metrics
python exercises/ex04_classification.py

# Exercise 5: Model Comparison
python exercises/ex05_models.py

# Exercise 6: Grid Search
python exercises/ex06_grid_search.py
```

### Run Jupyter Notebook
To explore the exercises interactively:

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `exercises/ML_Model_Evaluation.ipynb`.
3. Run the cells to execute the code.

## Key Concepts Covered

- **Regression Metrics**: R2 Score, Mean Squared Error (MSE), Mean Absolute Error (MAE).
- **Classification Metrics**: Accuracy, Precision, Recall, F1 Score, ROC AUC, Confusion Matrix.
- **Model Comparison**: Evaluating Linear Regression, SVM, Decision Trees, Random Forests, and Gradient Boosting.
- **Hyperparameter Tuning**: Using `GridSearchCV` to optimize model parameters.
