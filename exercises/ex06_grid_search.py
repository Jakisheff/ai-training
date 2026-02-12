import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# data
housing = fetch_california_housing()
X, y = housing['data'], housing['target']

# Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=43, n_jobs=-1))
])

# Grid Search Parameters
param_grid = {
    'rf__n_estimators': [10, 50, 75],
    'rf__max_depth': [3, 5, 7],
    'rf__min_samples_leaf': [10, 20, 30]
}

# Grid Search
# cv set to use the provided indices logic or standard cv=3 if not strictly required to recreate exact indices
# The prompt mentions specific indices logic, but simplest interpretation for broad Grid Search first:
# "cv parameter to [(np.arange(18576), np.arange(18576,20640))]"
cv_indices = [(np.arange(18576), np.arange(18576, 20640))]

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv_indices,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print("Starting Grid Search...")
grid_search.fit(X, y)
print("Grid Search Finished.")

def select_model_verbose(gs):
    best_score = gs.best_score_
    best_params = gs.best_params_
    trained_model = gs.best_estimator_
    return trained_model, best_params, best_score

best_model, best_params, best_score = select_model_verbose(grid_search)

print(f"Best Params: {best_params}")
print(f"Best Score (Negative MSE): {best_score:.4f}")

# Predict on new point
new_point = np.array([[3.2031, 52., 5.47761194, 1.07960199, 910., 2.26368159, 37.85, -122.26]])
prediction = best_model.predict(new_point)
print(f"Prediction for new point: {prediction[0]:.4f}")
