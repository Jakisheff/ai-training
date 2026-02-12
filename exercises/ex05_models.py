from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# data
housing = fetch_california_housing()
X, y = housing['data'], housing['target']

# split data train test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    shuffle=True,
                                                    random_state=43)

# Models
models = [
    ("Linear Regression", LinearRegression()),
    ("SVM", SVR()),
    ("Decision Tree", DecisionTreeRegressor(random_state=43)),
    ("Random Forest", RandomForestRegressor(random_state=43)),
    ("Gradient Boosting", GradientBoostingRegressor(random_state=43))
]

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    # pipeline
    pipeline = [('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', model)]
    pipe = Pipeline(pipeline)
    
    # fit
    pipe.fit(X_train, y_train)
    
    # Predict
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)
    
    print(f"--- {name} ---")
    print(f"Train R2: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Train MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"Test R2: {r2_score(y_test, y_test_pred):.4f}")
    print(f"Test MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
    print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
    print()

for name, model in models:
    evaluate_model(name, model, X_train, X_test, y_train, y_test)
