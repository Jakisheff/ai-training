from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
                                                    random_state=13)
# pipeline
pipeline = [('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())]
pipe = Pipeline(pipeline)

# fit
pipe.fit(X_train, y_train)

# Predict
y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)

print("--- 10 first values Train ---")
print(y_train_pred[:10])
print("\n--- 10 first values Test ---")
print(y_test_pred[:10])
print()

def print_metrics(y_true, y_pred, set_name):
    print(f"--- {set_name} Metrics ---")
    print(f"R2: {r2_score(y_true, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print()

print_metrics(y_train, y_train_pred, "Train")
print_metrics(y_test, y_test_pred, "Test")
