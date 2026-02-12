from sklearn.metrics import mean_squared_error

y_true = [91, 51, 2.5, 2, -5]
y_pred = [90, 48, 2, 2, -4]

mse = mean_squared_error(y_true, y_pred)
print("Mean Squared Error:", mse)
