from sklearn.metrics import accuracy_score

y_pred = [0, 1, 0, 1, 0, 1, 0]
y_true = [0, 0, 1, 1, 1, 1, 0]

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
