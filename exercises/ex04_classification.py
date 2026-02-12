from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=43)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
classifier = LogisticRegression()
classifier.fit(X_train_scaled, y_train)

# Predict
y_train_pred = classifier.predict(X_train_scaled)
y_test_pred = classifier.predict(X_test_scaled)
y_test_probs = classifier.predict_proba(X_test_scaled)[:, 1] # Probabilities for positive class

print("--- 10 first values Train ---")
print(y_train_pred[:10])
print("\n--- 10 first values Test ---")
print(y_test_pred[:10])
print()

def print_metrics(y_true, y_pred, y_probs=None, set_name="Test"):
    print(f"--- {set_name} Metrics ---")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    if y_probs is not None:
        print(f"ROC AUC: {roc_auc_score(y_true, y_probs):.4f}")
    print()

print_metrics(y_train, y_train_pred, None, "Train")
print_metrics(y_test, y_test_pred, y_test_probs, "Test")

print("--- Confusion Matrix (Test) ---")
print(confusion_matrix(y_test, y_test_pred))

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.savefig('exercises/roc_curve.png')
print("\nROC curve saved to exercises/roc_curve.png")
