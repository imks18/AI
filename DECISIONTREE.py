#Decision tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 1, 1])

# Create a decision tree classifier
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict
X_new = np.array([[5]])
y_pred = model.predict(X_new)
print("Prediction:", y_pred)
