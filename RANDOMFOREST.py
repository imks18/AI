#random forest
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 1, 1])

# Create a random forest classifier
model = RandomForestClassifier()
model.fit(X, y)

# Predict
X_new = np.array([[5]])
y_pred = model.predict(X_new)
print("Prediction:", y_pred)
