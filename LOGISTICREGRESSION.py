#logistic regression
from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 1, 1])

# Create a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict
X_new = np.array([[5]])
y_pred = model.predict(X_new)
print("Prediction:", y_pred)
