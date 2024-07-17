#linear regression

from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 5, 8])

# Create a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict
X_new = np.array([[5]])
y_pred = model.predict(X_new)
print("Prediction:", y_pred)
