#K-means clusturing
from sklearn.cluster import KMeans
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [10], [11], [12]])

# Create a KMeans model
model = KMeans(n_clusters=2)
model.fit(X)

# Predict clusters
clusters = model.predict(X)
print("Clusters:", clusters)
