import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("Shopping.csv")
#names = df["Name"]
df = df[["Age", "Gender", "Annual Income", "Spending Score"]]
df["Gender"] = df["Gender"].map({'Male' : 1, 'Female' : 0})
X = df.copy().to_numpy() * 1.0
std = np.std(X, axis=0)
av = np.average(X, axis=0)
X -= av
X /= std

# s, e = 1, 40
# inertia = np.zeros(e - s + 1)
# for i in range(s, e + 1):
#     print(i)
#     kmeans = KMeans(n_clusters=i, n_init=30).fit(X)
#     inertia[i - s] = kmeans.inertia_

kmeans = KMeans(n_clusters=20, n_init=30).fit(X)
centroids = kmeans.cluster_centers_ * std + av
cent = pd.DataFrame(centroids, columns=["Age", "Gender", "Annual Income", "Spending Score"]).round(2)
print(cent.to_string(index=False))

# plt.plot(np.arange(s, e + 1), inertia)
# plt.show()