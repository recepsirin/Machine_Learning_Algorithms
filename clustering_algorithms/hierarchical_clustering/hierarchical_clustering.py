import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

data_set = pd.read_csv('customers.csv')

X = data_set.iloc[:, 2:].values  # picking age, consumption rate and salary columns and making assignment to X

# Model Building

# n_clusters: The number of clusters to find.

# affinity: Metric used to compute the linkage. Our linkage is 'ward' so it only accepts 'euclidean'

# linkage: The linkage criterion determines which distance to use between sets of observation.
# The algorithm will merge the pairs of cluster that minimize this criterion.

ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')

y_pred = ac.fit_predict(X)  # making prediction

# Data Visualization

plt.title('Hierarchical Clustering')

plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s=100, c='purple')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s=100, c='blue')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s=100, c='yellow')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s=100, c='gray')

plt.show()
