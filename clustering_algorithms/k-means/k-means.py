import pandas as pd
from sklearn.cluster import KMeans

data_set = pd.read_csv('customers.csv')

X = data_set.iloc[:, 1:].values  # picking age, consumption rate and salary columns and making assignment to X

# Model Building
# n_clusters: The number of clusters to form as well as the number of centroids to generate.
# init : ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence

k_means = KMeans(n_clusters=3, init='k-means++')
k_means.fit(X)

# Outputting with K-Means attributes
print('cluster_centers_ \n', k_means.cluster_centers_)  # Coordinates of cluster centers.

# Sum of squared distances of samples to their closest cluster center.
print('inertia_ i.e. WCSS : ', k_means.inertia_)

print('Labels of each point  \n', k_means.labels_)
print('Number of iterations run : ', k_means.n_iter_)
