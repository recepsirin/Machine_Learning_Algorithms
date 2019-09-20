import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

data_set = pd.read_csv('customers.csv')

X = data_set.iloc[:, 1:].values  # picking age, consumption rate and salary columns and making assignment to X

# Model Building

ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_pred = ac.fit_predict(X)
# print(y_pred)

