from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

main_data = pd.read_csv('bmi_info.csv')

x = main_data.iloc[:, 1:3].values  # independent variable
y = main_data.iloc[:, 3:].values  # dependent variable

# split data as train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Scaling
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Model Building

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')  # n_neighbors=5 by default
# the biggest secret to get the right result that we want it is using right parameters with appropriate data

# Distance Metrics -> Distance Function
# euclidean -> sqrt(sum((x - y)^2)),
# manhattan -> sum(|x - y|),
# minkowski  -> sum(|x - y|^p)^(1/p),
# mahalanobis -> sqrt((x - y)' V^-1 (x - y))

# visit the below url to get more information
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

knn.fit(X_train, y_train.ravel())  # x -> y

y_pred = knn.predict(X_test)  # making prediction

# Outputting on the console to examine our prediction
for i in range(8):
    print('predicted value -> ', y_pred[i], 'original value -> ', y_test[i])

# Evaluating with Confusion Matrix

# Definitions -> a, b, c, d

# a -> the count of that we predict female and the result is female
# b -> the count of that we predict female and the result is not female
# c -> the count of that we predict male and the result is female
# d -> the count of that we predict male and the result is male

# [True Positive('a' ), False Negative('b']
# [False Positive('c'), True Negative('d')]

cm = confusion_matrix(y_pred, y_test)
print(cm)
