from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

main_data = pd.read_csv('bmi_info.csv')

x = main_data.iloc[:, 1:3].values  # independent variable
y = main_data.iloc[:, 3:].values  # dependent variable

# Split data as train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Scaling
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Model Building

# Information gain is the main key of Decision Tree Algorithms that builds it the actual Decision Tree
# by inspecting the qualification of data to pick it up which of the starting variable

dt = DecisionTreeClassifier(criterion='entropy')  # set criterion as entropy for information gain.

# criterion is the function to measure the quality of a split.

# max_depth is the maximum depth of the tree. If None, then nodes are expanded until all leaves are pure
# or until all leaves contain less than min_samples_split samples.

dt.fit(X_train, y_train.ravel())

y_pred = dt.predict(X_test)  # making prediction as always

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
