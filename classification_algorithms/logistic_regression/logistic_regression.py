from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd

main_data = pd.read_csv('bmi_info.csv')

x = main_data.iloc[:, 1:3].values  # independent variable
y = main_data.iloc[:, 3:].values  # dependent variable

# split data as train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Scaling
sc = StandardScaler()

X_train = sc.fit_transform(x_train)  # it means that simply first learns X_train after then transforms it
X_test = sc.transform(
    x_test)  # it means that does not relearn for x_test, just uses the transformation method

# Model Building

lr = LogisticRegression(random_state=0, solver='lbfgs')  # lbfgs -> 0.22 as default
lr.fit(X_train, y_train.ravel())  # x -> y

y_pred = lr.predict(X_test)  # making prediction

# Outputting on the console to examine our prediction
for i in range(8):
    print('predicted value -> ', y_pred[i], 'original value -> ', y_test[i])

# Evaluating with Confusion Matrix

# Confusion Matrix describes the performance of our classification model

# a -> the count of that we predict female and the result is female
# b -> the count of that we predict female and the result is not female
# c -> the count of that we predict male and the result is female)
# d -> the count of that we predict male and the result is male)

# [True Positive(read the above definition 'a' ), False Negative(read the above definition 'b']
# [False Positive(read the above definition 'c'), True Negative(read the above definition 'd')]
cm = confusion_matrix(y_pred, y_test)  # y_pred -> y_test
print(cm)
