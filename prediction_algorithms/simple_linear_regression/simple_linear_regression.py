import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

main_data = pd.read_csv('sales.csv')  # reading our sales.csv file and importing it as pandas data frame

months = main_data[['Months']]  # slicing months column to months variable
sales_amount = main_data[['Sales Amount']]  # slicing sales column to sales_amount variable

x_train, x_test, y_train, y_test = train_test_split(months, sales_amount, test_size=0.33, random_state=0)
# split 67% of the main data into train data and allocated the other of 33% part for testing.


# feature scaling -- applied the feature scaling approach with StandardScaler
# we use feature scaling for preventing overfitting case, improving accuracy values
# you can also search 'normalization approach' to apply your data for feature scaling
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

# model building
lr = LinearRegression()
lr.fit(x_train, y_train)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

# visualization
plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))  # making prediction on chart area

plt.title('Sales by Months')
plt.xlabel('Months')
plt.ylabel('Sales Amount')

plt.show()
