from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

main_data = pd.read_csv('staff_info.csv')

# picking columns as variable
edu_level = main_data.iloc[:, 1:2]  # independent variable
salary = main_data.iloc[:, 2:]  # dependent variable

# Model Building
dtr = DecisionTreeRegressor(random_state=None)
dtr.fit(edu_level, salary)

z = edu_level + 0.5
k = edu_level - 0.49

# Visualization
plt.scatter(edu_level, salary)
plt.plot(edu_level, dtr.predict(salary))

plt.plot(edu_level, dtr.predict(z))
plt.plot(edu_level, dtr.predict(k))

plt.show()

# Making Prediction
# The biggest problem of Decision Tree is memorizing values which means overfitting.
# Decision Tree memorizes values and tries to predict based on same values. It's making a kind of Quantization

print('Education Level -> 3.6  && Monthly Salary -> ', dtr.predict(np.array(3.6).reshape(1, 1)))
print('Education Level -> 7.5  && Monthly Salary -> ', dtr.predict(np.array(7.5).reshape(1, 1)))

# In the below prediction sample, if our max value is 50000 based on edu level is 10 ->
# which value we give has one condition is if the education edu_level greater than 10, it will always return 50000.
# it'll return same value which's our max value because the closest salary is depend on given edu_level

print('Education Level -> 11  && Monthly Salary -> ', dtr.predict(np.array(11).reshape(1, 1)))
print('Education Level -> 14  && Monthly Salary -> ', dtr.predict(np.array(14).reshape(1, 1)))
print('Education Level -> 47  && Monthly Salary -> ', dtr.predict(np.array(47).reshape(1, 1)))
