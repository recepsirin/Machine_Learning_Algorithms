from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

main_data = pd.read_csv('staff_info.csv')

# picking columns as variable
edu_level = main_data.iloc[:, 1:2].values  # independent variable
salary = main_data.iloc[:, 2:].values  # dependent variable

# Model Building
# There is a finding in random forest algorithms that data increases and success decreases.
rfr = RandomForestRegressor(n_estimators=10, random_state=0)
# random forest algorithm draws 10 different decision tree with n_estimators
rfr.fit(edu_level, salary.ravel())

# Visualization
plt.scatter(edu_level, salary, color='red')
plt.plot(edu_level, rfr.predict(edu_level), color='blue')
plt.plot(edu_level, rfr.predict(edu_level + 0.5), color='green')  # 0.5 more
plt.plot(edu_level, rfr.predict(edu_level - 0.5), color='yellow')  # 0.5 less

plt.show()

# Making Prediction
print('Education Level -> 3.6  && Monthly Salary -> ', rfr.predict(np.array(3.6).reshape(1, 1)))
print('Education Level -> 7.5  && Monthly Salary -> ', rfr.predict(np.array(7.5).reshape(1, 1)))
print('Education Level -> 11  && Monthly Salary -> ', rfr.predict(np.array(11).reshape(1, 1)))
