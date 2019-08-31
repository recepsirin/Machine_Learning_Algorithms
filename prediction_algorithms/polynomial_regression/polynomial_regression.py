from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

main_data = pd.read_csv('staff_info.csv')

# picking columns
education_level = main_data.iloc[:, 1:2]
salary = main_data.iloc[:, 2:]

# Model Building
pr = PolynomialFeatures(degree=4)  # it draws a fourth degree polynomial regression. it's second degree as default
el_poly = pr.fit_transform(education_level)
lr2 = LinearRegression()
lr2.fit(el_poly, salary)  # it increases exponentially

# Visualization
plt.scatter(education_level, salary)
plt.plot(education_level, lr2.predict(pr.fit_transform(education_level)))

plt.title('Monthly Salaries by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Monthly Salary')

plt.show()

# Making Prediction
# Although we don't have some education level below, our model will predict their monthly salary
print('Education Level -> 3.7  && Monthly Salary -> ', lr2.predict(pr.fit_transform(np.array(3.7).reshape(1, 1))))
print('Education Level -> 7.2  && Monthly Salary -> ', lr2.predict(pr.fit_transform(np.array(7.2).reshape(1, 1))))
print('Education Level -> 11  && Monthly Salary -> ', lr2.predict(pr.fit_transform(np.array(11).reshape(1, 1))))
print('Education Level -> 14  && Monthly Salary -> ', lr2.predict(pr.fit_transform(np.array(14).reshape(1, 1))))
