from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt

main_data = pd.read_csv('staff_info.csv')

# picking columns as variable
edu_level = main_data.iloc[:, 1:2]  # independent variable
salary = main_data.iloc[:, 2:]  # dependent variable

# Feature Scaling
# The feature scaling process takes an important place in support vector regression
sc = StandardScaler()
el_scaled = sc.fit_transform(edu_level)
s_scaled = sc.fit_transform(salary)

# Model Building
svr = SVR(kernel='rbf')  # kernel can be also linear, poly, rbf, sigmoid and precomputed. we used radial basis function
svr.fit(el_scaled, s_scaled.ravel())

# Visualization
plt.scatter(el_scaled, s_scaled)
plt.plot(el_scaled, svr.predict(el_scaled))
plt.show()
