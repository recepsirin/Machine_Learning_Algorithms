import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

main_data = pd.read_csv('personal_information.csv')

age = main_data.iloc[:, 1:4].values
country = main_data.iloc[:, 0:1].values

# encoding Categorical(Nominal, Ordinal) -> Numerical

# we have three different country.if we had enumerated countries as 0,1 and 2
# this numerical data type might have brought into a different state and that could have mislead our algorithm.

# ohe -> creates a binary column for each category and returns a sparse matrix or dense array.
ohe = OneHotEncoder(categories='auto')
country = ohe.fit_transform(country).toarray()

genders_with_letters = main_data.iloc[:, -1:].values  # genders_with_letters that means Female and Male
gender_ohe = ohe.fit_transform(genders_with_letters).toarray()  # it means 0, 1, 0 || 1, 0, 0 || 0, 0, 1

# Combining all columns into a DataFrame
# Generating DataFrame

gender = main_data.iloc[:, -1].values  # picking gender column from main data

df_country = pd.DataFrame(data=country, index=range(22), columns=['fr', 'tr', 'us'])
df_height_weight_age = pd.DataFrame(data=age, index=range(22), columns=['height', 'weight', 'age'])
df_gender = pd.DataFrame(data=gender_ohe[:, :1], index=range(22), columns=['gender'])

# concatenation with concat function
df_all_except_gender = pd.concat([df_country, df_height_weight_age],
                                 axis=1)  # axis -> 0 is index based, 1 is column based
df_all = pd.concat([df_all_except_gender, df_gender], axis=1)

height = df_all.iloc[:, 3:4].values  # data manipulation for height column

left_side = df_all.iloc[:, :3]
right_side = df_all.iloc[:, 4:]

# our data variable and we'll try to predict height values by using columns which including in data variable

data = pd.concat([left_side, right_side], axis=1)

# split data as train and test

x_train, x_test, y_train, y_test = train_test_split(data, height, test_size=0.33, random_state=0)

# Model Building

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

for i in range(8):
    print('original values -> ', y_test[i], 'predicted values ->', y_pred[i])

# it's optional, you can uncomment it out to see and examine which variable could not contribute to our model
# i'll give more information about backward elimination, forward selection and etc in the pre-processing section
"""
import numpy as np
import statsmodels.api as sm

X = np.append(arr=np.ones((22, 1)).astype(int), values=data, axis=1)
X_l = data.iloc[:, [0, 1, 2, 3, 4]].values
r_ols = sm.OLS(endog=height, exog=X_l).fit()
print(r_ols.summary())
"""
