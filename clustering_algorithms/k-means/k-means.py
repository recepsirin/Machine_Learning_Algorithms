import pandas as pd

data_set = pd.read_csv('customers.csv')

X = data_set.iloc[:, 1:].values  # picking age, consumption rate and salary columns and making assignment to X

