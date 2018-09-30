import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from clean import clean_data
from train import get_scores

df = pd.read_csv('avocado.csv')

df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)

df_train = clean_data(df_train)

avoc_labels = df_train['AveragePrice']
df_train.drop(['AveragePrice'], axis = 1, inplace = True)

lin_reg = LinearRegression()
lin_scores = get_scores(lin_reg, df_train, avoc_labels)
print(lin_scores.mean())

tree_reg = DecisionTreeRegressor()
tree_scores = get_scores(tree_reg, df_train, avoc_labels)
print(tree_scores.mean())

rf_reg = RandomForestRegressor()
rf_scores = get_scores(rf_reg, df_train, avoc_labels)
print(rf_scores.mean())

df_test = clean_data(df_test)

df_test_labels = df_test['AveragePrice']
df_test.drop(['AveragePrice'], axis = 1, inplace = True)

# Best rmse for Random Forest, so evaluate it on test set
rf_reg.fit(df_train, avoc_labels)
pred_labels = rf_reg.predict(df_test)
test_mse = mean_squared_error(df_test_labels, pred_labels)
test_rmse = np.sqrt(test_mse)
print(test_rmse)
