
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.pipeline import Pipeline

os.chdir("C:\\Users\\brian\\Downloads")

df = pd.read_csv("mlb_teams.csv", index_col="TeamName")
df = df.drop(columns=["WAR", "W", "L"])  # drop wins above replacement wins and losses to prevent data leakage

X = df.drop(columns=["W-L%"])
y = df["W-L%"]


# you do not have to standardize for random forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_regressor.fit(X_train, y_train)
#
# y_pred = rf_regressor.predict(X_test)
# #
# # # metrics/prediction accuracy
# r2 = r2_score(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# print("r2 score: ", r2)
# print("mse: ", mse)

#
#
# pred_vs_actual_df = pd.DataFrame({"Actual W-L%": y_test, "Predicted W-L%": y_pred})
# print(pred_vs_actual_df)

pipeline = Pipeline([
    ("XGB", xgb.XGBRegressor())
])

param_grid = {"XGB__n_estimators": [11],
              "XGB__learning_rate": [0.30],
              "XGB__max_depth": [None],

}

cv = KFold(n_splits=2, random_state=42, shuffle=True) # have to switch it to kfold for regression
# cross validation, only splits twice (train and test split) so it doesnt take a lot of time
best_model = GridSearchCV(param_grid=param_grid, cv =cv, estimator=pipeline)

best_model.fit(X_train, y_train)

test_score = best_model.score(X_test, y_test)
print("Test Score: ", test_score)

y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("r2 score: ", r2)
print("mse: ", mse)



pred_vs_actual_df = pd.DataFrame({"Actual W-L%": y_test, "Predicted W-L%": y_pred})
print(pred_vs_actual_df)

print(best_model.best_params_)