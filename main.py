import pandas as pd
import numpy as np 

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("income.csv")
df = df[["education", "age", "sex", "hours.per.week", "income"]]
tertiaryEducation = set(df["education"].unique()) - {"HS-grad", "7th-8th", "10th", "1st-4th", "5th-6th",
                        "12th", "9th", "Preschool", "11th"} # {All degrees} ∩ ¬{Tertiary degrees}
                        
"""1 (AKA True) corresponds to conditions we expect to yield a higher income. 0 is the opposite."""
df["sex"] = df["sex"].apply(lambda x: 1 if x == "Male" else 0) # 0 corresponds to Female. 0 to Male.
df["age"] = df["age"].apply(lambda x: 1 if int(x) >= 40 else 0) # Median age in U.S ~ 40. 0 corresponds to equal or below. 1 is above
df["education"] = df["education"].apply(lambda x: 1 if x in tertiaryEducation else 0) # 1 if tertiary education 0 otherwise.
df["hours.per.week"] = df["hours.per.week"].apply(lambda x: 1 if x >= 40 else 0) # 1 if work hours >= 40 otherwise 0
df["income"] = df["income"].replace({">50K" : 1, "<=50K" : 0}) # 0 corresponds to below 50k. 1 to above
df = df.astype({"sex" : bool, "age" : bool, "education" : bool, "hours.per.week" : bool, "income" : bool})

inputFeatures = df.loc[:, :"hours.per.week"] # splice columns
target = df["income"]
input_train, input_test, target_train, target_test = train_test_split(inputFeatures, target, test_size=0.2, random_state=42)

classifier = GradientBoostingClassifier(max_depth = 2, random_state=42)
loss = ["deviance", "exponential"]
n_estimators = [300, 200, 100, 50]
subsample = [1, 0.9, 0.8, 0.7]
criterion = ["friedman_mse", "squared_error", "mse", "mae"]
learning_rate =[0.5, 0.25, 0.1, 0.05, 0.01]


grid = dict(loss=loss, n_estimators=n_estimators, subsample=subsample, criterion=criterion, learning_rate=learning_rate)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
grid_search = GridSearchCV(estimator=classifier, param_grid=grid, n_jobs=-1, cv=cv, scoring="accuracy",error_score=0)
grid_result = grid_search.fit(input_train, target_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
for mean, stdev, param in zip(means, stds, params):
    print(f"{mean} {stdev} with: {param}")






