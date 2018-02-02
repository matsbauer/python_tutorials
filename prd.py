from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("melb_data.csv")

df = df.fillna(df.mean())
predictors = ["Landsize", "Rooms", "Bathroom"]
X = df[predictors]
y = df.Price

model = KNeighborsRegressor(n_neighbors=2)
model.fit(X, y)

X_new = [[355, 2, 1],[355, 3, 1]]
print(model.predict(X_new))