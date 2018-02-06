#from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures
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

model = Perceptron(max_iter=5, tol=None)
model.fit(X, y)

X_new = [[355, 2, 1]]
print(model.predict(X_new))