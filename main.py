"""
i dati devono essere presenti in una cartella "data", con nome del file "diamonds.csv"
link dataset: https://www.kaggle.com/datasets/shivam2503/diamonds
i grafici sono presenti nel file "main.ipynb"
"""

import pandas as pd

# import seaborn as sns // grafici nel notebook (ipynb)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Data Gathering
filepath = "data/diamonds.csv"
data = pd.read_csv(filepath)
print("Prime 5 righe:\n", data.head())
print("Shape:", data.shape)
print("Tipi colonne:\n", data.dtypes)

print("_" * 50, "\n")

# Data Modeling
# Data Modeling di dati categorici
# Cut
print("Valori taglio:\n", data.cut.unique())
cut_values = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
data.replace({"cut": cut_values}, inplace=True)

# Color
print("Valori colore:\n", data.color.unique())
color_values = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
data.replace({"color": color_values}, inplace=True)

# Clarity
print("Valori chiarezza:\n", data.clarity.unique())
clarity_values = {
    "I1": 0,
    "SI2": 1,
    "SI1": 2,
    "VS2": 3,
    "VS1": 4,
    "VVS2": 5,
    "VVS1": 6,
    "IF": 7,
}
data.replace({"clarity": clarity_values}, inplace=True)

print("_" * 50, "\n")

# dati numerici
print("Dati:\n", data.head())

# Correlazioni
print("Correlazioni:\n", data.corr())

correlazione_prezzo = data.corr()["price"].abs().sort_values(ascending=False)
print(
    "Correlazioni migliori con il prezzo:\n",
    correlazione_prezzo[correlazione_prezzo > 0.5],
)

print("_" * 50, "\n")

# Suddivisione dati x, y (variabile target: price , "prezzo")
print("Suddivisione dati in Features per il training e Target")
dataX = data.loc[
    :, ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
]
print("Features per il training:", dataX.columns.values)
datay = data.loc[:, "price"]
print("Target:", datay.name)

# Suddivisione training e test (80% training, 20% test)
print("Suddivisione training e test (80% training, 20% test)")
X_train, X_test, y_train, y_test = train_test_split(dataX, datay, test_size=0.2)

# Standardizzazione dei dati
print("Standardizzazione dei dati con un StandardScaler")
st_scaler = StandardScaler()
X_train = st_scaler.fit_transform(X_train)
X_test = st_scaler.transform(X_test)

print("_" * 50, "\n")

# Model Selection
# Decision tree regressor:
print("Decision Tree Regressor")
decision_tree_regressor = DecisionTreeRegressor()
print("Training")
decision_tree_regressor.fit(X_train, y_train)

pred_decision_tree_regressor = decision_tree_regressor.predict(X_test)
error_decision_tree = pred_decision_tree_regressor - y_test
mae = mean_absolute_error(y_test, pred_decision_tree_regressor)
mse = mean_squared_error(y_test, pred_decision_tree_regressor)
r2 = r2_score(y_test, pred_decision_tree_regressor)

score_decision_tree = {
    "mean_squared_error": mse,
    "mean_absolute_error": mae,
    "r2_score": r2,
}
print("Metriche:\n", score_decision_tree)

print("_-" * 25, "\n")

# Random forest regressor
print("Random Forest Regressor")
random_forest_regressor = RandomForestRegressor()
print("Training")
random_forest_regressor.fit(X_train, y_train)

pred_random_forest_regressor = random_forest_regressor.predict(X_test)
error_random_forest = pred_random_forest_regressor - y_test
mae = mean_absolute_error(y_test, pred_random_forest_regressor)
mse = mean_squared_error(y_test, pred_random_forest_regressor)
r2 = r2_score(y_test, pred_random_forest_regressor)

score_random_forest = {
    "mean_squared_error": mse,
    "mean_absolute_error": mae,
    "r2_score": r2,
}
print("Metriche:\n", score_random_forest)

print("_-" * 25, "\n")

# Linear Regressor
print("Linear Regressor")
linear_regression = LinearRegression()
print("Training")
linear_regression.fit(X_train, y_train)

pred_linear_regression = linear_regression.predict(X_test)
error_linear_regression = pred_linear_regression - y_test
mae = mean_absolute_error(y_test, pred_linear_regression)
mse = mean_squared_error(y_test, pred_linear_regression)
r2 = r2_score(y_test, pred_linear_regression)

score_linear_regression = {
    "mean_squared_error": mse,
    "mean_absolute_error": mae,
    "r2_score": r2,
}
print("Metriche:\n", score_linear_regression)

# Vedendo i risultati, random forest e' migliore, lo miglioro ancora di piu'

# Uso una grid search per trovare i migliori iperparametri per la random forest

print("Grid search per migliorare la random forest")
grid_search_params_for_random_forest_regressor = {
    "min_samples_split": [11, 12, 13],
    "min_samples_leaf": [1, 2, 3],
}

random_forest_regressor = RandomForestRegressor()
grid_search = GridSearchCV(
    random_forest_regressor,
    param_grid=grid_search_params_for_random_forest_regressor,
    verbose=2,
)
# Alleno la grid search
print(
    "Training grid search con iperparametri:\n",
    grid_search_params_for_random_forest_regressor,
)
grid_search.fit(X_train, y_train)

print("Migliori parametri:\n", grid_search.best_estimator_)
# i parametri migliori usciti sono min_samples_split => 12 e min_samples_lead =>2

print("_-" * 25, "\n")

# Random forest regressor con iperparametri
print("Random Forest Regressor con iperparametri")
random_forest_regressor_improved = RandomForestRegressor(
    min_samples_split=12, min_samples_leaf=2
)
print("Training")
random_forest_regressor_improved.fit(X_train, y_train)

pred_random_forest_regressor_improved = random_forest_regressor_improved.predict(X_test)
error_random_forest_improved = pred_random_forest_regressor_improved - y_test
mae = mean_absolute_error(y_test, pred_random_forest_regressor_improved)
mse = mean_squared_error(y_test, pred_random_forest_regressor_improved)
r2 = r2_score(y_test, pred_random_forest_regressor_improved)

score_random_forest_improved = {
    "mean_squared_error": mse,
    "mean_absolute_error": mae,
    "r2_score": r2,
}
print("Metriche:\n", score_random_forest_improved)
