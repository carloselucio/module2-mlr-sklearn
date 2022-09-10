import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.model_selection import train_test_split

# Autor: Carlos Enrique Lucio Domínguez | A00828524
# Objetivo: Programación de un modelo de regresión lineal múltiple utilizando la técnica de Gradiente Descendiente con el
# uso de un framework como scikit-learn.
# Problema: Encontrar un modelo de predicción para conocer el precio de un hogar a partir de características como número
# de recámaras, número de baños, tamaño del terreno total y de construcción, número de plantas/pisos, calificación de la
# vista y calificación de en qué condiciones se encuentra el hogar.
# Dataset Source: https://www.kaggle.com/datasets/shree1992/housedata?select=data.csv

df = pd.read_csv('data.csv') # Lectura del archivo de datos encontrado en el mismo folder que el archivo .py.

X= df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4, train_size=0.7) # Separación de datos de entrenamiento (70%) y pruebas (30%)

lr_gd = SGDRegressor(eta0=0.00000000001, max_iter=10000, shuffle=False) # Modelo de Gradiente Descendiente donde eta0 es el learning rate.
lr_gd.fit(X_train, y_train) # Entrenamiento del modelo

print("\nMétrica de desempeño del modelo:\nCoeficiente de determinación R^2 para los datos de prueba:", lr_gd.score(X_test, y_test))

y_pred = lr_gd.predict(X_test)

# Comparación entre valores reales y predicciones
comparison = pd.DataFrame(data={'Predicted Price': y_pred, 'Real Price': y_test})
print('\nComparación de valores reales y predicciones:')
print(comparison.head())
print()

# Array de residuos.
residuals = y_pred-y_test

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Gráfica de valores predichos vs valores reales
axes[0].scatter(y_test, y_pred, edgecolors=(0, 0, 0), alpha = 0.4)
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
                'k--', color = 'black', lw=2)
axes[0].set_title('Valor Predicho vs Valor Real', fontsize = 10, fontweight = "bold")
axes[0].set_xlabel('Precio Real')
axes[0].set_ylabel('Precio Predicho')
axes[0].tick_params(labelsize = 6)

# Gráfica de residuos del modelo
axes[1].scatter(list(range(len(y_test))), residuals,
                   edgecolors=(0, 0, 0), alpha = 0.4)
axes[1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
axes[1].set_title('Residuos del Modelo', fontsize = 10, fontweight = "bold")
axes[1].set_xlabel('Id')
axes[1].set_ylabel('Residuo')
axes[1].tick_params(labelsize = 6)

plt.show()