# module2-mlr-sklearn

## Especificaciones

* Programa uno de los algoritmos vistos en el módulo (o que tu profesor de módulo autorice) haciendo uso de una biblioteca o framework de aprendizaje máquina. Lo que se busca es que demuestres tu conocimiento sobre el framework y como configurar el algoritmo. 
* Prueba tu implementación con un set de datos y realiza algunas predicciones. Las predicciones las puedes correr en consola o las puedes implementar con una interfaz gráfica apoyándote en los visto en otros módulos.

## Librerías utilizadas

* **Pandas** para el manejo del dataset.
* **Scikit-learn** para la instanciación de modelos y separación de datos de entrenamiento y pruebas.
* **Matplotlib** para la visualización gráfica.

## Dataset utilizado

**Nombre:** House price prediction

**Kaggle URL:** https://www.kaggle.com/datasets/shree1992/housedata?select=data.csv

## Métrica de desempeño

Para medir el desempeño del modelo se utilizó el coeficiente de determinación $R^2$ sobre el subset de prueba. El resultado fue de **0.464**, lo cuál es significativamente bueno tomando en cuenta su complejidad y que es un modelo de regresión lineal.

## Predicciones de prueba

A continuación se muestran algunas entradas, valores esperados y valores obtenidos del modelo.

bedrooms | bathrooms | sqft_living | sqft_lot | floors | waterfront | view | condition | sqft_above | sqft_basement | yr_built | yr_renovated | expected_price | obtained_price
---------|-----------|-------------|----------|--------|------------|------|-----------|------------|---------------|----------|--------------|----------------|---------------
4.0      | 2.75 |	2310 | 5000 | 2.0 | 0 | 0 | 3 | 2310 | 0 | 2006 | 0 | 309950.0 | 598045.2
2.0      | 2.00 |	1460 | 9052 | 1.0 | 0 | 2 | 5 | 1460 | 0 | 1900 | 0 | 1010000.0 | 383834.1
3.0	| 1.75 |	1500 |	7200 |	1.0 |	0 |	0 |	3 |	1500 |	0 |	1957 |	2000 | 360000.0 | 420501.5
3.0	| 2.00 |	1670 |	7757 |	1.0 |	0 |	0 |	3 |	1670 |	0 |	1992 |	0 | 289950.0 | 437838.6
4.0	| 2.50 |	4620 |	20793 |	2.0 |	0 |	0 |	4 |	4620 |	0 |	1991 |	0 | 575000.0 | 1161926

## Nombre del archivo a revisar

main.py
