# T4 modelamiento

# T4 – Modelamiento

## Objetivo
En este módulo desarrollamos y comparamos distintos modelos de regresión para predecir el **tiempo de resolución de incidentes (TTR)**, a partir de los datos previamente procesados en T3.  
El propósito fue establecer un modelo baseline y luego probar un modelo más flexible, evaluando su desempeño mediante validación cruzada y un conjunto de test.

---

## Datos utilizados
- Dataset: `T3_preparacion/incidents_clean_for_model.csv`
- Registros originales: **24,918**
- Registros válidos tras limpieza del target: **23,362**
- Variables predictoras finales: **6** (3 numéricas, 3 categóricas)
- Target: `ttr_h_winsor` (winsorizado para reducir impacto de outliers)

---

## Modelos entrenados
1. **Baseline – Ridge Regression**  
   Modelo lineal regularizado usado como referencia inicial.

2. **RandomForestRegressor**  
   Modelo de ensamble basado en árboles de decisión, capaz de capturar relaciones no lineales y efectos de interacción entre variables.

---

## Resultados
Los modelos fueron evaluados mediante validación cruzada de 5 folds en train y posteriormente en un set de test (20% de los datos).  
Las métricas consideradas fueron: **MAE**, **RMSE** y **R²**.

=== Comparación de modelos ===
               MAE     RMSE     R2  CV_MAE_mean  CV_MAE_std
Ridge         181.388  361.063  0.234      181.814       3.580
RandomForest  174.376  356.594  0.253      175.885       3.077

---

## Conclusión
El modelo **RandomForest** logró un desempeño ligeramente superior al baseline de Ridge, reduciendo el MAE y el RMSE y alcanzando un R² de 0.25 frente a 0.23 del baseline.  
Aunque el poder explicativo global es limitado (lo que refleja la complejidad y el ruido en el target TTR), consideramos que **RandomForest es el modelo a seleccionar para despliegue** en etapas posteriores (T7).

---

## Artefactos generados
En la carpeta "T4_modelamiento/artifacts/" quedaron guardados los modelos y métricas:

- "model_ridge.joblib" / "metrics_ridge.json" / "schema_ridge.json"
- "model_random_forest.joblib" / "metrics_random_forest.json` / "schema_random_forest.json"