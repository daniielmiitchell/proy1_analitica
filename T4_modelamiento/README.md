# T4 – Modelamiento

## Objetivo
En este módulo desarrollamos y comparamos distintos modelos de regresión para predecir el **tiempo de resolución de incidentes (TTR)**, a partir de los datos previamente procesados en T3.  
El propósito fue establecer un modelo baseline y luego probar un modelo más flexible, evaluando su desempeño mediante validación cruzada y un conjunto de test.

---

## Datos utilizados
- Dataset: `T3_preparacion/incidents_clean_for_model.csv`
- Registros originales: **24,918**
- Registros eliminados por target inválido: **1,556**
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

### Comparación de modelos

| Modelo        | MAE ↓   | RMSE ↓  | R² ↑   | CV_MAE_mean | CV_MAE_std |
|---------------|---------|---------|--------|-------------|------------|
| Ridge         | 176.40  | 354.46  | 0.262  | 178.23      | 3.07       |
| RandomForest  | 174.38  | 356.59  | 0.253  | 175.89      | 3.08       |

---

## Conclusión
El modelo **RandomForest** logra un MAE ligeramente mejor que el baseline de Ridge (~2 horas menos de error promedio), aunque empeora en RMSE y R².  
Esto sugiere que, con las variables actuales, el problema es difícil de explicar y que **el modelo lineal (Ridge) es competitivo frente al RandomForest**.  

Para futuras iteraciones se recomienda:
- Explorar nuevas variables y características derivadas.  
- Probar algoritmos adicionales (Gradient Boosting, XGBoost, LightGBM).  
- Ajustar hiperparámetros para mejorar estabilidad en presencia de outliers.  

---

## Artefactos generados
En la carpeta `T4_modelamiento/artifacts/` quedaron guardados los modelos y métricas:

- `model_ridge.joblib` / `metrics_ridge.json` / `schema_ridge.json`
- `model_random_forest.joblib` / `metrics_random_forest.json` / `schema_random_forest.json`

---