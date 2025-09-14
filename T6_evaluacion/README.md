# T6 – Evaluación de Modelos (Regresión)

## Objetivo
En esta etapa realizamos la **evaluación robusta del modelo RandomForest** utilizando validación cruzada Out-Of-Fold (OOF).  
El objetivo es estimar de manera consistente el desempeño del modelo, evitar fugas de información (data leakage) y analizar estabilidad por subgrupos clave como `priority_level`.

---

## Datos utilizados
- Dataset: `T3_preparacion/incidents_clean_for_model.csv`
- Registros originales: **24,918**
- Registros eliminados por target inválido: **1,556**
- Registros válidos tras limpieza del target: **23,362**
- Variables predictoras finales: filtradas para excluir:
  - Columnas con tokens peligrosos (`ttr`, `resolve`, `close`, etc.)
  - Columnas de altísima cardinalidad (ej. IDs, timestamps)

Target: `ttr_h_winsor` (winsorizado para reducir el impacto de outliers)

---

## Metodología
1. **Filtrado seguro de features** para prevenir fugas.  
2. **Validación cruzada (KFold, 5 folds, shuffle, SEED=42)** con RandomForestRegressor.  
3. Generación de métricas OOF globales y por `priority_level` (cuando aplica).  
4. Exportación de predicciones y métricas como artefactos reutilizables.  

---

## Cómo ejecutar
Desde la raíz del proyecto:

bash
python3 T6_evaluacion/evaluacion_regresion.py

##Resultados esperados (referencia)

En nuestra corrida obtuvimos resultados en el orden de:
	•	RandomForest (OOF)
	•	MAE ≈ 175–176
	•	RMSE ≈ 354–357
	•	R² ≈ 0.25

Por priority_level, se observan diferencias en error: para prioridades bajas el error es mayor, mientras que para prioridades medias/altas el ajuste es algo más estable.

⸻

Conclusión
	•	El RandomForest mantiene un MAE ligeramente mejor que el baseline (Ridge), aunque no logra mejorar en RMSE o R².
	•	El poder explicativo global es limitado (~25% de la variabilidad).
	•	La evaluación OOF confirma que, con las features actuales, el modelo tiene un techo de desempeño bajo.

Recomendaciones:
	•	Enriquecer las variables predictoras con ingeniería de características.
	•	Explorar modelos de boosting (XGBoost, LightGBM).
	•	Profundizar en ajuste de hiperparámetros para mejorar estabilidad.