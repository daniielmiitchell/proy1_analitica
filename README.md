# Proyecto 1 – Analítica Computacional

**Objetivo:** preparar, explorar y modelar datos de incidentes para apoyar decisiones y construir un tablero en Dash.

## Estructura
- `T1_negocio/` – contexto y métricas
- `T2_exploracion/` – EDA (`EDA.ipynb`, figuras en `figs/`)
- `T3_preparacion/` – limpieza (`T2yT3.ipynb`) → `incidents_clean_for_model.csv`
- `T4_modelamiento/` – modelos y evaluación
- `T5_tablero/` – app Dash
- `T6_evaluacion/` – resultados finales
- `T7_despliegue/` – instrucciones/archivos para EC2
- `data/` – crudo local **(ignorado)** + `README.md`

## Reproducir
1. Coloca `data/incident_event_log.csv` (no versionado).
2. Ejecuta `T3_preparacion/T2yT3.ipynb` → genera `T3_preparacion/incidents_clean_for_model.csv`.
3. Ejecuta `T2_exploracion/EDA.ipynb` para la EDA.

## Requisitos
