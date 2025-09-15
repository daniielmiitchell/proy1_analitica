# T7 — Despliegue en EC2 (CLI)

## 🎯 Objetivo
Desplegar el modelo de TTR en una instancia **AWS EC2** y ejecutar predicciones **desde la nube** usando `predict.py`.  
La evidencia final es la salida JSON de una predicción **single** y otra **batch** generadas en la EC2.

---

## 1) Requisitos
- Ubuntu 22.04 en EC2 (t3.micro o similar)
- Python 3.10+ con `venv`
- Artefactos disponibles:
  - Modelo: `T4_modelamiento/artifacts/model_random_forest.joblib` (o `model_ridge.joblib`)
  - Schema: `T4_modelamiento/artifacts/schema_random_forest.json` (o `schema_ridge.json`)
- Dependencias:
- pip install -r T7_despliegue/requirements.txt

(incluye `pandas`, `numpy`, `scikit-learn`, `joblib`)

---

## 2) Preparación en EC2 (una vez)
```bash
# Dentro del repo
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r T7_despliegue/requirements.txt
predict.py autodetecta el modelo y el schema desde T4_modelamiento/artifacts/.
python T7_despliegue/predict.py \
  --mode single \
  --input T7_despliegue/examples/single_payload.json \
  --verbose
