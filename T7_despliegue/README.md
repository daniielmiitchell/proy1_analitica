# T7 – Despliegue en EC2 (CLI)

## 🎯 Objetivo
Desplegar el modelo de TTR en una instancia EC2 y ejecutar predicciones **desde la nube** usando un script de línea de comandos (`predict.py`).  
La evidencia final será la salida JSON de una predicción hecha en la EC2.

---

## 🚀 1. Lanzar la instancia EC2
- **AMI**: Ubuntu Server 22.04/24.04 LTS (x86_64)  
- **Tipo**: t2.micro o t3.micro (free tier elegible)  
- **Security group**:
  - SSH (22) abierto
  - (Opcional: TCP 5000 si querés exponer API en el futuro)

---

## 🔑 2. Conectarse por SSH
En tu máquina local:

```bash
chmod 400 /ruta/a/tu-llave.pem
ssh -i /ruta/a/tu-llave.pem ubuntu@<PUBLIC_IP>

# Actualizar paquetes e instalar dependencias
sudo apt update && sudo apt -y install git python3-pip python3-venv

# Clonar repo
git clone https://github.com/daniielmiitchell/proy1_analitica.git
cd proy1_analitica

# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar requirements
pip install -r T7_despliegue/requirements.txt

# Verificar modelos
ls -l T4_modelamiento/artifacts/

# Regenerar corriendo
python3 T4_modelamiento/baseline_regresion.py
python3 T4_modelamiento/random_forest_regresion.py

# Usar el script predict.py
# ejemplo (ajustar a features reales)
cat > T7_despliegue/input.json <<'JSON'
{
  "priority_level": "3 - Moderate",
  "category": "Hardware",
  "assignment_group": "Service Desk",
  "feature_num_1": 12.0,
  "feature_num_2": 0.35,
  "feature_num_3": 7
}
JSON

# Ejecutar prediccion
python3 T7_despliegue/predict.py \
  --model T4_modelamiento/artifacts/model_ridge.joblib \
  --json  T7_despliegue/input.json
  
