# T7_despliegue/predict.py
# Carga un pipeline entrenado (Ridge o RF) y predice TTR para uno o varios registros.
# Uso:
#   python3 T7_despliegue/predict.py --model T4_modelamiento/artifacts/model_ridge.joblib --json input.json
#   cat input.json | python3 T7_despliegue/predict.py --model ... --stdin
#   python3 T7_despliegue/predict.py --model ... --print-schema

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Union

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART4 = ROOT / "T4_modelamiento" / "artifacts"

DEFAULT_MODEL = ART4 / "model_ridge.joblib"        # por estabilidad
FALLBACK_RF   = ART4 / "model_random_forest.joblib"  # si prefieres MAE

# intentamos cargar un esquema si existe (para validar columnas)
SCHEMA_CANDIDATES = [
    ART4 / "schema_ridge.json",
    ART4 / "schema_random_forest.json",
]

def load_schema() -> Dict[str, Any] | None:
    for p in SCHEMA_CANDIDATES:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return None

def expected_columns_from_schema(schema: Dict[str, Any] | None) -> List[str] | None:
    if not schema:
        return None
    # buscá nombres típicos; ajusta si tu script guardó otra clave
    for key in ["expected_features", "feature_names", "features", "columns"]:
        if key in schema and isinstance(schema[key], list):
            return list(schema[key])
    # a veces el schema guarda por grupos
    num = schema.get("num_cols") or schema.get("numericas") or []
    cat = schema.get("cat_cols") or schema.get("categoricas") or []
    if isinstance(num, list) and isinstance(cat, list) and (num or cat):
        return list(num) + list(cat)
    return None

def read_payload(source: Union[str, Path, None], use_stdin: bool) -> List[Dict[str, Any]]:
    if use_stdin:
        data = sys.stdin.read().strip()
        if not data:
            raise SystemExit("STDIN vacío. Pasa un JSON por pipe o usa --json file.json")
        return json.loads(data) if data.lstrip().startswith("[") else [json.loads(data)]
    elif source:
        text = Path(source).read_text()
        return json.loads(text) if text.lstrip().startswith("[") else [json.loads(text)]
    else:
        raise SystemExit("Necesitas --json <archivo> o --stdin")

def validate_and_frame(rows: List[Dict[str, Any]], expected_cols: List[str] | None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if expected_cols:
        # advertimos si faltan o sobran
        missing = [c for c in expected_cols if c not in df.columns]
        extra   = [c for c in df.columns if c not in expected_cols]
        if missing:
            print(f"[WARN] Faltan columnas esperadas: {missing}", file=sys.stderr)
        if extra:
            print(f"[WARN] Columnas no esperadas en payload: {extra}", file=sys.stderr)
        # reordenamos cuando sea posible
        df = df.reindex(columns=expected_cols, fill_value=None)
    return df

def main():
    parser = argparse.ArgumentParser(description="Predictor TTR con pipeline entrenado")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="Ruta al .joblib del pipeline (por defecto Ridge).")
    parser.add_argument("--json", type=str, help="Archivo JSON con 1 o más registros.")
    parser.add_argument("--stdin", action="store_true", help="Leer el JSON desde STDIN.")
    parser.add_argument("--print-schema", action="store_true",
                        help="Imprime columnas esperadas si hay schema disponible.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        # fallback rápido a RF si el de Ridge no existe
        if Path(FALLBACK_RF).exists():
            print(f"[INFO] No se encontró {model_path.name}. Usando RF por defecto.", file=sys.stderr)
            model_path = Path(FALLBACK_RF)
        else:
            raise SystemExit(f"No existe el modelo: {model_path}")

    print(f"[INFO] Cargando modelo: {model_path}")
    pipe = joblib.load(model_path)

    schema = load_schema()
    exp_cols = expected_columns_from_schema(schema)
    if args.print-schema:
        print(json.dumps({"expected_columns": exp_cols}, indent=2, ensure_ascii=False))
        return

    rows = read_payload(args.json, args.stdin)
    df = validate_and_frame(rows, exp_cols)

    # predict
    preds = pipe.predict(df)
    out = [{"prediction_ttr_h": float(p)} for p in preds]
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()