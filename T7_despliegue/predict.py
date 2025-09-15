#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import pandas as pd
import numpy as np

def _try_import():
    import joblib
    return joblib

def log(msg, verbose):
    if verbose:
        print(f"[INFO] {msg}", file=sys.stderr)

def load_artifacts(model_dir: Path, verbose=False):
    joblib = _try_import()
    model_path = model_dir / "best_model.joblib"
    cols_path = model_dir / "feature_columns.json"
    enc_path  = model_dir / "encoder.joblib"
    scl_path  = model_dir / "scaler.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not cols_path.exists():
        raise FileNotFoundError(f"Definición de columnas no encontrada: {cols_path}")

    model = joblib.load(model_path)
    with open(cols_path, "r") as f:
        feature_cols = json.load(f)

    encoder = joblib.load(enc_path) if enc_path.exists() else None
    scaler  = joblib.load(scl_path) if scl_path.exists() else None
    log("Artifacts cargados", verbose)
    return model, feature_cols, encoder, scaler

def preprocess(df: pd.DataFrame, feature_cols, encoder=None, scaler=None, verbose=False):
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_cols].copy()

    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("missing")

    if encoder is not None:
        cat_cols = [c for c in df.columns if df[c].dtype == "object"]
        if cat_cols:
            enc_arr = encoder.transform(df[cat_cols])
            enc_df = pd.DataFrame(enc_arr, index=df.index)
            num_df = df[[c for c in df.columns if c not in cat_cols]].reset_index(drop=True)
            df = pd.concat([num_df.reset_index(drop=True), enc_df.reset_index(drop=True)], axis=1)

    if scaler is not None:
        df[:] = scaler.transform(df.values)

    if verbose:
        log(f"Shape post-proc: {df.shape}", verbose)
    return df

def predict_single(payload: dict, model_dir: Path, verbose=False):
    model, feature_cols, encoder, scaler = load_artifacts(model_dir, verbose)
    df = pd.DataFrame([payload])
    X = preprocess(df, feature_cols, encoder, scaler, verbose)
    y_hat = float(model.predict(X)[0])
    return {
        "ok": True,
        "mode": "single",
        "n_predictions": 1,
        "predictions": [{"y_hat": round(y_hat, 2), "id": payload.get("id")}],
        "model_info": {
            "artifact_dir": str(model_dir),
            "features_version": "feature_columns.json",
            "model_file": "best_model.joblib"
        }
    }

def predict_batch(csv_path: Path, model_dir: Path, verbose=False):
    model, feature_cols, encoder, scaler = load_artifacts(model_dir, verbose)
    df = pd.read_csv(csv_path)
    ids = df["id"].astype(str).tolist() if "id" in df.columns else [None] * len(df)
    X = preprocess(df, feature_cols, encoder, scaler, verbose)
    preds = model.predict(X)
    out = [{"id": _id, "y_hat": round(float(p), 2)} for _id, p in zip(ids, preds)]
    return {
        "ok": True,
        "mode": "batch",
        "n_predictions": len(out),
        "predictions": out,
        "model_info": {
            "artifact_dir": str(model_dir),
            "features_version": "feature_columns.json",
            "model_file": "best_model.joblib"
        }
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single","batch"], required=True)
    ap.add_argument("--input", required=True, help="Ruta a .json (single) o .csv (batch)")
    ap.add_argument("--output", help="Archivo de salida JSON")
    ap.add_argument("--model-dir", default="models", help="Carpeta con artifacts")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    try:
        if args.mode == "single":
            with open(args.input, "r") as f:
                payload = json.load(f)
            result = predict_single(payload, model_dir, verbose=args.verbose)
        else:
            result = predict_batch(Path(args.input), model_dir, verbose=args.verbose)

        out_str = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(out_str, encoding="utf-8")
        print(out_str)
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
