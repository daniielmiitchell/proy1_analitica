# T6_evaluacion/evaluacion_regresion.py
# Evaluación OOF (cross_val_predict) con limpieza robusta del target y features “seguras”.

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "T3_preparacion" / "incidents_clean_for_model.csv"
ART  = ROOT / "T6_evaluacion" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# ---------- utilidades de filtrado seguro (evita fugas) ----------
FORBIDDEN_TOKENS = [
    "ttr", "resolve", "resolved", "close", "closed",
    "duration", "elapsed", "breach", "sla", "tt_resolve",
    "time_to", "tt_", "resolution",
]
def _is_forbidden(colname: str) -> bool:
    c = colname.lower()
    return any(tok in c for tok in FORBIDDEN_TOKENS)

def _select_safe_features(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    df = df.copy()
    # quita target y variantes obviamente peligrosas
    drop_targets = [c for c in df.columns if ("ttr" in c.lower()) or (c == y_col)]
    safe = df.drop(columns=drop_targets, errors="ignore")
    # quita por nombre tokens peligrosos
    safe = safe[[c for c in safe.columns if not _is_forbidden(c)]]
    # quita ids/timestamps de altísima cardinalidad (memorizarían filas)
    n = len(safe)
    to_drop = []
    for c in safe.select_dtypes(include="object").columns:
        try:
            ratio = safe[c].nunique(dropna=True) / max(1, n)
            if ratio > 0.5:
                to_drop.append(c)
        except Exception:
            pass
    if to_drop:
        print(f"[INFO] Se descartan columnas de alta cardinalidad: {to_drop}")
        safe = safe.drop(columns=to_drop)
    return safe

def _build_ohe():
    # compat con distintas versiones de scikit-learn
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)  # >=1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)         # <1.2

# ---------- main ----------
def main():
    print("=== T6: Evaluación OOF de RandomForest ===")
    print(f"Leyendo: {DATA}")
    df = pd.read_csv(DATA)
    print(f"Shape original: {df.shape}")

    # target
    y_col = "ttr_h_winsor" if "ttr_h_winsor" in df.columns else "ttr_h"
    if y_col not in df.columns:
        raise RuntimeError("No encuentro target: ni 'ttr_h_winsor' ni 'ttr_h'.")

    # limpieza robusta del target
    y_raw = pd.to_numeric(df[y_col], errors="coerce")
    mask = y_raw.notna() & np.isfinite(y_raw) & (y_raw >= 0)
    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"[WARN] Filas eliminadas por target inválido ({y_col}): {dropped}")
    df = df.loc[mask].copy().reset_index(drop=True)
    y = y_raw.loc[mask].astype(float).reset_index(drop=True)

    # selección de features seguras
    X_full = df.drop(columns=[y_col], errors="ignore")
    X = _select_safe_features(X_full, y_col=y_col)

    # columnas num/cat
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if not num_cols and not cat_cols:
        raise RuntimeError("No quedaron columnas para entrenar tras el filtrado seguro.")
    print(f"Numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)} | y: {y.shape}")

    # preprocesamiento
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", _build_ohe()),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop", sparse_threshold=1.0)

    # modelo
    rf = RandomForestRegressor(
        n_estimators=350, max_depth=None, n_jobs=-1, random_state=SEED
    )
    pipe = Pipeline([
        ("prep", pre),
        ("model", rf),
    ])

    # OOF (Out-Of-Fold)
    print("Generando predicciones OOF con KFold(5, shuffle=True)...")
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)

    # métricas globales
    mse = mean_squared_error(y, oof_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y, oof_pred))
    r2  = float(r2_score(y, oof_pred))
    metrics_global = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print("=== Métricas OOF (global) ===")
    print(json.dumps(metrics_global, indent=2, ensure_ascii=False))

    # por prioridad (si existe)
    if "priority_level" in df.columns:
        df_eval = df.copy()
        df_eval["y"] = y.values
        df_eval["pred"] = oof_pred
        rows = []
        for p, g in df_eval.groupby("priority_level"):
            if len(g) < 5:
                continue
            _mse = mean_squared_error(g["y"], g["pred"])
            rows.append({
                "priority_level": p,
                "n": int(len(g)),
                "MAE": float(mean_absolute_error(g["y"], g["pred"])),
                "RMSE": float(np.sqrt(_mse)),
                "R2": float(r2_score(g["y"], g["pred"])),
            })
        if rows:
            by_pri = pd.DataFrame(rows).sort_values("priority_level")
            print("\n=== Métricas OOF por priority_level ===")
            print(by_pri.to_string(index=False))
            by_pri.to_csv(ART / "metrics_by_priority.csv", index=False)

    # guarda OOF para trazabilidad
    out = pd.DataFrame({"y": y, "pred": oof_pred})
    out.to_csv(ART / "oof_predictions.csv", index=False)
    print(f"\nArtefactos guardados en: {ART}")

if __name__ == "__main__":
    main()
