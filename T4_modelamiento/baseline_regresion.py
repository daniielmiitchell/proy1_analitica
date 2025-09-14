# T4_modelamiento/baseline_regresion.py
# Baseline de regresión (Ridge) con preprocesamiento y validación en español.

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "T3_preparacion" / "incidents_clean_for_model.csv"
ART  = ROOT / "T4_modelamiento" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

FORBIDDEN_TOKENS = [
    "ttr", "resolve", "resolved", "close", "closed",
    "duration", "elapsed", "breach", "sla", "tt_resolve",
    "time_to", "tt_", "resolution"
]
def is_forbidden(colname: str) -> bool:
    c = colname.lower()
    return any(tok in c for tok in FORBIDDEN_TOKENS)

def select_safe_features(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    df = df.copy()
    drop_targets = [c for c in df.columns if ("ttr" in c.lower()) or (c == y_col)]
    safe = df.drop(columns=drop_targets, errors="ignore")
    safe = safe[[c for c in safe.columns if not is_forbidden(c)]]
    n = len(safe)
    to_drop = []
    for c in safe.select_dtypes(include="object").columns:
        try:
            if safe[c].nunique(dropna=True) / max(1, n) > 0.5:
                to_drop.append(c)
        except Exception:
            pass
    if to_drop:
        print(f"[INFO] Se descartan columnas de alta cardinalidad: {to_drop}")
        safe = safe.drop(columns=to_drop)
        # al final de select_safe_features, antes de return:
        safe = safe.select_dtypes(exclude=["datetime64[ns]", "datetime64[ns, UTC]"])

    return safe

def _build_ohe():
    """OneHotEncoder compatible con distintas versiones de scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)  # >=1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)         # <1.2

def main():
    print("=== 1) Carga de datos ===")
    df = pd.read_csv(DATA)
    print(f"Shape original: {df.shape}")

    # 1.1 Selección de target
    y_col = "ttr_h_winsor" if "ttr_h_winsor" in df.columns else "ttr_h"
    if y_col not in df.columns:
        raise ValueError("No encuentro columnas de target 'ttr_h' ni 'ttr_h_winsor' en el CSV.")

    # 1.2 Limpieza robusta del target (numérico, finito, >=0)
    y_num = pd.to_numeric(df[y_col], errors="coerce")
    mask_ok = y_num.notna() & np.isfinite(y_num.values) & (y_num.values >= 0)
    dropped = int((~mask_ok).sum())
    if dropped > 0:
        print(f"[WARN] Filas eliminadas por target inválido ({y_col}): {dropped}")
    df = df.loc[mask_ok].copy().reset_index(drop=True)
    y = pd.to_numeric(df[y_col], errors="coerce")

   # 1.3 Definición de X sin fugas (quitamos variantes del target y filtramos features “seguras”)
    drop_targets = [c for c in ["ttr_h", "ttr_h_winsor", "ttr_h_log", "ttr_outlier"] if c in df.columns]
    X_full = df.drop(columns=drop_targets)
    X = select_safe_features(X_full, y_col=y_col)
    print(f"Shape tras limpiar: X={X.shape}, y={y.shape}")


    print("=== 2) Split train/test ===")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, shuffle=True)
    print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

    print("=== 3) Columnas numéricas y categóricas ===")
    num_cols = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = Xtr.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"Numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}")

    if not num_cols and not cat_cols:
        raise RuntimeError("No quedaron columnas para entrenar tras el filtrado seguro.")

    print("=== 4) Preprocesamiento ===")
    num_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        # StandardScaler con with_mean=False mantiene compatibilidad con matrices dispersas
        ("scaler", StandardScaler(with_mean=False))
    ])
    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", _build_ohe())
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0
    )

    print("=== 5) Modelo baseline (Ridge) ===")
    model = Ridge(alpha=1.0)

    pipe = Pipeline([
        ("prep", pre),
        ("model", model)
    ])

    print("=== 6) Validación cruzada (5 folds) en TRAIN ===")
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_mae = -cross_val_score(pipe, Xtr, ytr, scoring="neg_mean_absolute_error", cv=cv)
    print(f"Target usado: {y_col}")
    print(f"CV MAE por fold: {np.round(cv_mae, 3)} | mean: {np.round(cv_mae.mean(), 3)}")

    print("=== 7) Entrenamiento final y evaluación en TEST ===")
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    # Compatibilidad con versiones antiguas: RMSE = sqrt(MSE)
    mse = mean_squared_error(yte, pred)
    rmse = float(np.sqrt(mse))

    metrics = {
        "target": y_col,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "MAE": float(mean_absolute_error(yte, pred)),
        "RMSE": rmse,
        "R2": float(r2_score(yte, pred)),
        "cv_mae_mean": float(cv_mae.mean()),
        "cv_mae_std": float(cv_mae.std()),
    }
    print("Holdout metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print("=== 8) Guardar artefactos ===")
    joblib.dump(pipe, ART / "model_ridge.joblib")
    with open(ART / "metrics_ridge.json","w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(ART / "schema_ridge.json","w") as f:
        json.dump({"num_cols": num_cols, "cat_cols": cat_cols}, f, indent=2, ensure_ascii=False)

    print("Artefactos guardados en:", ART)

if __name__ == "__main__":
    main()
    
print("Ejemplo num_cols:", num_cols[:5])
print("Ejemplo cat_cols:", cat_cols[:5])
