# T4_modelamiento/random_forest_regresion.py
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "T3_preparacion" / "incidents_clean_for_model.csv"
ART  = ROOT / "T4_modelamiento" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# Para que no “se pegue” mientras pruebas:
FAST_DEBUG = False   # pon True si quieres cv=3 y menos árboles para probar rápido

FORBIDDEN_TOKENS = [
    "ttr", "resolve", "resolved", "close", "closed",
    "duration", "elapsed", "breach", "sla", "tt_resolve",
    "time_to", "tt_", "resolution"
]

def is_forbidden(colname: str) -> bool:
    c = colname.lower()
    return any(tok in c for tok in FORBIDDEN_TOKENS)

def select_safe_features(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """
    Selecciona un subconjunto 'seguro' de columnas:
    - elimina cualquier columna que contenga tokens peligrosos (arriba)
    - elimina el target y derivados obvios
    - elimina IDs/timestamps de altísima cardinalidad (memorizarían filas)
      regla simple: si nunique / nfilas > 0.5 y dtype es object, se descarta
    """
    df = df.copy()

    # Quita target y variantes por nombre
    drop_targets = [c for c in df.columns if ("ttr" in c.lower()) or (c == y_col)]
    safe = df.drop(columns=drop_targets, errors="ignore")

    # Quita columnas con tokens prohibidos
    safe = safe[[c for c in safe.columns if not is_forbidden(c)]]

    # Quita columnas con cardinalidad altísima (IDs, timestamps crudos)
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

def main():
    print("=== 1) Carga de datos ===")
    df = pd.read_csv(DATA)

    # Target (usa winsor si existe)
    y_col = "ttr_h_winsor" if "ttr_h_winsor" in df.columns else "ttr_h"
    if y_col not in df.columns:
        raise RuntimeError("No encuentro columnas de target: ni 'ttr_h_winsor' ni 'ttr_h'.")

    # Diagnóstico rápido del target
    print("\n=== Diagnóstico del target ===")
    print(df[y_col].describe())

    # Limpieza target (sin NaN/Inf y no negativos)
    y_raw = pd.to_numeric(df[y_col], errors="coerce")
    mask = y_raw.notna() & np.isfinite(y_raw) & (y_raw >= 0)
    drop_count = (~mask).sum()
    if drop_count > 0:
        print(f"[WARN] Filas eliminadas por target inválido ({y_col}): {drop_count}")
    df = df.loc[mask].copy().reset_index(drop=True)

    # Selección de features seguras
    # Selección de features seguras
    X_full = df.drop(columns=[y_col], errors="ignore")
    X = select_safe_features(X_full, y_col=y_col)
    y = df[y_col].astype(float)

    print("\n=== 2) Split train/test (80/20) ===")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"Train: {Xtr.shape} | Test: {Xte.shape}")

    print("\n=== 3) Columnas numéricas y categóricas (post-filtro seguro) ===")
    num_cols = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = Xtr.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"Numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}")
    if len(num_cols) + len(cat_cols) == 0:
        raise RuntimeError("No quedaron columnas seguras para entrenar. Revisa el filtro.")

    print("\n=== 4) Preprocesamiento ===")
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    print("\n=== 5) Modelo: RandomForestRegressor ===")
    rf = RandomForestRegressor(
        n_estimators=150 if FAST_DEBUG else 350,
        max_depth=None,
        n_jobs=-1,
        random_state=SEED
    )
    pipe = Pipeline([
        ("prep", pre),
        ("model", rf),
    ])

    print("\n=== 6) Validación cruzada en TRAIN ===")
    cv = KFold(n_splits=(3 if FAST_DEBUG else 5), shuffle=True, random_state=SEED)
    cv_mae = -cross_val_score(pipe, Xtr, ytr, scoring="neg_mean_absolute_error", cv=cv)
    print(f"Target usado: {y_col}")
    print("CV MAE por fold:", np.round(cv_mae, 2), "| mean:", round(cv_mae.mean(), 3))

    print("\n=== 7) Entrenamiento final y evaluación en TEST ===")
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    mse = mean_squared_error(yte, pred)
    rmse = float(np.sqrt(mse))
    metrics = {
        "target": y_col,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "MAE": float(mean_absolute_error(yte, pred)),
        "RMSE": rmse,
        "R2": float(r2_score(yte, pred)),
        "cv_mae_mean": float(cv_mae.mean()),
        "cv_mae_std": float(cv_mae.std()),
    }
    print("Holdout metrics:")
    for k, v in metrics.items():
        print(f'  "{k}": {v}')

    print("\n=== 8) Guardar artefactos ===")
    joblib.dump(pipe, ART / "model_random_forest.joblib")
    with open(ART / "metrics_random_forest.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(ART / "schema_random_forest.json", "w") as f:
        json.dump({"num_cols": num_cols, "cat_cols": cat_cols}, f, indent=2)

    print("Artefactos guardados en:", ART)

if __name__ == "__main__":
    main()

