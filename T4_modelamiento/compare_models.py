# T4_modelamiento/compare_models.py
from pathlib import Path
import json
import pandas as pd

ART = Path(__file__).resolve().parent / "artifacts"

def load_json(p: Path):
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return None

def main():
    files = {
        "Ridge": ART / "metrics_ridge.json",
        "RandomForest": ART / "metrics_random_forest.json",
    }

    rows = []
    for name, path in files.items():
        m = load_json(path)
        if m:
            rows.append({
                "modelo": name,
                "MAE": m.get("MAE"),
                "RMSE": m.get("RMSE"),
                "R2": m.get("R2"),
                "CV_MAE_mean": m.get("cv_mae_mean"),
                "CV_MAE_std": m.get("cv_mae_std"),
            })

    if not rows:
        print("No se encontraron métricas. Corre primero los entrenamientos.")
        print("Esperados:", [str(p) for p in files.values()])
        return

    df = pd.DataFrame(rows).set_index("modelo")
    print("\n=== Comparación de modelos ===")
    print(df.round(3))
    print("\nMejor (MAE) ->", df["MAE"].astype(float).idxmin())

if __name__ == "__main__":
    main()