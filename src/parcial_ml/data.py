
from pathlib import Path
import pandas as pd

# Raíz del proyecto: segundo_parcial/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_data(filename):
    """Carga un CSV que contiene las columnas x e y."""
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {"x", "y"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{filename} debe contener las columnas x e y."
        )

    return df[["x", "y"]].copy()
