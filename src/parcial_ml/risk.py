
import numpy as np


def quadratic_risk(y_true, y_pred):
    """Calcula el promedio de los errores al cuadrado."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Las observaciones reales y predichas "
            "deben tener la misma longitud."
        )

    return float(np.mean((y_true - y_pred) ** 2))
