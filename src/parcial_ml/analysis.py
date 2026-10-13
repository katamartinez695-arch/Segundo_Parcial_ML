
import numpy as np
import pandas as pd

from .data import load_data
from .models import fit_polynomial, predict_polynomial
from .risk import quadratic_risk


# EJERCICIO 1: grados de 1 a 15 con train_A

def exercise1():
    train = load_data("train_A.csv")
    validation = load_data("validation.csv")

    results = []
    models = {}

    for degree in range(1, 16):
        fitted = fit_polynomial(
            train["x"], train["y"], degree
        )
        models[degree] = fitted

        train_pred = predict_polynomial(
            fitted, train["x"]
        )
        val_pred = predict_polynomial(
            fitted, validation["x"]
        )

        results.append({
            "grado": degree,
            "riesgo_entrenamiento": quadratic_risk(
                train["y"], train_pred
            ),
            "riesgo_validacion": quadratic_risk(
                validation["y"], val_pred
            )
        })

    table = pd.DataFrame(results)

    best_train = table.loc[
        table["riesgo_entrenamiento"].idxmin()
    ]
    best_val = table.loc[
        table["riesgo_validacion"].idxmin()
    ]

    return table, models, best_train, best_val


# EJERCICIO 2: comparar las tres muestras

def exercise2():
    degrees = [2, 6, 15]
    datasets = {
        "A": load_data("train_A.csv"),
        "B": load_data("train_B.csv"),
        "C": load_data("train_C.csv")
    }

    validation = load_data("validation.csv")

    # Malla común de 300 puntos entre -2.5 y 2.5
    grid = np.linspace(-2.5, 2.5, 300)

    models = {}
    predictions = {}
    mean_predictions = {}
    variances = {}
    mean_validation_risks = {}

    for degree in degrees:
        models[degree] = {}
        predictions[degree] = {}

        for sample_name, df in datasets.items():
            fitted = fit_polynomial(
                df["x"], df["y"], degree
            )
            models[degree][sample_name] = fitted

            predictions[degree][sample_name] = (
                predict_polynomial(fitted, grid)
            )

        # Predicciones de A, B y C sobre la misma malla
        pred_matrix = np.vstack([
            predictions[degree][name]
            for name in ["A", "B", "C"]
        ])

        mean_pred = pred_matrix.mean(axis=0)
        mean_predictions[degree] = mean_pred

        # V_d: promedio sobre puntos y muestras
        variances[degree] = float(
            np.mean((pred_matrix - mean_pred) ** 2)
        )

        # Riesgo de validación promedio de los 3 modelos
        risks = []

        for sample_name in ["A", "B", "C"]:
            fitted = models[degree][sample_name]
            val_pred = predict_polynomial(
                fitted, validation["x"]
            )

            risks.append(
                quadratic_risk(validation["y"], val_pred)
            )

        mean_validation_risks[degree] = float(
            np.mean(risks)
        )

    variance_table = pd.DataFrame([
        {
            "grado": degree,
            "V_d": variances[degree],
            "riesgo_validacion_promedio":
                mean_validation_risks[degree]
        }
        for degree in degrees
    ])

    return (
        grid, models, predictions, mean_predictions,
        variances, mean_validation_risks, variance_table
    )


# EJERCICIO 3: selección del grado

def exercise3():
    (
        grid, models, predictions, mean_predictions,
        variances, mean_validation_risks, variance_table
    ) = exercise2()

    table = variance_table[
        ["grado", "riesgo_validacion_promedio", "V_d"]
    ].rename(columns={
        "riesgo_validacion_promedio": "Rval_d"
    })

    return table
