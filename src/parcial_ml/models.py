
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def fit_polynomial(x, y, degree):
    """Ajusta una regresión polinomial del grado indicado."""
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).ravel()

    polynomial = PolynomialFeatures(
        degree=degree,
        include_bias=False
    )

    x_poly = polynomial.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)

    return polynomial, model


def predict_polynomial(fitted_model, x):
    """Predice usando el transformador y modelo entrenados."""
    polynomial, model = fitted_model

    x = np.asarray(x).reshape(-1, 1)
    x_poly = polynomial.transform(x)

    return model.predict(x_poly)
