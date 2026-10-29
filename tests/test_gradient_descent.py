"""
Pruebas unitarias para el laboratorio de descenso del gradiente con un solo peso.
Verifica las propiedades matemáticas y los valores analíticos esperados.
"""

import pytest
from modules.mathematics.labs.gradient_descent import (
    predict,
    compute_loss,
    compute_gradient,
    gradient_descent_step,
)


def test_initial_values_x2_y10_w1():
    """Verifica los valores analíticos exactos pedidos en la especificación."""
    x = 2.0
    y = 10.0
    w = 1.0

    # 1. Predicción: ŷ = 1 * 2 = 2
    y_pred = predict(w, x)
    assert y_pred == 2.0

    # 2. Pérdida: L(1) = (2 - 10)^2 = 64
    loss = compute_loss(w, x, y)
    assert loss == 64.0

    # 3. Gradiente: dL/dw = 2 * (2 - 10) * 2 = -32
    grad = compute_gradient(w, x, y)
    assert grad == -32.0

    # 4. Actualización con alpha = 0.05
    alpha = 0.05
    w_new, g, l_before, l_after = gradient_descent_step(w, x, y, alpha)
    assert g == -32.0
    assert l_before == 64.0
    # w_new = 1.0 - 0.05 * (-32.0) = 2.6
    assert pytest.approx(w_new, rel=1e-5) == 2.6
    # Pérdida tras el paso: (2.6 * 2 - 10)^2 = 23.04
    assert pytest.approx(l_after, rel=1e-5) == 23.04
    # La pérdida debe disminuir
    assert l_after < l_before


def test_minimum_point_has_zero_gradient():
    """El mínimo analítico w* = y / x = 10 / 2 = 5 debe tener gradiente 0 y pérdida 0."""
    x = 2.0
    y = 10.0
    w_star = y / x

    assert compute_loss(w_star, x, y) == 0.0
    assert compute_gradient(w_star, x, y) == 0.0

    # Un paso en el mínimo no debe modificar el peso
    w_new, grad, l_before, l_after = gradient_descent_step(w_star, x, y, alpha=0.1)
    assert w_new == w_star
    assert grad == 0.0
    assert l_after == 0.0


def test_positive_slope_when_w_is_greater_than_optimal():
    """
    Cuando w > w* (por ejemplo w = 7), la pendiente dL/dw es positiva.
    La actualización resta un término positivo, desplazando w a la izquierda hacia w*.
    """
    x = 2.0
    y = 10.0
    w = 7.0
    alpha = 0.05

    grad = compute_gradient(w, x, y)
    assert grad > 0.0  # dL/dw = 2 * (14 - 10) * 2 = +16.0

    w_new, _, l_before, l_after = gradient_descent_step(w, x, y, alpha)
    assert w_new < w  # Se mueve hacia el mínimo (de 7 a 6.2)
    assert l_after < l_before

