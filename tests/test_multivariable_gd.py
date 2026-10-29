"""
Pruebas unitarias para el laboratorio de gradiente multivariable y álgebra matricial.
"""

import pytest
import numpy as np
from modules.mathematics.labs.multivariable_gd import (
    get_default_dataset,
    predict_matrix,
    compute_residuals,
    compute_mse_loss,
    compute_gradient_vector,
    multivariable_gradient_descent_step,
)


def test_matrix_dimensions():
    """Verifica que las dimensiones de las matrices cumplan con el álgebra lineal."""
    X, Y, W_opt = get_default_dataset()

    assert X.shape == (3, 2)
    assert Y.shape == (3, 1)
    assert W_opt.shape == (2, 1)

    # Predicción: (3, 2) x (2, 1) -> (3, 1)
    Y_pred = predict_matrix(X, W_opt)
    assert Y_pred.shape == (3, 1)

    # Residuos: (3, 1)
    residuals = compute_residuals(X, Y, W_opt)
    assert residuals.shape == (3, 1)

    # Gradiente: (2, 1)
    grad = compute_gradient_vector(X, Y, W_opt)
    assert grad.shape == (2, 1)


def test_analytical_optimum_zero_loss_and_gradient():
    """En W* = [2.0, 3.0]^T el error es nulo y el gradiente es el vector cero."""
    X, Y, W_opt = get_default_dataset()

    loss = compute_mse_loss(X, Y, W_opt)
    assert pytest.approx(loss, abs=1e-7) == 0.0

    grad = compute_gradient_vector(X, Y, W_opt)
    np.testing.assert_allclose(grad, np.zeros((2, 1)), atol=1e-7)


def test_multivariable_descent_step_decreases_loss():
    """Un paso de gradiente desde una posición no óptima debe reducir el MSE."""
    X, Y, W_opt = get_default_dataset()
    W_init = np.array([[0.5], [1.0]], dtype=float)

    loss_initial = compute_mse_loss(X, Y, W_init)
    assert loss_initial > 0.0

    alpha = 0.05
    W_new, grad, l_before, l_after = multivariable_gradient_descent_step(X, Y, W_init, alpha)

    assert l_before == loss_initial
    assert l_after < l_before  # La pérdida debe disminuir

    # La distancia euclidiana hacia W* debe reducirse
    dist_before = np.linalg.norm(W_init - W_opt)
    dist_after = np.linalg.norm(W_new - W_opt)
    assert dist_after < dist_before

