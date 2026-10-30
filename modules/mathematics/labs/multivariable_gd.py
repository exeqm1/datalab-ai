"""
Laboratorio interactivo: Descenso del gradiente multivariable y álgebra matricial (2 pesos).

Modelo lineal multivariable:
    ŷ = X · W

Dimensiones:
    X: (N x D)  -> N observaciones, D características
    W: (D x 1)  -> Vector de pesos
    ŷ: (N x 1)  -> Vector de predicciones
    Y: (N x 1)  -> Vector de objetivos
    E: (N x 1)  -> Vector de errores (ŷ - Y)

Pérdida (MSE medio):
    L(W) = (1 / N) * ||X · W - Y||^2 = (1 / N) * E^T · E

Gradiente matricial respecto a W:
    ∇_W L = (2 / N) * X^T · (X · W - Y) = (2 / N) * X^T · E  -> (D x 1)

Regla de actualización matricial:
    W_nuevo = W_actual - alpha * ∇_W L
"""

from typing import Tuple, List, Optional
import json
import numpy as np
import streamlit.components.v1 as components


# ==========================================
# CÁLCULOS MATEMÁTICOS PUROS (Desacoplados)
# ==========================================

def get_default_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna el dataset pedagógico estándar con N=3 muestras y D=2 características.
    Relación analítica exacta: y = 2*x1 + 3*x2
    Pesos óptimos: W* = [2.0, 3.0]^T
    """
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 2.0]
    ], dtype=float)

    W_opt = np.array([[2.0], [3.0]], dtype=float)
    Y = X @ W_opt  # [[8.0], [7.0], [12.0]]
    return X, Y, W_opt


def predict_matrix(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Calcula ŷ = X · W con producto matricial."""
    return X @ W


def compute_residuals(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Calcula el vector de residuos E = ŷ - Y."""
    return predict_matrix(X, W) - Y


def compute_mse_loss(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> float:
    """
    Calcula el error cuadrático medio:
        L(W) = (1 / N) * sum((ŷ_i - y_i)^2) = (1 / N) * E^T · E
    """
    error = compute_residuals(X, Y, W)
    return float(np.mean(error ** 2))


def compute_gradient_vector(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Calcula el vector gradiente respecto a W:
        ∇_W L = (2 / N) * X^T · (X · W - Y)
    Retorna un array de forma (D, 1).
    """
    N = X.shape[0]
    error = compute_residuals(X, Y, W)
    grad = (2.0 / N) * (X.T @ error)
    return grad


def multivariable_gradient_descent_step(
    X: np.ndarray, Y: np.ndarray, W: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Aplica una actualización del vector de pesos:
        W_nuevo = W - alpha * ∇_W L
    """
    loss_before = compute_mse_loss(X, Y, W)
    grad = compute_gradient_vector(X, Y, W)
    W_new = W - alpha * grad
    loss_after = compute_mse_loss(X, Y, W_new)
    return W_new, grad, loss_before, loss_after


def compute_loss_surface(
    X: np.ndarray,
    Y: np.ndarray,
    w1_center: float = 2.0,
    w2_center: float = 3.0,
    span: float = 4.0,
    resolution: int = 35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula una malla (grid) para la superficie de pérdida 3D:
        w1_grid, w2_grid -> Z (L(w1, w2))
    """
    w1_vals = np.linspace(w1_center - span, w1_center + span, resolution)
    w2_vals = np.linspace(w2_center - span, w2_center + span, resolution)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    Z = np.zeros_like(W1)
    for i in range(resolution):
        for j in range(resolution):
            w_candidate = np.array([[W1[i, j]], [W2[i, j]]])
            err = X @ w_candidate - Y
            Z[i, j] = np.mean(err ** 2)

    return W1, W2, Z


# ==========================================
# INTERFAZ CLIENTE 60FPS (PLOTLY.JS EMBEDDED)
# ==========================================

HTML_MULTIVARIABLE_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            -webkit-font-smoothing: antialiased;
        }
        body {
            background-color: #09090B;
            color: #F4F4F5;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
            overflow: hidden;
            padding: 0 4px;
        }
        .shell {
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-width: 100%;
        }

        /* Barra de título */
        .top-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #1F1F24;
            padding-bottom: 8px;
        }
        .top-left {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .title {
            font-size: 1.05rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: #F4F4F5;
        }
        .badge {
            font-family: ui-monospace, SFMono-Regular, monospace;
            font-size: 0.7rem;
            padding: 1px 6px;
            border-radius: 4px;
            background: #18181B;
            border: 1px solid #27272A;
            color: #A1A1AA;
        }
        .badge-opt {
            color: #38BDF8;
            border-color: #0369A1;
            background: #082F49;
            font-weight: 500;
        }

        /* Visor de Matrices Interactivo */
        .matrix-board {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #0D0D11;
            border: 1px solid #1F1F24;
            border-radius: 6px;
            padding: 8px 12px;
            overflow-x: auto;
        }
        .matrix-box {
            background: #121216;
            border: 1px solid #27272A;
            border-radius: 5px;
            padding: 6px 10px;
            font-family: ui-monospace, monospace;
            font-size: 0.76rem;
            min-width: 80px;
        }
        .matrix-box.active-w {
            border-color: #3B82F6;
        }
        .matrix-label {
            font-size: 0.65rem;
            color: #71717A;
            font-weight: 600;
            margin-bottom: 3px;
        }
        .matrix-row {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            color: #E4E4E7;
            line-height: 1.35;
        }
        .matrix-op {
            font-size: 1.1rem;
            color: #71717A;
            font-weight: 500;
        }

        /* Layout en 2 columnas */
        .workspace {
            display: grid;
            grid-template-columns: 1.35fr 1fr;
            gap: 14px;
            align-items: start;
        }

        /* Columna izquierda: Gráfico */
        .chart-panel {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .chart-box {
            background: #0D0D11;
            border: 1px solid #1F1F24;
            border-radius: 8px;
            padding: 4px;
            position: relative;
        }
        #plotDiv {
            width: 100%;
            height: 280px;
        }

        .math-banner {
            background: #121216;
            border: 1px solid #1F1F24;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 0.78rem;
            color: #A1A1AA;
            line-height: 1.4;
        }
        .math-banner code {
            font-family: ui-monospace, monospace;
            color: #F4F4F5;
            background: #18181B;
            padding: 1px 4px;
            border-radius: 3px;
        }

        /* Columna derecha: Controles y Métricas */
        .controls-panel {
            background: #121216;
            border: 1px solid #1F1F24;
            border-radius: 8px;
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .view-toggle {
            display: flex;
            gap: 6px;
            background: #09090B;
            border: 1px solid #27272A;
            border-radius: 6px;
            padding: 2px;
        }
        .view-btn {
            flex: 1;
            font-size: 0.74rem;
            font-weight: 600;
            padding: 5px 8px;
            background: transparent;
            color: #71717A;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .view-btn.active {
            background: #18181B;
            color: #F4F4F5;
            box-shadow: 0 1px 3px rgba(0,0,0,0.4);
        }

        .slider-group {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .slider-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.76rem;
            color: #A1A1AA;
            font-weight: 500;
        }
        .slider-val {
            font-family: ui-monospace, monospace;
            font-size: 0.76rem;
            color: #F4F4F5;
            background: #18181B;
            padding: 1px 5px;
            border-radius: 3px;
            border: 1px solid #27272A;
        }
        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 4px;
            background: #27272A;
            border-radius: 2px;
            outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: #F4F4F5;
            cursor: pointer;
            border: 2px solid #09090B;
        }
        input[type="range"]::-webkit-slider-thumb:hover {
            background: #3B82F6;
            transform: scale(1.15);
        }

        .action-row {
            display: grid;
            grid-template-columns: 1.35fr 1fr;
            gap: 8px;
            margin-top: 2px;
        }
        button.action-btn {
            font-family: inherit;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 7px 10px;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .btn-run {
            background: #F4F4F5;
            color: #09090B;
            border: 1px solid #F4F4F5;
        }
        .btn-run:hover {
            background: #FFFFFF;
            box-shadow: 0 0 8px rgba(255, 255, 255, 0.2);
        }
        .btn-reset {
            background: #18181B;
            color: #D4D4D8;
            border: 1px solid #27272A;
        }
        .btn-reset:hover {
            border-color: #3F3F46;
            color: #FFFFFF;
        }

        /* Cuadrícula de métricas */
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
        }
        .metric-cell {
            background: #0D0D11;
            border: 1px solid #1F1F24;
            border-radius: 5px;
            padding: 6px 8px;
        }
        .metric-label {
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #71717A;
            font-weight: 600;
        }
        .metric-val {
            font-family: ui-monospace, monospace;
            font-size: 1.1rem;
            font-weight: 600;
            color: #F4F4F5;
            margin-top: 1px;
        }
        .metric-sub {
            font-size: 0.68rem;
            color: #71717A;
            font-family: ui-monospace, monospace;
        }

        /* Tarjeta de paso */
        .step-info {
            background: #0D0D11;
            border: 1px solid #1F1F24;
            border-left: 3px solid #3B82F6;
            border-radius: 5px;
            padding: 6px 10px;
            font-size: 0.74rem;
            color: #D4D4D8;
            display: none;
        }
        .step-info code {
            font-family: ui-monospace, monospace;
            color: #38BDF8;
        }
    </style>
</head>
<body>
    <div class="shell">
        <!-- Encabezado -->
        <div class="top-row">
            <div class="top-left">
                <span class="title">Álgebra Matricial y Superficie 3D</span>
                <span class="badge">N=3 &bull; D=2</span>
                <span class="badge">ŷ = X · W</span>
            </div>
            <span class="badge badge-opt">W* óptimo = [2.00, 3.00]ᵀ</span>
        </div>

        <!-- Visor de Matrices Interactivo -->
        <div class="matrix-board">
            <!-- Matriz X -->
            <div class="matrix-box">
                <div class="matrix-label">Matriz X (3&times;2)</div>
                <div class="matrix-row"><span>1.0</span><span>2.0</span></div>
                <div class="matrix-row"><span>2.0</span><span>1.0</span></div>
                <div class="matrix-row"><span>3.0</span><span>2.0</span></div>
            </div>

            <div class="matrix-op">&times;</div>

            <!-- Vector W -->
            <div class="matrix-box active-w">
                <div class="matrix-label" style="color: #3B82F6;">W (2&times;1)</div>
                <div class="matrix-row"><span id="mW1" style="color: #38BDF8; font-weight: 600;">__INIT_W1__</span></div>
                <div class="matrix-row"><span id="mW2" style="color: #38BDF8; font-weight: 600;">__INIT_W2__</span></div>
            </div>

            <div class="matrix-op">=</div>

            <!-- Vector ŷ -->
            <div class="matrix-box">
                <div class="matrix-label">Pred ŷ (3&times;1)</div>
                <div class="matrix-row"><span id="mYp1">0.00</span></div>
                <div class="matrix-row"><span id="mYp2">0.00</span></div>
                <div class="matrix-row"><span id="mYp3">0.00</span></div>
            </div>

            <div class="matrix-op">vs</div>

            <!-- Vector Y -->
            <div class="matrix-box">
                <div class="matrix-label">Obj Y (3&times;1)</div>
                <div class="matrix-row"><span>8.0</span></div>
                <div class="matrix-row"><span>7.0</span></div>
                <div class="matrix-row"><span>12.0</span></div>
            </div>

            <div class="matrix-op">&rarr;</div>

            <!-- Vector E -->
            <div class="matrix-box">
                <div class="matrix-label" style="color: #F87171;">Error E (3&times;1)</div>
                <div class="matrix-row"><span id="mE1" style="color: #F87171;">+0.00</span></div>
                <div class="matrix-row"><span id="mE2" style="color: #F87171;">+0.00</span></div>
                <div class="matrix-row"><span id="mE3" style="color: #F87171;">+0.00</span></div>
            </div>
        </div>

        <!-- Área de Trabajo -->
        <div class="workspace">
            <!-- Gráfica 3D / 2D -->
            <div class="chart-panel">
                <div class="chart-box">
                    <div id="plotDiv"></div>
                </div>
                <div class="math-banner">
                    <b>Gradiente Matricial:</b> <code>∇_W L = (2/3) · Xᵀ · E</code><br>
                    <span id="gradText">∇ = [0.00, 0.00]ᵀ</span> &bull; 
                    Regla: <code>W ← W - α · ∇L</code> actualiza ambos pesos en el cuenco 3D.
                </div>
            </div>

            <!-- Controles y Métricas -->
            <div class="controls-panel">
                <!-- Selector de Vista -->
                <div class="view-toggle">
                    <button class="view-btn active" id="btnView3D">3D (Superficie)</button>
                    <button class="view-btn" id="btnView2D">2D (Contorno)</button>
                </div>

                <!-- Sliders -->
                <div class="slider-group">
                    <div class="slider-header">
                        <span>Peso w1</span>
                        <span class="slider-val" id="valW1">__INIT_W1__</span>
                    </div>
                    <input type="range" id="sliderW1" min="-1.5" max="5.5" step="0.02" value="__INIT_W1__">
                </div>

                <div class="slider-group">
                    <div class="slider-header">
                        <span>Peso w2</span>
                        <span class="slider-val" id="valW2">__INIT_W2__</span>
                    </div>
                    <input type="range" id="sliderW2" min="-1.0" max="6.5" step="0.02" value="__INIT_W2__">
                </div>

                <div class="slider-group">
                    <div class="slider-header">
                        <span>Tasa de aprendizaje (α)</span>
                        <span class="slider-val" id="valAlpha">__INIT_ALPHA__</span>
                    </div>
                    <input type="range" id="sliderAlpha" min="0.005" max="0.20" step="0.005" value="__INIT_ALPHA__">
                </div>

                <!-- Botones -->
                <div class="action-row">
                    <button class="action-btn btn-run" id="btnStepMat">Ejecutar paso</button>
                    <button class="action-btn btn-reset" id="btnResetMat">Reiniciar</button>
                </div>

                <!-- Métricas 2x2 -->
                <div class="metrics-grid">
                    <div class="metric-cell">
                        <div class="metric-label">Pérdida MSE</div>
                        <div class="metric-val" id="metricMse">0.00</div>
                        <div class="metric-sub" id="metricDist">dist: 0.00</div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-label">||∇L|| Magnitud</div>
                        <div class="metric-val" id="metricGradNorm">0.00</div>
                        <div class="metric-sub" id="metricGradShort">[0.0, 0.0]</div>
                    </div>
                </div>

                <!-- Detalle del paso -->
                <div class="step-info" id="stepBox">
                    <div style="font-size: 0.68rem; text-transform: uppercase; color: #71717A; font-weight: 600;">Paso Matricial</div>
                    <div id="stepFormula" style="margin: 2px 0;"></div>
                    <div id="stepLossDiff" style="color: #A1A1AA; font-size: 0.7rem;"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Datos matemáticos fijos
        const X_mat = [[1.0, 2.0], [2.0, 1.0], [3.0, 2.0]];
        const Y_vec = [8.0, 7.0, 12.0];
        const W_opt = [2.0, 3.0];

        // Malla precalculada en Python
        const gridW1 = __GRID_W1__;
        const gridW2 = __GRID_W2__;
        const gridZ  = __GRID_Z__;

        let curW1 = __INIT_W1__;
        let curW2 = __INIT_W2__;
        let curAlpha = __INIT_ALPHA__;
        let currentMode = '3D';
        let trajectory = [[curW1, curW2, computeLoss(curW1, curW2)]];

        // Elementos DOM
        const sliderW1 = document.getElementById('sliderW1');
        const sliderW2 = document.getElementById('sliderW2');
        const sliderAlpha = document.getElementById('sliderAlpha');
        const valW1 = document.getElementById('valW1');
        const valW2 = document.getElementById('valW2');
        const valAlpha = document.getElementById('valAlpha');
        const mW1 = document.getElementById('mW1');
        const mW2 = document.getElementById('mW2');
        const mYp1 = document.getElementById('mYp1');
        const mYp2 = document.getElementById('mYp2');
        const mYp3 = document.getElementById('mYp3');
        const mE1 = document.getElementById('mE1');
        const mE2 = document.getElementById('mE2');
        const mE3 = document.getElementById('mE3');
        const metricMse = document.getElementById('metricMse');
        const metricDist = document.getElementById('metricDist');
        const metricGradNorm = document.getElementById('metricGradNorm');
        const metricGradShort = document.getElementById('metricGradShort');
        const gradText = document.getElementById('gradText');
        const btnStepMat = document.getElementById('btnStepMat');
        const btnResetMat = document.getElementById('btnResetMat');
        const btnView3D = document.getElementById('btnView3D');
        const btnView2D = document.getElementById('btnView2D');
        const stepBox = document.getElementById('stepBox');
        const stepFormula = document.getElementById('stepFormula');
        const stepLossDiff = document.getElementById('stepLossDiff');

        function computePred(w1, w2) {
            return [
                X_mat[0][0] * w1 + X_mat[0][1] * w2,
                X_mat[1][0] * w1 + X_mat[1][1] * w2,
                X_mat[2][0] * w1 + X_mat[2][1] * w2,
            ];
        }

        function computeResiduals(w1, w2) {
            const yP = computePred(w1, w2);
            return [yP[0] - Y_vec[0], yP[1] - Y_vec[1], yP[2] - Y_vec[2]];
        }

        function computeLoss(w1, w2) {
            const e = computeResiduals(w1, w2);
            return (e[0]*e[0] + e[1]*e[1] + e[2]*e[2]) / 3.0;
        }

        function computeGrad(w1, w2) {
            const e = computeResiduals(w1, w2);
            // ∇ = (2/3) * X^T * e
            const g1 = (2.0 / 3.0) * (X_mat[0][0]*e[0] + X_mat[1][0]*e[1] + X_mat[2][0]*e[2]);
            const g2 = (2.0 / 3.0) * (X_mat[0][1]*e[0] + X_mat[1][1]*e[1] + X_mat[2][1]*e[2]);
            return [g1, g2];
        }

        // Inicialización de Gráficas con Plotly.js (WebGL)
        function initPlot3D() {
            const lossCurr = computeLoss(curW1, curW2);

            const surfaceTrace = {
                type: 'surface',
                x: gridW1,
                y: gridW2,
                z: gridZ,
                colorscale: 'Viridis',
                opacity: 0.85,
                showscale: false,
                contours: {
                    z: { show: true, usecolormap: true, project: { z: true } }
                },
                hovertemplate: 'w1: %{x:.2f}<br>w2: %{y:.2f}<br>L: %{z:.2f}<extra></extra>'
            };

            const optTrace = {
                type: 'scatter3d',
                x: [W_opt[0]],
                y: [W_opt[1]],
                z: [0.0],
                mode: 'markers+text',
                name: 'Óptimo W*',
                marker: { size: 6, color: '#38BDF8', symbol: 'diamond' },
                text: ['W*'],
                textposition: 'top center',
                hoverinfo: 'skip'
            };

            const trajTrace = {
                type: 'scatter3d',
                x: trajectory.map(p => p[0]),
                y: trajectory.map(p => p[1]),
                z: trajectory.map(p => p[2]),
                mode: 'lines+markers',
                name: 'Trayectoria',
                line: { color: '#10B981', width: 4 },
                marker: { size: 3, color: '#10B981' },
                hoverinfo: 'skip'
            };

            const pointTrace = {
                type: 'scatter3d',
                x: [curW1],
                y: [curW2],
                z: [lossCurr],
                mode: 'markers',
                name: 'W actual',
                marker: { size: 8, color: '#EF4444', line: { color: '#FFFFFF', width: 2 } },
                hovertemplate: 'Actual:<br>w1: %{x:.2f}<br>w2: %{y:.2f}<br>L: %{z:.2f}<extra></extra>'
            };

            const layout = {
                template: 'plotly_dark',
                paper_bgcolor: '#09090B',
                scene: {
                    bgcolor: '#09090B',
                    xaxis: { title: { text: 'w1', font: { color: '#A1A1AA', size: 10 } }, gridcolor: '#1F1F24', tickfont: { size: 8 } },
                    yaxis: { title: { text: 'w2', font: { color: '#A1A1AA', size: 10 } }, gridcolor: '#1F1F24', tickfont: { size: 8 } },
                    zaxis: { title: { text: 'L(W)', font: { color: '#A1A1AA', size: 10 } }, gridcolor: '#1F1F24', tickfont: { size: 8 } },
                    camera: { eye: { x: 1.45, y: -1.55, z: 1.15 } }
                },
                margin: { l: 0, r: 0, t: 0, b: 0 },
                showlegend: false
            };

            Plotly.newPlot('plotDiv', [surfaceTrace, optTrace, trajTrace, pointTrace], layout, { displayModeBar: false, responsive: true });
        }

        function initPlot2D() {
            const lossCurr = computeLoss(curW1, curW2);

            const contourTrace = {
                type: 'contour',
                x: gridW1[0],
                y: gridW2.map(row => row[0]),
                z: gridZ,
                colorscale: 'Viridis',
                showscale: false,
                contours: { coloring: 'heatmap', showlabels: true, labelfont: { size: 8, color: '#F4F4F5' } },
                hovertemplate: 'w1: %{x:.2f}<br>w2: %{y:.2f}<br>L: %{z:.2f}<extra></extra>'
            };

            const optTrace = {
                type: 'scatter',
                x: [W_opt[0]],
                y: [W_opt[1]],
                mode: 'markers+text',
                name: 'Óptimo W*',
                marker: { size: 8, color: '#38BDF8', symbol: 'diamond' },
                text: ['W*'],
                textposition: 'top center',
                hoverinfo: 'skip'
            };

            const trajTrace = {
                type: 'scatter',
                x: trajectory.map(p => p[0]),
                y: trajectory.map(p => p[1]),
                mode: 'lines+markers',
                name: 'Trayectoria',
                line: { color: '#10B981', width: 2, dash: 'dot' },
                marker: { size: 4, color: '#10B981' },
                hoverinfo: 'skip'
            };

            const pointTrace = {
                type: 'scatter',
                x: [curW1],
                y: [curW2],
                mode: 'markers',
                name: 'W actual',
                marker: { size: 10, color: '#EF4444', line: { color: '#FFFFFF', width: 2 } },
                hoverinfo: 'skip'
            };

            const layout = {
                template: 'plotly_dark',
                paper_bgcolor: '#09090B',
                plot_bgcolor: '#09090B',
                xaxis: { title: { text: 'Peso w1', font: { color: '#A1A1AA', size: 10 } }, gridcolor: '#1F1F24', tickfont: { size: 8 } },
                yaxis: { title: { text: 'Peso w2', font: { color: '#A1A1AA', size: 10 } }, gridcolor: '#1F1F24', tickfont: { size: 8 } },
                margin: { l: 30, r: 10, t: 10, b: 30 },
                showlegend: false
            };

            Plotly.newPlot('plotDiv', [contourTrace, optTrace, trajTrace, pointTrace], layout, { displayModeBar: false, responsive: true });
        }

        // Actualización Instantánea (0 ms - 60 FPS)
        function updateUI() {
            const yp = computePred(curW1, curW2);
            const e = computeResiduals(curW1, curW2);
            const loss = computeLoss(curW1, curW2);
            const g = computeGrad(curW1, curW2);
            const gNorm = Math.sqrt(g[0]*g[0] + g[1]*g[1]);
            const distOpt = Math.sqrt((curW1 - W_opt[0])**2 + (curW2 - W_opt[1])**2);

            // 1. Textos y Sliders
            valW1.textContent = curW1.toFixed(2);
            valW2.textContent = curW2.toFixed(2);
            valAlpha.textContent = curAlpha.toFixed(3);
            mW1.textContent = curW1.toFixed(2);
            mW2.textContent = curW2.toFixed(2);

            // 2. Valores en la Matriz
            mYp1.textContent = yp[0].toFixed(2);
            mYp2.textContent = yp[1].toFixed(2);
            mYp3.textContent = yp[2].toFixed(2);
            mE1.textContent = (e[0] >= 0 ? '+' : '') + e[0].toFixed(2);
            mE2.textContent = (e[1] >= 0 ? '+' : '') + e[1].toFixed(2);
            mE3.textContent = (e[2] >= 0 ? '+' : '') + e[2].toFixed(2);

            // 3. Métricas
            metricMse.textContent = loss.toFixed(2);
            metricDist.textContent = 'dist: ' + distOpt.toFixed(2);
            metricGradNorm.textContent = gNorm.toFixed(2);
            metricGradShort.textContent = '[' + (g[0]>=0?'+':'') + g[0].toFixed(1) + ', ' + (g[1]>=0?'+':'') + g[1].toFixed(1) + ']';
            gradText.innerHTML = '∇ = [' + (g[0]>=0?'+':'') + g[0].toFixed(2) + ', ' + (g[1]>=0?'+':'') + g[1].toFixed(2) + ']ᵀ';

            // 4. Actualización ultrarrápida del punto en WebGL (sin redibujar la superficie)
            if (currentMode === '3D') {
                Plotly.restyle('plotDiv', {
                    x: [[curW1]],
                    y: [[curW2]],
                    z: [[loss]]
                }, [3]);

                // Actualizar línea de trayectoria
                Plotly.restyle('plotDiv', {
                    x: [trajectory.map(p => p[0])],
                    y: [trajectory.map(p => p[1])],
                    z: [trajectory.map(p => p[2])]
                }, [2]);
            } else {
                Plotly.restyle('plotDiv', {
                    x: [[curW1]],
                    y: [[curW2]]
                }, [3]);

                Plotly.restyle('plotDiv', {
                    x: [trajectory.map(p => p[0])],
                    y: [trajectory.map(p => p[1])]
                }, [2]);
            }
        }

        // Eventos de sliders (reactivos en vivo a 60 FPS)
        sliderW1.addEventListener('input', (e) => {
            curW1 = parseFloat(e.target.value);
            trajectory = [[curW1, curW2, computeLoss(curW1, curW2)]];
            stepBox.style.display = 'none';
            updateUI();
        });

        sliderW2.addEventListener('input', (e) => {
            curW2 = parseFloat(e.target.value);
            trajectory = [[curW1, curW2, computeLoss(curW1, curW2)]];
            stepBox.style.display = 'none';
            updateUI();
        });

        sliderAlpha.addEventListener('input', (e) => {
            curAlpha = parseFloat(e.target.value);
            valAlpha.textContent = curAlpha.toFixed(3);
        });

        // Alternar vista 3D / 2D
        btnView3D.addEventListener('click', () => {
            if (currentMode !== '3D') {
                currentMode = '3D';
                btnView3D.classList.add('active');
                btnView2D.classList.remove('active');
                initPlot3D();
                updateUI();
            }
        });

        btnView2D.addEventListener('click', () => {
            if (currentMode !== '2D') {
                currentMode = '2D';
                btnView2D.classList.add('active');
                btnView3D.classList.remove('active');
                initPlot2D();
                updateUI();
            }
        });

        // Ejecutar paso con animación fluida
        btnStepMat.addEventListener('click', () => {
            const g = computeGrad(curW1, curW2);
            const lBefore = computeLoss(curW1, curW2);
            const nextW1 = curW1 - curAlpha * g[0];
            const nextW2 = curW2 - curAlpha * g[1];
            const lAfter = computeLoss(nextW1, nextW2);

            stepBox.style.display = 'block';
            stepFormula.innerHTML = `W_nuevo = [${curW1.toFixed(2)}, ${curW2.toFixed(2)}]ᵀ - ${curAlpha.toFixed(2)} &times; ∇ = <code>[${nextW1.toFixed(2)}, ${nextW2.toFixed(2)}]ᵀ</code>`;
            const dL = lAfter - lBefore;
            stepLossDiff.innerHTML = `L: <b>${lBefore.toFixed(2)}</b> &rarr; <b>${lAfter.toFixed(2)}</b> (ΔL: <span style="color: ${dL <= 0 ? '#10B981' : '#EF4444'};">${dL >= 0 ? '+' : ''}${dL.toFixed(2)}</span>)`;

            // Animación suave de 180ms
            const startW1 = curW1;
            const startW2 = curW2;
            const startTime = performance.now();
            const dur = 180;

            function animate(now) {
                const elapsed = now - startTime;
                const progress = Math.min(elapsed / dur, 1);
                const ease = 1 - Math.pow(1 - progress, 3);
                curW1 = startW1 + (nextW1 - startW1) * ease;
                curW2 = startW2 + (nextW2 - startW2) * ease;
                sliderW1.value = curW1;
                sliderW2.value = curW2;
                updateUI();

                if (progress < 1) {
                    requestAnimationFrame(animate);
                } else {
                    curW1 = nextW1;
                    curW2 = nextW2;
                    sliderW1.value = curW1;
                    sliderW2.value = curW2;
                    trajectory.push([curW1, curW2, lAfter]);
                    updateUI();
                }
            }
            requestAnimationFrame(animate);
        });

        // Reiniciar
        btnResetMat.addEventListener('click', () => {
            curW1 = 0.50;
            curW2 = 1.00;
            curAlpha = 0.05;
            sliderW1.value = curW1;
            sliderW2.value = curW2;
            sliderAlpha.value = curAlpha;
            trajectory = [[curW1, curW2, computeLoss(curW1, curW2)]];
            stepBox.style.display = 'none';
            updateUI();
        });

        // Inicializar gráfico 3D y estado
        initPlot3D();
        updateUI();
    </script>
</body>
</html>
"""


def render_multivariable_lab(st):
    """Renderiza el laboratorio 2 con Plotly.js en el cliente (0 ms latencia, 60 FPS)."""
    X, Y, W_opt = get_default_dataset()
    w1_opt = float(W_opt[0, 0])
    w2_opt = float(W_opt[1, 0])

    # Precomputamos la malla de superficie de pérdida una sola vez en Python
    W1, W2, Z = compute_loss_surface(X, Y, w1_center=w1_opt, w2_center=w2_opt, span=4.0, resolution=35)

    rendered_html = (
        HTML_MULTIVARIABLE_TEMPLATE
        .replace("__INIT_W1__", "0.50")
        .replace("__INIT_W2__", "1.00")
        .replace("__INIT_ALPHA__", "0.050")
        .replace("__GRID_W1__", json.dumps(W1.tolist()))
        .replace("__GRID_W2__", json.dumps(W2.tolist()))
        .replace("__GRID_Z__", json.dumps(Z.tolist()))
    )

    # Altura ajustada con margen para evitar cortes
    components.html(rendered_html, height=590, scrolling=False)
