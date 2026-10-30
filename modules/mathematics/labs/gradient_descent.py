"""
Laboratorio interactivo: Descenso del gradiente con un solo peso.

Modelo lineal:
    ŷ = w * x

Función de pérdida (MSE para una observación):
    L(w) = (w * x - y)^2

Gradiente respecto a w:
    dL/dw = 2 * (w * x - y) * x

Regla de actualización:
    w_nuevo = w_actual - alpha * (dL/dw)
"""

from typing import Tuple
import numpy as np
import streamlit.components.v1 as components


# ==========================================
# CÁLCULOS MATEMÁTICOS PUROS (Desacoplados)
# ==========================================

def predict(w: float, x: float) -> float:
    """Calcula la predicción del modelo: ŷ = w * x."""
    return float(w * x)


def compute_loss(w: float, x: float, y: float) -> float:
    """Calcula la pérdida cuadrática: L(w) = (w * x - y)^2."""
    error = predict(w, x) - y
    return float(error ** 2)


def compute_gradient(w: float, x: float, y: float) -> float:
    """
    Calcula la derivada analítica: dL/dw = 2 * (w * x - y) * x.
    Representa la pendiente de la función de pérdida en el punto w.
    """
    error = predict(w, x) - y
    return float(2.0 * error * x)


def gradient_descent_step(
    w: float, x: float, y: float, alpha: float
) -> Tuple[float, float, float, float]:
    """
    Ejecuta un único paso de actualización según la regla del gradiente.
    """
    loss_before = compute_loss(w, x, y)
    grad = compute_gradient(w, x, y)
    w_new = float(w - alpha * grad)
    loss_after = compute_loss(w_new, x, y)
    return w_new, grad, loss_before, loss_after


def get_loss_curve(
    x: float, y: float, w_min: float = -1.0, w_max: float = 11.0, points: int = 180
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera los vectores de valores (w, L(w)) para graficar la curva analítica."""
    w_vals = np.linspace(w_min, w_max, points)
    loss_vals = (w_vals * x - y) ** 2
    return w_vals, loss_vals


def get_tangent_line(
    w: float, x: float, y: float, delta: float = 1.8
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula la recta tangente en w."""
    m = compute_gradient(w, x, y)
    loss_w = compute_loss(w, x, y)
    w_range = np.array([w - delta, w + delta])
    loss_range = loss_w + m * (w_range - w)
    return w_range, loss_range


# ==========================================
# INTERFAZ CLIENTE COMPACTA 60FPS (LINEAR STYLE)
# ==========================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
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
        .app-shell {
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-width: 100%;
        }
        
        /* Barra superior de estado */
        .top-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #1F1F24;
            padding-bottom: 10px;
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

        /* Grid balanceado en dos columnas */
        .workspace-grid {
            display: grid;
            grid-template-columns: 1.35fr 1fr;
            gap: 16px;
            align-items: start;
        }

        /* Columna Izquierda: Canvas y Análisis */
        .left-col {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .canvas-card {
            background: #0D0D11;
            border: 1px solid #1F1F24;
            border-radius: 8px;
            padding: 10px;
            position: relative;
        }
        canvas {
            width: 100%;
            height: 280px;
            display: block;
            cursor: crosshair;
        }
        .canvas-legend {
            display: flex;
            align-items: center;
            gap: 14px;
            font-size: 0.72rem;
            color: #A1A1AA;
            margin-top: 8px;
            padding-top: 6px;
            border-top: 1px solid #18181B;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        .legend-line {
            width: 14px;
            height: 2px;
        }

        .analysis-box {
            background: #121216;
            border: 1px solid #1F1F24;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 0.82rem;
            color: #D4D4D8;
            line-height: 1.45;
            min-height: 62px;
        }
        .analysis-box code {
            font-family: ui-monospace, monospace;
            background: #18181B;
            padding: 1px 5px;
            border-radius: 3px;
            color: #F4F4F5;
        }

        /* Columna Derecha: Controles, Métricas y Desglose */
        .right-col {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .panel-card {
            background: #121216;
            border: 1px solid #1F1F24;
            border-radius: 8px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .control-row {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .control-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.78rem;
            color: #A1A1AA;
            font-weight: 500;
        }
        .control-val {
            font-family: ui-monospace, monospace;
            font-size: 0.78rem;
            color: #F4F4F5;
            background: #18181B;
            padding: 1px 6px;
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

        .buttons-row {
            display: grid;
            grid-template-columns: 1.3fr 1fr;
            gap: 8px;
        }
        button {
            font-family: inherit;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 7px 10px;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .btn-primary {
            background: #F4F4F5;
            color: #09090B;
            border: 1px solid #F4F4F5;
        }
        .btn-primary:hover {
            background: #FFFFFF;
            box-shadow: 0 0 8px rgba(255, 255, 255, 0.2);
        }
        .btn-secondary {
            background: #18181B;
            color: #D4D4D8;
            border: 1px solid #27272A;
        }
        .btn-secondary:hover {
            border-color: #3F3F46;
            color: #FFFFFF;
        }

        /* Métricas 2x2 */
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .metric-cell {
            background: #0D0D11;
            border: 1px solid #1F1F24;
            border-radius: 6px;
            padding: 8px 10px;
        }
        .metric-label {
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #71717A;
            font-weight: 600;
        }
        .metric-value {
            font-family: ui-monospace, monospace;
            font-size: 1.15rem;
            font-weight: 600;
            color: #F4F4F5;
            margin-top: 1px;
        }
        .metric-sub {
            font-size: 0.7rem;
            color: #71717A;
            font-family: ui-monospace, monospace;
        }

        /* Detalle del paso */
        .step-panel {
            background: #0D0D11;
            border: 1px solid #1F1F24;
            border-left: 3px solid #3B82F6;
            border-radius: 6px;
            padding: 8px 12px;
            display: none;
        }
        .step-title {
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #71717A;
            font-weight: 600;
        }
        .step-code {
            font-family: ui-monospace, monospace;
            font-size: 0.8rem;
            color: #E4E4E7;
            margin: 4px 0;
            line-height: 1.35;
        }
        .step-stats {
            display: flex;
            gap: 14px;
            font-size: 0.72rem;
            color: #71717A;
            border-top: 1px solid #18181B;
            padding-top: 4px;
            margin-top: 4px;
        }
        .step-stats b {
            color: #D4D4D8;
        }
    </style>
</head>
<body>
    <div class="app-shell">
        <!-- Barra de estado -->
        <div class="top-bar">
            <div class="top-left">
                <span class="title">Descenso del Gradiente</span>
                <span class="badge">1 Peso</span>
                <span class="badge">ŷ = X · W</span>
            </div>
            <span class="badge badge-opt">Óptimo w* = 5.00</span>
        </div>

        <!-- Visor de Matrices Interactivo (1D) -->
        <div class="matrix-board">
            <!-- Matriz X -->
            <div class="matrix-box">
                <div class="matrix-label">Matriz X (1&times;1)</div>
                <div class="matrix-row"><span>__X__</span></div>
            </div>

            <div class="matrix-op">&times;</div>

            <!-- Vector W -->
            <div class="matrix-box active-w">
                <div class="matrix-label" style="color: #3B82F6;">W (1&times;1)</div>
                <div class="matrix-row"><span id="mW" style="color: #38BDF8; font-weight: 600;">__INIT_W__</span></div>
            </div>

            <div class="matrix-op">=</div>

            <!-- Vector ŷ -->
            <div class="matrix-box">
                <div class="matrix-label">Pred ŷ (1&times;1)</div>
                <div class="matrix-row"><span id="mYp">2.00</span></div>
            </div>

            <div class="matrix-op">vs</div>

            <!-- Vector Y -->
            <div class="matrix-box">
                <div class="matrix-label">Obj Y (1&times;1)</div>
                <div class="matrix-row"><span>__Y__</span></div>
            </div>

            <div class="matrix-op">&rarr;</div>

            <!-- Vector E -->
            <div class="matrix-box">
                <div class="matrix-label" style="color: #F87171;">Error E (1&times;1)</div>
                <div class="matrix-row"><span id="mE" style="color: #F87171;">-8.00</span></div>
            </div>
        </div>

        <!-- Disposición en 2 Columnas -->
        <div class="workspace-grid">
            <!-- Izquierda: Gráfico y Lectura -->
            <div class="left-col">
                <div class="canvas-card">
                    <canvas id="lossCanvas"></canvas>
                    <div class="canvas-legend">
                        <div class="legend-item">
                            <span class="legend-line" style="background: #3B82F6;"></span>
                            <span>L(w)</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-line" style="background: #F59E0B; border-top: 1px dashed #F59E0B;"></span>
                            <span id="tangentLegendText">Tangente dL/dw</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-dot" style="background: #EF4444; border: 1px solid #FFF;"></span>
                            <span>Posición w</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-line" style="background: #52525B; border-top: 1px dotted #52525B;"></span>
                            <span>w* = 5.0</span>
                        </div>
                    </div>
                </div>

                <div class="analysis-box" id="slopeAnalysis"></div>
            </div>

            <!-- Derecha: Parámetros, Métricas y Actualización -->
            <div class="right-col">
                <div class="panel-card">
                    <div class="control-row">
                        <div class="control-label">
                            <span>Peso actual (w)</span>
                            <span class="control-val" id="wDisplay">__INIT_W__</span>
                        </div>
                        <input type="range" id="wSlider" min="-1.5" max="11.5" step="0.02" value="__INIT_W__">
                    </div>

                    <div class="control-row">
                        <div class="control-label">
                            <span>Tasa de aprendizaje (α)</span>
                            <span class="control-val" id="alphaDisplay">__INIT_ALPHA__</span>
                        </div>
                        <input type="range" id="alphaSlider" min="0.005" max="0.25" step="0.005" value="__INIT_ALPHA__">
                    </div>

                    <div class="buttons-row">
                        <button class="btn-primary" id="btnStep">Ejecutar paso</button>
                        <button class="btn-secondary" id="btnReset">Reiniciar</button>
                    </div>

                    <!-- Métricas 2x2 integradas en el panel -->
                    <div class="metrics-grid">
                        <div class="metric-cell">
                            <div class="metric-label">Peso actual (w)</div>
                            <div class="metric-value" id="metricW">__INIT_W__</div>
                            <div class="metric-sub" id="metricWDelta">-4.00 a w*</div>
                        </div>
                        <div class="metric-cell">
                            <div class="metric-label">Predicción (ŷ)</div>
                            <div class="metric-value" id="metricYPred">2.00</div>
                            <div class="metric-sub" id="metricError">error: -8.00</div>
                        </div>
                        <div class="metric-cell">
                            <div class="metric-label">Pérdida L(w)</div>
                            <div class="metric-value" id="metricLoss">64.00</div>
                            <div class="metric-sub" id="metricLossFormula">(2.0 - 10)²</div>
                        </div>
                        <div class="metric-cell">
                            <div class="metric-label">Derivada dL/dw</div>
                            <div class="metric-value" id="metricGrad">-32.00</div>
                            <div class="metric-sub" id="metricSlopeSign">Pendiente -</div>
                        </div>
                    </div>

                    <!-- Desglose compacto del paso -->
                    <div class="step-panel" id="stepBreakdown">
                        <div class="step-title">Última actualización</div>
                        <div class="step-code" id="formulaDetail">
                            w_nuevo = w - α &times; (dL/dw)
                        </div>
                        <div class="step-stats" id="stepSummary"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const x_obs = __X__;
        const y_obs = __Y__;
        const w_opt = y_obs / x_obs;

        let currentW = __INIT_W__;
        let currentAlpha = __INIT_ALPHA__;
        let lastStep = null;

        const canvas = document.getElementById('lossCanvas');
        const ctx = canvas.getContext('2d');
        const wSlider = document.getElementById('wSlider');
        const alphaSlider = document.getElementById('alphaSlider');
        const wDisplay = document.getElementById('wDisplay');
        const alphaDisplay = document.getElementById('alphaDisplay');
        const btnStep = document.getElementById('btnStep');
        const btnReset = document.getElementById('btnReset');

        const mW = document.getElementById('mW');
        const mYp = document.getElementById('mYp');
        const mE = document.getElementById('mE');

        const metricW = document.getElementById('metricW');
        const metricWDelta = document.getElementById('metricWDelta');
        const metricYPred = document.getElementById('metricYPred');
        const metricError = document.getElementById('metricError');
        const metricLoss = document.getElementById('metricLoss');
        const metricLossFormula = document.getElementById('metricLossFormula');
        const metricGrad = document.getElementById('metricGrad');
        const metricSlopeSign = document.getElementById('metricSlopeSign');
        const slopeAnalysis = document.getElementById('slopeAnalysis');
        const tangentLegendText = document.getElementById('tangentLegendText');
        const stepBreakdown = document.getElementById('stepBreakdown');
        const formulaDetail = document.getElementById('formulaDetail');
        const stepSummary = document.getElementById('stepSummary');

        function resizeCanvas() {
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
        }

        window.addEventListener('resize', () => {
            resizeCanvas();
            draw();
        });

        function computeLoss(w) {
            const err = w * x_obs - y_obs;
            return err * err;
        }

        function computeGrad(w) {
            return 2 * (w * x_obs - y_obs) * x_obs;
        }

        function draw() {
            const rect = canvas.getBoundingClientRect();
            const width = rect.width;
            const height = rect.height;

            ctx.clearRect(0, 0, width, height);

            const padL = 44;
            const padR = 20;
            const padT = 20;
            const padB = 36;

            const plotW = width - padL - padR;
            const plotH = height - padT - padB;

            const margin = Math.max(Math.abs(currentW - w_opt) * 1.55, 4.5);
            const wMin = w_opt - margin;
            const wMax = w_opt + margin;

            const lossAtW = computeLoss(currentW);
            const maxLoss = Math.max(lossAtW * 1.35, 45);
            const minLoss = -maxLoss * 0.05;

            function toCanvasX(w) {
                return padL + ((w - wMin) / (wMax - wMin)) * plotW;
            }
            function toCanvasY(loss) {
                return padT + (1 - (loss - minLoss) / (maxLoss - minLoss)) * plotH;
            }

            // Cuadrícula
            ctx.strokeStyle = '#18181B';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 4]);

            for (let i = 0; i <= 6; i++) {
                const wGrid = wMin + (i / 6) * (wMax - wMin);
                const gx = toCanvasX(wGrid);
                ctx.beginPath();
                ctx.moveTo(gx, padT);
                ctx.lineTo(gx, padT + plotH);
                ctx.stroke();

                ctx.fillStyle = '#71717A';
                ctx.font = '10px ui-monospace, monospace';
                ctx.textAlign = 'center';
                ctx.fillText(wGrid.toFixed(1), gx, padT + plotH + 16);
            }

            for (let j = 0; j <= 4; j++) {
                const lGrid = (j / 4) * maxLoss;
                const gy = toCanvasY(lGrid);
                ctx.beginPath();
                ctx.moveTo(padL, gy);
                ctx.lineTo(padL + plotW, gy);
                ctx.stroke();

                ctx.fillStyle = '#71717A';
                ctx.font = '10px ui-monospace, monospace';
                ctx.textAlign = 'right';
                ctx.fillText(lGrid.toFixed(0), padL - 8, gy + 3);
            }
            ctx.setLineDash([]);

            // Óptimo w*
            const optX = toCanvasX(w_opt);
            ctx.strokeStyle = '#3F3F46';
            ctx.lineWidth = 1.2;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(optX, padT);
            ctx.lineTo(optX, padT + plotH);
            ctx.stroke();
            ctx.setLineDash([]);

            // Curva L(w)
            ctx.strokeStyle = '#3B82F6';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            const steps = 140;
            for (let i = 0; i <= steps; i++) {
                const w = wMin + (i / steps) * (wMax - wMin);
                const l = computeLoss(w);
                const cx = toCanvasX(w);
                const cy = toCanvasY(l);
                if (i === 0) ctx.moveTo(cx, cy);
                else ctx.lineTo(cx, cy);
            }
            ctx.stroke();

            // Tangente
            const grad = computeGrad(currentW);
            const deltaW = Math.max(margin * 0.22, 1.2);
            const w1 = currentW - deltaW;
            const w2 = currentW + deltaW;
            const l1 = lossAtW + grad * (w1 - currentW);
            const l2 = lossAtW + grad * (w2 - currentW);

            ctx.strokeStyle = '#F59E0B';
            ctx.lineWidth = 1.8;
            ctx.setLineDash([5, 4]);
            ctx.beginPath();
            ctx.moveTo(toCanvasX(w1), toCanvasY(l1));
            ctx.lineTo(toCanvasX(w2), toCanvasY(l2));
            ctx.stroke();
            ctx.setLineDash([]);

            // Flecha de paso
            if (lastStep) {
                const xPrev = toCanvasX(lastStep.w_prev);
                const yPrev = toCanvasY(lastStep.loss_prev);
                const xNext = toCanvasX(lastStep.w_new);
                const yNext = toCanvasY(lastStep.loss_after);

                ctx.strokeStyle = '#10B981';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(xPrev, yPrev);
                ctx.lineTo(xNext, yNext);
                ctx.stroke();

                ctx.fillStyle = '#10B981';
                ctx.fillRect(xNext - 4, yNext - 4, 8, 8);
            }

            // Punto actual
            const ptX = toCanvasX(currentW);
            const ptY = toCanvasY(lossAtW);

            ctx.fillStyle = 'rgba(239, 68, 68, 0.25)';
            ctx.beginPath();
            ctx.arc(ptX, ptY, 9, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = '#EF4444';
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(ptX, ptY, 5.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            // Coordenadas
            ctx.fillStyle = '#F4F4F5';
            ctx.font = '10px ui-monospace, monospace';
            ctx.textAlign = 'center';
            ctx.fillText('(' + currentW.toFixed(2) + ', ' + lossAtW.toFixed(1) + ')', ptX, ptY - 12);
        }

        function updateState() {
            const yPred = currentW * x_obs;
            const loss = computeLoss(currentW);
            const grad = computeGrad(currentW);
            const diffOpt = currentW - w_opt;

            wDisplay.textContent = currentW.toFixed(2);
            alphaDisplay.textContent = currentAlpha.toFixed(3);

            if (mW) mW.textContent = currentW.toFixed(2);
            if (mYp) mYp.textContent = yPred.toFixed(2);
            const errVal = yPred - y_obs;
            if (mE) mE.textContent = (errVal >= 0 ? '+' : '') + errVal.toFixed(2);

            metricW.textContent = currentW.toFixed(2);
            metricWDelta.textContent = (diffOpt >= 0 ? '+' : '') + diffOpt.toFixed(2) + ' a w*';
            metricYPred.textContent = yPred.toFixed(2);
            metricError.textContent = 'error: ' + (yPred - y_obs >= 0 ? '+' : '') + (yPred - y_obs).toFixed(2);
            metricLoss.textContent = loss.toFixed(2);
            metricLossFormula.textContent = '(' + yPred.toFixed(2) + ' - ' + y_obs + ')²';
            metricGrad.textContent = (grad >= 0 ? '+' : '') + grad.toFixed(2);

            if (grad < -0.01) {
                metricSlopeSign.textContent = 'Pendiente -';
                slopeAnalysis.innerHTML = '<b>Pendiente negativa (dL/dw = ' + grad.toFixed(2) + ' &lt; 0):</b><br>' +
                    'La curva desciende hacia la derecha. Regla: <code>w_nuevo = w - α(' + grad.toFixed(2) + ') = w + |paso|</code>. ' +
                    'Al restar un negativo, <b>w aumenta</b> hacia el mínimo (w* = 5.0).';
            } else if (grad > 0.01) {
                metricSlopeSign.textContent = 'Pendiente +';
                slopeAnalysis.innerHTML = '<b>Pendiente positiva (dL/dw = +' + grad.toFixed(2) + ' &gt; 0):</b><br>' +
                    'La curva asciende hacia la derecha. Regla: <code>w_nuevo = w - α(+' + grad.toFixed(2) + ') = w - |paso|</code>. ' +
                    'Al restar un positivo, <b>w disminuye</b> hacia el mínimo (w* = 5.0).';
            } else {
                metricSlopeSign.textContent = 'Cero (Óptimo)';
                slopeAnalysis.innerHTML = '<b>Mínimo óptimo (dL/dw = 0.00):</b><br>' +
                    'La recta tangente es horizontal. ŷ = 10.0 y la pérdida es exactamente 0.00.';
            }

            tangentLegendText.textContent = 'Tangente (dL/dw = ' + (grad >= 0 ? '+' : '') + grad.toFixed(1) + ')';

            draw();
        }

        wSlider.addEventListener('input', (e) => {
            currentW = parseFloat(e.target.value);
            lastStep = null;
            stepBreakdown.style.display = 'none';
            updateState();
        });

        alphaSlider.addEventListener('input', (e) => {
            currentAlpha = parseFloat(e.target.value);
            alphaDisplay.textContent = currentAlpha.toFixed(3);
        });

        btnStep.addEventListener('click', () => {
            const grad = computeGrad(currentW);
            const lossBefore = computeLoss(currentW);
            const wNew = currentW - currentAlpha * grad;
            const lossAfter = computeLoss(wNew);

            lastStep = {
                w_prev: currentW,
                alpha: currentAlpha,
                grad: grad,
                w_new: wNew,
                loss_prev: lossBefore,
                loss_after: lossAfter
            };

            stepBreakdown.style.display = 'block';
            formulaDetail.innerHTML = 'w_nuevo = ' + lastStep.w_prev.toFixed(2) + ' - (' + lastStep.alpha.toFixed(3) + ') &times; (' + (lastStep.grad >= 0 ? '+' : '') + lastStep.grad.toFixed(2) + ') = <b style="color: #38BDF8;">' + lastStep.w_new.toFixed(2) + '</b>';

            const deltaL = lastStep.loss_after - lastStep.loss_prev;
            const dColor = deltaL <= 0 ? '#10B981' : '#EF4444';
            stepSummary.innerHTML = '<span>L_ant: <b>' + lastStep.loss_prev.toFixed(1) + '</b></span>' +
                '<span>L_desp: <b>' + lastStep.loss_after.toFixed(1) + '</b></span>' +
                '<span>ΔL: <b style="color: ' + dColor + ';">' + (deltaL >= 0 ? '+' : '') + deltaL.toFixed(1) + '</b></span>';

            const startW = currentW;
            const targetW = wNew;
            const startTime = performance.now();
            const duration = 200;

            function animateStep(now) {
                const elapsed = now - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const ease = 1 - Math.pow(1 - progress, 3);
                currentW = startW + (targetW - startW) * ease;
                wSlider.value = currentW;
                updateState();

                if (progress < 1) {
                    requestAnimationFrame(animateStep);
                } else {
                    currentW = targetW;
                    wSlider.value = currentW;
                    updateState();
                }
            }
            requestAnimationFrame(animateStep);
        });

        btnReset.addEventListener('click', () => {
            currentW = 1.00;
            currentAlpha = 0.05;
            wSlider.value = currentW;
            alphaSlider.value = currentAlpha;
            lastStep = null;
            stepBreakdown.style.display = 'none';
            updateState();
        });

        resizeCanvas();
        updateState();
    </script>
</body>
</html>
"""


def render_gradient_descent_lab(st, charts_module=None):
    """Renderiza el laboratorio con alto rendimiento y cero parpadeo."""
    rendered_html = (
        HTML_TEMPLATE.replace("__INIT_W__", "1.00")
        .replace("__INIT_ALPHA__", "0.050")
        .replace("__X__", "2.0")
        .replace("__Y__", "10.0")
    )
    # 590px permite que el visor de matrices y las tarjetas encajen holgadamente
    components.html(rendered_html, height=590, scrolling=False)
