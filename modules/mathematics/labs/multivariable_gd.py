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
import numpy as np
import plotly.graph_objects as go


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

    # W* exacto = [2.0, 3.0]
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
    N = X.shape[0]
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
    span: float = 3.5,
    resolution: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula una malla (grid) para la superficie de pérdida 3D:
        w1_grid, w2_grid -> Z (L(w1, w2))
    """
    w1_vals = np.linspace(w1_center - span, w1_center + span, resolution)
    w2_vals = np.linspace(w2_center - span, w2_center + span, resolution)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    Z = np.zeros_like(W1)
    N = X.shape[0]

    for i in range(resolution):
        for j in range(resolution):
            w_candidate = np.array([[W1[i, j]], [W2[i, j]]])
            err = X @ w_candidate - Y
            Z[i, j] = np.mean(err ** 2)

    return W1, W2, Z


# ==========================================
# VISUALIZACIÓN 3D Y DE CONTORNO CON PLOTLY
# ==========================================

def create_3d_loss_surface_chart(
    X: np.ndarray,
    Y: np.ndarray,
    W_current: np.ndarray,
    W_opt: np.ndarray,
    trajectory: Optional[List[Tuple[float, float, float]]] = None,
) -> go.Figure:
    """
    Genera la figura 3D interactiva del paraboloide de pérdida usando Plotly WebGL.
    Permite rotación 360°, zoom e inspección espacial.
    """
    w1_curr = float(W_current[0, 0])
    w2_curr = float(W_current[1, 0])
    loss_curr = compute_mse_loss(X, Y, W_current)

    w1_opt = float(W_opt[0, 0])
    w2_opt = float(W_opt[1, 0])

    span = max(abs(w1_curr - w1_opt) * 1.5, abs(w2_curr - w2_opt) * 1.5, 3.2)
    W1, W2, Z = compute_loss_surface(X, Y, w1_center=w1_opt, w2_center=w2_opt, span=span, resolution=42)

    fig = go.Figure()

    # 1. Superficie paraboloide 3D
    fig.add_trace(
        go.Surface(
            x=W1,
            y=W2,
            z=Z,
            colorscale="Viridis",
            opacity=0.82,
            showscale=False,
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="#F4F4F5",
                    project=dict(z=True),  # Proyección de curvas de nivel en el plano base
                )
            ),
            hovertemplate="w1: %{x:.2f}<br>w2: %{y:.2f}<br>Pérdida L: %{z:.2f}<extra></extra>",
        )
    )

    # 2. Mínimo analítico global (fondo del cuenco)
    fig.add_trace(
        go.Scatter3d(
            x=[w1_opt],
            y=[w2_opt],
            z=[0.0],
            mode="markers+text",
            name="Mínimo W*",
            marker=dict(size=7, color="#38BDF8", symbol="diamond"),
            text=["W* (Óptimo)"],
            textposition="top center",
            textfont=dict(color="#38BDF8", size=11, family="monospace"),
            hoverinfo="skip",
        )
    )

    # 3. Trayectoria de pasos si existe historial
    if trajectory and len(trajectory) > 1:
        traj_w1 = [p[0] for p in trajectory]
        traj_w2 = [p[1] for p in trajectory]
        traj_loss = [p[2] for p in trajectory]

        fig.add_trace(
            go.Scatter3d(
                x=traj_w1,
                y=traj_w2,
                z=traj_loss,
                mode="lines+markers",
                name="Trayectoria",
                line=dict(color="#10B981", width=5),
                marker=dict(size=4, color="#10B981"),
                hoverinfo="skip",
            )
        )

    # 4. Posición actual (w1, w2, L)
    fig.add_trace(
        go.Scatter3d(
            x=[w1_curr],
            y=[w2_curr],
            z=[loss_curr],
            mode="markers+text",
            name="Posición actual W",
            marker=dict(
                size=9,
                color="#EF4444",
                symbol="circle",
                line=dict(color="#FFFFFF", width=2),
            ),
            text=[f"W = [{w1_curr:.2f}, {w2_curr:.2f}]"],
            textposition="top center",
            textfont=dict(color="#F4F4F5", size=11, family="monospace"),
            hovertemplate=f"Actual:<br>w1: {w1_curr:.3f}<br>w2: {w2_curr:.3f}<br>L(W): {loss_curr:.3f}<extra></extra>",
        )
    )

    # Configuración de diseño y cámara oscura tipo Linear / Vercel
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#09090B",
        scene=dict(
            bgcolor="#09090B",
            xaxis=dict(
                title=dict(text="Peso w1", font=dict(color="#A1A1AA", size=11)),
                gridcolor="#1F1F24",
                zerolinecolor="#27272A",
                backgroundcolor="#09090B",
                showbackground=False,
                tickfont=dict(color="#71717A", size=9),
            ),
            yaxis=dict(
                title=dict(text="Peso w2", font=dict(color="#A1A1AA", size=11)),
                gridcolor="#1F1F24",
                zerolinecolor="#27272A",
                backgroundcolor="#09090B",
                showbackground=False,
                tickfont=dict(color="#71717A", size=9),
            ),
            zaxis=dict(
                title=dict(text="Pérdida L(W)", font=dict(color="#A1A1AA", size=11)),
                gridcolor="#1F1F24",
                zerolinecolor="#27272A",
                backgroundcolor="#09090B",
                showbackground=False,
                tickfont=dict(color="#71717A", size=9),
            ),
            camera=dict(
                eye=dict(x=1.5, y=-1.6, z=1.2),
            ),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=390,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#A1A1AA", size=10),
        ),
    )

    return fig


def create_contour_loss_chart(
    X: np.ndarray,
    Y: np.ndarray,
    W_current: np.ndarray,
    W_opt: np.ndarray,
    grad_current: np.ndarray,
    trajectory: Optional[List[Tuple[float, float, float]]] = None,
) -> go.Figure:
    """
    Genera el mapa de curvas de nivel 2D (visión cenital) del paraboloide,
    mostrando el vector gradiente ∇L perpendicular a las líneas de nivel.
    """
    w1_curr = float(W_current[0, 0])
    w2_curr = float(W_current[1, 0])
    w1_opt = float(W_opt[0, 0])
    w2_opt = float(W_opt[1, 0])

    span = max(abs(w1_curr - w1_opt) * 1.5, abs(w2_curr - w2_opt) * 1.5, 3.2)
    W1, W2, Z = compute_loss_surface(X, Y, w1_center=w1_opt, w2_center=w2_opt, span=span, resolution=50)

    fig = go.Figure()

    # 1. Curvas de nivel
    fig.add_trace(
        go.Contour(
            x=W1[0, :],
            y=W2[:, 0],
            z=Z,
            colorscale="Viridis",
            contours=dict(
                coloring="heatmap",
                showlabels=True,
                labelfont=dict(size=9, color="#F4F4F5"),
            ),
            showscale=False,
            hovertemplate="w1: %{x:.2f}<br>w2: %{y:.2f}<br>L(W): %{z:.2f}<extra></extra>",
        )
    )

    # 2. Óptimo
    fig.add_trace(
        go.Scatter(
            x=[w1_opt],
            y=[w2_opt],
            mode="markers+text",
            name="Óptimo W*",
            marker=dict(size=10, color="#38BDF8", symbol="diamond"),
            text=["W* = [2, 3]"],
            textposition="top center",
            textfont=dict(color="#38BDF8", size=11, family="monospace"),
            hoverinfo="skip",
        )
    )

    # 3. Trayectoria histórica
    if trajectory and len(trajectory) > 1:
        traj_w1 = [p[0] for p in trajectory]
        traj_w2 = [p[1] for p in trajectory]
        fig.add_trace(
            go.Scatter(
                x=traj_w1,
                y=traj_w2,
                mode="lines+markers",
                name="Trayectoria",
                line=dict(color="#10B981", width=2.5, dash="dot"),
                marker=dict(size=5, color="#10B981"),
                hoverinfo="skip",
            )
        )

    # 4. Vector gradiente (flecha que apunta hacia el máximo ascenso)
    g1 = float(grad_current[0, 0])
    g2 = float(grad_current[1, 0])
    g_norm = np.sqrt(g1**2 + g2**2)
    if g_norm > 1e-4:
        # Escalamos la flecha visualmente para que sea legible en la gráfica
        arrow_len = min(span * 0.22, 1.0)
        u1 = (g1 / g_norm) * arrow_len
        u2 = (g2 / g_norm) * arrow_len

        fig.add_annotation(
            x=w1_curr + u1,
            y=w2_curr + u2,
            ax=w1_curr,
            ay=w2_curr,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.3,
            arrowwidth=2.2,
            arrowcolor="#F59E0B",
            text="∇L",
            font=dict(color="#F59E0B", size=12, family="monospace"),
        )

    # 5. Punto actual
    fig.add_trace(
        go.Scatter(
            x=[w1_curr],
            y=[w2_curr],
            mode="markers+text",
            name="W actual",
            marker=dict(
                size=12,
                color="#EF4444",
                line=dict(color="#FFFFFF", width=2),
            ),
            text=[f"W = [{w1_curr:.2f}, {w2_curr:.2f}]"],
            textposition="bottom center",
            textfont=dict(color="#F4F4F5", size=10, family="monospace"),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#09090B",
        margin=dict(l=35, r=15, t=20, b=35),
        height=390,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color="#A1A1AA", size=10),
        ),
        xaxis=dict(
            title=dict(text="Peso w1", font=dict(color="#A1A1AA", size=11)),
            gridcolor="#1F1F24",
            zerolinecolor="#27272A",
            tickfont=dict(color="#71717A", size=9),
        ),
        yaxis=dict(
            title=dict(text="Peso w2", font=dict(color="#A1A1AA", size=11)),
            gridcolor="#1F1F24",
            zerolinecolor="#27272A",
            tickfont=dict(color="#71717A", size=9),
        ),
    )

    return fig


# ==========================================
# INTERFAZ STREAMLIT DEL LABORATORIO 2
# ==========================================

def render_multivariable_lab(st):
    """Renderiza el laboratorio con el visor de matrices y gráfico 3D/Contorno."""

    # Inicializar estado
    X, Y, W_opt = get_default_dataset()

    if "multi_w1" not in st.session_state:
        st.session_state.multi_w1 = 0.5
    if "multi_w2" not in st.session_state:
        st.session_state.multi_w2 = 1.0
    if "multi_alpha" not in st.session_state:
        st.session_state.multi_alpha = 0.05
    if "multi_history" not in st.session_state:
        st.session_state.multi_history = []
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "3D (Superficie)"

    w1 = float(st.session_state.multi_w1)
    w2 = float(st.session_state.multi_w2)
    alpha = float(st.session_state.multi_alpha)
    W = np.array([[w1], [w2]], dtype=float)

    # Cálculos en tiempo real
    Y_pred = predict_matrix(X, W)
    residuals = compute_residuals(X, Y, W)
    mse_loss = compute_mse_loss(X, Y, W)
    grad_vector = compute_gradient_vector(X, Y, W)

    # 1. Barra de encabezado
    st.markdown(
        """
        <div style="display: flex; align-items: baseline; justify-content: space-between; border-bottom: 1px solid #1F1F24; padding-bottom: 8px; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <h2 style="margin: 0; font-size: 1.1rem; font-weight: 600; color: #F4F4F5;">
                    Matrices y Superficie 3D de Pérdida
                </h2>
                <span style="font-family: monospace; font-size: 0.7rem; color: #A1A1AA; border: 1px solid #27272A; padding: 1px 6px; border-radius: 4px;">2 pesos</span>
            </div>
            <div style="font-family: monospace; font-size: 0.78rem; color: #38BDF8; background: #082F49; border: 1px solid #0369A1; padding: 2px 8px; border-radius: 4px;">
                W* óptimo = [2.00, 3.00]ᵀ
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Visor de Matrices Interactivo (Álgebra Lineal visual)
    st.markdown(
        """
        <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: #71717A; font-weight: 600; margin-bottom: 6px;">
            Visor de Álgebra Matricial: ŷ = X · W
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabla visual estilizada para las matrices
    c_x, c_mul, c_w, c_eq, c_ypred, c_y, c_err = st.columns([2.4, 0.4, 1.4, 0.4, 1.4, 1.4, 1.4])

    with c_x:
        st.markdown(
            f"""
            <div style="background: #121216; border: 1px solid #27272A; border-radius: 6px; padding: 8px 10px; font-family: monospace; font-size: 0.8rem;">
                <div style="color: #71717A; font-size: 0.68rem; margin-bottom: 4px;">Matriz X (3x2)</div>
                <div style="color: #E4E4E7;">[{X[0,0]:.1f}, {X[0,1]:.1f}]</div>
                <div style="color: #E4E4E7;">[{X[1,0]:.1f}, {X[1,1]:.1f}]</div>
                <div style="color: #E4E4E7;">[{X[2,0]:.1f}, {X[2,1]:.1f}]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_mul:
        st.markdown("<div style='text-align: center; font-size: 1.3rem; color: #71717A; padding-top: 24px;'>&times;</div>", unsafe_allow_html=True)

    with c_w:
        st.markdown(
            f"""
            <div style="background: #121216; border: 1px solid #3B82F6; border-radius: 6px; padding: 8px 10px; font-family: monospace; font-size: 0.8rem;">
                <div style="color: #3B82F6; font-size: 0.68rem; font-weight: 600; margin-bottom: 4px;">W (2x1)</div>
                <div style="color: #38BDF8; font-weight: 600;">[{w1:.2f}]</div>
                <div style="color: #38BDF8; font-weight: 600;">[{w2:.2f}]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_eq:
        st.markdown("<div style='text-align: center; font-size: 1.3rem; color: #71717A; padding-top: 24px;'>=</div>", unsafe_allow_html=True)

    with c_ypred:
        st.markdown(
            f"""
            <div style="background: #121216; border: 1px solid #27272A; border-radius: 6px; padding: 8px 10px; font-family: monospace; font-size: 0.8rem;">
                <div style="color: #71717A; font-size: 0.68rem; margin-bottom: 4px;">Pred ŷ (3x1)</div>
                <div style="color: #E4E4E7;">[{Y_pred[0,0]:.2f}]</div>
                <div style="color: #E4E4E7;">[{Y_pred[1,0]:.2f}]</div>
                <div style="color: #E4E4E7;">[{Y_pred[2,0]:.2f}]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_y:
        st.markdown(
            f"""
            <div style="background: #121216; border: 1px solid #27272A; border-radius: 6px; padding: 8px 10px; font-family: monospace; font-size: 0.8rem;">
                <div style="color: #71717A; font-size: 0.68rem; margin-bottom: 4px;">Obj Y (3x1)</div>
                <div style="color: #A1A1AA;">[{Y[0,0]:.1f}]</div>
                <div style="color: #A1A1AA;">[{Y[1,0]:.1f}]</div>
                <div style="color: #A1A1AA;">[{Y[2,0]:.1f}]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_err:
        st.markdown(
            f"""
            <div style="background: #121216; border: 1px solid #27272A; border-radius: 6px; padding: 8px 10px; font-family: monospace; font-size: 0.8rem;">
                <div style="color: #71717A; font-size: 0.68rem; margin-bottom: 4px;">Error E (3x1)</div>
                <div style="color: #F87171;">[{residuals[0,0]:+.2f}]</div>
                <div style="color: #F87171;">[{residuals[1,0]:+.2f}]</div>
                <div style="color: #F87171;">[{residuals[2,0]:+.2f}]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 3. Disposición en dos columnas: Gráfico Espacial (izq) y Panel de Control (der)
    col_chart, col_controls = st.columns([1.5, 1.0], gap="medium")

    with col_controls:
        # Selector de vista
        view_opt = st.radio(
            "Perspectiva geométrica:",
            options=["3D (Superficie)", "2D (Curvas de Nivel)"],
            horizontal=True,
            label_visibility="collapsed",
            key="radio_view_mode",
        )

        st.markdown(
            """
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: #71717A; font-weight: 600; margin: 4px 0;">
                Ajuste de Pesos
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Sliders de w1 y w2
        new_w1 = st.slider(
            "Peso w1",
            min_value=-2.0,
            max_value=6.0,
            value=float(w1),
            step=0.05,
            key="slider_w1",
            help="Modifica el peso w1 asignado a la primera característica."
        )
        new_w2 = st.slider(
            "Peso w2",
            min_value=-2.0,
            max_value=7.0,
            value=float(w2),
            step=0.05,
            key="slider_w2",
            help="Modifica el peso w2 asignado a la segunda característica."
        )
        new_alpha = st.slider(
            "Tasa de aprendizaje (α)",
            min_value=0.005,
            max_value=0.20,
            value=float(alpha),
            step=0.005,
            format="%.3f",
            key="slider_multi_alpha",
        )

        st.session_state.multi_w1 = new_w1
        st.session_state.multi_w2 = new_w2
        st.session_state.multi_alpha = new_alpha

        # Botones de acción
        b1, b2 = st.columns([1.3, 1.0])
        with b1:
            step_btn = st.button("Ejecutar paso matricial", type="primary", use_container_width=True)
        with b2:
            reset_btn = st.button("Reiniciar", type="secondary", use_container_width=True)

        if reset_btn:
            st.session_state.multi_w1 = 0.5
            st.session_state.multi_w2 = 1.0
            st.session_state.multi_alpha = 0.05
            st.session_state.multi_history = []
            st.rerun()

        if step_btn:
            # Registrar posición previa en historial
            if not st.session_state.multi_history:
                st.session_state.multi_history.append((w1, w2, mse_loss))

            W_new, grad_step, l_prev, l_after = multivariable_gradient_descent_step(X, Y, W, new_alpha)
            new_w1_val = float(W_new[0, 0])
            new_w2_val = float(W_new[1, 0])

            st.session_state.multi_w1 = new_w1_val
            st.session_state.multi_w2 = new_w2_val
            st.session_state.multi_history.append((new_w1_val, new_w2_val, l_after))
            st.rerun()

        # Métricas resumidas
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(
                label="Pérdida MSE",
                value=f"{mse_loss:.2f}",
                delta=f"Dist a W*: {np.linalg.norm(W - W_opt):.2f}",
                delta_color="off",
            )
        with m_col2:
            grad_norm = float(np.linalg.norm(grad_vector))
            st.metric(
                label="||∇L|| (Magnitud)",
                value=f"{grad_norm:.2f}",
                delta=f"∇ = [{grad_vector[0,0]:+.1f}, {grad_vector[1,0]:+.1f}]",
                delta_color="off",
            )

    # 4. Columna gráfica (3D o Contorno)
    with col_chart:
        history = st.session_state.multi_history
        if view_opt == "3D (Superficie)":
            fig3d = create_3d_loss_surface_chart(X, Y, W, W_opt, trajectory=history)
            st.plotly_chart(fig3d, use_container_width=True, config={"displayModeBar": False})
        else:
            fig_cont = create_contour_loss_chart(X, Y, W, W_opt, grad_vector, trajectory=history)
            st.plotly_chart(fig_cont, use_container_width=True, config={"displayModeBar": False})

    # 5. Explicación del cálculo matricial
    st.markdown(
        f"""
        <div style="background: #121216; border: 1px solid #1F1F24; border-radius: 6px; padding: 10px 14px; font-size: 0.8rem; color: #A1A1AA; line-height: 1.5; margin-top: 4px;">
            <b style="color: #F4F4F5;">Fórmula del Gradiente Matricial:</b> 
            <code>∇_W L = (2/N) · Xᵀ · E = (2/3) · Xᵀ · (ŷ - Y)</code><br>
            Vector actual: <b>∇_W L = [{grad_vector[0,0]:+.3f}, {grad_vector[1,0]:+.3f}]ᵀ</b>. 
            La regla matricial resta este vector: <code>W_nuevo = W - α · ∇_W L</code>, desplazando simultáneamente ambos pesos hacia el fondo del cuenco (W* = [2.0, 3.0]).
        </div>
        """,
        unsafe_allow_html=True,
    )

