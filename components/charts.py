"""
Componente visual de alto rendimiento para graficar la función de pérdida y gradiente.
Utiliza Plotly con renderizado vectorial por WebGL/SVG acelerado por hardware
para respuesta instantánea y fluida en tema oscuro minimalista (Linear / Vercel style).
"""

from typing import Optional
import plotly.graph_objects as go
import numpy as np
from modules.mathematics.labs.gradient_descent import (
    compute_loss,
    compute_gradient,
    get_loss_curve,
    get_tangent_line,
)


def create_gradient_descent_chart(
    w_current: float,
    x: float,
    y: float,
    w_next: Optional[float] = None,
    alpha: Optional[float] = None,
) -> go.Figure:
    """
    Construye la figura interactiva vectorial ultrarrápida para Streamlit.
    """
    w_opt = y / x if x != 0 else 0.0
    margin = max(abs(w_current - w_opt) * 1.5, 4.5)
    w_min = w_opt - margin
    w_max = w_opt + margin

    # 150 puntos son suficientes para una curva suave sin sobrecargar el cliente
    w_vals, loss_vals = get_loss_curve(x, y, w_min=w_min, w_max=w_max, points=180)
    loss_current = compute_loss(w_current, x, y)
    grad_current = compute_gradient(w_current, x, y)

    fig = go.Figure()

    # 1. Curva analítica de pérdida L(w)
    fig.add_trace(
        go.Scatter(
            x=w_vals,
            y=loss_vals,
            mode="lines",
            name="L(w) = (wx - y)²",
            line=dict(color="#3B82F6", width=2.5),
            hovertemplate="w: %{x:.2f}<br>L(w): %{y:.2f}<extra></extra>",
        )
    )

    # 2. Línea vertical del óptimo analítico w*
    fig.add_trace(
        go.Scatter(
            x=[w_opt, w_opt],
            y=[0, max(loss_current * 1.25, 40.0)],
            mode="lines",
            name=f"Óptimo w* = {w_opt:.1f}",
            line=dict(color="#3F3F46", width=1.5, dash="dot"),
            hoverinfo="skip",
        )
    )

    # 3. Recta tangente (pendiente / derivada analítica)
    delta_tan = max(margin * 0.22, 1.2)
    w_tan, loss_tan = get_tangent_line(w_current, x, y, delta=delta_tan)
    signo_str = "+" if grad_current > 0 else ("-" if grad_current < 0 else "0")
    fig.add_trace(
        go.Scatter(
            x=w_tan,
            y=loss_tan,
            mode="lines",
            name=f"Tangente (dL/dw = {grad_current:+.1f} [{signo_str}])",
            line=dict(color="#F59E0B", width=1.8, dash="dash"),
            hoverinfo="skip",
        )
    )

    # 4. Punto actual sobre la curva (w, L(w))
    fig.add_trace(
        go.Scatter(
            x=[w_current],
            y=[loss_current],
            mode="markers+text",
            name="Posición actual",
            marker=dict(
                size=12,
                color="#EF4444",
                line=dict(color="#FFFFFF", width=2),
            ),
            text=[f"w = {w_current:.2f}"],
            textposition="top center",
            textfont=dict(color="#F4F4F5", size=11, family="monospace"),
            hovertemplate=f"Actual: w = {w_current:.3f}<br>L(w) = {loss_current:.3f}<br>dL/dw = {grad_current:+.3f}<extra></extra>",
        )
    )

    # 5. Punto proyectado tras la actualización (si se ejecutó el paso)
    if w_next is not None and abs(w_next - w_current) > 1e-4:
        loss_next = compute_loss(w_next, x, y)
        fig.add_trace(
            go.Scatter(
                x=[w_next],
                y=[loss_next],
                mode="markers+text",
                name=f"Nuevo w = {w_next:.2f}",
                marker=dict(
                    size=12,
                    color="#10B981",
                    symbol="square",
                    line=dict(color="#FFFFFF", width=1.5),
                ),
                text=[f"w_nuevo = {w_next:.2f}"],
                textposition="bottom center",
                textfont=dict(color="#10B981", size=11, family="monospace"),
                hovertemplate=f"Actualizado: w = {w_next:.3f}<br>L(w) = {loss_next:.3f}<extra></extra>",
            )
        )

        # Flecha de transición
        fig.add_annotation(
            x=w_next,
            y=loss_next,
            ax=w_current,
            ay=loss_current,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="#10B981",
            opacity=0.8,
        )

    # Configuración de límites y escala
    y_max = max(loss_current * 1.35, 45.0)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#09090B",
        margin=dict(l=48, r=24, t=28, b=48),
        height=420,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#A1A1AA"),
        ),
        xaxis=dict(
            title=dict(text="Peso w", font=dict(size=12, color="#A1A1AA")),
            range=[w_min, w_max],
            gridcolor="#18181B",
            zerolinecolor="#27272A",
            tickfont=dict(color="#71717A", size=10),
        ),
        yaxis=dict(
            title=dict(text="Pérdida L(w)", font=dict(size=12, color="#A1A1AA")),
            range=[-max(y_max * 0.04, 2.0), y_max],
            gridcolor="#18181B",
            zerolinecolor="#27272A",
            tickfont=dict(color="#71717A", size=10),
        ),
        hoverlabel=dict(
            bgcolor="#18181B",
            font_size=11,
            font_family="monospace",
            bordercolor="#27272A",
        ),
    )

    return fig
