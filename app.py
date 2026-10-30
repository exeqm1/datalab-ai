"""
DataLab AI - Aplicación Principal
Interfaz minimalista de alto rendimiento para experimentación en Machine Learning.
"""

import streamlit as st
import components.charts as charts
from components.icons import icon
from modules.mathematics.labs.gradient_descent import render_gradient_descent_lab
from modules.mathematics.labs.multivariable_gd import render_multivariable_lab

# 1. Configuración de la página
st.set_page_config(
    page_title="DataLab AI",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Estilos globales minimalistas (Vercel / Linear style)
st.markdown(
    """
    <style>
        /* Importar fuente Inter */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #F4F4F5;
        }

        /* Barra superior transparente sin bloquear los controles de la barra lateral */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            height: 2.2rem !important;
            z-index: 99 !important;
        }

        /* Ocultar elementos innecesarios (menú y footer) */
        #MainMenu, footer {
            display: none !important;
            height: 0 !important;
        }

        /* Botón para abrir la barra lateral cuando está oculta (siempre visible y accesible) */
        [data-testid="collapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            color: #F4F4F5 !important;
            background-color: #121216 !important;
            border: 1px solid #27272A !important;
            border-radius: 6px !important;
            padding: 3px !important;
            margin-top: 4px !important;
            margin-left: 8px !important;
            transition: all 0.15s ease !important;
        }
        [data-testid="collapsedControl"]:hover {
            border-color: #3B82F6 !important;
            background-color: #18181B !important;
        }

        /* Contenedor principal ajustado a pantalla completa */
        .block-container {
            padding-top: 0.8rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            max-width: 100% !important;
        }

        /* Barra lateral oscura y minimalista */
        [data-testid="stSidebar"] {
            background-color: #0D0D11 !important;
            border-right: 1px solid #1F1F24 !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: #1F1F24 !important;
            margin: 1rem 0 !important;
        }

        /* Tarjetas de métricas nativas */
        [data-testid="stMetric"] {
            background-color: #121216;
            border: 1px solid #27272A;
            border-radius: 8px;
            padding: 12px 14px;
        }
        [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.35rem !important;
            font-weight: 600 !important;
            color: #F4F4F5 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #71717A !important;
            font-weight: 600 !important;
        }

        /* Botones primarios y secundarios */
        .stButton button[kind="primary"] {
            background-color: #F4F4F5 !important;
            color: #09090B !important;
            border: 1px solid #F4F4F5 !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            transition: all 0.15s ease-in-out !important;
        }
        .stButton button[kind="primary"]:hover {
            background-color: #FFFFFF !important;
            box-shadow: 0 0 12px rgba(255, 255, 255, 0.2) !important;
        }
        .stButton button[kind="secondary"] {
            background-color: #18181B !important;
            color: #D4D4D8 !important;
            border: 1px solid #27272A !important;
            border-radius: 6px !important;
            font-size: 0.85rem !important;
        }
        .stButton button[kind="secondary"]:hover {
            border-color: #3F3F46 !important;
            color: #FFFFFF !important;
        }

        /* Deslizadores / Sliders */
        div[data-testid="stSlider"] label {
            font-size: 0.84rem !important;
            color: #A1A1AA !important;
            font-weight: 500 !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background-color: #121216 !important;
            border: 1px solid #27272A !important;
            border-radius: 6px !important;
            font-size: 0.84rem !important;
            color: #A1A1AA !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. Barra lateral minimalista
with st.sidebar:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 4px 0 10px 0;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background: #3B82F6;"></div>
                <span style="font-weight: 700; font-size: 0.98rem; letter-spacing: -0.02em; color: #F4F4F5;">DataLab</span>
            </div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #52525B; border: 1px solid #27272A; padding: 1px 6px; border-radius: 4px;">v0.1</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #52525B; font-weight: 600; margin-bottom: 8px;">
            Módulos
        </div>
        """,
        unsafe_allow_html=True,
    )

    modules_map = {
        "Matemáticas": "activity",
        "Datasets": "database",
        "Machine Learning": "cpu",
        "Experimentos": "flask",
    }

    selected_module = st.radio(
        "Módulos:",
        options=list(modules_map.keys()),
        index=0,
        label_visibility="collapsed",
    )

    selected_lab = "1. Gradiente 1D (1 Peso)"
    if selected_module == "Matemáticas":
        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #52525B; font-weight: 600; margin-bottom: 8px;">
                Laboratorios
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_lab = st.radio(
            "Laboratorios:",
            options=[
                "1. Gradiente 1D (1 Peso)",
                "2. Matrices y Superficie 3D (2 Pesos)",
            ],
            index=0,
            label_visibility="collapsed",
        )

    st.markdown("---")

    context_desc = "Modelo lineal 1D" if "1." in selected_lab else "Álgebra matricial & 3D"
    st.markdown(
        f"""
        <div style="background: #121216; border: 1px solid #27272A; border-radius: 6px; padding: 10px 12px;">
            <div style="font-size: 0.75rem; color: #71717A; font-weight: 500; margin-bottom: 4px;">Contexto activo</div>
            <div style="font-size: 0.82rem; color: #D4D4D8; line-height: 1.4;">
                {selected_module} &bull; {context_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 4. Enrutamiento
if selected_module == "Matemáticas":
    if "1." in selected_lab:
        render_gradient_descent_lab(st, charts)
    else:
        render_multivariable_lab(st)
else:
    st.markdown(
        f"""
        <div style="border: 1px solid #27272A; border-radius: 8px; padding: 32px; background: #121216; margin-top: 16px;">
            <h3 style="margin: 0 0 8px 0; color: #F4F4F5; font-size: 1.1rem; font-weight: 600;">{selected_module}</h3>
            <p style="margin: 0; color: #71717A; font-size: 0.88rem;">
                Módulo reservado en la arquitectura. Podrás alternar entre módulos sin perder el estado de tus experimentos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
