# DataLab AI 🧪

Plataforma interactiva para aprender, experimentar y visualizar conceptos matemáticos y algoritmos de Inteligencia Artificial y Machine Learning.

---

##  Laboratorios Disponibles

### 1. Descenso del Gradiente (1 Peso Escalar)
Optimización analítica para $y = w \cdot x$ con pérdida cuadrática sobre un plano 2D $(w, L(w))$.

### 2. Matrices y Gradiente 2D (Superficie 3D y Contorno)
- **Álgebra Lineal:** $\hat{Y} = X \cdot W$ con matrices $X \ (3 \times 2)$, $W \ (2 \times 1)$ y $\hat{Y} \ (3 \times 1)$.
- **Pérdida MSE:** $L(W) = \frac{1}{N} \|X \cdot W - Y\|^2$.
- **Gradiente Matricial:** $\nabla_W L = \frac{2}{N} X^T (\hat{Y} - Y)$.
- **Superficie 3D interactiva:** paraboloide de pérdida generado con Plotly WebGL con rotación en 360°, punto 3D y curvas de nivel proyectadas.

---

## 🚀 Cómo ejecutar la aplicación

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar pruebas unitarias:**
   ```bash
   pytest tests
   ```

3. **Iniciar el laboratorio interactivo en el navegador:**
   ```bash
   streamlit run app.py
   ```

---

##  Estructura del Proyecto

```text
DataLabAI/
│
├── app.py                            # Punto de entrada y barra de navegación persistente
├── requirements.txt                  # Dependencias del proyecto
├── .gitignore                        # Archivos excluidos del control de versiones
├── README.md                         # Documentación general
│
├── modules/
│   └── mathematics/
│       └── labs/
│           └── gradient_descent.py   # Lógica matemática pura y componente UI del laboratorio
│
├── components/
│   └── charts.py                     # Generación de la gráfica de pérdida y tangente
│
└── tests/
    └── test_gradient_descent.py      # Verificación analítica con pytest
```

