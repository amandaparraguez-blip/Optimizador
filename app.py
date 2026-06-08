"""
app.py — Punto de entrada de la aplicación web (Streamlit).
Ejecutar:  streamlit run app.py

Minimización de funciones por Gradiente, Gradiente Conjugado y Newton,
con búsqueda de línea de Wolfe.
"""
import numpy as np
import pandas as pd
import streamlit as st

from core import ObjectiveFunction, minimize, METHOD_NAMES, classify_point
from ui import plot_convergence, plot_contour, plot_basins, plot_step_sizes

st.set_page_config(
    page_title="Optimizador | Métodos de Optimización",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Tema visual (oscuro con acentos en degradado)
# ----------------------------------------------------------------------
st.markdown("""
<style>
/* Fondos base */
[data-testid="stAppViewContainer"] { background-color: #0F172A; }
[data-testid="stHeader"] { background-color: rgba(15,23,42,0); }
[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1E293B; }

/* Tipografía de encabezados */
h1, h2, h3, h4 { color: #F8FAFC; letter-spacing: -0.01em; }

/* Botón principal con degradado */
.stButton > button {
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
    transition: filter .15s ease, transform .05s ease;
}
.stButton > button:hover { filter: brightness(1.12); color: white; }
.stButton > button:active { transform: translateY(1px); }

/* Botón de descarga acorde al tema */
.stDownloadButton > button {
    background: #1E293B;
    color: #E2E8F0;
    border: 1px solid #334155;
    border-radius: 10px;
}
.stDownloadButton > button:hover { border-color: #7C3AED; color: #FFFFFF; }

/* Tarjetas de métricas */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] { color: #94A3B8; }
[data-testid="stMetricValue"] { color: #F8FAFC; }

/* Pestañas */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: #111827;
    border-radius: 10px 10px 0 0;
    padding: 8px 16px;
    color: #94A3B8;
}
.stTabs [aria-selected="true"] {
    background: #1E293B;
    color: #F8FAFC;
    border-bottom: 2px solid #7C3AED;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid #1E293B;
    border-radius: 12px;
    background: #0B1220;
}
</style>
""", unsafe_allow_html=True)

st.title("📉 Minimización de funciones con condiciones de Wolfe")
st.caption(
    "Métodos de Optimización — Gradiente · Gradiente Conjugado · Newton  |  "
    "Búsqueda de línea con condiciones de Wolfe (Armijo + curvatura)"
)

# ----------------------------------------------------------------------
# Panel lateral: datos de entrada
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Datos de entrada")

    # --- Problema ---
    st.markdown("##### 1 · Problema")
    n_vars = st.number_input("Número de variables (n)", min_value=1, max_value=10, value=2, step=1)
    expr_str = st.text_area(
        "Función objetivo  f(x) =",
        value="100*(x2 - x1**2)**2 + (1 - x1)**2",
        height=100,
        help="Usa x1, x2, ...  •  `**` o `^` para potencias  •  sin, cos, exp, log, sqrt, tan. "
             "El cuadro crece y es redimensionable.",
    )
    x0_str = st.text_input(
        "Punto de partida  x0 =",
        value="-1.2, 1.0",
        help="Valores separados por comas. Debe tener exactamente n valores.",
    )

    st.divider()

    # --- Método ---
    st.markdown("##### 2 · Método")
    method_key = st.selectbox("Método de optimización", options=list(METHOD_NAMES.keys()),
                              format_func=lambda k: METHOD_NAMES[k])

    cg_variant = "PR"
    if method_key == "conjugate_gradient":
        cg_variant = st.radio("Variante de β", ["PR", "FR"],
                              format_func=lambda v: "Polak-Ribiere (PR+)" if v == "PR" else "Fletcher-Reeves (FR)",
                              horizontal=True)

    st.divider()

    # --- Criterios de parada ---
    st.markdown("##### 3 · Criterios de parada")
    col_a, col_b = st.columns(2)
    with col_a:
        max_iter = st.number_input("Máx. iteraciones", min_value=1, max_value=100000, value=1000, step=100)
    with col_b:
        tol = st.number_input("Tolerancia", min_value=1e-15, max_value=1.0, value=1e-6, format="%.1e")

    st.divider()

    # --- Búsqueda de línea ---
    st.markdown("##### 4 · Búsqueda de línea")
    ls_label = st.selectbox(
        "Tipo de búsqueda de línea",
        options=["wolfe", "backtracking", "fixed"],
        format_func=lambda v: {"wolfe": "Wolfe (1ra y 2da condición)",
                               "backtracking": "Backtracking (Armijo + factor de reducción)",
                               "fixed": "Paso fijo (α constante)"}[v],
    )

    alpha_label = "α (paso fijo)" if ls_label == "fixed" else "A₀ (paso inicial)"
    alpha0 = st.number_input(alpha_label, min_value=1e-6, max_value=100.0,
                             value=1.0, step=0.01, format="%.4f")

    rho = 0.5
    sigma = 0.5
    if ls_label == "wolfe":
        col_c, col_d = st.columns(2)
        with col_c:
            c1 = st.number_input("c1 (Armijo)", min_value=1e-6, max_value=0.5, value=1e-4, format="%.1e")
        with col_d:
            c2 = st.number_input("c2 (curvatura)", min_value=0.1, max_value=0.999, value=0.9, step=0.05)
        st.caption("Debe cumplirse 0 < c1 < c2 < 1. Sugerencia: c2=0.9 (Newton), c2=0.1 (CG).")
    elif ls_label == "backtracking":
        col_c, col_d = st.columns(2)
        with col_c:
            c1 = st.number_input("β (Armijo)", min_value=1e-6, max_value=0.9,
                                 value=0.0001, step=0.05, format="%.4f")
        with col_d:
            rho = st.number_input("ρ (factor de reducción)", min_value=0.05, max_value=0.95,
                                  value=0.5, step=0.05)
        sigma = st.number_input("σ (2ª condición de Wolfe, solo verificación)",
                                min_value=0.0, max_value=0.999, value=0.5, step=0.05)
        c2 = 0.9  # no se usa en backtracking, pero se mantiene por compatibilidad
        st.caption("Backtracking: reduce el paso multiplicándolo por ρ hasta cumplir Armijo (1ra condición). "
                   "El paso encontrado depende solo de β y ρ.")
        st.caption("ℹ️ σ solo afecta la verificación de la 2ª condición de Wolfe (✓/✗). "
                   "No cambia el punto encontrado ni las iteraciones.")
    else:  # fixed
        c1, c2 = 1e-4, 0.9  # no se usan en paso fijo
        st.caption("Paso fijo: avanza con  x_{k+1} = x_k + α·d_k  usando α constante (no hace búsqueda de "
                   "línea). Para el método del gradiente equivale a  x_{k+1} = x_k − α·∇f(x_k).")

    st.divider()

    # --- Opciones extra ---
    st.markdown("##### 5 · Opciones")
    compare_all = st.checkbox("🆚 Comparar los 3 métodos", value=False)
    show_basins = st.checkbox("🗺️ Mapa de cuencas de atracción (solo n=2)", value=False,
                              help="Corre el método desde una grilla de puntos y pinta el plano "
                                   "según a qué mínimo llega cada uno. Puede tardar unos segundos.")
    run = st.button("▶️ Ejecutar optimización", type="primary", use_container_width=True)


def show_result_block(key, r, tol, obj):
    st.subheader(f"📌 {METHOD_NAMES[key]}")
    c1_, c2_, c3_, c4_ = st.columns(4)
    c1_.metric("f(x*) — valor mínimo", f"{r.f_min:.6g}")
    c2_.metric("Iteraciones", r.n_iter)
    c3_.metric("Error final ||∇f||", f"{r.final_error:.3e}")
    c4_.metric("¿Convergió?", "Sí ✅" if r.success and r.final_error < tol * 10 else "Parcial ⚠️")
    xstr = ",  ".join(f"x{i+1} = {v:.6f}" for i, v in enumerate(r.x_min))
    st.success(f"**Punto mínimo encontrado:**  ({xstr})")
    st.info(f"**Criterio de parada:** {r.stop_reason}")

    # Clasificación del punto final según el Hessiano (mínimo / máximo / silla)
    tipo, eig, es_critico = classify_point(obj, r.x_min)
    eig_txt = ", ".join(f"{e:.4g}" for e in eig)
    etiqueta = {"mínimo": "MÍNIMO local ✅", "máximo": "MÁXIMO local 🔺",
                "silla": "PUNTO SILLA ⚠️", "degenerado": "DEGENERADO (no concluyente) ❔"}[tipo]
    detalle = {"mínimo": "Hessiano definido positivo (todos los autovalores > 0).",
               "máximo": "Hessiano definido negativo (todos los autovalores < 0).",
               "silla": "Hessiano indefinido (autovalores de distinto signo).",
               "degenerado": "Hay un autovalor ≈ 0; la 2ª derivada no concluye."}[tipo]
    msg = f"**Clasificación del punto:** {etiqueta} — {detalle}  \nAutovalores del Hessiano: [{eig_txt}]"
    if not es_critico:
        msg += "  \n⚠️ Nota: el gradiente aún no es cero aquí, así que todavía no es un punto crítico (faltan iteraciones)."
    (st.success if tipo == "mínimo" and es_critico else st.warning)(msg)

    # Resumen de la 2ª condición de Wolfe sobre el paso final (solo backtracking)
    if r.line_search == "backtracking" and r.wolfe2_history:
        if r.wolfe2_history[-1]:
            st.success(f"**2ª condición de Wolfe:** el paso final ✅ **cumple** la condición de "
                       f"curvatura (σ = {r.sigma}).")
        else:
            st.warning(f"**2ª condición de Wolfe:** el paso final ❌ **no cumple** la condición de "
                       f"curvatura (σ = {r.sigma}). Es normal en backtracking y no es un error.")


def show_history_block(key, r, n_vars):
    """Tabla de historial completo de iteraciones + descarga CSV (un método)."""
    st.markdown(f"**{METHOD_NAMES[key]}**")
    path = np.array(r.path)
    data = {"iteración": list(range(len(path)))}
    for i in range(n_vars):
        data[f"x{i+1}"] = path[:, i]
    data["f(x)"] = r.f_history
    data["||∇f||"] = r.error_history

    # Helper para alinear listas de longitud n_iter con las n_iter+1 filas
    def _align(values, fmt="{:.6g}"):
        col = ["—"] + [fmt.format(v) for v in values]
        return (col + ["—"] * len(path))[:len(path)]

    # α (paso aceptado en cada iteración) — útil sobre todo en backtracking/paso fijo
    if r.alpha_history:
        data["α usado"] = _align(r.alpha_history)
    # Lados de la condición de Armijo:  LHS = f(x_k+αp)  ≤  RHS = f(x_k)+c1·α·∇f·p
    if r.armijo_lhs_history:
        data["Armijo izq: f(x+αp)"] = _align(r.armijo_lhs_history)
        data["Armijo der (≥)"] = _align(r.armijo_rhs_history)
        cumple = [l <= rr for l, rr in zip(r.armijo_lhs_history, r.armijo_rhs_history)]
        col = ["—"] + ["✓" if c else "✗" for c in cumple]
        data["¿Cumple Armijo?"] = (col + ["—"] * len(path))[:len(path)]
    # Columna de la 2ª condición de Wolfe (solo backtracking)
    if r.line_search == "backtracking" and r.wolfe2_history:
        col = ["—"] + ["✓ cumple" if w else "✗ no cumple" for w in r.wolfe2_history]
        col = (col + ["—"] * len(path))[:len(path)]  # alinear longitud
        data[f"2ª Wolfe (σ={r.sigma})"] = col
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, height=250)
    st.download_button(
        f"⬇️ Descargar CSV ({METHOD_NAMES[key]})",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"iteraciones_{key}.csv", mime="text/csv", key=f"dl_{key}")


# ----------------------------------------------------------------------
# Lógica principal
# ----------------------------------------------------------------------
if run:
    try:
        x0 = np.array([float(v.strip()) for v in x0_str.split(",")], dtype=float)
        if len(x0) != n_vars:
            st.error(f"El punto de partida tiene {len(x0)} valores pero indicaste {n_vars} variables.")
            st.stop()
        if ls_label == "wolfe" and not (0 < c1 < c2 < 1):
            st.error("Las condiciones de Wolfe requieren 0 < c1 < c2 < 1.")
            st.stop()

        obj = ObjectiveFunction(expr_str.replace("\n", " ").strip(), n_vars)

        with st.spinner("Calculando..."):
            methods = ("gradient", "conjugate_gradient", "newton") if compare_all else (method_key,)
            results = {
                m: minimize(obj, x0, method=m, max_iter=int(max_iter),
                            tol=tol, c1=c1, c2=c2, cg_variant=cg_variant,
                            line_search=ls_label, alpha0=alpha0, rho=rho, sigma=sigma)
                for m in methods
            }

        tab_resumen, tab_graficos, tab_detalle = st.tabs(
            ["📊 Resumen", "📈 Gráficos", "🔢 Detalle"]
        )

        # --- Pestaña 1: Resumen (métricas + clasificación por método) ---
        with tab_resumen:
            for key, r in results.items():
                show_result_block(key, r, tol, obj)
                st.divider()

        # --- Pestaña 2: Gráficos ---
        with tab_graficos:
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                st.markdown("#### Gráfico de convergencia")
                st.pyplot(plot_convergence(results))
            with gcol2:
                if n_vars == 2:
                    st.markdown("#### Trayectoria de optimización")
                    st.pyplot(plot_contour(obj, results))
                else:
                    st.info("La trayectoria sobre curvas de nivel solo se grafica para n = 2 variables.")

            # Gráfico del tamaño de paso α — solo en backtracking (donde se usa el factor de reducción)
            if ls_label == "backtracking":
                st.markdown("#### Tamaño de paso α por iteración")
                st.pyplot(plot_step_sizes(results))
                st.caption("Muestra el α aceptado en cada iteración (parte en α₀ y baja por ρ si hace falta). "
                           "Si comparas los 3 métodos, aparecen los tres.")

            # Mapa de cuencas de atracción (característica distintiva, solo n=2)
            if show_basins:
                st.markdown("#### 🗺️ Mapa de cuencas de atracción")
                if n_vars != 2:
                    st.info("El mapa de cuencas solo está disponible para funciones de 2 variables.")
                else:
                    ls_kwargs = dict(c1=c1, c2=c2, cg_variant=cg_variant,
                                     line_search=ls_label, alpha0=alpha0, rho=rho, sigma=sigma)
                    with st.spinner("Corriendo el método desde cientos de puntos de partida..."):
                        fig_b, n_min = plot_basins(obj, minimize, method_key, ls_kwargs)
                    st.pyplot(fig_b)
                    st.caption(f"Cada color es una región cuyos puntos de partida terminan en el mismo "
                               f"mínimo (se encontraron {n_min}). El gris indica puntos que no convergieron. "
                               f"Método usado: {METHOD_NAMES[method_key]}.")

        # --- Pestaña 3: Detalle (simbólico + historial + CSV) ---
        with tab_detalle:
            with st.expander("Gradiente y Hessiano simbólicos", expanded=True):
                st.latex(r"f(x) = " + obj.latex_f())
                st.latex(r"\nabla f(x) = " + obj.latex_grad())
                st.latex(r"\nabla^2 f(x) = " + obj.latex_hess())

            with st.expander("Historial completo de iteraciones", expanded=True):
                for key, r in results.items():
                    show_history_block(key, r, n_vars)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar la función o los parámetros:\n\n`{e}`")
        st.caption("Revisa que la función use x1, x2, ... y sintaxis válida (ej: `x1**2 + sin(x2)`).")
else:
    st.info("👈 Configura los datos en el panel lateral y presiona **Ejecutar optimización**.")
    with st.expander("📖 Funciones de ejemplo para probar", expanded=True):
        st.markdown(
            "- **Cuadrática:** `(x1-1)**2 + (x2-2)**2`  → mínimo en (1, 2), x0 = `0,0`\n"
            "- **Rosenbrock:** `100*(x2 - x1**2)**2 + (1 - x1)**2`  → mínimo en (1, 1), x0 = `-1.2, 1.0`\n"
            "- **Himmelblau:** `(x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2`, x0 = `0,0`\n"
            "- **3 variables:** `(x1-3)**2 + (x2+1)**2 + (x3-2)**2`, x0 = `0,0,0`\n"
            "- Funciones disponibles: `sin, cos, exp, log, sqrt, tan`, etc."
        )
