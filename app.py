"""
app.py — Punto de entrada de la aplicación web (Streamlit).
Ejecutar:  streamlit run app.py

Minimización de funciones por Gradiente, Gradiente Conjugado y Newton,
con búsqueda de línea de Wolfe.
"""
import numpy as np
import pandas as pd
import streamlit as st

from core import ObjectiveFunction, minimize, METHOD_NAMES
from ui import plot_convergence, plot_contour

st.set_page_config(page_title="Optimizador | Métodos de Optimización", page_icon="📉", layout="wide")

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
    n_vars = st.number_input("Número de variables (n)", min_value=1, max_value=10, value=2, step=1)
    method_key = st.selectbox("Método de optimización", options=list(METHOD_NAMES.keys()),
                              format_func=lambda k: METHOD_NAMES[k])

    cg_variant = "PR"
    if method_key == "conjugate_gradient":
        cg_variant = st.radio("Variante de β", ["PR", "FR"],
                              format_func=lambda v: "Polak-Ribiere (PR+)" if v == "PR" else "Fletcher-Reeves (FR)",
                              horizontal=True)

    st.markdown("**Función objetivo** — usa `x1, x2, ...`, `**` o `^` para potencias")
    expr_str = st.text_input("f(x) =", value="100*(x2 - x1**2)**2 + (1 - x1)**2")

    st.markdown("**Punto de partida** (separado por comas)")
    x0_str = st.text_input("x0 =", value="-1.2, 1.0")

    st.markdown("**Criterios de parada**")
    col_a, col_b = st.columns(2)
    with col_a:
        max_iter = st.number_input("Máx. iteraciones", min_value=1, max_value=100000, value=1000, step=100)
    with col_b:
        tol = st.number_input("Tolerancia", min_value=1e-15, max_value=1.0, value=1e-6, format="%.1e")

    st.markdown("**Parámetros de Wolfe**")
    col_c, col_d = st.columns(2)
    with col_c:
        c1 = st.number_input("c1 (Armijo)", min_value=1e-6, max_value=0.5, value=1e-4, format="%.1e")
    with col_d:
        c2 = st.number_input("c2 (curvatura)", min_value=0.1, max_value=0.999, value=0.9, step=0.05)
    st.caption("Debe cumplirse 0 < c1 < c2 < 1. Sugerencia: c2=0.9 (Newton), c2=0.1 (CG).")

    compare_all = st.checkbox("🆚 Comparar los 3 métodos (valor agregado)", value=False)
    run = st.button("▶️ Ejecutar optimización", type="primary", use_container_width=True)


def show_result_block(key, r, tol):
    st.subheader(f"📌 Resultados — {METHOD_NAMES[key]}")
    c1_, c2_, c3_, c4_ = st.columns(4)
    c1_.metric("f(x*) — valor mínimo", f"{r.f_min:.6g}")
    c2_.metric("Iteraciones", r.n_iter)
    c3_.metric("Error final ||∇f||", f"{r.final_error:.3e}")
    c4_.metric("¿Convergió?", "Sí ✅" if r.success and r.final_error < tol * 10 else "Parcial ⚠️")
    xstr = ",  ".join(f"x{i+1} = {v:.6f}" for i, v in enumerate(r.x_min))
    st.success(f"**Punto mínimo encontrado:**  ({xstr})")
    st.info(f"**Criterio de parada:** {r.stop_reason}")


# ----------------------------------------------------------------------
# Lógica principal
# ----------------------------------------------------------------------
if run:
    try:
        x0 = np.array([float(v.strip()) for v in x0_str.split(",")], dtype=float)
        if len(x0) != n_vars:
            st.error(f"El punto de partida tiene {len(x0)} valores pero indicaste {n_vars} variables.")
            st.stop()
        if not (0 < c1 < c2 < 1):
            st.error("Las condiciones de Wolfe requieren 0 < c1 < c2 < 1.")
            st.stop()

        obj = ObjectiveFunction(expr_str, n_vars)

        with st.spinner("Calculando..."):
            methods = ("gradient", "conjugate_gradient", "newton") if compare_all else (method_key,)
            results = {
                m: minimize(obj, x0, method=m, max_iter=int(max_iter),
                            tol=tol, c1=c1, c2=c2, cg_variant=cg_variant)
                for m in methods
            }

        for key, r in results.items():
            show_result_block(key, r, tol)
            st.divider()

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown("#### Gráfico de convergencia")
            st.pyplot(plot_convergence(results))
        with gcol2:
            if n_vars == 2:
                st.markdown("#### Trayectoria (valor agregado)")
                st.pyplot(plot_contour(obj, results))
            else:
                st.info("La trayectoria sobre curvas de nivel solo se grafica para n = 2 variables.")

        with st.expander("🧮 Valor agregado: gradiente y Hessiano simbólicos"):
            st.latex(r"f(x) = " + obj.latex_f())
            st.latex(r"\nabla f(x) = " + obj.latex_grad())
            st.latex(r"\nabla^2 f(x) = " + obj.latex_hess())

        with st.expander("📋 Valor agregado: historial completo de iteraciones"):
            for key, r in results.items():
                st.markdown(f"**{METHOD_NAMES[key]}**")
                path = np.array(r.path)
                data = {"iteración": list(range(len(path)))}
                for i in range(n_vars):
                    data[f"x{i+1}"] = path[:, i]
                data["f(x)"] = r.f_history
                data["||∇f||"] = r.error_history
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, height=250)
                st.download_button(
                    f"⬇️ Descargar CSV ({METHOD_NAMES[key]})",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name=f"iteraciones_{key}.csv", mime="text/csv", key=f"dl_{key}")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar la función o los parámetros:\n\n`{e}`")
        st.caption("Revisa que la función use x1, x2, ... y sintaxis válida (ej: `x1**2 + sin(x2)`).")
else:
    st.info("👈 Configura los datos en el panel lateral y presiona **Ejecutar optimización**.")
    with st.expander("📖 Funciones de ejemplo para probar"):
        st.markdown(
            "- **Cuadrática:** `(x1-1)**2 + (x2-2)**2`  → mínimo en (1, 2), x0 = `0,0`\n"
            "- **Rosenbrock:** `100*(x2 - x1**2)**2 + (1 - x1)**2`  → mínimo en (1, 1), x0 = `-1.2, 1.0`\n"
            "- **Himmelblau:** `(x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2`, x0 = `0,0`\n"
            "- **3 variables:** `(x1-3)**2 + (x2+1)**2 + (x3-2)**2`, x0 = `0,0,0`\n"
            "- Funciones disponibles: `sin, cos, exp, log, sqrt, tan`, etc."
        )
