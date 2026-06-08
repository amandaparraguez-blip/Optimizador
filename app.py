"""
app.py — Punto de entrada de la aplicación web (Streamlit).
Ejecutar:  streamlit run app.py

Minimización de funciones por Gradiente, Gradiente Conjugado y Newton,
con búsqueda de línea de Wolfe.

Interfaz: HUD / dashboard técnico con tema claro/oscuro conmutable.
Toda la matemática vive en core/ y los gráficos en ui/ (no se modifican aquí).
"""
import contextlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import streamlit as st

from core import ObjectiveFunction, minimize, METHOD_NAMES, classify_point
from ui import plot_convergence, plot_contour, plot_basins, plot_step_sizes

st.set_page_config(
    page_title="Optimizador | Métodos de Optimización",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# TEMA — paletas claro/oscuro y conmutador real (no depende del tema nativo)
# ======================================================================
THEMES = {
    "dark": {
        "bg":        "#0A0F1E",   # fondo app
        "bg2":       "#0E1628",   # paneles / sidebar
        "panel":     "#111B30",   # tarjetas
        "panel2":    "#0B1424",   # tarjetas alternas
        "border":    "#1E2D49",   # bordes sutiles
        "border_hi": "#2E4677",   # borde resaltado
        "text":      "#E8EEF9",   # texto principal
        "muted":     "#8499BD",   # texto secundario
        "accent":    "#5B8CFF",   # acento neón (azul)
        "accent2":   "#A06BFF",   # acento neón (violeta)
        "good":      "#34D399",
        "warn":      "#FBBF24",
        "bad":       "#F87171",
        "grid":      "#1B2945",
        "mpl_face":  "#0E1628",
        "mpl_axes":  "#0B1424",
        "mpl_text":  "#E8EEF9",
        "mpl_grid":  "#22304F",
    },
    "light": {
        "bg":        "#F4F7FC",
        "bg2":       "#FFFFFF",
        "panel":     "#FFFFFF",
        "panel2":    "#F2F6FD",
        "border":    "#D7E0EE",
        "border_hi": "#9FBDFF",
        "text":      "#0E1A2F",
        "muted":     "#5A6B86",
        "accent":    "#2563EB",
        "accent2":   "#7C3AED",
        "good":      "#059669",
        "warn":      "#B45309",
        "bad":       "#DC2626",
        "grid":      "#E2E9F4",
        "mpl_face":  "#FFFFFF",
        "mpl_axes":  "#FFFFFF",
        "mpl_text":  "#0E1A2F",
        "mpl_grid":  "#D7E0EE",
    },
}

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"


def current_palette():
    return THEMES[st.session_state.theme_mode]


@contextlib.contextmanager
def mpl_theme(pal):
    """Aplica colores del tema a matplotlib SIN tocar la lógica de ui/plots.py."""
    rc = {
        "figure.facecolor": pal["mpl_face"],
        "axes.facecolor": pal["mpl_axes"],
        "savefig.facecolor": pal["mpl_face"],
        "text.color": pal["mpl_text"],
        "axes.labelcolor": pal["mpl_text"],
        "axes.edgecolor": pal["mpl_grid"],
        "axes.titlecolor": pal["mpl_text"],
        "xtick.color": pal["mpl_text"],
        "ytick.color": pal["mpl_text"],
        "grid.color": pal["mpl_grid"],
        "legend.facecolor": pal["mpl_axes"],
        "legend.edgecolor": pal["mpl_grid"],
        "legend.labelcolor": pal["mpl_text"],
    }
    with mpl.rc_context(rc):
        yield


def inject_css(pal):
    st.markdown(f"""
<style>
:root {{
  --bg:{pal['bg']}; --bg2:{pal['bg2']}; --panel:{pal['panel']}; --panel2:{pal['panel2']};
  --border:{pal['border']}; --border-hi:{pal['border_hi']};
  --text:{pal['text']}; --muted:{pal['muted']};
  --accent:{pal['accent']}; --accent2:{pal['accent2']};
  --good:{pal['good']}; --warn:{pal['warn']}; --bad:{pal['bad']};
}}

/* ---- Fondos base ---- */
[data-testid="stAppViewContainer"] {{ background: var(--bg); }}
[data-testid="stAppViewContainer"] > .main {{
    background:
      radial-gradient(900px 420px at 12% -8%, color-mix(in srgb, var(--accent) 14%, transparent), transparent 60%),
      radial-gradient(780px 400px at 100% 0%, color-mix(in srgb, var(--accent2) 12%, transparent), transparent 60%);
}}
[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
[data-testid="stSidebar"] {{ background: var(--bg2); border-right: 1px solid var(--border); }}
.block-container {{ padding-top: 1.4rem; }}

/* ---- Tipografía ---- */
html, body, [class*="css"] {{ color: var(--text); }}
h1, h2, h3, h4, h5 {{ color: var(--text); letter-spacing: -0.01em; }}
p, label, span, li {{ color: var(--text); }}
[data-testid="stMarkdownContainer"] small, .stCaption, [data-testid="stCaptionContainer"] {{ color: var(--muted) !important; }}

/* numbers in mono for the HUD feel */
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}

/* ---- Barra superior (HUD header) ---- */
.hud-header {{
    display:flex; align-items:center; justify-content:space-between;
    gap:16px; padding:14px 18px; margin-bottom:6px;
    background: linear-gradient(180deg, var(--panel), var(--panel2));
    border:1px solid var(--border); border-radius:16px;
    box-shadow: 0 1px 0 color-mix(in srgb, var(--accent) 25%, transparent) inset;
}}
.hud-title {{ display:flex; align-items:center; gap:12px; }}
.hud-title .logo {{
    display:grid; place-items:center; width:42px; height:42px; border-radius:12px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    box-shadow: 0 6px 18px color-mix(in srgb, var(--accent) 40%, transparent);
}}
.hud-title h1 {{ font-size:1.18rem; margin:0; line-height:1.1; }}
.hud-title .sub {{ color:var(--muted); font-size:.80rem; }}
.hud-badges {{ display:flex; gap:8px; flex-wrap:wrap; }}
.badge {{
    display:inline-flex; align-items:center; gap:6px;
    font-size:.72rem; color:var(--muted);
    border:1px solid var(--border); border-radius:999px; padding:5px 11px;
    background: var(--panel2);
}}
.badge svg {{ width:13px; height:13px; }}

/* ---- Section labels (sidebar) ---- */
.sec {{
    display:flex; align-items:center; gap:8px;
    font-size:.72rem; font-weight:700; letter-spacing:.10em; text-transform:uppercase;
    color:var(--muted); margin:6px 0 2px;
}}
.sec svg {{ width:15px; height:15px; color:var(--accent); }}
.sec .n {{
    display:grid; place-items:center; width:18px; height:18px; border-radius:6px;
    font-size:.66rem; color:#fff;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
}}

/* ---- Inputs ---- */
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea,
[data-baseweb="select"] > div {{
    background: var(--panel2) !important; color: var(--text) !important;
    border-radius: 10px !important;
}}
[data-testid="stSidebar"] [data-baseweb="input"] {{ border-radius:10px; }}

/* ---- Botones ---- */
.stButton > button {{
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color:#fff; border:none; border-radius:12px; padding:11px 20px; font-weight:700;
    box-shadow: 0 8px 22px color-mix(in srgb, var(--accent) 35%, transparent);
    transition: filter .15s, transform .05s;
}}
.stButton > button:hover {{ filter:brightness(1.1); color:#fff; }}
.stButton > button:active {{ transform: translateY(1px); }}
.stDownloadButton > button {{
    background: var(--panel2); color: var(--text);
    border:1px solid var(--border); border-radius:10px; font-weight:600;
}}
.stDownloadButton > button:hover {{ border-color: var(--accent); }}

/* ---- Métricas (tarjetas HUD) ---- */
[data-testid="stMetric"] {{
    background: linear-gradient(180deg, var(--panel), var(--panel2));
    border:1px solid var(--border); border-radius:14px; padding:14px 16px;
    position:relative; overflow:hidden;
}}
[data-testid="stMetric"]::before {{
    content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
}}
[data-testid="stMetricLabel"] p {{ color:var(--muted) !important; font-size:.74rem; }}
[data-testid="stMetricValue"] {{ color:var(--text); }}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{
    background: var(--panel2); border:1px solid var(--border); border-bottom:none;
    border-radius:10px 10px 0 0; padding:8px 16px; color:var(--muted);
}}
.stTabs [aria-selected="true"] {{
    background: var(--panel); color:var(--text);
    box-shadow: inset 0 -2px 0 var(--accent);
}}

/* ---- Expanders / alerts / dataframe ---- */
[data-testid="stExpander"] {{ border:1px solid var(--border); border-radius:12px; background: var(--panel2); }}
[data-testid="stExpander"] summary {{ color:var(--text); }}
[data-testid="stAlert"] {{ border-radius:12px; border:1px solid var(--border); }}
[data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius:12px; }}
hr {{ border-color: var(--border); }}

/* ---- Tarjeta de resultado (encabezado por método) ---- */
.result-head {{
    display:flex; align-items:center; gap:10px; margin: 6px 0 12px;
    padding:10px 14px; border:1px solid var(--border); border-radius:12px;
    background: linear-gradient(90deg, color-mix(in srgb, var(--accent) 10%, var(--panel)), var(--panel));
}}
.result-head svg {{ width:18px; height:18px; color:var(--accent); }}
.result-head b {{ font-size:1.02rem; }}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# ICONOS — Lucide (SVG inline, sin dependencias; heredan el color del texto)
# ======================================================================
_ICON_PATHS = {
    "activity":  '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "function":  '<path d="M9 17c0-3.87 0-7.42 2-9.5C13.5 5 16 6 16 6"/><path d="M3 12h6"/>',
    "settings":  '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    "target":    '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "git":       '<line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/>',
    "ruler":     '<path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.4 2.4 0 0 1 0-3.4l2.6-2.6a2.4 2.4 0 0 1 3.4 0Z"/><path d="m14.5 12.5 2-2"/><path d="m11.5 9.5 2-2"/><path d="m8.5 6.5 2-2"/><path d="m17.5 15.5 2-2"/>',
    "sliders":   '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "layers":    '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "play":      '<polygon points="5 3 19 12 5 21 5 3"/>',
    "chart":     '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "grid":      '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>',
    "list":      '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    "sigma":     '<path d="M18 7V5H6l6 7-6 7h12v-2"/>',
    "route":     '<circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/>',
    "map":       '<polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/>',
    "info":      '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "book":      '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    "moon":      '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
    "sun":       '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.2" y1="4.2" x2="5.6" y2="5.6"/><line x1="18.4" y1="18.4" x2="19.8" y2="19.8"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.2" y1="19.8" x2="5.6" y2="18.4"/><line x1="18.4" y1="5.6" x2="19.8" y2="4.2"/>',
    "pin":       '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>',
}


def icon(name, size=16):
    """Devuelve un SVG Lucide inline que hereda currentColor."""
    body = _ICON_PATHS.get(name, "")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


def sec_label(name, n, icon_name):
    st.markdown(
        f'<div class="sec"><span class="n">{n}</span>{icon(icon_name)}<span>{name}</span></div>',
        unsafe_allow_html=True,
    )


# ======================================================================
# RENDER — header HUD + conmutador de tema
# ======================================================================
pal = current_palette()
inject_css(pal)

hcol1, hcol2 = st.columns([5, 1.2], vertical_alignment="center")
with hcol1:
    is_dark = st.session_state.theme_mode == "dark"
    badge_theme = (icon("moon") + " Oscuro") if is_dark else (icon("sun") + " Claro")
    st.markdown(f"""
    <div class="hud-header">
      <div class="hud-title">
        <div class="logo" style="color:#fff">{icon("activity", 22)}</div>
        <div>
          <h1>Optimizador · Métodos de Optimización</h1>
          <div class="sub">Gradiente · Gradiente Conjugado · Newton — búsqueda de línea con condiciones de Wolfe</div>
        </div>
      </div>
      <div class="hud-badges">
        <span class="badge">{icon("target")} Wolfe</span>
        <span class="badge">{icon("git")} 3 métodos</span>
        <span class="badge">{badge_theme}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
with hcol2:
    new_dark = st.toggle("Modo oscuro", value=is_dark, key="theme_toggle",
                         help="Conmuta entre tema oscuro y claro.")
    target = "dark" if new_dark else "light"
    if target != st.session_state.theme_mode:
        st.session_state.theme_mode = target
        st.rerun()


# ======================================================================
# Panel lateral: datos de entrada
# ======================================================================
with st.sidebar:
    st.markdown(f'<div class="sec" style="font-size:.82rem">{icon("sliders")}<span>Panel de control</span></div>',
                unsafe_allow_html=True)

    sec_label("Problema", 1, "function")
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
    sec_label("Método", 2, "git")
    method_key = st.selectbox("Método de optimización", options=list(METHOD_NAMES.keys()),
                              format_func=lambda k: METHOD_NAMES[k])

    cg_variant = "PR"
    if method_key == "conjugate_gradient":
        cg_variant = st.radio("Variante de β", ["PR", "FR"],
                              format_func=lambda v: "Polak-Ribiere (PR+)" if v == "PR" else "Fletcher-Reeves (FR)",
                              horizontal=True)

    st.divider()
    sec_label("Criterios de parada", 3, "target")
    col_a, col_b = st.columns(2)
    with col_a:
        max_iter = st.number_input("Máx. iteraciones", min_value=1, max_value=100000, value=1000, step=100)
    with col_b:
        tol = st.number_input("Tolerancia", min_value=1e-15, max_value=1.0, value=1e-6, format="%.1e")

    st.divider()
    sec_label("Búsqueda de línea", 4, "ruler")
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
    sec_label("Opciones", 5, "layers")
    compare_all = st.checkbox("Comparar los 3 métodos", value=False)
    show_basins = st.checkbox("Mapa de cuencas de atracción (solo n=2)", value=False,
                              help="Corre el método desde una grilla de puntos y pinta el plano "
                                   "según a qué mínimo llega cada uno. Puede tardar unos segundos.")
    run = st.button("Ejecutar optimización", type="primary", use_container_width=True)


# ======================================================================
# Bloques de presentación
# ======================================================================
def show_result_block(key, r, tol, obj):
    st.markdown(
        f'<div class="result-head">{icon("pin")}<b>{METHOD_NAMES[key]}</b></div>',
        unsafe_allow_html=True,
    )
    c1_, c2_, c3_, c4_ = st.columns(4)
    c1_.metric("f(x*) — valor mínimo", f"{r.f_min:.6g}")
    c2_.metric("Iteraciones", r.n_iter)
    c3_.metric("Error final ||∇f||", f"{r.final_error:.3e}")
    c4_.metric("¿Convergió?", "Sí ✓" if r.success and r.final_error < tol * 10 else "Parcial !")
    xstr = ",  ".join(f"x{i+1} = {v:.6f}" for i, v in enumerate(r.x_min))
    st.success(f"**Punto mínimo encontrado:**  ({xstr})")
    st.info(f"**Criterio de parada:** {r.stop_reason}")

    # Clasificación del punto final según el Hessiano (mínimo / máximo / silla)
    tipo, eig, es_critico = classify_point(obj, r.x_min)
    eig_txt = ", ".join(f"{e:.4g}" for e in eig)
    etiqueta = {"mínimo": "MÍNIMO local", "máximo": "MÁXIMO local",
                "silla": "PUNTO SILLA", "degenerado": "DEGENERADO (no concluyente)"}[tipo]
    detalle = {"mínimo": "Hessiano definido positivo (todos los autovalores > 0).",
               "máximo": "Hessiano definido negativo (todos los autovalores < 0).",
               "silla": "Hessiano indefinido (autovalores de distinto signo).",
               "degenerado": "Hay un autovalor ≈ 0; la 2ª derivada no concluye."}[tipo]
    msg = f"**Clasificación del punto:** {etiqueta} — {detalle}  \nAutovalores del Hessiano: [{eig_txt}]"
    if not es_critico:
        msg += "  \nNota: el gradiente aún no es cero aquí, así que todavía no es un punto crítico (faltan iteraciones)."
    (st.success if tipo == "mínimo" and es_critico else st.warning)(msg)

    # Resumen de la 2ª condición de Wolfe sobre el paso final (solo backtracking)
    if r.line_search == "backtracking" and r.wolfe2_history:
        if r.wolfe2_history[-1]:
            st.success(f"**2ª condición de Wolfe:** el paso final **cumple** la condición de "
                       f"curvatura (σ = {r.sigma}).")
        else:
            st.warning(f"**2ª condición de Wolfe:** el paso final **no cumple** la condición de "
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
        f"Descargar CSV ({METHOD_NAMES[key]})",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"iteraciones_{key}.csv", mime="text/csv", key=f"dl_{key}")


def panel_title(text, icon_name):
    st.markdown(
        f'<div class="result-head" style="margin-top:2px">{icon(icon_name)}<b>{text}</b></div>',
        unsafe_allow_html=True,
    )


# ======================================================================
# Lógica principal
# ======================================================================
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
            ["  Resumen", "  Gráficos", "  Detalle"]
        )

        # --- Pestaña 1: Resumen (métricas + clasificación por método) ---
        with tab_resumen:
            for key, r in results.items():
                show_result_block(key, r, tol, obj)
                st.divider()

        # --- Pestaña 2: Gráficos (matplotlib con tema aplicado) ---
        with tab_graficos:
            with mpl_theme(pal):
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    panel_title("Gráfico de convergencia", "chart")
                    st.pyplot(plot_convergence(results))
                with gcol2:
                    if n_vars == 2:
                        panel_title("Trayectoria de optimización", "route")
                        st.pyplot(plot_contour(obj, results))
                    else:
                        st.info("La trayectoria sobre curvas de nivel solo se grafica para n = 2 variables.")

                # Tamaño de paso α — solo en backtracking
                if ls_label == "backtracking":
                    panel_title("Tamaño de paso α por iteración", "sliders")
                    st.pyplot(plot_step_sizes(results))
                    st.caption("Muestra el α aceptado en cada iteración (parte en α₀ y baja por ρ si hace falta). "
                               "Si comparas los 3 métodos, aparecen los tres.")

                # Mapa de cuencas de atracción (característica distintiva, solo n=2)
                if show_basins:
                    panel_title("Mapa de cuencas de atracción", "map")
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
            panel_title("Gradiente y Hessiano simbólicos", "sigma")
            st.latex(r"f(x) = " + obj.latex_f())
            st.latex(r"\nabla f(x) = " + obj.latex_grad())
            st.latex(r"\nabla^2 f(x) = " + obj.latex_hess())
            st.divider()
            panel_title("Historial completo de iteraciones", "list")
            for key, r in results.items():
                show_history_block(key, r, n_vars)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar la función o los parámetros:\n\n`{e}`")
        st.caption("Revisa que la función use x1, x2, ... y sintaxis válida (ej: `x1**2 + sin(x2)`).")
else:
    st.info("Configura los datos en el panel lateral y presiona **Ejecutar optimización**.")
    with st.expander("Funciones de ejemplo para probar", expanded=True):
        st.markdown(
            "- **Cuadrática:** `(x1-1)**2 + (x2-2)**2`  → mínimo en (1, 2), x0 = `0,0`\n"
            "- **Rosenbrock:** `100*(x2 - x1**2)**2 + (1 - x1)**2`  → mínimo en (1, 1), x0 = `-1.2, 1.0`\n"
            "- **Himmelblau:** `(x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2`, x0 = `0,0`\n"
            "- **3 variables:** `(x1-3)**2 + (x2+1)**2 + (x3-2)**2`, x0 = `0,0,0`\n"
            "- Funciones disponibles: `sin, cos, exp, log, sqrt, tan`, etc."
        )
