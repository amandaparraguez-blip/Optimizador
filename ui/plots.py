"""
ui/plots.py
-----------
Funciones de gráficos para la interfaz:
    - plot_convergence: error ||grad|| vs. iteraciones (escala log)
    - plot_contour: curvas de nivel + trayectoria de iterados (solo n=2)
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from core import METHOD_NAMES


def plot_convergence(results: dict):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, r in results.items():
        ax.semilogy(range(len(r.error_history)), r.error_history,
                    marker="o", markersize=3, linewidth=1.6, label=METHOD_NAMES[key])
    ax.set_xlabel("Número de iteraciones")
    ax.set_ylabel("Error  ||∇f(x)||  (escala log)")
    ax.set_title("Gráfico de convergencia: error vs. iteraciones")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_contour(obj, results: dict):
    """Curvas de nivel + trayectoria de iterados (solo para n = 2)."""
    all_pts = np.vstack([np.array(r.path) for r in results.values()])
    x_min_, x_max_ = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min_, y_max_ = all_pts[:, 1].min(), all_pts[:, 1].max()
    pad_x = max(0.5, 0.25 * (x_max_ - x_min_))
    pad_y = max(0.5, 0.25 * (y_max_ - y_min_))
    xs = np.linspace(x_min_ - pad_x, x_max_ + pad_x, 250)
    ys = np.linspace(y_min_ - pad_y, y_max_ + pad_y, 250)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = obj.f(np.array([X[i, j], Y[i, j]]))
            except Exception:
                Z[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(7, 5.5))
    levels = np.linspace(np.nanmin(Z), np.nanpercentile(Z, 92), 30)
    cs = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.85)
    ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.3, alpha=0.4)
    fig.colorbar(cs, ax=ax, label="f(x)")
    colors = ["#ff3b3b", "#ffd23b", "#3bff8f"]
    for c, (key, r) in zip(colors, results.items()):
        path = np.array(r.path)
        ax.plot(path[:, 0], path[:, 1], "-o", color=c, markersize=3,
                linewidth=1.4, label=METHOD_NAMES[key])
        ax.plot(path[-1, 0], path[-1, 1], "*", color=c, markersize=16,
                markeredgecolor="black")
    p0 = np.array(list(results.values())[0].path)[0]
    ax.plot(p0[0], p0[1], "ks", markersize=8, label="Punto inicial")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title("Trayectoria de optimización sobre curvas de nivel")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_basins(obj, minimize_fn, method, ls_kwargs, region=(-6, 6, -6, 6),
                grid_n=40, max_iter=200, tol=1e-4):
    """
    Mapa de cuencas de atracción (solo n=2): corre el método desde una grilla
    de puntos de partida y pinta el plano según a qué mínimo llega cada uno.
    """
    import matplotlib.colors as mcolors

    x_lo, x_hi, y_lo, y_hi = region
    xs = np.linspace(x_lo, x_hi, grid_n)
    ys = np.linspace(y_lo, y_hi, grid_n)
    labels = np.full((grid_n, grid_n), -1, dtype=int)
    minima = []  # representantes de cada mínimo encontrado

    def find_label(pt):
        for idx, m in enumerate(minima):
            if np.linalg.norm(pt - m) < 0.25:
                return idx
        minima.append(pt)
        return len(minima) - 1

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            try:
                r = minimize_fn(obj, np.array([x, y]), method=method,
                                max_iter=max_iter, tol=tol, **ls_kwargs)
                if r.final_error < 1e-2 and np.all(np.isfinite(r.x_min)):
                    labels[iy, ix] = find_label(np.round(r.x_min, 1))
            except Exception:
                pass

    n_min = len(minima)
    # Paleta: gris para "no convergió" (-1) + colores para cada mínimo
    palette = ["#888888"] + list(plt.cm.tab10(np.linspace(0, 1, max(n_min, 1))))
    cmap = mcolors.ListedColormap(palette[: n_min + 1])
    disp = labels + 1  # desplazar para que -1 -> 0 (gris)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.pcolormesh(xs, ys, disp, cmap=cmap, vmin=0, vmax=n_min, shading="auto")
    # marcar los mínimos encontrados
    for idx, m in enumerate(minima):
        ax.plot(m[0], m[1], "*", color="white", markersize=16, markeredgecolor="black")
        ax.annotate(f"({m[0]:.2f}, {m[1]:.2f})", (m[0], m[1]),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=8, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title(f"Mapa de cuencas de atracción ({n_min} mínimos encontrados)")
    fig.tight_layout()
    return fig, n_min


def plot_step_sizes(results):
    """
    Gráfico del tamaño de paso α aceptado en cada iteración (útil en backtracking,
    donde α puede reducirse). Muestra los métodos disponibles en 'results'.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    algo = False
    for key, r in results.items():
        if r.alpha_history:
            iters = range(1, len(r.alpha_history) + 1)
            ax.plot(iters, r.alpha_history, marker="o", markersize=4, linewidth=1.6,
                    drawstyle="steps-post", label=METHOD_NAMES[key])
            algo = True
    ax.set_xlabel("Número de iteración")
    ax.set_ylabel("Tamaño de paso α")
    ax.set_title("Tamaño de paso α por iteración")
    ax.grid(True, alpha=0.3)
    if algo:
        ax.legend()
    fig.tight_layout()
    return fig