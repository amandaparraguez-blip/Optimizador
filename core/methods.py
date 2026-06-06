"""
core/methods.py
---------------
Métodos de optimización con búsqueda de línea de Wolfe:
    - Gradiente (descenso más pronunciado)
    - Gradiente Conjugado (Fletcher-Reeves "FR" / Polak-Ribiere "PR")
    - Newton (con regularización para Hessianos no definidos positivos)

Criterio de parada: ||grad(x)|| < tol  o  max_iter alcanzado.
"""
from __future__ import annotations
import numpy as np

from core.line_search import strong_wolfe_line_search, backtracking_line_search

# Nombres legibles de cada método (para mostrar en la interfaz)
METHOD_NAMES = {
    "gradient": "Método del Gradiente",
    "conjugate_gradient": "Gradiente Conjugado",
    "newton": "Método de Newton",
}


class OptimizationResult:
    """Resultado estándar devuelto por cualquier método."""
    def __init__(self):
        self.x_min = None            # punto mínimo encontrado
        self.f_min = None            # valor de f en el mínimo
        self.n_iter = 0              # iteraciones realizadas
        self.final_error = None      # error final (||gradiente||)
        self.stop_reason = ""        # criterio de parada alcanzado
        self.error_history = []      # ||gradiente|| por iteración (para gráfico)
        self.f_history = []          # f(x) por iteración
        self.path = []               # lista de puntos x (para trayectoria 2D)
        self.success = True
        self.line_search = "wolfe"   # tipo de búsqueda de línea usada
        self.sigma = None            # parámetro de la 2da condición (solo backtracking)
        self.wolfe2_history = []     # ✓/✗ de la 2da condición de Wolfe por paso (backtracking)
        self.alpha_history = []      # α (tamaño de paso) aceptado en cada iteración
        self.armijo_lhs_history = [] # lado izquierdo de Armijo: f(x_k + α p)
        self.armijo_rhs_history = [] # lado derecho de Armijo: f(x_k) + c1·α·∇f(x_k)^T p


def _solve_newton_direction(H, g, n):
    """
    Resuelve H p = -g. Si H no es definida positiva, la regulariza sumando
    tau*I hasta que lo sea (modificación tipo Levenberg-Marquardt),
    garantizando que p sea dirección de descenso.
    """
    I = np.eye(n)
    tau = 0.0
    beta = 1e-3
    min_diag = np.min(np.diag(H))
    if min_diag <= 0:
        tau = -min_diag + beta
    for _ in range(50):
        try:
            np.linalg.cholesky(H + tau * I)  # éxito => definida positiva
            return np.linalg.solve(H + tau * I, -g)
        except np.linalg.LinAlgError:
            tau = max(2.0 * tau, beta)
    return -g  # último recurso: descenso de gradiente


def classify_point(obj, x, tol=1e-6):
    """
    Clasifica el punto x según el Hessiano:
        - todos los autovalores > 0  -> mínimo local (Hessiano definido positivo)
        - todos < 0                  -> máximo local (definido negativo)
        - signos mezclados           -> punto silla
        - alguno ≈ 0                 -> caso degenerado (no concluyente con 2do orden)
    Devuelve (tipo, autovalores, es_critico).
    """
    g = obj.grad(np.array(x, dtype=float))
    es_critico = bool(np.linalg.norm(g) < 1e-3)
    H = obj.hess(np.array(x, dtype=float))
    eig = np.linalg.eigvalsh(H) if H.shape[0] > 1 else np.array([H[0, 0]])
    eps = 1e-8
    if np.any(np.abs(eig) < eps):
        tipo = "degenerado"
    elif np.all(eig > 0):
        tipo = "mínimo"
    elif np.all(eig < 0):
        tipo = "máximo"
    else:
        tipo = "silla"
    return tipo, eig, es_critico


def minimize(obj, x0, method="gradient", max_iter=1000, tol=1e-6,
             c1=1e-4, c2=0.9, cg_variant="PR",
             line_search="wolfe", alpha0=1.0, rho=0.5, sigma=0.5) -> OptimizationResult:
    res = OptimizationResult()
    res.line_search = line_search
    res.sigma = sigma
    x = np.array(x0, dtype=float)
    n = obj.n_vars

    g = obj.grad(x)
    res.path.append(x.copy())
    res.error_history.append(float(np.linalg.norm(g)))
    res.f_history.append(obj.f(x))

    p = -g
    g_prev = g.copy()
    k = 0

    for k in range(1, max_iter + 1):
        grad_norm = float(np.linalg.norm(g))
        if grad_norm < tol:
            res.stop_reason = f"Convergencia: ||gradiente|| = {grad_norm:.3e} < tol = {tol:.1e}"
            break

        # --- Dirección de descenso según el método ---
        if method == "gradient":
            p = -g
        elif method == "conjugate_gradient":
            if k == 1:
                p = -g
            else:
                if cg_variant == "FR":
                    beta = (g @ g) / (g_prev @ g_prev)
                else:  # Polak-Ribiere con salvaguarda (PR+)
                    beta = max(0.0, (g @ (g - g_prev)) / (g_prev @ g_prev))
                p = -g + beta * p
                if g @ p >= 0:  # si pierde el descenso, reiniciar
                    p = -g
        elif method == "newton":
            p = _solve_newton_direction(obj.hess(x), g, n)
        else:
            raise ValueError(f"Método desconocido: {method}")

        # --- Búsqueda de línea (Wolfe fuerte, Backtracking o Paso fijo) ---
        def _line_search(direction):
            if line_search == "fixed":
                return alpha0, 0  # paso constante, sin búsqueda
            if line_search == "backtracking":
                return backtracking_line_search(obj, x, direction, c1=c1,
                                                alpha0=alpha0, rho=rho)
            return strong_wolfe_line_search(obj, x, direction, c1=c1, c2=c2,
                                            alpha0=alpha0)

        alpha, flag = _line_search(p)
        # En paso fijo NO se reinicia ni se busca: se respeta el alpha tal cual.
        if line_search != "fixed" and (alpha is None or flag != 0 or alpha == 0):
            p = -g  # reiniciar a gradiente
            alpha, flag = _line_search(p)
            if alpha is None:
                res.stop_reason = "Detenido: no se encontró paso válido (búsqueda de línea)."
                res.success = False
                break

        # --- Actualizar punto ---
        x_new = x + alpha * p
        g_new = obj.grad(x_new)
        f_old = res.f_history[-1]
        f_new = obj.f(x_new)
        dphi0 = float(g @ p)

        # Registrar el paso α y los dos lados de la condición de Armijo:
        #   LHS = f(x_k + α p)        ;   RHS = f(x_k) + c1·α·∇f(x_k)^T p
        res.alpha_history.append(float(alpha))
        res.armijo_lhs_history.append(float(f_new))
        res.armijo_rhs_history.append(float(f_old + c1 * alpha * dphi0))

        # Chequeo (informativo) de la 2da condición de Wolfe sobre el paso aceptado:
        #   grad(x_new)^T p  >=  sigma * grad(x)^T p     (condición de curvatura)
        if line_search == "backtracking":
            dphi_new = float(g_new @ p)
            res.wolfe2_history.append(bool(dphi_new >= sigma * dphi0))

        g_prev, x, g = g, x_new, g_new

        res.path.append(x.copy())
        res.error_history.append(float(np.linalg.norm(g)))
        res.f_history.append(f_new)
        res.n_iter = k

        if not np.all(np.isfinite(x)):
            res.stop_reason = "Detenido: divergencia numérica (valores no finitos)."
            res.success = False
            break
    else:
        res.stop_reason = f"Máximo de iteraciones alcanzado ({max_iter})."

    res.x_min = x
    res.f_min = obj.f(x)
    res.final_error = float(np.linalg.norm(obj.grad(x)))
    res.n_iter = res.n_iter or k
    return res