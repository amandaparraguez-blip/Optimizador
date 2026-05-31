"""
core/line_search.py
-------------------
Búsqueda de línea que satisface las CONDICIONES DE WOLFE (versión fuerte):
    (1ra) f(x + a*p) <= f(x) + c1 * a * grad(x)^T p           [Armijo / decrecimiento suficiente]
    (2da) |grad(x + a*p)^T p| <= c2 * |grad(x)^T p|           [curvatura]
Requiere 0 < c1 < c2 < 1 y que p sea dirección de descenso.

Algoritmo de Nocedal & Wright, "Numerical Optimization",
Algoritmos 3.5 (line search) y 3.6 (zoom).
"""
from __future__ import annotations
import numpy as np


def strong_wolfe_line_search(obj, x, p, c1=1e-4, c2=0.9, alpha_max=10.0, max_ls_iter=50):
    phi0 = obj.f(x)
    grad0 = obj.grad(x)
    dphi0 = float(grad0 @ p)  # phi'(0) = grad(x)^T p

    if dphi0 >= 0:  # p no es dirección de descenso
        return None, 1

    def phi(a):
        return obj.f(x + a * p)

    def dphi(a):
        return float(obj.grad(x + a * p) @ p)

    def zoom(a_lo, a_hi, phi_lo):
        for _ in range(max_ls_iter):
            a_j = 0.5 * (a_lo + a_hi)  # bisección (robusta)
            phi_j = phi(a_j)
            if (phi_j > phi0 + c1 * a_j * dphi0) or (phi_j >= phi_lo):
                a_hi = a_j
            else:
                dphi_j = dphi(a_j)
                if abs(dphi_j) <= -c2 * dphi0:
                    return a_j  # cumple ambas condiciones de Wolfe
                if dphi_j * (a_hi - a_lo) >= 0:
                    a_hi = a_lo
                a_lo = a_j
                phi_lo = phi_j
        return 0.5 * (a_lo + a_hi)

    a_prev, a_cur = 0.0, 1.0
    phi_prev = phi0
    for i in range(1, max_ls_iter + 1):
        phi_cur = phi(a_cur)
        if (phi_cur > phi0 + c1 * a_cur * dphi0) or (phi_cur >= phi_prev and i > 1):
            return zoom(a_prev, a_cur, phi_prev), 0
        dphi_cur = dphi(a_cur)
        if abs(dphi_cur) <= -c2 * dphi0:
            return a_cur, 0
        if dphi_cur >= 0:
            return zoom(a_cur, a_prev, phi_cur), 0
        a_prev, phi_prev = a_cur, phi_cur
        a_cur = min(2.0 * a_cur, alpha_max)
    return a_cur, 0
