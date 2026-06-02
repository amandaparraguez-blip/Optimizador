"""
core/line_search.py
-------------------
Búsquedas de línea disponibles:

1) WOLFE FUERTE (Armijo + curvatura), Nocedal & Wright Alg. 3.5/3.6:
       (1ra) f(x + a*p) <= f(x) + c1 * a * grad(x)^T p          [Armijo]
       (2da) |grad(x + a*p)^T p| <= c2 * |grad(x)^T p|          [curvatura fuerte]
   Usa paso inicial alpha0 y requiere 0 < c1 < c2 < 1.

2) BACKTRACKING (solo Armijo) con factor de reducción rho:
       parte en alpha0 y lo multiplica por rho hasta cumplir Armijo.
   Parámetros: alpha0 (paso inicial), c1 (= beta de Armijo), rho (factor de reducción).
"""
from __future__ import annotations
import numpy as np


def strong_wolfe_line_search(obj, x, p, c1=1e-4, c2=0.9, alpha0=1.0,
                             alpha_max=10.0, max_ls_iter=50):
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

    a_prev, a_cur = 0.0, alpha0   # <-- ahora el paso inicial es alpha0
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


def backtracking_line_search(obj, x, p, c1=1e-4, alpha0=1.0, rho=0.5, max_ls_iter=50):
    """
    Backtracking de Armijo: parte en alpha0 y reduce alpha <- rho*alpha
    hasta cumplir  f(x + a*p) <= f(x) + c1 * a * grad(x)^T p.
    Devuelve (alpha, flag). flag=1 si p no es dirección de descenso.
    """
    phi0 = obj.f(x)
    dphi0 = float(obj.grad(x) @ p)
    if dphi0 >= 0:  # p no es dirección de descenso
        return None, 1
    alpha = alpha0
    for _ in range(max_ls_iter):
        if obj.f(x + alpha * p) <= phi0 + c1 * alpha * dphi0:
            return alpha, 0
        alpha *= rho
    return alpha, 0