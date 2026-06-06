"""
core/objective.py
-----------------
Convierte una función escrita como texto (ej. "100*(x2 - x1**2)**2 + (1 - x1)**2")
en funciones numéricas rápidas (f, gradiente, Hessiano) usando SymPy + NumPy.
Las variables se llaman x1, x2, ..., xn.
"""
from __future__ import annotations
import numpy as np
import sympy as sp


class ObjectiveFunction:
    def __init__(self, expr_str: str, n_vars: int):
        self.expr_str = expr_str
        self.n_vars = n_vars
        self.symbols = sp.symbols(f"x1:{n_vars + 1}", real=True)  # x1..xn

        local_dict = {f"x{i+1}": self.symbols[i] for i in range(n_vars)}
        transformations = sp.parsing.sympy_parser.standard_transformations + (
            sp.parsing.sympy_parser.implicit_multiplication_application,
            sp.parsing.sympy_parser.convert_xor,  # permite usar ^ como potencia
        )
        self.expr = sp.parsing.sympy_parser.parse_expr(
            expr_str, local_dict=local_dict, transformations=transformations
        )

        self.grad_expr = [sp.diff(self.expr, s) for s in self.symbols]
        self.hess_expr = sp.hessian(self.expr, self.symbols)

        self._f = sp.lambdify(self.symbols, self.expr, "numpy")
        self._grad = sp.lambdify(self.symbols, self.grad_expr, "numpy")
        self._hess = sp.lambdify(self.symbols, self.hess_expr, "numpy")

    # --- Evaluaciones numéricas ---
    def f(self, x: np.ndarray) -> float:
        return float(self._f(*x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._grad(*x), dtype=float).flatten()

    def hess(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._hess(*x), dtype=float).reshape(self.n_vars, self.n_vars)

    # --- Representaciones LaTeX (para mostrar en la web) ---
    def latex_f(self) -> str:
        return sp.latex(self.expr)

    def latex_grad(self) -> str:
        return sp.latex(sp.Matrix(self.grad_expr))

    def latex_hess(self) -> str:
        return sp.latex(self.hess_expr)