"""
Paquete core: toda la lógica matemática del optimizador.
Permite importar de forma limpia:  from core import ObjectiveFunction, minimize
"""
from core.objective import ObjectiveFunction
from core.line_search import strong_wolfe_line_search
from core.methods import minimize, OptimizationResult, METHOD_NAMES

__all__ = [
    "ObjectiveFunction",
    "strong_wolfe_line_search",
    "minimize",
    "OptimizationResult",
    "METHOD_NAMES",
]
