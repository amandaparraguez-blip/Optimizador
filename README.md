# Optimizador Web — Métodos de Optimización

Aplicación web que encuentra el mínimo de una función mediante el **método del
gradiente**, el **gradiente conjugado** y el **método de Newton**, todos con
**búsqueda de línea que cumple las condiciones de Wolfe** (Armijo + curvatura).

## Estructura del proyecto

```
optimizador/
│
├── app.py                  # ① Punto de entrada de la web (Streamlit apunta aquí)
├── requirements.txt        # Dependencias
├── README.md               # Este archivo
├── .gitignore
│
├── core/                   # ② NÚCLEO MATEMÁTICO (independiente de la interfaz)
│   ├── __init__.py         #    expone: ObjectiveFunction, minimize, METHOD_NAMES
│   ├── objective.py        #    función objetivo: f, gradiente y Hessiano (SymPy)
│   ├── line_search.py      #    búsqueda de línea con condiciones de Wolfe
│   └── methods.py          #    los 3 métodos + driver minimize() + OptimizationResult
│
├── ui/                     # ③ COMPONENTES DE INTERFAZ
│   ├── __init__.py
│   └── plots.py            #    gráfico de convergencia + trayectoria (curvas de nivel)
│
└── tests/                  # ④ PRUEBAS DE CORRECTITUD
    └── test_methods.py     #    valida contra funciones de mínimo conocido
```

**Idea del diseño:** `core/` no sabe nada de la web (puro Python/NumPy/SymPy), así
que se puede probar solo o reutilizar en otra interfaz. `app.py` y `ui/` solo se
encargan de mostrar. Esto hace el proyecto fácil de mantener y de explicar al profesor.

## A. Ejecutar localmente en VS Code

1. Instala **Python 3.10+** (marca "Add Python to PATH").
2. Instala **VS Code** + la extensión **Python** de Microsoft.
3. Abre la carpeta `optimizador` y en la terminal:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate            # Windows
   source .venv/bin/activate         # macOS/Linux
   pip install -r requirements.txt
   streamlit run app.py
   ```

   Se abre en http://localhost:8501

4. Para correr las pruebas:  `python -m tests.test_methods`

## B. Publicar el enlace público (GRATIS — Streamlit Community Cloud)

1. Sube todo el proyecto a un repositorio de **GitHub** (respetando las carpetas).
2. Entra a **https://share.streamlit.io** e inicia sesión con GitHub.
3. **New app** → elige el repo, rama `main` y archivo principal `app.py`.
4. **Deploy**. En 1–2 minutos tendrás un link `https://...streamlit.app`
   para enviar al profesor. Cada `git push` actualiza la app.

## Cumplimiento del enunciado

| Requisito | Dónde |
|---|---|
| N° variables, método, función, punto inicial, máx. iteraciones, tolerancia, parámetros de Wolfe | `app.py` (panel lateral) |
| Punto mínimo, valor de f, n° iteraciones, error final, criterio de parada | `app.py` → `show_result_block` |
| Gráfico de convergencia (error vs. iteraciones) | `ui/plots.py` → `plot_convergence` |
| Valor agregado | Trayectoria 2D, gradiente/Hessiano simbólicos, historial CSV, comparación de los 3 métodos |
