"""
figure_utils.py — Única fuente de estilo IEEE/A4 — Proyecto WTI Tesis
=====================================================================
Versión : 2.3
Fecha   : 2026-03-21
Cambios v2.3:
  - Paletas canónicas completas para los 4 notebooks
  - COLORS_SIGNALS agregado (NB03B: OVX, VIX, DXY, EIA)
  - COLORES_MODELOS agregado (NB02: benchmark completo)
  - COLOR_PRECIO, COLOR_CLEAN, COLOR_VAR, COLOR_DYN unificados
  - __all__ y NOTEBOOK_IMPORT_SNIPPET actualizados
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# CONSTANTES DE PÁGINA (pulgadas)
# =============================================================================
_PAGE_DIM = {
    "A4": {"w": 8.27, "h": 11.69},
    "letter": {"w": 8.50, "h": 11.00},
}

_MARGINS = {
    "top": 0.98,
    "bottom": 0.98,
    "left": 1.18,
    "right": 0.98,
}


def _text_area(fmt: str = "A4") -> tuple[float, float]:
    """Retorna (ancho, alto) de área de texto útil en pulgadas."""
    if fmt not in _PAGE_DIM:
        raise ValueError(f"fmt debe ser 'A4' o 'letter'. Recibido: {fmt!r}")
    p = _PAGE_DIM[fmt]
    w = p["w"] - _MARGINS["left"] - _MARGINS["right"]
    h = p["h"] - _MARGINS["top"] - _MARGINS["bottom"]
    return round(w, 4), round(h, 4)


# =============================================================================
# FIG_SIZE
# =============================================================================
def fig_size(layout: str = "full", fmt: str = "A4", aspect: float = 0.618) -> tuple[float, float]:
    """
    Retorna figsize (w, h) en pulgadas para impresión A4/Letter.

    Layouts:
      full, half, wide, 2x1, 2x2, 3x1, square, tall, page,
      nb03_ts, nb03_dist, nb03_corr
    """
    tw, th = _text_area(fmt)

    opts = {
        "full":      (tw, tw * aspect),
        "half":      (tw / 2, (tw / 2) * aspect),
        "wide":      (tw, tw * 0.45),
        "2x1":       (tw, tw * 0.50),
        "2x2":       (tw, tw * 0.75),
        "3x1":       (tw, tw * 0.38),
        "square":    (tw, tw),
        "tall":      (tw, min(tw * 1.40, th * 0.85)),
        "page":      (tw, th * 0.90),
        # Presets específicos NB03
        "nb03_ts":   (tw, min(tw * 1.45, th * 0.92)),
        "nb03_dist": (tw, tw * 0.62),
        "nb03_corr": (tw, tw * 0.95),
    }

    if layout not in opts:
        warnings.warn(
            f"Layout '{layout}' no reconocido. Usando 'full'. Opciones: {list(opts.keys())}",
            UserWarning,
            stacklevel=2,
        )
        layout = "full"

    return tuple(round(x, 4) for x in opts[layout])


# =============================================================================
# SAVE_FIG
# =============================================================================
def save_fig(
    fig_or_none=None,
    filename: str = "",
    output_path: str = ".",
    dpi: int = 300,
    fmt: str = "png",
    fmt_page: str = "A4",
    warn: bool = True,
    **kwargs,
) -> str:
    """
    Guarda figura con parámetros IEEE y validación dimensional.
    """
    if not filename or not filename.strip():
        raise ValueError("filename no puede estar vacío en save_fig().")

    fig = fig_or_none if fig_or_none is not None else plt.gcf()
    w, h = fig.get_size_inches()
    tw, th = _text_area(fmt_page)

    # Tall/page suelen ser altos por diseño
    is_tall_layout = h > tw * 1.30
    if warn and not is_tall_layout and (w > tw * 1.05 or h > th * 0.93):
        warnings.warn(
            f"'{filename}': {w:.2f}\"×{h:.2f}\" excede área {fmt_page} "
            f"({tw:.2f}\"×{th:.2f}\"). Considera fig_size().",
            UserWarning,
            stacklevel=2,
        )

    os.makedirs(output_path, exist_ok=True)
    filename_safe = filename.replace(" ", "_").replace("/", "_")
    path = os.path.join(output_path, f"{filename_safe}.{fmt}")

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        edgecolor="none",
        **kwargs,
    )
    print(f"  [OK] {path}  [{w:.1f}\"×{h:.1f}\" @ {dpi}dpi]")
    return path


# =============================================================================
# APPLY_THESIS_STYLE
# =============================================================================
def apply_thesis_style(fmt_page: str = "A4") -> None:
    """
    Aplica estilo IEEE del proyecto WTI.
    Llamar una sola vez al inicio del notebook.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass

    plt.rcParams.update({
        # Fuentes
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern", "DejaVu Serif"],
        "font.size": 10,

        # Títulos / ejes
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "center",
        "axes.titlepad": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.pad": 3.5,
        "ytick.major.pad": 3.5,

        # Leyendas
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "gray",

        # Resolución / fondo
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "figure.edgecolor": "none",
        "axes.facecolor": "white",

        # Grid / bordes
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,

        # Líneas
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    })

    tw, th = _text_area(fmt_page)
    print(f"[IEEE v2.3] Estilo aplicado — {fmt_page}: {tw:.2f}\"×{th:.2f}\" ({tw*2.54:.1f}×{th*2.54:.1f} cm)")
    print("  Layouts (figsize):")
    for lyt in ["full", "half", "wide", "2x1", "2x2", "3x1", "tall", "page", "nb03_ts", "nb03_dist", "nb03_corr"]:
        ww, hh = fig_size(lyt, fmt_page)
        print(f"    fig_size('{lyt:<9}') -> ({ww:.2f}, {hh:.2f})\"")


# =============================================================================
# HELPER OPCIONAL: LEYENDA ARRIBA
# =============================================================================
def add_top_legend(
    handles: Iterable,
    labels: Optional[Iterable[str]] = None,
    ax=None,
    fig=None,
    ncol: int = 4,
    y: float = 0.98,
    fontsize: float = 8.5,
    frameon: bool = True,
):
    """
    Coloca leyenda arriba.
    - Si pasas ax: leyenda del eje.
    - Si pasas fig: leyenda global.
    """
    if fig is None and ax is None:
        raise ValueError("Debes pasar ax o fig a add_top_legend().")

    target = fig if fig is not None else ax.figure
    if fig is not None:
        return target.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=ncol,
            fontsize=fontsize,
            frameon=frameon,
            columnspacing=1.2,
            handletextpad=0.6,
        )
    else:
        return ax.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=ncol,
            fontsize=fontsize,
            frameon=frameon,
            columnspacing=1.2,
            handletextpad=0.6,
        )


# =============================================================================
# PALETAS CANÓNICAS
# =============================================================================
# 
# PALETAS CANÓNICAS — Proyecto WTI Tesis

#  1. SPLITS TEMPORALES 
SPLIT_COLORS = {
    'train':      '#2166ac',   # azul IEEE — período base
    'validation': '#f4a582',   # salmón — período COVID (cálido, distinto)
    'val':        '#f4a582',   # alias
    'test':       '#d6604d',   # rojo oscuro — período post-pandemia
}

#  2. CLASES (Normal vs Extremo) 
# CAMBIO: unifica COLOR_NORMAL_NB01 y COLOR_NORMAL en uno solo
# El azul del train = azul del normal (semánticamente coherente)
COLOR_NORMAL  = '#2166ac'   # azul IEEE
COLOR_EXTREMO = '#d73027'   # rojo fuerte — alerta/peligro

#  3. SERIES PRINCIPALES 
# NUEVO: COLOR_PRECIO faltaba completamente
COLOR_PRECIO  = '#253494'   # azul profundo para precio WTI (más oscuro que vol)
COLOR_VOL     = '#2c7bb6'   # azul medio para vol21_clean
# CAMBIO: COLOR_CLEAN era verde (#27ae60) — no coordinaba con azules
COLOR_CLEAN   = '#4575b4'   # azul claro — serie CLEAN (misma familia que VOL)
# CAMBIO: COLOR_VAR era rojo oscuro (#8B2020) — chocaba con COLOR_EXTREMO
COLOR_VAR     = '#74add1'   # azul grisáceo — VaR (más frío, sin alarma)
# CAMBIO: COLOR_DYN era darkorange — demasiado brillante
COLOR_DYN     = '#f4a582'   # salmón — umbral dinámico (mismo que val, es temporal)

# ── 4. PERCENTILES 
# CAMBIO: P90 era #f0a500 (dorado brillante) → amarillo-naranja más sobrio
COLORS_P = {
    90: '#fdae61',   # naranja claro — umbral conservador
    95: '#d73027',   # rojo — umbral principal (mismo que COLOR_EXTREMO)
    99: '#7b2d8b',   # morado — umbral más estricto
}

# ── 5. MODELOS DEL BENCHMARK (NB02) 
# NUEVO — faltaba completamente, cada notebook lo hardcodeaba
COLORES_MODELOS = {
    'GARCH':               '#7b2d8b',   # morado — econométrico clásico
    'LSTM-Simple-Uni':     '#74add1',   # azul claro
    'LSTM-Simple-Multi':   '#4575b4',   # azul medio
    'BiDir-LSTM-Uni':      '#d73027',   # rojo — GANADOR del benchmark
    'BiDir-LSTM-Multi':    '#f46d43',   # naranja-rojo
    'Hybrid-Sent':         '#fdae61',   # naranja claro — resultado negativo
    'Hybrid-MktSignals':   '#fee090',   # amarillo suave
    'Baseline-B1':         '#aaaaaa',   # gris — baseline ingenuo
    'Baseline-B5':         '#555555',   # gris oscuro — baseline cuantitativo
}

# ── 6. BASELINES NB01 (fig_16)
# CAMBIO: reordena para que B5 (cuantitativo) sea gris oscuro,
#         coherente con COLORES_MODELOS
COLORES_B = [
    '#aaaaaa',   # B0 — Siempre Normal (gris claro)
    '#2166ac',   # B1 — |ret_lag1| > P95 (azul — referencia mínima)
    '#f4a582',   # B2 — |ret_lag1| > P75 (salmón)
    '#4575b4',   # B3 — vol21 > P95 (azul medio)
    '#d6604d',   # B4 — VaR violation (rojo oscuro)
    '#7b2d8b',   # B5 — Christoffersen (morado — cuantitativo)
]

# ── 7. SENTIMIENTO (NB03) 
# Sin cambios — ya estaban bien coordinados
COLORS_SENT = {
    'positive': '#1a9641',   # verde — sentimiento positivo
    'negative': '#d7191c',   # rojo — sentimiento negativo
    'neutral':  '#b0b0b0',   # gris — sentimiento neutro
}

COLORS_SIGNALS = {
    'OVX': '#d73027',    # rojo — volatilidad implícita crudo (alarma)
    'VIX': '#4575b4',    # azul — volatilidad implícita mercado
    'DXY': '#1a9641',    # verde — dólar (convención forex)
    'EIA': '#7b2d8b',    # morado — inventarios (estructural)
}

# =============================================================================
# SNIPPET DE IMPORT PARA NOTEBOOKS
# =============================================================================
NOTEBOOK_IMPORT_SNIPPET = r"""
# --- Importar figure_utils ---
import os, sys
import matplotlib as mpl

_utils_paths = [
    '/content/drive/My Drive/2410VDSO Trabajo de Grado/Modelos/utils',
    os.path.abspath('./utils'),
    os.path.abspath('../utils'),
]

for p in _utils_paths:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from figure_utils import (
    fig_size, save_fig, apply_thesis_style, add_top_legend,
    SPLIT_COLORS, COLOR_NORMAL, COLOR_EXTREMO,
    COLOR_PRECIO, COLOR_VOL, COLOR_CLEAN, COLOR_VAR, COLOR_DYN,
    COLORS_P, COLORES_MODELOS, COLORES_B,
    COLORS_SENT, COLORS_SIGNALS,
)

mpl.rcParams.update(mpl.rcParamsDefault)
apply_thesis_style('A4')
"""


__all__ = [
    "fig_size", "save_fig", "apply_thesis_style", "add_top_legend",
    # Splits y clases
    "SPLIT_COLORS", "COLOR_NORMAL", "COLOR_EXTREMO",
    # Series principales
    "COLOR_PRECIO", "COLOR_VOL", "COLOR_CLEAN", "COLOR_VAR", "COLOR_DYN",
    # Percentiles y modelos
    "COLORS_P", "COLORES_MODELOS", "COLORES_B",
    # NB03A y NB03B
    "COLORS_SENT", "COLORS_SIGNALS",
    # Utilidad
    "NOTEBOOK_IMPORT_SNIPPET",
]


# =============================================================================
# TEST RÁPIDO
# =============================================================================
if __name__ == "__main__":
    import numpy as np

    print("=" * 55)
    print("TEST figure_utils v3")
    print("=" * 55)

    apply_thesis_style("A4")

    fig, ax = plt.subplots(figsize=fig_size("full"))
    x = np.linspace(0, 2 * np.pi, 200)
    ax.plot(x, np.sin(x), label="sin(x)", color=COLOR_VOL)
    ax.plot(x, np.cos(x), label="cos(x)", ls="--", color=COLOR_EXTREMO)
    ax.set_title("Test figure_utils v2.3")
    ax.legend(loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_fig(fig, "test_figure_utils_v22", output_path=".", dpi=150)
    plt.close(fig)

    print("[PASS] Test básico OK")