# config.py
# Central configuration for all simulation settings.

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
FLUENT_EXE  = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\fluent\ntbin\win64\fluent.exe"
BASE_CASE   = r"D:\Ansys_works\2D_airfoil_NACA2412li.cas.h5"
RESULTS_DIR = r"D:\Ansys_works\results"

# ── Fluent Solver Settings ────────────────────────────────────────────────────
FLUENT_DIM  = "2ddp"      # 2ddp = 2D double precision
CORES       = 4           # Student version max = 4 cores (hard license limit)
ITERATIONS  = 2000        # max iterations per simulation

# ── Airfoil / Reference Values ────────────────────────────────────────────────
CHORD       = 1.0         # m
REF_AREA    = 1.0         # m²

# ── Fluid Properties (Air at standard conditions) ─────────────────────────────
DENSITY     = 1.225       # kg/m³
VISCOSITY   = 1.7894e-5   # Pa·s

# ── Turbulence Settings (k-omega SST) ────────────────────────────────────────
TURB_INTENSITY  = 5       # % turbulence intensity
TURB_VISC_RATIO = 10      # turbulent viscosity ratio

# ── Parameter Sweep ───────────────────────────────────────────────────────────
# 5 velocities × 10 AoA = 50 total simulations
VELOCITIES = np.linspace(10, 50, 5).tolist()    # [10, 20, 30, 40, 50] m/s
AOA_LIST   = np.linspace(-5, 20, 10).tolist()   # 10 AoA values from -5° to 20°

# ── Zone Names ────────────────────────────────────────────────────────────────
INLET_ZONE   = "inlet"
OUTLET_ZONE  = "outlet"
AIRFOIL_ZONE = "airfoil"
TOP_ZONE     = "top"
BOTTOM_ZONE  = "bottom"