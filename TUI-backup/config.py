# config.py
# Central configuration for all simulation settings.
# Edit this file to change sweep ranges, paths, and solver settings.

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
FLUENT_EXE  = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\fluent\ntbin\win64\fluent.exe"
BASE_CASE   = r"D:\Ansys_works\2D_airfoil_NACA2412li.cas.h5"
RESULTS_DIR = r"D:\Ansys_works\results"

# ── Fluent Solver Settings ────────────────────────────────────────────────────
FLUENT_DIM  = "2ddp"        # 2d or 3d
CORES       = 4           # CPU cores (max 4 for student version)
ITERATIONS  = 2000         # iterations per simulation

# ── Airfoil / Reference Values ────────────────────────────────────────────────
CHORD       = 1.0         # m
REF_AREA    = 1.0         # m² (chord × unit span for 2D)

# ── Fluid Properties (Air at standard conditions) ─────────────────────────────
DENSITY     = 1.225       # kg/m³
VISCOSITY   = 1.7894e-5   # Pa·s

# ── Turbulence Settings (k-omega SST) ────────────────────────────────────────
TURB_INTENSITY    = 5     # % — freestream turbulence intensity
TURB_VISC_RATIO   = 10    # turbulent viscosity ratio

# ── Parameter Sweep ───────────────────────────────────────────────────────────
# Velocity range (m/s)
VELOCITIES  = np.linspace(10, 50, 5).tolist()   # [10, 20, 30, 40, 50] m/s

# Angle of attack range (degrees)
AOA_LIST    = np.linspace(-5, 20, 6).tolist()   # [-5, 1, 7, 13, 19, 20] deg

# ── Zone Names (must match exactly what's in Fluent) ─────────────────────────
INLET_ZONE   = "inlet"
OUTLET_ZONE  = "outlet"
AIRFOIL_ZONE = "airfoil"
TOP_ZONE     = "top"
BOTTOM_ZONE  = "bottom"