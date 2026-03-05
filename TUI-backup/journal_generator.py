# journal_generator.py
# Generates Fluent TUI journal files for each simulation run.
# Handles velocity decomposition for angle of attack.

import os
import math
from config import *


def decompose_velocity(velocity, aoa_deg):
    """
    Decomposes freestream velocity into X and Y components
    based on angle of attack.

    At AoA=0:  Vx = V,    Vy = 0
    At AoA=10: Vx = V*cos(10°), Vy = V*sin(10°)

    This simulates angle of attack WITHOUT rotating the mesh —
    one mesh works for all angles!
    """
    aoa_rad = math.radians(aoa_deg)
    Vx = velocity * math.cos(aoa_rad)
    Vy = velocity * math.sin(aoa_rad)
    return Vx, Vy


def generate_journal(run_id, velocity, aoa_deg, run_dir):
    """
    Generates a complete Fluent TUI journal file for one simulation.

    Parameters:
        run_id   : unique run identifier
        velocity : freestream velocity (m/s)
        aoa_deg  : angle of attack (degrees)
        run_dir  : folder to save results

    Returns:
        path to the generated .jou file
    """
    os.makedirs(run_dir, exist_ok=True)

    # Decompose velocity into components
    Vx, Vy = decompose_velocity(velocity, aoa_deg)

    # Use forward slashes — required by Fluent TUI
    case_path    = BASE_CASE.replace("\\", "/")
    run_dir_fwd  = run_dir.replace("\\", "/")

    # Output file paths
    cl_cd_file   = f"{run_dir_fwd}/cl_cd.txt"
    cp_file      = f"{run_dir_fwd}/cp.txt"
    wss_file     = f"{run_dir_fwd}/wall_shear.txt"
    case_out     = f"{run_dir_fwd}/solved.cas.h5"
    jou_path     = os.path.join(run_dir, "run.jou")

    lines = []

    # ── Read base case ────────────────────────────────────────────────────────
    lines.append(f'; Run {run_id:04d} | V={velocity} m/s | AoA={aoa_deg} deg')
    lines.append(f'/file/read-case "{case_path}"')
    lines.append('')

    # ── Set inlet velocity components ─────────────────────────────────────────
    # TUI sequence confirmed from Fluent 2025 R2 log:
    # 1. Magnitude and Direction? → yes
    # 2. Absolute reference frame? → yes (when yes, skips Relative question)
    # 3. Use profile for velocity magnitude? → no
    # 4. Velocity magnitude value
    # 5. Supersonic gauge pressure? → no
    # 6. X direction component
    # 7. Y direction component
    # 8. Turbulence spec → intensity-and-viscosity-ratio
    # 9. Intensity value
    # 10. Viscosity ratio
    lines.append('; Set inlet boundary condition')
    lines.append(f'/define/boundary-conditions/velocity-inlet {INLET_ZONE}')
    lines.append('yes')              # Velocity Specification: Magnitude and Direction
    lines.append('yes')              # Reference Frame: Absolute
    lines.append('no')               # Use Profile for Velocity Magnitude?
    lines.append(f'{velocity}')      # Velocity Magnitude value
    lines.append('no')               # Use Profile for Supersonic Gauge Pressure?
    lines.append('0')                # Supersonic/Initial Gauge Pressure value (Pa)
    lines.append('no')               # Use Profile for X-Component of Flow Direction?
    lines.append(f'{Vx/velocity}')   # X direction component (normalized)
    lines.append('no')               # Use Profile for Y-Component of Flow Direction?
    lines.append(f'{Vy/velocity}')   # Y direction component (normalized)
    lines.append('no')               # Turbulence: K and Omega? no
    lines.append('no')               # Turbulence: Intensity and Length Scale? no
    lines.append('yes')              # Turbulence: Intensity and Viscosity Ratio? yes
    lines.append(f'{TURB_INTENSITY}')        # 5 (percentage, not fraction)
    lines.append(f'{TURB_VISC_RATIO}')       # 10
    lines.append('')

    # ── Set reference values ──────────────────────────────────────────────────
    lines.append('; Set reference values for Cl/Cd calculation')
    lines.append(f'/report/reference-values/area {REF_AREA}')
    lines.append(f'/report/reference-values/density {DENSITY}')
    lines.append(f'/report/reference-values/velocity {velocity}')
    lines.append(f'/report/reference-values/length {CHORD}')
    lines.append(f'/report/reference-values/compute/velocity-inlet {INLET_ZONE}')
    lines.append('')
    # ── Set convergence criteria ──────────────────────────────────────────────
    lines.append('; Set convergence criteria')
    lines.append('/solve/monitors/residual/convergence-criteria')
    lines.append('1e-6')    # continuity
    lines.append('1e-6')    # x-velocity
    lines.append('1e-6')    # y-velocity
    lines.append('1e-6')    # k
    lines.append('1e-6')    # omega
    lines.append('')
    # ── Initialize ────────────────────────────────────────────────────────────
    lines.append('; Initialize solution')
    lines.append('/solve/initialize/hyb-initialization')
    lines.append('')

    # ── Solve ─────────────────────────────────────────────────────────────────
    lines.append(f'; Iterate for {ITERATIONS} steps')
    lines.append(f'/solve/iterate {ITERATIONS}')
    lines.append('')

    # ── Extract Cl and Cd ─────────────────────────────────────────────────────
    # drag direction = freestream direction
    # lift direction = perpendicular to freestream
    drag_x =  Vx / velocity
    drag_y =  Vy / velocity
    lift_x = -Vy / velocity
    lift_y =  Vx / velocity

    lines.append('; Extract drag coefficient')
    lines.append(f'/report/forces/wall-forces')
    lines.append(f'no')                       # all wall zones? no
    lines.append(f'{AIRFOIL_ZONE}')
    lines.append(f'()')                       # end of zones
    lines.append(f'{drag_x}')                 # drag X direction
    lines.append(f'{drag_y}')                 # drag Y direction
    lines.append(f'yes')                      # write to file?
    lines.append(f'"{cl_cd_file}_cd.txt"')    # output file
    lines.append('')

    lines.append('; Extract lift coefficient')
    lines.append(f'/report/forces/wall-forces')
    lines.append(f'no')                       # all wall zones? no
    lines.append(f'{AIRFOIL_ZONE}')           # zone name
    lines.append(f'()')                       # end of zones
    lines.append(f'{lift_x}')                 # lift X direction
    lines.append(f'{lift_y}')                 # lift Y direction
    lines.append(f'yes')                      # write to file?
    lines.append(f'"{cl_cd_file}_cl.txt"')    # output file
    lines.append('')

    # ── Extract Pressure Coefficient (Cp) ─────────────────────────────────────
    lines.append('; Extract pressure coefficient on airfoil')
    lines.append(f'/report/surface-integrals/area-weighted-avg')
    lines.append(f'{AIRFOIL_ZONE}')
    lines.append(f'()')
    lines.append(f'pressure-coefficient')
    lines.append(f'yes')
    lines.append(f'"{cp_file}"')
    lines.append('')

    # ── Extract Wall Shear Stress ─────────────────────────────────────────────
    lines.append('; Extract wall shear stress on airfoil')
    lines.append(f'/report/surface-integrals/area-weighted-avg')
    lines.append(f'{AIRFOIL_ZONE}')
    lines.append(f'()')
    lines.append(f'wall-shear')
    lines.append(f'yes')
    lines.append(f'"{wss_file}"')
    lines.append('')

    # ── Save solved case ──────────────────────────────────────────────────────
    lines.append(f'; Save solved case')
    lines.append(f'/file/write-case-data "{case_out}"')
    lines.append('')

    # ── Exit ──────────────────────────────────────────────────────────────────
    lines.append('/exit')
    lines.append('yes')

    with open(jou_path, 'w') as f:
        f.write('\n'.join(lines))

    return jou_path