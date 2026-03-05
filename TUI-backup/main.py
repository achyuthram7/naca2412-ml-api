# main.py
# Orchestrates the complete simulation pipeline:
# Parameter sweep → Journal generation → Fluent runs → Results collection → CSV

import os
import json
import itertools
import pandas as pd
from config import *
from journal_generator import generate_journal
from fluent_runner import run_fluent, read_run_results

# ── Generate all combinations of velocity and AoA ─────────────────────────────
def generate_run_matrix():
    """
    Creates all combinations of velocity and angle of attack.
    Example: 5 velocities × 6 AoA = 30 total simulations
    """
    runs = []
    for velocity, aoa in itertools.product(VELOCITIES, AOA_LIST):
        runs.append({"velocity": velocity, "aoa_deg": aoa})
    return runs


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate run matrix
    run_matrix = generate_run_matrix()
    total = len(run_matrix)

    print("=" * 60)
    print(f"NACA 2412 Airfoil — Parameter Sweep")
    print(f"Velocities : {VELOCITIES}")
    print(f"AoA range  : {AOA_LIST}")
    print(f"Total runs : {total}")
    print("=" * 60)

    all_results = []
    failed_runs = []

    for run_id, params in enumerate(run_matrix, start=1):
        velocity = params["velocity"]
        aoa_deg  = params["aoa_deg"]

        print(f"\n{'─'*55}")
        print(f"Run {run_id:03d}/{total} | V={velocity} m/s | AoA={aoa_deg:.1f}°")
        print(f"{'─'*55}")

        # Create run directory
        run_dir = os.path.join(RESULTS_DIR, f"run_{run_id:04d}")
        os.makedirs(run_dir, exist_ok=True)

        # Save run parameters
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)

        # Generate journal file
        jou_path = generate_journal(run_id, velocity, aoa_deg, run_dir)
        print(f"  [Journal] Written -> run_{run_id:04d}/run.jou")

        # Run Fluent
        success = run_fluent(jou_path, run_dir)

        if not success:
            print(f"  [Main] Run {run_id} FAILED — skipping")
            failed_runs.append(run_id)
            continue

        # Read results
        results = read_run_results(run_dir)

        # Combine inputs + outputs into one record
        record = {
            "run_id"          : run_id,
            "velocity_ms"     : velocity,
            "aoa_deg"         : aoa_deg,
            "reynolds_number" : (DENSITY * velocity * CHORD) / VISCOSITY,
            "Cl"              : results.get("Cl"),
            "Cd"              : results.get("Cd"),
            "Cp"              : results.get("Cp"),
            "wall_shear_stress": results.get("wall_shear_stress"),
            "converged"       : results.get("converged"),
        }

        # Save individual run results
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(record, f, indent=2)

        all_results.append(record)

        # Print extracted values
        print(f"  [Results] Cl={record['Cl']}  Cd={record['Cd']}  "
              f"Cp={record['Cp']}  Converged={record['converged']}")

    # ── Save all results to CSV ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(RESULTS_DIR, "all_results.csv")
        df.to_csv(csv_path, index=False)

        print(f"Completed: {len(all_results)}/{total} runs")
        if failed_runs:
            print(f"Failed runs: {failed_runs}")
        print(f"\nResults saved -> {csv_path}")
        print(f"\n{df.to_string(index=False)}")
    else:
        print("No successful runs to report.")
    print("=" * 60)


if __name__ == "__main__":
    main()