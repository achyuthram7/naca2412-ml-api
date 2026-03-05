# main.py
# Orchestrates the complete simulation pipeline with resume capability.
# If interrupted, re-running will skip already completed runs.

import os
import json
import itertools
import pandas as pd
from config import *
from journal_generator import generate_journal
from fluent_runner import run_fluent, read_run_results


def generate_run_matrix():
    """Creates all combinations of velocity and AoA — 50 total runs."""
    runs = []
    for velocity, aoa in itertools.product(VELOCITIES, AOA_LIST):
        runs.append({"velocity": velocity, "aoa_deg": aoa})
    return runs


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    run_matrix = generate_run_matrix()
    total = len(run_matrix)

    print("=" * 60)
    print(f"NACA 2412 Airfoil — Parameter Sweep")
    print(f"Velocities  : {[round(v,1) for v in VELOCITIES]}")
    print(f"AoA values  : {[round(a,1) for a in AOA_LIST]}")
    print(f"Total runs  : {total}")
    print(f"Results dir : {RESULTS_DIR}")
    print("=" * 60)

    # Count already completed runs
    already_done = sum(
        1 for i in range(1, total + 1)
        if os.path.exists(os.path.join(RESULTS_DIR, f"run_{i:04d}", "results.json"))
    )
    if already_done > 0:
        print(f"\nResume mode: {already_done}/{total} runs already completed — skipping them")

    all_results = []
    failed_runs = []

    for run_id, params in enumerate(run_matrix, start=1):
        velocity = params["velocity"]
        aoa_deg  = params["aoa_deg"]
        run_dir  = os.path.join(RESULTS_DIR, f"run_{run_id:04d}")

        print(f"\n{'─'*55}")
        print(f"Run {run_id:03d}/{total} | V={velocity:.1f} m/s | AoA={aoa_deg:.1f}°")
        print(f"{'─'*55}")

        # ── Resume: skip if already completed ────────────────────────────────
        result_file = os.path.join(run_dir, "results.json")
        if os.path.exists(result_file):
            print(f"  [Main] Already completed — skipping")
            with open(result_file) as f:
                all_results.append(json.load(f))
            continue

        os.makedirs(run_dir, exist_ok=True)

        # Save run parameters
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)

        # Generate journal
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

        # Combine inputs + outputs
        record = {
            "run_id"           : run_id,
            "velocity_ms"      : velocity,
            "aoa_deg"          : aoa_deg,
            "reynolds_number"  : (DENSITY * velocity * CHORD) / VISCOSITY,
            "Cl"               : results.get("Cl"),
            "Cd"               : results.get("Cd"),
            "Cp"               : results.get("Cp"),
            "wall_shear_stress" : results.get("wall_shear_stress"),
            "converged"        : results.get("converged"),
        }

        # Save individual run results
        with open(result_file, "w") as f:
            json.dump(record, f, indent=2)

        all_results.append(record)

        print(f"  [Results] Cl={record['Cl']}  Cd={record['Cd']}  "
              f"Cp={record['Cp']}  Converged={record['converged']}")

    # ── Consolidate all results ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"SIMULATION COMPLETE")
    print(f"{'=' * 60}")

    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(RESULTS_DIR, "all_results.csv")
        df.to_csv(csv_path, index=False)

        print(f"Completed  : {len(all_results)}/{total} runs")
        print(f"Converged  : {df['converged'].sum()}/{len(df)} runs")
        if failed_runs:
            print(f"Failed     : {failed_runs}")
        print(f"CSV saved  : {csv_path}")
        print(f"\n{df[['run_id','velocity_ms','aoa_deg','Cl','Cd','converged']].to_string(index=False)}")
    else:
        print("No successful runs to report.")


if __name__ == "__main__":
    main()