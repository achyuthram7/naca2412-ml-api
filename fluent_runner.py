# fluent_runner.py + results_reader.py
# Launches Fluent headlessly and reads results.

import subprocess
import os
import re
import time
from config import FLUENT_EXE, FLUENT_DIM, CORES


# ══════════════════════════════════════════════════════════════════════════════
# FLUENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_fluent(journal_path, run_dir):
    """
    Runs Fluent in batch/headless mode using a journal file.
    Returns True if successful, False otherwise.
    """
    log_path = os.path.join(run_dir, "fluent.log")

    cmd = [
        FLUENT_EXE,
        FLUENT_DIM,           # "2ddp" = 2D double precision
        f"-t{CORES}",         # number of CPU cores
        "-g",                 # no GUI
        "-i", journal_path,   # journal file to execute
        "-driver", "null",    # no graphics driver (headless)
    ]

    print(f"  [Fluent] Launching...")
    t0 = time.time()

    try:
        with open(log_path, "w") as log:
            result = subprocess.run(
                cmd,
                stdout=log,
                stderr=log,
                timeout=7200    # 2 hour max per run
            )

        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)

        if result.returncode == 0:
            print(f"  [Fluent] Completed in {mins}m {secs}s")
            return True
        else:
            print(f"  [Fluent] Failed (code {result.returncode}) after {mins}m {secs}s")
            return False

    except subprocess.TimeoutExpired:
        print(f"  [Fluent] Timed out after 2 hours!")
        return False
    except FileNotFoundError:
        print(f"  [Fluent] ERROR: Executable not found at:\n  {FLUENT_EXE}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS READER
# ══════════════════════════════════════════════════════════════════════════════

def parse_report_file(filepath):
    """
    Reads a Fluent surface report .txt file and extracts the numeric value.
    Returns None if file not found or can't be parsed.
    """
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r") as f:
            content = f.read()
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", content)
        return float(numbers[-1]) if numbers else None
    except Exception:
        return None


def parse_force_file(filepath):
    """
    Parses Fluent wall-forces report file to extract force coefficient.
    Returns the coefficient value or None.
    """
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r") as f:
            content = f.read()
        # Extract all numbers and return the last meaningful value
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", content)
        if numbers:
            # Filter out very small numbers that are likely formatting artifacts
            valid = [float(n) for n in numbers if abs(float(n)) > 1e-10]
            return valid[-1] if valid else float(numbers[-1])
        return None
    except Exception:
        return None


def check_convergence(log_path):
    """
    Checks Fluent log for convergence message.
    Returns True if solution converged.
    """
    if not os.path.exists(log_path):
        return False
    try:
        with open(log_path, "r") as f:
            content = f.read()
        return "solution is converged" in content.lower()
    except Exception:
        return False


def read_run_results(run_dir):
    """
    Reads all result files from a completed run folder.
    Returns a dict of extracted values.
    """
    results = {}

    # Cl and Cd from separate force files
    results["Cl"] = parse_force_file(os.path.join(run_dir, "cl_cd_cl.txt"))
    results["Cd"] = parse_force_file(os.path.join(run_dir, "cl_cd_cd.txt"))

    # Cp and Wall Shear from surface integral files
    results["Cp"]               = parse_report_file(os.path.join(run_dir, "cp.txt"))
    results["wall_shear_stress"] = parse_report_file(os.path.join(run_dir, "wall_shear.txt"))

    # Convergence status
    results["converged"] = check_convergence(os.path.join(run_dir, "fluent.log"))

    return results