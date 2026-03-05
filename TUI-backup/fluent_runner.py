# fluent_runner.py
# Launches Fluent headlessly using a journal file.

import subprocess
import os
import time
from config import FLUENT_EXE, FLUENT_DIM, CORES


def run_fluent(journal_path, run_dir):
    """
    Runs Fluent in batch mode using the given journal file.
    Returns True if successful, False otherwise.
    """
    log_path = os.path.join(run_dir, "fluent.log")

    cmd = [
        FLUENT_EXE,
        FLUENT_DIM,          # "2d"
        f"-t{CORES}",        # number of cores
        "-g",                # no GUI
        "-i", journal_path,  # journal file
        "-driver", "null",   # no graphics driver (headless)
    ]

    print(f"  [Fluent] Launching...")
    t0 = time.time()

    try:
        with open(log_path, "w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=log, timeout=7200)

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
        print(f"  [Fluent] ERROR: Fluent not found at:\n  {FLUENT_EXE}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# results_reader.py — Parse Fluent output files
# ─────────────────────────────────────────────────────────────────────────────

import re


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


def parse_cl_cd_file(run_dir):
    """
    Parses Fluent's separate Cl and Cd report files.
    Returns (Cl, Cd) or (None, None) if not found.
    """
    cl_file = os.path.join(run_dir, "cl_cd.txt_cl.txt")
    cd_file = os.path.join(run_dir, "cl_cd.txt_cd.txt")

    cl = parse_report_file(cl_file)
    cd = parse_report_file(cd_file)
    return cl, cd


def check_convergence(log_path):
    """
    Checks Fluent log file for convergence message.
    Returns True if converged.
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

    # Cl and Cd — now reads from separate files
    cl, cd = parse_cl_cd_file(run_dir)
    results["Cl"] = cl
    results["Cd"] = cd

    # Pressure coefficient
    results["Cp"] = parse_report_file(os.path.join(run_dir, "cp.txt"))

    # Wall shear stress
    results["wall_shear_stress"] = parse_report_file(os.path.join(run_dir, "wall_shear.txt"))

    # Convergence
    results["converged"] = check_convergence(os.path.join(run_dir, "fluent.log"))

    return results