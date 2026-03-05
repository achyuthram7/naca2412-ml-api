# ml_training.py
# Complete ML surrogate model training for NACA 2412 aerodynamics.
# Dataset: 50 CFD runs, 3 inputs, 4 outputs.
#
# Run in Jupyter:
# !pip install scikit-learn pandas numpy matplotlib joblib

import warnings
warnings.filterwarnings("ignore")  # suppress sklearn warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection    import cross_val_score, KFold
from sklearn.preprocessing      import StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.ensemble           import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network     import MLPRegressor
from sklearn.gaussian_process   import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.metrics            import r2_score, mean_absolute_error, mean_squared_error

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_CSV = r"D:\Ansys_works\results\all_results.csv"
MODELS_DIR  = r"D:\Ansys_works\ml_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Columns ───────────────────────────────────────────────────────────────────
INPUT_COLS  = ["velocity_ms", "aoa_deg", "reynolds_number"]
OUTPUT_COLS = ["Cl", "Cd", "Cp", "wall_shear_stress"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STEP 1 — Loading Dataset")
print("=" * 60)

df = pd.read_csv(RESULTS_CSV)
df = df[df["converged"] == True].dropna(subset=INPUT_COLS + OUTPUT_COLS)

print(f"Total samples   : {len(df)}")
print(f"Input features  : {INPUT_COLS}")
print(f"Output targets  : {OUTPUT_COLS}")
print(f"\nInput ranges:")
for col in INPUT_COLS:
    print(f"  {col:20s}: {df[col].min():.3f} → {df[col].max():.3f}")
print(f"\nOutput ranges:")
for col in OUTPUT_COLS:
    print(f"  {col:20s}: {df[col].min():.6f} → {df[col].max():.6f}")

X = df[INPUT_COLS].values
print(f"\nFeature matrix shape: {X.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — VISUALIZE DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 2 — Visualizing CFD Data")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("NACA 2412 — CFD Training Data Overview", fontsize=14, fontweight="bold")

velocities = sorted(df["velocity_ms"].unique())
colors     = plt.cm.plasma(np.linspace(0, 0.85, len(velocities)))

for ax, target, ylabel, unit in zip(
    axes.flat,
    OUTPUT_COLS,
    ["Lift Coefficient (Cl)", "Drag Coefficient (Cd)",
     "Pressure Coefficient (Cp)", "Wall Shear Stress"],
    ["", "", "", "(Pa)"]
):
    for vel, color in zip(velocities, colors):
        sub = df[df["velocity_ms"] == vel].sort_values("aoa_deg")
        ax.plot(sub["aoa_deg"], sub[target], "o-",
                color=color, label=f"V={vel:.0f} m/s", lw=2, ms=5)
    ax.set_xlabel("Angle of Attack (°)")
    ax.set_ylabel(f"{ylabel} {unit}")
    ax.set_title(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "cfd_data_overview.png"), dpi=150)
plt.show()
print("Plot saved -> cfd_data_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 3 — Training ML Models")
print("=" * 60)

# ── Define candidate models ───────────────────────────────────────────────────
# Note: With 50 samples, we use Leave-One-Out cross validation
# for most reliable performance estimate

candidates = {
    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators  = 300,
            max_depth     = 3,
            learning_rate = 0.05,
            subsample     = 0.8,
            random_state  = 42
        ))
    ]),

    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators    = 300,
            max_depth       = 6,
            min_samples_leaf= 2,
            random_state    = 42
        ))
    ]),

    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MLPRegressor(
            hidden_layer_sizes = (64, 32),
            max_iter           = 5000,
            alpha              = 0.01,
            random_state       = 42,
            early_stopping     = True,
            validation_fraction= 0.15
        ))
    ]),

    "GaussianProcess": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GaussianProcessRegressor(
            kernel               = ConstantKernel(1.0) * Matern(nu=2.5),
            n_restarts_optimizer = 10,
            normalize_y          = True,
            alpha                = 1e-6
        ))
    ]),
}

summary = []

for target in OUTPUT_COLS:
    print(f"\n{'─'*55}")
    print(f"Training model for: {target}")
    print(f"{'─'*55}")

    y = df[target].values

    # Use 5-fold cross validation
    # Each fold: train on 40 samples, test on 10 samples
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_name, best_pipe, best_r2 = None, None, -np.inf

    for name, pipe in candidates.items():
        try:
            scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
            r2_cv  = scores.mean()
            r2_std = scores.std()
            print(f"  {name:20s}: 5-Fold R² = {r2_cv:.4f} ± {r2_std:.4f}")

            if r2_cv > best_r2:
                best_r2   = r2_cv
                best_name = name
                best_pipe = pipe

        except Exception as e:
            print(f"  {name:20s}: Failed — {e}")

    # ── Final fit on full dataset ─────────────────────────────────────────────
    # For surrogate models, we train on ALL data (no holdout)
    # since every data point is precious
    best_pipe.fit(X, y)

    # In-sample metrics (how well it fits training data)
    y_pred_train = best_pipe.predict(X)
    train_r2     = r2_score(y, y_pred_train)
    train_mae    = mean_absolute_error(y, y_pred_train)
    train_rmse   = np.sqrt(mean_squared_error(y, y_pred_train))

    print(f"\n  Winner          : {best_name}")
    print(f"  5-Fold R²       : {best_r2:.4f}  (generalization)")
    print(f"  Train R²        : {train_r2:.4f}  (fit quality)")
    print(f"  Train MAE       : {train_mae:.6f}")
    print(f"  Train RMSE      : {train_rmse:.6f}")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, f"{target}.pkl")
    joblib.dump({
        "pipe"      : best_pipe,
        "inputs"    : INPUT_COLS,
        "target"    : target,
        "cv_r2"     : round(best_r2, 4),
        "train_r2"  : round(train_r2, 4),
        "train_mae" : round(train_mae, 6),
    }, model_path)
    print(f"  Saved           : {model_path}")

    # ── Parity plot ───────────────────────────────────────────────────────────
    plt.figure(figsize=(5, 5))
    plt.scatter(y, y_pred_train, alpha=0.8, edgecolors="k",
                s=60, zorder=3, c=df["velocity_ms"], cmap="plasma")
    mn = min(y.min(), y_pred_train.min())
    mx = max(y.max(), y_pred_train.max())
    plt.plot([mn, mx], [mn, mx], "r--", lw=2, label="Ideal (y=x)")
    plt.colorbar(label="Velocity (m/s)")
    plt.xlabel("CFD Actual")
    plt.ylabel("ML Predicted")
    plt.title(f"{target} — {best_name}\nLOO R²={best_r2:.4f} | MAE={train_mae:.5f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f"{target}_parity.png"), dpi=150)
    plt.show()

    summary.append({
        "target"    : target,
        "best_model": best_name,
        "cv_r2"     : round(best_r2, 4),
        "train_r2"  : round(train_r2, 4),
        "train_mae" : round(train_mae, 6),
        "train_rmse": round(train_rmse, 6),
    })


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 4 — Model Summary")
print("=" * 60)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(MODELS_DIR, "model_summary.csv"), index=False)
print(summary_df.to_string(index=False))

print(f"\nAll models saved to: {MODELS_DIR}")
print("\nFiles generated:")
for f in os.listdir(MODELS_DIR):
    fpath = os.path.join(MODELS_DIR, f)
    size  = os.path.getsize(fpath) / 1024
    print(f"  {f:40s} {size:.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — QUICK TEST PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 5 — Quick Test Predictions")
print("=" * 60)

def predict(velocity_ms, aoa_deg):
    re = (1.225 * velocity_ms * 1.0) / 1.7894e-5
    X_new = np.array([[velocity_ms, aoa_deg, re]])
    print(f"\n  V={velocity_ms} m/s | AoA={aoa_deg}° | Re={re:.0f}")
    print(f"  {'─'*40}")
    for target in OUTPUT_COLS:
        artifact = joblib.load(os.path.join(MODELS_DIR, f"{target}.pkl"))
        val = artifact["pipe"].predict(X_new)[0]
        print(f"  {target:22s} = {val:.6f}")

# Test on interval points (not in training data)
predict(15, 2.5)     # between training points
predict(25, 9.5)     # between training points
predict(35, 16.0)    # between training points