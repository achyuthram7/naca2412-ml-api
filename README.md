# NACA 2412 Airfoil — CFD + ML Surrogate Model

## Overview
This project builds a Machine Learning surrogate model for NACA 2412 
airfoil aerodynamics using ANSYS Fluent CFD simulations.

## Workflow
1. **CFD Simulations** — 50 runs using ANSYS Fluent (TUI automation)
2. **ML Training** — Gradient Boosting surrogate model (R² > 0.999)
3. **FastAPI** — REST API for instant aerodynamic predictions

## Parameters
- Velocity: 10–50 m/s
- Angle of Attack: -5° to 20°
- Turbulence Model: k-ω SST

## Results
| Output | Model | R² |
|--------|-------|-----|
| Cl | Gradient Boosting | 0.9997 |
| Cd | Gradient Boosting | 0.9998 |
| Cp | Gradient Boosting | 0.9906 |
| Wall Shear Stress | Gaussian Process | 1.0000 |

## Installation
pip install -r requirements.txt

## Usage
python api_server.py
```

### File 3 — `requirements.txt`
```
fastapi==0.104.1
uvicorn==0.24.0
joblib==1.3.2
numpy==1.26.0
pandas==2.1.0
scikit-learn==1.3.2
matplotlib==3.8.0