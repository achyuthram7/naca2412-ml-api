# api_server.py
# Intelligent Aerodynamics API powered by Claude AI.
# Users can ask ANYTHING in natural language.
# Claude decides what tools to use and how to answer.
#
# Install:
#   pip install fastapi uvicorn anthropic joblib numpy pandas scikit-learn
#
# Run:
#   uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

import os
import json
import math
import numpy as np
import pandas as pd
import joblib
from groq import Groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "NACA 2412 Intelligent Aerodynamics API",
    description = "Ask anything about NACA 2412 aerodynamics in natural language.",
    version     = "3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "ml_models")
DATA_FILE  = os.path.join(BASE_DIR, "results", "all_results.csv")

# ── Constants ─────────────────────────────────────────────────────────────────
DENSITY      = 1.225
VISCOSITY    = 1.7894e-5
CHORD        = 1.0
TARGET_NAMES = ["Cl", "Cd", "Cp", "wall_shear_stress"]

# ── Global storage ────────────────────────────────────────────────────────────
models = {}
cfd_df = None
client = None


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global cfd_df, client

    # Load ML models
    for target in TARGET_NAMES:
        path = os.path.join(MODELS_DIR, f"{target}.pkl")
        if os.path.exists(path):
            models[target] = joblib.load(path)
            print(f"Loaded: {target}")

    # Load CFD data
    if os.path.exists(DATA_FILE):
        cfd_df = pd.read_csv(DATA_FILE)
        print(f"Loaded CFD data: {len(cfd_df)} rows")

    # Init Groq client
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY", "")
    )
    print("Groq client initialized")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS — These are called by Claude when needed
# ══════════════════════════════════════════════════════════════════════════════

def tool_ml_predict(velocity_ms: float, aoa_deg: float) -> dict:
    """
    Tool: Predict aerodynamic coefficients using ML surrogate model.
    Use this for any prediction request within range:
    velocity 10-50 m/s, AoA -5 to 20 degrees.
    """
    if not models:
        return {"error": "ML models not loaded"}
    if not (10 <= velocity_ms <= 50):
        return {"error": f"Velocity {velocity_ms} out of range (10-50 m/s)"}
    if not (-5 <= aoa_deg <= 20):
        return {"error": f"AoA {aoa_deg} out of range (-5 to 20 degrees)"}

    re = (DENSITY * velocity_ms * CHORD) / VISCOSITY
    X  = np.array([[velocity_ms, aoa_deg, re]])

    result = {
        "source"        : "ML Surrogate Model (Gradient Boosting, R²=0.9997)",
        "velocity_ms"   : velocity_ms,
        "aoa_deg"       : aoa_deg,
        "reynolds_number": round(re, 0),
    }
    for target in TARGET_NAMES:
        if target in models:
            result[target] = round(float(models[target]["pipe"].predict(X)[0]), 6)
    return result


def tool_cfd_lookup(velocity_ms: float, aoa_deg: float) -> dict:
    """
    Tool: Look up actual CFD simulation result closest to requested conditions.
    Returns real ANSYS Fluent simulation data.
    """
    if cfd_df is None:
        return {"error": "CFD data not available"}

    distances = np.sqrt(
        ((cfd_df["velocity_ms"] - velocity_ms) / 10) ** 2 +
        ((cfd_df["aoa_deg"]    - aoa_deg)      / 5)  ** 2
    )
    idx = distances.idxmin()
    row = cfd_df.iloc[idx]

    return {
        "source"           : "Actual ANSYS Fluent CFD Simulation",
        "requested"        : {"velocity_ms": velocity_ms, "aoa_deg": aoa_deg},
        "nearest_point"    : {
            "velocity_ms"      : float(row["velocity_ms"]),
            "aoa_deg"          : round(float(row["aoa_deg"]), 4),
        },
        "Cl"               : round(float(row["Cl"]), 6),
        "Cd"               : round(float(row["Cd"]), 6),
        "Cp"               : round(float(row["Cp"]), 6),
        "wall_shear_stress" : round(float(row["wall_shear_stress"]), 6),
        "converged"        : bool(row["converged"]),
        "reynolds_number"  : round(float(row["reynolds_number"]), 0),
    }


def tool_aoa_sweep(velocity_ms: float, aoa_start: float = -5,
                   aoa_end: float = 20, n_points: int = 50) -> dict:
    """
    Tool: Generate ML predictions over a range of angles of attack.
    Use this when user asks for a curve, sweep, or range of AoA values.
    """
    if not models:
        return {"error": "ML models not loaded"}

    aoa_values = np.linspace(aoa_start, aoa_end, n_points)
    re         = (DENSITY * velocity_ms * CHORD) / VISCOSITY
    X          = np.column_stack([
        np.full(n_points, velocity_ms),
        aoa_values,
        np.full(n_points, re)
    ])

    data = []
    preds = {t: models[t]["pipe"].predict(X) for t in TARGET_NAMES if t in models}
    for i, aoa in enumerate(aoa_values):
        data.append({
            "aoa_deg"          : round(float(aoa), 3),
            "Cl"               : round(float(preds.get("Cl", [None]*n_points)[i]), 6),
            "Cd"               : round(float(preds.get("Cd", [None]*n_points)[i]), 6),
            "Cp"               : round(float(preds.get("Cp", [None]*n_points)[i]), 6),
            "wall_shear_stress": round(float(preds.get("wall_shear_stress",
                                               [None]*n_points)[i]), 6),
        })

    # Find stall (max Cl)
    cl_values  = [d["Cl"] for d in data]
    stall_idx  = cl_values.index(max(cl_values))
    stall_aoa  = data[stall_idx]["aoa_deg"]
    stall_cl   = data[stall_idx]["Cl"]

    return {
        "source"      : "ML Surrogate Model Sweep",
        "velocity_ms" : velocity_ms,
        "aoa_range"   : f"{aoa_start}° to {aoa_end}°",
        "n_points"    : n_points,
        "stall_angle" : stall_aoa,
        "max_Cl"      : stall_cl,
        "data"        : data
    }


def tool_thin_airfoil_theory(aoa_deg: float) -> dict:
    """
    Tool: Calculate aerodynamic coefficients using thin airfoil theory.
    Analytical formula — useful for comparison with ML and CFD.
    NACA 2412 zero-lift angle = -2.08 degrees.
    """
    aoa_rad   = math.radians(aoa_deg)
    alpha_L0  = math.radians(-2.08)   # zero-lift angle for NACA 2412
    Cl        = 2 * math.pi * (aoa_rad - alpha_L0)
    Cd        = 0.006 + (Cl ** 2) / (math.pi * 6)
    Cp        = -Cl * 0.5

    return {
        "source"    : "Thin Airfoil Theory (Analytical)",
        "aoa_deg"   : aoa_deg,
        "Cl"        : round(Cl, 6),
        "Cd"        : round(Cd, 6),
        "Cp"        : round(Cp, 6),
        "note"      : "Most accurate for AoA < 10°. Ignores viscous effects.",
        "formula"   : "Cl = 2π(α - α_L0), α_L0 = -2.08° for NACA 2412"
    }


def tool_get_dataset_info() -> dict:
    """
    Tool: Get information about the CFD dataset used to train the ML model.
    Use this when user asks about training data, simulation details, or dataset.
    """
    if cfd_df is None:
        return {"error": "Dataset not available"}
    return {
        "total_simulations" : len(cfd_df),
        "converged"         : int(cfd_df["converged"].sum()),
        "velocity_range"    : f"{cfd_df['velocity_ms'].min()} - {cfd_df['velocity_ms'].max()} m/s",
        "aoa_range"         : f"{cfd_df['aoa_deg'].min():.2f}° to {cfd_df['aoa_deg'].max():.2f}°",
        "Cl_range"          : f"{cfd_df['Cl'].min():.4f} to {cfd_df['Cl'].max():.4f}",
        "Cd_range"          : f"{cfd_df['Cd'].min():.4f} to {cfd_df['Cd'].max():.4f}",
        "solver"            : "ANSYS Fluent 2025 R2 Student",
        "turbulence_model"  : "k-omega SST",
        "mesh_elements"     : 49765,
        "convergence"       : "1e-6 residuals",
        "airfoil"           : "NACA 2412",
        "chord_length"      : "1.0 m",
        "fluid"             : "Air (1.225 kg/m³, 1.7894e-5 Pa·s)",
    }


def tool_get_ml_model_info() -> dict:
    """
    Tool: Get information about the trained ML surrogate models.
    Use this when user asks about model accuracy, training, or performance.
    """
    summary_path = os.path.join(MODELS_DIR, "model_summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        return {
            "models"       : df.to_dict(orient="records"),
            "input_features": ["velocity_ms", "aoa_deg", "reynolds_number"],
            "outputs"      : TARGET_NAMES,
            "training_data": "50 CFD simulations",
            "cv_method"    : "5-fold cross validation",
            "note"         : "GBR = Gradient Boosting Regressor, GP = Gaussian Process"
        }
    return {"error": "Model summary not found"}


# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE TOOL DEFINITIONS
# These tell Claude what tools are available and when to use them
# ══════════════════════════════════════════════════════════════════════════════

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name"       : "ml_predict",
            "description": "Predict Cl, Cd, Cp, wall shear stress using ML surrogate model.",
            "parameters" : {
                "type"      : "object",
                "properties": {
                    "velocity_ms": {"type": "number", "description": "Velocity in m/s (10-50)"},
                    "aoa_deg"    : {"type": "number", "description": "Angle of attack in degrees (-5 to 20)"}
                },
                "required": ["velocity_ms", "aoa_deg"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name"       : "cfd_lookup",
            "description": "Look up actual ANSYS Fluent CFD simulation result nearest to requested conditions.",
            "parameters" : {
                "type"      : "object",
                "properties": {
                    "velocity_ms": {"type": "number"},
                    "aoa_deg"    : {"type": "number"}
                },
                "required": ["velocity_ms", "aoa_deg"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name"       : "aoa_sweep",
            "description": "Generate ML predictions over a range of angles of attack. Use when user asks for a curve or range.",
            "parameters" : {
                "type"      : "object",
                "properties": {
                    "velocity_ms": {"type": "number"},
                    "aoa_start"  : {"type": "number"},
                    "aoa_end"    : {"type": "number"},
                    "n_points"   : {"type": "integer"}
                },
                "required": ["velocity_ms"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name"       : "thin_airfoil_theory",
            "description": "Calculate using thin airfoil theory analytical formula for comparison.",
            "parameters" : {
                "type"      : "object",
                "properties": {
                    "aoa_deg": {"type": "number"}
                },
                "required": ["aoa_deg"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name"       : "get_dataset_info",
            "description": "Get information about the CFD simulation dataset and training data.",
            "parameters" : {
                "type"      : "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name"       : "get_ml_model_info",
            "description": "Get ML model performance metrics, R² scores and accuracy details.",
            "parameters" : {
                "type"      : "object",
                "properties": {}
            }
        }
    }
]

# Map tool names to functions
TOOL_FUNCTIONS = {
    "ml_predict"         : tool_ml_predict,
    "cfd_lookup"         : tool_cfd_lookup,
    "aoa_sweep"          : tool_aoa_sweep,
    "thin_airfoil_theory": tool_thin_airfoil_theory,
    "get_dataset_info"   : tool_get_dataset_info,
    "get_ml_model_info"  : tool_get_ml_model_info,
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INTELLIGENT ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert aerodynamics assistant specializing in NACA 2412 airfoil analysis.
You have access to tools for:
- ML surrogate model predictions (Gradient Boosting, R² > 0.999)
- Actual ANSYS Fluent CFD simulation data (50 runs)
- Thin airfoil theory analytical formulas
- Dataset and model information

Your primary focus is the ML model and CFD data. However, you can also:
- Explain aerodynamic concepts and theory
- Compare different prediction methods
- Discuss the physics behind the results
- Answer questions about CFD, turbulence models, meshing
- Explain what the coefficients mean physically
- Discuss limitations of the model
- Answer anything related to aerodynamics, fluid mechanics, or aerospace

Always use the ML model as the primary prediction source.
Use CFD data to validate ML predictions when asked.
Use thin airfoil theory for comparison and educational purposes.
Be concise but thorough. Include numerical results when available.
If asked something outside your tools, use your general aerodynamics knowledge."""


class ChatRequest(BaseModel):
    message     : str = Field(..., description="Natural language question or request")
    history     : Optional[List[dict]] = Field(default=[], description="Previous conversation messages")


class ChatResponse(BaseModel):
    answer      : str
    tools_used  : List[str]
    data        : Optional[dict]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main intelligent endpoint — ask anything in natural language!"""
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=503,
                            detail="Groq API key not configured")
    try:
        # Build message history
        messages = list(req.history) if req.history else []
        messages.append({"role": "user", "content": req.message})

        tools_used = []
        tool_data  = {}

        # Agentic loop
        while True:
            response = client.chat.completions.create(
                model       = "llama-3.3-70b-versatile",
                messages    = [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                tools       = GROQ_TOOLS,
                tool_choice = "auto",
                max_tokens  = 2000,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append({
                    "role"      : "assistant",
                    "content"   : msg.content or "",
                    "tool_calls": [
                        {
                            "id"      : tc.id,
                            "type"    : "function",
                            "function": {
                                "name"     : tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })

                for tc in msg.tool_calls:
                    tool_name  = tc.function.name
                    tool_input = json.loads(tc.function.arguments)
                    tools_used.append(tool_name)

                    fn     = TOOL_FUNCTIONS.get(tool_name)
                    result = fn(**tool_input) if fn else {"error": f"Unknown tool: {tool_name}"}
                    tool_data[tool_name] = result

                    messages.append({
                        "role"        : "tool",
                        "tool_call_id": tc.id,
                        "content"     : json.dumps(result)
                    })
            else:
                return ChatResponse(
                    answer     = msg.content or "No response generated.",
                    tools_used = tools_used,
                    data       = tool_data if tool_data else None
                )

    except Exception as e:
        # Return actual error message instead of generic 500
        raise HTTPException(status_code=500, detail=str(e))


# ── Standard endpoints still available ───────────────────────────────────────
@app.get("/")
def root():
    return {
        "name"     : "NACA 2412 Intelligent Aerodynamics API",
        "version"  : "3.0.0",
        "main"     : "POST /chat — ask anything in natural language",
        "docs"     : "/docs",
        "examples" : [
            "What is Cl at 30 m/s and 7 degrees?",
            "Show Cl vs AoA curve at 40 m/s",
            "Compare ML with CFD at 20 m/s, 5 degrees",
            "What is the stall angle?",
            "How accurate is the ML model?",
            "Explain wall shear stress physically",
            "What turbulence model was used?"
        ]
    }

@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "ml_models"   : list(models.keys()),
        "cfd_rows"    : len(cfd_df) if cfd_df is not None else 0,
        "groq"        : "configured" if os.environ.get("GROQ_API_KEY") else "missing API key"
    }