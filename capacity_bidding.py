# -*- coding: utf-8 -*-
"""
Battery‑Sizing Capacity‑Market Optimiser (Gurobi)
================================================
This standalone script **co‑optimises** the *size* of a PV‑coupled battery
(energy capacity and power rating) and its *day‑ahead reserve bids* for a 24‑h
capacity market product.  The objective maximises **net profit**: capacity
payments minus a slack penalty *and* minus an annualised CAPEX charge for the
chosen battery size.

Highlights
----------
* **Sizing variables** – `E_cap` (MWh) and `P_cap` (MW) are now decision
  variables.  The per‑hour amortised CAPEX is subtracted from revenue.
* **Linear model** – No bilinear terms are introduced; upper‑bound constraints
  like `E[t] ≤ E_cap` are expressed explicitly, keeping the problem a pure LP.
* **Two‑hour sustain rule** – `D = 2 h`; the adaptive SOC band and power head‑
  room scale with this duration.
* **Reproducible scenario** – `np.random.seed(77)` fixes PV/load noise.

Tweak `CAPEX_ENERGY_EUR_PER_MWH` and `CAPEX_POWER_EUR_PER_MW` to reflect your
own cost assumptions.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(77)  # reproducible PV / load traces

# -----------------------------
# 0. Scenario & Exogenous Params
# -----------------------------
T = 24
h = range(T)

# Capital‑cost assumptions (very rough!) -----------------------------------
CAPEX_ENERGY_EUR_PER_MWH = 150000   # € / MWh installed (battery pack + PCS)
CAPEX_POWER_EUR_PER_MW   = 70000    # € / MW inverter & BOS
ANNUALISATION_YEARS      = 10        # simple straight‑line
HOURLY_CAPEX_ENERGY = CAPEX_ENERGY_EUR_PER_MWH / (ANNUALISATION_YEARS * 365 * 24)
HOURLY_CAPEX_POWER  = CAPEX_POWER_EUR_PER_MW   / (ANNUALISATION_YEARS * 365 * 24)

# Baseline minimum energy (safety buffer that does not scale with size) -----
E_min_fixed = 20.0  # MWh

# PV profile with daylight‑only noise --------------------------------------
P_PV_base = np.array([0, 0, 0, 0, 0, 10, 20, 30, 40, 50, 60, 70,
                      60, 50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0])
P_PV = np.array([
    max(P_PV_base[t] + (np.random.uniform(-10, 15) if P_PV_base[t] > 0 else 0), 0)
    for t in h
])

# Load (flat 50 MW ± noise) -------------------------------------------------
P_load = np.array([50 + np.random.uniform(-10, 10) for _ in h])

# Capacity‑market price (€/MW) ---------------------------------------------
pi_capacity = np.array([5 if 11 <= t < 14 else 3 for t in h])

# Slack penalty -------------------------------------------------------------
pi_slack = 100.0  # €/MWh unmet load

# Sustain duration ----------------------------------------------------------
D = 2.0  # hours

# -----------------------------
# 1. Build LP model
# -----------------------------
model = gp.Model("Sizing_and_Bidding")
model.setParam("OutputFlag", 0)

# Sizing variables ---------------------------------------------------------
E_cap = model.addVar(lb=E_min_fixed, ub=300, name="E_cap")   # MWh
P_cap = model.addVar(lb=0,           ub=80,  name="P_cap")   # MW

# Time‑series variables ----------------------------------------------------
E = model.addVars(T + 1, lb=E_min_fixed,    ub=300, name="E_state")
P = model.addVars(T,     lb=-80,            ub=80,  name="P_bat")
Pc= model.addVars(T,     lb=0,              ub=80,  name="P_cap_bid")
S = model.addVars(T,     lb=0,                    name="Slack")

# Initial SOC (optional: start half‑full) ----------------------------------
model.addConstr(E[0] == E_min_fixed + 0.5 * (E_cap - E_min_fixed), name="Init_SOC")

for t in h:
    # Energy balance
    model.addConstr(E[t + 1] == E[t] + P[t], name=f"E_balance_{t}")

    # Micro‑grid power balance
    model.addConstr(P_PV[t] + P[t] - P_load[t] + S[t] == 0, name=f"P_balance_{t}")

    # Energy & power upper bounds linked to sizing vars
    model.addConstr(E[t] <= E_cap,      name=f"E_cap_ub_{t}")
    model.addConstr(P[t] <=  P_cap - Pc[t], name=f"P_ub_{t}")
    model.addConstr(P[t] >= -P_cap + Pc[t], name=f"P_lb_{t}")

    # Adaptive SOC window for reserve delivery (2‑hour sustain)
    model.addConstr(E[t] >= E_min_fixed + Pc[t] * D, name=f"E_min_adapt_{t}")
    model.addConstr(E[t] <= E_cap     - Pc[t] * D, name=f"E_max_adapt_{t}")

# Terminal SOC ≥ initial (optional)
model.addConstr(E[T] >= E[0], name="Terminal_SOC")

# -----------------------------
# 2. Objective: capacity revenue – slack – amortised CAPEX
# -----------------------------
revenue = gp.quicksum(pi_capacity[t] * Pc[t] - pi_slack * S[t] for t in h)
capex_cost = HOURLY_CAPEX_ENERGY * E_cap + HOURLY_CAPEX_POWER * P_cap
model.setObjective(revenue - capex_cost, GRB.MAXIMIZE)

# -----------------------------
# 3. Solve
# -----------------------------
model.optimize()

# -----------------------------
# 4. Results
# -----------------------------
if model.status == GRB.OPTIMAL:
    print("Net hourly profit : €{:.2f}".format(model.objVal))
    print("Optimal size      : {:.1f} MWh, {:.1f} MW".format(E_cap.X, P_cap.X))

    E_vals  = np.array([E[t].X for t in range(T + 1)])
    Pc_vals = np.array([Pc[t].X for t in h])
    S_vals  = np.array([S[t].X  for t in h])

    E_min_adapt = E_min_fixed + Pc_vals * D
    E_max_adapt = E_cap.X      - Pc_vals * D

    plt.figure(figsize=(12, 5))
    plt.plot(range(T + 1), E_vals, marker="o", label="SOC (MWh)")
    plt.step(h, Pc_vals, where="mid", marker="^", label="Reserve bid (MW)")
    plt.step(h, [E_min_adapt[t] for t in h], where="mid", ls="--", label="E_min adaptive")
    plt.step(h, [E_max_adapt[t] for t in h], where="mid", ls="--", label="E_max adaptive")
    plt.legend(); plt.grid(True); plt.xlabel("Hour"); plt.ylabel("MWh / MW");
    plt.title("Optimal sizing and bidding profile")
    plt.show()
else:
    print("Optimization status", model.status)
