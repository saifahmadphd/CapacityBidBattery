# -*- coding: utf-8 -*-
"""
PV‑Battery Sizing & Capacity‑Market Co‑Optimiser (LP, Gurobi)
=============================================================
This script **co‑optimises** the *size* of a PV‑coupled battery (energy `E_cap`, power
`P_cap`) and *hourly* capacity-market bids over a 24‑hour horizon. It remains a
**pure LP**, featuring:

- Separate charge/discharge variables (`Pch`, `Pdis`) with efficiencies.
- Two‑hour sustain rule (configurable `D`).
- Curtailment to avoid infeasibility under PV surplus.
- Optional grid import/export pricing.
- C‑rate linkage, ramp limits, degradation, CAPEX & OPEX amortisation.
- Realistic PV GHI from PVGIS via pvlib for Marseille.

Tweak the *Configuration* section to your parameters (costs, dates, bounds, etc.).
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pvlib.iotools import get_pvgis_tmy
import pvdata as pv
from datetime import datetime



# -----------------------------
# Configuration
# -----------------------------
# Time horizon
days = 5
hours_per_day = 24
T = days * hours_per_day
h = range(T)
# Time step
dt = 1.0  # hours

# Sustain duration (adaptive window)
D = 1.0  # hours

#--------------------
# Fetch data
#---------------------
# PV data
lat, lon = 43.3026, 5.3691 # for Marseille
start = datetime(2023, 1, 5, 5) # format YYYY, MM, DD, hh
end   = datetime(2023, 1, 4+days, 5)
area = 2000000 # in m^2
data = pv.fetch_data(lat, lon, start, end,
                email='saifiitp16@gmail.com',
                identifier='cams_radiation')
power = pv.compute_pv_power(data,area)


# Capital costs (amortised)
CAPEX_ENERGY_EUR_PER_MWH = 150_000
CAPEX_POWER_EUR_PER_MW   = 70_000
ANNUAL_YEARS = 10
HOURLY_CAPEX_ENERGY = CAPEX_ENERGY_EUR_PER_MWH / (ANNUAL_YEARS * 365 * 24)
HOURLY_CAPEX_POWER  = CAPEX_POWER_EUR_PER_MW   / (ANNUAL_YEARS * 365 * 24)

# Operating costs (amortised)
OPEX_ENERGY_EUR_PER_MWH_PER_YEAR = 5_000
OPEX_POWER_EUR_PER_MW_PER_YEAR   = 500
HOURLY_OPEX_ENERGY = OPEX_ENERGY_EUR_PER_MWH_PER_YEAR / (365 * 24)
HOURLY_OPEX_POWER  = OPEX_POWER_EUR_PER_MW_PER_YEAR   / (365 * 24)

# Physical bounds
E_MIN_FIXED = 20.0   # MWh safety buffer
E_CAP_MAX   = 300.0  # MWh
P_CAP_MAX   = 80.0   # MW
C_RATE_MAX  = 1.0    # max C-rate: P_cap ≤ C_RATE_MAX * E_cap

# Efficiencies
ETA_CH  = 0.96
ETA_DIS = 0.96

# Market & penalties
dt = 1.0
pi_capacity = np.array([5.0 if 11 <= t < 14 else 3.0 for t in h])  # €/MW-h
PI_SLACK    = 10.0  # €/MWh unmet load
PI_CURT     = 0.0    # €/MWh curtailment
PI_DEG      = 1.0    # €/MWh throughput
PI_TERM     = 0.5    # €/MWh SOC end penalty
USE_ENERGY_PRICES = True
PI_IMPORT = 120.0  # €/MWh
PI_EXPORT = 30.0   # €/MWh

# PVGIS location
lat, lon = 43.3, 5.4  # Marseille

# -----------------------------
# Data: PV and Load
# -----------------------------
# try:
#     tmy, meta = get_pvgis_tmy(latitude=lat, longitude=lon)
#     year = tmy.index.year[0]
#     date0 = pd.to_datetime(f"{year}-06-21")
#     ghi = tmy['G(h)'].loc[date0: date0 + pd.Timedelta(hours=T-1)]
#     # Scale to MW
#     AREA = 450_000.0  # m^2
#     EFF  = 0.20
#     P_PV = ghi.values * AREA * EFF / 1e6
# except Exception as e:
#     # Fallback synthetic if PVGIS fails
#     np.random.seed(0)
#     P_PV = np.clip(50*np.sin(np.linspace(0, 2*np.pi, T)) + 50, 0, None)
#     print("Warning: PVGIS fetch failed, using synthetic PV profile.")
P_PV = power['pv_power_MW'].tolist()
# Load: 50 ±10 MW noise
def make_load(seed=0):
    rng = np.random.RandomState(seed)
    return 50 + rng.uniform(-10, 10, size=T)
P_load = make_load(77)
# Precompute constant net-load demand per hour (clip negatives to 0)
net_load = [max(P_load[t] - P_PV[t], 0.0) for t in h]   # MW

# -----------------------------
# Build LP model
# -----------------------------
model = gp.Model('PV_Battery_Sizing')
model.Params.OutputFlag = 0

# Decision vars: sizing
E_cap = model.addVar(lb=E_MIN_FIXED, ub=E_CAP_MAX, name='E_cap')
P_cap = model.addVar(lb=0.0, ub=P_CAP_MAX, name='P_cap')

# Time series vars
E    = model.addVars(T+1, lb=E_MIN_FIXED, ub=E_CAP_MAX, name='E_state')
Pch  = model.addVars(T, lb=0.0, name='P_ch')
Pdis = model.addVars(T, lb=0.0, name='P_dis')
Pc   = model.addVars(T, lb=0.0, ub=P_CAP_MAX, name='P_cap_bid')
Curt = model.addVars(T, lb=0.0, name='Curtail')
S    = model.addVars(T, lb=0.0, name='Slack')
# Grid import/export if needed
if USE_ENERGY_PRICES:
    Gimp = model.addVars(T, lb=0.0, name='Grid_Imp')
    Gexp = model.addVars(T, lb=0.0, name='Grid_Exp')
else:
    Gimp = {t:0.0 for t in h}
    Gexp = {t:0.0 for t in h}
# Terminal SOC slack
E_pos = model.addVar(lb=0.0, name='E_term_pos')
E_neg = model.addVar(lb=0.0, name='E_term_neg')

# Constraints
model.addConstr(E[0] == E_MIN_FIXED + 0.5*(E_cap-E_MIN_FIXED), 'Init_SOC')
model.addConstr(
    gp.quicksum(S[t] * dt for t in h) <= 0.10 * sum(n * dt for n in net_load),
    name="SlackBudget_10pctNetLoad"
)
for t in h:
    # Energy balance
    model.addConstr(E[t+1] == E[t] + dt*(ETA_CH*Pch[t] - Pdis[t]/ETA_DIS), f'E_bal_{t}')
    # Power balance
    model.addConstr(
        P_PV[t] - Curt[t] + Pdis[t] - Pch[t]
        + (Gimp[t] if USE_ENERGY_PRICES else 0)
        - (Gexp[t] if USE_ENERGY_PRICES else 0)
        - P_load[t] + S[t] == 0, f'P_bal_{t}'
    )
    # Headroom
    model.addConstr(Pdis[t] + Pc[t] <= P_cap, f'dis_head_{t}')
    model.addConstr(Pch[t]  + Pc[t] <= P_cap, f'ch_head_{t}')
    model.addConstr(Pc[t] <= P_cap, f'Pc_le_{t}')
    # SOC windows
    model.addConstr(E[t] >= E_MIN_FIXED + Pc[t]*D/ETA_DIS, f'E_min_{t}')
    model.addConstr(E[t] <= E_cap - Pc[t]*D*ETA_CH, f'E_max_{t}')
# Terminal SOC
model.addConstr(E[T] + E_pos - E_neg == E[0], 'Term_SOC')
# C-rate
model.addConstr(P_cap <= C_RATE_MAX*E_cap, 'C_rate')
# Duration tightener
for t in h:
    model.addConstr(Pc[t] <= (E_cap-E_MIN_FIXED)/(D*(ETA_CH+1/ETA_DIS)), f'dur_tight_{t}')

# Objective components
cap_rev = gp.quicksum(pi_capacity[t]*Pc[t]*dt for t in h)
energy_val = gp.quicksum((PI_EXPORT*Gexp[t] - PI_IMPORT*Gimp[t])*dt for t in h) if USE_ENERGY_PRICES else 0
curt_cost  = gp.quicksum(PI_CURT*Curt[t]*dt for t in h)
slack_cost = gp.quicksum(PI_SLACK*S[t]*dt for t in h)
deg_cost   = gp.quicksum(PI_DEG*(Pch[t]+Pdis[t])*dt for t in h)
term_cost  = PI_TERM*(E_pos+E_neg)
capex_cost = (HOURLY_CAPEX_ENERGY*E_cap + HOURLY_CAPEX_POWER*P_cap)*(T*dt)
opex_cost  = (HOURLY_OPEX_ENERGY*E_cap + HOURLY_OPEX_POWER*P_cap)*(T*dt)
# Set objective
model.setObjective(cap_rev + energy_val - curt_cost - slack_cost - deg_cost - term_cost - capex_cost - opex_cost, GRB.MAXIMIZE)

# Solve
model.optimize()

# Results if optimal
if model.status == GRB.OPTIMAL:
    print(f"Profit/day: €{model.objVal:,.2f}, Profit/hr: €{model.objVal/(T*dt):.2f}")
    print(f"E_cap={E_cap.X:.1f} MWh, P_cap={P_cap.X:.1f} MW")
    Pc_vals = [Pc[t].X for t in h]
    print(f"Avg bid={np.mean(Pc_vals):.1f} MW, Max bid={np.max(Pc_vals):.1f} MW")
        # Plot State of Charge and Capacity Bids
    E_vals   = np.array([E[t].X for t in range(T+1)])
    Pc_vals  = np.array([Pc[t].X for t in h])
    E_min_adapt = E_MIN_FIXED + Pc_vals*D/ETA_DIS
    E_max_adapt = E_cap.X - Pc_vals*D*ETA_CH
    S_vals   = np.array([S[t].X for t in h])

    if USE_ENERGY_PRICES:
        Gimp_vals = np.array([Gimp[t].X for t in h])
        Gexp_vals = np.array([Gexp[t].X for t in h])
    else:
        Gimp_vals = np.zeros(T)
        Gexp_vals = np.zeros(T)

    plt.figure(figsize=(12,5))
    plt.plot(range(T+1), E_vals, marker='o', label='SOC (MWh)')
    plt.step(h, E_min_adapt, where='mid', linestyle='--', label='E_min adaptive')
    plt.step(h, E_max_adapt, where='mid', linestyle='--', label='E_max adaptive')
    plt.step(h, Pc_vals, where='mid', marker='^', label='Capacity bid (MW)')
    plt.grid(True)
    plt.xlabel('Hour')
    plt.legend()
    plt.title('State of Charge & Capacity Bids')
    plt.show()

    # Plot Power Flows
    Pch_vals  = np.array([Pch[t].X for t in h])
    Pdis_vals = np.array([Pdis[t].X for t in h])
    Curt_vals = np.array([Curt[t].X for t in h])
    plt.figure(figsize=(12,5))
    plt.step(h, Pdis_vals, where='mid', label='Discharge (MW)')
    plt.step(h, -Pch_vals, where='mid', label='Charge (MW)')
    plt.step(h, P_PV, where='mid', label='PV generation (MW)')
    plt.step(h, -P_load, where='mid', label='Load (MW)')
    plt.step(h, S_vals, where='mid', linestyle='--', label='Slack (MW)')
    plt.step(h, Gimp_vals, where='mid', linestyle='--', label='Grid import (MW)')
    plt.step(h, -Gexp_vals, where='mid', linestyle='--', label='Grid export (MW)')
    plt.grid(True); plt.xlabel('Hour'); plt.legend()
    plt.title('Power Flows (incl. Slack & Grid)')
    plt.show()
    tot_load    = float(np.sum(P_load)     * dt)
    tot_pv      = float(np.sum(P_PV)       * dt)
    tot_dis     = float(np.sum(Pdis_vals)  * dt)
    tot_ch      = float(np.sum(Pch_vals)   * dt)
    tot_import  = float(np.sum(Gimp_vals)  * dt)
    tot_export  = float(np.sum(Gexp_vals)  * dt)
    tot_slack   = float(np.sum(S_vals)     * dt)
    allowed_slk = 0.10 * sum(n * dt for n in net_load)
    
    print("Energy totals over horizon (MWh):")
    print(f"  Load        : {tot_load:10.1f}")
    print(f"  PV          : {tot_pv:10.1f}")
    print(f"  Discharge   : {tot_dis:10.1f}")
    print(f"  Charge      : {tot_ch:10.1f}")
    print(f"  Grid import : {tot_import:10.1f}")
    print(f"  Grid export : {tot_export:10.1f}")
    print(f"  Slack       : {tot_slack:10.1f}  (limit {allowed_slk:.1f}, "
          f"{100*tot_slack/allowed_slk:.1f}% used)")


else:
    print("No optimal solution, status", model.status)
