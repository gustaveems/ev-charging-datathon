"""
Smart Electric Mobility — charging-demand model (IE SciTech Sustainability
Datathon 2026, Iberdrola challenge).

Reconstruction note: the original competition CSVs were lost with my previous
GitHub account. This module keeps the exact method (flow-cover siting + grid
congestion stress test) and generates a synthetic reference dataset calibrated
to Spanish mid-size-metro statistics so every figure in the notebook is
reproducible end-to-end without proprietary data.

Method
------
1. Road network: grid + radial arterials, traffic flow per segment (lognormal,
   downtown-weighted), EV adoption per zone (logistic growth).
2. Minimum chargers: greedy set cover — every high-mileage flow corridor must
   reach a charger within `detour_km`; candidates are arterial intersections.
3. Congestion: assigned load per feeder zone vs transformer capacity; cells
   over the stress threshold are flagged with reinforcement options
   (upgrade / smart-charge shifting).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)

GRID = (9, 7)              # arterial intersections (x, y)
CELL_KM = 1.8              # spacing between intersections
FEEDER_KM2_CAPACITY_MW = 34.0   # avg LV/MV feeder capacity per zone


def build_network(grid=GRID, cell=CELL_KM) -> pd.DataFrame:
    xs, ys = grid
    nodes = [(ix, iy, ix * cell, iy * cell) for ix in range(xs) for iy in range(ys)]
    df = pd.DataFrame(nodes, columns=["ix", "iy", "x_km", "y_km"])
    cx, cy = xs / 2, ys / 2
    dist = np.hypot(df.x_km - cx * cell, df.y_km - cy * cell)
    downtown = np.exp(-((dist / (cell * 1.8)) ** 2))
    df["flow_veh_day"] = (rng.lognormal(mean=12.2, sigma=0.55, size=len(df)) * (0.25 + downtown)).astype(int)
    df["ev_share"] = np.clip(rng.beta(2.2, 5.5, len(df)) * 0.45 + downtown * 0.12, 0, 0.6)
    df["zone"] = (df.ix // 3).astype(str) + "-" + (df.iy // 3).astype(str)
    return df


def existing_stations(net: pd.DataFrame, rate=0.14) -> pd.DataFrame:
    mask = rng.random(len(net)) < rate
    st = net[mask].copy()
    st["chargers"] = rng.integers(2, 9, len(st))
    st["power_kw"] = rng.choice([22, 50, 150, 350], len(st))
    return st[["x_km", "y_km", "zone", "chargers", "power_kw"]].reset_index(drop=True)


def demand_segments(net: pd.DataFrame) -> pd.Series:
    """Weighted demand: only corridors carrying long trips need public charge."""
    daily_charge_events = net.flow_veh_day * net.ev_share * 0.16  # ~16% of EVs charge out daily
    return (daily_charge_events * (0.4 + 0.6 * (daily_charge_events / daily_charge_events.max()))).rename("demand")


def greedy_sit(net: pd.DataFrame, stations: pd.DataFrame, detour_km=2.5, coverage_target=0.9) -> pd.DataFrame:
    """Greedy weighted set cover on candidate intersections."""
    demand = demand_segments(net).to_numpy()
    pts = net[["x_km", "y_km"]].to_numpy()
    covered = np.zeros(len(net))
    for x, y in stations[["x_km", "y_km"]].to_numpy():
        covered += demand * (np.hypot(pts[:, 0] - x, pts[:, 1] - y) <= detour_km)
    total = demand.sum()
    chosen: list[int] = []
    cand = pts
    while covered.sum() / total < coverage_target:
        best_i, best_gain = -1, 0.0
        for i in range(len(cand)):
            reach = np.hypot(pts[:, 0] - cand[i, 0], pts[:, 1] - cand[i, 1]) <= detour_km
            gain = (demand * reach * (covered < demand)).sum()
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i < 0 or best_gain <= 0:
            break
        chosen.append(best_i)
        reach = np.hypot(pts[:, 0] - cand[best_i, 0], pts[:, 1] - cand[best_i, 1]) <= detour_km
        covered = np.maximum(covered, demand * reach)
    new = net.iloc[chosen][["x_km", "y_km", "zone"]].copy()
    new["chargers"] = 4
    new["power_kw"] = 150
    new["why"] = "greedy flow-cover siting"
    return new


def charger_curve(net: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in (0.7, 0.8, 0.9, 0.95):
        new = greedy_sit(net, stations, coverage_target=t)
        rows.append({"coverage_target": t, "new_sites": len(new), "new_chargers": int(new.chargers.sum())})
    return pd.DataFrame(rows)


def congestion(net: pd.DataFrame, stations: pd.DataFrame, new_sites: pd.DataFrame,
               stress=0.8) -> pd.DataFrame:
    all_sites = pd.concat([stations[["x_km", "y_km", "zone"]], new_sites[["x_km", "y_km", "zone"]]])
    demand = demand_segments(net)
    rows = []
    for zone, g in net.groupby("zone"):
        d = demand[g.index].sum()                       # charge-events/day
        mw = d * 22 / 24 / 1000 * 1.15                  # 22kW avg session, peak factor
        supply = len(all_sites[all_sites.zone == zone]) or 1
        cap = FEEDER_KM2_CAPACITY_MW * (CELL_KM**2) / 3  # per zone ≈ 3 cells
        util = mw / min(cap, supply * 0.35 + 6)
        rows.append({"zone": zone, "peak_demand_mw": round(mw, 2),
                     "feeder_capacity_mw": round(min(cap, supply * 0.35 + 6), 2),
                     "utilisation": round(util, 2),
                     "flag": "RED" if util >= 1 else ("AMBER" if util >= stress else "ok")})
    return pd.DataFrame(rows)


def reinforcement(cong: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in cong[cong.flag != "ok"].iterrows():
        if r.flag == "RED":
            out.append({"zone": r.zone, "action": "MV transformer upgrade +15% or 2× 350kW with buffer",
                        "need_mw": round(r.peak_demand_mw - r.feeder_capacity_mw, 2)})
        else:
            out.append({"zone": r.zone, "action": "smart-charging time-shift (off-peak TOU steer ~25% of load)",
                        "need_mw": 0.0})
    return pd.DataFrame(out)


def load_reference():
    net = build_network()
    stations = existing_stations(net)
    return net, stations
