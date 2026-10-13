# IE SciTech Sustainability Datathon 2026 — reconstruction

Challenge entries for **Iberdrola (Smart Electric Mobility)** and **Aqualia (ESG
materiality)**. The original notebooks and competition CSVs were lost along with a previous
GitHub account — this repo reconstructs the **method and findings** I presented,
running on a synthetic reference dataset calibrated to Spanish mid-size-metro statistics,
so everything is reproducible without proprietary data.

## Contents

| Notebook | Challenge | Core technique |
|---|---|---|
| `notebooks/01_smart_electric_mobility.ipynb` | Iberdrola | weighted greedy set-cover siting → minimum chargers vs coverage guarantee; feeder-level congestion stress test with reinforcement options |
| `notebooks/02_aqualia_esg_materiality.ipynb` | Aqualia | CSRD/ESRS double-materiality matrix, emerging-topic shortlist |

`src/datathon_model.py` holds the shareable model (network, demand, siting, congestion,
reinforcement) so the notebook stays narrative.

## Key results (Iberdrola)

- 90 % detour-reachable coverage needs only **4 new sites / 16 fast chargers**;
  pushing 90 → 95 % costs ~50 % more capacity (the marginal-cost curve was the pitch).
- Two feeder zones land RED after siting → MV upgrade or buffered 350 kW;
  AMBER zones solved by TOU smart-charging time-shift (~25 % peak shave, no capex).

## Run

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

## Swap in real data

`build_network()` / `existing_stations()` accept any DataFrame with
`x_km, y_km, flow_veh_day, ev_share, zone` — feed them from OSM road-graph extracts,
`carga.miteco.es` charging census, and grid open data; the model functions are unchanged.
