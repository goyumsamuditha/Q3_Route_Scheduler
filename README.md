# Sri Lanka Last-Mile Delivery Optimisation — GA vs MIP

This project tackles a **Vehicle Routing Problem with Time Windows** for a delivery hub based in Colombo, Sri Lanka. Unlike a textbook CVRP, the fleet
cannot reach every customer within a single working day, so the model has to decide **which** orders to serve as well as **how** to route them, in order to maximise the total delivery value collected. The problem is solved two ways — a **Genetic Algorithm (GA)** built from scratch in NumPy, and a **Mixed Integer Programme (MIP)** built with PuLP/CBC — so the two approaches can be compared on solution quality, speed and scalability.

## Project structure

```
.
├── data/
│   ├── raw/
│   │   └── VRP_Sri_Lanka_Dataset.xlsx     
│   └── processed/
│       ├── cleaned_delivery_orders.csv     
│       └── vrp_mathematical_matrices.npz   
│
├── notebooks/
│   ├── ETL_Data_Prep.ipynb                 
│   ├── GA_Evolutionary_Engine.ipynb        
│   ├── MIP_Baseline.ipynb                 
│   └── Report_Analysis.ipynb  
│   ├── ga_results.json                         
│   ├── mip_results.json                       
│   ├── mip_scalability_results.json            
│   ├── vrp_debug.lp                           
│   ├── dispatch_map.html
│   └── hyperparameter_tuning_results.png             
│
├── src/
│   ├── ga_engine.py                        
│   ├── dashboard.py 
│   └── visualization.py                    
│                                               
├── requirements.txt
└── README.md
```

## Setting it up

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in this order — each one reads what the previous one wrote:

1. **`ETL_Data_Prep.ipynb`** — loads `VRP_Sri_Lanka_Dataset.xlsx`, fixes the missing values,
   inconsistent text and corrupted numeric fields, then builds the distance/time matrices used
   by both solvers and writes them to `data/processed/`.
2. **`GA_Evolutionary_Engine.ipynb`** — runs the Genetic Algorithm on all 191 orders, plots the
   convergence curve, sweeps population size and mutation rate, and saves `ga_results.json`.
3. **`MIP_Baseline.ipynb`** — builds the exact MIP formulation in PuLP, solves it on progressively
   larger slices of the problem (5, 10, 15, 20 orders) and saves the two `mip_*.json` files.
4. **`Report_Analysis.ipynb`** — reloads both result sets, lines them up in one table, and
   generates the interactive dispatch map (`dispatch_map.html`).

If you just want to look at the results without re-running anything, the `outputs/` folder has
static PNGs of the convergence curve, the route map and the GA-vs-MIP comparison, and
`dispatch_map.html` can be opened directly in a browser.

## The optimisation problem in one paragraph

Twenty vehicles start from a single depot in Colombo. There are 191 delivery orders spread
across Sri Lanka, each with a cargo weight, a monetary value, a priority level, a service time
and a delivery time window. A vehicle cannot carry more than its rated capacity and cannot work
longer than its shift limit. Because there isn't enough vehicle-time in a day to reach every
order, the objective is to choose the subset of orders to serve and the routes to serve them
with, so that the total value collected is as high as possible, minus travel cost and minus a
penalty for leaving a Priority-5 (contracted) customer unserved.

## Where the numbers in the report come from

All figures quoted in the accompanying assignment report are taken directly from the JSON result
files in this repository (`ga_results.json`, `mip_results.json`, `mip_scalability_results.json`)
and from re-running `run_ga_engine()` with the same random seed (42) used in the original
notebook, which reproduces the same result to the LKR. The full-scale GA run visited 191 orders
in 212 generations and 30–125 seconds depending on machine load; the MIP was only tested up to 20
orders because beyond that CBC could not close the optimality gap within a 300-second limit.

## Notes and assumptions

- All monetary values are illustrative LKR figures for coursework purposes, not real tariffs.
- Distances are computed with the Haversine (great-circle) formula rather than road-network
  distances, then adjusted with a road-condition speed multiplier