import sys
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

file_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(file_dir)                  

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Now Python can cleanly see the 'src' package
from src.ga_engine import GAEngine

# 1. Securely load environment variables
load_dotenv()
MAP_API_KEY = os.getenv("MAP_API_KEY")
px.set_mapbox_access_token(MAP_API_KEY)

# 2. Configure the page layout
st.set_page_config(page_title="Dispatch Dashboard", layout="wide")

# 3. Inject Modern Soft UI CSS
st.markdown("""
<style>
    .soft-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #f0f0f0;
    }
    .badge-success {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9em;
        border: 1px solid #28a745;
        display: inline-block;
        margin-bottom: 5px;
    }
    .badge-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9em;
        border: 1px solid #ffc107;
        display: inline-block;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📦 National Dispatch Control Center")
st.markdown("Automated Routing via Evolutionary Heuristic Engine")

# 4. Load Real Data
@st.cache_resource
def load_data():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(file_dir)
    data_dir = os.path.join(project_root, 'data', 'processed')
    
    df = pd.read_csv(os.path.join(data_dir, "cleaned_delivery_orders.csv"))
    # Load the file, but keep it as a resource
    matrices = np.load(os.path.join(data_dir, "vrp_mathematical_matrices.npz"), allow_pickle=True)
    return df, matrices

df, matrices = load_data()

# 5. Run Optimization Button
if st.button("🚀 Run Evolutionary Optimization"):
    with st.spinner('Running Genetic Algorithm (Evaluating 200 nodes)...'):
        # Instantiate GA Engine
        n_nodes = len(matrices['cargo_weights'])
        ga = GAEngine(
            n_nodes, 
        population_size=50, 
        penalty_value=10000, 
        vehicle_capacity=matrices['vehicle_capacities'], 
        vehicle_shifts_min=matrices['vehicle_shifts_min']
        )
        
        # Run a quick 20-generation evolution for dashboard speed
        population = ga.initialize_population(50, n_nodes)
        best_chromosome = population[0]
        max_fitness = -np.inf
        
        for generation in range(20):
            fitness_scores = np.array([ga.calculate_fitness(
                ga.decode_chromosome(chrom, matrices),
                matrices['order_values'], 
                df['Priority_Level'].to_numpy(), 
                matrices['cargo_weights']
            ) for chrom in population])
            
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > max_fitness:
                max_fitness = fitness_scores[best_idx]
                best_chromosome = population[best_idx]
            
            # Simplified next-gen logic for UI speed
            population = [ga.mutate_swap(chrom) if np.random.rand() < 0.1 else chrom for chrom in population]

        # Decode the absolute best result
        best_routes = ga.decode_chromosome(best_chromosome, matrices)
        
        # Parse output for UI mapping
        route_nodes = [node for route in best_routes for node in route if node != 0]
        deferred_nodes = [n for n in range(1, n_nodes) if n not in route_nodes]
        
        # Append Map Status to DataFrame
        df['Mapping_Status'] = 'Deferred'
        df.loc[route_nodes, 'Mapping_Status'] = 'Routed - Standard'
        df.loc[(df.index.isin(route_nodes)) & (df['Priority_Level'] == 5), 'Mapping_Status'] = 'Routed - Priority 5 (SLA Met)'
        df.loc[0, 'Mapping_Status'] = 'Depot'

        # 6. Build the UI
        col1, col2 = st.columns([2, 1])

        # Primary Map Panel
        with col1:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("🗺️ Live Network Map")
            
            fig = go.Figure()
            
            # Order Points
            fig.add_trace(go.Scattermapbox(
                lat=df['Latitude'],
                lon=df['Longitude'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=9,
                    color='gray',
                    opacity=0.7
                ),
                name='All Orders'
            ))
            
            # Actual Routed Orders
            routed_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            for i, route in enumerate(best_routes):
                if len(route) > 2:
                    route_df = df.iloc[route]
                    
                    fig.add_trace(go.Scattermapbox(
                        lat = route_df['Latitude'],
                        lon = route_df['Longitude'],
                        mode = 'lines+markers',
                        line = dict(width=4, color=routed_colors[i % len(routed_colors)]),
                        marker = dict(size=10, color=routed_colors[i % len(routed_colors)]),
                        name = f'Vehicle {i+1} Route'
                    ))

            # map layout
            fig.update_layout(
                mapbox = dict(
                    accesstoken = MAP_API_KEY,
                    center = dict(lat=7.8731, lon=80.7718),
                    zoom = 7,
                    style = "carto-positron"
                ),
                showlegend = True,
                margin = {"r":0,"t":0,"l":0,"b":0}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Secondary Sidebar: Deferred Queue
        with col2:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("🟡 Deferred Queue")
            
            if len(deferred_nodes) > 0:
                for node in deferred_nodes[:15]: # Show top 15 to avoid clutter
                    st.markdown(f'<div class="badge-warning">Order {node} - Pending</div>', unsafe_allow_html=True)
                if len(deferred_nodes) > 15:
                    st.write(f"...and {len(deferred_nodes) - 15} more.")
            else:
                st.write("All orders successfully routed!")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Success SLA Panel
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("🟢 High-Priority Dispatches")
            priority_routed = df[df['Mapping_Status'] == 'Routed - Priority 5 (SLA Met)'].index.tolist()
            for node in priority_routed[:15]:
                st.markdown(f'<div class="badge-success">Order {node} - SLA Met</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Click the button above to run the GA Engine and map the logistics network.")