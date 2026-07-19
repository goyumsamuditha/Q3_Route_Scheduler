import  folium
import pandas as pd

def generate_dispatch_map(df_path, routes):
    """
    Generates an interactive Folium map showing the depot, deliveries, and vehicle routes.
    """
    
    #  load the data processed delivery orders
    df = pd.read_csv(df_path)
    
    # Get the depot location 
    depot_lat = df.iloc[0]['Latitude']
    depot_lon = df.iloc[0]['Longitude']
    
    # Initiate map centered on sri lanka
    dispatch_map = folium.Map(location=[7.8731, 80.7718], zoom_start=7, tiles='cartodbpositron') 
    
    
    # Depot marker
    folium.Marker(
        location=[depot_lat, depot_lon],
        popup='Depot',
        icon=folium.Icon(color='green', icon='home')
    ).add_to(dispatch_map)
    
    # colour palette for vehicles
    route_colors = [
    'blue', 'green', 'purple', 'orange', 
    'darkred', 'cadetblue', 'darkblue', 
    'darkgreen', 'pink', 'gray'
    ]
        
    # Iterate through each vehicle's route
    for vehicle_id, route in enumerate(routes):
        # empty routes
        if len(route) == 0:
            continue
        
        route_coordinates = []
        route_color = route_colors[vehicle_id % len(route_colors)]
        
        for step, node_idx in enumerate(route):
            lat = df.iloc[node_idx]['Latitude']
            lon = df.iloc[node_idx]['Longitude']
            route_coordinates.append((lat, lon))
            
            # Add a marker for each delivery point
            if node_idx != 0:  # Skip the depot
                order_id = df.iloc[node_idx]['Order_ID']
                weight = df.iloc[node_idx]['Cargo_Weight_kg']
                revenue = df.iloc[node_idx]['Order_Value_LKR']
                
                popup_text = f"""
                <b>Order ID:</b> {order_id}<br>
                <b>Vehicle :</b> {vehicle_id + 1}<br>
                <b>Weight:</b> {weight} kg<br>
                <b>Revenue:</b> {revenue} LKR
                """
                
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color=route_color, icon='info-sign')
                ).add_to(dispatch_map)
                
        # Draw the polyline for the vehicle's route
        folium.PolyLine(
            locations=route_coordinates,
            color=route_color,
            weight=5,
            opacity=0.7,
            tooltip=f'Vehicle {vehicle_id + 1} Route'
        ).add_to(dispatch_map)
    return dispatch_map