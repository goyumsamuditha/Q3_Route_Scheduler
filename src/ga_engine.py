import numpy as np

class GAEngine:
    def __init__(self, n_nodes, population_size, penalty_value, vehicle_capacity, vehicle_shifts_min):
        self.n_nodes = n_nodes
        self.population_size = population_size
        self.penalty_value = penalty_value
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_shifts_min = vehicle_shifts_min
        
    def initialize_population(self, population_size, n_nodes):
        """logic to generate population number of arrays.
        each array is a permutation of numbers from 0 to n_nodes-1"""
        population = [np.random.permutation(n_nodes) for _ in range(population_size)]
        return np.array(population)

    def decode_chromosome(self, chromosome, matrices):
        """logic to decode a chromosome into a list of routes"""
        
        routes = []
        current_vehicle_id = 0
        current_time = 0
        current_load = 0
        current_route = [0]  # Start with the depot (node 0)
        # pull required arrays from matrices
        cargo_weights = matrices['cargo_weights']
        time_cost = matrices['time_cost_matrix']
        service = matrices['service_times']
        
        for node in chromosome:
            if node == 0:
                continue  # Skip depot in the chromosome
                
            cap = self.vehicle_capacity[current_vehicle_id]
            limit = self.vehicle_shifts_min[current_vehicle_id]
        
            if (current_load + cargo_weights[node] <= cap and
                current_time + service[node] + time_cost[current_route[-1], node] <= limit):
                
                current_load += cargo_weights[node]
                current_time += service[node] + time_cost[current_route[-1], node]
                current_route.append(node)
            else:
                routes.append(current_route + [0])  # Return to depot
                current_vehicle_id += 1
                if current_vehicle_id >= len(self.vehicle_capacity):
                    break  # No more vehicles available
                
                current_load = cargo_weights[node]
                current_time = service[node] + time_cost[0, node]
                current_route = [0, node]  # Start new route from depot
                
        routes.append(current_route + [0])  # Append the last route returning to depot
        return routes

    def calculate_fitness(self, decoded_routes, order_values, priorities, cargo_weights):
        """logic to calculate fitness of a chromosome based on decoded routes"""
        route_nodes = [node for route in decoded_routes for node in route if node != 0]
        fitness = sum(order_values[node] for node in route_nodes)
        
        all_nodes = range(len(order_values))
        missed_nodes = [n for n in all_nodes if n not in route_nodes and n != 0]
        for n in missed_nodes:
            if priorities[n] == 5:
                fitness -= self.penalty_value
                
        infeasible_penalty = 1000000
        for route in decoded_routes:
            total_load = sum(cargo_weights[n] for n in route)
            # Use the singular name defined in __init__
            if total_load > self.vehicle_capacity[0]: 
                fitness -= infeasible_penalty 
                
        return fitness

    def crossover_ox(self, parent1, parent2):
        """logic to perform crossover between two parents and return a child chromosome"""
        # Implement Order Crossover (OX)
        size = len(parent1)
        start, end = sorted(np.random.choice(range(size), 2, replace=False))
        child = np.full(size, -1)
        child[start:end] = parent1[start:end]
        
        p2_elements = [item for item in parent2 if item not in child]
        idx = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = p2_elements[idx]
                idx += 1
        return child
    
    
    def mutate_swap(self, chromosome):
        """logic to perform swap mutation on a chromosome"""
        mutated_chrom = chromosome.copy()
        idx1, idx2 = np.random.choice(len(mutated_chrom), 2, replace=False)
        mutated_chrom[idx1], mutated_chrom[idx2] = mutated_chrom[idx2], mutated_chrom[idx1]
        return mutated_chrom
        
        
        
        