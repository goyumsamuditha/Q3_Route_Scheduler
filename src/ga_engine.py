import numpy as np
import time

def run_ga_engine(n_nodes, population_size, penalty_value, vehicle_capacity, vehicle_shifts_min,
                  matrices, df, mx_generations=500, patience=20,mutation_rate=0.05, crossover_rate=0.8,
                  tournament_size=3, elite_fraction=0.05, seed=None, verbose=False):
    """Run the Genetic Algorithm engine for route optimization.""" 
    if seed is not None:
        np.random.seed(seed)
    ga = GAEngine(n_nodes, population_size, penalty_value, vehicle_capacity, vehicle_shifts_min)
    population = ga.initialize_population(population_size, n_nodes)
    
    history = []
    best_fitness = -np.inf
    no_improve_count = 0
    generation = 0
    start_time = time.time()
    
    for generation in range(mx_generations):
        fitness_scores = np.array([
            ga.calculate_fitness(
                ga.decode_chromosome(chrom, matrices),
                matrices['order_values'], df['Priority_Level'].to_numpy(), matrices['cargo_weights']
            ) for chrom in population
        ])

        num_elites = max(1, int(elite_fraction * population_size))
        elite_indices = np.argsort(fitness_scores)[-num_elites:]
        new_population = [population[i] for i in elite_indices]

        current_max = np.max(fitness_scores)
        if current_max > best_fitness + 1e-3:
            best_fitness = current_max
            no_improve_count = 0
        else:
            no_improve_count += 1

        history.append(current_max)
        if verbose and generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness Score = {current_max}")

        if no_improve_count >= patience:
            if verbose:
                print(f"Convergence achieved at generation {generation}. Stopping early.")
            break

        while len(new_population) < population_size:
            parent1 = ga.tournament_selection(population, fitness_scores, tournament_size)
            parent2 = ga.tournament_selection(population, fitness_scores, tournament_size)

            if np.random.rand() < crossover_rate:
                offspring = ga.crossover_ox(parent1, parent2)
            else:
                offspring = parent1.copy()

            if np.random.rand() < mutation_rate:
                offspring = ga.mutate_swap(offspring)

            new_population.append(offspring)

        population = np.array(new_population)

    execution_time = time.time() - start_time

    best_idx = np.argmax(fitness_scores)
    best_chromosome = population[best_idx]
    best_routes = ga.decode_chromosome(best_chromosome, matrices)

    routed_nodes = [node for route in best_routes for node in route if node != 0]
    deferred_nodes = [node for node in range(1, n_nodes) if node not in routed_nodes]
    actual_revenue = float(sum(matrices['order_values'][n] for n in routed_nodes))

    return {
        'history': history, 'best_routes': best_routes, 'deferred_nodes': deferred_nodes,
        'execution_time': execution_time, 'fitness_score': float(history[-1]),
        'actual_revenue': actual_revenue, 'generations_run': generation + 1
    }

def slice_matrices(matrices, N):
    """Slice the matrices to only include the first N nodes."""
    limit = N + 1
    return {
        'cargo_weights': matrices['cargo_weights'][:limit],
        'order_values': matrices['order_values'][:limit],
        'service_times': matrices['service_times'][:limit],
        'window_starts': matrices['window_starts'][:limit],
        'window_ends': matrices['window_ends'][:limit],
        'distance_matrix_km': matrices['distance_matrix_km'][:limit, :limit],
        'time_cost_matrix': matrices['time_cost_matrix'][:limit, :limit],
    }
    
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
        for vehicle_idx, route in enumerate(decoded_routes):
            total_load = sum(cargo_weights[n] for n in route)
            capacity = self.vehicle_capacity[min(vehicle_idx, len(self.vehicle_capacity)-1)]
            if total_load > capacity: 
                fitness -= infeasible_penalty 
                
        return fitness
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """logic to perform tournament selection and return a selected parent"""
        contender_idx = np.random.choice(len(population), size=tournament_size, replace=False)
        best_idx = contender_idx[np.argmax(fitness_scores[contender_idx])]
        return population[best_idx]

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
        
        
        
        