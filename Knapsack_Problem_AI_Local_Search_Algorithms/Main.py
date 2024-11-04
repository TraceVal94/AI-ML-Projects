
# imports
import numpy as np
import random
import matplotlib.pyplot as plt
from time import time 

## 1 DATA LOADING AND ANALYSIS ##

# Load the CSV file to examine the dataset
dataset_path = r'C:\Users\valba\Desktop\Knapsack_Problem\Task dataset (knapsack problem).csv'
knapsack_data = pd.read_csv(dataset_path)

# Display the first few rows of the dataset to understand its structure and check for any issues
knapsack_data.head(), knapsack_data.info()

## 2 EXTRACTING WEIGHTS AND VALUES ##

# Extract weights and values from the dataset
weights = knapsack_data['weights'].values
values = knapsack_data['values'].values

# Knapsack capacity
capacity = 1500

## 3 DEFINE FITNESS FUNCTION ##

# Define the fitness function for the Knapsack problem
def fitness(solution):
    total_weight = np.dot(solution, weights)
    if total_weight > capacity:
        return 0  # Penalize infeasible solutions
    return np.dot(solution, values)

## 4 IMPLEMENT CHOSEN ALGORITHMS ##

## ALGORITHM 1 ## 
# Simulated Annealing implementation
def simulated_annealing(weights, values, capacity, max_iterations=1000, initial_temp=1000, cooling_rate=0.95):
    num_items = len(weights)
    current_solution = np.random.randint(2, size=num_items)
    current_fitness = fitness(current_solution)
    best_solution = np.copy(current_solution)
    best_fitness = current_fitness
    temperature = initial_temp
    
    convergence = []
    
    for iteration in range(max_iterations):
        neighbor = np.copy(current_solution)
        index = np.random.randint(num_items)
        neighbor[index] = 1 - neighbor[index]  # Flip the inclusion status
        neighbor_fitness = fitness(neighbor)
        
        if neighbor_fitness > current_fitness or np.random.random() < np.exp((neighbor_fitness - current_fitness) / temperature):
            current_solution = neighbor
            current_fitness = neighbor_fitness
        
        if current_fitness > best_fitness:
            best_solution = np.copy(current_solution)
            best_fitness = current_fitness
        
        temperature *= cooling_rate
        convergence.append(best_fitness)
        
        if temperature < 0.1:
            break
    
    return best_solution, best_fitness, convergence

##ALGORITHM 2 ## 
def tabu_search(weights, values, capacity, max_iterations=1000, tabu_list_size=10):
    num_items = len(weights)
    current_solution = np.random.randint(2, size=num_items)
    current_fitness = fitness(current_solution)
    best_solution = np.copy(current_solution)
    best_fitness = current_fitness
    
    tabu_list = []
    convergence = []
    
    for iteration in range(max_iterations):
        neighbors = []
        for i in range(num_items):
            neighbor = np.copy(current_solution)
            neighbor[i] = 1 - neighbor[i]  # Flip the inclusion status
            neighbors.append((neighbor, fitness(neighbor), i))
        
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        for neighbor, neighbor_fitness, index in neighbors:
            if index not in tabu_list or neighbor_fitness > best_fitness:
                current_solution = neighbor
                current_fitness = neighbor_fitness
                tabu_list.append(index)
                if len(tabu_list) > tabu_list_size:
                    tabu_list.pop(0)
                break
        
        if current_fitness > best_fitness:
            best_solution = np.copy(current_solution)
            best_fitness = current_fitness
        
        convergence.append(best_fitness)
        
        if iteration - tabu_list_size > 0 and all(neighbors[j][1] <= best_fitness for j in range(len(neighbors))):
            break
    
    return best_solution, best_fitness, convergence

# Function to run an algorithm multiple times
def run_algorithm(algorithm, weights, values, capacity, num_runs=10, **kwargs):
    results = []
    for _ in range(num_runs):
        start_time = time()
        best_solution, best_fitness, convergence = algorithm(weights, values, capacity, **kwargs)
        end_time = time()
        results.append({
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'total_weight': np.dot(best_solution, weights),
            'execution_time': end_time - start_time,
            'convergence': convergence
        })
    return results

# Function to analyze results
def analyze_results(results, algorithm_name):
    df = pd.DataFrame(results)
    print(f"\nResults for {algorithm_name}:")
    print(f"Best solution: {df['best_fitness'].max()}")
    print(f"Average solution: {df['best_fitness'].mean():.2f}")
    print(f"Worst solution: {df['best_fitness'].min()}")
    print(f"Average execution time: {df['execution_time'].mean():.2f} seconds")
    print(f"Solution with highest fitness:\n{df.loc[df['best_fitness'].idxmax()]}")
    return df

# Function to plot results
def plot_results(sa_results, ts_results):
    sa_fitness = [r['best_fitness'] for r in sa_results]
    ts_fitness = [r['best_fitness'] for r in ts_results]

    plt.figure(figsize=(10, 6))
    plt.boxplot([sa_fitness, ts_fitness], labels=['Simulated Annealing', 'Tabu Search'])
    plt.title('Comparison of Algorithm Performance')
    plt.ylabel('Best Fitness (Value)')
    plt.savefig('algorithm_comparison.png')
    plt.close()

# Function to plot convergence
def plot_convergence(sa_convergence, ts_convergence):
    plt.figure(figsize=(10, 6))
    plt.plot(sa_convergence, label='Simulated Annealing')
    plt.plot(ts_convergence, label='Tabu Search')
    plt.title('Convergence of Algorithms')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.savefig('convergence_comparison.png')
    plt.close()

# Function for sensitivity analysis
def sensitivity_analysis(algorithm, weights, values, capacity, param_name, param_values):
    results = []
    for value in param_values:
        kwargs = {param_name: value}
        run_results = run_algorithm(algorithm, weights, values, capacity, num_runs=10, **kwargs)
        avg_fitness = np.mean([r['best_fitness'] for r in run_results])
        results.append((value, avg_fitness))
    
    values, fitnesses = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(values, fitnesses, marker='o')
    plt.title(f'Sensitivity Analysis: {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Average Best Fitness')
    plt.savefig(f'sensitivity_{param_name}.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Run Simulated Annealing
    sa_results = run_algorithm(simulated_annealing, weights, values, capacity, max_iterations=1000, initial_temp=1000, cooling_rate=0.95)
    sa_df = analyze_results(sa_results, "Simulated Annealing")

    # Run Tabu Search
    ts_results = run_algorithm(tabu_search, weights, values, capacity, max_iterations=1000, tabu_list_size=10)
    ts_df = analyze_results(ts_results, "Tabu Search")

    # Plot results
    plot_results(sa_results, ts_results)

    # Plot convergence (using the first run of each algorithm)
    plot_convergence(sa_results[0]['convergence'], ts_results[0]['convergence'])

    # Sensitivity analysis
    sensitivity_analysis(simulated_annealing, weights, values, capacity, 'cooling_rate',[0.8, 0.85, 0.9, 0.95, 0.99])
    sensitivity_analysis(tabu_search, weights, values, capacity, 'tabu_list_size',[5, 10, 15, 20, 25])

    print("\nAnalysis complete. Check the generated plots for visual comparisons.")

# To save the results to a CSV file for further analysis
sa_df.to_csv('simulated_annealing_results.csv', index=False)
ts_df.to_csv('tabu_search_results.csv', index=False)