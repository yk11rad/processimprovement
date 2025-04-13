# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate Synthetic Data
n_orders = 1000
orders = pd.DataFrame({
    'order_id': range(1, n_orders + 1),
    'item_count': np.random.randint(1, 50, n_orders),
    'customer_priority': np.random.choice(['low', 'medium', 'high'], n_orders, p=[0.5, 0.3, 0.2])
})

# Define process steps with fixed/variable types
process_steps = {
    'order_receipt': {'time': lambda: np.random.normal(5, 1), 'cost': 2.0, 'type': 'fixed'},
    'order_processing': {'time': lambda: np.random.normal(10, 2), 'cost': 5.0, 'type': 'fixed'},
    'inventory_check': {'time': lambda: np.random.normal(8, 1.5), 'cost': 3.0, 'type': 'variable'},
    'packing': {'time': lambda: np.random.normal(15, 3), 'cost': 7.0, 'type': 'variable'},
    'shipping': {'time': lambda: np.random.normal(20, 4), 'cost': 10.0, 'type': 'variable'}
}

# Priority impact: high priority is faster
priority_impact = {'low': 1.0, 'medium': 0.9, 'high': 0.8}

# Step 2: Simulate Baseline Process
def simulate_baseline(order):
    total_time = 0
    total_cost = 0
    for step, params in process_steps.items():
        # Apply priority impact
        time = params['time']() * priority_impact[order['customer_priority']]
        cost = params['cost']
        # Handle fixed vs variable
        if params['type'] == 'variable':
            time *= (order['item_count'] / 10)
            cost *= (order['item_count'] / 10)
        # Add rework (5% chance)
        if np.random.rand() < 0.05:
            time += params['time']() * priority_impact[order['customer_priority']]
            cost += params['cost'] * 0.5
        total_time += max(0, time)
        total_cost += cost
    return {'order_id': order['order_id'], 'total_time': total_time, 'total_cost': total_cost}

# Step 3: Simulate Lean Process
lean_process_steps = process_steps.copy()
lean_process_steps['inventory_check']['time'] = lambda: np.random.normal(4, 0.75)
lean_process_steps['order_processing']['time'] = lambda: np.random.normal(7, 1.5)

def simulate_lean(order):
    total_time = 0
    total_cost = 0
    for step, params in lean_process_steps.items():
        time = params['time']() * priority_impact[order['customer_priority']]
        cost = params['cost']
        if params['type'] == 'variable':
            time *= (order['item_count'] / 10)
            cost *= (order['item_count'] / 10)
        if np.random.rand() < 0.05:
            time += params['time']() * priority_impact[order['customer_priority']]
            cost += params['cost'] * 0.5
        total_time += max(0, time)
        total_cost += cost
    return {'order_id': order['order_id'], 'total_time': total_time, 'total_cost': total_cost}

# Step 4: Simulate Agile Process
def simulate_agile(order):
    total_time = 0
    total_cost = 0
    batches = (order['item_count'] + 9) // 10  # Ceiling division
    
    # Fixed steps (once per order)
    for step in ['order_receipt', 'order_processing']:
        params = process_steps[step]
        time = params['time']() * priority_impact[order['customer_priority']]
        cost = params['cost']
        if np.random.rand() < 0.05:
            time += params['time']() * priority_impact[order['customer_priority']]
            cost += params['cost'] * 0.5
        total_time += max(0, time)
        total_cost += cost
    
    # Variable steps (per batch)
    for _ in range(batches):
        inventory_time = process_steps['inventory_check']['time']() * priority_impact[order['customer_priority']]
        packing_time = process_steps['packing']['time']() * 0.8 * priority_impact[order['customer_priority']]
        shipping_time = process_steps['shipping']['time']() * 0.8 * priority_impact[order['customer_priority']]
        # Parallel packing/shipping
        batch_time = max(packing_time, shipping_time) + inventory_time
        total_time += max(0, batch_time)
        
        # Costs per batch
        inventory_cost = process_steps['inventory_check']['cost']
        packing_cost = process_steps['packing']['cost']
        shipping_cost = process_steps['shipping']['cost']
        if np.random.rand() < 0.05:
            inventory_cost += process_steps['inventory_check']['cost'] * 0.5
            packing_cost += process_steps['packing']['cost'] * 0.5
            shipping_cost += process_steps['shipping']['cost'] * 0.5
        total_cost += (inventory_cost + packing_cost + shipping_cost) * (10 / max(10, order['item_count']))
    
    return {'order_id': order['order_id'], 'total_time': total_time, 'total_cost': total_cost}

# Step 5: Simulate Hybrid (Lean + Agile)
hybrid_process_steps = lean_process_steps.copy()
hybrid_process_steps['packing']['time'] = lambda: np.random.normal(12, 2)  # Agile tweak

def simulate_hybrid(order):
    total_time = 0
    total_cost = 0
    batches = (order['item_count'] + 9) // 10
    
    # Fixed steps
    for step in ['order_receipt', 'order_processing']:
        params = hybrid_process_steps[step]
        time = params['time']() * priority_impact[order['customer_priority']]
        cost = params['cost']
        if np.random.rand() < 0.05:
            time += params['time']() * priority_impact[order['customer_priority']]
            cost += params['cost'] * 0.5
        total_time += max(0, time)
        total_cost += cost
    
    # Variable steps
    for _ in range(batches):
        inventory_time = hybrid_process_steps['inventory_check']['time']() * priority_impact[order['customer_priority']]
        packing_time = hybrid_process_steps['packing']['time']() * 0.8 * priority_impact[order['customer_priority']]
        shipping_time = hybrid_process_steps['shipping']['time']() * 0.8 * priority_impact[order['customer_priority']]
        batch_time = max(packing_time, shipping_time) + inventory_time
        total_time += max(0, batch_time)
        
        inventory_cost = hybrid_process_steps['inventory_check']['cost']
        packing_cost = hybrid_process_steps['packing']['cost']
        shipping_cost = hybrid_process_steps['shipping']['cost']
        if np.random.rand() < 0.05:
            inventory_cost += hybrid_process_steps['inventory_check']['cost'] * 0.5
            packing_cost += hybrid_process_steps['packing']['cost'] * 0.5
            shipping_cost += hybrid_process_steps['shipping']['cost'] * 0.5
        total_cost += (inventory_cost + packing_cost + shipping_cost) * (10 / max(10, order['item_count']))
    
    return {'order_id': order['order_id'], 'total_time': total_time, 'total_cost': total_cost}

# Step 6: Run Simulations with Multiprocessing
with Pool() as pool:
    baseline_results = pd.DataFrame(pool.map(simulate_baseline, orders.to_dict('records')))
    lean_results = pd.DataFrame(pool.map(simulate_lean, orders.to_dict('records')))
    agile_results = pd.DataFrame(pool.map(simulate_agile, orders.to_dict('records')))
    hybrid_results = pd.DataFrame(pool.map(simulate_hybrid, orders.to_dict('records')))

# Step 7: Combine Results
results = pd.DataFrame({
    'Baseline_Time': baseline_results['total_time'],
    'Baseline_Cost': baseline_results['total_cost'],
    'Lean_Time': lean_results['total_time'],
    'Lean_Cost': lean_results['total_cost'],
    'Agile_Time': agile_results['total_time'],
    'Agile_Cost': agile_results['total_cost'],
    'Hybrid_Time': hybrid_results['total_time'],
    'Hybrid_Cost': hybrid_results['total_cost']
})

# Step 8: Enhanced Analysis
# Summary with percentiles
summary = results.describe(percentiles=[0.25, 0.5, 0.75, 0.9])
print("Summary Statistics:\n", summary)

# Improvement percentages
mean_times = results[['Baseline_Time', 'Lean_Time', 'Agile_Time', 'Hybrid_Time']].mean()
mean_costs = results[['Baseline_Cost', 'Lean_Cost', 'Agile_Cost', 'Hybrid_Cost']].mean()
for method in ['Lean', 'Agile', 'Hybrid']:
    time_improv = 100 * (mean_times['Baseline_Time'] - mean_times[f'{method}_Time']) / mean_times['Baseline_Time']
    cost_improv = 100 * (mean_costs['Baseline_Cost'] - mean_costs[f'{method}_Cost']) / mean_costs['Baseline_Cost']
    print(f"{method} Time Improvement: {time_improv:.2f}%")
    print(f"{method} Cost Improvement: {cost_improv:.2f}%")

# Step 9: Visualizations
plt.figure(figsize=(14, 10))

# Boxplot for time
plt.subplot(2, 2, 1)
sns.boxplot(data=results[['Baseline_Time', 'Lean_Time', 'Agile_Time', 'Hybrid_Time']])
plt.title('Processing Time Distributions')
plt.ylabel('Time (minutes)')

# Boxplot for cost
plt.subplot(2, 2, 2)
sns.boxplot(data=results[['Baseline_Cost', 'Lean_Cost', 'Agile_Cost', 'Hybrid_Cost']])
plt.title('Cost Distributions')
plt.ylabel('Cost ($)')

# Bar plot for mean time
plt.subplot(2, 2, 3)
mean_times.plot(kind='bar', color=['blue', 'orange', 'green', 'purple'])
plt.title('Mean Processing Time')
plt.ylabel('Time (minutes)')

# Bar plot for mean cost
plt.subplot(2, 2, 4)
mean_costs.plot(kind='bar', color=['blue', 'orange', 'green', 'purple'])
plt.title('Mean Cost')
plt.ylabel('Cost ($)')

plt.tight_layout()
plt.show()

# Step 10: Sensitivity Analysis (Batch Size Impact)
batch_sizes = [5, 10, 20]
sensitivity_results = []

def simulate_agile_with_batch_size(order, batch_size):
    total_time = 0
    total_cost = 0
    batches = (order['item_count'] + batch_size - 1) // batch_size
    for step in ['order_receipt', 'order_processing']:
        params = process_steps[step]
        time = params['time']() * priority_impact[order['customer_priority']]
        total_time += max(0, time)
        total_cost += params['cost']
    for _ in range(batches):
        inventory_time = process_steps['inventory_check']['time']() * priority_impact[order['customer_priority']]
        packing_time = process_steps['packing']['time']() * 0.8 * priority_impact[order['customer_priority']]
        shipping_time = process_steps['shipping']['time']() * 0.8 * priority_impact[order['customer_priority']]
        batch_time = max(packing_time, shipping_time) + inventory_time
        total_time += max(0, batch_time)
        total_cost += (process_steps['inventory_check']['cost'] + 
                       process_steps['packing']['cost'] + 
                       process_steps['shipping']['cost']) * (batch_size / max(batch_size, order['item_count']))
    return {'total_time': total_time, 'total_cost': total_cost}

for batch_size in batch_sizes:
    with Pool() as pool:
        results = pd.DataFrame(pool.starmap(simulate_agile_with_batch_size, 
                                          [(order, batch_size) for order in orders.to_dict('records')]))
    sensitivity_results.append({
        'batch_size': batch_size,
        'mean_time': results['total_time'].mean(),
        'mean_cost': results['total_cost'].mean()
    })

# Plot sensitivity results
sensitivity_df = pd.DataFrame(sensitivity_results)
plt.figure(figsize=(8, 5))
plt.plot(sensitivity_df['batch_size'], sensitivity_df['mean_time'], marker='o', label='Mean Time')
plt.plot(sensitivity_df['batch_size'], sensitivity_df['mean_cost'], marker='s', label='Mean Cost')
plt.xlabel('Batch Size')
plt.ylabel('Value')
plt.title('Sensitivity Analysis: Batch Size Impact')
plt.legend()
plt.grid(True)
plt.show()

# README Section (Markdown)
"""
# README: Order Fulfillment Process Improvement Simulation

## Overview
This Google Colab notebook simulates an order fulfillment process to compare the performance of different process improvement methodologies: **Baseline**, **Lean**, **Agile**, and **Hybrid (Lean + Agile)**. The simulation uses synthetic data to model 1,000 orders, tracking **total processing time** and **total cost** across five steps: order receipt, order processing, inventory check, packing, and shipping.

### Key Features
- **Realistic Modeling**: 
  - Fixed steps (e.g., order receipt) have constant time/cost; variable steps (e.g., packing) scale with item count.
  - Customer priority (low, medium, high) impacts processing speed.
  - 5% chance of rework per step adds variability.
- **Improvements**:
  - **Lean**: Reduces inventory check time by 50% and order processing by 30%.
  - **Agile**: Processes orders in batches with parallel packing/shipping.
  - **Hybrid**: Combines Lean optimizations with Agile batching and faster packing.
- **Performance**: Multiprocessing speeds up simulations for large datasets.
- **Analysis**: Includes percentiles, boxplots, bar plots, and sensitivity analysis for batch sizes (5, 10, 20).

## Requirements
- Google Colab environment (no additional setup needed).
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `multiprocessing` (pre-installed in Colab).

## How to Run
1. Open this notebook in Google Colab: https://colab.research.google.com/
2. Copy and paste the entire code into a new notebook.
3. Run all cells (`Ctrl+F9`).
4. Review the outputs:
   - **Summary Statistics**: Mean, median, percentiles (25th, 50th, 75th, 90th).
   - **Improvement Percentages**: Time and cost reductions for Lean, Agile, and Hybrid.
   - **Visualizations**: Boxplots for distributions, bar plots for means, and a sensitivity plot for batch size impact.

## Outputs
- **Console**:
  - Summary table with detailed statistics.
  - Percentage improvements (e.g., "Lean Time Improvement: 15.23%").
- **Plots**:
  - Boxplots showing time and cost distributions.
  - Bar plots comparing mean time and cost across methods.
  - Sensitivity plot analyzing Agile batch size effects.
- **Insights**: Hybrid typically offers the best time savings, while Agile balances flexibility and cost.

## Customization
- Adjust `n_orders` to change the number of simulated orders.
- Modify `process_steps` to alter time/cost distributions or add steps.
- Experiment with `batch_sizes` in sensitivity analysis.
- Extend with constraints (e.g., worker limits) or additional metrics (e.g., error rates).

## Notes
- Random seed (`np.random.seed(42)`) ensures reproducibility.
- Simulations assume normal distributions for processing times; adjust for other distributions if needed.
- Rework probability (5%) and priority impacts (e.g., high = 80% time) can be tuned for realism.

## Author
Edward Antwi 
"""