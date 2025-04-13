# processimprovement
 README: Order Fulfillment Process Improvement Simulation

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
