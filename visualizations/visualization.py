import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import sys

# --- 1. ROBUST PATH PATCHING ---
# Ensures agents find their .pkl, .csv, and .json files correctly by
# setting the project root as the working directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)
os.chdir(project_root)

from travel_agent import DataHandler

# --- 2. CONFIGURATION ---
OUTPUT_DIR = "visualizations/poster_visuals_pure"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global Visual Style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.titlesize': 16})


def main():
    print("--- 🚀 STARTING PURE STATE-BASED EVALUATION ---")

    # Initialize handler after chdir to ensure relative paths in agents work
    try:
        handler = DataHandler()
    except Exception as e:
        print(f"❌ Critical Error: Could not initialize DataHandler. {e}")
        return

    # Define the states for comparison
    target_states = ['HI', 'MA', 'NY', 'CA', 'FL', 'PA', 'TX', 'AZ', 'NV']

    # Define variation scenarios
    scenarios = [
        {"name": "Solo Weekender\n(1 Pax, 3 Days)", "ppl": 1, "days": 3},
        {"name": "Couple's Week\n(2 Pax, 7 Days)", "ppl": 2, "days": 7},
        {"name": "Family Roadtrip\n(4 Pax, 10 Days)", "ppl": 4, "days": 10}
    ]

    plot_data = []

    print("--- 🔍 COLLECTING STATE DATA ---")
    for state in target_states:
        for scen in scenarios:
            try:
                # Passing city="" triggers internal state-level fallbacks in:
                # 1. TransportAgent (state_lookup JSON)
                # 2. EntertainmentAgent (City profile average)
                # 3. FoodCostAgent (State-level grocery index)

                demo = {"males": scen['ppl'], "females": 0, "children": 0}

                est = handler.get_trip_estimate(
                    city="",  # Triggers pure state-level fallback
                    state=state,
                    days=scen['days'],
                    travelers=scen['ppl'],
                    budget_level="Moderate",
                    demographics=demo
                )

                bd = est.get('breakdown', {})
                plot_data.append({
                    'State': state,
                    'Scenario': scen['name'],
                    'Lodging': bd.get('Lodging', 0),
                    'Food': bd.get('Food', 0),
                    'Entertainment': bd.get('Entertainment', 0),
                    'Transport': bd.get('Transportation', 0)
                })
            except Exception as e:
                print(f"⚠️ Failed state {state} ({scen['name']}): {e}")

    # --- 3. VISUALIZATION ---
    if not plot_data:
        print("❌ No data was collected. Check agent logs.")
        return

    df_plot = pd.DataFrame(plot_data)
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=False)

    # Standard color palette for travel categories
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']  # Lodging, Food, Ent, Trans
    categories = ['Lodging', 'Food', 'Entertainment', 'Transport']

    for i, scenario_name in enumerate([s['name'] for s in scenarios]):
        ax = axes[i]

        # Filter by scenario and set State as the index for the x-axis labels
        subset = df_plot[df_plot['Scenario'] == scenario_name].copy()
        if subset.empty:
            continue
        subset = subset.set_index('State')

        # Plotting only the numeric categories to avoid TypeError
        subset[categories].plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=colors,
            width=0.7,
            legend=False
        )

        ax.set_title(scenario_name, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=0)

        # Sum only numeric categories for the total label
        # Calculation: $Total = \sum_{i \in Categories} Cost_i$
        totals = subset[categories].sum(axis=1)
        for idx, val in enumerate(totals):
            ax.text(
                idx,
                val + (val * 0.01),
                f"${int(val)}",
                ha='center',
                fontsize=9,
                fontweight='bold'
            )

    axes[0].set_ylabel("Pure Model State Estimate ($)", fontsize=13)

    # Unified Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        fontsize=12
    )

    plt.tight_layout()

    # Save Output
    save_path = os.path.join(OUTPUT_DIR, "pure_state_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ State-Based Chart Saved: {save_path}")


if __name__ == "__main__":
    main()