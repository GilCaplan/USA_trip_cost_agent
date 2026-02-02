import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
import logging
import os
import sys

# --- 1. ROBUST PATH PATCHING ---
# Ensures agents find their model files by setting the project root as the working directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)
os.chdir(project_root)

# Import your agents
from sub_agents.entertainment_agent import EntertainmentAgent
from sub_agents.food_cost_agent import FoodCostAgent
from sub_agents.lodging_cost_agent import LodgingCostAgent
from sub_agents.transport_agent import TransportAgent

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION: State Grid Layout ---
STATE_GRID = {
    'WA': (0, 7), 'ID': (1, 6), 'MT': (2, 7), 'ND': (3, 7), 'MN': (4, 7), 'IL': (5, 6), 'WI': (5, 7), 'MI': (6, 7),
    'NY': (9, 6), 'MA': (10, 6), 'VT': (9, 7), 'NH': (10, 7), 'ME': (11, 7),
    'OR': (0, 6), 'NV': (1, 5), 'WY': (2, 6), 'SD': (3, 6), 'IA': (4, 6), 'IN': (5, 5), 'OH': (6, 5), 'PA': (7, 5),
    'NJ': (8, 5), 'CT': (9, 5), 'RI': (10, 5),
    'CA': (0, 4), 'UT': (1, 4), 'CO': (2, 4), 'NE': (3, 5), 'MO': (4, 5), 'KY': (5, 4), 'WV': (6, 4), 'VA': (7, 4),
    'MD': (8, 4), 'DE': (9, 4),
    'AZ': (1, 3), 'NM': (2, 3), 'KS': (3, 4), 'AR': (4, 4), 'TN': (5, 3), 'NC': (6, 3), 'SC': (7, 3), 'DC': (8, 3),
    'OK': (3, 3), 'LA': (4, 2), 'MS': (5, 2), 'AL': (6, 2), 'GA': (7, 2),
    'TX': (3, 1), 'FL': (8, 1),
    'AK': (0, 1), 'HI': (1, 1)
}

def get_aggregated_costs():
    logger.info("Initializing Agents...")
    try:
        # Paths are now relative to the project root due to os.chdir()
        ent_agent = EntertainmentAgent(city_data_path="sub_agents/models/city_profiles.csv")
        food_agent = FoodCostAgent()
        lodging_agent = LodgingCostAgent()
        trans_agent = TransportAgent()
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        return pd.DataFrame()

    results = []
    TRAVELERS = 1
    DAYS = 1

    logger.info("Querying agents for all 50 states...")
    print(f"{'State':<6} {'Ent':<8} {'Food':<8} {'Lodge':<8} {'Trans':<8} | {'TOTAL':<8}")
    print("-" * 55)

    for code, coords in STATE_GRID.items():
        try:
            # 1. Entertainment: Use city="" to trigger the internal State Average fallback.
            ent_cost = ent_agent.predict_cost(city="", state=code, travelers=TRAVELERS, days=DAYS)

            # 2. Food: Use the demographics dictionary as required by the updated agent.
            demo = {"males": TRAVELERS, "females": 0, "children": 0}
            food_res = food_agent.predict_cost(location=code, days=DAYS, demographics=demo)
            food_cost = food_res.get('total_cost', 0)

            # 3. Lodging: Pass state abbreviation.
            lodging_res = lodging_agent.predict_cost(state=code, travelers=TRAVELERS, nights=DAYS)
            lodging_cost = lodging_res.get('total_cost', 0)

            # 4. Transport: Update to use 'state' and 'travel_mode' (1 = Moderate).
            trans_cost = trans_agent.predict_cost(state=code, days=DAYS, travelers=TRAVELERS, travel_mode=1)

            total_daily = ent_cost + food_cost + lodging_cost + trans_cost

            print(f"{code:<6} ${ent_cost:<7.0f} ${food_cost:<7.0f} ${lodging_cost:<7.0f} ${trans_cost:<7.0f} | ${total_daily:<7.0f}")

            results.append({
                'state_code': code,
                'cost': int(total_daily)
            })

        except Exception as e:
            logger.warning(f"Could not calculate for {code}: {e}")
            results.append({'state_code': code, 'cost': 0})

    return pd.DataFrame(results)

def generate_heatmap(data):
    if data.empty:
        logger.error("No data to plot.")
        return

    # Visual refinement: Ensure the directory exists
    os.makedirs("visualizations", exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    norm = plt.Normalize(vmin=data['cost'].min(), vmax=data['cost'].max())

    for _, row in data.iterrows():
        code = row['state_code']
        cost = row['cost']
        if code not in STATE_GRID: continue

        col, grid_row = STATE_GRID[code]
        color = cmap(norm(cost))

        rect = patches.Rectangle((col, grid_row), 0.9, 0.9, linewidth=1, edgecolor='white', facecolor=color)
        ax.add_patch(rect)

        ax.text(col + 0.45, grid_row + 0.6, code, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax.text(col + 0.45, grid_row + 0.3, f"${cost}", ha='center', va='center', fontsize=9, color='white')

    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(0.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.title("USA Travel Cost Index\n(Average Daily Cost per Person - All Categories)", fontsize=18, fontweight='bold', loc='left')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.65, 0.05, 0.25, 0.03])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Cost ($)', labelpad=5)

    save_path = "visualizations/usa_travel_cost_index.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Graph saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    df_costs = get_aggregated_costs()
    if not df_costs.empty:
        generate_heatmap(df_costs)