import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

# For reproducible color choice
random.seed(0)

# -----------------------------------------------------------
# 1. Get the data for a random adset
# -----------------------------------------------------------

def plot_campaign(current_dp):
    current_dp.date = current_dp.date.astype(str)
    # -----------------------------------------------------------
    # 2. Prepare placement information and color map
    # -----------------------------------------------------------
    # List of all possible placements
    unique_positions = ['feed_facebook_status', 'instant_article_facebook_status', 'facebook_stories_facebook_status', 'marketplace_facebook_status', 'right_hand_column_facebook_status']

    # Create a color map for each possible placement
    color_map = {pos: random.choice(list(mcolors.CSS4_COLORS.keys())) for pos in unique_positions}

    # -----------------------------------------------------------
    # 3. Plot main data
    # -----------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Year extraction (assuming date has format YYYY-MM-DD, etc.)
    year = current_dp.date.iloc[0].split("-")[0]

    # Primary axis: Total impressions and target impressions
    ax1.plot(current_dp.date, current_dp.total_impressions, label="Total Impressions", color='blue')
    ax1.plot(current_dp.date, current_dp.target_total_impressions, label="Target Total Impressions", color='orange')
    ax1.axhline(y=current_dp.impression_goal.iloc[0], color='red', linestyle='--', label="Campaign Target Value")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Impressions")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # -----------------------------------------------------------
    # 4. Secondary axis: Set CPM
    # -----------------------------------------------------------
    ax2 = ax1.twinx()
    ax2.plot(current_dp.date, current_dp.set_cpm, label="Set CPM (€)", color='green')
    ax2.set_ylabel("Set CPM (€)")
    ax2.legend(loc="upper right")

    # -----------------------------------------------------------
    # 5. Rotate x-axis labels for readability
    # -----------------------------------------------------------
    ax1.set_xticks(current_dp.date)
    ax1.set_xticklabels(current_dp.date, rotation=90)

    # -----------------------------------------------------------
    # 6. Mark the ACTIVE placements with colored dots
    #    (Invert the logic: show where the ad was active)
    # -----------------------------------------------------------
    # For each row, we subtract the inactive placements from the total set of placements
    for idx, row in current_dp.iterrows():
        # A small vertical offset so multiple placements can be plotted above each other
        y_offset = 0.05 * max(current_dp.total_impressions)  
        i = 1

        # Determine active placements
        # row.inactive_placement is a list of placements that are INACTIVE for this row
        # So, we get the set of ACTIVE placements by taking all placements minus the inactive ones.
        active_placements = row[unique_positions][row[unique_positions] == "Active"].index.tolist()

        for placement in active_placements:
            ax1.scatter(
                row.date,
                row.total_impressions + y_offset * i,
                color=color_map[placement],
                alpha=0.7,
                # Show a label only if it's not already in the legend
                label=placement if placement not in ax1.get_legend_handles_labels()[1] else None
            )
            i += 1

    # -----------------------------------------------------------
    # 7. Combine legends and place them outside the plot
    # -----------------------------------------------------------
    handles, labels = ax1.get_legend_handles_labels()

    ax1.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.1, 0.5),
        title="Legend"
    )

    # -----------------------------------------------------------
    # 8. Finalize and show
    # -----------------------------------------------------------
    plt.title(f"Total Impressions, Target Impressions, and Set CPM - {year}")
    plt.tight_layout()
    plt.show()
