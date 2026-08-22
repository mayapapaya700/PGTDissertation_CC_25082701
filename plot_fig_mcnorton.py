import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap

# Set font properties
font_path = "XXX.ttf"
font_bold='XXX.ttf'

custom_font = FontProperties(fname=font_path, size=24)
custom_font_bold = FontProperties(fname=font_bold, size=24, weight='bold')
custom_fontx = FontProperties(fname=font_bold, size=14, weight='bold')
custom_fonty = FontProperties(fname=font_path, size=14, weight='bold')

# File path
file_path = "ALL_REGION_DATA.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Ensure Lag (months) is sorted in descending order
df["Lag (months)"] = df["Lag (months)"].astype(int)

# Define the variable order and their display names
variable_order = ["SPEI3", "VPD", "SM1", "LFMC", "DFMC_Foliage", "DFMC_Wood", "EVI", "LAI"]
variable_labels = {
    "SPEI3": "SPEI",
    "VPD": "Vapour Pressure Def.*",
    "SM1": "Soil Moisture",
    "LFMC": "Live Fuel Moisture",
    "DFMC_Foliage": "Dead Foliage Moisture",
    "DFMC_Wood": "Dead Wood Moisture",
    "EVI": "Enhanced Veg. Index",
    "LAI": "LAI"
}

# Flip the sign of VPD values
df.loc[df["Variable"] == "VPD", "Lagged Correlation"] *= -1

# Replace variable names with labels
df["Variable"] = df["Variable"].map(variable_labels)

# Ensure the order of the variables
df["Variable"] = pd.Categorical(df["Variable"], categories=[variable_labels[v] for v in variable_order], ordered=True)

# Get unique regions (excluding the Marine West Coast Forest plot)
regions = [region for region in df["Region"].unique() if region != "Marine West Coast Forest"]

# Set up subplots
fig, axes = plt.subplots(len(regions), 1, figsize=(10, 10), gridspec_kw={'hspace': 0.15})

# Define colormap limits
vmin, vmax = -0.15, 0.15

# Create a custom color palette
colors = ['#8a3b00', '#e57100', '#ffb271', '#ffd8b8', '#ffffff', '#b6e0db', '#6bbbaf', '#008786', '#004c4b']
custom_cmap = LinearSegmentedColormap.from_list("custom_palette", colors)

# Create a shared colorbar axis
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])

# Iterate over each region for heatmap generation
for i, (region, ax) in enumerate(zip(regions, axes)):
    region_df = df[df["Region"] == region]
    heatmap_data = region_df.pivot(index="Variable", columns="Lag (months)", values="Lagged Correlation")
    heatmap_data = heatmap_data.sort_index(axis=1, ascending=False)
    
    # Create heatmap
    sns.heatmap(heatmap_data, cmap=custom_cmap, ax=ax, center=0, vmin=vmin, vmax=vmax,
                cbar=False, square=True, linewidths=0.5, linecolor='gray')
    
    # Highlight significant correlations (p-value < 0.05)
    for _, row in region_df.iterrows():
        if row['p-value'] < 0.1 and row["Lag (months)"] <= 20:
            col_idx = heatmap_data.columns.get_loc(row["Lag (months)"])
            row_idx = heatmap_data.index.get_loc(row["Variable"])
            ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='black', lw=2))
    
    ax.set_title(f"{region}", fontsize=18, pad=8, fontproperties=custom_font_bold)
    if i < len(regions) - 1:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Lag (Months)", fontsize=16, fontproperties=custom_font)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_xticklabels([30, 25, 20, 15, 10, 5, 0], fontsize=12, fontproperties=custom_fontx)
    
    ax.set_ylabel("", fontsize=14, fontproperties=custom_font)
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)

# Add a single colorbar
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, label="Lagged Correlation")
cbar.ax.tick_params(labelsize=16)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(custom_fontx)
cbar.set_label("Lagged Correlation", fontsize=14, fontproperties=custom_fontx)
