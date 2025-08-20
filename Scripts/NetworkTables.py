from pathlib import Path
import igraph as ig
from tqdm.auto import tqdm
import xnetwork as xn
import pandas as pd
import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib

# enabling illustrator font editing
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

networkFolder = Path("Networks")
dataFolder = Path("Data")

networks = {
    "CRS0": "expressionAndMicrobiomeCorrelation_CRC0UMAP_Pearson_CLR_Disparity_T0.05",
    "CRS1": "expressionAndMicrobiomeCorrelation_CRC1UMAP_Pearson_CLR_Disparity_T0.05"
}

# Load the two networks and compute the degree if not already present
g1 = xn.load(dataFolder / f"{networks['CRS0']}_attributes.xnet")
g2 = xn.load(dataFolder / f"{networks['CRS1']}_attributes.xnet")
g1.vs["Degree"] = g1.degree()
g2.vs["Degree"] = g2.degree()

# Parameters
attributes = [
    "Degree",
    "MEDegrees",
    "BetweenessCentrality",
    "BetweenessInCommunity",
    "Coreness",
]
topN = 20

def create_rank_plot(g1, g2, type_filter, attribute, topN=10, ax=None, left_title="CRS0", right_title="CRS1"):
    """
    Creates a rank plot on the given axis `ax` for vertices of type `type_filter`
    (e.g., "E" for genes or "M" for bugs). It selects the topN vertices sorted by the given
    attribute from both networks, places the labels on the left (from g1) and right (from g2),
    and draws a connecting line between the labels if the same vertex appears in both.
    
    New modifications:
    - Each label is prefixed with its ranking position (e.g. "#1.").
    - An extra entry is added at the end (with ranking position topN+1) that shows "< topN"
      and is separated by an ellipsis ("...") from the top entries.
    """
    # Filter vertices by type and sort by attribute (descending)
    verts1 = list(g1.vs.select(Type_eq=type_filter))
    verts2 = list(g2.vs.select(Type_eq=type_filter))

    # filter nans
    verts1 = [v for v in verts1 if not np.isnan(v[attribute])]
    verts2 = [v for v in verts2 if not np.isnan(v[attribute])]


    totalMoreThanTopN1 = len(verts1) > topN
    totalMoreThanTopN2 = len(verts2) > topN

    verts1_sorted = sorted(verts1, key=lambda v: v[attribute], reverse=True)[:topN]
    verts2_sorted = sorted(verts2, key=lambda v: v[attribute], reverse=True)[:topN]
    
    # Extract names (assuming each vertex has a "Label" attribute)
    names1 = [v["Label"] for v in verts1_sorted]
    names2 = [v["Label"] for v in verts2_sorted]

    # for all names replace _ by space
    names1 = [name.replace("_", " ") for name in names1]
    names2 = [name.replace("_", " ") for name in names2]

    values1 = [v[attribute] for v in verts1_sorted]
    values2 = [v[attribute] for v in verts2_sorted]


    # colormap = plt.cm.get_cmap("CET_CBL2")
    colormap = cc.m_CET_CBL2_r
    
    # Equally spaced y positions (highest rank at the top)
    y_positions1 = np.linspace(1, 0, topN)
    y_positions2 = np.linspace(1, 0, topN)
    
    # Create mappings from name to y position (for connecting lines)
    mapping1 = {name: y for name, y in zip(names1, y_positions1)}
    mapping2 = {name: y for name, y in zip(names2, y_positions2)}

    allValues = values1 + values2
    minVal = min(allValues)
    maxVal = max(allValues)
    
    # # Retrieve community attribute if available; otherwise use None.
    # communities1 = [v["community"] if "community" in v.attribute_names() else None for v in verts1_sorted]
    # communities2 = [v["community"] if "community" in v.attribute_names() else None for v in verts2_sorted]
    
    # Create a color mapping for communities (using tab10 colormap)
    # cmap = plt.get_cmap("tab10")
    # unique_comms1 = sorted(set(c for c in communities1 if c is not None))
    # comm_color_dict1 = {comm: cmap(i % 10) for i, comm in enumerate(unique_comms1)}
    
    # unique_comms2 = sorted(set(c for c in communities2 if c is not None))
    # comm_color_dict2 = {comm: cmap(i % 10) for i, comm in enumerate(unique_comms2)}
    
    # create colors for all the values

    # Plot left-side labels (from g1) with ranking positions
    squareSize = 0.025
    halfSquare = squareSize / 2


    for i, (name, y, value) in enumerate(zip(names1, y_positions1, values1)):
        normValue = colormap((value - minVal) / (maxVal - minVal))
        # small square with the color centered at the endingpoint
        ax.text(0-squareSize, y, f" {name[2:]}", ha="right", va="center", fontsize=10)
        ax.add_patch(plt.Rectangle((0.0-halfSquare, y-halfSquare), squareSize, squareSize, facecolor=normValue, edgecolor="gray"))
    
    # Plot right-side labels (from g2) with ranking positions
    for i, (name, y, value) in enumerate(zip(names2, y_positions2, values2)):
        normValue = colormap((value - minVal) / (maxVal - minVal))
        # small square with the color centered at the startingpoint
        ax.text(1+squareSize, y, f"{name[2:]}", ha="left", va="center", fontsize=10)
        ax.add_patch(plt.Rectangle((1-halfSquare, y-halfSquare), squareSize, squareSize, facecolor=normValue, edgecolor="gray"))
    
    # Draw connecting lines for nodes that appear in both lists
    common_names = set(names1) & set(names2)
    for name in common_names:
        y1 = mapping1[name]
        y2 = mapping2[name]
        ax.plot([0, 1], [y1, y2], color="gray", lw=1, zorder=0)

    # Add extra entry at the end with ellipsis separation.
    # Compute y positions for ellipsis and extra entry (placed below the last top entry)
    ellipsis_y = y_positions1[-1] - 0.05  # Position for ellipsis; adjust as needed
    extra_y = y_positions1[-1] - 0.1       # Position for the extra label


    if(totalMoreThanTopN1):
    # Draw lines for nodes that are in one list but not the other to the last entry
        for name in names1:
            if name not in common_names:
                y1 = mapping1[name]
                ax.plot([0, 1], [y1, extra_y], color="gray", lw=1, zorder=0)

    if(totalMoreThanTopN2):
        # and for the other way around
        for name in names2:
            if name not in common_names:
                y2 = mapping2[name]
                ax.plot([1, 0], [y2, extra_y], color="gray", lw=1, zorder=0)

    
    # Left column extra entry
    if(totalMoreThanTopN1):
        ax.text(0, ellipsis_y, '...', ha="right", va="center", color="black", fontsize=10)
        ax.text(0, extra_y, f"> #{topN+1}", ha="right", va="center", color="black", fontsize=10)
    

    # Right column extra entry
    if(totalMoreThanTopN2):

        ax.text(1, ellipsis_y, '...', ha="left", va="center", color="black", fontsize=10)
        ax.text(1, extra_y, f"> #{topN+1}", ha="left", va="center", color="black", fontsize=10)
    


    # Add column titles
    ax.text(0, 1.05, left_title, ha="center", va="bottom", fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(1, 1.05, right_title, ha="center", va="bottom", fontsize=12, fontweight="bold", transform=ax.transAxes)
    
    # add a horizontal colorbar on the bottom
    norm = matplotlib.colors.Normalize(vmin=minVal, vmax=maxVal)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation="horizontal", pad=0.05)
    # cbar.set_label(attribute, fontsize=12)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")



    # Adjust limits to accommodate the extra entry
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(extra_y - 0.1, 1.1)
    ax.axis("off")
    


    return mapping1, mapping2


topMicrobiome = set()
for attribute in attributes:
    # -----------------------------
    # Plot for Genes (Type "E")
    # -----------------------------
    fig_genes, ax_genes = plt.subplots(figsize=(6, 6))
    genes_results = create_rank_plot(
        g1, g2, type_filter="E", attribute=attribute, topN=topN,
        ax=ax_genes, left_title="CRS0", right_title="CRS1"
    )
    ax_genes.set_title(f"{attribute} ranking (Genes)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"Figures/Top_{topN}_Genes_{attribute}.pdf", bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Plot for Bugs (Type "M")
    # -----------------------------
    fig_bugs, ax_bugs = plt.subplots(figsize=(8, 6))
    microbiome_results = create_rank_plot(
        g1, g2, type_filter="M", attribute=attribute, topN=topN,
        ax=ax_bugs, left_title="CRS0", right_title="CRS1"
    )
    ax_bugs.set_title(f"{attribute} ranking (Microbiome)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"Figures/Top_{topN}_Bugs_{attribute}.pdf", bbox_inches="tight")
    plt.close()

    # add top 10 microbiome to the set
    topMicrobiome.update(list(microbiome_results[0].keys())[:5])
    topMicrobiome.update(list(microbiome_results[1].keys())[:5])


