""" Figures related to Ocean Science and Climate"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse

from ocpy.utils import fig_utils


def fig_stommel():
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot data
    #ax.plot(x, y, label="Example Data")

    # Set logarithmic scales
    #ax.set_xscale('log')
    #ax.set_yscale('log')

    # Set x-axis range

    # Customize x-axis tick labels
    tick_labels = ['1 mm', '1 cm', '1 dm', '1 m', '10 m', '100 m', '1 km', '10 km',
                   r'$10^2$ km', r'$10^3$ km', r'$10^4$ km']  # Corresponding labels
    ticks = np.arange(0, 11)  # Ticks from 0 to 10
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-0.5, 11.0)  # Set x-axis limits

    # Y-axis ticks are in time scale
    y_ticks = np.arange(9)
    ytick_labels = ['1 s', '1 min', '1 hour', '1 day', '1 week', '1 month', 
                    '1 year', '10 years', '100 years']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_ylim(-0.5, 8)  # Set y-axis limits

    #ax.set_title("Logarithmic Spatial and Time Scales")

    # Add a legend
    #ax.legend()

    # Add ellipses for the physical processes
    ellipse_data = [
        # Physical processes
        {"center": (0.5, 0.7), "width": 2, "height": 1.5, "angle": 0, "color": 
            "lightblue", "text": "Molecular\nProcesses", "label": "Molecular",
            "label_off": (0.0, 0.0)},
        {"center": (2.7, 1.5), "width": 2, "height": 1.5, "angle": 0, "color": 
            "lightblue", "text": "Tubulent\nPatches", "label": "Molecular",
            "label_off": (0.0, -0.2)},
        {"center": (4.1, 1.7), "width": 2, "height": 1.5, "angle": 0, "color": 
            "lightblue", "text": "Langmuir\nCells", "label": "Molecular",
            "label_off": (0.0, 0.0)},
        {"center": (4.0, 1.3), "width": 3, "height": 1.5, "angle": 0, "color": 
            "lightblue", "text": "Surface\nWaves", "label": "Molecular",
            "label_off": (0.0, -0.5)},
        {"center": (5.1, 2.0), "width": 3, "height": 2.0, "angle": 0, "color": 
            "lightblue", "text": "Inertial\nWaves", "label": "Molecular",
            "label_off": (0.2, +0.1)},
        {"text": "Internal\nTides", "center": (7.1, 2.8), "width": 1, "height": 0.5, "angle": 0, 
            "color": "lightblue", "label": "Molecular", "label_off": (0.0, -0.1)},
        {"text": "Fronts\nEddies\nUpwelling", "center": (7.2, 5.0), 
         "width": 2.1, "height": 1.7, "angle": 0, "color": "lightblue", "label": "Molecular", "label_off": (0.0, +0.0)},
        {"text": "Coastal\nWaves", "center": (7.3, 4.1), 
         "width": 0.5, "height": 1.0, "angle": 0, "color": "lightblue", "label": "Molecular", "label_off": (0.0, +0.0)},
        {"text": "Surface\nTides", "center": (8.5, 2.7), 
         "width": 1.0, "height": 0.5, "angle": 0, "color": "lightblue", "label": "Molecular", "label_off": (0.0, +0.0)},
        {"text": "Mixed Layer\nDepth", "center": (8.5, 5.5), 
         "width": 1.0, "height": 1.0, "angle": 0, "color": "lightblue", 
         "label": "Molecular", "label_off": (0.0, -0.2)},
        {"text": "Rossby\nWaves", "center": (9.5, 6.0), 
         "width": 1.7, "height": 1.0, "angle": 0, "color": "lightblue", 
         "label": "Molecular", "label_off": (0.0, -0.0)},
        {"text": "ENSO", "center": (10.2, 6.5), 
         "width": 1.0, "height": 0.5, "angle": 0, "color": "lightblue", "label": "Molecular", "label_off": (0.0, +0.0)},
        # Biological processes
        {"center": (1.2, 1.5), "width": 2.5, "height": 1.5, "angle": 0, "color": 
            "pink", "text": "Individual\nMovement", "label": "Molecular",
            "label_off": (0.0, 0.0)},
        {"center": (5.0, 3.2), "width": 2.5, "height": 1.5, "angle": 0, "color": 
            "pink", "text": "Plankton\nMigration", "label": "Molecular",
            "label_off": (0.0, 0.0)},
        {"center": (5.5, 4.5), "width": 3.2, "height": 1.5, "angle": 0, "color": 
            "pink", "text": "Phytoplankton\nBlooms", "label": "Molecular",
            "label_off": (0.0, 0.0)},
        {"text": "Biomass\nCycles", "center": (8.0, 5.5), 
         "width": 1.3, "height": 1.0, "angle": 0, "color": "pink", 
         "label": "Molecular", "label_off": (-0.4, +0.5)},
        # Climate
        {"center": (10.2, 8.1), "width": 1.0, "height": 2.0, "angle": 0, "color": 
            "darkorchid", "text": "Climate", "label": "Molecular",
            "label_off": (0.0, -0.5)},
        {"center": (10.2, 7.0), "width": 1.0, "height": 0.3, "angle": 0, "color": 
            "darkorchid", "text": "Decadal\nOscillations", "label": "Molecular",
            "label_off": (0.0, 0.0)},
    ]

    # Add all ellipses and their text to the plot
    for data in ellipse_data:
        # Create and add the ellipse
        ellipse = Ellipse(data["center"], width=data["width"], 
                          height=data["height"], angle=data["angle"],
                        facecolor=data["color"], edgecolor=data['color'], alpha=0.7)
        ax.add_patch(ellipse)
        
        # Add text to the center of each ellipse
        #fclr = 'black' if data["color"] in ['lightblue','lightgreen'] else 'gray'
        fclr = 'black' 
        ax.text(data["center"][0]+data['label_off'][0],
                data["center"][1]+data['label_off'][1], 
                data["text"], 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8,
                fontweight='bold',
                color=fclr)

    # Add an arrow with two heads
    aclr = 'r'
    ax.annotate('', xy=(0.5, 0.5), xytext=(9.5, 7.5),
                arrowprops=dict(arrowstyle='<->', lw=2, color=aclr),
                fontsize=8)
    # Add text to the arrow, rotated
    ax.text(3.0, 2.8, "Cascade of energy", fontsize=10,
            horizontalalignment='center', verticalalignment='center',
            rotation=40, color=aclr, fontweight='bold')

    # Add an empty rectangle for Satellites
    sat_clr = 'gold'
    x_sat = (5.7, 10.8)
    y_sat = (3.0, 7.3)
    xy_sat = (x_sat[0], y_sat[0])
    dx_sat = np.abs(x_sat[1] - x_sat[0])
    dy_sat = np.abs(y_sat[1] - y_sat[0])
    rect = plt.Rectangle(xy_sat, dx_sat, dy_sat, angle=0.0,
                         facecolor='none', edgecolor=sat_clr, lw=2)
    ax.add_patch(rect)
    ax.text(8.5, 3.2, "Satellites", fontsize=10,
            horizontalalignment='left', verticalalignment='center',
            color=sat_clr, fontweight='bold')
    
    # Calypso Monterey Bay
    cmb_clr = 'green'
    x_cmb = (4.0, 8.0)
    y_cmb = (1.8, 6.5)
    xy_cmb = (x_cmb[0], y_cmb[0])
    dx_cmb = np.abs(x_cmb[1] - x_cmb[0])
    dy_cmb = np.abs(y_cmb[1] - y_cmb[0])
    rect = plt.Rectangle(xy_cmb, dx_cmb, dy_cmb, angle=0.0,
                         facecolor='none', edgecolor=cmb_clr, lw=2)
    ax.add_patch(rect)
    ax.text(2.0, 6.2, "Calypso\nMonterey\nBay", fontsize=12,
            horizontalalignment='left', verticalalignment='center',
            color=cmb_clr, fontweight='bold')

    # Axes fonts
    fig_utils.set_fontsize(ax, 8)
    #fig_utils.set_fontweight(ax, 'bold')

    # Set axis labels and title
    fsz = 12
    ax.set_xlabel("Horizontal Spatial Scales", fontsize=fsz)
    ax.set_ylabel("Time Scales", fontsize=fsz)

    # Save the figure
    plt.tight_layout()
    plt.savefig("fig_stommel.png", dpi=300, bbox_inches='tight')
    print("Figure saved as fig_stommel.png")

# Command line
if __name__ == "__main__":
    fig_stommel()
    # Uncomment the following line to run the function
    # fig_stommel()