""" Figures related to Ocean Science and Climate"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse


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
                   '100 km', '1000 km', '10,000 km']  # Corresponding labels
    ticks = np.arange(0, 11)  # Ticks from 0 to 10
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(0, 10)  # Set x-axis limits

    # Y-axis ticks are in time scale
    y_ticks = np.arange(9)
    ytick_labels = ['1 s', '1 min', '1 hour', '1 day', '1 week', '1 month', 
                    '1 year', '10 years', '100 years']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_ylim(0, 8)  # Set y-axis limits

    # Set axis labels and title
    ax.set_xlabel("Spatial Scale")
    ax.set_ylabel("Time Scale (log)")
    #ax.set_title("Logarithmic Spatial and Time Scales")

    # Add a legend
    #ax.legend()

    # Add ellipses for the physical processes
    ellipse_data = [
        {"center": (0.5, 0.7), "width": 2, "height": 1.5, "angle": 0, "color": 
            "lightblue", "text": "Molecular\nProcesses", "label": "Molecular"},
    ]

    # Add all ellipses and their text to the plot
    for data in ellipse_data:
        # Create and add the ellipse
        ellipse = Ellipse(data["center"], width=data["width"], 
                          height=data["height"], angle=data["angle"],
                        facecolor=data["color"], edgecolor=data['color'], alpha=0.7)
        ax.add_patch(ellipse)
        
        # Add text to the center of each ellipse
        ax.text(data["center"][0], data["center"][1], data["text"],
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8,
                fontweight='bold',
                color='black')

    # Save the figure
    plt.savefig("fig_stommel.png", dpi=300, bbox_inches='tight')
    print("Figure saved as fig_stommel.png")

# Command line
if __name__ == "__main__":
    fig_stommel()
    # Uncomment the following line to run the function
    # fig_stommel()