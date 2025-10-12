"""
Utility functions (classes) used by other scripts.

This module provides:
- plot_results: for generating 2D plots from the provided data.
"""

import matplotlib.pyplot as plt

def plot_results(data, labels, scales={}):
    """
    Plot the results.

    Parameters
    ----------
    data : list of dict
        Each dta dictionary contains the keys:
        'x', 'y', 'label', 'linestyle', 'color'.
    labels : dict
        Dict of label strings for x and y axis with keys: 'x', 'y'.
    scales : dict, default {}
        Dict of scale strings for x and y axis with keys: 'x', 'y'.

    Returns
    -------
    plt.Axes
        The axes object the results have been plotted to.
    """
    fig, ax = plt.subplots(figsize=(11,6))

    for d in data:
        ax.plot(
            d['x'],
            d['y'],
            label=d['label'],
            linestyle=d['linestyle'],
            color=d['color']
        )
    ax.set_xlabel(labels['x'])
    ax.set_ylabel(labels['y'])
    if 'x' in scales: ax.set_xscale(scales['x'])
    if 'y' in scales: ax.set_yscale(scales['y'])
    ax.legend()

    plt.show()

    return ax
