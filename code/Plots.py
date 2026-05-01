import numpy as np
import matplotlib.pyplot as plt
from RamanModel import *

def PlotNeighbors(best_params, neighbors_params, target_x, target_y):
    # --- 3. Plotting the Results ---
    plt.figure(figsize=(12, 7))

    # Plot the Target Curve (The one we are trying to match)
    plt.plot(target_x, target_y, color='black', linewidth=1, label='Target (Original)', zorder=5)

    # Plot each of the K-Nearest Neighbors found in the final round
    for i, neighbor in enumerate(neighbors_params):
        neighbor_y = np.zeros(len(target_x))
        for peak in neighbor:
            neighbor_y += lorentzian(target_x, peak['center'], peak['gamma'], peak['amplitude'])

        # Normalize neighbor to match target scale
        if np.max(neighbor_y) > 0:
            neighbor_y /= np.max(neighbor_y)

        plt.plot(target_x, neighbor_y, alpha=0.4, linestyle='--', label=f'Neighbor {i+1}' if i < 5 else "")

    plt.title(f"KNN Lorentzian Fitting: Target vs {len(neighbors_params)} Nearest Neighbors")
    plt.xlabel("X")
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    plt.show()