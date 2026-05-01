import numpy as np
import heapq

def euclidean_distance(y1, y2):
    return np.sqrt(np.sum((y1 - y2)**2))

def get_param_distance(p1, p2, ranges):
    """
    Calculates normalized Euclidean distance between two peaks.
    Normalization ensures that a small change in Gamma is as significant
    as a large change in Center.
    """
    # Use the initial broad ranges for normalization consistency
    d_c = (p1['center'] - p2['center']) / (ranges['center_range'][1] - ranges['center_range'][0])
    d_a = (p1['amplitude'] - p2['amplitude']) / (ranges['amplitude_range'][1] - ranges['amplitude_range'][0])
    d_g = (p1['gamma'] - p2['gamma']) / (ranges['gamma_range'][1] - ranges['gamma_range'][0])
    return np.sqrt(d_c**2 + d_a**2 + d_g**2)

def print_peak_params(params):
    """Prints peak parameters in a clean, tabular format."""
    # Define header and column widths
    header = f"{'Peak ID':<10} | {'Center':<12} | {'Amplitude':<12} | {'Gamma':<12}"
    separator = "-" * len(header)

    print("\nGenerated Curve Parameters:")
    print(separator)
    print(header)
    print(separator)

    for p in params:
        print(f"{p['peak_id']:<10} | {p['center']:<12.4f} | {p['amplitude']:<12.4f} | {p['gamma']:<12.4f}")

    print(separator + "\n")