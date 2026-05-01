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

def simulated_annealing(spectrum, kmax, max_temp, cooling_factor, samples, num_repetitions):
    map = []

    best_state = None
    best_energy = -np.inf

    for i in range(num_repetitions):

        s_current = np.random.randint(0, len(spectrum)-1)
        energy_current = spectrum[s_current]
        T = max_temp

        for j in range(kmax):
            #T = max_temp * (1 - (j + 1) / kmax)
            T = T/((cooling_factor)**(1/kmax))
            #if T < 1e-12: break

            for k in range(samples):
                s_new = np.random.randint(0, len(spectrum)-1)
                energy_new = spectrum[s_new]

                diff = energy_new - energy_current
                if diff > 0 or np.random.random() < np.exp(diff / max(T, 1e-12)):
                    s_current = s_new
                    energy_current = energy_new

                    if energy_current > best_energy:
                        best_energy = energy_current
                        best_state = s_current

            map.append((T, s_current))

    return best_energy, best_state, map

def log_transform(x):
    """Applies a logarithmic transformation to the input array."""
    return np.log(x + 1e-12)  # Adding a small constant to avoid log(0)

def flatten_SA_map(map, spectrum_length, samples_per_pixel, function=(lambda x: x)):
    """Flattens the SA map into a 2D array for easier plotting."""

    new_length = spectrum_length // samples_per_pixel
    flattened_map = np.zeros(new_length)

    for i in range(len(map)):
        T, s = map[i]
        flattened_map[s // samples_per_pixel] += 1  # Ensure s is within spectrum length

    flattened_map[flattened_map == 0] = 1
    
    # Normalize the flattened map for better visualization
    flattened_map = function(flattened_map)
    
    return flattened_map