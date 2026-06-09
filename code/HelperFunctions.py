import numpy as np
import heapq
from numba import njit

@njit(fastmath=True)
def euclidean_distance(y1, y2):
    """
    Calculates the Euclidean distance between two points.
    """
    #return np.sqrt(np.sum((y1 - y2)**2))
    diff = y1 - y2
    return np.dot(diff, diff)

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
    """
    Creates a map of likely peak positions by recording posisitons and temperatures. This can be later
    flattened to produce a probability map. This function is likely deprecated in favor of less computationally
    expensive approaches.
    """

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
    print(f'Number of samples generated: {len(map)}')
    
    return best_energy, best_state, map

def log_transform(x):
    """Applies a logarithmic transformation to the input array."""
    return np.log(x + 1e-12)  # Adding a small constant to avoid log(0)

def flatten_SA_map(map, spectrum_length, samples_per_pixel, function=(lambda x: x)):
    """Flattens the SA map into a 2D array for easier plotting."""

    new_length = spectrum_length // samples_per_pixel
    flattened_map = np.zeros(new_length)
    max_temperature = max(map, key=lambda x: x[1])[1]

    for i in range(len(map)):
        T, s = map[i]
        flattened_map[s // samples_per_pixel] += 1/T

    flattened_map[flattened_map == 0] = np.median(flattened_map)
    
    # Normalize the flattened map for better visualization
    flattened_map = function(flattened_map)
    
    return flattened_map

def downsample_and_normalize(spectrum, resolution=100):
    """
    Downsamples and normalizes spectra into the standard 100 pixel resolution. 
    Downsampling helps reduce noise signatures.
    Additionally places the spectrum into the amplitude range of 0,1.
    """
    size = len(spectrum)
    
    factor = size/resolution
    output = np.zeros(resolution)

    accum = 0
    idx = 0
    
    for i in range(1, size + 1):
        
        accum += spectrum[i-1]
        
        if (i%(np.ceil(factor)) == 0):
            output[idx] = accum/(np.ceil(factor))
            accum = 0
            idx += 1
            
    output = output/max(output)
    return output

def create_probability_map2(spectrum, resolution = 50):
    """
    Generates the probability map for use in the KNN algorithm.
    It downsamples the spectrum and then performs area normalization.
    The return value is like a probability density funtion.
    """
    spectrum = downsample_and_normalize(spectrum, resolution)
    spectrum = list(map(lambda x: max(0, x), spectrum))
    spectrum = spectrum/sum(spectrum)

    return spectrum


def create_probability_map(spectrum, num_peaks, resolution=50):
    """
    Generates a probability map where no single bin exceeds 1/num_peaks.
    """
    spectrum = downsample_and_normalize(spectrum, resolution)
    prob_map = np.maximum(0, spectrum)
    
    if np.sum(prob_map) > 0:
        prob_map = prob_map / np.sum(prob_map)
    else:
        return prob_map

    ceiling = 1.0 / num_peaks
    
    for _ in range(resolution):
        over_limit_mask = prob_map > ceiling
        if not np.any(over_limit_mask):
            break
            
        # Calculate excess probability
        excess = np.sum(prob_map[over_limit_mask] - ceiling)
        
        # Cap the bins that were over the limit
        prob_map[over_limit_mask] = ceiling
        
        # Find bins that can still "absorb" probability (those under the limit)
        under_limit_mask = prob_map < ceiling
        
        if np.any(under_limit_mask):
            under_limit_sum = np.sum(prob_map[under_limit_mask])
            if under_limit_sum > 0:
                # Add excess weighted by the existing distribution
                prob_map[under_limit_mask] += (prob_map[under_limit_mask] / under_limit_sum) * excess
            else:
                # If everything else is 0, distribute uniformly among under-limit bins
                prob_map[under_limit_mask] += excess / np.sum(under_limit_mask)
        else:
            # Fallback: if all bins are capped, the map is uniform
            break

    #return prob_map
    return np.cumsum(prob_map)

#def position_probability_map(probability_map, initial_resolution = 1876):
@njit
def position_probability_map(cdf, x_range):
    """
    Outputs a position range given the probability of peak positions generated
    by the probability map.
    """
    #resolution = len(probability_map)
    resolution = len(cdf)
    
    #shift_per_pixel = 1/initial_resolution
    #scaling_factor = np.floor(initial_resolution/resolution)
    scaling_factor = len(x_range)//resolution
    
    #index = np.random.choice(resolution, p=probability_map)
    r = np.random.random()
    index = np.searchsorted(cdf, r)
    
    lower_bound = index * scaling_factor
    upper_bound = lower_bound + scaling_factor - 1

    #x_range = np.linspace(0, 1, initial_resolution)

    pixel = np.random.randint(lower_bound, upper_bound)

    #return (x_range[lower_bound], x_range[upper_bound])
    return x_range[pixel]


def shift_axis(spectrum, resolution=200):
    """
    Shifts the spectrum to the y-axis, such that the new curve represents the number of
    times a horizontal line intersects the original curve.
    """
    spec = np.asarray(spectrum)
    
    step = 1 / resolution
    start = step / 2
    y_axis_lines = np.arange(start, 1, step)
    counts = np.zeros(len(y_axis_lines))

    # We look at each adjacent pair of points in the spectrum
    # A horizontal line 'h' intersects the segment between spec[i] and spec[i+1]
    
    y_min = np.minimum(spec[:-1], spec[1:])
    y_max = np.maximum(spec[:-1], spec[1:])

    for i, h in enumerate(y_axis_lines):
        # A line at height 'h' intersects a segment if y_min < h <= y_max
        counts[i] = np.sum((y_min < h) & (y_max >= h))
        
    return counts

    
    
    
    