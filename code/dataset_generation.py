# CODE TO GENERATE A SYNTHETIC RAMAN DATASET FOR NEURAL NETWORK TRAINING

import numpy as np
import h5py
import json
import itertools
from RamanModel import generateCurve
from HelperFunctions import downsample_and_normalize

def generate_raman_dataset(n, peak_counts, noise_sigmas, parameters, resolution, filename="raman_training.h5"):
    """
    Generates a balanced dataset of Raman curves and saves them to an HDF5 file.
    
    Args:
        n (int): Total number of curves to generate.
        peak_counts (list): List of possible number of peaks (e.g., [1, 2, 3]).
        noise_sigmas (list): List of noise standard deviations (e.g., [0.01, 0.05]).
        parameters (dict): The range parameters for centers, amplitudes, and gammas.
        resolution (int): The number of points in each curve.
        filename (str): The name of the output HDF5 file.
    """
    # 1. Calculate combinations and distribution for balance
    combinations = list(itertools.product(peak_counts, noise_sigmas))
    num_combinations = len(combinations)
    
    # Calculate samples per combination
    samples_per_combo = n // num_combinations
    remainder = n % num_combinations
    
    # Distribute the total n samples (handling the remainder if n is not divisible)
    samples_distribution = [samples_per_combo + (1 if i < remainder else 0) for i in range(num_combinations)]
    
    # 2. Data Containers
    all_y = []
    all_peak_params = []
    all_num_peaks = []
    x_values_shared = None

    print(f"Generating balanced dataset of {n} curves...")
    current_idx = 0
    # 3. Generate Curves
    for (num_peaks, sigma), count in zip(combinations, samples_distribution):
        for _ in range(count):
            print(f"Generating curve {current_idx} of {n} (Peaks: {num_peaks}, Sigma: {sigma})...", end="\r", flush=True)
            current_idx += 1

            x, y, params = generateCurve(parameters, num_peaks, resolution, sigma)
            y = downsample_and_normalize(y, resolution) # Normalization only
            
            # Save x values only once
            if x_values_shared is None:
                x_values_shared = x
            
            all_y.append(y)
            all_num_peaks.append(num_peaks)
            # Parameters are saved as JSON strings to maintain structure in HDF5
            all_peak_params.append(json.dumps(params))

    # Convert to numpy arrays
    all_y = np.array(all_y)
    all_num_peaks = np.array(all_num_peaks)

    # 4. Save to HDF5
    with h5py.File(filename, 'w') as hf:
        # Save shared x values
        hf.create_dataset("x_values", data=x_values_shared)
        
        # Save intensities (y values)
        hf.create_dataset("y_values", data=all_y)
        
        # Save number of peaks per curve
        hf.create_dataset("num_peaks", data=all_num_peaks)
        
        # Save peak parameters as variable-length strings (JSON formatted)
        dt = h5py.special_dtype(vlen=str)
        ds_params = hf.create_dataset("peak_parameters", (n,), dtype=dt)
        ds_params[:] = all_peak_params

    print(f"Successfully saved dataset to '{filename}'.")

# EXECUTION BLOCK TO GENERATE TRAINING AND TEST DATASETS

peak_counts_list = [1, 2, 3, 4, 5]
noise_list = np.linspace(0.01, 0.5, num=50)
raman_params = {
    "center_range": (0.1, 0.9),
    "amplitude_range": (0.1, 1.0),
    "gamma_range": (0.01, 0.05),
    "x_min": 0,
    "x_max": 1
 }
 
generate_raman_dataset(n=20000, 
                        peak_counts=peak_counts_list, 
                        noise_sigmas=noise_list, 
                        parameters=raman_params, 
                        resolution=1000,
                        filename="training_data/raman_training.h5"
                        )

generate_raman_dataset(n=2000, 
                        peak_counts=peak_counts_list, 
                        noise_sigmas=noise_list, 
                        parameters=raman_params, 
                        resolution=1000,
                        filename="training_data/raman_test.h5"
                        )