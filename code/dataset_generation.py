from RamanModel import *
from HelperFunctions import *
import numpy as np
import h5py

def generate_dataset(num_curves, parameters, resolution, noise_range, num_peaks_range, filename="dataset.h5"):

    noise_range = np.linspace(noise_range[0], noise_range[1], 100)
    num_peaks_range = list(range(num_peaks_range[0], num_peaks_range[1]))

    samples = num_curves // (len(noise_range) * len(num_peaks_range))
    total_samples = samples * len(noise_range) * len(num_peaks_range)

    # Preallocate datasets
    spectra = np.zeros((total_samples, 100))
    peak_counts = np.zeros(total_samples, dtype=np.int32)
    noise_levels = np.zeros(total_samples, dtype=np.float32)

    with h5py.File(filename, "w") as f:
        spectra_ds = f.create_dataset("spectra", data=spectra)
        peaks_ds = f.create_dataset("num_peaks", data=peak_counts)
        noise_ds = f.create_dataset("noise", data=noise_levels)

        params_group = f.create_group("parameters")

        idx = 0

        for num_peaks in num_peaks_range:
            for noise in noise_range:
                for _ in range(samples):

                    _, y, params = generateCurve(parameters, num_peaks, resolution, noise)
                    y_ds = downsample_and_normalize(y)

                    # Store main data
                    spectra_ds[idx] = y_ds
                    peaks_ds[idx] = num_peaks
                    noise_ds[idx] = noise

                    # Store detailed parameters (optional, but useful)
                    sample_group = params_group.create_group(f"sample_{idx}")

                    for peak in params:
                        peak_id = peak["peak_id"]
                        peak_group = sample_group.create_group(f"peak_{peak_id}")

                        peak_group.attrs["center"] = peak["center"]
                        peak_group.attrs["amplitude"] = peak["amplitude"]
                        peak_group.attrs["gamma"] = peak["gamma"]

                    idx += 1

    print(f"Dataset saved to {filename} with {total_samples} samples.")

parameters = {
    "center_range": (0.1, 0.9),
    "amplitude_range": (0.1, 1),
    "gamma_range": (0.001, 0.05),
    "x_min": 0,
    "x_max": 1
}
generate_dataset(num_curves=10000, parameters=parameters, resolution=1876, noise_range=(0.01, 0.6), num_peaks_range=(1, 5), filename="dataset.h5")