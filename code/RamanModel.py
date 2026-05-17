import numpy as np

def lorentzian(x, x0, gamma, maximum):
    return maximum / (1 + ((x - x0) / gamma)**2)

def addWhiteNoiseRandom(xData, yData, factor):
    y_noisy = yData + np.random.normal(0, factor, len(xData))
    return y_noisy

def addNoiseSNR(yData, SNR):
    signal_power = np.sum(yData**2) / len(yData)
    noise_power = np.sqrt(signal_power / SNR)
    y_noisy = yData + np.random.normal(0, noise_power, len(yData))
    return y_noisy, noise_power

def generateCurve(parameters, number_of_peaks, resolution, noise_sigma):
    center_params = parameters["center_range"]
    amplitude_params = parameters["amplitude_range"]
    gamma_params = parameters["gamma_range"]
    x_min = parameters["x_min"]
    x_max = parameters["x_max"]

    y = np.zeros(resolution)
    x = np.linspace(x_min, x_max, resolution)
    params = []

    for i in range(number_of_peaks):
        center = np.random.uniform(center_params[0], center_params[1])
        amplitude = np.random.uniform(amplitude_params[0], amplitude_params[1])
        gamma = np.random.uniform(gamma_params[0], gamma_params[1])

        y += lorentzian(x, center, gamma, amplitude)

        # Append individual peak params
        params.append({
            "peak_id": i,
            "center": center,
            "amplitude": amplitude,
            "gamma": gamma
        })

    # Normalization
    #max_y = np.max(y)
    #if max_y > 1:
    #    y /= max_y

    # Add noise
    if noise_sigma > 0:
        y += addWhiteNoiseRandom(x, y, noise_sigma)

    return x, y, params

def print_params(params, precision=4):
    if not params:
        print("No parameters to display.")
        return

    # Header
    print(f"{'Peak':<6} {'Center':<12} {'Amplitude':<12} {'Gamma':<12}")
    print("-" * 44)

    # Rows
    for p in params:
        print(
            f"{p['peak_id']:<6} "
            f"{p['center']:<12.{precision}f} "
            f"{p['amplitude']:<12.{precision}f} "
            f"{p['gamma']:<12.{precision}f}"
        )