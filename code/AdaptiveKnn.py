from RamanModel import *
from HelperFunctions import *
from Plots import *
from lmfit import Model, Parameters
from numba import njit

@njit(fastmath=True)
def add_lorentzian(curve, x, x0, gamma, maximum):
    gamma_sq = gamma * gamma

    for i in range(len(x)):
        dx = x[i] - x0
        curve[i] += (maximum * gamma_sq) / (gamma_sq + dx * dx)

@njit(fastmath=True)
def normalize_curve(curve):
    top = np.max(curve)
    if top > 0:
        curve /= top

@njit(fastmath=True)
def preallocate_centers(array, samples, num_peaks, probs, x_range):
    for sample_idx in range(samples):
        for peak_idx in range(num_peaks):
            array[sample_idx, peak_idx] = position_probability_map(probs, x_range)

def knnFit_persistent(samples, base_parameters, target_x, target_y, num_peaks=4, k=10, rounds=3, eps=0.15):
    peak_search_spaces = []
    
    for _ in range(num_peaks):
        peak_search_spaces.append({
            "center_range": list(base_parameters["center_range"]),
            "amplitude_range": list(base_parameters["amplitude_range"]),
            "gamma_range": list(base_parameters["gamma_range"])
        })

    # MOVE HEAP OUTSIDE THE ROUNDS LOOP
    heap = []
    best_neighbor_params = []

    # DEFINE X RANGE FOR PROB MAP SAMPLING
    x_range = np.linspace(0, 1, len(target_y))

    # CALCULATE PROBABILITY MAP FOR FIRST PASS
    probs = create_probability_map(target_y, num_peaks=num_peaks, resolution=50)
    first_round_centers = np.empty((samples, num_peaks))
    preallocate_centers(first_round_centers, samples, num_peaks, probs, x_range)
    

    #######################################################
    ##################### Nth PASS ########################
    #######################################################
    
    for r in range(rounds):
        current_y = np.empty_like(target_x)
        
        # INITIALIZE PARAMETERS

        centers = np.empty((samples, num_peaks))
        amps    = np.empty((samples, num_peaks))
        gammas  = np.empty((samples, num_peaks))

        for i in range(num_peaks):
            p = peak_search_spaces[i]
        
            centers[:, i] = np.random.uniform(
                p["center_range"][0],
                p["center_range"][1],
                size=samples
            )
        
            amps[:, i] = np.random.uniform(
                p["amplitude_range"][0],
                p["amplitude_range"][1],
                size=samples
            )
        
            gammas[:, i] = np.random.uniform(
                p["gamma_range"][0],
                p["gamma_range"][1],
                size=samples
            )

        # START LOOP
        for sample_idx in range(samples):
            current_y.fill(0.0)
            current_params = []

            for i in range(num_peaks):
                p = peak_search_spaces[i]

                if (r == 0):
                    c = first_round_centers[sample_idx, i]
                else:
                    c = centers[sample_idx, i]
                a = amps[sample_idx, i]
                g = gammas[sample_idx, i]

                add_lorentzian(current_y, target_x, c, g, a)
                current_params.append({"center": c, "amplitude": a, "gamma": g})

            """
            max_point = np.max(current_y)
            if max_point > 0:
                current_y /= max_point
            """
            normalize_curve(current_y)

            dist = euclidean_distance(target_y, current_y)

            if len(heap) < k:
                heapq.heappush(heap, (-dist, sample_idx, current_params))
            else:
                if -dist > heap[0][0]:
                    heapq.heapreplace(heap, (-dist, sample_idx, current_params))
                    
        # --- Clustering Peaks ---
        #nearest_neighbors = [item[1] for item in heap]
        nearest_neighbors = [item[2] for item in heap]

        all_candidate_peaks = []
        for neighbor_id, neighbor_params in enumerate(nearest_neighbors):
            for peak in neighbor_params:
                # Store the peak data AND which curve it belongs to
                all_candidate_peaks.append({
                    "data": peak,
                    "origin_curve": neighbor_id
                })

        visited = [False] * len(all_candidate_peaks)
        valid_clusters = []

        for i in range(len(all_candidate_peaks)):
            if visited[i]: continue

            cluster_indices = [i]
            # Tracking which curves have contributed to this cluster
            curves_in_cluster = {all_candidate_peaks[i]["origin_curve"]}

            for j in range(len(all_candidate_peaks)):
                if i == j: continue

                dist = get_param_distance(
                    all_candidate_peaks[i]["data"],
                    all_candidate_peaks[j]["data"],
                    base_parameters
                )

                if dist < eps:
                    cluster_indices.append(j)
                    curves_in_cluster.add(all_candidate_peaks[j]["origin_curve"])

            # CRITICAL CHECK: Does a majority of DIFFERENT curves agree?
            if len(curves_in_cluster) > (k // 2):
                valid_clusters.append(cluster_indices)
                # Mark peaks in this cluster as visited
                for idx in cluster_indices:
                    visited[idx] = True

        # Sort and update search spaces
        valid_clusters.sort(key=lambda c_idx: np.mean([all_candidate_peaks[idx]['data']['center'] for idx in c_idx]))

        for p_idx, cluster_indices in enumerate(valid_clusters[:num_peaks]):
            c_vals = [all_candidate_peaks[i]['data']['center'] for i in cluster_indices]
            a_vals = [all_candidate_peaks[i]['data']['amplitude'] for i in cluster_indices]
            g_vals = [all_candidate_peaks[i]['data']['gamma'] for i in cluster_indices]

            peak_search_spaces[p_idx]["center_range"] = [min(c_vals), max(c_vals)]
            peak_search_spaces[p_idx]["amplitude_range"] = [min(a_vals), max(a_vals)]
            peak_search_spaces[p_idx]["gamma_range"] = [min(g_vals), max(g_vals)]


        best_neighbor_params = max(heap, key=lambda x: x[0])[2]

    return nearest_neighbors


def composite_lorentzian_model(x, **params):
    """Sums multiple Lorentzians based on prefixes (p0_, p1_, etc.)"""
    y = np.zeros_like(x)
    prefixes = sorted(set(p.split('_')[0] for p in params.keys()))
    for pref in prefixes:
        y += lorentzian(x, params[f'{pref}_x0'], params[f'{pref}_gamma'], params[f'{pref}_maximum'])
    return y

def refine_knn_with_lmfit(neighbors, target_x, target_y):
    """
    Refines KNN neighbors using lmfit and returns the single best result 
    based on the lowest chi-square residual.
    """
    model = Model(composite_lorentzian_model)
    all_results = []
    
    # 1. Align neighbors by center to ensure consistency
    sorted_neighbors = [sorted(n, key=lambda p: p['center']) for n in neighbors]
    num_peaks = len(sorted_neighbors[0])
    
    # --- Step A: Refine every individual neighbor ---
    for peaks in sorted_neighbors:
        pars = Parameters()
        for i, p in enumerate(peaks):
            pref = f"p{i}_"
            pars.add(f"{pref}x0", value=p['center'], min=target_x.min(), max=target_x.max())
            pars.add(f"{pref}gamma", value=p['gamma'], min=1e-5)
            pars.add(f"{pref}maximum", value=p['amplitude'], min=0)
            
        res = model.fit(target_y, pars, x=target_x)
        all_results.append(res)

    # --- Step B: Refine the mean parameters ---
    mean_pars = Parameters()
    for i in range(num_peaks):
        pref = f"p{i}_"
        avg_x0 = np.mean([n[i]['center'] for n in sorted_neighbors])
        avg_gamma = np.mean([n[i]['gamma'] for n in sorted_neighbors])
        avg_max = np.mean([n[i]['amplitude'] for n in sorted_neighbors])
        
        mean_pars.add(f"{pref}x0", value=avg_x0, min=target_x.min(), max=target_x.max())
        mean_pars.add(f"{pref}gamma", value=avg_gamma, min=1e-5)
        mean_pars.add(f"{pref}maximum", value=avg_max, min=0)

    mean_res = model.fit(target_y, mean_pars, x=target_x)
    all_results.append(mean_res)

    # --- Step C: Select the best fit (smallest chi-square) ---
    # chisqr is the sum of squared residuals
    best_result = min(all_results, key=lambda r: r.chisqr)
    
    print(f"Best fit found with Chi-Square: {best_result.chisqr:.6f}")
    return best_result