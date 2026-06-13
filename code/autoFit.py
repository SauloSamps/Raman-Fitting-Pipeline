import os
import torch
import numpy as np
from multiprocessing import Process, Queue
from slmk import SLMK_CNN_PeakClassifier, predict_top_k_peaks
from AdaptiveKnn import knnFit_persistent, refine_knn_with_lmfit


def _fit_worker(num_peaks, samples, base_parameters, target_x, target_y, k, rounds, eps, output_queue):
    """
    Worker function executed in a separate process.
    Runs the adaptive KNN fit followed by the lmfit refinement for a specific peak count.
    """
    try:
        # 1. Run the persistent KNN fit to get neighboring candidate parameter sets
        neighbors = knnFit_persistent(
            samples=samples,
            base_parameters=base_parameters,
            target_x=target_x,
            target_y=target_y,
            num_peaks=num_peaks,
            k=k,
            rounds=rounds,
            eps=eps
        )
        
        # 2. Refine the parameters using non-linear least squares optimization
        best_fit_result = refine_knn_with_lmfit(neighbors, target_x, target_y)
        
        # 3. Extract unbinarized, serializable metrics to safely send across the process boundary
        worker_output = {
            "num_peaks": num_peaks,
            "chisqr": float(best_fit_result.chisqr),
            "best_values": best_fit_result.best_values,  # Dictionary of optimized parameters
            "bic": float(best_fit_result.bic),
            "fitted_values": best_fit_result.best_fit.tolist(),  # Best fit y-curve array
            "success": True
        }
    except Exception as e:
        worker_output = {
            "num_peaks": num_peaks,
            "success": False,
            "error": str(e)
        }
        
    output_queue.put(worker_output)

def autoFit(target_x, target_y, base_parameters, model_path="models/model_5_peaks.pth", num_classes=5, samples=1000, k=5, rounds=3, eps=0.15):
    """
    Main pipeline function:
    1. Runs the target spectrum through the CNN to identify the top 3 most likely peak counts.
    2. Spawns 3 concurrent background processes to perform fitting optimizations.
    3. Aggregates results back into a single process.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Check if the user passed a relative path, and make it absolute if so
    if not os.path.isabs(model_path):
        resolved_model_path = os.path.join(current_file_dir, model_path)
    else:
        resolved_model_path = model_path

    # Double check if the file is actually resolved correctly
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(f"Dynamic path resolution failed. Target file not found at: '{resolved_model_path}'")
    
    # --- Step 1: Neural Network Inference ---
    # Instantiate the multi-scale CNN classifier architecture
    model = SLMK_CNN_PeakClassifier(input_length=len(target_y), num_classes=num_classes)
    model.load_state_dict(torch.load(resolved_model_path, map_location=device))
    model.to(device)
    
    # Predict the top 3 most likely peak counts
    top_3_predictions = predict_top_k_peaks(target_y, model, use_savgol=True, device=device, top_k=3)
    top_3_counts = [pred[0] for pred in top_3_predictions]
    
    #print(f"CNN predicted top 3 peak counts: {top_3_counts}")
    
    # --- Step 2: Multi-processing Spawn ---
    output_queue = Queue()
    processes = []
    
    #print("Spawning 3 parallel fitting processes...")
    for peak_count in top_3_counts:
        p = Process(
            target=_fit_worker,
            args=(peak_count, samples, base_parameters, target_x, target_y, k, rounds, eps, output_queue)
        )
        processes.append(p)
        p.start()
        
    # --- Step 3: Rejoin & Collect Results ---
    results_list = []
    for _ in range(3):
        # Retrieve computed dictionary blocks from the queue as workers finish
        worker_res = output_queue.get()
        results_list.append(worker_res)
        
    for p in processes:
        p.join()
        
    #print("All fitting processes have concluded and rejoined the main loop.")
    
    # --- Step 4: Final Selection Logic ---
    # We're gonna use the Bayesian Information Criterion to select the best result (lowest val)
    successful_fits = [f for f in results_list if f["success"]]
    
    if not successful_fits:
        print("Error: All parallel fitting paths failed.")
        return None

    # Find the result block with the absolute lowest BIC value
    best_overall_result = min(successful_fits, key=lambda x: x["bic"])
    
    return best_overall_result