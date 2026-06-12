import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import savgol_filter


class SLMK_CNN_PeakClassifier(nn.Module):

    def __init__(self, input_length=100, num_classes=5):

        super(SLMK_CNN_PeakClassifier, self).__init__()

        # Multi-scale convolutions
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1, bias=False) # biases were not previously false
        self.conv2 = nn.Conv1d(1, 16, kernel_size=7, padding=3, bias=False)
        self.conv3 = nn.Conv1d(1, 32, kernel_size=13, padding=6, bias=False)
        self.conv4 = nn.Conv1d(1, 64, kernel_size=27, padding=13, bias=False)
        self.conv5 = nn.Conv1d(1, 128, kernel_size=45, padding=22, bias=False)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.flattened_size = (8 + 16 + 32 + 64 + 128) * (input_length // 2)

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(self.flattened_size, 512, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128, bias=False)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128,num_classes)


        self._initialize_weights()

    def forward(self, x):
        x1 = self.bn1(self.conv1(x))
        x2 = self.bn2(self.conv2(x))
        x3 = self.bn3(self.conv3(x))
        x4 = self.bn4(self.conv4(x))
        x5 = self.bn5(self.conv5(x))

        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)
        x4 = F.relu(x4)
        x5 = F.relu(x5)

        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)
        x4 = self.pool(x4)
        x5 = self.pool(x5)

        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x_flat = x_cat.view(x_cat.size(0), -1)


        x = self.fc1(x_flat)
        x = F.relu(self.bn_fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(self.bn_fc2(x))
        x = self.dropout(x)

        logits = self.fc3(x)
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def predict_top_k_peaks(spectrum, model, use_savgol=True, device="cuda", top_k=3):
    """
    Predicts the top K most likely peak counts for a single raw spectrum.
    
    Args:
        spectrum (np.ndarray or list): 1D array representing the target Raman spectrum.
        model (nn.Module): The trained instance of SLMK_CNN_PeakClassifier.
        use_savgol (bool): Whether to apply the Savitzky-Golay filter.
        device (str): Device to run inference on ('cuda' or 'cpu').
        top_k (int): Number of top predictions to return.
        
    Returns:
        list of tuples: Formatted as [(peak_count, probability_percentage), ...]
    """
    # 1. Ensure input is a floating-point NumPy array
    spectrum = np.array(spectrum, dtype=np.float32)
    
    # 2. Replicate dataset preprocessing pipeline exactly
    if use_savgol:
        spectrum = savgol_filter(spectrum, window_length=21, polyorder=3)
        
    spectrum = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-8)
    
    # 3. Convert to tensor and add dimensions: [1, 1, resolution] (Batch, Channel, Length)
    input_tensor = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    
    # 4. Inference mode
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0) # [Batch, Classes] -> [Classes]
        
    # 5. Extract top-K values and indices
    top_probs, top_indices = torch.topk(probabilities, k=top_k)
    
    # 6. Map indices back to peak counts (index 0 -> 1 peak, index 9 -> 10 peaks)
    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        peak_count = int(idx) + 1
        prob_percentage = float(prob) * 100
        results.append((peak_count, prob_percentage))
        
    return results