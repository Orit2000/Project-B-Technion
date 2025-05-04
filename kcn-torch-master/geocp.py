import numpy as np
import torch

class GeoCPWrapper:
    def __init__(self, model, calibration_coords, calibration_features, calibration_labels, eps=0.1, decay_beta=1.0, device="cpu"):
        """
        model: Trained model
        calibration_coords: (n_calib, 2)
        calibration_features: (n_calib, d)
        calibration_labels: (n_calib,)
        eps: miscoverage level (default 0.1 for 90% confidence)
        decay_beta: decay rate for distance weighting
        """
        self.model = model.to(device)
        self.device = device
        self.calib_coords = calibration_coords.to(self.device)
        self.calib_features = calibration_features.to(self.device)
        self.calib_labels = calibration_labels.to(self.device)
        self.eps = eps
        self.decay_beta = decay_beta
        # Predict calibration set outputs
        with torch.no_grad():
            preds = model(calibration_coords, calibration_features).cpu().numpy()
        
        self.nonconformity_scores = np.abs(preds.flatten() - calibration_labels.cpu().numpy().flatten())

    def decay(self, distances):
        """Decay function (e.g., exponential)"""
        return np.exp(-self.decay_beta * distances)

    def predict_interval(self, test_coord, test_feature):
        """
        Return (lower_bound, upper_bound) prediction interval for a test point.
        """
        # Predict model point estimate
        with torch.no_grad():
            pred = self.model(test_coord.unsqueeze(0), test_feature.unsqueeze(0)).cpu().numpy().item()

        # Compute distances to calibration set
        test_coord_np = test_coord.cpu().numpy()
        distances = np.linalg.norm(self.calib_coords - test_coord_np, axis=1)

        # Compute weights
        weights = self.decay(distances)
        weights /= np.sum(weights)

        # Weighted quantile
        sorted_idx = np.argsort(self.nonconformity_scores)
        sorted_scores = self.nonconformity_scores[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cumulative_weights = np.cumsum(sorted_weights)

        idx = np.searchsorted(cumulative_weights, 1 - self.eps)
        quantile = sorted_scores[min(idx, len(sorted_scores) - 1)]

        return pred - quantile, pred + quantile

