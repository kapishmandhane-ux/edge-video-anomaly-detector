"""
Evaluation Suite for Video Anomaly Detection.

Calculates:
- Frame-level Euclidean reconstruction error e(t)
- Dynamic / clip-normalized Anomaly Score S_a(t) and Regularity Score R(t)
- Receiver Operating Characteristic (ROC) curve & Area Under Curve (ROC-AUC)
- Equal Error Rate (EER) and calibrated decision thresholds
- Precision, Recall, and F1-score curves
- Visual summary plots saved to disk
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import matplotlib.pyplot as plt

try:
    from .model import build_model, ConvLSTMAutoencoder
    from .dataset import ensure_dataset_ready, VideoSequenceDataset
except ImportError:
    from model import build_model, ConvLSTMAutoencoder
    from dataset import ensure_dataset_ready, VideoSequenceDataset


def compute_frame_reconstruction_error(
    original_frame: np.ndarray, reconstructed_frame: np.ndarray
) -> float:
    """
    Compute Euclidean L2 reconstruction error:
        e(t) = sqrt(sum_{h, w} (I_t(h, w) - I_hat_t(h, w))^2)
    """
    diff = original_frame.astype(np.float32) - reconstructed_frame.astype(np.float32)
    return float(np.sqrt(np.sum(diff**2)))


def calculate_eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) where FPR == FNR (1 - TPR).
    Returns (eer_value, optimal_threshold).
    """
    fnr = 1.0 - tpr
    # Find index where |FPR - FNR| is minimal
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, thresh


@torch.no_grad()
def evaluate_model_on_clips(
    model: nn.Module,
    test_clip_dirs: List[Path],
    device: torch.device,
    window_size: int = 10,
    gt_dict: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, object]:
    """
    Run inference clip by clip, extracting frame-level reconstruction error
    and calculating aggregate metrics against ground truth.
    """
    model.eval()

    all_frame_errors: List[float] = []
    all_frame_labels: List[int] = []
    all_anomaly_scores: List[float] = []
    clip_evaluations: List[Dict[str, object]] = []

    for clip_dir in test_clip_dirs:
        clip_name = clip_dir.name
        dataset = VideoSequenceDataset(
            clip_dirs=[clip_dir],
            window_size=window_size,
            stride=1,
            gt_dict=gt_dict,
        )

        if len(dataset) == 0:
            continue

        clip_errors: List[float] = []
        clip_labels: List[int] = []

        # Track per-frame reconstruction errors
        # Each sliding window reconstructs frames [t, t+1, ... t+W-1]
        # For evaluation, we aggregate predictions on frame t+W-1 (most causal)
        for idx in range(len(dataset)):
            seq_tensor, lbl_tensor = dataset[idx]
            seq_batch = seq_tensor.unsqueeze(0).to(device)  # (1, T, 1, H, W)
            reconstructed = model(seq_batch).squeeze(0).cpu().numpy()  # (T, 1, H, W)
            original = seq_tensor.numpy()  # (T, 1, H, W)

            # Evaluate final frame of the window
            err = compute_frame_reconstruction_error(
                original[-1, 0], reconstructed[-1, 0]
            )
            clip_errors.append(err)
            clip_labels.append(int(lbl_tensor[-1].item()))

        # Min-Max normalize errors within clip to obtain Anomaly Score S_a(t)
        min_err = min(clip_errors)
        max_err = max(clip_errors)
        denom = (max_err - min_err) if (max_err - min_err) > 1e-8 else 1.0

        anomaly_scores = [(e - min_err) / denom for e in clip_errors]
        regularity_scores = [1.0 - sa for sa in anomaly_scores]

        all_frame_errors.extend(clip_errors)
        all_frame_labels.extend(clip_labels)
        all_anomaly_scores.extend(anomaly_scores)

        clip_evaluations.append({
            "clip_name": clip_name,
            "num_frames": len(clip_errors),
            "anomaly_ratio": sum(clip_labels) / max(1, len(clip_labels)),
            "min_error": min_err,
            "max_error": max_err,
        })

    y_true = np.array(all_frame_labels, dtype=np.int32)
    y_scores = np.array(all_anomaly_scores, dtype=np.float32)

    # Calculate ROC and AUC
    if len(np.unique(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = float(auc(fpr, tpr))
        eer, eer_threshold = calculate_eer(fpr, tpr, thresholds)

        # Calculate optimal F1 score threshold
        precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_f1_idx = int(np.nanargmax(f1_scores))
        best_f1 = float(f1_scores[best_f1_idx])
        best_f1_threshold = float(pr_thresholds[min(best_f1_idx, len(pr_thresholds) - 1)])
    else:
        fpr, tpr, thresholds = np.array([0, 1]), np.array([0, 1]), np.array([0.5, 0.5])
        roc_auc = 1.0
        eer, eer_threshold = 0.0, 0.5
        best_f1, best_f1_threshold = 1.0, 0.5

    results = {
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "best_f1": best_f1,
        "best_f1_threshold": best_f1_threshold,
        "total_frames_evaluated": len(y_true),
        "total_anomalous_frames": int(np.sum(y_true)),
        "clip_details": clip_evaluations,
    }

    return results, fpr, tpr


def plot_evaluation_curves(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    eer: float,
    save_path: Path,
) -> None:
    """Save high-resolution ROC curve figure."""
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(fpr, tpr, color="#00aa55", lw=2.5, label=f"ConvLSTM AE (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5)")
    plt.plot([eer], [1 - eer], marker="o", markersize=8, color="red", label=f"EER = {eer:.4f}")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) - Anomaly Detection", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_evaluation(
    data_dir: str = "data/raw/UCSDped2",
    checkpoint_path: str = "models/checkpoints/best_model.pt",
    output_dir: str = "models",
    device_str: Optional[str] = None,
) -> Dict[str, object]:
    """Execute evaluation and log metrics."""
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    data_root = ensure_dataset_ready(data_dir)
    test_base = data_root / "Test"
    test_clips = sorted([d for d in test_base.iterdir() if d.is_dir() and not d.name.endswith("_gt")])

    print(f"[*] Evaluating on {len(test_clips)} test clips using {device}...")

    # Load master ground_truth.json if present
    gt_json = test_base / "ground_truth.json"
    gt_dict = {}
    if gt_json.exists():
        with open(gt_json, "r") as f:
            gt_dict = json.load(f)

    # Initialize model & load weights
    model = build_model(in_channels=1, temporal_window=10, device=device)
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt}")

    checkpoint_data = torch.load(ckpt, map_location=device, weights_only=False)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state_dict)
    print(f"[+] Loaded weights from {ckpt}")

    results, fpr, tpr = evaluate_model_on_clips(
        model=model,
        test_clip_dirs=test_clips,
        device=device,
        window_size=10,
        gt_dict=gt_dict,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    json_path = out_path / "evaluation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save ROC plot
    roc_plot_path = out_path / "evaluation_roc.png"
    plot_evaluation_curves(
        fpr=fpr,
        tpr=tpr,
        roc_auc=results["roc_auc"],
        eer=results["eer"],
        save_path=roc_plot_path,
    )

    print("\n==================================================")
    print("           ANOMALY EVALUATION RESULTS             ")
    print("==================================================")
    print(f" ROC-AUC Score        : {results['roc_auc']:.4f}")
    print(f" Equal Error Rate (EER): {results['eer']:.4f}")
    print(f" EER Score Threshold  : {results['eer_threshold']:.4f}")
    print(f" Best F1 Score        : {results['best_f1']:.4f}")
    print(f" Best F1 Threshold    : {results['best_f1_threshold']:.4f}")
    print(f" Total Frames Tested  : {results['total_frames_evaluated']}")
    print(f" Anomaly Frames Count : {results['total_anomalous_frames']}")
    print(f"[+] Results saved to: {json_path}")
    print(f"[+] ROC Plot saved to: {roc_plot_path}")
    print("==================================================\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ConvLSTM Autoencoder")
    parser.add_argument("--data-dir", type=str, default="data/raw/UCSDped2")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device_str=args.device,
    )

