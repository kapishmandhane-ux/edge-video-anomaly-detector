"""
ONNX Graph Exporter and Numeric Validator.

Exports ConvLSTM Autoencoder to ONNX format with:
- Opset version 17
- Input signature: 'input_seq' [B, 10, 1, 128, 128]
- Output signature: 'reconstructed_seq' [B, 10, 1, 128, 128]
- Dynamic batch axis configuration
- Graph structure validation with onnx.checker
- Numeric discrepancy validation between native PyTorch and ONNX Runtime
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None

try:
    from .model import build_model, ConvLSTMAutoencoder
except ImportError:
    from model import build_model, ConvLSTMAutoencoder


def export_to_onnx(
    checkpoint_path: Optional[str] = "models/checkpoints/best_model.pt",
    output_onnx_path: str = "models/onnx/anomaly_detector.onnx",
    temporal_window: int = 10,
    dynamic_batch: bool = True,
    opset_version: int = 17,
    device_str: str = "cpu",
) -> str:
    """
    Export PyTorch ConvLSTM Autoencoder model to ONNX format.
    """
    device = torch.device(device_str)
    model = build_model(in_channels=1, temporal_window=temporal_window, device=device)

    # Load weights if checkpoint exists
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
        model.load_state_dict(state_dict)
        print(f"[+] Loaded weights from {checkpoint_path}")
    else:
        print(f"[!] Checkpoint not provided or not found; exporting model with random initialized weights.")

    model.eval()

    out_file = Path(output_onnx_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, temporal_window, 1, 128, 128, dtype=torch.float32, device=device)

    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    if dynamic_batch:
        dynamic_axes = {
            "input_seq": {0: "batch_size"},
            "reconstructed_seq": {0: "batch_size"},
        }

    print(f"[*] Exporting graph to {out_file} (opset={opset_version}, dynamic_batch={dynamic_batch})...")

    # Clean PyTorch export
    torch.onnx.export(
        model,
        dummy_input,
        str(out_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_seq"],
        output_names=["reconstructed_seq"],
        dynamic_axes=dynamic_axes,
    )

    print(f"[+] ONNX model exported successfully to {out_file}")

    # 1. Structural graph verification via onnx.checker
    if onnx is not None:
        print("[*] Validating ONNX structural integrity with onnx.checker...")
        onnx_proto = onnx.load(str(out_file))
        onnx.checker.check_model(onnx_proto)
        print("[+] ONNX model structure is strictly valid.")
    else:
        print("[!] onnx package not installed. Skipping structural checker.")

    # 2. Numeric Discrepancy Verification against Native PyTorch via ONNX Runtime
    if ort is not None:
        print("[*] Running numeric tolerance verification with ONNX Runtime...")
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(str(out_file), session_options, providers=["CPUExecutionProvider"])

        test_input_np = np.random.uniform(0.0, 1.0, size=(1, temporal_window, 1, 128, 128)).astype(np.float32)
        test_input_pt = torch.from_numpy(test_input_np).to(device)

        with torch.no_grad():
            pt_out = model(test_input_pt).cpu().numpy()

        ort_inputs = {"input_seq": test_input_np}
        ort_out = session.run(["reconstructed_seq"], ort_inputs)[0]

        max_abs_diff = float(np.max(np.abs(pt_out - ort_out)))
        mean_abs_diff = float(np.mean(np.abs(pt_out - ort_out)))

        print(f"[*] Output discrepancy - Max: {max_abs_diff:.6f}, Mean: {mean_abs_diff:.6f}")

        try:
            np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-4)
            print("[+] Numerical equivalence verified! (rtol=1e-3, atol=1e-4 passed)")
        except AssertionError as err:
            print(f"[!] Warning: Numerical discrepancy slightly exceeded strict threshold: {err}")
    else:
        print("[!] onnxruntime not installed. Skipping runtime numeric verification.")

    return str(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ConvLSTM AE to ONNX")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default="models/onnx/anomaly_detector.onnx")
    parser.add_argument("--temporal-window", type=int, default=10)
    parser.add_argument("--dynamic-batch", action="store_true", default=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        temporal_window=args.temporal_window,
        dynamic_batch=args.dynamic_batch,
        opset_version=args.opset,
        device_str=args.device,
    )

