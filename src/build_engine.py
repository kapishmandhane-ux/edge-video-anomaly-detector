"""
NVIDIA TensorRT Engine Compilation and Edge Optimizer.

Compiles exported ONNX graph to an optimized TensorRT engine:
- Configures dynamic batch optimization profile:
    Min: [1, 10, 1, 128, 128]
    Opt: [1, 10, 1, 128, 128]
    Max: [4, 10, 1, 128, 128]
- Precision acceleration: FP16 and INT8 Calibration (IInt8EntropyCalibrator2)
- Serializes plan file (.engine) to disk
- Graceful Fallback: Seamlessly falls back to onnxruntime-gpu with
  TensorrtExecutionProvider or CUDAExecutionProvider if native TensorRT is missing.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Union

import numpy as np

# Check TensorRT availability
try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    trt = None
    HAS_TRT = False

# Check PyCUDA availability
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_PYCUDA = True
except ImportError:
    cuda = None
    HAS_PYCUDA = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    ort = None
    HAS_ORT = False

try:
    from .dataset import ensure_dataset_ready, VideoSequenceDataset
except ImportError:
    from dataset import ensure_dataset_ready, VideoSequenceDataset


if HAS_TRT:
    class IInt8EntropyCalibrator2(trt.IInt8EntropyCalibrator2):
        """
        Entropy Calibrator for TensorRT INT8 Quantization.
        Streams representative calibration batches from normal video sequences.
        """

        def __init__(
            self,
            data_dir: Union[str, Path],
            cache_file: Path,
            batch_size: int = 1,
            num_batches: int = 100,
            temporal_window: int = 10,
        ) -> None:
            super().__init__()
            self.cache_file = Path(cache_file)
            self.batch_size = batch_size
            self.num_batches = num_batches
            self.temporal_window = temporal_window
            self.current_idx = 0

            # Prepare dataset
            root = ensure_dataset_ready(data_dir)
            train_clips = sorted([d for d in (root / "Train").iterdir() if d.is_dir()])
            self.dataset = VideoSequenceDataset(
                clip_dirs=train_clips,
                window_size=temporal_window,
                stride=2,
            )

            # Pre-allocate host batch buffer and CUDA device memory if pycuda available
            self.sample_shape = (batch_size, temporal_window, 1, 128, 128)
            self.host_allocation = np.zeros(self.sample_shape, dtype=np.float32)
            self.nbytes = self.host_allocation.nbytes

            if HAS_PYCUDA:
                self.device_allocation = cuda.mem_alloc(self.nbytes)
            else:
                self.device_allocation = None

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names: List[str]) -> Optional[List[int]]:
            if self.current_idx >= min(self.num_batches, len(self.dataset)):
                return None

            # Collect batch samples
            for b_idx in range(self.batch_size):
                seq_idx = (self.current_idx + b_idx) % len(self.dataset)
                seq_tensor, _ = self.dataset[seq_idx]
                self.host_allocation[b_idx] = seq_tensor.numpy()

            self.current_idx += self.batch_size

            if HAS_PYCUDA and self.device_allocation is not None:
                cuda.memcpy_htod(self.device_allocation, self.host_allocation)
                return [int(self.device_allocation)]
            return None

        def read_calibration_cache(self) -> Optional[bytes]:
            if self.cache_file.exists():
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "wb") as f:
                f.write(cache)


def build_tensorrt_engine(
    onnx_path: str = "models/onnx/anomaly_detector.onnx",
    engine_path: str = "models/engine/anomaly_detector_fp16.engine",
    precision: str = "fp16",
    calib_data_dir: str = "data/raw/UCSDped2",
    workspace_gb: int = 2,
) -> Optional[str]:
    """
    Build native TensorRT engine with optimization profile and FP16/INT8 modes.
    """
    if not HAS_TRT:
        print("[!] Native TensorRT library ('tensorrt') not found in environment.")
        return None

    onnx_file = Path(onnx_path)
    engine_file = Path(engine_path)
    engine_file.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX model file not found: {onnx_file}")

    print(f"[*] Building TensorRT Engine from {onnx_file} (Precision: {precision.upper()})...")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            print("[!] Failed to parse ONNX file with TensorRT parser:")
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # Optimization Profile for Dynamic Input Shape
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input_seq",
        min=(1, 10, 1, 128, 128),
        opt=(1, 10, 1, 128, 128),
        max=(4, 10, 1, 128, 128),
    )
    config.add_optimization_profile(profile)

    # Precision Configuration
    if precision.lower() == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[+] FP16 precision mode enabled on hardware.")
        else:
            print("[!] Platform does not have fast native FP16; building FP32 baseline.")

    elif precision.lower() == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            cache_file = engine_file.parent / "int8_calibration.cache"
            calibrator = IInt8EntropyCalibrator2(
                data_dir=calib_data_dir,
                cache_file=cache_file,
                batch_size=1,
                num_batches=100,
            )
            config.int8_calibrator = calibrator
            print("[+] INT8 Entropy Calibrator II configured.")
        else:
            print("[!] Hardware lacks fast INT8; falling back to FP16/FP32.")

    # Build and serialize engine
    print("[*] Serializing TensorRT optimization plan (this may take 1-2 minutes)...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("[!] TensorRT engine build failed.")
        return None

    with open(engine_file, "wb") as f:
        f.write(plan)

    print(f"[+] TensorRT Engine serialized and saved to: {engine_file}")
    return str(engine_file)


def setup_edge_runtime(
    onnx_path: str = "models/onnx/anomaly_detector.onnx",
    engine_path: str = "models/engine/anomaly_detector_fp16.engine",
    precision: str = "fp16",
    calib_data_dir: str = "data/raw/UCSDped2",
) -> str:
    """
    Unified entrypoint: Attempts native TensorRT compilation first.
    If unavailable, falls back cleanly to ONNX Runtime GPU/CPU without failing.
    """
    engine_file = Path(engine_path)
    engine_file.parent.mkdir(parents=True, exist_ok=True)

    # Attempt native TensorRT build
    trt_result = None
    if HAS_TRT:
        try:
            trt_result = build_tensorrt_engine(
                onnx_path=onnx_path,
                engine_path=engine_path,
                precision=precision,
                calib_data_dir=calib_data_dir,
            )
        except Exception as e:
            print(f"[!] TensorRT build exception encountered: {e}. Transitioning to ORT edge fallback.")

    if trt_result is not None:
        return trt_result

    # Graceful Fallback Path: ONNX Runtime Edge Execution
    print("\n--------------------------------------------------------------")
    print("[*] INITIATING EDGE RUNTIME FALLBACK (ONNX Runtime Edge Pipeline)")
    print("--------------------------------------------------------------")

    providers = []
    if HAS_ORT:
        available_providers = ort.get_available_providers()
        print(f"[*] Available ONNX Runtime execution providers: {available_providers}")

        if "TensorrtExecutionProvider" in available_providers:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
    else:
        providers = ["PyTorch/Native"]

    print(f"[+] Fallback Execution Provider Hierarchy: {providers}")

    # Generate edge deployment descriptor
    edge_config = {
        "engine_type": "onnxruntime_edge_fallback",
        "onnx_model_path": onnx_path,
        "precision": precision,
        "input_signature": {"name": "input_seq", "shape": [1, 10, 1, 128, 128]},
        "output_signature": {"name": "reconstructed_seq", "shape": [1, 10, 1, 128, 128]},
        "execution_providers": providers,
        "dynamic_batch_profile": {
            "min": [1, 10, 1, 128, 128],
            "opt": [1, 10, 1, 128, 128],
            "max": [4, 10, 1, 128, 128],
        },
    }

    config_path = engine_file.parent / "edge_runtime_config.json"
    with open(config_path, "w") as f:
        json.dump(edge_config, f, indent=2)

    print(f"[+] Edge runtime configuration manifest saved to: {config_path}")
    print("[+] Edge engine preparation complete. Ready for real-time inference streaming.\n")

    return str(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile TensorRT Engine or Setup Edge Fallback")
    parser.add_argument("--onnx", type=str, default="models/onnx/anomaly_detector.onnx")
    parser.add_argument("--engine", type=str, default="models/engine/anomaly_detector_fp16.engine")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "int8", "fp32"])
    parser.add_argument("--calib-dir", type=str, default="data/raw/UCSDped2")
    args = parser.parse_args()

    setup_edge_runtime(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        calib_data_dir=args.calib_dir,
    )

