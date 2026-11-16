"""
Real-Time Edge Video Anomaly Detection Stream Inference Pipeline.

Features:
- Asynchronous thread-safe circular frame buffer (temporal window T=10)
- Zero-copy / pre-allocated memory buffers for GPU execution
- Multi-backend inference: Native TensorRT Engine, ONNX Runtime GPU/CPU, or PyTorch
- Stream Ingestion: RTSP URL, Webcam (0), Video files (.mp4/.avi), or Synthetic Live Generator
- Real-Time HUD:
    1. Raw camera feed with bounding guides
    2. Reconstruction difference heatmap (JET colormap overlay)
    3. Temporal Regularity Score R(t) sparkline graph with safety threshold line
    4. Red Flashing "ANOMALY DETECTED" alert banner
    5. Real-time telemetry: Acquisition, Preprocessing, Inference, Postprocessing, and FPS
- Video recording support (.mp4) for headless and continuous edge surveillance
"""

import os
import sys
import time
import math
import queue
import argparse
import threading
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List

import numpy as np
import cv2

# Check optional TensorRT & PyCUDA
try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    trt = None
    HAS_TRT = False

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
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from .model import build_model
except (ImportError, ValueError):
    try:
        from model import build_model
    except ImportError:
        build_model = None


class CircularFrameBuffer:
    """Thread-safe circular frame buffer for sliding window temporal inference."""

    def __init__(self, capacity: int = 10, target_size: Tuple[int, int] = (128, 128)) -> None:
        self.capacity = capacity
        self.target_size = target_size
        self._lock = threading.Lock()
        self._buffer: deque = deque(maxlen=capacity)
        self._norm_buffer: deque = deque(maxlen=capacity)

    def append(self, raw_frame_bgr: np.ndarray) -> None:
        """Add a new raw frame and preprocess to normalized grayscale."""
        # Convert to grayscale and resize to (128, 128)
        gray = cv2.cvtColor(raw_frame_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / 255.0)[np.newaxis, :, :]  # (1, H, W)

        with self._lock:
            self._buffer.append(raw_frame_bgr)
            self._norm_buffer.append(normalized)

    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) == self.capacity

    def get_latest_raw_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1].copy()

    def get_temporal_window_batch(self) -> Optional[np.ndarray]:
        """
        Returns:
            np.ndarray of shape (1, T, 1, 128, 128) float32 in [0, 1]
        """
        with self._lock:
            if len(self._norm_buffer) < self.capacity:
                return None
            # Stack into (T, 1, H, W) -> add batch dimension (1, T, 1, H, W)
            stacked = np.stack(list(self._norm_buffer), axis=0)[np.newaxis, ...]
            return stacked.astype(np.float32)


class BaseInferenceEngine:
    """Abstract base class for inference execution backends."""

    def infer(self, input_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TensorRTInferenceEngine(BaseInferenceEngine):
    """Native TensorRT Engine execution with pre-allocated pinned host/device buffers."""

    def __init__(self, engine_path: str) -> None:
        if not HAS_TRT or not HAS_PYCUDA:
            raise RuntimeError("TensorRT or PyCUDA not available for native engine execution.")

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate host and device memory buffers
        self.host_input = cuda.pagelocked_empty((1, 10, 1, 128, 128), dtype=np.float32)
        self.host_output = cuda.pagelocked_empty((1, 10, 1, 128, 128), dtype=np.float32)

        self.d_input = cuda.mem_alloc(self.host_input.nbytes)
        self.d_output = cuda.mem_alloc(self.host_output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

    def infer(self, input_batch: np.ndarray) -> np.ndarray:
        # Copy input into pinned host memory
        np.copyto(self.host_input, input_batch)
        # Asynchronously transfer Host to Device
        cuda.memcpy_htod_async(self.d_input, self.host_input, self.stream)
        # Execute TensorRT context
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Asynchronously transfer Device to Host
        cuda.memcpy_dtoh_async(self.host_output, self.d_output, self.stream)
        # Synchronize CUDA stream
        self.stream.synchronize()
        return self.host_output.copy()


class ONNXRuntimeInferenceEngine(BaseInferenceEngine):
    """ONNX Runtime Edge Engine with GPU / CPU fallback."""

    def __init__(self, onnx_model_path: str) -> None:
        if not HAS_ORT:
            raise RuntimeError("ONNX Runtime is not installed.")

        session_opts = ort.SessionOptions()
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_opts.intra_op_num_threads = 4

        available = ort.get_available_providers()
        chosen_providers = []
        if "TensorrtExecutionProvider" in available:
            chosen_providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available:
            chosen_providers.append("CUDAExecutionProvider")
        chosen_providers.append("CPUExecutionProvider")

        print(f"[*] Initializing ONNX Runtime Engine with providers: {chosen_providers}")
        self.session = ort.InferenceSession(str(onnx_model_path), session_opts, providers=chosen_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, input_batch: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: input_batch})
        return outputs[0]


class PyTorchInferenceEngine(BaseInferenceEngine):
    """PyTorch Native Engine fallback using torch.cuda or CPU."""

    def __init__(self, checkpoint_path: str, device_str: Optional[str] = None) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed.")

        if device_str is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.model = build_model(in_channels=1, temporal_window=10, device=self.device)

        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            checkpoint_data = torch.load(ckpt, map_location=self.device, weights_only=False)
            state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
            self.model.load_state_dict(state_dict)
            print(f"[+] PyTorch Inference Engine loaded weights from: {ckpt} (Device: {self.device})")
        else:
            print(f"[!] Warning: Checkpoint {ckpt} not found. Running with initialized weights.")

        self.model.eval()

    @torch.no_grad()
    def infer(self, input_batch: np.ndarray) -> np.ndarray:
        tensor_in = torch.from_numpy(input_batch).to(self.device)
        tensor_out = self.model(tensor_in)
        return tensor_out.cpu().numpy()


def create_engine(
    model_path: str,
    device_str: Optional[str] = None,
) -> BaseInferenceEngine:
    """Unified engine factory that inspects model path and selects optimal backend."""
    path = Path(model_path)

    # 1. Check for native TensorRT engine
    if path.suffix == ".engine":
        if HAS_TRT and HAS_PYCUDA:
            print(f"[+] Instantiating native TensorRT Engine: {path}")
            return TensorRTInferenceEngine(str(path))
        else:
            # Look for companion ONNX or PT
            print("[!] TensorRT/PyCUDA not available for .engine; searching for ONNX/PyTorch companion...")
            onnx_alt = path.parent.parent / "onnx" / "anomaly_detector.onnx"
            if onnx_alt.exists() and HAS_ORT:
                return ONNXRuntimeInferenceEngine(str(onnx_alt))
            pt_alt = path.parent.parent / "checkpoints" / "best_model.pt"
            return PyTorchInferenceEngine(str(pt_alt), device_str=device_str)

    # 2. Check for ONNX model
    if path.suffix == ".onnx":
        if HAS_ORT:
            return ONNXRuntimeInferenceEngine(str(path))
        else:
            pt_alt = path.parent.parent / "checkpoints" / "best_model.pt"
            return PyTorchInferenceEngine(str(pt_alt), device_str=device_str)

    # 3. Check for PyTorch checkpoint or edge runtime config
    if path.suffix == ".json":
        # Load edge config
        import json
        with open(path, "r") as f:
            cfg = json.load(f)
        onnx_file = cfg.get("onnx_model_path", "models/onnx/anomaly_detector.onnx")
        # If GPU is available, PyTorch CUDA provides ultra-fast 60+ FPS edge execution
        if HAS_TORCH and torch.cuda.is_available() and device_str != "cpu":
            pt_alt = path.parent.parent / "checkpoints" / "best_model.pt"
            if pt_alt.exists():
                return PyTorchInferenceEngine(str(pt_alt), device_str="cuda")
        if HAS_ORT and Path(onnx_file).exists():
            return ONNXRuntimeInferenceEngine(onnx_file)
        return PyTorchInferenceEngine("models/checkpoints/best_model.pt", device_str=device_str)

    # Default: PyTorch Engine
    return PyTorchInferenceEngine(str(path), device_str=device_str)


class SyntheticLiveVideoFeed:
    """Generates continuous animated frames with periodic anomalies for real-time testing."""

    def __init__(self, width: int = 480, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_idx = 0

        # Normal entities
        self.entities = [
            {"x": 50.0, "y": 200.0, "vx": 1.5, "vy": 0.0, "r": 12, "color": (220, 220, 220), "phase": 0.0},
            {"x": 200.0, "y": 260.0, "vx": -1.2, "vy": 0.0, "r": 10, "color": (200, 200, 200), "phase": 1.2},
            {"x": 380.0, "y": 220.0, "vx": -1.8, "vy": 0.0, "r": 14, "color": (230, 230, 230), "phase": 2.5},
            {"x": 100.0, "y": 300.0, "vx": 1.3, "vy": 0.0, "r": 11, "color": (190, 190, 190), "phase": 4.0},
        ]
        self.intruder = {"x": -20.0, "y": 240.0, "vx": 8.0, "r": 18, "active": False}

    def read(self) -> Tuple[bool, np.ndarray]:
        self.frame_idx += 1
        canvas = np.full((self.height, self.width, 3), 40, dtype=np.uint8)

        # Draw walkway boundaries
        cv2.line(canvas, (0, 150), (self.width, 150), (70, 70, 70), 2)
        cv2.line(canvas, (0, 350), (self.width, 350), (70, 70, 70), 2)

        # Trigger anomaly every 150 frames for 60 frames
        cycle = self.frame_idx % 220
        is_anom_time = 120 <= cycle <= 180

        # Update normal entities
        for ent in self.entities:
            ent["x"] += ent["vx"]
            ent["phase"] += 0.2
            ent["y"] += math.sin(ent["phase"]) * 0.5
            if ent["x"] > self.width + 20:
                ent["x"] = -20
            elif ent["x"] < -20:
                ent["x"] = self.width + 20

            cv2.circle(
                canvas,
                (int(ent["x"]), int(ent["y"])),
                ent["r"],
                ent["color"],
                -1,
            )

        # Update intruder if anomaly is active
        if is_anom_time:
            self.intruder["active"] = True
            self.intruder["x"] += self.intruder["vx"]
            self.intruder["y"] = 240.0 + 35.0 * math.sin(self.frame_idx * 0.3)
            if self.intruder["x"] > self.width + 30:
                self.intruder["x"] = -30

            # Draw anomalous entity (square fast mover)
            ix, iy, ir = int(self.intruder["x"]), int(self.intruder["y"]), self.intruder["r"]
            cv2.rectangle(
                canvas,
                (ix - ir, iy - ir),
                (ix + ir, iy + ir),
                (255, 255, 255),
                -1,
            )
        else:
            self.intruder["active"] = False
            self.intruder["x"] = -20.0

        # Subtle noise
        noise = np.random.normal(0, 4.0, (self.height, self.width, 3)).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Simulate frame rate delay
        time.sleep(1.0 / self.fps)
        return True, canvas

    def release(self) -> None:
        pass


def draw_hud(
    raw_frame: np.ndarray,
    diff_heatmap_color: np.ndarray,
    regularity_history: List[float],
    regularity_score: float,
    threshold: float,
    telemetry: Dict[str, float],
    frame_idx: int,
) -> np.ndarray:
    """
    Construct integrated real-time surveillance Heads-Up Display (HUD).
    Combines:
    1. Raw camera feed (Left Panel)
    2. Reconstruction error heatmap overlay (Right Panel)
    3. Regularity Score Sparkline History Chart (Bottom Panel)
    4. Flashing warning banner and detailed performance telemetry
    """
    h, w, _ = raw_frame.shape

    # 1. Overlay heatmap on copy of raw frame
    blend_heatmap = cv2.addWeighted(raw_frame, 0.6, diff_heatmap_color, 0.4, 0)

    # 2. Top Banner: Anomaly Status
    is_anomaly = regularity_score < threshold
    flash_on = (frame_idx // 8) % 2 == 0  # 4 Hz flashing indicator

    top_banner_h = 50
    banner = np.zeros((top_banner_h, w * 2, 3), dtype=np.uint8)

    if is_anomaly:
        banner_color = (0, 0, 200) if flash_on else (0, 0, 120)
        cv2.rectangle(banner, (0, 0), (w * 2, top_banner_h), banner_color, -1)
        alert_text = f"CRITICAL: ANOMALY DETECTED | REGULARITY: {regularity_score:.3f} < {threshold:.2f}"
        cv2.putText(
            banner, alert_text, (20, 33), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2
        )
    else:
        cv2.rectangle(banner, (0, 0), (w * 2, top_banner_h), (35, 95, 35), -1)
        ok_text = f"NORMAL STATE: RECONSTRUCTION NOMINAL | REGULARITY: {regularity_score:.3f}"
        cv2.putText(
            banner, ok_text, (20, 33), cv2.FONT_HERSHEY_DUPLEX, 0.75, (220, 255, 220), 2
        )

    # 3. Label panels
    cv2.putText(
        raw_frame, "RAW VIDEO FEED", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
    )
    cv2.putText(
        blend_heatmap,
        "SPATIO-TEMPORAL RECONSTRUCTION ERROR",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    # 4. Telemetry overlay box on raw frame
    t_lines = [
        f"FPS: {telemetry.get('fps', 0.0):.1f}",
        f"Acq Latency: {telemetry.get('acq_ms', 0.0):.1f} ms",
        f"Pre Latency: {telemetry.get('pre_ms', 0.0):.1f} ms",
        f"Inference:   {telemetry.get('infer_ms', 0.0):.1f} ms",
        f"Post Latency:{telemetry.get('post_ms', 0.0):.1f} ms",
    ]
    cv2.rectangle(raw_frame, (10, h - 120), (220, h - 10), (20, 20, 20), -1)
    cv2.rectangle(raw_frame, (10, h - 120), (220, h - 10), (80, 80, 80), 1)
    for i, line in enumerate(t_lines):
        cv2.putText(
            raw_frame, line, (18, h - 100 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
        )

    # Combine Top feeds horizontally
    top_row = np.hstack([raw_frame, blend_heatmap])

    # 5. Bottom Panel: Regularity Sparkline Graph
    graph_h = 130
    graph_w = w * 2
    graph = np.full((graph_h, graph_w, 3), 25, dtype=np.uint8)

    # Draw grid & threshold line
    thresh_y = int(graph_h - (threshold * (graph_h - 30)) - 15)
    cv2.line(graph, (0, thresh_y), (graph_w, thresh_y), (0, 0, 255), 1)
    cv2.putText(
        graph,
        f"Threshold ({threshold:.2f})",
        (graph_w - 140, thresh_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 255),
        1,
    )

    cv2.putText(
        graph,
        "TEMPORAL REGULARITY SCORE S_t in [0.0, 1.0]",
        (15, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    # Plot sparkline curve
    if len(regularity_history) > 1:
        points = []
        max_history = 120
        recent = regularity_history[-max_history:]
        step_x = (graph_w - 40) / max(1, max_history - 1)

        for i, val in enumerate(recent):
            px = int(20 + i * step_x)
            py = int(graph_h - (val * (graph_h - 35)) - 15)
            points.append((px, py))

        for i in range(len(points) - 1):
            pt_color = (0, 230, 0) if recent[i] >= threshold else (0, 0, 255)
            cv2.line(graph, points[i], points[i + 1], pt_color, 2)

    # Assemble complete HUD canvas
    hud_canvas = np.vstack([banner, top_row, graph])
    return hud_canvas


def run_stream_inference(
    video_source: Union[int, str] = 0,
    model_path: str = "models/engine/anomaly_detector_fp16.engine",
    threshold: float = 0.5,
    output_video_path: Optional[str] = "output_stream.mp4",
    max_frames: Optional[int] = None,
    device_str: Optional[str] = None,
    display: bool = True,
) -> None:
    """
    Main real-time stream inference pipeline.
    """
    print(f"[*] Initializing Edge Anomaly Detection Stream Engine...")
    print(f"[*] Target Model Path : {model_path}")
    print(f"[*] Threshold         : {threshold}")
    print(f"[*] Video Source      : {video_source}")

    # 1. Initialize Video Source
    cap = None
    use_synthetic = False

    if str(video_source).isdigit():
        src_int = int(video_source)
        cap = cv2.VideoCapture(src_int)
        if not cap.isOpened():
            print(f"[!] Physical camera #{src_int} could not be opened. Falling back to Synthetic Stream.")
            use_synthetic = True
    elif isinstance(video_source, str) and video_source.lower() == "synthetic":
        use_synthetic = True
    else:
        # File path or RTSP URL
        cap = cv2.VideoCapture(str(video_source))
        if not cap.isOpened():
            print(f"[!] Video source '{video_source}' failed to open. Falling back to Synthetic Stream.")
            use_synthetic = True

    if use_synthetic:
        cap = SyntheticLiveVideoFeed(width=480, height=360, fps=30)
        print("[+] Synthetic live video source online.")

    # 2. Initialize Frame Buffer and Engine
    frame_buffer = CircularFrameBuffer(capacity=10, target_size=(128, 128))
    engine = create_engine(model_path=model_path, device_str=device_str)

    # 3. Dynamic Normalization History Buffer for Anomaly Score Calculation
    error_history: deque = deque(maxlen=60)
    regularity_history: List[float] = []

    # 4. Video Output Writer
    writer = None
    fps_history: deque = deque(maxlen=30)

    frame_count = 0
    print("[*] Stream inference active. Press 'q' or ESC in HUD window to terminate.")

    try:
        while True:
            t_start = time.perf_counter()

            # --- Stage 1: Frame Acquisition ---
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[*] End of stream or stream disconnected.")
                break

            t_acq = (time.perf_counter() - t_start) * 1000.0
            frame_count += 1

            # Resize raw frame to standard display size
            frame_display = cv2.resize(frame, (480, 360))

            # --- Stage 2: Preprocessing ---
            t_pre_start = time.perf_counter()
            frame_buffer.append(frame_display)
            input_batch = frame_buffer.get_temporal_window_batch()
            t_pre = (time.perf_counter() - t_pre_start) * 1000.0

            if input_batch is None:
                # Buffer warming up (< 10 frames)
                cv2.putText(
                    frame_display,
                    f"Buffering frames: {frame_count}/10",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                if display:
                    try:
                        cv2.imshow("Edge Anomaly Detector HUD", frame_display)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                    except Exception:
                        pass
                continue

            # --- Stage 3: Engine Execution ---
            t_infer_start = time.perf_counter()
            reconstructed_batch = engine.infer(input_batch)  # (1, 10, 1, 128, 128)
            t_infer = (time.perf_counter() - t_infer_start) * 1000.0

            # --- Stage 4: Postprocessing & Anomaly Scoring ---
            t_post_start = time.perf_counter()

            # Target most recent frame in window (index -1)
            orig_frame_gray = input_batch[0, -1, 0]  # (128, 128)
            recon_frame_gray = reconstructed_batch[0, -1, 0]  # (128, 128)

            diff = np.abs(orig_frame_gray - recon_frame_gray)
            e_t = float(np.sqrt(np.sum(diff**2)))
            error_history.append(e_t)

            # Dynamic Temporal Normalization over recent buffer
            min_e = min(error_history)
            max_e = max(error_history)
            denom = (max_e - min_e) if (max_e - min_e) > 1e-6 else 1.0
            anomaly_score = float((e_t - min_e) / denom)
            regularity_score = float(1.0 - anomaly_score)
            regularity_history.append(regularity_score)

            # Upscale difference heatmap to display dimensions
            diff_norm = np.clip(diff / (np.max(diff) + 1e-6) * 255.0, 0, 255).astype(np.uint8)
            diff_norm_resized = cv2.resize(diff_norm, (480, 360), interpolation=cv2.INTER_LINEAR)
            diff_heatmap_color = cv2.applyColorMap(diff_norm_resized, cv2.COLORMAP_JET)

            t_post = (time.perf_counter() - t_post_start) * 1000.0

            # End-to-end FPS computation
            total_frame_time = time.perf_counter() - t_start
            fps = 1.0 / max(total_frame_time, 1e-5)
            fps_history.append(fps)
            avg_fps = sum(fps_history) / len(fps_history)

            telemetry = {
                "fps": avg_fps,
                "acq_ms": t_acq,
                "pre_ms": t_pre,
                "infer_ms": t_infer,
                "post_ms": t_post,
            }

            # --- Stage 5: Render HUD ---
            hud = draw_hud(
                raw_frame=frame_display.copy(),
                diff_heatmap_color=diff_heatmap_color,
                regularity_history=regularity_history,
                regularity_score=regularity_score,
                threshold=threshold,
                telemetry=telemetry,
                frame_idx=frame_count,
            )

            # Write to output video file if configured
            if output_video_path is not None:
                if writer is None:
                    hud_h, hud_w, _ = hud.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (hud_w, hud_h))
                writer.write(hud)

            # Print telemetry periodically to console
            if frame_count % 15 == 0:
                status_str = "ANOMALY!" if regularity_score < threshold else "NORMAL"
                print(
                    f"Frame [{frame_count:04d}] | State: {status_str:<8} | "
                    f"R(t): {regularity_score:.3f} | "
                    f"FPS: {avg_fps:.1f} | Latencies (ms): Acq={t_acq:.1f}, Pre={t_pre:.1f}, TRT/Infer={t_infer:.1f}, Post={t_post:.1f}"
                )

            if display:
                try:
                    cv2.imshow("Edge Anomaly Detector HUD", hud)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("[*] User quit signal received.")
                        break
                except Exception:
                    # In headless environments, cv2.imshow may fail
                    display = False

            if max_frames is not None and frame_count >= max_frames:
                print(f"[*] Reached configured max_frames ({max_frames}). Stopping stream.")
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[+] Annotated surveillance stream saved to: {output_video_path}")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print(f"[+] Stream processing terminated. Total processed frames: {frame_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Edge Stream Anomaly Detector")
    parser.add_argument("--source", default="synthetic", help="Webcam id (0), video path, or 'synthetic'")
    parser.add_argument("--model", type=str, default="models/engine/anomaly_detector_fp16.engine")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="output_stream.mp4")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-display", action="store_true", help="Disable cv2.imshow window")
    args = parser.parse_args()

    # Convert numeric string to int for webcam index
    src: Union[int, str] = int(args.source) if args.source.isdigit() else args.source

    run_stream_inference(
        video_source=src,
        model_path=args.model,
        threshold=args.threshold,
        output_video_path=args.output,
        max_frames=args.max_frames,
        device_str=args.device,
        display=not args.no_display,
    )

