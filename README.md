# Real-Time Edge Video Anomaly Detection Engine
### ConvLSTM Autoencoder + TensorRT & Edge GPU Acceleration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Opset 17](https://img.shields.io/badge/ONNX-Opset%2017-005ced.svg)](https://onnx.ai/)
[![NVIDIA TensorRT](https://img.shields.io/badge/NVIDIA-TensorRT%20FP16-76b900.svg)](https://developer.nvidia.com/tensorrt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade Spatio-Temporal Video Anomaly Detection engine engineered for high-throughput edge deployment (60+ FPS on edge NVIDIA GPUs). The system utilizes an unsupervised ConvLSTM Autoencoder trained solely on normal video sequences to reconstruct incoming video frames. When anomalous events occur (such as cyclists on pedestrian walkways, sudden falls, or erratic trajectory deviations), reconstruction divergence triggers a dynamic pixel-level heatmap and real-time Heads-Up Display (HUD) alarm.

---

## 1. System Architecture

```text
Input Sequence [B, 10, 1, 128, 128]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Spatial Encoder (Time-Distributed 2D Convolutions)    │
│  - Conv2D (1 -> 32, k=3, s=1, p=1) + BN + LeakyReLU    │
│  - Conv2D (32 -> 64, k=3, s=2, p=1) + BN + LeakyReLU   │
│  - Conv2D (64 -> 128, k=3, s=2, p=1) + BN + LeakyReLU  │
└────────────────────────────────────────────────────────┘
       │  Spatial Latent Representation [B, 10, 128, 32, 32]
       ▼
┌────────────────────────────────────────────────────────┐
│  Temporal Bottleneck (4-Layer Stacked ConvLSTM)        │
│  - Layer 1: ConvLSTM (128 -> 64, k=3)                  │
│  - Layer 2: ConvLSTM (64 -> 32, k=3)                   │
│  - Layer 3: ConvLSTM (32 -> 64, k=3)                   │
│  - Layer 4: ConvLSTM (64 -> 128, k=3)                  │
└────────────────────────────────────────────────────────┘
       │  Bottleneck State [B, 10, 128, 32, 32]
       ▼
┌────────────────────────────────────────────────────────┐
│  Spatial Decoder (Time-Distributed Transposed 2D Conv) │
│  - ConvTranspose2D (128 -> 64, k=3, s=2, p=1) + BN     │
│  - ConvTranspose2D (64 -> 32, k=3, s=2, p=1) + BN      │
│  - Conv2D (32 -> 1, k=3, s=1, p=1) + Sigmoid           │
└────────────────────────────────────────────────────────┘
       │
       ▼
Reconstructed Sequence [B, 10, 1, 128, 128]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Edge Inference, Dynamic Scoring & Real-Time HUD       │
│  - L2 Pixel Reconstruction Error: e(t) = ||I - Î||     │
│  - Normalized Anomaly Score: S_a(t) ∈ [0, 1]           │
│  - Regularity Score: R(t) = 1.0 - S_a(t)               │
│  - JET Colormap Heatmap Overlay + Flashing Alert       │
└────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulation

### 2.1 ConvLSTM Recurrent Unit
$$\begin{pmatrix} i_t \\ f_t \\ g_t \\ o_t \end{pmatrix} = \begin{pmatrix} \sigma \\ \sigma \\ \tanh \\ \sigma \end{pmatrix} \left( W * [X_t, H_{t-1}] + b \right)$$

$$C_t = f_t \odot C_{t-1} + i_t \odot g_t$$
$$H_t = o_t \odot \tanh(C_t)$$

### 2.2 Objective Loss Function
Trained using a weighted combination of Mean Squared Error (MSE) and Structural Similarity (SSIM):
$$\mathcal{L}_{total} = \alpha \cdot \text{MSE}(\mathbf{X}, \hat{\mathbf{X}}) + (1 - \alpha) \cdot (1 - \text{SSIM}(\mathbf{X}, \hat{\mathbf{X}})) \quad (\alpha = 0.7)$$

Where 2D SSIM is computed via Gaussian filtering:
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

### 2.3 Regularity & Anomaly Score
Reconstruction divergence for frame $I_t$:
$$e(t) = \Vert{}I_t - \hat{I}_t\Vert{}_2 = \sqrt{\sum_{h=1}^H \sum_{w=1}^W (I_t(h, w) - \hat{I}_t(h, w))^2}$$

Normalized over a sliding temporal buffer $\mathcal{W}$:
$$S_a(t) = \frac{e(t) - \min_{\tau \in \mathcal{W}} e(\tau)}{\max_{\tau \in \mathcal{W}} e(\tau) - \min_{\tau \in \mathcal{W}} e(\tau)}$$

$$R(t) = 1.0 - S_a(t)$$
An alarm is triggered whenever $R(t) < \theta_{threshold}$ (calibrated via EER).

---

## 3. Benchmark Results & Edge Performance

Evaluated on 1,332 test frames with 595 anomalous ground-truth frames:

| Metric | Result | Description |
| :--- | :--- | :--- |
| **GPU Target** | NVIDIA GeForce RTX 4050 Laptop GPU | Edge / Embedded GPU benchmark |
| **Parameters** | 1,845,889 | Compact spatio-temporal model |
| **Inference Latency** | **12.01 ms** | TensorRT / CUDA execution |
| **Throughput** | **83.3 FPS** | Exceeds 60 FPS real-time streaming target |
| **ROC-AUC** | **0.9479** | Benchmark discrimination accuracy |
| **Equal Error Rate (EER)** | **0.1057** | False acceptance / rejection trade-off |
| **Optimal EER Threshold** | **0.3624** | Calibrated operational threshold |
| **Best F1 Score** | **0.9249** | At threshold 0.4707 |
| **ONNX Verification** | Max error: `1e-6` | Numerical equivalence with PyTorch |

---

## 4. Repository Structure

```text
edge-video-anomaly-detector/
├── .gitignore                         # Standard exclusion rules
├── README.md                          # Architecture & user manual
├── requirements.txt                   # Production dependencies
├── run_pipeline.py                    # Master CLI orchestration pipeline
├── test_components.py                 # Comprehensive unit test suite
├── src/
│   ├── __init__.py
│   ├── convlstm.py                    # ConvLSTMCell & multi-layer stacked network
│   ├── model.py                       # Spatio-Temporal ConvLSTM Autoencoder
│   ├── loss.py                        # Pure PyTorch SSIM + MSE loss criterion
│   ├── dataset.py                     # Sliding window loader & synthetic fallback
│   ├── train.py                       # AdamW training loop with top-3 checkpointing
│   ├── evaluate.py                    # ROC-AUC, EER, and threshold calibration
│   ├── export_onnx.py                 # ONNX opset 17 export & numerical validation
│   ├── build_engine.py                # TensorRT builder (FP16/INT8) & edge fallback
│   └── infer_stream.py                # Real-time circular buffer stream runner & HUD
├── models/
│   ├── checkpoints/                   # Model weight checkpoints
│   ├── engine/                        # Compiled TensorRT plans & configs
│   ├── onnx/                          # Exported ONNX models
│   ├── evaluation_metrics.json        # Evaluation metrics log
│   └── evaluation_roc.png             # ROC curve visualization
└── data/
    ├── raw/UCSDped2/                  # UCSD Ped2 benchmark dataset
    └── processed/
```

---

## 5. Installation & Setup

### 5.1 Clone & Setup Environment
```bash
git clone https://github.com/kapishmandhane-ux/edge-video-anomaly-detector.git
cd edge-video-anomaly-detector

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Hardware Acceleration
For NVIDIA CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 6. Execution Guide

The master orchestrator [`run_pipeline.py`](run_pipeline.py) provides unified CLI access to all pipeline stages:

### 6.1 Run Complete Pipeline End-to-End
```bash
python run_pipeline.py --all
```

### 6.2 Train Model
```bash
python run_pipeline.py --train --epochs 15 --batch-size 4 --lr 1e-4
```
*Features:* Top-3 checkpointing, CosineAnnealingLR scheduling, gradient clipping (`max_norm=1.0`), and VRAM telemetry.

### 6.3 Benchmark Evaluation
```bash
python run_pipeline.py --evaluate
```
*Features:* Computes ROC-AUC, EER, Best F1 score, and saves [`evaluation_roc.png`](models/evaluation_roc.png).

### 6.4 Export to ONNX
```bash
python run_pipeline.py --export
```
*Features:* Exports to ONNX (opset 17) with dynamic batch dimensions, runs `onnx.checker`, and verifies numeric equivalence against native PyTorch.

### 6.5 Compile TensorRT Engine
```bash
# FP16 Precision
python run_pipeline.py --build-engine --precision fp16

# INT8 Precision (runs IInt8EntropyCalibrator2)
python run_pipeline.py --build-engine --precision int8
```

### 6.6 Real-Time Stream Inference & Heads-Up Display
```bash
# From physical webcam (index 0)
python run_pipeline.py --stream --source 0

# From synthetic animated generator with periodic anomalies
python run_pipeline.py --stream --source synthetic --stream-output output_stream.mp4

# From RTSP network camera
python run_pipeline.py --stream --source "rtsp://admin:pass@192.168.1.50:554/live"
```

---

## 7. Unit Tests

Run the complete test suite:
```bash
python test_components.py
```
Validates:
1. ConvLSTMCell single step forward pass
2. Multi-Layer ConvLSTM temporal unrolling
3. ConvLSTMAutoencoder end-to-end forward pass on CUDA
4. SSIM & combined loss function
5. Synthetic video generation & sliding window dataset loading
6. Thread-safe CircularFrameBuffer and real-time HUD rendering

---

## 8. License

MIT License. Designed and developed for edge computing and autonomous video analytics.
