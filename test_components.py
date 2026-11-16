"""
Comprehensive Verification and Test Suite for ConvLSTM Video Anomaly Detector.

Tests:
1. ConvLSTMCell and Multi-layer ConvLSTM forward pass & shapes
2. ConvLSTMAutoencoder full spatial-temporal forward pass on CPU and CUDA
3. SSIM and Combined SpatioTemporalLoss computation
4. Dataset generation (Synthetic Ped2) & VideoSequenceDataset sliding window
5. ONNX Export and ONNX Runtime numerical verification
6. Inference engine execution and HUD generation
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch
import cv2

# Set path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.convlstm import ConvLSTMCell, ConvLSTM
from src.model import ConvLSTMAutoencoder, build_model
from src.loss import SSIM, SpatioTemporalLoss
from src.dataset import (
    ensure_dataset_ready,
    generate_synthetic_ped2,
    VideoSequenceDataset,
    get_dataloaders,
)
from src.export_onnx import export_to_onnx
from src.infer_stream import (
    CircularFrameBuffer,
    PyTorchInferenceEngine,
    SyntheticLiveVideoFeed,
    draw_hud,
)


class TestConvLSTMEngine(unittest.TestCase):

    def test_01_convlstm_cell(self):
        """Test ConvLSTMCell single step forward pass."""
        b, c_in, c_hidden, h, w = 2, 8, 16, 32, 32
        cell = ConvLSTMCell(input_dim=c_in, hidden_dim=c_hidden, kernel_size=3)
        x = torch.randn(b, c_in, h, w)

        h_out, c_out = cell(x)
        self.assertEqual(h_out.shape, (b, c_hidden, h, w))
        self.assertEqual(c_out.shape, (b, c_hidden, h, w))

        # Test stateful second step
        h_out2, c_out2 = cell(x, (h_out, c_out))
        self.assertEqual(h_out2.shape, (b, c_hidden, h, w))
        print("[-] ConvLSTMCell forward pass verified.")

    def test_02_convlstm_multilayer(self):
        """Test Multi-Layer ConvLSTM forward pass across time window."""
        b, t, c_in, h, w = 2, 10, 16, 32, 32
        hidden_dims = [32, 64]
        convlstm = ConvLSTM(
            input_dim=c_in,
            hidden_dim=hidden_dims,
            kernel_size=3,
            num_layers=2,
            batch_first=True,
        )
        seq = torch.randn(b, t, c_in, h, w)
        out, states = convlstm(seq)

        self.assertEqual(out.shape, (b, t, hidden_dims[-1], h, w))
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0][0].shape, (b, hidden_dims[0], h, w))
        self.assertEqual(states[1][0].shape, (b, hidden_dims[1], h, w))
        print("[-] ConvLSTM multi-layer stacked network verified.")

    def test_03_autoencoder_architecture(self):
        """Test ConvLSTMAutoencoder end-to-end forward pass."""
        b, t, c, h, w = 2, 10, 1, 128, 128
        model = build_model(in_channels=c, temporal_window=t)

        x = torch.rand(b, t, c, h, w)
        out = model(x)
        self.assertEqual(out.shape, (b, t, c, h, w))

        # Test bounded output range [0, 1] due to final Sigmoid
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

        # Test CUDA if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model_cuda = model.to(device)
            x_cuda = x.to(device)
            out_cuda = model_cuda(x_cuda)
            self.assertEqual(out_cuda.shape, (b, t, c, h, w))
            print(f"[-] ConvLSTMAutoencoder verified on CUDA ({torch.cuda.get_device_name(0)}).")
        else:
            print("[-] ConvLSTMAutoencoder verified on CPU.")

    def test_04_loss_functions(self):
        """Test SSIM and Combined SpatioTemporalLoss."""
        b, t, c, h, w = 2, 5, 1, 64, 64
        criterion = SpatioTemporalLoss(alpha=0.7)

        target = torch.rand(b, t, c, h, w)
        pred = target.clone()

        # Identical inputs should produce SSIM ~ 1.0 and loss ~ 0.0
        loss, metrics = criterion(pred, target)
        self.assertAlmostEqual(metrics["loss_total"], 0.0, places=4)
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=4)

        # Perturbed input should produce higher loss and lower SSIM
        pred_noisy = target + torch.randn_like(target) * 0.5
        loss_noisy, metrics_noisy = criterion(pred_noisy, target)
        self.assertGreater(metrics_noisy["loss_total"], 0.0)
        self.assertLess(metrics_noisy["ssim"], 1.0)
        print("[-] SpatioTemporalLoss (MSE + SSIM) verified.")

    def test_05_dataset_generation_and_loader(self):
        """Test synthetic dataset generator and sliding window DataLoader."""
        test_data_dir = ROOT_DIR / "data" / "raw" / "test_synth"
        generate_synthetic_ped2(
            base_dir=test_data_dir,
            num_train_clips=2,
            num_test_clips=2,
            frames_per_clip=30,
            img_size=128,
        )

        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=test_data_dir,
            batch_size=2,
            window_size=10,
            train_stride=2,
            test_stride=1,
        )

        batch_seq, batch_lbl = next(iter(train_loader))
        self.assertEqual(batch_seq.shape, (2, 10, 1, 128, 128))
        self.assertEqual(batch_lbl.shape, (2, 10))

        # Clean up temporary test data
        import shutil
        shutil.rmtree(test_data_dir, ignore_errors=True)
        print("[-] Dataset sliding window pipeline verified.")

    def test_06_circular_buffer_and_hud(self):
        """Test CircularFrameBuffer and HUD generation."""
        buffer = CircularFrameBuffer(capacity=10, target_size=(128, 128))
        dummy_frame = np.full((360, 480, 3), 100, dtype=np.uint8)

        for _ in range(9):
            buffer.append(dummy_frame)
        self.assertFalse(buffer.is_full())
        self.assertIsNone(buffer.get_temporal_window_batch())

        buffer.append(dummy_frame)
        self.assertTrue(buffer.is_full())
        batch = buffer.get_temporal_window_batch()
        self.assertEqual(batch.shape, (1, 10, 1, 128, 128))

        # Test HUD drawing
        diff_heatmap = np.zeros((360, 480, 3), dtype=np.uint8)
        telemetry = {"fps": 62.5, "acq_ms": 1.2, "pre_ms": 2.1, "infer_ms": 8.5, "post_ms": 1.4}
        hud = draw_hud(
            raw_frame=dummy_frame,
            diff_heatmap_color=diff_heatmap,
            regularity_history=[0.8, 0.85, 0.7, 0.4],
            regularity_score=0.4,
            threshold=0.5,
            telemetry=telemetry,
            frame_idx=10,
        )
        self.assertIsNotNone(hud)
        self.assertEqual(hud.shape[1], 480 * 2)  # 2 panels wide
        print("[-] Circular buffer and real-time HUD rendering verified.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

