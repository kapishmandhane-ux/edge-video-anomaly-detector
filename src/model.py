"""
Spatio-Temporal ConvLSTM Autoencoder Architecture.

Combines:
1. Spatial Encoder: Time-Distributed Conv2D feature extractors.
2. Temporal Bottleneck: 4-layer stacked ConvLSTM network.
3. Spatial Decoder: Time-Distributed ConvTranspose2D reconstruction network.

Produces a pixel-level reconstructed sequence (B, T, 1, 128, 128) with values in [0, 1].
Fully traceable and compatible with ONNX opset 17 and TensorRT engines.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn

try:
    from .convlstm import ConvLSTM
except ImportError:
    from convlstm import ConvLSTM


class ConvLSTMAutoencoder(nn.Module):
    """
    Spatio-Temporal Autoencoder for Video Anomaly Detection.

    Input shape:  (B, T, 1, 128, 128)
    Output shape: (B, T, 1, 128, 128)
    """

    def __init__(
        self,
        in_channels: int = 1,
        temporal_window: int = 10,
        leakiness: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.temporal_window = temporal_window

        # -------------------------------------------------------------
        # 1. Spatial Encoder (Time-Distributed 2D Convolutions)
        # (B*T, 1, 128, 128) -> (B*T, 128, 32, 32)
        # -------------------------------------------------------------
        self.encoder = nn.Sequential(
            # Stage 1: (1 -> 32, k=3, s=1, p=1) -> (32, 128, 128)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(leakiness, inplace=True),

            # Stage 2: (32 -> 64, k=3, s=2, p=1) -> (64, 64, 64)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leakiness, inplace=True),

            # Stage 3: (64 -> 128, k=3, s=2, p=1) -> (128, 32, 32)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(leakiness, inplace=True),
        )

        # -------------------------------------------------------------
        # 2. Temporal Bottleneck (Multi-Layer ConvLSTM)
        # 4 Stacked Layers:
        # Layer 1: 128 -> 64
        # Layer 2: 64  -> 32
        # Layer 3: 32  -> 64
        # Layer 4: 64  -> 128
        # Preserves spatial resolution (32, 32) across temporal steps
        # -------------------------------------------------------------
        self.bottleneck = ConvLSTM(
            input_dim=128,
            hidden_dim=[64, 32, 64, 128],
            kernel_size=3,
            num_layers=4,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )

        # -------------------------------------------------------------
        # 3. Spatial Decoder (Time-Distributed Transposed 2D Convolutions)
        # (B*T, 128, 32, 32) -> (B*T, 1, 128, 128)
        # -------------------------------------------------------------
        self.decoder = nn.Sequential(
            # Stage 1: (128 -> 64, k=3, s=2, p=1, out_p=1) -> (64, 64, 64)
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leakiness, inplace=True),

            # Stage 2: (64 -> 32, k=3, s=2, p=1, out_p=1) -> (32, 128, 128)
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(leakiness, inplace=True),

            # Stage 3: (32 -> 1, k=3, s=1, p=1) + Sigmoid -> (1, 128, 128)
            nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequence tensor of shape (B, T, C, H, W)
               typically (B, 10, 1, 128, 128)

        Returns:
            reconstructed: Reconstructed sequence of shape (B, T, C, H, W)
        """
        b, t, c, h, w = x.shape

        # 1. Flatten temporal dimension into batch for Spatial 2D Encoder
        x_reshaped = x.view(b * t, c, h, w)
        encoded = self.encoder(x_reshaped)  # (B*T, 128, 32, 32)

        # 2. Reshape back to temporal sequence for ConvLSTM Bottleneck
        _, enc_c, enc_h, enc_w = encoded.shape
        encoded_seq = encoded.view(b, t, enc_c, enc_h, enc_w)

        bottleneck_out, _ = self.bottleneck(encoded_seq)  # (B, T, 128, 32, 32)

        # 3. Flatten temporal dimension for Spatial 2D Decoder
        bottleneck_reshaped = bottleneck_out.contiguous().view(b * t, enc_c, enc_h, enc_w)
        reconstructed = self.decoder(bottleneck_reshaped)  # (B*T, 1, 128, 128)

        # 4. Reshape back to (B, T, C, H, W)
        reconstructed_seq = reconstructed.view(b, t, c, h, w)

        return reconstructed_seq


def build_model(
    in_channels: int = 1,
    temporal_window: int = 10,
    device: Optional[torch.device] = None,
) -> ConvLSTMAutoencoder:
    """Factory helper to construct and initialize the ConvLSTM Autoencoder."""
    model = ConvLSTMAutoencoder(in_channels=in_channels, temporal_window=temporal_window)
    if device is not None:
        model = model.to(device)
    return model

