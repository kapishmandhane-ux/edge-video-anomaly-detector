"""
Combined MSE and SSIM Loss Criterion for Spatio-Temporal Sequences.

Implements pure PyTorch Structural Similarity Index (SSIM) using Gaussian 2D
filtering and combines it with Mean Squared Error (MSE):
    L_total = alpha * MSE(X, X_hat) + (1 - alpha) * (1 - SSIM(X, X_hat))
where alpha = 0.7 by default.
"""

from typing import Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_gaussian_window_1d(window_size: int, sigma: float) -> torch.Tensor:
    """Generate 1D normalized Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size // 2)
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    return gauss / gauss.sum()


def create_gaussian_window_2d(
    window_size: int = 11, sigma: float = 1.5, channels: int = 1
) -> torch.Tensor:
    """Generate 2D Gaussian window for depthwise convolution."""
    _1d = create_gaussian_window_1d(window_size, sigma)
    _2d = (_1d.unsqueeze(1) @ _1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


class SSIM(nn.Module):
    """
    Differentiable Structural Similarity (SSIM) module for 2D images and temporal sequences.
    Operates on tensors normalized to [0, 1].
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 1,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels
        self.data_range = data_range

        # Constants from SSIM paper
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2

        window = create_gaussian_window_2d(window_size, sigma, channels)
        self.register_buffer("window", window)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute mean SSIM between img1 and img2.
        Accepts:
            (N, C, H, W) or (B, T, C, H, W)
        """
        if img1.dim() == 5:
            b, t, c, h, w = img1.shape
            img1 = img1.view(b * t, c, h, w)
            img2 = img2.view(b * t, c, h, w)

        _, c, _, _ = img1.shape
        window = self.window.to(dtype=img1.dtype, device=img1.device)
        if c != self.channels:
            window = create_gaussian_window_2d(self.window_size, self.sigma, c).to(
                dtype=img1.dtype, device=img1.device
            )

        pad = self.window_size // 2

        mu1 = F.conv2d(img1, window, padding=pad, groups=c)
        mu2 = F.conv2d(img2, window, padding=pad, groups=c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=c) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=c) - mu1_mu2

        # Guard against minor floating point precision issues causing negative variance
        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

        numerator = (2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)
        denominator = (mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2)

        ssim_map = numerator / (denominator + 1e-8)
        return ssim_map.mean()


class SpatioTemporalLoss(nn.Module):
    """
    Combined Reconstruction Loss:
        L = alpha * MSE + (1 - alpha) * (1 - SSIM)
    """

    def __init__(
        self,
        alpha: float = 0.7,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 1,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.mse_fn = nn.MSELoss()
        self.ssim_fn = SSIM(window_size=window_size, sigma=sigma, channels=channels)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            pred: Reconstructed sequence (B, T, C, H, W)
            target: Ground-truth sequence (B, T, C, H, W)

        Returns:
            total_loss: scalar loss tensor
            metrics: dict with detached float values for 'mse', 'ssim', and 'total'
        """
        mse_loss = self.mse_fn(pred, target)
        ssim_val = self.ssim_fn(pred, target)
        ssim_loss = 1.0 - ssim_val

        total_loss = self.alpha * mse_loss + (1.0 - self.alpha) * ssim_loss

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_mse": float(mse_loss.item()),
            "loss_ssim": float(ssim_loss.item()),
            "ssim": float(ssim_val.item()),
        }
        return total_loss, metrics

