"""
Production Training Harness for Spatio-Temporal ConvLSTM Autoencoder.

Features:
- AdamW optimizer with CosineAnnealingLR scheduling
- Gradient clipping (max_norm=1.0)
- Validation tracking with top-3 model checkpointing
- Telemetry logging: Loss breakdown (MSE, SSIM), epoch progress, GPU VRAM memory usage
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from .model import build_model, ConvLSTMAutoencoder
    from .loss import SpatioTemporalLoss
    from .dataset import get_dataloaders
except ImportError:
    from model import build_model, ConvLSTMAutoencoder
    from loss import SpatioTemporalLoss
    from dataset import get_dataloaders


class CheckpointManager:
    """Maintains the top-K model checkpoints based on lowest validation loss."""

    def __init__(self, checkpoint_dir: Path, max_keep: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        # List of tuples: (val_loss, epoch, filepath)
        self.best_checkpoints: List[Tuple[float, int, Path]] = []

    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object],
        epoch: int,
        val_loss: float,
        metrics: Dict[str, float],
    ) -> bool:
        """
        Save checkpoint if it qualifies as top-K.
        Always updates best_model.pt and latest_model.pt.
        """
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
            "metrics": metrics,
        }

        # Save latest
        latest_path = self.checkpoint_dir / "latest_model.pt"
        torch.save(checkpoint_state, latest_path)

        # Check if this qualifies for top-K
        is_best = False
        ckpt_filename = f"model_epoch_{epoch:03d}_val_{val_loss:.5f}.pt"
        ckpt_path = self.checkpoint_dir / ckpt_filename

        if len(self.best_checkpoints) < self.max_keep or val_loss < self.best_checkpoints[-1][0]:
            torch.save(checkpoint_state, ckpt_path)
            self.best_checkpoints.append((val_loss, epoch, ckpt_path))
            self.best_checkpoints.sort(key=lambda x: x[0])

            # Prune worse checkpoints beyond max_keep
            if len(self.best_checkpoints) > self.max_keep:
                worst = self.best_checkpoints.pop(-1)
                if worst[2].exists() and worst[2] != ckpt_path:
                    try:
                        worst[2].unlink()
                    except OSError:
                        pass

            # Update best_model.pt
            if self.best_checkpoints[0][1] == epoch:
                is_best = True
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint_state, best_path)

        return is_best


def get_vram_usage() -> float:
    """Return current CUDA VRAM memory allocated in Megabytes (MB)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    return 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: SpatioTemporalLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 5,
) -> Dict[str, float]:
    """Execute one training epoch with gradient clipping."""
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_ssim = 0.0
    total_batches = len(loader)

    for step, (batch_seq, _) in enumerate(loader, start=1):
        batch_seq = batch_seq.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        reconstructed = model(batch_seq)

        loss, metrics = criterion(reconstructed, batch_seq)
        loss.backward()

        # Gradient clipping to stabilize recurrent ConvLSTM gates
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += metrics["loss_total"]
        running_mse += metrics["loss_mse"]
        running_ssim += metrics["loss_ssim"]

        if step % log_interval == 0 or step == total_batches:
            vram_mb = get_vram_usage()
            print(
                f"[Train] Epoch [{epoch:03d}] Step [{step:03d}/{total_batches:03d}] | "
                f"Loss: {metrics['loss_total']:.5f} | "
                f"MSE: {metrics['loss_mse']:.5f} | "
                f"SSIM: {metrics['ssim']:.4f} | "
                f"VRAM: {vram_mb:.1f} MB"
            )

    return {
        "loss_total": running_loss / max(1, total_batches),
        "loss_mse": running_mse / max(1, total_batches),
        "loss_ssim": running_ssim / max(1, total_batches),
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: SpatioTemporalLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Execute validation over normal validation sequences."""
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_ssim = 0.0
    running_ssim_val = 0.0
    total_batches = len(loader)

    for batch_seq, _ in loader:
        batch_seq = batch_seq.to(device, non_blocking=True)
        reconstructed = model(batch_seq)
        _, metrics = criterion(reconstructed, batch_seq)

        running_loss += metrics["loss_total"]
        running_mse += metrics["loss_mse"]
        running_ssim += metrics["loss_ssim"]
        running_ssim_val += metrics["ssim"]

    return {
        "val_loss": running_loss / max(1, total_batches),
        "val_mse": running_mse / max(1, total_batches),
        "val_ssim_loss": running_ssim / max(1, total_batches),
        "val_ssim": running_ssim_val / max(1, total_batches),
    }


def train_pipeline(
    data_dir: str = "data/raw/UCSDped2",
    checkpoints_dir: str = "models/checkpoints",
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    device_str: Optional[str] = None,
) -> str:
    """Full training pipeline execution."""
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[*] Training on target hardware device: {device}")
    if device.type == "cuda":
        print(f"[*] Device name: {torch.cuda.get_device_name(0)}")

    ckpt_dir = Path(checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        window_size=10,
        train_stride=2,
    )
    print(f"[+] Loaded {len(train_loader)} train batches, {len(val_loader)} validation batches.")

    # 2. Build Model
    model = build_model(in_channels=1, temporal_window=10, device=device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Model constructed. Total trainable parameters: {num_params:,}")

    # 3. Setup Optimizer, Scheduler, and Criterion
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = SpatioTemporalLoss(alpha=0.7)

    checkpoint_manager = CheckpointManager(checkpoint_dir=ckpt_dir, max_keep=3)
    history = []

    print("[*] Starting training loop...")
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        val_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = checkpoint_manager.update(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            val_loss=val_metrics["val_loss"],
            metrics=val_metrics,
        )

        best_flag = " [BEST MODEL SAVED]" if is_best else ""
        print(
            f"--> Epoch [{epoch:03d}/{epochs:03d}] Completed | "
            f"Train Loss: {train_metrics['loss_total']:.5f} | "
            f"Val Loss: {val_metrics['val_loss']:.5f} (SSIM: {val_metrics['val_ssim']:.4f}) | "
            f"LR: {current_lr:.2e}{best_flag}\n"
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": current_lr,
        })

    # Save training history
    history_path = ckpt_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    best_ckpt = ckpt_dir / "best_model.pt"
    print(f"[+] Training complete. Best model saved at: {best_ckpt}")
    return str(best_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConvLSTM Autoencoder Training")
    parser.add_argument("--data-dir", type=str, default="data/raw/UCSDped2")
    parser.add_argument("--checkpoints-dir", type=str, default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        checkpoints_dir=args.checkpoints_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device_str=args.device,
    )

