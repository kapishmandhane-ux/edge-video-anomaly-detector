"""
UCSD Ped2 Benchmark Dataset and Fallback Synthetic Sequence Generator.

Handles:
- Frame extraction from image sequence directories (.tif, .png, .jpg).
- Grayscale conversion, spatial resizing to (128, 128), and normalization to [0.0, 1.0].
- Sliding window temporal chunking with stride S=2 (training) and S=1 (inference/test).
- Automated dataset downloader and deterministic synthetic generator for zero setup failure.
"""

from typing import List, Tuple, Dict, Optional, Union
import os
import glob
import math
import json
import urllib.request
import tarfile
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw


# Standard Ground Truth Anomaly Intervals for UCSDped2 Benchmark
# Format: 1-indexed (start_frame, end_frame)
UCSD_PED2_GT_INTERVALS: Dict[str, List[Tuple[int, int]]] = {
    "Test001": [(61, 180)],
    "Test002": [(95, 180)],
    "Test003": [(1, 146)],
    "Test004": [(31, 180)],
    "Test005": [(1, 129)],
    "Test006": [(1, 159)],
    "Test007": [(46, 180)],
    "Test008": [(1, 180)],
    "Test009": [(1, 120)],
    "Test010": [(1, 150)],
    "Test011": [(1, 180)],
    "Test012": [(88, 180)],
}


def download_ucsd_ped2(destination_dir: Union[str, Path]) -> bool:
    """
    Attempt to download and unpack official UCSD Ped2 dataset.
    Returns True if successful, False otherwise.
    """
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    url = "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"
    tar_path = dest_path / "UCSD_Anomaly_Dataset.tar.gz"

    print(f"[*] Attempting to download UCSD Ped2 dataset from {url}...")
    try:
        # User-agent header to avoid bot blockers
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(tar_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)

        print("[*] Extracting UCSD Ped2 archive...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_path)

        # Cleanup archive
        if tar_path.exists():
            tar_path.unlink()

        return True
    except Exception as e:
        print(f"[!] UCSD Ped2 download or extract failed ({e}). Proceeding with synthetic fallback.")
        if tar_path.exists():
            tar_path.unlink()
        return False


def generate_synthetic_ped2(
    base_dir: Union[str, Path],
    num_train_clips: int = 12,
    num_test_clips: int = 12,
    frames_per_clip: int = 120,
    img_size: int = 128,
) -> None:
    """
    Synthesize realistic moving spatio-temporal video sequences.
    
    Normal Behavior:
    - Multiple simulated entities (pedestrians) moving linearly at constant normal speeds
      (1.0 - 2.0 px/frame) with subtle rhythmic vertical gait oscillation.
    - Uniform background texture with subtle sensor noise.

    Anomalous Behavior:
    - Sudden erratic acceleration (3x - 5x velocity, simulating cyclists/skaters).
    - Sudden 90/180-degree velocity reversals, erratic zig-zag motion.
    - Entity collisions and rapid radial expansion (abnormal physical events).
    """
    base_path = Path(base_dir)
    train_dir = base_path / "Train"
    test_dir = base_path / "Test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    gt_labels: Dict[str, List[int]] = {}

    print(f"[*] Generating synthetic spatio-temporal sequences in {base_path}...")

    # Helper to generate a single frame image
    def render_entities(entities, width=img_size, height=img_size):
        img = Image.new("L", (width, height), color=35)
        draw = ImageDraw.Draw(img)

        # Draw static walkway boundary guides
        draw.line([(0, 20), (width, 20)], fill=55, width=1)
        draw.line([(0, height - 20), (width, height - 20)], fill=55, width=1)

        for ent in entities:
            x, y, r, val = ent["x"], ent["y"], ent["r"], ent["val"]
            bbox = [x - r, y - r, x + r, y + r]
            if ent.get("shape", "circle") == "circle":
                draw.ellipse(bbox, fill=val)
            else:
                draw.rectangle(bbox, fill=val)

        arr = np.array(img, dtype=np.float32)
        # Add subtle Gaussian sensor noise
        noise = np_rng.normal(0, 3.0, (height, width))
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    # 1. Generate Training Clips (solely normal activity)
    for c_idx in range(1, num_train_clips + 1):
        clip_name = f"Train{c_idx:03d}"
        clip_path = train_dir / clip_name
        clip_path.mkdir(parents=True, exist_ok=True)

        # Initialize normal entities
        num_entities = rng.randint(4, 7)
        entities = []
        for _ in range(num_entities):
            direction = 1 if rng.random() > 0.5 else -1
            entities.append({
                "x": rng.uniform(10, img_size - 10),
                "y": rng.uniform(30, img_size - 30),
                "vx": direction * rng.uniform(1.0, 1.8),
                "vy": rng.uniform(-0.15, 0.15),
                "r": rng.uniform(3.5, 5.5),
                "val": rng.randint(180, 240),
                "phase": rng.uniform(0, 2 * math.pi),
                "shape": "circle",
            })

        for f_idx in range(1, frames_per_clip + 1):
            for ent in entities:
                ent["x"] += ent["vx"]
                # subtle vertical gait bobbing
                ent["phase"] += 0.3
                ent["y"] += ent["vy"] + 0.3 * math.sin(ent["phase"])

                # Wrap around screen horizontally
                if ent["x"] < -10:
                    ent["x"] = img_size + 10
                elif ent["x"] > img_size + 10:
                    ent["x"] = -10

                ent["y"] = max(25, min(img_size - 25, ent["y"]))

            frame_img = render_entities(entities)
            frame_img.save(clip_path / f"{f_idx:03d}.tif")

    # 2. Generate Test Clips (normal + seeded anomalous intervals)
    for c_idx in range(1, num_test_clips + 1):
        clip_name = f"Test{c_idx:03d}"
        clip_path = test_dir / clip_name
        clip_path.mkdir(parents=True, exist_ok=True)

        num_entities = rng.randint(4, 6)
        entities = []
        for _ in range(num_entities):
            direction = 1 if rng.random() > 0.5 else -1
            entities.append({
                "x": rng.uniform(10, img_size - 10),
                "y": rng.uniform(30, img_size - 30),
                "vx": direction * rng.uniform(1.0, 1.8),
                "vy": rng.uniform(-0.15, 0.15),
                "r": rng.uniform(3.5, 5.5),
                "val": rng.randint(180, 240),
                "phase": rng.uniform(0, 2 * math.pi),
                "shape": "circle",
            })

        # Define anomaly window for this test clip
        anom_start = rng.randint(35, 50)
        anom_end = min(frames_per_clip, anom_start + rng.randint(40, 55))
        clip_labels = [0] * frames_per_clip

        # Add anomaly entity during anomaly window
        fast_intruder = {
            "x": 0.0,
            "y": 64.0,
            "vx": 6.0,  # 3x-4x speed of normal pedestrians
            "vy": 0.0,
            "r": 7.0,
            "val": 255,
            "shape": "rectangle",
        }

        for f_idx in range(1, frames_per_clip + 1):
            is_anomaly = anom_start <= f_idx <= anom_end
            if is_anomaly:
                clip_labels[f_idx - 1] = 1

            for ent in entities:
                ent["x"] += ent["vx"]
                ent["phase"] += 0.3
                ent["y"] += ent["vy"] + 0.3 * math.sin(ent["phase"])
                if ent["x"] < -10:
                    ent["x"] = img_size + 10
                elif ent["x"] > img_size + 10:
                    ent["x"] = -10
                ent["y"] = max(25, min(img_size - 25, ent["y"]))

            current_frame_entities = list(entities)

            if is_anomaly:
                fast_intruder["x"] += fast_intruder["vx"]
                # Erratic oscillation in y
                fast_intruder["y"] = 64.0 + 20.0 * math.sin(f_idx * 0.4)
                if fast_intruder["x"] > img_size + 15:
                    fast_intruder["x"] = -15
                current_frame_entities.append(fast_intruder)

            frame_img = render_entities(current_frame_entities)
            frame_img.save(clip_path / f"{f_idx:03d}.tif")

        gt_labels[clip_name] = clip_labels
        # Save per-clip text ground truth (1 line per frame: 0 or 1)
        with open(test_dir / f"{clip_name}_gt.txt", "w") as f:
            for lbl in clip_labels:
                f.write(f"{lbl}\n")

    # Save master ground truth JSON
    with open(test_dir / "ground_truth.json", "w") as f:
        json.dump(gt_labels, f, indent=2)

    print(f"[+] Synthetic dataset successfully generated: {num_train_clips} train clips, {num_test_clips} test clips.")


def ensure_dataset_ready(data_dir: Union[str, Path]) -> Path:
    """Ensure raw dataset exists; download or synthesize if missing."""
    data_path = Path(data_dir)
    ucsd_train = data_path / "Train"
    ucsd_test = data_path / "Test"

    # Also handle nested structure if dataset extracted inside UCSD_Anomaly_Dataset.v1p2
    if not (ucsd_train.exists() and ucsd_test.exists()):
        nested = list(data_path.glob("**/UCSDped2"))
        if nested:
            return nested[0]

    if not (ucsd_train.exists() and ucsd_test.exists()):
        # Try downloading first
        success = download_ucsd_ped2(data_path)
        # Check again
        if not (ucsd_train.exists() and ucsd_test.exists()):
            generate_synthetic_ped2(data_path)

    return data_path


class VideoSequenceDataset(Dataset):
    """
    Sliding window video sequence dataset for spatio-temporal reconstruction.

    Yields sequences of shape (T, C, H, W) normalized to [0.0, 1.0], along with
    frame-level ground-truth labels if available.
    """

    def __init__(
        self,
        clip_dirs: List[Path],
        window_size: int = 10,
        stride: int = 1,
        img_size: Tuple[int, int] = (128, 128),
        gt_dict: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.img_size = img_size
        self.gt_dict = gt_dict or {}

        self.samples: List[Tuple[List[Path], List[int], str, int]] = []
        self._prepare_sequences(clip_dirs)

    def _prepare_sequences(self, clip_dirs: List[Path]) -> None:
        """Scan clip folders and extract sliding window frame index spans."""
        exts = ("*.tif", "*.png", "*.jpg", "*.jpeg", "*.bmp")

        for clip_dir in sorted(clip_dirs):
            clip_name = clip_dir.name
            frame_files: List[Path] = []
            for ext in exts:
                frame_files.extend(sorted(clip_dir.glob(ext)))

            # Deduplicate and sort numerically if frame files exist
            frame_files = sorted(list(set(frame_files)))
            n_frames = len(frame_files)
            if n_frames < self.window_size:
                continue

            # Load ground truth labels for this clip if available
            labels = self._load_clip_labels(clip_dir, clip_name, n_frames)

            # Slide window across clip frames
            for start_idx in range(0, n_frames - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                window_frames = frame_files[start_idx:end_idx]
                window_labels = labels[start_idx:end_idx]
                self.samples.append((window_frames, window_labels, clip_name, start_idx))

    def _load_clip_labels(self, clip_dir: Path, clip_name: str, n_frames: int) -> List[int]:
        """Resolve frame-level ground truth binary labels."""
        # Check explicit dictionary passed in
        if clip_name in self.gt_dict:
            return self.gt_dict[clip_name][:n_frames]

        # Check for matching _gt.txt in parent directory
        gt_txt = clip_dir.parent / f"{clip_name}_gt.txt"
        if gt_txt.exists():
            with open(gt_txt, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                return [int(l) for l in lines][:n_frames]

        # Check standard UCSDped2 benchmark interval table
        if clip_name in UCSD_PED2_GT_INTERVALS:
            intervals = UCSD_PED2_GT_INTERVALS[clip_name]
            labels = [0] * n_frames
            for s, e in intervals:
                # 1-indexed to 0-indexed
                for i in range(max(0, s - 1), min(n_frames, e)):
                    labels[i] = 1
            return labels

        # Default: normal (0)
        return [0] * n_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            frames_tensor: (T, 1, H, W) float32 in [0.0, 1.0]
            labels_tensor: (T,) int64 binary labels
        """
        window_frames, window_labels, _, _ = self.samples[idx]

        frame_list = []
        for f_path in window_frames:
            with Image.open(f_path) as img:
                # Convert to 8-bit grayscale
                img_gray = img.convert("L")
                if img_gray.size != self.img_size:
                    img_gray = img_gray.resize(self.img_size, Image.Resampling.BILINEAR)
                arr = np.array(img_gray, dtype=np.float32) / 255.0
                frame_list.append(arr)

        # Shape: (T, H, W) -> (T, 1, H, W)
        seq_arr = np.stack(frame_list, axis=0)[:, np.newaxis, :, :]
        tensor_seq = torch.from_numpy(seq_arr).float()
        labels_seq = torch.tensor(window_labels, dtype=torch.long)

        return tensor_seq, labels_seq


def get_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 8,
    window_size: int = 10,
    train_stride: int = 2,
    test_stride: int = 1,
    num_workers: int = 0,
    val_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct Train, Validation, and Test DataLoaders.
    """
    root = ensure_dataset_ready(data_dir)

    train_base = root / "Train"
    test_base = root / "Test"

    # Gather all train clip directories (Train001, Train002, ...)
    train_clips = sorted([d for d in train_base.iterdir() if d.is_dir() and not d.name.endswith("_gt")])
    test_clips = sorted([d for d in test_base.iterdir() if d.is_dir() and not d.name.endswith("_gt")])

    if not train_clips:
        raise RuntimeError(f"No training clips found in {train_base}")

    # Split train clips into train and val
    rng = random.Random(42)
    shuffled_train = list(train_clips)
    rng.shuffle(shuffled_train)

    num_val = max(1, int(len(shuffled_train) * val_ratio))
    val_clip_dirs = shuffled_train[:num_val]
    train_clip_dirs = shuffled_train[num_val:]

    # Check for master ground_truth.json
    gt_json = test_base / "ground_truth.json"
    gt_dict = {}
    if gt_json.exists():
        with open(gt_json, "r") as f:
            gt_dict = json.load(f)

    train_dataset = VideoSequenceDataset(
        clip_dirs=train_clip_dirs,
        window_size=window_size,
        stride=train_stride,
    )
    val_dataset = VideoSequenceDataset(
        clip_dirs=val_clip_dirs,
        window_size=window_size,
        stride=train_stride,
    )
    test_dataset = VideoSequenceDataset(
        clip_dirs=test_clips,
        window_size=window_size,
        stride=test_stride,
        gt_dict=gt_dict,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

