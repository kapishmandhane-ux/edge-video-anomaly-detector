"""
Master Orchestration Pipeline for Edge Video Anomaly Detection Engine.

Coordinates the end-to-end lifecycle:
1. --train        : Trains ConvLSTM Autoencoder with combined MSE + SSIM loss
2. --evaluate     : Evaluates checkpoints on test sequences, computes ROC-AUC and EER
3. --export       : Exports validated graph to ONNX opset 17 with dynamic batch
4. --build-engine : Compiles NVIDIA TensorRT engine (FP16/INT8) or edge fallback
5. --stream       : Launches real-time video inference loop with integrated HUD
6. --all          : Runs the full pipeline end-to-end
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Ensure edge_anomaly_detector root and src directory are in Python sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.dataset import ensure_dataset_ready
from src.train import train_pipeline
from src.evaluate import run_evaluation
from src.export_onnx import export_to_onnx
from src.build_engine import setup_edge_runtime
from src.infer_stream import run_stream_inference


def print_banner(title: str) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f" {title.center(66)} ")
    print(f"{sep}\n")


def execute_pipeline(args: argparse.Namespace) -> None:
    t_pipeline_start = time.time()

    # Determine execution stages
    run_all = args.all or not (args.train or args.evaluate or args.export or args.build_engine or args.stream)
    do_train = args.train or run_all
    do_eval = args.evaluate or run_all
    do_export = args.export or run_all
    do_build = args.build_engine or run_all
    do_stream = args.stream or run_all

    print_banner("EDGE VIDEO ANOMALY DETECTION ENGINE (CONVLSTM + TENSORRT)")
    print(f"[*] Workspace Root       : {SCRIPT_DIR}")
    print(f"[*] Target Data Directory: {args.data_dir}")
    print(f"[*] Precision Mode       : {args.precision.upper()}")
    print(f"[*] Execution Stages     : Train={do_train}, Eval={do_eval}, Export={do_export}, Build={do_build}, Stream={do_stream}")

    # Stage 0: Dataset Verification & Preparation
    print_banner("STAGE 0: DATASET VERIFICATION & PREPARATION")
    data_path = SCRIPT_DIR / args.data_dir
    ensure_dataset_ready(data_path)

    best_checkpoint = SCRIPT_DIR / "models/checkpoints/best_model.pt"
    onnx_output = SCRIPT_DIR / "models/onnx/anomaly_detector.onnx"
    engine_output = SCRIPT_DIR / f"models/engine/anomaly_detector_{args.precision}.engine"

    # Stage 1: Training
    if do_train:
        print_banner("STAGE 1: MODEL TRAINING")
        train_pipeline(
            data_dir=str(data_path),
            checkpoints_dir=str(SCRIPT_DIR / "models/checkpoints"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device_str=args.device,
        )

    # Stage 2: Evaluation
    if do_eval:
        print_banner("STAGE 2: EVALUATION & BENCHMARKING (ROC-AUC / EER)")
        run_evaluation(
            data_dir=str(data_path),
            checkpoint_path=str(best_checkpoint),
            output_dir=str(SCRIPT_DIR / "models"),
            device_str=args.device,
        )

    # Stage 3: ONNX Graph Export & Verification
    if do_export:
        print_banner("STAGE 3: ONNX GRAPH EXPORT & NUMERICAL VALIDATION")
        export_to_onnx(
            checkpoint_path=str(best_checkpoint),
            output_onnx_path=str(onnx_output),
            temporal_window=10,
            dynamic_batch=True,
            opset_version=17,
            device_str="cpu",
        )

    # Stage 4: TensorRT Engine Build / Edge Fallback
    runtime_target = str(onnx_output)
    if do_build:
        print_banner(f"STAGE 4: TENSORRT ENGINE COMPILATION ({args.precision.upper()})")
        compiled_target = setup_edge_runtime(
            onnx_path=str(onnx_output),
            engine_path=str(engine_output),
            precision=args.precision,
            calib_data_dir=str(data_path),
        )
        if compiled_target and Path(compiled_target).exists():
            runtime_target = compiled_target

    # Stage 5: Real-Time Stream Inference & HUD
    if do_stream:
        print_banner("STAGE 5: REAL-TIME STREAM INFERENCE & HEADS-UP DISPLAY")
        stream_out = str(SCRIPT_DIR / args.stream_output) if args.stream_output else None
        run_stream_inference(
            video_source=args.source,
            model_path=runtime_target,
            threshold=args.threshold,
            output_video_path=stream_out,
            max_frames=args.max_frames,
            device_str=args.device,
            display=not args.no_display,
        )

    elapsed = time.time() - t_pipeline_start
    print_banner(f"PIPELINE COMPLETED SUCCESSFULLY (Total Elapsed: {elapsed:.2f}s)")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master Orchestration for ConvLSTM Video Anomaly Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Mode selection
    group = parser.add_argument_group("Pipeline Stage Flags")
    group.add_argument("--all", action="store_true", help="Execute complete pipeline end-to-end")
    group.add_argument("--train", action="store_true", help="Execute model training")
    group.add_argument("--evaluate", action="store_true", help="Run benchmark evaluation (ROC-AUC / EER)")
    group.add_argument("--export", action="store_true", help="Export PyTorch model to ONNX")
    group.add_argument("--build-engine", action="store_true", help="Compile NVIDIA TensorRT engine")
    group.add_argument("--stream", action="store_true", help="Run real-time stream inference")

    # Hyperparameters & Paths
    parser.add_argument("--data-dir", type=str, default="data/raw/UCSDped2", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for AdamW")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "int8", "fp32"], help="Engine precision")
    parser.add_argument("--device", type=str, default=None, help="Hardware device ('cuda' or 'cpu')")

    # Stream configuration
    parser.add_argument("--source", default="synthetic", help="Video source: 'synthetic', webcam index (e.g. '0'), or file/RTSP URL")
    parser.add_argument("--threshold", type=float, default=0.5, help="Regularity score anomaly threshold")
    parser.add_argument("--stream-output", type=str, default="output_stream.mp4", help="Save annotated video stream")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit maximum frames for streaming")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI display window")

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_arguments()
    execute_pipeline(cli_args)

