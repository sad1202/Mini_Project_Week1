import argparse
import sys
from pathlib import Path

try:
    import onnxruntime as _onnxruntime  # noqa: F401
except Exception:
    _onnxruntime = None

try:
    import torch as _torch  # noqa: F401
except Exception:
    _torch = None

from PyQt5.QtWidgets import QApplication


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cam_ai.UI.UI import MonitoringWindow  # noqa: E402


def split_sources(raw_sources: list[str] | None) -> list[str] | None:
    if not raw_sources:
        return None

    sources: list[str] = []
    for item in raw_sources:
        sources.extend(source.strip() for source in item.split(",") if source.strip())
    return sources or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Camera Monitoring")
    parser.add_argument(
        "--source",
        action="append",
        help="Camera source: webcam index, video path, or RTSP URL.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Multiple camera sources. Accepts space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--rtsp-4cam",
        action="store_true",
        help="Run four default RTSP cameras: rtsp://localhost:8554/cam1..cam4.",
    )
    parser.add_argument(
        "--sample-4cam",
        action="store_true",
        help="Run four panels with the sample video.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="YOLO model path. Default searches project model/checkpoint files.",
    )
    parser.add_argument(
        "--model-format",
        choices=["onnx", "pt"],
        default=None,
        help="Shortcut model selector: onnx uses yolo26n.onnx, pt uses yolo26n.pt.",
    )
    parser.add_argument(
        "--mode",
        choices=["none", "grayscale", "blur"],
        default="none",
        help="Image preprocessing mode.",
    )
    parser.add_argument("--loiter-seconds", type=float, default=5.0)
    parser.add_argument("--crowd-threshold", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=320, help="Image size for YOLO model inference.")
    return parser.parse_args()


def resolve_sources(args: argparse.Namespace):
    sources = split_sources(args.sources) or split_sources(args.source)

    if args.rtsp_4cam:
        return MonitoringWindow.default_rtsp_sources()

    if args.sample_4cam:
        sample_source = MonitoringWindow._default_source()
        return [sample_source, sample_source, sample_source, sample_source]

    return sources


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    window = MonitoringWindow(
        sources=resolve_sources(args),
        model_path=args.model,
        model_format=args.model_format,
        mode=args.mode,
        loiter_seconds=args.loiter_seconds,
        crowd_threshold=args.crowd_threshold,
        imgsz=args.imgsz,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
