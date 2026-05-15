import sys
from pathlib import Path

try:
    import onnxruntime as _onnxruntime  # noqa: F401
except Exception:
    _onnxruntime = None

from PyQt5.QtWidgets import QApplication


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cam_ai.UI.UI import MonitoringWindow  # noqa: E402


if __name__ == "__main__":
    app = QApplication(sys.argv)
    sample_source = MonitoringWindow._default_source()
    window = MonitoringWindow(sources=[sample_source, sample_source, sample_source, sample_source])
    window.show()
    raise SystemExit(app.exec_())
