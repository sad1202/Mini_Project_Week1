import sys
import time
from PyQt5.QtCore import QThread, pyqtSignal, QObject



class DetectWorker(QObject):
    