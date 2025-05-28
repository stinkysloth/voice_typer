from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtCore import Qt

class RecordingIndicator(QWidget):
    """
    Visual indicator that shows when recording is active.
    """
    def __init__(self):
        super().__init__()
        self.setFixedSize(80, 80)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.recording = False

    def set_recording(self, recording: bool):
        self.recording = recording
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.recording:
            painter.setBrush(QColor(255, 0, 0, 180))
        else:
            painter.setBrush(QColor(150, 150, 150, 120))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(10, 10, 60, 60)
        painter.end()
