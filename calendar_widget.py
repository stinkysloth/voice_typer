#!/usr/bin/env python3
"""
Calendar widget for WhisperNotes application.
Displays a calendar with indicators for dates with recordings.
"""
import os
import datetime
import calendar
from typing import Dict, List, Optional, Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGridLayout, QMenu, QDialog
)
from PyQt6.QtCore import QObject
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QAction


class CalendarWidget(QDialog):
    """
    Calendar widget that displays months with indicators for dates with recordings.
    """
    date_selected = pyqtSignal(str)  # Signal emitted when a date with recordings is clicked
    
    def __init__(self, parent=None, recordings_dir: str = None):
        """
        Initialize the calendar widget.
        
        Args:
            parent: Parent widget
            recordings_dir: Directory containing recordings
        """
        super().__init__(parent)
        self.setWindowTitle("Recording Calendar")
        self.setMinimumSize(300, 350)
        
        # Initialize properties
        self.recordings_dir = recordings_dir
        self.current_date = datetime.datetime.now()
        self.recording_dates = {}  # Will store dates with recordings
        
        # Set up the UI
        self.setup_ui()
        
        # Load recording dates
        self.scan_recordings()
        
    def setup_ui(self):
        """Set up the calendar UI."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Month navigation
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedWidth(40)
        self.prev_btn.clicked.connect(self.previous_month)
        
        self.month_label = QLabel()
        self.month_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedWidth(40)
        self.next_btn.clicked.connect(self.next_month)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.month_label)
        nav_layout.addWidget(self.next_btn)
        
        main_layout.addLayout(nav_layout)
        
        # Calendar grid
        self.calendar_grid = QGridLayout()
        
        # Add day headers
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i, day in enumerate(days):
            label = QLabel(day)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.calendar_grid.addWidget(label, 0, i)
        
        # Create date buttons (will be populated in update_calendar)
        self.date_buttons = []
        for row in range(1, 7):  # 6 rows max needed for a month
            for col in range(7):  # 7 days in a week
                btn = DateButton()
                btn.clicked.connect(self.on_date_clicked)
                self.calendar_grid.addWidget(btn, row, col)
                self.date_buttons.append(btn)
                
        main_layout.addLayout(self.calendar_grid)
        
        # Add a "Close" button at the bottom
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        main_layout.addWidget(close_btn)
        
        # Update the calendar display
        self.update_calendar()
        
    def update_calendar(self):
        """Update the calendar display for the current month."""
        year = self.current_date.year
        month = self.current_date.month
        
        # Update month label
        self.month_label.setText(f"{calendar.month_name[month]} {year}")
        
        # Get the calendar for the current month
        cal = calendar.monthcalendar(year, month)
        
        # Reset all buttons
        for btn in self.date_buttons:
            btn.setText("")
            btn.setEnabled(False)
            btn.setProperty("has_recording", False)
            btn.setProperty("date_str", "")
            btn.update()
        
        # Fill in the dates
        btn_idx = 0
        for week in cal:
            for day in week:
                if day == 0:
                    # Empty cell
                    self.date_buttons[btn_idx].setText("")
                    self.date_buttons[btn_idx].setEnabled(False)
                else:
                    # Valid date
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    self.date_buttons[btn_idx].setText(str(day))
                    self.date_buttons[btn_idx].setEnabled(True)
                    self.date_buttons[btn_idx].setProperty("date_str", date_str)
                    
                    # Check if this date has recordings
                    has_recording = date_str in self.recording_dates
                    self.date_buttons[btn_idx].setProperty("has_recording", has_recording)
                    
                btn_idx += 1
                if btn_idx >= len(self.date_buttons):
                    break
    
    def previous_month(self):
        """Navigate to the previous month."""
        year = self.current_date.year
        month = self.current_date.month
        
        if month == 1:
            self.current_date = self.current_date.replace(year=year-1, month=12)
        else:
            self.current_date = self.current_date.replace(month=month-1)
            
        self.update_calendar()
    
    def next_month(self):
        """Navigate to the next month."""
        year = self.current_date.year
        month = self.current_date.month
        
        if month == 12:
            self.current_date = self.current_date.replace(year=year+1, month=1)
        else:
            self.current_date = self.current_date.replace(month=month+1)
            
        self.update_calendar()
    
    def scan_recordings(self):
        """Scan the recordings directory to find dates with recordings."""
        if not self.recordings_dir or not os.path.exists(self.recordings_dir):
            return
            
        self.recording_dates = {}
        
        # Look for audio files in the recordings directory
        for filename in os.listdir(self.recordings_dir):
            if filename.endswith(('.wav', '.mp3', '.m4a')):
                # Extract date from filename (assuming format like YYYYMMDD_HHMMSS.wav)
                try:
                    date_part = filename.split('_')[0]
                    if len(date_part) == 8:  # YYYYMMDD
                        year = int(date_part[:4])
                        month = int(date_part[4:6])
                        day = int(date_part[6:8])
                        
                        date_str = f"{year}-{month:02d}-{day:02d}"
                        
                        if date_str not in self.recording_dates:
                            self.recording_dates[date_str] = []
                            
                        self.recording_dates[date_str].append(filename)
                except (ValueError, IndexError):
                    # Skip files that don't match the expected format
                    continue
                    
        # Also look in the entries directory for markdown files
        entries_dir = os.path.join(os.path.dirname(self.recordings_dir), "entries")
        if os.path.exists(entries_dir):
            for filename in os.listdir(entries_dir):
                if filename.endswith('.md'):
                    # Extract date from filename (assuming format like YYYY-MM-DD - HH:MM:SS AM/PM - Audio Journal Entry.md)
                    try:
                        date_part = filename.split(' - ')[0]
                        if len(date_part.split('-')) == 3:  # YYYY-MM-DD
                            date_str = date_part
                            
                            if date_str not in self.recording_dates:
                                self.recording_dates[date_str] = []
                                
                            self.recording_dates[date_str].append(filename)
                    except (ValueError, IndexError):
                        # Skip files that don't match the expected format
                        continue
        
        # Update the calendar to show the recording indicators
        self.update_calendar()
    
    def on_date_clicked(self):
        """Handle date button clicks."""
        btn = self.sender()
        if btn.property("has_recording"):
            date_str = btn.property("date_str")
            self.date_selected.emit(date_str)
            
            # Show recordings for this date
            self.show_recordings_for_date(date_str)
    
    def show_recordings_for_date(self, date_str):
        """Show a list of recordings for the selected date."""
        if date_str not in self.recording_dates:
            return
        recordings = self.recording_dates[date_str]
        if not recordings:
            return
        # Create a menu with the recordings
        menu = QMenu(self)
        # Only add an action to open the .md journal file for this date
        open_md_action = QAction(f"Open journal for {date_str}", self)
        open_md_action.triggered.connect(lambda: self.open_md_for_date(date_str))
        menu.addAction(open_md_action)
        menu.exec(self.cursor().pos())

    def open_md_for_date(self, date_str):
        """Open the journal .md file for the given date in the default editor."""
        # Journal files are in 'entries' subfolder: {journal_dir}/entries/{date_str} - Audio Journal Entry.md
        journal_dir = os.path.dirname(self.recordings_dir) if self.recordings_dir else None
        if not journal_dir:
            self._show_error("Journal directory not found.")
            return
        entries_dir = os.path.join(journal_dir, "entries")
        md_filename = f"{date_str} - Audio Journal Entry.md"
        md_path = os.path.join(entries_dir, md_filename)
        if not os.path.exists(md_path):
            self._show_error(f"No journal file found for {date_str} at:\n{md_path}")
            return
        import subprocess
        try:
            subprocess.run(["open", md_path], check=True)
        except Exception as e:
            self._show_error(f"Failed to open journal file:\n{e}")

    def _show_error(self, message):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Open Journal", message)

        
    def set_recordings_dir(self, directory):
        """Set the recordings directory and rescan for recordings."""
        self.recordings_dir = directory
        self.scan_recordings()


class DateButton(QPushButton):
    """Custom button for calendar dates with recording indicators."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(QSize(35, 35))
        
    def paintEvent(self, event):
        """Custom paint event to draw the recording indicator."""
        super().paintEvent(event)
        
        if self.property("has_recording"):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Draw a colored dot at the top of the button
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(52, 152, 219))  # Blue dot
            
            dot_size = 8
            x = (self.width() - dot_size) / 2
            y = 5
            
            painter.drawEllipse(int(x), int(y), dot_size, dot_size)
            painter.end()


# For testing the widget directly
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Example usage
    calendar = CalendarWidget()
    calendar.show()
    
    sys.exit(app.exec())
