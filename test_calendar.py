#!/usr/bin/env python3
"""
Test script for the calendar widget.
This allows testing the calendar functionality without running the full application.
"""
import os
import sys
import datetime
from PyQt6.QtWidgets import QApplication
from calendar_widget import CalendarWidget

def create_test_data(base_dir):
    """Create test data for the calendar widget."""
    # Create directories if they don't exist
    recordings_dir = os.path.join(base_dir, "recordings")
    entries_dir = os.path.join(base_dir, "entries")
    
    os.makedirs(recordings_dir, exist_ok=True)
    os.makedirs(entries_dir, exist_ok=True)
    
    # Create some test recordings
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    last_week = today - datetime.timedelta(days=7)
    last_month = today - datetime.timedelta(days=30)
    
    # Create empty files with appropriate names
    dates = [today, yesterday, last_week, last_month]
    
    for date in dates:
        # Create recording file (format: YYYYMMDD_HHMMSS.wav)
        recording_name = f"{date.strftime('%Y%m%d_%H%M%S')}.wav"
        recording_path = os.path.join(recordings_dir, recording_name)
        
        with open(recording_path, 'w') as f:
            f.write("Test recording file")
        
        # Create entry file (format: YYYY-MM-DD - HH:MM:SS AM/PM - Audio Journal Entry.md)
        time_str = date.strftime("%I:%M:%S %p")
        entry_name = f"{date.strftime('%Y-%m-%d')} - {time_str} - Audio Journal Entry.md"
        entry_path = os.path.join(entries_dir, entry_name)
        
        with open(entry_path, 'w') as f:
            f.write(f"# Audio Journal Entry - {date.strftime('%Y-%m-%d')} {time_str}\n\n")
            f.write("### Summary\n")
            f.write("This is a test entry.\n\n")
            f.write("### Transcript\n")
            f.write("This is a test transcript for the calendar widget.\n\n")
    
    print(f"Created test data in {base_dir}")
    return recordings_dir

def on_date_selected(date_str):
    """Handle date selection from the calendar."""
    print(f"Date selected: {date_str}")

def main():
    """Run the calendar widget test."""
    app = QApplication(sys.argv)
    
    # Create test data in a temporary directory
    test_dir = os.path.join(os.path.expanduser("~"), "Documents", "WhisperNotes_Test")
    recordings_dir = create_test_data(test_dir)
    
    # Create and show the calendar widget
    calendar = CalendarWidget(None, recordings_dir)
    calendar.date_selected.connect(on_date_selected)
    calendar.show()
    
    print("Calendar widget is now visible. Close the window to exit.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
