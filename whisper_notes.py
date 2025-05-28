#!/usr/bin/env python3
import os
import sys
print("Main process Python executable:", sys.executable)
import time
import logging
import threading
import numpy as np
import sounddevice as sd
import librosa
from functools import partial
from datetime import datetime
from pynput import keyboard
import subprocess
import tempfile
import json
from PyQt6.QtWidgets import (QApplication, QSystemTrayIcon, QMenu, QMessageBox, QFileDialog,
                               QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton, QDialogButtonBox)
from PyQt6.QtGui import QIcon, QAction, QPixmap, QPainter, QColor
from PyQt6.QtCore import QObject, QThread, pyqtSignal as Signal, Qt, QTimer, QMutex, QCoreApplication, QSettings, QStandardPaths

# Import journaling module
try:
    from journaling import JournalingManager
except ImportError:
    # Fallback if journaling.py is not found
    class JournalingManager:
        def __init__(self, *args, **kwargs):
            pass
        def create_journal_entry(self, *args, **kwargs):
            return {'error': 'Journaling not available'}

# Import our custom calendar widget
from calendar_widget import CalendarWidget

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("voice_typer.log"),
        logging.StreamHandler()
    ]
)

class ModelLoader(QObject):
    """Worker for loading the Whisper model."""
    finished = Signal(object)
    error = Signal(str)
    
    def __init__(self, model_name="base"):
        super().__init__()
        self.model_name = model_name
        
    def run(self):
        """Load the Whisper model."""
        try:
            logging.info("Loading Whisper model...")
            import whisper
            model = whisper.load_model(self.model_name)
            logging.info("Whisper model loaded")
            self.finished.emit(model)
        except Exception as e:
            error_msg = f"Error loading Whisper model: {e}"
            logging.error(error_msg, exc_info=True)
            self.error.emit(error_msg)


# No longer needed: all transcription is now handled by an external subprocess


class TranscriptionWorker(QObject):
    """Worker for transcribing audio with Whisper via an external subprocess."""
    finished = Signal()
    transcription_ready = Signal(str)
    error = Signal(str)

    def __init__(self, model_name, audio_data):
        super().__init__()
        self.model_name = model_name
        self.audio_data = audio_data
        self.timeout = 120 # Timeout for subprocess in seconds
        self._should_stop = False
        self.process = None
        logging.debug(f"TranscriptionWorker initialized for model {self.model_name}")

    def request_stop(self):
        logging.info("TranscriptionWorker: Stop requested.")
        self._should_stop = True
        if self.process and self.process.poll() is None: # If process exists and is running
            logging.info("TranscriptionWorker: Terminating active subprocess due to stop request.")
            try:
                self.process.kill()
                self.process.wait(timeout=2) # Give it a moment to die
            except subprocess.TimeoutExpired:
                logging.warning("TranscriptionWorker: Subprocess did not terminate quickly after kill.")
            except Exception as e:
                logging.error(f"TranscriptionWorker: Error killing subprocess: {e}")
        self.process = None # Ensure process handle is cleared

    def run(self):
        logging.info("[TranscriptionWorker] RUN METHOD ENTERED (topmost line).")
        # Ensure imports are available if run in a very bare environment (though QThread usually inherits context)
        import numpy as np
        import soundfile as sf
        import tempfile
        import subprocess
        import json
        import os
        import sys
        import traceback # For detailed error logging

        try:
            logging.getLogger().handlers[0].flush()
            # print(f"RUN STARTED (thread: {threading.current_thread().name}, ident: {threading.get_ident()})", flush=True)
            # logging.info(f"[TranscriptionWorker] Qt thread: {QThread.currentThread()}, main thread: {QCoreApplication.instance().thread() if QCoreApplication.instance() else None}"); logging.getLogger().handlers[0].flush()
            # logging.info(f"[TranscriptionWorker] Python sys.executable: {sys.executable}"); logging.getLogger().handlers[0].flush()
            # logging.info(f"[TranscriptionWorker] CWD: {os.getcwd()}"); logging.getLogger().handlers[0].flush()
            
            logging.info("[TranscriptionWorker] Starting Whisper transcription (subprocess)..." ); logging.getLogger().handlers[0].flush()
            # logging.info(f"[TranscriptionWorker] Python executable: {sys.executable}"); logging.getLogger().handlers[0].flush()
            # logging.info(f"[TranscriptionWorker] Platform: {platform.platform()}"); logging.getLogger().handlers[0].flush()
            # logging.info(f"[TranscriptionWorker] ENV: {os.environ}"); logging.getLogger().handlers[0].flush()
            # start_time = time.time()

            if self._should_stop:
                logging.info("[TranscriptionWorker] Aborting _do_transcription at start: stop requested.")
                return

            # Create a temporary directory for intermediate files
            with tempfile.TemporaryDirectory() as tmpdir:
                logging.info(f"[TranscriptionWorker] Created tempdir: {tmpdir}"); logging.getLogger().handlers[0].flush()
                audio_path = os.path.join(tmpdir, "audio.wav")
                result_path = os.path.join(tmpdir, "result.json")

                try:
                    logging.info(f"[TranscriptionWorker] Saving audio to {audio_path}"); logging.getLogger().handlers[0].flush()
                    # Ensure audio_data is float32 for librosa, then convert to PCM 16-bit for WAV
                    if self.audio_data.dtype != np.float32:
                        audio_float32 = self.audio_data.astype(np.float32)
                    else:
                        audio_float32 = self.audio_data
                    
                    # Normalize if max abs value is > 1.0 (sf.write expects this range for float32)
                    max_val = np.max(np.abs(audio_float32))
                    if max_val > 1.0:
                        audio_float32 /= max_val # max_val cannot be 0 here if > 1.0
                    elif max_val == 0:
                        logging.info("[TranscriptionWorker] Audio data is all zeros, no normalization needed.")
                    # else: audio is already in [-1.0, 1.0] or quieter, no normalization needed

                    sf.write(audio_path, audio_float32, samplerate=16000, subtype='PCM_16')
                    logging.info(f"[TranscriptionWorker] Audio saved to {audio_path}"); logging.getLogger().handlers[0].flush()
                except Exception as e:
                    logging.error(f"[TranscriptionWorker] Error saving audio: {e}", exc_info=True); logging.getLogger().handlers[0].flush()
                    self.error.emit(f"Error saving audio: {e}")
                    return

                if self._should_stop: # Check after potentially lengthy file ops
                    logging.info("[TranscriptionWorker] Aborting _do_transcription before Popen: stop requested.")
                    return

                # Construct command for transcribe_worker.py
                script_path = os.path.join(os.path.dirname(__file__), "transcribe_worker.py")
                cmd = [
                    sys.executable, 
                    script_path,
                    self.model_name, 
                    audio_path,
                    result_path
                ]
                logging.info(f"[TranscriptionWorker] Subprocess command: {' '.join(cmd)}"); logging.getLogger().handlers[0].flush()

                try:
                    self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    logging.info(f"[TranscriptionWorker] Subprocess started with PID: {self.process.pid}"); logging.getLogger().handlers[0].flush()
                except Exception as e:
                    logging.error(f"[TranscriptionWorker] Failed to Popen subprocess: {e}", exc_info=True); logging.getLogger().handlers[0].flush()
                    self.error.emit(f"Failed to start transcription process: {e}")
                    self.process = None # Ensure process is None
                    return

                if self._should_stop: # Check immediately after Popen
                    logging.info("[TranscriptionWorker] Stop requested immediately after Popen, killing subprocess.")
                    if self.process and self.process.poll() is None:
                        self.process.kill()
                        try: self.process.wait(timeout=1)
                        except subprocess.TimeoutExpired: pass
                    self.process = None
                    return

                try:
                    stdout, stderr = self.process.communicate(timeout=self.timeout)
                    return_code = self.process.returncode
                    stdout_str = stdout.strip()
                    stderr_str = stderr.strip()
                    logging.info(f"[TranscriptionWorker] Subprocess stdout:\n{stdout_str}")
                    logging.info(f"[TranscriptionWorker] Subprocess stderr:\n{stderr_str}")

                    logging.info(f"[TranscriptionWorker] Subprocess return code (logged before check): {return_code}")
                    actual_return_code = return_code
                    logging.info(f"[TranscriptionWorker] About to check return code. Actual value: {actual_return_code}, Type: {type(actual_return_code)}")

                    if actual_return_code == 0:
                        logging.info("[TranscriptionWorker] Entered actual_return_code == 0 block.")
                        if not os.path.exists(result_path):
                            logging.error(f"[TranscriptionWorker] Result file not found: {result_path}")
                        else:
                            with open(result_path, 'r') as f:
                                result = json.load(f)
                            if result.get("status") == "success":
                                logging.info(f"[TranscriptionWorker] Transcription successful: {result['text']}"); logging.getLogger().handlers[0].flush()
                                self.transcription_ready.emit(result["text"])
                            else:
                                err_msg = result.get("error", "Transcription failed in worker script.")
                                logging.error(f"[TranscriptionWorker] Transcription failed in worker: {err_msg}"); logging.getLogger().handlers[0].flush()
                                self.error.emit(err_msg)
                    else: # actual_return_code != 0
                        logging.error(f"[TranscriptionWorker] Entered actual_return_code != 0 block. Actual value was: {actual_return_code}")
                        error_detail = "Unknown error from worker script."
                        if stderr_str:
                            error_detail = f"Worker script stderr: {stderr_str}"
                        elif stdout_str:
                            # Check if stdout contains error-like messages if stderr is empty
                            if "error" in stdout_str.lower() or "traceback" in stdout_str.lower():
                                error_detail = f"Worker script stdout (may contain error): {stdout_str}"
                        
                        logging.error(f"[TranscriptionWorker] Subprocess failed. {error_detail}")
                        self.error.emit(f"Transcription failed in worker script. {error_detail}")

                except subprocess.TimeoutExpired:
                    logging.warning(f"[TranscriptionWorker] Subprocess timed out after {self.timeout}s."); logging.getLogger().handlers[0].flush()
                    if self.process and self.process.poll() is None: # Check if still running
                        self.process.kill()
                        try: self.process.wait(timeout=1)
                        except subprocess.TimeoutExpired: pass
                    self.error.emit("Transcription timed out.")
                except Exception as e: # Broad catch for other Popen/communicate issues
                    if self._should_stop:
                        logging.info(f"[TranscriptionWorker] Exception during communicate, likely due to stop request: {e}")
                    else:
                        logging.error(f"[TranscriptionWorker] Error during subprocess communication: {e}", exc_info=True); logging.getLogger().handlers[0].flush()
                        self.error.emit(f"Transcription error: {e}")
                finally:
                    self.process = None # Clear the process reference

            # Fallback for tempdir cleanup issues, though 'with' should handle it
            try:
                if 'tmpdir' in locals() and os.path.exists(tmpdir):
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception as e:
                logging.warning(f"[TranscriptionWorker] Error cleaning up temp directory {tmpdir}: {e}")

            logging.info("[TranscriptionWorker] Main logic in run method finished."); logging.getLogger().handlers[0].flush()

        except Exception as e:
            # This is the catch-all for the entire run operation
            error_msg = f"[TranscriptionWorker] Unhandled error in run method: {e}\n{traceback.format_exc()}"
            logging.error(error_msg, exc_info=True)
            self.error.emit(f"Critical worker error: {e}")
        finally:
            logging.info("[TranscriptionWorker] Run method finally block: Emitting finished signal."); logging.getLogger().handlers[0].flush()
            self.finished.emit() # CRUCIAL: always emit finished


class RecordingThread(QThread):
    """A thread for recording audio."""
    finished = Signal(object)  # Emits audio data when done
    error = Signal(str)
    
    def __init__(self, max_duration=900.0):  # Changed from 10.0 to 900.0 (15 minutes)
        super().__init__()
        self.max_duration = max_duration
        self.stop_flag = False
    
    def run(self):
        """Record audio for up to max_duration seconds or until stopped."""
        try:
            # Setup audio parameters
            sample_rate = 44100
            channels = 1
            
            # Get available audio devices
            devices = sd.query_devices()
            logging.info(f"Available audio devices: {len(devices)}")
            
            # Find default input device
            default_device = sd.default.device[0]
            device_info = devices[default_device]
            logging.info(f"Using input device {default_device}: {device_info['name']}")
            
            # Record audio
            logging.info(f"Recording audio for up to {self.max_duration} seconds...")
            audio_data = sd.rec(
                int(self.max_duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype='float32'
            )
            
            # Keep track of recording time
            start_time = time.time()
            
            # Check for stop signal periodically
            while not self.stop_flag and (time.time() - start_time) < self.max_duration:
                # Check every 100ms
                time.sleep(0.1)
                # This won't affect the recording, which happens in another thread via sounddevice
            
            # Calculate how long we've been recording
            elapsed_time = time.time() - start_time
            
            # Stop recording
            sd.stop()
            
            # Trim audio data to actual recorded length
            if elapsed_time < self.max_duration:
                logging.info(f"Recording stopped after {elapsed_time:.2f} seconds")
                frames_recorded = int(elapsed_time * sample_rate)
                audio_data = audio_data[:frames_recorded]
            else:
                logging.info(f"Recording completed full {self.max_duration} seconds")
            
            # Convert to the format expected by Whisper
            if sample_rate != 16000:
                logging.info(f"Resampling from {sample_rate}Hz to 16000Hz...")
                audio_data = librosa.resample(audio_data.squeeze(), orig_sr=sample_rate, target_sr=16000)
            
            # Emit the audio data
            self.finished.emit(audio_data)
            
        except Exception as e:
            error_msg = f"Error recording audio: {e}"
            logging.error(error_msg, exc_info=True)
            self.error.emit(error_msg)
    
    def stop(self):
        """Signal the thread to stop recording."""
        logging.info("Stopping recording thread...")
        self.stop_flag = True


class WhisperNotes(QObject):
    """Main application class for WhisperNotes."""
    toggle_recording_signal = Signal()
    toggle_journal_signal = Signal()
    quit_signal = Signal()
    
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.model = None
        self.recording_thread = None
        self.transcription_thread = None
        self.transcriber = None  # Holds reference to the worker object
        self.last_recording_time = 0
        self.mutex = QMutex()  # For thread safety
        self.hotkey_active = False
        self.pressed_keys = set()
        self.journaling_mode = False  # Track if we're in journaling mode
        self.is_recording = False  # Track recording state
        
        # Initialize QSettings for persistent output file path
        self.settings = QSettings("VoiceTyper", "VoiceTyper")
        if not self.settings.contains("output_file"):
            documents_path = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
            default_output = os.path.join(documents_path, "VoiceTyperTranscriptions.md")
            self.settings.setValue("output_file", default_output)
        logging.info(f"Output Markdown file initialized to: {self.settings.value('output_file')}")
        
        # Initialize journaling manager with directory and prompts from settings if available
        journal_dir = self.settings.value("journal_dir")
        summary_prompt = self.settings.value("summary_prompt")
        format_prompt = self.settings.value("format_prompt")
        
        if journal_dir and os.path.isdir(journal_dir):
            self.journal_manager = JournalingManager(output_dir=journal_dir, summary_prompt=summary_prompt)
            logging.info(f"Using journal directory from settings: {journal_dir}")
        else:
            # Use default directory
            home_dir = os.path.expanduser("~")
            default_journal_dir = os.path.join(home_dir, "Documents", "Personal", "Audio Journal")
            self.journal_manager = JournalingManager(output_dir=default_journal_dir, summary_prompt=summary_prompt)
            logging.info(f"Using default journal directory: {default_journal_dir}")
            # Save the default to settings
            self.settings.setValue("journal_dir", default_journal_dir)
            
        # Set format prompt if available
        if format_prompt:
            self.journal_manager.set_format_prompt(format_prompt)
        
        self.toggle_recording_signal.connect(self.toggle_recording)
        self.toggle_journal_signal.connect(self.toggle_journal_mode)
        self.quit_signal.connect(self.quit)

        self.load_model()
        self.setup_tray()
        self.setup_hotkeys()
        self.setup_watchdog()
    
    def setup_watchdog(self):
        """Setup a watchdog timer to periodically check the application state."""
        self.watchdog_timer = QTimer(self)
        self.watchdog_timer.timeout.connect(self.check_application_state)
        self.watchdog_timer.start(1000)  # Check every second
    
    def check_application_state(self):
        """Check the application state and ensure everything is responsive."""
        # Check if recording has timed out
        if self.recording_thread and self.recording_thread.isRunning():
            elapsed = time.time() - self.last_recording_time
            max_duration = getattr(self.recording_thread, 'max_duration', 900.0)  # Default to 15 minutes if not set
            if elapsed > max_duration + 2.0:  # Add 2-second buffer
                logging.warning(f"Recording timeout detected ({elapsed:.1f}s > {max_duration}s). Forcing stop.")
                self.stop_recording()
    
    def load_model(self):
        """Load the Whisper model in a background thread."""
        self.model_thread = QThread()
        self.model_loader = ModelLoader(model_name="base")
        self.model_loader.moveToThread(self.model_thread)

        # Connect signals
        self.model_thread.started.connect(self.model_loader.run)
        self.model_loader.finished.connect(self.on_model_loaded)
        self.model_loader.error.connect(self.handle_error)
        self.model_loader.finished.connect(self.model_thread.quit)
        self.model_loader.finished.connect(self.model_loader.deleteLater)
        self.model_thread.finished.connect(self.model_thread.deleteLater)

        # Start loading
        self.model_thread.start()

        
    def on_model_loaded(self, model):
        """Handle when model is loaded."""
        self.model = model
        logging.info(f"Model 'base' loaded successfully.")
        self.update_icon(False)
        
    def update_icon(self, recording):
        """Update the system tray icon based on recording status."""
        try:
            if recording:
                pixmap = QPixmap(32, 32) # Increased size for better visibility
                pixmap.fill(Qt.GlobalColor.red)
                self.tray_icon.setToolTip("Voice Typer (Recording...)")
            else:
                pixmap = QPixmap(32, 32)
                pixmap.fill(Qt.GlobalColor.gray)
                self.tray_icon.setToolTip("Voice Typer (Idle)")
            
            if hasattr(self, 'tray_icon') and self.tray_icon:
                self.tray_icon.setIcon(QIcon(pixmap))
            else:
                logging.warning("update_icon called but tray_icon not yet initialized.")
        except Exception as e:
            logging.error(f"Error in update_icon: {e}", exc_info=True)
        
    def setup_tray(self):
        """Setup the system tray icon and menu."""
        self.tray_icon = QSystemTrayIcon(self)
        self.update_icon(False)  # Set initial icon to not recording

        menu = QMenu()

        self.status_action = QAction("Status: Idle", self)
        self.status_action.setEnabled(False)
        menu.addAction(self.status_action)

        self.toggle_action = QAction("Start Recording (Cmd+Shift+R)", self)
        self.toggle_action.triggered.connect(self.toggle_recording)
        menu.addAction(self.toggle_action)
        
        self.journal_action = QAction("Start Journal Entry (Cmd+Shift+J)", self)
        self.journal_action.triggered.connect(self.toggle_journal_mode)
        menu.addAction(self.journal_action)
        
        # Add calendar view action
        self.calendar_action = QAction("Calendar View", self)
        self.calendar_action.triggered.connect(self.show_calendar)
        menu.addAction(self.calendar_action)
        
        menu.addSeparator()
        
        # Output settings submenu
        output_settings_menu = QMenu("Output Settings", menu)
        
        set_output_file_action = QAction("Set Transcription Output File...", self)
        set_output_file_action.triggered.connect(self.prompt_set_output_file)
        output_settings_menu.addAction(set_output_file_action)
        
        set_journal_dir_action = QAction("Set Journal Directory...", self)
        set_journal_dir_action.triggered.connect(self.prompt_set_journal_dir)
        output_settings_menu.addAction(set_journal_dir_action)
        
        output_settings_menu.addSeparator()
        
        edit_summary_prompt_action = QAction("Edit Summary Prompt...", self)
        edit_summary_prompt_action.triggered.connect(self.prompt_edit_summary_prompt)
        output_settings_menu.addAction(edit_summary_prompt_action)
        
        edit_format_prompt_action = QAction("Edit Format Prompt...", self)
        edit_format_prompt_action.triggered.connect(self.prompt_edit_format_prompt)
        output_settings_menu.addAction(edit_format_prompt_action)
        
        menu.addMenu(output_settings_menu)
        
        # Hotkey settings submenu
        hotkey_settings_menu = QMenu("Hotkey Settings", menu)
        
        set_record_hotkey_action = QAction("Set Recording Hotkey...", self)
        set_record_hotkey_action.triggered.connect(self.prompt_set_record_hotkey)
        hotkey_settings_menu.addAction(set_record_hotkey_action)
        
        set_journal_hotkey_action = QAction("Set Journal Hotkey...", self)
        set_journal_hotkey_action.triggered.connect(self.prompt_set_journal_hotkey)
        hotkey_settings_menu.addAction(set_journal_hotkey_action)
        
        menu.addMenu(hotkey_settings_menu)

        menu.addSeparator()

        quit_action = QAction("Quit WhisperNotes (Cmd+Q)", self)
        quit_action.triggered.connect(self.quit)  # Connect to the existing quit method
        menu.addAction(quit_action)

        self.tray_icon.setContextMenu(menu)
        self.tray_icon.show()

        # Connect tray icon activation for showing status or simple interaction
        # self.tray_icon.activated.connect(self.on_tray_icon_activated) # Ensure on_tray_icon_activated exists or remove

    def setup_hotkeys(self):
        """Setup global hotkeys for toggling recording, journaling, and quitting."""
        self.pressed_keys = set()
        self.hotkey_active = True # Flag to enable/disable hotkey processing if needed

        # Get custom hotkeys from settings or use defaults
        # Recording hotkey (default: Cmd+Shift+R)
        record_hotkey = self.settings.value("record_hotkey")
        if record_hotkey:
            try:
                self.TOGGLE_HOTKEY = self._parse_hotkey_string(record_hotkey)
                logging.info(f"Using custom recording hotkey: {record_hotkey}")
            except Exception as e:
                logging.error(f"Error parsing custom recording hotkey: {e}")
                # Fall back to default
                self.TOGGLE_HOTKEY = {keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode(char='r')}
        else:
            self.TOGGLE_HOTKEY = {keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode(char='r')}
            
        # Journal hotkey (default: Cmd+Shift+J)
        journal_hotkey = self.settings.value("journal_hotkey")
        if journal_hotkey:
            try:
                self.JOURNAL_HOTKEY = self._parse_hotkey_string(journal_hotkey)
                logging.info(f"Using custom journal hotkey: {journal_hotkey}")
            except Exception as e:
                logging.error(f"Error parsing custom journal hotkey: {e}")
                # Fall back to default
                self.JOURNAL_HOTKEY = {keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode(char='j')}
        else:
            self.JOURNAL_HOTKEY = {keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode(char='j')}
        
        # For Cmd+Q (not customizable)
        self.QUIT_HOTKEY = {keyboard.Key.cmd, keyboard.KeyCode(char='q')}

        try:
            self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()
            logging.info("Hotkey listener started.")
        except Exception as e:
            logging.error(f"Failed to start hotkey listener: {e}", exc_info=True)
            # Potentially fall back to a non-hotkey mode or notify user
            
    def _parse_hotkey_string(self, hotkey_str):
        """Parse a hotkey string into a set of keyboard keys.
        
        Format: 'cmd+shift+r' or 'ctrl+alt+j'
        
        Returns:
            set: A set of keyboard.Key and keyboard.KeyCode objects
        """
        hotkey_set = set()
        parts = hotkey_str.lower().split('+')
        
        for part in parts:
            part = part.strip()
            if part == 'cmd' or part == 'command':
                hotkey_set.add(keyboard.Key.cmd)
            elif part == 'ctrl' or part == 'control':
                hotkey_set.add(keyboard.Key.ctrl)
            elif part == 'alt':
                hotkey_set.add(keyboard.Key.alt)
            elif part == 'shift':
                hotkey_set.add(keyboard.Key.shift)
            elif len(part) == 1:  # Single character
                hotkey_set.add(keyboard.KeyCode(char=part))
            else:
                raise ValueError(f"Unsupported key: {part}")
                
        return hotkey_set
        
    def _hotkey_to_string(self, hotkey_set):
        """Convert a set of keyboard keys to a string representation.
        
        Returns:
            str: A string representation of the hotkey (e.g., 'cmd+shift+r')
        """
        parts = []
        
        # Process special keys first
        if keyboard.Key.cmd in hotkey_set:
            parts.append("cmd")
        if keyboard.Key.ctrl in hotkey_set:
            parts.append("ctrl")
        if keyboard.Key.alt in hotkey_set:
            parts.append("alt")
        if keyboard.Key.shift in hotkey_set:
            parts.append("shift")
            
        # Then add character keys
        for key in hotkey_set:
            if isinstance(key, keyboard.KeyCode) and hasattr(key, 'char'):
                parts.append(key.char)
                
        return "+".join(parts)

    def on_press(self, key):
        """Handle key press events for hotkeys."""
        if not self.hotkey_active:
            return

        # Normalize key for comparison (e.g. KeyCode instances)
        if hasattr(key, 'char') and key.char:
            # For character keys, store the character itself if it's part of a combo
            # This helps differentiate 'r' from Key.shift for example.
            # However, for direct KeyCode comparison, using key directly is better.
            pass # self.pressed_keys.add(key.char) - let's add the raw key object
        
        self.pressed_keys.add(key)
        # logging.debug(f"Pressed keys: {self.pressed_keys}")

        # Check for Cmd+Shift+R to toggle recording
        if self.TOGGLE_HOTKEY.issubset(self.pressed_keys):
            logging.info("Toggle recording hotkey detected (Cmd+Shift+R).")
            # Emit signal to ensure toggle_recording runs on the main Qt thread
            self.toggle_recording_signal.emit()
            # Clear keys to prevent re-triggering until a key is released and re-pressed
            # This is a simple way to handle autorepeat or holding keys down.
            # self.pressed_keys.clear() # Reconsider this, might interfere with other hotkeys
            return # Hotkey handled

        # Check for Cmd+Shift+J to toggle journaling
        if self.JOURNAL_HOTKEY.issubset(self.pressed_keys):
            logging.info("Toggle journaling hotkey detected (Cmd+Shift+J).")
            # Emit signal to ensure toggle_journal_mode runs on the main Qt thread
            self.toggle_journal_signal.emit()
            # self.pressed_keys.clear()
            return # Hotkey handled

        # Check for Cmd+Q to quit
        if self.QUIT_HOTKEY.issubset(self.pressed_keys):
            logging.info("Quit hotkey detected (Cmd+Q).")
            self.quit_signal.emit()
            # self.pressed_keys.clear()
            return # Hotkey handled

    def on_release(self, key):
        """Handle key release events for hotkeys."""
        if not self.hotkey_active:
            return
        
        # logging.debug(f"Key released: {key}")
        try:
            self.pressed_keys.remove(key)
        except KeyError:
            # This can happen if a key is released that wasn't tracked (e.g., if listener started mid-press)
            # Or if a modifier was released that was part of a combo already cleared.
            # logging.debug(f"Key {key} not in pressed_keys to remove.")
            pass

    def handle_recording_finished(self, audio_data):
        """Handle the recorded audio data."""
        if audio_data is None or len(audio_data) == 0:
            logging.warning("No audio data to process.")
            self.update_icon(False)
            return
            
        # Store audio data for journaling if needed
        self.audio_data = audio_data
        logging.info(f"Recording finished, audio data shape: {audio_data.shape if hasattr(audio_data, 'shape') else 'unknown'}")
        
        # Update UI to show we're no longer recording
        self.update_icon(False)
        
        # Show transcription in progress notification
        mode_text = "journal entry" if self.journaling_mode else "transcription"
        self.tray_icon.showMessage(
            "Voice Typer", 
            f"Transcribing audio for {mode_text}...", 
            QSystemTrayIcon.Information, 
            2000
        )

        # Ensure previous transcriber and thread are cleaned up if they somehow exist
        if self.transcriber is not None or self.transcription_thread is not None:
            logging.warning("[VoiceTyper] Previous transcriber/thread not None, attempting to clean up.")
            if self.transcription_thread and self.transcription_thread.isRunning():
                self.transcription_thread.quit()
                self.transcription_thread.wait(1000) # Wait a bit
            self._clear_transcriber_references()
            self._clear_transcription_thread_references()

        # Create and start new transcription worker and thread
        self.transcriber = TranscriptionWorker("base", audio_data)
        self.transcription_thread = QThread()
        self.transcriber.moveToThread(self.transcription_thread)

        # Connect signals
        self.transcription_thread.started.connect(self.transcriber.run)
        self.transcription_thread.started.connect(lambda: logging.info("[VoiceTyper] transcription_thread 'started' SIGNAL EMITTED."))
        
        self.transcriber.finished.connect(self.transcription_thread.quit)
        self.transcriber.finished.connect(self.transcriber.deleteLater)
        self.transcriber.finished.connect(self._clear_transcriber_references)
        
        self.transcription_thread.finished.connect(self.transcription_thread.deleteLater)
        self.transcription_thread.finished.connect(self._clear_transcription_thread_references)

        self.transcriber.transcription_ready.connect(self.handle_transcription)
        self.transcriber.error.connect(self.handle_error)
        
        logging.info("[VoiceTyper] About to call self.transcription_thread.start()...")
        self.transcription_thread.start()
        logging.info("[VoiceTyper] Call to self.transcription_thread.start() returned.")

    def prompt_set_output_file(self):
        """Prompts the user to select a Markdown file for saving transcriptions."""
        current_path = self.settings.value("output_file")
        # Suggest a filename and directory based on current settings or defaults
        suggested_filename = os.path.basename(current_path) if current_path and not os.path.isdir(current_path) else "VoiceTyperTranscriptions.md"
        suggested_dir = os.path.dirname(current_path) if current_path and os.path.isdir(os.path.dirname(current_path)) else QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        
        initial_path = os.path.join(suggested_dir, suggested_filename)

        file_path, _ = QFileDialog.getSaveFileName(
            None,  # Parent widget; QMainWindow instance usually, None for simple dialogs
            "Set Transcription Output File",
            initial_path,
            "Markdown Files (*.md);;All Files (*)"
        )
        if file_path:
            self.settings.setValue("output_file", file_path)
            logging.info(f"Output Markdown file changed to: {self.settings.value('output_file')}")
            self.tray_icon.showMessage("Settings Updated", f"Transcription output file set to:\n{file_path}", QSystemTrayIcon.Information, 3000)
            
    def prompt_set_journal_dir(self):
        """Prompts the user to select a directory for saving journal entries."""
        # Get current journal directory from settings or use default
        current_dir = self.settings.value("journal_dir")
        if not current_dir:
            # Default to ~/Documents/Personal/Audio Journal/
            home_dir = os.path.expanduser("~")
            current_dir = os.path.join(home_dir, "Documents", "Personal", "Audio Journal")
            
        # Open directory selection dialog
        journal_dir = QFileDialog.getExistingDirectory(
            None,
            "Select Journal Directory",
            current_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if journal_dir:
            # Save the new journal directory in settings
            self.settings.setValue("journal_dir", journal_dir)
            logging.info(f"Journal directory changed to: {journal_dir}")
            
            # Update the journal manager with the new directory and existing prompt
            summary_prompt = self.settings.value("summary_prompt")
            self.journal_manager = JournalingManager(output_dir=journal_dir, summary_prompt=summary_prompt)
            
            # Show confirmation to user
            self.tray_icon.showMessage(
                "Settings Updated", 
                f"Journal directory set to:\n{journal_dir}", 
                QSystemTrayIcon.Information, 
                3000
            )
            
    def prompt_edit_summary_prompt(self):
        """Opens a dialog to edit the summary prompt template used for journal entries."""
        # Get current summary prompt from settings or use default
        default_prompt = "Provide a 1-2 sentence summary of this text. DO NOT add any commentary, analysis, or description of the text. Only extract and condense the main points:"
        current_prompt = self.settings.value("summary_prompt", default_prompt)
        
        # Create dialog
        dialog = QDialog(None)
        dialog.setWindowTitle("Edit Summary Prompt")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(300)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Add explanation label
        explanation = QLabel(
            "Customize the prompt used to generate summaries for journal entries. "
            "You can include personal context like your name or background to make summaries more relevant. "
            "The transcript will be appended to the end of this prompt."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        
        # Add text edit for prompt
        prompt_edit = QTextEdit(dialog)
        prompt_edit.setPlainText(current_prompt)
        layout.addWidget(prompt_edit)
        
        # Add reset button
        reset_button = QPushButton("Reset to Default", dialog)
        reset_button.clicked.connect(lambda: prompt_edit.setPlainText(default_prompt))
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add reset button to button box layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(reset_button)
        button_layout.addStretch()
        button_layout.addWidget(button_box)
        layout.addLayout(button_layout)
        
        # Show dialog and process result
        if dialog.exec_() == QDialog.Accepted:
            new_prompt = prompt_edit.toPlainText().strip()
            if new_prompt:
                # Save to settings
                self.settings.setValue("summary_prompt", new_prompt)
                logging.info("Summary prompt updated")
                
                # Update the journal manager with the new prompt
                self.journal_manager.set_summary_prompt(new_prompt)
                
                # Show confirmation to user
                self.tray_icon.showMessage(
                    "Settings Updated", 
                    "Summary prompt has been updated.", 
                    QSystemTrayIcon.Information, 
                    3000
                )
                
    def prompt_edit_format_prompt(self):
        """Opens a dialog to edit the format prompt template used for journal entries."""
        # Get current format prompt from settings or use default
        default_prompt = "Format this transcription into well-structured paragraphs. DO NOT add any commentary, analysis, or description. DO NOT change any words or meaning. Only add proper paragraph breaks, fix punctuation, and improve readability:"
        current_prompt = self.settings.value("format_prompt", default_prompt)
        
        # Create dialog
        dialog = QDialog(None)
        dialog.setWindowTitle("Edit Format Prompt")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(300)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Add explanation label
        explanation = QLabel(
            "Customize the prompt used to format transcriptions for journal entries. "
            "You can specify how you want paragraphs structured, punctuation handled, or any specific formatting preferences. "
            "The transcript will be appended to the end of this prompt."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        
        # Add text edit for prompt
        prompt_edit = QTextEdit(dialog)
        prompt_edit.setPlainText(current_prompt)
        layout.addWidget(prompt_edit)
        
        # Add reset button
        reset_button = QPushButton("Reset to Default", dialog)
        reset_button.clicked.connect(lambda: prompt_edit.setPlainText(default_prompt))
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add reset button to button box layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(reset_button)
        button_layout.addStretch()
        button_layout.addWidget(button_box)
        layout.addLayout(button_layout)
        
        # Show dialog and process result
        if dialog.exec_() == QDialog.Accepted:
            new_prompt = prompt_edit.toPlainText().strip()
            if new_prompt:
                # Save to settings
                self.settings.setValue("format_prompt", new_prompt)
                logging.info("Format prompt updated")
                
                # Update the journal manager with the new prompt
                self.journal_manager.set_format_prompt(new_prompt)
                
                # Show confirmation to user
                self.tray_icon.showMessage(
                    "Settings Updated", 
                    "Format prompt has been updated.", 
                    QSystemTrayIcon.Information, 
                    3000
                )
                
    def prompt_set_record_hotkey(self):
        """Opens a dialog to set the recording hotkey."""
        # Get current hotkey from settings or use default
        default_hotkey = "cmd+shift+r"
        current_hotkey = self.settings.value("record_hotkey", default_hotkey)
        
        # Create dialog
        dialog = QDialog(None)
        dialog.setWindowTitle("Set Recording Hotkey")
        dialog.setMinimumWidth(400)
        dialog.setMinimumHeight(200)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Add explanation label
        explanation = QLabel(
            "Set the hotkey combination for starting/stopping recording. "
            "Use format like 'cmd+shift+r' or 'ctrl+alt+r'. "
            "Changes will take effect after restarting the application."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        
        # Add text edit for hotkey
        hotkey_edit = QTextEdit(dialog)
        hotkey_edit.setPlainText(current_hotkey)
        hotkey_edit.setMaximumHeight(60)
        layout.addWidget(hotkey_edit)
        
        # Add platform-specific examples
        examples = QLabel(
            "Examples:\n"
            "- macOS: cmd+shift+r, cmd+alt+r\n"
            "- Windows: ctrl+shift+r, ctrl+alt+r"
        )
        examples.setWordWrap(True)
        layout.addWidget(examples)
        
        # Add reset button
        reset_button = QPushButton("Reset to Default", dialog)
        reset_button.clicked.connect(lambda: hotkey_edit.setPlainText(default_hotkey))
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add reset button to button box layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(reset_button)
        button_layout.addStretch()
        button_layout.addWidget(button_box)
        layout.addLayout(button_layout)
        
        # Show dialog and process result
        if dialog.exec_() == QDialog.Accepted:
            new_hotkey = hotkey_edit.toPlainText().strip()
            if new_hotkey:
                try:
                    # Validate the hotkey format
                    self._parse_hotkey_string(new_hotkey)
                    
                    # Save to settings
                    self.settings.setValue("record_hotkey", new_hotkey)
                    logging.info(f"Recording hotkey updated to: {new_hotkey}")
                    
                    # Show confirmation to user
                    self.tray_icon.showMessage(
                        "Settings Updated", 
                        f"Recording hotkey set to: {new_hotkey}\nRestart the application for changes to take effect.", 
                        QSystemTrayIcon.Information, 
                        5000
                    )
                except Exception as e:
                    # Show error message
                    QMessageBox.critical(
                        None,
                        "Invalid Hotkey Format",
                        f"The hotkey format is invalid: {str(e)}\n\nPlease use format like 'cmd+shift+r' or 'ctrl+alt+r'."
                    )
                    
    def prompt_set_journal_hotkey(self):
        """Opens a dialog to set the journal hotkey."""
        # Get current hotkey from settings or use default
        default_hotkey = "cmd+shift+j"
        current_hotkey = self.settings.value("journal_hotkey", default_hotkey)
        
        # Create dialog
        dialog = QDialog(None)
        dialog.setWindowTitle("Set Journal Hotkey")
        dialog.setMinimumWidth(400)
        dialog.setMinimumHeight(200)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Add explanation label
        explanation = QLabel(
            "Set the hotkey combination for starting/stopping journal recording. "
            "Use format like 'cmd+shift+j' or 'ctrl+alt+j'. "
            "Changes will take effect after restarting the application."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        
        # Add text edit for hotkey
        hotkey_edit = QTextEdit(dialog)
        hotkey_edit.setPlainText(current_hotkey)
        hotkey_edit.setMaximumHeight(60)
        layout.addWidget(hotkey_edit)
        
        # Add platform-specific examples
        examples = QLabel(
            "Examples:\n"
            "- macOS: cmd+shift+j, cmd+alt+j\n"
            "- Windows: ctrl+shift+j, ctrl+alt+j"
        )
        examples.setWordWrap(True)
        layout.addWidget(examples)
        
        # Add reset button
        reset_button = QPushButton("Reset to Default", dialog)
        reset_button.clicked.connect(lambda: hotkey_edit.setPlainText(default_hotkey))
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add reset button to button box layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(reset_button)
        button_layout.addStretch()
        button_layout.addWidget(button_box)
        layout.addLayout(button_layout)
        
        # Show dialog and process result
        if dialog.exec_() == QDialog.Accepted:
            new_hotkey = hotkey_edit.toPlainText().strip()
            if new_hotkey:
                try:
                    # Validate the hotkey format
                    self._parse_hotkey_string(new_hotkey)
                    
                    # Save to settings
                    self.settings.setValue("journal_hotkey", new_hotkey)
                    logging.info(f"Journal hotkey updated to: {new_hotkey}")
                    
                    # Show confirmation to user
                    self.tray_icon.showMessage(
                        "Settings Updated", 
                        f"Journal hotkey set to: {new_hotkey}\nRestart the application for changes to take effect.", 
                        QSystemTrayIcon.Information, 
                        5000
                    )
                except Exception as e:
                    # Show error message
                    QMessageBox.critical(
                        None,
                        "Invalid Hotkey Format",
                        f"The hotkey format is invalid: {str(e)}\n\nPlease use format like 'cmd+shift+j' or 'ctrl+alt+j'."
                    )
    
    def handle_transcription(self, text):
        """Handle the transcribed text."""
        if not text or not text.strip():
            logging.warning("Empty transcription received")
            self.tray_icon.showMessage(
                "Voice Typer", 
                "No speech detected. Please try again.", 
                QSystemTrayIcon.Warning, 
                3000
            )
            return

        try:
            if self.journaling_mode:
                # Handle as journal entry
                self.handle_journal_entry(text)
            else:
                # Regular transcription mode
                # Get the output file path from settings
                output_file = self.settings.value("output_file")
                if not output_file:
                    logging.error("No output file configured")
                    self.tray_icon.showMessage(
                        "Voice Typer", 
                        "No output file configured. Please set an output file.", 
                        QSystemTrayIcon.Critical, 
                        5000
                    )
                    return

                # Ensure the output directory exists
                output_dir = os.path.dirname(output_file)
                os.makedirs(output_dir, exist_ok=True)

                # Format the transcription with timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                formatted_text = f"## {timestamp}\n\n{text}\n\n---\n\n"

                # Append to the output file
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(formatted_text)

                # Type the text using keyboard module
                import keyboard as kb
                kb.write(text)

                # Show success notification
                self.tray_icon.showMessage(
                    "Voice Typer", 
                    f"Transcription saved to {output_file}", 
                    QSystemTrayIcon.Information, 
                    3000
                )

        except Exception as e:
            logging.error(f"Error handling transcription: {e}", exc_info=True)
            self.tray_icon.showMessage(
                "Voice Typer", 
                f"Error: {str(e)}", 
                QSystemTrayIcon.Critical, 
                5000
            )
    
    def toggle_recording(self):
        """Toggle recording state."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
            
    def toggle_journal_mode(self):
        """Toggle journaling mode and start/stop recording."""
        if self.is_recording:
            # If already recording, stop it
            logging.info("Stopping recording in journal mode")
            self.stop_recording()
        else:
            # Start recording in journal mode
            self.journaling_mode = True
            logging.info("Journal mode activated")
            self.start_recording()
            self.tray_icon.showMessage(
                "Voice Typer",
                "Journal mode activated. Recording will be saved as a journal entry.",
                QSystemTrayIcon.Information,
                2000
            )
            
    def handle_journal_entry(self, text):
        """Handle creating a journal entry with the transcribed text."""
        try:
            # Create journal entry
            entry = self.journal_manager.create_journal_entry(
                transcription=text,
                audio_data=self.audio_data if hasattr(self, 'audio_data') else None
            )
            
            # Show a notification
            self.tray_icon.showMessage(
                "Voice Typer",
                f"Journal entry created with timestamp: {entry.get('timestamp', 'Unknown')}",
                QSystemTrayIcon.Information,
                3000
            )
            
            # Reset journaling mode
            self.journaling_mode = False
            
        except Exception as e:
            logging.error(f"Error creating journal entry: {e}", exc_info=True)
            self.tray_icon.showMessage(
                "Voice Typer",
                f"Error creating journal entry: {e}",
                QSystemTrayIcon.Critical,
                5000
            )
            
    def start_recording(self):
        """Start recording audio in a separate thread."""
        if not hasattr(self, 'recording_thread') or not self.recording_thread or not self.recording_thread.isRunning():
            logging.info("Starting recording...")
            self.is_recording = True
            self.last_recording_time = time.time()
            
            # Update UI
            self.toggle_action.setText("Stop Recording")
            self.update_icon(True)
            self.tray_icon.setToolTip("Voice Typer (Recording...)")
            
            # Create recording thread
            self.recording_thread = RecordingThread()
            
            # Connect signals
            self.recording_thread.finished.connect(self.handle_recording_finished)
            self.recording_thread.error.connect(self.handle_error)
            
            # Start recording
            self.recording_thread.start()
    
    def stop_recording(self):
        """Stop the recording thread."""
        if not self.is_recording:
            logging.info("Not recording, nothing to stop")
            return
            
        logging.info("Stopping recording...")
        self.is_recording = False
        
        # Update UI immediately on the main thread
        self.toggle_action.setText("Start Recording")
        self.update_icon(False)
        self.tray_icon.setToolTip("Voice Typer (Ready)")
        
        # Stop recording thread if it exists
        if self.recording_thread and self.recording_thread.isRunning():
            logging.info("Sending stop signal to recording thread")
            self.recording_thread.stop()
    
    def handle_error(self, error_msg):
        """Handle errors from the worker thread."""
        logging.error(f"Error: {error_msg}")
        self.tray_icon.showMessage("Error", error_msg, QSystemTrayIcon.Critical)
        self.stop_recording()
    
    def _clear_transcriber_references(self):
        logging.info("[VoiceTyper] Clearing self.transcriber reference after it finished and deleteLater was called.")
        self.transcriber = None

    def show_calendar(self):
        """Show the calendar widget with recording dates."""
        try:
            # If the calendar is already open, bring it to the front
            if hasattr(self, "calendar_widget") and self.calendar_widget is not None:
                if self.calendar_widget.isVisible():
                    self.calendar_widget.raise_()
                    self.calendar_widget.activateWindow()
                    return
            # Get the journal recordings directory
            recordings_dir = os.path.join(self.journal_manager.output_dir, "recordings")
            # Create and show the calendar widget
            self.calendar_widget = CalendarWidget(None, recordings_dir)
            self.calendar_widget.date_selected.connect(self.on_calendar_date_selected)
            self.calendar_widget.destroyed.connect(lambda: setattr(self, "calendar_widget", None))
            self.calendar_widget.show()
        except Exception as e:
            logging.error(f"Error showing calendar: {e}", exc_info=True)
            self.tray_icon.showMessage(
                "WhisperNotes", 
                f"Error showing calendar: {str(e)}", 
                QSystemTrayIcon.MessageIcon.Critical, 
                5000
            )

    def on_calendar_date_selected(self, date_str):
        """
        Handle a date selection from the calendar.
        Args:
            date_str (str): The selected date in YYYY-MM-DD format.
        """
        try:
            logging.info(f"Calendar date selected: {date_str}")
            self.tray_icon.showMessage(
                "WhisperNotes",
                f"Selected recordings from {date_str}",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
        except Exception as e:
            logging.error(f"Error handling calendar date selection: {e}", exc_info=True)



    def quit(self):
        """Quit the application."""
        try:
            logging.info("Quitting application...")
            self.stop_recording() # This should ideally wait for the recording thread to finish.

            # Stop watchdog timer
            if hasattr(self, 'watchdog_timer'):
                self.watchdog_timer.stop()

            # Cleanup transcription thread if running
            try:
                if hasattr(self, 'transcription_thread') and self.transcription_thread and self.transcription_thread.isRunning():
                    logging.info("Requesting transcription worker to stop and waiting for thread to finish...")
                    if hasattr(self, 'transcriber') and self.transcriber: # self.transcriber is the QObject worker
                        self.transcriber.request_stop() # New method
                    
                    self.transcription_thread.quit() # Ask event loop to stop (harmless if no event loop)
                    if not self.transcription_thread.wait(5000): # Increased timeout
                        logging.warning("Transcription thread did not finish in time.")
                        # self.transcription_thread.terminate() # Avoid if possible, can lead to instability
            except RuntimeError as e:
                logging.warning(f"RuntimeError during transcription thread cleanup: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during transcription thread cleanup: {e}", exc_info=True)

            # Cleanup recording thread if running (stop_recording should handle its stopping, this is for waiting)
            try:
                if hasattr(self, 'recording_thread') and self.recording_thread and self.recording_thread.isRunning():
                    logging.info("Waiting for recording thread to finish...")
                    # self.recording_thread.stop() was already called by self.stop_recording()
                    self.recording_thread.quit() # Ask event loop to stop
                    if not self.recording_thread.wait(5000): # Increased timeout
                        logging.warning("Recording thread did not finish in time.")
            except RuntimeError as e:
                logging.warning(f"RuntimeError during recording thread cleanup: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during recording thread cleanup: {e}", exc_info=True)

            # Cleanup model loader thread if running
            try:
                if hasattr(self, 'model_thread') and self.model_thread and self.model_thread.isRunning():
                    logging.info("Waiting for model loader thread to finish...")
                    # ModelLoader.run() is blocking; no easy stop. quit() is a hint if it had an event loop.
                    self.model_thread.quit()
                    if not self.model_thread.wait(5000): # Increased timeout
                        logging.warning("Model loader thread did not finish in time.")
            except RuntimeError as e:
                logging.warning(f"RuntimeError during model loader thread cleanup: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during model loader thread cleanup: {e}", exc_info=True)

            # Cleanup hotkey listener
            if hasattr(self, 'listener'):
                logging.info("Stopping hotkey listener...")
                self.listener.stop()
                try:
                    # Attempt to join the listener thread to ensure it exits cleanly
                    # This might block, so use with caution or make listener a QThread
                    # For pynput, listener.stop() is usually sufficient and join isn't always exposed/needed this way.
                    if hasattr(self.listener, 'join') and callable(self.listener.join):
                         self.listener.join() # Remove timeout=1.0
                except Exception as e:
                    logging.warning(f"Error joining hotkey listener thread: {e}")

            logging.info("All cleanup attempts finished. Quitting QCoreApplication.")
            QCoreApplication.instance().quit()

        except Exception as e:
            logging.error(f"Error during quit: {e}", exc_info=True)
            QCoreApplication.instance().quit() # Ensure quit is called even if an error occurs in cleanup



if __name__ == "__main__":
    # Create the Qt Application
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    
    # Create and show the voice typer
    whisper_notes = WhisperNotes(app)
    
    # Start the event loop
    sys.exit(app.exec())
