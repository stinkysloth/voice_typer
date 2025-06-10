 HEAD

"""
WhisperNotes Main Application Module


This module implements the main application logic for WhisperNotes, a cross-platform (macOS, Windows, Linux) desktop voice journaling and note-taking app using PySide6 (Qt), Whisper ASR, and system tray integration.

Key Responsibilities:
---------------------
- Application entry point and event loop
- System tray icon, actions, and notifications
- Global hotkey registration (Cmd+Shift+R/J, Cmd+Q)
- Audio recording and transcription using Whisper
- Journaling integration and clipboard management
- Thread-safe communication via Qt signals/slots
- Robust error handling and resource cleanup

Architecture & Structure:
------------------------
- Platform-specific clipboard and accessibility support
- Worker threads for recording, transcription, and model loading
- Main `WhisperNotes` class orchestrates UI, hotkeys, and threading
- Modular fallback logic for missing dependencies (journaling, exceptions)
- Mutex-protected critical sections for thread safety
- Modular cleanup helpers for graceful shutdown

Constraints & Developer Notes:
-----------------------------
- Requires PySide6, pynput, sounddevice, librosa, and Whisper
- On macOS, terminal/Python must have Accessibility permissions for hotkeys
- All UI and signal/slot connections must be made from the main Qt thread
- File should be split into smaller modules if it exceeds 500 lines (see refactor plan at bottom)
- See README.md for setup, permissions, and troubleshooting

"""
 2819f8b (feat: Add task for batch audio import feature)
#!/usr/bin/env python3
import os
import sys
import time
import platform
from PySide6.QtCore import QObject
print("Main process Python executable:", sys.executable)

# Import platform-specific modules for auto-paste
if sys.platform == 'darwin':  # macOS
    import Quartz
    from AppKit import NSApplication, NSApp
    from PySide6.QtCore import QTimer
elif sys.platform == 'win32':  # Windows
    import win32clipboard
    import win32con
    import win32gui
    import win32api
    import win32process
    import win32ui
    from ctypes import wintypes
    import ctypes
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    from pywinauto import Application
    import uiautomation as auto
else:  # Linux
    import subprocess
    import gi
    gi.require_version('Gtk', '3.0')
    from gi.repository import Gtk, Gdk, GdkX11  # noqa: F401
    import Xlib.display
    import Xlib.XK
    import Xlib.X
    import Xlib.ext.xtest
    import Xlib.error
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
 HEAD
< HEAD:whisper_notes.py
from PySide6.QtWidgets import (QApplication, QSystemTrayIcon, QMenu, QMessageBox, QFileDialog,
                               QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton, QDialogButtonBox)
=

 2819f8b (feat: Add task for batch audio import feature)
import traceback
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (QApplication, QSystemTrayIcon, QMenu, QMessageBox, 
                             QFileDialog, QDialog, QVBoxLayout, QTextEdit, 
                             QDialogButtonBox, QLabel)
 HEAD
> 2819f8b (feat: Add task for batch audio import feature):archive/whisper_notes.py.bak

from PySide6.QtCore import Signal, Slot
 2819f8b (feat: Add task for batch audio import feature)
from PySide6.QtGui import QIcon, QAction, QPixmap, QPainter, QColor
from PySide6.QtCore import (QObject, QThread, Signal, Qt, QTimer, QMutex, 
                           QCoreApplication, QSettings, QStandardPaths)

 HEAD

# Modular imports
from tray import TrayManager
from hotkeys import HotkeyManager
from audio import RecordingThread
from transcription import ModelLoader, TranscriptionWorker
from journaling import JournalingManager

 2819f8b (feat: Add task for batch audio import feature)
# Import custom exceptions
try:
    from exceptions import (
        AudioRecordingError, AudioSaveError, TranscriptionError, ModelError,
        JournalingError, FileSystemError, ConfigurationError, handle_error
    )
    CUSTOM_EXCEPTIONS_AVAILABLE = True
except ImportError:
    # Fallback if custom exceptions aren't available
    CUSTOM_EXCEPTIONS_AVAILABLE = False
    logging.warning("Custom exceptions not found. Using basic error handling.")
    
    # Define basic exception classes for fallback
    class AudioRecordingError(Exception): pass
    class AudioSaveError(Exception): pass
    class TranscriptionError(Exception): pass
    class ModelError(Exception): pass
    class JournalingError(Exception): pass
    class FileSystemError(Exception): pass
    class ConfigurationError(Exception): pass
    
    def handle_error(error: Exception, context: str = "") -> str:
        logging.error(f"Error in {context or 'unknown context'}: {str(error)}\n{traceback.format_exc()}")
        return f"An error occurred: {str(error)}"

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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("whisper_notes.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('WhisperNotes')
logger.info("Logging system initialized")

 HEAD
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

# ModelLoader is now imported from transcription.py
 2819f8b (feat: Add task for batch audio import feature)


# No longer needed: all transcription is now handled by an external subprocess


 HEAD
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
            logging.debug("Starting audio recording")
            self.audio_data = []
            
            def callback(indata, frames, time, status):
                if status:
                    logging.warning(f"Audio status: {status}")
                    if status.input_overflow:
                        logging.warning("Input overflow in audio stream")
                        raise AudioRecordingError("Audio input overflow - system can't keep up with recording")
                    elif status.input_error:
                        raise AudioRecordingError("Error in audio input device")
                        
                if self.stop_flag:
                    raise sd.CallbackStop()
                    
                if indata is not None and len(indata) > 0:
                    self.audio_data.append(indata.copy())
            
            # Check if input device is available
            devices = sd.query_devices()
            if not devices:
                raise AudioRecordingError("No audio input devices found")
                
            default_input = sd.default.device[0]
            if default_input >= len(devices):
                raise AudioRecordingError(f"Default input device {default_input} is out of range")
                
            device_info = devices[default_input]
            if device_info['max_input_channels'] == 0:
                raise AudioRecordingError("Default device has no input channels")
            
            logging.info(f"Using audio input device: {device_info['name']} (sample rate: {device_info['default_samplerate']} Hz)")
            
            # Start recording with a larger queue size
            with sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype='float32',
                callback=callback,
                blocksize=4096,
                device=None,  # Use default device
                latency='high'  # Better for stability
            ) as stream:
                logging.debug("Audio stream started")
                start_time = time.time()
                
                try:
                    while not self.stop_flag and (time.time() - start_time) < self.max_duration:
                        # Process events while waiting
                        QCoreApplication.processEvents()
                        time.sleep(0.1)  # Small sleep to prevent busy waiting
                    
                    logging.debug("Stopping audio recording")
                    stream.stop()
                    
                    if not self.audio_data:
                        raise AudioRecordingError("No audio data was recorded. Please check your microphone.")
                        
                    audio_data = np.concatenate(self.audio_data, axis=0)
                    
                    # Simple audio validation
                    if np.max(np.abs(audio_data)) < 0.001:  # Very quiet audio
                        logging.warning("Audio signal is very quiet - possible microphone issue")
                    
                    self.finished.emit(audio_data)
                    
                except sd.PortAudioError as e:
                    raise AudioRecordingError(f"Audio device error: {str(e)}")
                except Exception as e:
                    raise AudioRecordingError(f"Error during recording: {str(e)}")
                    
        except Exception as e:
            # Use our custom error handling
            error_context = "audio recording"
            error_msg = handle_error(e, error_context)
            
            # Emit the error signal with the user-friendly message
            if not isinstance(e, (AudioRecordingError, AudioSaveError)):
                # If it's not one of our custom exceptions, wrap it in an AudioRecordingError
                e = AudioRecordingError(str(e))
                
            if hasattr(self, 'error'):
                self.error.emit(str(e))
            else:
                logging.error(f"Error signal not available: {error_msg}")
                
            # Re-raise the exception with the original traceback
            raise
    
    def stop(self):
        """Signal the thread to stop recording."""
        logging.info("Stopping recording thread...")
        self.stop_flag = True


class WhisperNotes(QObject):
    """Main application class for Whisper Notes."""

# TranscriptionWorker is now imported from transcription.py

# RecordingThread is now imported from audio.py



class WhisperNotes(QObject):
    # Signals to safely show dialogs from main thread
    show_error_dialog = Signal(str, str)
    show_info_dialog = Signal(str, str)
    show_warning_dialog = Signal(str, str)
    show_config_dialog = Signal()

    """
    Main application class for WhisperNotes.

    Orchestrates the system tray, hotkeys, audio recording, transcription,
    journaling, and thread management. Handles all user interactions and
    coordinates between UI and background workers. Entry point for the app logic.

    Args:
        app (QApplication): The main Qt application instance.

    Constraints:
        - All UI and signal/slot connections must be made from the main Qt thread.
        - Platform-specific permissions may be required (see README).
    """
 2819f8b (feat: Add task for batch audio import feature)
    toggle_recording_signal = Signal()
    toggle_journal_signal = Signal()
    quit_signal = Signal()
    
    def __init__(self, app):
        super().__init__()
 HEAD

        # Connect dialog signals to slots
        self.show_error_dialog.connect(self._show_error_dialog_slot)
        self.show_info_dialog.connect(self._show_info_dialog_slot)
        self.show_warning_dialog.connect(self._show_warning_dialog_slot)
        self.show_config_dialog.connect(self._show_config_dialog_slot)

 2819f8b (feat: Add task for batch audio import feature)
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
        self.auto_paste_enabled = True  # Enable auto-paste by default
 HEAD
        
        # Initialize platform-specific settings
        self._init_platform_specific()
        


        # Template system
        from template_manager import TemplateManager
        from template_config_dialog import TemplateConfigDialog
        self.template_manager = TemplateManager()
        self.TemplateConfigDialog = TemplateConfigDialog
        
        # Load template configurations and register hotkeys
        self._load_template_configs()

        # Initialize platform-specific settings
        self._init_platform_specific()

        # Define tray dialog handler before tray setup
        def open_template_config_dialog():
            """
            Open the Template Configuration dialog from the tray menu.
            """
            # Emit signal to open config dialog on main thread
            self.show_config_dialog.emit()
        self.open_template_config_dialog = open_template_config_dialog

        # Ensure tray and hotkeys are set up
        logging.info("Setting up TrayManager in __init__")
        self.tray_manager = TrayManager(
            app=self.app,
            parent=self,
            on_record=self.toggle_recording,
            on_journal=self.toggle_journal_mode,
            on_quit=self.quit,
            on_edit_prompt=self.prompt_edit_summary_prompt,
            on_set_journal_dir=self.prompt_set_journal_dir,
            on_configure_templates=self.open_template_config_dialog,
            on_import_audio=self.import_audio_files
        )
        logging.info("Setting up HotkeyManager in __init__")
        self.hotkey_manager = HotkeyManager(
            on_toggle_recording=self.toggle_recording,
            on_toggle_journal=self.toggle_journal_mode,
            on_quit=self.quit
        )

        # Initialize QSettings for persistent output file path
        self.settings = QSettings("WhisperNotes", "WhisperNotes")
        if not self.settings.contains("output_file"):
            documents_path = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
            default_output = os.path.join(documents_path, "WhisperNotesTranscriptions.md")
            self.settings.setValue("output_file", default_output)
        logging.info(f"Output Markdown file initialized to: {self.settings.value('output_file')}")

        # Initialize journaling manager with directory and prompt from settings if available
        journal_dir = self.settings.value("journal_dir")
        summary_prompt = self.settings.value("summary_prompt")
        
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
        
        self.toggle_recording_signal.connect(self.toggle_recording)
        self.toggle_journal_signal.connect(self.toggle_journal_mode)
        self.quit_signal.connect(self.quit)

        self.load_model()
        self.setup_watchdog()

        # Connect dialog signals to slots
        self.show_error_dialog.connect(self._show_error_dialog_slot)
        self.show_info_dialog.connect(self._show_info_dialog_slot)
        self.show_warning_dialog.connect(self._show_warning_dialog_slot)
        self.show_config_dialog.connect(self._show_config_dialog_slot)

 2819f8b (feat: Add task for batch audio import feature)
    def _init_platform_specific(self):
        """Initialize platform-specific settings and modules."""
        if sys.platform == 'darwin':  # macOS
            # Initialize NSApplication for macOS accessibility
            self.nsapp = NSApplication.sharedApplication()
        elif sys.platform == 'win32':  # Windows
            # Initialize Windows-specific settings
            self._init_windows()
        # Linux initialization handled in the paste method
        
    def _init_windows(self):
        """Initialize Windows-specific settings."""
        # Ensure the process is DPI aware for high-DPI displays
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)  # Process DPI aware
        except:
            pass  # Not critical if this fails
    
    def paste_at_cursor(self, text):
        """
        Paste text at the current cursor position in the active window.
        
        Args:
            text: The text to paste
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not text or not self.auto_paste_enabled:
            return False
            
        try:
            if sys.platform == 'darwin':  # macOS
                return self._paste_macos(text)
            elif sys.platform == 'win32':  # Windows
                return self._paste_windows(text)
            else:  # Linux
                return self._paste_linux(text)
        except Exception as e:
            logging.error(f"Error pasting text at cursor: {e}")
            return False
    
    def _paste_macos(self, text):
        """Paste text at cursor position on macOS."""
        try:
            # Save current clipboard content
            old_clipboard = QApplication.clipboard()
            old_text = old_clipboard.text()
            
            # Set new text to clipboard
            old_clipboard.setText(text)
            
            # Get the current application
            system_events = Quartz.NSApplication.sharedApplication()
            
            # Simulate Cmd+V to paste
            source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateCombinedSessionState)
            
            # Press Cmd down
            cmd_down = Quartz.CGEventCreateKeyboardEvent(source, 0x37, True)
            Quartz.CGEventPost(Quartz.kCGSessionEventTap, cmd_down)
            
            # Press V down
            v_down = Quartz.CGEventCreateKeyboardEvent(source, 0x09, True)
            Quartz.CGEventSetFlags(v_down, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGSessionEventTap, v_down)
            
            # Release V
            v_up = Quartz.CGEventCreateKeyboardEvent(source, 0x09, False)
            Quartz.CGEventSetFlags(v_up, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGSessionEventTap, v_up)
            
            # Release Cmd
            cmd_up = Quartz.CGEventCreateKeyboardEvent(source, 0x37, False)
            Quartz.CGEventPost(Quartz.kCGSessionEventTap, cmd_up)
            
            # Restore old clipboard after a short delay
            QTimer.singleShot(1000, lambda: old_clipboard.setText(old_text))
            
            return True
            
        except Exception as e:
            logging.error(f"macOS paste error: {e}")
            return False
    
    def _paste_windows(self, text):
        """Paste text at cursor position on Windows."""
        try:
            # Save current clipboard content
            win32clipboard.OpenClipboard()
            try:
                old_data = win32clipboard.GetClipboardData()
            except:
                old_data = None
            win32clipboard.CloseClipboard()
            
            # Set new text to clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text, win32con.CF_UNICODETEXT)
            win32clipboard.CloseClipboard()
            
            # Get the foreground window
            hwnd = user32.GetForegroundWindow()
            
            # Send Ctrl+V to paste
            user32.keybd_event(0x11, 0, 0, 0)  # Ctrl down
            user32.keybd_event(0x56, 0, 0, 0)  # V down
            user32.keybd_event(0x56, 0, 2, 0)  # V up
            user32.keybd_event(0x11, 0, 2, 0)  # Ctrl up
            
            # Restore old clipboard after a short delay
            def restore_clipboard():
                try:
                    win32clipboard.OpenClipboard()
                    win32clipboard.EmptyClipboard()
                    if old_data is not None:
                        win32clipboard.SetClipboardText(old_data, win32con.CF_UNICODETEXT)
                    win32clipboard.CloseClipboard()
                except:
                    pass
                    
            QTimer.singleShot(1000, restore_clipboard)
            
            return True
            
        except Exception as e:
            logging.error(f"Windows paste error: {e}")
            return False
    
    def _paste_linux(self, text):
        """Paste text at cursor position on Linux."""
        try:
            # Save current clipboard content
            old_clipboard = QApplication.clipboard()
            old_text = old_clipboard.text()
            
            # Set new text to clipboard
            old_clipboard.setText(text)
            
            # Get display and root window
            display = Xlib.display.Display()
            root = display.screen().root
            
            # Get the current focus window
            current_focus = display.get_input_focus().focus
            
            # Create the key press event
            keycode = display.keysym_to_keycode(Xlib.XK.string_to_keysym('v'))
            
            # Press Control
            Xlib.ext.xtest.fake_input(display, Xlib.X.KeyPress, 37, Xlib.X.CurrentTime, current_focus, 0, 0, 0)
            
            # Press V
            Xlib.ext.xtest.fake_input(display, Xlib.X.KeyPress, keycode, Xlib.X.CurrentTime, current_focus, 0, 0, 0)
            
            # Release V
            Xlib.ext.xtest.fake_input(display, Xlib.X.KeyRelease, keycode, Xlib.X.CurrentTime, current_focus, 0, 0, 0)
            
            # Release Control
            Xlib.ext.xtest.fake_input(display, Xlib.X.KeyRelease, 37, Xlib.X.CurrentTime, current_focus, 0, 0, 0)
            
            display.sync()
            display.close()
            
            # Restore old clipboard after a short delay
            QTimer.singleShot(1000, lambda: old_clipboard.setText(old_text))
            
            return True
            
        except Exception as e:
            logging.error(f"Linux paste error: {e}")
            return False
        
        # Initialize QSettings for persistent output file path
        self.settings = QSettings("WhisperNotes", "WhisperNotes")
        if not self.settings.contains("output_file"):
            documents_path = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
            default_output = os.path.join(documents_path, "WhisperNotesTranscriptions.md")
            self.settings.setValue("output_file", default_output)
        logging.info(f"Output Markdown file initialized to: {self.settings.value('output_file')}")
        
 HEAD
        # Initialize journaling manager with directory and prompts from settings if available
        journal_dir = self.settings.value("journal_dir")
        summary_prompt = self.settings.value("summary_prompt")
        format_prompt = self.settings.value("format_prompt")

        # Initialize journaling manager with directory and prompt from settings if available
        journal_dir = self.settings.value("journal_dir")
        summary_prompt = self.settings.value("summary_prompt")
 2819f8b (feat: Add task for batch audio import feature)
        
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
 HEAD
            
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
    

        
 2819f8b (feat: Add task for batch audio import feature)
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
 HEAD
        self.update_icon(False)
        
    def toggle_recording(self):
        """Toggle recording state."""

        self.tray_manager.update_icon(False)

    def toggle_recording(self):
        """Toggle recording state."""
        logging.info("[SLOT ENTRY] toggle_recording() called. is_recording=%s", self.is_recording)
 2819f8b (feat: Add task for batch audio import feature)
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
 HEAD
            
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


    def start_recording(self):
        """Start recording audio in a separate thread."""
        logging.info("start_recording() called. Checking if thread can be started...")
        if not hasattr(self, 'recording_thread') or not self.recording_thread or not self.recording_thread.isRunning():
            logging.info("Starting recording...")
            self.is_recording = True
            self.last_recording_time = time.time()

            # Update tray icon and tooltip only
            self.tray_manager.update_icon(True)
            self.tray_manager.tray_icon.setToolTip("Voice Typer (Recording...)")

            # Create recording thread
            self.recording_thread = RecordingThread()

            # Connect signals
            self.recording_thread.finished.connect(self.handle_recording_finished)
            self.recording_thread.error.connect(self.handle_error)

            # Start recording
            self.recording_thread.start()
        else:
            logging.info("Recording thread already running or exists; not starting a new one.")

    def toggle_journal_mode(self):
        """Toggle journaling mode and start/stop recording."""
        logging.info("[SLOT ENTRY] toggle_journal_mode() called. is_recording=%s", self.is_recording)
        if self.is_recording:
            logging.info("Stopping recording in journal mode")
            self.stop_recording()
        else:
            self.journaling_mode = True
            logging.info("Journal mode activated")
            self.start_recording()
            self.tray_manager.tray_icon.showMessage(
 2819f8b (feat: Add task for batch audio import feature)
                "WhisperNotes",
                "Journal mode activated. Recording will be saved as a journal entry.",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
 HEAD
            
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
        try:
            logger.debug("Setting up system tray...")
            
            # Ensure we have an application instance
            if not QApplication.instance():
                logger.error("No QApplication instance found!")
                return
                
            # Check if system tray is available
            if not QSystemTrayIcon.isSystemTrayAvailable():
                logger.error("System tray is not available on this system")
                # Show a message box if running in GUI mode
                if QApplication.instance().topLevelWindows():
                    QMessageBox.critical(None, "Error", "System tray is not available on this system")
                return
                
            # Create and configure the tray icon
            self.tray_icon = QSystemTrayIcon(self)
            if not self.tray_icon.isSystemTrayAvailable():
                logger.error("System tray is not available")
                return
                
            logger.debug("Creating menu...")
            menu = QMenu()

            # Add status indicator
            self.status_action = QAction("Status: Idle", self)
            self.status_action.setEnabled(False)
            menu.addAction(self.status_action)

            # Add recording actions
            self.toggle_action = QAction("Start Recording (Cmd+Shift+R)", self)
            self.toggle_action.triggered.connect(self.toggle_recording)
            menu.addAction(self.toggle_action)

< HEAD:whisper_notes.py
        self.toggle_action = QAction("Start Recording (Cmd+Shift+R)", self)
        self.toggle_action.triggered.connect(self.toggle_recording)
        menu.addAction(self.toggle_action)
        
        self.journal_action = QAction("Start Journal Entry (Cmd+Shift+J)", self)
        self.journal_action.triggered.connect(self.toggle_journal_mode)
        menu.addAction(self.journal_action)
        
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
=
            journal_action = QAction("Start Journal Entry (Cmd+Shift+J)", self)
            journal_action.triggered.connect(self.toggle_journal_mode)
            menu.addAction(journal_action)
> 2819f8b (feat: Add task for batch audio import feature):archive/whisper_notes.py.bak

            view_journal_action = QAction("View Journal", self)
            view_journal_action.triggered.connect(lambda: self.view_journal())
            menu.addAction(view_journal_action)

            menu.addSeparator()

            # Add settings submenu
            settings_menu = menu.addMenu("Settings")
            
            # Add output file setting
            output_file_action = QAction("Set Output File...", self)
            output_file_action.triggered.connect(self.prompt_set_output_file)
            settings_menu.addAction(output_file_action)
            
            # Add journal directory setting
            journal_dir_action = QAction("Set Journal Directory...", self)
            journal_dir_action.triggered.connect(self.prompt_set_journal_dir)
            settings_menu.addAction(journal_dir_action)
            
            # Add edit summary prompt action
            edit_prompt_action = QAction("Edit Summary Prompt...", self)
            edit_prompt_action.triggered.connect(self.prompt_edit_summary_prompt)
            settings_menu.addAction(edit_prompt_action)

            menu.addSeparator()

            # Add quit action
            quit_action = QAction("Quit WhisperNotes (Cmd+Q)", self)
            quit_action.triggered.connect(self.quit)
            menu.addAction(quit_action)

            # Set the menu and show the icon
            self.tray_icon.setContextMenu(menu)
            
            # Set initial icon
            self.update_icon(False)
            
            # Show the tray icon
            if not self.tray_icon.isVisible():
                logger.debug("Showing tray icon...")
                self.tray_icon.show()
                
                # Add a small delay and check again
                QTimer.singleShot(1000, self.check_tray_visibility)
                
            logger.info("System tray setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up system tray: {e}", exc_info=True)
            
    def check_tray_visibility(self):
        """Check if the tray icon is visible and show an error if not."""
        if not hasattr(self, 'tray_icon') or not self.tray_icon.isVisible():
            logger.error("Failed to show system tray icon")
            # Show a message box if running in GUI mode
            if QApplication.instance().topLevelWindows():
                QMessageBox.critical(
                    None, 
                    "Error", 
                    "Failed to show system tray icon. Please check your system tray settings."
                )

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


        
 2819f8b (feat: Add task for batch audio import feature)
    def handle_recording_finished(self, audio_data):
        """Handle the recorded audio data."""
        if audio_data is None or len(audio_data) == 0:
            logging.warning("No audio data to process.")
 HEAD
            self.update_icon(False)

            self.tray_manager.update_icon(False)
 2819f8b (feat: Add task for batch audio import feature)
            return
            
        # Store audio data for journaling if needed
        self.audio_data = audio_data
        logging.info(f"Recording finished, audio data shape: {audio_data.shape if hasattr(audio_data, 'shape') else 'unknown'}")
        
        # Update UI to show we're no longer recording
 HEAD
        self.update_icon(False)
        
        # Show transcription in progress notification
        mode_text = "journal entry" if self.journaling_mode else "transcription"
        self.tray_icon.showMessage(

        self.tray_manager.update_icon(False)
        
        # Show transcription in progress notification
        mode_text = "journal entry" if self.journaling_mode else "transcription"
        self.tray_manager.tray_icon.showMessage(
 2819f8b (feat: Add task for batch audio import feature)
            "Voice Typer", 
            f"Transcribing audio for {mode_text}...", 
            QSystemTrayIcon.MessageIcon.Information, 
            2000
        )

        # Ensure previous transcriber and thread are cleaned up if they somehow exist
        if self.transcriber is not None or self.transcription_thread is not None:
            logging.warning("[WhisperNotes] Previous transcriber/thread not None, attempting to clean up.")
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
        self.transcription_thread.started.connect(lambda: logging.info("[WhisperNotes] transcription_thread 'started' SIGNAL EMITTED."))
        
        self.transcriber.finished.connect(self.transcription_thread.quit)
        self.transcriber.finished.connect(self.transcriber.deleteLater)
        self.transcriber.finished.connect(self._clear_transcriber_references)
        
        self.transcription_thread.finished.connect(self.transcription_thread.deleteLater)
        self.transcription_thread.finished.connect(self._clear_transcription_thread_references)

        self.transcriber.transcription_ready.connect(self.handle_transcription)
        self.transcriber.error.connect(self.handle_error)
        
        logging.info("[WhisperNotes] About to call self.transcription_thread.start()...")
        self.transcription_thread.start()
        logging.info("[WhisperNotes] Call to self.transcription_thread.start() returned.")
 HEAD

        
    def _clear_transcriber_references(self):
        """Clear references to the transcriber worker."""
        logging.debug("Clearing transcriber references")
        self.transcriber = None
        
    def _clear_transcription_thread_references(self):
        """Clear references to the transcription thread."""
        logging.debug("Clearing transcription thread references")
        self.transcription_thread = None
 2819f8b (feat: Add task for batch audio import feature)

    def prompt_set_output_file(self):
        """Prompts the user to select a Markdown file for saving transcriptions."""
        current_path = self.settings.value("output_file")
        # Suggest a filename and directory based on current settings or defaults
        suggested_filename = os.path.basename(current_path) if current_path and not os.path.isdir(current_path) else "WhisperNotesTranscriptions.md"
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
 HEAD
            self.tray_icon.showMessage("Settings Updated", f"Transcription output file set to:\n{file_path}", QSystemTrayIcon.MessageIcon.Information, 3000)
            
    def prompt_set_journal_dir(self):
        """Prompts the user to select a directory for saving journal entries."""

            self.tray_manager.tray_icon.showMessage("Settings Updated", f"Transcription output file set to:\n{file_path}", QSystemTrayIcon.MessageIcon.Information, 3000)
            
    def import_audio_files(self):
        """Handle importing and transcribing multiple audio files."""
        try:
            # Get the directory containing the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Open directory selection dialog
            dir_path = QFileDialog.getExistingDirectory(
                None,
                "Select Directory with Audio Files",
                script_dir,  # Default to script directory
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            
            if not dir_path:  # User cancelled
                return
                
            # Get list of supported audio files
            supported_extensions = ('.wav', '.mp3', '.m4a', '.amr')
            audio_files = []
            
            # Scan directory for audio files
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.lower().endswith(supported_extensions):
                        audio_files.append(os.path.join(root, file))
            
            if not audio_files:
                self.show_warning_dialog.emit(
                    "No Audio Files Found",
                    f"No supported audio files (.wav, .mp3, .m4a, .amr) found in:\n{dir_path}"
                )
                return
                
            # Sort files by name (which often includes timestamp)
            audio_files.sort()
            
            # Ask for confirmation
            reply = QMessageBox.question(
                None,
                "Import Audio Files",
                f"Found {len(audio_files)} audio files. Process them now?\n\n"
                f"First file: {os.path.basename(audio_files[0])}\n"
                f"Last file: {os.path.basename(audio_files[-1])}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self._process_audio_files(audio_files)
                
        except Exception as e:
            error_msg = handle_error(e, "importing audio files")
            self.show_error_dialog.emit("Import Error", error_msg)
            logging.error(f"Error in import_audio_files: {str(e)}", exc_info=True)
    
    def _format_file_size(self, size_bytes):
        """Format file size in a human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _format_duration(self, seconds):
        """Format duration in a human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        seconds = seconds % 60
        if minutes < 60:
            return f"{minutes}m {seconds:.0f}s"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m {seconds:.0f}s"

    def _load_audio_as_array(self, file_path):
        """
        Load an audio file as a numpy array and return (audio_data, sample_rate).
        If the format is not supported by soundfile, auto-convert to WAV using pydub/ffmpeg.
        """
        import soundfile as sf
        import os, tempfile
        try:
            audio_data, sample_rate = sf.read(file_path)
            return audio_data, sample_rate
        except Exception as e:
            # Try conversion for unsupported formats
            try:
                from pydub import AudioSegment
            except ImportError:
                raise RuntimeError(f"Audio format not supported and pydub not installed: {e}")
            try:
                audio = AudioSegment.from_file(file_path)
            except Exception as conv_e:
                raise RuntimeError(f"Format not recognised and conversion failed: {conv_e}")
            import shutil
            debug_dir = os.path.join(os.path.dirname(__file__), 'converted_audio_debug')
            os.makedirs(debug_dir, exist_ok=True)
            debug_wav_path = os.path.join(debug_dir, os.path.splitext(os.path.basename(file_path))[0] + '.wav')
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                audio.export(tmp_wav.name, format='wav')
                tmp_wav_path = tmp_wav.name
            # Save a copy for inspection
            shutil.copy(tmp_wav_path, debug_wav_path)
            print(f"[DEBUG] Converted audio saved to: {debug_wav_path}")
            try:
                audio_data, sample_rate = sf.read(tmp_wav_path)
            finally:
                os.remove(tmp_wav_path)
            return audio_data, sample_rate

    def _transcribe_audio_file_blocking(self, audio_file_path: str) -> str:
        """
        Transcribe an audio file synchronously using TranscriptionWorker in a local thread.
        Returns the transcript as a string.
        """
        import threading
        from transcription import TranscriptionWorker
        from PySide6.QtCore import QEventLoop
        transcript_result = {'text': None, 'error': None}
        loop = QEventLoop()
        def on_ready(text):
            transcript_result['text'] = text
            loop.quit()
        def on_error(msg):
            transcript_result['error'] = msg
            loop.quit()
        audio_data, _ = self._load_audio_as_array(audio_file_path)
        worker = TranscriptionWorker(model_name="base", audio_data=audio_data)
        worker.transcription_ready.connect(on_ready)
        worker.error.connect(on_error)
        worker.finished.connect(loop.quit)
        thread = threading.Thread(target=worker.run)
        thread.start()
        loop.exec()
        thread.join()
        if transcript_result['error']:
            raise Exception(transcript_result['error'])
        return transcript_result['text']

    def _process_audio_files(self, audio_files):
        """Process a list of audio files for transcription with detailed progress dialog."""
        import os, time
        from PySide6.QtWidgets import (QDialog, QLabel, QVBoxLayout, QHBoxLayout, QProgressBar, QDialogButtonBox, QTextEdit, QGroupBox)
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QFont, QPixmap, QColor, QTextCharFormat, QTextCursor
        try:
            from pydub.utils import mediainfo
        except ImportError:
            mediainfo = None

        # Helper for file icons
        def get_icon_for_ext(ext):
            icon_map = {
                '.wav': '🎵', '.mp3': '🎶', '.m4a': '🎤', '.amr': '📞'
            }
            return icon_map.get(ext.lower(), '📁')

        def get_audio_duration(filepath):
            if mediainfo:
                try:
                    info = mediainfo(filepath)
                    return float(info.get('duration', 0))
                except Exception:
                    return None
            return None

        # --- Dialog Layout ---
        dialog = QDialog()
        dialog.setWindowTitle("Importing Audio Files")
        dialog.setMinimumWidth(600)
        main_layout = QVBoxLayout(dialog)

        # Progress Bars
        pb_group = QGroupBox("Progress")
        pb_layout = QVBoxLayout()
        overall_pb = QProgressBar()
        overall_pb.setMaximum(len(audio_files))
        overall_pb.setValue(0)
        overall_pb.setTextVisible(True)
        overall_pb.setFormat("%v/%m files (%p%)")
        file_pb = QProgressBar()
        file_pb.setMaximum(100)
        file_pb.setValue(0)
        file_pb.setTextVisible(True)
        file_pb.setFormat("Current file: %p%")
        pb_layout.addWidget(overall_pb)
        pb_layout.addWidget(file_pb)
        pb_group.setLayout(pb_layout)
        main_layout.addWidget(pb_group)

        # File Info Section
        info_group = QGroupBox("Current File Info")
        info_layout = QHBoxLayout()
        icon_label = QLabel()
        icon_label.setFont(QFont("Arial", 32))
        info_layout.addWidget(icon_label)
        fileinfo_v = QVBoxLayout()
        file_label = QLabel()
        file_label.setFont(QFont("Arial", 12, QFont.Bold))
        fileinfo_v.addWidget(file_label)
        details_label = QLabel()
        details_label.setFont(QFont("Arial", 10))
        fileinfo_v.addWidget(details_label)
        status_label = QLabel()
        status_label.setFont(QFont("Arial", 10))
        fileinfo_v.addWidget(status_label)
        fileinfo_v.addStretch()
        info_layout.addLayout(fileinfo_v)
        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)

        # Transcription Preview
        preview_group = QGroupBox("Transcription Preview")
        preview_layout = QVBoxLayout()
        preview_text = QTextEdit()
        preview_text.setReadOnly(True)
        preview_text.setMaximumHeight(80)
        preview_layout.addWidget(preview_text)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        # Log Panel
        log_group = QGroupBox("Import Log")
        log_layout = QVBoxLayout()
        log_text = QTextEdit()
        log_text.setReadOnly(True)
        log_layout.addWidget(log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        # Time/Rate
        time_label = QLabel()
        main_layout.addWidget(time_label)

        # Cancel Button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        button_box.rejected.connect(dialog.reject)
        main_layout.addWidget(button_box)

        dialog.setLayout(main_layout)
        dialog.show()

        # --- Logging Helper ---
        def log_message(text, color=None):
            cursor = log_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            fmt = QTextCharFormat()
            if color:
                fmt.setForeground(QColor(color))
            cursor.insertText(text + "\n", fmt)
            log_text.setTextCursor(cursor)
            log_text.ensureCursorVisible()

        # --- Processing Loop ---
        total_size = sum(os.path.getsize(f) for f in audio_files)
        processed_size = 0
        start_time = time.time()
        success_count = 0
        failed_files = []

        for idx, audio_file in enumerate(audio_files, 1):
            if not dialog.isVisible():
                log_message("Import cancelled by user.", "orange")
                break
            file_name = os.path.basename(audio_file)
            ext = os.path.splitext(file_name)[1]
            icon_label.setText(get_icon_for_ext(ext))
            file_label.setText(f"{file_name}")
            file_size = os.path.getsize(audio_file)
            duration = get_audio_duration(audio_file)
            size_str = self._format_file_size(file_size)
            dur_str = f"{duration:.1f}s" if duration else "Unknown"
            details_label.setText(f"<b>Type:</b> {ext.upper()} | <b>Size:</b> {size_str} | <b>Duration:</b> {dur_str}")
            preview_text.setPlainText("")
            status_label.setText("<span style='color:blue'>Queued...</span>")
            QApplication.processEvents()

            try:
                status_label.setText("<span style='color:blue'>Transcribing...</span>")
                t0 = time.time()
                # --- Actual transcription ---
                transcript = None
                try:
                    transcript = self._transcribe_audio_file_blocking(audio_file)
                except Exception as e:
                    raise Exception(f"Transcription failed: {e}")
                elapsed = time.time() - t0
                preview_text.setPlainText(transcript or "[No transcript generated]")
                file_pb.setValue(100)
                time_label.setText(f"Elapsed: {self._format_duration(elapsed)} | Rate: {self._format_file_size(file_size/(elapsed+0.01))}/s")
                QApplication.processEvents()
                if not transcript or not transcript.strip():
                    raise Exception("Empty transcript")
                # --- Copy audio file to main recordings storage ---
                import shutil, os
                from datetime import datetime
                home_dir = os.path.expanduser("~")
                recordings_dir = os.path.join(home_dir, "Documents", "Personal", "Audio Journal", "recordings")
                os.makedirs(recordings_dir, exist_ok=True)
                # Use unique filename to avoid collisions
                base_name = os.path.basename(audio_file)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_name = f"imported_{timestamp}_{base_name}"
                dest_path = os.path.join(recordings_dir, dest_name)
                try:
                    shutil.copy2(audio_file, dest_path)
                    log_message(f"Audio copied to {dest_path}", "gray")
                except Exception as e:
                    log_message(f"ERROR copying audio: {e}", "orange")
                    raise Exception(f"Audio copy failed: {e}")

                # --- Journal entry (main workflow) ---
                status_label.setText("<span style='color:blue'>Saving to Journal...</span>")
                QApplication.processEvents()
                try:
                    entry = self.handle_journal_entry(transcript)
                except Exception as e:
                    raise Exception(f"Journal save failed: {e}")
                status_label.setText("<span style='color:green'>Success</span>")
                log_message(f"SUCCESS: {file_name}", "green")
                success_count += 1
            except Exception as e:
                import logging
                error_msg = f"FAILED: {file_name} (Transcription failed: {e})"
                print(error_msg)
                logging.error(error_msg)
                status_label.setText(f"<span style='color:red'>Error: {e}</span>")
                log_message(f"FAILED: {file_name} ({e})", "red")
                failed_files.append((file_name, str(e)))
                QApplication.processEvents()
                time.sleep(0.5)
            overall_pb.setValue(idx)
            processed_size += file_size
            QApplication.processEvents()

        # Final status
        elapsed = time.time() - start_time
        status = f"Import complete: <span style='color:green'>{success_count} succeeded</span>, <span style='color:red'>{len(failed_files)} failed</span> in {self._format_duration(elapsed)}."
        log_message(status, "blue")
        status_label.setText(status)
        time_label.setText(f"Total elapsed: {self._format_duration(elapsed)} | Total size: {self._format_file_size(processed_size)}")
        QApplication.processEvents()
        time.sleep(1)
        QTimer.singleShot(4000, dialog.accept)

        return success_count, failed_files

        
        # Create a progress dialog
        progress_dialog = QProgressDialog(
            "Processing audio files...",
            "Cancel",
            0,
            len(audio_files),
            None
        )
        progress_dialog.setWindowTitle("Importing Audio Files")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(False)
        
        # Create a custom widget for detailed progress
        progress_widget = QDialog()
        progress_widget.setWindowTitle("Import Progress")
        progress_widget.setMinimumWidth(500)
        layout = QVBoxLayout(progress_widget)
        
        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        # Overall progress
        overall_progress = QProgressBar()
        overall_progress.setMaximum(len(audio_files))
        overall_progress.setTextVisible(True)
        overall_progress.setFormat("%v/%m files (%p%)")
        progress_layout.addWidget(QLabel("<b>Overall Progress:</b>"))
        progress_layout.addWidget(overall_progress)
        
        # Current file progress
        file_progress = QProgressBar()
        file_progress.setMaximum(100)
        file_progress.setTextVisible(True)
        file_progress.setFormat("%p%")
        progress_layout.addWidget(QLabel("<b>Current File Progress:</b>"))
        progress_layout.addWidget(file_progress)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Details group
        details_group = QGroupBox("File Details")
        details_layout = QVBoxLayout()
        
        # File info
        file_info = QLabel("<b>Current File:</b> Waiting to start...")
        file_info.setWordWrap(True)
        details_layout.addWidget(file_info)
        
        # Status info
        status_info = QLabel("<b>Status:</b> Idle")
        details_layout.addWidget(status_info)
        
        # File details
        details_text = QLabel()
        details_text.setWordWrap(True)
        details_layout.addWidget(details_text)
        
        # Time info
        time_info = QLabel()
        details_layout.addWidget(time_info)
        
        # Error info (hidden by default)
        error_info = QLabel()
        error_info.setWordWrap(True)
        error_info.setStyleSheet("color: red;")
        error_info.hide()
        details_layout.addWidget(error_info)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        # Add cancel button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        button_box.rejected.connect(progress_dialog.cancel)
        layout.addWidget(button_box)
        
        progress_widget.setLayout(layout)
        progress_widget.show()
        
        try:
            processed_count = 0
            success_count = 0
            failed_files = []
            
            # Show initial notification
            self.tray_manager.show_notification(
                "Import Started",
                f"Starting import of {len(audio_files)} audio files...",
                QSystemTrayIcon.Information
            )
            
            # Process each file sequentially
            for i, audio_file in enumerate(audio_files, 1):
                if progress_dialog.wasCanceled():
                    status_label.setText("<b>Import cancelled by user</b>")
                    logging.info(f"Import cancelled by user after processing {i-1} files")
                    break
                    
                # Update progress UI
                file_name = os.path.basename(audio_file)
                file_label.setText(f"<b>Processing file {i} of {len(audio_files)}:</b> {file_name}")
                status_label.setText("Converting audio format..." if audio_file.lower().endswith(('.amr', '.m4a')) 
                                   else "Transcribing audio...")
                progress_bar.setValue(i)
                progress_dialog.setValue(i)
                
                # Force UI update
                QApplication.processEvents()
                
                try:
                    start_time = time.time()
                    logging.info(f"Processing audio file ({i}/{len(audio_files)}): {file_name}")
                    
                    # TODO: Add actual transcription logic here
                    # For now, simulate processing time based on file size
                    file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)  # Size in MB
                    process_time = min(5, max(1, file_size_mb * 0.5))  # 0.5s per MB, max 5s
                    
                    # Simulate processing with progress updates
                    steps = int(process_time * 10)  # Update every 100ms
                    for step in range(steps):
                        if progress_dialog.wasCanceled():
                            raise Exception("Processing cancelled by user")
                        progress = int((step / steps) * 100)
                        status_label.setText(f"Processing... ({progress}%)")
                        QApplication.processEvents()
                        time.sleep(0.1)
                    
                    # TODO: Replace with actual transcription
                    # transcription = self.transcribe_audio_file(audio_file)
                    # self.save_transcription(transcription, audio_file)
                    
                    success_count += 1
                    processed_count += 1
                    elapsed = time.time() - start_time
                    logging.info(f"Successfully processed {file_name} in {elapsed:.1f} seconds")
                    
                except Exception as e:
                    error_msg = str(e)
                    logging.error(f"Error processing {file_name}: {error_msg}", exc_info=True)
                    failed_files.append((file_name, error_msg))
                    status_label.setText(f"<font color='red'>Error processing {file_name}</font>")
                    QApplication.processEvents()
                    time.sleep(1)  # Pause to show error
            
            # Show completion status
            if processed_count == 0:
                status_text = "No files were processed."
            else:
                status_text = (
                    f"<b>Import complete!</b><br>"
                    f"• Successfully processed: <b>{success_count}</b> file{'' if success_count == 1 else 's'}<br>"
                )
                if failed_files:
                    status_text += (
                        f"• <font color='red'>Failed: {len(failed_files)} file{'' if len(failed_files) == 1 else 's'}</font>"
                    )
            
            status_label.setText(status_text)
            
            # Show completion notification
            self.tray_manager.show_notification(
                "Import Complete" if not progress_dialog.wasCanceled() else "Import Cancelled",
                f"Processed {success_count} of {len(audio_files)} files" + 
                (f"\nFailed: {len(failed_files)}" if failed_files else ""),
                QSystemTrayIcon.Information if success_count > 0 else QSystemTrayIcon.Warning
            )
            
            # Log detailed results
            logging.info(f"Import completed. Success: {success_count}, Failed: {len(failed_files)}")
            for file_name, error in failed_files:
                logging.error(f"Failed to process {file_name}: {error}")
            
            # Keep the dialog open for 5 seconds after completion
            if not progress_dialog.wasCanceled():
                QTimer.singleShot(5000, progress_widget.accept)
            
        except Exception as e:
            error_msg = handle_error(e, "processing audio files")
            logging.error(f"Error in _process_audio_files: {error_msg}", exc_info=True)
            self.show_error_dialog.emit("Processing Error", error_msg)
            progress_widget.close()
        
        return success_count, failed_files
    
    def prompt_set_journal_dir(self):
        """Prompts the user to select a directory for saving journal entries."""
        # This method is only called from the tray (main thread), so safe to show dialog here
 2819f8b (feat: Add task for batch audio import feature)
        # Get current journal directory from settings or use default
        current_dir = self.settings.value("journal_dir")
        if not current_dir:
            # Default to ~/Documents/Personal/Audio Journal/
            home_dir = os.path.expanduser("~")
            current_dir = os.path.join(home_dir, "Documents", "Personal", "Audio Journal")
 HEAD
            

        
 2819f8b (feat: Add task for batch audio import feature)
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
 HEAD
            self.tray_icon.showMessage(

            self.tray_manager.tray_icon.showMessage(
 2819f8b (feat: Add task for batch audio import feature)
                "Settings Updated", 
                f"Journal directory set to:\n{journal_dir}", 
                QSystemTrayIcon.Information, 
                3000
            )
            
    def prompt_edit_summary_prompt(self):
        """Opens a dialog to edit the summary prompt template used for journal entries."""
 HEAD

        # This method is only called from the tray (main thread), so safe to show dialog here
 2819f8b (feat: Add task for batch audio import feature)
        # Get current summary prompt from settings or use default
        default_prompt = """Analyze the following journal entry and provide a clear, structured summary. Focus on identifying the main themes, emotional tone, and key points. Maintain a professional yet empathetic tone in your analysis.

Key elements to include:
1. Main Themes: 2-3 primary topics or recurring subjects
2. Emotional Tone: Note any strong emotions or mood shifts
3. Key Points: 3-5 significant points or insights
4. Action Items: Any mentioned tasks, decisions, or follow-ups
5. Notable Details: Important names, dates, or specific references

If any sections are unclear due to audio quality, note: "[Section unclear due to audio quality]" and continue with the analysis. If the entry is largely inaudible, state: "The audio quality is too poor for meaningful analysis."

Journal Entry:
{transcription}"""
        current_prompt = self.settings.value("summary_prompt", default_prompt)
        
        # Create dialog
        dialog = QDialog(None)
        dialog.setWindowTitle("Edit Summary Prompt")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(300)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Add label
        label = QLabel("Edit the summary prompt. Use {transcription} as a placeholder for the transcription text:")
        layout.addWidget(label)
        
        # Add text edit
        text_edit = QTextEdit()
        text_edit.setPlainText(current_prompt)
        text_edit.setAcceptRichText(False)
        layout.addWidget(text_edit)
        
        # Add button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog and get result
        if dialog.exec() == QDialog.Accepted:
            new_prompt = text_edit.toPlainText().strip()
            if new_prompt and new_prompt != current_prompt:
                # Save to settings
                self.settings.setValue("summary_prompt", new_prompt)
                logging.info("Summary prompt updated")
                
                # Update the journal manager with the new prompt if it exists
                if hasattr(self, 'journal_manager') and self.journal_manager:
                    self.journal_manager.set_summary_prompt(new_prompt)
                
                # Show confirmation to user
 HEAD
                self.tray_icon.showMessage(

                self.tray_manager.tray_icon.showMessage(
 2819f8b (feat: Add task for batch audio import feature)
                    "Settings Updated", 
                    "Summary prompt has been updated.", 
                    QSystemTrayIcon.Information, 
                    3000
                )
 HEAD
                
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

    
    def handle_transcription(self, text):
        # If called from a worker thread, use signals for dialogs
 2819f8b (feat: Add task for batch audio import feature)
        """
        Handle the transcribed text.
        
        Args:
            text: The transcribed text to process and save
            
        Returns:
            bool: True if transcription was handled successfully, False otherwise
        """
        if not text or not text.strip():
            logging.warning("Empty transcription received")
 HEAD
            self.tray_icon.showMessage(
                "Empty Transcription",
                "No speech was detected. Please try again.",
                QSystemTrayIcon.MessageIcon.Warning,
                3000
            )

            self.show_warning_dialog.emit("Empty Transcription", "No speech was detected. Please try again.")
 2819f8b (feat: Add task for batch audio import feature)
            return False

        try:
            logging.info(f"Transcription received (length: {len(text)} chars)")
            
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
            
            # Format the entry with timestamp
            entry = f"{timestamp} - {text}\n\n"
            
            # Save to the configured output file
            output_file = self.settings.value("output_file")
            if not output_file:
                raise ConfigurationError("No output file configured")
            
            # Ensure the directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except OSError as e:
                    raise FileSystemError(f"Could not create directory {output_dir}: {str(e)}")
            
            # Write to the file atomically
            temp_file = f"{output_file}.tmp"
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(entry)
                
                # On POSIX systems, this is atomic
                if os.path.exists(output_file):
                    os.remove(output_file)
                os.rename(temp_file, output_file)
                
                logging.info(f"Appended transcription to {output_file}")
                
                # Try to auto-paste the transcription at cursor position
                paste_success = self.paste_at_cursor(text)
                
                # Fallback to clipboard if auto-paste fails or isn't enabled
                if not paste_success:
                    try:
                        clipboard = QApplication.clipboard()
                        if clipboard:
                            clipboard.setText(text)
                            logging.info("Transcription copied to clipboard")
                            
                            # Show notification that text was copied to clipboard
 HEAD
                            self.tray_icon.showMessage(

                            self.tray_manager.tray_icon.showMessage(
 2819f8b (feat: Add task for batch audio import feature)
                                "Transcription Complete",
                                "Your transcription has been saved and copied to clipboard.",
                                QSystemTrayIcon.MessageIcon.Information,
                                3000
                            )
                    except Exception as e:
                        logging.warning(f"Failed to copy to clipboard: {e}")
                        
                        # Show error notification
 HEAD
                        self.tray_icon.showMessage(
                            "Transcription Saved",
                            "Transcription completed but could not paste or copy to clipboard.",
                            QSystemTrayIcon.MessageIcon.Warning,
                            3000
                        )
                else:
                    # Show success notification for auto-paste
                    self.tray_icon.showMessage(

                        self.show_error_dialog.emit("Transcription Saved", "Transcription completed but could not paste or copy to clipboard.")
                else:
                    # Show success notification for auto-paste
                    self.tray_manager.tray_icon.showMessage(
 2819f8b (feat: Add task for batch audio import feature)
                        "Transcription Complete",
                        "Your transcription has been pasted at cursor position.",
                        QSystemTrayIcon.MessageIcon.Information,
                        3000
                    )
                
                # If we're in journaling mode, create a journal entry
                if hasattr(self, 'journaling_mode') and self.journaling_mode:
                    logging.info(f"[Journal] Passing audio_data to handle_journal_entry: {'set' if hasattr(self, 'audio_data') and self.audio_data is not None else 'not set'}")
                    self.handle_journal_entry(text, audio_data=getattr(self, 'audio_data', None), sample_rate=getattr(self, 'sample_rate', 16000))
                    self._clear_audio_data()
                
                return True
            
            except (IOError, OSError) as e:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                raise FileSystemError(f"Failed to write to {output_file}: {str(e)}")
            
        except Exception as e:
            if not isinstance(e, (FileSystemError, ConfigurationError)):
                e = FileSystemError(f"Failed to save transcription: {str(e)}")
            
            # Use our custom error handling
            error_msg = handle_error(e, "transcription handling")
            
            # Show error to user
 HEAD
            try:
                self.tray_icon.showMessage(
                    "Error",
                    error_msg,
                    QSystemTrayIcon.Critical,
                    5000
                )
            except Exception as ui_error:
                logging.error(f"Failed to show error message: {ui_error}")

            self.show_error_dialog.emit("Error", error_msg)
 2819f8b (feat: Add task for batch audio import feature)
            
            # Log the error
            logging.error(f"Error in handle_transcription: {str(e)}", exc_info=True)
            return False


    def handle_journal_entry(self, text, audio_data=None, sample_rate=16000):
        """
        Handle creating a journal entry with the transcribed text.
        
        Args:
            text: The transcribed text to include in the journal entry
            
        Returns:
            dict: The created journal entry if successful
            
        Raises:
            JournalingError: If there's an error creating the journal entry
            ConfigurationError: If the journal manager is not properly initialized
            ValueError: If no text is provided for the journal entry
        """
        try:
            if not hasattr(self, 'journal_manager') or not self.journal_manager:
                raise ConfigurationError("Journal manager is not properly initialized")
            
            if not text or not text.strip():
                raise ValueError("No text provided for journal entry")
                
            logging.info("Creating journal entry...")
            
            try:
 HEAD
                # Create a journal entry with the transcribed text
                entry = self.journal_manager.create_journal_entry(text)
                
                if not entry or 'error' in entry:
                    error_msg = entry.get('error', 'Unknown error') if isinstance(entry, dict) else 'Unknown error'
                    raise JournalingError(f"Failed to create journal entry: {error_msg}")

                # Get the active template name if any
                active_template = self.journal_manager.active_template
                template_name = active_template if active_template else None
                
                # Create a journal entry with the transcribed text and template
                # Pass audio_data and sample_rate if available
                if audio_data is not None:
                    logging.info(f"[Journal] Passing audio_data to create_journal_entry (type: {type(audio_data)})")
                    entry_path = self.journal_manager.create_journal_entry(text, audio_data=audio_data, sample_rate=sample_rate, template_name=template_name)
                else:
                    entry_path = self.journal_manager.create_journal_entry(text, template_name=template_name)
                
                if not entry_path:
                    raise JournalingError("Failed to create journal entry: No entry path returned")
 2819f8b (feat: Add task for batch audio import feature)
                
                # Reset journaling mode
                self.journaling_mode = False
                self._clear_audio_data()
                
 HEAD
                # Update the tray icon to show we're back to normal mode
                self.update_icon(False)
                
                logging.info(f"Journal entry created: {entry.get('id', 'unknown')}")
                
                # Show a notification
                self.tray_icon.showMessage(
                    "Journal Entry Created",
                    "Your journal entry has been saved.",

                # Reset the active template and custom settings
    
                
                # Update the tray icon to show we're back to normal mode
                self.tray_manager.update_icon(False)
                
                logging.info(f"Journal entry created: {os.path.basename(entry_path)}")
                
                # Show a notification with template info if applicable
                notification_title = "Journal Entry Created"
                notification_message = "Your journal entry has been saved."
                
                if template_name:
                    notification_message = f"Your journal entry has been saved using the '{template_name}' template."
                
                self.tray_manager.tray_icon.showMessage(
                    notification_title,
                    notification_message,
 2819f8b (feat: Add task for batch audio import feature)
                    QSystemTrayIcon.MessageIcon.Information,
                    3000
                )
                
                # Reset UI elements
                if hasattr(self, 'toggle_action'):
                    self.toggle_action.setText("Start Recording")
 HEAD
                self.update_icon(False)
                self.tray_icon.setToolTip("WhisperNotes - Ready")
                
                return entry

                self.tray_manager.update_icon(False)
                self.tray_manager.tray_icon.setToolTip("WhisperNotes - Ready")
                
                return {"entry_path": entry_path, "template": template_name}
 2819f8b (feat: Add task for batch audio import feature)
                
            except Exception as e:
                if not isinstance(e, JournalingError):
                    raise JournalingError(f"Failed to create journal entry: {str(e)}")
                raise
                
        except Exception as e:
            # Use our custom error handling
            error_msg = handle_error(e, "journal entry creation")
            
            # Show error to user
 HEAD
            try:
                self.tray_icon.showMessage(
                    "Journal Error",
                    error_msg,
                    QSystemTrayIcon.MessageIcon.Critical,
                    5000
                )
            except Exception as ui_error:
                logging.error(f"Failed to show error message: {ui_error}")

            self.show_error_dialog.emit("Journal Error", error_msg)
 2819f8b (feat: Add task for batch audio import feature)
            
            # Log the error
            logging.error(f"Error in handle_journal_entry: {str(e)}", exc_info=True)
            raise
    
    def _clear_audio_data(self):
        """Clear stored audio data and sample rate to free memory."""
        self.audio_data = None
        self.sample_rate = 16000

    def stop_recording(self):
        """Stop the recording thread."""
        if not self.is_recording:
            logging.info("Not recording, nothing to stop")
            return
            
        logging.info("Stopping recording...")
        self.is_recording = False
        
 HEAD
        # Update UI immediately on the main thread
        self.toggle_action.setText("Start Recording")
        self.update_icon(False)
        self.tray_icon.setToolTip("Voice Typer (Ready)")

        # Update tray icon and tooltip only
        self.tray_manager.update_icon(False)
        self.tray_manager.tray_icon.setToolTip("Voice Typer (Ready)")
 2819f8b (feat: Add task for batch audio import feature)
        
        # Stop recording thread if it exists
        if self.recording_thread and self.recording_thread.isRunning():
            logging.info("Sending stop signal to recording thread")
            self.recording_thread.stop()
    
    def handle_error(self, error_msg):
        """Handle errors from the worker thread."""
        logging.error(f"Error: {error_msg}")
 HEAD
        self.tray_icon.showMessage("Error", error_msg, QSystemTrayIcon.MessageIcon.Critical)
        self.stop_recording()
    
    def _clear_transcriber_references(self):
        """
        Clean up and remove references to the transcriber worker.
        This is called when the worker has finished and is being deleted.
        """
        try:
            if not hasattr(self, 'transcriber') or self.transcriber is None:
                return
                
            logging.info("Cleaning up transcriber worker references")
            
            # Store reference and clear immediately to prevent reentrancy
            transcriber = self.transcriber
            self.transcriber = None
            
            # If the transcriber is already deleted, just return
            try:
                if isinstance(transcriber, QObject) and not transcriber.isWidgetType():
                    try:
                        # Safely disconnect signals if they exist
                        signal_names = ['error', 'finished', 'transcription_ready']
                        for signal_name in signal_names:
                            try:
                                if hasattr(transcriber, signal_name):
                                    signal = getattr(transcriber, signal_name, None)
                                    if signal is not None and hasattr(signal, 'disconnect'):
                                        try:
                                            signal.disconnect()
                                        except (RuntimeError, TypeError) as e:
                                            # Signal might be already disconnected or invalid
                                            logging.debug(f"Error disconnecting {signal_name} signal: {e}")
                            except Exception as e:
                                logging.warning(f"Error checking {signal_name} signal: {e}")
                        
                        # Request stop if the transcriber is still running
                        if hasattr(transcriber, 'request_stop'):
                            try:
                                transcriber.request_stop()
                            except Exception as e:
                                logging.warning(f"Error requesting stop on transcriber: {e}")
                        
                        # Schedule the transcriber for deletion if it's a QObject
                        if isinstance(transcriber, QObject):
                            transcriber.deleteLater()
                        
                        logging.debug("Transcriber worker references cleared")
                        
                    except Exception as e:
                        logging.error(f"Error during transcriber cleanup: {e}", exc_info=True)
                else:
                    logging.debug("Transcriber worker already deleted, skipping cleanup")
                    
            except Exception as e:
                logging.error(f"Error checking if transcriber is deleted: {e}")
                
        except Exception as e:
            logging.error(f"Unexpected error in _clear_transcriber_references: {e}", exc_info=True)

    def _clear_transcription_thread_references(self):
        """
        Clean up and remove references to the transcription thread.
        This is called when the thread has finished and is being deleted.
        """
        try:
            # Early return if no thread exists
            if not hasattr(self, 'transcription_thread') or self.transcription_thread is None:
                return
                
            logging.info("Cleaning up transcription thread references")
            
            # Store reference and clear immediately to prevent reentrancy
            thread = self.transcription_thread
            self.transcription_thread = None
            
            # If the thread is already deleted, just return
            if not isinstance(thread, QObject) or thread.thread() is None:
                logging.debug("Thread already deleted or invalid, skipping cleanup")
                return
            
            try:
                # Safely disconnect signals if they exist
                self._disconnect_thread_signals(thread)
                
                # Request stop if possible
                if hasattr(self, 'transcriber') and self.transcriber is not None:
                    try:
                        # Check if transcriber is valid and has request_stop method
                        if (isinstance(self.transcriber, QObject) and 
                            hasattr(self.transcriber, 'request_stop') and 
                            callable(getattr(self.transcriber, 'request_stop'))):
                            self.transcriber.request_stop()
                    except Exception as e:
                        logging.warning(f"Error requesting stop on transcriber: {e}")
                
                # Ensure the thread is properly terminated
                if thread.isRunning():
                    logging.warning("Transcription thread still running during cleanup")
                    
                    # First try to quit gracefully
                    thread.quit()
                    if not thread.wait(2000):  # Wait up to 2 seconds
                        logging.warning("Thread did not exit gracefully, terminating...")
                        try:
                            thread.terminate()
                            if not thread.wait(1000):  # Give it 1 more second
                                logging.error("Failed to terminate transcription thread")
                        except Exception as e:
                            logging.error(f"Error terminating thread: {e}")
                
                # Schedule the thread for deletion if it's still valid
                if isinstance(thread, QObject) and thread.thread() is not None:
                    try:
                        thread.deleteLater()
                    except Exception as e:
                        logging.error(f"Error scheduling thread for deletion: {e}")
                
                # Clear any remaining references
                if hasattr(self, 'transcriber') and self.transcriber is not None:
                    self.transcriber = None
                
                logging.debug("Transcription thread references cleared")
                
            except Exception as e:
                logging.error(f"Error during transcription thread cleanup: {e}", exc_info=True)
                
        except Exception as e:
            logging.error(f"Unexpected error in _clear_transcription_thread_references: {e}", exc_info=True)
            
    def _disconnect_thread_signals(self, thread):
        """Safely disconnect signals from a thread."""
        if not thread:
            return
            
        signal_names = ['started', 'finished', 'error', 'transcription_ready']
        
        for signal_name in signal_names:
            try:
                if hasattr(thread, signal_name):
                    signal = getattr(thread, signal_name)
                    if signal:
                        try:
                            signal.disconnect()
                        except (RuntimeError, TypeError) as e:
                            # Signal might be already disconnected or invalid
                            logging.debug(f"Error disconnecting {signal_name} signal: {e}")
            except Exception as e:
                logging.warning(f"Error checking {signal_name} signal: {e}")

        self.show_error_dialog.emit("Error", error_msg)
        self.stop_recording()
    
    @Slot(str, str)
    def _show_error_dialog_slot(self, title, msg):
        QMessageBox.critical(None, title, msg)

    @Slot(str, str)
    def _show_info_dialog_slot(self, title, msg):
        QMessageBox.information(None, title, msg)

    @Slot(str, str)
    def _show_warning_dialog_slot(self, title, msg):
        QMessageBox.warning(None, title, msg)

    @Slot()
    def _show_config_dialog_slot(self):
        dialog = self.TemplateConfigDialog(parent=None, template_manager=self.template_manager, settings=self.settings)
        result = dialog.exec()
        
        # If dialog was accepted, update template hotkeys
        if result == QDialog.DialogCode.Accepted:
            self._update_template_hotkeys()
            
    def _load_template_configs(self):
        """
        Load template configurations from settings and register hotkeys.
        """
        if not hasattr(self, 'settings') or not self.settings:
            logging.warning("Settings not initialized, cannot load template configurations")
            return
            
        config_str = self.settings.value("template_configs")
        if config_str:
            try:
                template_configs = json.loads(config_str)
                self.template_manager.load_template_configs(template_configs)
                self._register_template_hotkeys(template_configs)
                logging.info(f"Loaded {len(template_configs)} template configurations from settings")
            except json.JSONDecodeError:
                logging.error("Failed to parse template configurations from settings")
                
    def _update_template_hotkeys(self):
        """
        Update template hotkeys based on current template configurations.
        """
        # Get current template configurations
        template_configs = self.template_manager.template_configs
        
        # Unregister all existing template hotkeys
        self._unregister_all_template_hotkeys()
        
        # Register new template hotkeys
        self._register_template_hotkeys(template_configs)
        
    def _register_template_hotkeys(self, template_configs):
        """
        Register hotkeys for templates based on their configurations.
        
        Args:
            template_configs: Dictionary of template configurations
        """
        for template_name, config in template_configs.items():
            hotkey = config.get("hotkey")
            if hotkey and hotkey.strip():
                # Register the hotkey with the hotkey manager
                success = self.hotkey_manager.register_template_hotkey(
                    hotkey_str=hotkey.strip(),
                    template_name=template_name,
                    callback=self._on_template_hotkey
                )
                if success:
                    logging.info(f"Registered hotkey '{hotkey}' for template '{template_name}'")
                else:
                    logging.warning(f"Failed to register hotkey '{hotkey}' for template '{template_name}'")
                    
    def _unregister_all_template_hotkeys(self):
        """
        Unregister all template hotkeys.
        """
        # Get current template configurations
        template_configs = self.template_manager.template_configs
        
        # Unregister each hotkey
        for template_name, config in template_configs.items():
            hotkey = config.get("hotkey")
            if hotkey and hotkey.strip():
                self.hotkey_manager.unregister_template_hotkey(hotkey.strip())
                
    def _on_template_hotkey(self, template_name):
        """
        Handle template hotkey press.
        
        Args:
            template_name: Name of the template to use
        """
        logging.info(f"Template hotkey pressed for template: {template_name}")
        
        # Check if we're already recording
        if self.is_recording:
            logging.warning("Cannot start template recording while already recording")
            self.tray_manager.show_message(
                "Recording in Progress",
                "Please finish the current recording before using a template.",
                QSystemTrayIcon.MessageIcon.Warning
            )
            return
            
        # Start recording with the selected template
        self.journaling_mode = True
        self._set_active_template(template_name)
        self.toggle_recording()
        
    def _set_active_template(self, template_name):
        """
        Set the active template for the next journal entry.
        
        Args:
            template_name: Name of the template to use
        """
        if not hasattr(self, 'journal_manager') or not self.journal_manager:
            logging.error("Journal manager not initialized, cannot set active template")
            return
            
        # Store the template name in the journal manager
        self.journal_manager.active_template = template_name
        
        # Get template configuration
        template_config = self.template_manager.get_template_config(template_name)
        
        # Set custom save location if specified
        if template_config and "save_location" in template_config and template_config["save_location"]:
            self.journal_manager.custom_output_dir = template_config["save_location"]
        else:
            self.journal_manager.custom_output_dir = None
            
        # Set custom tags if specified
        if template_config and "tags" in template_config:
            self.journal_manager.custom_tags = template_config["tags"]
        else:
            self.journal_manager.custom_tags = None
            
        logging.info(f"Set active template to '{template_name}'")
 2819f8b (feat: Add task for batch audio import feature)

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
    
    # Set application attributes for better macOS integration
    if sys.platform == 'darwin':  # macOS specific
        from Foundation import NSBundle
        # Set the bundle name to show in the menu bar
        bundle = NSBundle.mainBundle()
        if bundle:
            info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if info:
                info['CFBundleName'] = 'WhisperNotes'
    
 HEAD
    # Create the main application with error handling and logging
    try:
        logger.info("Instantiating WhisperNotes...")
        whisper_notes = WhisperNotes(app)
        logger.info("WhisperNotes instantiated successfully.")
    except Exception as e:
        logger.exception("Exception during WhisperNotes instantiation: %s", e)
        import traceback
        print("Exception during WhisperNotes instantiation:")
        print(traceback.format_exc())
        sys.exit(1)

 2819f8b (feat: Add task for batch audio import feature)
    # On macOS, we need to ensure the application is properly activated
    if sys.platform == 'darwin':
        from AppKit import NSApp
        NSApp.activateIgnoringOtherApps_(True)
    
    # Start the event loop
    sys.exit(app.exec())
 HEAD


# ------------------------------------------------------------------------------
# Proposed Module Split / Refactor Plan
# ------------------------------------------------------------------------------
# To improve maintainability and adhere to the project rule of keeping files
# under 500 lines, consider splitting whisper_notes.py into the following modules:
#
# 1. tray.py
#    - System tray icon, menu, and notification logic
#    - Tray setup, icon updates, and user actions
#
# 2. hotkeys.py
#    - Global hotkey registration, handling, and signal emission
#    - Platform-specific hotkey logic and permission checks
#
# 3. audio.py
#    - Audio recording thread (RecordingThread), sounddevice integration
#    - Audio data management, memory cleanup
#
# 4. transcription.py
#    - ModelLoader, TranscriptionWorker, and related threading logic
#    - Whisper model loading, transcription, and error handling
#
# 5. journaling.py
#    - JournalingManager and journal entry creation
#    - Integration with templates and clipboard management
#
# 6. main.py (or app.py)
#    - Application entry point and orchestration
#    - Instantiates and wires together the above modules
#
# 7. utils.py
#    - Shared utility functions, error handling, and platform helpers
#
# Each module should have clear docstrings and be independently testable.
# This split will improve readability, testability, and future extensibility.
# ------------------------------------------------------------------------------
 2819f8b (feat: Add task for batch audio import feature)
