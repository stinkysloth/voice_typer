# WhisperNotes

A powerful, privacy-focused voice-to-text application that runs entirely on your computer (macOS and Windows supported). WhisperNotes serves two distinct purposes: instant transcription for everyday tasks and thoughtful voice journaling with AI-enhanced organization.

## What is WhisperNotes?

WhisperNotes combines two powerful tools in one seamless application:

### 1. Instant Voice-to-Text Transcription

Capture your spoken words instantly with a simple keyboard shortcut. Perfect for when you need to:

- Dictate information quickly (credit card numbers, addresses, passwords)
- Take notes while your hands are busy
- Draft emails or messages without typing
- Capture ideas on the fly

### 2. Voice Journaling System

Transform your spoken thoughts into organized, searchable journal entries with AI assistance. WhisperNotes:

- Creates markdown-formatted journal entries from your voice recordings
- Automatically generates concise summaries of your entries
- Stores both the audio recording and transcription for future reference
- Integrates seamlessly with Obsidian vaults (though works with any markdown system)
- Preserves your privacy by processing everything locally

## Features

- 🎙️ Global hotkeys for instant access (`Cmd+Shift+R` for transcription, `Cmd+Shift+J` for journaling)
- ✍️ Local speech processing using OpenAI's Whisper (no internet required)
- 🤖 AI-powered summaries and formatting via Ollama (customizable prompts)
- 📝 Automatic cursor placement for transcribed text
- 📊 Comprehensive logging with timestamps
- 🗂️ Organized storage of journal entries with audio recordings
- 🔍 Searchable journal entries compatible with Obsidian and other markdown systems
- 🎯 Visual recording indicator
- 🖥️ Unobtrusive system tray (menu bar) interface
- 🔒 Complete privacy - all processing happens on your device

## Installation

### Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning the repository)

### macOS Installation

1. **Install Python** if you haven't already:
   ```bash
   # Using Homebrew (recommended)
   brew install python
   ```

2. **Clone this repository** or download the source code:
   ```bash
   git clone https://github.com/pieChartsAreLies/WhisperNotes.git
   cd WhisperNotes
   ```

3. **Install additional audio dependencies** (required for PyAudio):
   ```bash
   brew install portaudio
   ```

4. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

5. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Install Ollama** (for AI-powered journaling features):
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull the Llama model (or another model of your choice)
   ollama pull llama3
   ```

### Windows Installation

1. **Install Python** from the [official website](https://www.python.org/downloads/windows/)

2. **Clone this repository** or download the source code:
   ```cmd
   git clone https://github.com/pieChartsAreLies/WhisperNotes.git
   cd WhisperNotes
   ```

3. **Create a virtual environment** (recommended):
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install the required dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```
   Note: On Windows, PyAudio might require additional steps. If you encounter issues, try:
   ```cmd
   pip install pipwin
   pipwin install pyaudio
   ```

5. **Install Ollama** (for AI-powered journaling features):
   - Download from [Ollama's website](https://ollama.com/download/windows)
   - Install and run Ollama
   - Pull the Llama model:
     ```cmd
     ollama pull llama3
     ```

## Usage

### Running the Application

**macOS**:
```bash
# If using a virtual environment
source venv/bin/activate
python whisper_notes.py
```

**Windows**:
```cmd
# If using a virtual environment
venv\Scripts\activate
python whisper_notes.py
```

### Setting Up Permissions

#### macOS
1. **Grant Accessibility Permissions**:
   - When prompted, go to System Preferences > Security & Privacy > Privacy > Accessibility
   - Click the lock icon and enter your password
   - Add the application to the list of allowed applications
   - Add your Terminal or iTerm app to the list
   - If you're using VS Code's terminal, add VS Code to the list

#### Windows
1. **No special permissions required** in most cases, but you may need to:
   - Run the application as Administrator the first time
   - Allow Python through the Windows Firewall if prompted

### Using the Features

#### For Instant Transcription
- Place your cursor where you want the text to appear
- Press the recording hotkey (default: `Cmd+Shift+R` on macOS, `Ctrl+Shift+R` on Windows) to start recording
- Speak clearly into your microphone
- Press the same hotkey again to stop recording
- The transcribed text will appear at your cursor position

#### For Voice Journaling
- Press the journal hotkey (default: `Cmd+Shift+J` on macOS, `Ctrl+Shift+J` on Windows) to start a journal entry recording
- Speak your thoughts, ideas, or reflections
- Press the same hotkey again to stop recording
   - WhisperNotes will:
     - Save the audio recording
     - Transcribe your speech
     - Generate a summary using Ollama
     - Format the text for readability
     - Save everything to your journal directory

5. **Customizing Settings**:
   - Click on the WhisperNotes icon in the menu bar
   - Select "Output Settings" to configure:
     - Transcription output file location
     - Journal directory location
     - Summary prompt customization
     - Format prompt customization
   - Click the + button and add your Terminal or iTerm app
   - If you're using VS Code's terminal, you'll need to add VS Code to the list

## Privacy & Security

WhisperNotes is designed with privacy as a core principle:

- **Fully Local Processing**: All speech recognition is performed on your device using Whisper
- **No Cloud Dependencies**: No audio or transcriptions are sent to external servers
- **Local AI**: Ollama runs locally on your machine for AI-powered features
- **Your Data Stays Yours**: All recordings and transcriptions are stored only on your computer

## Common Use Cases

### Instant Transcription
- Dictating addresses, phone numbers, or other information you need to type
- Taking quick notes during meetings or calls
- Drafting emails or messages hands-free
- Capturing ideas while multitasking

### Voice Journaling
- Daily reflections and personal journaling
- Meeting notes with automatic summaries
- Brainstorming sessions with organized output
- Research notes with searchable transcriptions
- Personal development tracking
- Idea capture for creative projects

## Why WhisperNotes?

Unlike cloud-based alternatives, WhisperNotes offers:

- **Complete Privacy**: No data ever leaves your computer
- **No Subscription Fees**: One-time setup, no recurring costs
- **Customizable Experience**: Adjust settings to match your workflow
- **Markdown Integration**: Works seamlessly with Obsidian and other knowledge management systems
- **Dual Functionality**: Both quick transcription and thoughtful journaling in one tool

## Journal Structure

When using the journaling feature:
- Entries are saved to `~/Documents/Personal/Audio Journal/Journal.md` (customizable)
- Each entry includes a timestamp, summary, and link to the full transcription
- Audio recordings are saved in the `recordings` subfolder
- Detailed entries are stored in the `entries` subfolder

## Logging

- Regular transcriptions are automatically saved to the configured output file (default: `~/Documents/WhisperNotesTranscriptions.md`) with timestamps.
- Journal entries are saved to `~/Documents/Personal/Audio Journal/Journal.md` with timestamps, summaries, and links to detailed entries.

## Customization

You can customize the following in the `CONFIG` dictionary in `whisper_notes.py`:

- Hotkey combination
- Whisper model size (`tiny`, `base`, `small`, `medium`, `large`)
- Audio recording settings
- Log file location

## Troubleshooting

- **App doesn't type text**: Make sure you've granted Accessibility permissions to your terminal app.
- **No audio input detected**: Check your microphone settings in System Preferences > Sound > Input.
- **Installation issues**: Try creating a virtual environment first:
  ```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

## Requirements

- macOS 10.15 or later
- Python 3.8+
- Microphone access
- Internet connection (only for first-time model download)
- Ollama running locally (for journal entry summarization)

## License

MIT
