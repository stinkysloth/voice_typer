#!/usr/bin/env python3
"""
Journaling module for WhisperNotes application.
Handles saving transcriptions with timestamps to a markdown file.
"""
import os
import logging
import datetime
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
import soundfile as sf
import ollama

class JournalingManager:
    """
    Handles journaling functionality including saving transcriptions and audio recordings.
    Integrates with Ollama for text summarization and formatting.
    Supports templates for customizing journal entry format.
    """
    
    def __init__(self, output_dir: Optional[str] = None, summary_prompt: Optional[str] = None):
        """
        Initialize the JournalingManager.
        
        Args:
            output_dir: Directory to store journal entries and audio files.
                       If None, uses '~/Documents/Personal/Audio Journal/'.
            summary_prompt: Custom prompt to use for generating summaries
        """
        # Use the specified directory or default to ~/Documents/Personal/Audio Journal/
        if output_dir is None:
            home_dir = os.path.expanduser("~")
            self.output_dir = os.path.join(home_dir, "Documents", "Personal", "Audio Journal")
        else:
            self.output_dir = output_dir
            
        # Set default summary prompt (updated version)
        self.default_summary_prompt = "Summarize the following text in 1-2 sentences. Only output the summary, with no preamble or extra instructions."
        self.summary_prompt = summary_prompt if summary_prompt else self.default_summary_prompt
        
        # Default formatting prompt
        self.default_format_prompt = "Format this transcription into well-structured paragraphs. DO NOT add any commentary, analysis, or description. DO NOT change any words or meaning. Only add proper paragraph breaks, fix punctuation, and improve readability:"
        self.format_prompt = None  # Will be loaded from settings if available
            
        # Ensure directories exist
        self.ensure_directory_exists(self.output_dir)
        self.recordings_dir = os.path.join(self.output_dir, "recordings")
        self.ensure_directory_exists(self.recordings_dir)
        self.entries_dir = os.path.join(self.output_dir, "entries")
        self.ensure_directory_exists(self.entries_dir)
        
        # Journal file path
        self.journal_file = os.path.join(self.output_dir, "Journal.md")
        
        # Create journal file if it doesn't exist
        if not os.path.exists(self.journal_file):
            with open(self.journal_file, 'w', encoding='utf-8') as f:
                f.write("# Audio Journal\n\n")
                f.write("*Created on {}*\n\n".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")
                ))
        
        # Configure Ollama
        self.ollama_model = "llama3"  # Default model
        self.ollama_available = self._check_ollama_availability()
        if not self.ollama_available:
            logging.warning("Ollama is not available or no models are installed. Summaries will not be generated.")
            
        # Template support
        self.active_template = None  # Currently active template name
        self.custom_output_dir = None  # Custom output directory for template
        self.custom_tags = None  # Custom tags for template
            
    def _check_ollama_availability(self) -> bool:
        """
        Check if Ollama is available and has models installed.
        
        Returns:
            bool: True if Ollama is available, False otherwise
        """
        try:
            # Try to list models
            models = ollama.list()
            if not models.get('models'):
                logging.warning("No Ollama models found. Please install models with 'ollama pull llama3'")
                return False
            return True
        except Exception as e:
            logging.error(f"Error checking Ollama availability: {e}")
            return False
    
    def ensure_directory_exists(self, directory: str) -> None:
        """Ensure the specified directory exists, create it if it doesn't."""
        os.makedirs(directory, exist_ok=True)
    
    def save_audio(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """
        Save audio data to a file and return the file path.

        Args:
            audio_data (bytes or np.ndarray): Raw audio data as numpy array or bytes
            sample_rate (int): Audio sample rate

        Returns:
            str: Path to the saved audio file
        """
        import numpy as np
        logging.debug(f"[save_audio] self.recordings_dir: {self.recordings_dir}")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        filepath = os.path.join(self.recordings_dir, filename)
        try:
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.float32)
            # Save audio file
            sf.write(filepath, audio_data, sample_rate)
            logging.info(f"Audio saved to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"Error saving audio file: {e}")
            return ""
    
    def set_summary_prompt(self, prompt: str) -> None:
        """
        Set a custom prompt for generating summaries.
        
        Args:
            prompt: The custom prompt to use
        """
        self.summary_prompt = prompt
        logging.info(f"Summary prompt updated: {prompt[:50]}..." if len(prompt) > 50 else f"Summary prompt updated: {prompt}")
        
    def set_format_prompt(self, prompt: str) -> None:
        """
        Set a custom prompt for formatting transcriptions.
        
        Args:
            prompt: The custom prompt to use
        """
        self.format_prompt = prompt
        logging.info(f"Format prompt updated: {prompt[:50]}..." if len(prompt) > 50 else f"Format prompt updated: {prompt}")
    
    def process_with_ollama(self, text: str) -> Tuple[str, str]:
        """
        Process text with Ollama to get a summary and formatted version.
        
        Args:
            text: The raw transcription text
            
        Returns:
            Tuple containing (summary, formatted_text)
        """
        # If Ollama is not available, return the original text
        if not hasattr(self, 'ollama_available') or not self.ollama_available:
            logging.warning("Ollama not available. Using original text without processing.")
            return text[:50] + ("..." if len(text) > 50 else ""), text
            
        try:
            # Get summary from Ollama using the custom or default prompt
            summary_prompt = f"{self.summary_prompt} {text}"
            summary_response = ollama.chat(model=self.ollama_model, messages=[{
                "role": "user",
                "content": summary_prompt
            }])
            summary = summary_response['message']['content'].strip()
            
            # Get formatted text from Ollama using custom or default prompt
            if self.format_prompt:
                format_prompt_text = self.format_prompt
            else:
                format_prompt_text = self.default_format_prompt
                
            format_prompt = f"{format_prompt_text} {text}"
            format_response = ollama.chat(model=self.ollama_model, messages=[{
                "role": "user",
                "content": format_prompt
            }])
            formatted_text = format_response['message']['content'].strip()
            
            logging.info("Successfully processed text with Ollama")
            return summary, formatted_text
        except Exception as e:
            logging.error(f"Error processing text with Ollama: {e}")
            # Create a simple summary as fallback
            simple_summary = text[:50] + ("..." if len(text) > 50 else "")
            return simple_summary, text  # Return original text if processing fails
    
    def create_journal_entry(self, transcription: str, audio_data=None, sample_rate: int = 16000, template_name: Optional[str] = None):
        # Always use the original recordings_dir regardless of template/custom_output_dir
        self.recordings_dir = os.path.join(self.output_dir, "recordings")
        """
        Create a new journal entry with optional audio and template.
        
        Args:
            transcription: The transcribed text
            audio_data: Path to audio file or raw audio data
            sample_rate: Audio sample rate
            template_name: Name of template to use (overrides self.active_template)
            
        Returns:
            Path to the saved entry file
        """
        timestamp = datetime.datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        timestamp_str = timestamp.strftime("%Y-%m-%d %I:%M:%S %p")  # Changed to 12-hour format with AM/PM
        entry_id = timestamp.strftime("%Y%m%d_%H%M%S")  # Keep 24-hour format for filenames
        
        # Process text with Ollama
        summary, formatted_text = self.process_with_ollama(transcription)
        
        # Handle audio path
        audio_path = None
        relative_audio_path = None
        
        # Check if audio_data is a string (path) or raw audio data
        if isinstance(audio_data, str) and os.path.exists(audio_data):
            # It's already a path to an audio file
            audio_path = audio_data
            # Calculate path relative to the journal file location
            relative_audio_path = os.path.relpath(audio_path, os.path.dirname(self.journal_file))
            logging.debug(f"[create_journal_entry] Using existing audio_path: {audio_path}, relative_audio_path: {relative_audio_path}")
        elif audio_data is not None:
            # It's raw audio data
            audio_path = self.save_audio(audio_data, sample_rate)
            if audio_path:
                relative_audio_path = os.path.relpath(audio_path, os.path.dirname(self.journal_file))
                logging.debug(f"[create_journal_entry] Saved audio_path: {audio_path}, relative_audio_path: {relative_audio_path}")
            else:
                logging.warning(f"[create_journal_entry] Audio save failed, audio_path is empty.")
        
        # Use template_name parameter if provided, otherwise use active_template
        active_template = template_name or self.active_template
        
        # Use custom output directory if specified for the template
        output_dir = self.custom_output_dir if self.custom_output_dir else self.entries_dir
        
        # Add tags if specified for the template
        tags = self.custom_tags if self.custom_tags else ""
        
        # Create entry data
        entry = {
            "id": entry_id,
            "date": date_str,
            "timestamp": timestamp_str,
            "transcription": transcription,
            "summary": summary,
            "formatted_text": formatted_text,
            "audio_file": audio_path,
            "relative_audio_path": relative_audio_path,
            "tags": tags,
            "template": active_template,
            "title": f"Journal Entry - {timestamp_str}"
        }
        
        # Use custom output directory if specified
        original_entries_dir = self.entries_dir
        if self.custom_output_dir and os.path.isdir(self.custom_output_dir):
            self.entries_dir = self.custom_output_dir
            self.ensure_directory_exists(self.entries_dir)
        
        # Save detailed entry markdown file
        entry_file_path = self._save_entry_file(entry)
        entry["entry_file"] = entry_file_path
        
        # Set the entry link for the main journal entry
        entry["entry_link"] = f"[[{os.path.basename(entry_file_path).replace('.md', '')}]]"
        
        # Save to main journal file
        self._save_markdown_entry(entry)
        
        # Restore original entries directory
        self.entries_dir = original_entries_dir
        
        return entry_file_path
    
    def _save_entry_file(self, entry: Dict[str, Any]) -> str:
        """
        Save a detailed journal entry to its own markdown file.
        
        Args:
            entry: The journal entry data
            
        Returns:
            str: Path to the saved entry file
        """
        try:
            # Only use template if it is a non-empty string
            template = entry.get('template')
            if template and isinstance(template, str) and template.strip():
                return self._save_entry_with_template(entry)

            # Create filename based on date, ensure uniqueness with a counter if needed
            base_filename = f"{entry['date']} - Audio Journal Entry"
            entry_filename = f"{base_filename}.md"
            entry_path = os.path.join(self.entries_dir, entry_filename)
            counter = 2
            while os.path.exists(entry_path):
                entry_filename = f"{base_filename} ({counter}).md"
                entry_path = os.path.join(self.entries_dir, entry_filename)
                counter += 1

            with open(entry_path, 'w', encoding='utf-8') as f:
                f.write(f"# Audio Journal Entry - {entry['timestamp']}\n\n")
                f.write("### Summary\n")
                f.write(f"{entry['summary']}\n\n")
                f.write("### Transcript\n")
                f.write(f"{entry['formatted_text']}\n\n")
                # Add tags if available
                if entry.get('tags'):
                    f.write(f"**Tags**: {entry['tags']}\n\n")
                # Add link to audio recording if available
                if entry.get('relative_audio_path'):
                    f.write(f" [Listen to recording]({entry['relative_audio_path']})\n\n")

            logging.info(f"Detailed journal entry saved to {entry_path}")
            return entry_path
        except Exception as e:
            logging.error(f"Error saving detailed journal entry: {e}")
            return ""

    def _save_entry_with_template(self, entry: Dict[str, Any]) -> str:
        """
        Save a journal entry using a template.
        
        Args:
            entry: The journal entry data
            
        Returns:
            str: Path to the saved entry file
        """
        try:
            template = entry.get('template')
            if not template or not isinstance(template, str) or not template.strip():
                logging.warning("No valid template specified. Falling back to standard entry format.")
                return self._save_entry_file({**entry, 'template': ""})
            # Import template manager here to avoid circular imports
            from template_manager import TemplateManager
            
            # Create a temporary template manager
            template_manager = TemplateManager()
            
            # Apply template to entry
            formatted_content = template_manager.apply_template(template, entry)
            
            # Create filename based on template and date
            template_name = template.replace(" ", "_")
            entry_filename = f"{entry['date']} - {template_name}.md"
            entry_path = os.path.join(self.entries_dir, entry_filename)
            counter = 2
            while os.path.exists(entry_path):
                entry_filename = f"{entry['date']} - {template_name} ({counter}).md"
                entry_path = os.path.join(self.entries_dir, entry_filename)
                counter += 1
            
            # Write the formatted content to file
            with open(entry_path, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
                
            logging.info(f"Template-based journal entry saved to {entry_path}")
            return entry_path
        except Exception as e:
            logging.error(f"Error saving template-based journal entry: {e}")
            return ""  # Do not recurse
    
    def _save_markdown_entry(self, entry: Dict[str, Any]) -> None:
        """
        Save a journal entry to the main markdown file.
        
        Args:
            entry: The journal entry data
        """
        try:
            with open(self.journal_file, 'a', encoding='utf-8') as f:
                f.write(f"\n### {entry['timestamp']}\n\n")
                
                logging.debug(f"[journal.md] relative_audio_path: {entry.get('relative_audio_path')}")
                if entry.get('relative_audio_path'):
                    f.write(f" [Listen to recording]({entry['relative_audio_path']})\n\n")
                
                # Add summary
                f.write(f"{entry['summary']}\n\n")
                
                # Add link to detailed entry
                # Extract time from the timestamp (format: 'YYYY-MM-DD HH:MM:SS AM/PM')
                try:
                    timestamp = datetime.datetime.strptime(entry['timestamp'], '%Y-%m-%d %I:%M:%S %p')
                    time_str = timestamp.strftime('%I%M%p').lower()
                except (ValueError, KeyError) as e:
                    logging.warning(f"Could not parse timestamp, using current time: {e}")
                    time_str = datetime.datetime.now().strftime('%I%M%p').lower()
                    
                entry_link = f"[[{entry['date']} - {time_str} - Audio Journal Entry]]"
                f.write(f"{entry_link}\n\n")
                
                f.write("---\n\n")
                
            logging.info(f"Journal entry saved to {self.journal_file}")
        except Exception as e:
            logging.error(f"Error saving journal entry: {e}")
            raise
