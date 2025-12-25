"""
LinguaBridge Kivy App
=====================
Main entrypoint for the Kivy mobile/desktop application.

Features:
- Real-time voice translation
- Chat bubble display
- Microphone capture and playback
- Socket.IO server connection
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Kivy configuration (must be before imports)
os.environ["KIVY_LOG_LEVEL"] = "warning"

from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.properties import (
    StringProperty,
    BooleanProperty,
    ListProperty,
    ObjectProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.core.audio import SoundLoader

# Socket.IO client
import socketio

# Local imports
try:
    from .audio_streamer import AudioStreamer
except ImportError:
    from audio_streamer import AudioStreamer


class ChatBubble(BoxLayout):
    """A single chat message bubble."""
    text = StringProperty("")
    is_user = BooleanProperty(False)
    timestamp = StringProperty("")


class MainLayout(BoxLayout):
    """Main app layout with chat and controls."""
    
    # Properties
    server_url = StringProperty("http://localhost:8000")
    is_connected = BooleanProperty(False)
    is_recording = BooleanProperty(False)
    status_text = StringProperty("Disconnected")
    source_lang = StringProperty("en")
    target_lang = StringProperty("hi")
    
    # References
    chat_container = ObjectProperty(None)
    mic_button = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Socket.IO client
        self.sio = socketio.Client(logger=False)
        self.setup_socket_events()
        
        # Audio streamer
        self.audio_streamer = None
        
        # Connect on startup
        Clock.schedule_once(lambda dt: self.connect_to_server(), 1)
    
    def setup_socket_events(self):
        """Configure Socket.IO event handlers."""
        
        @self.sio.event
        def connect():
            logger.info("Connected to server")
            Clock.schedule_once(lambda dt: self._on_connected())
        
        @self.sio.event
        def disconnect():
            logger.info("Disconnected from server")
            Clock.schedule_once(lambda dt: self._on_disconnected())
        
        @self.sio.on("transcription_result")
        def on_transcription(data):
            text = data.get("text", "")
            Clock.schedule_once(lambda dt: self.add_message(text, is_user=True))
        
        @self.sio.on("translation_result")
        def on_translation(data):
            translated = data.get("translated", "")
            Clock.schedule_once(lambda dt: self.add_message(translated, is_user=False))
        
        @self.sio.on("audio_result")
        def on_audio(data):
            # Play received audio
            Clock.schedule_once(lambda dt: self.play_audio(data))
        
        @self.sio.on("pipeline_result")
        def on_pipeline(data):
            original = data.get("original", "")
            translated = data.get("translated", "")
            audio = data.get("audio", b"")
            
            Clock.schedule_once(lambda dt: self.add_message(original, is_user=True))
            Clock.schedule_once(lambda dt: self.add_message(translated, is_user=False))
            Clock.schedule_once(lambda dt: self.play_audio(audio))
        
        @self.sio.on("error")
        def on_error(data):
            error_type = data.get("type", "unknown")
            message = data.get("message", "Unknown error")
            logger.error(f"Server error ({error_type}): {message}")
            Clock.schedule_once(lambda dt: self.show_error(message))
    
    def _on_connected(self):
        self.is_connected = True
        self.status_text = "Connected"
    
    def _on_disconnected(self):
        self.is_connected = False
        self.status_text = "Disconnected"
    
    def connect_to_server(self):
        """Connect to the backend server."""
        try:
            if not self.sio.connected:
                logger.info(f"Connecting to {self.server_url}...")
                self.status_text = "Connecting..."
                self.sio.connect(self.server_url)
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.status_text = f"Error: {e}"
    
    def toggle_recording(self):
        """Toggle microphone recording."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Start recording audio."""
        if not self.is_connected:
            self.show_error("Not connected to server")
            return
        
        try:
            if self.audio_streamer is None:
                self.audio_streamer = AudioStreamer(
                    on_audio_chunk=self.send_audio_chunk
                )
            
            self.audio_streamer.start()
            self.is_recording = True
            self.status_text = "Recording..."
            logger.info("Recording started")
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.show_error(str(e))
    
    def stop_recording(self):
        """Stop recording audio."""
        if self.audio_streamer:
            self.audio_streamer.stop()
        
        self.is_recording = False
        self.status_text = "Processing..."
        logger.info("Recording stopped")
    
    def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to server."""
        if self.sio.connected:
            self.sio.emit("voice_chunk", audio_data)
    
    def add_message(self, text: str, is_user: bool = True):
        """Add a message to the chat."""
        if not text.strip():
            return
        
        if self.chat_container:
            from datetime import datetime
            bubble = ChatBubble(
                text=text,
                is_user=is_user,
                timestamp=datetime.now().strftime("%H:%M")
            )
            self.chat_container.add_widget(bubble)
    
    def play_audio(self, audio_data: bytes):
        """Play audio data."""
        if not audio_data:
            return
        
        try:
            # Save to temp file and play
            temp_file = Path(__file__).parent / "temp_playback.wav"
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            sound = SoundLoader.load(str(temp_file))
            if sound:
                sound.play()
                logger.debug("Playing audio response")
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
    
    def show_error(self, message: str):
        """Show error message."""
        self.status_text = f"Error: {message}"
    
    def on_stop(self):
        """Cleanup on app stop."""
        if self.audio_streamer:
            self.audio_streamer.stop()
        
        if self.sio.connected:
            self.sio.disconnect()


class LinguaBridgeApp(App):
    """Main Kivy application."""
    
    title = "LinguaBridge"
    
    def build(self):
        """Build the app UI."""
        # Load KV file
        kv_file = Path(__file__).parent / "linguabridge.kv"
        if kv_file.exists():
            Builder.load_file(str(kv_file))
        
        return MainLayout()
    
    def on_stop(self):
        """Cleanup on app exit."""
        if hasattr(self.root, "on_stop"):
            self.root.on_stop()


def main():
    """Run the app."""
    LinguaBridgeApp().run()


if __name__ == "__main__":
    main()
