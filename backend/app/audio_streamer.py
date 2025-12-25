"""
LinguaBridge Audio Streamer
===========================
Capture audio from microphone and stream to server.
"""

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Sample rate for Whisper (16kHz mono)
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 500  # Audio chunk duration
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)


class AudioStreamer:
    """
    Capture and stream audio from microphone.
    
    Uses sounddevice for cross-platform audio capture.
    Chunks audio and sends via callback.
    """
    
    def __init__(
        self,
        on_audio_chunk: Optional[Callable[[bytes], None]] = None,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        chunk_duration_ms: int = CHUNK_DURATION_MS,
    ):
        """
        Initialize audio streamer.
        
        Args:
            on_audio_chunk: Callback for each audio chunk (WAV bytes)
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            chunk_duration_ms: Chunk duration in milliseconds
        """
        self.on_audio_chunk = on_audio_chunk
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        
        self._stream = None
        self._audio_queue = queue.Queue()
        self._is_running = False
        self._processor_thread = None
        
        # Check sounddevice availability
        self._check_sounddevice()
    
    def _check_sounddevice(self):
        """Check if sounddevice is available."""
        try:
            import sounddevice as sd
            self.sd = sd
            logger.info(f"Audio device: {sd.query_devices(None, 'input')['name']}")
        except ImportError:
            logger.error("sounddevice not installed. Run: pip install sounddevice")
            raise RuntimeError("sounddevice not available")
        except Exception as e:
            logger.warning(f"Audio device query failed: {e}")
            self.sd = None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream."""
        if status:
            logger.warning(f"Audio status: {status}")
        
        # Copy audio data to queue
        audio_copy = indata.copy()
        self._audio_queue.put(audio_copy)
    
    def _process_audio(self):
        """Process audio chunks in background thread."""
        import io
        import wave
        
        accumulated = []
        accumulated_samples = 0
        
        while self._is_running:
            try:
                # Get audio from queue with timeout
                chunk = self._audio_queue.get(timeout=0.1)
                
                # Accumulate samples
                accumulated.append(chunk)
                accumulated_samples += len(chunk)
                
                # When we have enough samples, process
                if accumulated_samples >= self.chunk_size:
                    # Combine accumulated audio
                    audio = np.concatenate(accumulated, axis=0)[:self.chunk_size]
                    
                    # Convert to int16
                    audio_int16 = (audio * 32767).astype(np.int16)
                    
                    # Create WAV bytes
                    buffer = io.BytesIO()
                    with wave.open(buffer, 'wb') as wav:
                        wav.setnchannels(self.channels)
                        wav.setsampwidth(2)  # 16-bit
                        wav.setframerate(self.sample_rate)
                        wav.writeframes(audio_int16.tobytes())
                    
                    wav_bytes = buffer.getvalue()
                    
                    # Send to callback
                    if self.on_audio_chunk:
                        self.on_audio_chunk(wav_bytes)
                    
                    # Reset accumulator
                    accumulated = []
                    accumulated_samples = 0
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
    
    def start(self):
        """Start audio capture."""
        if self._is_running:
            return
        
        try:
            import sounddevice as sd
            
            self._is_running = True
            
            # Start processor thread
            self._processor_thread = threading.Thread(
                target=self._process_audio,
                daemon=True
            )
            self._processor_thread.start()
            
            # Start input stream
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=1024,
            )
            self._stream.start()
            
            logger.info("Audio capture started")
            
        except Exception as e:
            self._is_running = False
            logger.error(f"Failed to start audio: {e}")
            raise
    
    def stop(self):
        """Stop audio capture."""
        self._is_running = False
        
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error stopping stream: {e}")
            self._stream = None
        
        if self._processor_thread:
            self._processor_thread.join(timeout=1)
            self._processor_thread = None
        
        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Audio capture stopped")
    
    def is_running(self) -> bool:
        """Check if streaming is active."""
        return self._is_running


def test_audio():
    """Test audio capture."""
    def on_chunk(data):
        print(f"Received chunk: {len(data)} bytes")
    
    streamer = AudioStreamer(on_audio_chunk=on_chunk)
    streamer.start()
    
    import time
    time.sleep(3)
    
    streamer.stop()
    print("Test complete")


if __name__ == "__main__":
    test_audio()
