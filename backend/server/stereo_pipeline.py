"""
LinguaBridge Stereo Pipeline
==============================
Dual-channel translation: Left=English, Right=Hindi.
Two speakers, two parallel pipelines, one stereo output.
"""

import asyncio
import logging
import time
import threading
from typing import Optional, Callable
from collections import deque

import numpy as np
import sounddevice as sd

from .engine_stt import get_stt_engine
from .engine_nmt import get_nmt_engine
from .engine_tts import get_tts_engine
from .stereo_mixer import StereoMixer, merge_mono_to_stereo
from .speaker_session import SpeakerSession, DualSpeakerManager, get_speaker_manager

logger = logging.getLogger(__name__)


# Configuration
SAMPLE_RATE_STT = 16000
SAMPLE_RATE_TTS = 22050
SILENCE_THRESHOLD = 0.008
PHRASE_SILENCE = 0.7
MIN_PHRASE_DURATION = 0.4


class ChannelPipeline:
    """
    Single-channel pipeline: STT -> NMT -> TTS.
    
    Processes audio from one speaker, outputs to one channel.
    """
    
    def __init__(
        self,
        speaker_id: str,
        source_lang: str,
        target_lang: str,
        output_channel: str,
        on_result: Optional[Callable] = None
    ):
        self.speaker_id = speaker_id
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.output_channel = output_channel
        self.on_result = on_result
        
        self.stt = get_stt_engine()
        self.nmt = get_nmt_engine()
        self.tts = get_tts_engine()
        
        self.running = False
        self.audio_chunks = []
        self.last_sound_time = 0
        self.phrase_count = 0
        self.processing = False
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """Detect if audio contains speech."""
        rms = np.sqrt(np.mean(audio**2))
        return rms > SILENCE_THRESHOLD
    
    def process_phrase(self, audio: np.ndarray):
        """Process complete phrase through pipeline."""
        if self.processing:
            return
        
        self.processing = True
        
        try:
            duration = len(audio) / SAMPLE_RATE_STT
            if duration < MIN_PHRASE_DURATION:
                return
            
            self.phrase_count += 1
            t_start = time.perf_counter()
            
            # STT
            text = self.stt.transcribe(audio)
            if not text or len(text.strip()) < 2:
                return
            
            t_stt = time.perf_counter()
            
            # NMT
            translated, meta = self.nmt.smart_translate(
                text.strip(),
                self.source_lang,
                self.target_lang,
                use_llm=True,
                use_memory=True
            )
            
            t_nmt = time.perf_counter()
            
            # TTS - get MONO audio buffer
            voice = "en_US" if self.target_lang == "en" else "hi_IN"
            audio_out = self.tts.synthesize(translated, voice_key=voice)
            
            t_tts = time.perf_counter()
            
            total_ms = int((t_tts - t_start) * 1000)
            
            logger.info(
                f"[{self.speaker_id}] {self.source_lang}->{self.target_lang}: "
                f"'{text[:30]}' -> '{translated[:30]}' ({total_ms}ms)"
            )
            
            # Callback with result
            if self.on_result:
                self.on_result(
                    speaker_id=self.speaker_id,
                    original=text.strip(),
                    translated=translated,
                    audio=audio_out,
                    channel=self.output_channel,
                    latency_ms=total_ms
                )
            
        except Exception as e:
            logger.error(f"[{self.speaker_id}] Pipeline error: {e}")
        finally:
            self.processing = False
    
    def feed_audio(self, audio: np.ndarray):
        """Feed audio to pipeline for processing."""
        has_speech = self.is_speech(audio)
        
        if has_speech:
            self.audio_chunks.append(audio.copy())
            self.last_sound_time = time.time()
        
        elif self.audio_chunks:
            # Check for end of phrase
            silence_duration = time.time() - self.last_sound_time
            
            if silence_duration >= PHRASE_SILENCE:
                # Phrase complete
                full_audio = np.concatenate(self.audio_chunks)
                self.audio_chunks = []
                
                # Process in background
                threading.Thread(
                    target=self.process_phrase,
                    args=(full_audio,),
                    daemon=True
                ).start()


class StereoPipeline:
    """
    Dual-channel stereo pipeline.
    
    Speaker A (Hindi) -> English TTS -> LEFT channel
    Speaker B (English) -> Hindi TTS -> RIGHT channel
    """
    
    def __init__(self, output_device: Optional[int] = None):
        self.output_device = output_device
        
        # Audio mixer
        self.mixer = StereoMixer(sample_rate=SAMPLE_RATE_TTS)
        
        # Speaker manager
        self.speakers = get_speaker_manager()
        
        # Pipelines (created on start)
        self.pipeline_a: Optional[ChannelPipeline] = None
        self.pipeline_b: Optional[ChannelPipeline] = None
        
        self.running = False
        self.playback_thread: Optional[threading.Thread] = None
    
    def _on_result(self, speaker_id, original, translated, audio, channel, latency_ms):
        """Handle pipeline result - route audio to correct channel."""
        if channel == "left":
            self.mixer.add_left(audio)
        else:
            self.mixer.add_right(audio)
        
        # Log for visibility
        print(f"  [{speaker_id.upper()}] {original[:40]}")
        print(f"        -> {translated[:40]} ({channel.upper()})")
    
    def _playback_loop(self):
        """Continuous stereo playback."""
        chunk_samples = int(SAMPLE_RATE_TTS * 0.1)  # 100ms chunks
        
        try:
            with sd.OutputStream(
                samplerate=SAMPLE_RATE_TTS,
                channels=2,
                dtype='float32',
                device=self.output_device,
                blocksize=chunk_samples
            ) as stream:
                while self.running:
                    if self.mixer.has_audio():
                        stereo = self.mixer.get_stereo_chunk(chunk_samples)
                        stream.write(stereo)
                    else:
                        time.sleep(0.01)
        except Exception as e:
            logger.error(f"Playback error: {e}")
    
    def start(
        self,
        device_a: Optional[int] = None,
        device_b: Optional[int] = None
    ):
        """
        Start stereo translation.
        
        Args:
            device_a: Mic for Hindi speaker (or None for demo mode)
            device_b: Mic for English speaker (or None for demo mode)
        """
        print("\n" + "="*60)
        print("STEREO TRANSLATION MODE")
        print("="*60)
        print("  LEFT EAR = English (translated Hindi)")
        print("  RIGHT EAR = Hindi (translated English)")
        print("="*60)
        
        # Setup speakers
        self.speakers.setup_speakers(device_a, device_b)
        
        # Create pipelines
        self.pipeline_a = ChannelPipeline(
            speaker_id="a",
            source_lang="hi",
            target_lang="en",
            output_channel="left",
            on_result=self._on_result
        )
        
        self.pipeline_b = ChannelPipeline(
            speaker_id="b",
            source_lang="en",
            target_lang="hi",
            output_channel="right",
            on_result=self._on_result
        )
        
        self.running = True
        
        # Start playback thread
        self.playback_thread = threading.Thread(
            target=self._playback_loop,
            daemon=True
        )
        self.playback_thread.start()
        
        print("\n  Pipelines ready. Use demo mode or connect two mics.\n")
    
    def feed_speaker_a(self, audio: np.ndarray):
        """Feed audio from Hindi speaker."""
        if self.pipeline_a:
            self.pipeline_a.feed_audio(audio)
    
    def feed_speaker_b(self, audio: np.ndarray):
        """Feed audio from English speaker."""
        if self.pipeline_b:
            self.pipeline_b.feed_audio(audio)
    
    def stop(self):
        """Stop all pipelines."""
        self.running = False
        self.speakers.stop()
        self.mixer.clear()
        print("\n  Stereo pipeline stopped.\n")
    
    def demo_translate(self, text: str, speaker: str = "a"):
        """
        Demo: Translate text and output to correct channel.
        
        Args:
            text: Text to translate
            speaker: "a" (Hindi speaker) or "b" (English speaker)
        """
        if speaker == "a":
            # Hindi -> English -> LEFT
            src, tgt = "hi", "en"
            voice = "en_US"
            channel = "left"
        else:
            # English -> Hindi -> RIGHT
            src, tgt = "en", "hi"
            voice = "hi_IN"
            channel = "right"
        
        try:
            # Translate
            translated, _ = self.nmt.smart_translate(text, src, tgt, use_llm=False)
            
            # TTS
            audio = self.tts.synthesize(translated, voice_key=voice)
            
            # Route to channel
            if channel == "left":
                self.mixer.add_left(audio)
            else:
                self.mixer.add_right(audio)
            
            print(f"  [{speaker.upper()}] {text} -> {translated} ({channel.upper()})")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    @property
    def nmt(self):
        return get_nmt_engine()
    
    @property
    def tts(self):
        return get_tts_engine()


# Singleton
_stereo_pipeline: Optional[StereoPipeline] = None


def get_stereo_pipeline() -> StereoPipeline:
    """Get singleton stereo pipeline."""
    global _stereo_pipeline
    if _stereo_pipeline is None:
        _stereo_pipeline = StereoPipeline()
    return _stereo_pipeline
