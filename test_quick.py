"""
LinguaBridge Live Console v8
=============================
With Bluetooth device selection and real-time translation.
"""

import sys
import time
import threading
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

import sounddevice as sd
import numpy as np

from backend.server.engine_nmt import get_nmt_engine
from backend.server.engine_tts import get_tts_engine
from backend.server.engine_stt import get_stt_engine
from backend.server import translation_memory

# =============================================================================
# CONFIG
# =============================================================================
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.008
PHRASE_SILENCE = 0.7
MIN_PHRASE_DURATION = 0.5

# Device selection
PLAYBACK_DEVICE = None  # None = system default
CAPTURE_DEVICE = None

# =============================================================================
# AUDIO DEVICE HELPERS
# =============================================================================

def list_devices():
    """List all audio devices."""
    devices = sd.query_devices()
    default_in, default_out = sd.default.device
    
    print("\n" + "="*60)
    print("🎧 AUDIO DEVICES")
    print("="*60)
    
    print("\n📤 OUTPUT (Playback) Devices:")
    for i, d in enumerate(devices):
        if d['max_output_channels'] > 0:
            marker = " ⭐" if i == default_out else ""
            print(f"  [{i}] {d['name']}{marker}")
    
    print("\n📥 INPUT (Capture) Devices:")
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            marker = " ⭐" if i == default_in else ""
            print(f"  [{i}] {d['name']}{marker}")
    
    print("\n  ⭐ = System Default")
    print("="*60)

def select_devices():
    """Interactive device selection."""
    global PLAYBACK_DEVICE, CAPTURE_DEVICE
    
    list_devices()
    
    try:
        print("\n📤 Select PLAYBACK device (for TTS output):")
        choice = input("   Device ID (or Enter for default): ").strip()
        if choice:
            PLAYBACK_DEVICE = int(choice)
            print(f"   ✅ Playback: {sd.query_devices(PLAYBACK_DEVICE)['name']}")
        
        print("\n📥 Select CAPTURE device (for STT input):")
        choice = input("   Device ID (or Enter for default): ").strip()
        if choice:
            CAPTURE_DEVICE = int(choice)
            print(f"   ✅ Capture: {sd.query_devices(CAPTURE_DEVICE)['name']}")
            
    except (ValueError, sd.PortAudioError) as e:
        print(f"   ❌ Invalid device: {e}")

def test_playback():
    """Test playback device with a beep."""
    global PLAYBACK_DEVICE
    print(f"\n🔊 Testing playback device: {PLAYBACK_DEVICE or 'default'}...")
    
    # Generate a test tone
    duration = 0.5
    freq = 440
    t = np.linspace(0, duration, int(22050 * duration))
    tone = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    
    try:
        sd.play(tone, 22050, device=PLAYBACK_DEVICE)
        sd.wait()
        print("   ✅ Playback OK!")
    except Exception as e:
        print(f"   ❌ Playback failed: {e}")

def test_capture():
    """Test capture device by recording 2s."""
    global CAPTURE_DEVICE
    print(f"\n🎤 Testing capture device: {CAPTURE_DEVICE or 'default'}...")
    print("   Recording 2 seconds...")
    
    try:
        audio = sd.rec(int(2 * SAMPLE_RATE), samplerate=SAMPLE_RATE, 
                       channels=1, dtype='float32', device=CAPTURE_DEVICE)
        sd.wait()
        
        rms = np.sqrt(np.mean(audio**2))
        level = int(rms * 100 * 10)  # Scale for display
        bar = "█" * level + "░" * (20 - level)
        
        print(f"   Level: [{bar}] {rms:.4f}")
        print("   ✅ Capture OK!")
    except Exception as e:
        print(f"   ❌ Capture failed: {e}")

# =============================================================================
# PHRASE TRANSLATOR
# =============================================================================

class PhraseTranslator:
    """Translates complete phrases with device selection support."""
    
    def __init__(self, stt, nmt, tts):
        self.stt = stt
        self.nmt = nmt
        self.tts = tts
        
        self.running = False
        self.recording = False
        self.audio_chunks = []
        self.last_sound_time = 0
        self.phrase_count = 0
        self.processing = False
    
    def is_speech(self, audio):
        rms = np.sqrt(np.mean(audio**2))
        return rms > SILENCE_THRESHOLD
    
    def process_phrase(self, audio):
        if self.processing:
            return
        
        self.processing = True
        
        try:
            duration = len(audio) / SAMPLE_RATE
            if duration < MIN_PHRASE_DURATION:
                return
            
            self.phrase_count += 1
            
            # STT
            t0 = time.perf_counter()
            text = self.stt.transcribe(audio)
            stt_time = time.perf_counter() - t0
            
            if not text or len(text.strip()) < 2:
                return
            
            # NMT
            t1 = time.perf_counter()
            translated, meta = self.nmt.smart_translate(
                text.strip(), "en", "hi",
                use_llm=True, use_memory=True
            )
            nmt_time = time.perf_counter() - t1
            
            # Display
            print(f"\n  ━━━ Phrase {self.phrase_count} ━━━")
            print(f"  🎤 YOU: \"{text.strip()}\"")
            print(f"  🌐 हिंदी: \"{translated}\"")
            print(f"  ⏱️  STT:{stt_time*1000:.0f}ms NMT:{nmt_time*1000:.0f}ms ({meta.get('source','')})")
            
            # TTS - play on selected device
            try:
                audio_out = self.tts.synthesize(translated, voice_key="hi_IN")
                sd.play(audio_out, 22050, device=PLAYBACK_DEVICE)
                sd.wait()
            except Exception as e:
                print(f"  🔊 TTS error: {e}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        finally:
            self.processing = False
    
    def audio_callback(self, indata, frames, time_info, status):
        if not self.running:
            return
        
        audio = indata[:, 0].copy()
        has_speech = self.is_speech(audio)
        
        if has_speech:
            self.recording = True
            self.audio_chunks.append(audio)
            self.last_sound_time = time.time()
        
        elif self.recording:
            self.audio_chunks.append(audio)
            silence_duration = time.time() - self.last_sound_time
            
            if silence_duration >= PHRASE_SILENCE:
                if self.audio_chunks:
                    full_audio = np.concatenate(self.audio_chunks)
                    self.audio_chunks = []
                    self.recording = False
                    
                    threading.Thread(
                        target=self.process_phrase,
                        args=(full_audio,),
                        daemon=True
                    ).start()
    
    def start(self):
        print("\n" + "="*60)
        print("🎙️  PHRASE-BASED TRANSLATION")
        print("="*60)
        print(f"   Capture: {CAPTURE_DEVICE or 'default'}")
        print(f"   Playback: {PLAYBACK_DEVICE or 'default'}")
        print("   Press Ctrl+C to stop")
        print("="*60)
        print("\n  🟢 LISTENING... Speak, pause, hear translation!\n")
        
        self.running = True
        self.audio_chunks = []
        self.phrase_count = 0
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=int(SAMPLE_RATE * 0.1),
                device=CAPTURE_DEVICE,
                callback=self.audio_callback
            ):
                while self.running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            print(f"\n  🛑 STOPPED | Phrases: {self.phrase_count}\n")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("🌐 LinguaBridge Translation Console v8")
    print("   With Bluetooth Device Selection")
    print("="*60)
    
    print("\n⏳ Loading engines...")
    translation_memory.init_db()
    
    stt = get_stt_engine()
    nmt = get_nmt_engine()
    tts = get_tts_engine()
    tts.load_voice("hi_IN")
    tts.load_voice("en_US")
    
    print("✅ Ready!\n")
    
    while True:
        print("="*60)
        print("  1 = 🎙️  Start Translation")
        print("  2 = 🎧  Select Audio Devices")
        print("  3 = 🔊  Test Playback")
        print("  4 = 🎤  Test Microphone")
        print("  5 = ⌨️  Text Mode")
        print("  q = Quit")
        print("="*60)
        
        try:
            choice = input("\nChoice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break
        
        if choice == 'q':
            print("👋 Goodbye!")
            break
        
        elif choice == '1':
            translator = PhraseTranslator(stt, nmt, tts)
            translator.start()
        
        elif choice == '2':
            select_devices()
        
        elif choice == '3':
            test_playback()
        
        elif choice == '4':
            test_capture()
        
        elif choice == '5':
            print("\n⌨️ Text mode (type 'q' to exit)")
            while True:
                try:
                    text = input("  > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if text.lower() == 'q':
                    break
                if text:
                    try:
                        trans, meta = nmt.smart_translate(text, "en", "hi", use_llm=False)
                        print(f"  🌐 {trans}")
                        audio = tts.synthesize(trans, voice_key="hi_IN")
                        sd.play(audio, 22050, device=PLAYBACK_DEVICE)
                        sd.wait()
                    except Exception as e:
                        print(f"  ❌ {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")