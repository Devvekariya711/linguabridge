"""
LinguaBridge Terminal Test
==========================
Test the full translation pipeline from terminal.
Records audio, transcribes, translates, speaks, and saves everything.
"""

import os
import sys
import time
import wave
from pathlib import Path
from datetime import datetime

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.server.engine_stt import get_stt_engine
from backend.server.engine_nmt import get_nmt_engine
from backend.server.engine_tts import get_tts_engine


# Output directory
OUTPUT_DIR = Path("backend/storage/test_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Record audio from microphone."""
    try:
        import sounddevice as sd
        print(f"\n🎤 Recording for {duration} seconds... SPEAK NOW!")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        print("✅ Recording complete!")
        return audio.flatten()
    except ImportError:
        print("❌ sounddevice not installed. Run: pip install sounddevice")
        return None
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return None


def save_audio(audio: np.ndarray, filepath: Path, sample_rate: int = 16000):
    """Save audio to WAV file."""
    # Convert to int16
    if audio.dtype == np.float32:
        audio_int16 = (audio * 32767).astype(np.int16)
    else:
        audio_int16 = audio.astype(np.int16)
    
    with wave.open(str(filepath), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    print(f"💾 Saved: {filepath}")


def play_audio(audio: np.ndarray, sample_rate: int = 22050):
    """Play audio through speakers."""
    try:
        import sounddevice as sd
        print("🔊 Playing audio...")
        sd.play(audio, sample_rate)
        sd.wait()
        print("✅ Playback complete!")
    except Exception as e:
        print(f"⚠️ Playback failed: {e}")


def run_pipeline(
    source_lang: str = "en",
    target_lang: str = "hi",
    record_duration: float = 5.0
):
    """Run the full translation pipeline."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*60)
    print("🌉 LinguaBridge Terminal Test")
    print("="*60)
    print(f"Source: {source_lang} -> Target: {target_lang}")
    print(f"Record duration: {record_duration}s")
    print("="*60)
    
    # 1. Record audio
    print("\n[STEP 1] Recording audio...")
    input_audio = record_audio(record_duration)
    
    if input_audio is None:
        return
    
    # Save input audio
    input_path = OUTPUT_DIR / f"input_{timestamp}.wav"
    save_audio(input_audio, input_path, sample_rate=16000)
    
    # 2. Speech-to-Text
    print("\n[STEP 2] Transcribing speech (STT)...")
    stt = get_stt_engine()
    original_text = stt.transcribe(input_audio)
    print(f"📝 Transcription: \"{original_text}\"")
    
    if not original_text.strip():
        print("⚠️ No speech detected. Try speaking louder.")
        return
    
    # 3. Translation
    print(f"\n[STEP 3] Translating {source_lang} -> {target_lang} (NMT)...")
    nmt = get_nmt_engine()
    translated_text = nmt.translate(original_text, source_lang, target_lang)
    print(f"🌐 Translation: \"{translated_text}\"")
    
    # 4. Text-to-Speech
    voice = "hi_IN" if target_lang == "hi" else "en_US"
    print(f"\n[STEP 4] Synthesizing speech (TTS - {voice})...")
    tts = get_tts_engine()
    
    if not tts.load_voice(voice):
        print(f"⚠️ Voice {voice} not available, using en_US")
        voice = "en_US"
        tts.load_voice(voice)
    
    output_audio = tts.synthesize(translated_text, voice_key=voice)
    print(f"🔊 Generated {len(output_audio)} audio samples")
    
    # Save output audio
    output_path = OUTPUT_DIR / f"output_{timestamp}.wav"
    save_audio(output_audio, output_path, sample_rate=22050)
    
    # 5. Play output
    print("\n[STEP 5] Playing translated speech...")
    play_audio(output_audio, sample_rate=22050)
    
    # 6. Save text log
    log_path = OUTPUT_DIR / f"log_{timestamp}.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"LinguaBridge Translation Log\n")
        f.write(f"============================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Direction: {source_lang} -> {target_lang}\n")
        f.write(f"\n")
        f.write(f"INPUT TEXT:\n{original_text}\n")
        f.write(f"\n")
        f.write(f"OUTPUT TEXT:\n{translated_text}\n")
        f.write(f"\n")
        f.write(f"INPUT AUDIO: {input_path.name}\n")
        f.write(f"OUTPUT AUDIO: {output_path.name}\n")
    
    print(f"\n💾 Log saved: {log_path}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"INPUT:  \"{original_text}\"")
    print(f"OUTPUT: \"{translated_text}\"")
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print("="*60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LinguaBridge Terminal Test")
    parser.add_argument("--source", "-s", default="en", help="Source language (default: en)")
    parser.add_argument("--target", "-t", default="hi", help="Target language (default: hi)")
    parser.add_argument("--duration", "-d", type=float, default=5.0, help="Record duration in seconds")
    parser.add_argument("--text", help="Text to translate (skip recording)")
    
    args = parser.parse_args()
    
    if args.text:
        # Text-only mode
        print("\n" + "="*60)
        print("🌉 LinguaBridge Text Translation")
        print("="*60)
        
        nmt = get_nmt_engine()
        translated = nmt.translate(args.text, args.source, args.target)
        
        print(f"INPUT:  {args.text}")
        print(f"OUTPUT: {translated}")
        
        # TTS
        voice = "hi_IN" if args.target == "hi" else "en_US"
        tts = get_tts_engine()
        tts.load_voice(voice)
        audio = tts.synthesize(translated, voice_key=voice)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"tts_{timestamp}.wav"
        save_audio(audio, output_path, sample_rate=22050)
        
        play_audio(audio, sample_rate=22050)
    else:
        # Full pipeline with recording
        run_pipeline(
            source_lang=args.source,
            target_lang=args.target,
            record_duration=args.duration
        )


if __name__ == "__main__":
    main()
