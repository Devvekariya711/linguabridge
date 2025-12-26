"""
LinguaBridge Quick Test
=======================
Interactive loop test for translation pipeline.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.server.engine_stt import get_stt_engine
from backend.server.engine_nmt import get_nmt_engine
from backend.server.engine_tts import get_tts_engine


def test_translation(text: str, source: str = "en", target: str = "hi"):
    """Test translation with timing."""
    nmt = get_nmt_engine()
    
    t0 = time.perf_counter()
    result = nmt.translate(text, source, target)
    t1 = time.perf_counter()
    
    return result, (t1 - t0)


def test_full_pipeline(text: str, source: str = "en", target: str = "hi"):
    """Test full NMT + TTS pipeline."""
    nmt = get_nmt_engine()
    tts = get_tts_engine()
    
    # Translate
    t0 = time.perf_counter()
    translated = nmt.translate(text, source, target)
    t_nmt = time.perf_counter() - t0
    
    # TTS
    voice = "hi_IN" if target == "hi" else "en_US"
    t0 = time.perf_counter()
    audio = tts.synthesize(translated, voice_key=voice)
    t_tts = time.perf_counter() - t0
    
    return translated, t_nmt, t_tts, audio


def play_audio(audio, sample_rate=22050):
    """Play audio."""
    try:
        import sounddevice as sd
        sd.play(audio, sample_rate)
        sd.wait()
    except:
        pass


def main():
    print("\n" + "="*60)
    print("LinguaBridge Interactive Test")
    print("="*60)
    print("\nCommands:")
    print("  Type text to translate EN->HI")
    print("  'hi:' prefix for HI->EN (e.g., 'hi: namaste')")
    print("  'q' to quit")
    print("  'loop N' to run N test iterations")
    print("="*60)
    
    # Pre-load engines
    print("\nLoading engines...")
    stt = get_stt_engine()
    nmt = get_nmt_engine()
    tts = get_tts_engine()
    
    # Warm up
    nmt.translate("hello", "en", "hi")
    tts.load_voice("en_US")
    tts.load_voice("hi_IN")
    print("Ready!\n")
    
    test_phrases = [
        "Hello, how are you?",
        "What is your name?",
        "I am learning Hindi.",
        "Nice to meet you.",
        "Thank you very much.",
    ]
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'q':
                print("Goodbye!")
                break
            
            # Loop test
            if user_input.lower().startswith('loop'):
                parts = user_input.split()
                n = int(parts[1]) if len(parts) > 1 else 3
                
                print(f"\nRunning {n} test iterations...")
                print("-"*60)
                
                total_time = 0
                for i, phrase in enumerate(test_phrases[:n]):
                    t0 = time.perf_counter()
                    translated, t_nmt, t_tts, audio = test_full_pipeline(phrase)
                    total = time.perf_counter() - t0
                    total_time += total
                    
                    print(f"[{i+1}] {phrase}")
                    print(f"    -> {translated}")
                    print(f"    Time: NMT={t_nmt:.2f}s, TTS={t_tts:.2f}s, Total={total:.2f}s")
                    
                    # Play audio
                    play_audio(audio)
                    time.sleep(0.5)
                
                print("-"*60)
                print(f"Average: {total_time/n:.2f}s per translation")
                continue
            
            # Detect direction
            if user_input.lower().startswith('hi:'):
                text = user_input[3:].strip()
                source, target = "hi", "en"
            else:
                text = user_input
                source, target = "en", "hi"
            
            # Translate and speak
            t0 = time.perf_counter()
            translated, t_nmt, t_tts, audio = test_full_pipeline(text, source, target)
            total = time.perf_counter() - t0
            
            print(f"\nIN:  {text}")
            print(f"OUT: {translated}")
            print(f"Time: {total:.2f}s (NMT={t_nmt:.2f}s, TTS={t_tts:.2f}s)")
            
            play_audio(audio)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
