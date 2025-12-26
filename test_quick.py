"""
LinguaBridge Quick Test
=======================
Interactive test for translation with LLM + memory.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.server.engine_nmt import get_nmt_engine
from backend.server.engine_tts import get_tts_engine
from backend.server import translation_memory
from backend.server.constants import LLM_ENABLED, LLM_MODEL


def play_audio(audio, sample_rate=22050):
    """Play audio."""
    try:
        import sounddevice as sd
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception:
        pass


def main():
    print("\n" + "="*60)
    print("LinguaBridge Interactive Test")
    print("="*60)
    print(f"LLM Enabled: {LLM_ENABLED}")
    print(f"LLM Model: {LLM_MODEL}")
    print("\nCommands:")
    print("  Type text to translate EN->HI")
    print("  'hi:' prefix for HI->EN")
    print("  'fast:' prefix to use Argos only (skip LLM)")
    print("  'stats' to show memory stats")
    print("  'q' to quit")
    print("="*60)
    
    # Init memory
    translation_memory.init_db()
    
    # Load engines
    print("\nLoading engines...")
    nmt = get_nmt_engine()
    tts = get_tts_engine()
    tts.load_voice("en_US")
    tts.load_voice("hi_IN")
    print("Ready!\n")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'q':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                stats = translation_memory.get_stats()
                print(f"Memory entries: {stats['total_entries']}")
                print(f"Language pairs: {stats['language_pairs']}")
                continue
            
            # Parse options
            use_llm = True
            source, target = "en", "hi"
            text = user_input
            
            if user_input.lower().startswith('fast:'):
                use_llm = False
                text = user_input[5:].strip()
            elif user_input.lower().startswith('hi:'):
                text = user_input[3:].strip()
                source, target = "hi", "en"
            
            # Translate
            t0 = time.perf_counter()
            translated, meta = nmt.smart_translate(text, source, target, use_llm=use_llm)
            elapsed = time.perf_counter() - t0
            
            # Print result (encode for Windows console)
            print(f"\nIN:  {text}")
            try:
                print(f"OUT: {translated}")
            except UnicodeEncodeError:
                print(f"OUT: {translated.encode('utf-8')}")
            print(f"Source: {meta.get('source')} | Time: {elapsed:.2f}s")
            
            # Play audio
            voice = "hi_IN" if target == "hi" else "en_US"
            try:
                audio = tts.synthesize(translated, voice_key=voice)
                play_audio(audio)
            except Exception as e:
                print(f"TTS failed: {e}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
