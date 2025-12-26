"""
LinguaBridge Latency Benchmark
==============================
Measure time for each step of the pipeline.
"""

import time
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from backend.server.engine_stt import get_stt_engine
from backend.server.engine_nmt import get_nmt_engine
from backend.server.engine_tts import get_tts_engine


def benchmark():
    """Benchmark each component."""
    
    print("\n" + "="*60)
    print("[BENCHMARK] LinguaBridge Latency Test")
    print("="*60)
    
    # Test data
    test_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1  # 3 sec noise
    test_text_en = "Hello, how are you today?"
    test_text_hi = "namaste aap kaise hain"
    
    results = {}
    
    # 1. STT FIRST LOAD
    print("\n[1] STT Engine (Whisper) - First load...")
    t0 = time.perf_counter()
    stt = get_stt_engine()
    # Force model load by doing first transcription
    _ = stt.transcribe(np.zeros(1600, dtype=np.float32))  # 0.1s silence
    t1 = time.perf_counter()
    results['stt_load'] = t1 - t0
    print(f"    Model load: {results['stt_load']:.2f}s")
    
    # 2. STT INFERENCE
    print("\n[2] STT Inference (3s audio)...")
    t0 = time.perf_counter()
    _ = stt.transcribe(test_audio)
    t1 = time.perf_counter()
    results['stt_infer'] = t1 - t0
    print(f"    Inference: {results['stt_infer']:.2f}s")
    
    # Second run (cached)
    t0 = time.perf_counter()
    _ = stt.transcribe(test_audio)
    t1 = time.perf_counter()
    results['stt_infer_cached'] = t1 - t0
    print(f"    Inference (cached): {results['stt_infer_cached']:.2f}s")
    
    # 3. NMT LOAD
    print("\n[3] NMT Engine (Argos) - Load...")
    t0 = time.perf_counter()
    nmt = get_nmt_engine()
    t1 = time.perf_counter()
    results['nmt_load'] = t1 - t0
    print(f"    Load: {results['nmt_load']:.2f}s")
    
    # 4. NMT INFERENCE
    print("\n[4] NMT Translation (EN->HI)...")
    t0 = time.perf_counter()
    _ = nmt.translate(test_text_en, "en", "hi")
    t1 = time.perf_counter()
    results['nmt_infer'] = t1 - t0
    print(f"    First call: {results['nmt_infer']:.2f}s")
    
    # Second run
    t0 = time.perf_counter()
    _ = nmt.translate(test_text_en, "en", "hi")
    t1 = time.perf_counter()
    results['nmt_infer_cached'] = t1 - t0
    print(f"    Cached: {results['nmt_infer_cached']:.2f}s")
    
    # 5. TTS LOAD
    print("\n[5] TTS Engine (Piper) - Load voice...")
    t0 = time.perf_counter()
    tts = get_tts_engine()
    tts.load_voice("en_US")
    t1 = time.perf_counter()
    results['tts_load'] = t1 - t0
    print(f"    Voice load: {results['tts_load']:.2f}s")
    
    # 6. TTS INFERENCE
    print("\n[6] TTS Synthesis...")
    t0 = time.perf_counter()
    _ = tts.synthesize(test_text_en, voice_key="en_US")
    t1 = time.perf_counter()
    results['tts_infer'] = t1 - t0
    print(f"    First call: {results['tts_infer']:.2f}s")
    
    # Second run
    t0 = time.perf_counter()
    _ = tts.synthesize(test_text_en, voice_key="en_US")
    t1 = time.perf_counter()
    results['tts_infer_cached'] = t1 - t0
    print(f"    Cached: {results['tts_infer_cached']:.2f}s")
    
    # Summary
    print("\n" + "="*60)
    print("[SUMMARY] LATENCY RESULTS")
    print("="*60)
    
    total_first = results['stt_load'] + results['stt_infer'] + results['nmt_infer'] + results['tts_load'] + results['tts_infer']
    total_cached = results['stt_infer_cached'] + results['nmt_infer_cached'] + results['tts_infer_cached']
    
    print(f"\n[COLD START] First run:")
    print(f"   STT Load:     {results['stt_load']:.2f}s")
    print(f"   STT Infer:    {results['stt_infer']:.2f}s")
    print(f"   NMT Infer:    {results['nmt_infer']:.2f}s")
    print(f"   TTS Load:     {results['tts_load']:.2f}s")
    print(f"   TTS Infer:    {results['tts_infer']:.2f}s")
    print(f"   -----------------------")
    print(f"   TOTAL:        {total_first:.2f}s")
    
    print(f"\n[WARM] Subsequent runs:")
    print(f"   STT Infer:    {results['stt_infer_cached']:.2f}s")
    print(f"   NMT Infer:    {results['nmt_infer_cached']:.2f}s")
    print(f"   TTS Infer:    {results['tts_infer_cached']:.2f}s")
    print(f"   -----------------------")
    print(f"   TOTAL:        {total_cached:.2f}s")
    
    print("\n[TIPS] Optimization suggestions:")
    if results['stt_load'] > 2:
        print("   - Pre-load STT model at server startup")
    if results['stt_infer'] > 2:
        print("   - Use 'tiny' Whisper model for faster STT")
    if results['tts_load'] > 1:
        print("   - Pre-load TTS voice at startup")
    if total_cached > 3:
        print("   - Consider using GPU for faster inference")
    
    print("\n" + "="*60)
    

if __name__ == "__main__":
    benchmark()
