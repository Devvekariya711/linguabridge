"""
LinguaBridge Bluetooth Manager
===============================
Helpers for Bluetooth audio device detection and management.
Uses sounddevice for device enumeration (cross-platform).
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import sounddevice
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice not installed. Audio device features disabled.")


# =============================================================================
# BLUETOOTH DETECTION KEYWORDS
# =============================================================================

BLUETOOTH_KEYWORDS = [
    # Generic
    'bluetooth', 'bt ', 'wireless',
    # Profiles
    'a2dp', 'hfp', 'hands-free', 'headset',
    # Common brands
    'airpods', 'galaxy buds', 'jabra', 'bose', 'sony wh', 'wf-',
    'beats', 'jbl', 'marshall', 'sennheiser', 'skullcandy',
    'anker', 'soundcore', 'realme buds', 'oneplus buds',
    'boat', 'noise', 'boult'  # Indian brands
]


# =============================================================================
# DEVICE DISCOVERY
# =============================================================================

def is_bluetooth_device(name: str) -> bool:
    """Check if device name suggests Bluetooth."""
    if not name:
        return False
    name_lower = name.lower()
    return any(kw in name_lower for kw in BLUETOOTH_KEYWORDS)


def find_bluetooth_audio_devices() -> List[Dict[str, Any]]:
    """
    Find all Bluetooth audio devices.
    
    Returns:
        List of device dicts with id, name, input/output channels
    """
    if not SOUNDDEVICE_AVAILABLE:
        logger.error("sounddevice not available")
        return []
    
    bluetooth_devices = []
    
    try:
        devices = sd.query_devices()
        
        for i, d in enumerate(devices):
            if is_bluetooth_device(d['name']):
                device_info = {
                    "id": i,
                    "name": d['name'],
                    "input_channels": d['max_input_channels'],
                    "output_channels": d['max_output_channels'],
                    "sample_rate": d['default_samplerate'],
                    "is_input": d['max_input_channels'] > 0,
                    "is_output": d['max_output_channels'] > 0,
                }
                bluetooth_devices.append(device_info)
                
    except Exception as e:
        logger.error(f"Error enumerating Bluetooth devices: {e}")
    
    return bluetooth_devices


def get_all_audio_devices() -> List[Dict[str, Any]]:
    """
    Get all audio devices with Bluetooth detection.
    
    Returns:
        List of device dicts
    """
    if not SOUNDDEVICE_AVAILABLE:
        return []
    
    devices = []
    
    try:
        raw_devices = sd.query_devices()
        default_input, default_output = sd.default.device
        
        for i, d in enumerate(raw_devices):
            device_info = {
                "id": i,
                "name": d['name'],
                "input_channels": d['max_input_channels'],
                "output_channels": d['max_output_channels'],
                "sample_rate": d['default_samplerate'],
                "is_input": d['max_input_channels'] > 0,
                "is_output": d['max_output_channels'] > 0,
                "is_default_input": (i == default_input),
                "is_default_output": (i == default_output),
                "is_bluetooth": is_bluetooth_device(d['name']),
            }
            devices.append(device_info)
            
    except Exception as e:
        logger.error(f"Error enumerating devices: {e}")
    
    return devices


def get_recommended_devices() -> Dict[str, Any]:
    """
    Get recommended playback and capture devices.
    
    Prefers:
    - Playback: Bluetooth A2DP (stereo output)
    - Capture: Built-in mic (better quality than HFP mic)
    
    Returns:
        Dict with recommended playback and capture device IDs
    """
    devices = get_all_audio_devices()
    
    recommended_playback = None
    recommended_capture = None
    
    # Find best playback (prefer Bluetooth with output)
    for d in devices:
        if d['is_output']:
            # Prefer Bluetooth for playback
            if d['is_bluetooth'] and recommended_playback is None:
                recommended_playback = d['id']
            # If no Bluetooth, use default
            elif d['is_default_output'] and recommended_playback is None:
                recommended_playback = d['id']
    
    # Find best capture (prefer built-in over Bluetooth)
    for d in devices:
        if d['is_input']:
            # Prefer non-Bluetooth for better quality
            if not d['is_bluetooth'] and recommended_capture is None:
                recommended_capture = d['id']
    
    # Fallback to Bluetooth capture if no other mic
    if recommended_capture is None:
        for d in devices:
            if d['is_input'] and d['is_bluetooth']:
                recommended_capture = d['id']
                break
    
    return {
        "recommended_playback": recommended_playback,
        "recommended_capture": recommended_capture,
        "note": "Recommended: Bluetooth for playback, built-in mic for capture (better quality)"
    }


def refresh_devices() -> List[Dict[str, Any]]:
    """
    Refresh device list (useful after pairing new device).
    
    Note: User should pair Bluetooth in Windows Settings first.
    """
    # Force sounddevice to refresh
    if SOUNDDEVICE_AVAILABLE:
        try:
            # Some versions support this
            sd._terminate()
            sd._initialize()
        except:
            pass
    
    return get_all_audio_devices()


# =============================================================================
# USER GUIDANCE
# =============================================================================

def get_pairing_instructions() -> str:
    """Get instructions for pairing Bluetooth audio."""
    return """
📱 How to Pair Bluetooth Earbuds:

1. Put earbuds in pairing mode (usually hold button 3-5 seconds)
2. Open Windows Settings → Bluetooth & devices
3. Click "Add device" → Bluetooth
4. Select your earbuds from the list
5. Wait for "Connected" status
6. Return to LinguaBridge and click "Refresh Devices"

💡 Tips:
- For best quality: Use earbuds for playback (TTS), laptop mic for speech
- Bluetooth adds ~100ms latency - this is normal
- If mic quality is poor, use built-in laptop microphone
"""


# =============================================================================
# DEVICE PREFERENCE (Simple in-memory storage)
# =============================================================================

class BluetoothPreference:
    """Store user's Bluetooth device preferences."""
    
    _playback_device: Optional[int] = None
    _capture_device: Optional[int] = None
    
    @classmethod
    def set_playback(cls, device_id: Optional[int]):
        cls._playback_device = device_id
    
    @classmethod
    def set_capture(cls, device_id: Optional[int]):
        cls._capture_device = device_id
    
    @classmethod
    def get_playback(cls) -> Optional[int]:
        return cls._playback_device
    
    @classmethod
    def get_capture(cls) -> Optional[int]:
        return cls._capture_device
    
    @classmethod
    def reset(cls):
        cls._playback_device = None
        cls._capture_device = None
