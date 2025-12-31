"""
LinguaBridge Audio Device Manager
==================================
Enumerate, select, and manage audio devices including Bluetooth.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import sounddevice as sd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AudioDevice:
    """Represents an audio device."""
    id: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default_input: bool
    is_default_output: bool
    is_bluetooth: bool
    device_type: str  # "input", "output", "duplex"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "max_input_channels": self.max_input_channels,
            "max_output_channels": self.max_output_channels,
            "default_samplerate": self.default_samplerate,
            "is_default_input": self.is_default_input,
            "is_default_output": self.is_default_output,
            "is_bluetooth": self.is_bluetooth,
            "device_type": self.device_type,
        }


# =============================================================================
# DEVICE ENUMERATION
# =============================================================================

def _is_bluetooth_device(name: str) -> bool:
    """Check if device name suggests Bluetooth."""
    name_lower = name.lower()
    bt_keywords = [
        'bluetooth', 'bt', 'wireless', 'airpods', 'galaxy buds',
        'jabra', 'bose', 'sony wh', 'wf-', 'beats', 'headset',
        'hands-free', 'a2dp', 'hfp', 'jbl', 'marshall'
    ]
    return any(kw in name_lower for kw in bt_keywords)


def _get_device_type(device: dict) -> str:
    """Determine device type from channels."""
    has_input = device['max_input_channels'] > 0
    has_output = device['max_output_channels'] > 0
    
    if has_input and has_output:
        return "duplex"
    elif has_input:
        return "input"
    elif has_output:
        return "output"
    return "unknown"


def list_all_devices() -> List[AudioDevice]:
    """
    List all available audio devices.
    
    Returns:
        List of AudioDevice objects
    """
    devices = []
    raw_devices = sd.query_devices()
    default_input, default_output = sd.default.device
    
    for i, d in enumerate(raw_devices):
        device = AudioDevice(
            id=i,
            name=d['name'],
            max_input_channels=d['max_input_channels'],
            max_output_channels=d['max_output_channels'],
            default_samplerate=d['default_samplerate'],
            is_default_input=(i == default_input),
            is_default_output=(i == default_output),
            is_bluetooth=_is_bluetooth_device(d['name']),
            device_type=_get_device_type(d),
        )
        devices.append(device)
    
    return devices


def list_input_devices() -> List[AudioDevice]:
    """List devices with input (microphone) capability."""
    return [d for d in list_all_devices() if d.max_input_channels > 0]


def list_output_devices() -> List[AudioDevice]:
    """List devices with output (playback) capability."""
    return [d for d in list_all_devices() if d.max_output_channels > 0]


def list_bluetooth_devices() -> List[AudioDevice]:
    """List Bluetooth audio devices."""
    return [d for d in list_all_devices() if d.is_bluetooth]


def get_device(device_id: int) -> Optional[AudioDevice]:
    """Get device by ID."""
    devices = list_all_devices()
    for d in devices:
        if d.id == device_id:
            return d
    return None


def get_default_input() -> Optional[int]:
    """Get default input device ID."""
    default_input, _ = sd.default.device
    return default_input


def get_default_output() -> Optional[int]:
    """Get default output device ID."""
    _, default_output = sd.default.device
    return default_output


# =============================================================================
# DEVICE TESTING
# =============================================================================

def test_playback_device(device_id: Optional[int] = None, duration: float = 0.5) -> bool:
    """
    Test playback device by playing a tone.
    
    Args:
        device_id: Device ID or None for default
        duration: Tone duration in seconds
    
    Returns:
        True if successful
    """
    try:
        # Generate 440 Hz sine wave
        samplerate = 22050
        t = np.linspace(0, duration, int(samplerate * duration))
        tone = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
        
        sd.play(tone, samplerate, device=device_id)
        sd.wait()
        
        logger.info(f"Playback test successful: device={device_id}")
        return True
        
    except Exception as e:
        logger.error(f"Playback test failed: {e}")
        return False


def test_capture_device(device_id: Optional[int] = None, duration: float = 2.0) -> Dict[str, Any]:
    """
    Test capture device by recording audio.
    
    Args:
        device_id: Device ID or None for default
        duration: Recording duration in seconds
    
    Returns:
        Dict with test results including audio level
    """
    try:
        samplerate = 16000
        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='float32',
            device=device_id
        )
        sd.wait()
        
        # Calculate RMS level
        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.max(np.abs(audio)))
        
        logger.info(f"Capture test successful: device={device_id}, rms={rms:.4f}")
        
        return {
            "success": True,
            "device_id": device_id,
            "duration": duration,
            "rms_level": rms,
            "peak_level": peak,
            "has_signal": rms > 0.001,
        }
        
    except Exception as e:
        logger.error(f"Capture test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "device_id": device_id,
        }


# =============================================================================
# DEVICE PREFERENCE STORAGE
# =============================================================================

class DevicePreference:
    """Stores user device preferences in memory."""
    
    _playback_device: Optional[int] = None
    _capture_device: Optional[int] = None
    
    @classmethod
    def set_playback(cls, device_id: Optional[int]):
        """Set preferred playback device."""
        cls._playback_device = device_id
        logger.info(f"Playback device set to: {device_id}")
    
    @classmethod
    def set_capture(cls, device_id: Optional[int]):
        """Set preferred capture device."""
        cls._capture_device = device_id
        logger.info(f"Capture device set to: {device_id}")
    
    @classmethod
    def get_playback(cls) -> Optional[int]:
        """Get preferred playback device."""
        return cls._playback_device
    
    @classmethod
    def get_capture(cls) -> Optional[int]:
        """Get preferred capture device."""
        return cls._capture_device
    
    @classmethod
    def reset(cls):
        """Reset to system defaults."""
        cls._playback_device = None
        cls._capture_device = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_devices_summary() -> Dict[str, Any]:
    """Get a summary of available devices."""
    all_devices = list_all_devices()
    
    return {
        "total": len(all_devices),
        "input_devices": len([d for d in all_devices if d.max_input_channels > 0]),
        "output_devices": len([d for d in all_devices if d.max_output_channels > 0]),
        "bluetooth_devices": len([d for d in all_devices if d.is_bluetooth]),
        "default_input": get_default_input(),
        "default_output": get_default_output(),
        "selected_playback": DevicePreference.get_playback(),
        "selected_capture": DevicePreference.get_capture(),
    }
