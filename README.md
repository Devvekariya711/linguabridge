# LinguaBridge 🌉

**Real-time offline voice translation** - Speak in one language, hear in another. No internet required.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20|%20Linux%20|%20Android-lightgrey.svg)]()

## ✨ Features

- 🎤 **Speech-to-Text** - Whisper AI (Faster-Whisper)
- 🌐 **Translation** - Argos Translate (Neural MT)
- 🔊 **Text-to-Speech** - Piper TTS (Natural voices)
- 📴 **100% Offline** - No cloud, no cost
- ⚡ **Fast** - ~1.2s latency after warmup
- 🔒 **Private** - Your data stays on device

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/Devvekariya711/linguabridge.git
cd linguabridge
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Download Models
```bash
python backend/download_models.py --all
```

### 3. Start Server
```bash
python -m uvicorn backend.server.server_main:asgi_app --port 8000
```

### 4. Test Translation
```bash
python test_quick.py
```

## 📁 Project Structure

```
linguabridge/
├── backend/
│   ├── server/          # FastAPI + Socket.IO
│   │   ├── engine_stt.py    # Whisper STT
│   │   ├── engine_nmt.py    # Argos Translation
│   │   ├── engine_tts.py    # Piper TTS
│   │   └── server_main.py   # Main server
│   ├── app/             # Kivy mobile app
│   └── database/        # SQLite storage
├── frontend/            # React web UI (coming soon)
└── git/                 # CI/CD, scripts
```

## 🌍 Supported Languages

| Language | STT | Translation | TTS |
|----------|-----|-------------|-----|
| English | ✅ | ✅ | ✅ |
| Hindi | ✅ | ✅ | ✅ |
| Japanese | ✅ | ✅ | ❌ |

## ⚡ Performance

| Metric | First Run | Warm |
|--------|-----------|------|
| Full Pipeline | ~9s | ~1.2s |
| STT only | ~0.6s | ~0.6s |
| Translation | ~0.5s | ~0.2s |
| TTS only | ~0.3s | ~0.3s |

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ping` | GET | Health check |
| `/api/status` | GET | Engine status |
| `/socket.io` | WS | Real-time events |

### Socket.IO Events

```javascript
// Send voice for transcription
socket.emit('voice_chunk', audioBlob);
socket.on('transcription_result', (data) => console.log(data.text));

// Translate text
socket.emit('translate_text', {text: 'Hello', source_lang: 'en', target_lang: 'hi'});
socket.on('translation_result', (data) => console.log(data.translated));

// Full pipeline
socket.emit('full_pipeline', {audio: audioBlob, source_lang: 'en', target_lang: 'hi'});
socket.on('pipeline_result', (data) => playAudio(data.audio));
```

## 🛠️ Development

```bash
# Run tests
python test_pipeline.py

# Run benchmark
python benchmark_latency.py

# Interactive test
python test_quick.py
```

## 📋 Requirements

- Python 3.10+
- ~2GB disk space for models
- Microphone for voice input
- Speakers for audio output

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🔒 Security

See [SECURITY.md](SECURITY.md) for security policy.

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Credits

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - STT
- [Argos Translate](https://github.com/argosopentech/argos-translate) - NMT
- [Piper TTS](https://github.com/rhasspy/piper) - TTS
- [FastAPI](https://fastapi.tiangolo.com/) - Server
- [Kivy](https://kivy.org/) - Mobile UI

---

Made with ❤️ by [Dev Vekariya](https://github.com/Devvekariya711)
