# LinguaBridge 🌉

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Platform-Windows%20|%20Linux%20|%20Android-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/Offline-100%25-orange" alt="Offline">
</p>

<p align="center">
  <b>Real-time offline voice translation</b> — Speak in one language, hear in another. No internet required.
</p>

---

## 🤔 Why LinguaBridge?

- 💡 **100% Offline** — All AI models run locally on your device
- 🔒 **Privacy First** — No data leaves your computer, ever
- 💸 **Zero Cost** — No API keys, no subscriptions, no cloud fees
- ⚡ **Fast** — ~1.2s latency after warmup
- 🌍 **Multi-Language** — English, Hindi, Japanese support

---

## ✨ Features

| Feature | Technology | Status |
|---------|------------|--------|
| 🎤 **Speech-to-Text** | Faster-Whisper (OpenAI Whisper) | ✅ |
| 🌐 **Translation** | Argos Translate (Neural MT) | ✅ |
| 🔊 **Text-to-Speech** | Piper TTS (ONNX voices) | ✅ |
| �️ **Server** | FastAPI + Socket.IO | ✅ |
| 📱 **Mobile App** | Kivy (Python) | ✅ |
| 🌐 **Web Frontend** | React (coming soon) | 🔧 |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Devvekariya711/linguabridge.git
cd linguabridge
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download AI models
```bash
python backend/download_models.py --all
```

### 5. Start the server
```bash
python -m uvicorn backend.server.server_main:asgi_app --port 8000
```

### 6. Test translation
```bash
python test_quick.py
```

---

## 📁 Project Structure

```
linguabridge/
├── backend/
│   ├── server/              # FastAPI + Socket.IO server
│   │   ├── engine_stt.py        # Whisper STT
│   │   ├── engine_nmt.py        # Argos + LLM Translation
│   │   ├── engine_tts.py        # Piper TTS
│   │   ├── engine_llm.py        # Ollama LLM wrapper
│   │   ├── embeddings.py        # Sentence-transformers
│   │   ├── vector_db.py         # ChromaDB vector store
│   │   ├── translation_memory.py # SQLite + RAG cache
│   │   └── server_main.py       # Main server
│   ├── app/                 # Kivy mobile app
│   │   ├── main.py              # App entry
│   │   └── audio_streamer.py    # Mic capture
│   └── database/            # SQLite + ChromaDB storage
├── frontend/                # React web UI (coming)
├── git/                     # CI/CD, scripts, docs
│   ├── .github/workflows/       # GitHub Actions
│   └── scripts/                 # Build scripts
└── requirements.txt         # All dependencies
```

---

## 🌍 Supported Languages

| Language | STT | Translation | TTS |
|----------|:---:|:-----------:|:---:|
| English | ✅ | ✅ | ✅ |
| Hindi | ✅ | ✅ | ✅ |
| Japanese | ✅ | ✅ | ❌ |

---

## ⚡ Performance

| Metric | Cold Start | Warm | With Cache |
|--------|-----------|------|------------|
| **Full Pipeline** | ~9s | ~1.2s | **<0.1s** |
| STT (3s audio) | ~5s | ~0.6s | - |
| Translation (LLM) | ~3s | ~2s | **<1ms** |
| Translation (Argos) | ~0.5s | ~0.2s | **<1ms** |
| TTS | ~2.5s | ~0.3s | - |

> 💡 **Translation Memory:** Cached phrases return in <1ms via exact match or vector search.

---

## 🔧 API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ping` | GET | Health check |
| `/api/status` | GET | Engine status |

### Socket.IO Events

```javascript
// Transcribe voice
socket.emit('voice_chunk', audioBlob);
socket.on('transcription_result', (data) => console.log(data.text));

// Translate text
socket.emit('translate_text', {
  text: 'Hello',
  source_lang: 'en',
  target_lang: 'hi'
});
socket.on('translation_result', (data) => console.log(data.translated));

// Full pipeline (STT → NMT → TTS)
socket.emit('full_pipeline', {
  audio: audioBlob,
  source_lang: 'en',
  target_lang: 'hi'
});
socket.on('pipeline_result', (data) => {
  console.log(data.original, '→', data.translated);
  playAudio(data.audio);
});
```

---

## 🛠️ Development

```bash
# Run interactive test
python test_quick.py

# Run full pipeline test
python test_pipeline.py

# Run latency benchmark
python benchmark_latency.py
```

---

## 📋 Requirements

- Python 3.10+
- ~3GB disk space for AI models
- ~90MB for embedding model
- Microphone (for voice input)
- Speakers (for audio output)
- **Optional:** Ollama for LLM translation
- **Optional:** GPU for faster inference

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 🔒 Security

See [SECURITY.md](SECURITY.md) for security policy and responsible disclosure.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Credits

| Component | Technology |
|-----------|------------|
| STT | [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) |
| Translation | [Argos Translate](https://github.com/argosopentech/argos-translate) |
| LLM | [Ollama](https://ollama.ai) |
| Vector Search | [ChromaDB](https://github.com/chroma-core/chroma) |
| Embeddings | [Sentence-Transformers](https://www.sbert.net/) |
| TTS | [Piper TTS](https://github.com/rhasspy/piper) |
| Server | [FastAPI](https://fastapi.tiangolo.com/) |
| Mobile UI | [Kivy](https://kivy.org/) |

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/Devvekariya711">Dev Vekariya</a>
</p>

<p align="center">
  ⭐ Star this repo if you find it useful!
</p>
