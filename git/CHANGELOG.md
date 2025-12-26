# Changelog

All notable changes to LinguaBridge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

---

## [1.0.0] - 2025-12-26

### ✨ Added
- Speech-to-Text using Faster-Whisper (base model)
- Neural Machine Translation using Argos Translate
- Text-to-Speech using Piper TTS (en_US, hi_IN voices)
- FastAPI + Socket.IO real-time server
- Kivy mobile/desktop application
- SQLite conversation storage
- Model pre-loading for fast first requests
- Interactive test scripts
- Latency benchmark tool
- Complete documentation (README, CONTRIBUTING, SECURITY)
- GitHub Actions CI workflow

### 🌍 Supported Languages
| Language | STT | NMT | TTS |
|----------|-----|-----|-----|
| English | ✅ | ✅ | ✅ |
| Hindi | ✅ | ✅ | ✅ |
| Japanese | ✅ | ✅ | ❌ |

### ⚠️ Known Issues
- Translation quality varies for complex sentences
- Proper names may not translate correctly
- Japanese TTS not available (no Piper voice)

---

## [Unreleased]

### 🔮 Planned
- React web frontend
- Android APK build (Buildozer)
- Streaming STT for real-time transcription
- Additional language support
- Improved translation models
- GPU acceleration support
