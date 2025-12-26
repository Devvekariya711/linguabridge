# Changelog

All notable changes to LinguaBridge will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - 2025-12-26

### Added
- Speech-to-Text engine using Faster-Whisper
- Neural Machine Translation using Argos Translate
- Text-to-Speech using Piper TTS
- FastAPI + Socket.IO server
- Kivy mobile application
- SQLite conversation storage
- Pre-loading for fast first requests
- Interactive test scripts
- Latency benchmark tool

### Supported Languages
- English (STT, NMT, TTS)
- Hindi (STT, NMT, TTS)
- Japanese (STT, NMT only)

### Known Issues
- Translation quality varies for complex sentences
- Proper names may not translate correctly
- Japanese TTS not available (no Piper voice)

---

## [Unreleased]

### Planned
- Web frontend (React)
- Android APK build
- Streaming STT for real-time transcription
- Additional language support
- Improved translation models
