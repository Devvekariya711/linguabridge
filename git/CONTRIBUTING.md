# Contributing to LinguaBridge 🤝

Thank you for your interest in contributing to LinguaBridge! 

## 📋 Ways to Contribute

- 🐛 **Report bugs** — Open an issue with reproduction steps
- 💡 **Suggest features** — Open an issue with `[Feature]` prefix
- 📖 **Improve docs** — Fix typos, add examples
- 🔧 **Submit PRs** — Bug fixes, new features

---

## 🚀 Development Setup

### 1. Fork & Clone
```bash
git clone https://github.com/YOUR_USERNAME/linguabridge.git
cd linguabridge
git checkout -b feature/your-feature-name
```

### 2. Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Tests
```bash
python test_quick.py
python test_pipeline.py
```

---

## 📝 Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

| Type | Description |
|------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation |
| `style:` | Formatting |
| `refactor:` | Code restructure |
| `test:` | Add/update tests |
| `chore:` | Maintenance |

**Examples:**
```
feat: add Japanese TTS voice support
fix: resolve duplicate UI in Kivy app
docs: update API documentation
```

---

## ✅ Pull Request Checklist

- [ ] Code follows project style
- [ ] Tests pass locally
- [ ] Documentation updated (if needed)
- [ ] Commit messages follow convention
- [ ] No merge conflicts

---

## 📁 Code Structure

```
backend/
├── server/       # FastAPI server & AI engines
├── app/          # Kivy mobile app
└── database/     # SQLite storage

frontend/         # React web UI
git/              # CI/CD, scripts
```

---

## 🔍 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn

---

<p align="center">Thank you for contributing! 🙏</p>
