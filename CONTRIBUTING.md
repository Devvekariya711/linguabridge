# Contributing to LinguaBridge

Thank you for your interest in contributing! 🎉

## 📋 How to Contribute

### 1. Fork & Clone
```bash
git clone https://github.com/YOUR_USERNAME/linguabridge.git
cd linguabridge
git checkout -b feature/your-feature-name
```

### 2. Set Up Development Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Make Changes

- Follow existing code style
- Add tests for new features
- Update documentation if needed

### 4. Test Your Changes
```bash
python test_pipeline.py
python test_quick.py
```

### 5. Commit & Push
```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### 6. Open Pull Request

Go to GitHub and create a Pull Request.

---

## 📝 Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

| Type | Description |
|------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation |
| `style:` | Formatting |
| `refactor:` | Code restructure |
| `test:` | Add tests |
| `chore:` | Maintenance |

Examples:
```
feat: add Japanese TTS voice support
fix: resolve duplicate UI in Kivy app
docs: update API documentation
```

---

## 🐛 Reporting Bugs

1. Check if issue already exists
2. Create new issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - System info (OS, Python version)
   - Error logs

---

## 💡 Feature Requests

1. Open an issue with `[Feature]` prefix
2. Describe the feature clearly
3. Explain why it's useful

---

## 📁 Code Structure

```
backend/
├── server/       # Add server features here
├── app/          # Kivy mobile app
└── database/     # Database models

frontend/         # Web UI (React)
```

---

## ✅ Checklist Before PR

- [ ] Code follows project style
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts

---

## 🤝 Code of Conduct

- Be respectful
- Be constructive
- Help others learn

---

Thank you for contributing! 🙏
