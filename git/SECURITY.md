# Security Policy 🔒

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ Active |

---

## 🛡️ Security Features

LinguaBridge is designed with **privacy-first** principles:

| Feature | Description |
|---------|-------------|
| 📴 **100% Offline** | No internet connection required |
| 🚫 **No Cloud** | All processing happens on-device |
| 🔇 **No Telemetry** | Zero usage tracking |
| 💾 **Local Storage** | SQLite database stays on device |

---

## ⚠️ Reporting Vulnerabilities

**DO NOT** open a public issue for security vulnerabilities.

### Contact
📧 Email: devvekariya711@gmail.com

### Include
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

| Stage | Time |
|-------|------|
| Initial response | 48 hours |
| Status update | 7 days |
| Fix release | 30 days |

---

## 🔐 Security Best Practices

### For Development
```env
# backend/.env
HOST=127.0.0.1          # Localhost only
DEBUG=false             # Disable debug mode
CORS_ALLOWED_ORIGINS=https://yourdomain.com
```

### For Production
- 🔒 Don't expose port 8000 to internet
- 🛡️ Add authentication layer
- 🔐 Use HTTPS/TLS if exposing server
- 🧹 Regularly clean `temp_audio/` folder

---

## ⚡ Known Limitations

| Issue | Mitigation |
|-------|------------|
| Server binds to `0.0.0.0` | Change to `127.0.0.1` in `.env` |
| No built-in auth | Add authentication for production |
| Temp audio files | Auto-cleaned after 24h |

---

<p align="center">Thank you for helping keep LinguaBridge secure! 🙏</p>
