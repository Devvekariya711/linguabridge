# Security Policy

## 🔒 Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ |

## 🛡️ Security Features

LinguaBridge is designed with privacy in mind:

- **100% Offline** - No data leaves your device
- **No Cloud** - All processing happens locally
- **No Telemetry** - No usage tracking
- **Local Storage** - SQLite database on device

## ⚠️ Reporting a Vulnerability

If you discover a security vulnerability:

1. **DO NOT** open a public issue
2. Email: devvekariya711@gmail.com
3. Include:
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

## 🔐 Best Practices for Users

1. **Keep Updated** - Use latest version
2. **Local Network** - Don't expose server to internet
3. **Firewall** - Block port 8000 from external access
4. **CORS** - Configure allowed origins in production

## 🚫 Known Limitations

- Server binds to `0.0.0.0` by default (change in `.env`)
- No authentication built-in (add for production)
- Audio files stored temporarily in `/storage/temp_audio/`

## 🛠️ Production Recommendations

```env
# backend/.env
HOST=127.0.0.1          # Localhost only
DEBUG=false             # Disable debug mode
CORS_ALLOWED_ORIGINS=https://yourdomain.com
```

---

Thank you for helping keep LinguaBridge secure! 🙏
