# LinguaBridge - Git Documentation
# =================================

This folder contains Git-related configuration, CI/CD workflows, and scripts.

## Structure

```
git/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI
│
├── scripts/
│   ├── build_frontend.sh   # Build React frontend
│   ├── run_local.sh        # Run dev server
│   └── migrate_db.sh       # Database migration
│
├── .gitattributes          # Line endings config
└── README.md               # This file
```

## Scripts

### run_local.sh
Start development server:
```bash
./git/scripts/run_local.sh
```

### build_frontend.sh
Build and deploy frontend:
```bash
./git/scripts/build_frontend.sh
```

### migrate_db.sh
Initialize database:
```bash
./git/scripts/migrate_db.sh
```

## CI/CD

GitHub Actions runs on every push/PR to `main`:
- Import tests for all engines
- Code style check with flake8

## Notes

- Main `.gitignore` is at repository root (required by Git)
- `.gitattributes` handles line endings for cross-platform compatibility
