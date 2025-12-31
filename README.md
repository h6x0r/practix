<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# KODLA - Interactive Coding Platform

Learn Go and Java through hands-on coding exercises with real-time code execution.

## Features

- **7 Courses** (4 Go + 3 Java) with ~403 tasks
- **Multi-language** support (EN, RU, UZ)
- **Real code execution** with Piston + BullMQ
- **Web IDE Playground** at `/playground`
- **AI Tutor** powered by Gemini 2.0 Flash
- **Progress tracking** and premium tiers

## Quick Start

**Prerequisites:** Node.js 18+, Docker, Docker Compose

```bash
# Clone and install
npm install && cd server && npm install && cd ..

# Start everything
make start-docker
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8080

## Documentation

| Doc | Description |
|-----|-------------|
| [RUN_GUIDE.md](RUN_GUIDE.md) | Detailed setup & commands |
| [ROADMAP.md](ROADMAP.md) | Feature status & course content |
| [TECH_STACK.md](TECH_STACK.md) | Architecture & integrations |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18 + TypeScript + Monaco Editor |
| Backend | NestJS + Prisma + PostgreSQL |
| Code Execution | Piston + BullMQ + Redis |
| AI | Google Gemini 2.0 Flash |
| Auth | JWT |

## Makefile Commands

```bash
make start-docker   # Start full stack
make start          # Local dev mode
make stop           # Stop containers
make db-refresh     # Reset database
make vet            # Type-check both apps
```

## Environment Variables

Create `server/.env`:
```env
DATABASE_URL="postgresql://kodla_user:kodla_secure_password@db:5432/kodla_db"
JWT_SECRET="your-secret"
GEMINI_API_KEY="your-google-ai-key"
```

## License

MIT
