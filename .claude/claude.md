# KODLA Project Configuration

## Agent Rules

**IMPORTANT: Parallel agents are FORBIDDEN unless explicitly requested in the prompt.**
- Always work synchronously, step by step
- Only use parallel Task tools when user explicitly says "in parallel" or "параллельно"
- Focus on careful, thorough implementation over speed

---

## MCP Tools Usage

### When to use Sequential Thinking (`mcp__sequentialthinking`)
Use for complex problem-solving that requires step-by-step analysis:
- Architecture decisions with multiple trade-offs
- Debugging complex issues
- Planning multi-step implementations
- Analyzing code for refactoring

### When to use Memory (`mcp__memory`)
Use for storing/retrieving persistent knowledge across sessions:
- User preferences and decisions
- Project-specific conventions discovered
- Important context that should persist
- Notes about code patterns used in this project

### When to use Context7 (`mcp__context7`)
Use for retrieving up-to-date documentation:
- Library/framework documentation (React, NestJS, Prisma, etc.)
- API references and examples
- Best practices from official docs

---

## Quick Reference

| File | Purpose |
|------|---------|
| `ROADMAP.md` | Complete course catalog and development status |
| `TECH_STACK.md` | Technology stack and integrations |
| `RUN_GUIDE.md` | Local development setup |
| `TASK_CREATION_GUIDE.md` | How to create new tasks/courses |
| `docs/TECH_DEBT_AND_ROADMAP.md` | Technical debt and implementation plans |

---

## Key Directories

### Backend (`/server/`)
```
server/
├── prisma/
│   ├── schema.prisma      # Database schema
│   ├── seed.ts            # Database seeder
│   └── seeds/             # Course content
│       └── courses/       # All course definitions
├── src/
│   ├── subscriptions/     # Subscription system
│   ├── submissions/       # Code execution & grading
│   ├── ai/                # AI Tutor (Gemini)
│   ├── piston/            # Code execution engine
│   ├── queue/             # BullMQ job queue
│   ├── health/            # Health checks & metrics
│   └── gamification/      # XP, levels, badges
```

### Frontend (`/src/`)
```
src/
├── features/
│   ├── subscriptions/     # Subscription UI & API
│   ├── tasks/             # Task workspace
│   ├── courses/           # Course catalog
│   ├── playground/        # Web IDE
│   └── dashboard/         # User stats
├── contexts/              # React contexts
└── components/            # Shared UI components
```

---

## Current Platform Status

### Production Ready
- **18 Courses** (~921 tasks) with full localization (EN/RU/UZ)
- **Piston Code Execution** - 8 languages
- **BullMQ Queue + Redis Caching**
- **Playground (Web IDE)** - /playground
- **AI Tutor** - Gemini 2.0 Flash (100 req/day premium)
- **Gamification** - XP, levels, badges, streaks
- **Health Checks** - /health, /health/metrics
- **Swagger Docs** - /api/docs (dev only)

### Planned Courses
- **Prompt Engineering** (Priority: HIGH) - 35+ tasks
- **Math for Data Science** (Priority: MEDIUM) - Discussion needed
- **System Design** (Priority: MEDIUM) - 30+ tasks

---

## AI Tutor Limits

| Tier | Daily Limit | Notes |
|------|-------------|-------|
| Free (no subscription) | 5 | Basic access |
| Course subscription | 30 | Per-course purchase |
| Global Premium | 100 | Full platform access |
| Prompt Engineering course | 100 | Special limit for PE course |

---

## Adding New Courses

See `TASK_CREATION_GUIDE.md` for complete instructions.

### Required Task Fields
```typescript
{
  slug: string,           // unique identifier
  title: string,          // English title
  difficulty: 'easy' | 'medium' | 'hard',
  tags: string[],
  estimatedTime: string,  // '15m', '30m', '1h'
  isPremium: boolean,
  description: string,
  initialCode: string,
  solutionCode: string,
  testCode: string,       // 10 test cases required
  hint1: string,
  hint2: string,
  whyItMatters: string,
  order: number,
  translations: { ru: {...}, uz: {...} }
}
```

---

## Session Notes

### Jan 4, 2025
- Added Swagger/OpenAPI documentation
- Added Prometheus metrics and health checks
- Implemented graceful shutdown for BullMQ
- Test coverage at 80%+ for all services
- Updated AI tutor limits: 100/day for premium

### Next Priority
1. Prompt Engineering course (with 100 AI req/day limit)
2. Math for Data Science (needs execution strategy)
