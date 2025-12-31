# KODLA Project Documentation Map

## Agent Rules

**IMPORTANT: Parallel agents are FORBIDDEN unless explicitly requested in the prompt.**
- Always work synchronously, step by step
- Only use parallel Task tools when user explicitly says "in parallel" or "параллельно"
- Focus on careful, thorough implementation over speed

---

## Quick Reference

| File | Purpose |
|------|---------|
| `docs/NEXT_STEPS.md` | Current development plan (subscriptions, AI/ML courses, gamification) |
| `docs/TECH_DEBT_AND_ROADMAP.md` | Technical debt tracker and long-term roadmap |
| `ROADMAP.md` | High-level product roadmap |
| `TECH_STACK.md` | Technology stack overview |
| `RUN_GUIDE.md` | How to run the project locally |
| `TASK_CREATION_GUIDE.md` | Guide for creating new tasks/courses |

---

## Documentation Structure

### `/docs/` - Technical Documentation

| File | Description |
|------|-------------|
| `NEXT_STEPS.md` | **ACTIVE PLAN** - Current sprint/iteration work |
| `TECH_DEBT_AND_ROADMAP.md` | Tech debt items and long-term technical roadmap |
| `API_ARCHITECTURE.md` | Backend API design and patterns |
| `API_CLIENT_USAGE.md` | Frontend API client usage guide |
| `API_SERVICES_MIGRATION.md` | Migration notes for API services |
| `API_UNIFICATION_SUMMARY.md` | Summary of API unification work |
| `README_API.md` | API documentation overview |
| `TASK_MIGRATION_GUIDE.md` | Guide for migrating tasks to new structure |
| `ai-ml-course-research.md` | Research for Python AI/ML course |
| `java-ai-ml-course-research.md` | Research for Java AI/ML course |
| `ai-ml-implementation-plan.md` | Implementation plan for AI/ML courses |

### Root Level `.md` Files

| File | Description |
|------|-------------|
| `README.md` | Project overview and quick start |
| `ROADMAP.md` | Product roadmap (features, timeline) |
| `RUN_GUIDE.md` | Local development setup |
| `TECH_STACK.md` | Technologies used |
| `TASK_CREATION_GUIDE.md` | How to create new course content |

---

## Key Directories

### Backend (`/server/`)
```
server/
├── prisma/
│   ├── schema.prisma      # Database schema
│   ├── seed.ts            # Database seeder
│   └── seeds/             # Course content (tasks, modules, topics)
│       └── courses/       # All course definitions
├── src/
│   ├── subscriptions/     # NEW: Subscription system
│   ├── submissions/       # Code execution & grading
│   ├── ai/                # AI Tutor (Gemini)
│   ├── piston/            # Code execution engine
│   └── queue/             # BullMQ job queue
```

### Frontend (`/src/`)
```
src/
├── features/
│   ├── subscriptions/     # NEW: Subscription types & API
│   ├── tasks/             # Task workspace
│   ├── courses/           # Course catalog
│   └── dashboard/         # User dashboard & stats
├── contexts/
│   ├── SubscriptionContext.tsx  # NEW: Subscription state
│   └── LanguageContext.tsx      # i18n
└── components/            # Shared UI components
```

---

## Current Focus Areas

### 1. Subscription System (DONE - needs testing)
- Backend: `/server/src/subscriptions/`
- Frontend: `/src/features/subscriptions/`, `/src/contexts/SubscriptionContext.tsx`
- Schema: `SubscriptionPlan`, `Subscription`, `Payment` models

### 2. AI/ML Courses (PLANNING)
- Research: `docs/ai-ml-course-research.md`, `docs/java-ai-ml-course-research.md`
- Plan: `docs/ai-ml-implementation-plan.md`

### 3. Gamification (TODO)
- Current: `/server/src/users/users.service.ts` (streak, skillPoints, rank)
- Planned: XP system, levels, badges, achievements

---

## Session Notes

### Dec 20, 2024
- Implemented subscription system (backend + frontend)
- Created plan for AI/ML courses
- Analyzed current gamification state
- Need to: test subscriptions, plan AI/ML UI for charts, implement XP/levels/badges
