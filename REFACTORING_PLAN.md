# Refactoring Plan

> Last Updated: 2026-02-14

---

## Completed

### ✅ RoadmapsService Configs (2026-01-16)
- Created `server/src/roadmaps/roadmap.config.ts`
- Extracted: SALARY_RANGES, COURSE_ICONS, PHASE_PALETTES, CATEGORY_PATTERNS
- AI model now configurable via `AI_MODEL_NAME` env variable

### ✅ AiService Configs (2026-01-16)
- Created `server/src/ai/ai.config.ts`
- Extracted: DEFAULT_AI_MODEL, AI_DAILY_LIMITS
- AI model uses env variable with fallback to default

### ✅ Dynamic Resize Panels (2026-01-16)
- Updated `useResizablePanel` with `maxWidthRatio` support
- Left panel now respects 55% max viewport width
- Fixes issues when switching between monitors

### ✅ AdminService Decomposition (2026-02-14)
- Split `admin.service.ts` (1744 lines) → 5 sub-services + facade
- Created: `AdminStatsService`, `AdminMetricsService`, `AdminRetentionService`, `AdminUsersService`, `AdminPaymentsService`
- `AdminService` remains as facade, delegating to sub-services
- All 64 admin tests passing

### ✅ RoadmapsService Decomposition (2026-02-14)
- Split `roadmaps.service.ts` (1090 lines) → 4 files
- Created: `roadmap.types.ts` (90), `roadmap-ai.service.ts` (314), `roadmap-variants.service.ts` (158)
- `RoadmapsService` (429 lines) remains as facade
- All 69 roadmap tests passing

### ✅ Judge0Service Decomposition (2026-02-14)
- Split `judge0.service.ts` (995 lines) → 3 files
- Created: `judge0.types.ts` (86), `judge0-test-builders.ts` (475)
- `Judge0Service` (335 lines) — core service with re-exports
- All 23 judge0 tests passing

### ✅ PaymentsPage Decomposition (2026-02-14)
- Split `PaymentsPage.tsx` (823 lines) → 4 components
- Created: `StatusCard.tsx` (67), `CheckoutPanel.tsx` (263), `PaymentHistoryTab.tsx` (95)
- `PaymentsPage.tsx` (450 lines) — main page with sub-components
- All data-testid attributes preserved, 41 frontend tests passing

### ✅ TypeScript Strict Mode on Backend (2026-02-14)
- Enabled `strict: true` in `server/tsconfig.json`
- Fixed 245+ type errors across the codebase
- Found real bug in `snippets.controller.ts` (missing null check)
- `strictPropertyInitialization: false` for NestJS DI compatibility

---

## Priority: HIGH

### 1. LanguageContext.tsx (1,937 lines)
**Location:** `src/contexts/LanguageContext.tsx`
**Status:** 🔴 Not Started

**Problem:** Monolithic context containing translations, translation logic, and UI state management all in one file.

**Proposed Solution:**
- Extract translation data to separate JSON files per language (`locales/en.json`, `locales/ru.json`, `locales/uz.json`)
- Split into smaller contexts:
  - `TranslationContext` - for i18n functionality
  - `UILanguageContext` - for UI language preferences
- Use a library like `react-i18next` or create a simpler custom solution

**Estimated Impact:** High - improves maintainability, enables lazy loading of translations

---

### 2. SubmissionsService (729 lines)
**Location:** `server/src/submissions/submissions.service.ts`
**Status:** 🟡 Partially Done (TestParserService extracted)

**Problem:** God class handling code execution, test parsing, scoring, gamification updates, and queue management.

**Proposed Solution:**
- Extract remaining concerns:
  - `CodeExecutionService` - handles Piston API calls
  - `ScoringService` - calculates scores and determines pass/fail
  - `SubmissionHistoryService` - manages submission records
- Use facade pattern to maintain backward compatibility

**Estimated Impact:** High - reduces coupling, enables better testing

---

### 3. RoadmapPage.tsx (956 lines)
**Location:** `src/features/roadmap/ui/RoadmapPage.tsx`
**Status:** 🔴 Not Started

**Problem:** Large component with mixed concerns - data fetching, rendering, and business logic.

**Proposed Solution:**
- Extract custom hooks:
  - `useRoadmapData()` - data fetching and caching
  - `useRoadmapWizard()` - wizard state management
  - `useRoadmapProgress()` - progress calculation
- Extract sub-components:
  - `RoadmapWizard` - wizard UI
  - `RoadmapVariantSelector` - variant selection
  - `RoadmapPhaseList` - phase display
  - `RoadmapStepCard` - individual step card

**Estimated Impact:** Medium - improves readability and reusability

---

### 4. ~~RoadmapsService~~ ✅ DONE (2026-02-14)
**Location:** `server/src/roadmaps/roadmaps.service.ts`
**Status:** ✅ Completed — see "Completed" section above

Split into: `roadmap.types.ts`, `roadmap-ai.service.ts`, `roadmap-variants.service.ts` + facade (429 lines)

---

## Priority: MEDIUM

### 5. useTaskRunner.ts (392 lines)
**Location:** `src/features/tasks/model/useTaskRunner.ts`
**Status:** 🔴 Not Started

**Problem:** Complex hook handling code execution, submission, cooldowns, and results management.

**Proposed Solution:**
- Extract into smaller hooks:
  - `useCodeExecution()` - handles run/submit API calls
  - `useCooldown()` - manages rate limiting UI
  - `useSubmissionHistory()` - fetches and manages past submissions
- Create a higher-order hook that composes these

**Estimated Impact:** Medium - improves testability and reusability

---

### 6. Feature Structure Standardization
**Status:** 🟡 Analysis Done

**Current State:**
| Feature | api | model | ui | Notes |
|---------|-----|-------|-----|-------|
| admin | ✅ | ❌ | ✅ | Add model/ if types needed |
| ai | ✅ | ❌ | ❌ | No UI needed |
| analytics | ✅ | ❌ | ✅ | Consider data/ → model/ |
| auth | ✅ | ✅ | ✅ | OK |
| config | ✅ | ✅ | ❌ | No UI needed |
| courses | ✅ | ✅ | ✅ | OK |
| dashboard | ✅ | ❌ | ✅ | Add model/ if types needed |
| gamification | ✅ | ❌ | ✅ | Add model/ if types needed |
| my-tasks | ❌ | ❌ | ✅ | Uses other features' APIs |
| payments | ✅ | ✅ | ✅ | OK |
| playground | ✅ | ❌ | ✅ | Add model/ if types needed |
| roadmap | ✅ | ✅ | ✅ | OK |
| settings | ❌ | ❌ | ✅ | Uses other features' APIs |
| subscriptions | ✅ | ✅ | ✅ | OK |
| tasks | ✅ | ✅ | ✅ | OK |

**Standard Structure:**
```
feature/
├── api/        # API services
├── model/      # Types, hooks, business logic
└── ui/         # React components
```

---

### 7. ~~Type Safety Improvements~~ ✅ DONE (2026-02-14)
**Status:** ✅ Completed — TypeScript strict mode enabled on backend

- `strict: true` in `server/tsconfig.json` (245+ errors fixed)
- Remaining: frontend strict mode, replace remaining `any` types, Zod validation at API boundaries

---

## Implementation Order

1. ✅ **Phase 1:** Fix UI issues, extract configs
2. ✅ **Phase 1.5:** Backend strict mode + monster file decomposition (admin, roadmaps, judge0, payments)
3. 🔴 **Phase 2:** Extract LanguageContext translations
4. 🔴 **Phase 3:** Refactor SubmissionsService (remaining)
5. 🔴 **Phase 4:** Split RoadmapPage component
6. 🔴 **Phase 5:** Refactor useTaskRunner hook

---

## Notes

- Each refactoring should be done incrementally with tests
- Maintain backward compatibility during transitions
- Large refactorings (LanguageContext, RoadmapPage) require dedicated sprints
- Document API changes in CHANGELOG.md
