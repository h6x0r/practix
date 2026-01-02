# План: Тестирование и Рефакторинг

## Цель
Покрыть кодовую базу тестами, затем безопасно провести рефакторинг с уверенностью что логика не сломана.

---

## Фаза 1: Backend Integration Tests

### 1.1 Controllers (API контракты)
- [ ] `AuthController` - login, register, logout, refresh
- [ ] `CoursesController` - getCourses, getCourseStructure
- [ ] `TasksController` - getTask, getTaskBySlug
- [ ] `SubmissionsController` - submit, runTests, runQuickTests, getHistory
- [ ] `UsersController` - getProfile, updateProfile, updatePreferences
- [ ] `SubscriptionsController` - getPlans, subscribe, cancel
- [ ] `RoadmapsController` - generate, getVariants, selectVariant
- [ ] `AdminController` - stats, users, submissions
- [ ] `GamificationController` - getProgress, getLeaderboard

### 1.2 Guards & Middleware
- [ ] `JwtAuthGuard` - token validation
- [ ] `AdminGuard` - admin access
- [ ] `PremiumGuard` - premium access

### Цель: 80%+ покрытие бэкенда

---

## Фаза 2: Frontend Unit Tests

### 2.1 Hooks
- [ ] `useTaskRunner` - code execution, submissions
- [ ] `useAiChat` - AI tutor interactions
- [ ] `useAuth` - authentication state
- [ ] `useTaskState` - task loading
- [ ] `useTaskNavigation` - navigation between tasks

### 2.2 Services
- [ ] `taskService` - API calls for tasks
- [ ] `authService` - auth API calls
- [ ] `courseService` - courses API
- [ ] `subscriptionService` - subscriptions API

### 2.3 Contexts
- [ ] `AuthContext` - auth state management
- [ ] `SubscriptionContext` - subscription state
- [ ] `LanguageContext` - i18n

### 2.4 Components (critical)
- [ ] `CodeEditorPanel` - editor functionality
- [ ] `TaskDescriptionPanel` - task display
- [ ] `AiTutorPanel` - AI chat
- [ ] `WorkspaceHeader` - navigation

### Цель: 70%+ покрытие фронтенда

---

## Фаза 3: E2E Tests (Playwright)

### 3.1 Critical User Flows
- [ ] **Auth Flow**: Register → Login → Logout
- [ ] **Task Flow**: Browse → Select Task → Write Code → Run Tests → Submit
- [ ] **Course Flow**: View Courses → Start Course → Track Progress
- [ ] **Subscription Flow**: View Plans → Check Access Control

### 3.2 Edge Cases
- [ ] Unauthorized access attempts
- [ ] Invalid submissions
- [ ] Network error handling

### Цель: Все критические пути покрыты

---

## Фаза 4: Рефакторинг (ПОСЛЕ тестов)

### 4.1 Backend Refactoring

#### High Priority
- [ ] **Разбить `SubmissionsService`** (600+ строк)
  - Извлечь `TestParsingService`
  - Извлечь `TestValidationService`
  - Оставить core submission logic

- [ ] **Добавить Repository Layer**
  - `TaskRepository`
  - `SubmissionRepository`
  - `CourseRepository`
  - Абстрагировать Prisma от сервисов

- [ ] **Устранить дублирование**
  - Task resolution logic → helper method
  - Language validation → shared validator

#### Medium Priority
- [ ] Стандартизировать error handling
- [ ] Добавить pagination на все list endpoints
- [ ] Добавить structured logging

### 4.2 Frontend Refactoring

#### High Priority
- [ ] **Разбить `TaskWorkspace`** (200+ строк)
  - Извлечь `<MobileNavigation />`
  - Извлечь `<TaskHeader />`
  - Извлечь keyboard shortcuts hook

- [ ] **Разбить `useTaskRunner`** (325 строк)
  - `useTaskCode()` - code state
  - `useTaskSubmissions()` - history
  - `useTaskExecution()` - run/submit

- [ ] **Упростить Contexts**
  - Вынести логику из `SubscriptionContext` в сервис
  - Оставить только state management

#### Medium Priority
- [ ] Добавить Zod валидацию API responses
- [ ] Консолидировать типы (TaskAccess и др.)
- [ ] Добавить request deduplication

---

## Метрики успеха

| Метрика | Текущее | Цель |
|---------|---------|------|
| Backend coverage | 62% | 80%+ |
| Frontend coverage | 0% | 70%+ |
| E2E critical paths | 0% | 100% |
| Largest service | 600 lines | <200 lines |
| Largest component | 200 lines | <100 lines |

---

## Порядок выполнения

```
Week 1-2: Backend Integration Tests
    ↓
Week 3-4: Frontend Unit Tests
    ↓
Week 5: E2E Tests
    ↓
Week 6-8: Refactoring (с уверенностью!)
```

---

## Важно!

⚠️ **НЕ начинать рефакторинг до завершения тестов!**

Тесты = страховочная сетка. Без них рефакторинг рискован.

После каждого изменения в рефакторинге:
1. Запустить все тесты
2. Убедиться что все проходят
3. Только тогда commit

---

*Создано: 2026-01-01*
*Статус: В процессе - Фаза 1*
