# KODLA: Технический Долг и План Развития

> Дата: 2025-12-17

---

## Оглавление

1. [Текущее состояние](#текущее-состояние)
2. [Технический долг](#технический-долг)
3. [План: Dashboard](#dashboard)
4. [План: Roadmap (Персонализированный)](#roadmap)
5. [План: Analytics](#analytics)
6. [План: Платежи (Humo/UzCard/Visa/MC)](#платежи)
7. [Приоритеты и этапы](#приоритеты)

---

## Текущее состояние

### Что работает (Production Ready)
- **13 курсов** (~605 задач) с полной локализацией (EN/RU/UZ)
- **Piston Code Execution** - 8 языков
- **BullMQ Queue + Redis Caching**
- **Playground (Web IDE)** - /playground
- **Progress tracking** - из passed submissions
- **AI Tutor** - Gemini 2.0 Flash

### Что НЕ работает / Моки
| Компонент | Статус | Проблема |
|-----------|--------|----------|
| **Auth** | MOCK | Симуляция через localStorage |
| **Dashboard Stats** | MOCK | Hardcoded: 142 solved, 48h, #842 |
| **Analytics** | MOCK | Рандомные данные при каждой загрузке |
| **Roadmap** | MOCK | Только 2 шаблона, hardcoded |
| **Payments** | MOCK | Нет интеграции, `upgrade()` = mock |

---

## Технический Долг

### 1. Критические проблемы

#### A. Auth в режиме симуляции
**Файл:** `src/features/auth/api/authService.ts`
```typescript
// Строка 22-26: Явно указано
// Currently operating in Simulation Mode using Repository pattern.
// To switch to Real API, implement the api.post calls instead of repository calls.
```

**Решение:**
1. Бэкенд уже имеет JWT auth (`server/src/auth/`)
2. Нужно подключить реальные endpoints:
   - `POST /auth/login`
   - `POST /auth/register`
   - `GET /auth/profile`

#### B. TypeScript `any` типы (14 мест)

| Файл | Проблема |
|------|----------|
| `src/lib/storage.ts` | 6 instances - функции с `any` |
| `src/lib/api/client.ts` | 4 instances - ApiError, body |
| `src/pages/Settings.tsx` | Toggle props: `any` |
| `src/features/*/api/*.ts` | catch (error: any) |

**Решение:** Создать proper types в `src/types/`:
```typescript
// src/types/storage.ts
interface TimerState { ... }
interface RoadmapPrefs { role: string; level: string; goal: string; }

// src/types/api.ts
interface ApiError { message: string; status: number; data?: unknown; }
```

#### C. Console.log в production
**Файл:** `src/lib/sentry.ts:52`
```typescript
console.log('[Sentry] Initialized for environment:', import.meta.env.MODE);
```
**Решение:** Удалить или использовать conditional logging.

### 2. Архитектурные проблемы

#### A. Дублирование слоёв
```
features/roadmap/
├── data/repository.ts    # Mock data
├── api/roadmapService.ts # Вызывает repository
└── model/types.ts
```
**Проблема:** `api/` должен вызывать бэкенд, не repository.

**Решение:**
- `data/` - удалить после подключения API
- `api/` - реальные HTTP запросы
- Или оставить `data/` как fallback для offline mode

#### B. Hardcoded значения

| Файл | Значение |
|------|----------|
| Dashboard | `chartData`, stats (142, 48h, #842) |
| Analytics | 1,240 submissions, 342 streak |
| Premium | $0, $19 |
| Playground | Monaco CDN URL |

### 3. Performance проблемы

#### A. Отсутствие мемоизации
```typescript
// src/pages/Dashboard.tsx:34-42
const chartData = [...]; // Создаётся при каждом render
```

**Компоненты без React.memo:**
- TaskDescriptionPanel
- CodeEditorPanel (Monaco - дорого!)
- HintsPanel
- CoursesPage

#### B. Тяжёлые импорты
- Monaco Editor - lazy load
- Recharts - dynamic import
- SolutionExplanationTab (600+ строк)

---

## Dashboard

### Текущее состояние
- ✅ Курсы загружаются из API
- ✅ Recent tasks загружаются
- ❌ Stats hardcoded (142 solved, 48h, #842)
- ❌ Chart data hardcoded
- ❌ Streak hardcoded (12 days)

### План реализации

#### Backend endpoints (новые)
```typescript
// GET /users/me/stats
interface UserStats {
  totalSolved: number;
  hoursSpent: number;
  globalRank: number;
  skillPoints: number;
  currentStreak: number;
  maxStreak: number;
}

// GET /users/me/activity?days=7
interface DailyActivity {
  date: string; // ISO
  tasksCompleted: number;
  minutesSpent: number;
}[]
```

#### Frontend изменения
```typescript
// src/pages/Dashboard.tsx
useEffect(() => {
  Promise.all([
    userService.getStats(),
    userService.getActivity(7),
    courseService.getAllCourses(),
    taskService.getRecentTasks()
  ]).then(([stats, activity, courses, tasks]) => {
    setStats(stats);
    setChartData(activity);
    setCourses(courses);
    setRecentTasks(tasks);
  });
}, []);
```

---

## Roadmap

### Текущее состояние
- ✅ UI Wizard (3 шага)
- ✅ Красивый результат с phases/steps
- ❌ Только 2 hardcoded шаблона (backend-go-mid, backend-go-senior)
- ❌ Java/Fullstack не работают
- ❌ Нет персонализации по истории пользователя

### Концепция нового Roadmap

#### 1. Расширенный Wizard (Квиз)

**Шаг 1: Направление**
```
- Backend (Go)
- Backend (Java)
- Fullstack (Go + React)
- Fullstack (Java + React)
- Algorithms & DS
- Software Engineering
```

**Шаг 2: Уровень (с Self-Assessment)**
```
Junior: "Я знаю основы синтаксиса"
  └── Тест: 5 простых вопросов по базе

Mid: "Я работаю с production кодом"
  └── Тест: 5 вопросов по concurrency, patterns

Senior: "Я проектирую системы"
  └── Тест: 5 вопросов по architecture
```

**Шаг 3: Цель**
```
- Найти работу (быстро, за 1-2 месяца)
- Повышение (глубокое изучение)
- Освоить конкретный навык (выбор модуля)
```

**Шаг 4: Доступное время**
```
- 30 мин/день (Relaxed - 6 месяцев)
- 1 час/день (Standard - 3 месяца)
- 2+ часа/день (Intensive - 1.5 месяца)
```

#### 2. Логика генерации Roadmap

**Вариант A: Rule-Based (Вшитые правила)**
```typescript
// Преимущества: Быстро, предсказуемо, дёшево
// Недостатки: Негибко, много кода

const generateRoadmap = (prefs: UserPrefs, history: UserHistory) => {
  const phases: Phase[] = [];

  // 1. Определить начальный уровень по тестам
  const assessedLevel = assessUserLevel(prefs.testResults);

  // 2. Выбрать курсы по направлению
  const baseCourses = COURSE_MAP[prefs.direction];

  // 3. Отфильтровать уже пройденное
  const remainingModules = filterCompleted(baseCourses, history);

  // 4. Упорядочить по dependencies
  const orderedModules = topologicalSort(remainingModules);

  // 5. Разбить на phases по времени
  return splitIntoPhases(orderedModules, prefs.dailyTime);
};
```

**Вариант B: AI-Assisted (LLM)**
```typescript
// Преимущества: Гибко, персонализировано
// Недостатки: Дорого (API calls), непредсказуемо

const generateRoadmapAI = async (prefs: UserPrefs, history: UserHistory) => {
  const prompt = `
    User profile:
    - Direction: ${prefs.direction}
    - Self-assessed level: ${prefs.level}
    - Test results: ${prefs.testResults}
    - Completed tasks: ${history.completedTasks}
    - Goal: ${prefs.goal}
    - Daily time: ${prefs.dailyTime}

    Available modules: ${JSON.stringify(ALL_MODULES)}

    Generate a personalized roadmap with phases...
  `;

  return await geminiService.generate(prompt);
};
```

**Рекомендация: Гибридный подход**
1. Rule-based для структуры (phases, order)
2. AI для персонализированных рекомендаций внутри phase
3. AI для объяснения "почему этот модуль"

#### 3. Backend Schema

```prisma
model UserRoadmap {
  id        String   @id @default(uuid())
  userId    String
  user      User     @relation(fields: [userId], references: [id])

  direction String   // backend-go, backend-java, fullstack, etc.
  level     String   // junior, mid, senior
  goal      String   // job, promo, skill
  dailyTime Int      // minutes

  phases    RoadmapPhase[]

  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
}

model RoadmapPhase {
  id          String   @id @default(uuid())
  roadmapId   String
  roadmap     UserRoadmap @relation(fields: [roadmapId], references: [id])

  title       String
  description String
  order       Int

  items       RoadmapItem[]
}

model RoadmapItem {
  id          String   @id @default(uuid())
  phaseId     String
  phase       RoadmapPhase @relation(fields: [phaseId], references: [id])

  type        String   // 'task' | 'module' | 'course' | 'external'
  targetId    String?  // ID задачи/модуля/курса
  externalUrl String?  // Для внешних ресурсов
  title       String
  description String?

  status      String   @default("pending") // pending, completed, skipped
  order       Int
}
```

#### 4. API Endpoints

```typescript
// POST /roadmap/generate
// Body: { direction, level, goal, dailyTime, testResults? }
// Returns: UserRoadmap with phases and items

// GET /roadmap/me
// Returns: Current user's roadmap or null

// PATCH /roadmap/items/:id
// Body: { status: 'completed' | 'skipped' }

// POST /roadmap/regenerate
// Regenerates roadmap based on current progress
```

---

## Analytics

### Текущее состояние
- ❌ Weekly chart - random data
- ❌ Heatmap - synthetic
- ❌ Stats (1,240, 342, 42) - hardcoded

### План реализации

#### Backend endpoints
```typescript
// GET /analytics/weekly?offset=0
interface WeeklyStats {
  data: { name: string; tasks: number; date: string }[];
  totalTasks: number;
}

// GET /analytics/yearly
interface YearlyContributions {
  data: { date: string; count: number; intensity: 0|1|2|3|4 }[];
  totalSubmissions: number;
  currentStreak: number;
  maxStreak: number;
}

// GET /analytics/overview
interface AnalyticsOverview {
  completionRate: number;  // % passed / total
  avgRuntime: string;      // Средний runtime submissions
  totalXP: number;         // Расчёт по сложности задач
}
```

#### Streak calculation (Backend)
```typescript
const calculateStreak = async (userId: string) => {
  const submissions = await prisma.submission.findMany({
    where: { userId, status: 'passed' },
    orderBy: { createdAt: 'desc' },
    select: { createdAt: true }
  });

  if (submissions.length === 0) return { current: 0, max: 0 };

  let currentStreak = 0;
  let maxStreak = 0;
  let tempStreak = 1;
  let lastDate = startOfDay(submissions[0].createdAt);

  // Check if streak is active (today or yesterday)
  const today = startOfDay(new Date());
  const yesterday = subDays(today, 1);
  if (lastDate >= yesterday) {
    currentStreak = 1;
  }

  for (let i = 1; i < submissions.length; i++) {
    const currentDate = startOfDay(submissions[i].createdAt);
    const diff = differenceInDays(lastDate, currentDate);

    if (diff === 1) {
      tempStreak++;
      if (currentStreak > 0) currentStreak++;
    } else if (diff > 1) {
      maxStreak = Math.max(maxStreak, tempStreak);
      tempStreak = 1;
      currentStreak = 0;
    }
    lastDate = currentDate;
  }

  maxStreak = Math.max(maxStreak, tempStreak);
  return { current: currentStreak, max: maxStreak };
};
```

---

## Платежи

### Платёжные системы Узбекистана

#### 1. Click (click.uz)
- **Популярность:** Высокая
- **Интеграция:** REST API
- **Комиссия:** ~1.5%
- **Документация:** https://docs.click.uz

#### 2. Payme (payme.uz)
- **Популярность:** Очень высокая
- **Интеграция:** Merchant API
- **Комиссия:** ~1.5-2%
- **Документация:** https://developer.payme.uz

#### 3. Uzum Bank (uzum.uz)
- **Карты:** Humo, UzCard
- **Интеграция:** REST API
- **Документация:** Через менеджера

#### 4. Международные карты (Visa/MC)
- **Stripe** - НЕ работает в Узбекистане напрямую
- **Через посредников:**
  - Fondy (fondy.eu) - работает с UZ
  - Paycom (paycom.uz) - поддерживает Visa/MC
  - LiqPay - ограниченно

### Рекомендуемая архитектура

```
┌─────────────────┐
│   Frontend      │
│   (React)       │
└────────┬────────┘
         │ 1. POST /payments/create
         ▼
┌─────────────────┐
│   Backend       │
│   (NestJS)      │
├─────────────────┤
│ PaymentService  │
│ - createOrder() │
│ - webhook()     │
│ - verify()      │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ Click │ │ Payme │
└───────┘ └───────┘
```

### Backend Schema

```prisma
model Subscription {
  id        String   @id @default(uuid())
  userId    String   @unique
  user      User     @relation(fields: [userId], references: [id])

  plan      String   // 'free' | 'pro' | 'team'
  status    String   // 'active' | 'cancelled' | 'expired'

  startDate DateTime
  endDate   DateTime

  payments  Payment[]
}

model Payment {
  id             String   @id @default(uuid())
  subscriptionId String?
  subscription   Subscription? @relation(fields: [subscriptionId], references: [id])
  userId         String

  amount         Int      // В тийинах (1 UZS = 100 тийин)
  currency       String   @default("UZS")
  provider       String   // 'click' | 'payme' | 'uzum' | 'stripe'

  externalId     String?  // ID транзакции в платёжной системе
  status         String   // 'pending' | 'completed' | 'failed' | 'refunded'

  metadata       Json?    // Дополнительные данные от провайдера

  createdAt      DateTime @default(now())
  completedAt    DateTime?
}
```

### API Endpoints

```typescript
// POST /payments/create
interface CreatePaymentRequest {
  provider: 'click' | 'payme' | 'uzum';
  plan: 'pro_monthly' | 'pro_yearly';
  returnUrl: string;
}
interface CreatePaymentResponse {
  paymentId: string;
  redirectUrl: string; // URL для редиректа на платёжную систему
}

// POST /payments/webhook/:provider
// Webhook от платёжной системы

// GET /payments/verify/:paymentId
// Проверка статуса платежа

// GET /payments/history
// История платежей пользователя
```

### Click Integration Example

```typescript
// server/src/payments/providers/click.service.ts
@Injectable()
export class ClickService {
  private readonly merchantId = process.env.CLICK_MERCHANT_ID;
  private readonly secretKey = process.env.CLICK_SECRET_KEY;

  async createPayment(amount: number, userId: string, returnUrl: string) {
    const orderId = generateOrderId();

    // Click требует подпись
    const signString = `${this.merchantId}${orderId}${amount}${this.secretKey}`;
    const sign = crypto.createHash('md5').update(signString).digest('hex');

    const params = new URLSearchParams({
      merchant_id: this.merchantId,
      amount: amount.toString(),
      transaction_param: orderId,
      return_url: returnUrl,
      sign: sign
    });

    return {
      orderId,
      redirectUrl: `https://my.click.uz/services/pay?${params.toString()}`
    };
  }

  async handleWebhook(data: ClickWebhookPayload) {
    // Verify signature
    // Update payment status
    // Activate subscription if success
  }
}
```

### Pricing (UZS)

```typescript
const PLANS = {
  pro_monthly: {
    priceUZS: 99_000,  // ~$8
    priceUSD: 8,
    duration: 30 // days
  },
  pro_yearly: {
    priceUZS: 799_000, // ~$64 (33% скидка)
    priceUSD: 64,
    duration: 365
  }
};
```

---

## Приоритеты и Этапы

### Фаза 1: Core Tech Debt (1-2 недели)
1. ✅ ~~Submissions tab - real data~~
2. [ ] Подключить реальную авторизацию (JWT)
3. [ ] Исправить TypeScript `any` типы
4. [ ] Удалить console.log из production
5. [ ] Добавить React.memo к тяжёлым компонентам

### Фаза 2: Dashboard & Analytics (1-2 недели)
1. [ ] Backend: User stats endpoint
2. [ ] Backend: Activity/streak endpoints
3. [ ] Frontend: Подключить Dashboard к real data
4. [ ] Frontend: Подключить Analytics к real data

### Фаза 3: Roadmap (2-3 недели)
1. [ ] Backend: Roadmap schema + endpoints
2. [ ] Backend: Rule-based generation logic
3. [ ] Frontend: Расширенный wizard (4 шага)
4. [ ] Frontend: Self-assessment mini-quiz
5. [ ] Интеграция с AI для персонализации (опционально)

### Фаза 4: Payments (2-3 недели)
1. [ ] Регистрация в Click/Payme
2. [ ] Backend: Payment service + webhooks
3. [ ] Frontend: Payment flow UI
4. [ ] Тестирование в sandbox
5. [ ] Production deploy

### Фаза 5: Polish
1. [ ] Lazy loading для Monaco
2. [ ] Code splitting для charts
3. [ ] Error boundaries
4. [ ] Performance audit
5. [ ] A11y audit

---

## Appendix: Курсы и их зависимости

```
go-basics ──────────────► go-concurrency ──► go-production
     │                          │
     └──► go-web-apis ──────────┘

java-core ──► java-modern ──► java-advanced

go-design-patterns (standalone)
java-design-patterns (standalone)

software-engineering (standalone, language-agnostic)

algo-fundamentals ──► algo-advanced
```

Эти зависимости используются для rule-based roadmap generation.

---

*Документ создан: 2025-12-17*
