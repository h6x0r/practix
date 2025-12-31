# План: Улучшение фильтрации курсов и URL slug-ов

**Статус**: 🟡 В процессе (Часть 1-2 завершены, Часть 3 ожидает)

**Дата начала**: 2024-12-24
**Последнее обновление**: 2024-12-24

---

## Обзор задач

| # | Задача | Статус |
|---|--------|--------|
| 1 | Система фильтрации (grouped chips UI) | ✅ Завершено |
| 2 | Переименование slug-ов в kebab-case | ✅ Завершено |
| 3 | Добавление ML курсов в ALL_COURSES | ⏳ Ожидает аудита |

---

## Часть 1: Система фильтрации ✅ ЗАВЕРШЕНО

### Было
- Фильтры: `all | go | java | algo_ds | design_patterns`
- UI: Простые pills в ряд
- Проблема: Python, SE, ML курсы не имели фильтров

### Стало
```
┌──────────────────────────────────────────────────────────────────────┐
│ [All Tracks]  │  LANGUAGES       │ CS FUNDAMENTALS  │ APPLIED       │
│               │  🐹 Go           │ 🧮 Algo & DS     │ 🤖 ML/AI      │
│               │  ☕ Java         │ 🏗 Patterns & SE │               │
│               │  🐍 Python       │                  │               │
└──────────────────────────────────────────────────────────────────────┘
```

### Изменённые файлы
- `src/features/courses/ui/CoursesPage.tsx` — новый UI + filterCourse() + getCourseBadge()
- `src/utils/themeUtils.ts` — getCourseTheme() для kebab-case slug-ов
- `src/contexts/LanguageContext.tsx` — переводы EN/RU/UZ для новых фильтров

### Новые ключи переводов
```
courses.groupLanguages, courses.groupCsFundamentals, courses.groupApplied
courses.filterPatternsSE, courses.filterMlAi
courses.softwareEng, courses.mlAi, courses.algoDs
```

---

## Часть 2: Переименование slug-ов ✅ ЗАВЕРШЕНО

### Обновлённые курсы (12 активных)

| Было | Стало | Статус |
|------|-------|--------|
| `c_go_basics` | `go-basics` | ✅ |
| `c_go_concurrency` | `go-concurrency` | ✅ |
| `c_go_web_apis` | `go-web-apis` | ✅ |
| `c_go_production` | `go-production` | ✅ |
| `c_go_design_patterns` | `go-design-patterns` | ✅ |
| `c_java_core` | `java-core` | ✅ |
| `c_java_modern` | `java-modern` | ✅ |
| `c_java_advanced` | `java-advanced` | ✅ |
| `c_java_design_patterns` | `java-design-patterns` | ✅ |
| `c_python_ml_fundamentals` | `python-ml-fundamentals` | ✅ |
| `c_software_engineering` | `software-engineering` | ✅ |
| `c_algo_fundamentals` | `algo-fundamentals` | ✅ |

### Изменённые файлы (slug в course.ts)
```
server/prisma/seeds/courses/go-basics/course.ts
server/prisma/seeds/courses/go-concurrency/course.ts
server/prisma/seeds/courses/go-web-apis/course.ts
server/prisma/seeds/courses/go-production/course.ts
server/prisma/seeds/courses/go-design-patterns/course.ts
server/prisma/seeds/courses/java-core/course.ts
server/prisma/seeds/courses/java-modern/course.ts
server/prisma/seeds/courses/java-advanced/course.ts
server/prisma/seeds/courses/java-design-patterns/course.ts
server/prisma/seeds/courses/c_python_ml_fundamentals/course.ts
server/prisma/seeds/courses/software-engineering/course.ts
server/prisma/seeds/courses/algo-fundamentals/course.ts
```

---

## Часть 3: Добавление ML курсов ⏳ ОЖИДАЕТ

### Курсы для добавления в ALL_COURSES

| Курс | Директория | Slug (обновлён) | Аудит |
|------|------------|-----------------|-------|
| Python Deep Learning | `c_python_deep_learning` | `python-deep-learning` | ⏳ Требуется |
| Python LLM | `c_python_llm` | `python-llm` | ⏳ Требуется |
| Java ML | `c_java_ml` | `java-ml` | ⏳ Требуется |
| Java NLP | `c_java_nlp` | `java-nlp` | ⏳ Требуется |
| Go ML Inference | `c_go_ml_inference` | `go-ml-inference` | ⏳ Требуется |

### Что проверить при аудите каждого курса

1. **Структура курса**
   - [ ] Корректный `course.ts` с метаданными и переводами
   - [ ] Все модули имеют `module.ts` и `topics/index.ts`
   - [ ] Правильный порядок (order) модулей и задач

2. **Задачи (tasks)**
   - [ ] Каждая задача имеет `slug`, `title`, `description`
   - [ ] `initialCode` — стартовый код для студента
   - [ ] `solutionCode` — эталонное решение
   - [ ] `testCode` — тесты для проверки
   - [ ] `hints` — подсказки (hint1, hint2)
   - [ ] `whyItMatters` — объяснение важности
   - [ ] `translations` — RU и UZ переводы

3. **Качество контента**
   - [ ] initialCode не содержит решения
   - [ ] solutionCode компилируется/работает
   - [ ] testCode корректно проверяет решение
   - [ ] Описания понятны и информативны

### Шаги для добавления курса

1. Провести аудит курса по чеклисту выше
2. Исправить найденные проблемы
3. Добавить импорт в `server/prisma/seeds/courses/index.ts`
4. Добавить в массив `ALL_COURSES`
5. Пересидить базу данных
6. Проверить в UI

### Файл для изменения
```typescript
// server/prisma/seeds/courses/index.ts

// Добавить импорты:
import pythonDeepLearning from './c_python_deep_learning';
import pythonLlm from './c_python_llm';
import javaMl from './c_java_ml';
import javaNlp from './c_java_nlp';
import goMlInference from './c_go_ml_inference';

// Добавить в ALL_COURSES:
export const ALL_COURSES = [
  // ... существующие курсы ...

  // ML/AI Courses
  pythonDeepLearning,
  pythonLlm,
  javaMl,
  javaNlp,
  goMlInference,
];
```

---

## Результаты (текущее состояние)

### API Response: GET /courses
```json
[
  "go-basics",
  "go-concurrency",
  "go-web-apis",
  "go-production",
  "go-design-patterns",
  "java-core",
  "java-modern",
  "java-advanced",
  "java-design-patterns",
  "python-ml-fundamentals",
  "software-engineering",
  "algo-fundamentals"
]
```

### Фильтры работают корректно
- **Go**: 5 курсов (go-*)
- **Java**: 4 курса (java-*)
- **Python**: 1 курс (python-ml-fundamentals)
- **Algo & DS**: 1 курс (algo-fundamentals)
- **Patterns & SE**: 3 курса (go-design-patterns, java-design-patterns, software-engineering)
- **ML/AI**: 1 курс (python-ml-fundamentals) — расширится после добавления ML курсов

---

## Заметки

- Директории курсов всё ещё имеют старые имена (`c_python_ml_fundamentals`), но это не влияет на работу — важен только `slug` в `course.ts`
- После добавления ML курсов фильтр "ML/AI" покажет все 6 курсов
- Data Engineering и Prompt Engineering фильтры можно добавить позже как disabled, когда появятся соответствующие курсы
