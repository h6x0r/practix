# Task Migration & Localization Guide

This document describes how to add, modify, and localize tasks in the kodla-starter platform.

## Architecture Overview

```
server/prisma/seeds/
├── go-concurrency.ts    # Go concurrency tasks
├── java-tasks.ts        # Java tasks (placeholder)
├── algorithms.ts        # Algorithm tasks (placeholder)
└── seed.ts              # Main seeder
```

## Task Schema

```typescript
interface Task {
  slug: string;              // Unique identifier (e.g., 'go-ctx-timeout')
  title: string;             // English title
  difficulty: 'easy' | 'medium' | 'hard';
  tags: string[];            // ['go', 'context', 'concurrency']
  estimatedTime: string;     // '15m', '30m', '1h'
  isPremium: boolean;        // Premium-only content
  description: string;       // Markdown description with examples
  initialCode: string;       // Starter template
  solutionCode: string;      // Solution with inline comments
  solutionExplanation: null; // DEPRECATED - use inline comments
  whyItMatters: string;      // Real-world relevance
  hint1: string;
  hint2: string;
  order: number;
  translations: {
    ru: TranslatedFields;
    uz: TranslatedFields;
  }
}

interface TranslatedFields {
  title: string;
  description: string;
  hint1: string;
  hint2: string;
  solutionCode: string;      // Same code, localized comments
  whyItMatters: string;
}
```

## Inline Comment Style Guide

### English Comments
```go
func DoWithTimeout(ctx context.Context, f func(), d time.Duration) error {
    if ctx == nil {                                         // Handle nil context safely
        ctx = context.Background()                          // Use Background as default
    }
    ctxWithTimeout, cancel := context.WithTimeout(ctx, d)   // Create context with timeout
    defer cancel()                                          // MUST call to release timer resources
}
```

### Russian Comments
```go
func DoWithTimeout(ctx context.Context, f func(), d time.Duration) error {
    if ctx == nil {                                         // Безопасная обработка nil контекста
        ctx = context.Background()                          // Используем Background как значение по умолчанию
    }
    ctxWithTimeout, cancel := context.WithTimeout(ctx, d)   // Создаём контекст с таймаутом
    defer cancel()                                          // ОБЯЗАТЕЛЬНО вызываем для освобождения таймера
}
```

### Uzbek Comments
```go
func DoWithTimeout(ctx context.Context, f func(), d time.Duration) error {
    if ctx == nil {                                         // nil contextni xavfsiz qayta ishlash
        ctx = context.Background()                          // Standart sifatida Background dan foydalaning
    }
    ctxWithTimeout, cancel := context.WithTimeout(ctx, d)   // Timeout bilan context yarating
    defer cancel()                                          // Timer resurslarini bo'shatish uchun ALBATTA chaqiring
}
```

## Description Template

### English
```markdown
Brief description of what the function should do.

Why this matters in real-world scenarios.

**Requirements:**
- First requirement
- Second requirement
- Third requirement

**Example:**
\`\`\`go
result := FunctionName(args)
// Expected: description of expected output
\`\`\`

**Constraints:**
- Constraint 1
- Constraint 2
```

### Russian
```markdown
Краткое описание того, что должна делать функция.

Почему это важно в реальных сценариях.

**Требования:**
- Первое требование
- Второе требование

**Пример:**
\`\`\`go
result := FunctionName(args)
// Ожидается: описание ожидаемого результата
\`\`\`

**Ограничения:**
- Ограничение 1
- Ограничение 2
```

### Uzbek
```markdown
Funktsiya nima qilishi kerakligi haqida qisqacha tavsif.

Haqiqiy stsenariylarda nima uchun bu muhim.

**Talablar:**
- Birinchi talab
- Ikkinchi talab

**Misol:**
\`\`\`go
result := FunctionName(args)
// Kutilgan: kutilayotgan natija tavsifi
\`\`\`

**Cheklovlar:**
- Cheklov 1
- Cheklov 2
```

## Adding a New Module

1. Create a new file in `server/prisma/seeds/`:

```typescript
// server/prisma/seeds/new-module.ts
export const NEW_MODULE = [
  {
    title: 'Module Title',
    description: 'Module description',
    section: 'core',
    order: 1,
    translations: {
      ru: { title: '...', description: '...' },
      uz: { title: '...', description: '...' }
    },
    topics: [
      {
        title: 'Topic Title',
        // ... topic structure
        tasks: [
          // ... task objects
        ]
      }
    ]
  }
];
```

2. Import and use in `seed.ts`:

```typescript
import { NEW_MODULE } from './new-module';

// In seed function:
await createModule(prisma, course.id, NEW_MODULE);
```

## Database Operations

### Reset and Reseed
```bash
# Drop and recreate schema
docker exec kodla_postgres psql -U kodla_user -d kodla_db -c \
  "DROP SCHEMA public CASCADE; CREATE SCHEMA public; GRANT ALL ON SCHEMA public TO kodla_user;"

# Rebuild backend (includes new seed data)
docker compose build backend --no-cache

# Start services
docker compose up -d

# Monitor seed progress
docker logs -f kodla-starter-backend-1
```

### Verify Task Data
```bash
# Check task via API
curl -s http://localhost:8080/tasks/task-slug | jq .

# Check specific fields
curl -s http://localhost:8080/tasks/task-slug | jq '.translations.uz.solutionCode'
```

## Frontend Integration

### Translatable Fields
Fields automatically translated based on user language:
- `title`
- `description`
- `hint1`, `hint2`
- `solutionCode`
- `whyItMatters`

Configured in `src/contexts/LanguageContext.tsx`:
```typescript
const TRANSLATABLE_FIELDS = [
  'title',
  'description',
  'hint1',
  'hint2',
  'solutionExplanation',
  'solutionCode',
  'whyItMatters',
];
```

### UI Translation Keys
Add new UI strings to `UI_TRANSLATIONS` in `LanguageContext.tsx`:

```typescript
const UI_TRANSLATIONS = {
  en: {
    'task.newKey': 'English text',
  },
  ru: {
    'task.newKey': 'Русский текст',
  },
  uz: {
    'task.newKey': 'O\'zbek matni',
  },
};
```

## Common Issues

### Task not appearing after seed
- Check Docker logs for seed errors
- Verify task slug is unique
- Ensure module/topic structure is correct

### Translation not showing
- Verify field is in TRANSLATABLE_FIELDS
- Check translations object structure
- Confirm language code matches ('ru', 'uz')

### Comments misaligned
- Use tabs for Go code indentation
- Align comments at column 50-55
- Test display in SolutionExplanationTab

## Checklist for New Task

- [ ] Unique slug
- [ ] All required fields present
- [ ] English description with Example section
- [ ] solutionCode with inline comments
- [ ] whyItMatters explains real-world relevance
- [ ] RU translation complete with solutionCode
- [ ] UZ translation complete with solutionCode
- [ ] Database reseeded
- [ ] API returns correct data
- [ ] UI displays properly in all languages
