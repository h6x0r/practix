# Task Creation and Migration Guide

This guide defines the rules and templates for creating tasks in the KODLA learning platform.

## Table of Contents
1. [Task Structure](#task-structure)
2. [Translation Requirements](#translation-requirements)
3. [solutionCode Comments Pattern](#solutioncode-comments-pattern)
4. [Language Rules](#language-rules)
5. [Complete Example](#complete-example)

---

## Task Structure

Every task file must follow this TypeScript structure:

```typescript
import { Task } from '../../../../../../../types';

export const task: Task = {
    slug: 'unique-task-slug',           // URL-safe identifier, globally unique
    title: 'Task Title in English',     // Clear, descriptive title
    difficulty: 'easy' | 'medium' | 'hard',
    tags: ['language', 'topic', 'concept'],
    estimatedTime: '20m',               // Realistic time estimate
    isPremium: false,                   // Premium content flag
    youtubeUrl: '',                     // Optional video URL
    description: `Full task description...`,
    initialCode: `// Starter code...`,
    solutionCode: `// Solution with line-by-line comments...`,
    hint1: `First hint - general direction...`,
    hint2: `Second hint - more specific...`,
    whyItMatters: `Real-world importance explanation...`,
    order: 0,                           // Order within topic
    translations: {
        ru: { /* Russian translations */ },
        uz: { /* Uzbek translations */ }
    }
};
```

---

## Translation Requirements

### Rule 1: 100% Complete Translations
Every translation must be **100% complete**. No field should be empty or abbreviated.

```typescript
translations: {
    ru: {
        title: 'Полный перевод заголовка',
        description: `Полное описание задачи...`, // FULL, not shortened
        hint1: `Полная первая подсказка...`,
        hint2: `Полная вторая подсказка...`,
        whyItMatters: `Полное объяснение важности...` // FULL, not abbreviated
    },
    uz: {
        title: `To'liq sarlavha tarjimasi`,
        description: `Vazifaning to'liq tavsifi...`,
        hint1: `To'liq birinchi maslahat...`,
        hint2: `To'liq ikkinchi maslahat...`,
        whyItMatters: `Ahamiyatining to'liq tushuntirishi...`
    }
}
```

### Rule 2: Preserve Structure
Translations must preserve the **exact structure** of the original:
- Same sections and headers
- Same code examples (with translated comments)
- Same bullet points and numbered lists
- Same formatting (bold, code blocks, etc.)

### Rule 3: Translate ALL User-Facing Content
- Titles
- Descriptions
- Hints
- Why It Matters explanations
- Comments in code examples within descriptions
- Error messages in examples

---

## solutionCode Comments Pattern

### Rule: Line-by-Line Explanation
Every line of solutionCode must have a comment explaining:
1. **What** it does
2. **Why** it's needed
3. **Logic** behind the approach

### Pattern Examples:

```go
// GOOD - Clear line-by-line comments
func SafeDelete[M ~map[K]V, K comparable, V any](m M, keys []K) M {
    if m == nil {                                  // guard against nil map to prevent panic
        var zero M
        return zero                                // return zero value for nil input
    }
    toDelete := make(map[K]struct{}, len(keys))   // create set of keys to delete for O(1) lookup
    for _, key := range keys {                    // iterate over keys to mark for deletion
        toDelete[key] = struct{}{}                // add key to deletion set with zero-byte value
    }
    cloned := make(map[K]V, len(m))               // allocate new map with same capacity as original
    for key, value := range m {                   // iterate over original map entries
        if _, skip := toDelete[key]; skip {       // check if current key should be deleted
            continue                               // skip keys marked for deletion
        }
        cloned[key] = value                       // copy entry to new map
    }
    return M(cloned)                              // return new map with entries removed
}
```

```go
// BAD - Missing or vague comments
func SafeDelete[M ~map[K]V, K comparable, V any](m M, keys []K) M {
    if m == nil {
        var zero M
        return zero
    }
    toDelete := make(map[K]struct{}, len(keys))   // make map
    for _, key := range keys {
        toDelete[key] = struct{}{}
    }
    // ... rest without comments
}
```

### Comment Language Rule
- Comments in the **main** `solutionCode` field are in **English**
- **Translations** should include localized `solutionCode` with comments in the respective language:

```typescript
solutionCode: `package example

func Zero[T any]() T {
    var zero T    // declare variable with zero value
    return zero   // return zero-initialized value
}`,
translations: {
    ru: {
        // ... other fields ...
        solutionCode: `package example

func Zero[T any]() T {
    var zero T    // объявляем переменную с нулевым значением
    return zero   // возвращаем значение инициализированное нулём
}`
    },
    uz: {
        // ... other fields ...
        solutionCode: `package example

func Zero[T any]() T {
    var zero T    // nol qiymat bilan o'zgaruvchi e'lon qilamiz
    return zero   // nol bilan initsializatsiya qilingan qiymatni qaytaramiz
}`
    }
}
```

**Key rules for translated solutionCode:**
1. Code logic remains **identical** - only comments are translated
2. Variable names, function names, keywords stay in English
3. Every line should have a comment in the translation language
4. Preserve exact same formatting and indentation

---

## Language Rules

### Rule 1: No Language Mixing in User Content
All user-facing content must be in the selected language. No mixing allowed.

**WRONG (mixed):**
```typescript
// Russian with English mixed
description: `Реализуйте **CopyN** с semantics \`io.CopyN\`...`
whyItMatters: `...используйте io.Reader и io.Writer для interface composition...`
```

**CORRECT (pure Russian):**
```typescript
description: `Реализуйте **CopyN** с семантикой \`io.CopyN\`...`
whyItMatters: `...используйте io.Reader и io.Writer для композиции интерфейсов...`
```

### Rule 2: Code Identifiers Stay English
Variable names, function names, type names, and Go keywords remain in English:

```typescript
// Uzbek description with English code identifiers
description: `**ZeroInt(p *int)** funksiyasini amalga oshiring - ko'rsatkich bo'yicha qiymatni 0 ga o'rnating.`
```

### Rule 3: Technical Terms
Some technical terms can remain in English if:
- There's no established translation
- The English term is more recognizable
- It's a proper name (e.g., "Go", "Kubernetes")

Examples:
- "nil" → "nil" (Go keyword)
- "pointer" → "указатель" (RU) / "ko'rsatkich" (UZ)
- "interface" → "интерфейс" (RU) / "interfeys" (UZ)
- "map" → "map" (can keep) or "словарь/карта" (RU) / "lug'at/xarita" (UZ)

---

## Complete Example

Here's a complete task with proper translations:

```typescript
import { Task } from '../../../../../../../types';

export const task: Task = {
    slug: 'go-example-zero-value',
    title: 'Zero Value Initialization',
    difficulty: 'easy',
    tags: ['go', 'basics', 'initialization'],
    estimatedTime: '10m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a function that returns the zero value for any type.

**Requirements:**
1. Function takes no arguments
2. Returns the zero value of the generic type T

**Example:**
\`\`\`go
// Integer zero value
result := Zero[int]()
// result == 0

// String zero value
str := Zero[string]()
// str == ""
\`\`\``,
    initialCode: `package example

// TODO: Implement Zero
// Return the zero value of type T
func Zero[T any]() T {
    panic("TODO")
}`,
    solutionCode: `package example

func Zero[T any]() T {
    var zero T    // declare variable of type T with zero value
    return zero   // return the zero-initialized value
}`,
    hint1: `In Go, declaring a variable without initialization gives it the zero value.`,
    hint2: `Use \`var zero T\` to get the zero value, then return it.`,
    whyItMatters: `Understanding zero values is fundamental to Go programming.

**Why Zero Values Matter:**

1. **Memory Safety**: Go initializes all memory to zero, preventing undefined behavior.

2. **Default Behavior**: Zero values provide sensible defaults:
   - \`int\`: 0
   - \`string\`: ""
   - \`bool\`: false
   - \`pointer\`: nil

3. **Production Use**: Zero values enable patterns like optional fields and lazy initialization.`,
    order: 0,
    translations: {
        ru: {
            title: 'Инициализация нулевым значением',
            description: `Реализуйте функцию, которая возвращает нулевое значение для любого типа.

**Требования:**
1. Функция не принимает аргументов
2. Возвращает нулевое значение generic типа T

**Пример:**
\`\`\`go
// Нулевое значение для целого числа
result := Zero[int]()
// result == 0

// Нулевое значение для строки
str := Zero[string]()
// str == ""
\`\`\``,
            hint1: `В Go объявление переменной без инициализации даёт ей нулевое значение.`,
            hint2: `Используйте \`var zero T\` для получения нулевого значения, затем верните его.`,
            whyItMatters: `Понимание нулевых значений фундаментально для программирования на Go.

**Почему нулевые значения важны:**

1. **Безопасность памяти**: Go инициализирует всю память нулями, предотвращая неопределённое поведение.

2. **Поведение по умолчанию**: Нулевые значения предоставляют разумные значения по умолчанию:
   - \`int\`: 0
   - \`string\`: ""
   - \`bool\`: false
   - \`pointer\`: nil

3. **Production использование**: Нулевые значения позволяют использовать паттерны опциональных полей и ленивой инициализации.`
        },
        uz: {
            title: `Nol qiymat bilan initsializatsiya`,
            description: `Har qanday tur uchun nol qiymat qaytaruvchi funksiyani amalga oshiring.

**Talablar:**
1. Funksiya argumentlar qabul qilmaydi
2. Generic T turining nol qiymatini qaytaradi

**Misol:**
\`\`\`go
// Butun son uchun nol qiymat
result := Zero[int]()
// result == 0

// Satr uchun nol qiymat
str := Zero[string]()
// str == ""
\`\`\``,
            hint1: `Go'da o'zgaruvchini initsializatsiyasiz e'lon qilish unga nol qiymat beradi.`,
            hint2: `Nol qiymat olish uchun \`var zero T\` dan foydalaning, keyin uni qaytaring.`,
            whyItMatters: `Nol qiymatlarni tushunish Go dasturlash uchun asosiy hisoblanadi.

**Nol qiymatlar nima uchun muhim:**

1. **Xotira xavfsizligi**: Go barcha xotirani nolga initsializatsiya qiladi, noaniq xatti-harakatlarni oldini oladi.

2. **Standart xatti-harakat**: Nol qiymatlar oqilona standart qiymatlarni ta'minlaydi:
   - \`int\`: 0
   - \`string\`: ""
   - \`bool\`: false
   - \`pointer\`: nil

3. **Production foydalanish**: Nol qiymatlar ixtiyoriy maydonlar va dangasa initsializatsiya patternlaridan foydalanishga imkon beradi.`
        }
    }
};
```

---

## Translation Formatting Rules

### Rule 1: Use Backticks for Code References
ALWAYS use backticks (\`) for inline code in descriptions, NEVER single quotes (').

**WRONG:**
```typescript
// Uzbek with single quotes - WRONG
description: `**Talablar:**
1. 'SafeDelete[M ~map[K]V](m M, keys []K) M' funksiyasini yarating`
```

**CORRECT:**
```typescript
// Uzbek with backticks - CORRECT
description: `**Talablar:**
1. \`SafeDelete[M ~map[K]V](m M, keys []K) M\` funksiyasini yarating`
```

### Rule 2: Proper Line Breaks in Numbered Lists
Each numbered item in a list must be on its own line with proper spacing.

**WRONG:**
```typescript
// Items merged together - WRONG
description: `**Talablar:**
1. \`FunctionName\` funksiyasini yarating 2. nil ni ishlang 3. Yangi map qaytaring`
```

**CORRECT:**
```typescript
// Each item on separate line - CORRECT
description: `**Talablar:**
1. \`FunctionName\` funksiyasini yarating
2. nil ni ishlang
3. Yangi map qaytaring`
```

### Rule 3: Consistent Formatting Across Languages
The Uzbek translation must have EXACTLY the same structure as English and Russian:
- Same number of list items
- Same headers and sections
- Same code blocks
- Same formatting (bold, backticks, etc.)

**Checklist for Uzbek translations:**
- [ ] All code references use backticks, not single quotes
- [ ] Numbered lists have proper line breaks
- [ ] Structure matches English/Russian versions
- [ ] All sections are translated completely

### Rule 4: Escape Backticks in TypeScript Strings
When using backticks inside template literals, escape them with backslash:

```typescript
description: `Implement \`FunctionName[T any]()\` that returns...`
//                      ^       ^    ^  ^  ^
//                      These are all escaped backticks
```

---

## Checklist Before Committing

- [ ] `slug` is globally unique
- [ ] All fields have content (no empty strings)
- [ ] `solutionCode` has line-by-line comments
- [ ] Russian translation is 100% complete
- [ ] Uzbek translation is 100% complete
- [ ] Uzbek uses backticks (not single quotes) for code
- [ ] Numbered lists have proper line breaks
- [ ] No language mixing in translations
- [ ] Code examples work correctly
- [ ] Difficulty matches actual complexity
- [ ] `estimatedTime` is realistic
- [ ] `whyItMatters` explains real-world value
- [ ] Tags are relevant and consistent

---

## Common Uzbek Technical Terms

| English | Russian | Uzbek |
|---------|---------|-------|
| pointer | указатель | ko'rsatkich |
| function | функция | funksiya |
| interface | интерфейс | interfeys |
| struct | структура | struktura |
| method | метод | metod |
| variable | переменная | o'zgaruvchi |
| constant | константа | konstanta |
| slice | срез | kesim / slice |
| map | карта/словарь | lug'at / map |
| channel | канал | kanal |
| goroutine | горутина | gorutin |
| mutex | мьютекс | myuteks |
| buffer | буфер | bufer |
| error | ошибка | xato |
| validation | валидация | validatsiya |
| nil | nil | nil |
| panic | паника | panic |
| defer | defer | defer |
| context | контекст | kontekst |
| timeout | таймаут | taymaut |
| concurrent | конкурентный | parallel |
| synchronization | синхронизация | sinxronizatsiya |

---

## File Naming Convention

```
modules/
  module-name/
    module.ts           # Module definition
    index.ts            # Module exports
    topics/
      topic-name/
        topic.ts        # Topic definition
        index.ts        # Topic exports
        tasks/
          01-task-name.ts   # First task
          02-task-name.ts   # Second task
          index.ts          # Tasks export
```

Task file naming:
- Prefix with two-digit order number: `01-`, `02-`, etc.
- Use kebab-case for task name
- Keep names concise but descriptive

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12 | Initial version |
| 1.1 | 2025-12 | Added Translation Formatting Rules (backticks vs single quotes, line breaks) |
