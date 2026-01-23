# Task Creation Guide

> Last updated: 2026-01-04

## Table of Contents
1. [Complete Task Interface](#complete-task-interface)
2. [Translation Rules](#translation-rules)
3. [WhyItMatters Requirements](#whyitmatters-translation-requirements)
4. [Estimated Time Rules](#estimated-time-rules)
5. [Test Code Requirements](#test-code-requirements)
6. [Python ML Tasks](#python-ml-tasks)
7. [Task Quality Checklist](#task-quality-checklist)

---

## Complete Task Interface

Every task file must export a `Task` object with the following structure:

```typescript
import { Task } from '../../types';

export const task: Task = {
    // REQUIRED FIELDS
    slug: string,           // Unique identifier: 'module-topic-task-name'
    title: string,          // English title (descriptive)
    difficulty: 'easy' | 'medium' | 'hard',
    tags: string[],         // ['go', 'concurrency', 'channels']
    estimatedTime: string,  // '15m', '30m', '1h'
    isPremium: boolean,     // Free (false) or Premium (true)

    description: string,    // Markdown task description with examples
    initialCode: string,    // Starting code template with TODO
    solutionCode: string,   // Complete solution with English comments
    testCode: string,       // 10 unit tests (language-specific)

    hint1: string,          // Gentle hint (concept direction)
    hint2: string,          // More specific hint (code pattern)
    whyItMatters: string,   // Real-world relevance + production pattern

    order: number,          // Position within topic (0-indexed)

    // REQUIRED TRANSLATIONS
    translations: {
        ru: {
            title: string,
            solutionCode: string,   // Solution with Russian comments
            description: string,
            hint1: string,
            hint2: string,
            whyItMatters: string    // Includes "Продакшен паттерн:" section
        },
        uz: {
            title: string,
            solutionCode: string,   // Solution with Uzbek comments
            description: string,
            hint1: string,
            hint2: string,
            whyItMatters: string    // Includes "Ishlab chiqarish patterni:" section
        }
    },

    // OPTIONAL FIELDS
    youtubeUrl?: string,            // YouTube tutorial link
    visualizationType?: 'none' | 'line' | 'bar' | 'scatter' | 'heatmap'
};

export default task;
```

### Directory Structure

```
server/prisma/seeds/
├── types.ts                    # Task interface definition
├── courses/                    # Course definitions
│   └── {course-slug}/
│       ├── index.ts            # Course export
│       └── modules/{module}/
│           └── topics/{topic}/
│               └── tasks/
│                   ├── 01-task-name.ts
│                   ├── 02-task-name.ts
│                   └── index.ts
└── shared/
    └── modules/                # Reusable modules
        ├── go/                 # 25+ Go modules
        ├── java/               # 35+ Java modules
        └── python/             # Python modules
```

### Current Statistics

| Category | Count |
|----------|-------|
| Total Courses | 18 |
| Total Tasks | ~921 |
| Languages | Go, Java, Python, TypeScript |
| Translations | EN, RU, UZ |

---

## Translation Rules

### General Principles

1. **Don't translate literally** - Translations should sound natural in the target language
2. **Add context** - If a term is ambiguous, add clarifying words
3. **Keep technical terms** - Java/Go/Python class names, library names stay in English
4. **Be descriptive** - A title should explain WHAT the task does, not just name a concept

### Russian Translation Rules

#### DO:
- "Safe Delete" → "Безопасное удаление из слайса" (добавлен контекст)
- "Reverse In Place" → "Разворот массива без выделения памяти" (объяснено что делает)
- "Batch Processing" → "Пакетная обработка данных" (добавлен объект)
- "Unique" → "Удаление дубликатов" (описывает действие)
- "Linear Regression" → "Линейная регрессия" (ML термин)
- "Feature Scaling" → "Масштабирование признаков" (ML контекст)

#### DON'T:
- "Переворот на месте" (звучит как сальто)
- "Уникальные элементы" (не описывает что делает функция)
- "Пакетная обработка" (чего? непонятно)

#### Title Format:
- Используйте **отглагольные существительные**: "Удаление", "Создание", "Обработка"
- Добавляйте **объект действия**: "...из слайса", "...строк", "...данных"
- Указывайте **ключевую особенность**: "...без выделения памяти", "...с таймаутом"

### Uzbek Translation Rules

#### DO:
- "Safe Delete" → "Slaysdan xavfsiz o'chirish" (kontekst qo'shilgan)
- "Reverse In Place" → "Massivni joyida teskari aylantirish" (aniq)
- "K-Means Clustering" → "K-Means klasterlash" (ML termini)

#### DON'T:
- "Joyida teskari burish" (noaniq)
- "Noyob elementlar" (funksiya nimani qilishini ko'rsatmaydi)

### Technical Terms

Keep these in English:
- Class names: `ConcurrentHashMap`, `LocalDateTime`, `DataFrame`
- Library names: `numpy`, `pandas`, `sklearn`, `torch`
- Patterns: `Singleton`, `Factory`, `Worker Pool`
- ML terms: `XGBoost`, `LightGBM`, `LSTM`, `GRU`

Translate the explanation part:
- "DataFrame basics" → "Основы DataFrame" / "DataFrame asoslari"
- "K-Means clustering" → "Кластеризация K-Means" / "K-Means klasterlash"

### Examples by Module

#### Data Structures
| English | Russian | Uzbek |
|---------|---------|-------|
| Flatten Nested | Развёртывание вложенных слайсов | Ichma-ich slayslarni tekislash |
| Safe Delete | Безопасное удаление из слайса | Slaysdan xavfsiz o'chirish |
| Unique | Удаление дубликатов | Takroriylarni olib tashlash |

#### Machine Learning
| English | Russian | Uzbek |
|---------|---------|-------|
| Linear Regression | Линейная регрессия | Chiziqli regressiya |
| Decision Trees | Деревья решений | Qaror daraxtlari |
| Feature Scaling | Масштабирование признаков | Xususiyatlarni masshtablash |
| Cross-Validation | Кросс-валидация | Kross-validatsiya |
| Confusion Matrix | Матрица ошибок | Xatolar matritsasi |
| Overfitting | Переобучение | Haddan tashqari o'rganish |

---

## WhyItMatters Translation Requirements

### Production Pattern Section (REQUIRED)

Every `whyItMatters` field in translations MUST include a production pattern section at the end.

#### Russian (ru) Format:

```markdown
**Продакшен паттерн:**
\`\`\`python
# Пример кода с комментариями на русском
from sklearn.linear_model import LinearRegression

# Создание и обучение модели
model = LinearRegression()
model.fit(X_train, y_train)
\`\`\`

**Практические преимущества:**
- Преимущество 1
- Преимущество 2
- Преимущество 3
```

#### Uzbek (uz) Format:

```markdown
**Ishlab chiqarish patterni:**
\`\`\`python
# Misol kodi o'zbek tilidagi izohlar bilan
from sklearn.linear_model import LinearRegression

# Model yaratish va o'rgatish
model = LinearRegression()
model.fit(X_train, y_train)
\`\`\`

**Amaliy foydalari:**
- Foyda 1
- Foyda 2
- Foyda 3
```

### Common Issues to Avoid:

- Empty whyItMatters in translations
- Missing production pattern section
- Untranslated code comments in production examples
- Copy-paste English content into translations

---

## Estimated Time Rules

### Time Format

| Format | Example | Minutes |
|--------|---------|---------|
| Minutes | `'15m'`, `'30m'`, `'45m'` | 15, 30, 45 |
| Range | `'15-20m'` | Uses first number (15) |
| Hours | `'1h'`, `'1.5h'`, `'2h'` | 60, 90, 120 |

### Recommended Task Times

| Difficulty | Time Range | Description |
|------------|------------|-------------|
| Easy | 10-20m | Simple concepts, single function |
| Medium | 20-35m | Multiple steps, pattern application |
| Hard | 35-60m | Complex logic, multiple components |

### ML-Specific Time Guidelines

| Task Type | Difficulty | Time |
|-----------|------------|------|
| NumPy array creation | easy | 10-15m |
| Pandas data manipulation | medium | 15-25m |
| sklearn model training | medium | 20-30m |
| Visualization with data | medium | 20-25m |
| Full ML pipeline | hard | 30-45m |
| Hyperparameter tuning | hard | 25-35m |

---

## Test Code Requirements

### General Structure

Every task MUST have a `testCode` field with 10 test cases. The test runner automatically captures `expected` and `actual` values for display in the UI.

### Language-Specific Test Formats

| Language | Test Format | Expected/Actual Capture |
|----------|-------------|------------------------|
| Python | `unittest.TestCase` with `self.assert*` | Automatic via assertion parsing |
| Go | `func TestX(t *testing.T)` with `t.Error()` | Via comment `// TestN: description` |
| Java | `class TestN { void test() }` with `Assert.assertEquals()` | Automatic via Assert tracking |

---

### Python Test Template

```python
import unittest
import numpy as np

class Test(unittest.TestCase):
    def test_1(self):
        """Basic case"""
        result = create_array([1, 2, 3])
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_2(self):
        """Empty input"""
        result = create_array([])
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    def test_3(self):
        """Negative numbers"""
        result = create_array([-1, -2, -3])
        expected = np.array([-1, -2, -3])
        np.testing.assert_array_equal(result, expected)

    # ... 7 more tests

if __name__ == '__main__':
    unittest.main()
```

### NumPy/Pandas Test Assertions

```python
# NumPy arrays
np.testing.assert_array_equal(result, expected)
np.testing.assert_array_almost_equal(result, expected, decimal=5)
np.testing.assert_allclose(result, expected, rtol=1e-5)

# Pandas DataFrames
pd.testing.assert_frame_equal(result, expected)
pd.testing.assert_series_equal(result, expected)

# ML metrics (approximate)
self.assertAlmostEqual(accuracy, 0.95, places=2)
self.assertGreater(r2_score, 0.8)
```

---

### Go Test Template

Go tests use the standard `testing` package. For Input/Expected/Actual display in UI, use comments like `// TestN: description`.

```go
package main

import (
    "testing"
)

// Test1: Empty slice input
func Test1(t *testing.T) {
    result := YourFunction([]int{})
    expected := 0
    if result != expected {
        t.Errorf("Expected %d, got %d", expected, result)
    }
}

// Test2: Single element
func Test2(t *testing.T) {
    result := YourFunction([]int{5})
    expected := 5
    if result != expected {
        t.Errorf("Expected %d, got %d", expected, result)
    }
}

// Test3: Multiple elements
func Test3(t *testing.T) {
    result := YourFunction([]int{1, 2, 3, 4, 5})
    expected := 15
    if result != expected {
        t.Errorf("Expected %d, got %d", expected, result)
    }
}

// Test4: Negative numbers
func Test4(t *testing.T) {
    result := YourFunction([]int{-1, -2, -3})
    expected := -6
    if result != expected {
        t.Errorf("Expected %d, got %d", expected, result)
    }
}

// ... 6 more tests (total 10)
```

**Key points for Go tests:**
- Use `// TestN: description` comment BEFORE each test function for Input display
- Use `t.Errorf("Expected %v, got %v", expected, result)` format for clear error messages
- Expected/Actual values are extracted from the error message
- Test function names MUST match pattern `TestN` (Test1, Test2, etc.)

---

### Java Test Template

Java tests use a custom lightweight test framework. Each test is a class implementing `Testable` interface.

```java
// Test classes
class Test1 implements Testable {
    public void test() {
        // Test basic functionality
        Calculator calc = new Calculator();
        int result = calc.add(2, 3);
        Assert.assertEquals(5, result);
    }
}

class Test2 implements Testable {
    public void test() {
        // Test with zero
        Calculator calc = new Calculator();
        int result = calc.add(0, 5);
        Assert.assertEquals(5, result);
    }
}

class Test3 implements Testable {
    public void test() {
        // Test negative numbers
        Calculator calc = new Calculator();
        int result = calc.add(-1, -2);
        Assert.assertEquals(-3, result);
    }
}

class Test4 implements Testable {
    public void test() {
        // Test string result
        Greeter greeter = new Greeter();
        String result = greeter.hello("World");
        Assert.assertEquals("Hello, World!", result);
    }
}

// ... 6 more tests (total 10)
```

**Key points for Java tests:**
- Each test is a separate class: `class TestN implements Testable`
- Use `Assert.assertEquals(expected, actual)` for automatic value capture
- The test runner captures `lastExpected` and `lastActual` for UI display
- Do NOT add imports in testCode - they are handled by the test runner
- Supported assertions:
  - `Assert.assertEquals(Object expected, Object actual)`
  - `Assert.assertEquals(int expected, int actual)`
  - `Assert.assertTrue(boolean condition)`
  - `Assert.assertFalse(boolean condition)`

**Available Assert methods in Java tests:**

```java
// Object comparison (uses .equals())
Assert.assertEquals("expected string", result);

// Numeric comparison
Assert.assertEquals(42, result);

// Boolean checks
Assert.assertTrue(result > 0);
Assert.assertFalse(list.isEmpty());
```

---

### Test Runner Output Format

All language test runners produce JSON output in this format:

```json
{
  "tests": [
    {
      "name": "Test1",
      "passed": true,
      "expected": "5",
      "output": "5"
    },
    {
      "name": "Test2",
      "passed": false,
      "expected": "10",
      "output": "8",
      "error": "Expected: 10, Got: 8"
    }
  ],
  "passed": 1,
  "total": 2
}
```

The UI displays:
- **Input**: From test description/comment
- **Expected**: The expected value from assertion
- **Actual**: The actual value produced by student code

---

## Python ML Tasks

### Task Schema for ML

```typescript
export const task: Task = {
    slug: 'python-ml-linear-regression',
    title: 'Linear Regression',
    difficulty: 'medium',
    tags: ['python', 'ml', 'sklearn', 'regression'],
    estimatedTime: '25m',
    isPremium: false,

    // Visualization type (optional)
    visualizationType: 'scatter', // 'none' | 'line' | 'bar' | 'scatter' | 'heatmap'

    description: `...`,
    initialCode: `...`,
    solutionCode: `...`,
    testCode: `...`,

    hint1: `...`,
    hint2: `...`,
    whyItMatters: `...`,

    order: 0,
    translations: { ru: {...}, uz: {...} }
};
```

### Visualization Output Format

For tasks with visualization, student code should output JSON:

```python
import json

# After computation
chart_data = {
    "type": "scatter",
    "data": [
        {"x": x_val, "y": y_val, "label": "Actual"},
        {"x": x_val, "y": pred_val, "label": "Predicted"}
    ],
    "config": {
        "title": "Linear Regression Results",
        "xLabel": "X",
        "yLabel": "Y"
    }
}

# Output format: __CHART__<json>__CHART__
print("__CHART__" + json.dumps(chart_data) + "__CHART__")
```

### Visualization Types

| Type | Use Case | Data Format |
|------|----------|-------------|
| `line` | Training curves, time series | `[{x, y}, ...]` |
| `bar` | Feature importance, counts | `[{name, value}, ...]` |
| `scatter` | Regression, clustering | `[{x, y, label?}, ...]` |
| `heatmap` | Confusion matrix, correlation | `[[val, ...], ...]` |

### Example: NumPy Task

```typescript
export const task: Task = {
    slug: 'python-numpy-array-creation',
    title: 'Create Arrays from Lists',
    difficulty: 'easy',
    tags: ['python', 'numpy', 'arrays'],
    estimatedTime: '10m',
    isPremium: false,

    description: `Learn to create NumPy arrays from Python lists.

**Task:** Implement a function \`create_array\` that takes a Python list and returns a NumPy array.

**Requirements:**
- Import numpy as np
- Convert list to numpy array
- Return the array

**Example:**
\`\`\`python
>>> create_array([1, 2, 3])
array([1, 2, 3])
\`\`\``,

    initialCode: `import numpy as np

def create_array(lst):
    """Convert a Python list to a NumPy array.

    Args:
        lst: A Python list of numbers

    Returns:
        np.ndarray: A NumPy array
    """
    # TODO: Implement this function
    pass`,

    solutionCode: `import numpy as np

def create_array(lst):
    """Convert a Python list to a NumPy array.

    Args:
        lst: A Python list of numbers

    Returns:
        np.ndarray: A NumPy array
    """
    return np.array(lst)`,

    testCode: `import unittest
import numpy as np

class Test(unittest.TestCase):
    def test_1(self):
        result = create_array([1, 2, 3])
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_2(self):
        result = create_array([])
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    def test_3(self):
        result = create_array([1.5, 2.5, 3.5])
        expected = np.array([1.5, 2.5, 3.5])
        np.testing.assert_array_equal(result, expected)

    def test_4(self):
        result = create_array([-1, 0, 1])
        expected = np.array([-1, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_5(self):
        result = create_array([100])
        expected = np.array([100])
        np.testing.assert_array_equal(result, expected)

    def test_6(self):
        result = create_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(len(result), 10)

    def test_7(self):
        result = create_array([1, 2, 3])
        self.assertIsInstance(result, np.ndarray)

    def test_8(self):
        result = create_array([0, 0, 0])
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_9(self):
        result = create_array(list(range(100)))
        self.assertEqual(result.shape, (100,))

    def test_10(self):
        result = create_array([1e10, 2e10])
        expected = np.array([1e10, 2e10])
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()`,

    hint1: `Use np.array() to convert a list to a NumPy array.`,
    hint2: `Simply pass the list directly to np.array(): return np.array(lst)`,

    whyItMatters: `NumPy arrays are the foundation of scientific computing in Python. Unlike Python lists, NumPy arrays are:
- Faster (vectorized operations)
- Memory-efficient (contiguous storage)
- Required by most ML libraries (sklearn, tensorflow, pytorch)

**Production Pattern:**

\`\`\`python
import numpy as np

# Data from API/database as list
raw_data = [1.5, 2.3, 3.1, 4.2]

# Convert to numpy for ML processing
data = np.array(raw_data)

# Now you can use vectorized operations
normalized = (data - data.mean()) / data.std()
\`\`\`

**Practical Benefits:**
1. All ML frameworks expect NumPy arrays
2. Vectorized operations are 10-100x faster than loops
3. Memory layout enables GPU acceleration`,

    order: 0,
    translations: {
        ru: {
            title: 'Создание массивов из списков',
            solutionCode: `import numpy as np

def create_array(lst):
    """Преобразует Python список в массив NumPy.

    Args:
        lst: Python список чисел

    Returns:
        np.ndarray: Массив NumPy
    """
    return np.array(lst)`,
            description: `Научитесь создавать массивы NumPy из списков Python.

**Задача:** Реализуйте функцию \`create_array\`, которая принимает список Python и возвращает массив NumPy.

**Требования:**
- Импортируйте numpy как np
- Преобразуйте список в массив numpy
- Верните массив

**Пример:**
\`\`\`python
>>> create_array([1, 2, 3])
array([1, 2, 3])
\`\`\``,
            hint1: `Используйте np.array() для преобразования списка в массив NumPy.`,
            hint2: `Просто передайте список в np.array(): return np.array(lst)`,
            whyItMatters: `Массивы NumPy - основа научных вычислений в Python. В отличие от списков Python, массивы NumPy:
- Быстрее (векторизованные операции)
- Эффективнее по памяти (непрерывное хранение)
- Необходимы для большинства ML библиотек

**Продакшен паттерн:**

\`\`\`python
import numpy as np

# Данные из API/базы данных как список
raw_data = [1.5, 2.3, 3.1, 4.2]

# Преобразуем в numpy для ML обработки
data = np.array(raw_data)

# Теперь можно использовать векторизованные операции
normalized = (data - data.mean()) / data.std()
\`\`\`

**Практические преимущества:**
1. Все ML фреймворки ожидают массивы NumPy
2. Векторизованные операции в 10-100 раз быстрее циклов
3. Расположение в памяти позволяет ускорение на GPU`
        },
        uz: {
            title: `Ro'yxatlardan massivlar yaratish`,
            solutionCode: `import numpy as np

def create_array(lst):
    """Python ro'yxatini NumPy massiviga aylantiradi.

    Args:
        lst: Python raqamlar ro'yxati

    Returns:
        np.ndarray: NumPy massivi
    """
    return np.array(lst)`,
            description: `Python ro'yxatlaridan NumPy massivlarini yaratishni o'rganing.

**Vazifa:** Python ro'yxatini qabul qilib, NumPy massivini qaytaradigan \`create_array\` funksiyasini amalga oshiring.

**Talablar:**
- numpy ni np sifatida import qiling
- Ro'yxatni numpy massiviga aylantiring
- Massivni qaytaring

**Misol:**
\`\`\`python
>>> create_array([1, 2, 3])
array([1, 2, 3])
\`\`\``,
            hint1: `Ro'yxatni NumPy massiviga aylantirish uchun np.array() dan foydalaning.`,
            hint2: `Ro'yxatni to'g'ridan-to'g'ri np.array() ga bering: return np.array(lst)`,
            whyItMatters: `NumPy massivlari Python da ilmiy hisoblashning asosidir. Python ro'yxatlaridan farqli, NumPy massivlari:
- Tezroq (vektorlashtirilgan operatsiyalar)
- Xotira bo'yicha samaraliroq
- Ko'pchilik ML kutubxonalari uchun talab qilinadi

**Ishlab chiqarish patterni:**

\`\`\`python
import numpy as np

# API/ma'lumotlar bazasidan ro'yxat ko'rinishidagi ma'lumotlar
raw_data = [1.5, 2.3, 3.1, 4.2]

# ML ishlov berish uchun numpy ga aylantirish
data = np.array(raw_data)

# Endi vektorlashtirilgan operatsiyalardan foydalanish mumkin
normalized = (data - data.mean()) / data.std()
\`\`\`

**Amaliy foydalari:**
1. Barcha ML freymvorklari NumPy massivlarini kutadi
2. Vektorlashtirilgan operatsiyalar sikllardan 10-100 marta tezroq
3. Xotirada joylashuv GPU tezlashtirish imkonini beradi`
        }
    }
};

export default task;
```

### Example: sklearn Task with Visualization

```typescript
export const task: Task = {
    slug: 'python-sklearn-linear-regression',
    title: 'Linear Regression',
    difficulty: 'medium',
    tags: ['python', 'ml', 'sklearn', 'regression'],
    estimatedTime: '25m',
    isPremium: false,
    visualizationType: 'scatter',

    description: `Train a Linear Regression model using scikit-learn.

**Task:** Implement a function \`train_linear_regression\` that:
1. Creates a LinearRegression model
2. Fits it on training data
3. Returns predictions on test data

**Requirements:**
- Use sklearn.linear_model.LinearRegression
- Return predictions as numpy array
- Output visualization data

**Expected output includes a scatter plot of actual vs predicted values.**`,

    // ... rest of task
};
```

---

## Task Quality Checklist

### Before Submitting Any Task

- [ ] `slug` follows format: `{language}-{module}-{task-name}`
- [ ] `title` is descriptive in English
- [ ] `translations.ru.title` sounds natural in Russian
- [ ] `translations.uz.title` sounds natural in Uzbek
- [ ] `difficulty` matches content (easy/medium/hard)
- [ ] `estimatedTime` is realistic
- [ ] `description` includes clear requirements and examples
- [ ] `initialCode` has TODO comment and function signature
- [ ] `solutionCode` is correct and follows best practices
- [ ] `testCode` has 10 test cases covering edge cases
- [ ] `hint1` gives a small nudge
- [ ] `hint2` provides more concrete guidance
- [ ] `whyItMatters` explains real-world relevance
- [ ] `whyItMatters` includes production pattern in all languages
- [ ] All translations have translated code comments

### ML-Specific Checks

- [ ] Library imports are correct (numpy, pandas, sklearn)
- [ ] Data shapes/types are documented
- [ ] Edge cases for empty/small datasets are tested
- [ ] Numerical precision is handled (use assertAlmostEqual)
- [ ] Visualization type is set if task produces charts
- [ ] Output format matches expected JSON structure
