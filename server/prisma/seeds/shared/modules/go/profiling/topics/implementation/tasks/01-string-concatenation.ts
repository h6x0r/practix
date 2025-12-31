import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-profiling-string-concat',
	title: 'Optimized String Concatenation',
	difficulty: 'easy',	tags: ['go', 'profiling', 'performance', 'strings'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Optimize string concatenation using strings.Builder with pre-allocated capacity.

**Requirements:**
1. **BetterConcat**: Concatenate strings efficiently using strings.Builder
2. **Pre-allocation**: Calculate total length and call builder.Grow()
3. **Avoid copies**: Use WriteString() instead of += operator
4. **Return result**: Convert builder to string with String()

**Naive Approach (Slow):**
\`\`\`go
func NaiveConcat(parts []string) string {
    var s string
    for _, p := range parts {
        s += p  // Creates new string on EVERY iteration!
    }
    return s
}
// For 1000 parts: ~500,000 allocations, ~5MB memory, 50ms
\`\`\`

**Optimized Approach:**
\`\`\`go
func BetterConcat(parts []string) string {
    // Calculate total size needed
    n := 0
    for _, p := range parts {
        n += len(p)
    }

    // Pre-allocate exactly what we need
    var builder strings.Builder
    builder.Grow(n)

    // Write without allocations
    for _, p := range parts {
        builder.WriteString(p)
    }

    return builder.String()
}
// For 1000 parts: 1 allocation, ~10KB memory, 0.5ms (100x faster!)
\`\`\`

**Key Concepts:**
- String concatenation with += creates a new string each time
- Each copy requires allocating new memory and copying all bytes
- O(n²) time complexity with naive approach
- strings.Builder reuses internal buffer efficiently
- Grow() pre-allocates exact capacity needed
- O(n) time complexity with Builder

**Benchmark Commands:**
\`\`\`bash
# See the performance difference
go test -bench=Concat -benchmem

# Expected output:
BenchmarkConcat_Naive-8     100   50000000 ns/op   5242880 B/op   1000 allocs/op
BenchmarkConcat_Better-8   2000      500000 ns/op     10240 B/op      1 allocs/op

# Better is 100x faster with 99.9% less allocations!
\`\`\`

**Example Usage:**
\`\`\`go
// Building SQL query
parts := []string{
    "SELECT * FROM users WHERE ",
    "status = 'active' AND ",
    "created_at > '2024-01-01' AND ",
    "email LIKE '%@example.com'",
}
query := BetterConcat(parts)

// Building HTML
htmlParts := []string{
    "<html><body>",
    "<h1>Welcome</h1>",
    "<p>Hello, user!</p>",
    "</body></html>",
}
html := BetterConcat(htmlParts)

// Building log message with many fields
logParts := []string{
    "[", timestamp, "] ",
    "[", level, "] ",
    "[", requestID, "] ",
    message,
}
logLine := BetterConcat(logParts)
\`\`\`

**Constraints:**
- Must use strings.Builder
- Must call Grow() with calculated capacity
- Must use WriteString() for appending
- Return final string with String() method`,
	initialCode: `package profilingx

import (
	"strings"
)

// TODO: Implement BetterConcat
// Calculate total length of all parts
// Pre-allocate builder capacity with Grow
// Use WriteString to append each part
// Return final string
func BetterConcat(parts []string) string {
	return "" // TODO: Implement
}`,
	solutionCode: `package profilingx

import (
	"strings"
)

func BetterConcat(parts []string) string {
	n := 0
	for _, p := range parts {                   // calculate total size needed
		n += len(p)
	}

	var builder strings.Builder                 // create builder with internal buffer
	builder.Grow(n)                             // pre-allocate exact capacity to avoid reallocations

	for _, p := range parts {
		builder.WriteString(p)                   // append without copying entire string
	}

	return builder.String()                     // convert to final string (one allocation)
}`,
			hint1: `Loop through parts first to calculate total length: n += len(p). Then call builder.Grow(n).`,
			hint2: `Use builder.WriteString(p) in the second loop to append each part efficiently.`,
			testCode: `package profilingx

import (
	"testing"
)

func Test1(t *testing.T) {
	// Test empty slice
	result := BetterConcat([]string{})
	if result != "" {
		t.Errorf("BetterConcat([]) = %q, want empty string", result)
	}
}

func Test2(t *testing.T) {
	// Test nil slice
	result := BetterConcat(nil)
	if result != "" {
		t.Errorf("BetterConcat(nil) = %q, want empty string", result)
	}
}

func Test3(t *testing.T) {
	// Test single string
	result := BetterConcat([]string{"hello"})
	if result != "hello" {
		t.Errorf("BetterConcat([hello]) = %q, want %q", result, "hello")
	}
}

func Test4(t *testing.T) {
	// Test two strings
	result := BetterConcat([]string{"hello", "world"})
	if result != "helloworld" {
		t.Errorf("BetterConcat([hello,world]) = %q, want %q", result, "helloworld")
	}
}

func Test5(t *testing.T) {
	// Test multiple strings
	parts := []string{"a", "b", "c", "d", "e"}
	result := BetterConcat(parts)
	if result != "abcde" {
		t.Errorf("BetterConcat([a,b,c,d,e]) = %q, want %q", result, "abcde")
	}
}

func Test6(t *testing.T) {
	// Test with spaces
	parts := []string{"hello", " ", "world"}
	result := BetterConcat(parts)
	if result != "hello world" {
		t.Errorf("BetterConcat([hello, ,world]) = %q, want %q", result, "hello world")
	}
}

func Test7(t *testing.T) {
	// Test with empty strings in slice
	parts := []string{"a", "", "b", "", "c"}
	result := BetterConcat(parts)
	if result != "abc" {
		t.Errorf("BetterConcat([a,,b,,c]) = %q, want %q", result, "abc")
	}
}

func Test8(t *testing.T) {
	// Test with unicode
	parts := []string{"привет", " ", "мир"}
	result := BetterConcat(parts)
	if result != "привет мир" {
		t.Errorf("BetterConcat(unicode) = %q, want %q", result, "привет мир")
	}
}

func Test9(t *testing.T) {
	// Test with large strings
	large := make([]string, 100)
	for i := range large {
		large[i] = "x"
	}
	result := BetterConcat(large)
	if len(result) != 100 {
		t.Errorf("len(BetterConcat(100x)) = %d, want 100", len(result))
	}
}

func Test10(t *testing.T) {
	// Test with newlines
	parts := []string{"line1", "\n", "line2", "\n", "line3"}
	expected := "line1\nline2\nline3"
	result := BetterConcat(parts)
	if result != expected {
		t.Errorf("BetterConcat(newlines) = %q, want %q", result, expected)
	}
}`,
			whyItMatters: `String concatenation optimization is crucial for high-performance Go applications handling text processing.

**Why This Matters:**

**1. The Hidden Cost of String Concatenation**
Strings in Go are immutable - every += creates a new string:
\`\`\`go
// What you write:
s := "Hello"
s += " "
s += "World"

// What actually happens:
s := "Hello"              // Allocate 5 bytes
temp1 := "Hello "         // Allocate 6 bytes, copy 5 bytes
s = temp1
temp2 := "Hello World"    // Allocate 11 bytes, copy 6 bytes
s = temp2
// Total: 22 bytes allocated, 11 bytes copied
// For just 2 concatenations!
\`\`\`

**2. Real Production Scenario: Log Aggregation**
A logging service concatenates thousands of log lines per second:
\`\`\`go
// BEFORE - Naive approach
func BuildLogLine(timestamp, level, requestID, message string) string {
    var line string
    line += "["
    line += timestamp
    line += "] ["
    line += level
    line += "] ["
    line += requestID
    line += "] "
    line += message
    // 7 concatenations = 7 allocations = slow!
    return line
}

// Handling 10,000 logs/sec:
// - 70,000 allocations/sec
// - GC pressure constantly
// - CPU at 80%
// - Latency: p99 = 50ms

// AFTER - strings.Builder
func BuildLogLine(timestamp, level, requestID, message string) string {
    var b strings.Builder
    b.Grow(len(timestamp) + len(level) + len(requestID) + len(message) + 10)
    b.WriteString("[")
    b.WriteString(timestamp)
    b.WriteString("] [")
    b.WriteString(level)
    b.WriteString("] [")
    b.WriteString(requestID)
    b.WriteString("] ")
    b.WriteString(message)
    return b.String()
}

// Handling 10,000 logs/sec:
// - 10,000 allocations/sec (1 per log)
// - Minimal GC pressure
// - CPU at 15%
// - Latency: p99 = 2ms (25x improvement!)
\`\`\`

**3. JSON Response Building**
API server building JSON responses:
\`\`\`go
// Building large JSON for 1000 users
users := fetchUsers() // 1000 users

// SLOW - Naive concatenation
func BuildJSON_Slow(users []User) string {
    json := "{"
    json += "\\"users\\": ["
    for i, user := range users {
        json += "{\\"id\\":" + strconv.Itoa(user.ID) + ","
        json += "\\"name\\":\\"" + user.Name + "\\","
        json += "\\"email\\":\\"" + user.Email + "\\"}"
        if i < len(users)-1 {
            json += ","
        }
    }
    json += "]}"
    return json
}
// Time: 450ms, Allocations: 4000+

// FAST - strings.Builder
func BuildJSON_Fast(users []User) string {
    var b strings.Builder
    b.Grow(len(users) * 100) // Estimate size
    b.WriteString("{\\"users\\": [")
    for i, user := range users {
        b.WriteString("{\\"id\\":")
        b.WriteString(strconv.Itoa(user.ID))
        b.WriteString(",\\"name\\":\\"")
        b.WriteString(user.Name)
        b.WriteString("\\",\\"email\\":\\"")
        b.WriteString(user.Email)
        b.WriteString("\\"}")
        if i < len(users)-1 {
            b.WriteString(",")
        }
    }
    b.WriteString("]}")
    return b.String()
}
// Time: 8ms (56x faster!), Allocations: 2
\`\`\`

**4. CSV Export Generation**
Exporting database records to CSV:
\`\`\`go
// Exporting 100,000 rows

// BEFORE - String concatenation
func ExportCSV_Slow(records []Record) string {
    csv := "id,name,email,created_at\\n"
    for _, r := range records {
        csv += fmt.Sprintf("%d,%s,%s,%s\\n",
            r.ID, r.Name, r.Email, r.CreatedAt)
    }
    return csv
}
// Time: 45 seconds, Memory: 2GB (OOM!)

// AFTER - strings.Builder
func ExportCSV_Fast(records []Record) string {
    var b strings.Builder
    b.Grow(len(records) * 80) // Avg row size
    b.WriteString("id,name,email,created_at\\n")
    for _, r := range records {
        b.WriteString(strconv.Itoa(r.ID))
        b.WriteString(",")
        b.WriteString(r.Name)
        b.WriteString(",")
        b.WriteString(r.Email)
        b.WriteString(",")
        b.WriteString(r.CreatedAt.String())
        b.WriteString("\\n")
    }
    return b.String()
}
// Time: 800ms (56x faster!), Memory: 8MB
\`\`\`

**5. Why Grow() Matters**
Pre-allocation prevents internal buffer resizing:
\`\`\`go
// WITHOUT Grow - buffer doubles each time it fills
var b strings.Builder
for i := 0; i < 1000; i++ {
    b.WriteString("item")
}
// Internal resizing: 4 → 8 → 16 → 32 → 64 → 128 → ...
// Multiple allocations and copies

// WITH Grow - one allocation
var b strings.Builder
b.Grow(4000)  // Exactly what we need
for i := 0; i < 1000; i++ {
    b.WriteString("item")
}
// Single allocation, no resizing, no copying
\`\`\`

**6. Profiling to Find Bottlenecks**
\`\`\`bash
# Generate CPU profile
go test -bench=Concat -cpuprofile=cpu.out

# Analyze with pprof
go tool pprof cpu.out
(pprof) top10
# Shows runtime.concatstrings taking 85% of time!

(pprof) list NaiveConcat
# Shows exact lines causing allocations

# After optimization:
go test -bench=Concat -cpuprofile=cpu2.out
go tool pprof cpu2.out
(pprof) top10
# runtime.concatstrings no longer in top 10!
\`\`\`

**7. Memory Profiling**
\`\`\`bash
# Generate memory profile
go test -bench=Concat -memprofile=mem.out

# Analyze allocations
go tool pprof mem.out
(pprof) top10
# BEFORE: runtime.concatstrings: 5MB
# AFTER: strings.Builder: 10KB (500x reduction!)
\`\`\`

**Real-World Impact:**
An e-commerce company built product search results:
- **Before**: Naive concatenation for HTML generation
  - 1000 products = 8 seconds to render
  - Servers constantly hitting memory limits
  - Users abandoning slow search results

- **After**: strings.Builder with Grow()
  - 1000 products = 80ms to render (100x faster!)
  - Memory usage dropped 95%
  - Search abandonment rate: 45% → 8%
  - Revenue increased by $2M/year

**Production Best Practices:**
1. Always use strings.Builder for multiple concatenations
2. Call Grow() if you know approximate final size
3. Use WriteString() not += or +
4. Profile with -benchmem to verify improvements
5. Consider bytes.Buffer for []byte operations
6. Benchmark before and after optimization
7. Use pprof to find actual bottlenecks

String concatenation seems trivial, but it's a common source of performance problems in production. Master these techniques and you'll write dramatically faster code.`,	order: 0,
	translations: {
		ru: {
			title: 'Профилирование конкатенации строк',
			solutionCode: `package profilingx

import (
	"strings"
)

func BetterConcat(parts []string) string {
	n := 0
	for _, p := range parts {                   // вычисляем общий нужный размер
		n += len(p)
	}

	var builder strings.Builder                 // создаем builder с внутренним буфером
	builder.Grow(n)                             // пре-аллоцируем точную capacity чтобы избежать реаллокаций

	for _, p := range parts {
		builder.WriteString(p)                   // добавляем без копирования всей строки
	}

	return builder.String()                     // конвертируем в финальную строку (одна аллокация)
}`,
			description: `Оптимизируйте конкатенацию строк используя strings.Builder с пре-аллокацией capacity.

**Требования:**
1. **BetterConcat**: Конкатенировать строки эффективно используя strings.Builder
2. **Pre-allocation**: Вычислить общую длину и вызвать builder.Grow()
3. **Избегать копий**: Использовать WriteString() вместо оператора +=
4. **Вернуть результат**: Конвертировать builder в строку через String()

**Наивный подход (медленный):**
\`\`\`go
func NaiveConcat(parts []string) string {
    var s string
    for _, p := range parts {
        s += p  // Создает новую строку на КАЖДОЙ итерации!
    }
    return s
}
// Для 1000 частей: ~500,000 аллокаций, ~5MB памяти, 50ms
\`\`\`

**Ключевые концепции:**
- Конкатенация строк с += создает новую строку каждый раз
- Каждое копирование требует аллокации новой памяти и копирования всех байтов
- O(n²) временная сложность с наивным подходом
- strings.Builder переиспользует внутренний буфер эффективно
- Grow() пре-аллоцирует точную нужную capacity
- O(n) временная сложность с Builder

**Ограничения:**
- Должен использовать strings.Builder
- Должен вызывать Grow() с вычисленной capacity
- Должен использовать WriteString() для append
- Вернуть финальную строку через метод String()`,
			hint1: `Сначала пройдите по parts чтобы вычислить общую длину: n += len(p). Затем вызовите builder.Grow(n).`,
			hint2: `Используйте builder.WriteString(p) во втором цикле для эффективного добавления каждой части.`,
			whyItMatters: `Оптимизация конкатенации строк критична для высокопроизводительных Go приложений, обрабатывающих текст.

**Почему это важно:**

**1. Скрытая цена конкатенации строк**

Строки в Go иммутабельны - каждый += создает новую строку:

\`\`\`go
// Что вы пишете:
s := "Hello"
s += " "
s += "World"

// Что реально происходит:
s := "Hello"              // Аллокация 5 байт
temp1 := "Hello "         // Аллокация 6 байт, копирование 5 байт
s = temp1
temp2 := "Hello World"    // Аллокация 11 байт, копирование 6 байт
s = temp2
// Итого: 22 байта выделено, 11 байт скопировано
// И это только для 2 конкатенаций!
\`\`\`

**2. Реальный продакшен: Агрегация логов**

Сервис логирования конкатенирует тысячи log строк в секунду:

\`\`\`go
// BEFORE - Naive approach
func BuildLogLine(timestamp, level, requestID, message string) string {
    var line string
    line += "["
    line += timestamp
    line += "] ["
    line += level
    line += "] ["
    line += requestID
    line += "] "
    line += message
    // 7 concatenations = 7 allocations = slow!
    return line
}

// Обработка 10,000 logs/sec:
// - 70,000 allocations/sec
// - Постоянное давление на GC
// - CPU at 80%
// - Latency: p99 = 50ms

// AFTER - strings.Builder
func BuildLogLine(timestamp, level, requestID, message string) string {
    var b strings.Builder
    b.Grow(len(timestamp) + len(level) + len(requestID) + len(message) + 10)
    b.WriteString("[")
    b.WriteString(timestamp)
    b.WriteString("] [")
    b.WriteString(level)
    b.WriteString("] [")
    b.WriteString(requestID)
    b.WriteString("] ")
    b.WriteString(message)
    return b.String()
}

// Обработка 10,000 logs/sec:
// - 10,000 allocations/sec (1 на log)
// - Минимальное давление на GC
// - CPU at 15%
// - Latency: p99 = 2ms (улучшение в 25x!)
\`\`\`

**3. JSON Response Building**

API сервер строит JSON ответы:

\`\`\`go
// Построение большого JSON для 1000 users
users := fetchUsers() // 1000 users

// SLOW - Naive concatenation
func BuildJSON_Slow(users []User) string {
    json := "{"
    json += "\\"users\\": ["
    for i, user := range users {
        json += "{\\"id\\":" + strconv.Itoa(user.ID) + ","
        json += "\\"name\\":\\"" + user.Name + "\\","
        json += "\\"email\\":\\"" + user.Email + "\\"}"
        if i < len(users)-1 {
            json += ","
        }
    }
    json += "]}"
    return json
}
// Time: 450ms, Allocations: 4000+

// FAST - strings.Builder
func BuildJSON_Fast(users []User) string {
    var b strings.Builder
    b.Grow(len(users) * 100) // Estimate size
    b.WriteString("{\\"users\\": [")
    for i, user := range users {
        b.WriteString("{\\"id\\":")
        b.WriteString(strconv.Itoa(user.ID))
        b.WriteString(",\\"name\\":\\"")
        b.WriteString(user.Name)
        b.WriteString("\\",\\"email\\":\\"")
        b.WriteString(user.Email)
        b.WriteString("\\"}")
        if i < len(users)-1 {
            b.WriteString(",")
        }
    }
    b.WriteString("]}")
    return b.String()
}
// Time: 8ms (56x быстрее!), Allocations: 2
\`\`\`

**4. CSV Export Generation**

Экспорт записей БД в CSV:

\`\`\`go
// Экспорт 100,000 строк

// BEFORE - String concatenation
func ExportCSV_Slow(records []Record) string {
    csv := "id,name,email,created_at\\n"
    for _, r := range records {
        csv += fmt.Sprintf("%d,%s,%s,%s\\n",
            r.ID, r.Name, r.Email, r.CreatedAt)
    }
    return csv
}
// Time: 45 seconds, Memory: 2GB (OOM!)

// AFTER - strings.Builder
func ExportCSV_Fast(records []Record) string {
    var b strings.Builder
    b.Grow(len(records) * 80) // Avg row size
    b.WriteString("id,name,email,created_at\\n")
    for _, r := range records {
        b.WriteString(strconv.Itoa(r.ID))
        b.WriteString(",")
        b.WriteString(r.Name)
        b.WriteString(",")
        b.WriteString(r.Email)
        b.WriteString(",")
        b.WriteString(r.CreatedAt.String())
        b.WriteString("\\n")
    }
    return b.String()
}
// Time: 800ms (56x быстрее!), Memory: 8MB
\`\`\`

**5. Почему Grow() важен**

Пре-аллокация предотвращает изменение размера внутреннего буфера:

\`\`\`go
// БЕЗ Grow - буфер удваивается при заполнении
var b strings.Builder
for i := 0; i < 1000; i++ {
    b.WriteString("item")
}
// Внутренние изменения: 4 → 8 → 16 → 32 → 64 → 128 → ...
// Множественные аллокации и копирования

// С Grow - одна аллокация
var b strings.Builder
b.Grow(4000)  // Exactly what we need
for i := 0; i < 1000; i++ {
    b.WriteString("item")
}
// Одна аллокация, без изменения размера, без копирования
\`\`\`

**6. Профилирование для поиска bottlenecks**

\`\`\`bash
# Generate CPU profile
go test -bench=Concat -cpuprofile=cpu.out

# Analyze with pprof
go tool pprof cpu.out
(pprof) top10
# Shows runtime.concatstrings taking 85% of time!

(pprof) list NaiveConcat
# Shows exact lines causing allocations

# After optimization:
go test -bench=Concat -cpuprofile=cpu2.out
go tool pprof cpu2.out
(pprof) top10
# runtime.concatstrings no longer in top 10!
\`\`\`

**7. Memory Profiling**

\`\`\`bash
# Generate memory profile
go test -bench=Concat -memprofile=mem.out

# Analyze allocations
go tool pprof mem.out
(pprof) top10
# BEFORE: runtime.concatstrings: 5MB
# AFTER: strings.Builder: 10KB (уменьшение в 500x!)
\`\`\`

**Реальное влияние:**

E-commerce компания строила результаты поиска продуктов:
- **До**: Naive concatenation для HTML generation
  - 1000 продуктов = 8 секунд рендеринга
  - Серверы постоянно достигали лимитов памяти
  - Пользователи покидали медленные результаты поиска

- **После**: strings.Builder с Grow()
  - 1000 продуктов = 80ms рендеринга (100x быстрее!)
  - Использование памяти упало на 95%
  - Abandonment rate поиска: 45% → 8%
  - Revenue увеличился на $2M/год

**Production Best Practices:**
1. Всегда используйте strings.Builder для множественных конкатенаций
2. Вызывайте Grow() если знаете примерный финальный размер
3. Используйте WriteString() не += или +
4. Профилируйте с -benchmem для проверки улучшений
5. Рассмотрите bytes.Buffer для операций с []byte
6. Делайте бенчмарки до и после оптимизации
7. Используйте pprof для поиска реальных bottlenecks

Конкатенация строк кажется тривиальной, но это частый источник проблем производительности в продакшене. Освойте эти техники и вы будете писать драматически более быстрый код.`
		},
		uz: {
			title: `String birlashtirish profiling`,
			solutionCode: `package profilingx

import (
	"strings"
)

func BetterConcat(parts []string) string {
	n := 0
	for _, p := range parts {                   // kerakli umumiy o'lchamni hisoblaymiz
		n += len(p)
	}

	var builder strings.Builder                 // ichki buferli builder yaratamiz
	builder.Grow(n)                             // qayta ajratishlarni oldini olish uchun aniq capacity ni oldindan ajratamiz

	for _, p := range parts {
		builder.WriteString(p)                   // butun satrni nusxalamasdan qo'shamiz
	}

	return builder.String()                     // yakuniy satrga aylantiramiz (bitta ajratish)
}`,
			description: `Oldindan ajratilgan capacity bilan strings.Builder yordamida satr konkatenatsiyasini optimallashtiring.

**Talablar:**
1. **BetterConcat**: strings.Builder yordamida satrlarni samarali birlashtiring
2. **Oldindan Ajratish**: Umumiy uzunlikni hisoblang va builder.Grow() ni chaqiring
3. **Nusxalardan Qoching**: += operatori o'rniga WriteString() dan foydalaning
4. **Natijani Qaytaring**: Builder ni String() bilan satrga aylantiring

**Naiv Yondashuv (Sekin):**
\`\`\`go
func NaiveConcat(parts []string) string {
    var s string
    for _, p := range parts {
        s += p  // HAR BIR iteratsiyada yangi satr yaratadi!
    }
    return s
}
// 1000 qism uchun: ~500,000 ajratish, ~5MB xotira, 50ms
\`\`\`

**Optimallashtirilgan Yondashuv:**
\`\`\`go
func BetterConcat(parts []string) string {
    n := 0
    for _, p := range parts {
        n += len(p)
    }
    var builder strings.Builder
    builder.Grow(n)
    for _, p := range parts {
        builder.WriteString(p)
    }
    return builder.String()
}
// 1000 qism uchun: 1 ajratish, ~10KB xotira, 0.5ms (100x tezroq!)
\`\`\`

**Cheklovlar:**
- strings.Builder ishlatish kerak
- Hisoblangan capacity bilan Grow() chaqirish kerak
- Qo'shish uchun WriteString() ishlatish kerak
- String() metodi bilan yakuniy satrni qaytaring`,
			hint1: `Avval umumiy uzunlikni hisoblash uchun parts bo'ylab aylanib chiqing: n += len(p). Keyin builder.Grow(n) chaqiring.`,
			hint2: `Har bir qismni samarali qo'shish uchun ikkinchi siklda builder.WriteString(p) ishlating.`,
			whyItMatters: `Satr konkatenatsiya optimizatsiyasi matn qayta ishlash bilan shug'ullanadigan yuqori ishlashli Go ilovalari uchun muhim.

**Nima uchun bu muhim:**

**1. Satr Konkatenatsiyasining Yashirin Narxi**

Go da satrlar o'zgarmas - har bir += yangi satr yaratadi:

\`\`\`go
// Siz yozasiz:
s := "Hello"
s += " "
s += "World"

// Aslida sodir bo'layotgan narsa:
s := "Hello"              // 5 bayt ajratish
temp1 := "Hello "         // 6 bayt ajratish, 5 bayt nusxalash
s = temp1
temp2 := "Hello World"    // 11 bayt ajratish, 6 bayt nusxalash
s = temp2
// Jami: 22 bayt ajratildi, 11 bayt nusxalandi
// Va bu faqat 2 ta konkatenatsiya uchun!
\`\`\`

**2. Haqiqiy ishlab chiqarish: Log Aggregation**

Jurnal yozish xizmati soniyada minglab log qatorlarini birlashtiradi:

\`\`\`go
// OLDIN - Naiv yondashuv
func BuildLogLine(timestamp, level, requestID, message string) string {
    var line string
    line += "["
    line += timestamp
    line += "] ["
    line += level
    line += "] ["
    line += requestID
    line += "] "
    line += message
    // 7 ta birlashtirish = 7 ta ajratish = sekin!
    return line
}

// 10,000 log/soniyani qayta ishlash:
// - 70,000 ajratish/soniya
// - Doimiy GC bosimi
// - CPU 80% da
// - Latency: p99 = 50ms

// KEYIN - strings.Builder
func BuildLogLine(timestamp, level, requestID, message string) string {
    var b strings.Builder
    b.Grow(len(timestamp) + len(level) + len(requestID) + len(message) + 10)
    b.WriteString("[")
    b.WriteString(timestamp)
    b.WriteString("] [")
    b.WriteString(level)
    b.WriteString("] [")
    b.WriteString(requestID)
    b.WriteString("] ")
    b.WriteString(message)
    return b.String()
}

// 10,000 log/soniyani qayta ishlash:
// - 10,000 ajratish/soniya (log uchun 1 ta)
// - Minimal GC bosimi
// - CPU 15% da
// - Latency: p99 = 2ms (25x yaxshilanish!)
\`\`\`

**3. JSON Response Building**

API server JSON javoblarini quradi:

\`\`\`go
// 1000 foydalanuvchi uchun katta JSON qurish
users := fetchUsers() // 1000 foydalanuvchi

// SEKIN - Naiv konkatenatsiya
func BuildJSON_Slow(users []User) string {
    json := "{"
    json += "\\"users\\": ["
    for i, user := range users {
        json += "{\\"id\\":" + strconv.Itoa(user.ID) + ","
        json += "\\"name\\":\\"" + user.Name + "\\","
        json += "\\"email\\":\\"" + user.Email + "\\"}"
        if i < len(users)-1 {
            json += ","
        }
    }
    json += "]}"
    return json
}
// Vaqt: 450ms, Ajratishlar: 4000+

// TEZ - strings.Builder
func BuildJSON_Fast(users []User) string {
    var b strings.Builder
    b.Grow(len(users) * 100) // O'lchamni taxmin qilish
    b.WriteString("{\\"users\\": [")
    for i, user := range users {
        b.WriteString("{\\"id\\":")
        b.WriteString(strconv.Itoa(user.ID))
        b.WriteString(",\\"name\\":\\"")
        b.WriteString(user.Name)
        b.WriteString("\\",\\"email\\":\\"")
        b.WriteString(user.Email)
        b.WriteString("\\"}")
        if i < len(users)-1 {
            b.WriteString(",")
        }
    }
    b.WriteString("]}")
    return b.String()
}
// Vaqt: 8ms (56x tezroq!), Ajratishlar: 2
\`\`\`

**4. CSV Export Yaratish**

Ma'lumotlar bazasi yozuvlarini CSV ga eksport qilish:

\`\`\`go
// 100,000 qatorni eksport qilish

// OLDIN - Satr konkatenatsiyasi
func ExportCSV_Slow(records []Record) string {
    csv := "id,name,email,created_at\\n"
    for _, r := range records {
        csv += fmt.Sprintf("%d,%s,%s,%s\\n",
            r.ID, r.Name, r.Email, r.CreatedAt)
    }
    return csv
}
// Vaqt: 45 soniya, Xotira: 2GB (OOM!)

// KEYIN - strings.Builder
func ExportCSV_Fast(records []Record) string {
    var b strings.Builder
    b.Grow(len(records) * 80) // O'rtacha qator hajmi
    b.WriteString("id,name,email,created_at\\n")
    for _, r := range records {
        b.WriteString(strconv.Itoa(r.ID))
        b.WriteString(",")
        b.WriteString(r.Name)
        b.WriteString(",")
        b.WriteString(r.Email)
        b.WriteString(",")
        b.WriteString(r.CreatedAt.String())
        b.WriteString("\\n")
    }
    return b.String()
}
// Vaqt: 800ms (56x tezroq!), Xotira: 8MB
\`\`\`

**5. Nima uchun Grow() muhim**

Oldindan ajratish ichki bufer o'lchamini o'zgartirishning oldini oladi:

\`\`\`go
// Grow SIZ - bufer to'lganda ikki barobar ortadi
var b strings.Builder
for i := 0; i < 1000; i++ {
    b.WriteString("item")
}
// Ichki o'lcham o'zgarishlari: 4 → 8 → 16 → 32 → 64 → 128 → ...
// Ko'p ajratishlar va nusxalashlar

// Grow BILAN - bitta ajratish
var b strings.Builder
b.Grow(4000)  // Aniq kerak bo'lgan narsa
for i := 0; i < 1000; i++ {
    b.WriteString("item")
}
// Bitta ajratish, o'lcham o'zgartirish yo'q, nusxalash yo'q
\`\`\`

**6. Bottlenecklarni topish uchun profillash**

\`\`\`bash
# CPU profileni yaratish
go test -bench=Concat -cpuprofile=cpu.out

# pprof bilan tahlil qilish
go tool pprof cpu.out
(pprof) top10
# runtime.concatstrings vaqtning 85% ini olayotganini ko'rsatadi!

(pprof) list NaiveConcat
# Ajratishlarni keltirib chiqaradigan aniq qatorlarni ko'rsatadi

# Optimallashtirdan keyin:
go test -bench=Concat -cpuprofile=cpu2.out
go tool pprof cpu2.out
(pprof) top10
# runtime.concatstrings endi top 10 da yo'q!
\`\`\`

**7. Xotira Profillash**

\`\`\`bash
# Xotira profileni yaratish
go test -bench=Concat -memprofile=mem.out

# Ajratishlarni tahlil qilish
go tool pprof mem.out
(pprof) top10
# OLDIN: runtime.concatstrings: 5MB
# KEYIN: strings.Builder: 10KB (500x kamayish!)
\`\`\`

**Haqiqiy ta'sir:**

E-commerce kompaniyasi mahsulot qidiruv natijalarini qurdi:
- **Oldin**: HTML generatsiyasi uchun naiv konkatenatsiya
  - 1000 mahsulot = render qilish uchun 8 soniya
  - Serverlar doimiy ravishda xotira limitlariga yetdi
  - Foydalanuvchilar sekin qidiruv natijalarini tark etdi

- **Keyin**: Grow() bilan strings.Builder
  - 1000 mahsulot = render qilish uchun 80ms (100x tezroq!)
  - Xotira foydalanishi 95% ga kamaydi
  - Qidiruv abandonment rate: 45% → 8%
  - Daromad yiliga $2M ga oshdi

**Production eng yaxshi amaliyotlari:**
1. Ko'p konkatenatsiyalar uchun har doim strings.Builder dan foydalaning
2. Taxminiy yakuniy o'lchamni bilsangiz Grow() ni chaqiring
3. += yoki + emas WriteString() dan foydalaning
4. Yaxshilanishlarni tekshirish uchun -benchmem bilan profillash qiling
5. []byte operatsiyalari uchun bytes.Buffer ni ko'rib chiqing
6. Optimallashtirish dan oldin va keyin benchmark oling
7. Haqiqiy bottlenecklarni topish uchun pprof dan foydalaning

Satr konkatenatsiyasi sodda ko'rinadi, lekin bu ishlab chiqarishda ishlash muammolarining keng tarqalgan manbai. Ushbu texnikalarni o'rganing va siz dramatik tarzda tezroq kod yozasiz.`
		}
	}
};

export default task;
