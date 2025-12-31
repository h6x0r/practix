import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-fundamentals-join-efficient',
	title: 'Efficient String Concatenation',
	difficulty: 'medium',	tags: ['go', 'data-structures', 'maps/slices/strings', 'generics'],
	estimatedTime: '15-20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **JoinEfficient** that concatenates strings without using string concatenation in loops.

**Requirements:**
1. Create function \`JoinEfficient(parts []string) string\`
2. Handle empty slice (return empty string)
3. Handle single element slice (return that element)
4. Use strings.Builder for efficient concatenation
5. Pre-calculate total size needed
6. Call Grow() before writing to pre-allocate buffer
7. Write all parts to the builder
8. Return the final string

**Example:**
\`\`\`go
result := JoinEfficient([]string{"Hello", "World", "!"})
// result = "HelloWorld!"

result2 := JoinEfficient([]string{"a"})
// result2 = "a"

result3 := JoinEfficient([]string{})
// result3 = ""

result4 := JoinEfficient([]string{"foo", "bar", "baz", "qux"})
// result4 = "foobarbazqux"
\`\`\`

**Constraints:**
- Must not use string concatenation with + operator in loops
- Must handle empty slice gracefully
- Must pre-allocate buffer using Grow()
- Must calculate total byte length needed
- Should not use strings.Join (implement the pattern itself)`,
	initialCode: `package datastructures

import "strings"

// TODO: Implement JoinEfficient
func JoinEfficient(parts []string) string {
	return "" // TODO: Implement
}`,
	solutionCode: `package datastructures

import "strings"

func JoinEfficient(parts []string) string {
	switch len(parts) {                                     // Switch on slice length
	case 0:                                                 // Empty slice
		return ""                                       // Return empty string
	case 1:                                                 // Single element
		return parts[0]                                 // Return that element
	}
	totalLen := 0                                           // Calculate total length
	for _, part := range parts {                            // Sum all part lengths
		totalLen += len(part)                           // Add each part length
	}
	var builder strings.Builder                             // Create builder
	builder.Grow(totalLen)                                  // Pre-allocate exact buffer size
	for _, part := range parts {                            // Iterate through parts
		builder.WriteString(part)                       // Write each part
	}
	return builder.String()                                 // Return concatenated result
}`,
	testCode: `package datastructures

import "testing"

func Test1(t *testing.T) {
	// Basic concatenation
	result := JoinEfficient([]string{"Hello", "World", "!"})
	expected := "HelloWorld!"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func Test2(t *testing.T) {
	// Single element
	result := JoinEfficient([]string{"a"})
	expected := "a"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func Test3(t *testing.T) {
	// Empty slice
	result := JoinEfficient([]string{})
	expected := ""
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func Test4(t *testing.T) {
	// Four elements
	result := JoinEfficient([]string{"foo", "bar", "baz", "qux"})
	expected := "foobarbazqux"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func Test5(t *testing.T) {
	// Two elements
	result := JoinEfficient([]string{"ab", "cd"})
	expected := "abcd"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func Test6(t *testing.T) {
	// Empty strings in slice
	result := JoinEfficient([]string{"a", "", "b"})
	expected := "ab"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func Test7(t *testing.T) {
	// Long strings
	result := JoinEfficient([]string{"hello", "world", "from", "go"})
	expected := "helloworldfromgo"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func Test8(t *testing.T) {
	// Unicode strings
	result := JoinEfficient([]string{"Привет", "Мир"})
	expected := "ПриветМир"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func Test9(t *testing.T) {
	// Single character strings
	result := JoinEfficient([]string{"a", "b", "c", "d", "e"})
	expected := "abcde"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func Test10(t *testing.T) {
	// Whitespace strings
	result := JoinEfficient([]string{" ", "x", " "})
	expected := " x "
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}`,
	hint1: `Use switch statement to handle 0, 1, and multiple string cases. Calculate total byte length first, then use builder.Grow().`,
			hint2: `strings.Builder with Grow() avoids reallocations. Each WriteString() appends without creating intermediate strings.`,
			whyItMatters: `JoinEfficient demonstrates the critical performance difference between naive string concatenation and efficient buffering, especially important for high-throughput systems processing thousands of strings.

**Why Efficient String Building:**
- **Performance:** No intermediate string allocation in loops
- **Memory:** Single allocation instead of n allocations for n strings
- **GC Pressure:** Reduces garbage collection overhead dramatically
- **Scalability:** Essential for processing millions of strings
- **Responsiveness:** Eliminates microsecond delays from repeated allocations

**Production Pattern:**
\`\`\`go
// JSON response builder for API
func BuildJSONResponse(fields map[string]string) string {
    var parts []string
    parts = append(parts, "{")

    first := true
    for key, value := range fields {
        if !first {
            parts = append(parts, ",")
        }
        parts = append(parts, fmt.Sprintf("\"%s\":\"%s\"", key, value))
        first = false
    }

    parts = append(parts, "}")
    return JoinEfficient(parts)
}

// Log message formatting with multiple fields
type LogFormatter struct {
    timestamp string
    level     string
    message   string
    fields    map[string]string
}

func (lf *LogFormatter) Format() string {
    parts := make([]string, 0, 10)

    parts = append(parts,
        "[", lf.timestamp, "]",
        "[", lf.level, "]",
        " ", lf.message,
    )

    for key, value := range lf.fields {
        parts = append(parts, " ", key, "=", value)
    }

    return JoinEfficient(parts)
}

// HTML template builder
func BuildHTMLTable(rows [][]string) string {
    parts := []string{"<table>"}

    for _, row := range rows {
        parts = append(parts, "<tr>")
        for _, cell := range row {
            parts = append(parts, "<td>", cell, "</td>")
        }
        parts = append(parts, "</tr>")
    }

    parts = append(parts, "</table>")
    return JoinEfficient(parts)
}

// SQL query builder
type QueryBuilder struct {
    parts []string
}

func (qb *QueryBuilder) Build() string {
    return JoinEfficient(qb.parts)
}

func (qb *QueryBuilder) Select(columns ...string) *QueryBuilder {
    qb.parts = append(qb.parts, "SELECT", strings.Join(columns, ", "))
    return qb
}

func (qb *QueryBuilder) From(table string) *QueryBuilder {
    qb.parts = append(qb.parts, "FROM", table)
    return qb
}

// Path construction
func BuildFilePath(segments []string) string {
    parts := make([]string, 0, len(segments)*2-1)

    for i, segment := range segments {
        if i > 0 {
            parts = append(parts, "/")
        }
        parts = append(parts, segment)
    }

    return JoinEfficient(parts)
}

// Cache key generation from parts
func GenerateCacheKey(components ...string) string {
    parts := make([]string, 0, len(components)*2-1)

    for i, comp := range components {
        if i > 0 {
            parts = append(parts, ":")
        }
        parts = append(parts, comp)
    }

    return JoinEfficient(parts)
}

// Error message builder with context
type ErrorContext struct {
    operation string
    details   string
    code      string
}

func (ec *ErrorContext) Message() string {
    return JoinEfficient([]string{
        "Error in ", ec.operation,
        ": ", ec.details,
        " (code: ", ec.code, ")",
    })
}

// Protocol message construction
type ProtocolMessage struct {
    header  string
    payload string
    footer  string
}

func (pm *ProtocolMessage) Serialize() string {
    return JoinEfficient([]string{
        pm.header,
        pm.payload,
        pm.footer,
    })
}

// CSV record formatter
func FormatCSVRecord(fields []string) string {
    parts := make([]string, 0, len(fields)*2-1)

    for i, field := range fields {
        if i > 0 {
            parts = append(parts, ",")
        }
        // Escape quotes in field
        escaped := strings.ReplaceAll(field, "\"", "\"\"")
        parts = append(parts, "\"", escaped, "\"")
    }

    return JoinEfficient(parts)
}

// Log aggregation with batched messages
type LogBatch struct {
    entries []string
}

func (lb *LogBatch) Serialize() string {
    return JoinEfficient(lb.entries)
}

// URL query parameter builder
func BuildQueryString(params map[string]string) string {
    parts := make([]string, 0, len(params)*2-1)

    i := 0
    for key, value := range params {
        if i > 0 {
            parts = append(parts, "&")
        }
        parts = append(parts, key, "=", url.QueryEscape(value))
        i++
    }

    return "?" + JoinEfficient(parts)
}

// Markdown builder
func BuildMarkdownLink(text, url string) string {
    return JoinEfficient([]string{"[", text, "](", url, ")"})
}

func BuildMarkdownHeader(level int, text string) string {
    return JoinEfficient([]string{
        strings.Repeat("#", level),
        " ",
        text,
    })
}

// Request header builder
type HeaderBuilder struct {
    headers map[string]string
}

func (hb *HeaderBuilder) Build() string {
    parts := make([]string, 0)

    for key, value := range hb.headers {
        parts = append(parts,
            key, ": ", value, "\r\n",
        )
    }

    return JoinEfficient(parts)
}

// Configuration file generator
func GenerateConfigContent(config map[string]interface{}) string {
    parts := make([]string, 0)

    for key, value := range config {
        parts = append(parts,
            key, " = ", fmt.Sprint(value), "\n",
        )
    }

    return JoinEfficient(parts)
}
\`\`\`

**Real-World Benefits:**
- **API Response Generation:** Build JSON/XML responses without allocations
- **Logging:** Format large log messages with multiple fields efficiently
- **Query Building:** Construct SQL queries from parts without overhead
- **Template Rendering:** Generate HTML/text templates without intermediate strings
- **Network Protocols:** Assemble protocol messages efficiently

**Performance Comparison:**
- **String + in loop:** O(n²) time complexity, n allocations
- **strings.Builder:** O(n) time complexity, 1 allocation
- **Performance gain:** 100-1000x faster for large strings

**Benchmark Results (typical):**
- Concatenating 1000 strings with +: ~100ms
- Same with JoinEfficient: ~1ms
- Memory allocation difference: 100MB vs 1MB

**Common Use Cases:**
- JSON/XML response building
- SQL query construction
- Log message formatting
- URL building
- HTML template generation
- Protocol message assembly

Without JoinEfficient, building responses with thousands of string parts would cause OOM errors or massive latency spikes in production systems.`,	order: 5,
	translations: {
		ru: {
			title: 'Эффективная конкатенация строк',
			description: `Реализуйте **JoinEfficient**, который объединяет строки без использования string concatenation в цикле.

**Требования:**
1. Создайте функцию \`JoinEfficient(parts []string) string\`
2. Обработайте пустой слайс (верните пустую строку)
3. Обработайте single element слайс (верните этот элемент)
4. Используйте strings.Builder для эффективного объединения
5. Предварительно вычислите необходимый общий размер
6. Вызовите Grow() перед записью для предварительного выделения буфера
7. Напишите все части в builder
8. Верните финальную строку

**Пример:**
\`\`\`go
result := JoinEfficient([]string{"Hello", "World", "!"})
// result = "HelloWorld!"

result2 := JoinEfficient([]string{"a"})
// result2 = "a"

result3 := JoinEfficient([]string{})
// result3 = ""

result4 := JoinEfficient([]string{"foo", "bar", "baz", "qux"})
// result4 = "foobarbazqux"
\`\`\`

**Ограничения:**
- Не должен использовать string concatenation с + operator в цикле
- Должен корректно обработать пустой слайс
- Должен предварительно выделить буфер используя Grow()
- Должен вычислить общую byte длину необходимую
- Не должен использовать strings.Join (реализуйте паттерн самостоятельно)`,
			hint1: `Используйте switch statement для обработки 0, 1 и multiple случаев. Сначала вычислите общую byte длину, затем используйте builder.Grow().`,
			hint2: `strings.Builder с Grow() избегает переаллокаций. Каждый WriteString() добавляет без создания промежуточных строк.`,
			whyItMatters: `JoinEfficient демонстрирует критическую разницу в производительности между наивной string concatenation и эффективной буферизацией, особенно важно для high-throughput систем обрабатывающих тысячи строк.

**Почему Efficient String Building:**
- **Производительность:** Нет промежуточного выделения строк в цикле
- **Память:** Одно выделение вместо n выделений для n строк
- **GC Pressure:** Драматически снижает overhead garbage collection
- **Масштабируемость:** Необходимо для обработки миллионов строк
- **Responsiveness:** Устраняет microsecond задержки от повторных выделений

**Production Pattern:**
\`\`\`go
// JSON response builder для API
func BuildJSONResponse(fields map[string]string) string {
    var parts []string
    parts = append(parts, "{")

    first := true
    for key, value := range fields {
        if !first {
            parts = append(parts, ",")
        }
        parts = append(parts, fmt.Sprintf("\"%s\":\"%s\"", key, value))
        first = false
    }

    parts = append(parts, "}")
    return JoinEfficient(parts)
}

// Форматирование log сообщений с несколькими полями
type LogFormatter struct {
    timestamp string
    level     string
    message   string
    fields    map[string]string
}

func (lf *LogFormatter) Format() string {
    parts := make([]string, 0, 10)

    parts = append(parts,
        "[", lf.timestamp, "]",
        "[", lf.level, "]",
        " ", lf.message,
    )

    for key, value := range lf.fields {
        parts = append(parts, " ", key, "=", value)
    }

    return JoinEfficient(parts)
}

// HTML шаблонов builder
func BuildHTMLTable(rows [][]string) string {
    parts := []string{"<table>"}

    for _, row := range rows {
        parts = append(parts, "<tr>")
        for _, cell := range row {
            parts = append(parts, "<td>", cell, "</td>")
        }
        parts = append(parts, "</tr>")
    }

    parts = append(parts, "</table>")
    return JoinEfficient(parts)
}

// SQL query builder
type QueryBuilder struct {
    parts []string
}

func (qb *QueryBuilder) Build() string {
    return JoinEfficient(qb.parts)
}

func (qb *QueryBuilder) Select(columns ...string) *QueryBuilder {
    qb.parts = append(qb.parts, "SELECT", strings.Join(columns, ", "))
    return qb
}

func (qb *QueryBuilder) From(table string) *QueryBuilder {
    qb.parts = append(qb.parts, "FROM", table)
    return qb
}

// Построение путей
func BuildFilePath(segments []string) string {
    parts := make([]string, 0, len(segments)*2-1)

    for i, segment := range segments {
        if i > 0 {
            parts = append(parts, "/")
        }
        parts = append(parts, segment)
    }

    return JoinEfficient(parts)
}

// Генерация cache ключей из частей
func GenerateCacheKey(components ...string) string {
    parts := make([]string, 0, len(components)*2-1)

    for i, comp := range components {
        if i > 0 {
            parts = append(parts, ":")
        }
        parts = append(parts, comp)
    }

    return JoinEfficient(parts)
}

// Error message builder с контекстом
type ErrorContext struct {
    operation string
    details   string
    code      string
}

func (ec *ErrorContext) Message() string {
    return JoinEfficient([]string{
        "Error in ", ec.operation,
        ": ", ec.details,
        " (code: ", ec.code, ")",
    })
}

// Построение protocol сообщений
type ProtocolMessage struct {
    header  string
    payload string
    footer  string
}

func (pm *ProtocolMessage) Serialize() string {
    return JoinEfficient([]string{
        pm.header,
        pm.payload,
        pm.footer,
    })
}

// CSV record formatter
func FormatCSVRecord(fields []string) string {
    parts := make([]string, 0, len(fields)*2-1)

    for i, field := range fields {
        if i > 0 {
            parts = append(parts, ",")
        }
        // Экранирование кавычек в поле
        escaped := strings.ReplaceAll(field, "\"", "\"\"")
        parts = append(parts, "\"", escaped, "\"")
    }

    return JoinEfficient(parts)
}

// Log агрегация с batched сообщениями
type LogBatch struct {
    entries []string
}

func (lb *LogBatch) Serialize() string {
    return JoinEfficient(lb.entries)
}

// URL query параметров builder
func BuildQueryString(params map[string]string) string {
    parts := make([]string, 0, len(params)*2-1)

    i := 0
    for key, value := range params {
        if i > 0 {
            parts = append(parts, "&")
        }
        parts = append(parts, key, "=", url.QueryEscape(value))
        i++
    }

    return "?" + JoinEfficient(parts)
}

// Markdown builder
func BuildMarkdownLink(text, url string) string {
    return JoinEfficient([]string{"[", text, "](", url, ")"})
}

func BuildMarkdownHeader(level int, text string) string {
    return JoinEfficient([]string{
        strings.Repeat("#", level),
        " ",
        text,
    })
}

// Request header builder
type HeaderBuilder struct {
    headers map[string]string
}

func (hb *HeaderBuilder) Build() string {
    parts := make([]string, 0)

    for key, value := range hb.headers {
        parts = append(parts,
            key, ": ", value, "\r\n",
        )
    }

    return JoinEfficient(parts)
}

// Configuration файлов генератор
func GenerateConfigContent(config map[string]interface{}) string {
    parts := make([]string, 0)

    for key, value := range config {
        parts = append(parts,
            key, " = ", fmt.Sprint(value), "\n",
        )
    }

    return JoinEfficient(parts)
}
\`\`\`

**Практические преимущества:**
- **Генерация API ответов:** Построение JSON/XML ответов без выделений памяти
- **Логирование:** Форматирование больших log сообщений с несколькими полями эффективно
- **Построение запросов:** Создание SQL запросов из частей без overhead
- **Рендеринг шаблонов:** Генерация HTML/text шаблонов без промежуточных строк
- **Сетевые протоколы:** Эффективная сборка protocol сообщений

**Сравнение производительности:**
- **String + в цикле:** O(n²) временная сложность, n выделений
- **strings.Builder:** O(n) временная сложность, 1 выделение
- **Выигрыш производительности:** 100-1000x быстрее для больших строк

**Результаты бенчмарков (типичные):**
- Конкатенация 1000 строк с +: ~100ms
- То же с JoinEfficient: ~1ms
- Разница выделения памяти: 100MB vs 1MB

**Частые случаи использования:**
- Построение JSON/XML ответов
- Создание SQL запросов
- Форматирование log сообщений
- Построение URL
- Генерация HTML шаблонов
- Сборка protocol сообщений

Без JoinEfficient построение ответов с тысячами строковых частей вызвало бы OOM ошибки или массивные latency спайки в production системах.`,
			solutionCode: `package datastructures

import "strings"

func JoinEfficient(parts []string) string {
	switch len(parts) {                                     // Switch по длине слайса
	case 0:                                                 // Пустой слайс
		return ""                                       // Вернуть пустую строку
	case 1:                                                 // Один элемент
		return parts[0]                                 // Вернуть этот элемент
	}
	totalLen := 0                                           // Вычислить общую длину
	for _, part := range parts {                            // Суммировать все длины частей
		totalLen += len(part)                           // Добавить длину каждой части
	}
	var builder strings.Builder                             // Создать builder
	builder.Grow(totalLen)                                  // Предварительно выделить точный размер буфера
	for _, part := range parts {                            // Итерация по частям
		builder.WriteString(part)                       // Записать каждую часть
	}
	return builder.String()                                 // Вернуть объединённый результат
}`
		},
		uz: {
			title: 'Samarali stringlarni birlashtirish',
			description: `String concatenation dan foydalanib string concat qilmaydigan **JoinEfficient** ni amalga oshiring.

**Talablar:**
1. \`JoinEfficient(parts []string) string\` funksiyasini yarating
2. Bo'sh slaysni ishlang (bo'sh string qaytaring)
3. Single element slaysni ishlang (o'sha elementni qaytaring)
4. Samarali concatenation uchun strings.Builder dan foydalaning
5. Zarur bo'lgan jami o'lchamni oldindan hisoblang
6. Bufferni oldindan ajratib qo'yish uchun Grow() ni chaqiring
7. Barcha qismlarni builderga yozing
8. Final stringni qaytaring

**Misol:**
\`\`\`go
result := JoinEfficient([]string{"Hello", "World", "!"})
// result = "HelloWorld!"

result2 := JoinEfficient([]string{"a"})
// result2 = "a"

result3 := JoinEfficient([]string{})
// result3 = ""

result4 := JoinEfficient([]string{"foo", "bar", "baz", "qux"})
// result4 = "foobarbazqux"
\`\`\`

**Cheklovlar:**
- Loopda + operatori bilan string concatenation ishlatmasligi kerak
- Bo'sh slaysni to'g'ri ishlashi kerak
- Grow() dan foydalanib bufferni oldindan ajratib qo'yishi kerak
- Zarur bo'lgan jami byte uzunligini hisoblashi kerak
- strings.Join dan foydalanmasligi kerak (patternni o'zingiz amalga oshiring)`,
			hint1: `0, 1 va multiple caslarni ishlash uchun switch statementdan foydalaning. Birinchi jami byte uzunligini hisoblang, keyin builder.Grow() dan foydalaning.`,
			hint2: `strings.Builder Grow() bilan reallokatsiyalardan qochadi. Har bir WriteString() oraliq stringlarni yaratmasdan qo'shadi.`,
			whyItMatters: `JoinEfficient naiviy string concatenation va samarali bufferizatsiya orasidagi kritik samaradorlik farqini ko'rsatadi, ayniqsa ko'p stringlarni qayta ishlash high-throughput tizimlari uchun muhim.

**Nima uchun Efficient String Building:**
- **Samaradorlik:** Loopda oraliq stringlarni ajratib qo'yish yo'q
- **Xotira:** n stringlar uchun n ajratilishi o'rniga bitta ajratilish
- **GC Pressure:** Garbage collection overheadni dramatik kamaytiradi
- **Scalability:** Milyonlab stringlarni qayta ishlash uchun zarur
- **Responsiveness:** Repeated ajratilishlardan microsecond kechikishni olib tashlaydi

**Production Pattern:**
\`\`\`go
// API uchun JSON response builder
func BuildJSONResponse(fields map[string]string) string {
    var parts []string
    parts = append(parts, "{")

    first := true
    for key, value := range fields {
        if !first {
            parts = append(parts, ",")
        }
        parts = append(parts, fmt.Sprintf("\"%s\":\"%s\"", key, value))
        first = false
    }

    parts = append(parts, "}")
    return JoinEfficient(parts)
}

// Ko'p maydonlar bilan log xabarlarini formatlash
type LogFormatter struct {
    timestamp string
    level     string
    message   string
    fields    map[string]string
}

func (lf *LogFormatter) Format() string {
    parts := make([]string, 0, 10)

    parts = append(parts,
        "[", lf.timestamp, "]",
        "[", lf.level, "]",
        " ", lf.message,
    )

    for key, value := range lf.fields {
        parts = append(parts, " ", key, "=", value)
    }

    return JoinEfficient(parts)
}

// HTML shablonlar builder
func BuildHTMLTable(rows [][]string) string {
    parts := []string{"<table>"}

    for _, row := range rows {
        parts = append(parts, "<tr>")
        for _, cell := range row {
            parts = append(parts, "<td>", cell, "</td>")
        }
        parts = append(parts, "</tr>")
    }

    parts = append(parts, "</table>")
    return JoinEfficient(parts)
}

// SQL query builder
type QueryBuilder struct {
    parts []string
}

func (qb *QueryBuilder) Build() string {
    return JoinEfficient(qb.parts)
}

func (qb *QueryBuilder) Select(columns ...string) *QueryBuilder {
    qb.parts = append(qb.parts, "SELECT", strings.Join(columns, ", "))
    return qb
}

func (qb *QueryBuilder) From(table string) *QueryBuilder {
    qb.parts = append(qb.parts, "FROM", table)
    return qb
}

// Yo'l qurilishi
func BuildFilePath(segments []string) string {
    parts := make([]string, 0, len(segments)*2-1)

    for i, segment := range segments {
        if i > 0 {
            parts = append(parts, "/")
        }
        parts = append(parts, segment)
    }

    return JoinEfficient(parts)
}

// Qismlardan cache kalitlarini yaratish
func GenerateCacheKey(components ...string) string {
    parts := make([]string, 0, len(components)*2-1)

    for i, comp := range components {
        if i > 0 {
            parts = append(parts, ":")
        }
        parts = append(parts, comp)
    }

    return JoinEfficient(parts)
}

// Kontekst bilan error message builder
type ErrorContext struct {
    operation string
    details   string
    code      string
}

func (ec *ErrorContext) Message() string {
    return JoinEfficient([]string{
        "Error in ", ec.operation,
        ": ", ec.details,
        " (code: ", ec.code, ")",
    })
}

// Protocol xabarlarini qurilishi
type ProtocolMessage struct {
    header  string
    payload string
    footer  string
}

func (pm *ProtocolMessage) Serialize() string {
    return JoinEfficient([]string{
        pm.header,
        pm.payload,
        pm.footer,
    })
}

// CSV record formatter
func FormatCSVRecord(fields []string) string {
    parts := make([]string, 0, len(fields)*2-1)

    for i, field := range fields {
        if i > 0 {
            parts = append(parts, ",")
        }
        // Maydon ichidagi qo'shtirnoqlarni escape qilish
        escaped := strings.ReplaceAll(field, "\"", "\"\"")
        parts = append(parts, "\"", escaped, "\"")
    }

    return JoinEfficient(parts)
}

// Batched xabarlar bilan log agregatsiya
type LogBatch struct {
    entries []string
}

func (lb *LogBatch) Serialize() string {
    return JoinEfficient(lb.entries)
}

// URL query parametrlari builder
func BuildQueryString(params map[string]string) string {
    parts := make([]string, 0, len(params)*2-1)

    i := 0
    for key, value := range params {
        if i > 0 {
            parts = append(parts, "&")
        }
        parts = append(parts, key, "=", url.QueryEscape(value))
        i++
    }

    return "?" + JoinEfficient(parts)
}

// Markdown builder
func BuildMarkdownLink(text, url string) string {
    return JoinEfficient([]string{"[", text, "](", url, ")"})
}

func BuildMarkdownHeader(level int, text string) string {
    return JoinEfficient([]string{
        strings.Repeat("#", level),
        " ",
        text,
    })
}

// Request header builder
type HeaderBuilder struct {
    headers map[string]string
}

func (hb *HeaderBuilder) Build() string {
    parts := make([]string, 0)

    for key, value := range hb.headers {
        parts = append(parts,
            key, ": ", value, "\r\n",
        )
    }

    return JoinEfficient(parts)
}

// Configuration fayllari generatori
func GenerateConfigContent(config map[string]interface{}) string {
    parts := make([]string, 0)

    for key, value := range config {
        parts = append(parts,
            key, " = ", fmt.Sprint(value), "\n",
        )
    }

    return JoinEfficient(parts)
}
\`\`\`

**Amaliy afzalliklar:**
- **API javoblarini yaratish:** Xotira ajratilmasdan JSON/XML javoblarini qurilishi
- **Loglar:** Ko'p maydonlar bilan katta log xabarlarini samarali formatlash
- **So'rovlar qurilishi:** Qismlardan SQL so'rovlarni overhead siz yaratish
- **Shablonlarni render qilish:** Oraliq stringlar siz HTML/text shablonlarini yaratish
- **Tarmoq protokollari:** Protocol xabarlarini samarali yig'ish

**Samaradorlik taqqoslash:**
- **Loopda String +:** O(n²) vaqt murakkabligi, n ta ajratilish
- **strings.Builder:** O(n) vaqt murakkabligi, 1 ta ajratilish
- **Samaradorlik yutuq:** Katta stringlar uchun 100-1000x tezroq

**Benchmark natijalari (odatiy):**
- 1000 ta stringni + bilan birlashtirish: ~100ms
- JoinEfficient bilan xuddi shunday: ~1ms
- Xotira ajratilish farqi: 100MB vs 1MB

**Umumiy foydalanish holatlari:**
- JSON/XML javoblarini qurilishi
- SQL so'rovlar yaratish
- Log xabarlarini formatlash
- URL qurilishi
- HTML shablonlar yaratish
- Protocol xabarlari yig'ish

JoinEfficient siz, minglab string qismlar bilan javoblar qurilishi production tizimlarda OOM xatolar yoki katta latency spiklariga olib kelgan bo'lar edi.`,
			solutionCode: `package datastructures

import "strings"

func JoinEfficient(parts []string) string {
	switch len(parts) {                                     // Slayz uzunligi bo'yicha switch
	case 0:                                                 // Bo'sh slayz
		return ""                                       // Bo'sh string qaytarish
	case 1:                                                 // Bitta element
		return parts[0]                                 // O'sha elementni qaytarish
	}
	totalLen := 0                                           // Jami uzunlikni hisoblash
	for _, part := range parts {                            // Barcha qismlar uzunliklarini jamlash
		totalLen += len(part)                           // Har bir qism uzunligini qo'shish
	}
	var builder strings.Builder                             // Builder yaratish
	builder.Grow(totalLen)                                  // Aniq bufer o'lchamini oldindan ajratish
	for _, part := range parts {                            // Qismlar bo'ylab iteratsiya
		builder.WriteString(part)                       // Har bir qismni yozish
	}
	return builder.String()                                 // Birlashtirilgan natijani qaytarish
}`
		}
	}
};

export default task;
