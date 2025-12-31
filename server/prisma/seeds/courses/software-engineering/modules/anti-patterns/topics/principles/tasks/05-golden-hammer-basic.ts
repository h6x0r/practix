import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-ap-golden-hammer-basic',
	title: 'Golden Hammer Anti-pattern - Basic',
	difficulty: 'easy',
	tags: ['go', 'anti-patterns', 'golden-hammer', 'refactoring'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Recognize and fix the Golden Hammer anti-pattern by choosing the right tool for each job.

**The Problem:**

Golden Hammer is when you use the same familiar solution for every problem, even when it's not appropriate. "If all you have is a hammer, everything looks like a nail."

**The Scenario:**

A developer who loves regular expressions uses them for EVERYTHING, even simple string operations.

**You will implement proper solutions:**
1. **IsValidEmail** - Email validation (regex is appropriate)
2. **ExtractNumbers** - Extract numbers from string (simple parsing)
3. **ReverseString** - Reverse a string (built-in functions)

**Your Task:**

Use the right tool for each job - regex when needed, simple operations when appropriate.`,
	initialCode: `package antipatterns

import (
	"regexp"
	"strconv"
	"strings"
)

func IsValidEmail(email string) bool {
}

func ExtractNumbers(s string) []int {
}

func ReverseString(s string) string {
}`,
	solutionCode: `package antipatterns

import (
	"regexp"
	"strconv"
	"strings"
)

// IsValidEmail validates email using regex - APPROPRIATE use
// Regex is the right tool for pattern matching
func IsValidEmail(email string) bool {
	pattern := ` + "`^[^@]+@[^@]+\\.[^@]+$`" + `	// simple email pattern
	matched, _ := regexp.MatchString(pattern, email)
	return matched
}

// ExtractNumbers extracts numbers using simple string operations
// NO regex needed - simple parsing is clearer and faster
func ExtractNumbers(s string) []int {
	var numbers []int
	fields := strings.Fields(s)	// split by whitespace

	for _, field := range fields {
		if num, err := strconv.Atoi(field); err == nil {
			numbers = append(numbers, num)	// successfully parsed as number
		}
	}

	return numbers	// returns empty slice if no numbers
}

// ReverseString reverses using built-in rune manipulation
// NO regex needed - simple slice operations are clearer
func ReverseString(s string) string {
	runes := []rune(s)	// convert to rune slice for proper Unicode handling

	// Reverse the slice in place
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]	// swap
	}

	return string(runes)	// convert back to string
}`,
	hint1: `IsValidEmail: use regexp.MatchString with pattern "^[^@]+@[^@]+\\.[^@]+$". ExtractNumbers: use strings.Fields to split, then strconv.Atoi to parse each field.`,
	hint2: `ReverseString: convert to []rune, use two-pointer approach to swap elements (i from start, j from end), convert back to string.`,
	whyItMatters: `Using the wrong tool for the job leads to overcomplicated, slow, and hard-to-maintain code.

**Why Golden Hammer is Problematic:**

**1. Performance Problems**

\`\`\`go
// BAD: Using regex for simple checks
func StartsWith(s, prefix string) bool {
	pattern := "^" + regexp.QuoteMeta(prefix)
	matched, _ := regexp.MatchString(pattern, s)
	return matched
}
// Regex compilation is SLOW!

// GOOD: Use built-in functions
func StartsWith(s, prefix string) bool {
	return strings.HasPrefix(s, prefix)
}
// 100x faster!
\`\`\`

**2. Overcomplicated Code**

\`\`\`go
// BAD: Using regex to parse simple CSV
func ParseCSV(line string) []string {
	re := regexp.MustCompile(` + "`" + `[^,]+` + "`" + `)
	return re.FindAllString(line, -1)
}
// Breaks on quotes, spaces, etc.

// GOOD: Use csv package
import "encoding/csv"

func ParseCSV(line string) []string {
	reader := csv.NewReader(strings.NewReader(line))
	record, _ := reader.Read()
	return record
}
// Handles edge cases correctly!
\`\`\`

**Real-World Examples:**

**Example 1: Database for Everything**

\`\`\`go
// BAD: Using database for caching (Golden Hammer: PostgreSQL)
type Cache struct {
	db *sql.DB
}

func (c *Cache) Get(key string) (string, error) {
	var value string
	err := c.db.QueryRow("SELECT value FROM cache WHERE key = $1", key).Scan(&value)
	return value, err
}
// Every cache read hits the database - SLOW!

// GOOD: Use in-memory cache
type Cache struct {
	data sync.Map  // concurrent map
}

func (c *Cache) Get(key string) (string, bool) {
	val, ok := c.data.Load(key)
	if !ok {
		return "", false
	}
	return val.(string), true
}
// Nanosecond access time vs milliseconds!
\`\`\`

**Example 2: JSON for Everything**

\`\`\`go
// BAD: Using JSON for configuration (Golden Hammer: JSON)
type Config struct {
	DatabaseURL string ` + "`json:\"database_url\"`" + `
	// Cannot have comments
	// Cannot have multi-line strings
	// No environment variable support
}
// JSON has no comments or variables!

// GOOD: Use YAML or TOML for config
# config.yaml
database:
  url: \${DATABASE_URL}  # environment variable
  max_connections: 25   # this is a comment
  connection_string: |  # multi-line string
    host=localhost
    port=5432
\`\`\`

**Example 3: Goroutines for Everything**

\`\`\`go
// BAD: Using goroutines unnecessarily
func Add(a, b int) int {
	result := make(chan int)
	go func() {
		result <- a + b  // goroutine overhead for simple addition!
	}()
	return <-result
}
// Goroutine + channel overhead is 1000x slower than addition!

// GOOD: Use goroutines for I/O or parallelism
func FetchURLs(urls []string) []Response {
	results := make(chan Response, len(urls))

	for _, url := range urls {
		go func(u string) {
			resp, _ := http.Get(u)  // parallel network I/O
			results <- processResponse(resp)
		}(url)
	}

	// Collect results...
}
// Appropriate use: parallel I/O operations
\`\`\`

**How to Avoid Golden Hammer:**

**1. Learn Multiple Tools**
- String manipulation: strings package
- Pattern matching: regex
- Data serialization: JSON, Protocol Buffers, MessagePack
- Caching: sync.Map, Redis, Memcached
- Storage: Files, SQLite, PostgreSQL, MongoDB

**2. Choose Based on Requirements**

\`\`\`go
// Small config file? Use JSON or YAML
type Config struct { ... }
json.Unmarshal(data, &config)

// Large datasets? Use binary format
protobuf.Unmarshal(data, &message)

// Need caching?
// - Small data: sync.Map
// - Large data, single server: Redis
// - Distributed: Memcached/Redis Cluster
\`\`\`

**3. Measure Performance**

\`\`\`go
// Benchmark different approaches
func BenchmarkRegex(b *testing.B) {
	for i := 0; i < b.N; i++ {
		regexp.MatchString("^hello", "hello world")
	}
}

func BenchmarkHasPrefix(b *testing.B) {
	for i := 0; i < b.N; i++ {
		strings.HasPrefix("hello world", "hello")
	}
}
// HasPrefix is 100x faster!
\`\`\`

**Common Golden Hammers:**

1. **Regex for everything** - Use strings package for simple ops
2. **Microservices for everything** - Monolith is fine for small apps
3. **NoSQL for everything** - SQL databases are powerful
4. **Channels for everything** - Mutex is often simpler
5. **Reflection for everything** - Code generation is clearer
6. **Dependency injection everywhere** - Simple constructors work too

**The Right Mindset:**

> "When all you have is a hammer, everything looks like a nail."
> "When you have many tools, you can choose the right one."

Each tool has its place - learn when to use what!`,
	order: 4,
	testCode: `package antipatterns

import (
	"testing"
)

// Test1: Valid email with @ and .
func Test1(t *testing.T) {
	if !IsValidEmail("user@example.com") {
		t.Error("user@example.com should be valid")
	}
}

// Test2: Invalid email without @
func Test2(t *testing.T) {
	if IsValidEmail("userexample.com") {
		t.Error("userexample.com should be invalid (no @)")
	}
}

// Test3: Invalid email without .
func Test3(t *testing.T) {
	if IsValidEmail("user@examplecom") {
		t.Error("user@examplecom should be invalid (no .)")
	}
}

// Test4: ExtractNumbers from sentence
func Test4(t *testing.T) {
	nums := ExtractNumbers("I have 3 cats and 2 dogs")
	if len(nums) != 2 || nums[0] != 3 || nums[1] != 2 {
		t.Errorf("Expected [3, 2], got %v", nums)
	}
}

// Test5: ExtractNumbers with no numbers
func Test5(t *testing.T) {
	nums := ExtractNumbers("hello world")
	if len(nums) != 0 {
		t.Errorf("Expected empty slice, got %v", nums)
	}
}

// Test6: ReverseString simple
func Test6(t *testing.T) {
	if ReverseString("hello") != "olleh" {
		t.Error("Expected 'olleh'")
	}
}

// Test7: ReverseString empty
func Test7(t *testing.T) {
	if ReverseString("") != "" {
		t.Error("Expected empty string")
	}
}

// Test8: ReverseString single char
func Test8(t *testing.T) {
	if ReverseString("a") != "a" {
		t.Error("Expected 'a'")
	}
}

// Test9: ReverseString palindrome
func Test9(t *testing.T) {
	if ReverseString("radar") != "radar" {
		t.Error("Expected 'radar'")
	}
}

// Test10: ExtractNumbers single number
func Test10(t *testing.T) {
	nums := ExtractNumbers("age 25")
	if len(nums) != 1 || nums[0] != 25 {
		t.Errorf("Expected [25], got %v", nums)
	}
}
`,
	translations: {
		ru: {
			title: 'Антипаттерн Golden Hammer - Базовый',
			description: `Распознайте и исправьте антипаттерн Golden Hammer, выбрав правильный инструмент для каждой задачи.

**Проблема:**

Golden Hammer — это когда вы используете одно и то же знакомое решение для каждой проблемы, даже когда оно не подходит. "Если у вас есть только молоток, всё выглядит как гвоздь."

**Сценарий:**

Разработчик, который любит регулярные выражения, использует их для ВСЕГО, даже для простых строковых операций.

**Вы реализуете правильные решения:**
1. **IsValidEmail** - Валидация email (regex уместен)
2. **ExtractNumbers** - Извлечение чисел из строки (простой парсинг)
3. **ReverseString** - Переворот строки (встроенные функции)`,
			hint1: `IsValidEmail: используйте regexp.MatchString с паттерном "^[^@]+@[^@]+\\.[^@]+$". ExtractNumbers: используйте strings.Fields для разделения, затем strconv.Atoi для парсинга каждого поля.`,
			hint2: `ReverseString: конвертируйте в []rune, используйте подход с двумя указателями для обмена элементами (i с начала, j с конца), конвертируйте обратно в string.`,
			whyItMatters: `Использование неправильного инструмента приводит к усложнённому, медленному и трудно поддерживаемому коду.`,
			solutionCode: `package antipatterns

import (
	"regexp"
	"strconv"
	"strings"
)

func IsValidEmail(email string) bool {
	pattern := ` + "`^[^@]+@[^@]+\\.[^@]+$`" + `
	matched, _ := regexp.MatchString(pattern, email)
	return matched
}

func ExtractNumbers(s string) []int {
	var numbers []int
	fields := strings.Fields(s)

	for _, field := range fields {
		if num, err := strconv.Atoi(field); err == nil {
			numbers = append(numbers, num)
		}
	}

	return numbers
}

func ReverseString(s string) string {
	runes := []rune(s)

	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}

	return string(runes)
}`
		},
		uz: {
			title: 'Golden Hammer Anti-pattern - Asosiy',
			description: `Golden Hammer anti-patternini tanib, har bir ish uchun to'g'ri vositani tanlab tuzating.

**Muammo:**

Golden Hammer — bu har bir muammo uchun bir xil tanish yechimni ishlatish, hatto u mos bo'lmasa ham. "Agar sizda faqat bolg'a bo'lsa, hamma narsa mix kabi ko'rinadi."

**Stsenariy:**

Muntazam ifodalarni yaxshi ko'radigan dasturchi ularni HAMMA NARSA uchun ishlatadi, hatto oddiy string operatsiyalari uchun ham.

**Siz to'g'ri yechimlarni amalga oshirasiz:**
1. **IsValidEmail** - Email validatsiyasi (regex mos)
2. **ExtractNumbers** - Stringdan raqamlarni chiqarish (oddiy parsing)
3. **ReverseString** - Stringni teskari aylantirish (o'rnatilgan funksiyalar)`,
			hint1: `IsValidEmail: regexp.MatchString dan "^[^@]+@[^@]+\\.[^@]+$" pattern bilan foydalaning. ExtractNumbers: bo'lish uchun strings.Fields, keyin har bir maydonni parsing qilish uchun strconv.Atoi ishlating.`,
			hint2: `ReverseString: []rune ga o'giring, elementlarni almashtirish uchun ikki ko'rsatgich yondashuvini ishlating (boshidan i, oxiridan j), string ga qaytaring.`,
			whyItMatters: `Ish uchun noto'g'ri vositani ishlatish murakkablashgan, sekin va qo'llab-quvvatlash qiyin bo'lgan kodga olib keladi.`,
			solutionCode: `package antipatterns

import (
	"regexp"
	"strconv"
	"strings"
)

func IsValidEmail(email string) bool {
	pattern := ` + "`^[^@]+@[^@]+\\.[^@]+$`" + `
	matched, _ := regexp.MatchString(pattern, email)
	return matched
}

func ExtractNumbers(s string) []int {
	var numbers []int
	fields := strings.Fields(s)

	for _, field := range fields {
		if num, err := strconv.Atoi(field); err == nil {
			numbers = append(numbers, num)
		}
	}

	return numbers
}

func ReverseString(s string) string {
	runes := []rune(s)

	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}

	return string(runes)
}`
		}
	}
};

export default task;
