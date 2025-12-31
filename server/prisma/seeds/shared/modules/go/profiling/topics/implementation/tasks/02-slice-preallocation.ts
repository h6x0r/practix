import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-profiling-slice-prealloc',
	title: 'Slice Pre-allocation for Performance',
	difficulty: 'easy',	tags: ['go', 'profiling', 'performance', 'memory'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Optimize slice allocation by pre-allocating capacity to avoid repeated memory reallocations.

**Requirements:**
1. **BetterAlloc**: Create slice with known capacity upfront
2. **Use make()**: Pre-allocate with make([]int, n) or make([]int, 0, n)
3. **Avoid growth**: Prevent automatic slice growth and copying
4. **Return slice**: Return properly sized slice

**Naive Approach (Slow):**
\`\`\`go
func NaiveAlloc(n int) []int {
    var out []int
    for i := 0; i < n; i++ {
        out = append(out, rand.Intn(1000))
        // Slice grows: 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128...
        // Each growth copies ALL elements to new array
    }
    return out
}
// For n=1000: 10+ reallocations, multiple copies
\`\`\`

**Optimized Approach:**
\`\`\`go
func BetterAlloc(n int) []int {
    // Pre-allocate exactly what we need
    out := make([]int, n)

    // Fill directly by index (no append needed)
    for i := 0; i < n; i++ {
        out[i] = rand.Intn(1000)
    }

    return out
}
// For n=1000: 1 allocation, no copying, 10x faster
\`\`\`

**Alternative with append:**
\`\`\`go
func BetterAlloc(n int) []int {
    // Pre-allocate capacity, start with length 0
    out := make([]int, 0, n)

    for i := 0; i < n; i++ {
        out = append(out, rand.Intn(1000))
        // No reallocation - capacity already sufficient
    }

    return out
}
\`\`\`

**Key Concepts:**
- Slices have length (len) and capacity (cap)
- append() grows capacity when len == cap
- Growth strategy: double capacity each time (1→2→4→8...)
- Each growth requires new allocation + copying all elements
- Pre-allocation eliminates growth and copying
- make([]T, n) creates length=n, capacity=n
- make([]T, 0, n) creates length=0, capacity=n

**Benchmark Results:**
\`\`\`bash
go test -bench=Alloc -benchmem

BenchmarkAlloc_Naive-8      50000    30000 ns/op    24576 B/op    10 allocs/op
BenchmarkAlloc_Better-8    500000     3000 ns/op     8192 B/op     1 allocs/op

# Better is 10x faster with 90% less memory and 1 allocation!
\`\`\`

**Example Usage:**
\`\`\`go
// Reading database results
func FetchUsers(db *sql.DB) []User {
    rows, _ := db.Query("SELECT COUNT(*) FROM users")
    var count int
    rows.Scan(&count)

    // Pre-allocate exact size
    users := make([]User, 0, count)

    rows, _ = db.Query("SELECT * FROM users")
    for rows.Next() {
        var u User
        rows.Scan(&u.ID, &u.Name)
        users = append(users, u)  // No reallocation!
    }
    return users
}

// Processing batch data
func ProcessBatch(items []Item) []Result {
    // Know result size in advance
    results := make([]Result, len(items))

    for i, item := range items {
        results[i] = process(item)  // Direct assignment
    }
    return results
}

// Building response array
func BuildResponse(ids []int) []Response {
    // Pre-allocate for known size
    responses := make([]Response, 0, len(ids))

    for _, id := range ids {
        resp := fetch(id)
        responses = append(responses, resp)
    }
    return responses
}
\`\`\`

**When to Pre-allocate:**
1. **Known size in advance** - use make([]T, n)
2. **Approximate size** - use make([]T, 0, estimate)
3. **Transform 1:1** - output size = input size
4. **Database results** - COUNT query first
5. **Batch processing** - size known from input

**Constraints:**
- Must use make() for pre-allocation
- Choose make([]int, n) for direct assignment
- Choose make([]int, 0, n) for append pattern
- Final slice must contain n elements`,
	initialCode: `package profilingx

import (
	"math/rand"
)

// TODO: Implement BetterAlloc
// Pre-allocate slice with make()
// Fill slice with random numbers
// Return properly sized slice
func BetterAlloc(n int) []int {
	// TODO: Implement
}`,
	solutionCode: `package profilingx

import (
	"math/rand"
)

func BetterAlloc(n int) []int {
	out := make([]int, n)                       // pre-allocate exact capacity needed
	for i := 0; i < n; i++ {                    // fill slice by index assignment
		out[i] = rand.Intn(1000)                 // no append, no reallocation, direct write
	}
	return out                                  // return fully populated slice
}`,
			hint1: `Use make([]int, n) to create a slice with length n and capacity n.`,
			hint2: `Fill the slice using index assignment: out[i] = rand.Intn(1000) instead of append.`,
			testCode: `package profilingx

import (
	"testing"
)

func Test1(t *testing.T) {
	// Test n=0
	result := BetterAlloc(0)
	if len(result) != 0 {
		t.Errorf("BetterAlloc(0) len = %d, want 0", len(result))
	}
}

func Test2(t *testing.T) {
	// Test n=1
	result := BetterAlloc(1)
	if len(result) != 1 {
		t.Errorf("BetterAlloc(1) len = %d, want 1", len(result))
	}
}

func Test3(t *testing.T) {
	// Test n=10
	result := BetterAlloc(10)
	if len(result) != 10 {
		t.Errorf("BetterAlloc(10) len = %d, want 10", len(result))
	}
}

func Test4(t *testing.T) {
	// Test n=100
	result := BetterAlloc(100)
	if len(result) != 100 {
		t.Errorf("BetterAlloc(100) len = %d, want 100", len(result))
	}
}

func Test5(t *testing.T) {
	// Test values are within range [0, 1000)
	result := BetterAlloc(50)
	for i, v := range result {
		if v < 0 || v >= 1000 {
			t.Errorf("BetterAlloc(50)[%d] = %d, want 0 <= v < 1000", i, v)
		}
	}
}

func Test6(t *testing.T) {
	// Test capacity equals length (efficient allocation)
	result := BetterAlloc(100)
	if cap(result) < 100 {
		t.Errorf("BetterAlloc(100) cap = %d, want >= 100", cap(result))
	}
}

func Test7(t *testing.T) {
	// Test large allocation
	result := BetterAlloc(1000)
	if len(result) != 1000 {
		t.Errorf("BetterAlloc(1000) len = %d, want 1000", len(result))
	}
}

func Test8(t *testing.T) {
	// Test returns different values (random)
	result := BetterAlloc(100)
	allSame := true
	first := result[0]
	for _, v := range result {
		if v != first {
			allSame = false
			break
		}
	}
	if allSame && len(result) > 1 {
		t.Error("BetterAlloc returns all same values, expected random")
	}
}

func Test9(t *testing.T) {
	// Test consistency of length
	for n := 0; n <= 50; n++ {
		result := BetterAlloc(n)
		if len(result) != n {
			t.Errorf("BetterAlloc(%d) len = %d, want %d", n, len(result), n)
		}
	}
}

func Test10(t *testing.T) {
	// Test n=5 specific
	result := BetterAlloc(5)
	if len(result) != 5 {
		t.Errorf("BetterAlloc(5) len = %d, want 5", len(result))
	}
	for i, v := range result {
		if v < 0 || v >= 1000 {
			t.Errorf("BetterAlloc(5)[%d] = %d, out of range", i, v)
		}
	}
}`,
			whyItMatters: `Slice pre-allocation is one of the most impactful Go performance optimizations you can make.

**Why This Matters:**

**1. The Hidden Cost of Slice Growth**
Go doubles slice capacity on each growth:
\`\`\`go
// What happens with append on empty slice:
var s []int
s = append(s, 1)    // Allocate [1]       cap=1
s = append(s, 2)    // Allocate [1,2]     cap=2, copy 1 element
s = append(s, 3)    // Allocate [1,2,3,_] cap=4, copy 2 elements
s = append(s, 4)    // No allocation      cap=4
s = append(s, 5)    // Allocate [...]     cap=8, copy 4 elements

// For 1000 elements:
// - 10 allocations
// - 1023 elements copied in total
// - Wasted capacity: ~1024 - 1000 = 24 elements
\`\`\`

**2. Real Production Scenario: API Response Building**
REST API returns array of products:
\`\`\`go
// BEFORE - No pre-allocation
func GetProducts(category string) []Product {
    var products []Product  // cap=0

    rows, _ := db.Query("SELECT * FROM products WHERE category = ?", category)
    for rows.Next() {
        var p Product
        rows.Scan(&p.ID, &p.Name, &p.Price)
        products = append(products, p)
        // Growing: 0→1→2→4→8→16→32→64→128→256→512→1024
        // For 1000 products: 11 reallocations!
    }
    return products
}

// Response time: 45ms per request
// Memory: 85MB for 10k products
// GC pressure: constant

// AFTER - Pre-allocation with COUNT
func GetProducts(category string) []Product {
    var count int
    db.QueryRow("SELECT COUNT(*) FROM products WHERE category = ?", category).Scan(&count)

    products := make([]Product, 0, count)  // Pre-allocate exact size

    rows, _ := db.Query("SELECT * FROM products WHERE category = ?", category)
    for rows.Next() {
        var p Product
        rows.Scan(&p.ID, &p.Name, &p.Price)
        products = append(products, p)  // No reallocation!
    }
    return products
}

// Response time: 8ms per request (5.6x faster!)
// Memory: 40MB for 10k products (53% reduction)
// GC pressure: minimal
\`\`\`

**3. Batch Processing**
Processing files in chunks:
\`\`\`go
// BEFORE - Growing slice
func ProcessFile(filename string) []Record {
    file, _ := os.Open(filename)
    scanner := bufio.NewScanner(file)

    var records []Record
    for scanner.Scan() {
        line := scanner.Text()
        records = append(records, parseLine(line))
        // Constant reallocations for large files
    }
    return records
}
// 1M lines: 21 reallocations, 1M copies, 120 seconds

// AFTER - Pre-allocated
func ProcessFile(filename string) []Record {
    // Count lines first
    lineCount := countLines(filename)

    file, _ := os.Open(filename)
    scanner := bufio.NewScanner(file)

    records := make([]Record, 0, lineCount)
    for scanner.Scan() {
        line := scanner.Text()
        records = append(records, parseLine(line))
    }
    return records
}
// 1M lines: 1 allocation, 0 copies, 45 seconds (2.7x faster!)
\`\`\`

**4. Data Transformation**
Transforming one slice to another:
\`\`\`go
// BEFORE
func TransformIDs(userIDs []int64) []string {
    var stringIDs []string
    for _, id := range userIDs {
        stringIDs = append(stringIDs, strconv.FormatInt(id, 10))
    }
    return stringIDs
}

// AFTER - Size known in advance
func TransformIDs(userIDs []int64) []string {
    stringIDs := make([]string, len(userIDs))  // Exact size
    for i, id := range userIDs {
        stringIDs[i] = strconv.FormatInt(id, 10)
    }
    return stringIDs
}
// 3x faster for 10k items
\`\`\`

**5. Memory Profiling Shows the Impact**
\`\`\`bash
# Profile memory allocations
go test -bench=Alloc -memprofile=mem.out

# Analyze with pprof
go tool pprof mem.out

# BEFORE (naive):
(pprof) top
Total: 245.3 MB
    234.1 MB  runtime.growslice
     11.2 MB  rand.Intn

# AFTER (pre-allocated):
(pprof) top
Total: 7.8 MB
      7.8 MB  rand.Intn
      0.0 MB  runtime.growslice  # Gone!
\`\`\`

**6. When Pre-allocation Really Matters**
High-traffic scenarios where every millisecond counts:

\`\`\`go
// Request handler processing 10k requests/second
func HandleBulkUpload(items []Item) Response {
    // SLOW: results slice grows 14 times for 10k items
    var results []Result
    for _, item := range items {
        results = append(results, process(item))
    }

    // FAST: single allocation
    results := make([]Result, len(items))
    for i, item := range items {
        results[i] = process(item)
    }

    // Impact per request:
    // - SLOW: 140 µs overhead from slice growth
    // - FAST: 5 µs overhead
    //
    // For 10k req/sec:
    // - SLOW: 1.4 seconds of CPU time wasted
    // - FAST: 0.05 seconds
    //
    // Saved: 1.35 CPU seconds per second!
}
\`\`\`

**7. Benchmark Comparison**
\`\`\`go
// Creating 100k element slices

// Naive approach:
// Time: 12ms
// Allocs: 17 (due to growth)
// Memory: 1.6MB (wasted capacity)

// Pre-allocated:
// Time: 2ms (6x faster)
// Allocs: 1
// Memory: 800KB (exact size)

// Benefit increases with size:
// 1M elements: 50x faster
// 10M elements: 100x faster
\`\`\`

**8. Combined with strings.Builder**
Building large text outputs:
\`\`\`go
func GenerateReport(records []Record) string {
    // Pre-allocate both slice and builder
    lines := make([]string, len(records))
    for i, r := range records {
        lines[i] = formatRecord(r)
    }

    var b strings.Builder
    b.Grow(len(lines) * 100)  // Estimate
    for _, line := range lines {
        b.WriteString(line)
        b.WriteString("\\n")
    }
    return b.String()
}
\`\`\`

**Real-World Impact:**
Analytics company processing time-series data:
- **Before**: Processing 1M data points took 3 minutes
  - 24 slice reallocations per batch
  - 85% time spent in memory allocation
  - Servers constantly at capacity

- **After**: Same processing in 8 seconds (22.5x faster!)
  - 1 allocation per batch
  - 5% time in memory allocation
  - Reduced server count from 12 to 2
  - Saved $120K/year in infrastructure costs

**Production Best Practices:**
1. Always pre-allocate when size is known
2. Use make([]T, n) for direct assignment by index
3. Use make([]T, 0, n) for append pattern
4. Profile with -benchmem to verify improvements
5. For unknown sizes, estimate high (better than growing)
6. Consider using sync.Pool for frequently allocated slices
7. Measure with benchmarks before and after

Slice pre-allocation is a simple change that can dramatically improve your Go application's performance. It's low-hanging fruit that every production Go developer should pick.`,	order: 1,
	translations: {
		ru: {
			title: 'Предварительное выделение слайсов',
			solutionCode: `package profilingx

import (
	"math/rand"
)

func BetterAlloc(n int) []int {
	out := make([]int, n)                       // пре-аллоцируем точно нужную capacity
	for i := 0; i < n; i++ {                    // заполняем слайс присваиванием по индексу
		out[i] = rand.Intn(1000)                 // без append, без реаллокации, прямая запись
	}
	return out                                  // возвращаем полностью заполненный слайс
}`,
			description: `Оптимизируйте аллокацию слайсов через пре-аллокацию capacity чтобы избежать повторных реаллокаций памяти.

**Требования:**
1. **BetterAlloc**: Создать слайс с известной capacity заранее
2. **Использовать make()**: Пре-аллоцировать через make([]int, n) или make([]int, 0, n)
3. **Избегать роста**: Предотвратить автоматический рост слайса и копирование
4. **Вернуть слайс**: Вернуть слайс правильного размера

**Наивный подход (медленный):**
\`\`\`go
func NaiveAlloc(n int) []int {
    var out []int
    for i := 0; i < n; i++ {
        out = append(out, rand.Intn(1000))
        // Слайс растет: 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128...
        // Каждый рост копирует ВСЕ элементы в новый массив
    }
    return out
}
// Для n=1000: 10+ реаллокаций, множественные копирования
\`\`\`

**Ключевые концепции:**
- Слайсы имеют длину (len) и capacity (cap)
- append() увеличивает capacity когда len == cap
- Стратегия роста: удвоение capacity каждый раз (1→2→4→8...)
- Каждый рост требует новой аллокации + копирования всех элементов
- Пре-аллокация устраняет рост и копирование
- make([]T, n) создает length=n, capacity=n
- make([]T, 0, n) создает length=0, capacity=n

**Ограничения:**
- Должен использовать make() для пре-аллокации
- Выбрать make([]int, n) для прямого присваивания
- Выбрать make([]int, 0, n) для паттерна append
- Финальный слайс должен содержать n элементов`,
			hint1: `Используйте make([]int, n) чтобы создать слайс с длиной n и capacity n.`,
			hint2: `Заполните слайс используя присваивание по индексу: out[i] = rand.Intn(1000) вместо append.`,
			whyItMatters: `Предварительное выделение слайсов - одна из самых эффективных оптимизаций производительности в Go.

**Почему это важно:**

**1. Скрытая цена роста слайса**

Go удваивает capacity слайса при каждом росте:

\`\`\`go
// Что происходит с append на пустом слайсе:
var s []int
s = append(s, 1)    // Allocate [1]       cap=1
s = append(s, 2)    // Allocate [1,2]     cap=2, copy 1 element
s = append(s, 3)    // Allocate [1,2,3,_] cap=4, copy 2 elements
s = append(s, 4)    // No allocation      cap=4
s = append(s, 5)    // Allocate [...]     cap=8, copy 4 elements

// Для 1000 элементов:
// - 10 аллокаций
// - 1023 элемента скопировано в сумме
// - Потраченная capacity: ~1024 - 1000 = 24 элемента
\`\`\`

**2. Реальный продакшен: API Response Building**

REST API возвращает массив продуктов:

\`\`\`go
// BEFORE - No pre-allocation
func GetProducts(category string) []Product {
    var products []Product  // cap=0

    rows, _ := db.Query("SELECT * FROM products WHERE category = ?", category)
    for rows.Next() {
        var p Product
        rows.Scan(&p.ID, &p.Name, &p.Price)
        products = append(products, p)
        // Growing: 0→1→2→4→8→16→32→64→128→256→512→1024
        // For 1000 products: 11 reallocations!
    }
    return products
}

// Response time: 45ms per request
// Memory: 85MB for 10k products
// GC pressure: constant

// AFTER - Pre-allocation with COUNT
func GetProducts(category string) []Product {
    var count int
    db.QueryRow("SELECT COUNT(*) FROM products WHERE category = ?", category).Scan(&count)

    products := make([]Product, 0, count)  // Pre-allocate exact size

    rows, _ := db.Query("SELECT * FROM products WHERE category = ?", category)
    for rows.Next() {
        var p Product
        rows.Scan(&p.ID, &p.Name, &p.Price)
        products = append(products, p)  // No reallocation!
    }
    return products
}

// Response time: 8ms per request (5.6x быстрее!)
// Memory: 40MB for 10k products (53% reduction)
// GC pressure: minimal
\`\`\`

**3. Batch Processing**

Обработка файлов по частям:

\`\`\`go
// BEFORE - Growing slice
func ProcessFile(filename string) []Record {
    file, _ := os.Open(filename)
    scanner := bufio.NewScanner(file)

    var records []Record
    for scanner.Scan() {
        line := scanner.Text()
        records = append(records, parseLine(line))
        // Constant reallocations for large files
    }
    return records
}
// 1M lines: 21 reallocations, 1M copies, 120 seconds

// AFTER - Pre-allocated
func ProcessFile(filename string) []Record {
    // Count lines first
    lineCount := countLines(filename)

    file, _ := os.Open(filename)
    scanner := bufio.NewScanner(file)

    records := make([]Record, 0, lineCount)
    for scanner.Scan() {
        line := scanner.Text()
        records = append(records, parseLine(line))
    }
    return records
}
// 1M lines: 1 allocation, 0 copies, 45 seconds (2.7x быстрее!)
\`\`\`

**4. Data Transformation**

Трансформация одного слайса в другой:

\`\`\`go
// BEFORE
func TransformIDs(userIDs []int64) []string {
    var stringIDs []string
    for _, id := range userIDs {
        stringIDs = append(stringIDs, strconv.FormatInt(id, 10))
    }
    return stringIDs
}

// AFTER - Size known in advance
func TransformIDs(userIDs []int64) []string {
    stringIDs := make([]string, len(userIDs))  // Exact size
    for i, id := range userIDs {
        stringIDs[i] = strconv.FormatInt(id, 10)
    }
    return stringIDs
}
// 3x быстрее для 10k элементов
\`\`\`

**5. Memory Profiling показывает влияние**

\`\`\`bash
# Profile memory allocations
go test -bench=Alloc -memprofile=mem.out

# Analyze with pprof
go tool pprof mem.out

# BEFORE (naive):
(pprof) top
Total: 245.3 MB
    234.1 MB  runtime.growslice
     11.2 MB  rand.Intn

# AFTER (pre-allocated):
(pprof) top
Total: 7.8 MB
      7.8 MB  rand.Intn
      0.0 MB  runtime.growslice  # Gone!
\`\`\`

**6. Когда пре-аллокация действительно важна**

High-traffic сценарии где каждая миллисекунда критична:

\`\`\`go
// Request handler processing 10k requests/second
func HandleBulkUpload(items []Item) Response {
    // SLOW: results slice grows 14 times for 10k items
    var results []Result
    for _, item := range items {
        results = append(results, process(item))
    }

    // FAST: single allocation
    results := make([]Result, len(items))
    for i, item := range items {
        results[i] = process(item)
    }

    // Impact per request:
    // - SLOW: 140 µs overhead from slice growth
    // - FAST: 5 µs overhead
    //
    // For 10k req/sec:
    // - SLOW: 1.4 seconds of CPU time wasted
    // - FAST: 0.05 seconds
    //
    // Saved: 1.35 CPU seconds per second!
}
\`\`\`

**7. Benchmark Comparison**

\`\`\`go
// Creating 100k element slices

// Naive approach:
// Time: 12ms
// Allocs: 17 (due to growth)
// Memory: 1.6MB (wasted capacity)

// Pre-allocated:
// Time: 2ms (6x faster)
// Allocs: 1
// Memory: 800KB (exact size)

// Benefit increases with size:
// 1M elements: 50x faster
// 10M elements: 100x faster
\`\`\`

**Реальное влияние:**

Analytics компания обрабатывала time-series данные:
- **До**: Обработка 1M data points занимала 3 минуты
  - 24 реаллокации слайсов на батч
  - 85% времени тратилось на аллокацию памяти
  - Серверы постоянно на пределе

- **После**: Та же обработка за 8 секунд (22.5x быстрее!)
  - 1 аллокация на батч
  - 5% времени на аллокацию памяти
  - Серверов уменьшено с 12 до 2
  - Экономия $120K/год на infrastructure

**Production Best Practices:**
1. Всегда пре-аллоцируйте когда размер известен
2. Используйте make([]T, n) для прямого присваивания по индексу
3. Используйте make([]T, 0, n) для паттерна append
4. Профилируйте с -benchmem для проверки улучшений
5. Для неизвестных размеров оценивайте выше (лучше чем рост)
6. Рассмотрите использование sync.Pool для часто аллоцируемых слайсов
7. Измеряйте с бенчмарками до и после

Пре-аллокация слайсов - простое изменение которое может драматически улучшить производительность вашего Go приложения. Это низко-висящий фрукт который каждый продакшен Go разработчик должен сорвать.`
		},
		uz: {
			title: `Slayzlarni oldindan ajratish`,
			solutionCode: `package profilingx

import (
	"math/rand"
)

func BetterAlloc(n int) []int {
	out := make([]int, n)                       // kerakli aniq capacity ni oldindan ajratamiz
	for i := 0; i < n; i++ {                    // slice ni indeks bo'yicha tayinlash orqali to'ldiramiz
		out[i] = rand.Intn(1000)                 // append yo'q, qayta ajratish yo'q, to'g'ridan-to'g'ri yozish
	}
	return out                                  // to'liq to'ldirilgan slice ni qaytaramiz
}`,
			description: `Takroriy xotira qayta ajratishlarini oldini olish uchun capacity ni oldindan ajratish orqali slice ajratishni optimallashtiring.

**Talablar:**
1. **BetterAlloc**: Ma'lum capacity bilan slice ni oldindan yarating
2. **make() dan Foydalaning**: make([]int, n) yoki make([]int, 0, n) bilan oldindan ajrating
3. **O'sishdan Qoching**: Avtomatik slice o'sishi va nusxalashni oldini oling
4. **Slice Qaytaring**: To'g'ri o'lchamdagi slice qaytaring

**Naiv Yondashuv (Sekin):**
\`\`\`go
func NaiveAlloc(n int) []int {
    var out []int
    for i := 0; i < n; i++ {
        out = append(out, rand.Intn(1000))
        // Slice o'sadi: 1 → 2 → 4 → 8 → 16 → 32...
        // Har bir o'sish BARCHA elementlarni yangi massivga nusxalaydi
    }
    return out
}
// n=1000 uchun: 10+ qayta ajratish, bir nechta nusxa
\`\`\`

**Optimallashtirilgan Yondashuv:**
\`\`\`go
func BetterAlloc(n int) []int {
    out := make([]int, n)
    for i := 0; i < n; i++ {
        out[i] = rand.Intn(1000)
    }
    return out
}
// n=1000 uchun: 1 ajratish, nusxalash yo'q, 10x tezroq
\`\`\`

**Cheklovlar:**
- Oldindan ajratish uchun make() ishlatish kerak
- To'g'ridan-to'g'ri tayinlash uchun make([]int, n) tanlang
- append pattern uchun make([]int, 0, n) tanlang
- Yakuniy slice n elementni o'z ichiga olishi kerak`,
			hint1: `n uzunligi va n capacity li slice yaratish uchun make([]int, n) dan foydalaning.`,
			hint2: `append o'rniga indeks bo'yicha tayinlash orqali slice ni to'ldiring: out[i] = rand.Intn(1000).`,
			whyItMatters: `Slice oldindan ajratish siz qilishingiz mumkin bo'lgan eng ta'sirli Go ishlash optimizatsiyalaridan biri.

**Nima uchun bu muhim:**

**1. Slice O'sishining Yashirin Narxi**

Go har bir o'sishda slice capacity ni ikki baravar oshiradi:

\`\`\`go
// Bo'sh slice da append bilan nima sodir bo'ladi:
var s []int
s = append(s, 1)    // Allocate [1]       cap=1
s = append(s, 2)    // Allocate [1,2]     cap=2, 1 elementni nusxalash
s = append(s, 3)    // Allocate [1,2,3,_] cap=4, 2 elementni nusxalash
s = append(s, 4)    // Ajratish yo'q      cap=4
s = append(s, 5)    // Allocate [...]     cap=8, 4 elementni nusxalash

// 1000 element uchun:
// - 10 ajratish
// - Jami 1023 element nusxalangan
// - Isrof qilingan capacity: ~1024 - 1000 = 24 element
\`\`\`

**2. Haqiqiy ishlab chiqarish: API Response Building**

REST API mahsulotlar massivini qaytaradi:

\`\`\`go
// OLDIN - Oldindan ajratish yo'q
func GetProducts(category string) []Product {
    var products []Product  // cap=0

    rows, _ := db.Query("SELECT * FROM products WHERE category = ?", category)
    for rows.Next() {
        var p Product
        rows.Scan(&p.ID, &p.Name, &p.Price)
        products = append(products, p)
        // O'sish: 0→1→2→4→8→16→32→64→128→256→512→1024
        // 1000 mahsulot uchun: 11 qayta ajratish!
    }
    return products
}

// Javob vaqti: so'rov uchun 45ms
// Xotira: 10k mahsulot uchun 85MB
// GC bosimi: doimiy

// KEYIN - COUNT bilan oldindan ajratish
func GetProducts(category string) []Product {
    var count int
    db.QueryRow("SELECT COUNT(*) FROM products WHERE category = ?", category).Scan(&count)

    products := make([]Product, 0, count)  // Aniq o'lchamni oldindan ajrating

    rows, _ := db.Query("SELECT * FROM products WHERE category = ?", category)
    for rows.Next() {
        var p Product
        rows.Scan(&p.ID, &p.Name, &p.Price)
        products = append(products, p)  // Qayta ajratish yo'q!
    }
    return products
}

// Javob vaqti: so'rov uchun 8ms (5.6x tezroq!)
// Xotira: 10k mahsulot uchun 40MB (53% kamayish)
// GC bosimi: minimal
\`\`\`

**3. Batch Processing**

Fayllarni qismlarda qayta ishlash:

\`\`\`go
// OLDIN - O'suvchi slice
func ProcessFile(filename string) []Record {
    file, _ := os.Open(filename)
    scanner := bufio.NewScanner(file)

    var records []Record
    for scanner.Scan() {
        line := scanner.Text()
        records = append(records, parseLine(line))
        // Katta fayllar uchun doimiy qayta ajratishlar
    }
    return records
}
// 1M qator: 21 qayta ajratish, 1M nusxa, 120 soniya

// KEYIN - Oldindan ajratilgan
func ProcessFile(filename string) []Record {
    // Avval qatorlarni hisoblash
    lineCount := countLines(filename)

    file, _ := os.Open(filename)
    scanner := bufio.NewScanner(file)

    records := make([]Record, 0, lineCount)
    for scanner.Scan() {
        line := scanner.Text()
        records = append(records, parseLine(line))
    }
    return records
}
// 1M qator: 1 ajratish, 0 nusxa, 45 soniya (2.7x tezroq!)
\`\`\`

**4. Ma'lumotlarni Transformatsiya Qilish**

Bir slice ni boshqasiga transformatsiya qilish:

\`\`\`go
// OLDIN
func TransformIDs(userIDs []int64) []string {
    var stringIDs []string
    for _, id := range userIDs {
        stringIDs = append(stringIDs, strconv.FormatInt(id, 10))
    }
    return stringIDs
}

// KEYIN - O'lcham oldindan ma'lum
func TransformIDs(userIDs []int64) []string {
    stringIDs := make([]string, len(userIDs))  // Aniq o'lcham
    for i, id := range userIDs {
        stringIDs[i] = strconv.FormatInt(id, 10)
    }
    return stringIDs
}
// 10k element uchun 3x tezroq
\`\`\`

**5. Memory Profiling ta'sirni ko'rsatadi**

\`\`\`bash
# Xotira ajratishlarni profillash
go test -bench=Alloc -memprofile=mem.out

# pprof bilan tahlil qilish
go tool pprof mem.out

# OLDIN (naiv):
(pprof) top
Total: 245.3 MB
    234.1 MB  runtime.growslice
     11.2 MB  rand.Intn

# KEYIN (oldindan ajratilgan):
(pprof) top
Total: 7.8 MB
      7.8 MB  rand.Intn
      0.0 MB  runtime.growslice  # Yo'q!
\`\`\`

**6. Oldindan ajratish haqiqatan ham muhim bo'lganda**

Har bir millisekund muhim bo'lgan yuqori trafik stsenariylari:

\`\`\`go
// Soniyada 10k so'rovni qayta ishlovchi Request handler
func HandleBulkUpload(items []Item) Response {
    // SEKIN: 10k element uchun results slice 14 marta o'sadi
    var results []Result
    for _, item := range items {
        results = append(results, process(item))
    }

    // TEZ: bitta ajratish
    results := make([]Result, len(items))
    for i, item := range items {
        results[i] = process(item)
    }

    // So'rov uchun ta'sir:
    // - SEKIN: slice o'sishidan 140 µs overhead
    // - TEZ: 5 µs overhead
    //
    // 10k req/soniya uchun:
    // - SEKIN: 1.4 soniya CPU vaqti isrof
    // - TEZ: 0.05 soniya
    //
    // Tejaldi: soniyada 1.35 CPU soniya!
}
\`\`\`

**7. Benchmark Taqqoslash**

\`\`\`go
// 100k elementli slice yaratish

// Naiv yondashuv:
// Vaqt: 12ms
// Ajratishlar: 17 (o'sish tufayli)
// Xotira: 1.6MB (isrof qilingan capacity)

// Oldindan ajratilgan:
// Vaqt: 2ms (6x tezroq)
// Ajratishlar: 1
// Xotira: 800KB (aniq o'lcham)

// Foyda o'lcham bilan ortadi:
// 1M element: 50x tezroq
// 10M element: 100x tezroq
\`\`\`

**Haqiqiy ta'sir:**

Analytics kompaniyasi time-series ma'lumotlarini qayta ishladi:
- **Oldin**: 1M data point ni qayta ishlash 3 daqiqa oldi
  - Batch uchun 24 slice qayta ajratish
  - Vaqtning 85% i xotira ajratishga sarflandi
  - Serverlar doimiy sig'im chegarasida

- **Keyin**: Xuddi shu qayta ishlash 8 soniyada (22.5x tezroq!)
  - Batch uchun 1 ajratish
  - Ajratishga vaqtning 5% i
  - Serverlar 12 dan 2 ga kamaytirildi
  - Infrastructure uchun yiliga $120K tejash

**Production Eng Yaxshi Amaliyotlari:**
1. O'lcham ma'lum bo'lganda har doim oldindan ajrating
2. Indeks bo'yicha to'g'ridan-to'g'ri tayinlash uchun make([]T, n) dan foydalaning
3. append patterni uchun make([]T, 0, n) dan foydalaning
4. Yaxshilanishlarni tekshirish uchun -benchmem bilan profillash qiling
5. Noma'lum o'lchamlar uchun yuqori baholang (o'sishdan yaxshiroq)
6. Tez-tez ajratiladigan slice lar uchun sync.Pool dan foydalanishni ko'rib chiqing
7. Oldindan va keyin benchmark lar bilan o'lchang

Slice oldindan ajratish Go ilovangiz ishlashini dramatik ravishda yaxshilashi mumkin bo'lgan oddiy o'zgarish. Bu har bir production Go developeri terishi kerak bo'lgan past osiladigan meva.`
		}
	}
};

export default task;
