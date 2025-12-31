import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-profiling-map-prealloc',
	title: 'Map Pre-allocation Optimization',
	difficulty: 'easy',	tags: ['go', 'profiling', 'performance', 'maps'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Optimize map creation by pre-allocating capacity to avoid hash table rehashing and memory reallocations.

**Requirements:**
1. **BetterMapAlloc**: Create map with known size hint upfront
2. **Use make()**: Pre-allocate with make(map[K]V, n)
3. **Avoid rehashing**: Prevent automatic map growth and bucket reallocation
4. **Return map**: Return fully populated map

**Naive Approach (Slow):**
\`\`\`go
func NaiveMapAlloc(n int) map[int]string {
    m := make(map[int]string)  // No size hint!
    for i := 0; i < n; i++ {
        m[i] = fmt.Sprintf("value-%d", i)
        // Map rehashes at: 8, 16, 32, 64, 128, 256, 512, 1024...
        // Each rehash copies ALL entries to new buckets
    }
    return m
}
// For n=1000: 8+ rehashes, multiple full copies
\`\`\`

**Optimized Approach:**
\`\`\`go
func BetterMapAlloc(n int) map[int]string {
    // Pre-allocate with size hint
    m := make(map[int]string, n)

    for i := 0; i < n; i++ {
        m[i] = fmt.Sprintf("value-%d", i)
        // No rehashing - buckets already allocated
    }

    return m
}
// For n=1000: 1 allocation, no rehashing, 5-10x faster
\`\`\`

**Key Concepts:**
- Maps in Go use hash tables with buckets
- Each bucket holds 8 key-value pairs
- When map fills, it doubles bucket count and rehashes
- Rehashing requires allocating new buckets and copying all entries
- Growth strategy: 8 → 16 → 32 → 64 → 128 → 256...
- make(map[K]V, n) pre-allocates buckets for n elements
- Pre-allocation eliminates rehashing overhead
- Size hint is approximate - Go allocates sufficient buckets

**Benchmark Results:**
\`\`\`bash
go test -bench=MapAlloc -benchmem

BenchmarkMapAlloc_Naive-8     20000    80000 ns/op   65536 B/op    12 allocs/op
BenchmarkMapAlloc_Better-8   100000    15000 ns/op   45056 B/op     1 allocs/op

# Better is 5-10x faster with 92% fewer allocations!
\`\`\`

**Example Usage:**
\`\`\`go
// Building lookup table from database
func BuildUserCache(db *sql.DB) map[int64]User {
    var count int
    db.QueryRow("SELECT COUNT(*) FROM users").Scan(&count)

    // Pre-allocate map
    cache := make(map[int64]User, count)

    rows, _ := db.Query("SELECT id, name, email FROM users")
    for rows.Next() {
        var u User
        rows.Scan(&u.ID, &u.Name, &u.Email)
        cache[u.ID] = u  // No rehashing!
    }
    return cache
}

// Request ID tracking
func TrackRequests(ids []string) map[string]bool {
    // Know exact size needed
    seen := make(map[string]bool, len(ids))

    for _, id := range ids {
        seen[id] = true
    }
    return seen
}

// Grouping items by category
func GroupByCategory(items []Item) map[string][]Item {
    // Estimate number of unique categories
    groups := make(map[string][]Item, len(items)/10)

    for _, item := range items {
        groups[item.Category] = append(groups[item.Category], item)
    }
    return groups
}

// Building index from array
func BuildIndex(products []Product) map[string]Product {
    index := make(map[string]Product, len(products))

    for _, p := range products {
        index[p.SKU] = p
    }
    return index
}
\`\`\`

**When to Pre-allocate Maps:**
1. **Known size** - exact count of entries
2. **Database results** - COUNT query first
3. **Array to map conversion** - size = len(array)
4. **Batch processing** - size known from input
5. **Approximate size** - better to overestimate than grow

**Constraints:**
- Must use make(map[int]string, n) with size hint
- Must populate map with n entries
- Keys: 0 to n-1, Values: "value-{i}" format
- Return fully populated map`,
	initialCode: `package profilingx

import (
	"fmt"
)

// TODO: Implement BetterMapAlloc
// Create map with size hint using make(map[int]string, n)
// Fill map with entries: key=i, value="value-{i}"
// Return populated map
func BetterMapAlloc(n int) map[int]string {
	// TODO: Implement
}`,
	solutionCode: `package profilingx

import (
	"fmt"
)

func BetterMapAlloc(n int) map[int]string {
	m := make(map[int]string, n)               // pre-allocate buckets for n entries
	for i := 0; i < n; i++ {                    // populate map without triggering rehash
		m[i] = fmt.Sprintf("value-%d", i)        // direct insertion, no bucket reallocation
	}
	return m                                    // return fully populated map
}`,
		hint1: `Use make(map[int]string, n) to create a map with a size hint. This pre-allocates enough buckets to hold n entries.`,
		hint2: `Loop from 0 to n-1 and insert entries: m[i] = fmt.Sprintf("value-%d", i)`,
		testCode: `package profilingx

import (
	"fmt"
	"testing"
)

func Test1(t *testing.T) {
	// Test n=0
	result := BetterMapAlloc(0)
	if len(result) != 0 {
		t.Errorf("BetterMapAlloc(0) len = %d, want 0", len(result))
	}
}

func Test2(t *testing.T) {
	// Test n=1
	result := BetterMapAlloc(1)
	if len(result) != 1 {
		t.Errorf("BetterMapAlloc(1) len = %d, want 1", len(result))
	}
	if result[0] != "value-0" {
		t.Errorf("BetterMapAlloc(1)[0] = %q, want %q", result[0], "value-0")
	}
}

func Test3(t *testing.T) {
	// Test n=5
	result := BetterMapAlloc(5)
	if len(result) != 5 {
		t.Errorf("BetterMapAlloc(5) len = %d, want 5", len(result))
	}
}

func Test4(t *testing.T) {
	// Test value format
	result := BetterMapAlloc(10)
	for i := 0; i < 10; i++ {
		expected := fmt.Sprintf("value-%d", i)
		if result[i] != expected {
			t.Errorf("BetterMapAlloc(10)[%d] = %q, want %q", i, result[i], expected)
		}
	}
}

func Test5(t *testing.T) {
	// Test n=100
	result := BetterMapAlloc(100)
	if len(result) != 100 {
		t.Errorf("BetterMapAlloc(100) len = %d, want 100", len(result))
	}
}

func Test6(t *testing.T) {
	// Test all keys present
	result := BetterMapAlloc(50)
	for i := 0; i < 50; i++ {
		if _, ok := result[i]; !ok {
			t.Errorf("BetterMapAlloc(50) missing key %d", i)
		}
	}
}

func Test7(t *testing.T) {
	// Test n=1000
	result := BetterMapAlloc(1000)
	if len(result) != 1000 {
		t.Errorf("BetterMapAlloc(1000) len = %d, want 1000", len(result))
	}
}

func Test8(t *testing.T) {
	// Test first and last entries
	result := BetterMapAlloc(500)
	if result[0] != "value-0" {
		t.Errorf("BetterMapAlloc(500)[0] = %q, want %q", result[0], "value-0")
	}
	if result[499] != "value-499" {
		t.Errorf("BetterMapAlloc(500)[499] = %q, want %q", result[499], "value-499")
	}
}

func Test9(t *testing.T) {
	// Test n=3
	result := BetterMapAlloc(3)
	expected := map[int]string{0: "value-0", 1: "value-1", 2: "value-2"}
	for k, v := range expected {
		if result[k] != v {
			t.Errorf("BetterMapAlloc(3)[%d] = %q, want %q", k, result[k], v)
		}
	}
}

func Test10(t *testing.T) {
	// Test no extra keys
	result := BetterMapAlloc(10)
	if _, ok := result[10]; ok {
		t.Error("BetterMapAlloc(10) should not contain key 10")
	}
	if _, ok := result[-1]; ok {
		t.Error("BetterMapAlloc(10) should not contain key -1")
	}
}`,
		whyItMatters: `Map pre-allocation is crucial for high-performance Go applications that build large lookup tables and indices.

**Why This Matters:**

**1. The Hidden Cost of Map Growth**
Maps in Go grow by doubling bucket count and rehashing all entries:
\`\`\`go
// What happens without size hint:
m := make(map[int]string)
m[0] = "a"  // Allocate 1 bucket (8 slots)
m[8] = "b"  // Still fits
m[9] = "c"  // Trigger rehash! Allocate 2 buckets, copy all entries
m[17] = "d" // Trigger rehash! Allocate 4 buckets, copy all entries
m[33] = "e" // Trigger rehash! Allocate 8 buckets, copy all entries

// For 1000 entries:
// - 8-10 rehash operations
// - Each rehash copies ALL existing entries
// - Total: 8000+ entries copied
// - Time: ~80µs
// - Memory: 65KB allocated

// With size hint:
m := make(map[int]string, 1000)
// Single allocation for ~128 buckets
// No rehashing during insertions
// Time: ~15µs (5x faster!)
// Memory: 45KB allocated (optimal)
\`\`\`

**2. Real Production Scenario: User Session Cache**
Web service caching user sessions in memory:
\`\`\`go
// BEFORE - No size hint
func LoadActiveSessions() map[string]*Session {
    m := make(map[string]*Session)

    rows, _ := db.Query("SELECT session_id, user_id, data FROM sessions WHERE active = true")
    for rows.Next() {
        var s Session
        rows.Scan(&s.ID, &s.UserID, &s.Data)
        m[s.ID] = &s
        // Rehashing at: 8, 16, 32, 64, 128, 256, 512, 1024, 2048...
        // For 5000 active sessions: 10 rehashes!
    }
    return m
}

// Session loading time: 85ms
// Memory allocations: 14 (from rehashing)
// Peak memory during load: 420KB
// GC triggered during load: Yes

// AFTER - With size hint
func LoadActiveSessions() map[string]*Session {
    var count int
    db.QueryRow("SELECT COUNT(*) FROM sessions WHERE active = true").Scan(&count)

    m := make(map[string]*Session, count)  // Pre-allocate

    rows, _ := db.Query("SELECT session_id, user_id, data FROM sessions WHERE active = true")
    for rows.Next() {
        var s Session
        rows.Scan(&s.ID, &s.UserID, &s.Data)
        m[s.ID] = &s  // No rehashing!
    }
    return m
}

// Session loading time: 18ms (4.7x faster!)
// Memory allocations: 1
// Peak memory: 280KB (33% less)
// GC triggered: No
\`\`\`

**3. Building Product Catalog Index**
E-commerce site indexing products by SKU:
\`\`\`go
// BEFORE - Growing map
func BuildProductIndex(products []Product) map[string]Product {
    index := make(map[string]Product)

    for _, p := range products {
        index[p.SKU] = p
        // Constant rehashing for 10k products
    }
    return index
}
// 10k products: 95ms, 15 rehashes

// AFTER - Pre-allocated map
func BuildProductIndex(products []Product) map[string]Product {
    index := make(map[string]Product, len(products))

    for _, p := range products {
        index[p.SKU] = p
    }
    return index
}
// 10k products: 12ms (7.9x faster!), 1 allocation
\`\`\`

**4. Request Deduplication**
API gateway tracking seen request IDs:
\`\`\`go
// SLOW - No pre-allocation
func FilterDuplicates(requests []Request) []Request {
    seen := make(map[string]bool)  // Starts at 8 entries
    var unique []Request

    for _, req := range requests {
        if !seen[req.ID] {
            seen[req.ID] = true
            unique = append(unique, req)
        }
    }
    return unique
}
// 50k requests/sec: 400ms CPU time, constant rehashing

// FAST - Pre-allocated
func FilterDuplicates(requests []Request) []Request {
    seen := make(map[string]bool, len(requests))  // No rehashing
    unique := make([]Request, 0, len(requests))

    for _, req := range requests {
        if !seen[req.ID] {
            seen[req.ID] = true
            unique = append(unique, req)
        }
    }
    return unique
}
// 50k requests/sec: 85ms CPU time (4.7x faster!)
\`\`\`

**5. Memory Profiling Shows Rehashing Cost**
\`\`\`bash
# Generate memory profile
go test -bench=MapAlloc -memprofile=mem.out

# Analyze with pprof
go tool pprof mem.out

# BEFORE (no size hint):
(pprof) top
Total: 385.2 MB
    312.5 MB  runtime.makemap_small  # Initial small map
     45.8 MB  runtime.mapassign      # Assignments
     26.9 MB  runtime.growWork       # Rehashing overhead!

# AFTER (with size hint):
(pprof) top
Total: 156.3 MB
    145.1 MB  runtime.makemap        # Single allocation
     11.2 MB  runtime.mapassign      # Assignments
      0.0 MB  runtime.growWork       # Gone!

# 59% memory reduction from eliminating rehashing!
\`\`\`

**6. CPU Profiling Shows Rehashing Time**
\`\`\`bash
go test -bench=MapAlloc -cpuprofile=cpu.out
go tool pprof cpu.out

# BEFORE:
(pprof) top10
Total: 12.5s
     8.2s  65.6%  runtime.growWork      # Rehashing!
     2.1s  16.8%  runtime.mapassign
     1.5s  12.0%  runtime.makemap_small
     0.7s   5.6%  fmt.Sprintf

# AFTER:
(pprof) top10
Total: 2.8s
     1.6s  57.1%  fmt.Sprintf           # Main work
     0.8s  28.6%  runtime.mapassign
     0.4s  14.3%  runtime.makemap       # One-time cost
     0.0s   0.0%  runtime.growWork      # Gone!
\`\`\`

**7. Aggregation Pipeline**
Analytics processing grouping events by user:
\`\`\`go
// BEFORE - No size hints
func AggregateByUser(events []Event) map[int64][]Event {
    groups := make(map[int64][]Event)

    for _, event := range events {
        groups[event.UserID] = append(groups[event.UserID], event)
    }
    return groups
}
// 1M events, 100k users: 8.5 seconds

// AFTER - Estimated size hint
func AggregateByUser(events []Event) map[int64][]Event {
    // Estimate unique users (e.g., 10% of events)
    estimatedUsers := len(events) / 10
    groups := make(map[int64][]Event, estimatedUsers)

    for _, event := range events {
        groups[event.UserID] = append(groups[event.UserID], event)
    }
    return groups
}
// 1M events, 100k users: 1.2 seconds (7x faster!)
\`\`\`

**8. Configuration Loading**
Loading environment config into map:
\`\`\`go
// Building config map from file
func LoadConfig(lines []string) map[string]string {
    config := make(map[string]string, len(lines))

    for _, line := range lines {
        parts := strings.SplitN(line, "=", 2)
        if len(parts) == 2 {
            config[parts[0]] = parts[1]
        }
    }
    return config
}
// 1000 config entries: 0.8ms vs 4.5ms without hint (5.6x faster!)
\`\`\`

**9. Trace Profiling Visualization**
\`\`\`bash
# Generate trace
go test -bench=MapAlloc -trace=trace.out

# View in browser
go tool trace trace.out

# BEFORE (no size hint):
# - Multiple GC pauses during map growth
# - GC triggered by excessive allocations
# - Rehashing visible as CPU spikes

# AFTER (with size hint):
# - No GC pauses during map creation
# - Smooth CPU usage
# - Single allocation visible
\`\`\`

**10. Overestimation is OK**
\`\`\`go
// When exact size unknown, overestimate:
func BuildCache(data []Data) map[string]*Data {
    // Overestimate by 20%
    cache := make(map[string]*Data, len(data)*12/10)

    for _, d := range data {
        if d.Active {
            cache[d.Key] = &d
        }
    }
    return cache
}
// Slight memory overhead is better than rehashing!
\`\`\`

**Real-World Impact:**
Financial trading platform indexing market data:
- **Before**: Building orderbook map for 50k orders
  - Time: 450ms per update
  - 12 rehash operations per build
  - Memory allocations: 18 per build
  - System struggling with updates

- **After**: Pre-allocated maps with size hints
  - Time: 45ms per update (10x faster!)
  - 1 allocation per build
  - Reduced server count from 8 to 2
  - Saved $180K/year in infrastructure
  - System handling 5x more updates/sec

**Production Best Practices:**
1. Always provide size hint when size is known or estimatable
2. Use make(map[K]V, n) not make(map[K]V)
3. Overestimate rather than underestimate (slight memory overhead OK)
4. Profile with -benchmem to measure allocation reduction
5. Use pprof to identify rehashing bottlenecks (runtime.growWork)
6. For unknown sizes, estimate based on typical data patterns
7. Combine with slice pre-allocation for maximum performance
8. Benchmark before/after to verify improvements

**Map vs Slice Pre-allocation:**
\`\`\`go
// Both together for maximum performance
func ProcessBatch(items []Item) map[string][]Result {
    // Pre-allocate map for categories
    groups := make(map[string][]Result, 20)

    // Pre-allocate slice for each group
    for _, item := range items {
        if _, exists := groups[item.Category]; !exists {
            groups[item.Category] = make([]Result, 0, 100)
        }
        result := process(item)
        groups[item.Category] = append(groups[item.Category], result)
    }
    return groups
}
\`\`\`

Map pre-allocation is a simple optimization with massive impact. Combined with profiling tools, it's one of the fastest ways to improve Go application performance in production.`,	order: 2,
	translations: {
		ru: {
			title: 'Предварительное выделение map',
			solutionCode: `package profilingx

import (
	"fmt"
)

func BetterMapAlloc(n int) map[int]string {
	m := make(map[int]string, n)               // пре-аллоцируем buckets для n записей
	for i := 0; i < n; i++ {                    // заполняем map без запуска rehash
		m[i] = fmt.Sprintf("value-%d", i)        // прямая вставка, без реаллокации buckets
	}
	return m                                    // возвращаем полностью заполненную map
}`,
			description: `Оптимизируйте создание карт через пре-аллокацию capacity чтобы избежать rehashing хеш-таблицы и реаллокаций памяти.

**Требования:**
1. **BetterMapAlloc**: Создать map с указанием размера заранее
2. **Использовать make()**: Пре-аллоцировать через make(map[K]V, n)
3. **Избегать rehashing**: Предотвратить автоматический рост map и реаллокацию buckets
4. **Вернуть map**: Вернуть полностью заполненную map

**Наивный подход (медленный):**
\`\`\`go
func NaiveMapAlloc(n int) map[int]string {
    m := make(map[int]string)  // Без size hint!
    for i := 0; i < n; i++ {
        m[i] = fmt.Sprintf("value-%d", i)
        // Map делает rehash на: 8, 16, 32, 64, 128, 256, 512, 1024...
        // Каждый rehash копирует ВСЕ записи в новые buckets
    }
    return m
}
// Для n=1000: 8+ rehash операций, множественные полные копирования
\`\`\`

**Ключевые концепции:**
- Maps в Go используют hash tables с buckets
- Каждый bucket держит 8 пар ключ-значение
- Когда map заполняется, удваивается количество buckets и происходит rehash
- Rehashing требует аллокации новых buckets и копирования всех записей
- Стратегия роста: 8 → 16 → 32 → 64 → 128 → 256...
- make(map[K]V, n) пре-аллоцирует buckets для n элементов
- Pre-allocation устраняет overhead rehashing
- Size hint приблизительный - Go аллоцирует достаточно buckets

**Ограничения:**
- Должен использовать make(map[int]string, n) с size hint
- Должен заполнить map с n записями
- Ключи: 0 до n-1, Значения: формат "value-{i}"
- Вернуть полностью заполненную map`,
			hint1: `Используйте make(map[int]string, n) чтобы создать map с size hint. Это пре-аллоцирует достаточно buckets для n записей.`,
			hint2: `Пройдите в цикле от 0 до n-1 и вставьте записи: m[i] = fmt.Sprintf("value-%d", i)`,
			whyItMatters: `Map pre-allocation критична для высокопроизводительных Go приложений, строящих большие lookup tables и indices.

**Почему это важно:**

**1. Скрытая цена роста Map (The Hidden Cost of Map Growth)**
Maps в Go растут удваиванием bucket count и перехешированием всех записей:

\`\`\`go
// Что происходит без size hint:
m := make(map[int]string)
m[0] = "a"  // Аллоцируем 1 bucket (8 слотов)
m[8] = "b"  // Еще помещается
m[9] = "c"  // Триггер rehash! Аллоцируем 2 buckets, копируем все записи
m[17] = "d" // Триггер rehash! Аллоцируем 4 buckets, копируем все записи
m[33] = "e" // Триггер rehash! Аллоцируем 8 buckets, копируем все записи

// Для 1000 записей:
// - 8-10 rehash операций
// - Каждый rehash копирует ВСЕ существующие записи
// - Итого: 8000+ записей скопировано
// - Время: ~80µs
// - Память: 65KB выделено

// С size hint:
m := make(map[int]string, 1000)
// Одна аллокация для ~128 buckets
// Без rehashing во время вставок
// Время: ~15µs (5x быстрее!)
// Память: 45KB выделено (оптимально)
\`\`\`

**2. Real Production Сценарий: User Session Cache**
Web сервис кеширует user sessions в памяти:

\`\`\`go
// ДО - Без size hint
func LoadActiveSessions() map[string]*Session {
    m := make(map[string]*Session)

    rows, _ := db.Query("SELECT session_id, user_id, data FROM sessions WHERE active = true")
    for rows.Next() {
        var s Session
        rows.Scan(&s.ID, &s.UserID, &s.Data)
        m[s.ID] = &s
        // Rehashing на: 8, 16, 32, 64, 128, 256, 512, 1024, 2048...
        // Для 5000 активных сессий: 10 rehashes!
    }
    return m
}

// Время загрузки сессий: 85ms
// Memory allocations: 14 (от rehashing)
// Пиковая память при загрузке: 420KB
// GC triggered во время загрузки: Да

// ПОСЛЕ - С size hint
func LoadActiveSessions() map[string]*Session {
    var count int
    db.QueryRow("SELECT COUNT(*) FROM sessions WHERE active = true").Scan(&count)

    m := make(map[string]*Session, count)  // Pre-allocate

    rows, _ := db.Query("SELECT session_id, user_id, data FROM sessions WHERE active = true")
    for rows.Next() {
        var s Session
        rows.Scan(&s.ID, &s.UserID, &s.Data)
        m[s.ID] = &s  // Без rehashing!
    }
    return m
}

// Время загрузки сессий: 18ms (4.7x быстрее!)
// Memory allocations: 1
// Пиковая память: 280KB (33% меньше)
// GC triggered: Нет
\`\`\`

**3. Building Product Catalog Index (построение индекса каталога)**
E-commerce сайт индексирует продукты по SKU:

\`\`\`go
// ДО - Растущая map
func BuildProductIndex(products []Product) map[string]Product {
    index := make(map[string]Product)

    for _, p := range products {
        index[p.SKU] = p
        // Постоянное rehashing для 10k продуктов
    }
    return index
}
// 10k продуктов: 95ms, 15 rehashes

// ПОСЛЕ - Pre-allocated map
func BuildProductIndex(products []Product) map[string]Product {
    index := make(map[string]Product, len(products))

    for _, p := range products {
        index[p.SKU] = p
    }
    return index
}
// 10k продуктов: 12ms (7.9x быстрее!), 1 аллокация
\`\`\`

**4. Request Deduplication (дедупликация запросов)**
API gateway отслеживает seen request IDs:

\`\`\`go
// МЕДЛЕННО - Без pre-allocation
func FilterDuplicates(requests []Request) []Request {
    seen := make(map[string]bool)  // Начинается с 8 записей
    var unique []Request

    for _, req := range requests {
        if !seen[req.ID] {
            seen[req.ID] = true
            unique = append(unique, req)
        }
    }
    return unique
}
// 50k requests/sec: 400ms CPU time, постоянное rehashing

// БЫСТРО - Pre-allocated
func FilterDuplicates(requests []Request) []Request {
    seen := make(map[string]bool, len(requests))  // Без rehashing
    unique := make([]Request, 0, len(requests))

    for _, req := range requests {
        if !seen[req.ID] {
            seen[req.ID] = true
            unique = append(unique, req)
        }
    }
    return unique
}
// 50k requests/sec: 85ms CPU time (4.7x быстрее!)
\`\`\`

**5. Memory Profiling показывает цену Rehashing**
\`\`\`bash
# Генерируем memory profile
go test -bench=MapAlloc -memprofile=mem.out

# Анализируем с pprof
go tool pprof mem.out

# ДО (без size hint):
(pprof) top
Total: 385.2 MB
    312.5 MB  runtime.makemap_small  # Начальная маленькая map
     45.8 MB  runtime.mapassign      # Assignments
     26.9 MB  runtime.growWork       # Rehashing overhead!

# ПОСЛЕ (с size hint):
(pprof) top
Total: 156.3 MB
    145.1 MB  runtime.makemap        # Одна аллокация
     11.2 MB  runtime.mapassign      # Assignments
      0.0 MB  runtime.growWork       # Исчезло!

# 59% снижение памяти от устранения rehashing!
\`\`\`

**6. CPU Profiling показывает время Rehashing**
\`\`\`bash
go test -bench=MapAlloc -cpuprofile=cpu.out
go tool pprof cpu.out

# ДО:
(pprof) top10
Total: 12.5s
     8.2s  65.6%  runtime.growWork      # Rehashing!
     2.1s  16.8%  runtime.mapassign
     1.5s  12.0%  runtime.makemap_small
     0.7s   5.6%  fmt.Sprintf

# ПОСЛЕ:
(pprof) top10
Total: 2.8s
     1.6s  57.1%  fmt.Sprintf           # Основная работа
     0.8s  28.6%  runtime.mapassign
     0.4s  14.3%  runtime.makemap       # Одноразовая цена
     0.0s   0.0%  runtime.growWork      # Исчезло!
\`\`\`

**7. Aggregation Pipeline (конвейер агрегации)**
Analytics обработка группирует события по пользователям:

\`\`\`go
// ДО - Без size hints
func AggregateByUser(events []Event) map[int64][]Event {
    groups := make(map[int64][]Event)

    for _, event := range events {
        groups[event.UserID] = append(groups[event.UserID], event)
    }
    return groups
}
// 1M событий, 100k пользователей: 8.5 секунд

// ПОСЛЕ - Оценочный size hint
func AggregateByUser(events []Event) map[int64][]Event {
    // Оцениваем уникальных пользователей (например, 10% от событий)
    estimatedUsers := len(events) / 10
    groups := make(map[int64][]Event, estimatedUsers)

    for _, event := range events {
        groups[event.UserID] = append(groups[event.UserID], event)
    }
    return groups
}
// 1M событий, 100k пользователей: 1.2 секунды (7x быстрее!)
\`\`\`

**8. Configuration Loading (загрузка конфигурации)**
Загрузка environment config в map:

\`\`\`go
// Построение config map из файла
func LoadConfig(lines []string) map[string]string {
    config := make(map[string]string, len(lines))

    for _, line := range lines {
        parts := strings.SplitN(line, "=", 2)
        if len(parts) == 2 {
            config[parts[0]] = parts[1]
        }
    }
    return config
}
// 1000 config entries: 0.8ms vs 4.5ms без hint (5.6x быстрее!)
\`\`\`

**9. Trace Profiling Visualization**
\`\`\`bash
# Генерируем trace
go test -bench=MapAlloc -trace=trace.out

# Смотрим в браузере
go tool trace trace.out

# ДО (без size hint):
# - Множественные GC паузы во время роста map
# - GC triggered избыточными аллокациями
# - Rehashing видно как CPU spikes

# ПОСЛЕ (с size hint):
# - Без GC пауз во время создания map
# - Плавное использование CPU
# - Видна одна аллокация
\`\`\`

**10. Overestimation is OK (переоценка допустима)**
\`\`\`go
// Когда точный размер неизвестен, переоцените:
func BuildCache(data []Data) map[string]*Data {
    // Переоцениваем на 20%
    cache := make(map[string]*Data, len(data)*12/10)

    for _, d := range data {
        if d.Active {
            cache[d.Key] = &d
        }
    }
    return cache
}
// Небольшой memory overhead лучше, чем rehashing!
\`\`\`

**Real-World Impact (реальное влияние):**
Financial trading platform индексация market data:

**До:** Построение orderbook map для 50k orders
- Время: 450ms на обновление
- 12 rehash операций на построение
- Memory allocations: 18 на построение
- Система с трудом справляется с обновлениями

**После:** Pre-allocated maps с size hints
- Время: 45ms на обновление (10x быстрее!)
- 1 аллокация на построение
- Уменьшили количество серверов с 8 до 2
- Сэкономили $180K/год на инфраструктуре
- Система обрабатывает 5x больше updates/sec

**Production Best Practices:**
1. **Всегда указывайте size hint**, когда размер известен или оценим
2. **Используйте make(map[K]V, n)**, не make(map[K]V)
3. **Переоценивайте, а не недооценивайте** (небольшой memory overhead OK)
4. **Профилируйте с -benchmem** для измерения снижения аллокаций
5. **Используйте pprof** для выявления bottlenecks rehashing (runtime.growWork)
6. **Для неизвестных размеров** оценивайте на основе типичных паттернов данных
7. **Комбинируйте с slice pre-allocation** для максимальной производительности
8. **Бенчмаркируйте до/после** для проверки улучшений

**Map vs Slice Pre-allocation:**
\`\`\`go
// Оба вместе для максимальной производительности
func ProcessBatch(items []Item) map[string][]Result {
    // Pre-allocate map для категорий
    groups := make(map[string][]Result, 20)

    // Pre-allocate slice для каждой группы
    for _, item := range items {
        if _, exists := groups[item.Category]; !exists {
            groups[item.Category] = make([]Result, 0, 100)
        }
        result := process(item)
        groups[item.Category] = append(groups[item.Category], result)
    }
    return groups
}
\`\`\`

**Вывод:**
Map pre-allocation - это простая, но мощная оптимизация. Одна дополнительная цифра в make() может дать 5-10x улучшение производительности и значительно снизить нагрузку на GC. В production системах с высокой нагрузкой это различие между системой, которая еле справляется, и системой с запасом мощности.`
		},
		uz: {
			title: `Maplarni oldindan ajratish`,
			solutionCode: `package profilingx

import (
	"fmt"
)

func BetterMapAlloc(n int) map[int]string {
	m := make(map[int]string, n)               // n yozuvlar uchun bucket larni oldindan ajratamiz
	for i := 0; i < n; i++ {                    // rehash ni ishga tushirmasdan map ni to'ldiramiz
		m[i] = fmt.Sprintf("value-%d", i)        // to'g'ridan-to'g'ri kiritish, bucket qayta ajratish yo'q
	}
	return m                                    // to'liq to'ldirilgan map ni qaytaramiz
}`,
			description: `Hash jadval qayta hashlashni va xotira qayta ajratishlarini oldini olish uchun capacity ni oldindan ajratish orqali map yaratishni optimallashtiring.

**Talablar:**
1. **BetterMapAlloc**: Oldindan ma'lum o'lcham ko'rsatkichi bilan map yarating
2. **make() dan Foydalaning**: make(map[K]V, n) bilan oldindan ajrating
3. **Qayta Hashlashdan Qoching**: Avtomatik map o'sishi va bucket qayta ajratishni oldini oling
4. **Map Qaytaring**: To'liq to'ldirilgan map qaytaring

**Naiv Yondashuv (Sekin):**
\`\`\`go
func NaiveMapAlloc(n int) map[int]string {
    m := make(map[int]string)  // O'lcham ko'rsatkichi yo'q!
    for i := 0; i < n; i++ {
        m[i] = fmt.Sprintf("value-%d", i)
        // Map quyidagilarda qayta hashlaydi: 8, 16, 32, 64, 128, 256, 512, 1024...
        // Har bir qayta hashlash BARCHA yozuvlarni yangi bucket larga nusxalaydi
    }
    return m
}
// n=1000 uchun: 8+ qayta hashlash, bir nechta to'liq nusxa
\`\`\`

**Optimallashtirilgan Yondashuv:**
\`\`\`go
func BetterMapAlloc(n int) map[int]string {
    m := make(map[int]string, n)
    for i := 0; i < n; i++ {
        m[i] = fmt.Sprintf("value-%d", i)
    }
    return m
}
// n=1000 uchun: 1 ajratish, qayta hashlash yo'q, 5-10x tezroq
\`\`\`

**Asosiy Tushunchalar:**
- Go dagi map lar bucket li hash jadvallardan foydalanadi
- Har bir bucket 8 ta kalit-qiymat juftligini saqlaydi
- Map to'lganda, bucket soni ikki baravar oshadi va qayta hashlash sodir bo'ladi
- Qayta hashlash yangi bucket larni ajratishni va barcha yozuvlarni nusxalashni talab qiladi
- O'sish strategiyasi: 8 → 16 → 32 → 64 → 128 → 256...
- make(map[K]V, n) n elementlar uchun bucket larni oldindan ajratadi

**Cheklovlar:**
- O'lcham ko'rsatkichi bilan make(map[int]string, n) ishlatish kerak
- Map ni n yozuvlar bilan to'ldirish kerak
- Kalitlar: 0 dan n-1 gacha, Qiymatlar: "value-{i}" formati
- To'liq to'ldirilgan map ni qaytaring`,
			hint1: `n yozuvlarni saqlash uchun yetarli bucket larni oldindan ajratadigan map yaratish uchun make(map[int]string, n) dan foydalaning.`,
			hint2: `0 dan n-1 gacha sikl aylanib yozuvlarni kiriting: m[i] = fmt.Sprintf("value-%d", i)`,
			whyItMatters: `Map oldindan ajratish katta lookup jadvallar va indekslarni quradigan yuqori ishlashli Go ilovalari uchun muhim.

**Nima uchun bu muhim:**

**1. Map O'sishining Yashirin Narxi**
Go'dagi maplar bucket sonini ikki baravar oshirish va barcha yozuvlarni qayta hashlash orqali o'sadi:

\`\`\`go
// O'lcham ko'rsatkichisiz nima bo'ladi:
m := make(map[int]string)
m[0] = "a"  // 1 bucket ajratamiz (8 slot)
m[8] = "b"  // Hali sig'adi
m[9] = "c"  // Rehash trigger! 2 bucket ajratamiz, barcha yozuvlarni nusxalaymiz
m[17] = "d" // Rehash trigger! 4 bucket ajratamiz, barcha yozuvlarni nusxalaymiz
m[33] = "e" // Rehash trigger! 8 bucket ajratamiz, barcha yozuvlarni nusxalaymiz

// 1000 yozuv uchun:
// - 8-10 rehash operatsiyasi
// - Har bir rehash BARCHA mavjud yozuvlarni nusxalaydi
// - Jami: 8000+ yozuv nusxalangan
// - Vaqt: ~80µs
// - Xotira: 65KB ajratilgan

// O'lcham ko'rsatkichi bilan:
m := make(map[int]string, 1000)
// ~128 bucket uchun bitta ajratish
// Kiritishlar paytida rehashing yo'q
// Vaqt: ~15µs (5x tezroq!)
// Xotira: 45KB ajratilgan (optimal)
\`\`\`

**2. Real Production Stsenariysi: User Session Cache**
Veb xizmati user sessiyalarini xotirada keshlaydi:

\`\`\`go
// OLDIN - O'lcham ko'rsatkichisiz
func LoadActiveSessions() map[string]*Session {
    m := make(map[string]*Session)

    rows, _ := db.Query("SELECT session_id, user_id, data FROM sessions WHERE active = true")
    for rows.Next() {
        var s Session
        rows.Scan(&s.ID, &s.UserID, &s.Data)
        m[s.ID] = &s
        // Rehashing: 8, 16, 32, 64, 128, 256, 512, 1024, 2048...
        // 5000 faol sessiya uchun: 10 rehash!
    }
    return m
}

// Sessiyalarni yuklash vaqti: 85ms
// Xotira ajratishlari: 14 (rehashingdan)
// Yuklash paytida eng yuqori xotira: 420KB
// Yuklash paytida GC ishga tushdi: Ha

// KEYIN - O'lcham ko'rsatkichi bilan
func LoadActiveSessions() map[string]*Session {
    var count int
    db.QueryRow("SELECT COUNT(*) FROM sessions WHERE active = true").Scan(&count)

    m := make(map[string]*Session, count)  // Pre-allocate

    rows, _ := db.Query("SELECT session_id, user_id, data FROM sessions WHERE active = true")
    for rows.Next() {
        var s Session
        rows.Scan(&s.ID, &s.UserID, &s.Data)
        m[s.ID] = &s  // Rehashingsiz!
    }
    return m
}

// Sessiyalarni yuklash vaqti: 18ms (4.7x tezroq!)
// Xotira ajratishlari: 1
// Eng yuqori xotira: 280KB (33% kam)
// GC ishga tushdi: Yo'q
\`\`\`

**3. Product Catalog Index Qurish**
E-commerce sayti mahsulotlarni SKU bo'yicha indekslashtiradi:

\`\`\`go
// OLDIN - O'sayotgan map
func BuildProductIndex(products []Product) map[string]Product {
    index := make(map[string]Product)

    for _, p := range products {
        index[p.SKU] = p
        // 10k mahsulot uchun doimiy rehashing
    }
    return index
}
// 10k mahsulot: 95ms, 15 rehash

// KEYIN - Oldindan ajratilgan map
func BuildProductIndex(products []Product) map[string]Product {
    index := make(map[string]Product, len(products))

    for _, p := range products {
        index[p.SKU] = p
    }
    return index
}
// 10k mahsulot: 12ms (7.9x tezroq!), 1 ajratish
\`\`\`

**4. Request Deduplication (so'rovlarni deduplikatsiya qilish)**
API gateway ko'rilgan request IDlarni kuzatadi:

\`\`\`go
// SEKIN - Pre-allocationsiz
func FilterDuplicates(requests []Request) []Request {
    seen := make(map[string]bool)  // 8 yozuvdan boshlanadi
    var unique []Request

    for _, req := range requests {
        if !seen[req.ID] {
            seen[req.ID] = true
            unique = append(unique, req)
        }
    }
    return unique
}
// 50k so'rov/sek: 400ms CPU vaqti, doimiy rehashing

// TEZ - Pre-allocated
func FilterDuplicates(requests []Request) []Request {
    seen := make(map[string]bool, len(requests))  // Rehashingsiz
    unique := make([]Request, 0, len(requests))

    for _, req := range requests {
        if !seen[req.ID] {
            seen[req.ID] = true
            unique = append(unique, req)
        }
    }
    return unique
}
// 50k so'rov/sek: 85ms CPU vaqti (4.7x tezroq!)
\`\`\`

**5. Memory Profiling Rehashing Narxini Ko'rsatadi**
\`\`\`bash
# Memory profile yaratamiz
go test -bench=MapAlloc -memprofile=mem.out

# pprof bilan tahlil qilamiz
go tool pprof mem.out

# OLDIN (o'lcham ko'rsatkichisiz):
(pprof) top
Total: 385.2 MB
    312.5 MB  runtime.makemap_small  # Boshlang'ich kichik map
     45.8 MB  runtime.mapassign      # Tayinlashlar
     26.9 MB  runtime.growWork       # Rehashing overhead!

# KEYIN (o'lcham ko'rsatkichi bilan):
(pprof) top
Total: 156.3 MB
    145.1 MB  runtime.makemap        # Bitta ajratish
     11.2 MB  runtime.mapassign      # Tayinlashlar
      0.0 MB  runtime.growWork       # Yo'qoldi!

# Rehashingni bartaraf etishdan 59% xotira qisqarishi!
\`\`\`

**6. CPU Profiling Rehashing Vaqtini Ko'rsatadi**
\`\`\`bash
go test -bench=MapAlloc -cpuprofile=cpu.out
go tool pprof cpu.out

# OLDIN:
(pprof) top10
Total: 12.5s
     8.2s  65.6%  runtime.growWork      # Rehashing!
     2.1s  16.8%  runtime.mapassign
     1.5s  12.0%  runtime.makemap_small
     0.7s   5.6%  fmt.Sprintf

# KEYIN:
(pprof) top10
Total: 2.8s
     1.6s  57.1%  fmt.Sprintf           # Asosiy ish
     0.8s  28.6%  runtime.mapassign
     0.4s  14.3%  runtime.makemap       # Bir martalik narx
     0.0s   0.0%  runtime.growWork      # Yo'qoldi!
\`\`\`

**7. Aggregation Pipeline (agregatsiya konveyeri)**
Analytics qayta ishlash hodisalarni foydalanuvchilar bo'yicha guruhlaydi:

\`\`\`go
// OLDIN - O'lcham ko'rsatkichlarisiz
func AggregateByUser(events []Event) map[int64][]Event {
    groups := make(map[int64][]Event)

    for _, event := range events {
        groups[event.UserID] = append(groups[event.UserID], event)
    }
    return groups
}
// 1M hodisa, 100k foydalanuvchi: 8.5 soniya

// KEYIN - Taxminiy o'lcham ko'rsatkichi
func AggregateByUser(events []Event) map[int64][]Event {
    // Noyob foydalanuvchilarni baholaymiz (masalan, hodisalarning 10%)
    estimatedUsers := len(events) / 10
    groups := make(map[int64][]Event, estimatedUsers)

    for _, event := range events {
        groups[event.UserID] = append(groups[event.UserID], event)
    }
    return groups
}
// 1M hodisa, 100k foydalanuvchi: 1.2 soniya (7x tezroq!)
\`\`\`

**8. Configuration Loading (konfiguratsiyani yuklash)**
Environment configni mapga yuklash:

\`\`\`go
// Fayldan config map qurish
func LoadConfig(lines []string) map[string]string {
    config := make(map[string]string, len(lines))

    for _, line := range lines {
        parts := strings.SplitN(line, "=", 2)
        if len(parts) == 2 {
            config[parts[0]] = parts[1]
        }
    }
    return config
}
// 1000 config yozuvlar: 0.8ms vs 4.5ms ko'rsatkichsiz (5.6x tezroq!)
\`\`\`

**9. Trace Profiling Visualization**
\`\`\`bash
# Trace yaratamiz
go test -bench=MapAlloc -trace=trace.out

# Brauzerda ko'ramiz
go tool trace trace.out

# OLDIN (o'lcham ko'rsatkichisiz):
# - Map o'sishi paytida ko'plab GC pauzalari
# - Ortiqcha ajratishlar GC ni ishga tushiradi
# - Rehashing CPU spikes sifatida ko'rinadi

# KEYIN (o'lcham ko'rsatkichi bilan):
# - Map yaratish paytida GC pauzalari yo'q
# - Silliq CPU foydalanishi
# - Bitta ajratish ko'rinadi
\`\`\`

**10. Ortiqcha Baholash Yaxshi**
\`\`\`go
// Aniq o'lcham noma'lum bo'lganda, ortiqcha baholang:
func BuildCache(data []Data) map[string]*Data {
    // 20% ga ortiqcha baholaymiz
    cache := make(map[string]*Data, len(data)*12/10)

    for _, d := range data {
        if d.Active {
            cache[d.Key] = &d
        }
    }
    return cache
}
// Ozgina xotira overhead rehashingdan yaxshiroq!
\`\`\`

**Real-World Ta'siri:**
Financial trading platformasi bozor ma'lumotlarini indekslashtirish:

**Oldin:** 50k orderlar uchun orderbook map qurish
- Vaqt: yangilanish uchun 450ms
- Qurish uchun 12 rehash operatsiyasi
- Xotira ajratishlari: qurish uchun 18
- Tizim yangilanishlar bilan qiynalmoqda

**Keyin:** O'lcham ko'rsatkichlari bilan oldindan ajratilgan maplar
- Vaqt: yangilanish uchun 45ms (10x tezroq!)
- Qurish uchun 1 ajratish
- Serverlar sonini 8 dan 2 ga kamaytirdik
- Infratuzilmada yiliga $180K tejalmoqda
- Tizim soniyasiga 5x ko'p yangilanishni qayta ishlaydi

**Production Best Practices:**
1. **Har doim o'lcham ko'rsatkichini bering**, o'lcham ma'lum yoki taxmin qilish mumkin bo'lganda
2. **make(map[K]V, n) ishlatiladi**, make(map[K]V) emas
3. **Ortiqcha baholang, kam baholamang** (ozgina xotira overhead yaxshi)
4. **-benchmem bilan profil qiling** ajratishlar kamayishini o'lchash uchun
5. **pprof ishlatiladi** rehashing bottlenecklarini aniqlash uchun (runtime.growWork)
6. **Noma'lum o'lchamlar uchun** odatiy ma'lumot patternlariga asoslangan holda baholang
7. **Slice pre-allocation bilan birlashtiriladi** maksimal ishlash uchun
8. **Oldin/keyin benchmark qiling** yaxshilanishlarni tekshirish uchun

**Map vs Slice Pre-allocation:**
\`\`\`go
// Maksimal ishlash uchun ikkalasi birgalikda
func ProcessBatch(items []Item) map[string][]Result {
    // Kategoriyalar uchun mapni oldindan ajratish
    groups := make(map[string][]Result, 20)

    // Har bir guruh uchun sliceni oldindan ajratish
    for _, item := range items {
        if _, exists := groups[item.Category]; !exists {
            groups[item.Category] = make([]Result, 0, 100)
        }
        result := process(item)
        groups[item.Category] = append(groups[item.Category], result)
    }
    return groups
}
\`\`\`

**Xulosa:**
Map oldindan ajratish oddiy, lekin kuchli optimallashtirish. make() da bitta qo'shimcha raqam 5-10x ishlash yaxshilanishini berishi va GC yukini sezilarli darajada kamaytirishi mumkin. Yuqori yukli production tizimlarida bu zo'rg'a kurashayotgan tizim va quvvat zaxirasi bo'lgan tizim o'rtasidagi farqdir.`
		}
	}
};

export default task;
