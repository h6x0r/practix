import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-ap-premature-optimization-advanced',
	title: 'Premature Optimization - Advanced',
	difficulty: 'medium',
	tags: ['go', 'anti-patterns', 'premature-optimization', 'refactoring'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Understand when NOT to optimize and write clear, maintainable code instead.

**The Problem:**

Developers often sacrifice code clarity for marginal performance gains that don't matter in practice.

**You will implement:**

A simple cache with clear, maintainable code - no premature complexity.

**Implement:**
1. **Cache** - Simple in-memory cache
2. **Get** - Retrieve from cache
3. **Set** - Store in cache
4. **Clear** - Clear all cached values

**Your Task:**

Write simple, thread-safe code. Don't add complexity like LRU, TTL, or sharding until you measure the need.`,
	initialCode: `package antipatterns

import "sync"

type Cache struct {
	data map[string]string
	mu   sync.RWMutex
}

func NewCache() *Cache {
}

func (c *Cache) Get(key string) (string, bool) {
}

func (c *Cache) Set(key, value string) {
}

func (c *Cache) Clear() {
}`,
	solutionCode: `package antipatterns

import "sync"

// Cache is a simple, thread-safe in-memory cache
// KISS: No LRU, no TTL, no sharding - add only when needed
type Cache struct {
	data map[string]string	// simple map storage
	mu   sync.RWMutex		// thread-safe access
}

// NewCache creates a simple cache
// Start simple - complexity can be added later if profiling shows need
func NewCache() *Cache {
	return &Cache{
		data: make(map[string]string),	// empty map
	}
}

// Get retrieves value from cache
// Thread-safe read operation
func (c *Cache) Get(key string) (string, bool) {
	c.mu.RLock()			// read lock - allows concurrent reads
	defer c.mu.RUnlock()	// always unlock

	value, exists := c.data[key]	// check map
	return value, exists			// return value and existence flag
}

// Set stores value in cache
// Thread-safe write operation
func (c *Cache) Set(key, value string) {
	c.mu.Lock()			// write lock - exclusive access
	defer c.mu.Unlock()	// always unlock

	c.data[key] = value	// store in map
}

// Clear removes all cached entries
// Simple reset - create new map
func (c *Cache) Clear() {
	c.mu.Lock()			// write lock - exclusive access
	defer c.mu.Unlock()	// always unlock

	c.data = make(map[string]string)	// new empty map
}`,
	hint1: `NewCache returns &Cache{data: make(map[string]string)}. Get uses RLock, gets from map, returns value and bool. Set uses Lock, sets in map.`,
	hint2: `Clear uses Lock and creates a new map: c.data = make(map[string]string). All locking methods use defer to ensure unlock.`,
	whyItMatters: `Starting with simple, clear code allows you to add complexity only where it's proven necessary.

**The Trap of Over-Engineering:**

\`\`\`go
// BAD: Premature complexity - LRU cache with TTL, sharding, metrics
type AdvancedCache struct {
	shards    [256]*CacheShard  // sharding for "performance"
	ttl       time.Duration     // time-to-live
	maxSize   int               // LRU eviction
	hits      *atomic.Int64     // metrics
	misses    *atomic.Int64     // metrics
	evictions *atomic.Int64     // metrics
}

type CacheShard struct {
	data     map[string]*CacheEntry
	mu       sync.RWMutex
	lru      *list.List
	elements map[string]*list.Element
}

type CacheEntry struct {
	value     interface{}
	expiresAt time.Time
	lastUsed  time.Time
}

func (c *AdvancedCache) Get(key string) (interface{}, bool) {
	// 50 lines of complex logic
	// Sharding, LRU updates, TTL checks, metrics...
}

// 500+ lines total for a cache!
// Is this complexity justified? Profile first!

// GOOD: Start simple
type Cache struct {
	data map[string]string
	mu   sync.RWMutex
}

func (c *Cache) Get(key string) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	value, exists := c.data[key]
	return value, exists
}

// 20 lines total. Works for 99% of cases!
// Add complexity ONLY if profiling shows:
// - Memory issues (then add LRU)
// - Stale data issues (then add TTL)
// - Lock contention (then add sharding)
\`\`\`

**Evolution Based on Real Needs:**

\`\`\`go
// Version 1: Simple cache (start here!)
type Cache struct {
	data map[string]string
	mu   sync.RWMutex
}
// Serves 1000 requests/sec fine

// After profiling: memory growing unbounded
// Version 2: Add size limit (only what's needed!)
type Cache struct {
	data     map[string]string
	mu       sync.RWMutex
	maxSize  int  // NEW: only added because profiling showed need
}

func (c *Cache) Set(key, value string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(c.data) >= c.maxSize {
		// Simple eviction: delete random entry
		for k := range c.data {
			delete(c.data, k)
			break
		}
	}

	c.data[key] = value
}
// Still simple! Only added what profiling showed was necessary

// After profiling: stale data causing bugs
// Version 3: Add TTL (only what's needed!)
type CacheEntry struct {
	value     string
	expiresAt time.Time  // NEW: only added because we measured the problem
}

type Cache struct {
	data    map[string]CacheEntry
	mu      sync.RWMutex
	maxSize int
}
// Complexity grows ONLY based on measured needs!
\`\`\`

**The Right Questions:**

Before adding complexity, ask:
1. **Do we have profiling data showing a problem?**
2. **What is the measured impact?** (latency, memory, CPU)
3. **What's the simplest fix for this specific problem?**
4. **Can we test the improvement?** (benchmarks)

**Real-World Example:**

\`\`\`go
// Company built ultra-optimized cache:
// - 64-way sharding
// - LRU with 5 eviction strategies
// - TTL with sliding windows
// - Built-in metrics and monitoring
// - 2000 lines of code
// - 3 months of development

// After deployment with profiling:
// - Cache hit rate: 15% (low!)
// - 99% of time spent in... database queries
// - Cache "optimization" had zero impact on total performance

// The real solution:
// - Add database indexes (1 day of work)
// - 10x performance improvement
// - Simple map[string]string cache was fine all along!
\`\`\`

**Benchmark to Prove Optimizations:**

\`\`\`go
// Always benchmark before and after
func BenchmarkCacheSimple(b *testing.B) {
	cache := NewSimpleCache()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			cache.Set("key", "value")
			cache.Get("key")
		}
	})
}

func BenchmarkCacheOptimized(b *testing.B) {
	cache := NewOptimizedCache()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			cache.Set("key", "value")
			cache.Get("key")
		}
	})
}

// Results:
// BenchmarkCacheSimple     10000000    150 ns/op
// BenchmarkCacheOptimized  10000001    140 ns/op

// 10ns improvement (7%) for 500 lines of complexity?
// Not worth it! Keep it simple!
\`\`\`

**Golden Rules:**

1. **Readability > Performance** (until profiling proves otherwise)
2. **Simple > Clever** (you're not as smart as you think)
3. **Working > Perfect** (ship and measure)
4. **Measured > Assumed** (profile, don't guess)`,
	order: 11,
	testCode: `package antipatterns

import (
	"testing"
)

// Test1: NewCache returns non-nil
func Test1(t *testing.T) {
	cache := NewCache()
	if cache == nil {
		t.Error("NewCache should return non-nil")
	}
}

// Test2: Set and Get value
func Test2(t *testing.T) {
	cache := NewCache()
	cache.Set("key", "value")
	val, ok := cache.Get("key")
	if !ok || val != "value" {
		t.Error("Should get 'value'")
	}
}

// Test3: Get missing key returns false
func Test3(t *testing.T) {
	cache := NewCache()
	_, ok := cache.Get("missing")
	if ok {
		t.Error("Missing key should return false")
	}
}

// Test4: Clear removes all entries
func Test4(t *testing.T) {
	cache := NewCache()
	cache.Set("a", "1")
	cache.Set("b", "2")
	cache.Clear()
	_, ok := cache.Get("a")
	if ok {
		t.Error("Clear should remove all entries")
	}
}

// Test5: Set overwrites existing value
func Test5(t *testing.T) {
	cache := NewCache()
	cache.Set("key", "first")
	cache.Set("key", "second")
	val, _ := cache.Get("key")
	if val != "second" {
		t.Error("Should overwrite with 'second'")
	}
}

// Test6: Empty cache Get returns empty string
func Test6(t *testing.T) {
	cache := NewCache()
	val, ok := cache.Get("any")
	if ok || val != "" {
		t.Error("Empty cache should return empty string and false")
	}
}

// Test7: Multiple keys can be stored
func Test7(t *testing.T) {
	cache := NewCache()
	cache.Set("a", "1")
	cache.Set("b", "2")
	v1, _ := cache.Get("a")
	v2, _ := cache.Get("b")
	if v1 != "1" || v2 != "2" {
		t.Error("Both keys should be retrievable")
	}
}

// Test8: Clear on empty cache is safe
func Test8(t *testing.T) {
	cache := NewCache()
	cache.Clear() // should not panic
}

// Test9: Set empty string value
func Test9(t *testing.T) {
	cache := NewCache()
	cache.Set("key", "")
	val, ok := cache.Get("key")
	if !ok || val != "" {
		t.Error("Should store empty string")
	}
}

// Test10: Get after Clear returns false
func Test10(t *testing.T) {
	cache := NewCache()
	cache.Set("key", "value")
	cache.Clear()
	_, ok := cache.Get("key")
	if ok {
		t.Error("Key should not exist after Clear")
	}
}
`,
	translations: {
		ru: {
			title: 'Преждевременная оптимизация - Продвинутый',
			description: `Поймите, когда НЕ надо оптимизировать, и вместо этого пишите понятный, поддерживаемый код.`,
			hint1: `NewCache возвращает &Cache{data: make(map[string]string)}. Get использует RLock, получает из map, возвращает значение и bool. Set использует Lock, устанавливает в map.`,
			hint2: `Clear использует Lock и создаёт новый map: c.data = make(map[string]string). Все методы блокировки используют defer для гарантии разблокировки.`,
			whyItMatters: `Начиная с простого, понятного кода, вы можете добавлять сложность только там, где это доказано необходимым.`,
			solutionCode: `package antipatterns

import "sync"

type Cache struct {
	data map[string]string
	mu   sync.RWMutex
}

func NewCache() *Cache {
	return &Cache{
		data: make(map[string]string),
	}
}

func (c *Cache) Get(key string) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	value, exists := c.data[key]
	return value, exists
}

func (c *Cache) Set(key, value string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.data[key] = value
}

func (c *Cache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.data = make(map[string]string)
}`
		},
		uz: {
			title: 'Erta Optimizatsiya - Ilg\'or',
			description: `Qachon optimallashtirmaslik kerakligini tushuning va buning o'rniga aniq, qo'llab-quvvatlanadigan kod yozing.`,
			hint1: `NewCache &Cache{data: make(map[string]string)} ni qaytaradi. Get RLock ishlatadi, map dan oladi, qiymat va bool qaytaradi. Set Lock ishlatadi, map ga o'rnatadi.`,
			hint2: `Clear Lock ishlatadi va yangi map yaratadi: c.data = make(map[string]string). Barcha qulflash metodlari qulfni ochishni kafolatlash uchun defer ishlatadi.`,
			whyItMatters: `Oddiy, aniq kod bilan boshlash faqat zarur deb isbotlangan joylarda murakkablikni qo'shish imkonini beradi.`,
			solutionCode: `package antipatterns

import "sync"

type Cache struct {
	data map[string]string
	mu   sync.RWMutex
}

func NewCache() *Cache {
	return &Cache{
		data: make(map[string]string),
	}
}

func (c *Cache) Get(key string) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	value, exists := c.data[key]
	return value, exists
}

func (c *Cache) Set(key, value string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.data[key] = value
}

func (c *Cache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.data = make(map[string]string)
}`
		}
	}
};

export default task;
