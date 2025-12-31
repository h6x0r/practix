import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-result-caching',
	title: 'Result Caching',
	difficulty: 'medium',
	tags: ['go', 'ml', 'caching', 'lru'],
	estimatedTime: '25m',
	isPremium: true,
	order: 1,
	description: `# Result Caching

Implement inference result caching with LRU eviction.

## Task

Build a result cache that:
- Caches inference outputs by input hash
- Implements LRU eviction policy
- Supports TTL-based expiration
- Handles cache invalidation

## Example

\`\`\`go
cache := NewResultCache(maxSize: 1000, ttl: 5*time.Minute)
result, hit := cache.Get(inputHash)
if !hit {
    result = model.Infer(input)
    cache.Set(inputHash, result)
}
\`\`\``,

	initialCode: `package main

import (
	"fmt"
	"time"
)

// ResultCache caches inference results
type ResultCache struct {
	// Your fields here
}

// NewResultCache creates a result cache
func NewResultCache(maxSize int, ttl time.Duration) *ResultCache {
	// Your code here
	return nil
}

// Get retrieves a cached result
func (c *ResultCache) Get(key string) ([]float32, bool) {
	// Your code here
	return nil, false
}

// Set stores a result in cache
func (c *ResultCache) Set(key string, value []float32) {
	// Your code here
}

// HashInput creates a cache key from input
func HashInput(input []float32) string {
	// Your code here
	return ""
}

func main() {
	fmt.Println("Result Caching")
}`,

	solutionCode: `package main

import (
	"container/list"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"time"
)

// cacheEntry holds cached value and metadata
type cacheEntry struct {
	key       string
	value     []float32
	expiresAt time.Time
	element   *list.Element
}

// ResultCache caches inference results
type ResultCache struct {
	maxSize int
	ttl     time.Duration

	cache   map[string]*cacheEntry
	lru     *list.List
	mu      sync.RWMutex

	hits   int64
	misses int64
}

// NewResultCache creates a result cache
func NewResultCache(maxSize int, ttl time.Duration) *ResultCache {
	c := &ResultCache{
		maxSize: maxSize,
		ttl:     ttl,
		cache:   make(map[string]*cacheEntry),
		lru:     list.New(),
	}

	// Start cleanup goroutine
	go c.cleanupLoop()

	return c
}

// Get retrieves a cached result
func (c *ResultCache) Get(key string) ([]float32, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, exists := c.cache[key]
	if !exists {
		c.misses++
		return nil, false
	}

	// Check expiration
	if time.Now().After(entry.expiresAt) {
		c.removeEntry(entry)
		c.misses++
		return nil, false
	}

	// Move to front (most recently used)
	c.lru.MoveToFront(entry.element)
	c.hits++

	// Return a copy to prevent modification
	result := make([]float32, len(entry.value))
	copy(result, entry.value)

	return result, true
}

// Set stores a result in cache
func (c *ResultCache) Set(key string, value []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if already exists
	if entry, exists := c.cache[key]; exists {
		entry.value = make([]float32, len(value))
		copy(entry.value, value)
		entry.expiresAt = time.Now().Add(c.ttl)
		c.lru.MoveToFront(entry.element)
		return
	}

	// Evict if at capacity
	for c.lru.Len() >= c.maxSize {
		c.evictOldest()
	}

	// Create new entry
	entry := &cacheEntry{
		key:       key,
		value:     make([]float32, len(value)),
		expiresAt: time.Now().Add(c.ttl),
	}
	copy(entry.value, value)

	entry.element = c.lru.PushFront(key)
	c.cache[key] = entry
}

// Delete removes an entry from cache
func (c *ResultCache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if entry, exists := c.cache[key]; exists {
		c.removeEntry(entry)
	}
}

// removeEntry removes entry from cache and LRU
func (c *ResultCache) removeEntry(entry *cacheEntry) {
	c.lru.Remove(entry.element)
	delete(c.cache, entry.key)
}

// evictOldest removes the least recently used entry
func (c *ResultCache) evictOldest() {
	oldest := c.lru.Back()
	if oldest == nil {
		return
	}

	key := oldest.Value.(string)
	if entry, exists := c.cache[key]; exists {
		c.removeEntry(entry)
	}
}

// cleanupLoop periodically removes expired entries
func (c *ResultCache) cleanupLoop() {
	ticker := time.NewTicker(c.ttl / 2)
	defer ticker.Stop()

	for range ticker.C {
		c.cleanup()
	}
}

// cleanup removes expired entries
func (c *ResultCache) cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	for key, entry := range c.cache {
		if now.After(entry.expiresAt) {
			c.removeEntry(entry)
			_ = key // silence unused warning
		}
	}
}

// Stats returns cache statistics
func (c *ResultCache) Stats() (hits, misses int64, hitRate float64) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := c.hits + c.misses
	if total > 0 {
		hitRate = float64(c.hits) / float64(total)
	}
	return c.hits, c.misses, hitRate
}

// Size returns current cache size
func (c *ResultCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.cache)
}

// Clear removes all entries
func (c *ResultCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache = make(map[string]*cacheEntry)
	c.lru.Init()
}

// HashInput creates a cache key from input
func HashInput(input []float32) string {
	h := sha256.New()
	for _, v := range input {
		bits := math.Float32bits(v)
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, bits)
		h.Write(buf)
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

// HashInputWithPrefix creates a cache key with model prefix
func HashInputWithPrefix(modelName string, input []float32) string {
	return modelName + ":" + HashInput(input)
}

func main() {
	cache := NewResultCache(100, 1*time.Minute)

	// Simulate inference with caching
	input := []float32{1.0, 2.0, 3.0}
	key := HashInput(input)

	// First call - cache miss
	result, hit := cache.Get(key)
	fmt.Printf("First call: hit=%v, result=%v\\n", hit, result)

	// Simulate inference
	result = []float32{2.0, 4.0, 6.0}
	cache.Set(key, result)

	// Second call - cache hit
	result, hit = cache.Get(key)
	fmt.Printf("Second call: hit=%v, result=%v\\n", hit, result)

	hits, misses, hitRate := cache.Stats()
	fmt.Printf("Stats: hits=%d, misses=%d, hitRate=%.2f\\n", hits, misses, hitRate)
}`,

	testCode: `package main

import (
	"testing"
	"time"
)

func TestResultCache(t *testing.T) {
	cache := NewResultCache(10, 1*time.Minute)

	cache.Set("key1", []float32{1, 2, 3})
	result, hit := cache.Get("key1")

	if !hit {
		t.Error("Should hit cache")
	}
	if len(result) != 3 {
		t.Errorf("Expected 3 elements, got %d", len(result))
	}
}

func TestCacheMiss(t *testing.T) {
	cache := NewResultCache(10, 1*time.Minute)

	_, hit := cache.Get("nonexistent")
	if hit {
		t.Error("Should miss cache")
	}
}

func TestCacheExpiration(t *testing.T) {
	cache := NewResultCache(10, 50*time.Millisecond)

	cache.Set("key1", []float32{1, 2, 3})

	// Should hit before expiration
	_, hit := cache.Get("key1")
	if !hit {
		t.Error("Should hit before expiration")
	}

	time.Sleep(100 * time.Millisecond)

	// Should miss after expiration
	_, hit = cache.Get("key1")
	if hit {
		t.Error("Should miss after expiration")
	}
}

func TestLRUEviction(t *testing.T) {
	cache := NewResultCache(3, 1*time.Minute)

	cache.Set("key1", []float32{1})
	cache.Set("key2", []float32{2})
	cache.Set("key3", []float32{3})
	cache.Set("key4", []float32{4}) // Should evict key1

	_, hit := cache.Get("key1")
	if hit {
		t.Error("key1 should be evicted")
	}

	_, hit = cache.Get("key4")
	if !hit {
		t.Error("key4 should exist")
	}
}

func TestHashInput(t *testing.T) {
	input1 := []float32{1, 2, 3}
	input2 := []float32{1, 2, 3}
	input3 := []float32{1, 2, 4}

	hash1 := HashInput(input1)
	hash2 := HashInput(input2)
	hash3 := HashInput(input3)

	if hash1 != hash2 {
		t.Error("Same inputs should have same hash")
	}
	if hash1 == hash3 {
		t.Error("Different inputs should have different hash")
	}
}

func TestCacheStats(t *testing.T) {
	cache := NewResultCache(10, 1*time.Minute)

	cache.Set("key1", []float32{1})
	cache.Get("key1") // hit
	cache.Get("key2") // miss

	hits, misses, hitRate := cache.Stats()
	if hits != 1 {
		t.Errorf("Expected 1 hit, got %d", hits)
	}
	if misses != 1 {
		t.Errorf("Expected 1 miss, got %d", misses)
	}
	if hitRate != 0.5 {
		t.Errorf("Expected 0.5 hit rate, got %f", hitRate)
	}
}

func TestDelete(t *testing.T) {
	cache := NewResultCache(10, 1*time.Minute)

	cache.Set("key1", []float32{1, 2, 3})
	cache.Delete("key1")

	_, hit := cache.Get("key1")
	if hit {
		t.Error("Should not find deleted key")
	}
}

func TestClear(t *testing.T) {
	cache := NewResultCache(10, 1*time.Minute)

	cache.Set("key1", []float32{1})
	cache.Set("key2", []float32{2})
	cache.Clear()

	if cache.Size() != 0 {
		t.Errorf("Expected size 0 after clear, got %d", cache.Size())
	}
}

func TestSize(t *testing.T) {
	cache := NewResultCache(10, 1*time.Minute)

	if cache.Size() != 0 {
		t.Error("Empty cache should have size 0")
	}

	cache.Set("key1", []float32{1})
	cache.Set("key2", []float32{2})

	if cache.Size() != 2 {
		t.Errorf("Expected size 2, got %d", cache.Size())
	}
}

func TestHashInputWithPrefix(t *testing.T) {
	input := []float32{1, 2, 3}
	key := HashInputWithPrefix("model-v1", input)

	if len(key) == 0 {
		t.Error("Hash should not be empty")
	}
	if key[:9] != "model-v1:" {
		t.Error("Should have model prefix")
	}
}`,

	hint1: 'Use container/list for O(1) LRU operations',
	hint2: 'Hash inputs with SHA256 for consistent cache keys',

	whyItMatters: `Result caching dramatically improves inference performance:

- **Latency reduction**: Cache hits are orders of magnitude faster
- **Cost savings**: Avoid redundant GPU computation
- **Traffic patterns**: Many ML workloads have repeating inputs
- **Scalability**: Handle more requests with same resources

Effective caching can reduce inference costs by 50% or more.`,

	translations: {
		ru: {
			title: 'Кэширование результатов',
			description: `# Кэширование результатов

Реализуйте кэширование результатов инференса с LRU вытеснением.

## Задача

Создайте кэш результатов:
- Кэширование выходов инференса по хешу входа
- Реализация политики вытеснения LRU
- Поддержка истечения по TTL
- Обработка инвалидации кэша

## Пример

\`\`\`go
cache := NewResultCache(maxSize: 1000, ttl: 5*time.Minute)
result, hit := cache.Get(inputHash)
if !hit {
    result = model.Infer(input)
    cache.Set(inputHash, result)
}
\`\`\``,
			hint1: 'Используйте container/list для O(1) операций LRU',
			hint2: 'Хешируйте входы с SHA256 для консистентных ключей кэша',
			whyItMatters: `Кэширование результатов драматически улучшает производительность инференса:

- **Снижение латентности**: Попадания в кэш на порядки быстрее
- **Экономия затрат**: Избежание избыточных вычислений GPU
- **Паттерны трафика**: Многие ML нагрузки имеют повторяющиеся входы
- **Масштабируемость**: Обработка большего количества запросов с теми же ресурсами`,
		},
		uz: {
			title: 'Natijalarni keshlash',
			description: `# Natijalarni keshlash

LRU chiqarish bilan inference natijalarini keshlashni amalga oshiring.

## Topshiriq

Natijalar keshini yarating:
- Kirish heshi bo'yicha inference chiqishlarini keshlash
- LRU chiqarish siyosatini amalga oshirish
- TTL asosida muddati tugashni qo'llab-quvvatlash
- Kesh bekor qilishni qayta ishlash

## Misol

\`\`\`go
cache := NewResultCache(maxSize: 1000, ttl: 5*time.Minute)
result, hit := cache.Get(inputHash)
if !hit {
    result = model.Infer(input)
    cache.Set(inputHash, result)
}
\`\`\``,
			hint1: "O(1) LRU operatsiyalari uchun container/list dan foydalaning",
			hint2: "Izchil kesh kalitlari uchun kirishlarni SHA256 bilan heshlang",
			whyItMatters: `Natijalarni keshlash inference ishlashini keskin yaxshilaydi:

- **Latency ni kamaytirish**: Kesh urishlari bir necha tartibda tezroq
- **Xarajatlarni tejash**: Ortiqcha GPU hisoblashlardan qochish
- **Trafik patternlari**: Ko'plab ML ish yuklari takrorlanuvchi kirishlarga ega
- **Masshtablilik**: Xuddi shu resurslar bilan ko'proq so'rovlarni qayta ishlash`,
		},
	},
};

export default task;
