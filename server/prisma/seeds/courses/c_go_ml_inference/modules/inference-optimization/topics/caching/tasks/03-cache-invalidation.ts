import { Task } from '../../../../../../../types';

const task: Task = {
  slug: 'goml-cache-invalidation',
  title: 'Cache Invalidation Strategies',
  difficulty: 'medium',
  tags: ['go', 'ml', 'caching', 'invalidation', 'versioning'],
  estimatedTime: '25m',
  isPremium: false,
  order: 3,

  description: `
## Cache Invalidation Strategies

Implement cache invalidation strategies for ML inference that handle model updates, TTL expiration, and dependency tracking.

### Requirements

1. **VersionedCache** - Cache with model version tracking:
   - \`NewVersionedCache(modelVersion string)\` - Create with model version
   - \`Get(key string) ([]float32, bool)\` - Get if version matches
   - \`Set(key string, value []float32)\` - Store with current version
   - \`UpdateVersion(newVersion string)\` - Invalidate all entries
   - \`Stats() CacheStats\` - Get cache statistics

2. **InvalidationStrategies**:
   - **TTL-based**: Entries expire after fixed duration
   - **Version-based**: Entries invalidated on model update
   - **Dependency-based**: Invalidate related entries together
   - **LRU eviction**: Remove least recently used when full

3. **CacheStats**:
   - Hit rate
   - Miss rate
   - Invalidation count
   - Current size

4. **Advanced Features**:
   - Lazy invalidation (check on access)
   - Background cleanup goroutine
   - Atomic version updates

### Example

\`\`\`go
cache := NewVersionedCache("v1.0")

// Cache inference result
cache.Set("user_123_features", result)

// Get cached result
if cached, ok := cache.Get("user_123_features"); ok {
    return cached
}

// Model update invalidates all
cache.UpdateVersion("v1.1")

// Previous cache entries now miss
_, ok := cache.Get("user_123_features") // ok == false
\`\`\`
`,

  initialCode: `package cacheinvalidation

import (
	"sync"
	"time"
)

type CacheStats struct {
	Hits          int64
	Misses        int64
	Invalidations int64
	Size          int
	HitRate       float64
}

type CacheEntry struct {
	Value     []float32
	Version   string
	ExpiresAt time.Time
	CreatedAt time.Time
}

type VersionedCache struct {
}

func NewVersionedCache(modelVersion string) *VersionedCache {
	return nil
}

func (c *VersionedCache) Get(key string) ([]float32, bool) {
	return nil, false
}

func (c *VersionedCache) Set(key string, value []float32) {
}

func (c *VersionedCache) UpdateVersion(newVersion string) {
}

func (c *VersionedCache) Stats() CacheStats {
	return CacheStats{}
}

func (c *VersionedCache) SetTTL(ttl time.Duration) {
}

type DependencyCache struct {
}

func NewDependencyCache() *DependencyCache {
	return nil
}

func (c *DependencyCache) SetWithDependencies(key string, value []float32, deps []string) {
}

func (c *DependencyCache) InvalidateDependency(dep string) {
}`,

  solutionCode: `package cacheinvalidation

import (
	"sync"
	"sync/atomic"
	"time"
)

// CacheStats holds cache statistics
type CacheStats struct {
	Hits          int64
	Misses        int64
	Invalidations int64
	Size          int
	HitRate       float64
}

// CacheEntry holds a cached value with metadata
type CacheEntry struct {
	Value     []float32
	Version   string
	ExpiresAt time.Time
	CreatedAt time.Time
}

// VersionedCache caches inference results with version tracking
type VersionedCache struct {
	version       string
	entries       map[string]*CacheEntry
	ttl           time.Duration
	maxSize       int
	mu            sync.RWMutex

	hits          int64
	misses        int64
	invalidations int64

	cleanupDone   chan struct{}
}

// NewVersionedCache creates a new versioned cache
func NewVersionedCache(modelVersion string) *VersionedCache {
	c := &VersionedCache{
		version:     modelVersion,
		entries:     make(map[string]*CacheEntry),
		ttl:         0, // No TTL by default
		maxSize:     10000,
		cleanupDone: make(chan struct{}),
	}

	go c.cleanupLoop()

	return c
}

func (c *VersionedCache) cleanupLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.cleanup()
		case <-c.cleanupDone:
			return
		}
	}
}

func (c *VersionedCache) cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	for key, entry := range c.entries {
		if !entry.ExpiresAt.IsZero() && now.After(entry.ExpiresAt) {
			delete(c.entries, key)
			atomic.AddInt64(&c.invalidations, 1)
		}
	}
}

// Get retrieves a value if it exists and version matches
func (c *VersionedCache) Get(key string) ([]float32, bool) {
	c.mu.RLock()
	entry, exists := c.entries[key]
	currentVersion := c.version
	c.mu.RUnlock()

	if !exists {
		atomic.AddInt64(&c.misses, 1)
		return nil, false
	}

	// Check version
	if entry.Version != currentVersion {
		c.mu.Lock()
		delete(c.entries, key)
		c.mu.Unlock()
		atomic.AddInt64(&c.misses, 1)
		atomic.AddInt64(&c.invalidations, 1)
		return nil, false
	}

	// Check TTL
	if !entry.ExpiresAt.IsZero() && time.Now().After(entry.ExpiresAt) {
		c.mu.Lock()
		delete(c.entries, key)
		c.mu.Unlock()
		atomic.AddInt64(&c.misses, 1)
		atomic.AddInt64(&c.invalidations, 1)
		return nil, false
	}

	atomic.AddInt64(&c.hits, 1)
	return entry.Value, true
}

// Set stores a value with the current model version
func (c *VersionedCache) Set(key string, value []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Evict if at capacity
	if len(c.entries) >= c.maxSize {
		c.evictOldest()
	}

	entry := &CacheEntry{
		Value:     value,
		Version:   c.version,
		CreatedAt: time.Now(),
	}

	if c.ttl > 0 {
		entry.ExpiresAt = time.Now().Add(c.ttl)
	}

	c.entries[key] = entry
}

func (c *VersionedCache) evictOldest() {
	var oldestKey string
	var oldestTime time.Time

	for key, entry := range c.entries {
		if oldestTime.IsZero() || entry.CreatedAt.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.CreatedAt
		}
	}

	if oldestKey != "" {
		delete(c.entries, oldestKey)
	}
}

// UpdateVersion updates the model version and invalidates old entries
func (c *VersionedCache) UpdateVersion(newVersion string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	oldCount := len(c.entries)
	c.version = newVersion
	// Clear all entries with old version
	c.entries = make(map[string]*CacheEntry)
	atomic.AddInt64(&c.invalidations, int64(oldCount))
}

// Stats returns cache statistics
func (c *VersionedCache) Stats() CacheStats {
	c.mu.RLock()
	size := len(c.entries)
	c.mu.RUnlock()

	hits := atomic.LoadInt64(&c.hits)
	misses := atomic.LoadInt64(&c.misses)

	total := hits + misses
	var hitRate float64
	if total > 0 {
		hitRate = float64(hits) / float64(total)
	}

	return CacheStats{
		Hits:          hits,
		Misses:        misses,
		Invalidations: atomic.LoadInt64(&c.invalidations),
		Size:          size,
		HitRate:       hitRate,
	}
}

// SetTTL sets time-to-live for cache entries
func (c *VersionedCache) SetTTL(ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ttl = ttl
}

// SetMaxSize sets maximum cache size
func (c *VersionedCache) SetMaxSize(maxSize int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.maxSize = maxSize
}

// Close stops the cleanup goroutine
func (c *VersionedCache) Close() {
	close(c.cleanupDone)
}

// DependencyCache tracks dependencies between cache entries
type DependencyCache struct {
	entries      map[string][]float32
	dependencies map[string]map[string]bool // dep -> set of keys
	keyDeps      map[string][]string        // key -> list of deps
	mu           sync.RWMutex
}

// NewDependencyCache creates a cache with dependency tracking
func NewDependencyCache() *DependencyCache {
	return &DependencyCache{
		entries:      make(map[string][]float32),
		dependencies: make(map[string]map[string]bool),
		keyDeps:      make(map[string][]string),
	}
}

// Get retrieves a cached value
func (c *DependencyCache) Get(key string) ([]float32, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	val, ok := c.entries[key]
	return val, ok
}

// SetWithDependencies stores a value with dependency keys
func (c *DependencyCache) SetWithDependencies(key string, value []float32, deps []string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries[key] = value
	c.keyDeps[key] = deps

	for _, dep := range deps {
		if c.dependencies[dep] == nil {
			c.dependencies[dep] = make(map[string]bool)
		}
		c.dependencies[dep][key] = true
	}
}

// InvalidateDependency invalidates all entries depending on a key
func (c *DependencyCache) InvalidateDependency(dep string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	keys, exists := c.dependencies[dep]
	if !exists {
		return
	}

	for key := range keys {
		delete(c.entries, key)
		// Clean up key's dependency entries
		if deps, ok := c.keyDeps[key]; ok {
			for _, d := range deps {
				if c.dependencies[d] != nil {
					delete(c.dependencies[d], key)
				}
			}
			delete(c.keyDeps, key)
		}
	}

	delete(c.dependencies, dep)
}

// Clear removes all entries
func (c *DependencyCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries = make(map[string][]float32)
	c.dependencies = make(map[string]map[string]bool)
	c.keyDeps = make(map[string][]string)
}
`,

  testCode: `package cacheinvalidation

import (
	"testing"
	"time"
)

func TestNewVersionedCache(t *testing.T) {
	cache := NewVersionedCache("v1.0")
	defer cache.Close()

	if cache == nil {
		t.Fatal("Expected non-nil cache")
	}
}

func TestGetSet(t *testing.T) {
	cache := NewVersionedCache("v1.0")
	defer cache.Close()

	value := []float32{1.0, 2.0, 3.0}
	cache.Set("key1", value)

	result, ok := cache.Get("key1")
	if !ok {
		t.Fatal("Expected cache hit")
	}

	if len(result) != 3 || result[0] != 1.0 {
		t.Error("Value mismatch")
	}
}

func TestVersionInvalidation(t *testing.T) {
	cache := NewVersionedCache("v1.0")
	defer cache.Close()

	cache.Set("key1", []float32{1.0})

	// Should hit
	_, ok := cache.Get("key1")
	if !ok {
		t.Error("Expected cache hit before version update")
	}

	// Update version
	cache.UpdateVersion("v1.1")

	// Should miss
	_, ok = cache.Get("key1")
	if ok {
		t.Error("Expected cache miss after version update")
	}
}

func TestTTLExpiration(t *testing.T) {
	cache := NewVersionedCache("v1.0")
	defer cache.Close()

	cache.SetTTL(50 * time.Millisecond)
	cache.Set("key1", []float32{1.0})

	// Should hit immediately
	_, ok := cache.Get("key1")
	if !ok {
		t.Error("Expected cache hit before TTL")
	}

	// Wait for expiration
	time.Sleep(100 * time.Millisecond)

	// Should miss
	_, ok = cache.Get("key1")
	if ok {
		t.Error("Expected cache miss after TTL")
	}
}

func TestCacheStats(t *testing.T) {
	cache := NewVersionedCache("v1.0")
	defer cache.Close()

	cache.Set("key1", []float32{1.0})

	cache.Get("key1")     // hit
	cache.Get("key1")     // hit
	cache.Get("missing")  // miss

	stats := cache.Stats()

	if stats.Hits != 2 {
		t.Errorf("Expected 2 hits, got %d", stats.Hits)
	}
	if stats.Misses != 1 {
		t.Errorf("Expected 1 miss, got %d", stats.Misses)
	}
	if stats.Size != 1 {
		t.Errorf("Expected size 1, got %d", stats.Size)
	}
}

func TestHitRate(t *testing.T) {
	cache := NewVersionedCache("v1.0")
	defer cache.Close()

	cache.Set("key1", []float32{1.0})

	for i := 0; i < 4; i++ {
		cache.Get("key1") // 4 hits
	}
	cache.Get("missing") // 1 miss

	stats := cache.Stats()
	expectedRate := 0.8 // 4/5

	if stats.HitRate < expectedRate-0.01 || stats.HitRate > expectedRate+0.01 {
		t.Errorf("Expected hit rate ~%.2f, got %.2f", expectedRate, stats.HitRate)
	}
}

func TestMaxSize(t *testing.T) {
	cache := NewVersionedCache("v1.0")
	defer cache.Close()

	cache.SetMaxSize(3)

	cache.Set("key1", []float32{1.0})
	cache.Set("key2", []float32{2.0})
	cache.Set("key3", []float32{3.0})
	cache.Set("key4", []float32{4.0}) // Should evict oldest

	stats := cache.Stats()
	if stats.Size > 3 {
		t.Errorf("Expected size <= 3, got %d", stats.Size)
	}
}

func TestDependencyCache(t *testing.T) {
	cache := NewDependencyCache()

	cache.SetWithDependencies("user_123_profile", []float32{1.0}, []string{"user_123"})
	cache.SetWithDependencies("user_123_prefs", []float32{2.0}, []string{"user_123"})
	cache.SetWithDependencies("user_456_profile", []float32{3.0}, []string{"user_456"})

	// All should exist
	_, ok1 := cache.Get("user_123_profile")
	_, ok2 := cache.Get("user_123_prefs")
	_, ok3 := cache.Get("user_456_profile")

	if !ok1 || !ok2 || !ok3 {
		t.Error("Expected all entries to exist")
	}

	// Invalidate user_123
	cache.InvalidateDependency("user_123")

	// user_123 entries should be gone
	_, ok1 = cache.Get("user_123_profile")
	_, ok2 = cache.Get("user_123_prefs")
	_, ok3 = cache.Get("user_456_profile")

	if ok1 || ok2 {
		t.Error("Expected user_123 entries to be invalidated")
	}
	if !ok3 {
		t.Error("Expected user_456 entry to still exist")
	}
}

func TestDependencyCacheClear(t *testing.T) {
	cache := NewDependencyCache()

	cache.SetWithDependencies("key1", []float32{1.0}, []string{"dep1"})
	cache.SetWithDependencies("key2", []float32{2.0}, []string{"dep1"})

	cache.Clear()

	_, ok := cache.Get("key1")
	if ok {
		t.Error("Expected cache to be empty after clear")
	}
}

func TestMultipleDependencies(t *testing.T) {
	cache := NewDependencyCache()

	// Key depends on multiple dependencies
	cache.SetWithDependencies("combined_data", []float32{1.0}, []string{"user_123", "product_456"})
	cache.SetWithDependencies("user_only", []float32{2.0}, []string{"user_123"})

	// Invalidating product should only affect combined_data
	cache.InvalidateDependency("product_456")

	_, ok1 := cache.Get("combined_data")
	_, ok2 := cache.Get("user_only")

	if ok1 {
		t.Error("combined_data should be invalidated")
	}
	if !ok2 {
		t.Error("user_only should still exist")
	}
}
`,

  hint1: `Store model version with each cache entry. On Get, compare entry version with current version. If mismatch, treat as cache miss and delete the stale entry.`,

  hint2: `For dependency tracking, maintain two maps: one from dependency to set of keys, another from key to its dependencies. When invalidating, use the first map to find affected keys.`,

  whyItMatters: `Cache invalidation is critical when ML models are updated. Stale predictions from old model versions can cause inconsistent user experiences. Version-based invalidation ensures fresh predictions without manual cache clearing.`,

  translations: {
    ru: {
      title: 'Стратегии Инвалидации Кэша',
      description: `
## Стратегии Инвалидации Кэша

Реализуйте стратегии инвалидации кэша для ML-инференса, которые обрабатывают обновления модели, TTL истечение и отслеживание зависимостей.

### Требования

1. **VersionedCache** - Кэш с отслеживанием версии модели:
   - \`NewVersionedCache(modelVersion string)\` - Создание с версией модели
   - \`Get(key string) ([]float32, bool)\` - Получение если версия совпадает
   - \`Set(key string, value []float32)\` - Сохранение с текущей версией
   - \`UpdateVersion(newVersion string)\` - Инвалидация всех записей
   - \`Stats() CacheStats\` - Получение статистики кэша

2. **Стратегии инвалидации**:
   - **На основе TTL**: Записи истекают после фиксированного времени
   - **На основе версии**: Записи инвалидируются при обновлении модели
   - **На основе зависимостей**: Инвалидация связанных записей вместе
   - **LRU вытеснение**: Удаление наименее используемых при переполнении

3. **CacheStats**:
   - Hit rate
   - Miss rate
   - Количество инвалидаций
   - Текущий размер

4. **Расширенные возможности**:
   - Ленивая инвалидация (проверка при доступе)
   - Фоновая горутина очистки
   - Атомарные обновления версии

### Пример

\`\`\`go
cache := NewVersionedCache("v1.0")

// Кэширование результата инференса
cache.Set("user_123_features", result)

// Получение кэшированного результата
if cached, ok := cache.Get("user_123_features"); ok {
    return cached
}

// Обновление модели инвалидирует всё
cache.UpdateVersion("v1.1")

// Предыдущие записи кэша теперь промахи
_, ok := cache.Get("user_123_features") // ok == false
\`\`\`
`,
      hint1: 'Храните версию модели с каждой записью кэша. При Get сравнивайте версию записи с текущей. Если не совпадает, считайте промахом и удаляйте устаревшую запись.',
      hint2: 'Для отслеживания зависимостей используйте две map: от зависимости к множеству ключей и от ключа к его зависимостям. При инвалидации используйте первую map для нахождения затронутых ключей.',
      whyItMatters: 'Инвалидация кэша критически важна при обновлении ML-моделей. Устаревшие предсказания от старых версий модели могут вызвать несогласованный пользовательский опыт. Инвалидация на основе версий обеспечивает свежие предсказания без ручной очистки кэша.',
    },
    uz: {
      title: 'Kesh Invalidatsiya Strategiyalari',
      description: `
## Kesh Invalidatsiya Strategiyalari

Model yangilanishlarini, TTL tugashini va bog'liqlik kuzatishini boshqaradigan ML inference uchun kesh invalidatsiya strategiyalarini amalga oshiring.

### Talablar

1. **VersionedCache** - Model versiyasini kuzatuvchi kesh:
   - \`NewVersionedCache(modelVersion string)\` - Model versiyasi bilan yaratish
   - \`Get(key string) ([]float32, bool)\` - Versiya mos kelsa olish
   - \`Set(key string, value []float32)\` - Joriy versiya bilan saqlash
   - \`UpdateVersion(newVersion string)\` - Barcha yozuvlarni invalidatsiya qilish
   - \`Stats() CacheStats\` - Kesh statistikasini olish

2. **Invalidatsiya strategiyalari**:
   - **TTL asosida**: Yozuvlar belgilangan vaqtdan keyin tugaydi
   - **Versiya asosida**: Model yangilanganda yozuvlar invalidatsiya qilinadi
   - **Bog'liqlik asosida**: Bog'liq yozuvlarni birga invalidatsiya qilish
   - **LRU chiqarish**: To'lganda eng kam ishlatilganlarni o'chirish

3. **CacheStats**:
   - Hit rate
   - Miss rate
   - Invalidatsiyalar soni
   - Joriy hajm

4. **Kengaytirilgan xususiyatlar**:
   - Lazy invalidatsiya (kirish vaqtida tekshirish)
   - Fon tozalash goroutinei
   - Atomik versiya yangilanishlari

### Misol

\`\`\`go
cache := NewVersionedCache("v1.0")

// Inference natijasini keshlash
cache.Set("user_123_features", result)

// Keshlangan natijani olish
if cached, ok := cache.Get("user_123_features"); ok {
    return cached
}

// Model yangilanishi hammasini invalidatsiya qiladi
cache.UpdateVersion("v1.1")

// Oldingi kesh yozuvlari endi miss
_, ok := cache.Get("user_123_features") // ok == false
\`\`\`
`,
      hint1: "Model versiyasini har bir kesh yozuvi bilan saqlang. Get da yozuv versiyasini joriy versiya bilan solishtiring. Mos kelmasa, kesh miss deb hisoblang va eskirgan yozuvni o'chiring.",
      hint2: "Bog'liqlik kuzatish uchun ikkita map saqlang: bog'liqlikdan kalitlar to'plamiga va kalitdan uning bog'liqliklariga. Invalidatsiya qilganda, ta'sirlangan kalitlarni topish uchun birinchi mapdan foydalaning.",
      whyItMatters: "ML modellar yangilanganda kesh invalidatsiyasi juda muhim. Eski model versiyalaridan eskirgan bashoratlar nomuvofiq foydalanuvchi tajribasiga sabab bo'lishi mumkin. Versiya asosidagi invalidatsiya qo'lda kesh tozalashsiz yangi bashoratlarni ta'minlaydi.",
    },
  },
};

export default task;
