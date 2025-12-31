import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-ttlcache-len',
	title: 'Len Method',
	difficulty: 'medium',	tags: ['go', 'cache', 'metrics', 'cleanup'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Len** method to return the count of live (non-expired) entries.

**Requirements:**
1. Implement \`Len() int\` method on TTLCache
2. Return \`0\` if cache is nil
3. Capture current time once: \`now := time.Now()\`
4. Use exclusive lock (Lock/Unlock) for atomic cleanup and count
5. Iterate over all entries and delete expired ones: \`!ent.exp.IsZero() && !now.Before(ent.exp)\`
6. Return \`len(c.m)\` after cleanup (count of remaining live entries)

**Example:**
\`\`\`go
cache := NewTTLCache(100 * time.Millisecond)
cache.Set("key1", "value1")
cache.Set("key2", "value2")

count := cache.Len()  // 2 (both alive)

time.Sleep(150 * time.Millisecond)
count = cache.Len()   // 0 (both expired and cleaned up)
\`\`\`

**Constraints:**
- Must clean up expired entries before counting (ensure accurate count)
- Must capture time once to ensure consistent expiration check
- Must use Lock (not RLock) because cleanup mutates map`,
	initialCode: `package cache

import (
	"sync"
	"time"
)

type entry struct {
	v   any
	exp time.Time
}

type TTLCache struct {
	mu  sync.RWMutex
	m   map[string]entry
	ttl time.Duration
}

func NewTTLCache(ttl time.Duration) *TTLCache {
	return &TTLCache{m: make(map[string]entry), ttl: ttl}
}

func (c *TTLCache) Set(key string, v any) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	expire := time.Time{}
	if c.ttl > 0 {
		expire = time.Now().Add(c.ttl)
	}
	c.m[key] = entry{v: v, exp: expire}
}

func (c *TTLCache) Get(key string) (any, bool) {
	if c == nil {
		return nil, false
	}
	c.mu.RLock()
	ent, ok := c.m[key]
	c.mu.RUnlock()
	if !ok {
		return nil, false
	}
	if !ent.exp.IsZero() && time.Now().After(ent.exp) {
		c.mu.Lock()
		defer c.mu.Unlock()
		if entCur, still := c.m[key]; still && entCur.exp == ent.exp {
			delete(c.m, key)
		}
		return nil, false
	}
	return ent.v, true
}

func (c *TTLCache) cleanupNow(now time.Time) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for k, ent := range c.m {
		if !ent.exp.IsZero() && !now.Before(ent.exp) {
			delete(c.m, k)
		}
	}
}

func (c *TTLCache) Delete(key string) bool {
	if c == nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.m[key]; ok {
		delete(c.m, key)
		return true
	}
	return false
}

// TODO: Implement Len method
// 1. Check if c is nil, return 0
// 2. Capture current time once
// 3. Use Lock for exclusive access
// 4. Iterate and delete expired entries (like cleanupNow)
// 5. Return len(c.m) after cleanup
func (c *TTLCache) Len() int {
	return 0 // TODO: Implement
}`,
	solutionCode: `package cache

import (
	"sync"
	"time"
)

type entry struct {
	v   any
	exp time.Time
}

type TTLCache struct {
	mu  sync.RWMutex
	m   map[string]entry
	ttl time.Duration
}

func NewTTLCache(ttl time.Duration) *TTLCache {
	return &TTLCache{m: make(map[string]entry), ttl: ttl}
}

func (c *TTLCache) Set(key string, v any) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	expire := time.Time{}
	if c.ttl > 0 {
		expire = time.Now().Add(c.ttl)
	}
	c.m[key] = entry{v: v, exp: expire}
}

func (c *TTLCache) Get(key string) (any, bool) {
	if c == nil {
		return nil, false
	}
	c.mu.RLock()
	ent, ok := c.m[key]
	c.mu.RUnlock()
	if !ok {
		return nil, false
	}
	if !ent.exp.IsZero() && time.Now().After(ent.exp) {
		c.mu.Lock()
		defer c.mu.Unlock()
		if entCur, still := c.m[key]; still && entCur.exp == ent.exp {
			delete(c.m, key)
		}
		return nil, false
	}
	return ent.v, true
}

func (c *TTLCache) cleanupNow(now time.Time) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for k, ent := range c.m {
		if !ent.exp.IsZero() && !now.Before(ent.exp) {
			delete(c.m, k)
		}
	}
}

func (c *TTLCache) Delete(key string) bool {
	if c == nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.m[key]; ok {
		delete(c.m, key)
		return true
	}
	return false
}

func (c *TTLCache) Len() int {
	if c == nil {	// Nil cache behaves like empty cache
		return 0
	}
	now := time.Now()	// Capture current time once to avoid repeat calls
	c.mu.Lock()	// Lock to safely mutate and count
	defer c.mu.Unlock()	// Ensure lock release even if counting fails
	for k, ent := range c.m {	// Visit every entry to check expiration
		if !ent.exp.IsZero() && !now.Before(ent.exp) {	// Expired entries are removed eagerly
			delete(c.m, k)	// Delete stale entry before counting
		}
	}
	return len(c.m)	// Return number of remaining live entries
}`,
	testCode: `package cache

import (
	"testing"
	"time"
)

func TestLen_EmptyCache(t *testing.T) {
	cache := NewTTLCache(time.Second)
	length := cache.Len()
	if length != 0 {
		t.Errorf("expected length 0 for empty cache, got %d", length)
	}
}

func TestLen_SingleEntry(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	length := cache.Len()
	if length != 1 {
		t.Errorf("expected length 1, got %d", length)
	}
}

func TestLen_MultipleEntries(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	cache.Set("key3", "value3")
	length := cache.Len()
	if length != 3 {
		t.Errorf("expected length 3, got %d", length)
	}
}

func TestLen_NilCache(t *testing.T) {
	var cache *TTLCache
	length := cache.Len()
	if length != 0 {
		t.Errorf("expected length 0 for nil cache, got %d", length)
	}
}

func TestLen_AfterExpiration(t *testing.T) {
	cache := NewTTLCache(50 * time.Millisecond)
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	time.Sleep(100 * time.Millisecond)
	length := cache.Len()
	if length != 0 {
		t.Errorf("expected length 0 after expiration, got %d", length)
	}
}

func TestLen_PartialExpiration(t *testing.T) {
	cache := NewTTLCache(100 * time.Millisecond)
	cache.Set("key1", "value1")
	time.Sleep(50 * time.Millisecond)
	cache.Set("key2", "value2")
	time.Sleep(60 * time.Millisecond)
	length := cache.Len()
	if length != 1 {
		t.Errorf("expected length 1 after partial expiration, got %d", length)
	}
}

func TestLen_ZeroTTL(t *testing.T) {
	cache := NewTTLCache(0)
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	time.Sleep(100 * time.Millisecond)
	length := cache.Len()
	if length != 2 {
		t.Errorf("expected length 2 (zero ttl never expires), got %d", length)
	}
}

func TestLen_AfterDelete(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	cache.Delete("key1")
	length := cache.Len()
	if length != 1 {
		t.Errorf("expected length 1 after delete, got %d", length)
	}
}

func TestLen_CleansUpExpired(t *testing.T) {
	cache := NewTTLCache(50 * time.Millisecond)
	cache.Set("key1", "value1")
	time.Sleep(100 * time.Millisecond)
	length := cache.Len()
	if len(cache.m) != 0 {
		t.Error("expected Len to clean up expired entries from map")
	}
	if length != 0 {
		t.Errorf("expected length 0, got %d", length)
	}
}

func TestLen_Overwrite(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	cache.Set("key1", "value2")
	length := cache.Len()
	if length != 1 {
		t.Errorf("expected length 1 after overwrite, got %d", length)
	}
}
`,
			hint1: `Capture time.Now() once at the start. Use it for all expiration checks to ensure consistency.`,
			hint2: `Inline the cleanup logic instead of calling cleanupNow() to hold single lock throughout.`,
			whyItMatters: `Len() provides accurate cache metrics by eagerly cleaning expired entries and counting live ones.

**Why Clean Before Count:**
\`\`\`go
// WRONG: Count includes expired entries
func (c *TTLCache) Len() int {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return len(c.m)  // Includes expired entries!
}

// CORRECT: Clean then count
func (c *TTLCache) Len() int {
    now := time.Now()
    c.mu.Lock()
    defer c.mu.Unlock()
    for k, ent := range c.m {
        if expired(ent, now) {
            delete(c.m, k)  // Remove before counting
        }
    }
    return len(c.m)  // Accurate live count
}
\`\`\`

**Why Capture Time Once:**
\`\`\`go
// WRONG: Time changes during iteration
for k, ent := range c.m {
    if time.Now().After(ent.exp) {  // Different time each iteration!
        delete(c.m, k)
    }
}

// CORRECT: Consistent snapshot
now := time.Now()  // Single capture
for k, ent := range c.m {
    if now.After(ent.exp) {  // Same time for all entries
        delete(c.m, k)
    }
}
\`\`\`

**Why Lock (not RLock):**
\`\`\`go
// Len() does TWO things:
// 1. Delete expired entries (requires Lock)
// 2. Count remaining entries (could use RLock)
// Since we delete, we need Lock for the entire operation
\`\`\`

**Production Use Cases:**

1. **Metrics and Monitoring:**
\`\`\`go
// Expose cache size to Prometheus
func (c *TTLCache) RecordMetrics() {
    cacheSize.Set(float64(c.Len()))  // Accurate live count
}
\`\`\`

2. **Health Checks:**
\`\`\`go
func (c *TTLCache) HealthCheck() error {
    size := c.Len()
    if size > maxCacheSize {
        return fmt.Errorf("cache too large: %d entries", size)
    }
    return nil
}
\`\`\`

3. **Capacity Planning:**
\`\`\`go
func (c *TTLCache) ShouldEvict() bool {
    return c.Len() > targetCapacity  // Check against threshold
}
\`\`\`

4. **Debug Logging:**
\`\`\`go
func (c *TTLCache) LogStats() {
    log.Printf("Cache contains %d live entries", c.Len())
}
\`\`\`

**Why Inline Cleanup (not call cleanupNow):**
\`\`\`go
// WRONG: Acquires lock twice
func (c *TTLCache) Len() int {
    c.cleanupNow(time.Now())  // Lock acquired
    c.mu.RLock()              // Lock acquired again!
    defer c.mu.RUnlock()
    return len(c.m)
}

// CORRECT: Single lock acquisition
func (c *TTLCache) Len() int {
    now := time.Now()
    c.mu.Lock()               // Lock once
    defer c.mu.Unlock()
    // Inline cleanup logic
    for k, ent := range c.m {
        if !ent.exp.IsZero() && !now.Before(ent.exp) {
            delete(c.m, k)
        }
    }
    return len(c.m)           // Count while holding same lock
}
\`\`\`

**Real-World Examples:**

1. **Redis INFO command:**
\`\`\`
> INFO keyspace
db0:keys=1000,expires=850  # Shows live key count
\`\`\`

2. **Memcached stats:**
\`\`\`
STAT curr_items 1543  # Current live items
STAT expired_unfetched 847  # Expired but not yet deleted
\`\`\`

3. **groupcache Stats():**
\`\`\`go
type Stats struct {
    Items int64  // Number of items in cache (after cleanup)
}
\`\`\`

**Performance Considerations:**
- **O(n) operation:** Must scan entire map
- **Infrequent calls:** Len() typically called for metrics, not hot path
- **Side effect:** Cleans cache as side effect, reduces memory

**Common Pattern - Cache Stats:**
\`\`\`go
type CacheStats struct {
    Size     int
    Hits     int64
    Misses   int64
}

func (c *TTLCache) Stats() CacheStats {
    return CacheStats{
        Size:   c.Len(),  // Accurate after cleanup
        Hits:   atomic.LoadInt64(&c.hits),
        Misses: atomic.LoadInt64(&c.misses),
    }
}
\`\`\`

**Testing Pattern:**
\`\`\`go
func TestTTLExpiration(t *testing.T) {
    cache := NewTTLCache(50 * time.Millisecond)
    cache.Set("key", "value")

    if cache.Len() != 1 {
        t.Fatal("expected 1 entry")
    }

    time.Sleep(100 * time.Millisecond)

    if cache.Len() != 0 {  // Len() cleans up expired entries
        t.Fatal("expected 0 entries after expiration")
    }
}
\`\`\`

Without eager cleanup in Len(), you'd report inflated sizes including expired entries that consume memory but provide no value.`,	order: 5,
	translations: {
		ru: {
			title: 'Подсчёт элементов в кэше',
			solutionCode: `package cache

import (
	"sync"
	"time"
)

type entry struct {
	v   any
	exp time.Time
}

type TTLCache struct {
	mu  sync.RWMutex
	m   map[string]entry
	ttl time.Duration
}

func (c *TTLCache) Len() int {
	if c == nil {	// Nil cache ведёт себя как пустой cache
		return 0
	}
	now := time.Now()	// Захватываем время один раз для избежания повторных вызовов
	c.mu.Lock()	// Lock для безопасной модификации и подсчёта
	defer c.mu.Unlock()	// Гарантируем освобождение lock даже при ошибке
	for k, ent := range c.m {	// Посещаем каждую запись для проверки истечения
		if !ent.exp.IsZero() && !now.Before(ent.exp) {	// Истёкшие записи удаляем активно
			delete(c.m, k)	// Удаляем устаревшую запись перед подсчётом
		}
	}
	return len(c.m)	// Возвращаем количество оставшихся живых записей
}`,
			description: `Реализуйте метод **Len** для возврата количества живых (не истёкших) записей.

**Требования:**
1. Реализуйте метод \`Len() int\` для TTLCache
2. Верните \`0\` если cache nil
3. Захватите текущее время один раз: \`now := time.Now()\`
4. Используйте эксклюзивную блокировку для атомарной очистки и подсчёта
5. Итерируйте по всем записям и удалите истёкшие: \`!ent.exp.IsZero() && !now.Before(ent.exp)\`
6. Верните \`len(c.m)\` после очистки (количество оставшихся живых записей)

**Пример:**
\`\`\`go
cache := NewTTLCache(100 * time.Millisecond)
cache.Set("key1", "value1")
cache.Set("key2", "value2")

count := cache.Len()  // 2 (обе живые)

time.Sleep(150 * time.Millisecond)
count = cache.Len()   // 0 (обе истекли и очищены)
\`\`\`

**Ограничения:**
- Очистите истёкшие записи перед подсчётом
- Захватите время один раз для консистентности
- Используйте Lock (не RLock) т.к. очистка изменяет map`,
			hint1: `Захватите time.Now() один раз в начале. Используйте для всех проверок истечения для консистентности.`,
			hint2: `Встройте логику очистки вместо вызова cleanupNow() для удержания одной блокировки.`,
			whyItMatters: `Len() предоставляет точные метрики кеша, активно очищая истёкшие записи и считая живые.

**Почему очищать перед подсчётом:**
\`\`\`go
// НЕПРАВИЛЬНО: Подсчёт включает истёкшие записи
func (c *TTLCache) Len() int {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return len(c.m)  // Включает истёкшие записи!
}

// ПРАВИЛЬНО: Очистить затем посчитать
func (c *TTLCache) Len() int {
    now := time.Now()
    c.mu.Lock()
    defer c.mu.Unlock()
    for k, ent := range c.m {
        if expired(ent, now) {
            delete(c.m, k)  // Удалить перед подсчётом
        }
    }
    return len(c.m)  // Точный подсчёт живых
}
\`\`\`

**Почему захватывать время один раз:**
\`\`\`go
// НЕПРАВИЛЬНО: Время меняется при итерации
for k, ent := range c.m {
    if time.Now().After(ent.exp) {  // Разное время каждую итерацию!
        delete(c.m, k)
    }
}

// ПРАВИЛЬНО: Консистентный снимок
now := time.Now()  // Один захват
for k, ent := range c.m {
    if now.After(ent.exp) {  // То же время для всех записей
        delete(c.m, k)
    }
}
\`\`\`

**Почему Lock (не RLock):**
\`\`\`go
// Len() делает ДВА действия:
// 1. Удаляет истёкшие записи (требует Lock)
// 2. Считает оставшиеся записи (могло бы использовать RLock)
// Так как удаляем, нужен Lock для всей операции
\`\`\`

**Случаи использования в продакшене:**

1. **Метрики и мониторинг:**
\`\`\`go
// Выставить размер кеша в Prometheus
func (c *TTLCache) RecordMetrics() {
    cacheSize.Set(float64(c.Len()))  // Точный подсчёт живых
}
\`\`\`

2. **Health Checks:**
\`\`\`go
func (c *TTLCache) HealthCheck() error {
    size := c.Len()
    if size > maxCacheSize {
        return fmt.Errorf("cache too large: %d entries", size)
    }
    return nil
}
\`\`\`

3. **Планирование ёмкости:**
\`\`\`go
func (c *TTLCache) ShouldEvict() bool {
    return c.Len() > targetCapacity  // Проверка против порога
}
\`\`\`

4. **Отладочное логирование:**
\`\`\`go
func (c *TTLCache) LogStats() {
    log.Printf("Cache contains %d live entries", c.Len())
}
\`\`\`

**Почему встроенная очистка (не вызов cleanupNow):**
\`\`\`go
// НЕПРАВИЛЬНО: Получает блокировку дважды
func (c *TTLCache) Len() int {
    c.cleanupNow(time.Now())  // Блокировка получена
    c.mu.RLock()              // Блокировка получена снова!
    defer c.mu.RUnlock()
    return len(c.m)
}

// ПРАВИЛЬНО: Одно получение блокировки
func (c *TTLCache) Len() int {
    now := time.Now()
    c.mu.Lock()               // Блокировка один раз
    defer c.mu.Unlock()
    // Встроенная логика очистки
    for k, ent := range c.m {
        if !ent.exp.IsZero() && !now.Before(ent.exp) {
            delete(c.m, k)
        }
    }
    return len(c.m)           // Подсчёт держа ту же блокировку
}
\`\`\`

**Примеры из реального мира:**

1. **Команда Redis INFO:**
\`\`\`
> INFO keyspace
db0:keys=1000,expires=850  # Показывает подсчёт живых ключей
\`\`\`

2. **Статистика Memcached:**
\`\`\`
STAT curr_items 1543  # Текущие живые элементы
STAT expired_unfetched 847  # Истёкшие но ещё не удалённые
\`\`\`

3. **Stats() groupcache:**
\`\`\`go
type Stats struct {
    Items int64  // Количество элементов в кеше (после очистки)
}
\`\`\`

**Соображения производительности:**
- **O(n) операция:** Должна просканировать весь map
- **Редкие вызовы:** Len() обычно вызывается для метрик, не в горячем пути
- **Побочный эффект:** Очищает кеш как побочный эффект, уменьшает память

**Общий паттерн - статистика кеша:**
\`\`\`go
type CacheStats struct {
    Size     int
    Hits     int64
    Misses   int64
}

func (c *TTLCache) Stats() CacheStats {
    return CacheStats{
        Size:   c.Len(),  // Точно после очистки
        Hits:   atomic.LoadInt64(&c.hits),
        Misses: atomic.LoadInt64(&c.misses),
    }
}
\`\`\`

**Паттерн тестирования:**
\`\`\`go
func TestTTLExpiration(t *testing.T) {
    cache := NewTTLCache(50 * time.Millisecond)
    cache.Set("key", "value")

    if cache.Len() != 1 {
        t.Fatal("expected 1 entry")
    }

    time.Sleep(100 * time.Millisecond)

    if cache.Len() != 0 {  // Len() очищает истёкшие записи
        t.Fatal("expected 0 entries after expiration")
    }
}
\`\`\`

**Практические преимущества:**

Без активной очистки в Len() сообщались бы завышенные размеры включая истёкшие записи, которые потребляют память но не дают ценности.`
		},
		uz: {
			title: `Keshdagi elementlar sonini hisoblash`,
			solutionCode: `package cache

import (
	"sync"
	"time"
)

type entry struct {
	v   any
	exp time.Time
}

type TTLCache struct {
	mu  sync.RWMutex
	m   map[string]entry
	ttl time.Duration
}

func (c *TTLCache) Len() int {
	if c == nil {	// Nil cache bo'sh cache kabi ishlaydi
		return 0
	}
	now := time.Now()	// Takroriy chaqiruvlardan qochish uchun vaqtni bir marta qo'lga olamiz
	c.mu.Lock()	// Xavfsiz o'zgartirish va hisoblash uchun qulf
	defer c.mu.Unlock()	// Hisoblash muvaffaqiyatsiz bo'lsa ham qulf ochilishini kafolatlash
	for k, ent := range c.m {	// Muddati o'tishni tekshirish uchun har bir yozuvga tashrif buyuramiz
		if !ent.exp.IsZero() && !now.Before(ent.exp) {	// Muddati o'tgan yozuvlar faol o'chiriladi
			delete(c.m, k)	// Hisoblashdan oldin eskirgan yozuvni o'chirish
		}
	}
	return len(c.m)	// Qolgan tirik yozuvlar sonini qaytarish
}`,
			description: `Tirik (muddati o'tmagan) yozuvlar sonini qaytarish uchun **Len** metodini amalga oshiring.

**Talablar:**
1. TTLCache uchun \`Len() int\` metodini amalga oshiring
2. cache nil bo'lsa \`0\` qaytaring
3. Joriy vaqtni bir marta qo'lga oling: \`now := time.Now()\`
4. Atomik tozalash va hisoblash uchun eksklyuziv qulf (Lock/Unlock) ishlating
5. Barcha yozuvlarni aylanib chiqing va muddati o'tganlarni o'chiring: \`!ent.exp.IsZero() && !now.Before(ent.exp)\`
6. Tozalashdan keyin \`len(c.m)\` qaytaring (qolgan tirik yozuvlar soni)

**Misol:**
\`\`\`go
cache := NewTTLCache(100 * time.Millisecond)
cache.Set("key1", "value1")
cache.Set("key2", "value2")

count := cache.Len()  // 2 (ikkisi ham tirik)

time.Sleep(150 * time.Millisecond)
count = cache.Len()   // 0 (ikkisi ham muddati o'tdi va tozalandi)
\`\`\`

**Cheklovlar:**
- Hisoblashdan oldin muddati o'tgan yozuvlarni tozalang (aniq hisobni ta'minlash)
- Izchil muddati o'tish tekshiruvini ta'minlash uchun vaqtni bir marta qo'lga oling
- Tozalash map ni o'zgartirishi uchun Lock ishlating (RLock emas)`,
			hint1: `Boshida time.Now() ni bir marta qo'lga oling. Izchillik uchun barcha muddati o'tish tekshiruvlarida ishlating.`,
			hint2: `Bitta qulfni davomida ushlab turish uchun cleanupNow() ni chaqirish o'rniga tozalash mantiqini ichki joylashtiring.`,
			whyItMatters: `Len() muddati o'tgan yozuvlarni faol tozalab va tiriklar sonini hisoblash orqali aniq kesh metrikalarini taqdim etadi.

**Nima uchun hisoblashdan oldin tozalash:**
\`\`\`go
// NOTO'G'RI: Hisob muddati o'tgan yozuvlarni o'z ichiga oladi
func (c *TTLCache) Len() int {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return len(c.m)  // Muddati o'tgan yozuvlarni o'z ichiga oladi!
}

// TO'G'RI: Tozalash keyin hisoblash
func (c *TTLCache) Len() int {
    now := time.Now()
    c.mu.Lock()
    defer c.mu.Unlock()
    for k, ent := range c.m {
        if expired(ent, now) {
            delete(c.m, k)  // Hisoblashdan oldin o'chirish
        }
    }
    return len(c.m)  // Aniq tirik hisob
}
\`\`\`

**Nima uchun vaqtni bir marta qo'lga olish:**
\`\`\`go
// NOTO'G'RI: Iteratsiya paytida vaqt o'zgaradi
for k, ent := range c.m {
    if time.Now().After(ent.exp) {  // Har bir iteratsiyada boshqa vaqt!
        delete(c.m, k)
    }
}

// TO'G'RI: Izchil surat
now := time.Now()  // Bir marta qo'lga olish
for k, ent := range c.m {
    if now.After(ent.exp) {  // Barcha yozuvlar uchun bir xil vaqt
        delete(c.m, k)
    }
}
\`\`\`

**Nima uchun Lock (RLock emas):**
\`\`\`go
// Len() IKKITA narsa qiladi:
// 1. Muddati o'tgan yozuvlarni o'chiradi (Lock talab qiladi)
// 2. Qolgan yozuvlarni hisoblaydi (RLock ishlatilishi mumkin edi)
// O'chirganimiz uchun butun operatsiya uchun Lock kerak
\`\`\`

**Ishlab chiqarishda foydalanish holatlari:**

1. **Metrikalar va monitoring:**
\`\`\`go
// Prometheus ga kesh hajmini chiqarish
func (c *TTLCache) RecordMetrics() {
    cacheSize.Set(float64(c.Len()))  // Aniq tirik hisob
}
\`\`\`

2. **Health Checks:**
\`\`\`go
func (c *TTLCache) HealthCheck() error {
    size := c.Len()
    if size > maxCacheSize {
        return fmt.Errorf("cache too large: %d entries", size)
    }
    return nil
}
\`\`\`

3. **Sig'im rejalashtirish:**
\`\`\`go
func (c *TTLCache) ShouldEvict() bool {
    return c.Len() > targetCapacity  // Chegara bilan tekshirish
}
\`\`\`

4. **Debug logging:**
\`\`\`go
func (c *TTLCache) LogStats() {
    log.Printf("Cache contains %d live entries", c.Len())
}
\`\`\`

**Nima uchun inline tozalash (cleanupNow ni chaqirmaslik):**
\`\`\`go
// NOTO'G'RI: Qulfni ikki marta oladi
func (c *TTLCache) Len() int {
    c.cleanupNow(time.Now())  // Qulf olindi
    c.mu.RLock()              // Qulf yana olindi!
    defer c.mu.RUnlock()
    return len(c.m)
}

// TO'G'RI: Bitta qulf olish
func (c *TTLCache) Len() int {
    now := time.Now()
    c.mu.Lock()               // Bir marta qulf
    defer c.mu.Unlock()
    // Inline tozalash mantiqi
    for k, ent := range c.m {
        if !ent.exp.IsZero() && !now.Before(ent.exp) {
            delete(c.m, k)
        }
    }
    return len(c.m)           // Bir xil qulfni ushlab turib hisoblash
}
\`\`\`

**Haqiqiy dunyodan misollar:**

1. **Redis INFO buyrug'i:**
\`\`\`
> INFO keyspace
db0:keys=1000,expires=850  # Tirik kalitlar sonini ko'rsatadi
\`\`\`

2. **Memcached statistikasi:**
\`\`\`
STAT curr_items 1543  # Hozirgi tirik elementlar
STAT expired_unfetched 847  # Muddati o'tgan lekin hali o'chirilmagan
\`\`\`

3. **groupcache Stats():**
\`\`\`go
type Stats struct {
    Items int64  // Keshdagi elementlar soni (tozalashdan keyin)
}
\`\`\`

**Ishlash ko'rsatkichlari:**
- **O(n) operatsiya:** Butun map ni skanerlash kerak
- **Kamdan-kam chaqiruvlar:** Len() odatda metrikalar uchun chaqiriladi, issiq yo'lda emas
- **Yon ta'sir:** Yon ta'sir sifatida keshni tozalaydi, xotirani kamaytiradi

**Umumiy pattern - Kesh statistikasi:**
\`\`\`go
type CacheStats struct {
    Size     int
    Hits     int64
    Misses   int64
}

func (c *TTLCache) Stats() CacheStats {
    return CacheStats{
        Size:   c.Len(),  // Tozalashdan keyin aniq
        Hits:   atomic.LoadInt64(&c.hits),
        Misses: atomic.LoadInt64(&c.misses),
    }
}
\`\`\`

**Testlash patterni:**
\`\`\`go
func TestTTLExpiration(t *testing.T) {
    cache := NewTTLCache(50 * time.Millisecond)
    cache.Set("key", "value")

    if cache.Len() != 1 {
        t.Fatal("expected 1 entry")
    }

    time.Sleep(100 * time.Millisecond)

    if cache.Len() != 0 {  // Len() muddati o'tgan yozuvlarni tozalaydi
        t.Fatal("expected 0 entries after expiration")
    }
}
\`\`\`

**Amaliy foydalari:**

Len() da faol tozalash bo'lmasa, xotirani egallagan lekin qiymat bermaydigan muddati o'tgan yozuvlarni o'z ichiga olgan shishirilgan hajmlar haqida xabar berilardi.`
		}
	}
};

export default task;
