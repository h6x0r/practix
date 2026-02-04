import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-ttlcache-cleanup-now',
	title: 'cleanupNow Method',
	difficulty: 'medium',	tags: ['go', 'cache', 'cleanup', 'iteration'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **cleanupNow** method to remove all expired entries at a given time.

**Requirements:**
1. Implement \`cleanupNow(now time.Time)\` method on TTLCache
2. Check if cache is nil and return early
3. Use exclusive lock (Lock/Unlock) for map iteration and deletion
4. Iterate over all entries and delete if: \`!ent.exp.IsZero() && !now.Before(ent.exp)\`
5. This is an internal method used by Len() and potential background cleanup

**Example:**
\`\`\`go
cache := NewTTLCache(100 * time.Millisecond)
cache.Set("key1", "value1")
cache.Set("key2", "value2")

time.Sleep(150 * time.Millisecond)
cache.cleanupNow(time.Now())  // Removes both expired entries

// Internal use:
cache.Len()  // Calls cleanupNow first, then counts remaining entries
\`\`\`

**Constraints:**
- Must use Lock (not RLock) because we're deleting from map
- Must check !ent.exp.IsZero() to skip never-expire entries
- Must use !now.Before(ent.exp) which is equivalent to now >= ent.exp`,
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

func (c *TTLCache) Get(key string) (interface{}, bool) {
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

// TODO: Implement cleanupNow method
// 1. Check if c is nil and return early
// 2. Use Lock for exclusive access during iteration and deletion
// 3. Iterate over all entries in c.m
// 4. Delete entries where expiration <= now (and exp is not zero)
func (c *TTLCache) cleanupNow(now time.Time) {
	// TODO: Implement
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

func (c *TTLCache) Get(key string) (interface{}, bool) {
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
	if c == nil {	// Nothing to clean when cache is nil
		return
	}
	c.mu.Lock()	// Ensure exclusive access during cleanup
	defer c.mu.Unlock()	// Release lock after iteration
	for k, ent := range c.m {	// Iterate over all entries to check expiry
		if !ent.exp.IsZero() && !now.Before(ent.exp) {	// Expired entries meet the removal criteria
			delete(c.m, k)	// Drop stale entry from map
		}
	}
}`,
	testCode: `package cache

import (
	"testing"
	"time"
)

func TestCleanupNow_BasicCleanup(t *testing.T) {
	cache := NewTTLCache(50 * time.Millisecond)
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	time.Sleep(100 * time.Millisecond)
	now := time.Now()
	cache.cleanupNow(now)
	if len(cache.m) != 0 {
		t.Errorf("expected empty cache after cleanup, got %d entries", len(cache.m))
	}
}

func TestCleanupNow_NilCache(t *testing.T) {
	var cache *TTLCache
	now := time.Now()
	cache.cleanupNow(now)
	// Should not panic
}

func TestCleanupNow_PartialCleanup(t *testing.T) {
	cache := NewTTLCache(100 * time.Millisecond)
	cache.Set("key1", "value1")
	time.Sleep(50 * time.Millisecond)
	cache.Set("key2", "value2")
	time.Sleep(60 * time.Millisecond)
	now := time.Now()
	cache.cleanupNow(now)
	if len(cache.m) != 1 {
		t.Errorf("expected 1 entry after cleanup, got %d", len(cache.m))
	}
	if _, ok := cache.m["key2"]; !ok {
		t.Error("expected key2 to remain after cleanup")
	}
}

func TestCleanupNow_ZeroTTLNotCleaned(t *testing.T) {
	cache := NewTTLCache(0)
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	now := time.Now().Add(time.Hour)
	cache.cleanupNow(now)
	if len(cache.m) != 2 {
		t.Errorf("expected 2 entries (zero ttl never expires), got %d", len(cache.m))
	}
}

func TestCleanupNow_EmptyCache(t *testing.T) {
	cache := NewTTLCache(time.Second)
	now := time.Now()
	cache.cleanupNow(now)
	if len(cache.m) != 0 {
		t.Errorf("expected empty cache, got %d entries", len(cache.m))
	}
}

func TestCleanupNow_AllFresh(t *testing.T) {
	cache := NewTTLCache(time.Hour)
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	now := time.Now()
	cache.cleanupNow(now)
	if len(cache.m) != 2 {
		t.Errorf("expected 2 entries (all fresh), got %d", len(cache.m))
	}
}

func TestCleanupNow_ExactExpiration(t *testing.T) {
	cache := NewTTLCache(100 * time.Millisecond)
	cache.Set("key1", "value1")
	expTime := cache.m["key1"].exp
	cache.cleanupNow(expTime)
	if len(cache.m) != 0 {
		t.Errorf("expected empty cache at exact expiration time, got %d entries", len(cache.m))
	}
}

func TestCleanupNow_BeforeExpiration(t *testing.T) {
	cache := NewTTLCache(time.Hour)
	cache.Set("key1", "value1")
	now := time.Now()
	cache.cleanupNow(now)
	if len(cache.m) != 1 {
		t.Errorf("expected 1 entry before expiration, got %d", len(cache.m))
	}
}

func TestCleanupNow_MultipleExpired(t *testing.T) {
	cache := NewTTLCache(50 * time.Millisecond)
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	cache.Set("key3", "value3")
	time.Sleep(100 * time.Millisecond)
	now := time.Now()
	cache.cleanupNow(now)
	if len(cache.m) != 0 {
		t.Errorf("expected all expired entries removed, got %d", len(cache.m))
	}
}

func TestCleanupNow_FutureTime(t *testing.T) {
	cache := NewTTLCache(100 * time.Millisecond)
	cache.Set("key1", "value1")
	futureTime := time.Now().Add(time.Hour)
	cache.cleanupNow(futureTime)
	if len(cache.m) != 0 {
		t.Errorf("expected empty cache with future time, got %d entries", len(cache.m))
	}
}
`,
			hint1: `Safe to delete from map during range iteration in Go. Use !now.Before(exp) which means now >= exp.`,
			hint2: `Check !ent.exp.IsZero() first to skip entries with no expiration (zero time).`,
			whyItMatters: `Bulk cleanup prevents memory leaks and provides a consistent view of cache size.

**Why Bulk Cleanup:**
- **Memory management:** Expired entries consume memory until removed
- **Accurate metrics:** Len() needs to report live entries, not stale ones
- **Performance:** Batch deletion more efficient than one-by-one in Get()
- **Consistency:** Atomic cleanup provides snapshot of live entries

**Why Lock (not RLock):**
\`\`\`go
// Deleting from map requires exclusive access
for k, ent := range c.m {
    delete(c.m, k)  // MUST hold Lock, not RLock
}
\`\`\`

**Why Safe to Delete During Iteration:**
\`\`\`go
// Go allows deletion during range iteration
for k, v := range m {
    if condition {
        delete(m, k)  // Safe: iteration continues correctly
    }
}
// Note: Adding keys during iteration may or may not visit them
\`\`\`

**Why !now.Before(exp) Instead of now.After(exp):**
\`\`\`go
// These are equivalent:
now.After(exp)      // true if now > exp
!now.Before(exp)    // true if now >= exp

// Solution uses >= (includes exact match):
!now.Before(exp)    // Treats exact match as expired
// More conservative: expires at exact deadline
\`\`\`

**Production Use Cases:**

1. **Periodic Background Cleanup:**
\`\`\`go
func (c *TTLCache) StartCleanup(interval time.Duration) {
    ticker := time.NewTicker(interval)
    go func() {
        for now := range ticker.C {
            c.cleanupNow(now)  // Remove all expired entries
        }
    }()
}

// Usage:
cache := NewTTLCache(5 * time.Minute)
cache.StartCleanup(1 * time.Minute)  // Clean every minute
\`\`\`

2. **Before Size Checks:**
\`\`\`go
func (c *TTLCache) Len() int {
    c.cleanupNow(time.Now())  // Get accurate count
    return len(c.m)
}
\`\`\`

3. **Memory Pressure Response:**
\`\`\`go
func (c *TTLCache) HandleMemoryPressure() {
    c.cleanupNow(time.Now())  // Free expired entries immediately
}
\`\`\`

**Real-World Examples:**
- **Redis:** SCAN + expiration checks for background cleanup
- **Memcached:** Lazy expiration + periodic cleanup
- **groupcache:** LRU eviction combined with TTL
- **go-cache:** Uses similar cleanupNow for periodic maintenance

**Performance Considerations:**
\`\`\`go
// Worst case: O(n) where n = map size
// Called infrequently (background) or when accurate size needed
// More efficient than checking expiration on every Get
\`\`\`

**Common Pattern:**
\`\`\`go
// Combine lazy (in Get) + eager (periodic) cleanup:
// - Get: Remove single expired entry on access (fast path)
// - cleanupNow: Batch remove all expired (background)
// Result: Memory stays bounded without sacrificing Get performance
\``,	order: 3,
	translations: {
		ru: {
			title: 'Очистка просроченных записей',
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

func (c *TTLCache) cleanupNow(now time.Time) {
	if c == nil {	// Нечего очищать когда cache nil
		return
	}
	c.mu.Lock()	// Обеспечиваем эксклюзивный доступ при очистке
	defer c.mu.Unlock()	// Освобождаем lock после итерации
	for k, ent := range c.m {	// Итерируем по всем записям для проверки истечения
		if !ent.exp.IsZero() && !now.Before(ent.exp) {	// Истёкшие записи соответствуют критерию удаления
			delete(c.m, k)	// Удаляем устаревшую запись из map
		}
	}
}`,
			description: `Реализуйте метод **cleanupNow** для удаления всех истёкших записей в заданное время.

**Требования:**
1. Реализуйте метод \`cleanupNow(now time.Time)\` для TTLCache
2. Проверьте nil cache и верните рано
3. Используйте эксклюзивную блокировку для итерации и удаления
4. Удалите записи где: \`!ent.exp.IsZero() && !now.Before(ent.exp)\`
5. Внутренний метод, используется в Len() и фоновой очистке

**Пример:**
\`\`\`go
cache := NewTTLCache(100 * time.Millisecond)
cache.Set("key1", "value1")
cache.Set("key2", "value2")

time.Sleep(150 * time.Millisecond)
cache.cleanupNow(time.Now())  // Удаляет обе истёкшие записи

// Внутреннее использование:
cache.Len()  // Сначала вызывает cleanupNow, затем считает
\`\`\`

**Ограничения:**
- Используйте Lock (не RLock) для удаления из map
- Проверяйте !ent.exp.IsZero() для пропуска never-expire
- Используйте !now.Before(ent.exp) что равно now >= ent.exp`,
			hint1: `В Go безопасно удалять из map во время range итерации. Используйте !now.Before(exp) что означает now >= exp.`,
			hint2: `Проверьте !ent.exp.IsZero() сначала для пропуска записей без истечения (нулевое время).`,
			whyItMatters: `Массовая очистка предотвращает утечки памяти и обеспечивает консистентное представление размера кеша.

**Почему массовая очистка:**
- **Управление памятью:** Истёкшие записи потребляют память до удаления
- **Точные метрики:** Len() нужно сообщать живые записи, а не устаревшие
- **Производительность:** Батч удаление эффективнее чем по одной в Get()
- **Консистентность:** Атомарная очистка даёт снимок живых записей

**Почему Lock (не RLock):**
\`\`\`go
// Удаление из map требует эксклюзивного доступа
for k, ent := range c.m {
    delete(c.m, k)  // ДОЛЖЕН держать Lock, не RLock
}
\`\`\`

**Почему безопасно удалять во время итерации:**
\`\`\`go
// Go позволяет удаление во время range итерации
for k, v := range m {
    if condition {
        delete(m, k)  // Безопасно: итерация продолжается корректно
    }
}
// Примечание: Добавление ключей при итерации может их посетить или нет
\`\`\`

**Почему !now.Before(exp) вместо now.After(exp):**
\`\`\`go
// Эти эквивалентны:
now.After(exp)      // true если now > exp
!now.Before(exp)    // true если now >= exp

// Решение использует >= (включает точное совпадение):
!now.Before(exp)    // Трактует точное совпадение как истёкшее
// Более консервативно: истекает в точный дедлайн
\`\`\`

**Случаи использования в продакшене:**

1. **Периодическая фоновая очистка:**
\`\`\`go
func (c *TTLCache) StartCleanup(interval time.Duration) {
    ticker := time.NewTicker(interval)
    go func() {
        for now := range ticker.C {
            c.cleanupNow(now)  // Удалить все истёкшие записи
        }
    }()
}

// Использование:
cache := NewTTLCache(5 * time.Minute)
cache.StartCleanup(1 * time.Minute)  // Очищать каждую минуту
\`\`\`

2. **Перед проверками размера:**
\`\`\`go
func (c *TTLCache) Len() int {
    c.cleanupNow(time.Now())  // Получить точный счёт
    return len(c.m)
}
\`\`\`

3. **Ответ на давление памяти:**
\`\`\`go
func (c *TTLCache) HandleMemoryPressure() {
    c.cleanupNow(time.Now())  // Освободить истёкшие записи немедленно
}
\`\`\`

**Примеры из реального мира:**
- **Redis:** SCAN + проверки истечения для фоновой очистки
- **Memcached:** Ленивое истечение + периодическая очистка
- **groupcache:** LRU вытеснение комбинированное с TTL
- **go-cache:** Использует похожий cleanupNow для периодического обслуживания

**Соображения производительности:**
\`\`\`go
// Худший случай: O(n) где n = размер map
// Вызывается редко (фон) или когда нужен точный размер
// Эффективнее чем проверять истечение при каждом Get
\`\`\`

**Общий паттерн:**
\`\`\`go
// Комбинация lazy (в Get) + eager (периодическая) очистки:
// - Get: Удалить одну истёкшую запись при обращении (быстрый путь)
// - cleanupNow: Батч удаление всех истёкших (фон)
// Результат: Память остаётся ограниченной без ущерба производительности Get
\`\`\`

**Практические преимущества:**

Этот подход обеспечивает баланс между производительностью и эффективным использованием памяти в продакшене.`
		},
		uz: {
			title: `Muddati o'tgan yozuvlarni tozalash`,
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

func (c *TTLCache) cleanupNow(now time.Time) {
	if c == nil {	// cache nil bo'lganda tozalash uchun hech narsa yo'q
		return
	}
	c.mu.Lock()	// Tozalash paytida eksklyuziv kirishni ta'minlaymiz
	defer c.mu.Unlock()	// Iteratsiyadan keyin qulfni bo'shatamiz
	for k, ent := range c.m {	// Muddati o'tishni tekshirish uchun barcha yozuvlarni aylanamiz
		if !ent.exp.IsZero() && !now.Before(ent.exp) {	// Muddati o'tgan yozuvlar o'chirish kriteriyasiga mos
			delete(c.m, k)	// Eskirgan yozuvni map dan o'chiramiz
		}
	}
}`,
			description: `Berilgan vaqtda barcha muddati o'tgan yozuvlarni o'chirish uchun **cleanupNow** metodini amalga oshiring.

**Talablar:**
1. TTLCache uchun \`cleanupNow(now time.Time)\` metodini amalga oshiring
2. nil cache tekshiring va erta qaytaring
3. Iteratsiya va o'chirish uchun eksklyuziv qulf (Lock/Unlock) ishlating
4. Barcha yozuvlarni aylanib chiqing va muddati o'tganlarni o'chiring: \`!ent.exp.IsZero() && !now.Before(ent.exp)\`
5. Bu ichki metod, Len() va potentsial fon tozalashda ishlatiladi

**Misol:**
\`\`\`go
cache := NewTTLCache(100 * time.Millisecond)
cache.Set("key1", "value1")
cache.Set("key2", "value2")

time.Sleep(150 * time.Millisecond)
cache.cleanupNow(time.Now())  // Ikkala muddati o'tgan yozuvni o'chiradi

// Ichki foydalanish:
cache.Len()  // Avval cleanupNow ni chaqiradi, keyin qolganlarni hisoblaydi
\`\`\`

**Cheklovlar:**
- map dan o'chirish uchun Lock ishlating (RLock emas)
- never-expire yozuvlarni o'tkazib yuborish uchun !ent.exp.IsZero() tekshiring
- now >= ent.exp ga teng bo'lgan !now.Before(ent.exp) ishlating`,
			hint1: `Go da range iteratsiyasi paytida map dan o'chirish xavfsiz. now >= exp ni bildiruvchi !now.Before(exp) ishlating.`,
			hint2: `Avval muddati o'tish yo'q yozuvlarni (nol vaqt) o'tkazib yuborish uchun !ent.exp.IsZero() tekshiring.`,
			whyItMatters: `Ommaviy tozalash xotira oqishlarini oldini oladi va kesh hajmining izchil ko'rinishini ta'minlaydi.

**Nima uchun ommaviy tozalash:**
- **Xotira boshqarish:** Muddati o'tgan yozuvlar o'chirilmaguncha xotira egallaydi
- **Aniq metrikalar:** Len() eskirgan emas, tirik yozuvlarni xabar qilishi kerak
- **Ishlash:** Partiyaviy o'chirish Get() da birma-bir o'chirishdan samaraliroq
- **Izchillik:** Atomik tozalash tirik yozuvlarning suratini beradi

**Nima uchun Lock (RLock emas):**
\`\`\`go
// map dan o'chirish eksklyuziv kirishni talab qiladi
for k, ent := range c.m {
    delete(c.m, k)  // Lock ni ushlab turish KERAK, RLock emas
}
\`\`\`

**Nima uchun iteratsiya paytida o'chirish xavfsiz:**
\`\`\`go
// Go range iteratsiyasi paytida o'chirishga ruxsat beradi
for k, v := range m {
    if condition {
        delete(m, k)  // Xavfsiz: iteratsiya to'g'ri davom etadi
    }
}
// Eslatma: Iteratsiya paytida kalitlarni qo'shish ularga tashrif buyurishi yoki buyurmasligi mumkin
\`\`\`

**Nima uchun !now.Before(exp) o'rniga now.After(exp):**
\`\`\`go
// Bular ekvivalent:
now.After(exp)      // true agar now > exp
!now.Before(exp)    // true agar now >= exp

// Yechim >= dan foydalanadi (aniq moslikni o'z ichiga oladi):
!now.Before(exp)    // Aniq moslikni muddati o'tgan deb hisoblaydi
// Konservativroq: aniq muddatda muddati o'tadi
\`\`\`

**Ishlab chiqarishda foydalanish holatlari:**

1. **Davriy fon tozalash:**
\`\`\`go
func (c *TTLCache) StartCleanup(interval time.Duration) {
    ticker := time.NewTicker(interval)
    go func() {
        for now := range ticker.C {
            c.cleanupNow(now)  // Barcha muddati o'tgan yozuvlarni o'chirish
        }
    }()
}

// Foydalanish:
cache := NewTTLCache(5 * time.Minute)
cache.StartCleanup(1 * time.Minute)  // Har daqiqa tozalash
\`\`\`

2. **Hajm tekshiruvlaridan oldin:**
\`\`\`go
func (c *TTLCache) Len() int {
    c.cleanupNow(time.Now())  // Aniq hisobni olish
    return len(c.m)
}
\`\`\`

3. **Xotira bosimiga javob:**
\`\`\`go
func (c *TTLCache) HandleMemoryPressure() {
    c.cleanupNow(time.Now())  // Muddati o'tgan yozuvlarni darhol bo'shatish
}
\`\`\`

**Haqiqiy dunyodan misollar:**
- **Redis:** Fon tozalash uchun SCAN + muddati o'tish tekshiruvlari
- **Memcached:** Dangasa muddati o'tish + davriy tozalash
- **groupcache:** TTL bilan birlashtirilgan LRU chiqarish
- **go-cache:** Davriy xizmat ko'rsatish uchun o'xshash cleanupNow dan foydalanadi

**Ishlash ko'rsatkichlari:**
\`\`\`go
// Eng yomon holat: O(n) bu yerda n = map hajmi
// Kamdan-kam chaqiriladi (fon) yoki aniq hajm kerak bo'lganda
// Har bir Get da muddati o'tishni tekshirishdan samaraliroq
\`\`\`

**Umumiy pattern:**
\`\`\`go
// Lazy (Get da) + eager (davriy) tozalashni birlashtirish:
// - Get: Kirishda bitta muddati o'tgan yozuvni o'chirish (tez yo'l)
// - cleanupNow: Barcha muddati o'tganlarni partiyaviy o'chirish (fon)
// Natija: Get ishlashiga zarar bermasdan xotira chegaralangan bo'lib qoladi
\`\`\`

**Amaliy foydalari:**

Bu yondashuv ishlab chiqarishda ishlash va xotirani samarali ishlatish o'rtasida muvozanatni ta'minlaydi.`
		}
	}
};

export default task;
