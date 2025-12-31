import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-ttlcache-set',
	title: 'Set Method',
	difficulty: 'easy',	tags: ['go', 'cache', 'concurrency', 'mutex'],
	estimatedTime: '15m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement thread-safe **Set** method to store values with automatic expiration.

**Requirements:**
1. Implement \`Set(key string, v any)\` method on TTLCache
2. Check if cache is nil and return early (no-op for nil cache)
3. Use \`c.mu.Lock()\` for exclusive write access
4. Calculate expiration: \`expire := time.Time{}\` (zero) if ttl <= 0, else \`time.Now().Add(c.ttl)\`
5. Store entry in map: \`c.m[key] = entry{v: v, exp: expire}\`

**Example:**
\`\`\`go
cache := NewTTLCache(5 * time.Second)
cache.Set("user:123", &User{Name: "Alice"})
// Stored with expiration = now + 5s

cache.Set("user:456", &User{Name: "Bob"})
// Overwrites if key exists
\`\`\`

**Constraints:**
- Must handle nil cache gracefully (no panic)
- Must use Lock/Unlock (not RLock) for map writes
- Must use defer to ensure unlock even if panic occurs`,
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

// TODO: Implement Set method
// 1. Check if c is nil and return early
// 2. Lock mutex for exclusive access
// 3. Calculate expiration time
// 4. Store entry in map
func (c *TTLCache) Set(key string, v any) {
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
	if c == nil {	// Nil cache acts as a no-op for callers
		return
	}
	c.mu.Lock()	// Exclusive lock protects the underlying map
	defer c.mu.Unlock()	// Ensure lock release even on panic
	expire := time.Time{}	// Zero expiration means no TTL enforcement
	if c.ttl > 0 {	// Only compute expiration when ttl is positive
		expire = time.Now().Add(c.ttl)	// Schedule the absolute expiration moment
	}
	c.m[key] = entry{v: v, exp: expire}	// Store the payload together with expiry timestamp
}`,
	testCode: `package cache

import (
	"testing"
	"time"
)

func TestSet_BasicSet(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	if len(cache.m) != 1 {
		t.Errorf("expected map length 1, got %d", len(cache.m))
	}
	if cache.m["key1"].v != "value1" {
		t.Errorf("expected value1, got %v", cache.m["key1"].v)
	}
}

func TestSet_Overwrite(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	cache.Set("key1", "value2")
	if cache.m["key1"].v != "value2" {
		t.Errorf("expected value2, got %v", cache.m["key1"].v)
	}
}

func TestSet_NilCache(t *testing.T) {
	var cache *TTLCache
	cache.Set("key1", "value1")
	// Should not panic
}

func TestSet_MultipleKeys(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	cache.Set("key3", "value3")
	if len(cache.m) != 3 {
		t.Errorf("expected map length 3, got %d", len(cache.m))
	}
}

func TestSet_ExpirationSet(t *testing.T) {
	cache := NewTTLCache(5 * time.Second)
	before := time.Now()
	cache.Set("key1", "value1")
	after := time.Now().Add(6 * time.Second)
	exp := cache.m["key1"].exp
	if exp.Before(before.Add(4*time.Second)) || exp.After(after) {
		t.Errorf("expiration time not in expected range")
	}
}

func TestSet_ZeroTTL(t *testing.T) {
	cache := NewTTLCache(0)
	cache.Set("key1", "value1")
	exp := cache.m["key1"].exp
	if !exp.IsZero() {
		t.Errorf("expected zero expiration for zero ttl, got %v", exp)
	}
}

func TestSet_NegativeTTL(t *testing.T) {
	cache := NewTTLCache(-1 * time.Second)
	cache.Set("key1", "value1")
	exp := cache.m["key1"].exp
	if !exp.IsZero() {
		t.Errorf("expected zero expiration for negative ttl, got %v", exp)
	}
}

func TestSet_DifferentTypes(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("int", 42)
	cache.Set("string", "hello")
	cache.Set("slice", []int{1, 2, 3})
	if cache.m["int"].v != 42 {
		t.Errorf("expected 42, got %v", cache.m["int"].v)
	}
	if cache.m["string"].v != "hello" {
		t.Errorf("expected hello, got %v", cache.m["string"].v)
	}
}

func TestSet_EmptyKey(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("", "empty key value")
	if cache.m[""].v != "empty key value" {
		t.Errorf("expected empty key value, got %v", cache.m[""].v)
	}
}

func TestSet_NilValue(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("nilkey", nil)
	if cache.m["nilkey"].v != nil {
		t.Errorf("expected nil value, got %v", cache.m["nilkey"].v)
	}
}
`,
			hint1: `Use defer c.mu.Unlock() immediately after Lock() to guarantee unlock even if panic occurs.`,
			hint2: `Zero time.Time{} means no expiration. Only calculate time.Now().Add(c.ttl) if ttl > 0.`,
			whyItMatters: `The Set method must be thread-safe because multiple goroutines will write to the cache concurrently in production.

**Why Thread Safety Matters:**
- **Data races:** Without locks, concurrent map writes cause runtime panics
- **Correctness:** Ensure entries are written atomically with their expiration
- **Production reality:** Caches receive writes from many concurrent HTTP handlers

**Why Defer Unlock:**
\`\`\`go
// BAD: If panic occurs before unlock, mutex stays locked forever
c.mu.Lock()
c.m[key] = entry{v: v, exp: expire}
c.mu.Unlock()  // Never called if panic happens

// GOOD: Defer ensures unlock even during panic
c.mu.Lock()
defer c.mu.Unlock()  // Always called during stack unwinding
c.m[key] = entry{v: v, exp: expire}
\`\`\`

**Why Lock (not RLock):**
- **Map writes:** Go maps require exclusive access for writes
- **Data integrity:** Prevent concurrent writes from corrupting internal state
- **RWMutex:** Use Lock for writes, RLock only for reads

**Production Pattern:**
\`\`\`go
// HTTP handler - multiple concurrent requests
func HandleLogin(w http.ResponseWriter, r *http.Request) {
    session := generateSession()
    sessionCache.Set(session.ID, session)  // Thread-safe write
    // Multiple handlers call this simultaneously - no races!
}
\`\`\`

**Real-World Examples:**
- **sync.Map:** Standard library's concurrent map (more complex)
- **groupcache:** Google's distributed cache (similar Set pattern)
- **go-cache:** Popular library using same Lock/defer pattern

**Why Nil Check:**
\`\`\`go
var cache *TTLCache  // nil cache
cache.Set("key", "value")  // Would panic without nil check
// With nil check: no-op, caller doesn't need to check
\`\`\`

**Real-World Benefits:**

This pattern makes caches optional dependencies - if initialization fails, operations become no-ops instead of crashing.`,	order: 1,
	translations: {
		ru: {
			title: 'Добавление элемента в кэш',
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
	if c == nil {	// Nil cache работает как no-op для вызывающих
		return
	}
	c.mu.Lock()	// Эксклюзивная блокировка защищает underlying map
	defer c.mu.Unlock()	// Гарантируем освобождение lock даже при панике
	expire := time.Time{}	// Нулевое истечение означает нет TTL enforcement
	if c.ttl > 0 {	// Вычисляем истечение только если ttl положительный
		expire = time.Now().Add(c.ttl)	// Планируем абсолютный момент истечения
	}
	c.m[key] = entry{v: v, exp: expire}	// Сохраняем payload вместе с timestamp истечения
}`,
			description: `Реализуйте потокобезопасный метод **Set** для хранения значений с автоматическим истечением.

**Требования:**
1. Реализуйте метод \`Set(key string, v any)\` для TTLCache
2. Проверьте nil cache и верните рано (no-op для nil)
3. Используйте \`c.mu.Lock()\` для эксклюзивного доступа
4. Вычислите expiration: \`time.Time{}\` если ttl <= 0, иначе \`time.Now().Add(c.ttl)\`
5. Сохраните entry в map: \`c.m[key] = entry{v: v, exp: expire}\`

**Пример:**
\`\`\`go
cache := NewTTLCache(5 * time.Second)
cache.Set("user:123", &User{Name: "Alice"})
// Сохранено с истечением = now + 5s

cache.Set("user:456", &User{Name: "Bob"})
// Перезаписывает если ключ существует
\`\`\`

**Ограничения:**
- Обрабатывайте nil cache без паники
- Используйте Lock (не RLock) для записи в map
- Используйте defer для гарантии unlock`,
			hint1: `Используйте defer c.mu.Unlock() сразу после Lock() для гарантии unlock даже при панике.`,
			hint2: `Нулевой time.Time{} означает отсутствие истечения. Вычисляйте time.Now().Add(c.ttl) только если ttl > 0.`,
			whyItMatters: `Метод Set должен быть thread-safe, потому что множество goroutines будут писать в кеш одновременно в продакшене.

**Почему важна потокобезопасность:**
- **Data races:** Без блокировок concurrent записи в map вызывают runtime панику
- **Корректность:** Гарантия, что записи сохраняются атомарно с истечением
- **Production реальность:** Кеши получают записи от многих concurrent HTTP handlers

**Почему Defer Unlock:**
\`\`\`go
// ПЛОХО: Если паника до unlock, мьютекс заблокирован навсегда
c.mu.Lock()
c.m[key] = entry{v: v, exp: expire}
c.mu.Unlock()  // Никогда не вызовется если паника

// ХОРОШО: Defer гарантирует unlock даже при панике
c.mu.Lock()
defer c.mu.Unlock()  // Всегда вызывается при раскрутке стека
c.m[key] = entry{v: v, exp: expire}
\`\`\`

**Почему Lock (не RLock):**
- **Записи в map:** Go maps требуют эксклюзивного доступа для записи
- **Целостность данных:** Предотвращает повреждение внутреннего состояния от concurrent записей
- **RWMutex:** Используйте Lock для записи, RLock только для чтения

**Продакшен паттерн:**
\`\`\`go
// HTTP handler - множество concurrent запросов
func HandleLogin(w http.ResponseWriter, r *http.Request) {
    session := generateSession()
    sessionCache.Set(session.ID, session)  // Thread-safe запись
    // Множество handlers вызывают это одновременно - нет гонок!
}
\`\`\`

**Примеры из реального мира:**
- **sync.Map:** Concurrent map из стандартной библиотеки (более сложная)
- **groupcache:** Распределенный кеш Google (похожий паттерн Set)
- **go-cache:** Популярная библиотека с тем же паттерном Lock/defer

**Почему проверка Nil:**
\`\`\`go
var cache *TTLCache  // nil кеш
cache.Set("key", "value")  // Паника без проверки nil
// С проверкой nil: no-op, вызывающему не нужно проверять
\`\`\`

**Практические преимущества:**

Этот паттерн делает кеши опциональными зависимостями - если инициализация не удалась, операции становятся no-ops вместо краха.`
		},
		uz: {
			title: `Keshga element qo'shish`,
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
	if c == nil {	// Nil cache chaqiruvchilar uchun no-op sifatida ishlaydi
		return
	}
	c.mu.Lock()	// Eksklyuziv qulf asosiy map ni himoya qiladi
	defer c.mu.Unlock()	// Panik bo'lsa ham qulf bo'shatilishini kafolatlash
	expire := time.Time{}	// Nol muddati o'tish TTL majburlash yo'q degani
	if c.ttl > 0 {	// Faqat ttl ijobiy bo'lganda muddati o'tishni hisoblash
		expire = time.Now().Add(c.ttl)	// Mutlaq muddati o'tish paytini rejalashtirish
	}
	c.m[key] = entry{v: v, exp: expire}	// Yuklarni muddati o'tish timestamp bilan birga saqlash
}`,
			description: `Avtomatik muddati o'tish bilan qiymatlarni saqlash uchun thread-safe **Set** metodini amalga oshiring.

**Talablar:**
1. TTLCache uchun \`Set(key string, v any)\` metodini amalga oshiring
2. cache nil bo'lsa tekshiring va erta qaytaring (nil uchun no-op)
3. Eksklyuziv yozish kirishi uchun \`c.mu.Lock()\` dan foydalaning
4. Muddati o'tishni hisoblang: ttl <= 0 bo'lsa \`time.Time{}\` (nol), aks holda \`time.Now().Add(c.ttl)\`
5. Yozuvni map da saqlang: \`c.m[key] = entry{v: v, exp: expire}\`

**Misol:**
\`\`\`go
cache := NewTTLCache(5 * time.Second)
cache.Set("user:123", &User{Name: "Alice"})
// now + 5s muddati o'tish bilan saqlandi

cache.Set("user:456", &User{Name: "Bob"})
// Agar kalit mavjud bo'lsa qayta yozadi
\`\`\`

**Cheklovlar:**
- nil cache ni panik qilmasdan qayta ishlashi kerak
- map yozishlari uchun Lock/Unlock ishlating (RLock emas)
- Panik bo'lsa ham unlock bo'lishini ta'minlash uchun defer ishlating`,
			hint1: `Panik bo'lsa ham unlock bo'lishini kafolatlash uchun Lock() dan keyin darhol defer c.mu.Unlock() dan foydalaning.`,
			hint2: `Nol time.Time{} muddati o'tmaslikni bildiradi. Faqat ttl > 0 bo'lsa time.Now().Add(c.ttl) hisoblang.`,
			whyItMatters: `Set metodi thread-safe bo'lishi kerak, chunki production da bir nechta goroutine bir vaqtda keshga yozadi.

**Nima uchun thread xavfsizligi muhim:**
- **Data races:** Quflarsiz bir vaqtda map yozishlari runtime panikga sabab bo'ladi
- **To'g'rilik:** Yozuvlar muddati o'tish bilan atomik ravishda yozilishini ta'minlash
- **Production haqiqati:** Keshlar ko'plab bir vaqtda ishlaydigan HTTP handler lardan yozuvlar oladi

**Nima uchun Defer Unlock:**
\`\`\`go
// YOMON: Agar unlock dan oldin panik bo'lsa, mutex abadiy qulflangan bo'lib qoladi
c.mu.Lock()
c.m[key] = entry{v: v, exp: expire}
c.mu.Unlock()  // Panik bo'lsa hech qachon chaqirilmaydi

// YAXSHI: Defer panik paytida ham unlock bo'lishini kafolatlaydi
c.mu.Lock()
defer c.mu.Unlock()  // Stek ochilishi paytida har doim chaqiriladi
c.m[key] = entry{v: v, exp: expire}
\`\`\`

**Nima uchun Lock (RLock emas):**
- **Map yozishlari:** Go map lari yozish uchun eksklyuziv kirishni talab qiladi
- **Ma'lumot yaxlitligi:** Bir vaqtda yozishlardan ichki holatning buzilishini oldini oladi
- **RWMutex:** Yozish uchun Lock, faqat o'qish uchun RLock ishlating

**Ishlab chiqarish patterni:**
\`\`\`go
// HTTP handler - bir nechta bir vaqtdagi so'rovlar
func HandleLogin(w http.ResponseWriter, r *http.Request) {
    session := generateSession()
    sessionCache.Set(session.ID, session)  // Thread-safe yozish
    // Bir nechta handler lar buni bir vaqtda chaqiradi - poyga yo'q!
}
\`\`\`

**Haqiqiy dunyodan misollar:**
- **sync.Map:** Standart kutubxonaning bir vaqtdagi map i (murakkabroq)
- **groupcache:** Google ning taqsimlangan keshi (o'xshash Set patterni)
- **go-cache:** Bir xil Lock/defer patternidan foydalanadigan mashhur kutubxona

**Nima uchun Nil tekshiruvi:**
\`\`\`go
var cache *TTLCache  // nil kesh
cache.Set("key", "value")  // Nil tekshiruvisiz panik bo'ladi
// Nil tekshiruvi bilan: no-op, chaqiruvchi tekshirishi shart emas
\`\`\`

**Amaliy foydalari:**

Bu pattern keshlarni ixtiyoriy bog'liqliklar qiladi - agar ishga tushirish muvaffaqiyatsiz bo'lsa, operatsiyalar buzilish o'rniga no-ops bo'ladi.`
		}
	}
};

export default task;
