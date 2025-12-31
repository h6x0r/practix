import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-ttlcache-get',
	title: 'Get Method',
	difficulty: 'medium',	tags: ['go', 'cache', 'concurrency', 'expiration'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement thread-safe **Get** method that returns values only if not expired.

**Requirements:**
1. Implement \`Get(key string) (any, bool)\` method on TTLCache
2. Return \`(nil, false)\` if cache is nil or key not found
3. Use RLock for initial lookup (optimistic read)
4. Check if entry is expired: \`!ent.exp.IsZero() && time.Now().After(ent.exp)\`
5. If expired, upgrade to write lock and delete entry (with double-check)
6. Return \`(nil, false)\` for expired/missing entries, \`(ent.v, true)\` for valid entries

**Example:**
\`\`\`go
cache := NewTTLCache(100 * time.Millisecond)
cache.Set("key", "value")

v, ok := cache.Get("key")  // v = "value", ok = true

time.Sleep(150 * time.Millisecond)
v, ok = cache.Get("key")  // v = nil, ok = false (expired)
\`\`\`

**Constraints:**
- Must use RLock first (optimize for read-heavy workloads)
- Must upgrade to Lock before deleting expired entries
- Must double-check expiration after acquiring write lock (prevent race)`,
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

// TODO: Implement Get method
// 1. Check if c is nil, return (nil, false)
// 2. Use RLock to read entry from map
// 3. Check if key exists and if entry is expired
// 4. If expired, upgrade to Lock and delete (with double-check)
// 5. Return value and existence status
func (c *TTLCache) Get(key string) (any, bool) {
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

func (c *TTLCache) Get(key string) (any, bool) {
	if c == nil {	// Nil cache never stores values
		return nil, false
	}
	c.mu.RLock()	// Take read lock for optimistic lookup
	ent, ok := c.m[key]	// Fetch entry if it exists
	c.mu.RUnlock()	// Release read lock before potential writes
	if !ok {	// Absent key yields miss immediately
		return nil, false
	}
	if !ent.exp.IsZero() && time.Now().After(ent.exp) {	// Expire stale entry regardless of ttl setting
		c.mu.Lock()	// Upgrade to write lock to delete entry
		defer c.mu.Unlock()	// Ensure lock release after cleanup
		if entCur, still := c.m[key]; still && entCur.exp == ent.exp {	// Confirm the same entry is present
			delete(c.m, key)	// Remove expired record from storage
		}
		return nil, false	// Report miss for expired value
	}
	return ent.v, true	// Return cached value while it is still valid
}`,
	testCode: `package cache

import (
	"testing"
	"time"
)

func TestGet_BasicGet(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	v, ok := cache.Get("key1")
	if !ok {
		t.Fatal("expected key1 to exist")
	}
	if v != "value1" {
		t.Errorf("expected value1, got %v", v)
	}
}

func TestGet_NonExistent(t *testing.T) {
	cache := NewTTLCache(time.Second)
	v, ok := cache.Get("nonexistent")
	if ok {
		t.Error("expected key to not exist")
	}
	if v != nil {
		t.Errorf("expected nil, got %v", v)
	}
}

func TestGet_NilCache(t *testing.T) {
	var cache *TTLCache
	v, ok := cache.Get("key1")
	if ok {
		t.Error("expected nil cache to return false")
	}
	if v != nil {
		t.Errorf("expected nil, got %v", v)
	}
}

func TestGet_Expired(t *testing.T) {
	cache := NewTTLCache(50 * time.Millisecond)
	cache.Set("key1", "value1")
	time.Sleep(100 * time.Millisecond)
	v, ok := cache.Get("key1")
	if ok {
		t.Error("expected expired key to return false")
	}
	if v != nil {
		t.Errorf("expected nil for expired key, got %v", v)
	}
}

func TestGet_NotExpired(t *testing.T) {
	cache := NewTTLCache(500 * time.Millisecond)
	cache.Set("key1", "value1")
	time.Sleep(100 * time.Millisecond)
	v, ok := cache.Get("key1")
	if !ok {
		t.Fatal("expected key1 to still exist")
	}
	if v != "value1" {
		t.Errorf("expected value1, got %v", v)
	}
}

func TestGet_ZeroTTLNeverExpires(t *testing.T) {
	cache := NewTTLCache(0)
	cache.Set("key1", "value1")
	time.Sleep(100 * time.Millisecond)
	v, ok := cache.Get("key1")
	if !ok {
		t.Fatal("expected key1 to exist (zero ttl never expires)")
	}
	if v != "value1" {
		t.Errorf("expected value1, got %v", v)
	}
}

func TestGet_DeletesExpired(t *testing.T) {
	cache := NewTTLCache(50 * time.Millisecond)
	cache.Set("key1", "value1")
	time.Sleep(100 * time.Millisecond)
	cache.Get("key1")
	if _, exists := cache.m["key1"]; exists {
		t.Error("expected expired key to be deleted from map")
	}
}

func TestGet_MultipleKeys(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	v1, ok1 := cache.Get("key1")
	v2, ok2 := cache.Get("key2")
	if !ok1 || !ok2 {
		t.Fatal("expected both keys to exist")
	}
	if v1 != "value1" || v2 != "value2" {
		t.Errorf("expected value1 and value2, got %v and %v", v1, v2)
	}
}

func TestGet_OverwrittenKey(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "old")
	cache.Set("key1", "new")
	v, ok := cache.Get("key1")
	if !ok {
		t.Fatal("expected key1 to exist")
	}
	if v != "new" {
		t.Errorf("expected new, got %v", v)
	}
}

func TestGet_DifferentTypes(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("int", 42)
	cache.Set("string", "hello")
	vInt, ok1 := cache.Get("int")
	vStr, ok2 := cache.Get("string")
	if !ok1 || !ok2 {
		t.Fatal("expected both keys to exist")
	}
	if vInt != 42 {
		t.Errorf("expected 42, got %v", vInt)
	}
	if vStr != "hello" {
		t.Errorf("expected hello, got %v", vStr)
	}
}
`,
			hint1: `Use RLock first for fast reads. Release it before upgrading to Lock to avoid deadlock.`,
			hint2: `After getting write lock, double-check that the entry still exists and has the same expiration (another goroutine might have updated it).`,
			whyItMatters: `The Get method demonstrates critical concurrency patterns: optimistic locking, lock upgrades, and double-checking.

**Why RLock First:**
- **Read optimization:** Multiple goroutines can read simultaneously
- **Performance:** 10-100x faster than exclusive locks for read-heavy workloads
- **Common case:** Most cache accesses are hits on non-expired entries

**Why Lock Upgrade Pattern:**
\`\`\`go
// WRONG: Hold RLock while deleting (deadlock!)
c.mu.RLock()
if expired {
    c.mu.Lock()  // DEADLOCK: Can't upgrade while holding RLock
    delete(c.m, key)
    c.mu.Unlock()
}
c.mu.RUnlock()

// CORRECT: Release RLock before acquiring Lock
c.mu.RLock()
ent, ok := c.m[key]
c.mu.RUnlock()  // Release before upgrade
if expired {
    c.mu.Lock()
    // Must double-check here!
    c.mu.Unlock()
}
\`\`\`

**Why Double-Check:**
\`\`\`go
// Timeline of race condition:
// T1: RLock, read entry (expires at 10:00:00.100)
// T1: RUnlock
// T2: Lock, Set same key (new expiration 10:00:05.000)
// T2: Unlock
// T1: Lock, delete key (WRONG: deletes fresh entry!)

// Solution: Verify expiration hasn't changed
if entCur, still := c.m[key]; still && entCur.exp == ent.exp {
    delete(c.m, key)  // Safe: same entry we checked earlier
}
\`\`\`

**Production Impact:**
\`\`\`go
// Read-heavy workload: 10,000 reads/sec, 100 writes/sec
// RLock:  ~100ns per read  = 1ms total
// Lock:   ~1μs per read    = 10ms total (10x slower)
\`\`\`

**Real-World Examples:**
- **groupcache:** Uses same RLock → Lock upgrade pattern
- **sync.Map:** Optimistic read path for hot keys
- **Database connections:** Read-heavy connection pools use RWMutex

**Why Zero Time Check:**
\`\`\`go
!ent.exp.IsZero()  // Check if expiration was set
// Zero time means "never expires" (when ttl <= 0 in Set)
// Prevents treating all entries as expired
\`\`\`

**Common Mistakes:**
1. **Forgetting to unlock before upgrade:** Causes deadlock
2. **Not double-checking after upgrade:** Race condition deletes fresh entries
3. **Using Lock for reads:** Kills cache performance
4. **Checking IsZero() without !:** Treats never-expire entries as expired

This pattern appears in every high-performance concurrent cache implementation.`,	order: 2,
	translations: {
		ru: {
			title: 'Получение элемента из кэша',
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

func (c *TTLCache) Get(key string) (any, bool) {
	if c == nil {	// Nil cache никогда не хранит значений
		return nil, false
	}
	c.mu.RLock()	// Берём read lock для оптимистичного поиска
	ent, ok := c.m[key]	// Получаем entry если существует
	c.mu.RUnlock()	// Освобождаем read lock перед потенциальной записью
	if !ok {	// Отсутствующий ключ немедленно возвращает miss
		return nil, false
	}
	if !ent.exp.IsZero() && time.Now().After(ent.exp) {	// Истёкшую entry удаляем независимо от ttl
		c.mu.Lock()	// Повышаемся до write lock для удаления
		defer c.mu.Unlock()	// Гарантируем освобождение lock после очистки
		if entCur, still := c.m[key]; still && entCur.exp == ent.exp {	// Подтверждаем та же entry
			delete(c.m, key)	// Удаляем истёкшую запись из хранилища
		}
		return nil, false	// Сообщаем miss для истёкшего значения
	}
	return ent.v, true	// Возвращаем кешированное значение пока валидно
}`,
			description: `Реализуйте потокобезопасный метод **Get**, возвращающий значения только если не истекли.

**Требования:**
1. Реализуйте метод \`Get(key string) (any, bool)\` для TTLCache
2. Верните \`(nil, false)\` если cache nil или ключ не найден
3. Используйте RLock для начального поиска (оптимистичное чтение)
4. Проверьте истечение: \`!ent.exp.IsZero() && time.Now().After(ent.exp)\`
5. Если истекло, обновитесь до write lock и удалите (с двойной проверкой)
6. Верните \`(nil, false)\` для истёкших/отсутствующих, \`(ent.v, true)\` для валидных

**Пример:**
\`\`\`go
cache := NewTTLCache(100 * time.Millisecond)
cache.Set("key", "value")

v, ok := cache.Get("key")  // v = "value", ok = true

time.Sleep(150 * time.Millisecond)
v, ok = cache.Get("key")  // v = nil, ok = false (истекло)
\`\`\`

**Ограничения:**
- Используйте RLock сначала
- Обновитесь до Lock перед удалением
- Двойная проверка после write lock`,
			hint1: `Используйте RLock для быстрого чтения. Освободите перед обновлением до Lock во избежание deadlock.`,
			hint2: `После получения write lock дважды проверьте, что entry существует и имеет то же истечение (другая goroutine могла обновить).`,
			whyItMatters: `Метод Get демонстрирует критические паттерны concurrency: оптимистичную блокировку, обновление блокировок и двойную проверку.

**Почему RLock сначала:**
- **Оптимизация чтения:** Множество goroutines могут читать одновременно
- **Производительность:** В 10-100 раз быстрее эксклюзивных блокировок для read-heavy нагрузок
- **Обычный случай:** Большинство обращений к кешу - попадания на неистёкшие записи

**Почему паттерн обновления блокировки:**
\`\`\`go
// НЕПРАВИЛЬНО: Держать RLock при удалении (deadlock!)
c.mu.RLock()
if expired {
    c.mu.Lock()  // DEADLOCK: Нельзя обновиться держа RLock
    delete(c.m, key)
    c.mu.Unlock()
}
c.mu.RUnlock()

// ПРАВИЛЬНО: Освободить RLock перед получением Lock
c.mu.RLock()
ent, ok := c.m[key]
c.mu.RUnlock()  // Освободить перед обновлением
if expired {
    c.mu.Lock()
    // Должна быть двойная проверка здесь!
    c.mu.Unlock()
}
\`\`\`

**Почему двойная проверка:**
\`\`\`go
// Временная шкала состояния гонки:
// T1: RLock, прочитать entry (истекает в 10:00:00.100)
// T1: RUnlock
// T2: Lock, Set того же ключа (новое истечение 10:00:05.000)
// T2: Unlock
// T1: Lock, delete ключа (НЕПРАВИЛЬНО: удаляет свежую запись!)

// Решение: Проверить что истечение не изменилось
if entCur, still := c.m[key]; still && entCur.exp == ent.exp {
    delete(c.m, key)  // Безопасно: та же запись что проверяли
}
\`\`\`

**Влияние на продакшен:**
\`\`\`go
// Read-heavy нагрузка: 10,000 чтений/сек, 100 записей/сек
// RLock:  ~100ns на чтение  = 1ms всего
// Lock:   ~1μs на чтение    = 10ms всего (в 10x медленнее)
\`\`\`

**Примеры из реального мира:**
- **groupcache:** Использует тот же паттерн обновления RLock → Lock
- **sync.Map:** Оптимистичный путь чтения для горячих ключей
- **Пулы подключений к БД:** Read-heavy пулы используют RWMutex

**Почему проверка нулевого времени:**
\`\`\`go
!ent.exp.IsZero()  // Проверить установлено ли истечение
// Нулевое время означает "никогда не истекает" (когда ttl <= 0 в Set)
// Предотвращает трактовку всех записей как истёкших
\`\`\`

**Распространённые ошибки:**
1. **Забыть unlock перед обновлением:** Вызывает deadlock
2. **Не перепроверять после обновления:** Состояние гонки удаляет свежие записи
3. **Использовать Lock для чтения:** Убивает производительность кеша
4. **Проверять IsZero() без !:** Трактует never-expire записи как истёкшие

**Практические преимущества:**

Этот паттерн встречается в каждой высокопроизводительной реализации concurrent кеша.`
		},
		uz: {
			title: `Keshdan element olish`,
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

func (c *TTLCache) Get(key string) (any, bool) {
	if c == nil {	// Nil cache hech qachon qiymat saqlamaydi
		return nil, false
	}
	c.mu.RLock()	// Optimistik qidirish uchun o'qish qulfini olamiz
	ent, ok := c.m[key]	// Mavjud bo'lsa yozuvni olamiz
	c.mu.RUnlock()	// Potentsial yozishlardan oldin o'qish qulfini bo'shatamiz
	if !ok {	// Mavjud bo'lmagan kalit darhol miss qaytaradi
		return nil, false
	}
	if !ent.exp.IsZero() && time.Now().After(ent.exp) {	// Muddati o'tgan yozuvni ttl sozlamasidan qat'i nazar o'chiramiz
		c.mu.Lock()	// Yozuvni o'chirish uchun yozish qulfiga ko'taramiz
		defer c.mu.Unlock()	// Tozalashdan keyin qulf bo'shatilishini kafolatlaymiz
		if entCur, still := c.m[key]; still && entCur.exp == ent.exp {	// Bir xil yozuv mavjudligini tasdiqlaymiz
			delete(c.m, key)	// Muddati o'tgan yozuvni xotiradan o'chiramiz
		}
		return nil, false	// Muddati o'tgan qiymat uchun miss xabar qilamiz
	}
	return ent.v, true	// Keshlangan qiymatni hali yaroqli ekan qaytaramiz
}`,
			description: `Faqat muddati o'tmagan qiymatlarni qaytaradigan thread-safe **Get** metodini amalga oshiring.

**Talablar:**
1. TTLCache uchun \`Get(key string) (any, bool)\` metodini amalga oshiring
2. cache nil yoki kalit topilmasa \`(nil, false)\` qaytaring
3. Dastlabki qidirish uchun RLock dan foydalaning (optimistik o'qish)
4. Muddati o'tganligini tekshiring: \`!ent.exp.IsZero() && time.Now().After(ent.exp)\`
5. Agar muddati o'tgan bo'lsa, yozish qulfiga ko'taring va yozuvni o'chiring (ikki marta tekshirish bilan)
6. Muddati o'tgan/yo'q yozuvlar uchun \`(nil, false)\`, yaroqli uchun \`(ent.v, true)\` qaytaring

**Misol:**
\`\`\`go
cache := NewTTLCache(100 * time.Millisecond)
cache.Set("key", "value")

v, ok := cache.Get("key")  // v = "value", ok = true

time.Sleep(150 * time.Millisecond)
v, ok = cache.Get("key")  // v = nil, ok = false (muddati o'tdi)
\`\`\`

**Cheklovlar:**
- Avval RLock ishlating (read-heavy ish yuklari uchun optimallash)
- Muddati o'tgan yozuvlarni o'chirishdan oldin Lock ga ko'taring
- Yozish qulfini olgandan keyin muddati o'tishni ikki marta tekshiring (poyga oldini olish)`,
			hint1: `Tez o'qishlar uchun avval RLock ishlating. Deadlock dan qochish uchun Lock ga ko'tarishdan oldin bo'shating.`,
			hint2: `Yozish qulfini olgandan keyin, yozuv hali mavjud va bir xil muddati o'tishga ega ekanligini ikki marta tekshiring (boshqa goroutine yangilagan bo'lishi mumkin).`,
			whyItMatters: `Get metodi muhim concurrency pattern larni ko'rsatadi: optimistik qulflash, qulf ko'tarish va ikki marta tekshirish.

**Nima uchun avval RLock:**
- **O'qishni optimallashtirish:** Bir nechta goroutine bir vaqtda o'qishi mumkin
- **Ishlash:** Read-heavy ish yuklari uchun eksklyuziv qulflardan 10-100 marta tezroq
- **Umumiy holat:** Kesh kirishlarining aksariyati muddati o'tmagan yozuvlarga tegishlar

**Nima uchun qulf ko'tarish patterni:**
\`\`\`go
// NOTO'G'RI: O'chirishda RLock ni ushlab turish (deadlock!)
c.mu.RLock()
if expired {
    c.mu.Lock()  // DEADLOCK: RLock ni ushlab turib ko'tarib bo'lmaydi
    delete(c.m, key)
    c.mu.Unlock()
}
c.mu.RUnlock()

// TO'G'RI: Lock olishdan oldin RLock ni bo'shatish
c.mu.RLock()
ent, ok := c.m[key]
c.mu.RUnlock()  // Ko'tarishdan oldin bo'shatish
if expired {
    c.mu.Lock()
    // Bu yerda ikki marta tekshirish kerak!
    c.mu.Unlock()
}
\`\`\`

**Nima uchun ikki marta tekshirish:**
\`\`\`go
// Poyga holatining vaqt chizig'i:
// T1: RLock, yozuvni o'qish (10:00:00.100 da muddati o'tadi)
// T1: RUnlock
// T2: Lock, bir xil kalitni Set (yangi muddati o'tish 10:00:05.000)
// T2: Unlock
// T1: Lock, kalitni delete (NOTO'G'RI: yangi yozuvni o'chiradi!)

// Yechim: Muddati o'tish o'zgarmaganligini tekshirish
if entCur, still := c.m[key]; still && entCur.exp == ent.exp {
    delete(c.m, key)  // Xavfsiz: avval tekshirgan bir xil yozuv
}
\`\`\`

**Ishlab chiqarishga ta'siri:**
\`\`\`go
// Read-heavy ish yuklamasi: 10,000 o'qish/sek, 100 yozish/sek
// RLock:  ~100ns har bir o'qish  = 1ms jami
// Lock:   ~1μs har bir o'qish    = 10ms jami (10x sekinroq)
\`\`\`

**Haqiqiy dunyodan misollar:**
- **groupcache:** Bir xil RLock → Lock ko'tarish patternidan foydalanadi
- **sync.Map:** Issiq kalitlar uchun optimistik o'qish yo'li
- **Ma'lumotlar bazasi ulanishlari:** Read-heavy ulanish havzalari RWMutex dan foydalanadi

**Nima uchun nol vaqt tekshiruvi:**
\`\`\`go
!ent.exp.IsZero()  // Muddati o'tish o'rnatilgan yoki yo'qligini tekshirish
// Nol vaqt "hech qachon muddati o'tmaydi" degani (Set da ttl <= 0 bo'lganda)
// Barcha yozuvlarni muddati o'tgan deb hisoblanishini oldini oladi
\`\`\`

**Keng tarqalgan xatolar:**
1. **Ko'tarishdan oldin unlock ni unutish:** Deadlock ga olib keladi
2. **Ko'tarishdan keyin qayta tekshirmaslik:** Poyga holati yangi yozuvlarni o'chiradi
3. **O'qish uchun Lock ishlatish:** Kesh ishlashini buzadi
4. **IsZero() ni ! siz tekshirish:** Never-expire yozuvlarni muddati o'tgan deb hisoblaydi

**Amaliy foydalari:**

Bu pattern har bir yuqori ishlashli concurrent kesh implementatsiyasida uchraydi.`
		}
	}
};

export default task;
