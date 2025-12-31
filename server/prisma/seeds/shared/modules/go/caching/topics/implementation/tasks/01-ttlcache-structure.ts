import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-ttlcache-structure',
	title: 'TTLCache Structure & New',
	difficulty: 'easy',	tags: ['go', 'cache', 'concurrency', 'sync'],
	estimatedTime: '15m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **TTLCache** structure with thread-safe initialization.

**Requirements:**
1. Define \`entry\` struct with fields: \`v any\` (value), \`exp time.Time\` (expiration)
2. Define \`TTLCache\` struct with: \`mu sync.RWMutex\`, \`m map[string]entry\`, \`ttl time.Duration\`
3. Implement \`NewTTLCache(ttl time.Duration) *TTLCache\` constructor
4. Initialize map and set default TTL to 1ms if ttl <= 0

**Example:**
\`\`\`go
cache := NewTTLCache(5 * time.Second)
// cache.ttl = 5s
// cache.m = empty map
// cache.mu = RWMutex ready to use

invalidCache := NewTTLCache(-1 * time.Second)
// invalidCache.ttl = 1ms (default minimum)
\`\`\`

**Constraints:**
- Must initialize map in constructor (avoid nil map panics)
- TTL must be positive (minimum 1ms)
- Use RWMutex for concurrent read/write access`,
	initialCode: `package cache

import (
	"sync"
	"time"
)

// TODO: Define entry struct to store value and expiration time
type entry struct {
	v   any
	exp time.Time
}

// TODO: Define TTLCache with mutex, map, and ttl
type TTLCache struct {
	mu  sync.RWMutex
	m   map[string]entry
	ttl time.Duration
}

// TODO: Implement constructor that initializes cache
// If ttl <= 0, set ttl to 1ms
func NewTTLCache(ttl time.Duration) *TTLCache {
	// TODO: Implement
}`,
	solutionCode: `package cache

import (
	"sync"
	"time"
)

type entry struct {
	v   any	// Cached value (can be any type)
	exp time.Time	// Absolute expiration timestamp
}

type TTLCache struct {
	mu  sync.RWMutex	// Protects concurrent access to map
	m   map[string]entry	// Stores key-value pairs with expiration
	ttl time.Duration	// Time-to-live for all entries
}

func NewTTLCache(ttl time.Duration) *TTLCache {
	if ttl <= 0 {	// Validate TTL is positive
		ttl = time.Millisecond	// Set minimum TTL to prevent immediate expiration
	}
	return &TTLCache{
		m:   make(map[string]entry),	// Initialize map to avoid nil panics
		ttl: ttl,	// Store TTL for future Set operations
	}
}`,
	testCode: `package cache

import (
	"testing"
	"time"
)

func TestNewTTLCache_ValidTTL(t *testing.T) {
	cache := NewTTLCache(5 * time.Second)
	if cache == nil {
		t.Fatal("expected cache to be created, got nil")
	}
	if cache.ttl != 5*time.Second {
		t.Errorf("expected ttl to be 5s, got %v", cache.ttl)
	}
}

func TestNewTTLCache_MapInitialized(t *testing.T) {
	cache := NewTTLCache(time.Second)
	if cache.m == nil {
		t.Fatal("expected map to be initialized, got nil")
	}
	if len(cache.m) != 0 {
		t.Errorf("expected empty map, got length %d", len(cache.m))
	}
}

func TestNewTTLCache_NegativeTTL(t *testing.T) {
	cache := NewTTLCache(-1 * time.Second)
	if cache.ttl != time.Millisecond {
		t.Errorf("expected ttl to be 1ms for negative input, got %v", cache.ttl)
	}
}

func TestNewTTLCache_ZeroTTL(t *testing.T) {
	cache := NewTTLCache(0)
	if cache.ttl != time.Millisecond {
		t.Errorf("expected ttl to be 1ms for zero input, got %v", cache.ttl)
	}
}

func TestNewTTLCache_MinimumTTL(t *testing.T) {
	cache := NewTTLCache(time.Nanosecond)
	if cache.ttl != time.Millisecond {
		t.Errorf("expected ttl to be 1ms for nanosecond input, got %v", cache.ttl)
	}
}

func TestNewTTLCache_LargeTTL(t *testing.T) {
	cache := NewTTLCache(24 * time.Hour)
	if cache.ttl != 24*time.Hour {
		t.Errorf("expected ttl to be 24h, got %v", cache.ttl)
	}
}

func TestNewTTLCache_DefaultMinimumTTL(t *testing.T) {
	cache := NewTTLCache(-100 * time.Second)
	if cache.ttl != time.Millisecond {
		t.Errorf("expected ttl to be 1ms for large negative input, got %v", cache.ttl)
	}
}

func TestNewTTLCache_MultipleCaches(t *testing.T) {
	cache1 := NewTTLCache(time.Second)
	cache2 := NewTTLCache(time.Minute)
	if cache1 == cache2 {
		t.Error("expected different cache instances")
	}
	if cache1.ttl == cache2.ttl {
		t.Error("expected different ttl values")
	}
}

func TestNewTTLCache_StructFields(t *testing.T) {
	cache := NewTTLCache(10 * time.Second)
	if cache.m == nil {
		t.Error("expected m to be initialized")
	}
	if cache.ttl != 10*time.Second {
		t.Errorf("expected ttl to be 10s, got %v", cache.ttl)
	}
}

func TestNewTTLCache_EdgeCaseBoundary(t *testing.T) {
	cache := NewTTLCache(time.Millisecond)
	if cache.ttl != time.Millisecond {
		t.Errorf("expected ttl to be 1ms, got %v", cache.ttl)
	}
	if cache.m == nil {
		t.Error("expected map to be initialized")
	}
}
`,
			hint1: `Use sync.RWMutex for read-heavy concurrent access. Initialize map with make() in constructor.`,
			hint2: `Check if ttl <= 0 and set it to time.Millisecond as a safe minimum value.`,
			whyItMatters: `TTL (Time-To-Live) caches automatically expire entries, preventing stale data and memory leaks in production systems.

**Why TTL Caches:**
- **Automatic cleanup:** No manual invalidation needed
- **Memory safety:** Old entries don't accumulate indefinitely
- **Data freshness:** Ensures clients get reasonably current data
- **Simplicity:** Single expiration policy for all entries

**Production Use Cases:**
\`\`\`go
// Session cache (30 minute TTL)
sessions := NewTTLCache(30 * time.Minute)

// API rate limiting (1 minute TTL)
rateLimits := NewTTLCache(time.Minute)

// DNS lookup cache (5 minute TTL)
dnsCache := NewTTLCache(5 * time.Minute)
\`\`\`

**Why RWMutex:**
- **Read optimization:** Multiple goroutines can read simultaneously
- **Write safety:** Exclusive lock for Set/Delete operations
- **Performance:** Better than sync.Mutex for read-heavy workloads

**Real-World Examples:**
- Redis: TTL for automatic key expiration
- Memcached: Expiration time per item
- groupcache: Time-based eviction policies
- go-cache: Popular TTL cache library

**Production Pattern:**
\`\`\`go
var userCache = NewTTLCache(15 * time.Minute)

func GetUser(id string) (*User, error) {
    if cached, ok := userCache.Get(id); ok {
        return cached.(*User), nil  // Cache hit
    }
    user, err := db.QueryUser(id)
    if err == nil {
        userCache.Set(id, user)     // Cache for future requests
    }
    return user, err
}
\`\`\`

**Real-World Benefits:**

Without TTL, you'd need manual invalidation logic, which is error-prone and often leads to stale data bugs in production.`,	order: 0,
	translations: {
		ru: {
			title: 'Структура кэша с временем жизни (TTL)',
			solutionCode: `package cache

import (
	"sync"
	"time"
)

type entry struct {
	v   any	// Кешированное значение (любой тип)
	exp time.Time	// Абсолютный timestamp истечения
}

type TTLCache struct {
	mu  sync.RWMutex	// Защищает конкурентный доступ к map
	m   map[string]entry	// Хранит пары ключ-значение с истечением
	ttl time.Duration	// Time-to-live для всех записей
}

func NewTTLCache(ttl time.Duration) *TTLCache {
	if ttl <= 0 {	// Валидация TTL положительный
		ttl = time.Millisecond	// Минимальный TTL для предотвращения немедленного истечения
	}
	return &TTLCache{
		m:   make(map[string]entry),	// Инициализация map для избежания nil паники
		ttl: ttl,	// Сохраняем TTL для будущих Set операций
	}
}`,
			description: `Реализуйте структуру **TTLCache** с потокобезопасной инициализацией.

**Требования:**
1. Определите структуру \`entry\` с полями: \`v any\`, \`exp time.Time\`
2. Определите структуру \`TTLCache\` с: \`mu sync.RWMutex\`, \`m map[string]entry\`, \`ttl time.Duration\`
3. Реализуйте конструктор \`NewTTLCache(ttl time.Duration) *TTLCache\`
4. Инициализируйте map и установите TTL в 1ms, если ttl <= 0

**Пример:**
\`\`\`go
cache := NewTTLCache(5 * time.Second)
// cache.ttl = 5s
// cache.m = пустая map
// cache.mu = RWMutex готов к использованию

invalidCache := NewTTLCache(-1 * time.Second)
// invalidCache.ttl = 1ms (минимум по умолчанию)
\`\`\`

**Ограничения:**
- Инициализируйте map в конструкторе
- TTL должен быть положительным
- Используйте RWMutex для concurrent доступа`,
			hint1: `Используйте sync.RWMutex для read-heavy нагрузки. Инициализируйте map через make().`,
			hint2: `Проверьте ttl <= 0 и установите time.Millisecond как безопасный минимум.`,
			whyItMatters: `TTL (Time-To-Live) кеши автоматически удаляют устаревшие записи, предотвращая утечки памяти и устаревшие данные.

**Почему TTL кеши:**
- **Автоматическая очистка:** Не нужна ручная инвалидация
- **Безопасность памяти:** Старые записи не накапливаются бесконечно
- **Свежесть данных:** Клиенты получают разумно актуальные данные
- **Простота:** Единая политика истечения для всех записей

**Примеры использования в продакшене:**
\`\`\`go
// Кеш сессий (TTL 30 минут)
sessions := NewTTLCache(30 * time.Minute)

// API rate limiting (TTL 1 минута)
rateLimits := NewTTLCache(time.Minute)

// Кеш DNS lookup (TTL 5 минут)
dnsCache := NewTTLCache(5 * time.Minute)
\`\`\`

**Почему RWMutex:**
- **Оптимизация чтения:** Множественные goroutines могут читать одновременно
- **Безопасность записи:** Эксклюзивная блокировка для операций Set/Delete
- **Производительность:** Лучше чем sync.Mutex для read-heavy нагрузок

**Примеры из реального мира:**
- Redis: TTL для автоматического истечения ключей
- Memcached: Время истечения для каждого элемента
- groupcache: Политики вытеснения на основе времени
- go-cache: Популярная библиотека TTL кеша

**Продакшен паттерн:**
\`\`\`go
var userCache = NewTTLCache(15 * time.Minute)

func GetUser(id string) (*User, error) {
    if cached, ok := userCache.Get(id); ok {
        return cached.(*User), nil  // Попадание в кеш
    }
    user, err := db.QueryUser(id)
    if err == nil {
        userCache.Set(id, user)     // Кешируем для будущих запросов
    }
    return user, err
}
\`\`\`

**Практические преимущества:**

Без TTL потребовалась бы логика ручной инвалидации, которая подвержена ошибкам и часто приводит к багам с устаревшими данными в продакшене.`
		},
		uz: {
			title: `Vaqt chegarali (TTL) kesh strukturasi`,
			solutionCode: `package cache

import (
	"sync"
	"time"
)

type entry struct {
	v   any	// Keshlangan qiymat (har qanday tur)
	exp time.Time	// Mutlaq muddati o'tish timestamp i
}

type TTLCache struct {
	mu  sync.RWMutex	// map ga parallel kirishni himoya qiladi
	m   map[string]entry	// Kalit-qiymat juftlarini muddati o'tish bilan saqlaydi
	ttl time.Duration	// Barcha yozuvlar uchun yashash vaqti
}

func NewTTLCache(ttl time.Duration) *TTLCache {
	if ttl <= 0 {	// TTL ijobiy ekanligini tekshirish
		ttl = time.Millisecond	// Darhol muddati o'tishni oldini olish uchun minimal TTL
	}
	return &TTLCache{
		m:   make(map[string]entry),	// nil panikni oldini olish uchun map ni ishga tushirish
		ttl: ttl,	// Kelajakdagi Set operatsiyalari uchun TTL ni saqlash
	}
}`,
			description: `Thread-safe ishga tushirish bilan **TTLCache** strukturasini amalga oshiring.

**Talablar:**
1. Maydonlari bilan \`entry\` strukturasini aniqlang: \`v any\`, \`exp time.Time\`
2. \`TTLCache\` strukturasini aniqlang: \`mu sync.RWMutex\`, \`m map[string]entry\`, \`ttl time.Duration\`
3. \`NewTTLCache(ttl time.Duration) *TTLCache\` konstruktorini amalga oshiring
4. map ni ishga tushiring va ttl <= 0 bo'lsa TTL ni 1ms ga o'rnating

**Misol:**
\`\`\`go
cache := NewTTLCache(5 * time.Second)
// cache.ttl = 5s
// cache.m = bo'sh map
// cache.mu = RWMutex ishlatishga tayyor

invalidCache := NewTTLCache(-1 * time.Second)
// invalidCache.ttl = 1ms (standart minimal)
\`\`\`

**Cheklovlar:**
- Konstruktorda map ni ishga tushiring
- TTL ijobiy bo'lishi kerak
- Bir vaqtda o'qish/yozish uchun RWMutex ishlating`,
			hint1: `Read-heavy concurrent kirish uchun sync.RWMutex dan foydalaning. Konstruktorda map ni make() bilan ishga tushiring.`,
			hint2: `ttl <= 0 tekshiring va xavfsiz minimal qiymat sifatida time.Millisecond o'rnating.`,
			whyItMatters: `TTL (Time-To-Live) keshlar avtomatik ravishda yozuvlarni muddati o'tganda o'chiradi, bu eskirgan ma'lumotlar va xotira oqishlarini oldini oladi.

**Nima uchun TTL keshlar:**
- **Avtomatik tozalash:** Qo'lda invalidatsiya kerak emas
- **Xotira xavfsizligi:** Eski yozuvlar cheksiz to'planmaydi
- **Ma'lumotlar yangiligi:** Mijozlar oqilona joriy ma'lumotlarni olishini ta'minlaydi
- **Oddiylik:** Barcha yozuvlar uchun yagona muddati o'tish siyosati

**Ishlab chiqarishda foydalanish misollari:**
\`\`\`go
// Sessiya keshi (30 daqiqa TTL)
sessions := NewTTLCache(30 * time.Minute)

// API rate limiting (1 daqiqa TTL)
rateLimits := NewTTLCache(time.Minute)

// DNS lookup keshi (5 daqiqa TTL)
dnsCache := NewTTLCache(5 * time.Minute)
\`\`\`

**Nima uchun RWMutex:**
- **O'qishni optimallashtirish:** Bir nechta goroutine bir vaqtda o'qishi mumkin
- **Yozish xavfsizligi:** Set/Delete operatsiyalari uchun eksklyuziv qulf
- **Ishlash:** Read-heavy ish yuklari uchun sync.Mutex dan yaxshiroq

**Haqiqiy dunyodan misollar:**
- Redis: Kalitlar avtomatik muddati o'tishi uchun TTL
- Memcached: Har bir element uchun muddati o'tish vaqti
- groupcache: Vaqtga asoslangan chiqarish siyosatlari
- go-cache: Mashhur TTL kesh kutubxonasi

**Ishlab chiqarish patterni:**
\`\`\`go
var userCache = NewTTLCache(15 * time.Minute)

func GetUser(id string) (*User, error) {
    if cached, ok := userCache.Get(id); ok {
        return cached.(*User), nil  // Keshga tegish
    }
    user, err := db.QueryUser(id)
    if err == nil {
        userCache.Set(id, user)     // Kelajakdagi so'rovlar uchun keshlash
    }
    return user, err
}
\`\`\`

**Amaliy foydalari:**

TTL bo'lmasa, qo'lda invalidatsiya mantiqi kerak bo'lar edi, bu xatolarga moyil va ko'pincha production da eskirgan ma'lumot xatolariga olib keladi.`
		}
	}
};

export default task;
