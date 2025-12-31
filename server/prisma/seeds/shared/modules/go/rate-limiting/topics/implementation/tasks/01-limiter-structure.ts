import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-ratelimit-limiter-structure',
	title: 'Limiter Structure',
	difficulty: 'easy',	tags: ['go', 'rate-limiting', 'concurrency', 'sync'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Design a **Limiter** struct that stores rate limiting configuration and state without using external libraries.

**Requirements:**
1. Create struct \`Limiter\` with fields for configuration and state
2. Store interval between tokens (\`time.Duration\`)
3. Store burst size (\`int\`)
4. Track token reservations as slice of \`time.Time\`
5. Use \`sync.Mutex\` to protect concurrent access

**Example:**
\`\`\`go
type Limiter struct {
    mu           sync.Mutex
    interval     time.Duration  // Time between tokens
    burst        int            // Maximum tokens
    reservations []time.Time    // When tokens become available
}
\`\`\`

**Constraints:**
- Must use mutex for thread safety
- Must track reservations as time moments
- No external rate limiting libraries`,
	initialCode: `package ratelimit

import (
	"sync"
	"time"
)

// TODO: Design Limiter struct with configuration and state
type Limiter struct {
	// Your fields here
}`,
	solutionCode: `package ratelimit

import (
	"sync"
	"time"
)

// Limiter implements token bucket rate limiting without external dependencies
type Limiter struct {
	mu           sync.Mutex    // Protects concurrent access to reservations
	interval     time.Duration // Duration between tokens (e.g., time.Second / rps)
	burst        int           // Maximum tokens available (bucket capacity)
	reservations []time.Time   // Timestamps when reserved tokens become available again
}`,
			hint1: `Store interval as time.Duration (calculated from RPS), burst as int, and reservations as []time.Time.`,
			hint2: `Use sync.Mutex to protect the reservations slice from race conditions when multiple goroutines access it.`,
			testCode: `package ratelimit

import (
	"reflect"
	"sync"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Test Limiter struct exists
	l := &Limiter{}
	if l == nil {
		t.Error("Limiter struct should be creatable")
	}
}

func Test2(t *testing.T) {
	// Test Limiter has mu field of type sync.Mutex
	l := Limiter{}
	typ := reflect.TypeOf(l)
	field, ok := typ.FieldByName("mu")
	if !ok {
		t.Error("Limiter should have 'mu' field")
	}
	if field.Type != reflect.TypeOf(sync.Mutex{}) {
		t.Error("'mu' field should be sync.Mutex")
	}
}

func Test3(t *testing.T) {
	// Test Limiter has interval field of type time.Duration
	l := Limiter{}
	typ := reflect.TypeOf(l)
	field, ok := typ.FieldByName("interval")
	if !ok {
		t.Error("Limiter should have 'interval' field")
	}
	if field.Type != reflect.TypeOf(time.Duration(0)) {
		t.Error("'interval' field should be time.Duration")
	}
}

func Test4(t *testing.T) {
	// Test Limiter has burst field of type int
	l := Limiter{}
	typ := reflect.TypeOf(l)
	field, ok := typ.FieldByName("burst")
	if !ok {
		t.Error("Limiter should have 'burst' field")
	}
	if field.Type.Kind() != reflect.Int {
		t.Error("'burst' field should be int")
	}
}

func Test5(t *testing.T) {
	// Test Limiter has reservations field of type []time.Time
	l := Limiter{}
	typ := reflect.TypeOf(l)
	field, ok := typ.FieldByName("reservations")
	if !ok {
		t.Error("Limiter should have 'reservations' field")
	}
	if field.Type != reflect.TypeOf([]time.Time{}) {
		t.Error("'reservations' field should be []time.Time")
	}
}

func Test6(t *testing.T) {
	// Test struct has exactly 4 fields
	l := Limiter{}
	typ := reflect.TypeOf(l)
	if typ.NumField() != 4 {
		t.Errorf("Limiter should have 4 fields, got %d", typ.NumField())
	}
}

func Test7(t *testing.T) {
	// Test setting interval
	l := Limiter{interval: 100 * time.Millisecond}
	if l.interval != 100*time.Millisecond {
		t.Errorf("interval = %v, want 100ms", l.interval)
	}
}

func Test8(t *testing.T) {
	// Test setting burst
	l := Limiter{burst: 10}
	if l.burst != 10 {
		t.Errorf("burst = %d, want 10", l.burst)
	}
}

func Test9(t *testing.T) {
	// Test setting reservations
	now := time.Now()
	l := Limiter{reservations: []time.Time{now}}
	if len(l.reservations) != 1 {
		t.Errorf("reservations len = %d, want 1", len(l.reservations))
	}
}

func Test10(t *testing.T) {
	// Test mutex can be locked/unlocked
	l := Limiter{}
	l.mu.Lock()
	l.mu.Unlock()
}`,
			whyItMatters: `Understanding rate limiter structure is fundamental to implementing production-grade API throttling.

**Why This Design:**
- **Token Bucket Algorithm:** Track when tokens become available using timestamps
- **No External Dependencies:** Native implementation without golang.org/x/time/rate
- **Thread-Safe:** Mutex protects state across concurrent requests
- **Memory Efficient:** Only store active reservations, not entire token history

**Production Pattern:**
\`\`\`go
// Token bucket: interval controls refill rate, burst controls capacity
type Limiter struct {
    mu           sync.Mutex
    interval     time.Duration  // time.Second / 100 = 10ms per token at 100 RPS
    burst        int            // 10 tokens = handle bursts of 10 requests instantly
    reservations []time.Time    // Only store future availability times
}

// Example: 100 RPS with burst of 10
limiter := &Limiter{
    interval:     10 * time.Millisecond,  // 100 RPS
    burst:        10,
    reservations: make([]time.Time, 0, 10),
}
\`\`\`

**Real-World Applications:**
- **API Gateway:** Rate limit requests per API key (1000 req/min, burst 50)
- **Database Throttling:** Limit query rate (100 queries/sec, burst 10)
- **External API Calls:** Respect third-party rate limits (Twitter: 300 req/15min)
- **Background Jobs:** Throttle email sending (10 emails/sec, burst 5)

**Key Concepts:**
- **Interval:** \`time.Second / rps\` gives time between tokens
- **Burst:** Maximum requests that can execute immediately
- **Reservations:** Track when each token becomes available again
- **Mutex:** Essential for goroutine safety in high-concurrency scenarios

**Alternatives:**
- **golang.org/x/time/rate:** Official implementation (we're building from scratch)
- **Token Bucket:** What we're implementing (flexible, allows bursts)
- **Leaky Bucket:** Fixed rate without bursts (stricter)
- **Sliding Window:** Complex but precise rate limiting

This native implementation gives you full control and deep understanding of rate limiting internals.`,	order: 0,
	translations: {
		ru: {
			title: 'Структура лимитера запросов',
			solutionCode: `package ratelimit

import (
	"sync"
	"time"
)

// Limiter реализует token bucket rate limiting без внешних зависимостей
type Limiter struct {
	mu           sync.Mutex    // Защищает конкурентный доступ к reservations
	interval     time.Duration // Интервал между токенами (например time.Second / rps)
	burst        int           // Максимум доступных токенов (ёмкость bucket)
	reservations []time.Time   // Timestamps когда зарезервированные токены снова станут доступны
}`,
			description: `Спроектируйте структуру **Limiter**, которая хранит конфигурацию и состояние rate limiting без использования внешних библиотек.

**Требования:**
1. Создайте структуру \`Limiter\` с полями для конфигурации и состояния
2. Сохраните интервал между токенами (\`time.Duration\`)
3. Сохраните размер burst (\`int\`)
4. Отслеживайте резервирования токенов как срез \`time.Time\`
5. Используйте \`sync.Mutex\` для защиты конкурентного доступа

**Пример:**
\`\`\`go
type Limiter struct {
    mu           sync.Mutex
    interval     time.Duration  // Время между токенами
    burst        int            // Максимум токенов
    reservations []time.Time    // Когда токены станут доступны
}
\`\`\`

**Ограничения:**
- Должен использовать mutex для потокобезопасности
- Должен отслеживать резервирования как моменты времени
- Без внешних библиотек rate limiting`,
			hint1: `Храните interval как time.Duration (вычисляется из RPS), burst как int, и reservations как []time.Time.`,
			hint2: `Используйте sync.Mutex для защиты среза reservations от race conditions при многопоточном доступе.`,
			whyItMatters: `Понимание структуры rate limiter фундаментально для реализации production-grade API throttling.

**Почему такой дизайн:**
- **Алгоритм Token Bucket:** Отслеживание доступности токенов через timestamps
- **Без внешних зависимостей:** Нативная реализация без golang.org/x/time/rate
- **Потокобезопасность:** Mutex защищает состояние при конкурентных запросах
- **Эффективность памяти:** Хранятся только активные резервирования

**Продакшен паттерн:**
\`\`\`go
// Token bucket: interval управляет скоростью пополнения, burst управляет ёмкостью
type Limiter struct {
    mu           sync.Mutex
    interval     time.Duration  // time.Second / 100 = 10ms на токен при 100 RPS
    burst        int            // 10 токенов = обработка burst из 10 запросов мгновенно
    reservations []time.Time    // Хранятся только будущие моменты доступности
}

// Пример: 100 RPS с burst 10
limiter := &Limiter{
    interval:     10 * time.Millisecond,  // 100 RPS
    burst:        10,
    reservations: make([]time.Time, 0, 10),
}
\`\`\`

**Практические преимущества:**

**Реальные применения:**
- **API Gateway:** Rate limit по API ключу (1000 req/min, burst 50)
- **Database Throttling:** Ограничение запросов (100 queries/sec, burst 10)
- **Внешние API:** Соблюдение лимитов (Twitter: 300 req/15min)
- **Background Jobs:** Throttling отправки email (10 emails/sec, burst 5)

Эта нативная реализация дает полный контроль и глубокое понимание внутренней работы rate limiting.`
		},
		uz: {
			title: `So'rovlar limiter strukturasi`,
			solutionCode: `package ratelimit

import (
	"sync"
	"time"
)

// Limiter tashqi bog'liqliklarsiz token bucket rate limiting ni amalga oshiradi
type Limiter struct {
	mu           sync.Mutex    // reservations ga parallel kirishni himoya qiladi
	interval     time.Duration // Tokenlar orasidagi davomiylik (masalan time.Second / rps)
	burst        int           // Mavjud maksimal tokenlar (bucket sig'imi)
	reservations []time.Time   // Rezerv qilingan tokenlar qachon yana mavjud bo'lishini bildiruvchi timestamp lar
}`,
			description: `Tashqi kutubxonalardan foydalanmasdan rate limiting konfiguratsiyasi va holatini saqlaydigan **Limiter** strukturasini loyihalang.

**Talablar:**
1. Konfiguratsiya va holat uchun maydonlarga ega \`Limiter\` strukturasini yarating
2. Tokenlar orasidagi intervalni saqlang (\`time.Duration\`)
3. Burst hajmini saqlang (\`int\`)
4. Token rezervatsiyalarini \`time.Time\` slice sifatida kuzating
5. Bir vaqtda kirishni himoya qilish uchun \`sync.Mutex\` dan foydalaning

**Misol:**
\`\`\`go
type Limiter struct {
    mu           sync.Mutex
    interval     time.Duration  // Tokenlar orasidagi vaqt
    burst        int            // Maksimal tokenlar
    reservations []time.Time    // Tokenlar qachon mavjud bo'ladi
}
\`\`\`

**Cheklovlar:**
- Thread xavfsizligi uchun mutex ishlatish kerak
- Rezervatsiyalarni vaqt momentlari sifatida kuzatish kerak
- Tashqi rate limiting kutubxonalarsiz`,
			hint1: `interval ni time.Duration (RPS dan hisoblanadi), burst ni int, va reservations ni []time.Time sifatida saqlang.`,
			hint2: `Bir nechta goroutine kirganda reservations slice ni race condition lardan himoya qilish uchun sync.Mutex dan foydalaning.`,
			whyItMatters: `Rate limiter strukturasini tushunish production-grade API throttling amalga oshirish uchun asosiy hisoblanadi.

**Nima uchun bunday dizayn:**
- **Token Bucket Algoritmi:** Tokenlar mavjudligini timestamp lar orqali kuzatish
- **Tashqi bog'liqliksiz:** golang.org/x/time/rate siz nativ amalga oshirish
- **Thread-Safe:** Mutex bir vaqtdagi so'rovlarda holatni himoya qiladi
- **Xotira samaradorligi:** Faqat faol rezervatsiyalar saqlanadi

**Ishlab chiqarish patterni:**
\`\`\`go
// Token bucket: interval to'ldirish tezligini boshqaradi, burst sig'imni boshqaradi
type Limiter struct {
    mu           sync.Mutex
    interval     time.Duration  // time.Second / 100 = 100 RPS da token uchun 10ms
    burst        int            // 10 token = 10 so'rovning burst ini darhol qayta ishlash
    reservations []time.Time    // Faqat kelajakdagi mavjudlik vaqtlarini saqlash
}

// Misol: 100 RPS burst 10 bilan
limiter := &Limiter{
    interval:     10 * time.Millisecond,  // 100 RPS
    burst:        10,
    reservations: make([]time.Time, 0, 10),
}
\`\`\`

**Amaliy foydalari:**

**Haqiqiy qo'llanmalar:**
- **API Gateway:** API kaliti bo'yicha rate limit (1000 req/min, burst 50)
- **Database Throttling:** So'rovlar chastotasini cheklash (100 queries/sec, burst 10)
- **Tashqi API:** Uchinchi tomon limitlariga rioya qilish (Twitter: 300 req/15min)
- **Background Jobs:** Email jo'natishni throttling (10 emails/sec, burst 5)

**Asosiy tushunchalar:**
- **Interval:** \`time.Second / rps\` tokenlar orasidagi vaqtni beradi
- **Burst:** Darhol bajarilishi mumkin bo'lgan maksimal so'rovlar
- **Reservations:** Har bir token qachon yana mavjud bo'lishini kuzatish
- **Mutex:** Yuqori bir vaqtdalik senarilarida goroutine xavfsizligi uchun muhim

Bu nativ amalga oshirish to'liq nazoratni va rate limiting ichki ishlashini chuqur tushunishni beradi.`
		}
	}
};

export default task;
