import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-ratelimit-reserve-method',
	title: 'Reserve Method',
	difficulty: 'medium',	tags: ['go', 'rate-limiting', 'token-bucket', 'concurrency'],
	estimatedTime: '35m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement the **reserve** method that attempts to reserve a token and returns wait time if unavailable.

**Requirements:**
1. Create private method \`reserve(now time.Time) (time.Duration, bool)\`
2. Lock mutex at start, defer unlock
3. Release expired reservations (timestamps before \`now\`)
4. If tokens available (\`len(reservations) < burst\`):
   4.1. Append \`now.Add(interval)\` to reservations
   4.2. Return \`(0, true)\` - token granted immediately
5. If no tokens available:
   5.1. Calculate wait as \`reservations[0].Sub(now)\`
   5.2. Return \`(wait, false)\` - token unavailable
6. Handle negative wait times (return \`(0, false)\`)

**Example:**
\`\`\`go
limiter := New(10, 2)  // 10 RPS, burst 2

// First request - token available
wait, ok := limiter.reserve(time.Now())
// Returns: (0, true) - granted immediately

// Second request - token available
wait, ok = limiter.reserve(time.Now())
// Returns: (0, true) - granted immediately

// Third request - no tokens available
wait, ok = limiter.reserve(time.Now())
// Returns: (100ms, false) - must wait 100ms
\`\`\`

**Constraints:**
- Must be thread-safe (use mutex)
- Must release expired tokens before checking availability
- Must not modify reservations if token unavailable`,
	initialCode: `package ratelimit

import (
	"sync"
	"time"
)

type Limiter struct {
	mu           sync.Mutex
	interval     time.Duration
	burst        int
	reservations []time.Time
}

// TODO: Implement reserve method
func (l *Limiter) reserve(now time.Time) (time.Duration, bool) {
	// TODO: Implement
}

// Helper: release expired reservations (implement this too)
func (l *Limiter) releaseExpiredLocked(now time.Time) {
	// TODO: Implement
}`,
	solutionCode: `package ratelimit

import (
	"sync"
	"time"
)

type Limiter struct {
	mu           sync.Mutex
	interval     time.Duration
	burst        int
	reservations []time.Time
}

func (l *Limiter) reserve(now time.Time) (time.Duration, bool) {
	l.mu.Lock()                                          // Acquire lock for thread-safe access
	defer l.mu.Unlock()                                  // Always release lock when function exits

	l.releaseExpiredLocked(now)                          // Remove tokens that have been released (time passed)

	if len(l.reservations) < l.burst {                   // Check if token available (under capacity)
		l.reservations = append(l.reservations, now.Add(l.interval))  // Reserve token: mark it unavailable until now+interval
		return 0, true                                   // Token granted immediately, no wait needed
	}

	wait := l.reservations[0].Sub(now)                   // Calculate wait time until earliest token becomes available
	if wait < 0 {                                        // Handle race condition: token just became available
		return 0, false                                  // No wait but retry needed
	}
	return wait, false                                   // Token unavailable, return wait duration
}

func (l *Limiter) releaseExpiredLocked(now time.Time) {
	idx := 0
	for _, r := range l.reservations {                   // Find first reservation that hasn't expired yet
		if r.After(now) {                                // If reservation time is in future, stop here
			break
		}
		idx++                                            // Count expired reservations
	}
	if idx == 0 {                                        // No expired reservations
		return
	}
	copy(l.reservations, l.reservations[idx:])           // Shift remaining reservations to start
	l.reservations = l.reservations[:len(l.reservations)-idx]  // Truncate slice to new length
}`,
			hint1: `Lock mutex, call releaseExpiredLocked(now), then check if len(reservations) < burst.`,
			hint2: `If token available: append now.Add(interval) and return (0, true). If unavailable: calculate reservations[0].Sub(now).`,
			testCode: `package ratelimit

import (
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Test reserve returns immediately available when empty
	l := New(100, 10)
	now := time.Now()
	wait, ok := l.reserve(now)
	if !ok {
		t.Error("reserve on empty limiter should return ok=true")
	}
	if wait != 0 {
		t.Errorf("wait = %v, want 0", wait)
	}
}

func Test2(t *testing.T) {
	// Test reserve consumes tokens up to burst
	l := New(100, 3)
	now := time.Now()
	for i := 0; i < 3; i++ {
		_, ok := l.reserve(now)
		if !ok {
			t.Errorf("reserve %d should be ok", i)
		}
	}
}

func Test3(t *testing.T) {
	// Test reserve returns wait time when burst exceeded
	l := New(100, 2)
	now := time.Now()
	l.reserve(now)
	l.reserve(now)
	wait, ok := l.reserve(now)
	if ok {
		t.Error("reserve beyond burst should return ok=false")
	}
	if wait <= 0 {
		t.Errorf("wait = %v, should be > 0", wait)
	}
}

func Test4(t *testing.T) {
	// Test reservations expire after interval
	l := New(100, 1)
	now := time.Now()
	l.reserve(now)
	future := now.Add(11 * time.Millisecond)
	_, ok := l.reserve(future)
	if !ok {
		t.Error("reserve after interval should be ok=true")
	}
}

func Test5(t *testing.T) {
	// Test releaseExpiredLocked removes expired reservations
	l := New(100, 5)
	now := time.Now()
	l.reserve(now)
	l.reserve(now)
	future := now.Add(20 * time.Millisecond)
	l.releaseExpiredLocked(future)
	if len(l.reservations) != 0 {
		t.Errorf("reservations len = %d after expiry, want 0", len(l.reservations))
	}
}

func Test6(t *testing.T) {
	// Test burst of 1
	l := New(10, 1)
	now := time.Now()
	_, ok := l.reserve(now)
	if !ok {
		t.Error("first reserve should be ok")
	}
	_, ok = l.reserve(now)
	if ok {
		t.Error("second reserve at same time should not be ok")
	}
}

func Test7(t *testing.T) {
	// Test wait duration is correct
	l := New(10, 1) // 100ms interval
	now := time.Now()
	l.reserve(now)
	wait, _ := l.reserve(now)
	if wait < 90*time.Millisecond || wait > 110*time.Millisecond {
		t.Errorf("wait = %v, want ~100ms", wait)
	}
}

func Test8(t *testing.T) {
	// Test consecutive reserves track time correctly
	l := New(100, 2)
	now := time.Now()
	l.reserve(now)
	l.reserve(now)
	if len(l.reservations) != 2 {
		t.Errorf("reservations len = %d, want 2", len(l.reservations))
	}
}

func Test9(t *testing.T) {
	// Test releaseExpiredLocked doesn't remove future reservations
	l := New(100, 5)
	now := time.Now()
	l.reserve(now)
	l.releaseExpiredLocked(now.Add(time.Millisecond))
	if len(l.reservations) == 0 {
		t.Error("should not remove future reservations")
	}
}

func Test10(t *testing.T) {
	// Test high RPS limiter
	l := New(1000, 5)
	now := time.Now()
	for i := 0; i < 5; i++ {
		_, ok := l.reserve(now)
		if !ok {
			t.Errorf("reserve %d at high RPS should be ok", i)
		}
	}
}`,
			whyItMatters: `The reserve method is the core of token bucket rate limiting, determining when requests can proceed.

**Why This Algorithm:**
- **Token Bucket:** Tokens refill at constant rate (interval), burst allows temporary excess
- **Time-Based:** Track when each token becomes available again (more efficient than tick-based)
- **Lock-Free Optimization:** Only locks briefly during reservation, not during wait
- **Expired Token Cleanup:** Automatically releases tokens that have "refilled"

**Token Bucket Mechanics:**
\`\`\`go
// Example: 10 RPS (100ms interval), burst 3
limiter := New(10, 3)

// Time 0ms: 3 tokens available
reserve(T0) → (0, true)       // Token 1 reserved until T0+100ms
reserve(T0) → (0, true)       // Token 2 reserved until T0+100ms
reserve(T0) → (0, true)       // Token 3 reserved until T0+100ms
reserve(T0) → (100ms, false)  // No tokens, wait 100ms

// Time 100ms: 1 token refilled (earliest expired)
reserve(T100) → (0, true)     // Token 1 available again
reserve(T100) → (100ms, false) // Still need to wait

// Time 200ms: 2 tokens refilled
reserve(T200) → (0, true)     // Tokens available
\`\`\`

**Real-World Scenarios:**

**API Gateway Rate Limiting:**
\`\`\`go
limiter := New(100, 10)  // 100 RPS, burst 10

func handleRequest(w http.ResponseWriter, r *http.Request) {
    wait, ok := limiter.reserve(time.Now())
    if !ok && wait > time.Second {
        // Too much wait, reject request
        http.Error(w, "Rate limit exceeded", 429)
        return
    }
    if !ok {
        // Short wait, sleep and retry
        time.Sleep(wait)
        limiter.reserve(time.Now())  // Retry after wait
    }
    // Process request...
}
\`\`\`

**Background Job Throttling:**
\`\`\`go
limiter := New(10, 5)  // 10 jobs/sec, burst 5

for _, job := range jobs {
    for {
        wait, ok := limiter.reserve(time.Now())
        if ok {
            processJob(job)
            break
        }
        time.Sleep(wait)  // Wait until token available
    }
}
\`\`\`

**Database Query Rate Limiting:**
\`\`\`go
limiter := New(1000, 50)  // 1000 queries/sec, burst 50

func queryDatabase(query string) (Result, error) {
    wait, ok := limiter.reserve(time.Now())
    if !ok {
        time.Sleep(wait)  // Throttle query rate
    }
    return db.Execute(query)
}
\`\`\`

**Key Concepts:**
- **Reservations:** Each entry represents when a token becomes available
- **releaseExpiredLocked:** Removes reservations where \`time >= reservation\`
- **Burst:** Maximum tokens available simultaneously (len(reservations) < burst)
- **Wait Time:** \`reservations[0].Sub(now)\` gives time until next token

**Performance:**
- **Time Complexity:** O(n) for cleanup where n = burst size
- **Space Complexity:** O(burst) - only store active reservations
- **Lock Contention:** Minimal - lock held only during reservation

**Common Pitfalls:**
- Forgetting to release expired tokens (memory leak, false rate limiting)
- Not handling negative wait times (race condition between cleanup and check)
- Modifying reservations when token unavailable (incorrect state)

This method enables precise, efficient rate limiting without external dependencies.`,	order: 2,
	translations: {
		ru: {
			title: 'Резервирование слота для запроса',
			solutionCode: `package ratelimit

import (
	"sync"
	"time"
)

type Limiter struct {
	mu           sync.Mutex
	interval     time.Duration
	burst        int
	reservations []time.Time
}

func (l *Limiter) reserve(now time.Time) (time.Duration, bool) {
	l.mu.Lock()                                          // Захватываем lock для потокобезопасного доступа
	defer l.mu.Unlock()                                  // Всегда освобождаем lock при выходе из функции

	l.releaseExpiredLocked(now)                          // Удаляем токены которые были освобождены (время прошло)

	if len(l.reservations) < l.burst {                   // Проверяем доступен ли токен (под ёмкостью)
		l.reservations = append(l.reservations, now.Add(l.interval))  // Резервируем токен: отмечаем недоступным до now+interval
		return 0, true                                   // Токен выдан немедленно, ожидание не нужно
	}

	wait := l.reservations[0].Sub(now)                   // Вычисляем время ожидания до освобождения первого токена
	if wait < 0 {                                        // Обработка race condition: токен только что стал доступен
		return 0, false                                  // Без ожидания но нужен retry
	}
	return wait, false                                   // Токен недоступен, возвращаем время ожидания
}

func (l *Limiter) releaseExpiredLocked(now time.Time) {
	idx := 0
	for _, r := range l.reservations {                   // Находим первое резервирование которое ещё не истекло
		if r.After(now) {                                // Если время резервирования в будущем, останавливаемся
			break
		}
		idx++                                            // Считаем истекшие резервирования
	}
	if idx == 0 {                                        // Нет истекших резервирований
		return
	}
	copy(l.reservations, l.reservations[idx:])           // Сдвигаем оставшиеся резервирования в начало
	l.reservations = l.reservations[:len(l.reservations)-idx]  // Обрезаем срез до новой длины
}`,
			description: `Реализуйте метод **reserve**, который пытается зарезервировать токен и возвращает время ожидания, если недоступен.

**Требования:**
1. Создайте приватный метод \`reserve(now time.Time) (time.Duration, bool)\`
2. Залокируйте mutex в начале, defer unlock
3. Освободите просроченные резервирования (timestamp до \`now\`)
4. Если токены доступны (\`len(reservations) < burst\`):
   4.1. Добавьте \`now.Add(interval)\` в reservations
   4.2. Верните \`(0, true)\` - токен выдан немедленно
5. Если нет токенов:
   5.1. Вычислите wait как \`reservations[0].Sub(now)\`
   5.2. Верните \`(wait, false)\` - токен недоступен
6. Обработайте отрицательное время ожидания (вернуть \`(0, false)\`)

**Пример:**
\`\`\`go
limiter := New(10, 2)  // 10 RPS, burst 2

// Первый запрос - токен доступен
wait, ok := limiter.reserve(time.Now())
// Возвращает: (0, true) - выдано немедленно

// Третий запрос - нет токенов
wait, ok = limiter.reserve(time.Now())
// Возвращает: (100ms, false) - нужно ждать 100ms
\`\`\`

**Ограничения:**
- Должен быть потокобезопасным (использовать mutex)
- Должен освобождать просроченные токены перед проверкой
- Не должен модифицировать reservations если токен недоступен`,
			hint1: `Залокируйте mutex, вызовите releaseExpiredLocked(now), затем проверьте len(reservations) < burst.`,
			hint2: `Если токен доступен: добавьте now.Add(interval) и верните (0, true). Если нет: вычислите reservations[0].Sub(now).`,
			whyItMatters: `Метод reserve - ядро token bucket rate limiting, определяющее когда запросы могут продолжиться.

**Почему этот алгоритм:**
- **Token Bucket:** Токены пополняются с постоянной скоростью (interval), burst позволяет временный избыток
- **Time-Based:** Отслеживание когда каждый токен снова станет доступным (эффективнее чем tick-based)
- **Lock-Free оптимизация:** Блокирует только кратко во время резервирования, не во время ожидания
- **Очистка просроченных токенов:** Автоматически освобождает токены которые "пополнились"

**Механика Token Bucket:**
\`\`\`go
// Пример: 10 RPS (100ms interval), burst 3
limiter := New(10, 3)

// Время 0ms: 3 токена доступны
reserve(T0) → (0, true)       // Токен 1 зарезервирован до T0+100ms
reserve(T0) → (0, true)       // Токен 2 зарезервирован до T0+100ms
reserve(T0) → (0, true)       // Токен 3 зарезервирован до T0+100ms
reserve(T0) → (100ms, false)  // Нет токенов, ждать 100ms

// Время 100ms: 1 токен пополнился (первый истёк)
reserve(T100) → (0, true)     // Токен 1 снова доступен
reserve(T100) → (100ms, false) // Всё ещё нужно ждать
\`\`\`

**Реальные сценарии:**

**API Gateway Rate Limiting:**
\`\`\`go
limiter := New(100, 10)  // 100 RPS, burst 10

func handleRequest(w http.ResponseWriter, r *http.Request) {
    wait, ok := limiter.reserve(time.Now())
    if !ok && wait > time.Second {
        http.Error(w, "Rate limit exceeded", 429)
        return
    }
    if !ok {
        time.Sleep(wait)
        limiter.reserve(time.Now())
    }
    // Обработать запрос...
}
\`\`\`

**Background Job Throttling:**
\`\`\`go
limiter := New(10, 5)  // 10 jobs/sec, burst 5

for _, job := range jobs {
    for {
        wait, ok := limiter.reserve(time.Now())
        if ok {
            processJob(job)
            break
        }
        time.Sleep(wait)
    }
}
\`\`\`

**Ключевые концепции:**
- **Reservations:** Каждая запись представляет когда токен станет доступным
- **releaseExpiredLocked:** Удаляет резервирования где \`time >= reservation\`
- **Burst:** Максимум токенов доступных одновременно (len(reservations) < burst)
- **Wait Time:** \`reservations[0].Sub(now)\` даёт время до следующего токена

**Производительность:**
- **Временная сложность:** O(n) для очистки где n = размер burst
- **Пространственная сложность:** O(burst) - только активные резервирования
- **Lock Contention:** Минимальная - блокировка удерживается только во время резервирования

**Частые ошибки:**
- Забыть освободить истёкшие токены (утечка памяти, ложное ограничение)
- Не обрабатывать отрицательное время ожидания (race condition)
- Изменять резервирования когда токен недоступен (некорректное состояние)

Этот метод позволяет точное, эффективное rate limiting без внешних зависимостей.`
		},
		uz: {
			title: `So'rov uchun slot zahiralash`,
			solutionCode: `package ratelimit

import (
	"sync"
	"time"
)

type Limiter struct {
	mu           sync.Mutex
	interval     time.Duration
	burst        int
	reservations []time.Time
}

func (l *Limiter) reserve(now time.Time) (time.Duration, bool) {
	l.mu.Lock()                                          // Thread-safe kirish uchun qulfni olamiz
	defer l.mu.Unlock()                                  // Funksiyadan chiqqanda har doim qulfni bo'shatamiz

	l.releaseExpiredLocked(now)                          // Bo'shatilgan tokenlarni olib tashlaymiz (vaqt o'tgan)

	if len(l.reservations) < l.burst {                   // Token mavjudligini tekshiramiz (sig'im ostida)
		l.reservations = append(l.reservations, now.Add(l.interval))  // Tokenni rezerv qilamiz: now+interval gacha mavjud emas deb belgilaymiz
		return 0, true                                   // Token darhol berildi, kutish kerak emas
	}

	wait := l.reservations[0].Sub(now)                   // Eng erta token mavjud bo'lguncha kutish vaqtini hisoblaymiz
	if wait < 0 {                                        // Race condition qayta ishlash: token hozirgina mavjud bo'ldi
		return 0, false                                  // Kutish yo'q lekin qayta urinish kerak
	}
	return wait, false                                   // Token mavjud emas, kutish davomiyligini qaytaramiz
}

func (l *Limiter) releaseExpiredLocked(now time.Time) {
	idx := 0
	for _, r := range l.reservations {                   // Hali muddati o'tmagan birinchi rezervatsiyani topamiz
		if r.After(now) {                                // Agar rezervatsiya vaqti kelajakda bo'lsa, bu yerda to'xtaymiz
			break
		}
		idx++                                            // Muddati o'tgan rezervatsiyalarni sanab chiqamiz
	}
	if idx == 0 {                                        // Muddati o'tgan rezervatsiyalar yo'q
		return
	}
	copy(l.reservations, l.reservations[idx:])           // Qolgan rezervatsiyalarni boshiga siljitamiz
	l.reservations = l.reservations[:len(l.reservations)-idx]  // Slice ni yangi uzunlikka qisqartiramiz
}`,
			description: `Token rezervatsiya qilishga urinadigan va mavjud bo'lmasa kutish vaqtini qaytaradigan **reserve** metodini amalga oshiring.

**Talablar:**
1. Xususiy metod \`reserve(now time.Time) (time.Duration, bool)\` yarating
2. Boshida mutex ni qulflang, defer unlock
3. Muddati o'tgan rezervatsiyalarni bo'shating (\`now\` dan oldingi timestamp lar)
4. Agar tokenlar mavjud bo'lsa (\`len(reservations) < burst\`):
   4.1. reservations ga \`now.Add(interval)\` qo'shing
   4.2. \`(0, true)\` qaytaring - token darhol berildi
5. Agar tokenlar mavjud bo'lmasa:
   5.1. wait ni \`reservations[0].Sub(now)\` sifatida hisoblang
   5.2. \`(wait, false)\` qaytaring - token mavjud emas
6. Manfiy kutish vaqtlarini qayta ishlang (\`(0, false)\` qaytaring)

**Misol:**
\`\`\`go
limiter := New(10, 2)  // 10 RPS, burst 2

// Birinchi so'rov - token mavjud
wait, ok := limiter.reserve(time.Now())
// Qaytaradi: (0, true) - darhol berildi

// Uchinchi so'rov - tokenlar yo'q
wait, ok = limiter.reserve(time.Now())
// Qaytaradi: (100ms, false) - 100ms kutish kerak
\`\`\`

**Cheklovlar:**
- Thread-safe bo'lishi kerak (mutex ishlatish)
- Mavjudlikni tekshirishdan oldin muddati o'tgan tokenlarni bo'shatish kerak
- Token mavjud bo'lmasa reservations ni o'zgartirmasligi kerak`,
			hint1: `Mutex ni qulflang, releaseExpiredLocked(now) ni chaqiring, keyin len(reservations) < burst tekshiring.`,
			hint2: `Agar token mavjud: now.Add(interval) qo'shing va (0, true) qaytaring. Agar mavjud emas: reservations[0].Sub(now) hisoblang.`,
			whyItMatters: `reserve metodi token bucket rate limiting ning yadrosi bo'lib, so'rovlar qachon davom etishi mumkinligini belgilaydi.

**Nima uchun bu algoritm:**
- **Token Bucket:** Tokenlar doimiy tezlikda (interval) to'ldiriladi, burst vaqtinchalik ortiqcha imkon beradi
- **Time-Based:** Har bir token qachon yana mavjud bo'lishini kuzatish (tick-based dan samaraliroq)
- **Lock-Free optimallashtirish:** Faqat rezervatsiya paytida qisqa qulflaydi, kutish paytida emas
- **Muddati o'tgan tokenlarni tozalash:** "To'ldirilgan" tokenlarni avtomatik bo'shatadi

**Token Bucket mexanikasi:**
\`\`\`go
// Misol: 10 RPS (100ms interval), burst 3
limiter := New(10, 3)

// Vaqt 0ms: 3 token mavjud
reserve(T0) → (0, true)       // Token 1 T0+100ms gacha rezervatsiya qilingan
reserve(T0) → (0, true)       // Token 2 T0+100ms gacha rezervatsiya qilingan
reserve(T0) → (0, true)       // Token 3 T0+100ms gacha rezervatsiya qilingan
reserve(T0) → (100ms, false)  // Tokenlar yo'q, 100ms kutish

// Vaqt 100ms: 1 token to'ldirildi (birinchi muddati o'tdi)
reserve(T100) → (0, true)     // Token 1 yana mavjud
reserve(T100) → (100ms, false) // Hali kutish kerak
\`\`\`

**Haqiqiy senarilar:**

**API Gateway Rate Limiting:**
\`\`\`go
limiter := New(100, 10)  // 100 RPS, burst 10

func handleRequest(w http.ResponseWriter, r *http.Request) {
    wait, ok := limiter.reserve(time.Now())
    if !ok && wait > time.Second {
        http.Error(w, "Rate limit exceeded", 429)
        return
    }
    if !ok {
        time.Sleep(wait)
        limiter.reserve(time.Now())
    }
    // So'rovni qayta ishlash...
}
\`\`\`

**Background Job Throttling:**
\`\`\`go
limiter := New(10, 5)  // 10 jobs/sec, burst 5

for _, job := range jobs {
    for {
        wait, ok := limiter.reserve(time.Now())
        if ok {
            processJob(job)
            break
        }
        time.Sleep(wait)
    }
}
\`\`\`

**Asosiy tushunchalar:**
- **Reservations:** Har bir yozuv token qachon mavjud bo'lishini ko'rsatadi
- **releaseExpiredLocked:** \`time >= reservation\` bo'lgan rezervatsiyalarni o'chiradi
- **Burst:** Bir vaqtda mavjud bo'lgan maksimal tokenlar (len(reservations) < burst)
- **Wait Time:** \`reservations[0].Sub(now)\` keyingi tokengacha vaqtni beradi

**Ishlash:**
- **Vaqt murakkabligi:** O(n) tozalash uchun, bu yerda n = burst hajmi
- **Xotira murakkabligi:** O(burst) - faqat faol rezervatsiyalar
- **Lock Contention:** Minimal - lock faqat rezervatsiya paytida ushlab turiladi

**Keng tarqalgan xatolar:**
- Muddati o'tgan tokenlarni bo'shatishni unutish (xotira oqishi, yolg'on cheklash)
- Manfiy kutish vaqtlarini qayta ishlamaslik (race condition)
- Token mavjud bo'lmaganda rezervatsiyalarni o'zgartirish (noto'g'ri holat)

Bu metod tashqi bog'liqliklarsiz aniq, samarali rate limiting ga imkon beradi.`
		}
	}
};

export default task;
