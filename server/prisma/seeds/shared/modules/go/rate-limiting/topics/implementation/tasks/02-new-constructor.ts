import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-ratelimit-new-constructor',
	title: 'New Constructor',
	difficulty: 'easy',	tags: ['go', 'rate-limiting', 'constructor', 'validation'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement a **New** constructor that creates and configures a native rate limiter with proper validation.

**Requirements:**
1. Create function \`New(rps float64, burst int) *Limiter\`
2. Normalize negative/zero \`rps\` to 1.0 (prevent division by zero)
3. Normalize negative/zero \`burst\` to 1 (minimum capacity)
4. Calculate \`interval\` as \`time.Second / rps\`
5. Initialize \`reservations\` slice with zero capacity
6. Handle edge cases: extremely high RPS causing zero interval

**Example:**
\`\`\`go
// 100 RPS with burst of 10
limiter := New(100, 10)
// interval: 10ms, burst: 10, reservations: []

// Invalid input normalization
limiter := New(-5, 0)
// Normalized to: interval: 1s, burst: 1
\`\`\`

**Constraints:**
- Must validate and normalize all inputs
- Must prevent division by zero
- Must handle extremely high RPS (>= 1 billion)`,
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

// TODO: Implement New constructor with validation
func New(rps float64, burst int) *Limiter {
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

func New(rps float64, burst int) *Limiter {
	if burst <= 0 {                                          // Validate burst: minimum 1 token capacity
		burst = 1
	}
	if rps <= 0 {                                            // Validate RPS: prevent division by zero
		rps = 1
	}
	perToken := time.Duration(float64(time.Second) / rps)    // Calculate interval between tokens
	if perToken <= 0 {                                       // Handle extreme RPS (>= 1 billion): prevent zero interval
		perToken = time.Nanosecond
	}
	return &Limiter{
		interval:     perToken,                              // Store calculated interval
		burst:        burst,                                 // Store validated burst capacity
		reservations: make([]time.Time, 0),                  // Initialize empty reservations slice
	}
}`,
			hint1: `Validate inputs first: normalize burst to 1 if <= 0, normalize rps to 1.0 if <= 0.`,
			hint2: `Calculate interval as time.Duration(float64(time.Second) / rps), then check if result is <= 0 (extreme RPS).`,
			testCode: `package ratelimit

import (
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Test basic constructor
	l := New(100, 10)
	if l == nil {
		t.Error("New(100, 10) returned nil")
	}
}

func Test2(t *testing.T) {
	// Test burst is stored correctly
	l := New(100, 10)
	if l.burst != 10 {
		t.Errorf("burst = %d, want 10", l.burst)
	}
}

func Test3(t *testing.T) {
	// Test interval is calculated correctly
	l := New(100, 10)
	expected := 10 * time.Millisecond
	if l.interval != expected {
		t.Errorf("interval = %v, want %v", l.interval, expected)
	}
}

func Test4(t *testing.T) {
	// Test negative burst is normalized to 1
	l := New(100, -5)
	if l.burst != 1 {
		t.Errorf("burst with negative input = %d, want 1", l.burst)
	}
}

func Test5(t *testing.T) {
	// Test zero burst is normalized to 1
	l := New(100, 0)
	if l.burst != 1 {
		t.Errorf("burst with zero input = %d, want 1", l.burst)
	}
}

func Test6(t *testing.T) {
	// Test negative RPS is normalized
	l := New(-5, 10)
	if l.interval != time.Second {
		t.Errorf("interval with negative RPS = %v, want 1s", l.interval)
	}
}

func Test7(t *testing.T) {
	// Test zero RPS is normalized
	l := New(0, 10)
	if l.interval != time.Second {
		t.Errorf("interval with zero RPS = %v, want 1s", l.interval)
	}
}

func Test8(t *testing.T) {
	// Test reservations is initialized empty
	l := New(100, 10)
	if len(l.reservations) != 0 {
		t.Errorf("reservations len = %d, want 0", len(l.reservations))
	}
}

func Test9(t *testing.T) {
	// Test 1 RPS gives 1 second interval
	l := New(1, 1)
	if l.interval != time.Second {
		t.Errorf("interval at 1 RPS = %v, want 1s", l.interval)
	}
}

func Test10(t *testing.T) {
	// Test extreme RPS doesn't cause zero interval
	l := New(1e10, 1)
	if l.interval <= 0 {
		t.Errorf("interval at extreme RPS = %v, should be > 0", l.interval)
	}
}`,
			whyItMatters: `Proper constructor validation prevents runtime panics and ensures predictable rate limiting behavior.

**Why Input Validation Matters:**
- **Division by Zero:** \`time.Second / 0\` causes panic
- **Negative Values:** Negative RPS/burst makes no sense, must normalize
- **Extreme Values:** RPS >= 1 billion causes integer overflow in duration calculation
- **Predictable Behavior:** Always return valid limiter, never panic

**Production Pattern:**
\`\`\`go
// API rate limiting: 1000 requests per minute, burst 50
limiter := New(1000.0/60.0, 50)  // ~16.67 RPS
// interval: 60ms per token, burst: 50

// Database query throttling: 100 queries per second, burst 10
limiter := New(100, 10)
// interval: 10ms per token, burst: 10

// External API: Twitter rate limit (300 requests per 15 minutes)
limiter := New(300.0/(15*60), 20)  // 0.33 RPS, burst 20
// interval: 3000ms per token, burst: 20

// Extreme case handling
limiter := New(1_000_000_000, 1)  // 1 billion RPS
// interval: 1ns (minimum), burst: 1
// Prevents overflow while maintaining functionality
\`\`\`

**Real-World Applications:**
- **API Gateway:** Rate limit per user/API key with configurable limits
- **Background Workers:** Throttle job processing to avoid overwhelming external services
- **Retry Logic:** Control retry rate after failures (e.g., 5 retries per minute)
- **Cache Warming:** Limit cache refresh rate (100 keys per second)

**Edge Cases Handled:**
\`\`\`go
New(-10, -5)        // Normalized to: rps=1, burst=1
New(0, 0)           // Normalized to: rps=1, burst=1
New(1e9, 100)       // interval=1ns (prevents overflow)
New(0.001, 1)       // 1 request per 1000 seconds (valid)
\`\`\`

**Common Configurations:**
- **Strict:** \`New(10, 1)\` - 10 RPS, no burst tolerance
- **Flexible:** \`New(10, 10)\` - 10 RPS, handle bursts of 10
- **Bursty:** \`New(10, 100)\` - 10 RPS, very tolerant to bursts

**Key Formula:**
- **interval = time.Second / rps**
  - 100 RPS → 10ms interval
  - 10 RPS → 100ms interval
  - 1 RPS → 1s interval

Without proper validation, invalid inputs cause production crashes. This constructor ensures robustness.`,	order: 1,
	translations: {
		ru: {
			title: 'Создание лимитера с параметрами',
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

func New(rps float64, burst int) *Limiter {
	if burst <= 0 {                                          // Валидация burst: минимум 1 токен
		burst = 1
	}
	if rps <= 0 {                                            // Валидация RPS: предотвращение деления на ноль
		rps = 1
	}
	perToken := time.Duration(float64(time.Second) / rps)    // Вычисляем интервал между токенами
	if perToken <= 0 {                                       // Обработка экстремального RPS (>= 1 миллиард): предотвращение нулевого интервала
		perToken = time.Nanosecond
	}
	return &Limiter{
		interval:     perToken,                              // Сохраняем вычисленный интервал
		burst:        burst,                                 // Сохраняем валидированную ёмкость burst
		reservations: make([]time.Time, 0),                  // Инициализируем пустой срез reservations
	}
}`,
			description: `Реализуйте конструктор **New**, который создаёт и настраивает нативный rate limiter с корректной валидацией.

**Требования:**
1. Создайте функцию \`New(rps float64, burst int) *Limiter\`
2. Нормализуйте отрицательные/нулевые \`rps\` к 1.0 (предотвращение деления на ноль)
3. Нормализуйте отрицательные/нулевые \`burst\` к 1 (минимальная ёмкость)
4. Вычислите \`interval\` как \`time.Second / rps\`
5. Инициализируйте срез \`reservations\` с нулевой ёмкостью
6. Обработайте крайние случаи: очень высокий RPS, вызывающий нулевой интервал

**Пример:**
\`\`\`go
// 100 RPS с burst 10
limiter := New(100, 10)
// interval: 10ms, burst: 10, reservations: []

// Нормализация невалидных входных данных
limiter := New(-5, 0)
// Нормализовано к: interval: 1s, burst: 1
\`\`\`

**Ограничения:**
- Должен валидировать и нормализовать все входные данные
- Должен предотвращать деление на ноль
- Должен обрабатывать очень высокий RPS (>= 1 миллиард)`,
			hint1: `Сначала валидируйте входные данные: нормализуйте burst к 1 если <= 0, нормализуйте rps к 1.0 если <= 0.`,
			hint2: `Вычислите interval как time.Duration(float64(time.Second) / rps), затем проверьте результат <= 0 (экстремальный RPS).`,
			whyItMatters: `Правильная валидация конструктора предотвращает runtime паники и обеспечивает предсказуемое поведение rate limiting.

**Почему важна валидация входных данных:**
- **Деление на ноль:** \`time.Second / 0\` вызывает панику
- **Отрицательные значения:** Отрицательные RPS/burst бессмысленны, нужна нормализация
- **Экстремальные значения:** RPS >= 1 миллиард вызывает integer overflow
- **Предсказуемое поведение:** Всегда возвращать валидный limiter, никогда не паниковать

**Реальные применения:**
- **API Gateway:** Rate limit по пользователю/API ключу
- **Background Workers:** Throttling обработки задач
- **Retry Logic:** Контроль частоты повторов после ошибок
- **Cache Warming:** Ограничение частоты обновления кэша

**Продакшен паттерн:**
\`\`\`go
// API rate limiting: 1000 запросов в минуту, burst 50
limiter := New(1000.0/60.0, 50)  // ~16.67 RPS
// interval: 60ms на токен, burst: 50

// Database query throttling: 100 запросов в секунду, burst 10
limiter := New(100, 10)
// interval: 10ms на токен, burst: 10

// External API: Twitter rate limit (300 запросов за 15 минут)
limiter := New(300.0/(15*60), 20)  // 0.33 RPS, burst 20
// interval: 3000ms на токен, burst: 20

// Экстремальный случай
limiter := New(1_000_000_000, 1)  // 1 миллиард RPS
// interval: 1ns (минимум), burst: 1
// Предотвращает overflow сохраняя функциональность
\`\`\`

**Практические преимущества:**

**Типичные конфигурации:**
- **Строгий:** \`New(10, 1)\` - 10 RPS, без burst tolerance
- **Гибкий:** \`New(10, 10)\` - 10 RPS, обработка burst из 10
- **Bursty:** \`New(10, 100)\` - 10 RPS, очень толерантен к burst

Без правильной валидации некорректные входные данные вызывают краши в продакшене. Этот конструктор обеспечивает надежность.`
		},
		uz: {
			title: `Parametrlar bilan limiter yaratish`,
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

func New(rps float64, burst int) *Limiter {
	if burst <= 0 {                                          // Burst validatsiyasi: minimal 1 token sig'imi
		burst = 1
	}
	if rps <= 0 {                                            // RPS validatsiyasi: nolga bo'lishni oldini olish
		rps = 1
	}
	perToken := time.Duration(float64(time.Second) / rps)    // Tokenlar orasidagi intervalni hisoblaymiz
	if perToken <= 0 {                                       // Ekstremal RPS (>= 1 milliard) qayta ishlash: nol intervalni oldini olish
		perToken = time.Nanosecond
	}
	return &Limiter{
		interval:     perToken,                              // Hisoblangan intervalni saqlaymiz
		burst:        burst,                                 // Validatsiya qilingan burst sig'imini saqlaymiz
		reservations: make([]time.Time, 0),                  // Bo'sh reservations slice ni ishga tushiramiz
	}
}`,
			description: `To'g'ri validatsiya bilan nativ rate limiter yaratadigan va sozlaydigan **New** konstruktorni amalga oshiring.

**Talablar:**
1. \`New(rps float64, burst int) *Limiter\` funksiyasini yarating
2. Manfiy/nol \`rps\` ni 1.0 ga normalizatsiya qiling (nolga bo'lishni oldini olish)
3. Manfiy/nol \`burst\` ni 1 ga normalizatsiya qiling (minimal sig'im)
4. \`interval\` ni \`time.Second / rps\` sifatida hisoblang
5. \`reservations\` slice ni nol sig'im bilan ishga tushiring
6. Chekka holatlarni qayta ishlang: juda yuqori RPS nol intervalni keltirib chiqaradi

**Misol:**
\`\`\`go
// 100 RPS burst 10 bilan
limiter := New(100, 10)
// interval: 10ms, burst: 10, reservations: []

// Noto'g'ri kiritishni normalizatsiya qilish
limiter := New(-5, 0)
// Normalizatsiya qilingan: interval: 1s, burst: 1
\`\`\`

**Cheklovlar:**
- Barcha kiritishlarni validatsiya qilish va normalizatsiya qilish kerak
- Nolga bo'lishni oldini olish kerak
- Juda yuqori RPS ni qayta ishlash kerak (>= 1 milliard)`,
			hint1: `Avval kiritishlarni validatsiya qiling: burst ni 1 ga normalizatsiya qiling agar <= 0, rps ni 1.0 ga normalizatsiya qiling agar <= 0.`,
			hint2: `interval ni time.Duration(float64(time.Second) / rps) sifatida hisoblang, keyin natija <= 0 (ekstremal RPS) ekanligini tekshiring.`,
			whyItMatters: `To'g'ri konstruktor validatsiyasi runtime panikalarni oldini oladi va bashorat qilinadigan rate limiting xatti-harakatini ta'minlaydi.

**Nima uchun kiritish validatsiyasi muhim:**
- **Nolga bo'lish:** \`time.Second / 0\` panikaga sabab bo'ladi
- **Manfiy qiymatlar:** Manfiy RPS/burst ma'nosiz, normalizatsiya kerak
- **Ekstremal qiymatlar:** RPS >= 1 milliard integer overflow ga sabab bo'ladi
- **Bashorat qilinadigan xatti-harakat:** Har doim yaroqli limiter qaytaring, hech qachon panik qilmang

**Haqiqiy qo'llanmalar:**
- **API Gateway:** Foydalanuvchi/API kaliti bo'yicha rate limit
- **Background Workers:** Vazifalarni qayta ishlashni throttling
- **Retry Logic:** Xatolardan keyin qayta urinishlar chastotasini nazorat qilish
- **Cache Warming:** Kesh yangilash chastotasini cheklash

**Ishlab chiqarish patterni:**
\`\`\`go
// API rate limiting: daqiqada 1000 so'rov, burst 50
limiter := New(1000.0/60.0, 50)  // ~16.67 RPS
// interval: token uchun 60ms, burst: 50

// Database query throttling: soniyada 100 so'rov, burst 10
limiter := New(100, 10)
// interval: token uchun 10ms, burst: 10

// External API: Twitter rate limit (15 daqiqada 300 so'rov)
limiter := New(300.0/(15*60), 20)  // 0.33 RPS, burst 20
// interval: token uchun 3000ms, burst: 20

// Ekstremal holat
limiter := New(1_000_000_000, 1)  // 1 milliard RPS
// interval: 1ns (minimal), burst: 1
// Funksionallikni saqlab overflow ni oldini oladi
\`\`\`

**Amaliy foydalari:**

**Tipik konfiguratsiyalar:**
- **Qat'iy:** \`New(10, 1)\` - 10 RPS, burst tolerance yo'q
- **Moslashuvchan:** \`New(10, 10)\` - 10 RPS, 10 burst ni qayta ishlash
- **Bursty:** \`New(10, 100)\` - 10 RPS, burst larga juda tolerant

To'g'ri validatsiyasiz noto'g'ri kiritishlar ishlab chiqarishda buzilishlarga olib keladi. Bu konstruktor ishonchlilikni ta'minlaydi.`
		}
	}
};

export default task;
