import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-ratelimit-allow-method',
	title: 'Allow Method',
	difficulty: 'medium',	tags: ['go', 'rate-limiting', 'context', 'timer'],
	estimatedTime: '40m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement the **Allow** method that waits for a token until the context is cancelled or a token becomes available.

**Requirements:**
1. Create method \`Allow(ctx context.Context) error\`
2. Handle nil limiter (return nil - no rate limiting)
3. Handle nil context (use context.Background())
4. Loop until token available:
   4.1. Call \`reserve(time.Now())\` to attempt token acquisition
   4.2. If \`ok == true\`, return nil (token acquired)
   4.3. If \`wait > 0\`, wait for duration using timer
   4.4. Check context cancellation during wait
5. Implement helper \`waitOnContext(ctx, duration)\`:
   5.1. If duration <= 0, check context immediately
   5.2. Create timer for duration
   5.3. Select between timer and context.Done()
   5.4. Always stop timer properly (prevent goroutine leak)

**Example:**
\`\`\`go
limiter := New(10, 1)  // 10 RPS, burst 1

// First request - immediate
err := limiter.Allow(ctx)  // nil - granted immediately

// Second request - waits 100ms
err = limiter.Allow(ctx)  // nil - waited then granted

// Context cancelled during wait
ctx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
err = limiter.Allow(ctx)  // context.DeadlineExceeded
\`\`\`

**Constraints:**
- Must respect context cancellation
- Must properly stop timers to prevent leaks
- Must handle nil limiter and nil context`,
	initialCode: `package ratelimit

import (
	"context"
	"time"
)

type Limiter struct {
	mu           sync.Mutex
	interval     time.Duration
	burst        int
	reservations []time.Time
}

// TODO: Implement Allow method
func (l *Limiter) Allow(ctx context.Context) error {
	return nil // TODO: Implement
}

// TODO: Implement waitOnContext helper
func waitOnContext($2) error {
	return nil // TODO: Implement
}`,
	solutionCode: `package ratelimit

import (
	"context"
	"time"
)

type Limiter struct {
	mu           sync.Mutex
	interval     time.Duration
	burst        int
	reservations []time.Time
}

func (l *Limiter) Allow(ctx context.Context) error {
	if l == nil {                                        // Handle nil limiter: no rate limiting
		return nil
	}
	if ctx == nil {                                      // Handle nil context: use background context
		ctx = context.Background()
	}
	for {                                                // Loop until token acquired or context cancelled
		wait, ok := l.reserve(time.Now())                // Attempt to reserve a token
		if ok {                                          // Token acquired successfully
			return nil
		}
		if wait <= 0 {                                   // No wait needed, retry immediately
			continue
		}
		if err := waitOnContext(ctx, wait); err != nil { // Wait for token or context cancellation
			return err                                   // Context cancelled during wait
		}
	}
}

func waitOnContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {                                          // No wait needed, check context immediately
		select {
		case <-ctx.Done():                               // Context already cancelled
			return ctx.Err()
		default:                                         // Context still active
			return nil
		}
	}
	timer := time.NewTimer(d)                            // Create timer for wait duration
	defer timer.Stop()                                   // Always stop timer to prevent goroutine leak
	select {
	case <-timer.C:                                      // Wait completed successfully
		return nil
	case <-ctx.Done():                                   // Context cancelled during wait
		if !timer.Stop() {                               // Try to stop timer
			select {
			case <-timer.C:                              // Drain timer channel if already fired
			default:                                     // Timer not fired, already stopped
			}
		}
	}
	return ctx.Err()                                     // Return context error (DeadlineExceeded or Canceled)
}`,
			hint1: `In Allow loop: call reserve(), if ok return nil, if wait > 0 call waitOnContext() and check error.`,
			hint2: `In waitOnContext: create time.NewTimer(d), use select with timer.C and ctx.Done(), always defer timer.Stop().`,
			testCode: `package ratelimit

import (
	"context"
	"errors"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Test Allow returns nil immediately when token available
	l := New(100, 10)
	ctx := context.Background()
	err := l.Allow(ctx)
	if err != nil {
		t.Errorf("Allow = %v, want nil", err)
	}
}

func Test2(t *testing.T) {
	// Test Allow waits when no tokens
	l := New(100, 1)
	ctx := context.Background()
	l.Allow(ctx)
	start := time.Now()
	l.Allow(ctx)
	elapsed := time.Since(start)
	if elapsed < 5*time.Millisecond {
		t.Errorf("Allow should wait, elapsed = %v", elapsed)
	}
}

func Test3(t *testing.T) {
	// Test Allow respects context cancellation
	l := New(1, 1)
	l.Allow(context.Background())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := l.Allow(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Allow = %v, want context.Canceled", err)
	}
}

func Test4(t *testing.T) {
	// Test Allow respects context timeout
	l := New(1, 1)
	l.Allow(context.Background())
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	err := l.Allow(ctx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Allow = %v, want context.DeadlineExceeded", err)
	}
}

func Test5(t *testing.T) {
	// Test waitOnContext returns nil when d=0
	ctx := context.Background()
	err := waitOnContext(ctx, 0)
	if err != nil {
		t.Errorf("waitOnContext(0) = %v, want nil", err)
	}
}

func Test6(t *testing.T) {
	// Test waitOnContext waits for duration
	ctx := context.Background()
	start := time.Now()
	err := waitOnContext(ctx, 10*time.Millisecond)
	elapsed := time.Since(start)
	if err != nil {
		t.Errorf("waitOnContext = %v, want nil", err)
	}
	if elapsed < 8*time.Millisecond {
		t.Errorf("waited only %v, want ~10ms", elapsed)
	}
}

func Test7(t *testing.T) {
	// Test waitOnContext returns on context cancel
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(5 * time.Millisecond)
		cancel()
	}()
	err := waitOnContext(ctx, time.Second)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("waitOnContext = %v, want context.Canceled", err)
	}
}

func Test8(t *testing.T) {
	// Test Allow with burst consumption
	l := New(100, 3)
	ctx := context.Background()
	for i := 0; i < 3; i++ {
		err := l.Allow(ctx)
		if err != nil {
			t.Errorf("Allow %d = %v, want nil", i, err)
		}
	}
}

func Test9(t *testing.T) {
	// Test waitOnContext with negative duration
	ctx := context.Background()
	err := waitOnContext(ctx, -time.Millisecond)
	if err != nil {
		t.Errorf("waitOnContext(-1ms) = %v, want nil", err)
	}
}

func Test10(t *testing.T) {
	// Test Allow with cancelled context returns immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	l := New(100, 10)
	l.Allow(context.Background())
	start := time.Now()
	l.Allow(ctx)
	elapsed := time.Since(start)
	if elapsed > 10*time.Millisecond {
		t.Errorf("should return immediately, elapsed = %v", elapsed)
	}
}`,
			whyItMatters: `Allow method enables blocking rate limiting with graceful cancellation, essential for production services.

**Why Context-Aware Rate Limiting:**
- **Graceful Shutdown:** Context cancellation allows clean service termination
- **Request Timeouts:** HTTP request timeouts propagate through rate limiter
- **Resource Management:** Prevents goroutine leaks from abandoned timers
- **User Experience:** Fail fast on cancelled requests instead of waiting unnecessarily

**Real-World Pattern:**

**HTTP Server with Rate Limiting:**
\`\`\`go
limiter := New(100, 10)  // 100 RPS, burst 10

func handler(w http.ResponseWriter, r *http.Request) {
    // Use request context for automatic cancellation
    if err := limiter.Allow(r.Context()); err != nil {
        if errors.Is(err, context.DeadlineExceeded) {
            http.Error(w, "Request timeout", 408)
        } else {
            http.Error(w, "Request cancelled", 499)
        }
        return
    }
    // Process request...
}
\`\`\`

**Background Worker with Timeout:**
\`\`\`go
limiter := New(10, 2)

func processJobs(ctx context.Context, jobs []Job) error {
    for _, job := range jobs {
        // Each job respects global context cancellation
        if err := limiter.Allow(ctx); err != nil {
            return fmt.Errorf("rate limiter: %w", err)
        }

        if err := processJob(ctx, job); err != nil {
            return err
        }
    }
    return nil
}

// Usage with timeout
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
defer cancel()
processJobs(ctx, jobs)
\`\`\`

**External API Client:**
\`\`\`go
type APIClient struct {
    limiter *Limiter
    client  *http.Client
}

func NewAPIClient() *APIClient {
    return &APIClient{
        limiter: New(50, 5),  // Twitter API: 50 req/sec, burst 5
        client:  &http.Client{Timeout: 10 * time.Second},
    }
}

func (c *APIClient) GetTweets(ctx context.Context, query string) ([]Tweet, error) {
    // Rate limit before API call
    if err := c.limiter.Allow(ctx); err != nil {
        return nil, fmt.Errorf("rate limit: %w", err)
    }

    req, _ := http.NewRequestWithContext(ctx, "GET", "/tweets", nil)
    resp, err := c.client.Do(req)
    // ... handle response
}
\`\`\`

**Graceful Shutdown:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

// Signal handling for graceful shutdown
sigChan := make(chan os.Signal, 1)
signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

go func() {
    <-sigChan
    cancel()  // Cancel all in-flight operations
}()

// All rate-limited operations respect cancellation
for _, task := range tasks {
    if err := limiter.Allow(ctx); err != nil {
        log.Printf("Shutting down: %v", err)
        break
    }
    processTask(task)
}
\`\`\`

**Timer Management Best Practices:**
\`\`\`go
// CORRECT: Always stop timer and drain channel
timer := time.NewTimer(duration)
defer timer.Stop()
select {
case <-timer.C:
    return nil
case <-ctx.Done():
    if !timer.Stop() {        // Timer already fired
        <-timer.C             // Drain channel to prevent goroutine leak
    }
    return ctx.Err()
}

// WRONG: Goroutine leak if context cancelled
timer := time.NewTimer(duration)
select {
case <-timer.C:
    return nil
case <-ctx.Done():
    // Timer goroutine leaks! Never stopped.
    return ctx.Err()
}
\`\`\`

**Key Concepts:**
- **Context Propagation:** Pass request context through all rate-limited calls
- **Error Handling:** Distinguish DeadlineExceeded vs Canceled for proper retry logic
- **Timer Lifecycle:** Always stop timers to prevent goroutine leaks
- **Nil Safety:** Handle nil limiter (no limiting) and nil context (use Background)

**Performance:**
- **Blocking:** Allow blocks goroutine until token available or context cancelled
- **Fair:** FIFO order for waiting requests
- **Efficient:** Only creates timer when wait needed

**Error Types:**
\`\`\`go
err := limiter.Allow(ctx)
switch {
case err == nil:
    // Token acquired, proceed
case errors.Is(err, context.DeadlineExceeded):
    // Timeout exceeded, fail request
case errors.Is(err, context.Canceled):
    // Request cancelled by client, abort
}
\`\`\`

Context-aware rate limiting is critical for building resilient, user-friendly services that handle cancellation gracefully.`,	order: 3,
	translations: {
		ru: {
			title: 'Проверка доступности запроса',
			solutionCode: `package ratelimit

import (
	"context"
	"time"
)

type Limiter struct {
	mu           sync.Mutex
	interval     time.Duration
	burst        int
	reservations []time.Time
}

func (l *Limiter) Allow(ctx context.Context) error {
	if l == nil {                                        // Обработка nil limiter: нет rate limiting
		return nil
	}
	if ctx == nil {                                      // Обработка nil context: используем background context
		ctx = context.Background()
	}
	for {                                                // Цикл пока токен не получен или контекст не отменён
		wait, ok := l.reserve(time.Now())                // Пытаемся зарезервировать токен
		if ok {                                          // Токен успешно получен
			return nil
		}
		if wait <= 0 {                                   // Ожидание не нужно, retry немедленно
			continue
		}
		if err := waitOnContext(ctx, wait); err != nil { // Ждём токен или отмену контекста
			return err                                   // Контекст отменён во время ожидания
		}
	}
}

func waitOnContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {                                          // Ожидание не нужно, проверяем контекст немедленно
		select {
		case <-ctx.Done():                               // Контекст уже отменён
			return ctx.Err()
		default:                                         // Контекст ещё активен
			return nil
		}
	}
	timer := time.NewTimer(d)                            // Создаём таймер на время ожидания
	defer timer.Stop()                                   // Всегда останавливаем таймер для предотвращения утечки горутин
	select {
	case <-timer.C:                                      // Ожидание успешно завершено
		return nil
	case <-ctx.Done():                                   // Контекст отменён во время ожидания
		if !timer.Stop() {                               // Пытаемся остановить таймер
			select {
			case <-timer.C:                              // Дренируем канал если уже сработал
			default:                                     // Таймер не сработал, уже остановлен
			}
		}
	}
	return ctx.Err()                                     // Возвращаем ошибку контекста (DeadlineExceeded или Canceled)
}`,
			description: `Реализуйте метод **Allow**, который ждёт токена пока контекст не отменён или токен не станет доступен.

**Требования:**
1. Создайте метод \`Allow(ctx context.Context) error\`
2. Обработайте nil limiter (вернуть nil - нет rate limiting)
3. Обработайте nil context (использовать context.Background())
4. Цикл пока токен не станет доступен:
   4.1. Вызовите \`reserve(time.Now())\` для попытки получить токен
   4.2. Если \`ok == true\`, верните nil (токен получен)
   4.3. Если \`wait > 0\`, ждите через таймер
   4.4. Проверяйте отмену контекста во время ожидания
5. Реализуйте helper \`waitOnContext(ctx, duration)\`:
   5.1. Если duration <= 0, проверьте контекст немедленно
   5.2. Создайте таймер на duration
   5.3. Select между таймером и context.Done()
   5.4. Всегда корректно останавливайте таймер (предотвращение утечки горутин)

**Пример:**
\`\`\`go
limiter := New(10, 1)  // 10 RPS, burst 1

// Первый запрос - немедленно
err := limiter.Allow(ctx)  // nil - выдано сразу

// Второй запрос - ждёт 100ms
err = limiter.Allow(ctx)  // nil - дождался и получил

// Контекст отменён во время ожидания
ctx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
err = limiter.Allow(ctx)  // context.DeadlineExceeded
\`\`\`

**Ограничения:**
- Должен учитывать отмену контекста
- Должен корректно останавливать таймеры во избежание утечек
- Должен обрабатывать nil limiter и nil context`,
			hint1: `В цикле Allow: вызовите reserve(), если ok верните nil, если wait > 0 вызовите waitOnContext() и проверьте ошибку.`,
			hint2: `В waitOnContext: создайте time.NewTimer(d), используйте select с timer.C и ctx.Done(), всегда defer timer.Stop().`,
			whyItMatters: `Метод Allow обеспечивает блокирующее rate limiting с graceful отменой, необходимое для production сервисов.

**Почему Context-Aware Rate Limiting:**
- **Graceful Shutdown:** Отмена контекста позволяет чистое завершение сервиса
- **Request Timeouts:** HTTP таймауты запросов распространяются через rate limiter
- **Resource Management:** Предотвращает утечки горутин от заброшенных таймеров
- **User Experience:** Быстрый fail на отменённых запросах вместо бесполезного ожидания

**Реальные паттерны:**

**HTTP Server с Rate Limiting:**
\`\`\`go
limiter := New(100, 10)  // 100 RPS, burst 10

func handler(w http.ResponseWriter, r *http.Request) {
    // Использовать request context для автоматической отмены
    if err := limiter.Allow(r.Context()); err != nil {
        if errors.Is(err, context.DeadlineExceeded) {
            http.Error(w, "Request timeout", 408)
        } else {
            http.Error(w, "Request cancelled", 499)
        }
        return
    }
    // Обработать запрос...
}
\`\`\`

**Background Worker с таймаутом:**
\`\`\`go
limiter := New(10, 2)

func processJobs(ctx context.Context, jobs []Job) error {
    for _, job := range jobs {
        // Каждая задача уважает глобальную отмену контекста
        if err := limiter.Allow(ctx); err != nil {
            return fmt.Errorf("rate limiter: %w", err)
        }

        if err := processJob(ctx, job); err != nil {
            return err
        }
    }
    return nil
}

// Использование с таймаутом
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
defer cancel()
processJobs(ctx, jobs)
\`\`\`

**External API Client:**
\`\`\`go
type APIClient struct {
    limiter *Limiter
    client  *http.Client
}

func NewAPIClient() *APIClient {
    return &APIClient{
        limiter: New(50, 5),  // Twitter API: 50 req/sec, burst 5
        client:  &http.Client{Timeout: 10 * time.Second},
    }
}

func (c *APIClient) GetTweets(ctx context.Context, query string) ([]Tweet, error) {
    // Rate limit перед вызовом API
    if err := c.limiter.Allow(ctx); err != nil {
        return nil, fmt.Errorf("rate limit: %w", err)
    }

    req, _ := http.NewRequestWithContext(ctx, "GET", "/tweets", nil)
    resp, err := c.client.Do(req)
    // ... обработка ответа
}
\`\`\`

**Graceful Shutdown:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

// Обработка сигналов для graceful shutdown
sigChan := make(chan os.Signal, 1)
signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

go func() {
    <-sigChan
    cancel()  // Отменить все выполняющиеся операции
}()

// Все rate-limited операции уважают отмену
for _, task := range tasks {
    if err := limiter.Allow(ctx); err != nil {
        log.Printf("Shutting down: %v", err)
        break
    }
    processTask(task)
}
\`\`\`

**Управление таймерами - лучшие практики:**
\`\`\`go
// ПРАВИЛЬНО: Всегда останавливайте таймер и дренируйте канал
timer := time.NewTimer(duration)
defer timer.Stop()
select {
case <-timer.C:
    return nil
case <-ctx.Done():
    if !timer.Stop() {        // Таймер уже сработал
        <-timer.C             // Дренируйте канал для предотвращения утечки горутин
    }
    return ctx.Err()
}

// НЕПРАВИЛЬНО: Утечка горутин если контекст отменён
timer := time.NewTimer(duration)
select {
case <-timer.C:
    return nil
case <-ctx.Done():
    // Горутина таймера утечка! Никогда не остановлена.
    return ctx.Err()
}
\`\`\`

**Ключевые концепции:**
- **Распространение контекста:** Передавайте request context через все rate-limited вызовы
- **Обработка ошибок:** Различайте DeadlineExceeded и Canceled для корректной логики retry
- **Жизненный цикл таймера:** Всегда останавливайте таймеры для предотвращения утечек горутин
- **Nil Safety:** Обрабатывайте nil limiter (нет limiting) и nil context (используйте Background)

**Производительность:**
- **Блокирующий:** Allow блокирует горутину до доступности токена или отмены контекста
- **Справедливый:** FIFO порядок для ожидающих запросов
- **Эффективный:** Создаёт таймер только когда нужно ожидание

**Типы ошибок:**
\`\`\`go
err := limiter.Allow(ctx)
switch {
case err == nil:
    // Токен получен, продолжайте
case errors.Is(err, context.DeadlineExceeded):
    // Таймаут превышен, fail request
case errors.Is(err, context.Canceled):
    // Запрос отменён клиентом, abort
}
\`\`\`

Context-aware rate limiting критически важен для построения устойчивых, user-friendly сервисов с graceful обработкой отмены.`
		},
		uz: {
			title: `So'rov mavjudligini tekshirish`,
			solutionCode: `package ratelimit

import (
	"context"
	"time"
)

type Limiter struct {
	mu           sync.Mutex
	interval     time.Duration
	burst        int
	reservations []time.Time
}

func (l *Limiter) Allow(ctx context.Context) error {
	if l == nil {                                        // nil limiter ni qayta ishlash: rate limiting yo'q
		return nil
	}
	if ctx == nil {                                      // nil context ni qayta ishlash: background context ishlatamiz
		ctx = context.Background()
	}
	for {                                                // Token olinmaguncha yoki kontekst bekor qilinmaguncha sikl
		wait, ok := l.reserve(time.Now())                // Token rezervatsiya qilishga urinamiz
		if ok {                                          // Token muvaffaqiyatli olindi
			return nil
		}
		if wait <= 0 {                                   // Kutish kerak emas, darhol qayta urinamiz
			continue
		}
		if err := waitOnContext(ctx, wait); err != nil { // Token yoki kontekst bekor qilishni kutamiz
			return err                                   // Kutish paytida kontekst bekor qilindi
		}
	}
}

func waitOnContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {                                          // Kutish kerak emas, kontekstni darhol tekshiramiz
		select {
		case <-ctx.Done():                               // Kontekst allaqachon bekor qilingan
			return ctx.Err()
		default:                                         // Kontekst hali faol
			return nil
		}
	}
	timer := time.NewTimer(d)                            // Kutish davomiyligi uchun timer yaratamiz
	defer timer.Stop()                                   // Goroutine leak ni oldini olish uchun har doim timer ni to'xtatamiz
	select {
	case <-timer.C:                                      // Kutish muvaffaqiyatli tugadi
		return nil
	case <-ctx.Done():                                   // Kutish paytida kontekst bekor qilindi
		if !timer.Stop() {                               // Timer ni to'xtatishga urinamiz
			select {
			case <-timer.C:                              // Agar allaqachon ishga tushgan bo'lsa kanalni bo'shatamiz
			default:                                     // Timer ishga tushmagan, allaqachon to'xtatilgan
			}
		}
	}
	return ctx.Err()                                     // Kontekst xatosini qaytaramiz (DeadlineExceeded yoki Canceled)
}`,
			description: `Kontekst bekor qilinmaguncha yoki token mavjud bo'lguncha kutadigan **Allow** metodini amalga oshiring.

**Talablar:**
1. \`Allow(ctx context.Context) error\` metodini yarating
2. nil limiter ni qayta ishlang (nil qaytaring - rate limiting yo'q)
3. nil context ni qayta ishlang (context.Background() ishlating)
4. Token mavjud bo'lguncha sikl:
   4.1. Token olish uchun \`reserve(time.Now())\` chaqiring
   4.2. Agar \`ok == true\`, nil qaytaring (token olindi)
   4.3. Agar \`wait > 0\`, timer orqali kuting
   4.4. Kutish paytida kontekst bekor qilishni tekshiring
5. \`waitOnContext(ctx, duration)\` helper ni amalga oshiring:
   5.1. Agar duration <= 0, kontekstni darhol tekshiring
   5.2. duration uchun timer yarating
   5.3. timer va context.Done() o'rtasida select qiling
   5.4. Har doim timer ni to'g'ri to'xtating (goroutine leak ni oldini olish)

**Misol:**
\`\`\`go
limiter := New(10, 1)  // 10 RPS, burst 1

// Birinchi so'rov - darhol
err := limiter.Allow(ctx)  // nil - darhol berildi

// Ikkinchi so'rov - 100ms kutadi
err = limiter.Allow(ctx)  // nil - kutdi va oldi

// Kutish paytida kontekst bekor qilindi
ctx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
err = limiter.Allow(ctx)  // context.DeadlineExceeded
\`\`\`

**Cheklovlar:**
- Kontekst bekor qilishni hurmat qilishi kerak
- Leak larni oldini olish uchun timer larni to'g'ri to'xtatishi kerak
- nil limiter va nil context ni qayta ishlashi kerak`,
			hint1: `Allow siklida: reserve() chaqiring, agar ok nil qaytaring, agar wait > 0 waitOnContext() chaqiring va xatoni tekshiring.`,
			hint2: `waitOnContext da: time.NewTimer(d) yarating, timer.C va ctx.Done() bilan select ishlating, har doim defer timer.Stop().`,
			whyItMatters: `Allow metodi graceful bekor qilish bilan blokirovka qiluvchi rate limiting ni ta'minlaydi, bu production xizmatlar uchun muhim.

**Nima uchun Context-Aware Rate Limiting:**
- **Graceful Shutdown:** Kontekst bekor qilish toza xizmat tugatishga imkon beradi
- **Request Timeouts:** HTTP so'rov timeout lari rate limiter orqali tarqaladi
- **Resource Management:** Tashlab ketilgan timer lardan goroutine leak larni oldini oladi
- **User Experience:** Bekor qilingan so'rovlarda keraksiz kutish o'rniga tez muvaffaqiyatsizlik

**Haqiqiy patternlar:**

**HTTP Server bilan Rate Limiting:**
\`\`\`go
limiter := New(100, 10)  // 100 RPS, burst 10

func handler(w http.ResponseWriter, r *http.Request) {
    // Avtomatik bekor qilish uchun request context ishlatish
    if err := limiter.Allow(r.Context()); err != nil {
        if errors.Is(err, context.DeadlineExceeded) {
            http.Error(w, "Request timeout", 408)
        } else {
            http.Error(w, "Request cancelled", 499)
        }
        return
    }
    // So'rovni qayta ishlash...
}
\`\`\`

**Timeout bilan Background Worker:**
\`\`\`go
limiter := New(10, 2)

func processJobs(ctx context.Context, jobs []Job) error {
    for _, job := range jobs {
        // Har bir vazifa global kontekst bekor qilishni hurmat qiladi
        if err := limiter.Allow(ctx); err != nil {
            return fmt.Errorf("rate limiter: %w", err)
        }

        if err := processJob(ctx, job); err != nil {
            return err
        }
    }
    return nil
}

// Timeout bilan foydalanish
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
defer cancel()
processJobs(ctx, jobs)
\`\`\`

**External API Client:**
\`\`\`go
type APIClient struct {
    limiter *Limiter
    client  *http.Client
}

func NewAPIClient() *APIClient {
    return &APIClient{
        limiter: New(50, 5),  // Twitter API: 50 req/sec, burst 5
        client:  &http.Client{Timeout: 10 * time.Second},
    }
}

func (c *APIClient) GetTweets(ctx context.Context, query string) ([]Tweet, error) {
    // API chaqiruvidan oldin rate limit
    if err := c.limiter.Allow(ctx); err != nil {
        return nil, fmt.Errorf("rate limit: %w", err)
    }

    req, _ := http.NewRequestWithContext(ctx, "GET", "/tweets", nil)
    resp, err := c.client.Do(req)
    // ... javobni qayta ishlash
}
\`\`\`

**Graceful Shutdown:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

// Graceful shutdown uchun signallarni qayta ishlash
sigChan := make(chan os.Signal, 1)
signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

go func() {
    <-sigChan
    cancel()  // Barcha ishlab turgan operatsiyalarni bekor qilish
}()

// Barcha rate-limited operatsiyalar bekor qilishni hurmat qiladi
for _, task := range tasks {
    if err := limiter.Allow(ctx); err != nil {
        log.Printf("Shutting down: %v", err)
        break
    }
    processTask(task)
}
\`\`\`

**Timer boshqarish - eng yaxshi amaliyotlar:**
\`\`\`go
// TO'G'RI: Har doim timer ni to'xtating va kanalni bo'shating
timer := time.NewTimer(duration)
defer timer.Stop()
select {
case <-timer.C:
    return nil
case <-ctx.Done():
    if !timer.Stop() {        // Timer allaqachon ishga tushgan
        <-timer.C             // Goroutine leakni oldini olish uchun kanalni bo'shating
    }
    return ctx.Err()
}

// NOTO'G'RI: Kontekst bekor qilinsa goroutine leak
timer := time.NewTimer(duration)
select {
case <-timer.C:
    return nil
case <-ctx.Done():
    // Timer goroutine leak! Hech qachon to'xtatilmagan.
    return ctx.Err()
}
\`\`\`

**Asosiy tushunchalar:**
- **Kontekst tarqatish:** Barcha rate-limited chaqiruvlar orqali request kontekstini o'tkazing
- **Xatolarni ishlash:** To'g'ri retry mantiqi uchun DeadlineExceeded va Canceled ni farqlang
- **Timer hayot tsikli:** Goroutine oqishining oldini olish uchun har doim timerlarni to'xtating
- **Nil Safety:** nil limiter (limiting yo'q) va nil kontekstni (Background ishlating) ishlang

**Performans:**
- **Blokirovka:** Allow token mavjud yoki kontekst bekor qilinguncha goroutineni blokirovka qiladi
- **Adolatli:** Kutayotgan requestlar uchun FIFO tartibi
- **Samarali:** Faqat kutish kerak bo'lganda timer yaratadi

**Xato turlari:**
\`\`\`go
err := limiter.Allow(ctx)
switch {
case err == nil:
    // Token olindi, davom eting
case errors.Is(err, context.DeadlineExceeded):
    // Timeout oshib ketdi, requestni rad eting
case errors.Is(err, context.Canceled):
    // Request mijoz tomonidan bekor qilindi, to'xtating
}
\`\`\`

Context-aware rate limiting graceful bekor qilish bilan barqaror, foydalanuvchiga qulay xizmatlarni qurish uchun juda muhim.`
		}
	}
};

export default task;
