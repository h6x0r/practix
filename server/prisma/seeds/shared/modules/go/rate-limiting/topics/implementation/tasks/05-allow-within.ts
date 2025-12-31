import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-ratelimit-allow-within',
	title: 'AllowWithin Method',
	difficulty: 'medium',	tags: ['go', 'rate-limiting', 'timeout', 'context'],
	estimatedTime: '35m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **AllowWithin** method that limits maximum wait time for token acquisition.

**Requirements:**
1. Create method \`AllowWithin(ctx context.Context, maxWait time.Duration) error\`
2. Handle nil limiter (return nil - no rate limiting)
3. Handle nil context (use context.Background())
4. If \`maxWait <= 0\`:
   4.1. Try immediate token acquisition via \`tryImmediate(time.Now())\`
   4.2. Return nil if token granted
   4.3. Return \`context.DeadlineExceeded\` if no token available
5. If \`maxWait > 0\`:
   5.1. Create timeout context with \`context.WithTimeout(ctx, maxWait)\`
   5.2. Defer cancel to prevent context leak
   5.3. Call \`Allow(timeoutCtx)\` and return its result
6. Implement helper \`tryImmediate(now time.Time) bool\`:
   6.1. Lock mutex
   6.2. Release expired reservations
   6.3. If tokens available, append reservation and return true
   6.4. Otherwise return false

**Example:**
\`\`\`go
limiter := New(10, 1)  // 10 RPS, burst 1

// Immediate acquisition - no wait allowed
err := limiter.AllowWithin(ctx, 0)
// Returns: nil (token available) or DeadlineExceeded (no token)

// Wait up to 50ms for token
err = limiter.AllowWithin(ctx, 50*time.Millisecond)
// Returns: nil (got token within 50ms) or DeadlineExceeded (timeout)

// Wait up to 200ms for token
err = limiter.AllowWithin(ctx, 200*time.Millisecond)
// Returns: nil (got token within 200ms)
\`\`\`

**Constraints:**
- Must enforce maximum wait time strictly
- Must clean up context resources (defer cancel)
- Must return DeadlineExceeded when maxWait is 0 and no token available`,
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

// TODO: Implement AllowWithin method
func (l *Limiter) AllowWithin(ctx context.Context, maxWait time.Duration) error {
	return nil // TODO: Implement
}

// TODO: Implement tryImmediate helper
func (l *Limiter) tryImmediate(now time.Time) bool {
	// TODO: Implement
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

func (l *Limiter) AllowWithin(ctx context.Context, maxWait time.Duration) error {
	if l == nil {                                        // Handle nil limiter: no rate limiting
		return nil
	}
	if ctx == nil {                                      // Handle nil context: use background context
		ctx = context.Background()
	}
	if maxWait <= 0 {                                    // No wait allowed, try immediate acquisition
		if l.tryImmediate(time.Now()) {                  // Token available immediately
			return nil
		}
		return context.DeadlineExceeded                  // No token available, fail immediately
	}
	ctx, cancel := context.WithTimeout(ctx, maxWait)    // Create timeout context with maxWait duration
	defer cancel()                                       // Always cancel to release context resources
	return l.Allow(ctx)                                  // Reuse Allow with timeout context
}

func (l *Limiter) tryImmediate(now time.Time) bool {
	l.mu.Lock()                                          // Acquire lock for thread-safe access
	defer l.mu.Unlock()                                  // Always release lock when function exits
	l.releaseExpiredLocked(now)                          // Remove expired reservations
	if len(l.reservations) >= l.burst {                  // No tokens available (at capacity)
		return false
	}
	l.reservations = append(l.reservations, now.Add(l.interval))  // Reserve token immediately
	return true                                          // Token granted
}`,
			hint1: `For maxWait <= 0: call tryImmediate(time.Now()), return nil if true, else return context.DeadlineExceeded.`,
			hint2: `For maxWait > 0: create timeout context with context.WithTimeout(ctx, maxWait), defer cancel(), then call Allow().`,
			testCode: `package ratelimit

import (
	"context"
	"errors"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Test AllowWithin returns nil immediately when token available
	l := New(100, 10)
	ctx := context.Background()
	err := l.AllowWithin(ctx, time.Second)
	if err != nil {
		t.Errorf("AllowWithin = %v, want nil", err)
	}
}

func Test2(t *testing.T) {
	// Test AllowWithin with maxWait=0 (immediate mode)
	l := New(100, 1)
	ctx := context.Background()
	l.Allow(ctx)
	err := l.AllowWithin(ctx, 0)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("AllowWithin(0) = %v, want DeadlineExceeded", err)
	}
}

func Test3(t *testing.T) {
	// Test AllowWithin times out correctly
	l := New(1, 1) // 1 RPS
	ctx := context.Background()
	l.Allow(ctx)
	start := time.Now()
	err := l.AllowWithin(ctx, 50*time.Millisecond)
	elapsed := time.Since(start)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("AllowWithin = %v, want DeadlineExceeded", err)
	}
	if elapsed < 40*time.Millisecond || elapsed > 100*time.Millisecond {
		t.Errorf("elapsed = %v, want ~50ms", elapsed)
	}
}

func Test4(t *testing.T) {
	// Test tryImmediate returns true when token available
	l := New(100, 10)
	now := time.Now()
	if !l.tryImmediate(now) {
		t.Error("tryImmediate should return true when tokens available")
	}
}

func Test5(t *testing.T) {
	// Test tryImmediate returns false when burst exceeded
	l := New(100, 1)
	now := time.Now()
	l.tryImmediate(now)
	if l.tryImmediate(now) {
		t.Error("tryImmediate should return false when burst exceeded")
	}
}

func Test6(t *testing.T) {
	// Test AllowWithin succeeds when wait is within maxWait
	l := New(100, 1) // 10ms interval
	ctx := context.Background()
	l.Allow(ctx)
	err := l.AllowWithin(ctx, 50*time.Millisecond)
	if err != nil {
		t.Errorf("AllowWithin = %v, want nil", err)
	}
}

func Test7(t *testing.T) {
	// Test AllowWithin respects context cancellation
	l := New(1, 1)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := l.AllowWithin(ctx, time.Second)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("AllowWithin = %v, want context.Canceled", err)
	}
}

func Test8(t *testing.T) {
	// Test AllowWithin with negative maxWait
	l := New(100, 1)
	ctx := context.Background()
	l.Allow(ctx)
	err := l.AllowWithin(ctx, -time.Millisecond)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("AllowWithin(-1ms) = %v, want DeadlineExceeded", err)
	}
}

func Test9(t *testing.T) {
	// Test tryImmediate consumes token
	l := New(100, 2)
	now := time.Now()
	l.tryImmediate(now)
	l.tryImmediate(now)
	if l.tryImmediate(now) {
		t.Error("should be out of tokens")
	}
}

func Test10(t *testing.T) {
	// Test AllowWithin burst consumption
	l := New(100, 3)
	ctx := context.Background()
	for i := 0; i < 3; i++ {
		err := l.AllowWithin(ctx, 0)
		if err != nil {
			t.Errorf("AllowWithin %d = %v, want nil", i, err)
		}
	}
}`,
			whyItMatters: `AllowWithin provides bounded waiting for rate-limited operations, essential for maintaining SLAs and preventing indefinite blocking.

**Why Maximum Wait Time:**
- **SLA Guarantees:** "API responds within 100ms or fails fast"
- **User Experience:** Prevent users from waiting indefinitely
- **Resource Management:** Limit thread/goroutine blocking time
- **Circuit Breaker Integration:** Fail fast after threshold instead of accumulating backlog

**Real-World Patterns:**

**HTTP API with Response Time SLA:**
\`\`\`go
limiter := New(100, 10)

func handler(w http.ResponseWriter, r *http.Request) {
    // SLA: Must respond within 100ms total
    // Allow max 50ms wait for rate limiter
    if err := limiter.AllowWithin(r.Context(), 50*time.Millisecond); err != nil {
        // Failed to get token within 50ms
        http.Error(w, "Service overloaded, try again", 503)
        metrics.RecordRateLimitReject()
        return
    }

    // Process request (must complete in remaining 50ms)
    processRequest(w, r)
}
\`\`\`

**Background Job Processing:**
\`\`\`go
limiter := New(10, 2)  // 10 jobs/sec, burst 2

func processJobs(ctx context.Context, jobs []Job) {
    for _, job := range jobs {
        // Try to get token immediately, don't wait
        if err := limiter.AllowWithin(ctx, 0); err != nil {
            // No token available, queue job for later
            requeueJob(job)
            continue
        }

        // Got token immediately, process now
        if err := processJob(ctx, job); err != nil {
            log.Printf("Job failed: %v", err)
        }
    }
}
\`\`\`

**External API with Timeout:**
\`\`\`go
type APIClient struct {
    limiter *Limiter
}

func NewAPIClient() *APIClient {
    return &APIClient{
        limiter: New(50, 5),  // 50 RPS, burst 5
    }
}

func (c *APIClient) FetchData(ctx context.Context) (*Data, error) {
    // Total timeout: 5 seconds
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()

    // Wait max 1 second for rate limiter
    if err := c.limiter.AllowWithin(ctx, time.Second); err != nil {
        return nil, fmt.Errorf("rate limit timeout: %w", err)
    }

    // Make API call with remaining 4 seconds timeout
    return c.callAPI(ctx)
}
\`\`\`

**Load Shedding Pattern:**
\`\`\`go
limiter := New(1000, 100)

func handler(w http.ResponseWriter, r *http.Request) {
    // Immediate check: load shedding
    if err := limiter.AllowWithin(r.Context(), 0); err != nil {
        // Server overloaded, reject immediately
        w.Header().Set("Retry-After", "1")
        http.Error(w, "Server overloaded", 503)
        return
    }

    // Got token, process request
    processRequest(w, r)
}
\`\`\`

**Database Connection Pool:**
\`\`\`go
type DBPool struct {
    limiter *Limiter
    db      *sql.DB
}

func NewDBPool(db *sql.DB, maxQPS int) *DBPool {
    return &DBPool{
        limiter: New(float64(maxQPS), 10),
        db:      db,
    }
}

func (p *DBPool) Query(ctx context.Context, query string) (*sql.Rows, error) {
    // Database must respond within 2 seconds
    // Allow max 500ms wait for rate limiter
    if err := p.limiter.AllowWithin(ctx, 500*time.Millisecond); err != nil {
        return nil, fmt.Errorf("query rate limit timeout: %w", err)
    }

    // Execute query with remaining 1.5 seconds
    return p.db.QueryContext(ctx, query)
}
\`\`\`

**Retry Logic with Backoff:**
\`\`\`go
limiter := New(5, 1)  // 5 retries per second

func retryOperation(ctx context.Context, op func() error) error {
    maxRetries := 3
    for i := 0; i < maxRetries; i++ {
        // Try operation
        if err := op(); err == nil {
            return nil  // Success
        }

        // Backoff with rate limiting
        backoff := time.Duration(i+1) * 100 * time.Millisecond
        if err := limiter.AllowWithin(ctx, backoff); err != nil {
            // Exceeded backoff time or context cancelled
            return fmt.Errorf("retry timeout after %d attempts", i+1)
        }
    }
    return fmt.Errorf("max retries exceeded")
}
\`\`\`

**Key Concepts:**
- **Bounded Waiting:** Never wait longer than maxWait
- **Fail Fast:** Return DeadlineExceeded immediately when maxWait is 0
- **Context Composition:** Combine parent context timeout with maxWait timeout
- **Resource Cleanup:** Always defer cancel() to prevent context leaks

**Common Use Cases:**
- **maxWait = 0:** Immediate check, load shedding (reject if no token)
- **maxWait = small:** Fail fast if slightly overloaded (e.g., 50-100ms)
- **maxWait = medium:** Balance between user experience and throughput (e.g., 500ms-1s)
- **maxWait = infinite:** Use Allow() instead for unlimited waiting

**Error Handling:**
\`\`\`go
err := limiter.AllowWithin(ctx, 100*time.Millisecond)
switch {
case err == nil:
    // Token acquired within maxWait
case errors.Is(err, context.DeadlineExceeded):
    // Timeout: either maxWait exceeded or parent context timeout
    // Check parent context to distinguish
    if ctx.Err() == nil {
        // maxWait timeout (too slow)
        return ErrRateLimitTimeout
    } else {
        // Parent context timeout (request timeout)
        return ErrRequestTimeout
    }
case errors.Is(err, context.Canceled):
    // Client cancelled request
    return ErrRequestCanceled
}
\`\`\`

**Performance Considerations:**
- **Zero Allocation:** tryImmediate avoids timer creation for immediate check
- **Context Reuse:** Only creates new context when maxWait > 0
- **Early Return:** Fails fast when no token available and maxWait is 0

AllowWithin is critical for building responsive services that maintain SLAs while gracefully handling overload.`,	order: 4,
	translations: {
		ru: {
			title: 'Проверка запроса с таймаутом',
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

func (l *Limiter) AllowWithin(ctx context.Context, maxWait time.Duration) error {
	if l == nil {                                        // Обработка nil limiter: нет rate limiting
		return nil
	}
	if ctx == nil {                                      // Обработка nil context: используем background context
		ctx = context.Background()
	}
	if maxWait <= 0 {                                    // Ожидание не разрешено, пытаемся получить немедленно
		if l.tryImmediate(time.Now()) {                  // Токен доступен немедленно
			return nil
		}
		return context.DeadlineExceeded                  // Нет токена, fail немедленно
	}
	ctx, cancel := context.WithTimeout(ctx, maxWait)    // Создаём timeout context с maxWait
	defer cancel()                                       // Всегда отменяем для освобождения ресурсов контекста
	return l.Allow(ctx)                                  // Переиспользуем Allow с timeout context
}

func (l *Limiter) tryImmediate(now time.Time) bool {
	l.mu.Lock()                                          // Захватываем lock для потокобезопасного доступа
	defer l.mu.Unlock()                                  // Всегда освобождаем lock при выходе
	l.releaseExpiredLocked(now)                          // Удаляем истекшие резервирования
	if len(l.reservations) >= l.burst {                  // Нет доступных токенов (на ёмкости)
		return false
	}
	l.reservations = append(l.reservations, now.Add(l.interval))  // Резервируем токен немедленно
	return true                                          // Токен выдан
}`,
			description: `Реализуйте метод **AllowWithin**, который ограничивает максимальное время ожидания получения токена.

**Требования:**
1. Создайте метод \`AllowWithin(ctx context.Context, maxWait time.Duration) error\`
2. Обработайте nil limiter (вернуть nil - нет rate limiting)
3. Обработайте nil context (использовать context.Background())
4. Если \`maxWait <= 0\`:
   4.1. Попытайтесь получить токен немедленно через \`tryImmediate(time.Now())\`
   4.2. Верните nil если токен выдан
   4.3. Верните \`context.DeadlineExceeded\` если нет токена
5. Если \`maxWait > 0\`:
   5.1. Создайте timeout context через \`context.WithTimeout(ctx, maxWait)\`
   5.2. Defer cancel для предотвращения утечки контекста
   5.3. Вызовите \`Allow(timeoutCtx)\` и верните результат
6. Реализуйте helper \`tryImmediate(now time.Time) bool\`:
   6.1. Залокируйте mutex
   6.2. Освободите просроченные резервирования
   6.3. Если токены доступны, добавьте резервирование и верните true
   6.4. Иначе верните false

**Пример:**
\`\`\`go
limiter := New(10, 1)  // 10 RPS, burst 1

// Немедленное получение - ожидание не разрешено
err := limiter.AllowWithin(ctx, 0)
// Возвращает: nil (токен доступен) или DeadlineExceeded (нет токена)

// Ждать до 50ms для токена
err = limiter.AllowWithin(ctx, 50*time.Millisecond)
// Возвращает: nil (получил токен за 50ms) или DeadlineExceeded (таймаут)
\`\`\`

**Ограничения:**
- Должен строго соблюдать максимальное время ожидания
- Должен очищать ресурсы контекста (defer cancel)
- Должен возвращать DeadlineExceeded когда maxWait 0 и нет токена`,
			hint1: `Для maxWait <= 0: вызовите tryImmediate(time.Now()), верните nil если true, иначе context.DeadlineExceeded.`,
			hint2: `Для maxWait > 0: создайте timeout context через context.WithTimeout(ctx, maxWait), defer cancel(), затем вызовите Allow().`,
			whyItMatters: `AllowWithin обеспечивает ограниченное ожидание для rate-limited операций, необходимое для соблюдения SLA и предотвращения бесконечной блокировки.

**Почему важно максимальное время ожидания:**
- **SLA Гарантии:** "API отвечает за 100ms или fails fast"
- **User Experience:** Предотвращение бесконечного ожидания пользователей
- **Resource Management:** Ограничение времени блокировки потоков/горутин
- **Circuit Breaker интеграция:** Fail fast после порога вместо накопления backlog

**Реальные Паттерны:**

**HTTP API с Response Time SLA:**
\`\`\`go
limiter := New(100, 10)

func handler(w http.ResponseWriter, r *http.Request) {
    // SLA: Должен ответить в течение 100ms всего
    // Разрешаем максимум 50ms ожидания для rate limiter
    if err := limiter.AllowWithin(r.Context(), 50*time.Millisecond); err != nil {
        // Не удалось получить токен за 50ms
        http.Error(w, "Service overloaded, try again", 503)
        metrics.RecordRateLimitReject()
        return
    }

    // Обработать запрос (должен завершиться в оставшиеся 50ms)
    processRequest(w, r)
}
\`\`\`

**Обработка Background Job:**
\`\`\`go
limiter := New(10, 2)  // 10 jobs/sec, burst 2

func processJobs(ctx context.Context, jobs []Job) {
    for _, job := range jobs {
        // Попытка получить токен немедленно, не ждём
        if err := limiter.AllowWithin(ctx, 0); err != nil {
            // Токена нет, ставим job в очередь на потом
            requeueJob(job)
            continue
        }

        // Получили токен немедленно, обрабатываем сейчас
        if err := processJob(ctx, job); err != nil {
            log.Printf("Job провалился: %v", err)
        }
    }
}
\`\`\`

**External API с Timeout:**
\`\`\`go
type APIClient struct {
    limiter *Limiter
}

func NewAPIClient() *APIClient {
    return &APIClient{
        limiter: New(50, 5),  // 50 RPS, burst 5
    }
}

func (c *APIClient) FetchData(ctx context.Context) (*Data, error) {
    // Общий таймаут: 5 секунд
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()

    // Ждём максимум 1 секунду для rate limiter
    if err := c.limiter.AllowWithin(ctx, time.Second); err != nil {
        return nil, fmt.Errorf("rate limit timeout: %w", err)
    }

    // Делаем API вызов с оставшимися 4 секундами таймаута
    return c.callAPI(ctx)
}
\`\`\`

**Load Shedding Паттерн:**
\`\`\`go
limiter := New(1000, 100)

func handler(w http.ResponseWriter, r *http.Request) {
    // Немедленная проверка: load shedding
    if err := limiter.AllowWithin(r.Context(), 0); err != nil {
        // Сервер перегружен, отклоняем немедленно
        w.Header().Set("Retry-After", "1")
        http.Error(w, "Server перегружен", 503)
        return
    }

    // Получили токен, обрабатываем запрос
    processRequest(w, r)
}
\`\`\`

**Database Connection Pool:**
\`\`\`go
type DBPool struct {
    limiter *Limiter
    db      *sql.DB
}

func NewDBPool(db *sql.DB, maxQPS int) *DBPool {
    return &DBPool{
        limiter: New(float64(maxQPS), 10),
        db:      db,
    }
}

func (p *DBPool) Query(ctx context.Context, query string) (*sql.Rows, error) {
    // База данных должна ответить за 2 секунды
    // Разрешаем максимум 500ms ожидания для rate limiter
    if err := p.limiter.AllowWithin(ctx, 500*time.Millisecond); err != nil {
        return nil, fmt.Errorf("query rate limit timeout: %w", err)
    }

    // Выполняем запрос с оставшимися 1.5 секундами
    return p.db.QueryContext(ctx, query)
}
\`\`\`

**Retry Logic с Backoff:**
\`\`\`go
limiter := New(5, 1)  // 5 повторов в секунду

func retryOperation(ctx context.Context, op func() error) error {
    maxRetries := 3
    for i := 0; i < maxRetries; i++ {
        // Пробуем операцию
        if err := op(); err == nil {
            return nil  // Успех
        }

        // Backoff с rate limiting
        backoff := time.Duration(i+1) * 100 * time.Millisecond
        if err := limiter.AllowWithin(ctx, backoff); err != nil {
            // Превышено время backoff или контекст отменён
            return fmt.Errorf("retry timeout после %d попыток", i+1)
        }
    }
    return fmt.Errorf("превышено максимум повторов")
}
\`\`\`

**Ключевые Концепции:**
- **Ограниченное Ожидание:** Никогда не ждите дольше maxWait
- **Fail Fast:** Возвращайте DeadlineExceeded немедленно когда maxWait равен 0
- **Композиция Контекста:** Комбинируйте таймаут родительского контекста с maxWait таймаутом
- **Очистка Ресурсов:** Всегда defer cancel() для предотвращения утечек контекста

**Типичные Случаи Использования:**
- **maxWait = 0:** Немедленная проверка, load shedding (отклонить если нет токена)
- **maxWait = малый:** Fail fast при небольшой перегрузке (например, 50-100ms)
- **maxWait = средний:** Баланс между user experience и throughput (например, 500ms-1s)
- **maxWait = бесконечность:** Используйте Allow() вместо для неограниченного ожидания

**Обработка Ошибок:**
\`\`\`go
err := limiter.AllowWithin(ctx, 100*time.Millisecond)
switch {
case err == nil:
    // Токен получен в пределах maxWait
case errors.Is(err, context.DeadlineExceeded):
    // Таймаут: либо maxWait превышен либо таймаут родительского контекста
    // Проверяем родительский контекст для различения
    if ctx.Err() == nil {
        // maxWait таймаут (слишком медленно)
        return ErrRateLimitTimeout
    } else {
        // Таймаут родительского контекста (таймаут запроса)
        return ErrRequestTimeout
    }
case errors.Is(err, context.Canceled):
    // Клиент отменил запрос
    return ErrRequestCanceled
}
\`\`\`

**Соображения Производительности:**
- **Нулевая Аллокация:** tryImmediate избегает создания таймера для немедленной проверки
- **Переиспользование Контекста:** Создаёт новый контекст только когда maxWait > 0
- **Ранний Возврат:** Fails fast когда токен недоступен и maxWait равен 0

**Реальные Преимущества:**
- **Надёжность:** Поддержание стабильности системы при перегрузке
- **Отзывчивость:** Пользователи не ждут бесконечно
- **Соответствие SLA:** Response time гарантии выполняются
- **Эффективность Ресурсов:** Экономия ресурсов goroutine/thread

**Опыт Production:**
Компании сообщают:
- **99.9% Uptime:** Fail fast при перегрузках
- **50% меньше времени блокировки:** С ограничениями maxWait
- **Лучший UX:** Предсказуемые response time
- **Более простой debugging:** Чёткие таймауты

AllowWithin критически важен для построения отзывчивых сервисов, которые поддерживают SLA при graceful обработке перегрузки. Это необходимый инструмент для улучшения user experience и обеспечения стабильности системы в production.`
		},
		uz: {
			title: `Timeout bilan so'rovni tekshirish`,
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

func (l *Limiter) AllowWithin(ctx context.Context, maxWait time.Duration) error {
	if l == nil {                                        // nil limiter ni qayta ishlash: rate limiting yo'q
		return nil
	}
	if ctx == nil {                                      // nil context ni qayta ishlash: background context ishlatamiz
		ctx = context.Background()
	}
	if maxWait <= 0 {                                    // Kutish ruxsat etilmagan, darhol olishga urinamiz
		if l.tryImmediate(time.Now()) {                  // Token darhol mavjud
			return nil
		}
		return context.DeadlineExceeded                  // Token yo'q, darhol muvaffaqiyatsiz
	}
	ctx, cancel := context.WithTimeout(ctx, maxWait)    // maxWait bilan timeout context yaratamiz
	defer cancel()                                       // Kontekst resurslarini bo'shatish uchun har doim bekor qilamiz
	return l.Allow(ctx)                                  // timeout context bilan Allow ni qayta ishlatamiz
}

func (l *Limiter) tryImmediate(now time.Time) bool {
	l.mu.Lock()                                          // Thread-safe kirish uchun qulfni olamiz
	defer l.mu.Unlock()                                  // Chiqishda har doim qulfni bo'shatamiz
	l.releaseExpiredLocked(now)                          // Muddati o'tgan rezervatsiyalarni olib tashlaymiz
	if len(l.reservations) >= l.burst {                  // Tokenlar mavjud emas (sig'imda)
		return false
	}
	l.reservations = append(l.reservations, now.Add(l.interval))  // Tokenni darhol rezerv qilamiz
	return true                                          // Token berildi
}`,
			description: `Token olish uchun maksimal kutish vaqtini cheklaydigan **AllowWithin** metodini amalga oshiring.

**Talablar:**
1. \`AllowWithin(ctx context.Context, maxWait time.Duration) error\` metodini yarating
2. nil limiter ni qayta ishlang (nil qaytaring - rate limiting yo'q)
3. nil context ni qayta ishlang (context.Background() ishlating)
4. Agar \`maxWait <= 0\`:
   4.1. \`tryImmediate(time.Now())\` orqali darhol token olishga urining
   4.2. Token berilsa nil qaytaring
   4.3. Token mavjud bo'lmasa \`context.DeadlineExceeded\` qaytaring
5. Agar \`maxWait > 0\`:
   5.1. \`context.WithTimeout(ctx, maxWait)\` orqali timeout context yarating
   5.2. Kontekst leak ni oldini olish uchun defer cancel
   5.3. \`Allow(timeoutCtx)\` chaqiring va natijasini qaytaring
6. \`tryImmediate(now time.Time) bool\` helper ni amalga oshiring:
   6.1. Mutex ni qulflang
   6.2. Muddati o'tgan rezervatsiyalarni bo'shating
   6.3. Agar tokenlar mavjud, rezervatsiya qo'shing va true qaytaring
   6.4. Aks holda false qaytaring

**Misol:**
\`\`\`go
limiter := New(10, 1)  // 10 RPS, burst 1

// Darhol olish - kutish ruxsat etilmagan
err := limiter.AllowWithin(ctx, 0)
// Qaytaradi: nil (token mavjud) yoki DeadlineExceeded (token yo'q)

// Token uchun 50ms gacha kuting
err = limiter.AllowWithin(ctx, 50*time.Millisecond)
// Qaytaradi: nil (50ms ichida token oldi) yoki DeadlineExceeded (timeout)
\`\`\`

**Cheklovlar:**
- Maksimal kutish vaqtini qat'iy bajarishi kerak
- Kontekst resurslarini tozalashi kerak (defer cancel)
- maxWait 0 va token mavjud bo'lmasa DeadlineExceeded qaytarishi kerak`,
			hint1: `maxWait <= 0 uchun: tryImmediate(time.Now()) chaqiring, true bo'lsa nil qaytaring, aks holda context.DeadlineExceeded.`,
			hint2: `maxWait > 0 uchun: context.WithTimeout(ctx, maxWait) orqali timeout context yarating, defer cancel(), keyin Allow() chaqiring.`,
			whyItMatters: `AllowWithin rate-limited operatsiyalar uchun cheklangan kutishni ta'minlaydi, bu SLA larni saqlash va cheksiz blokirovkani oldini olish uchun muhim.

**Nima uchun maksimal kutish vaqti muhim:**
- **SLA Kafolatlari:** "API 100ms ichida javob beradi yoki tez muvaffaqiyatsiz bo'ladi"
- **User Experience:** Foydalanuvchilarni cheksiz kutishdan saqlash
- **Resource Management:** Thread/goroutine blokirovka vaqtini cheklash
- **Circuit Breaker integratsiyasi:** Backlog to'plash o'rniga chegaradan keyin tez muvaffaqiyatsizlik

**Haqiqiy Dunyo Patternlari:**

**HTTP API bilan Response Time SLA:**
\`\`\`go
limiter := New(100, 10)

func handler(w http.ResponseWriter, r *http.Request) {
    // SLA: Jami 100ms ichida javob berishi kerak
    // Rate limiter uchun maksimum 50ms kutishga ruxsat beramiz
    if err := limiter.AllowWithin(r.Context(), 50*time.Millisecond); err != nil {
        // 50ms ichida token ololmadik
        http.Error(w, "Service overloaded, try again", 503)
        metrics.RecordRateLimitReject()
        return
    }

    // So'rovni qayta ishlash (qolgan 50ms ichida tugatilishi kerak)
    processRequest(w, r)
}
\`\`\`

**Background Job qayta ishlash:**
\`\`\`go
limiter := New(10, 2)  // 10 jobs/sec, burst 2

func processJobs(ctx context.Context, jobs []Job) {
    for _, job := range jobs {
        // Darhol token olishga urining, kutmang
        if err := limiter.AllowWithin(ctx, 0); err != nil {
            // Token mavjud emas, vazifani keyinga qoldiring
            requeueJob(job)
            continue
        }

        // Darhol token oldik, hozir qayta ishlang
        if err := processJob(ctx, job); err != nil {
            log.Printf("Job muvaffaqiyatsiz: %v", err)
        }
    }
}
\`\`\`

**Timeout bilan Tashqi API:**
\`\`\`go
type APIClient struct {
    limiter *Limiter
}

func NewAPIClient() *APIClient {
    return &APIClient{
        limiter: New(50, 5),  // 50 RPS, burst 5
    }
}

func (c *APIClient) FetchData(ctx context.Context) (*Data, error) {
    // Jami timeout: 5 soniya
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()

    // Rate limiter uchun maksimum 1 soniya kuting
    if err := c.limiter.AllowWithin(ctx, time.Second); err != nil {
        return nil, fmt.Errorf("rate limit timeout: %w", err)
    }

    // Qolgan 4 soniya timeout bilan API chaqiruvini amalga oshiring
    return c.callAPI(ctx)
}
\`\`\`

**Load Shedding Patterni:**
\`\`\`go
limiter := New(1000, 100)

func handler(w http.ResponseWriter, r *http.Request) {
    // Darhol tekshirish: load shedding
    if err := limiter.AllowWithin(r.Context(), 0); err != nil {
        // Server ortiqcha yuklangan, darhol rad eting
        w.Header().Set("Retry-After", "1")
        http.Error(w, "Server ortiqcha yuklangan", 503)
        return
    }

    // Token oldik, so'rovni qayta ishlang
    processRequest(w, r)
}
\`\`\`

**Database Connection Pool:**
\`\`\`go
type DBPool struct {
    limiter *Limiter
    db      *sql.DB
}

func NewDBPool(db *sql.DB, maxQPS int) *DBPool {
    return &DBPool{
        limiter: New(float64(maxQPS), 10),
        db:      db,
    }
}

func (p *DBPool) Query(ctx context.Context, query string) (*sql.Rows, error) {
    // Ma'lumotlar bazasi 2 soniya ichida javob berishi kerak
    // Rate limiter uchun maksimum 500ms kutishga ruxsat bering
    if err := p.limiter.AllowWithin(ctx, 500*time.Millisecond); err != nil {
        return nil, fmt.Errorf("query rate limit timeout: %w", err)
    }

    // Qolgan 1.5 soniya bilan so'rovni bajarish
    return p.db.QueryContext(ctx, query)
}
\`\`\`

**Backoff bilan Retry mantiq:**
\`\`\`go
limiter := New(5, 1)  // Soniyada 5 ta qayta urinish

func retryOperation(ctx context.Context, op func() error) error {
    maxRetries := 3
    for i := 0; i < maxRetries; i++ {
        // Operatsiyani sinab ko'ring
        if err := op(); err == nil {
            return nil  // Muvaffaqiyat
        }

        // Rate limiting bilan backoff
        backoff := time.Duration(i+1) * 100 * time.Millisecond
        if err := limiter.AllowWithin(ctx, backoff); err != nil {
            // Backoff vaqti oshdi yoki kontekst bekor qilindi
            return fmt.Errorf("%d urinishdan keyin timeout", i+1)
        }
    }
    return fmt.Errorf("maksimal urinishlar oshdi")
}
\`\`\`

**Asosiy Tushunchalar:**
- **Cheklangan Kutish:** Hech qachon maxWait dan uzoqroq kutmang
- **Tez Muvaffaqiyatsizlik:** maxWait 0 bo'lganda darhol DeadlineExceeded qaytaring
- **Kontekst Kompozitsiyasi:** Ota kontekst timeout ni maxWait timeout bilan birlashtiring
- **Resurs Tozalash:** Kontekst leak larni oldini olish uchun har doim defer cancel()

**Odatiy Foydalanish Holatlari:**
- **maxWait = 0:** Darhol tekshirish, load shedding (token bo'lmasa rad eting)
- **maxWait = kichik:** Biroz ortiqcha yuklanganda tez muvaffaqiyatsizlik (masalan, 50-100ms)
- **maxWait = o'rtacha:** Foydalanuvchi tajribasi va o'tkazuvchanlik o'rtasida muvozanat (masalan, 500ms-1s)
- **maxWait = cheksiz:** Cheksiz kutish uchun o'rniga Allow() dan foydalaning

**Xatolarni Qayta Ishlash:**
\`\`\`go
err := limiter.AllowWithin(ctx, 100*time.Millisecond)
switch {
case err == nil:
    // maxWait ichida token olindi
case errors.Is(err, context.DeadlineExceeded):
    // Timeout: maxWait oshdi yoki ota kontekst timeout
    // Farqlash uchun ota kontekstni tekshiring
    if ctx.Err() == nil {
        // maxWait timeout (juda sekin)
        return ErrRateLimitTimeout
    } else {
        // Ota kontekst timeout (so'rov timeout)
        return ErrRequestTimeout
    }
case errors.Is(err, context.Canceled):
    // Mijoz so'rovni bekor qildi
    return ErrRequestCanceled
}
\`\`\`

**Ishlash Mulohazalari:**
- **Zero Allocation:** tryImmediate darhol tekshirish uchun timer yaratishdan qochadi
- **Kontekstni Qayta Ishlash:** maxWait > 0 bo'lganda faqat yangi kontekst yaratadi
- **Erta Qaytish:** Token mavjud bo'lmasa va maxWait 0 bo'lganda tez muvaffaqiyatsiz bo'ladi

**Real-World Foydalari:**
- **Ishonchlilik:** Ortiqcha yuk vaqtida tizimni barqaror saqlash
- **Javob Berish:** Foydalanuvchilar cheksiz kutmaydilar
- **SLA Muvofiqlik:** Response time kafolatlari bajariladi
- **Resurs Samaradorligi:** Goroutine/thread resurslarini tejash

**Ishlab Chiqarish Tajribasi:**
Kompaniyalar quyidagilarni xabar qilishadi:
- **99.9% Uptime:** Tez muvaffaqiyatsizlik bilan overload holatlarida
- **50% kam blokirovka vaqti:** maxWait chegaralari bilan
- **Yaxshi foydalanuvchi tajribasi:** Predictable response time lar
- **Osonroq debugging:** Timeout lari aniq belgilangan

AllowWithin ortiqcha yukni graceful qayta ishlagan holda SLA larni saqlaydigan javob beradigan xizmatlarni qurish uchun juda muhim. Bu production tizimlarida foydalanuvchi tajribasini yaxshilash va tizim barqarorligini ta'minlash uchun zarur vosita.`
		}
	}
};

export default task;
