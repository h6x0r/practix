import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-concurrency-retry-with-context',
	title: 'Retry With Context',
	difficulty: 'hard',	tags: ['go', 'concurrency', 'context', 'retry', 'resilience'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **RetryWithContext** that retries a function with exponential backoff while respecting context cancellation.

**Requirements:**
1. Create function \`RetryWithContext(ctx context.Context, attempts int, delay time.Duration, fn func(context.Context) error) error\`
2. Handle nil context (use Background)
3. Handle attempts <= 0 (return nil)
4. Try fn up to attempts times
5. Return nil on first success
6. Wait delay between attempts
7. Check context before each attempt
8. Check context during delay using timer
9. Return last error if all attempts fail
10. Properly cleanup timer if context canceled during delay

**Example:**
\`\`\`go
attempt := 0
err := RetryWithContext(context.Background(), 3, 100*time.Millisecond, func(ctx context.Context) error {
    attempt++
    if attempt < 3 {
        return fmt.Errorf("failed")
    }
    return nil // Success on 3rd attempt
})
// err = nil, attempt = 3

ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
defer cancel()

attempt = 0
err = RetryWithContext(ctx, 5, 100*time.Millisecond, func(ctx context.Context) error {
    attempt++
    return fmt.Errorf("always fails")
})
// err = context.DeadlineExceeded (canceled during retry)
\`\`\`

**Constraints:**
- Must respect context cancellation
- Must cleanup timer properly
- Must not delay after last attempt
- Must return last error if all fail`,
	initialCode: `package concurrency

import (
	"context"
	"time"
)

// TODO: Implement RetryWithContext
func RetryWithContext(ctx context.Context, attempts int, delay time.Duration, fn func(context.Context) error) error {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
	"time"
)

func RetryWithContext(ctx context.Context, attempts int, delay time.Duration, fn func(context.Context) error) error {
	if ctx == nil {                                             // Handle nil context
		ctx = context.Background()                          // Use Background as fallback
	}
	if attempts <= 0 {                                          // No attempts requested
		return nil                                          // Return immediately
	}
	var lastErr error                                           // Track last error
	for attempt := 0; attempt < attempts; attempt++ {           // Try up to attempts times
		if err := ctx.Err(); err != nil {                   // Check context before attempt
			return err                                  // Return context error
		}
		if err := fn(ctx); err == nil {                     // Try function
			return nil                                  // Success, return immediately
		} else {
			lastErr = err                               // Save error for later
		}
		if attempt == attempts-1 {                          // Last attempt
			break                                       // Don't delay after last
		}
		timer := time.NewTimer(delay)                       // Create timer for delay
		select {
		case <-ctx.Done():                                  // Context canceled during delay
			if !timer.Stop() {                          // Try to stop timer
				<-timer.C                           // Drain channel if too late
			}
			return ctx.Err()                            // Return context error
		case <-timer.C:                                     // Delay completed
		}
	}
	return lastErr                                              // All attempts failed, return last error
}`,
			hint1: `Use a for loop for attempts. Check ctx.Err() before each attempt. Use time.NewTimer for delay and select between ctx.Done() and timer.C.`,
			hint2: `Important: Stop timer if context cancels and drain channel if stop returns false. Don\`t delay after last attempt (use break).`,
			whyItMatters: `RetryWithContext provides resilient error handling with proper cancellation, essential for dealing with transient failures in distributed systems.

**Why Retry Logic:**
- **Resilience:** Recover from transient failures
- **Network Issues:** Retry on temporary network problems
- **Rate Limiting:** Back off when rate limited
- **Service Recovery:** Wait for services to recover
- **Graceful Degradation:** Try multiple times before giving up

**Production Pattern:**
\`\`\`go
// Retry API call with exponential backoff
func CallExternalAPI(ctx context.Context, endpoint string) (*Response, error) {
    var response *Response

    err := RetryWithContext(ctx, 5, time.Second, func(ctx context.Context) error {
        resp, err := http.Get(endpoint)
        if err != nil {
            return err
        }
        defer resp.Body.Close()

        if resp.StatusCode == 429 {
            return fmt.Errorf("rate limited")
        }

        if resp.StatusCode >= 500 {
            return fmt.Errorf("server error")
        }

        response = parseResponse(resp)
        return nil
    })

    return response, err
}

// Retry database connection
func ConnectDatabase(ctx context.Context, dsn string) (*sql.DB, error) {
    var db *sql.DB

    err := RetryWithContext(ctx, 10, 2*time.Second, func(ctx context.Context) error {
        var err error
        db, err = sql.Open("postgres", dsn)
        if err != nil {
            return err
        }

        if err := db.PingContext(ctx); err != nil {
            db.Close()
            return err
        }

        return nil
    })

    return db, err
}

// Retry message send with backoff
func SendMessage(ctx context.Context, queue *Queue, msg Message) error {
    return RetryWithContext(ctx, 3, 500*time.Millisecond, func(ctx context.Context) error {
        return queue.Send(ctx, msg)
    })
}

// Retry file operation
func WriteFileWithRetry(ctx context.Context, path string, data []byte) error {
    return RetryWithContext(ctx, 5, time.Second, func(ctx context.Context) error {
        return os.WriteFile(path, data, 0644)
    })
}

// Retry cache operation
func GetFromCacheWithRetry(ctx context.Context, cache *Cache, key string) (Value, error) {
    var value Value

    err := RetryWithContext(ctx, 3, 100*time.Millisecond, func(ctx context.Context) error {
        var err error
        value, err = cache.Get(ctx, key)
        return err
    })

    return value, err
}

// Exponential backoff with jitter
func RetryWithExponentialBackoff(ctx context.Context, fn func(context.Context) error) error {
    delay := 100 * time.Millisecond
    maxDelay := 10 * time.Second

    for attempt := 0; attempt < 10; attempt++ {
        if err := fn(ctx); err == nil {
            return nil
        }

        // Add jitter (randomness) to prevent thundering herd
        jitter := time.Duration(rand.Int63n(int64(delay) / 2))
        actualDelay := delay + jitter

        timer := time.NewTimer(actualDelay)
        select {
        case <-ctx.Done():
            timer.Stop()
            return ctx.Err()
        case <-timer.C:
        }

        // Exponential increase
        delay *= 2
        if delay > maxDelay {
            delay = maxDelay
        }
    }

    return fmt.Errorf("max retries exceeded")
}

// Circuit breaker with retry
type CircuitBreaker struct {
    failures int
    threshold int
}

func (cb *CircuitBreaker) Call(ctx context.Context, fn func(context.Context) error) error {
    if cb.failures >= cb.threshold {
        return fmt.Errorf("circuit breaker open")
    }

    err := RetryWithContext(ctx, 3, time.Second, fn)
    if err != nil {
        cb.failures++
    } else {
        cb.failures = 0
    }

    return err
}

// Retry with custom backoff strategy
func RetryWithBackoff(ctx context.Context, strategy BackoffStrategy, fn func(context.Context) error) error {
    for attempt := 0; ; attempt++ {
        if err := ctx.Err(); err != nil {
            return err
        }

        if err := fn(ctx); err == nil {
            return nil
        }

        delay := strategy.NextDelay(attempt)
        if delay < 0 {
            return fmt.Errorf("max retries exceeded")
        }

        timer := time.NewTimer(delay)
        select {
        case <-ctx.Done():
            timer.Stop()
            return ctx.Err()
        case <-timer.C:
        }
    }
}

// Retry with logging
func RetryWithLogging(ctx context.Context, name string, fn func(context.Context) error) error {
    return RetryWithContext(ctx, 5, time.Second, func(ctx context.Context) error {
        err := fn(ctx)
        if err != nil {
            log.Printf("%s: attempt failed: %v", name, err)
        }
        return err
    })
}
\`\`\`

**Real-World Benefits:**
- **Reliability:** Services recover from transient failures
- **User Experience:** Operations succeed despite temporary issues
- **Resource Efficiency:** Smart backoff prevents overwhelming systems
- **Observability:** Know how many retries needed

**Common Use Cases:**
- **API Calls:** Retry on network failures or 5xx errors
- **Database Operations:** Retry on connection issues
- **Message Queues:** Retry message delivery
- **File Operations:** Retry on temporary file locks
- **Cache Operations:** Retry on temporary unavailability
- **Service Discovery:** Retry finding available service

**Retry Strategies:**
- **Fixed Delay:** Same delay between retries
- **Exponential Backoff:** 100ms, 200ms, 400ms, 800ms...
- **Exponential with Jitter:** Add randomness to prevent thundering herd
- **Fibonacci Backoff:** 100ms, 100ms, 200ms, 300ms, 500ms...

**Best Practices:**
- **Idempotent Operations:** Safe to retry multiple times
- **Timeout Per Attempt:** Each attempt has own timeout
- **Log Retries:** Track retry patterns
- **Circuit Breaker:** Stop retrying if service is down
- **Backoff:** Don't hammer failing services

**Anti-Patterns:**
- **Infinite Retries:** Always have max attempts
- **No Delay:** Always wait between attempts
- **Ignoring Context:** Always respect cancellation
- **Same Error:** Distinguish transient from permanent errors

Without RetryWithContext, implementing reliable retry logic with proper cancellation, backoff, and cleanup is complex and error-prone.`,
	testCode: `package concurrency

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	err := RetryWithContext(context.Background(), 0, time.Millisecond, func(ctx context.Context) error { return nil })
	if err != nil { t.Errorf("expected nil for 0 attempts, got %v", err) }
}

func Test2(t *testing.T) {
	var count int64
	err := RetryWithContext(context.Background(), 3, time.Millisecond, func(ctx context.Context) error {
		atomic.AddInt64(&count, 1)
		return nil
	})
	if err != nil { t.Errorf("expected nil, got %v", err) }
	if count != 1 { t.Errorf("expected 1 call on success, got %d", count) }
}

func Test3(t *testing.T) {
	var count int64
	err := RetryWithContext(context.Background(), 3, time.Millisecond, func(ctx context.Context) error {
		atomic.AddInt64(&count, 1)
		return fmt.Errorf("always fails")
	})
	if err == nil { t.Error("expected error for all failures") }
	if count != 3 { t.Errorf("expected 3 attempts, got %d", count) }
}

func Test4(t *testing.T) {
	var count int64
	err := RetryWithContext(context.Background(), 5, time.Millisecond, func(ctx context.Context) error {
		c := atomic.AddInt64(&count, 1)
		if c < 3 { return fmt.Errorf("fail") }
		return nil
	})
	if err != nil { t.Errorf("expected nil after 3 attempts, got %v", err) }
	if count != 3 { t.Errorf("expected 3 attempts, got %d", count) }
}

func Test5(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	err := RetryWithContext(ctx, 10, 100*time.Millisecond, func(ctx context.Context) error { return fmt.Errorf("fail") })
	if !errors.Is(err, context.DeadlineExceeded) { t.Errorf("expected DeadlineExceeded, got %v", err) }
}

func Test6(t *testing.T) {
	err := RetryWithContext(nil, 1, time.Millisecond, func(ctx context.Context) error { return nil })
	if err != nil { t.Errorf("expected nil for nil context, got %v", err) }
}

func Test7(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := RetryWithContext(ctx, 5, time.Millisecond, func(ctx context.Context) error { return nil })
	if !errors.Is(err, context.Canceled) { t.Errorf("expected Canceled, got %v", err) }
}

func Test8(t *testing.T) {
	start := time.Now()
	_ = RetryWithContext(context.Background(), 3, 20*time.Millisecond, func(ctx context.Context) error { return fmt.Errorf("fail") })
	elapsed := time.Since(start)
	if elapsed < 35*time.Millisecond { t.Error("should have waited between retries") }
}

func Test9(t *testing.T) {
	err := RetryWithContext(context.Background(), -1, time.Millisecond, func(ctx context.Context) error { return fmt.Errorf("fail") })
	if err != nil { t.Errorf("expected nil for negative attempts, got %v", err) }
}

func Test10(t *testing.T) {
	var received bool
	_ = RetryWithContext(context.Background(), 1, time.Millisecond, func(ctx context.Context) error {
		if ctx != nil { received = true }
		return nil
	})
	if !received { t.Error("expected function to receive context") }
}
`,
	order: 8,
	translations: {
		ru: {
			title: 'Повторные попытки с учётом контекста',
			description: `Реализуйте **RetryWithContext**, который повторяет функцию с экспоненциальным backoff учитывая отмену контекста.

**Требования:**
1. Создайте функцию \`RetryWithContext(ctx context.Context, attempts int, delay time.Duration, fn func(context.Context) error) error\`
2. Обработайте nil context (используйте Background)
3. Обработайте attempts <= 0 (верните nil)
4. Попробуйте fn до attempts раз
5. Верните nil при первом успехе
6. Ждите delay между попытками
7. Проверяйте контекст перед каждой попыткой
8. Проверяйте контекст во время delay используя timer
9. Верните последнюю ошибку если все попытки провалились
10. Правильно очищайте timer если контекст отменён во время delay

**Пример:**
\`\`\`go
attempt := 0
err := RetryWithContext(context.Background(), 3, 100*time.Millisecond, func(ctx context.Context) error {
    attempt++
    if attempt < 3 {
        return fmt.Errorf("failed")
    }
    return nil // Успех на 3-й попытке
})
// err = nil, attempt = 3

ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
defer cancel()

attempt = 0
err = RetryWithContext(ctx, 5, 100*time.Millisecond, func(ctx context.Context) error {
    attempt++
    return fmt.Errorf("always fails")
})
// err = context.DeadlineExceeded (отменён во время retry)
\`\`\`

**Ограничения:**
- Должен учитывать отмену контекста
- Должен правильно очищать timer
- Не должен задерживаться после последней попытки
- Должен возвращать последнюю ошибку если все провалились`,
			hint1: `Используйте for цикл для попыток. Проверяйте ctx.Err() перед каждой попыткой. Используйте time.NewTimer для delay и select между ctx.Done() и timer.C.`,
			hint2: `Важно: Остановите timer если контекст отменяется и дренируйте канал если stop возвращает false. Не задерживайтесь после последней попытки (используйте break).`,
			whyItMatters: `RetryWithContext обеспечивает устойчивую обработку ошибок с правильной отменой, необходим для борьбы с временными сбоями в распределённых системах.

**Почему Retry Logic критична:**
- **Устойчивость:** Восстановление от временных сбоев
- **Проблемы сети:** Повтор при временных сетевых проблемах
- **Rate Limiting:** Backoff при rate limit
- **Восстановление сервиса:** Ожидание восстановления сервисов
- **Graceful Degradation:** Множество попыток перед отказом

**Production паттерны:**
\`\`\`go
// Повтор API вызова с exponential backoff
func CallExternalAPI(ctx context.Context, endpoint string) (*Response, error) {
    var response *Response

    err := RetryWithContext(ctx, 5, time.Second, func(ctx context.Context) error {
        resp, err := http.Get(endpoint)
        if err != nil {
            return err
        }
        defer resp.Body.Close()

        if resp.StatusCode == 429 {
            return fmt.Errorf("rate limited")
        }

        if resp.StatusCode >= 500 {
            return fmt.Errorf("server error")
        }

        response = parseResponse(resp)
        return nil
    })

    return response, err
}

// Повтор подключения к БД
func ConnectDatabase(ctx context.Context, dsn string) (*sql.DB, error) {
    var db *sql.DB

    err := RetryWithContext(ctx, 10, 2*time.Second, func(ctx context.Context) error {
        var err error
        db, err = sql.Open("postgres", dsn)
        if err != nil {
            return err
        }

        if err := db.PingContext(ctx); err != nil {
            db.Close()
            return err
        }

        return nil
    })

    return db, err
}

// Повтор отправки сообщения с backoff
func SendMessage(ctx context.Context, queue *Queue, msg Message) error {
    return RetryWithContext(ctx, 3, 500*time.Millisecond, func(ctx context.Context) error {
        return queue.Send(ctx, msg)
    })
}

// Повтор файловой операции
func WriteFileWithRetry(ctx context.Context, path string, data []byte) error {
    return RetryWithContext(ctx, 5, time.Second, func(ctx context.Context) error {
        return os.WriteFile(path, data, 0644)
    })
}

// Повтор cache операции
func GetFromCacheWithRetry(ctx context.Context, cache *Cache, key string) (Value, error) {
    var value Value

    err := RetryWithContext(ctx, 3, 100*time.Millisecond, func(ctx context.Context) error {
        var err error
        value, err = cache.Get(ctx, key)
        return err
    })

    return value, err
}

// Exponential backoff с jitter
func RetryWithExponentialBackoff(ctx context.Context, fn func(context.Context) error) error {
    delay := 100 * time.Millisecond
    maxDelay := 10 * time.Second

    for attempt := 0; attempt < 10; attempt++ {
        if err := fn(ctx); err == nil {
            return nil
        }

        // Добавляем jitter (случайность) для предотвращения thundering herd
        jitter := time.Duration(rand.Int63n(int64(delay) / 2))
        actualDelay := delay + jitter

        timer := time.NewTimer(actualDelay)
        select {
        case <-ctx.Done():
            timer.Stop()
            return ctx.Err()
        case <-timer.C:
        }

        // Экспоненциальное увеличение
        delay *= 2
        if delay > maxDelay {
            delay = maxDelay
        }
    }

    return fmt.Errorf("max retries exceeded")
}

// Circuit breaker с retry
type CircuitBreaker struct {
    failures int
    threshold int
}

func (cb *CircuitBreaker) Call(ctx context.Context, fn func(context.Context) error) error {
    if cb.failures >= cb.threshold {
        return fmt.Errorf("circuit breaker open")
    }

    err := RetryWithContext(ctx, 3, time.Second, fn)
    if err != nil {
        cb.failures++
    } else {
        cb.failures = 0
    }

    return err
}

// Retry с логированием
func RetryWithLogging(ctx context.Context, name string, fn func(context.Context) error) error {
    return RetryWithContext(ctx, 5, time.Second, func(ctx context.Context) error {
        err := fn(ctx)
        if err != nil {
            log.Printf("%s: попытка провалилась: %v", name, err)
        }
        return err
    })
}
\`\`\`

**Реальные преимущества:**
- **Надёжность:** Сервисы восстанавливаются от временных сбоев
- **Пользовательский опыт:** Операции успешны несмотря на временные проблемы
- **Эффективность ресурсов:** Умный backoff предотвращает перегрузку систем
- **Наблюдаемость:** Знать сколько повторов потребовалось

**Типичные сценарии использования:**
- **API вызовы:** Повтор при сетевых сбоях или 5xx ошибках
- **Операции с БД:** Повтор при проблемах соединения
- **Очереди сообщений:** Повтор доставки сообщения
- **Файловые операции:** Повтор при временных блокировках файлов
- **Cache операции:** Повтор при временной недоступности
- **Service Discovery:** Повтор поиска доступного сервиса

**Стратегии повтора:**
- **Fixed Delay:** Одинаковая задержка между повторами
- **Exponential Backoff:** 100ms, 200ms, 400ms, 800ms...
- **Exponential with Jitter:** Добавить случайность для предотвращения thundering herd
- **Fibonacci Backoff:** 100ms, 100ms, 200ms, 300ms, 500ms...

**Лучшие практики:**
- **Идемпотентные операции:** Безопасны для множественных повторов
- **Timeout на попытку:** Каждая попытка имеет собственный timeout
- **Логирование повторов:** Отслеживание паттернов повторов
- **Circuit Breaker:** Прекращение повторов если сервис недоступен
- **Backoff:** Не бомбардировать падающие сервисы

**Анти-паттерны:**
- **Бесконечные повторы:** Всегда имейте максимум попыток
- **Без задержки:** Всегда ждите между попытками
- **Игнорирование контекста:** Всегда учитывайте отмену
- **Одна ошибка:** Различайте временные и постоянные ошибки

Без RetryWithContext реализация надёжной логики повторов с правильной отменой, backoff и cleanup сложна и подвержена ошибкам.`,
			solutionCode: `package concurrency

import (
	"context"
	"time"
)

func RetryWithContext(ctx context.Context, attempts int, delay time.Duration, fn func(context.Context) error) error {
	if ctx == nil {                                             // Обработка nil контекста
		ctx = context.Background()                          // Используем Background как fallback
	}
	if attempts <= 0 {                                          // Не запрошено попыток
		return nil                                          // Возвращаемся сразу
	}
	var lastErr error                                           // Отслеживаем последнюю ошибку
	for attempt := 0; attempt < attempts; attempt++ {           // Пробуем до attempts раз
		if err := ctx.Err(); err != nil {                   // Проверяем контекст перед попыткой
			return err                                  // Возвращаем ошибку контекста
		}
		if err := fn(ctx); err == nil {                     // Пробуем функцию
			return nil                                  // Успех, возвращаемся сразу
		} else {
			lastErr = err                               // Сохраняем ошибку на потом
		}
		if attempt == attempts-1 {                          // Последняя попытка
			break                                       // Не задерживаемся после последней
		}
		timer := time.NewTimer(delay)                       // Создаём таймер для задержки
		select {
		case <-ctx.Done():                                  // Контекст отменён во время задержки
			if !timer.Stop() {                          // Пробуем остановить таймер
				<-timer.C                           // Дренируем канал если слишком поздно
			}
			return ctx.Err()                            // Возвращаем ошибку контекста
		case <-timer.C:                                     // Задержка завершена
		}
	}
	return lastErr                                              // Все попытки провалились, возвращаем последнюю ошибку
}`
		},
		uz: {
			title: 'Kontekstni hisobga olgan holda qayta urinish',
			description: `Kontekst bekor qilinishini hisobga olgan holda eksponensial backoff bilan funksiyani takrorlaydigan **RetryWithContext** ni amalga oshiring.

**Talablar:**
1. \`RetryWithContext(ctx context.Context, attempts int, delay time.Duration, fn func(context.Context) error) error\` funksiyasini yarating
2. nil kontekstni ishlang (Background dan foydalaning)
3. attempts <= 0 ni ishlang (nil qaytaring)
4. fn ni attempts martagacha sinab ko'ring
5. Birinchi muvaffaqiyatda nil qaytaring
6. Urinishlar o'rtasida delay kuting
7. Har bir urinishdan oldin kontekstni tekshiring
8. Timer dan foydalanib delay paytida kontekstni tekshiring
9. Agar barcha urinishlar muvaffaqiyatsiz bo'lsa oxirgi xatoni qaytaring
10. Agar delay paytida kontekst bekor qilinsa timerni to'g'ri tozalang

**Misol:**
\`\`\`go
attempt := 0
err := RetryWithContext(context.Background(), 3, 100*time.Millisecond, func(ctx context.Context) error {
    attempt++
    if attempt < 3 {
        return fmt.Errorf("failed")
    }
    return nil // 3-urinishda muvaffaqiyat
})
// err = nil, attempt = 3

ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
defer cancel()

attempt = 0
err = RetryWithContext(ctx, 5, 100*time.Millisecond, func(ctx context.Context) error {
    attempt++
    return fmt.Errorf("always fails")
})
// err = context.DeadlineExceeded (retry paytida bekor qilindi)
\`\`\`

**Cheklovlar:**
- Kontekst bekor qilinishini hisobga olishi kerak
- Timerni to'g'ri tozalashi kerak
- Oxirgi urinishdan keyin kechikmasligi kerak
- Agar hammasi muvaffaqiyatsiz bo'lsa oxirgi xatoni qaytarishi kerak`,
			hint1: `Urinishlar uchun for siklidan foydalaning. Har bir urinishdan oldin ctx.Err() ni tekshiring. Delay uchun time.NewTimer dan va ctx.Done() va timer.C o'rtasida select dan foydalaning.`,
			hint2: `Muhim: Agar kontekst bekor qilinsa timerni to'xtating va agar stop false qaytarsa kanalni drenaj qiling. Oxirgi urinishdan keyin kechiktirmang (break dan foydalaning).`,
			whyItMatters: `RetryWithContext to'g'ri bekor qilish bilan barqaror xatolarni qayta ishlashni ta'minlaydi, taqsimlangan tizimlardagi vaqtinchalik nosozliklarga qarshi kurashish uchun zarur.

**Nima uchun Retry Logic muhim:**
- **Barqarorlik:** Vaqtinchalik nosozliklardan tiklanish
- **Tarmoq muammolari:** Vaqtinchalik tarmoq muammolarida qayta urinish
- **Rate Limiting:** Rate limitda backoff
- **Xizmat tiklanishi:** Xizmatlar tiklanishini kutish
- **Graceful Degradation:** Taslim bo'lishdan oldin ko'p urinishlar

**Production patternlar:**
\`\`\`go
// Eksponensial backoff bilan API chaqiruvini qayta urinish
func CallExternalAPI(ctx context.Context, endpoint string) (*Response, error) {
    var response *Response

    err := RetryWithContext(ctx, 5, time.Second, func(ctx context.Context) error {
        resp, err := http.Get(endpoint)
        if err != nil {
            return err
        }
        defer resp.Body.Close()

        if resp.StatusCode == 429 {
            return fmt.Errorf("rate limited")
        }

        if resp.StatusCode >= 500 {
            return fmt.Errorf("server error")
        }

        response = parseResponse(resp)
        return nil
    })

    return response, err
}

// Ma'lumotlar bazasiga ulanishni qayta urinish
func ConnectDatabase(ctx context.Context, dsn string) (*sql.DB, error) {
    var db *sql.DB

    err := RetryWithContext(ctx, 10, 2*time.Second, func(ctx context.Context) error {
        var err error
        db, err = sql.Open("postgres", dsn)
        if err != nil {
            return err
        }

        if err := db.PingContext(ctx); err != nil {
            db.Close()
            return err
        }

        return nil
    })

    return db, err
}

// Backoff bilan xabar yuborishni qayta urinish
func SendMessage(ctx context.Context, queue *Queue, msg Message) error {
    return RetryWithContext(ctx, 3, 500*time.Millisecond, func(ctx context.Context) error {
        return queue.Send(ctx, msg)
    })
}

// Fayl operatsiyasini qayta urinish
func WriteFileWithRetry(ctx context.Context, path string, data []byte) error {
    return RetryWithContext(ctx, 5, time.Second, func(ctx context.Context) error {
        return os.WriteFile(path, data, 0644)
    })
}

// Kesh operatsiyasini qayta urinish
func GetFromCacheWithRetry(ctx context.Context, cache *Cache, key string) (Value, error) {
    var value Value

    err := RetryWithContext(ctx, 3, 100*time.Millisecond, func(ctx context.Context) error {
        var err error
        value, err = cache.Get(ctx, key)
        return err
    })

    return value, err
}

// Jitter bilan exponential backoff
func RetryWithExponentialBackoff(ctx context.Context, fn func(context.Context) error) error {
    delay := 100 * time.Millisecond
    maxDelay := 10 * time.Second

    for attempt := 0; attempt < 10; attempt++ {
        if err := fn(ctx); err == nil {
            return nil
        }

        // Thundering herd oldini olish uchun jitter (tasodifiylik) qo'shamiz
        jitter := time.Duration(rand.Int63n(int64(delay) / 2))
        actualDelay := delay + jitter

        timer := time.NewTimer(actualDelay)
        select {
        case <-ctx.Done():
            timer.Stop()
            return ctx.Err()
        case <-timer.C:
        }

        // Eksponensial o'sish
        delay *= 2
        if delay > maxDelay {
            delay = maxDelay
        }
    }

    return fmt.Errorf("max retries exceeded")
}

// Retry bilan circuit breaker
type CircuitBreaker struct {
    failures int
    threshold int
}

func (cb *CircuitBreaker) Call(ctx context.Context, fn func(context.Context) error) error {
    if cb.failures >= cb.threshold {
        return fmt.Errorf("circuit breaker open")
    }

    err := RetryWithContext(ctx, 3, time.Second, fn)
    if err != nil {
        cb.failures++
    } else {
        cb.failures = 0
    }

    return err
}

// Loglash bilan retry
func RetryWithLogging(ctx context.Context, name string, fn func(context.Context) error) error {
    return RetryWithContext(ctx, 5, time.Second, func(ctx context.Context) error {
        err := fn(ctx)
        if err != nil {
            log.Printf("%s: urinish muvaffaqiyatsiz: %v", name, err)
        }
        return err
    })
}
\`\`\`

**Haqiqiy foydalari:**
- **Ishonchlilik:** Xizmatlar vaqtinchalik nosozliklardan tiklanadi
- **Foydalanuvchi tajribasi:** Vaqtinchalik muammolarga qaramay operatsiyalar muvaffaqiyatli
- **Resurs samaradorligi:** Aqlli backoff tizimlarni ortiqcha yuklashdan saqlaydi
- **Kuzatish:** Qancha qayta urinishlar kerak bo'lganini bilish

**Umumiy foydalanish stsenariylari:**
- **API chaqiruvlari:** Tarmoq nosozliklarida yoki 5xx xatolarida qayta urinish
- **DB operatsiyalari:** Ulanish muammolarida qayta urinish
- **Xabar navbatlari:** Xabar yetkazishni qayta urinish
- **Fayl operatsiyalari:** Vaqtinchalik fayl bloklarida qayta urinish
- **Kesh operatsiyalari:** Vaqtinchalik mavjud bo'lmaganda qayta urinish
- **Service Discovery:** Mavjud xizmatni topishni qayta urinish

**Qayta urinish strategiyalari:**
- **Fixed Delay:** Qayta urinishlar o'rtasida bir xil kechikish
- **Exponential Backoff:** 100ms, 200ms, 400ms, 800ms...
- **Exponential with Jitter:** Thundering herd oldini olish uchun tasodifiylik qo'shish
- **Fibonacci Backoff:** 100ms, 100ms, 200ms, 300ms, 500ms...

**Eng yaxshi amaliyotlar:**
- **Idempotent operatsiyalar:** Ko'p qayta urinishlar uchun xavfsiz
- **Har urinish uchun timeout:** Har bir urinish o'z timeoutiga ega
- **Qayta urinishlarni loglash:** Qayta urinish patternlarini kuzatish
- **Circuit Breaker:** Xizmat ishlamasa qayta urinishni to'xtatish
- **Backoff:** Ishlamayotgan xizmatlarni bombardimon qilmaslik

**Anti-patternlar:**
- **Cheksiz qayta urinishlar:** Har doim maksimal urinishlarga ega bo'ling
- **Kechikishsiz:** Har doim urinishlar o'rtasida kuting
- **Kontekstni e'tiborsiz qoldirish:** Har doim bekor qilinishni hurmat qiling
- **Bir xato:** Vaqtinchalik va doimiy xatolarni farqlang

RetryWithContext bo'lmasa, to'g'ri bekor qilish, backoff va cleanup bilan ishonchli qayta urinish mantiqini amalga oshirish murakkab va xatolarga moyildir.`,
			solutionCode: `package concurrency

import (
	"context"
	"time"
)

func RetryWithContext(ctx context.Context, attempts int, delay time.Duration, fn func(context.Context) error) error {
	if ctx == nil {                                             // nil kontekstni ishlash
		ctx = context.Background()                          // Fallback sifatida Background ishlatamiz
	}
	if attempts <= 0 {                                          // Urinishlar so'ralmagan
		return nil                                          // Darhol qaytamiz
	}
	var lastErr error                                           // Oxirgi xatoni kuzatamiz
	for attempt := 0; attempt < attempts; attempt++ {           // attempts martagacha sinab ko'ramiz
		if err := ctx.Err(); err != nil {                   // Urinishdan oldin kontekstni tekshiramiz
			return err                                  // Kontekst xatosini qaytaramiz
		}
		if err := fn(ctx); err == nil {                     // Funksiyani sinab ko'ramiz
			return nil                                  // Muvaffaqiyat, darhol qaytamiz
		} else {
			lastErr = err                               // Xatoni keyinroq uchun saqlaymiz
		}
		if attempt == attempts-1 {                          // Oxirgi urinish
			break                                       // Oxirgidan keyin kechiktirmaymiz
		}
		timer := time.NewTimer(delay)                       // Kechikish uchun timer yaratamiz
		select {
		case <-ctx.Done():                                  // Kechikish paytida kontekst bekor qilindi
			if !timer.Stop() {                          // Timerni to'xtatishga harakat qilamiz
				<-timer.C                           // Juda kech bo'lsa kanalni drenaj qilamiz
			}
			return ctx.Err()                            // Kontekst xatosini qaytaramiz
		case <-timer.C:                                     // Kechikish tugadi
		}
	}
	return lastErr                                              // Barcha urinishlar muvaffaqiyatsiz, oxirgi xatoni qaytaramiz
}`
		}
	}
};

export default task;
