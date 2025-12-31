import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-timeout',
	title: 'Timeout Middleware',
	difficulty: 'medium',	tags: ['go', 'http', 'middleware', 'context'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Timeout** middleware that creates a child context with timeout and passes it to the next handler.

**Requirements:**
1. Create function \`Timeout(d time.Duration, next http.Handler) http.Handler\`
2. Skip middleware if duration <= 0
3. Create child context with timeout using context.WithTimeout
4. Always call cancel() in defer
5. Pass request with timeout context to next handler
6. Handle nil next handler

**Example:**
\`\`\`go
handler := Timeout(2*time.Second, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    select {
    case <-time.After(3 * time.Second):
        fmt.Fprintf(w, "Never reached")
    case <-r.Context().Done():
	// Context cancelled after 2s timeout
        return
    }
}))

// Request is cancelled after 2 seconds
\`\`\`

**Constraints:**
- Must use context.WithTimeout to create timeout context
- Must call cancel() in defer to prevent context leak
- Must pass modified request with timeout context`,
	initialCode: `package httpx

import (
	"context"
	"net/http"
	"time"
)

// TODO: Implement Timeout middleware
func Timeout(d time.Duration, next http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"context"
	"net/http"
	"time"
)

func Timeout(d time.Duration, next http.Handler) http.Handler {
	if next == nil {	// Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if d <= 0 {	// Check if duration is valid
			next.ServeHTTP(w, r)	// Skip timeout if invalid
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), d)	// Create timeout context
		defer cancel()	// Always cancel to prevent leak
		next.ServeHTTP(w, r.WithContext(ctx))	// Pass request with timeout context
	})
}`,
			hint1: `Use context.WithTimeout(r.Context(), d) to create a timeout context from the request context.`,
			hint2: `Always defer cancel() immediately after creating the context to prevent resource leaks.`,
			testCode: `package httpx

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	h := Timeout(100*time.Millisecond, handler)
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rec.Code)
	}
}

func Test2(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		deadline, ok := r.Context().Deadline()
		if !ok {
			t.Error("expected deadline to be set")
		}
		if time.Until(deadline) > 100*time.Millisecond {
			t.Error("deadline too far in future")
		}
	})
	h := Timeout(100*time.Millisecond, handler)
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
}

func Test3(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, ok := r.Context().Deadline()
		if ok {
			t.Error("expected no deadline when duration <= 0")
		}
		w.WriteHeader(http.StatusOK)
	})
	h := Timeout(0, handler)
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rec.Code)
	}
}

func Test4(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, ok := r.Context().Deadline()
		if ok {
			t.Error("expected no deadline when duration < 0")
		}
	})
	h := Timeout(-1*time.Second, handler)
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
}

func Test5(t *testing.T) {
	h := Timeout(100*time.Millisecond, nil)
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected 200 for nil handler, got %d", rec.Code)
	}
}

func Test6(t *testing.T) {
	done := make(chan struct{})
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
			close(done)
		case <-time.After(1 * time.Second):
			t.Error("handler should have been cancelled")
		}
	})
	h := Timeout(50*time.Millisecond, handler)
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	select {
	case <-done:
	case <-time.After(200 * time.Millisecond):
		t.Error("context was not cancelled")
	}
}

func Test7(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Test", "value")
		w.WriteHeader(http.StatusCreated)
	})
	h := Timeout(100*time.Millisecond, handler)
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Errorf("expected 201, got %d", rec.Code)
	}
	if rec.Header().Get("X-Test") != "value" {
		t.Error("expected header to be preserved")
	}
}

func Test8(t *testing.T) {
	parentCtx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	var ctxErr error
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
		ctxErr = r.Context().Err()
	})
	h := Timeout(50*time.Millisecond, handler)
	req := httptest.NewRequest("GET", "/", nil).WithContext(parentCtx)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if ctxErr != context.DeadlineExceeded {
		t.Errorf("expected DeadlineExceeded, got %v", ctxErr)
	}
}

func Test9(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		deadline, _ := r.Context().Deadline()
		if time.Until(deadline) < 400*time.Millisecond {
			t.Error("deadline should be ~500ms in future")
		}
	})
	h := Timeout(500*time.Millisecond, handler)
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
}

func Test10(t *testing.T) {
	counter := 0
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		counter++
		w.WriteHeader(http.StatusOK)
	})
	h := Timeout(100*time.Millisecond, handler)
	for i := 0; i < 3; i++ {
		req := httptest.NewRequest("GET", "/", nil)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
	}
	if counter != 3 {
		t.Errorf("expected 3 calls, got %d", counter)
	}
}
`,
			whyItMatters: `Timeout middleware prevents slow handlers from blocking resources indefinitely, ensuring responsive services and preventing resource leaks.

**Why Request Timeouts:**
- **Resource Protection:** Prevent handlers from running forever
- **Responsiveness:** Fail fast instead of hanging indefinitely
- **Client Experience:** Return errors quickly vs waiting forever
- **Resource Cleanup:** Cancel database queries, API calls when timeout

**Production Pattern:**
\`\`\`go
// Different timeouts for different endpoints
func Router() *http.ServeMux {
    mux := http.NewServeMux()

	// Fast endpoints: short timeout
    mux.Handle("/health", Timeout(1*time.Second, healthHandler))

	// Normal endpoints: medium timeout
    mux.Handle("/api/users", Timeout(5*time.Second, usersHandler))

	// Slow endpoints: long timeout
    mux.Handle("/reports", Timeout(30*time.Second, reportsHandler))

	// Background jobs: no timeout
    mux.Handle("/jobs", longRunningHandler)

    return mux
}

// Context-aware database query
func GetUser(ctx context.Context, userID string) (*User, error) {
	// Query will be cancelled if context times out
    query := "SELECT * FROM users WHERE id = $1"

    var user User
    err := db.QueryRowContext(ctx, query, userID).Scan(&user.ID, &user.Name)
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            return nil, fmt.Errorf("query timeout: %w", ctx.Err())
        }
        return nil, err
    }

    return &user, nil
}

// Timeout with custom error handler
func TimeoutWithHandler(d time.Duration, timeoutHandler http.Handler) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            ctx, cancel := context.WithTimeout(r.Context(), d)
            defer cancel()

            done := make(chan struct{})
            go func() {
                next.ServeHTTP(w, r.WithContext(ctx))
                close(done)
            }()

            select {
            case <-done:
	// Request completed successfully
            case <-ctx.Done():
	// Timeout occurred
                timeoutHandler.ServeHTTP(w, r)
            }
        })
    }
}

// Graceful timeout (allow cleanup)
func GracefulTimeout(requestTimeout, cleanupTimeout time.Duration) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Request context with timeout
            reqCtx, reqCancel := context.WithTimeout(r.Context(), requestTimeout)
            defer reqCancel()

	// Cleanup context with additional time
            cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), cleanupTimeout)
            defer cleanupCancel()

            done := make(chan struct{})
            go func() {
                next.ServeHTTP(w, r.WithContext(reqCtx))
                close(done)
            }()

            select {
            case <-done:
	// Completed successfully
            case <-reqCtx.Done():
	// Request timeout, allow cleanup
                log.Printf("Request timeout, allowing %v for cleanup", cleanupTimeout)
                select {
                case <-done:
                    log.Printf("Cleanup completed")
                case <-cleanupCtx.Done():
                    log.Printf("Cleanup timeout exceeded")
                }
                http.Error(w, "request timeout", http.StatusRequestTimeout)
            }
        })
    }
}

// Dynamic timeout based on endpoint
func DynamicTimeout(getTimeout func(*http.Request) time.Duration) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            timeout := getTimeout(r)
            if timeout > 0 {
                Timeout(timeout, next).ServeHTTP(w, r)
            } else {
                next.ServeHTTP(w, r)
            }
        })
    }
}

// Usage: Timeout based on user tier
timeoutMiddleware := DynamicTimeout(func(r *http.Request) time.Duration {
    tier := r.Context().Value("user_tier").(string)
    switch tier {
    case "premium":
        return 30 * time.Second
    case "standard":
        return 10 * time.Second
    default:
        return 5 * time.Second
    }
})

// External API call with timeout
func CallExternalAPI(ctx context.Context, url string) (*Response, error) {
    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }

    client := &http.Client{Timeout: 10 * time.Second}
    resp, err := client.Do(req)
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            return nil, fmt.Errorf("API call timeout: %w", err)
        }
        return nil, err
    }
    defer resp.Body.Close()

	// Process response...
    return processResponse(resp)
}

// Database transaction with timeout
func ExecuteTransaction(ctx context.Context, fn func(tx *sql.Tx) error) error {
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }

    defer tx.Rollback()

    if err := fn(tx); err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            return fmt.Errorf("transaction timeout: %w", err)
        }
        return err
    }

    return tx.Commit()
}
\`\`\`

**Real-World Benefits:**
- **Server Protection:** Prevent resource exhaustion from slow operations
- **Cascade Prevention:** Stop timeouts from propagating to dependencies
- **SLA Compliance:** Enforce maximum response times
- **Cost Control:** Limit expensive operation durations

**Context Timeout Best Practices:**
- **Always Cancel:** Defer cancel() to prevent context leaks
- **Check Context:** Handlers should check ctx.Done() regularly
- **Propagate Context:** Pass context to all downstream operations
- **Set Realistic Timeouts:** Too short = false failures, too long = resource waste

**Common Timeout Values:**
- **Health Checks:** 1-2 seconds
- **API Endpoints:** 5-10 seconds
- **Database Queries:** 5-30 seconds
- **External APIs:** 10-30 seconds
- **Reports/Exports:** 30-300 seconds
- **Background Jobs:** No timeout or hours

**Context Cancellation Signals:**
- **Deadline Exceeded:** Timeout occurred
- **Canceled:** Explicit cancellation
- **Database Drivers:** Honor context cancellation
- **HTTP Clients:** Cancel in-flight requests

Without Timeout middleware, slow operations can accumulate, exhausting connections and memory, eventually crashing the server.`,	order: 15,
	translations: {
		ru: {
			title: 'Установка таймаута для обработки запроса',
			description: `Реализуйте middleware **Timeout**, который создаёт дочерний контекст с тайм-аутом и передаёт его следующему handler.

**Требования:**
1. Создайте функцию \`Timeout(d time.Duration, next http.Handler) http.Handler\`
2. Пропустите middleware если duration <= 0
3. Создайте дочерний контекст с тайм-аутом используя context.WithTimeout
4. Всегда вызывайте cancel() в defer
5. Передайте запрос с timeout контекстом следующему handler
6. Обработайте nil handler

**Пример:**
\`\`\`go
handler := Timeout(2*time.Second, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    select {
    case <-time.After(3 * time.Second):
        fmt.Fprintf(w, "Never reached")
    case <-r.Context().Done():
	// Контекст отменён после 2s таймаута
        return
    }
}))

// Запрос отменяется через 2 секунды
\`\`\`

**Ограничения:**
- Должен использовать context.WithTimeout для создания timeout контекста
- Должен вызывать cancel() в defer для предотвращения утечки контекста
- Должен передавать модифицированный запрос с timeout контекстом`,
			hint1: `Используйте context.WithTimeout(r.Context(), d) для создания timeout контекста из контекста запроса.`,
			hint2: `Всегда defer cancel() сразу после создания контекста для предотвращения утечек ресурсов.`,
			whyItMatters: `Timeout middleware предотвращает блокирование ресурсов медленными handlers навечно, обеспечивая отзывчивые сервисы и предотвращая утечки ресурсов.

**Почему Request Timeouts:**
- **Защита ресурсов:** Предотвращение вечного выполнения handlers
- **Отзывчивость:** Быстрый fail вместо бесконечного ожидания
- **Опыт клиента:** Быстрый возврат ошибок vs вечное ожидание
- **Очистка ресурсов:** Отмена database запросов, API вызовов при таймауте

**Продакшен паттерн:**
\`\`\`go
// Разные таймауты для разных endpoints
func Router() *http.ServeMux {
    mux := http.NewServeMux()

	// Быстрые endpoints: короткий таймаут
    mux.Handle("/health", Timeout(1*time.Second, healthHandler))

	// Обычные endpoints: средний таймаут
    mux.Handle("/api/users", Timeout(5*time.Second, usersHandler))

	// Медленные endpoints: длинный таймаут
    mux.Handle("/reports", Timeout(30*time.Second, reportsHandler))

	// Фоновые задачи: без таймаута
    mux.Handle("/jobs", longRunningHandler)

    return mux
}

// Context-aware database запрос
func GetUser(ctx context.Context, userID string) (*User, error) {
	// Запрос будет отменён если контекст истечёт
    query := "SELECT * FROM users WHERE id = $1"

    var user User
    err := db.QueryRowContext(ctx, query, userID).Scan(&user.ID, &user.Name)
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            return nil, fmt.Errorf("query timeout: %w", ctx.Err())
        }
        return nil, err
    }

    return &user, nil
}

// Timeout с кастомным обработчиком ошибок
func TimeoutWithHandler(d time.Duration, timeoutHandler http.Handler) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            ctx, cancel := context.WithTimeout(r.Context(), d)
            defer cancel()

            done := make(chan struct{})
            go func() {
                next.ServeHTTP(w, r.WithContext(ctx))
                close(done)
            }()

            select {
            case <-done:
	// Запрос выполнен успешно
            case <-ctx.Done():
	// Произошёл таймаут
                timeoutHandler.ServeHTTP(w, r)
            }
        })
    }
}

// Graceful timeout (позволяет очистку)
func GracefulTimeout(requestTimeout, cleanupTimeout time.Duration) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Контекст запроса с таймаутом
            reqCtx, reqCancel := context.WithTimeout(r.Context(), requestTimeout)
            defer reqCancel()

	// Контекст очистки с дополнительным временем
            cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), cleanupTimeout)
            defer cleanupCancel()

            done := make(chan struct{})
            go func() {
                next.ServeHTTP(w, r.WithContext(reqCtx))
                close(done)
            }()

            select {
            case <-done:
	// Завершено успешно
            case <-reqCtx.Done():
	// Таймаут запроса, разрешаем очистку
                log.Printf("Request timeout, allowing %v for cleanup", cleanupTimeout)
                select {
                case <-done:
                    log.Printf("Cleanup completed")
                case <-cleanupCtx.Done():
                    log.Printf("Cleanup timeout exceeded")
                }
                http.Error(w, "request timeout", http.StatusRequestTimeout)
            }
        })
    }
}

// Динамический таймаут на основе endpoint
func DynamicTimeout(getTimeout func(*http.Request) time.Duration) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            timeout := getTimeout(r)
            if timeout > 0 {
                Timeout(timeout, next).ServeHTTP(w, r)
            } else {
                next.ServeHTTP(w, r)
            }
        })
    }
}

// Использование: Таймаут на основе уровня пользователя
timeoutMiddleware := DynamicTimeout(func(r *http.Request) time.Duration {
    tier := r.Context().Value("user_tier").(string)
    switch tier {
    case "premium":
        return 30 * time.Second
    case "standard":
        return 10 * time.Second
    default:
        return 5 * time.Second
    }
})

// Внешний API вызов с таймаутом
func CallExternalAPI(ctx context.Context, url string) (*Response, error) {
    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }

    client := &http.Client{Timeout: 10 * time.Second}
    resp, err := client.Do(req)
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            return nil, fmt.Errorf("API call timeout: %w", err)
        }
        return nil, err
    }
    defer resp.Body.Close()

	// Обработка ответа...
    return processResponse(resp)
}

// Database транзакция с таймаутом
func ExecuteTransaction(ctx context.Context, fn func(tx *sql.Tx) error) error {
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }

    defer tx.Rollback()

    if err := fn(tx); err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            return fmt.Errorf("transaction timeout: %w", err)
        }
        return err
    }

    return tx.Commit()
}
\`\`\`

**Практические преимущества:**
- **Защита сервера:** Предотвращение исчерпания ресурсов медленными операциями
- **Предотвращение каскадов:** Остановка таймаутов от распространения на зависимости
- **SLA Compliance:** Принудительное соблюдение максимального времени ответа
- **Контроль затрат:** Ограничение длительности дорогих операций

**Best practices для Context Timeout:**
- **Всегда Cancel:** Defer cancel() для предотвращения утечек контекста
- **Проверяйте Context:** Handlers должны регулярно проверять ctx.Done()
- **Пропагируйте Context:** Передавайте context всем downstream операциям
- **Реалистичные таймауты:** Слишком короткие = ложные сбои, слишком длинные = трата ресурсов

**Типичные значения таймаутов:**
- **Health Checks:** 1-2 секунды
- **API Endpoints:** 5-10 секунд
- **Database Queries:** 5-30 секунд
- **External APIs:** 10-30 секунд
- **Reports/Exports:** 30-300 секунд
- **Background Jobs:** Без таймаута или часы

**Сигналы отмены контекста:**
- **Deadline Exceeded:** Произошёл таймаут
- **Canceled:** Явная отмена
- **Database Drivers:** Соблюдают отмену контекста
- **HTTP Clients:** Отменяют выполняемые запросы

Без Timeout middleware медленные операции могут накапливаться, исчерпывая соединения и память, в конечном счёте крашнув сервер.`,
			solutionCode: `package httpx

import (
	"context"
	"net/http"
	"time"
)

func Timeout(d time.Duration, next http.Handler) http.Handler {
	if next == nil {	// Обработка nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if d <= 0 {	// Проверка валидности длительности
			next.ServeHTTP(w, r)	// Пропуск таймаута если невалидный
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), d)	// Создание контекста с таймаутом
		defer cancel()	// Всегда отменяйте чтобы предотвратить утечку
		next.ServeHTTP(w, r.WithContext(ctx))	// Передача запроса с контекстом таймаута
	})
}`
		},
		uz: {
			title: 'Request qayta ishlash uchun timeout o\'rnatish',
			description: `Timeout bilan child kontekstni yaratuvchi va uni keyingi handlerga o'tkazuvchi **Timeout** middleware ni amalga oshiring.

**Talablar:**
1. \`Timeout(d time.Duration, next http.Handler) http.Handler\` funksiyasini yarating
2. Agar duration <= 0 bo'lsa middleware ni o'tkazing
3. context.WithTimeout dan foydalanib timeout bilan child kontekstni yarating
4. Har doim defer da cancel() ni chaqiring
5. Timeout konteksti bilan requestni keyingi handlerga o'tkazing
6. nil handlerni ishlang

**Misol:**
\`\`\`go
handler := Timeout(2*time.Second, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    select {
    case <-time.After(3 * time.Second):
        fmt.Fprintf(w, "Never reached")
    case <-r.Context().Done():
	// 2s timeoutdan keyin kontekst bekor qilindi
        return
    }
}))

// 2 sekunddan keyin request bekor qilinadi
\`\`\`

**Cheklovlar:**
- Timeout kontekstni yaratish uchun context.WithTimeout dan foydalanishi kerak
- Kontekst oqishining oldini olish uchun defer da cancel() ni chaqirishi kerak
- Timeout konteksti bilan o'zgartirilgan requestni o'tkazishi kerak`,
			hint1: `Request kontekstidan timeout kontekstni yaratish uchun context.WithTimeout(r.Context(), d) dan foydalaning.`,
			hint2: `Resurs oqishining oldini olish uchun kontekst yaratilgandan keyin darhol defer cancel() qiling.`,
			whyItMatters: `Timeout middleware sekin handlerlar resurslarni abadiy bloklashining oldini oladi, tezkor xizmatlarni ta'minlaydi va resurs oqishining oldini oladi.

**Nima uchun Request Timeouts:**
- **Resurslarni himoya qilish:** Handlerlarning abadiy bajarilishining oldini olish
- **Tezkorlik:** Abadiy kutish o'rniga tez fail
- **Mijoz tajribasi:** Abadiy kutish o'rniga tez xato qaytarish
- **Resurslarni tozalash:** Timeoutda database so'rovlari, API chaqiruvlarini bekor qilish

**Ishlab chiqarish patterni:**
\`\`\`go
// Turli endpointlar uchun turli timeoutlar
func Router() *http.ServeMux {
    mux := http.NewServeMux()

	// Tez endpointlar: qisqa timeout
    mux.Handle("/health", Timeout(1*time.Second, healthHandler))

	// Oddiy endpointlar: o'rtacha timeout
    mux.Handle("/api/users", Timeout(5*time.Second, usersHandler))

	// Sekin endpointlar: uzun timeout
    mux.Handle("/reports", Timeout(30*time.Second, reportsHandler))

	// Fon vazifalar: timeoutsiz
    mux.Handle("/jobs", longRunningHandler)

    return mux
}

// Context-aware database so'rovi
func GetUser(ctx context.Context, userID string) (*User, error) {
	// Kontekst tugasa so'rov bekor qilinadi
    query := "SELECT * FROM users WHERE id = $1"

    var user User
    err := db.QueryRowContext(ctx, query, userID).Scan(&user.ID, &user.Name)
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            return nil, fmt.Errorf("query timeout: %w", ctx.Err())
        }
        return nil, err
    }

    return &user, nil
}

// Maxsus xato ishlovchisi bilan timeout
func TimeoutWithHandler(d time.Duration, timeoutHandler http.Handler) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            ctx, cancel := context.WithTimeout(r.Context(), d)
            defer cancel()

            done := make(chan struct{})
            go func() {
                next.ServeHTTP(w, r.WithContext(ctx))
                close(done)
            }()

            select {
            case <-done:
	// Request muvaffaqiyatli tugadi
            case <-ctx.Done():
	// Timeout yuz berdi
                timeoutHandler.ServeHTTP(w, r)
            }
        })
    }
}

// Graceful timeout (tozalashga ruxsat beradi)
func GracefulTimeout(requestTimeout, cleanupTimeout time.Duration) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Timeout bilan request konteksti
            reqCtx, reqCancel := context.WithTimeout(r.Context(), requestTimeout)
            defer reqCancel()

	// Qo'shimcha vaqt bilan tozalash konteksti
            cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), cleanupTimeout)
            defer cleanupCancel()

            done := make(chan struct{})
            go func() {
                next.ServeHTTP(w, r.WithContext(reqCtx))
                close(done)
            }()

            select {
            case <-done:
	// Muvaffaqiyatli tugadi
            case <-reqCtx.Done():
	// Request timeoutga tushdi, tozalashga ruxsat berildi
                log.Printf("Request timeout, allowing %v for cleanup", cleanupTimeout)
                select {
                case <-done:
                    log.Printf("Cleanup completed")
                case <-cleanupCtx.Done():
                    log.Printf("Cleanup timeout exceeded")
                }
                http.Error(w, "request timeout", http.StatusRequestTimeout)
            }
        })
    }
}

// Endpointga asoslangan dinamik timeout
func DynamicTimeout(getTimeout func(*http.Request) time.Duration) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            timeout := getTimeout(r)
            if timeout > 0 {
                Timeout(timeout, next).ServeHTTP(w, r)
            } else {
                next.ServeHTTP(w, r)
            }
        })
    }
}

// Foydalanish: Foydalanuvchi darajasiga asoslangan timeout
timeoutMiddleware := DynamicTimeout(func(r *http.Request) time.Duration {
    tier := r.Context().Value("user_tier").(string)
    switch tier {
    case "premium":
        return 30 * time.Second
    case "standard":
        return 10 * time.Second
    default:
        return 5 * time.Second
    }
})

// Timeout bilan tashqi API chaqiruvi
func CallExternalAPI(ctx context.Context, url string) (*Response, error) {
    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }

    client := &http.Client{Timeout: 10 * time.Second}
    resp, err := client.Do(req)
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            return nil, fmt.Errorf("API call timeout: %w", err)
        }
        return nil, err
    }
    defer resp.Body.Close()

	// Javobni qayta ishlash...
    return processResponse(resp)
}

// Timeout bilan database tranzaksiyasi
func ExecuteTransaction(ctx context.Context, fn func(tx *sql.Tx) error) error {
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }

    defer tx.Rollback()

    if err := fn(tx); err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            return fmt.Errorf("transaction timeout: %w", err)
        }
        return err
    }

    return tx.Commit()
}
\`\`\`

**Amaliy foydalari:**
- **Server himoyasi:** Sekin operatsiyalardan resurslar tugashining oldini olish
- **Kaskadlarni oldini olish:** Timeoutlarning bog'liqliklarga tarqalishini to'xtatish
- **SLA Compliance:** Maksimal javob vaqtini majburiy qilish
- **Xarajatlarni nazorat qilish:** Qimmat operatsiyalar davomiyligini cheklash

**Context Timeout uchun best practices:**
- **Har doim Cancel:** Kontekst oqishining oldini olish uchun defer cancel()
- **Contextni tekshiring:** Handlerlar muntazam ctx.Done() ni tekshirishi kerak
- **Contextni tarqating:** Barcha downstream operatsiyalarga contextni o'tkazing
- **Realistik timeoutlar:** Juda qisqa = noto'g'ri xatolar, juda uzun = resurslarni isrof qilish

**Odatiy timeout qiymatlari:**
- **Health Checks:** 1-2 sekund
- **API Endpointlari:** 5-10 sekund
- **Database So'rovlari:** 5-30 sekund
- **Tashqi API lar:** 10-30 sekund
- **Hisobotlar/Eksportlar:** 30-300 sekund
- **Fon Vazifalar:** Timeoutsiz yoki soatlar

**Kontekst bekor qilish signallari:**
- **Deadline Exceeded:** Timeout yuz berdi
- **Canceled:** Aniq bekor qilish
- **Database Drivers:** Kontekst bekor qilishga rioya qiladi
- **HTTP Clients:** Bajarilayotgan requestlarni bekor qiladi

Timeout middleware siz sekin operatsiyalar to'planishi, ulanishlar va xotirani tugashiga va oxir-oqibat serverni crashlashga olib kelishi mumkin.`,
			solutionCode: `package httpx

import (
	"context"
	"net/http"
	"time"
)

func Timeout(d time.Duration, next http.Handler) http.Handler {
	if next == nil {	// nil next handlerni ishlash
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if d <= 0 {	// Davomiylik haqiqiyligini tekshirish
			next.ServeHTTP(w, r)	// Agar noto'g'ri bo'lsa timeoutni o'tkazish
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), d)	// Timeout bilan kontekst yaratish
		defer cancel()	// Oqishni oldini olish uchun har doim bekor qiling
		next.ServeHTTP(w, r.WithContext(ctx))	// Timeout konteksti bilan requestni o'tkazish
	})
}`
		}
	}
};

export default task;
