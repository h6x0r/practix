import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-concurrency-limit',
	title: 'Concurrency Limit Middleware',
	difficulty: 'medium',	tags: ['go', 'http', 'middleware', 'concurrency'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **ConcurrencyLimit** middleware that restricts the number of concurrent request handlers using a semaphore pattern.

**Requirements:**
1. Create function \`ConcurrencyLimit(limit int, next http.Handler) http.Handler\`
2. Skip middleware if limit <= 0
3. Use buffered channel as semaphore
4. Acquire slot before calling next
5. Release slot after next completes (use defer)
6. Block new requests when limit is reached
7. Handle nil next handler

**Example:**
\`\`\`go
handler := ConcurrencyLimit(2, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    time.Sleep(1 * time.Second) // Simulate slow handler
    fmt.Fprintf(w, "Done")
}))

// Only 2 requests run concurrently, others wait for a slot
\`\`\`

**Constraints:**
- Must use buffered channel as semaphore
- Must release slot even if handler panics (use defer)
- Must not start processing until slot is available`,
	initialCode: `package httpx

import (
	"net/http"
)

// TODO: Implement ConcurrencyLimit middleware
func ConcurrencyLimit(limit int, next http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"net/http"
)

func ConcurrencyLimit(limit int, next http.Handler) http.Handler {
	if next == nil {	// Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	if limit <= 0 {	// Check if limit is valid
		return next	// Skip middleware if no limit
	}
	sem := make(chan struct{}, limit)	// Create buffered channel as semaphore
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sem <- struct{}{}	// Acquire slot (blocks if full)
		defer func() { <-sem }()	// Release slot after handler completes
		next.ServeHTTP(w, r)	// Execute handler
	})
}`,
			hint1: `Create a buffered channel with size=limit. Send to acquire, receive in defer to release.`,
			hint2: `Use defer func() { <-sem }() to ensure slot is released even if handler panics.`,
			testCode: `package httpx

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Test middleware returns non-nil handler
	h := ConcurrencyLimit(2, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	if h == nil {
		t.Error("handler should not be nil")
	}
}

func Test2(t *testing.T) {
	// Test nil handler returns non-nil
	h := ConcurrencyLimit(2, nil)
	if h == nil {
		t.Error("nil handler should return non-nil")
	}
}

func Test3(t *testing.T) {
	// Test limit <= 0 skips middleware
	called := false
	h := ConcurrencyLimit(0, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil))
	if !called {
		t.Error("limit 0 should skip middleware")
	}
}

func Test4(t *testing.T) {
	// Test single request passes through
	h := ConcurrencyLimit(1, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Body.String() != "ok" {
		t.Error("single request should pass through")
	}
}

func Test5(t *testing.T) {
	// Test concurrent requests limited
	var current int32
	var maxConcurrent int32
	h := ConcurrencyLimit(2, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c := atomic.AddInt32(&current, 1)
		if c > atomic.LoadInt32(&maxConcurrent) {
			atomic.StoreInt32(&maxConcurrent, c)
		}
		time.Sleep(50 * time.Millisecond)
		atomic.AddInt32(&current, -1)
	}))
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil))
		}()
	}
	wg.Wait()
	if maxConcurrent > 2 {
		t.Errorf("max concurrent = %d, want <= 2", maxConcurrent)
	}
}

func Test6(t *testing.T) {
	// Test method preserved
	var method string
	h := ConcurrencyLimit(1, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method = r.Method
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", nil))
	if method != "POST" {
		t.Errorf("method = %q, want POST", method)
	}
}

func Test7(t *testing.T) {
	// Test headers preserved
	var header string
	h := ConcurrencyLimit(1, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		header = r.Header.Get("X-Custom")
	}))
	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("X-Custom", "value")
	h.ServeHTTP(httptest.NewRecorder(), req)
	if header != "value" {
		t.Errorf("header = %q, want 'value'", header)
	}
}

func Test8(t *testing.T) {
	// Test response status preserved
	h := ConcurrencyLimit(1, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Code != http.StatusCreated {
		t.Errorf("status = %d, want 201", rec.Code)
	}
}

func Test9(t *testing.T) {
	// Test negative limit skips middleware
	called := false
	h := ConcurrencyLimit(-1, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil))
	if !called {
		t.Error("negative limit should skip middleware")
	}
}

func Test10(t *testing.T) {
	// Test multiple sequential requests work
	count := 0
	h := ConcurrencyLimit(1, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count++
	}))
	for i := 0; i < 3; i++ {
		h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil))
	}
	if count != 3 {
		t.Errorf("count = %d, want 3", count)
	}
}`,
			whyItMatters: `ConcurrencyLimit protects servers from overload by limiting concurrent request processing, preventing resource exhaustion and maintaining performance.

**Why Concurrency Limiting:**
- **Resource Protection:** Prevent memory/CPU exhaustion
- **Stability:** Avoid cascading failures from overload
- **Fair Service:** Ensure consistent response times
- **Graceful Degradation:** Queue requests instead of crashing

**Production Pattern:**
\`\`\`go
// Protect expensive endpoints
expensiveHandler := ConcurrencyLimit(10, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Expensive database query or computation
    result := expensiveQuery(r.Context())
    json.NewEncoder(w).Encode(result)
}))

// Different limits for different endpoints
func Router() *http.ServeMux {
    mux := http.NewServeMux()

	// High limit for cheap endpoints
    mux.Handle("/health", ConcurrencyLimit(1000, healthHandler))

	// Low limit for expensive endpoints
    mux.Handle("/report", ConcurrencyLimit(5, reportHandler))
    mux.Handle("/export", ConcurrencyLimit(3, exportHandler))

	// No limit for static files
    mux.Handle("/static/", http.FileServer(http.Dir("./static")))

    return mux
}

// Dynamic limits based on system load
func AdaptiveConcurrencyLimit(baseLimit int) func(http.Handler) http.Handler {
    currentLimit := baseLimit

	// Monitor system and adjust limit
    go func() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()

        for range ticker.C {
            load := getSystemLoad()
            if load > 0.8 {
                currentLimit = max(1, currentLimit-1) // Decrease limit
            } else if load < 0.5 && currentLimit < baseLimit {
                currentLimit++ // Increase limit
            }
        }
    }()

    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            sem := make(chan struct{}, currentLimit)
            handler := ConcurrencyLimit(currentLimit, next)
            handler.ServeHTTP(w, r)
        })
    }
}

// Timeout + Concurrency limit
func ProtectedHandler(limit int, timeout time.Duration) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return Chain(
            ConcurrencyLimit(limit),
            Timeout(timeout),
        )(next)
    }
}

// Per-user concurrency limits
func PerUserConcurrencyLimit(limit int) func(http.Handler) http.Handler {
    userSems := sync.Map{} // map[string]chan struct{}

    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            userID := r.Context().Value("user_id").(string)

	// Get or create semaphore for user
            val, _ := userSems.LoadOrStore(userID, make(chan struct{}, limit))
            sem := val.(chan struct{})

            select {
            case sem <- struct{}{}:
                defer func() { <-sem }()
                next.ServeHTTP(w, r)
            case <-time.After(5 * time.Second):
                http.Error(w, "too many concurrent requests", http.StatusTooManyRequests)
            }
        })
    }
}

// Priority-based concurrency
type PriorityLimit struct {
    highPriority chan struct{}
    lowPriority  chan struct{}
}

func NewPriorityLimit(highLimit, lowLimit int) *PriorityLimit {
    return &PriorityLimit{
        highPriority: make(chan struct{}, highLimit),
        lowPriority:  make(chan struct{}, lowLimit),
    }
}

func (pl *PriorityLimit) Middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        priority := r.Header.Get("X-Priority")

        var sem chan struct{}
        if priority == "high" {
            sem = pl.highPriority
        } else {
            sem = pl.lowPriority
        }

        select {
        case sem <- struct{}{}:
            defer func() { <-sem }()
            next.ServeHTTP(w, r)
        case <-time.After(10 * time.Second):
            http.Error(w, "service unavailable", http.StatusServiceUnavailable)
        }
    })
}

// Metrics tracking
func ConcurrencyLimitWithMetrics(limit int, next http.Handler) http.Handler {
    sem := make(chan struct{}, limit)

    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Record current concurrency
        currentConcurrency := len(sem)
        metrics.RecordGauge("http_concurrent_requests", float64(currentConcurrency), map[string]string{
            "path": r.URL.Path,
        })

	// Try to acquire with timeout
        select {
        case sem <- struct{}{}:
            defer func() { <-sem }()
            next.ServeHTTP(w, r)
        case <-time.After(30 * time.Second):
            metrics.IncrementCounter("http_concurrency_limit_rejections", map[string]string{
                "path": r.URL.Path,
            })
            http.Error(w, "too many requests", http.StatusTooManyRequests)
        }
    })
}
\`\`\`

**Real-World Benefits:**
- **Server Stability:** Prevent crashes from resource exhaustion
- **Predictable Performance:** Consistent response times under load
- **Cost Control:** Limit expensive operations (DB queries, external APIs)
- **Fair Access:** Prevent single users from monopolizing resources

**Semaphore Pattern:**
- **Buffered Channel:** Simple, Go-idiomatic semaphore
- **Blocking Acquire:** Natural backpressure mechanism
- **Defer Release:** Ensures slot is freed even on panic
- **Zero Overhead:** Channel operations are fast

**Common Limits:**
- **Database Endpoints:** 10-50 concurrent connections
- **External API Calls:** 5-20 concurrent requests
- **File Uploads:** 3-10 concurrent uploads
- **Heavy Computation:** 2-10 concurrent operations
- **Websockets:** 1000-10000 concurrent connections

**Important Notes:**
- **Too Low:** Underutilizes resources, increases latency
- **Too High:** Resource exhaustion, crashes
- **Monitor:** Track rejection rates, adjust limits
- **Timeouts:** Combine with timeouts to prevent indefinite waiting

Without ConcurrencyLimit, a spike in traffic can overwhelm the server, exhausting memory/connections and crashing the entire service.`,	order: 14,
	translations: {
		ru: {
			title: 'Ограничение числа параллельных запросов',
			description: `Реализуйте middleware **ConcurrencyLimit**, который ограничивает количество параллельных обработчиков запросов используя паттерн семафора.

**Требования:**
1. Создайте функцию \`ConcurrencyLimit(limit int, next http.Handler) http.Handler\`
2. Пропустите middleware если limit <= 0
3. Используйте буферизированный канал как семафор
4. Захватите слот перед вызовом next
5. Освободите слот после завершения next (используйте defer)
6. Блокируйте новые запросы когда достигнут лимит
7. Обработайте nil handler

**Пример:**
\`\`\`go
handler := ConcurrencyLimit(2, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    time.Sleep(1 * time.Second) // Симуляция медленного handler
    fmt.Fprintf(w, "Done")
}))

// Только 2 запроса выполняются параллельно, остальные ждут слот
\`\`\`

**Ограничения:**
- Должен использовать буферизированный канал как семафор
- Должен освобождать слот даже если handler паникует (использовать defer)
- Не должен начинать обработку пока слот не доступен`,
			hint1: `Создайте буферизированный канал с size=limit. Отправка для захвата, прием в defer для освобождения.`,
			hint2: `Используйте defer func() { <-sem }() для гарантии освобождения слота даже при панике handler.`,
			whyItMatters: `ConcurrencyLimit защищает серверы от перегрузки ограничивая параллельную обработку запросов, предотвращая исчерпание ресурсов и поддерживая производительность.

**Почему Concurrency Limiting:**
- **Защита ресурсов:** Предотвращение исчерпания memory/CPU
- **Стабильность:** Избежание каскадных сбоев от перегрузки
- **Справедливое обслуживание:** Обеспечение стабильных времён ответа
- **Graceful Degradation:** Очередь запросов вместо краша

**Продакшен паттерн:**
\`\`\`go
// Защита дорогих endpoints
expensiveHandler := ConcurrencyLimit(10, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Дорогой database запрос или вычисление
    result := expensiveQuery(r.Context())
    json.NewEncoder(w).Encode(result)
}))

// Разные лимиты для разных endpoints
func Router() *http.ServeMux {
    mux := http.NewServeMux()

	// Высокий лимит для дешевых endpoints
    mux.Handle("/health", ConcurrencyLimit(1000, healthHandler))

	// Низкий лимит для дорогих endpoints
    mux.Handle("/report", ConcurrencyLimit(5, reportHandler))
    mux.Handle("/export", ConcurrencyLimit(3, exportHandler))

	// Без лимита для статических файлов
    mux.Handle("/static/", http.FileServer(http.Dir("./static")))

    return mux
}

// Динамические лимиты на основе системной нагрузки
func AdaptiveConcurrencyLimit(baseLimit int) func(http.Handler) http.Handler {
    currentLimit := baseLimit

	// Мониторинг системы и корректировка лимита
    go func() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()

        for range ticker.C {
            load := getSystemLoad()
            if load > 0.8 {
                currentLimit = max(1, currentLimit-1) // Уменьшить лимит
            } else if load < 0.5 && currentLimit < baseLimit {
                currentLimit++ // Увеличить лимит
            }
        }
    }()

    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            sem := make(chan struct{}, currentLimit)
            handler := ConcurrencyLimit(currentLimit, next)
            handler.ServeHTTP(w, r)
        })
    }
}

// Timeout + Concurrency limit
func ProtectedHandler(limit int, timeout time.Duration) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return Chain(
            ConcurrencyLimit(limit),
            Timeout(timeout),
        )(next)
    }
}

// Per-user лимиты concurrency
func PerUserConcurrencyLimit(limit int) func(http.Handler) http.Handler {
    userSems := sync.Map{} // map[string]chan struct{}

    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            userID := r.Context().Value("user_id").(string)

	// Получить или создать семафор для пользователя
            val, _ := userSems.LoadOrStore(userID, make(chan struct{}, limit))
            sem := val.(chan struct{})

            select {
            case sem <- struct{}{}:
                defer func() { <-sem }()
                next.ServeHTTP(w, r)
            case <-time.After(5 * time.Second):
                http.Error(w, "too many concurrent requests", http.StatusTooManyRequests)
            }
        })
    }
}

// Concurrency на основе приоритета
type PriorityLimit struct {
    highPriority chan struct{}
    lowPriority  chan struct{}
}

func NewPriorityLimit(highLimit, lowLimit int) *PriorityLimit {
    return &PriorityLimit{
        highPriority: make(chan struct{}, highLimit),
        lowPriority:  make(chan struct{}, lowLimit),
    }
}

func (pl *PriorityLimit) Middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        priority := r.Header.Get("X-Priority")

        var sem chan struct{}
        if priority == "high" {
            sem = pl.highPriority
        } else {
            sem = pl.lowPriority
        }

        select {
        case sem <- struct{}{}:
            defer func() { <-sem }()
            next.ServeHTTP(w, r)
        case <-time.After(10 * time.Second):
            http.Error(w, "service unavailable", http.StatusServiceUnavailable)
        }
    })
}

// Отслеживание метрик
func ConcurrencyLimitWithMetrics(limit int, next http.Handler) http.Handler {
    sem := make(chan struct{}, limit)

    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Запись текущей concurrency
        currentConcurrency := len(sem)
        metrics.RecordGauge("http_concurrent_requests", float64(currentConcurrency), map[string]string{
            "path": r.URL.Path,
        })

	// Попытка захватить с таймаутом
        select {
        case sem <- struct{}{}:
            defer func() { <-sem }()
            next.ServeHTTP(w, r)
        case <-time.After(30 * time.Second):
            metrics.IncrementCounter("http_concurrency_limit_rejections", map[string]string{
                "path": r.URL.Path,
            })
            http.Error(w, "too many requests", http.StatusTooManyRequests)
        }
    })
}
\`\`\`

**Практические преимущества:**
- **Стабильность сервера:** Предотвращение крашей от исчерпания ресурсов
- **Предсказуемая производительность:** Стабильные времена ответа под нагрузкой
- **Контроль затрат:** Ограничение дорогих операций (DB запросы, внешние API)
- **Справедливый доступ:** Предотвращение монополизации ресурсов одним пользователем

**Паттерн семафора:**
- **Буферизированный канал:** Простой, Go-идиоматичный семафор
- **Блокирующий захват:** Естественный механизм backpressure
- **Defer Release:** Гарантирует освобождение слота даже при панике
- **Нулевые накладные расходы:** Операции с каналами быстрые

**Типичные лимиты:**
- **Database Endpoints:** 10-50 параллельных соединений
- **External API Calls:** 5-20 параллельных запросов
- **File Uploads:** 3-10 параллельных загрузок
- **Heavy Computation:** 2-10 параллельных операций
- **Websockets:** 1000-10000 параллельных соединений

**Важные заметки:**
- **Слишком низкий:** Недоиспользование ресурсов, увеличение задержки
- **Слишком высокий:** Исчерпание ресурсов, краши
- **Мониторинг:** Отслеживать уровень отклонений, корректировать лимиты
- **Таймауты:** Комбинировать с таймаутами для предотвращения бесконечного ожидания

Без ConcurrencyLimit всплеск трафика может перегрузить сервер, исчерпав память/соединения и крашнув весь сервис.`,
			solutionCode: `package httpx

import (
	"net/http"
)

func ConcurrencyLimit(limit int, next http.Handler) http.Handler {
	if next == nil {	// Обработка nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	if limit <= 0 {	// Проверка валидности лимита
		return next	// Пропуск middleware если нет лимита
	}
	sem := make(chan struct{}, limit)	// Создание буферизированного канала как семафора
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sem <- struct{}{}	// Захват слота (блокировка если заполнен)
		defer func() { <-sem }()	// Освобождение слота после завершения handler
		next.ServeHTTP(w, r)	// Выполнение handler
	})
}`
		},
		uz: {
			title: 'Parallel requestlar sonini cheklash',
			description: `Semafor patternidan foydalanib, parallel request handlerlar sonini cheklovchi **ConcurrencyLimit** middleware ni amalga oshiring.

**Talablar:**
1. \`ConcurrencyLimit(limit int, next http.Handler) http.Handler\` funksiyasini yarating
2. Agar limit <= 0 bo'lsa middleware ni o'tkazing
3. Semafor sifatida buferli kanaldan foydalaning
4. next ni chaqirishdan oldin slot oling
5. next tugagandan keyin slot ni bo'shating (defer dan foydalaning)
6. Limit yetganda yangi requestlarni bloklang
7. nil handlerni ishlang

**Misol:**
\`\`\`go
handler := ConcurrencyLimit(2, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    time.Sleep(1 * time.Second) // Sekin handlerni simulyatsiya qilish
    fmt.Fprintf(w, "Done")
}))

// Faqat 2 ta request parallel bajariladi, qolganlari slot kutadi
\`\`\`

**Cheklovlar:**
- Semafor sifatida buferli kanaldan foydalanishi kerak
- Handler panic qilsa ham slot ni bo'shatishi kerak (defer dan foydalaning)
- Slot mavjud bo'lmaguncha qayta ishlashni boshlamasligi kerak`,
			hint1: `size=limit bilan buferli kanal yarating. Olish uchun yuborish, bo'shatish uchun defer da qabul qilish.`,
			hint2: `Handler panic qilganda ham slot bo'shatishni kafolatlash uchun defer func() { <-sem }() dan foydalaning.`,
			whyItMatters: `ConcurrencyLimit parallel request qayta ishlashni cheklash orqali serverlarni ortiqcha yuklanishdan himoya qiladi, resurslar tugashining oldini oladi va performansni saqlaydi.

**Nima uchun Concurrency Limiting:**
- **Resurslarni himoya qilish:** Memory/CPU tugashining oldini olish
- **Barqarorlik:** Ortiqcha yuklanishdan kaskadli nosozliklarni oldini olish
- **Adolatli xizmat:** Barqaror javob vaqtlarini ta'minlash
- **Graceful Degradation:** Crash o'rniga requestlarni navbatga qo'yish

**Ishlab chiqarish patterni:**
\`\`\`go
// Qimmat endpointlarni himoya qilish
expensiveHandler := ConcurrencyLimit(10, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Qimmat database so'rovi yoki hisoblash
    result := expensiveQuery(r.Context())
    json.NewEncoder(w).Encode(result)
}))

// Turli endpointlar uchun turli limitlar
func Router() *http.ServeMux {
    mux := http.NewServeMux()

	// Arzon endpointlar uchun yuqori limit
    mux.Handle("/health", ConcurrencyLimit(1000, healthHandler))

	// Qimmat endpointlar uchun past limit
    mux.Handle("/report", ConcurrencyLimit(5, reportHandler))
    mux.Handle("/export", ConcurrencyLimit(3, exportHandler))

	// Statik fayllar uchun limitsiz
    mux.Handle("/static/", http.FileServer(http.Dir("./static")))

    return mux
}

// Tizim yuklanishiga asoslangan dinamik limitlar
func AdaptiveConcurrencyLimit(baseLimit int) func(http.Handler) http.Handler {
    currentLimit := baseLimit

	// Tizimni monitoring qilish va limitni sozlash
    go func() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()

        for range ticker.C {
            load := getSystemLoad()
            if load > 0.8 {
                currentLimit = max(1, currentLimit-1) // Limitni kamaytirish
            } else if load < 0.5 && currentLimit < baseLimit {
                currentLimit++ // Limitni oshirish
            }
        }
    }()

    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            sem := make(chan struct{}, currentLimit)
            handler := ConcurrencyLimit(currentLimit, next)
            handler.ServeHTTP(w, r)
        })
    }
}

// Timeout + Concurrency limit
func ProtectedHandler(limit int, timeout time.Duration) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return Chain(
            ConcurrencyLimit(limit),
            Timeout(timeout),
        )(next)
    }
}

// Har bir foydalanuvchi uchun concurrency limitlar
func PerUserConcurrencyLimit(limit int) func(http.Handler) http.Handler {
    userSems := sync.Map{} // map[string]chan struct{}

    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            userID := r.Context().Value("user_id").(string)

	// Foydalanuvchi uchun semafor olish yoki yaratish
            val, _ := userSems.LoadOrStore(userID, make(chan struct{}, limit))
            sem := val.(chan struct{})

            select {
            case sem <- struct{}{}:
                defer func() { <-sem }()
                next.ServeHTTP(w, r)
            case <-time.After(5 * time.Second):
                http.Error(w, "too many concurrent requests", http.StatusTooManyRequests)
            }
        })
    }
}

// Ustuvorlikka asoslangan concurrency
type PriorityLimit struct {
    highPriority chan struct{}
    lowPriority  chan struct{}
}

func NewPriorityLimit(highLimit, lowLimit int) *PriorityLimit {
    return &PriorityLimit{
        highPriority: make(chan struct{}, highLimit),
        lowPriority:  make(chan struct{}, lowLimit),
    }
}

func (pl *PriorityLimit) Middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        priority := r.Header.Get("X-Priority")

        var sem chan struct{}
        if priority == "high" {
            sem = pl.highPriority
        } else {
            sem = pl.lowPriority
        }

        select {
        case sem <- struct{}{}:
            defer func() { <-sem }()
            next.ServeHTTP(w, r)
        case <-time.After(10 * time.Second):
            http.Error(w, "service unavailable", http.StatusServiceUnavailable)
        }
    })
}

// Metrikalarni kuzatish
func ConcurrencyLimitWithMetrics(limit int, next http.Handler) http.Handler {
    sem := make(chan struct{}, limit)

    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Joriy concurrencyni yozish
        currentConcurrency := len(sem)
        metrics.RecordGauge("http_concurrent_requests", float64(currentConcurrency), map[string]string{
            "path": r.URL.Path,
        })

	// Timeout bilan olishga urinish
        select {
        case sem <- struct{}{}:
            defer func() { <-sem }()
            next.ServeHTTP(w, r)
        case <-time.After(30 * time.Second):
            metrics.IncrementCounter("http_concurrency_limit_rejections", map[string]string{
                "path": r.URL.Path,
            })
            http.Error(w, "too many requests", http.StatusTooManyRequests)
        }
    })
}
\`\`\`

**Amaliy foydalari:**
- **Server barqarorligi:** Resurslar tugashidan crashlarning oldini olish
- **Bashorat qilinadigan performans:** Yuklanish ostida barqaror javob vaqtlari
- **Xarajatlarni nazorat qilish:** Qimmat operatsiyalarni cheklash (DB so'rovlari, tashqi API lar)
- **Adolatli kirish:** Bitta foydalanuvchining resurslarni monopolizatsiya qilishining oldini olish

**Semafor patterni:**
- **Buferli kanal:** Oddiy, Go-idiomatik semafor
- **Blokirovka qiluvchi olish:** Tabiiy backpressure mexanizmi
- **Defer Release:** Panic qilganda ham slot bo'shatishni kafolatlaydi
- **Nol xarajatlar:** Kanal operatsiyalari tez

**Odatiy limitlar:**
- **Database Endpointlari:** 10-50 parallel ulanishlar
- **Tashqi API chaqiruvlari:** 5-20 parallel requestlar
- **Fayl yuklanishi:** 3-10 parallel yuklashlar
- **Og'ir hisoblashlar:** 2-10 parallel operatsiyalar
- **Websockets:** 1000-10000 parallel ulanishlar

**Muhim eslatmalar:**
- **Juda past:** Resurslardan kam foydalanish, kechikishni oshirish
- **Juda yuqori:** Resurslar tugashi, crashlar
- **Monitoring:** Rad etish darajasini kuzatish, limitlarni sozlash
- **Timeoutlar:** Cheksiz kutishning oldini olish uchun timeoutlar bilan birlashtirish

ConcurrencyLimit siz trafik ko'tarilishi serverni ortiqcha yuklashi, xotira/ulanishlarni tugatishi va butun xizmatni crashlashi mumkin.`,
			solutionCode: `package httpx

import (
	"net/http"
)

func ConcurrencyLimit(limit int, next http.Handler) http.Handler {
	if next == nil {	// nil next handlerni ishlash
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	if limit <= 0 {	// Limit haqiqiyligini tekshirish
		return next	// Agar limit bo'lmasa middleware ni o'tkazish
	}
	sem := make(chan struct{}, limit)	// Semafor sifatida buferli kanal yaratish
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sem <- struct{}{}	// Slot olish (to'lsa bloklash)
		defer func() { <-sem }()	// Handler tugagandan keyin slotni bo'shatish
		next.ServeHTTP(w, r)	// Handlerni bajarish
	})
}`
		}
	}
};

export default task;
