import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-request-id',
	title: 'Request ID Middleware',
	difficulty: 'easy',	tags: ['go', 'http', 'middleware', 'tracing'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **RequestID** middleware that adds a unique request identifier to context and response headers for distributed tracing.

**Requirements:**
1. Create function \`RequestID(next http.Handler) http.Handler\`
2. Generate unique ID using \`time.Now().UTC().Format(time.RFC3339Nano)\`
3. Store ID in request context using key \`RequestIDKey\`
4. Set \`X-Request-ID\` response header
5. Handle nil next handler by returning empty handler

**Example:**
\`\`\`go
handler := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    id := r.Context().Value(RequestIDKey).(string)
    fmt.Fprintf(w, "Request ID: %s", id)
}))

// Response headers: X-Request-ID: 2024-01-15T10:30:45.123456Z
// Response body: Request ID: 2024-01-15T10:30:45.123456Z
\`\`\`

**Constraints:**
- Must add ID to both context and response header
- Must handle nil handler gracefully`,
	initialCode: `package httpx

import (
	"context"
	"net/http"
	"time"
)

type ctxKey string

const RequestIDKey ctxKey = "rid"

// TODO: Implement RequestID middleware
func RequestID(next http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"context"
	"net/http"
	"time"
)

type ctxKey string

const RequestIDKey ctxKey = "rid"

func RequestID(next http.Handler) http.Handler {
	if next == nil {	// Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})	// Return no-op handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := time.Now().UTC().Format(time.RFC3339Nano)	// Generate unique ID with nanosecond precision
		ctx := context.WithValue(r.Context(), RequestIDKey, id)	// Add ID to request context for downstream handlers
		w.Header().Set("X-Request-ID", id)	// Add ID to response header for client tracking
		next.ServeHTTP(w, r.WithContext(ctx))	// Pass modified request to next handler
	})
}`,
			hint1: `Use time.Now().UTC().Format(time.RFC3339Nano) to generate unique IDs with high precision.`,
			hint2: `Add ID to both context (WithValue) and response header (Set) before calling next.ServeHTTP.`,
			testCode: `package httpx

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func Test1(t *testing.T) {
	// Test middleware returns non-nil handler
	h := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	if h == nil {
		t.Error("handler should not be nil")
	}
}

func Test2(t *testing.T) {
	// Test nil handler returns empty handler
	h := RequestID(nil)
	if h == nil {
		t.Error("nil handler should return non-nil")
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)
	h.ServeHTTP(rec, req)
}

func Test3(t *testing.T) {
	// Test X-Request-ID header is set
	h := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)
	h.ServeHTTP(rec, req)
	if rec.Header().Get("X-Request-ID") == "" {
		t.Error("X-Request-ID header should be set")
	}
}

func Test4(t *testing.T) {
	// Test request ID is in context
	var ctxID interface{}
	h := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctxID = r.Context().Value(RequestIDKey)
	}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)
	h.ServeHTTP(rec, req)
	if ctxID == nil {
		t.Error("request ID should be in context")
	}
}

func Test5(t *testing.T) {
	// Test context ID matches header ID
	var ctxID string
	h := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctxID = r.Context().Value(RequestIDKey).(string)
	}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)
	h.ServeHTTP(rec, req)
	headerID := rec.Header().Get("X-Request-ID")
	if ctxID != headerID {
		t.Errorf("context ID %q != header ID %q", ctxID, headerID)
	}
}

func Test6(t *testing.T) {
	// Test ID format looks like timestamp
	h := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)
	h.ServeHTTP(rec, req)
	id := rec.Header().Get("X-Request-ID")
	if len(id) < 20 {
		t.Errorf("ID %q too short for RFC3339Nano", id)
	}
}

func Test7(t *testing.T) {
	// Test next handler is called
	called := false
	h := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)
	h.ServeHTTP(rec, req)
	if !called {
		t.Error("next handler should be called")
	}
}

func Test8(t *testing.T) {
	// Test different requests get different IDs
	h := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec1 := httptest.NewRecorder()
	rec2 := httptest.NewRecorder()
	h.ServeHTTP(rec1, httptest.NewRequest("GET", "/", nil))
	h.ServeHTTP(rec2, httptest.NewRequest("GET", "/", nil))
	id1 := rec1.Header().Get("X-Request-ID")
	id2 := rec2.Header().Get("X-Request-ID")
	if id1 == id2 {
		t.Error("different requests should have different IDs")
	}
}

func Test9(t *testing.T) {
	// Test request body and method preserved
	var method string
	h := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method = r.Method
	}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/test", nil)
	h.ServeHTTP(rec, req)
	if method != "POST" {
		t.Errorf("method = %q, want POST", method)
	}
}

func Test10(t *testing.T) {
	// Test response writing works
	h := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte("created"))
	}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/", nil)
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Errorf("status = %d, want 201", rec.Code)
	}
	if rec.Body.String() != "created" {
		t.Errorf("body = %q, want 'created'", rec.Body.String())
	}
}`,
			whyItMatters: `Request IDs enable distributed tracing and log correlation across microservices.

**Why Request IDs:**
- **Distributed Tracing:** Track a single request as it flows through multiple services
- **Log Correlation:** Group all log entries for one request using \`grep request_id=xyz\`
- **Debugging:** Reproduce issues by finding all operations for a specific request ID
- **Monitoring:** Identify slow requests and track them end-to-end across services

**Production Pattern:**
\`\`\`go
func RequestID(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Check if client already provided an ID
        id := r.Header.Get("X-Request-ID")
        if id == "" {
            id = uuid.New().String()	// Use UUID for better uniqueness
        }

        ctx := context.WithValue(r.Context(), RequestIDKey, id)
        w.Header().Set("X-Request-ID", id)

	// Add to structured logs
        log.WithField("request_id", id).Info("handling request")

        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

// Propagate to downstream services
func CallAPI(ctx context.Context, url string) {
    req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
    if id := ctx.Value(RequestIDKey); id != nil {
        req.Header.Set("X-Request-ID", id.(string))	// Forward request ID
    }
    http.DefaultClient.Do(req)
}
\`\`\`

**Real-World Benefits:**
- **Incident Response:** \`grep "request_id=abc-123" logs/*\` finds all logs for a failed request
- **Performance Analysis:** Track slow requests through the entire system
- **Client Support:** Give customers the request ID for support tickets

**Standard Practice:**
- AWS X-Ray: \`X-Amzn-Trace-Id\`
- Google Cloud: \`X-Cloud-Trace-Context\`
- Industry standard: \`X-Request-ID\`

Without request IDs, debugging multi-service issues becomes nearly impossible—logs from different services can't be correlated.`,	order: 0,
	translations: {
		ru: {
			title: 'Добавление уникального ID к запросу',
			description: `Реализуйте middleware **RequestID**, который добавляет уникальный идентификатор запроса в контекст и заголовки ответа для распределённой трассировки.

**Требования:**
1. Создайте функцию \`RequestID(next http.Handler) http.Handler\`
2. Генерируйте уникальный ID используя \`time.Now().UTC().Format(time.RFC3339Nano)\`
3. Сохраните ID в контексте запроса с ключом \`RequestIDKey\`
4. Установите заголовок ответа \`X-Request-ID\`
5. Обрабатывайте nil next handler, возвращая пустой handler

**Пример:**
\`\`\`go
handler := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    id := r.Context().Value(RequestIDKey).(string)
    fmt.Fprintf(w, "Request ID: %s", id)
}))

// Заголовки ответа: X-Request-ID: 2024-01-15T10:30:45.123456Z
\`\`\`

**Ограничения:**
- Должен добавлять ID и в контекст, и в заголовок
- Должен корректно обрабатывать nil handler`,
			hint1: `Используйте time.Now().UTC().Format(time.RFC3339Nano) для генерации уникальных ID.`,
			hint2: `Добавьте ID в контекст (WithValue) и в заголовок (Set) перед вызовом next.ServeHTTP.`,
			whyItMatters: `Request ID позволяет отслеживать запросы через микросервисы и коррелировать логи.

**Почему Request ID:**
- **Распределённая трассировка:** Отслеживание одного запроса через несколько сервисов
- **Корреляция логов:** Группировка всех логов одного запроса через \`grep request_id=xyz\`
- **Отладка:** Воспроизведение проблем по конкретному request ID
- **Мониторинг:** Идентификация медленных запросов и сквозное отслеживание по сервисам

**Продакшен паттерн:**
\`\`\`go
func RequestID(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Проверяем, предоставил ли клиент ID
        id := r.Header.Get("X-Request-ID")
        if id == "" {
            id = uuid.New().String()	// Используем UUID для лучшей уникальности
        }

        ctx := context.WithValue(r.Context(), RequestIDKey, id)
        w.Header().Set("X-Request-ID", id)

	// Добавляем в структурированные логи
        log.WithField("request_id", id).Info("handling request")

        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

// Передача в downstream сервисы
func CallAPI(ctx context.Context, url string) {
    req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
    if id := ctx.Value(RequestIDKey); id != nil {
        req.Header.Set("X-Request-ID", id.(string))	// Пробрасываем request ID
    }
    http.DefaultClient.Do(req)
}
\`\`\`

**Практические преимущества:**
- **Реагирование на инциденты:** \`grep "request_id=abc-123" logs/*\` находит все логи неудачного запроса
- **Анализ производительности:** Отслеживание медленных запросов через всю систему
- **Поддержка клиентов:** Предоставление request ID клиентам для обращений в поддержку

**Стандартная практика:**
- AWS X-Ray: \`X-Amzn-Trace-Id\`
- Google Cloud: \`X-Cloud-Trace-Context\`
- Отраслевой стандарт: \`X-Request-ID\`

Без request ID отладка проблем в многосервисных системах становится практически невозможной.`,
			solutionCode: `package httpx

import (
	"context"
	"net/http"
	"time"
)

type ctxKey string

const RequestIDKey ctxKey = "rid"

func RequestID(next http.Handler) http.Handler {
	if next == nil {	// Обработка nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})	// Возврат no-op handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := time.Now().UTC().Format(time.RFC3339Nano)	// Генерация уникального ID с наносекундной точностью
		ctx := context.WithValue(r.Context(), RequestIDKey, id)	// Добавление ID в контекст запроса для downstream handlers
		w.Header().Set("X-Request-ID", id)	// Добавление ID в заголовок ответа для отслеживания клиентом
		next.ServeHTTP(w, r.WithContext(ctx))	// Передача модифицированного запроса следующему handler
	})
}`
		},
		uz: {
			title: 'Requestga noyob ID qo\'shish',
			description: `Tarqatilgan tracing uchun kontekst va response headerlarga noyob request identifikatorini qo'shadigan **RequestID** middleware ni amalga oshiring.

**Talablar:**
1. \`RequestID(next http.Handler) http.Handler\` funksiyasini yarating
2. \`time.Now().UTC().Format(time.RFC3339Nano)\` yordamida noyob ID yarating
3. ID ni 'RequestIDKey' kalit bilan request kontekstida saqlang
4. \`X-Request-ID\` response headerini o'rnating
5. nil next handlerni bo'sh handler qaytarish orqali ishlang

**Misol:**
\`\`\`go
handler := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    id := r.Context().Value(RequestIDKey).(string)
    fmt.Fprintf(w, "Request ID: %s", id)
}))

// Response headers: X-Request-ID: 2024-01-15T10:30:45.123456Z
\`\`\`

**Cheklovlar:**
- ID ni kontekst va header ga qo'shishi kerak
- nil handlerni to'g'ri ishlashi kerak`,
			hint1: `Yuqori aniqlikdagi noyob ID lar uchun time.Now().UTC().Format(time.RFC3339Nano) dan foydalaning.`,
			hint2: `next.ServeHTTP chaqirishdan oldin ID ni kontekst (WithValue) va headerga (Set) qo'shing.`,
			whyItMatters: `Request ID lar mikroservislar bo'ylab tarqatilgan tracing va log korrelyatsiyasini yoqadi.

**Nima uchun Request ID:**
- **Tarqatilgan Tracing:** Bir requestni bir nechta servislar orqali kuzatish
- **Log korrelyatsiyasi:** \`grep request_id=xyz\` orqali bitta requestning barcha loglarini guruhlash
- **Debugging:** Muayyan request ID orqali muammolarni qayta yaratish
- **Monitoring:** Sekin requestlarni aniqlash va ularni servislar bo'ylab kuzatish

**Ishlab chiqarish patterni:**
\`\`\`go
func RequestID(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Klient allaqachon ID taqdim etganligini tekshirish
        id := r.Header.Get("X-Request-ID")
        if id == "" {
            id = uuid.New().String()	// Yaxshiroq noyoblik uchun UUID ishlatish
        }

        ctx := context.WithValue(r.Context(), RequestIDKey, id)
        w.Header().Set("X-Request-ID", id)

	// Strukturalangan loglarga qo'shish
        log.WithField("request_id", id).Info("handling request")

        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

// Downstream servislarga uzatish
func CallAPI(ctx context.Context, url string) {
    req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
    if id := ctx.Value(RequestIDKey); id != nil {
        req.Header.Set("X-Request-ID", id.(string))	// Request ID ni uzatish
    }
    http.DefaultClient.Do(req)
}
\`\`\`

**Amaliy foydalari:**
- **Hodisalarga javob:** \`grep "request_id=abc-123" logs/*\` muvaffaqiyatsiz requestning barcha loglarini topadi
- **Ishlash tahlili:** Sekin requestlarni butun tizim bo'ylab kuzatish
- **Mijozlarni qo'llab-quvvatlash:** Qo'llab-quvvatlash so'rovlari uchun mijozlarga request ID berish

**Standart amaliyot:**
- AWS X-Ray: \`X-Amzn-Trace-Id\`
- Google Cloud: \`X-Cloud-Trace-Context\`
- Sanoat standarti: \`X-Request-ID\`

Request ID larsiz ko'p xizmatli tizimlardagi muammolarni tuzatish deyarli imkonsiz bo'ladi.`,
			solutionCode: `package httpx

import (
	"context"
	"net/http"
	"time"
)

type ctxKey string

const RequestIDKey ctxKey = "rid"

func RequestID(next http.Handler) http.Handler {
	if next == nil {	// nil next handlerni ishlash
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})	// No-op handlerni qaytarish
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := time.Now().UTC().Format(time.RFC3339Nano)	// Nanosekund aniqlik bilan noyob ID yaratish
		ctx := context.WithValue(r.Context(), RequestIDKey, id)	// Downstream handlerlar uchun request kontekstiga ID qo'shish
		w.Header().Set("X-Request-ID", id)	// Client tomonidan kuzatish uchun response headeriga ID qo'shish
		next.ServeHTTP(w, r.WithContext(ctx))	// O'zgartirilgan requestni keyingi handlerga o'tkazish
	})
}`
		}
	}
};

export default task;
