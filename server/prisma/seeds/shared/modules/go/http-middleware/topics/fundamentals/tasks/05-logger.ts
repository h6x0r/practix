import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-logger',
	title: 'Logger Middleware',
	difficulty: 'easy',	tags: ['go', 'http', 'middleware', 'logging'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Logger** middleware that measures request duration and logs method, path, and elapsed time.

**Requirements:**
1. Create function \`Logger(next http.Handler) http.Handler\`
2. Record start time before calling next
3. Calculate duration after next completes
4. Log method, path, and duration using \`log.Printf\`
5. Handle nil next handler

**Example:**
\`\`\`go
handler := Logger(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    time.Sleep(100 * time.Millisecond)
    fmt.Fprintf(w, "OK")
}))

// Console output: GET /api/users took 100.5ms
\`\`\`

**Constraints:**
- Must measure actual request duration
- Must log after request completes`,
	initialCode: `package httpx

import (
	"log"
	"net/http"
	"time"
)

// TODO: Implement Logger middleware
func Logger(next http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"log"
	"net/http"
	"time"
)

func Logger(next http.Handler) http.Handler {
	if next == nil {	// Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()	// Record start time
		next.ServeHTTP(w, r)	// Execute request
		duration := time.Since(start)	// Calculate elapsed time
		log.Printf("%s %s took %v", r.Method, r.URL.Path, duration)	// Log request info
	})
}`,
			hint1: `Use time.Now() before calling next.ServeHTTP() to record the start time.`,
			hint2: `Use time.Since(start) after next.ServeHTTP() to calculate duration, then log with log.Printf().`,
			testCode: `package httpx

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Test middleware returns non-nil handler
	h := Logger(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	if h == nil {
		t.Error("handler should not be nil")
	}
}

func Test2(t *testing.T) {
	// Test nil handler returns non-nil
	h := Logger(nil)
	if h == nil {
		t.Error("nil handler should return non-nil")
	}
}

func Test3(t *testing.T) {
	// Test next handler is called
	called := false
	h := Logger(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil))
	if !called {
		t.Error("next handler should be called")
	}
}

func Test4(t *testing.T) {
	// Test response is passed through
	h := Logger(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte("created"))
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/", nil))
	if rec.Code != http.StatusCreated {
		t.Errorf("status = %d, want 201", rec.Code)
	}
}

func Test5(t *testing.T) {
	// Test request method is preserved
	var method string
	h := Logger(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method = r.Method
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("PUT", "/", nil))
	if method != "PUT" {
		t.Errorf("method = %q, want PUT", method)
	}
}

func Test6(t *testing.T) {
	// Test request path is preserved
	var path string
	h := Logger(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/api/users", nil))
	if path != "/api/users" {
		t.Errorf("path = %q, want /api/users", path)
	}
}

func Test7(t *testing.T) {
	// Test logs after handler completes (no panic)
	h := Logger(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(10 * time.Millisecond)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil))
}

func Test8(t *testing.T) {
	// Test multiple requests work independently
	h := Logger(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(r.URL.Path))
	}))
	rec1 := httptest.NewRecorder()
	rec2 := httptest.NewRecorder()
	h.ServeHTTP(rec1, httptest.NewRequest("GET", "/path1", nil))
	h.ServeHTTP(rec2, httptest.NewRequest("GET", "/path2", nil))
	if rec1.Body.String() != "/path1" || rec2.Body.String() != "/path2" {
		t.Error("requests should be independent")
	}
}

func Test9(t *testing.T) {
	// Test headers are preserved
	var contentType string
	h := Logger(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		contentType = r.Header.Get("Content-Type")
	}))
	req := httptest.NewRequest("POST", "/", nil)
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(httptest.NewRecorder(), req)
	if contentType != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", contentType)
	}
}

func Test10(t *testing.T) {
	// Test slow handler still works
	h := Logger(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(50 * time.Millisecond)
		w.Write([]byte("slow"))
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Body.String() != "slow" {
		t.Errorf("body = %q, want 'slow'", rec.Body.String())
	}
}`,
			whyItMatters: `Logger middleware provides visibility into request handling, essential for debugging, performance monitoring, and audit trails.

**Why Request Logging:**
- **Performance Monitoring:** Identify slow endpoints and optimization opportunities
- **Debugging:** See exact request flow and timing
- **Audit Trail:** Track all API access for compliance
- **Alerting:** Trigger alerts for slow requests (>1s)

**Production Pattern:**
\`\`\`go
// Structured logging with more context
func StructuredLogger(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()

	// Capture response status
        rec := &statusRecorder{ResponseWriter: w, status: 200}

        next.ServeHTTP(rec, r)

        duration := time.Since(start)

	// Structured log entry
        log.Printf(
            "method=%s path=%s status=%d duration=%v ip=%s user_agent=%s",
            r.Method,
            r.URL.Path,
            rec.status,
            duration,
            r.RemoteAddr,
            r.UserAgent(),
        )

	// Alert on slow requests
        if duration > time.Second {
            log.Printf("SLOW REQUEST: %s %s took %v", r.Method, r.URL.Path, duration)
        }
    })
}

// JSON logging for production
func JSONLogger(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        rec := &statusRecorder{ResponseWriter: w, status: 200}

        next.ServeHTTP(rec, r)

        logEntry := map[string]interface{}{
            "timestamp":  time.Now().UTC().Format(time.RFC3339),
            "method":     r.Method,
            "path":       r.URL.Path,
            "status":     rec.status,
            "duration_ms": time.Since(start).Milliseconds(),
            "ip":         r.RemoteAddr,
            "request_id": r.Context().Value(RequestIDKey),
        }

        json.NewEncoder(os.Stdout).Encode(logEntry)
    })
}

// Metrics collection
func MetricsLogger(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        rec := &statusRecorder{ResponseWriter: w, status: 200}

        next.ServeHTTP(rec, r)

        duration := time.Since(start)

	// Send to metrics system (Prometheus, StatsD, etc.)
        metrics.RecordRequest(r.Method, r.URL.Path, rec.status, duration)
    })
}
\`\`\`

**Real-World Benefits:**
- **SLO Tracking:** Monitor if 99% of requests complete under 100ms
- **Error Investigation:** See which endpoints fail most often
- **Capacity Planning:** Identify traffic patterns and peak times
- **Security:** Detect suspicious patterns (many 401s, unusual paths)

**Log Aggregation:**
- **ELK Stack:** Elasticsearch, Logstash, Kibana
- **Splunk:** Enterprise log management
- **CloudWatch:** AWS log aggregation
- **Datadog:** APM and logging platform

Simple logging middleware is the foundation for observability—without it, you're flying blind in production.`,	order: 4,
	translations: {
		ru: {
			title: 'Логирование HTTP запросов и ответов',
			description: `Реализуйте middleware **Logger**, который измеряет время обработки запроса и логирует метод, путь и длительность.

**Требования:**
1. Создайте функцию \`Logger(next http.Handler) http.Handler\`
2. Запишите время начала перед вызовом next
3. Вычислите длительность после завершения next
4. Залогируйте метод, путь и длительность через \`log.Printf\`
5. Обработайте nil handler

**Пример:**
\`\`\`go
handler := Logger(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    time.Sleep(100 * time.Millisecond)
    fmt.Fprintf(w, "OK")
}))

// Консоль: GET /api/users took 100.5ms
\`\`\`

**Ограничения:**
- Должен измерять реальное время обработки запроса
- Должен логировать после завершения запроса`,
			hint1: `Используйте time.Now() перед вызовом next.ServeHTTP() для записи времени начала.`,
			hint2: `Используйте time.Since(start) после next.ServeHTTP() для вычисления длительности, затем залогируйте через log.Printf().`,
			whyItMatters: `Logger middleware обеспечивает видимость обработки запросов, необходимую для отладки, мониторинга производительности и аудита.

**Почему важно логирование запросов:**
- **Мониторинг производительности:** Определение медленных endpoints
- **Отладка:** Видение точного flow запроса и timing
- **Audit trail:** Отслеживание всех обращений к API
- **Алертинг:** Триггеры на медленные запросы (>1s)

**Продакшен паттерн:**
\`\`\`go
// Структурированное логирование с расширенным контекстом
func StructuredLogger(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()

	// Захват статуса ответа
        rec := &statusRecorder{ResponseWriter: w, status: 200}

        next.ServeHTTP(rec, r)

        duration := time.Since(start)

	// Структурированная запись лога
        log.Printf(
            "method=%s path=%s status=%d duration=%v ip=%s user_agent=%s",
            r.Method,
            r.URL.Path,
            rec.status,
            duration,
            r.RemoteAddr,
            r.UserAgent(),
        )

	// Алерт на медленные запросы
        if duration > time.Second {
            log.Printf("SLOW REQUEST: %s %s took %v", r.Method, r.URL.Path, duration)
        }
    })
}

// JSON логирование для продакшена
func JSONLogger(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        rec := &statusRecorder{ResponseWriter: w, status: 200}

        next.ServeHTTP(rec, r)

        logEntry := map[string]interface{}{
            "timestamp":  time.Now().UTC().Format(time.RFC3339),
            "method":     r.Method,
            "path":       r.URL.Path,
            "status":     rec.status,
            "duration_ms": time.Since(start).Milliseconds(),
            "ip":         r.RemoteAddr,
            "request_id": r.Context().Value(RequestIDKey),
        }

        json.NewEncoder(os.Stdout).Encode(logEntry)
    })
}
\`\`\`

**Практические преимущества:**
- **SLO Tracking:** Мониторинг 99% запросов с временем <100ms
- **Анализ ошибок:** Определение наиболее часто падающих endpoints
- **Планирование мощности:** Определение паттернов трафика и пиковых нагрузок
- **Безопасность:** Обнаружение подозрительных паттернов (много 401, необычные пути)

**Агрегация логов:**
- **ELK Stack:** Elasticsearch, Logstash, Kibana
- **Splunk:** Enterprise log management
- **CloudWatch:** AWS log aggregation
- **Datadog:** APM and logging platform

Простой logging middleware — основа наблюдаемости. Без него вы летите вслепую в продакшене.`,
			solutionCode: `package httpx

import (
	"log"
	"net/http"
	"time"
)

func Logger(next http.Handler) http.Handler {
	if next == nil {	// Обработка nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()	// Запись времени начала
		next.ServeHTTP(w, r)	// Выполнение запроса
		duration := time.Since(start)	// Вычисление времени выполнения
		log.Printf("%s %s took %v", r.Method, r.URL.Path, duration)	// Логирование информации о запросе
	})
}`
		},
		uz: {
			title: 'HTTP requestlar va responselarni loglash',
			description: `Request davomiyligini o'lchaydigan va metod, yo'l va vaqtni loglaydigan **Logger** middleware ni amalga oshiring.

**Talablar:**
1. \`Logger(next http.Handler) http.Handler\` funksiyasini yarating
2. next chaqirishdan oldin boshlang'ich vaqtni yozing
3. next tugagandan keyin davomiylikni hisoblang
4. \`log.Printf\` orqali metod, yo'l va davomiylikni loglang
5. nil handlerni ishlang

**Misol:**
\`\`\`go
handler := Logger(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    time.Sleep(100 * time.Millisecond)
    fmt.Fprintf(w, "OK")
}))

// Konsol chiqishi: GET /api/users took 100.5ms
\`\`\`

**Cheklovlar:**
- Haqiqiy request davomiyligini o'lchashi kerak
- Request tugagandan keyin loglashi kerak`,
			hint1: `Boshlang'ich vaqtni yozish uchun next.ServeHTTP() chaqirishdan oldin time.Now() dan foydalaning.`,
			hint2: `next.ServeHTTP() dan keyin davomiylikni hisoblash uchun time.Since(start) dan, keyin log.Printf() bilan loglang.`,
			whyItMatters: `Logger middleware request ishlashga ko'rinish beradi, debug, performance monitoring va audit uchun zarur.

**Nima uchun request logging:**
- **Performance monitoring:** Sekin endpointlarni aniqlash
- **Debugging:** Aniq request flow va timingni ko'rish
- **Audit trail:** Barcha API kirishlarini kuzatish
- **Alerting:** Sekin requestlar uchun alertlar (>1s)

**Ishlab chiqarish patterni:**
\`\`\`go
// Kengaytirilgan kontekst bilan strukturali logging
func StructuredLogger(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()

	// Response statusini ushlash
        rec := &statusRecorder{ResponseWriter: w, status: 200}

        next.ServeHTTP(rec, r)

        duration := time.Since(start)

	// Strukturali log yozuvi
        log.Printf(
            "method=%s path=%s status=%d duration=%v ip=%s user_agent=%s",
            r.Method,
            r.URL.Path,
            rec.status,
            duration,
            r.RemoteAddr,
            r.UserAgent(),
        )

	// Sekin requestlar uchun alert
        if duration > time.Second {
            log.Printf("SLOW REQUEST: %s %s took %v", r.Method, r.URL.Path, duration)
        }
    })
}

// Production uchun JSON logging
func JSONLogger(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        rec := &statusRecorder{ResponseWriter: w, status: 200}

        next.ServeHTTP(rec, r)

        logEntry := map[string]interface{}{
            "timestamp":  time.Now().UTC().Format(time.RFC3339),
            "method":     r.Method,
            "path":       r.URL.Path,
            "status":     rec.status,
            "duration_ms": time.Since(start).Milliseconds(),
            "ip":         r.RemoteAddr,
            "request_id": r.Context().Value(RequestIDKey),
        }

        json.NewEncoder(os.Stdout).Encode(logEntry)
    })
}
\`\`\`

**Amaliy foydalari:**
- **SLO Tracking:** 99% requestlarning <100ms da tugashini monitoring qilish
- **Xato tahlili:** Eng ko'p tushib qoladigan endpointlarni aniqlash
- **Quvvatni rejalashtirish:** Trafik patternlari va cho'qqi yuklanishlarni aniqlash
- **Xavfsizlik:** Shubhali patternlarni aniqlash (ko'p 401lar, noodatiy yo'llar)

**Log agregatsiyasi:**
- **ELK Stack:** Elasticsearch, Logstash, Kibana
- **Splunk:** Enterprise log management
- **CloudWatch:** AWS log agregatsiyasi
- **Datadog:** APM va logging platformasi

Oddiy logging middleware kuzatuvchanlikning asosi — usiz productionda ko'r uchyapsiz.`,
			solutionCode: `package httpx

import (
	"log"
	"net/http"
	"time"
)

func Logger(next http.Handler) http.Handler {
	if next == nil {	// nil next handlerni ishlash
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()	// Boshlang'ich vaqtni yozish
		next.ServeHTTP(w, r)	// Requestni bajarish
		duration := time.Since(start)	// O'tgan vaqtni hisoblash
		log.Printf("%s %s took %v", r.Method, r.URL.Path, duration)	// Request haqida ma'lumot loglash
	})
}`
		}
	}
};

export default task;
