/**
 * HTTP & gRPC Module Seeds - Production middleware and interceptor patterns
 *
 * Structure:
 * - 18 HTTP middleware tasks (httpx package)
 * - 5 gRPC interceptor tasks (grpcx package)
 * - Full EN/RU/UZ translations
 * - Line-by-line solution explanations
 */

export const GO_HTTP_GRPC_MODULES = [
	{
		title: 'HTTP Middleware & gRPC Interceptors',
		description: 'Master production-ready HTTP middleware and gRPC interceptor patterns for building robust web services.',
		section: 'web',
		order: 6,
		topics: [
			{
				title: 'HTTP Middleware Fundamentals',
				description: 'Build essential HTTP middleware for request handling, headers, and validation.',
				difficulty: 'easy',
				estimatedTime: '2.5h',
				order: 1,
				tasks: [
					{
						slug: 'go-http-request-id',
						title: 'Request ID Middleware',
						difficulty: 'easy',
						tags: ['go', 'http', 'middleware', 'tracing'],
						estimatedTime: '20m',
						isPremium: false,
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
	if next == nil {                                                    // Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})  // Return no-op handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := time.Now().UTC().Format(time.RFC3339Nano)             // Generate unique ID with nanosecond precision
		ctx := context.WithValue(r.Context(), RequestIDKey, id)     // Add ID to request context for downstream handlers
		w.Header().Set("X-Request-ID", id)                          // Add ID to response header for client tracking
		next.ServeHTTP(w, r.WithContext(ctx))                       // Pass modified request to next handler
	})
}`,
						hint1: 'Use time.Now().UTC().Format(time.RFC3339Nano) to generate unique IDs with high precision.',
						hint2: 'Add ID to both context (WithValue) and response header (Set) before calling next.ServeHTTP.',
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
            id = uuid.New().String()  // Use UUID for better uniqueness
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
        req.Header.Set("X-Request-ID", id.(string))  // Forward request ID
    }
    http.DefaultClient.Do(req)
}
\`\`\`

**Real-World Benefits:**
- **Incident Response:** \`grep "request_id=abc-123" logs/*\` finds all logs for a failed request
- **Performance Analysis:** Track slow requests through the entire system
- **Client Support:** Give customers the request ID for support tickets
- **A/B Testing:** Track experiment cohorts by request ID

**Standard Practice:**
- AWS X-Ray: \`X-Amzn-Trace-Id\`
- Google Cloud: \`X-Cloud-Trace-Context\`
- Industry standard: \`X-Request-ID\`

Without request IDs, debugging multi-service issues becomes nearly impossible—logs from different services can't be correlated.`,
						translations: {
							ru: {
								title: 'Request ID Middleware',
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
// Тело ответа: Request ID: 2024-01-15T10:30:45.123456Z
\`\`\`

**Ограничения:**
- Должен добавлять ID и в контекст, и в заголовок
- Должен корректно обрабатывать nil handler`,
								hint1: 'Используйте time.Now().UTC().Format(time.RFC3339Nano) для генерации уникальных ID с высокой точностью.',
								hint2: 'Добавьте ID в контекст (WithValue) и в заголовок (Set) перед вызовом next.ServeHTTP.',
								whyItMatters: `Request ID позволяет отслеживать запросы через микросервисы и коррелировать логи.

**Почему Request ID:**
- **Распределённая трассировка:** Отслеживание одного запроса через несколько сервисов
- **Корреляция логов:** Группировка всех логов одного запроса через \`grep request_id=xyz\`
- **Отладка:** Воспроизведение проблем по конкретному request ID
- **Мониторинг:** Определение медленных запросов и их отслеживание во всей системе

**Production паттерн:**
\`\`\`go
func RequestID(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Проверяем, предоставил ли клиент ID
        id := r.Header.Get("X-Request-ID")
        if id == "" {
            id = uuid.New().String()  // UUID для лучшей уникальности
        }

        ctx := context.WithValue(r.Context(), RequestIDKey, id)
        w.Header().Set("X-Request-ID", id)

        // Добавляем в структурированные логи
        log.WithField("request_id", id).Info("обработка запроса")

        next.ServeHTTP(w, r.WithContext(ctx))
    })
}
\`\`\`

**Реальные преимущества:**
- **Реагирование на инциденты:** \`grep "request_id=abc-123" logs/*\` находит все логи упавшего запроса
- **Анализ производительности:** Отслеживание медленных запросов во всей системе
- **Поддержка клиентов:** Предоставление request ID для тикетов поддержки
- **A/B тестирование:** Отслеживание экспериментальных когорт по request ID

**Стандартная практика:**
- AWS X-Ray: \`X-Amzn-Trace-Id\`
- Google Cloud: \`X-Cloud-Trace-Context\`
- Индустриальный стандарт: \`X-Request-ID\`

Без request ID отладка проблем в микросервисной архитектуре практически невозможна—логи из разных сервисов невозможно соотнести.`
							},
							uz: {
								title: 'Request ID Middleware',
								description: `Taqsimlangan tracing uchun so rov konteksti va javob headerlariga noyob so rov identifikatorini qo shadigan **RequestID** middleware ni amalga oshiring.

**Talablar:**
1. \`RequestID(next http.Handler) http.Handler\` funksiyasini yarating
2. \`time.Now().UTC().Format(time.RFC3339Nano)\` yordamida noyob ID yarating
3. ID ni \`RequestIDKey\` kalit bilan so rov kontekstida saqlang
4. \`X-Request-ID\` javob headerini o rnating
5. nil next handler ni bo sh handler qaytarish orqali ishlang

**Misol:**
\`\`\`go
handler := RequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    id := r.Context().Value(RequestIDKey).(string)
    fmt.Fprintf(w, "Request ID: %s", id)
}))

// Javob headerlari: X-Request-ID: 2024-01-15T10:30:45.123456Z
// Javob tanasi: Request ID: 2024-01-15T10:30:45.123456Z
\`\`\`

**Cheklovlar:**
- ID ni kontekst va headerga qo shishi kerak
- nil handler ni to g ri ishlashi kerak`,
								hint1: 'Yuqori aniqlik bilan noyob ID lar yaratish uchun time.Now().UTC().Format(time.RFC3339Nano) dan foydalaning.',
								hint2: 'next.ServeHTTP chaqirishdan oldin ID ni kontekst (WithValue) va header (Set) ga qo shing.',
								whyItMatters: `Request ID mikroservislar orqali so rovlarni kuzatish va loglarni korrelyatsiya qilishga imkon beradi.

**Nima uchun Request ID:**
- **Taqsimlangan tracing:** Bitta so rovni bir necha servislar orqali kuzatish
- **Log korrelyatsiyasi:** Bitta so rovning barcha loglarini \`grep request_id=xyz\` orqali guruhlash
- **Debugging:** Muayyan request ID bo yicha muammolarni takrorlash
- **Monitoring:** Sekin so rovlarni aniqlash va butun tizim bo ylab kuzatish

**Production namuna:**
\`\`\`go
func RequestID(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Mijoz ID taqdim etdimi tekshirish
        id := r.Header.Get("X-Request-ID")
        if id == "" {
            id = uuid.New().String()  // Yaxshiroq noyoblik uchun UUID
        }

        ctx := context.WithValue(r.Context(), RequestIDKey, id)
        w.Header().Set("X-Request-ID", id)

        // Strukturali loglarga qo shish
        log.WithField("request_id", id).Info("so rovni ishlov berish")

        next.ServeHTTP(w, r.WithContext(ctx))
    })
}
\`\`\`

**Haqiqiy foydalari:**
- **Hodisalarga javob:** \`grep "request_id=abc-123" logs/*\` muvaffaqiyatsiz so rovning barcha loglarini topadi
- **Performance tahlili:** Sekin so rovlarni butun tizimda kuzatish
- **Mijozlarni qo llab-quvvatlash:** Support ticketlar uchun request ID berish
- **A/B testing:** Request ID bo yicha eksperimental kohortalarni kuzatish

**Standart amaliyot:**
- AWS X-Ray: \`X-Amzn-Trace-Id\`
- Google Cloud: \`X-Cloud-Trace-Context\`
- Sanoat standarti: \`X-Request-ID\`

Request ID siz mikroservis arxitekturasida muammolarni debug qilish deyarli imkonsiz—turli servislardan loglarni bog lash mumkin emas.`
							}
						}
					},
					{
						slug: 'go-http-set-header',
						title: 'Set Header Middleware',
						difficulty: 'easy',
						tags: ['go', 'http', 'middleware', 'headers'],
						estimatedTime: '15m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement **SetHeader** middleware that ensures every response contains a specified header.

**Requirements:**
1. Create function \`SetHeader(name, value string, next http.Handler) http.Handler\`
2. Set the header using \`w.Header().Set(name, value)\`
3. Trim whitespace from header name
4. Return next handler unchanged if name is empty
5. Handle nil next handler gracefully

**Example:**
\`\`\`go
handler := SetHeader("X-API-Version", "v1.0", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    w.Write([]byte("Hello"))
}))

// Response will have header: X-API-Version: v1.0
\`\`\`

**Constraints:**
- Must set header before calling next handler
- Must handle empty header names`,
						initialCode: `package httpx

import (
	"net/http"
	"strings"
)

// TODO: Implement SetHeader middleware
func SetHeader(name, value string, next http.Handler) http.Handler {
	panic("TODO")
}`,
						solutionCode: `package httpx

import (
	"net/http"
	"strings"
)

func SetHeader(name, value string, next http.Handler) http.Handler {
	if next == nil {                                          // Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	headerName := strings.TrimSpace(name)                     // Remove leading/trailing whitespace
	if headerName == "" {                                     // Check if header name is empty
		return next                                       // No header to set, return next unchanged
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(headerName, value)                 // Set header in response
		next.ServeHTTP(w, r)                              // Pass to next handler
	})
}`,
						hint1: 'Trim whitespace from header name and check if it is empty.',
						hint2: 'Set the header using w.Header().Set() before calling next.ServeHTTP.',
						whyItMatters: `SetHeader middleware enables consistent response headers across all endpoints without modifying individual handlers.

**Why Set Headers:**
- **Security Headers:** Add CORS, CSP, or X-Frame-Options globally
- **API Versioning:** Indicate API version in every response
- **Caching:** Set Cache-Control headers consistently
- **Content Type:** Ensure proper Content-Type for all responses

**Production Pattern:**
\`\`\`go
// Security headers middleware
func SecurityHeaders(next http.Handler) http.Handler {
    return Chain(
        SetHeader("X-Content-Type-Options", "nosniff"),
        SetHeader("X-Frame-Options", "DENY"),
        SetHeader("X-XSS-Protection", "1; mode=block"),
        SetHeader("Strict-Transport-Security", "max-age=31536000"),
    )(next)
}

// CORS headers
func CORS(next http.Handler) http.Handler {
    return Chain(
        SetHeader("Access-Control-Allow-Origin", "*"),
        SetHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        SetHeader("Access-Control-Allow-Headers", "Content-Type"),
    )(next)
}
\`\`\`

**Real-World Benefits:**
- **Security Compliance:** OWASP recommends security headers on all responses
- **Performance:** Set caching headers to reduce server load
- **Client Compatibility:** Consistent headers prevent client-side issues
- **Observability:** Add server/version headers for debugging

Without centralized header setting, developers must remember to set headers in every handler—error-prone and unmaintainable.`,
						translations: {
							ru: {
								title: 'Set Header Middleware',
								description: `Реализуйте middleware **SetHeader**, который гарантирует, что каждый ответ содержит указанный заголовок.

**Требования:**
1. Создайте функцию \`SetHeader(name, value string, next http.Handler) http.Handler\`
2. Установите заголовок используя \`w.Header().Set(name, value)\`
3. Обрежьте пробелы из имени заголовка
4. Возвращайте next handler без изменений если name пустое
5. Обрабатывайте nil next handler корректно

**Пример:**
\`\`\`go
handler := SetHeader("X-API-Version", "v1.0", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    w.Write([]byte("Hello"))
}))

// Ответ будет иметь заголовок: X-API-Version: v1.0
\`\`\`

**Ограничения:**
- Должен устанавливать заголовок перед вызовом next
- Должен обрабатывать пустые имена заголовков`,
								hint1: 'Обрежьте пробелы из имени заголовка и проверьте на пустоту.',
								hint2: 'Установите заголовок через w.Header().Set() перед вызовом next.ServeHTTP.',
								whyItMatters: `SetHeader позволяет устанавливать консистентные заголовки для всех endpoints без изменения отдельных handlers.

**Почему устанавливать заголовки:**
- **Заголовки безопасности:** Добавление CORS, CSP или X-Frame-Options глобально
- **Версионирование API:** Указание версии API в каждом ответе
- **Кэширование:** Консистентная установка Cache-Control заголовков
- **Content Type:** Обеспечение правильного Content-Type для всех ответов

**Production паттерн:**
\`\`\`go
// Middleware для заголовков безопасности
func SecurityHeaders(next http.Handler) http.Handler {
    return Chain(
        SetHeader("X-Content-Type-Options", "nosniff"),
        SetHeader("X-Frame-Options", "DENY"),
        SetHeader("X-XSS-Protection", "1; mode=block"),
    )(next)
}
\`\`\`

Без централизованной установки заголовков разработчики должны помнить об установке в каждом handler—это подвержено ошибкам и не поддерживается.`
							},
							uz: {
								title: 'Set Header Middleware',
								description: `Har bir javob ko rsatilgan header ga ega bo lishini ta minlaydigan **SetHeader** middleware ni amalga oshiring.

**Talablar:**
1. \`SetHeader(name, value string, next http.Handler) http.Handler\` funksiyasini yarating
2. \`w.Header().Set(name, value)\` yordamida header ni o rnating
3. Header nomidan bo shliqlarni olib tashlang
4. Agar name bo sh bo lsa, next handler ni o zgarishsiz qaytaring
5. nil next handler ni to g ri ishlang

**Misol:**
\`\`\`go
handler := SetHeader("X-API-Version", "v1.0", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    w.Write([]byte("Hello"))
}))

// Javob headerga ega bo ladi: X-API-Version: v1.0
\`\`\`

**Cheklovlar:**
- next chaqirishdan oldin header ni o rnatishi kerak
- Bo sh header nomlarini ishlashi kerak`,
								hint1: 'Header nomidan bo shliqlarni olib tashlang va bo shligini tekshiring.',
								hint2: 'next.ServeHTTP chaqirishdan oldin w.Header().Set() orqali header ni o rnating.',
								whyItMatters: `SetHeader middleware alohida handlerlarni o zgartirmasdan barcha endpointlar uchun izchil headerlar o rnatishga imkon beradi.

**Nima uchun headerlar o rnatish:**
- **Xavfsizlik headerlari:** CORS, CSP yoki X-Frame-Options ni global qo shish
- **API versiyalash:** Har bir javobda API versiyasini ko rsatish
- **Kesh:** Cache-Control headerlarini izchil o rnatish
- **Content Type:** Barcha javoblar uchun to g ri Content-Type ni ta minlash

**Production namuna:**
\`\`\`go
// Xavfsizlik headerlari middleware
func SecurityHeaders(next http.Handler) http.Handler {
    return Chain(
        SetHeader("X-Content-Type-Options", "nosniff"),
        SetHeader("X-Frame-Options", "DENY"),
    )(next)
}
\`\`\`

Markazlashtirilgan header o rnatishsiz, dasturchilar har bir handlerda o rnatishni eslab qolishlari kerak—bu xatolarga moyil va qo llab-quvvatlanmaydi.`
							}
						}
					},
					{
						slug: 'go-http-require-method',
						title: 'Require Method Middleware',
						difficulty: 'easy',
						tags: ['go', 'http', 'middleware', 'validation'],
						estimatedTime: '15m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement **RequireMethod** middleware that only allows requests with a specific HTTP method and rejects others with 405 Method Not Allowed.

**Requirements:**
1. Create function \`RequireMethod(method string, next http.Handler) http.Handler\`
2. Compare \`r.Method\` with expected method (case-insensitive)
3. Return 405 status with \`Allow\` header if method doesn't match
4. Trim and uppercase both methods for comparison
5. Handle nil next handler and empty method gracefully

**Example:**
\`\`\`go
handler := RequireMethod("POST", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    w.Write([]byte("Data saved"))
}))

// GET request -> 405 Method Not Allowed, Allow: POST
// POST request -> 200 OK, "Data saved"
\`\`\`

**Constraints:**
- Must return 405 for mismatched methods
- Must set Allow header with expected method`,
						initialCode: `package httpx

import (
	"net/http"
	"strings"
)

// TODO: Implement RequireMethod middleware
func RequireMethod(method string, next http.Handler) http.Handler {
	panic("TODO")
}`,
						solutionCode: `package httpx

import (
	"net/http"
	"strings"
)

func RequireMethod(method string, next http.Handler) http.Handler {
	if next == nil {                                          // Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	want := strings.ToUpper(strings.TrimSpace(method))       // Normalize expected method
	if want == "" {                                           // If method is empty
		return next                                       // Skip validation, return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.ToUpper(r.Method) != want {            // Compare methods (case-insensitive)
			w.Header().Set("Allow", want)             // Set allowed method in header
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)  // Return 405
			return
		}
		next.ServeHTTP(w, r)                              // Method matches, proceed
	})
}`,
						hint1: 'Normalize both methods to uppercase using strings.ToUpper for case-insensitive comparison.',
						hint2: 'Set Allow header and return 405 status if methods don\'t match.',
						whyItMatters: `Method validation prevents security vulnerabilities and ensures API contracts are followed.

**Why Validate Methods:**
- **Security:** Prevent CSRF attacks by restricting GET requests on state-changing endpoints
- **API Contract:** Enforce REST conventions (GET for read, POST for create, etc.)
- **Client Errors:** Provide clear 405 responses instead of confusing 404/500 errors
- **Documentation:** Allowed methods self-document endpoint capabilities

**Production Pattern:**
\`\`\`go
// REST API with method validation
router.Handle("/users", RequireMethod("GET", ListUsersHandler))
router.Handle("/users", RequireMethod("POST", CreateUserHandler))
router.Handle("/users/{id}", RequireMethod("PUT", UpdateUserHandler))
router.Handle("/users/{id}", RequireMethod("DELETE", DeleteUserHandler))

// Combine with other middleware
func CreateUserEndpoint(next http.Handler) http.Handler {
    return Chain(
        RequireMethod("POST"),                    // Only POST allowed
        RequireHeader("Content-Type"),            // Must have Content-Type
        MaxBytes(1024 * 1024),                    // Limit body size to 1MB
    )(next)
}
\`\`\`

**Real-World Benefits:**
- **Security:** Many frameworks were vulnerable to CSRF because they didn't validate methods
- **Client Experience:** 405 with Allow header tells client exactly what methods are supported
- **Monitoring:** Track method-based errors separately from routing errors
- **Compliance:** REST API standards require 405 for incorrect methods

**HTTP Status Codes:**
- 405 Method Not Allowed: Request method not supported
- Allow header: Required with 405, lists allowed methods

Without method validation, DELETE requests might accidentally succeed on GET-only endpoints—catastrophic for data integrity.`,
						translations: {
							ru: {
								title: 'Require Method Middleware',
								description: `Реализуйте middleware **RequireMethod**, который разрешает запросы только с указанным HTTP методом и отклоняет остальные с 405 Method Not Allowed.

**Требования:**
1. Создайте функцию \`RequireMethod(method string, next http.Handler) http.Handler\`
2. Сравните \`r.Method\` с ожидаемым методом (без учёта регистра)
3. Верните 405 статус с заголовком \`Allow\` если метод не совпадает
4. Обрежьте и приведите оба метода к верхнему регистру для сравнения
5. Обрабатывайте nil next handler и пустой method корректно

**Пример:**
\`\`\`go
handler := RequireMethod("POST", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    w.Write([]byte("Data saved"))
}))

// GET запрос -> 405 Method Not Allowed, Allow: POST
// POST запрос -> 200 OK, "Data saved"
\`\`\`

**Ограничения:**
- Должен возвращать 405 для несоответствующих методов
- Должен устанавливать Allow заголовок с ожидаемым методом`,
								hint1: 'Нормализуйте оба метода в верхний регистр через strings.ToUpper для сравнения без учёта регистра.',
								hint2: 'Установите Allow заголовок и верните 405 статус если методы не совпадают.',
								whyItMatters: `Валидация методов предотвращает уязвимости безопасности и обеспечивает соблюдение API контрактов.

**Почему валидировать методы:**
- **Безопасность:** Предотвращение CSRF атак через ограничение GET запросов на endpoints изменяющих состояние
- **API Контракт:** Соблюдение REST конвенций (GET для чтения, POST для создания)
- **Ошибки клиента:** Предоставление чётких 405 ответов вместо путающих 404/500 ошибок
- **Документация:** Разрешённые методы самодокументируют возможности endpoint

**Production паттерн:**
\`\`\`go
// REST API с валидацией методов
router.Handle("/users", RequireMethod("GET", ListUsersHandler))
router.Handle("/users", RequireMethod("POST", CreateUserHandler))

// Комбинирование с другими middleware
func CreateUserEndpoint(next http.Handler) http.Handler {
    return Chain(
        RequireMethod("POST"),         // Только POST разрешён
        RequireHeader("Content-Type"),  // Должен иметь Content-Type
        MaxBytes(1024 * 1024),         // Ограничить размер тела до 1MB
    )(next)
}
\`\`\`

**Реальные преимущества:**
- **Безопасность:** Многие фреймворки были уязвимы к CSRF из-за отсутствия валидации методов
- **Клиентский опыт:** 405 с Allow заголовком точно сообщает какие методы поддерживаются
- **Мониторинг:** Отслеживание ошибок методов отдельно от ошибок роутинга
- **Соответствие:** REST API стандарты требуют 405 для неправильных методов

Без валидации методов DELETE запросы могут случайно успешно выполниться на GET-only endpoints—катастрофа для целостности данных.`
							},
							uz: {
								title: 'Require Method Middleware',
								description: `Faqat ko rsatilgan HTTP metodi bilan so rovlarga ruxsat beradigan va qolganlarini 405 Method Not Allowed bilan rad etadigan **RequireMethod** middleware ni amalga oshiring.

**Talablar:**
1. \`RequireMethod(method string, next http.Handler) http.Handler\` funksiyasini yarating
2. \`r.Method\` ni kutilgan metod bilan solishtiring (katta-kichik harflar farqsiz)
3. Agar metod mos kelmasa 405 status va \`Allow\` headeri bilan qaytaring
4. Solishtirish uchun ikkala metoddan bo shliqlarni olib tashlang va katta harfga o tkazing
5. nil next handler va bo sh metodlarni to g ri ishlang

**Misol:**
\`\`\`go
handler := RequireMethod("POST", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    w.Write([]byte("Data saved"))
}))

// GET so rov -> 405 Method Not Allowed, Allow: POST
// POST so rov -> 200 OK, "Data saved"
\`\`\`

**Cheklovlar:**
- Mos kelmaydigan metodlar uchun 405 qaytarishi kerak
- Kutilgan metod bilan Allow headerini o rnatishi kerak`,
								hint1: 'Katta-kichik harflar farqsiz solishtirish uchun ikkala metoddan strings.ToUpper orqali normalizatsiya qiling.',
								hint2: 'Agar metodlar mos kelmasa, Allow headerini o rnating va 405 statusni qaytaring.',
								whyItMatters: `Metod validatsiyasi xavfsizlik zaifliklarini oldini oladi va API shartnomalarga rioya qilinishini ta minlaydi.

**Nima uchun metodlarni validatsiya qilish:**
- **Xavfsizlik:** Holat o zgartiradigan endpointlarda GET so rovlarini cheklash orqali CSRF hujumlarini oldini olish
- **API Shartnoma:** REST konvensiyalariga rioya qilish (GET o qish uchun, POST yaratish uchun)
- **Mijoz xatolari:** Chalkash 404/500 xatolar o rniga aniq 405 javoblar berish
- **Hujjatlashtirish:** Ruxsat berilgan metodlar endpoint imkoniyatlarini o z-o zidan hujjatlashtiradi

**Production namuna:**
\`\`\`go
// Metod validatsiyali REST API
router.Handle("/users", RequireMethod("GET", ListUsersHandler))
router.Handle("/users", RequireMethod("POST", CreateUserHandler))

// Boshqa middleware lar bilan birlashtirishSecurity vulnerability example: Without method validation, someone could accidentally expose DELETE functionality via GET, allowing attackers to delete data just by visiting a URL in their browser—which automatically sends GET requests.`
							}
						}
					},
					{
						slug: 'go-http-require-header',
						title: 'Require Header Middleware',
						difficulty: 'easy',
						tags: ['go', 'http', 'middleware', 'validation'],
						estimatedTime: '15m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement **RequireHeader** middleware that validates presence of a required request header and returns 400 Bad Request if missing.

**Requirements:**
1. Create function \`RequireHeader(name string, next http.Handler) http.Handler\`
2. Check if header exists using \`r.Header.Get(name)\`
3. Trim whitespace from header value
4. Return 400 if header is missing or empty
5. Handle nil next handler and empty name gracefully

**Example:**
\`\`\`go
handler := RequireHeader("Authorization", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    token := r.Header.Get("Authorization")
    w.Write([]byte("Authenticated: " + token))
}))

// Request without Authorization header -> 400 Bad Request
// Request with Authorization header -> 200 OK
\`\`\`

**Constraints:**
- Must check header is not empty after trimming
- Must return 400 for missing headers`,
						initialCode: `package httpx

import (
	"net/http"
	"strings"
)

// TODO: Implement RequireHeader middleware
func RequireHeader(name string, next http.Handler) http.Handler {
	panic("TODO")
}`,
						solutionCode: `package httpx

import (
	"net/http"
	"strings"
)

func RequireHeader(name string, next http.Handler) http.Handler {
	if next == nil {                                          // Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	headerName := strings.TrimSpace(name)                     // Remove whitespace from header name
	if headerName == "" {                                     // If header name is empty
		return next                                       // Skip validation, return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.TrimSpace(r.Header.Get(headerName)) == "" {  // Check header exists and non-empty
			http.Error(w, "missing required header", http.StatusBadRequest)  // Return 400
			return
		}
		next.ServeHTTP(w, r)                              // Header present, proceed
	})
}`,
						hint1: 'Use r.Header.Get() to retrieve header value and trim whitespace.',
						hint2: 'Return 400 Bad Request if header is missing or empty after trimming.',
						whyItMatters: `Header validation ensures API contracts are met and prevents downstream errors from missing required data.

**Why Require Headers:**
- **Authentication:** Validate Authorization header before processing requests
- **Content Negotiation:** Require Content-Type for POST/PUT requests
- **API Versioning:** Require API-Version header for version-aware endpoints
- **Idempotency:** Require Idempotency-Key for duplicate-sensitive operations

**Production Pattern:**
\`\`\`go
// Authentication middleware
func AuthenticatedEndpoint(next http.Handler) http.Handler {
    return Chain(
        RequireHeader("Authorization"),
        validateToken,  // Custom middleware to validate JWT
    )(next)
}

// Content-Type validation for POST endpoints
func JSONEndpoint(next http.Handler) http.Handler {
    return Chain(
        RequireMethod("POST"),
        RequireHeader("Content-Type"),
        validateContentType("application/json"),
    )(next)
}

// Idempotent operations
func IdempotentEndpoint(next http.Handler) http.Handler {
    return Chain(
        RequireHeader("Idempotency-Key"),
        checkDuplicateKey,  // Prevent duplicate operations
    )(next)
}
\`\`\`

**Real-World Benefits:**
- **Early Validation:** Fail fast at middleware layer instead of deep in business logic
- **Clear Errors:** 400 response tells client exactly what's missing
- **Security:** Prevent unauthenticated requests from reaching handlers
- **Compliance:** Many APIs require specific headers (OAuth, CORS, etc.)

**Common Required Headers:**
- Authorization: Bearer tokens, API keys
- Content-Type: application/json, application/xml
- Accept: Response format preference
- X-Request-ID: Request tracking
- X-API-Key: API authentication

Without header validation, missing Authorization headers could crash your authentication logic—better to reject at the middleware layer with clear 400 errors.`,
						translations: {
							ru: {
								title: 'Require Header Middleware',
								description: `Реализуйте middleware **RequireHeader**, который валидирует наличие обязательного заголовка запроса и возвращает 400 Bad Request если отсутствует.

**Требования:**
1. Создайте функцию \`RequireHeader(name string, next http.Handler) http.Handler\`
2. Проверьте наличие заголовка используя \`r.Header.Get(name)\`
3. Обрежьте пробелы из значения заголовка
4. Верните 400 если заголовок отсутствует или пустой
5. Обрабатывайте nil next handler и пустой name корректно

**Пример:**
\`\`\`go
handler := RequireHeader("Authorization", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    token := r.Header.Get("Authorization")
    w.Write([]byte("Authenticated: " + token))
}))

// Запрос без Authorization заголовка -> 400 Bad Request
// Запрос с Authorization заголовком -> 200 OK
\`\`\`

**Ограничения:**
- Должен проверять что заголовок не пустой после обрезки
- Должен возвращать 400 для отсутствующих заголовков`,
								hint1: 'Используйте r.Header.Get() для получения значения заголовка и обрежьте пробелы.',
								hint2: 'Верните 400 Bad Request если заголовок отсутствует или пустой после обрезки.',
								whyItMatters: `Валидация заголовков обеспечивает соблюдение API контрактов и предотвращает ошибки downstream от отсутствующих данных.

**Почему требовать заголовки:**
- **Аутентификация:** Валидация Authorization заголовка перед обработкой запросов
- **Content Negotiation:** Требование Content-Type для POST/PUT запросов
- **Версионирование API:** Требование API-Version заголовка для version-aware endpoints
- **Идемпотентность:** Требование Idempotency-Key для операций чувствительных к дубликатам

**Production паттерн:**
\`\`\`go
// Middleware аутентификации
func AuthenticatedEndpoint(next http.Handler) http.Handler {
    return Chain(
        RequireHeader("Authorization"),
        validateToken,  // Пользовательский middleware для валидации JWT
    )(next)
}

// Валидация Content-Type для POST endpoints
func JSONEndpoint(next http.Handler) http.Handler {
    return Chain(
        RequireMethod("POST"),
        RequireHeader("Content-Type"),
        validateContentType("application/json"),
    )(next)
}
\`\`\`

**Реальные преимущества:**
- **Ранняя валидация:** Быстрый отказ на уровне middleware вместо глубоко в бизнес-логике
- **Чёткие ошибки:** 400 ответ точно сообщает что отсутствует
- **Безопасность:** Предотвращение не аутентифицированных запросов от достижения handlers
- **Соответствие:** Многие API требуют специфические заголовки (OAuth, CORS и т.д.)

Без валидации заголовков отсутствующие Authorization заголовки могут обрушить вашу логику аутентификации—лучше отклонить на уровне middleware с чёткими 400 ошибками.`
							},
							uz: {
								title: 'Require Header Middleware',
								description: `Majburiy so rov headerining mavjudligini validatsiya qiladigan va yo q bo lsa 400 Bad Request qaytaradigan **RequireHeader** middleware ni amalga oshiring.

**Talablar:**
1. \`RequireHeader(name string, next http.Handler) http.Handler\` funksiyasini yarating
2. \`r.Header.Get(name)\` yordamida header mavjudligini tekshiring
3. Header qiymatidan bo shliqlarni olib tashlang
4. Agar header yo q yoki bo sh bo lsa 400 qaytaring
5. nil next handler va bo sh name ni to g ri ishlang

**Misol:**
\`\`\`go
handler := RequireHeader("Authorization", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    token := r.Header.Get("Authorization")
    w.Write([]byte("Authenticated: " + token))
}))

// Authorization headersiz so rov -> 400 Bad Request
// Authorization headerli so rov -> 200 OK
\`\`\`

**Cheklovlar:**
- Bo shliqlarni olib tashlgandan keyin header bo sh emasligini tekshirishi kerak
- Yo q headerlar uchun 400 qaytarishi kerak`,
								hint1: 'Header qiymatini olish uchun r.Header.Get() dan foydalaning va bo shliqlarni olib tashlang.',
								hint2: 'Agar header yo q yoki bo shliqlarni olib tashlgandan keyin bo sh bo lsa 400 Bad Request qaytaring.',
								whyItMatters: `Header validatsiyasi API shartnomalarga rioya qilinishini ta minlaydi va yo q ma lumotlardan downstream xatolarni oldini oladi.

**Nima uchun headerlarni talab qilish:**
- **Autentifikatsiya:** So rovlarni ishlov berishdan oldin Authorization headerini validatsiya qilish
- **Content Negotiation:** POST/PUT so rovlari uchun Content-Type ni talab qilish
- **API versiyalash:** Version-aware endpointlar uchun API-Version headerini talab qilish
- **Idempotentlik:** Dublikatga sezgir operatsiyalar uchun Idempotency-Key ni talab qilish

**Production namuna:**
\`\`\`go
// Autentifikatsiya middleware
func AuthenticatedEndpoint(next http.Handler) http.Handler {
    return Chain(
        RequireHeader("Authorization"),
        validateToken,  // JWT ni validatsiya qilish uchun custom middleware
    )(next)
}
\`\`\`

Header validatsiyasisiz yo q Authorization headerlari autentifikatsiya mantiqingizni buzishi mumkin—middleware qatlamida aniq 400 xatolar bilan rad etish yaxshiroq.`
							}
						}
					}
				]
			}
		]
	}
];
