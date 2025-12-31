import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-header-to-context',
	title: 'Header to Context Middleware',
	difficulty: 'medium',	tags: ['go', 'http', 'middleware', 'context'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **HeaderToContext** middleware that extracts a header value and stores it in request context.

**Requirements:**
1. Create function \`HeaderToContext(key ctxKey, header string, next http.Handler) http.Handler\`
2. Skip middleware if header name is empty
3. Get header value from request
4. Store value in context using provided key
5. Pass modified request to next handler
6. Handle nil next handler

**Example:**
\`\`\`go
const UserIDKey ctxKey = "user_id"

handler := HeaderToContext(UserIDKey, "X-User-ID", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    userID := r.Context().Value(UserIDKey).(string)
    fmt.Fprintf(w, "User ID: %s", userID)
}))

// Request with X-User-ID: 12345 → Response: User ID: 12345
\`\`\`

**Constraints:**
- Must use context.WithValue to store header
- Must pass modified request with r.WithContext()`,
	initialCode: `package httpx

import (
	"context"
	"net/http"
)

type ctxKey string

// TODO: Implement HeaderToContext middleware
func HeaderToContext(key ctxKey, header string, next http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"context"
	"net/http"
)

type ctxKey string

func HeaderToContext(key ctxKey, header string, next http.Handler) http.Handler {
	if next == nil {	// Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	if header == "" {	// Empty header check
		return next	// Skip middleware if no header name
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		val := r.Header.Get(header)	// Get header value from request
		ctx := context.WithValue(r.Context(), key, val)	// Store value in context with key
		next.ServeHTTP(w, r.WithContext(ctx))	// Pass request with updated context
	})
}`,
	testCode: `package httpx

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

type ctxKey string

const TestKey ctxKey = "test_key"

func Test1HeaderToContextWithValidHeader(t *testing.T) {
	handler := HeaderToContext(TestKey, "X-Test", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		val := r.Context().Value(TestKey).(string)
		if val != "test-value" {
			t.Errorf("expected 'test-value', got '%s'", val)
		}
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("X-Test", "test-value")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rec.Code)
	}
}

func Test2HeaderToContextMissingHeader(t *testing.T) {
	handler := HeaderToContext(TestKey, "X-Test", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		val := r.Context().Value(TestKey).(string)
		if val != "" {
			t.Errorf("expected empty string, got '%s'", val)
		}
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rec.Code)
	}
}

func Test3HeaderToContextEmptyHeaderName(t *testing.T) {
	nextCalled := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		nextCalled = true
		w.WriteHeader(http.StatusOK)
	})
	handler := HeaderToContext(TestKey, "", next)
	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if !nextCalled {
		t.Error("expected next handler to be called when header name is empty")
	}
}

func Test4HeaderToContextNilHandler(t *testing.T) {
	handler := HeaderToContext(TestKey, "X-Test", nil)
	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("X-Test", "test-value")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rec.Code)
	}
}

func Test5HeaderToContextMultipleValues(t *testing.T) {
	handler := HeaderToContext(TestKey, "X-Test", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		val := r.Context().Value(TestKey).(string)
		if val != "value1" {
			t.Errorf("expected 'value1', got '%s'", val)
		}
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Add("X-Test", "value1")
	req.Header.Add("X-Test", "value2")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
}

func Test6HeaderToContextPreservesExistingContext(t *testing.T) {
	const ExistingKey ctxKey = "existing"
	handler := HeaderToContext(TestKey, "X-Test", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		existingVal := r.Context().Value(ExistingKey).(string)
		newVal := r.Context().Value(TestKey).(string)
		if existingVal != "existing-value" || newVal != "new-value" {
			t.Errorf("context values not preserved correctly")
		}
		w.WriteHeader(http.StatusOK)
	}))
	ctx := context.WithValue(context.Background(), ExistingKey, "existing-value")
	req := httptest.NewRequest("GET", "/test", nil).WithContext(ctx)
	req.Header.Set("X-Test", "new-value")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
}

func Test7HeaderToContextCaseInsensitive(t *testing.T) {
	handler := HeaderToContext(TestKey, "X-Test", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		val := r.Context().Value(TestKey).(string)
		if val != "test-value" {
			t.Errorf("expected 'test-value', got '%s'", val)
		}
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("x-test", "test-value")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
}

func Test8HeaderToContextEmptyHeaderValue(t *testing.T) {
	handler := HeaderToContext(TestKey, "X-Test", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		val := r.Context().Value(TestKey).(string)
		if val != "" {
			t.Errorf("expected empty string, got '%s'", val)
		}
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("X-Test", "")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
}

func Test9HeaderToContextChaining(t *testing.T) {
	const Key2 ctxKey = "key2"
	handler := HeaderToContext(TestKey, "X-Test1",
		HeaderToContext(Key2, "X-Test2", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			val1 := r.Context().Value(TestKey).(string)
			val2 := r.Context().Value(Key2).(string)
			if val1 != "value1" || val2 != "value2" {
				t.Errorf("chaining failed: val1=%s, val2=%s", val1, val2)
			}
			w.WriteHeader(http.StatusOK)
		})))
	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("X-Test1", "value1")
	req.Header.Set("X-Test2", "value2")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
}

func Test10HeaderToContextSpecialCharacters(t *testing.T) {
	handler := HeaderToContext(TestKey, "X-Test", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		val := r.Context().Value(TestKey).(string)
		expected := "special!@#$%^&*()_+-={}[]|:;<>?,./~"
		if val != expected {
			t.Errorf("expected '%s', got '%s'", expected, val)
		}
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("X-Test", "special!@#$%^&*()_+-={}[]|:;<>?,./~")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
}`,
			hint1: `Use r.Header.Get(header) to retrieve the header value.`,
			hint2: `Use context.WithValue() to add the value to context, then pass r.WithContext(ctx) to next.`,
			whyItMatters: `HeaderToContext bridges HTTP headers and Go contexts, enabling downstream handlers to access header values without re-parsing.

**Why Header-to-Context:**
- **Clean Handlers:** Handlers access context values instead of parsing headers
- **Middleware Chain:** Context flows through entire handler chain
- **Decoupling:** Handlers don't need to know about HTTP headers
- **Testing:** Easy to inject values via context in tests

**Production Pattern:**
\`\`\`go
const (
    UserIDKey     ctxKey = "user_id"
    TenantIDKey   ctxKey = "tenant_id"
    TraceIDKey    ctxKey = "trace_id"
    SessionIDKey  ctxKey = "session_id"
)

// Authentication middleware
func AuthMiddleware(next http.Handler) http.Handler {
    return HeaderToContext(UserIDKey, "X-User-ID", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Validate user ID from context
        userID := r.Context().Value(UserIDKey).(string)
        if userID == "" {
            http.Error(w, "missing user ID", http.StatusUnauthorized)
            return
        }

	// Add more user data to context
        user := fetchUser(userID)
        ctx := context.WithValue(r.Context(), "user", user)
        next.ServeHTTP(w, r.WithContext(ctx))
    }))
}

// Multi-tenant application
func TenantMiddleware(next http.Handler) http.Handler {
    return Chain(
        HeaderToContext(TenantIDKey, "X-Tenant-ID"),
        HeaderToContext(TraceIDKey, "X-Trace-ID"),
    )(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        tenantID := r.Context().Value(TenantIDKey).(string)

	// Configure database connection for tenant
        db := getDatabase(tenantID)
        ctx := context.WithValue(r.Context(), "db", db)

        next.ServeHTTP(w, r.WithContext(ctx))
    }))
}

// Handler using context values
func GetUser(w http.ResponseWriter, r *http.Request) {
    userID := r.Context().Value(UserIDKey).(string)
    tenantID := r.Context().Value(TenantIDKey).(string)
    traceID := r.Context().Value(TraceIDKey).(string)

    log.Printf("trace=%s tenant=%s user=%s", traceID, tenantID, userID)

    user := fetchUser(tenantID, userID)
    json.NewEncoder(w).Encode(user)
}

// Propagate context to downstream services
func CallDownstream(ctx context.Context, url string) (*Response, error) {
    req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)

	// Propagate values as headers
    if userID := ctx.Value(UserIDKey); userID != nil {
        req.Header.Set("X-User-ID", userID.(string))
    }
    if traceID := ctx.Value(TraceIDKey); traceID != nil {
        req.Header.Set("X-Trace-ID", traceID.(string))
    }

    return http.DefaultClient.Do(req)
}
\`\`\`

**Real-World Benefits:**
- **Request Scoping:** Context values are request-scoped, no global state
- **Cancellation:** Context carries cancellation signals
- **Deadlines:** Context propagates timeouts
- **Tracing:** Trace IDs flow through entire request lifecycle

**Context Best Practices:**
- **Type Safety:** Use typed keys (ctxKey type) to avoid collisions
- **Immutability:** Contexts are immutable, WithValue creates new context
- **No Mutation:** Never store pointers to mutable data
- **Cancel Properly:** Always defer cancel() when using WithTimeout/WithCancel

Without HeaderToContext, every handler must manually extract and validate headers, leading to duplicate code and missed validations.`,	order: 7,
	translations: {
		ru: {
			title: 'Сохранение заголовка в контексте запроса',
			description: `Реализуйте middleware **HeaderToContext**, который извлекает значение заголовка и сохраняет его в контексте запроса.

**Требования:**
1. Создайте функцию \`HeaderToContext(key ctxKey, header string, next http.Handler) http.Handler\`
2. Пропустите middleware если имя заголовка пустое
3. Получите значение заголовка из запроса
4. Сохраните значение в контексте с предоставленным ключом
5. Передайте модифицированный запрос следующему handler
6. Обработайте nil handler

**Пример:**
\`\`\`go
const UserIDKey ctxKey = "user_id"

handler := HeaderToContext(UserIDKey, "X-User-ID", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    userID := r.Context().Value(UserIDKey).(string)
    fmt.Fprintf(w, "User ID: %s", userID)
}))

// Запрос с X-User-ID: 12345 → Ответ: User ID: 12345
\`\`\`

**Ограничения:**
- Должен использовать context.WithValue для сохранения заголовка
- Должен передавать модифицированный запрос через r.WithContext()`,
			hint1: `Используйте r.Header.Get(header) для получения значения заголовка.`,
			hint2: `Используйте context.WithValue() для добавления значения в контекст, затем передайте r.WithContext(ctx) в next.`,
			whyItMatters: `HeaderToContext связывает HTTP заголовки и Go контексты, позволяя downstream handlers получать значения заголовков без повторного парсинга.

**Почему Header-to-Context:**
- **Чистые handlers:** Handlers получают значения из контекста вместо парсинга заголовков
- **Цепочка middleware:** Контекст проходит через всю цепочку handlers
- **Развязка:** Handlers не нужно знать об HTTP заголовках
- **Тестирование:** Легко инжектить значения через контекст в тестах

**Продакшен паттерн:**
\`\`\`go
const (
    UserIDKey     ctxKey = "user_id"
    TenantIDKey   ctxKey = "tenant_id"
    TraceIDKey    ctxKey = "trace_id"
    SessionIDKey  ctxKey = "session_id"
)

// Middleware аутентификации
func AuthMiddleware(next http.Handler) http.Handler {
    return HeaderToContext(UserIDKey, "X-User-ID", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        userID := r.Context().Value(UserIDKey).(string)
        if userID == "" {
            http.Error(w, "missing user ID", http.StatusUnauthorized)
            return
        }

        user := fetchUser(userID)
        ctx := context.WithValue(r.Context(), "user", user)
        next.ServeHTTP(w, r.WithContext(ctx))
    }))
}

// Multi-tenant приложение
func TenantMiddleware(next http.Handler) http.Handler {
    return Chain(
        HeaderToContext(TenantIDKey, "X-Tenant-ID"),
        HeaderToContext(TraceIDKey, "X-Trace-ID"),
    )(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        tenantID := r.Context().Value(TenantIDKey).(string)

        db := getDatabase(tenantID)
        ctx := context.WithValue(r.Context(), "db", db)

        next.ServeHTTP(w, r.WithContext(ctx))
    }))
}

// Handler использующий значения контекста
func GetUser(w http.ResponseWriter, r *http.Request) {
    userID := r.Context().Value(UserIDKey).(string)
    tenantID := r.Context().Value(TenantIDKey).(string)
    traceID := r.Context().Value(TraceIDKey).(string)

    log.Printf("trace=%s tenant=%s user=%s", traceID, tenantID, userID)

    user := fetchUser(tenantID, userID)
    json.NewEncoder(w).Encode(user)
}
\`\`\`

**Практические преимущества:**
- **Scope запроса:** Значения контекста привязаны к запросу, нет глобального состояния
- **Отмена:** Контекст несёт сигналы отмены
- **Дедлайны:** Контекст пропагирует таймауты
- **Трассировка:** Trace ID проходит через весь жизненный цикл запроса

**Best practices для контекста:**
- **Типобезопасность:** Используйте типизированные ключи (тип ctxKey) для избежания коллизий
- **Иммутабельность:** Контексты неизменяемы, WithValue создаёт новый контекст
- **Без мутаций:** Никогда не храните указатели на изменяемые данные
- **Правильная отмена:** Всегда defer cancel() при использовании WithTimeout/WithCancel

Без HeaderToContext каждый handler должен вручную извлекать и валидировать заголовки, что ведёт к дублированию кода.`,
			solutionCode: `package httpx

import (
	"context"
	"net/http"
)

type ctxKey string

func HeaderToContext(key ctxKey, header string, next http.Handler) http.Handler {
	if next == nil {	// Обработка nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	if header == "" {	// Проверка на пустой заголовок
		return next	// Пропуск middleware если нет имени заголовка
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		val := r.Header.Get(header)	// Получение значения заголовка из запроса
		ctx := context.WithValue(r.Context(), key, val)	// Сохранение значения в контексте с ключом
		next.ServeHTTP(w, r.WithContext(ctx))	// Передача запроса с обновленным контекстом
	})
}`
		},
		uz: {
			title: 'Headerni request kontekstida saqlash',
			description: `Header qiymatini ajratib oluvchi va request kontekstida saqlaydigan **HeaderToContext** middleware ni amalga oshiring.

**Talablar:**
1. \`HeaderToContext(key ctxKey, header string, next http.Handler) http.Handler\` funksiyasini yarating
2. Agar header nomi bo'sh bo'lsa middleware ni o'tkazing
3. Requestdan header qiymatini oling
4. Berilgan kalit bilan kontekstda qiymatni saqlang
5. Keyingi handlerga o'zgartirilgan requestni o'tkazing
6. nil handlerni ishlang

**Misol:**
\`\`\`go
const UserIDKey ctxKey = "user_id"

handler := HeaderToContext(UserIDKey, "X-User-ID", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    userID := r.Context().Value(UserIDKey).(string)
    fmt.Fprintf(w, "User ID: %s", userID)
}))

// X-User-ID: 12345 bilan request → Response: User ID: 12345
\`\`\`

**Cheklovlar:**
- Headerni saqlash uchun context.WithValue dan foydalanishi kerak
- O'zgartirilgan requestni r.WithContext() orqali o'tkazishi kerak`,
			hint1: `Header qiymatini olish uchun r.Header.Get(header) dan foydalaning.`,
			hint2: `Kontekstga qiymat qo'shish uchun context.WithValue() dan, keyin next ga r.WithContext(ctx) ni o'tkazing.`,
			whyItMatters: `HeaderToContext HTTP headerlar va Go kontekstlarini bog'laydi, downstream handlerlar headerlarda qayta parsing qilmasdan qiymatlarni olish imkonini beradi.

**Nima uchun Header-to-Context:**
- **Toza handlerlar:** Handlerlar headerlarni parsing qilish o'rniga kontekstdan qiymatlarni oladi
- **Middleware zanjiri:** Kontekst butun handler zanjiri orqali o'tadi
- **Bog'liqlikni kamaytirish:** Handlerlar HTTP headerlar haqida bilishi shart emas
- **Testlash:** Testlarda kontekst orqali qiymatlarni osonlikcha inject qilish

**Ishlab chiqarish patterni:**
\`\`\`go
const (
    UserIDKey     ctxKey = "user_id"
    TenantIDKey   ctxKey = "tenant_id"
    TraceIDKey    ctxKey = "trace_id"
    SessionIDKey  ctxKey = "session_id"
)

// Autentifikatsiya middlewaresi
func AuthMiddleware(next http.Handler) http.Handler {
    return HeaderToContext(UserIDKey, "X-User-ID", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        userID := r.Context().Value(UserIDKey).(string)
        if userID == "" {
            http.Error(w, "missing user ID", http.StatusUnauthorized)
            return
        }

        user := fetchUser(userID)
        ctx := context.WithValue(r.Context(), "user", user)
        next.ServeHTTP(w, r.WithContext(ctx))
    }))
}

// Multi-tenant ilova
func TenantMiddleware(next http.Handler) http.Handler {
    return Chain(
        HeaderToContext(TenantIDKey, "X-Tenant-ID"),
        HeaderToContext(TraceIDKey, "X-Trace-ID"),
    )(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        tenantID := r.Context().Value(TenantIDKey).(string)

        db := getDatabase(tenantID)
        ctx := context.WithValue(r.Context(), "db", db)

        next.ServeHTTP(w, r.WithContext(ctx))
    }))
}

// Kontekst qiymatlaridan foydalanuvchi handler
func GetUser(w http.ResponseWriter, r *http.Request) {
    userID := r.Context().Value(UserIDKey).(string)
    tenantID := r.Context().Value(TenantIDKey).(string)
    traceID := r.Context().Value(TraceIDKey).(string)

    log.Printf("trace=%s tenant=%s user=%s", traceID, tenantID, userID)

    user := fetchUser(tenantID, userID)
    json.NewEncoder(w).Encode(user)
}
\`\`\`

**Amaliy foydalari:**
- **Request scope:** Kontekst qiymatlari requestga bog'langan, global holat yo'q
- **Bekor qilish:** Kontekst bekor qilish signallarini o'tkazadi
- **Deadlinelar:** Kontekst timeoutlarni tarqatadi
- **Tracing:** Trace ID butun request hayot sikli orqali o'tadi

**Kontekst uchun best practices:**
- **Tip xavfsizligi:** To'qnashuvlardan qochish uchun tiplangan kalitlardan foydalaning (ctxKey tipi)
- **O'zgarmaslik:** Kontekstlar o'zgarmas, WithValue yangi kontekst yaratadi
- **Mutatsiyasiz:** O'zgaruvchan ma'lumotlarga ko'rsatkichlarni hech qachon saqlamang
- **To'g'ri bekor qilish:** WithTimeout/WithCancel ishlatganda har doim defer cancel()

HeaderToContext siz har bir handler headerlarni qo'lda ajratib olishi va validatsiya qilishi kerak, bu kodning takrorlanishiga olib keladi.`,
			solutionCode: `package httpx

import (
	"context"
	"net/http"
)

type ctxKey string

func HeaderToContext(key ctxKey, header string, next http.Handler) http.Handler {
	if next == nil {	// nil next handlerni ishlash
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	if header == "" {	// Bo'sh header tekshiruvi
		return next	// Header nomi bo'lmasa middleware ni o'tkazish
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		val := r.Header.Get(header)	// Requestdan header qiymatini olish
		ctx := context.WithValue(r.Context(), key, val)	// Kalit bilan kontekstda qiymatni saqlash
		next.ServeHTTP(w, r.WithContext(ctx))	// Yangilangan kontekst bilan requestni o'tkazish
	})
}`
		}
	}
};

export default task;
