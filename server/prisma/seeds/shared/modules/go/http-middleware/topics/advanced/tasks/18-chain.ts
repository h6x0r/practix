import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-chain',
	title: 'Chain Middleware Composer',
	difficulty: 'medium',	tags: ['go', 'http', 'middleware', 'composition'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Chain** function that combines multiple middleware functions into a single middleware, applying them left-to-right.

**Requirements:**
1. Create function \`Chain(middlewares ...func(http.Handler) http.Handler) func(http.Handler) http.Handler\`
2. Return a function that takes final handler
3. Apply middlewares from right to left (to execute left to right)
4. Skip nil middlewares
5. Handle nil final handler
6. Return composed handler

**Example:**
\`\`\`go
handler := Chain(
    Logger,	// Executes first
    RequestID,	// Executes second
    Recover,	// Executes third
)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Final handler")
}))

// Request flow: Logger → RequestID → Recover → Final handler
\`\`\`

**Constraints:**
- Must apply middlewares right-to-left to achieve left-to-right execution
- Must skip nil middleware functions
- Must handle nil final handler`,
	initialCode: `package httpx

import (
	"net/http"
)

// TODO: Implement Chain middleware composer
func Chain(middlewares ...func(http.Handler) http.Handler) func(http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"net/http"
)

func Chain(middlewares ...func(http.Handler) http.Handler) func(http.Handler) http.Handler {
	return func(final http.Handler) http.Handler {	// Return function that takes final handler
		if final == nil {	// Check if final handler is nil
			final = http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})	// Use no-op handler
		}
		wrapped := final	// Start with final handler
		for i := len(middlewares) - 1; i >= 0; i-- {	// Iterate backwards (right-to-left)
			mw := middlewares[i]	// Get current middleware
			if mw == nil {	// Skip nil middlewares
				continue
			}
			wrapped = mw(wrapped)	// Wrap handler with middleware
		}
		return wrapped	// Return fully wrapped handler
	}
}`,
			hint1: `Iterate through middlewares backwards (len-1 to 0) to wrap them right-to-left for left-to-right execution.`,
			hint2: `Start with final handler, wrap it with each middleware in reverse order, skip nil middlewares.`,
			testCode: `package httpx

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func Test1(t *testing.T) {
	order := ""
	mw1 := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			order += "1"
			next.ServeHTTP(w, r)
		})
	}
	mw2 := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			order += "2"
			next.ServeHTTP(w, r)
		})
	}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		order += "H"
	})
	h := Chain(mw1, mw2)(handler)
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if order != "12H" {
		t.Errorf("expected order '12H', got %q", order)
	}
}

func Test2(t *testing.T) {
	order := ""
	mw1 := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			order += "A"
			next.ServeHTTP(w, r)
			order += "a"
		})
	}
	mw2 := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			order += "B"
			next.ServeHTTP(w, r)
			order += "b"
		})
	}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		order += "H"
	})
	h := Chain(mw1, mw2)(handler)
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if order != "ABHba" {
		t.Errorf("expected order 'ABHba', got %q", order)
	}
}

func Test3(t *testing.T) {
	h := Chain()(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
	}))
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Errorf("expected 201, got %d", rec.Code)
	}
}

func Test4(t *testing.T) {
	order := ""
	mw1 := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			order += "1"
			next.ServeHTTP(w, r)
		})
	}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		order += "H"
	})
	h := Chain(mw1, nil)(handler)
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if order != "1H" {
		t.Errorf("expected order '1H' with nil skipped, got %q", order)
	}
}

func Test5(t *testing.T) {
	order := ""
	mw := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			order += "M"
			next.ServeHTTP(w, r)
		})
	}
	h := Chain(nil, mw, nil)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		order += "H"
	}))
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if order != "MH" {
		t.Errorf("expected 'MH', got %q", order)
	}
}

func Test6(t *testing.T) {
	h := Chain()(nil)
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected 200 for nil final handler, got %d", rec.Code)
	}
}

func Test7(t *testing.T) {
	order := ""
	mw1 := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			order += "1"
			next.ServeHTTP(w, r)
		})
	}
	mw2 := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			order += "2"
			next.ServeHTTP(w, r)
		})
	}
	mw3 := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			order += "3"
			next.ServeHTTP(w, r)
		})
	}
	h := Chain(mw1, mw2, mw3)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		order += "H"
	}))
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if order != "123H" {
		t.Errorf("expected '123H', got %q", order)
	}
}

func Test8(t *testing.T) {
	mw := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusForbidden)
		})
	}
	h := Chain(mw)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Errorf("expected 403 from short-circuit, got %d", rec.Code)
	}
}

func Test9(t *testing.T) {
	counter := 0
	mw := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			counter++
			next.ServeHTTP(w, r)
		})
	}
	h := Chain(mw, mw, mw)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if counter != 3 {
		t.Errorf("expected counter 3, got %d", counter)
	}
}

func Test10(t *testing.T) {
	mw := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("X-Middleware", "applied")
			next.ServeHTTP(w, r)
		})
	}
	h := Chain(mw)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
	}))
	req := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Header().Get("X-Middleware") != "applied" {
		t.Error("expected middleware header")
	}
	if rec.Code != http.StatusAccepted {
		t.Errorf("expected 202, got %d", rec.Code)
	}
}
`,
			whyItMatters: `Chain enables clean, readable middleware composition, making complex middleware stacks easy to build and maintain.

**Why Middleware Chaining:**
- **Readability:** Stack middlewares in execution order
- **Reusability:** Create reusable middleware combinations
- **Maintainability:** Easy to add/remove/reorder middleware
- **Composition:** Build complex behavior from simple pieces

**Production Pattern:**
\`\`\`go
// Standard middleware stack for all endpoints
var StandardStack = Chain(
    Recover,	// Catch panics first
    Logger,	// Log all requests
    RequestID,	// Add request ID
    Timeout(30 * time.Second),	// Set timeout
)

// API-specific stack
var APIStack = Chain(
    StandardStack,	// Include standard stack
    RequireHeader("Authorization"),	// Require auth
    ConcurrencyLimit(100),	// Limit concurrency
    MaxBytes(10 * KB),	// Limit payload size
)

// Usage
mux.Handle("/api/users", APIStack(usersHandler))
mux.Handle("/api/orders", APIStack(ordersHandler))

// Custom stacks for specific needs
var PublicAPIStack = Chain(
    Recover,
    Logger,
    RequestID,
    ConcurrencyLimit(1000),
    MaxBytes(1 * KB),
)

var AdminAPIStack = Chain(
    Recover,
    Logger,
    RequestID,
    RequireHeader("X-Admin-Key"),
    Timeout(60 * time.Second),
    MaxBytes(50 * KB),
)

var UploadStack = Chain(
    Recover,
    Logger,
    RequestID,
    RequireHeader("Authorization"),
    Timeout(300 * time.Second),	// Long timeout for uploads
    MaxBytes(100 * MB),
    DecompressGZIP,
)

// Conditional middleware in chain
func ConditionalChain(condition bool, mw func(http.Handler) http.Handler) func(http.Handler) http.Handler {
    if condition {
        return mw
    }
    return func(h http.Handler) http.Handler { return h }	// Pass-through
}

// Usage: Only add auth in production
var stack = Chain(
    Recover,
    Logger,
    ConditionalChain(isProduction, RequireAuth),
    RequestID,
)

// Per-route stacks
type Router struct {
    public    func(http.Handler) http.Handler
    protected func(http.Handler) http.Handler
    admin     func(http.Handler) http.Handler
}

func NewRouter() *Router {
    return &Router{
        public: Chain(
            Recover,
            Logger,
            ConcurrencyLimit(1000),
        ),
        protected: Chain(
            Recover,
            Logger,
            RequireAuth,
            ConcurrencyLimit(100),
        ),
        admin: Chain(
            Recover,
            Logger,
            RequireAuth,
            RequireAdmin,
            ConcurrencyLimit(10),
        ),
    }
}

func (r *Router) SetupRoutes(mux *http.ServeMux) {
	// Public routes
    mux.Handle("/health", r.public(healthHandler))
    mux.Handle("/login", r.public(loginHandler))

	// Protected routes
    mux.Handle("/api/profile", r.protected(profileHandler))
    mux.Handle("/api/orders", r.protected(ordersHandler))

	// Admin routes
    mux.Handle("/admin/users", r.admin(adminUsersHandler))
    mux.Handle("/admin/stats", r.admin(adminStatsHandler))
}

// Middleware groups for common patterns
var (
	// Minimal: Just essentials
    MinimalStack = Chain(
        Recover,
        Logger,
    )

	// Standard: Basic production stack
    StandardStack = Chain(
        Recover,
        Logger,
        RequestID,
        Timeout(30 * time.Second),
    )

	// Secure: Authentication + rate limiting
    SecureStack = Chain(
        Recover,
        Logger,
        RequestID,
        RequireAuth,
        ConcurrencyLimit(50),
        Timeout(15 * time.Second),
    )

	// Heavy: For expensive operations
    HeavyStack = Chain(
        Recover,
        Logger,
        RequestID,
        RequireAuth,
        ConcurrencyLimit(5),
        Timeout(300 * time.Second),
    )
)

// Dynamic chain builder
type ChainBuilder struct {
    middlewares []func(http.Handler) http.Handler
}

func NewChainBuilder() *ChainBuilder {
    return &ChainBuilder{}
}

func (cb *ChainBuilder) Use(mw func(http.Handler) http.Handler) *ChainBuilder {
    cb.middlewares = append(cb.middlewares, mw)
    return cb
}

func (cb *ChainBuilder) UseIf(condition bool, mw func(http.Handler) http.Handler) *ChainBuilder {
    if condition {
        cb.middlewares = append(cb.middlewares, mw)
    }
    return cb
}

func (cb *ChainBuilder) Build() func(http.Handler) http.Handler {
    return Chain(cb.middlewares...)
}

// Usage: Build chain dynamically
func BuildStack(config Config) func(http.Handler) http.Handler {
    builder := NewChainBuilder().
        Use(Recover).
        Use(Logger).
        UseIf(config.EnableAuth, RequireAuth).
        UseIf(config.EnableRateLimit, ConcurrencyLimit(config.MaxConcurrency)).
        UseIf(config.EnableCompression, DecompressGZIP)

    return builder.Build()
}
\`\`\`

**Real-World Benefits:**
- **DRY Principle:** Define middleware stacks once, reuse everywhere
- **Consistency:** All endpoints in a group use same middleware
- **Flexibility:** Easy to create specialized stacks
- **Testing:** Test middleware stacks independently

**Chain Execution Order:**
\`\`\`go
// Middlewares execute in order they appear
Chain(A, B, C)(handler)

// Execution flow:
// Request → A → B → C → handler → C → B → A → Response

// Why reverse iteration works:
// Chain wraps: A(B(C(handler)))
// To build this, we iterate backwards:
// 1. wrapped = handler
// 2. wrapped = C(wrapped) = C(handler)
// 3. wrapped = B(wrapped) = B(C(handler))
// 4. wrapped = A(wrapped) = A(B(C(handler)))
\`\`\`

**Best Practices:**
- **Order Matters:** Recover first, Auth before business logic
- **Create Stacks:** Define reusable middleware combinations
- **Name Clearly:** MinimalStack, SecureStack, HeavyStack
- **Document Order:** Comment why middlewares are ordered this way

**Common Stack Patterns:**
1. **Recover → Logger → RequestID** (essential trio)
2. **Recover → Logger → Auth → Business** (secured endpoint)
3. **Recover → Logger → RateLimit → Business** (public endpoint)
4. **Recover → Logger → Auth → Heavy → Business** (expensive operation)

Without Chain, middleware composition requires nested function calls like \`A(B(C(handler)))\` which is hard to read and maintain.`,	order: 17,
	translations: {
		ru: {
			title: 'Последовательное применение middleware',
			description: `Реализуйте функцию **Chain**, которая объединяет несколько middleware функций в одну, применяя их слева направо.

**Требования:**
1. Создайте функцию \`Chain(middlewares ...func(http.Handler) http.Handler) func(http.Handler) http.Handler\`
2. Верните функцию которая принимает финальный handler
3. Применяйте middlewares справа налево (для выполнения слева направо)
4. Пропускайте nil middlewares
5. Обработайте nil финальный handler
6. Верните скомпонованный handler

**Пример:**
\`\`\`go
handler := Chain(
    Logger,	// Выполняется первым
    RequestID,	// Выполняется вторым
    Recover,	// Выполняется третьим
)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Final handler")
}))

// Поток запроса: Logger → RequestID → Recover → Final handler
\`\`\`

**Ограничения:**
- Должен применять middlewares справа налево для достижения выполнения слева направо
- Должен пропускать nil middleware функции
- Должен обрабатывать nil финальный handler`,
			hint1: `Итерируйте через middlewares в обратном порядке (len-1 до 0) чтобы обернуть их справа налево для выполнения слева направо.`,
			hint2: `Начните с финального handler, оберните его каждым middleware в обратном порядке, пропускайте nil middlewares.`,
			whyItMatters: `Chain обеспечивает чистую, читаемую композицию middleware, делая сложные middleware стеки легко строить и поддерживать.

**Почему Middleware Chaining:**
- **Читаемость:** Стек middlewares в порядке выполнения
- **Переиспользуемость:** Создание переиспользуемых комбинаций middleware
- **Поддерживаемость:** Легко добавлять/удалять/переупорядочивать middleware
- **Композиция:** Построение сложного поведения из простых компонентов

**Продакшен паттерн:**
\`\`\`go
// Стандартный middleware стек для всех endpoints
var StandardStack = Chain(
    Recover,	// Ловить паники первым
    Logger,	// Логировать все запросы
    RequestID,	// Добавить request ID
    Timeout(30 * time.Second),	// Установить таймаут
)

// API-специфичный стек
var APIStack = Chain(
    StandardStack,	// Включить стандартный стек
    RequireHeader("Authorization"),	// Требовать auth
    ConcurrencyLimit(100),	// Ограничить concurrency
    MaxBytes(10 * KB),	// Ограничить размер payload
)

// Использование
mux.Handle("/api/users", APIStack(usersHandler))
mux.Handle("/api/orders", APIStack(ordersHandler))

// Кастомные стеки для специфичных нужд
var PublicAPIStack = Chain(
    Recover,
    Logger,
    RequestID,
    ConcurrencyLimit(1000),
    MaxBytes(1 * KB),
)

var AdminAPIStack = Chain(
    Recover,
    Logger,
    RequestID,
    RequireHeader("X-Admin-Key"),
    Timeout(60 * time.Second),
    MaxBytes(50 * KB),
)

var UploadStack = Chain(
    Recover,
    Logger,
    RequestID,
    RequireHeader("Authorization"),
    Timeout(300 * time.Second),	// Длинный таймаут для загрузок
    MaxBytes(100 * MB),
    DecompressGZIP,
)

// Условный middleware в chain
func ConditionalChain(condition bool, mw func(http.Handler) http.Handler) func(http.Handler) http.Handler {
    if condition {
        return mw
    }
    return func(h http.Handler) http.Handler { return h }	// Pass-through
}

// Использование: Добавить auth только в продакшене
var stack = Chain(
    Recover,
    Logger,
    ConditionalChain(isProduction, RequireAuth),
    RequestID,
)

// Per-route стеки
type Router struct {
    public    func(http.Handler) http.Handler
    protected func(http.Handler) http.Handler
    admin     func(http.Handler) http.Handler
}

func NewRouter() *Router {
    return &Router{
        public: Chain(
            Recover,
            Logger,
            ConcurrencyLimit(1000),
        ),
        protected: Chain(
            Recover,
            Logger,
            RequireAuth,
            ConcurrencyLimit(100),
        ),
        admin: Chain(
            Recover,
            Logger,
            RequireAuth,
            RequireAdmin,
            ConcurrencyLimit(10),
        ),
    }
}

func (r *Router) SetupRoutes(mux *http.ServeMux) {
	// Публичные маршруты
    mux.Handle("/health", r.public(healthHandler))
    mux.Handle("/login", r.public(loginHandler))

	// Защищённые маршруты
    mux.Handle("/api/profile", r.protected(profileHandler))
    mux.Handle("/api/orders", r.protected(ordersHandler))

	// Админские маршруты
    mux.Handle("/admin/users", r.admin(adminUsersHandler))
    mux.Handle("/admin/stats", r.admin(adminStatsHandler))
}

// Middleware группы для общих паттернов
var (
	// Minimal: Только основное
    MinimalStack = Chain(
        Recover,
        Logger,
    )

	// Standard: Базовый продакшен стек
    StandardStack = Chain(
        Recover,
        Logger,
        RequestID,
        Timeout(30 * time.Second),
    )

	// Secure: Аутентификация + rate limiting
    SecureStack = Chain(
        Recover,
        Logger,
        RequestID,
        RequireAuth,
        ConcurrencyLimit(50),
        Timeout(15 * time.Second),
    )

	// Heavy: Для дорогих операций
    HeavyStack = Chain(
        Recover,
        Logger,
        RequestID,
        RequireAuth,
        ConcurrencyLimit(5),
        Timeout(300 * time.Second),
    )
)

// Динамический chain builder
type ChainBuilder struct {
    middlewares []func(http.Handler) http.Handler
}

func NewChainBuilder() *ChainBuilder {
    return &ChainBuilder{}
}

func (cb *ChainBuilder) Use(mw func(http.Handler) http.Handler) *ChainBuilder {
    cb.middlewares = append(cb.middlewares, mw)
    return cb
}

func (cb *ChainBuilder) UseIf(condition bool, mw func(http.Handler) http.Handler) *ChainBuilder {
    if condition {
        cb.middlewares = append(cb.middlewares, mw)
    }
    return cb
}

func (cb *ChainBuilder) Build() func(http.Handler) http.Handler {
    return Chain(cb.middlewares...)
}

// Использование: Динамическое построение chain
func BuildStack(config Config) func(http.Handler) http.Handler {
    builder := NewChainBuilder().
        Use(Recover).
        Use(Logger).
        UseIf(config.EnableAuth, RequireAuth).
        UseIf(config.EnableRateLimit, ConcurrencyLimit(config.MaxConcurrency)).
        UseIf(config.EnableCompression, DecompressGZIP)

    return builder.Build()
}
\`\`\`

**Практические преимущества:**
- **DRY принцип:** Определить middleware стеки один раз, использовать везде
- **Консистентность:** Все endpoints в группе используют одни middleware
- **Гибкость:** Легко создавать специализированные стеки
- **Тестирование:** Тестировать middleware стеки независимо

**Порядок выполнения Chain:**
\`\`\`go
// Middlewares выполняются в порядке их появления
Chain(A, B, C)(handler)

// Поток выполнения:
// Request → A → B → C → handler → C → B → A → Response

// Почему обратная итерация работает:
// Chain оборачивает: A(B(C(handler)))
// Чтобы построить это, мы итерируем в обратном порядке:
// 1. wrapped = handler
// 2. wrapped = C(wrapped) = C(handler)
// 3. wrapped = B(wrapped) = B(C(handler))
// 4. wrapped = A(wrapped) = A(B(C(handler)))
\`\`\`

**Лучшие практики:**
- **Порядок важен:** Recover первым, Auth перед бизнес-логикой
- **Создавайте стеки:** Определяйте переиспользуемые комбинации middleware
- **Называйте ясно:** MinimalStack, SecureStack, HeavyStack
- **Документируйте порядок:** Комментируйте почему middlewares упорядочены так

**Общие паттерны стеков:**
1. **Recover → Logger → RequestID** (основная тройка)
2. **Recover → Logger → Auth → Business** (защищённый endpoint)
3. **Recover → Logger → RateLimit → Business** (публичный endpoint)
4. **Recover → Logger → Auth → Heavy → Business** (дорогая операция)

Без Chain, композиция middleware требует вложенных вызовов функций вроде \`A(B(C(handler)))\` которые сложно читать и поддерживать.`,
			solutionCode: `package httpx

import (
	"net/http"
)

func Chain(middlewares ...func(http.Handler) http.Handler) func(http.Handler) http.Handler {
	return func(final http.Handler) http.Handler {	// Возврат функции которая принимает финальный handler
		if final == nil {	// Проверка является ли финальный handler nil
			final = http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})	// Использование no-op handler
		}
		wrapped := final	// Начало с финального handler
		for i := len(middlewares) - 1; i >= 0; i-- {	// Итерация в обратном порядке (справа налево)
			mw := middlewares[i]	// Получение текущего middleware
			if mw == nil {	// Пропуск nil middlewares
				continue
			}
			wrapped = mw(wrapped)	// Обёртывание handler middleware
		}
		return wrapped	// Возврат полностью обёрнутого handler
	}
}`
		},
		uz: {
			title: 'Middlewarelarni ketma-ket qo\'llash',
			description: `Bir nechta middleware funksiyalarini bittaga birlashtirib, ularni chapdan o'ngga qo'llaydigan **Chain** funksiyasini amalga oshiring.

**Talablar:**
1. \`Chain(middlewares ...func(http.Handler) http.Handler) func(http.Handler) http.Handler\` funksiyasini yarating
2. Yakuniy handlerni qabul qiluvchi funksiyani qaytaring
3. Middlewarelarni o'ngdan chapga qo'llang (chapdan o'ngga bajarilishi uchun)
4. nil middlewarelarni o'tkazing
5. nil yakuniy handlerni ishlang
6. Kompozitsiya qilingan handlerni qaytaring

**Misol:**
\`\`\`go
handler := Chain(
    Logger,	// Birinchi bajariladi
    RequestID,	// Ikkinchi bajariladi
    Recover,	// Uchinchi bajariladi
)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Final handler")
}))

// Request oqimi: Logger → RequestID → Recover → Final handler
\`\`\`

**Cheklovlar:**
- Chapdan o'ngga bajarilishga erishish uchun middlewarelarni o'ngdan chapga qo'llashi kerak
- nil middleware funksiyalarini o'tkazishi kerak
- nil yakuniy handlerni ishlashi kerak`,
			hint1: `Chapdan o'ngga bajarilish uchun ularni o'ngdan chapga o'rash uchun middlewarelar orqali teskari (len-1 dan 0 gacha) iteratsiya qiling.`,
			hint2: `Yakuniy handlerdan boshlang, har bir middleware bilan teskari tartibda o'rang, nil middlewarelarni o'tkazing.`,
			whyItMatters: `Chain toza, o'qilishi mumkin middleware kompozitsiyasini ta'minlaydi, murakkab middleware stacklarini osongina qurish va saqlashni amalga oshiradi.

**Nima uchun Middleware Chaining:**
- **O'qilish:** Middleware stacki bajarilish tartibida
- **Qayta foydalanish:** Qayta ishlatiladigan middleware kombinatsiyalarini yaratish
- **Saqlanish:** Middleware ni qo'shish/o'chirish/qayta tartibga solish oson
- **Kompozitsiya:** Oddiy komponentlardan murakkab xatti-harakatni qurish

**Ishlab chiqarish patterni:**
\`\`\`go
// Barcha endpointlar uchun standart middleware stack
var StandardStack = Chain(
    Recover,	// Birinchi bo'lib paniklarni ushlash
    Logger,	// Barcha requestlarni log qilish
    RequestID,	// Request ID qo'shish
    Timeout(30 * time.Second),	// Timeout o'rnatish
)

// API-specific stack
var APIStack = Chain(
    StandardStack,	// Standart stackni qo'shish
    RequireHeader("Authorization"),	// Auth talab qilish
    ConcurrencyLimit(100),	// Concurrency ni cheklash
    MaxBytes(10 * KB),	// Payload hajmini cheklash
)

// Foydalanish
mux.Handle("/api/users", APIStack(usersHandler))
mux.Handle("/api/orders", APIStack(ordersHandler))

// Maxsus ehtiyojlar uchun stacklar
var PublicAPIStack = Chain(
    Recover,
    Logger,
    RequestID,
    ConcurrencyLimit(1000),
    MaxBytes(1 * KB),
)

var AdminAPIStack = Chain(
    Recover,
    Logger,
    RequestID,
    RequireHeader("X-Admin-Key"),
    Timeout(60 * time.Second),
    MaxBytes(50 * KB),
)

var UploadStack = Chain(
    Recover,
    Logger,
    RequestID,
    RequireHeader("Authorization"),
    Timeout(300 * time.Second),	// Yuklashlar uchun uzun timeout
    MaxBytes(100 * MB),
    DecompressGZIP,
)

// Chainda shartli middleware
func ConditionalChain(condition bool, mw func(http.Handler) http.Handler) func(http.Handler) http.Handler {
    if condition {
        return mw
    }
    return func(h http.Handler) http.Handler { return h }	// Pass-through
}

// Foydalanish: Faqat production muhitida auth qo'shish
var stack = Chain(
    Recover,
    Logger,
    ConditionalChain(isProduction, RequireAuth),
    RequestID,
)

// Har bir route uchun stacklar
type Router struct {
    public    func(http.Handler) http.Handler
    protected func(http.Handler) http.Handler
    admin     func(http.Handler) http.Handler
}

func NewRouter() *Router {
    return &Router{
        public: Chain(
            Recover,
            Logger,
            ConcurrencyLimit(1000),
        ),
        protected: Chain(
            Recover,
            Logger,
            RequireAuth,
            ConcurrencyLimit(100),
        ),
        admin: Chain(
            Recover,
            Logger,
            RequireAuth,
            RequireAdmin,
            ConcurrencyLimit(10),
        ),
    }
}

func (r *Router) SetupRoutes(mux *http.ServeMux) {
	// Ochiq routelar
    mux.Handle("/health", r.public(healthHandler))
    mux.Handle("/login", r.public(loginHandler))

	// Himoyalangan routelar
    mux.Handle("/api/profile", r.protected(profileHandler))
    mux.Handle("/api/orders", r.protected(ordersHandler))

	// Admin routelar
    mux.Handle("/admin/users", r.admin(adminUsersHandler))
    mux.Handle("/admin/stats", r.admin(adminStatsHandler))
}

// Umumiy patternlar uchun middleware guruhlari
var (
	// Minimal: Faqat asosiylar
    MinimalStack = Chain(
        Recover,
        Logger,
    )

	// Standard: Asosiy production stack
    StandardStack = Chain(
        Recover,
        Logger,
        RequestID,
        Timeout(30 * time.Second),
    )

	// Secure: Autentifikatsiya + rate limiting
    SecureStack = Chain(
        Recover,
        Logger,
        RequestID,
        RequireAuth,
        ConcurrencyLimit(50),
        Timeout(15 * time.Second),
    )

	// Heavy: Qimmat operatsiyalar uchun
    HeavyStack = Chain(
        Recover,
        Logger,
        RequestID,
        RequireAuth,
        ConcurrencyLimit(5),
        Timeout(300 * time.Second),
    )
)

// Dinamik chain builder
type ChainBuilder struct {
    middlewares []func(http.Handler) http.Handler
}

func NewChainBuilder() *ChainBuilder {
    return &ChainBuilder{}
}

func (cb *ChainBuilder) Use(mw func(http.Handler) http.Handler) *ChainBuilder {
    cb.middlewares = append(cb.middlewares, mw)
    return cb
}

func (cb *ChainBuilder) UseIf(condition bool, mw func(http.Handler) http.Handler) *ChainBuilder {
    if condition {
        cb.middlewares = append(cb.middlewares, mw)
    }
    return cb
}

func (cb *ChainBuilder) Build() func(http.Handler) http.Handler {
    return Chain(cb.middlewares...)
}

// Foydalanish: Dinamik chain qurish
func BuildStack(config Config) func(http.Handler) http.Handler {
    builder := NewChainBuilder().
        Use(Recover).
        Use(Logger).
        UseIf(config.EnableAuth, RequireAuth).
        UseIf(config.EnableRateLimit, ConcurrencyLimit(config.MaxConcurrency)).
        UseIf(config.EnableCompression, DecompressGZIP)

    return builder.Build()
}
\`\`\`

**Amaliy foydalari:**
- **DRY prinsipi:** Middleware stacklarni bir marta aniqlash, hamma joyda ishlatish
- **Izchillik:** Gruppadagi barcha endpointlar bir xil middlewaredan foydalanadi
- **Moslashuvchanlik:** Maxsus stacklarni osongina yaratish
- **Test qilish:** Middleware stacklarni mustaqil test qilish

**Chain bajarilish tartibi:**
\`\`\`go
// Middlewarelar ko'rinish tartibida bajariladi
Chain(A, B, C)(handler)

// Bajarilish oqimi:
// Request → A → B → C → handler → C → B → A → Response

// Nima uchun teskari iteratsiya ishlaydi:
// Chain o'raydi: A(B(C(handler)))
// Buni qurish uchun teskari tartibda iteratsiya qilamiz:
// 1. wrapped = handler
// 2. wrapped = C(wrapped) = C(handler)
// 3. wrapped = B(wrapped) = B(C(handler))
// 4. wrapped = A(wrapped) = A(B(C(handler)))
\`\`\`

**Eng yaxshi amaliyotlar:**
- **Tartib muhim:** Recover birinchi, Auth biznes-logikadan oldin
- **Stacklar yarating:** Qayta ishlatiladigan middleware kombinatsiyalarini aniqlang
- **Aniq nomlang:** MinimalStack, SecureStack, HeavyStack
- **Tartibni hujjatlantiring:** Nima uchun middlewarelar shunday tartibda ekanligini izohlab bering

**Umumiy stack patternlari:**
1. **Recover → Logger → RequestID** (asosiy uchlik)
2. **Recover → Logger → Auth → Business** (himoyalangan endpoint)
3. **Recover → Logger → RateLimit → Business** (ochiq endpoint)
4. **Recover → Logger → Auth → Heavy → Business** (qimmat operatsiya)

Chain siz middleware kompozitsiyasi \`A(B(C(handler)))\` kabi ichma-ich funksiya chaqiruvlarini talab qiladi, bu esa o'qish va saqlash uchun qiyin.`,
			solutionCode: `package httpx

import (
	"net/http"
)

func Chain(middlewares ...func(http.Handler) http.Handler) func(http.Handler) http.Handler {
	return func(final http.Handler) http.Handler {	// Yakuniy handlerni qabul qiluvchi funksiyani qaytarish
		if final == nil {	// Yakuniy handler nil ekanligini tekshirish
			final = http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})	// No-op handlerdan foydalanish
		}
		wrapped := final	// Yakuniy handlerdan boshlash
		for i := len(middlewares) - 1; i >= 0; i-- {	// Teskari tartibda iteratsiya qilish (o'ngdan chapga)
			mw := middlewares[i]	// Joriy middleware ni olish
			if mw == nil {	// nil middlewarelarni o'tkazish
				continue
			}
			wrapped = mw(wrapped)	// Handlerni middleware bilan o'rash
		}
		return wrapped	// To'liq o'ralgan handlerni qaytarish
	}
}`
		}
	}
};

export default task;
