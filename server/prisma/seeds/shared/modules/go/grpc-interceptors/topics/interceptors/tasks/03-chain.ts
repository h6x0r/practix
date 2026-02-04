import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-grpc-chain',
	title: 'Chain Interceptor Composer',
	difficulty: 'medium',	tags: ['go', 'grpc', 'interceptors', 'composition'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Chain** function that combines multiple interceptors into a single interceptor, applying them in order.

**Requirements:**
1. Create function \`Chain(interceptors ...UnaryServerInterceptor) UnaryServerInterceptor\`
2. Handle nil final handler (use no-op)
3. Apply interceptors from right to left to achieve left-to-right execution
4. Skip nil interceptors
5. Return composed interceptor that executes all interceptors in sequence

**Example:**
\`\`\`go
logging := LoggingInterceptor(logger)
timeout := TimeoutInterceptor(5 * time.Second)

combined := Chain(logging, timeout) // Logging executes first, then timeout

handler := func(ctx context.Context, req interface{}) (interface{}, error) {
    return "response", nil
}

resp, err := combined(ctx, "request", handler)
// Execution: logging → timeout → handler
\`\`\`

**Constraints:**
- Must apply interceptors in order they appear (left to right)
- Must skip nil interceptors
- Must handle nil handler gracefully`,
	initialCode: `package grpcx

import (
	"context"
)

type Handler func(ctx context.Context, req interface{}) (interface{}, error)

type UnaryServerInterceptor func(ctx context.Context, req interface{}, next Handler) (interface{}, error)

// TODO: Implement Chain interceptor composer
func Chain(interceptors ...UnaryServerInterceptor) UnaryServerInterceptor {
	// TODO: Implement
}`,
	solutionCode: `package grpcx

import (
	"context"
)

type Handler func(ctx context.Context, req interface{}) (interface{}, error)

type UnaryServerInterceptor func(ctx context.Context, req interface{}, next Handler) (interface{}, error)

func Chain(interceptors ...UnaryServerInterceptor) UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, handler Handler) (interface{}, error) {
		if handler == nil {	// Check if handler is nil
			handler = func(context.Context, interface{}) (interface{}, error) { return nil, nil }	// Use no-op handler
		}
		wrapped := handler	// Start with final handler
		for i := len(interceptors) - 1; i >= 0; i-- {	// Iterate backwards (right-to-left)
			current := interceptors[i]	// Get current interceptor
			if current == nil {	// Skip nil interceptors
				continue
			}
			next := wrapped	// Capture current wrapped handler
			wrapped = func(c context.Context, r interface{}) (interface{}, error) {	// Create new handler
				return current(c, r, next)	// Call interceptor with next handler
			}
		}
		return wrapped(ctx, req)	// Execute composed chain
	}
}`,
			hint1: `Iterate backwards through interceptors (len-1 to 0) to wrap them right-to-left for left-to-right execution.`,
			hint2: `Capture the "next" handler in a closure to avoid variable capture issues in the loop.`,
			testCode: `package grpcx

import (
	"context"
	"errors"
	"testing"
)

func Test1(t *testing.T) {
	// Test empty chain returns handler result
	chain := Chain()
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return "response", nil
	}
	resp, err := chain(context.Background(), nil, handler)
	if err != nil || resp != "response" {
		t.Errorf("got (%v, %v), want ('response', nil)", resp, err)
	}
}

func Test2(t *testing.T) {
	// Test single interceptor
	order := []string{}
	interceptor := func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
		order = append(order, "interceptor")
		return next(ctx, req)
	}
	chain := Chain(interceptor)
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		order = append(order, "handler")
		return nil, nil
	}
	chain(context.Background(), nil, handler)
	if len(order) != 2 || order[0] != "interceptor" || order[1] != "handler" {
		t.Errorf("order = %v, want [interceptor, handler]", order)
	}
}

func Test3(t *testing.T) {
	// Test multiple interceptors execute in order
	order := []string{}
	first := func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
		order = append(order, "first")
		return next(ctx, req)
	}
	second := func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
		order = append(order, "second")
		return next(ctx, req)
	}
	chain := Chain(first, second)
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		order = append(order, "handler")
		return nil, nil
	}
	chain(context.Background(), nil, handler)
	if len(order) != 3 || order[0] != "first" || order[1] != "second" {
		t.Errorf("order = %v, want [first, second, handler]", order)
	}
}

func Test4(t *testing.T) {
	// Test nil interceptors are skipped
	order := []string{}
	interceptor := func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
		order = append(order, "interceptor")
		return next(ctx, req)
	}
	chain := Chain(nil, interceptor, nil)
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		order = append(order, "handler")
		return nil, nil
	}
	chain(context.Background(), nil, handler)
	if len(order) != 2 {
		t.Errorf("order = %v, expected 2 entries", order)
	}
}

func Test5(t *testing.T) {
	// Test nil handler is replaced with no-op
	interceptor := func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
		return next(ctx, req)
	}
	chain := Chain(interceptor)
	resp, err := chain(context.Background(), nil, nil)
	if err != nil || resp != nil {
		t.Errorf("got (%v, %v), want (nil, nil)", resp, err)
	}
}

func Test6(t *testing.T) {
	// Test error propagation through chain
	expectedErr := errors.New("handler error")
	chain := Chain()
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return nil, expectedErr
	}
	_, err := chain(context.Background(), nil, handler)
	if err != expectedErr {
		t.Errorf("error = %v, want %v", err, expectedErr)
	}
}

func Test7(t *testing.T) {
	// Test interceptor can modify request
	modifier := func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
		return next(ctx, "modified-"+req.(string))
	}
	chain := Chain(modifier)
	var receivedReq interface{}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		receivedReq = req
		return nil, nil
	}
	chain(context.Background(), "original", handler)
	if receivedReq != "modified-original" {
		t.Errorf("request = %v, want 'modified-original'", receivedReq)
	}
}

func Test8(t *testing.T) {
	// Test interceptor can modify response
	modifier := func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
		resp, err := next(ctx, req)
		return "modified-" + resp.(string), err
	}
	chain := Chain(modifier)
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return "response", nil
	}
	resp, _ := chain(context.Background(), nil, handler)
	if resp != "modified-response" {
		t.Errorf("response = %v, want 'modified-response'", resp)
	}
}

func Test9(t *testing.T) {
	// Test context is passed through chain
	ctx := context.WithValue(context.Background(), "key", "value")
	interceptor := func(c context.Context, req interface{}, next Handler) (interface{}, error) {
		if c.Value("key") != "value" {
			t.Error("context value not preserved in interceptor")
		}
		return next(c, req)
	}
	chain := Chain(interceptor)
	handler := func(c context.Context, req interface{}) (interface{}, error) {
		if c.Value("key") != "value" {
			t.Error("context value not preserved in handler")
		}
		return nil, nil
	}
	chain(ctx, nil, handler)
}

func Test10(t *testing.T) {
	// Test interceptor can short-circuit chain
	short := func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
		return "short-circuit", nil // Don't call next
	}
	handlerCalled := false
	chain := Chain(short)
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		handlerCalled = true
		return nil, nil
	}
	resp, _ := chain(context.Background(), nil, handler)
	if handlerCalled {
		t.Error("handler should not be called when short-circuited")
	}
	if resp != "short-circuit" {
		t.Errorf("response = %v, want 'short-circuit'", resp)
	}
}`,
			whyItMatters: `Chain enables clean, readable interceptor composition, making complex gRPC middleware stacks easy to build and maintain.

**Why Interceptor Chaining:**
- **Readability:** Stack interceptors in execution order
- **Reusability:** Create reusable interceptor combinations
- **Maintainability:** Easy to add/remove/reorder interceptors
- **Composition:** Build complex behavior from simple pieces

**Production Pattern:**
\`\`\`go
// Standard interceptor stack for all services
var StandardInterceptors = Chain(
    RecoveryInterceptor(),     // Catch panics first
    LoggingInterceptor(),      // Log all requests
    TimeoutInterceptor(30 * time.Second),  // Set timeout
)

// Authentication stack
var AuthInterceptors = Chain(
    StandardInterceptors,      // Include standard stack
    AuthInterceptor(),         // Require authentication
    RateLimitInterceptor(100), // Limit request rate
)

// Usage with gRPC server
server := grpc.NewServer(
    grpc.UnaryInterceptor(AuthInterceptors),
)

// Custom stacks for specific needs
var PublicServiceInterceptors = Chain(
    RecoveryInterceptor(),
    LoggingInterceptor(),
    TimeoutInterceptor(10 * time.Second),
    RateLimitInterceptor(1000),
)

var AdminServiceInterceptors = Chain(
    RecoveryInterceptor(),
    LoggingInterceptor(),
    AuthInterceptor(),
    RequireAdminInterceptor(),
    TimeoutInterceptor(60 * time.Second),
)

var StreamingServiceInterceptors = Chain(
    RecoveryInterceptor(),
    LoggingInterceptor(),
    TimeoutInterceptor(300 * time.Second),  // Long timeout for streams
)

// Per-method interceptor selection
func MethodInterceptorSelector() grpc.UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
        var interceptor UnaryServerInterceptor

        switch {
        case strings.HasPrefix(info.FullMethod, "/admin."):
            interceptor = AdminServiceInterceptors
        case strings.HasPrefix(info.FullMethod, "/public."):
            interceptor = PublicServiceInterceptors
        default:
            interceptor = StandardInterceptors
        }

        return interceptor(ctx, req, handler)
    }
}

// Conditional interceptor
func ConditionalChain(condition bool, interceptor UnaryServerInterceptor) UnaryServerInterceptor {
    if condition {
        return interceptor
    }
    return func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
        return next(ctx, req) // Pass-through
    }
}

// Usage: Only add auth in production
var stack = Chain(
    RecoveryInterceptor(),
    LoggingInterceptor(),
    ConditionalChain(isProduction, AuthInterceptor()),
)

// Dynamic chain builder
type InterceptorChainBuilder struct {
    interceptors []UnaryServerInterceptor
}

func NewChainBuilder() *InterceptorChainBuilder {
    return &InterceptorChainBuilder{}
}

func (b *InterceptorChainBuilder) Use(interceptor UnaryServerInterceptor) *InterceptorChainBuilder {
    b.interceptors = append(b.interceptors, interceptor)
    return b
}

func (b *InterceptorChainBuilder) UseIf(condition bool, interceptor UnaryServerInterceptor) *InterceptorChainBuilder {
    if condition {
        b.interceptors = append(b.interceptors, interceptor)
    }
    return b
}

func (b *InterceptorChainBuilder) Build() UnaryServerInterceptor {
    return Chain(b.interceptors...)
}

// Usage: Build chain dynamically
func BuildServiceInterceptors(config Config) UnaryServerInterceptor {
    return NewChainBuilder().
        Use(RecoveryInterceptor()).
        Use(LoggingInterceptor()).
        UseIf(config.EnableAuth, AuthInterceptor()).
        UseIf(config.EnableRateLimit, RateLimitInterceptor(config.RateLimit)).
        UseIf(config.EnableMetrics, MetricsInterceptor()).
        Build()
}

// Multiple service interceptor chains
type ServiceInterceptors struct {
    User   UnaryServerInterceptor
    Admin  UnaryServerInterceptor
    Public UnaryServerInterceptor
}

func NewServiceInterceptors() *ServiceInterceptors {
    baseChain := Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
    )

    return &ServiceInterceptors{
        User: Chain(
            baseChain,
            AuthInterceptor(),
            TimeoutInterceptor(10 * time.Second),
        ),
        Admin: Chain(
            baseChain,
            AuthInterceptor(),
            RequireAdminInterceptor(),
            AuditLogInterceptor(),
            TimeoutInterceptor(30 * time.Second),
        ),
        Public: Chain(
            baseChain,
            RateLimitInterceptor(1000),
            TimeoutInterceptor(5 * time.Second),
        ),
    }
}

// Interceptor groups for common patterns
var (
    // Minimal: Just essentials
    MinimalChain = Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
    )

    // Standard: Basic production stack
    StandardChain = Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
        TimeoutInterceptor(30 * time.Second),
        MetricsInterceptor(),
    )

    // Secure: Authentication + authorization
    SecureChain = Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
        AuthInterceptor(),
        AuthzInterceptor(),
        TimeoutInterceptor(15 * time.Second),
    )

    // Heavy: For expensive operations
    HeavyChain = Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
        AuthInterceptor(),
        RateLimitInterceptor(5),
        TimeoutInterceptor(300 * time.Second),
    )
)
\`\`\`

**Real-World Benefits:**
- **DRY Principle:** Define interceptor chains once, reuse everywhere
- **Consistency:** All methods in a service use same interceptors
- **Flexibility:** Easy to create specialized chains
- **Testing:** Test interceptor chains independently

**Chain Execution Order:**
\`\`\`go
// Interceptors execute in order they appear
Chain(A, B, C)

// Execution flow:
// Request → A → B → C → handler → C → B → A → Response

// Why backward iteration works:
// Chain wraps: wrapped = C(wrapped)  // wrapped = handler
//              wrapped = B(wrapped)  // wrapped = C(handler)
//              wrapped = A(wrapped)  // wrapped = A(B(C(handler)))
\`\`\`

**Best Practices:**
- **Order Matters:** Recovery first, auth before business logic
- **Create Stacks:** Define reusable interceptor combinations
- **Name Clearly:** MinimalChain, SecureChain, HeavyChain
- **Document Order:** Comment why interceptors are ordered this way

**Common Chain Patterns:**
1. **Recovery → Logging → Timeout** (essential trio)
2. **Recovery → Logging → Auth → Business** (secured RPC)
3. **Recovery → Logging → RateLimit → Business** (public RPC)
4. **Recovery → Logging → Auth → Heavy → Business** (expensive operation)

Without Chain, interceptor composition requires manual nesting which is hard to read and maintain.`,	order: 2,
	translations: {
		ru: {
			title: 'Объединение нескольких интерсепторов',
			description: `Реализуйте функцию **Chain**, которая объединяет несколько interceptors в один, применяя их по порядку.

**Требования:**
1. Создайте функцию \`Chain(interceptors ...UnaryServerInterceptor) UnaryServerInterceptor\`
2. Обработайте nil финальный handler (используйте no-op)
3. Применяйте interceptors справа налево для достижения выполнения слева направо
4. Пропускайте nil interceptors
5. Верните скомпонованный interceptor который выполняет все interceptors последовательно

**Пример:**
\`\`\`go
logging := LoggingInterceptor(logger)
timeout := TimeoutInterceptor(5 * time.Second)

combined := Chain(logging, timeout) // Logging выполняется первым, затем timeout

handler := func(ctx context.Context, req interface{}) (interface{}, error) {
    return "response", nil
}

resp, err := combined(ctx, "request", handler)
// Выполнение: logging → timeout → handler
\`\`\`

**Ограничения:**
- Должен применять interceptors в порядке их появления (слева направо)
- Должен пропускать nil interceptors
- Должен корректно обрабатывать nil handler`,
			hint1: `Итерируйте в обратном порядке через interceptors (len-1 до 0) чтобы обернуть их справа налево для выполнения слева направо.`,
			hint2: `Захватите "next" handler в замыкании чтобы избежать проблем с захватом переменной в цикле.`,
			whyItMatters: `Chain обеспечивает чистую, читаемую композицию interceptors, делая сложные gRPC middleware стеки легко строить и поддерживать.

**Зачем нужен Interceptor Chaining:**
- **Читаемость:** Стек interceptors в порядке выполнения
- **Переиспользуемость:** Создание переиспользуемых комбинаций interceptors
- **Поддерживаемость:** Легко добавлять/удалять/переупорядочивать interceptors
- **Композиция:** Построение сложного поведения из простых частей

**Production Паттерн:**
\`\`\`go
// Стандартный стек interceptors для всех сервисов
var StandardInterceptors = Chain(
    RecoveryInterceptor(),     // Ловим панику первым
    LoggingInterceptor(),      // Логируем все запросы
    TimeoutInterceptor(30 * time.Second),  // Устанавливаем таймаут
)

// Authentication стек
var AuthInterceptors = Chain(
    StandardInterceptors,      // Включаем стандартный стек
    AuthInterceptor(),         // Требуем аутентификацию
    RateLimitInterceptor(100), // Ограничиваем частоту запросов
)

// Использование с gRPC сервером
server := grpc.NewServer(
    grpc.UnaryInterceptor(AuthInterceptors),
)

// Кастомные стеки для специфических нужд
var PublicServiceInterceptors = Chain(
    RecoveryInterceptor(),
    LoggingInterceptor(),
    TimeoutInterceptor(10 * time.Second),
    RateLimitInterceptor(1000),
)

var AdminServiceInterceptors = Chain(
    RecoveryInterceptor(),
    LoggingInterceptor(),
    AuthInterceptor(),
    RequireAdminInterceptor(),
    TimeoutInterceptor(60 * time.Second),
)

var StreamingServiceInterceptors = Chain(
    RecoveryInterceptor(),
    LoggingInterceptor(),
    TimeoutInterceptor(300 * time.Second),  // Долгий таймаут для потоков
)

// Выбор interceptor по методу
func MethodInterceptorSelector() grpc.UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
        var interceptor UnaryServerInterceptor

        switch {
        case strings.HasPrefix(info.FullMethod, "/admin."):
            interceptor = AdminServiceInterceptors
        case strings.HasPrefix(info.FullMethod, "/public."):
            interceptor = PublicServiceInterceptors
        default:
            interceptor = StandardInterceptors
        }

        return interceptor(ctx, req, handler)
    }
}

// Условный interceptor
func ConditionalChain(condition bool, interceptor UnaryServerInterceptor) UnaryServerInterceptor {
    if condition {
        return interceptor
    }
    return func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
        return next(ctx, req) // Сквозной проход
    }
}

// Использование: Добавляем auth только в production
var stack = Chain(
    RecoveryInterceptor(),
    LoggingInterceptor(),
    ConditionalChain(isProduction, AuthInterceptor()),
)

// Динамический построитель цепочек
type InterceptorChainBuilder struct {
    interceptors []UnaryServerInterceptor
}

func NewChainBuilder() *InterceptorChainBuilder {
    return &InterceptorChainBuilder{}
}

func (b *InterceptorChainBuilder) Use(interceptor UnaryServerInterceptor) *InterceptorChainBuilder {
    b.interceptors = append(b.interceptors, interceptor)
    return b
}

func (b *InterceptorChainBuilder) UseIf(condition bool, interceptor UnaryServerInterceptor) *InterceptorChainBuilder {
    if condition {
        b.interceptors = append(b.interceptors, interceptor)
    }
    return b
}

func (b *InterceptorChainBuilder) Build() UnaryServerInterceptor {
    return Chain(b.interceptors...)
}

// Использование: Динамическое построение цепочки
func BuildServiceInterceptors(config Config) UnaryServerInterceptor {
    return NewChainBuilder().
        Use(RecoveryInterceptor()).
        Use(LoggingInterceptor()).
        UseIf(config.EnableAuth, AuthInterceptor()).
        UseIf(config.EnableRateLimit, RateLimitInterceptor(config.RateLimit)).
        UseIf(config.EnableMetrics, MetricsInterceptor()).
        Build()
}

// Несколько цепочек interceptor для сервисов
type ServiceInterceptors struct {
    User   UnaryServerInterceptor
    Admin  UnaryServerInterceptor
    Public UnaryServerInterceptor
}

func NewServiceInterceptors() *ServiceInterceptors {
    baseChain := Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
    )

    return &ServiceInterceptors{
        User: Chain(
            baseChain,
            AuthInterceptor(),
            TimeoutInterceptor(10 * time.Second),
        ),
        Admin: Chain(
            baseChain,
            AuthInterceptor(),
            RequireAdminInterceptor(),
            AuditLogInterceptor(),
            TimeoutInterceptor(30 * time.Second),
        ),
        Public: Chain(
            baseChain,
            RateLimitInterceptor(1000),
            TimeoutInterceptor(5 * time.Second),
        ),
    }
}

// Группы interceptor для общих паттернов
var (
    // Минимальный: Только необходимое
    MinimalChain = Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
    )

    // Стандартный: Базовый production стек
    StandardChain = Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
        TimeoutInterceptor(30 * time.Second),
        MetricsInterceptor(),
    )

    // Защищённый: Аутентификация + авторизация
    SecureChain = Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
        AuthInterceptor(),
        AuthzInterceptor(),
        TimeoutInterceptor(15 * time.Second),
    )

    // Тяжёлый: Для дорогих операций
    HeavyChain = Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
        AuthInterceptor(),
        RateLimitInterceptor(5),
        TimeoutInterceptor(300 * time.Second),
    )
)
\`\`\`

**Реальные преимущества:**
- **DRY принцип:** Определите цепочки interceptors один раз, переиспользуйте везде
- **Консистентность:** Все методы в сервисе используют одни interceptors
- **Гибкость:** Легко создавать специализированные цепочки
- **Тестирование:** Тестируйте цепочки interceptors независимо

**Порядок выполнения цепочки:**
\`\`\`go
// Interceptors выполняются в порядке их появления
Chain(A, B, C)

// Поток выполнения:
// Request → A → B → C → handler → C → B → A → Response

// Почему обратная итерация работает:
// Chain оборачивает: wrapped = C(wrapped)  // wrapped = handler
//                     wrapped = B(wrapped)  // wrapped = C(handler)
//                     wrapped = A(wrapped)  // wrapped = A(B(C(handler)))
\`\`\`

**Лучшие практики:**
- **Порядок важен:** Recovery первым, auth перед бизнес-логикой
- **Создавайте стеки:** Определяйте переиспользуемые комбинации interceptors
- **Понятные имена:** MinimalChain, SecureChain, HeavyChain
- **Документируйте порядок:** Комментируйте почему interceptors в таком порядке

**Типичные паттерны цепочек:**
1. **Recovery → Logging → Timeout** (базовое трио)
2. **Recovery → Logging → Auth → Business** (защищённый RPC)
3. **Recovery → Logging → RateLimit → Business** (публичный RPC)
4. **Recovery → Logging → Auth → Heavy → Business** (дорогая операция)

Без Chain композиция interceptors требует ручного вложения, которое сложно читать и поддерживать.`,
			solutionCode: `package grpcx

import (
	"context"
)

type Handler func(ctx context.Context, req interface{}) (interface{}, error)

type UnaryServerInterceptor func(ctx context.Context, req interface{}, next Handler) (interface{}, error)

func Chain(interceptors ...UnaryServerInterceptor) UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, handler Handler) (interface{}, error) {
		if handler == nil {	// Проверка на nil handler
			handler = func(context.Context, interface{}) (interface{}, error) { return nil, nil }	// Используем no-op handler
		}
		wrapped := handler	// Начинаем с финального handler
		for i := len(interceptors) - 1; i >= 0; i-- {	// Итерируем в обратном порядке (справа налево)
			current := interceptors[i]	// Получаем текущий interceptor
			if current == nil {	// Пропускаем nil interceptors
				continue
			}
			next := wrapped	// Захватываем текущий обёрнутый handler
			wrapped = func(c context.Context, r interface{}) (interface{}, error) {	// Создаём новый handler
				return current(c, r, next)	// Вызываем interceptor со следующим handler
			}
		}
		return wrapped(ctx, req)	// Выполняем скомпонованную цепочку
	}
}`
		},
		uz: {
			title: 'Bir nechta interceptorlarni birlashtirish',
			description: `Bir nechta interceptorlarni bittaga birlashtirib, ularni tartib bilan qo'llaydigan **Chain** funksiyasini amalga oshiring.

**Talablar:**
1. \`Chain(interceptors ...UnaryServerInterceptor) UnaryServerInterceptor\` funksiyasini yarating
2. nil yakuniy handlerni ishlang (no-op dan foydalaning)
3. Chapdan o'ngga bajarilishga erishish uchun interceptorlarni o'ngdan chapga qo'llang
4. nil interceptorlarni o'tkazing
5. Barcha interceptorlarni ketma-ket bajariladigan kompozitsiya qilingan interceptorni qaytaring

**Misol:**
\`\`\`go
logging := LoggingInterceptor(logger)
timeout := TimeoutInterceptor(5 * time.Second)

combined := Chain(logging, timeout) // Logging birinchi bajariladi, keyin timeout

handler := func(ctx context.Context, req interface{}) (interface{}, error) {
    return "response", nil
}

resp, err := combined(ctx, "request", handler)
// Bajarilish: logging → timeout → handler
\`\`\`

**Cheklovlar:**
- Interceptorlarni ular paydo bo'lish tartibida (chapdan o'ngga) qo'llashi kerak
- nil interceptorlarni o'tkazishi kerak
- nil handlerni to'g'ri ishlashi kerak`,
			hint1: `Chapdan o'ngga bajarilish uchun ularni o'ngdan chapga o'rash uchun interceptorlar orqali teskari (len-1 dan 0 gacha) iteratsiya qiling.`,
			hint2: `Tsiklda o'zgaruvchini ushlab qolish muammolaridan qochish uchun "next" handlerni closurega ushlab qoling.`,
			whyItMatters: `Chain toza, o'qilishi mumkin interceptor kompozitsiyasini ta'minlaydi, murakkab gRPC middleware stacklarini osongina qurish va saqlashni amalga oshiradi.

**Nima uchun Interceptor Chaining kerak:**
- **O'qilish:** Interceptor stacki bajarilish tartibida
- **Qayta foydalanish:** Qayta ishlatiladigan interceptor kombinatsiyalarini yaratish
- **Saqlanish:** Interceptorlarni qo'shish/o'chirish/qayta tartibga solish oson
- **Kompozitsiya:** Oddiy qismlardan murakkab xatti-harakatni qurish

**Production Paterni:**
\`\`\`go
// Barcha servislar uchun standart interceptor stacki
var StandardInterceptors = Chain(
    RecoveryInterceptor(),     // Birinchi panikani ushlash
    LoggingInterceptor(),      // Barcha so'rovlarni log qilish
    TimeoutInterceptor(30 * time.Second),  // Timeoutni o'rnatish
)

// Authentication stacki
var AuthInterceptors = Chain(
    StandardInterceptors,      // Standart stackni kiritamiz
    AuthInterceptor(),         // Autentifikatsiyani talab qilish
    RateLimitInterceptor(100), // So'rov chastotasini cheklash
)

// gRPC server bilan foydalanish
server := grpc.NewServer(
    grpc.UnaryInterceptor(AuthInterceptors),
)

// Maxsus ehtiyojlar uchun moslashtirilgan stacklar
var PublicServiceInterceptors = Chain(
    RecoveryInterceptor(),
    LoggingInterceptor(),
    TimeoutInterceptor(10 * time.Second),
    RateLimitInterceptor(1000),
)

var AdminServiceInterceptors = Chain(
    RecoveryInterceptor(),
    LoggingInterceptor(),
    AuthInterceptor(),
    RequireAdminInterceptor(),
    TimeoutInterceptor(60 * time.Second),
)

var StreamingServiceInterceptors = Chain(
    RecoveryInterceptor(),
    LoggingInterceptor(),
    TimeoutInterceptor(300 * time.Second),  // Oqimlar uchun uzoq timeout
)

// Metod bo'yicha interceptor tanlash
func MethodInterceptorSelector() grpc.UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
        var interceptor UnaryServerInterceptor

        switch {
        case strings.HasPrefix(info.FullMethod, "/admin."):
            interceptor = AdminServiceInterceptors
        case strings.HasPrefix(info.FullMethod, "/public."):
            interceptor = PublicServiceInterceptors
        default:
            interceptor = StandardInterceptors
        }

        return interceptor(ctx, req, handler)
    }
}

// Shartli interceptor
func ConditionalChain(condition bool, interceptor UnaryServerInterceptor) UnaryServerInterceptor {
    if condition {
        return interceptor
    }
    return func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
        return next(ctx, req) // O'tkazib yuborish
    }
}

// Foydalanish: Faqat production da auth qo'shish
var stack = Chain(
    RecoveryInterceptor(),
    LoggingInterceptor(),
    ConditionalChain(isProduction, AuthInterceptor()),
)

// Dinamik zanjir quruvchisi
type InterceptorChainBuilder struct {
    interceptors []UnaryServerInterceptor
}

func NewChainBuilder() *InterceptorChainBuilder {
    return &InterceptorChainBuilder{}
}

func (b *InterceptorChainBuilder) Use(interceptor UnaryServerInterceptor) *InterceptorChainBuilder {
    b.interceptors = append(b.interceptors, interceptor)
    return b
}

func (b *InterceptorChainBuilder) UseIf(condition bool, interceptor UnaryServerInterceptor) *InterceptorChainBuilder {
    if condition {
        b.interceptors = append(b.interceptors, interceptor)
    }
    return b
}

func (b *InterceptorChainBuilder) Build() UnaryServerInterceptor {
    return Chain(b.interceptors...)
}

// Foydalanish: Zanjirni dinamik qurish
func BuildServiceInterceptors(config Config) UnaryServerInterceptor {
    return NewChainBuilder().
        Use(RecoveryInterceptor()).
        Use(LoggingInterceptor()).
        UseIf(config.EnableAuth, AuthInterceptor()).
        UseIf(config.EnableRateLimit, RateLimitInterceptor(config.RateLimit)).
        UseIf(config.EnableMetrics, MetricsInterceptor()).
        Build()
}

// Servislar uchun bir nechta interceptor zanjirlari
type ServiceInterceptors struct {
    User   UnaryServerInterceptor
    Admin  UnaryServerInterceptor
    Public UnaryServerInterceptor
}

func NewServiceInterceptors() *ServiceInterceptors {
    baseChain := Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
    )

    return &ServiceInterceptors{
        User: Chain(
            baseChain,
            AuthInterceptor(),
            TimeoutInterceptor(10 * time.Second),
        ),
        Admin: Chain(
            baseChain,
            AuthInterceptor(),
            RequireAdminInterceptor(),
            AuditLogInterceptor(),
            TimeoutInterceptor(30 * time.Second),
        ),
        Public: Chain(
            baseChain,
            RateLimitInterceptor(1000),
            TimeoutInterceptor(5 * time.Second),
        ),
    }
}

// Umumiy patternlar uchun interceptor guruhlari
var (
    // Minimal: Faqat kerakli
    MinimalChain = Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
    )

    // Standart: Asosiy production stack
    StandardChain = Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
        TimeoutInterceptor(30 * time.Second),
        MetricsInterceptor(),
    )

    // Himoyalangan: Autentifikatsiya + avtorizatsiya
    SecureChain = Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
        AuthInterceptor(),
        AuthzInterceptor(),
        TimeoutInterceptor(15 * time.Second),
    )

    // Og'ir: Qimmat operatsiyalar uchun
    HeavyChain = Chain(
        RecoveryInterceptor(),
        LoggingInterceptor(),
        AuthInterceptor(),
        RateLimitInterceptor(5),
        TimeoutInterceptor(300 * time.Second),
    )
)
\`\`\`

**Haqiqiy foydalari:**
- **DRY printsipi:** Interceptor zanjirlarini bir marta aniqlang, hamma joyda qayta ishlating
- **Izchillik:** Servisdagi barcha metodlar bir xil interceptorlarni ishlatadi
- **Moslashuvchanlik:** Maxsus zanjirlarni osongina yaratish
- **Sinov:** Interceptor zanjirlarini mustaqil sinash

**Zanjir bajarilish tartibi:**
\`\`\`go
// Interceptorlar paydo bo'lish tartibida bajariladi
Chain(A, B, C)

// Bajarilish oqimi:
// Request → A → B → C → handler → C → B → A → Response

// Nima uchun teskari iteratsiya ishlaydi:
// Chain o'raydi: wrapped = C(wrapped)  // wrapped = handler
//                wrapped = B(wrapped)  // wrapped = C(handler)
//                wrapped = A(wrapped)  // wrapped = A(B(C(handler)))
\`\`\`

**Eng yaxshi amaliyotlar:**
- **Tartib muhim:** Recovery birinchi, biznes mantiqdan oldin auth
- **Stacklar yarating:** Qayta ishlatiladigan interceptor kombinatsiyalarini aniqlang
- **Tushunarli nomlar:** MinimalChain, SecureChain, HeavyChain
- **Tartibni hujjatlantiring:** Nega interceptorlar bunday tartibda ekanligini izoh qiling

**Odatiy zanjir patternlari:**
1. **Recovery → Logging → Timeout** (asosiy trio)
2. **Recovery → Logging → Auth → Business** (himoyalangan RPC)
3. **Recovery → Logging → RateLimit → Business** (ommaviy RPC)
4. **Recovery → Logging → Auth → Heavy → Business** (qimmat operatsiya)

Chain siz interceptor kompozitsiyasi qo'lda ichma-ich kiritishni talab qiladi, buni o'qish va saqlash qiyin.`,
			solutionCode: `package grpcx

import (
	"context"
)

type Handler func(ctx context.Context, req interface{}) (interface{}, error)

type UnaryServerInterceptor func(ctx context.Context, req interface{}, next Handler) (interface{}, error)

func Chain(interceptors ...UnaryServerInterceptor) UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, handler Handler) (interface{}, error) {
		if handler == nil {	// Handler nil ekanligini tekshirish
			handler = func(context.Context, interface{}) (interface{}, error) { return nil, nil }	// No-op handler ishlatamiz
		}
		wrapped := handler	// Yakuniy handlerdan boshlaymiz
		for i := len(interceptors) - 1; i >= 0; i-- {	// Teskari tartibda iteratsiya qilamiz (o'ngdan chapga)
			current := interceptors[i]	// Hozirgi interceptorni olamiz
			if current == nil {	// nil interceptorlarni o'tkazamiz
				continue
			}
			next := wrapped	// Hozirgi o'ralgan handlerni ushlab qolamiz
			wrapped = func(c context.Context, r interface{}) (interface{}, error) {	// Yangi handler yaratamiz
				return current(c, r, next)	// Keyingi handler bilan interceptorni chaqiramiz
			}
		}
		return wrapped(ctx, req)	// Kompozitsiya qilingan zanjirni bajaramiz
	}
}`
		}
	}
};

export default task;
