import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-grpc-context-value-interceptor',
	title: 'Context Value Interceptor',
	difficulty: 'easy',	tags: ['go', 'grpc', 'interceptors', 'context'],
	estimatedTime: '15m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **ContextValueInterceptor** that adds a key-value pair to the context before passing it to the handler.

**Requirements:**
1. Create function \`ContextValueInterceptor(key any, value any) UnaryServerInterceptor\`
2. Handle nil handler (use no-op)
3. Add value to context using context.WithValue
4. Pass modified context to handler
5. Return handler's response and error

**Example:**
\`\`\`go
const RequestIDKey = "request_id"

interceptor := ContextValueInterceptor(RequestIDKey, "12345")

handler := func(ctx context.Context, req any) (any, error) {
    requestID := ctx.Value(RequestIDKey).(string)
    // requestID = "12345"
    return requestID, nil
}

resp, err := interceptor(ctx, "request", handler)
// resp = "12345"
\`\`\`

**Constraints:**
- Must use context.WithValue to add value
- Must pass modified context to handler`,
	initialCode: `package grpcx

import (
	"context"
)

type Handler func(ctx context.Context, req any) (any, error)

type UnaryServerInterceptor func(ctx context.Context, req any, next Handler) (any, error)

// TODO: Implement ContextValueInterceptor
func ContextValueInterceptor(key any, value any) UnaryServerInterceptor {
	// TODO: Implement
}`,
	solutionCode: `package grpcx

import (
	"context"
)

type Handler func(ctx context.Context, req any) (any, error)

type UnaryServerInterceptor func(ctx context.Context, req any, next Handler) (any, error)

func ContextValueInterceptor(key any, value any) UnaryServerInterceptor {
	return func(ctx context.Context, req any, handler Handler) (any, error) {
		if handler == nil {	// Check if handler is nil
			handler = func(context.Context, any) (any, error) { return nil, nil }	// Use no-op handler
		}
		ctx = context.WithValue(ctx, key, value)	// Add value to context
		return handler(ctx, req)	// Execute handler with modified context
	}
}`,
			hint1: `Use context.WithValue(ctx, key, value) to create a new context with the value.`,
			hint2: `Pass the new context to the handler by calling handler(ctx, req).`,
			testCode: `package grpcx

import (
	"context"
	"errors"
	"testing"
)

func Test1(t *testing.T) {
	// Test value is added to context
	interceptor := ContextValueInterceptor("key", "value")
	handler := func(ctx context.Context, req any) (any, error) {
		if ctx.Value("key") != "value" {
			t.Error("value not found in context")
		}
		return nil, nil
	}
	interceptor(context.Background(), nil, handler)
}

func Test2(t *testing.T) {
	// Test interceptor returns non-nil
	interceptor := ContextValueInterceptor("key", "value")
	if interceptor == nil {
		t.Error("interceptor should not be nil")
	}
}

func Test3(t *testing.T) {
	// Test response is passed through
	interceptor := ContextValueInterceptor("key", "value")
	handler := func(ctx context.Context, req any) (any, error) {
		return "response", nil
	}
	resp, err := interceptor(context.Background(), nil, handler)
	if err != nil || resp != "response" {
		t.Errorf("got (%v, %v), want (response, nil)", resp, err)
	}
}

func Test4(t *testing.T) {
	// Test error is passed through
	expectedErr := errors.New("handler error")
	interceptor := ContextValueInterceptor("key", "value")
	handler := func(ctx context.Context, req any) (any, error) {
		return nil, expectedErr
	}
	_, err := interceptor(context.Background(), nil, handler)
	if err != expectedErr {
		t.Errorf("error = %v, want %v", err, expectedErr)
	}
}

func Test5(t *testing.T) {
	// Test nil handler uses no-op
	interceptor := ContextValueInterceptor("key", "value")
	resp, err := interceptor(context.Background(), nil, nil)
	if err != nil || resp != nil {
		t.Errorf("got (%v, %v), want (nil, nil)", resp, err)
	}
}

func Test6(t *testing.T) {
	// Test request is passed to handler
	var receivedReq any
	interceptor := ContextValueInterceptor("key", "value")
	handler := func(ctx context.Context, req any) (any, error) {
		receivedReq = req
		return nil, nil
	}
	interceptor(context.Background(), "test-request", handler)
	if receivedReq != "test-request" {
		t.Errorf("request = %v, want 'test-request'", receivedReq)
	}
}

func Test7(t *testing.T) {
	// Test existing context values preserved
	ctx := context.WithValue(context.Background(), "existing", "preserved")
	interceptor := ContextValueInterceptor("new", "added")
	handler := func(c context.Context, req any) (any, error) {
		if c.Value("existing") != "preserved" {
			t.Error("existing value not preserved")
		}
		if c.Value("new") != "added" {
			t.Error("new value not added")
		}
		return nil, nil
	}
	interceptor(ctx, nil, handler)
}

func Test8(t *testing.T) {
	// Test different key types
	type customKey struct{}
	interceptor := ContextValueInterceptor(customKey{}, "custom-value")
	handler := func(ctx context.Context, req any) (any, error) {
		if ctx.Value(customKey{}) != "custom-value" {
			t.Error("custom key value not found")
		}
		return nil, nil
	}
	interceptor(context.Background(), nil, handler)
}

func Test9(t *testing.T) {
	// Test nil value can be stored
	interceptor := ContextValueInterceptor("key", nil)
	handler := func(ctx context.Context, req any) (any, error) {
		if ctx.Value("key") != nil {
			t.Error("nil value not stored correctly")
		}
		return nil, nil
	}
	interceptor(context.Background(), nil, handler)
}

func Test10(t *testing.T) {
	// Test multiple interceptors stack correctly
	first := ContextValueInterceptor("first", "1")
	second := ContextValueInterceptor("second", "2")
	handler := func(ctx context.Context, req any) (any, error) {
		if ctx.Value("first") != "1" || ctx.Value("second") != "2" {
			t.Error("not all values found")
		}
		return nil, nil
	}
	innerHandler := func(ctx context.Context, req any) (any, error) {
		return second(ctx, req, handler)
	}
	first(context.Background(), nil, innerHandler)
}`,
			whyItMatters: `ContextValueInterceptor enables request-scoped data propagation throughout the request lifecycle, essential for distributed tracing and request metadata.

**Why Context Values:**
- **Request Tracing:** Propagate trace IDs through service calls
- **User Context:** Pass authenticated user info to handlers
- **Tenant Isolation:** Store tenant ID for multi-tenant systems
- **Request Metadata:** Carry request-specific data without changing signatures

**Production Pattern:**
\`\`\`go
// Request ID propagation
func RequestIDInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Try to get request ID from metadata
        requestID := ""
        md, ok := metadata.FromIncomingContext(ctx)
        if ok {
            if ids := md.Get("x-request-id"); len(ids) > 0 {
                requestID = ids[0]
            }
        }

        // Generate new request ID if not present
        if requestID == "" {
            requestID = uuid.New().String()
        }

        // Add to context
        ctx = context.WithValue(ctx, "request_id", requestID)

        // Add to outgoing metadata for downstream services
        ctx = metadata.AppendToOutgoingContext(ctx, "x-request-id", requestID)

        return handler(ctx, req)
    }
}

// Authentication context
func AuthContextInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Extract JWT from metadata
        token := extractToken(ctx)
        if token == "" {
            return nil, status.Error(codes.Unauthenticated, "missing token")
        }

        // Validate and parse token
        claims, err := validateToken(token)
        if err != nil {
            return nil, status.Error(codes.Unauthenticated, "invalid token")
        }

        // Add user info to context
        ctx = context.WithValue(ctx, "user_id", claims.UserID)
        ctx = context.WithValue(ctx, "user_email", claims.Email)
        ctx = context.WithValue(ctx, "user_roles", claims.Roles)

        return handler(ctx, req)
    }
}

// Multi-tenant context
func TenantContextInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Extract tenant ID from metadata
        tenantID := ""
        md, ok := metadata.FromIncomingContext(ctx)
        if ok {
            if ids := md.Get("x-tenant-id"); len(ids) > 0 {
                tenantID = ids[0]
            }
        }

        if tenantID == "" {
            return nil, status.Error(codes.InvalidArgument, "missing tenant ID")
        }

        // Validate tenant
        tenant, err := getTenant(tenantID)
        if err != nil {
            return nil, status.Error(codes.NotFound, "tenant not found")
        }

        // Add tenant info to context
        ctx = context.WithValue(ctx, "tenant_id", tenantID)
        ctx = context.WithValue(ctx, "tenant", tenant)

        return handler(ctx, req)
    }
}

// Distributed tracing context
func TracingContextInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Extract trace context from metadata
        md, _ := metadata.FromIncomingContext(ctx)

        traceID := getMetadataValue(md, "x-trace-id")
        spanID := getMetadataValue(md, "x-span-id")
        parentSpanID := getMetadataValue(md, "x-parent-span-id")

        // Generate IDs if not present
        if traceID == "" {
            traceID = generateTraceID()
        }
        if spanID == "" {
            spanID = generateSpanID()
        }

        // Create trace context
        traceCtx := TraceContext{
            TraceID:      traceID,
            SpanID:       spanID,
            ParentSpanID: parentSpanID,
            Method:       info.FullMethod,
            StartTime:    time.Now(),
        }

        // Add to context
        ctx = context.WithValue(ctx, "trace_context", traceCtx)

        // Execute handler
        resp, err := handler(ctx, req)

        // Record span
        traceCtx.Duration = time.Since(traceCtx.StartTime)
        traceCtx.Success = err == nil
        recordSpan(traceCtx)

        return resp, err
    }
}

// Correlation ID for logging
func CorrelationIDInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Get or generate correlation ID
        correlationID := uuid.New().String()
        md, ok := metadata.FromIncomingContext(ctx)
        if ok {
            if ids := md.Get("x-correlation-id"); len(ids) > 0 {
                correlationID = ids[0]
            }
        }

        // Add to context for logging
        ctx = context.WithValue(ctx, "correlation_id", correlationID)

        // Add to logger
        logger := log.WithField("correlation_id", correlationID)
        ctx = context.WithValue(ctx, "logger", logger)

        return handler(ctx, req)
    }
}

// Chain multiple context values
func ContextChain() UnaryServerInterceptor {
    return Chain(
        RequestIDInterceptor(),
        CorrelationIDInterceptor(),
        AuthContextInterceptor(),
        TenantContextInterceptor(),
        TracingContextInterceptor(),
    )
}

// Usage in handler
func GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    // Extract context values
    requestID := ctx.Value("request_id").(string)
    userID := ctx.Value("user_id").(string)
    tenantID := ctx.Value("tenant_id").(string)
    logger := ctx.Value("logger").(*log.Entry)

    logger.Infof("GetUser called: request_id=%s user_id=%s tenant_id=%s",
        requestID, userID, tenantID)

    // Use tenant-specific database
    tenant := ctx.Value("tenant").(*Tenant)
    db := tenant.GetDatabase()

    // Query user
    user, err := db.GetUser(ctx, req.Id)
    if err != nil {
        logger.Errorf("Failed to get user: %v", err)
        return nil, err
    }

    return user, nil
}

// Context value helpers
func GetRequestID(ctx context.Context) string {
    if id, ok := ctx.Value("request_id").(string); ok {
        return id
    }
    return ""
}

func GetUserID(ctx context.Context) string {
    if id, ok := ctx.Value("user_id").(string); ok {
        return id
    }
    return ""
}

func GetTenantID(ctx context.Context) string {
    if id, ok := ctx.Value("tenant_id").(string); ok {
        return id
    }
    return ""
}

func GetLogger(ctx context.Context) *log.Entry {
    if logger, ok := ctx.Value("logger").(*log.Entry); ok {
        return logger
    }
    return log.NewEntry(log.StandardLogger())
}

// Type-safe context keys
type contextKey string

const (
    RequestIDKey    contextKey = "request_id"
    UserIDKey       contextKey = "user_id"
    TenantIDKey     contextKey = "tenant_id"
    TraceContextKey contextKey = "trace_context"
)

// Typed context value accessors
func WithRequestID(ctx context.Context, requestID string) context.Context {
    return context.WithValue(ctx, RequestIDKey, requestID)
}

func GetRequestIDTyped(ctx context.Context) (string, bool) {
    id, ok := ctx.Value(RequestIDKey).(string)
    return id, ok
}
\`\`\`

**Real-World Benefits:**
- **Clean Code:** Pass data without modifying function signatures
- **Request Scoping:** Values automatically scoped to request
- **Tracing:** Propagate trace IDs through entire call chain
- **Multi-Tenancy:** Isolate tenants via context values

**Context Value Best Practices:**
- **Type-Safe Keys:** Use typed keys to avoid collisions
- **Don't Overuse:** Context is not for passing optional parameters
- **Immutability:** Contexts are immutable, WithValue creates new context
- **Value Types:** Store small, immutable values
- **Documentation:** Document what values your code expects

**Common Context Values:**
- **Request ID:** Unique identifier for request tracing
- **User ID:** Authenticated user identifier
- **Tenant ID:** Multi-tenant isolation
- **Trace Context:** Distributed tracing metadata
- **Logger:** Request-scoped logger
- **Database Connection:** Tenant-specific database

**Anti-Patterns:**
- **Large Objects:** Don't store large objects in context
- **Mutable Data:** Don't store pointers to mutable data
- **Business Logic:** Don't use context for business logic parameters
- **String Keys:** Use typed keys to avoid collisions

Without ContextValueInterceptor, request-scoped data must be passed through every function parameter—verbose and error-prone.`,	order: 4,
	translations: {
		ru: {
			title: 'Передача значений через контекст в gRPC',
			description: `Реализуйте **ContextValueInterceptor**, который добавляет пару ключ-значение в контекст перед передачей в handler.

**Требования:**
1. Создайте функцию \`ContextValueInterceptor(key any, value any) UnaryServerInterceptor\`
2. Обработайте nil handler (используйте no-op)
3. Добавьте значение в контекст используя context.WithValue
4. Передайте модифицированный контекст в handler
5. Верните ответ и ошибку handler

**Пример:**
\`\`\`go
const RequestIDKey = "request_id"

interceptor := ContextValueInterceptor(RequestIDKey, "12345")

handler := func(ctx context.Context, req any) (any, error) {
    requestID := ctx.Value(RequestIDKey).(string)
    // requestID = "12345"
    return requestID, nil
}

resp, err := interceptor(ctx, "request", handler)
// resp = "12345"
\`\`\`

**Ограничения:**
- Должен использовать context.WithValue для добавления значения
- Должен передавать модифицированный контекст в handler`,
			hint1: `Используйте context.WithValue(ctx, key, value) для создания нового контекста со значением.`,
			hint2: `Передайте новый контекст в handler вызвав handler(ctx, req).`,
			whyItMatters: `ContextValueInterceptor обеспечивает распространение request-scoped данных на протяжении всего жизненного цикла запроса, необходимо для distributed tracing и request metadata.

**Зачем Нужны Context Values:**
- **Request Tracing:** Распространение trace IDs через service calls
- **User Context:** Передача информации об аутентифицированном пользователе handlers
- **Tenant Isolation:** Хранение tenant ID для multi-tenant систем
- **Request Metadata:** Передача request-специфичных данных без изменения сигнатур

**Production Паттерны:**

**Распространение Request ID:**
\`\`\`go
func RequestIDInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Пытаемся получить request ID из метаданных
        requestID := ""
        md, ok := metadata.FromIncomingContext(ctx)
        if ok {
            if ids := md.Get("x-request-id"); len(ids) > 0 {
                requestID = ids[0]
            }
        }

        // Генерируем новый request ID если отсутствует
        if requestID == "" {
            requestID = uuid.New().String()
        }

        // Добавляем в контекст
        ctx = context.WithValue(ctx, "request_id", requestID)

        // Добавляем в исходящие метаданные для downstream сервисов
        ctx = metadata.AppendToOutgoingContext(ctx, "x-request-id", requestID)

        return handler(ctx, req)
    }
}
\`\`\`

**Authentication Контекст:**
\`\`\`go
func AuthContextInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Извлекаем JWT из метаданных
        token := extractToken(ctx)
        if token == "" {
            return nil, status.Error(codes.Unauthenticated, "missing token")
        }

        // Валидируем и парсим токен
        claims, err := validateToken(token)
        if err != nil {
            return nil, status.Error(codes.Unauthenticated, "invalid token")
        }

        // Добавляем user info в контекст
        ctx = context.WithValue(ctx, "user_id", claims.UserID)
        ctx = context.WithValue(ctx, "user_email", claims.Email)
        ctx = context.WithValue(ctx, "user_roles", claims.Roles)

        return handler(ctx, req)
    }
}
\`\`\`

**Multi-Tenant Контекст:**
\`\`\`go
func TenantContextInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Извлекаем tenant ID из метаданных
        tenantID := ""
        md, ok := metadata.FromIncomingContext(ctx)
        if ok {
            if ids := md.Get("x-tenant-id"); len(ids) > 0 {
                tenantID = ids[0]
            }
        }

        if tenantID == "" {
            return nil, status.Error(codes.InvalidArgument, "missing tenant ID")
        }

        // Валидируем tenant
        tenant, err := getTenant(tenantID)
        if err != nil {
            return nil, status.Error(codes.NotFound, "tenant not found")
        }

        // Добавляем tenant info в контекст
        ctx = context.WithValue(ctx, "tenant_id", tenantID)
        ctx = context.WithValue(ctx, "tenant", tenant)

        return handler(ctx, req)
    }
}
\`\`\`

**Distributed Tracing Контекст:**
\`\`\`go
func TracingContextInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Извлекаем trace context из метаданных
        md, _ := metadata.FromIncomingContext(ctx)

        traceID := getMetadataValue(md, "x-trace-id")
        spanID := getMetadataValue(md, "x-span-id")
        parentSpanID := getMetadataValue(md, "x-parent-span-id")

        // Генерируем IDs если отсутствуют
        if traceID == "" {
            traceID = generateTraceID()
        }
        if spanID == "" {
            spanID = generateSpanID()
        }

        // Создаём trace context
        traceCtx := TraceContext{
            TraceID:      traceID,
            SpanID:       spanID,
            ParentSpanID: parentSpanID,
            Method:       info.FullMethod,
            StartTime:    time.Now(),
        }

        // Добавляем в контекст
        ctx = context.WithValue(ctx, "trace_context", traceCtx)

        // Выполняем handler
        resp, err := handler(ctx, req)

        // Записываем span
        traceCtx.Duration = time.Since(traceCtx.StartTime)
        traceCtx.Success = err == nil
        recordSpan(traceCtx)

        return resp, err
    }
}
\`\`\`

**Correlation ID для Логирования:**
\`\`\`go
func CorrelationIDInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Получаем или генерируем correlation ID
        correlationID := uuid.New().String()
        md, ok := metadata.FromIncomingContext(ctx)
        if ok {
            if ids := md.Get("x-correlation-id"); len(ids) > 0 {
                correlationID = ids[0]
            }
        }

        // Добавляем в контекст для логирования
        ctx = context.WithValue(ctx, "correlation_id", correlationID)

        // Добавляем к логгеру
        logger := log.WithField("correlation_id", correlationID)
        ctx = context.WithValue(ctx, "logger", logger)

        return handler(ctx, req)
    }
}
\`\`\`

**Цепочка Множественных Context Values:**
\`\`\`go
func ContextChain() UnaryServerInterceptor {
    return Chain(
        RequestIDInterceptor(),
        CorrelationIDInterceptor(),
        AuthContextInterceptor(),
        TenantContextInterceptor(),
        TracingContextInterceptor(),
    )
}
\`\`\`

**Использование в Handler:**
\`\`\`go
func GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    // Извлекаем значения контекста
    requestID := ctx.Value("request_id").(string)
    userID := ctx.Value("user_id").(string)
    tenantID := ctx.Value("tenant_id").(string)
    logger := ctx.Value("logger").(*log.Entry)

    logger.Infof("GetUser called: request_id=%s user_id=%s tenant_id=%s",
        requestID, userID, tenantID)

    // Используем tenant-специфичную базу данных
    tenant := ctx.Value("tenant").(*Tenant)
    db := tenant.GetDatabase()

    // Запрашиваем пользователя
    user, err := db.GetUser(ctx, req.Id)
    if err != nil {
        logger.Errorf("Failed to get user: %v", err)
        return nil, err
    }

    return user, nil
}
\`\`\`

**Context Value Helpers:**
\`\`\`go
func GetRequestID(ctx context.Context) string {
    if id, ok := ctx.Value("request_id").(string); ok {
        return id
    }
    return ""
}

func GetUserID(ctx context.Context) string {
    if id, ok := ctx.Value("user_id").(string); ok {
        return id
    }
    return ""
}

func GetTenantID(ctx context.Context) string {
    if id, ok := ctx.Value("tenant_id").(string); ok {
        return id
    }
    return ""
}

func GetLogger(ctx context.Context) *log.Entry {
    if logger, ok := ctx.Value("logger").(*log.Entry); ok {
        return logger
    }
    return log.NewEntry(log.StandardLogger())
}
\`\`\`

**Type-Safe Context Keys:**
\`\`\`go
type contextKey string

const (
    RequestIDKey    contextKey = "request_id"
    UserIDKey       contextKey = "user_id"
    TenantIDKey     contextKey = "tenant_id"
    TraceContextKey contextKey = "trace_context"
)

// Typed context value accessors
func WithRequestID(ctx context.Context, requestID string) context.Context {
    return context.WithValue(ctx, RequestIDKey, requestID)
}

func GetRequestIDTyped(ctx context.Context) (string, bool) {
    id, ok := ctx.Value(RequestIDKey).(string)
    return id, ok
}
\`\`\`

**Реальные Преимущества:**
- **Чистый Код:** Передача данных без изменения сигнатур функций
- **Request Scoping:** Значения автоматически привязаны к запросу
- **Tracing:** Распространение trace IDs через всю цепочку вызовов
- **Multi-Tenancy:** Изоляция tenants через значения контекста

**Лучшие Практики Context Value:**
- **Type-Safe Ключи:** Используйте типизированные ключи чтобы избежать коллизий
- **Не Злоупотребляйте:** Context не для передачи опциональных параметров
- **Неизменяемость:** Contexts неизменяемы, WithValue создаёт новый контекст
- **Типы Значений:** Храните маленькие, неизменяемые значения
- **Документация:** Документируйте какие значения ваш код ожидает

**Типичные Context Values:**
- **Request ID:** Уникальный идентификатор для request tracing
- **User ID:** Идентификатор аутентифицированного пользователя
- **Tenant ID:** Изоляция multi-tenant
- **Trace Context:** Метаданные distributed tracing
- **Logger:** Request-scoped логгер
- **Database Connection:** Tenant-специфичная база данных

**Anti-Patterns:**
- **Большие Объекты:** Не храните большие объекты в контексте
- **Изменяемые Данные:** Не храните указатели на изменяемые данные
- **Бизнес-Логика:** Не используйте контекст для параметров бизнес-логики
- **String Ключи:** Используйте типизированные ключи чтобы избежать коллизий

Без ContextValueInterceptor request-scoped данные должны передаваться через каждый параметр функции—многословно и подвержено ошибкам.`,
			solutionCode: `package grpcx

import (
	"context"
)

type Handler func(ctx context.Context, req any) (any, error)

type UnaryServerInterceptor func(ctx context.Context, req any, next Handler) (any, error)

func ContextValueInterceptor(key any, value any) UnaryServerInterceptor {
	return func(ctx context.Context, req any, handler Handler) (any, error) {
		if handler == nil {	// Проверка на nil handler
			handler = func(context.Context, any) (any, error) { return nil, nil }	// Используем no-op handler
		}
		ctx = context.WithValue(ctx, key, value)	// Добавляем значение в контекст
		return handler(ctx, req)	// Выполняем handler с модифицированным контекстом
	}
}`
		},
		uz: {
			title: 'gRPC da kontekst orqali qiymatlarni uzatish',
			description: `Handlerga o'tkazishdan oldin kontekstga kalit-qiymat juftligini qo'shuvchi **ContextValueInterceptor** ni amalga oshiring.

**Talablar:**
1. \`ContextValueInterceptor(key any, value any) UnaryServerInterceptor\` funksiyasini yarating
2. nil handlerni ishlang (no-op dan foydalaning)
3. context.WithValue dan foydalanib kontekstga qiymatni qo'shing
4. O'zgartirilgan kontekstni handlerga o'tkazing
5. Handler javob va xatosini qaytaring

**Misol:**
\`\`\`go
const RequestIDKey = "request_id"

interceptor := ContextValueInterceptor(RequestIDKey, "12345")

handler := func(ctx context.Context, req any) (any, error) {
    requestID := ctx.Value(RequestIDKey).(string)
    // requestID = "12345"
    return requestID, nil
}

resp, err := interceptor(ctx, "request", handler)
// resp = "12345"
\`\`\`

**Cheklovlar:**
- Qiymatni qo'shish uchun context.WithValue dan foydalanishi kerak
- O'zgartirilgan kontekstni handlerga o'tkazishi kerak`,
			hint1: `Qiymat bilan yangi kontekst yaratish uchun context.WithValue(ctx, key, value) dan foydalaning.`,
			hint2: `handler(ctx, req) ni chaqirib, yangi kontekstni handlerga o'tkazing.`,
			whyItMatters: `ContextValueInterceptor request hayot davri mobaynida request-scoped ma'lumotlarning tarqalishini ta'minlaydi, distributed tracing va request metadata uchun zarur.

**Nega Context Values Kerak:**
- **Request Tracing:** Service chaqiruvlar orqali trace ID larni tarqatish
- **User Context:** Autentifikatsiya qilingan foydalanuvchi ma'lumotini handlerlarga o'tkazish
- **Tenant Isolation:** Multi-tenant tizimlar uchun tenant ID ni saqlash
- **Request Metadata:** Imzolarni o'zgartirmasdan request-spetsifik ma'lumotlarni uzatish

**Production Patternlari:**

**Request ID ni Tarqatish:**
\`\`\`go
func RequestIDInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Metadatadan request ID ni olishga harakat qilamiz
        requestID := ""
        md, ok := metadata.FromIncomingContext(ctx)
        if ok {
            if ids := md.Get("x-request-id"); len(ids) > 0 {
                requestID = ids[0]
            }
        }

        // Yo'q bo'lsa yangi request ID yaratamiz
        if requestID == "" {
            requestID = uuid.New().String()
        }

        // Kontekstga qo'shamiz
        ctx = context.WithValue(ctx, "request_id", requestID)

        // Downstream servislar uchun chiquvchi metadataga qo'shamiz
        ctx = metadata.AppendToOutgoingContext(ctx, "x-request-id", requestID)

        return handler(ctx, req)
    }
}
\`\`\`

**Authentication Konteksti:**
\`\`\`go
func AuthContextInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Metadatadan JWT ni chiqarib olamiz
        token := extractToken(ctx)
        if token == "" {
            return nil, status.Error(codes.Unauthenticated, "missing token")
        }

        // Tokenni tekshiramiz va tahlil qilamiz
        claims, err := validateToken(token)
        if err != nil {
            return nil, status.Error(codes.Unauthenticated, "invalid token")
        }

        // User info ni kontekstga qo'shamiz
        ctx = context.WithValue(ctx, "user_id", claims.UserID)
        ctx = context.WithValue(ctx, "user_email", claims.Email)
        ctx = context.WithValue(ctx, "user_roles", claims.Roles)

        return handler(ctx, req)
    }
}
\`\`\`

**Multi-Tenant Konteksti:**
\`\`\`go
func TenantContextInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Metadatadan tenant ID ni chiqarib olamiz
        tenantID := ""
        md, ok := metadata.FromIncomingContext(ctx)
        if ok {
            if ids := md.Get("x-tenant-id"); len(ids) > 0 {
                tenantID = ids[0]
            }
        }

        if tenantID == "" {
            return nil, status.Error(codes.InvalidArgument, "missing tenant ID")
        }

        // Tenantni tekshiramiz
        tenant, err := getTenant(tenantID)
        if err != nil {
            return nil, status.Error(codes.NotFound, "tenant not found")
        }

        // Tenant info ni kontekstga qo'shamiz
        ctx = context.WithValue(ctx, "tenant_id", tenantID)
        ctx = context.WithValue(ctx, "tenant", tenant)

        return handler(ctx, req)
    }
}
\`\`\`

**Distributed Tracing Konteksti:**
\`\`\`go
func TracingContextInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Metadatadan trace kontekstni chiqarib olamiz
        md, _ := metadata.FromIncomingContext(ctx)

        traceID := getMetadataValue(md, "x-trace-id")
        spanID := getMetadataValue(md, "x-span-id")
        parentSpanID := getMetadataValue(md, "x-parent-span-id")

        // Yo'q bo'lsa ID larni yaratamiz
        if traceID == "" {
            traceID = generateTraceID()
        }
        if spanID == "" {
            spanID = generateSpanID()
        }

        // Trace kontekstni yaratamiz
        traceCtx := TraceContext{
            TraceID:      traceID,
            SpanID:       spanID,
            ParentSpanID: parentSpanID,
            Method:       info.FullMethod,
            StartTime:    time.Now(),
        }

        // Kontekstga qo'shamiz
        ctx = context.WithValue(ctx, "trace_context", traceCtx)

        // Handlerni bajaramiz
        resp, err := handler(ctx, req)

        // Spanni yozamiz
        traceCtx.Duration = time.Since(traceCtx.StartTime)
        traceCtx.Success = err == nil
        recordSpan(traceCtx)

        return resp, err
    }
}
\`\`\`

**Logging Uchun Correlation ID:**
\`\`\`go
func CorrelationIDInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler Handler) (any, error) {
        // Correlation ID ni olamiz yoki yaratamiz
        correlationID := uuid.New().String()
        md, ok := metadata.FromIncomingContext(ctx)
        if ok {
            if ids := md.Get("x-correlation-id"); len(ids) > 0 {
                correlationID = ids[0]
            }
        }

        // Logging uchun kontekstga qo'shamiz
        ctx = context.WithValue(ctx, "correlation_id", correlationID)

        // Loggerga qo'shamiz
        logger := log.WithField("correlation_id", correlationID)
        ctx = context.WithValue(ctx, "logger", logger)

        return handler(ctx, req)
    }
}
\`\`\`

**Ko'plab Context Values Zanjiri:**
\`\`\`go
func ContextChain() UnaryServerInterceptor {
    return Chain(
        RequestIDInterceptor(),
        CorrelationIDInterceptor(),
        AuthContextInterceptor(),
        TenantContextInterceptor(),
        TracingContextInterceptor(),
    )
}
\`\`\`

**Handler da Foydalanish:**
\`\`\`go
func GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    // Kontekst qiymatlarini chiqarib olamiz
    requestID := ctx.Value("request_id").(string)
    userID := ctx.Value("user_id").(string)
    tenantID := ctx.Value("tenant_id").(string)
    logger := ctx.Value("logger").(*log.Entry)

    logger.Infof("GetUser called: request_id=%s user_id=%s tenant_id=%s",
        requestID, userID, tenantID)

    // Tenant-spetsifik ma'lumotlar bazasidan foydalanamiz
    tenant := ctx.Value("tenant").(*Tenant)
    db := tenant.GetDatabase()

    // Foydalanuvchini so'raymiz
    user, err := db.GetUser(ctx, req.Id)
    if err != nil {
        logger.Errorf("Failed to get user: %v", err)
        return nil, err
    }

    return user, nil
}
\`\`\`

**Context Value Yordamchilar:**
\`\`\`go
func GetRequestID(ctx context.Context) string {
    if id, ok := ctx.Value("request_id").(string); ok {
        return id
    }
    return ""
}

func GetUserID(ctx context.Context) string {
    if id, ok := ctx.Value("user_id").(string); ok {
        return id
    }
    return ""
}

func GetTenantID(ctx context.Context) string {
    if id, ok := ctx.Value("tenant_id").(string); ok {
        return id
    }
    return ""
}

func GetLogger(ctx context.Context) *log.Entry {
    if logger, ok := ctx.Value("logger").(*log.Entry); ok {
        return logger
    }
    return log.NewEntry(log.StandardLogger())
}
\`\`\`

**Type-Safe Context Kalitlari:**
\`\`\`go
type contextKey string

const (
    RequestIDKey    contextKey = "request_id"
    UserIDKey       contextKey = "user_id"
    TenantIDKey     contextKey = "tenant_id"
    TraceContextKey contextKey = "trace_context"
)

// Turlangan kontekst qiymat accessorlari
func WithRequestID(ctx context.Context, requestID string) context.Context {
    return context.WithValue(ctx, RequestIDKey, requestID)
}

func GetRequestIDTyped(ctx context.Context) (string, bool) {
    id, ok := ctx.Value(RequestIDKey).(string)
    return id, ok
}
\`\`\`

**Haqiqiy Foydalari:**
- **Toza Kod:** Funksiya imzolarini o'zgartirmasdan ma'lumotlarni uzatish
- **Request Scoping:** Qiymatlar avtomatik ravishda so'rovga bog'langan
- **Tracing:** Butun chaqiruv zanjiri orqali trace ID larni tarqatish
- **Multi-Tenancy:** Kontekst qiymatlari orqali tenantlarni ajratish

**Context Value Eng Yaxshi Amaliyotlari:**
- **Type-Safe Kalitlar:** Kolliziyalardan qochish uchun turlangan kalitlardan foydalaning
- **Ortiqcha Foydalanmang:** Kontekst ixtiyoriy parametrlarni uzatish uchun emas
- **O'zgarmaslik:** Kontekstlar o'zgarmas, WithValue yangi kontekst yaratadi
- **Qiymat Turlari:** Kichik, o'zgarmas qiymatlarni saqlang
- **Hujjatlashtirish:** Kodingiz kutayotgan qiymatlarni hujjatlantiring

**Odatiy Context Values:**
- **Request ID:** Request kuzatish uchun noyob identifikator
- **User ID:** Autentifikatsiya qilingan foydalanuvchi identifikatori
- **Tenant ID:** Multi-tenant ajratish
- **Trace Context:** Distributed tracing metadatasi
- **Logger:** Request-scoped logger
- **Database Connection:** Tenant-spetsifik ma'lumotlar bazasi

**Anti-Patternlar:**
- **Katta Obyektlar:** Kontekstda katta obyektlarni saqlamang
- **O'zgaruvchan Ma'lumotlar:** O'zgaruvchan ma'lumotlarga ko'rsatkichlarni saqlamang
- **Biznes Mantiq:** Biznes mantiq parametrlari uchun kontekstdan foydalanmang
- **String Kalitlar:** Kolliziyalardan qochish uchun turlangan kalitlardan foydalaning

ContextValueInterceptor siz request-scoped ma'lumotlar har bir funksiya parametri orqali uzatilishi kerak—ko'p so'zli va xatolarga moyil.`,
			solutionCode: `package grpcx

import (
	"context"
)

type Handler func(ctx context.Context, req any) (any, error)

type UnaryServerInterceptor func(ctx context.Context, req any, next Handler) (any, error)

func ContextValueInterceptor(key any, value any) UnaryServerInterceptor {
	return func(ctx context.Context, req any, handler Handler) (any, error) {
		if handler == nil {	// Handler nil ekanligini tekshirish
			handler = func(context.Context, any) (any, error) { return nil, nil }	// No-op handler ishlatamiz
		}
		ctx = context.WithValue(ctx, key, value)	// Kontekstga qiymat qo'shamiz
		return handler(ctx, req)	// O'zgartirilgan kontekst bilan handlerni bajaramiz
	}
}`
		}
	}
};

export default task;
