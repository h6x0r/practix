import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-grpc-timeout-interceptor',
	title: 'Timeout Interceptor',
	difficulty: 'medium',	tags: ['go', 'grpc', 'interceptors', 'context'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **TimeoutInterceptor** that wraps handler execution in a context with timeout.

**Requirements:**
1. Create function \`TimeoutInterceptor(d time.Duration) UnaryServerInterceptor\`
2. Default to 1 second if duration <= 0
3. Create child context with timeout using context.WithTimeout
4. Always defer cancel() to prevent context leak
5. Pass timeout context to handler
6. Return handler's response and error

**Example:**
\`\`\`go
interceptor := TimeoutInterceptor(2 * time.Second)

handler := func(ctx context.Context, req interface{}) (interface{}, error) {
    select {
    case <-time.After(3 * time.Second):
        return "never", nil
    case <-ctx.Done():
        return nil, ctx.Err() // context deadline exceeded after 2s
    }
}

resp, err := interceptor(ctx, "request", handler)
// err = context.DeadlineExceeded
\`\`\`

**Constraints:**
- Must use context.WithTimeout to create timeout context
- Must always defer cancel() to prevent leaks
- Must pass timeout context to handler`,
	initialCode: `package grpcx

import (
	"context"
	"time"
)

type Handler func(ctx context.Context, req interface{}) (interface{}, error)

type UnaryServerInterceptor func(ctx context.Context, req interface{}, next Handler) (interface{}, error)

// TODO: Implement TimeoutInterceptor
func TimeoutInterceptor(d time.Duration) UnaryServerInterceptor {
	// TODO: Implement
}`,
	solutionCode: `package grpcx

import (
	"context"
	"time"
)

type Handler func(ctx context.Context, req interface{}) (interface{}, error)

type UnaryServerInterceptor func(ctx context.Context, req interface{}, next Handler) (interface{}, error)

func TimeoutInterceptor(d time.Duration) UnaryServerInterceptor {
	if d <= 0 {	// Check if duration is valid
		d = time.Second	// Default to 1 second
	}
	return func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
		ctx, cancel := context.WithTimeout(ctx, d)	// Create timeout context
		defer cancel()	// Always cancel to prevent leak
		return next(ctx, req)	// Execute handler with timeout context
	}
}`,
			hint1: `Use context.WithTimeout(ctx, d) to create a timeout context from the parent context.`,
			hint2: `Always defer cancel() immediately after creating the context to prevent resource leaks.`,
			testCode: `package grpcx

import (
	"context"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Test interceptor creation with valid duration
	interceptor := TimeoutInterceptor(2 * time.Second)
	if interceptor == nil {
		t.Error("interceptor should not be nil")
	}
}

func Test2(t *testing.T) {
	// Test default 1 second when duration is 0
	interceptor := TimeoutInterceptor(0)
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		deadline, ok := ctx.Deadline()
		if !ok {
			t.Error("context should have deadline")
		}
		remaining := time.Until(deadline)
		if remaining > time.Second || remaining < 900*time.Millisecond {
			t.Errorf("timeout ~1s expected, got %v", remaining)
		}
		return nil, nil
	}
	interceptor(context.Background(), nil, handler)
}

func Test3(t *testing.T) {
	// Test default 1 second when duration is negative
	interceptor := TimeoutInterceptor(-5 * time.Second)
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		_, ok := ctx.Deadline()
		if !ok {
			t.Error("context should have deadline")
		}
		return nil, nil
	}
	interceptor(context.Background(), nil, handler)
}

func Test4(t *testing.T) {
	// Test response is passed through
	interceptor := TimeoutInterceptor(time.Second)
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return "test-response", nil
	}
	resp, err := interceptor(context.Background(), nil, handler)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if resp != "test-response" {
		t.Errorf("response = %v, want 'test-response'", resp)
	}
}

func Test5(t *testing.T) {
	// Test error is passed through
	expectedErr := context.DeadlineExceeded
	interceptor := TimeoutInterceptor(time.Second)
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return nil, expectedErr
	}
	_, err := interceptor(context.Background(), nil, handler)
	if err != expectedErr {
		t.Errorf("error = %v, want %v", err, expectedErr)
	}
}

func Test6(t *testing.T) {
	// Test context times out with short duration
	interceptor := TimeoutInterceptor(50 * time.Millisecond)
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		select {
		case <-time.After(200 * time.Millisecond):
			return "completed", nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	_, err := interceptor(context.Background(), nil, handler)
	if err != context.DeadlineExceeded {
		t.Errorf("error = %v, want DeadlineExceeded", err)
	}
}

func Test7(t *testing.T) {
	// Test request is passed to handler
	var receivedReq any
	interceptor := TimeoutInterceptor(time.Second)
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		receivedReq = req
		return nil, nil
	}
	interceptor(context.Background(), "test-request", handler)
	if receivedReq != "test-request" {
		t.Errorf("request = %v, want 'test-request'", receivedReq)
	}
}

func Test8(t *testing.T) {
	// Test parent context values are preserved
	interceptor := TimeoutInterceptor(time.Second)
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		if ctx.Value("key") != "value" {
			t.Error("context value not preserved")
		}
		return nil, nil
	}
	ctx := context.WithValue(context.Background(), "key", "value")
	interceptor(ctx, nil, handler)
}

func Test9(t *testing.T) {
	// Test custom timeout duration
	interceptor := TimeoutInterceptor(500 * time.Millisecond)
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		deadline, ok := ctx.Deadline()
		if !ok {
			t.Error("context should have deadline")
		}
		remaining := time.Until(deadline)
		if remaining > 500*time.Millisecond || remaining < 400*time.Millisecond {
			t.Errorf("timeout ~500ms expected, got %v", remaining)
		}
		return nil, nil
	}
	interceptor(context.Background(), nil, handler)
}

func Test10(t *testing.T) {
	// Test multiple calls are independent
	interceptor := TimeoutInterceptor(time.Second)
	callCount := 0
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		callCount++
		return callCount, nil
	}
	resp1, _ := interceptor(context.Background(), nil, handler)
	resp2, _ := interceptor(context.Background(), nil, handler)
	if resp1 != 1 || resp2 != 2 {
		t.Errorf("calls should be independent, got %v and %v", resp1, resp2)
	}
}`,
			whyItMatters: `TimeoutInterceptor prevents long-running RPCs from blocking resources indefinitely, ensuring responsive microservices and preventing cascading failures.

**Why RPC Timeouts:**
- **Resource Protection:** Prevent handlers from running forever
- **Cascade Prevention:** Stop timeouts from propagating through services
- **SLA Compliance:** Enforce maximum RPC durations
- **Client Experience:** Fast failures vs hanging indefinitely

**Production Pattern:**
\`\`\`go
// Different timeouts for different methods
func MethodTimeoutInterceptor() UnaryServerInterceptor {
    timeouts := map[string]time.Duration{
        "/api.UserService/GetUser":    1 * time.Second,   // Fast reads
        "/api.UserService/CreateUser": 5 * time.Second,   // Slower writes
        "/api.ReportService/Generate": 30 * time.Second,  // Long operations
    }

    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        timeout, ok := timeouts[info.FullMethod]
        if !ok {
            timeout = 10 * time.Second // Default timeout
        }

        ctx, cancel := context.WithTimeout(ctx, timeout)
        defer cancel()

        return handler(ctx, req)
    }
}

// Dynamic timeout from client metadata
func ClientTimeoutInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        // Check if client specified timeout in metadata
        md, ok := metadata.FromIncomingContext(ctx)
        if ok {
            if timeoutStr, exists := md["timeout"]; exists && len(timeoutStr) > 0 {
                if clientTimeout, err := time.ParseDuration(timeoutStr[0]); err == nil {
                    // Use client timeout with maximum limit
                    maxTimeout := 60 * time.Second
                    if clientTimeout > maxTimeout {
                        clientTimeout = maxTimeout
                    }

                    ctx, cancel := context.WithTimeout(ctx, clientTimeout)
                    defer cancel()
                    return handler(ctx, req)
                }
            }
        }

        // Fall back to default timeout
        ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
        defer cancel()
        return handler(ctx, req)
    }
}

// Timeout with graceful degradation
func GracefulTimeoutInterceptor(requestTimeout, cleanupTimeout time.Duration) UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        // Request context with timeout
        reqCtx, reqCancel := context.WithTimeout(ctx, requestTimeout)
        defer reqCancel()

        // Cleanup context with additional time
        cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), cleanupTimeout)
        defer cleanupCancel()

        type result struct {
            resp any
            err  error
        }

        done := make(chan result, 1)
        go func() {
            resp, err := handler(reqCtx, req)
            done <- result{resp, err}
        }()

        select {
        case res := <-done:
            return res.resp, res.err
        case <-reqCtx.Done():
            // Request timeout, allow cleanup
            log.Printf("[Timeout] %s - allowing %v for cleanup", info.FullMethod, cleanupTimeout)
            select {
            case res := <-done:
                log.Printf("[Cleanup] completed")
                return res.resp, res.err
            case <-cleanupCtx.Done():
                log.Printf("[Cleanup] timeout exceeded")
                return nil, status.Error(codes.DeadlineExceeded, "request timeout")
            }
        }
    }
}

// Timeout with metrics
func TimeoutWithMetrics(d time.Duration) UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        ctx, cancel := context.WithTimeout(ctx, d)
        defer cancel()

        resp, err := handler(ctx, req)

        // Track timeout errors
        if err != nil && ctx.Err() == context.DeadlineExceeded {
            metrics.IncrementCounter("grpc_timeout_errors", map[string]string{
                "method": info.FullMethod,
            })
            log.Printf("[Timeout] method=%s duration=%v", info.FullMethod, d)
        }

        return resp, err
    }
}

// Adaptive timeout based on load
func AdaptiveTimeoutInterceptor(baseTimeout time.Duration) UnaryServerInterceptor {
    var (
        currentLoad     float64
        loadMutex       sync.RWMutex
        timeoutMultiplier = 1.0
    )

    // Monitor system load and adjust timeout
    go func() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()

        for range ticker.C {
            load := getSystemLoad()
            loadMutex.Lock()
            currentLoad = load

            // Increase timeout under high load
            if load > 0.8 {
                timeoutMultiplier = 2.0
            } else if load > 0.5 {
                timeoutMultiplier = 1.5
            } else {
                timeoutMultiplier = 1.0
            }
            loadMutex.Unlock()
        }
    }()

    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        loadMutex.RLock()
        multiplier := timeoutMultiplier
        loadMutex.RUnlock()

        timeout := time.Duration(float64(baseTimeout) * multiplier)

        ctx, cancel := context.WithTimeout(ctx, timeout)
        defer cancel()

        return handler(ctx, req)
    }
}

// Timeout chain (request timeout + database timeout)
func TimeoutChain(requestTimeout, dbTimeout time.Duration) UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        // Request-level timeout
        ctx, cancel := context.WithTimeout(ctx, requestTimeout)
        defer cancel()

        // Add database timeout to context
        ctx = context.WithValue(ctx, "db_timeout", dbTimeout)

        return handler(ctx, req)
    }
}

// Usage in handler
func GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    // Use database timeout from context
    dbTimeout := ctx.Value("db_timeout").(time.Duration)
    dbCtx, cancel := context.WithTimeout(ctx, dbTimeout)
    defer cancel()

    var user User
    err := db.QueryRowContext(dbCtx, "SELECT * FROM users WHERE id = $1", req.Id).Scan(&user)
    if err != nil {
        return nil, err
    }

    return &user, nil
}
\`\`\`

**Real-World Benefits:**
- **Server Protection:** Prevent resource exhaustion from slow operations
- **Cascade Prevention:** Stop timeouts from propagating to dependencies
- **SLA Compliance:** Enforce maximum response times
- **Cost Control:** Limit expensive operation durations

**Timeout Best Practices:**
- **Always Cancel:** Defer cancel() to prevent context leaks
- **Check Context:** Handlers should check ctx.Done() regularly
- **Propagate Context:** Pass context to all downstream operations
- **Set Realistic Timeouts:** Too short = false failures, too long = resource waste

**Common Timeout Values:**
- **Simple Queries:** 1-2 seconds
- **Complex Queries:** 5-10 seconds
- **External APIs:** 10-30 seconds
- **Heavy Processing:** 30-60 seconds
- **Long Operations:** 60-300 seconds

**Context Cancellation:**
- **Deadline Exceeded:** Timeout occurred
- **Canceled:** Explicit cancellation
- **gRPC:** Honors context cancellation automatically
- **Database Drivers:** Cancel queries when context times out

Without TimeoutInterceptor, slow RPCs accumulate, exhausting connections and memory, eventually crashing the service.`,	order: 1,
	translations: {
		ru: {
			title: 'Установка таймаута для gRPC вызовов',
			description: `Реализуйте **TimeoutInterceptor**, который оборачивает выполнение handler в контекст с тайм-аутом.

**Требования:**
1. Создайте функцию \`TimeoutInterceptor(d time.Duration) UnaryServerInterceptor\`
2. По умолчанию 1 секунда если duration <= 0
3. Создайте дочерний контекст с тайм-аутом используя context.WithTimeout
4. Всегда defer cancel() для предотвращения утечки контекста
5. Передайте timeout контекст в handler
6. Верните ответ и ошибку handler

**Пример:**
\`\`\`go
interceptor := TimeoutInterceptor(2 * time.Second)

handler := func(ctx context.Context, req interface{}) (interface{}, error) {
    select {
    case <-time.After(3 * time.Second):
        return "never", nil
    case <-ctx.Done():
        return nil, ctx.Err() // context deadline exceeded после 2s
    }
}

resp, err := interceptor(ctx, "request", handler)
// err = context.DeadlineExceeded
\`\`\`

**Ограничения:**
- Должен использовать context.WithTimeout для создания timeout контекста
- Должен всегда defer cancel() для предотвращения утечек
- Должен передавать timeout контекст в handler`,
			hint1: `Используйте context.WithTimeout(ctx, d) для создания timeout контекста из родительского контекста.`,
			hint2: `Всегда defer cancel() сразу после создания контекста для предотвращения утечек ресурсов.`,
			whyItMatters: `TimeoutInterceptor предотвращает долго выполняющиеся RPC от бесконечной блокировки ресурсов, обеспечивая отзывчивые микросервисы и предотвращая каскадные сбои.

**Зачем нужны RPC Timeouts:**
- **Защита ресурсов:** Предотвращение бесконечного выполнения handlers
- **Предотвращение каскадов:** Остановка распространения таймаутов через сервисы
- **Соответствие SLA:** Обеспечение максимальной длительности RPC
- **Опыт клиента:** Быстрые отказы вместо бесконечного зависания

**Production Паттерн:**
\`\`\`go
// Разные таймауты для разных методов
func MethodTimeoutInterceptor() UnaryServerInterceptor {
    timeouts := map[string]time.Duration{
        "/api.UserService/GetUser":    1 * time.Second,   // Быстрые чтения
        "/api.UserService/CreateUser": 5 * time.Second,   // Медленные записи
        "/api.ReportService/Generate": 30 * time.Second,  // Долгие операции
    }

    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        timeout, ok := timeouts[info.FullMethod]
        if !ok {
            timeout = 10 * time.Second // Таймаут по умолчанию
        }

        ctx, cancel := context.WithTimeout(ctx, timeout)
        defer cancel()

        return handler(ctx, req)
    }
}

// Динамический таймаут из метаданных клиента
func ClientTimeoutInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        // Проверяем указан ли клиентом таймаут в метаданных
        md, ok := metadata.FromIncomingContext(ctx)
        if ok {
            if timeoutStr, exists := md["timeout"]; exists && len(timeoutStr) > 0 {
                if clientTimeout, err := time.ParseDuration(timeoutStr[0]); err == nil {
                    // Используем клиентский таймаут с максимальным лимитом
                    maxTimeout := 60 * time.Second
                    if clientTimeout > maxTimeout {
                        clientTimeout = maxTimeout
                    }

                    ctx, cancel := context.WithTimeout(ctx, clientTimeout)
                    defer cancel()
                    return handler(ctx, req)
                }
            }
        }

        // Откат на таймаут по умолчанию
        ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
        defer cancel()
        return handler(ctx, req)
    }
}

// Таймаут с graceful degradation
func GracefulTimeoutInterceptor(requestTimeout, cleanupTimeout time.Duration) UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        // Контекст запроса с таймаутом
        reqCtx, reqCancel := context.WithTimeout(ctx, requestTimeout)
        defer reqCancel()

        // Контекст очистки с дополнительным временем
        cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), cleanupTimeout)
        defer cleanupCancel()

        type result struct {
            resp any
            err  error
        }

        done := make(chan result, 1)
        go func() {
            resp, err := handler(reqCtx, req)
            done <- result{resp, err}
        }()

        select {
        case res := <-done:
            return res.resp, res.err
        case <-reqCtx.Done():
            // Таймаут запроса, разрешаем очистку
            log.Printf("[Timeout] %s - разрешаем %v для очистки", info.FullMethod, cleanupTimeout)
            select {
            case res := <-done:
                log.Printf("[Cleanup] завершено")
                return res.resp, res.err
            case <-cleanupCtx.Done():
                log.Printf("[Cleanup] таймаут превышен")
                return nil, status.Error(codes.DeadlineExceeded, "request timeout")
            }
        }
    }
}

// Таймаут с метриками
func TimeoutWithMetrics(d time.Duration) UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        ctx, cancel := context.WithTimeout(ctx, d)
        defer cancel()

        resp, err := handler(ctx, req)

        // Отслеживаем ошибки таймаута
        if err != nil && ctx.Err() == context.DeadlineExceeded {
            metrics.IncrementCounter("grpc_timeout_errors", map[string]string{
                "method": info.FullMethod,
            })
            log.Printf("[Timeout] method=%s duration=%v", info.FullMethod, d)
        }

        return resp, err
    }
}

// Адаптивный таймаут на основе нагрузки
func AdaptiveTimeoutInterceptor(baseTimeout time.Duration) UnaryServerInterceptor {
    var (
        currentLoad     float64
        loadMutex       sync.RWMutex
        timeoutMultiplier = 1.0
    )

    // Мониторим системную нагрузку и регулируем таймаут
    go func() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()

        for range ticker.C {
            load := getSystemLoad()
            loadMutex.Lock()
            currentLoad = load

            // Увеличиваем таймаут при высокой нагрузке
            if load > 0.8 {
                timeoutMultiplier = 2.0
            } else if load > 0.5 {
                timeoutMultiplier = 1.5
            } else {
                timeoutMultiplier = 1.0
            }
            loadMutex.Unlock()
        }
    }()

    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        loadMutex.RLock()
        multiplier := timeoutMultiplier
        loadMutex.RUnlock()

        timeout := time.Duration(float64(baseTimeout) * multiplier)

        ctx, cancel := context.WithTimeout(ctx, timeout)
        defer cancel()

        return handler(ctx, req)
    }
}

// Цепочка таймаутов (request timeout + database timeout)
func TimeoutChain(requestTimeout, dbTimeout time.Duration) UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        // Таймаут уровня запроса
        ctx, cancel := context.WithTimeout(ctx, requestTimeout)
        defer cancel()

        // Добавляем таймаут базы данных в контекст
        ctx = context.WithValue(ctx, "db_timeout", dbTimeout)

        return handler(ctx, req)
    }
}

// Использование в handler
func GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    // Используем таймаут базы данных из контекста
    dbTimeout := ctx.Value("db_timeout").(time.Duration)
    dbCtx, cancel := context.WithTimeout(ctx, dbTimeout)
    defer cancel()

    var user User
    err := db.QueryRowContext(dbCtx, "SELECT * FROM users WHERE id = $1", req.Id).Scan(&user)
    if err != nil {
        return nil, err
    }

    return &user, nil
}
\`\`\`

**Реальные преимущества:**
- **Защита сервера:** Предотвращение истощения ресурсов от медленных операций
- **Предотвращение каскадов:** Остановка распространения таймаутов на зависимости
- **Соответствие SLA:** Обеспечение максимального времени ответа
- **Контроль затрат:** Ограничение длительности дорогих операций

**Лучшие практики таймаутов:**
- **Всегда Cancel:** Используйте defer cancel() для предотвращения утечек контекста
- **Проверка контекста:** Handlers должны регулярно проверять ctx.Done()
- **Распространение контекста:** Передавайте контекст всем downstream операциям
- **Реалистичные таймауты:** Слишком короткие = ложные отказы, слишком длинные = потеря ресурсов

**Типичные значения таймаутов:**
- **Простые запросы:** 1-2 секунды
- **Сложные запросы:** 5-10 секунд
- **Внешние API:** 10-30 секунд
- **Тяжёлая обработка:** 30-60 секунд
- **Долгие операции:** 60-300 секунд

**Отмена контекста:**
- **Deadline Exceeded:** Таймаут произошёл
- **Canceled:** Явная отмена
- **gRPC:** Автоматически учитывает отмену контекста
- **Database Drivers:** Отменяют запросы при таймауте контекста

Без TimeoutInterceptor медленные RPC накапливаются, исчерпывая соединения и память, в конечном итоге приводя к краху сервиса.`,
			solutionCode: `package grpcx

import (
	"context"
	"time"
)

type Handler func(ctx context.Context, req interface{}) (interface{}, error)

type UnaryServerInterceptor func(ctx context.Context, req interface{}, next Handler) (interface{}, error)

func TimeoutInterceptor(d time.Duration) UnaryServerInterceptor {
	if d <= 0 {	// Проверка валидности duration
		d = time.Second	// По умолчанию 1 секунда
	}
	return func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
		ctx, cancel := context.WithTimeout(ctx, d)	// Создаём timeout контекст
		defer cancel()	// Всегда отменяем для предотвращения утечки
		return next(ctx, req)	// Выполняем handler с timeout контекстом
	}
}`
		},
		uz: {
			title: 'gRPC chaqiruvlari uchun timeout o\'rnatish',
			description: `Handler bajarilishini timeout bilan kontekstga o'raydigan **TimeoutInterceptor** ni amalga oshiring.

**Talablar:**
1. \`TimeoutInterceptor(d time.Duration) UnaryServerInterceptor\` funksiyasini yarating
2. Agar duration <= 0 bo'lsa, standart 1 soniya
3. context.WithTimeout dan foydalanib timeout bilan child kontekstni yarating
4. Kontekst oqishining oldini olish uchun har doim defer cancel() qiling
5. Timeout kontekstini handlerga o'tkazing
6. Handler javob va xatosini qaytaring

**Misol:**
\`\`\`go
interceptor := TimeoutInterceptor(2 * time.Second)

handler := func(ctx context.Context, req interface{}) (interface{}, error) {
    select {
    case <-time.After(3 * time.Second):
        return "never", nil
    case <-ctx.Done():
        return nil, ctx.Err() // 2s dan keyin context deadline exceeded
    }
}

resp, err := interceptor(ctx, "request", handler)
// err = context.DeadlineExceeded
\`\`\`

**Cheklovlar:**
- Timeout kontekstni yaratish uchun context.WithTimeout dan foydalanishi kerak
- Oqishlarning oldini olish uchun har doim defer cancel() qilishi kerak
- Timeout kontekstini handlerga o'tkazishi kerak`,
			hint1: `Ota-kontekstdan timeout kontekstni yaratish uchun context.WithTimeout(ctx, d) dan foydalaning.`,
			hint2: `Resurs oqishining oldini olish uchun kontekst yaratilgandan keyin darhol defer cancel() qiling.`,
			whyItMatters: `TimeoutInterceptor uzoq davom etadigan RPC larning resurslarni abadiy blokirovka qilishining oldini oladi, sezgir mikroservislarni ta'minlaydi va kaskadli nosozliklarning oldini oladi.

**Nega RPC Timeoutlar kerak:**
- **Resurslarni himoya qilish:** Handlerlarning abadiy bajarilishining oldini olish
- **Kaskadni oldini olish:** Timeoutlarning servislar orqali tarqalishini to'xtatish
- **SLA muvofiqlik:** Maksimal RPC davomiyligini ta'minlash
- **Mijoz tajribasi:** Abadiy osilib qolish o'rniga tez muvaffaqiyatsizliklar

**Production Paterni:**
\`\`\`go
// Turli metodlar uchun turli timeoutlar
func MethodTimeoutInterceptor() UnaryServerInterceptor {
    timeouts := map[string]time.Duration{
        "/api.UserService/GetUser":    1 * time.Second,   // Tez o'qishlar
        "/api.UserService/CreateUser": 5 * time.Second,   // Sekin yozishlar
        "/api.ReportService/Generate": 30 * time.Second,  // Uzoq operatsiyalar
    }

    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        timeout, ok := timeouts[info.FullMethod]
        if !ok {
            timeout = 10 * time.Second // Standart timeout
        }

        ctx, cancel := context.WithTimeout(ctx, timeout)
        defer cancel()

        return handler(ctx, req)
    }
}

// Mijoz metadatasidan dinamik timeout
func ClientTimeoutInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        // Mijoz metadatada timeout ko'rsatganligini tekshiramiz
        md, ok := metadata.FromIncomingContext(ctx)
        if ok {
            if timeoutStr, exists := md["timeout"]; exists && len(timeoutStr) > 0 {
                if clientTimeout, err := time.ParseDuration(timeoutStr[0]); err == nil {
                    // Maksimal limit bilan mijoz timeout dan foydalanamiz
                    maxTimeout := 60 * time.Second
                    if clientTimeout > maxTimeout {
                        clientTimeout = maxTimeout
                    }

                    ctx, cancel := context.WithTimeout(ctx, clientTimeout)
                    defer cancel()
                    return handler(ctx, req)
                }
            }
        }

        // Standart timeout ga qaytish
        ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
        defer cancel()
        return handler(ctx, req)
    }
}

// Graceful degradation bilan timeout
func GracefulTimeoutInterceptor(requestTimeout, cleanupTimeout time.Duration) UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        // So'rov konteksti timeout bilan
        reqCtx, reqCancel := context.WithTimeout(ctx, requestTimeout)
        defer reqCancel()

        // Qo'shimcha vaqt bilan tozalash konteksti
        cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), cleanupTimeout)
        defer cleanupCancel()

        type result struct {
            resp any
            err  error
        }

        done := make(chan result, 1)
        go func() {
            resp, err := handler(reqCtx, req)
            done <- result{resp, err}
        }()

        select {
        case res := <-done:
            return res.resp, res.err
        case <-reqCtx.Done():
            // So'rov timeout, tozalashga ruxsat beramiz
            log.Printf("[Timeout] %s - tozalash uchun %v ga ruxsat beramiz", info.FullMethod, cleanupTimeout)
            select {
            case res := <-done:
                log.Printf("[Cleanup] tugadi")
                return res.resp, res.err
            case <-cleanupCtx.Done():
                log.Printf("[Cleanup] timeout oshib ketdi")
                return nil, status.Error(codes.DeadlineExceeded, "request timeout")
            }
        }
    }
}

// Metrikalar bilan timeout
func TimeoutWithMetrics(d time.Duration) UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        ctx, cancel := context.WithTimeout(ctx, d)
        defer cancel()

        resp, err := handler(ctx, req)

        // Timeout xatolarini kuzatamiz
        if err != nil && ctx.Err() == context.DeadlineExceeded {
            metrics.IncrementCounter("grpc_timeout_errors", map[string]string{
                "method": info.FullMethod,
            })
            log.Printf("[Timeout] method=%s duration=%v", info.FullMethod, d)
        }

        return resp, err
    }
}

// Yuklanish asosida moslashuvchan timeout
func AdaptiveTimeoutInterceptor(baseTimeout time.Duration) UnaryServerInterceptor {
    var (
        currentLoad     float64
        loadMutex       sync.RWMutex
        timeoutMultiplier = 1.0
    )

    // Tizim yuklanishini monitoring qilamiz va timeout ni sozlaymiz
    go func() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()

        for range ticker.C {
            load := getSystemLoad()
            loadMutex.Lock()
            currentLoad = load

            // Yuqori yuklanishda timeout ni oshiramiz
            if load > 0.8 {
                timeoutMultiplier = 2.0
            } else if load > 0.5 {
                timeoutMultiplier = 1.5
            } else {
                timeoutMultiplier = 1.0
            }
            loadMutex.Unlock()
        }
    }()

    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        loadMutex.RLock()
        multiplier := timeoutMultiplier
        loadMutex.RUnlock()

        timeout := time.Duration(float64(baseTimeout) * multiplier)

        ctx, cancel := context.WithTimeout(ctx, timeout)
        defer cancel()

        return handler(ctx, req)
    }
}

// Timeout zanjiri (request timeout + database timeout)
func TimeoutChain(requestTimeout, dbTimeout time.Duration) UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler Handler) (interface{}, error) {
        // So'rov darajasi timeout
        ctx, cancel := context.WithTimeout(ctx, requestTimeout)
        defer cancel()

        // Ma'lumotlar bazasi timeout ni kontekstga qo'shamiz
        ctx = context.WithValue(ctx, "db_timeout", dbTimeout)

        return handler(ctx, req)
    }
}

// Handler da foydalanish
func GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    // Kontekstdan database timeout ni ishlatamiz
    dbTimeout := ctx.Value("db_timeout").(time.Duration)
    dbCtx, cancel := context.WithTimeout(ctx, dbTimeout)
    defer cancel()

    var user User
    err := db.QueryRowContext(dbCtx, "SELECT * FROM users WHERE id = $1", req.Id).Scan(&user)
    if err != nil {
        return nil, err
    }

    return &user, nil
}
\`\`\`

**Haqiqiy foydalari:**
- **Server himoyasi:** Sekin operatsiyalardan resurs tugashining oldini olish
- **Kaskadni oldini olish:** Bog'liqlikka timeoutlar tarqalishini to'xtatish
- **SLA muvofiqlik:** Maksimal javob vaqtini ta'minlash
- **Xarajatlar nazorati:** Qimmat operatsiya davomiyligini cheklash

**Timeout eng yaxshi amaliyotlari:**
- **Har doim Cancel:** Kontekst oqishini oldini olish uchun defer cancel() ishlating
- **Kontekstni tekshirish:** Handlerlar muntazam ravishda ctx.Done() tekshirishi kerak
- **Kontekstni tarqatish:** Barcha downstream operatsiyalarga kontekstni o'tkazing
- **Real timeoutlar:** Juda qisqa = yolg'on muvaffaqiyatsizliklar, juda uzun = resurslar isrofi

**Odatiy timeout qiymatlari:**
- **Oddiy so'rovlar:** 1-2 soniya
- **Murakkab so'rovlar:** 5-10 soniya
- **Tashqi API lar:** 10-30 soniya
- **Og'ir qayta ishlash:** 30-60 soniya
- **Uzoq operatsiyalar:** 60-300 soniya

**Kontekstni bekor qilish:**
- **Deadline Exceeded:** Timeout yuz berdi
- **Canceled:** Aniq bekor qilish
- **gRPC:** Avtomatik ravishda kontekst bekor qilishni hisobga oladi
- **Database Drivers:** Kontekst timeoutida so'rovlarni bekor qiladi

TimeoutInterceptor siz sekin RPC lar to'planadi, ulanishlar va xotirani tugatadi va oxir-oqibat xizmat ishdan chiqadi.`,
			solutionCode: `package grpcx

import (
	"context"
	"time"
)

type Handler func(ctx context.Context, req interface{}) (interface{}, error)

type UnaryServerInterceptor func(ctx context.Context, req interface{}, next Handler) (interface{}, error)

func TimeoutInterceptor(d time.Duration) UnaryServerInterceptor {
	if d <= 0 {	// Duration haqiqiyligini tekshirish
		d = time.Second	// Standart 1 soniya
	}
	return func(ctx context.Context, req interface{}, next Handler) (interface{}, error) {
		ctx, cancel := context.WithTimeout(ctx, d)	// Timeout kontekstini yaratamiz
		defer cancel()	// Oqishni oldini olish uchun har doim bekor qilamiz
		return next(ctx, req)	// Handlerni timeout konteksti bilan bajaramiz
	}
}`
		}
	}
};

export default task;
