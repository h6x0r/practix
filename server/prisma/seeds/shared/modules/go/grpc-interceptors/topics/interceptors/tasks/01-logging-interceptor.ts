import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-grpc-logging-interceptor',
	title: 'Logging Interceptor',
	difficulty: 'easy',	tags: ['go', 'grpc', 'interceptors', 'logging'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **LoggingInterceptor** that logs the start and finish of gRPC handler execution using a provided logger function.

**Requirements:**
1. Create function \`LoggingInterceptor(logger func(context.Context, string)) UnaryServerInterceptor\`
2. Handle nil logger (use no-op function)
3. Call logger with "start" before executing handler
4. Call handler (next)
5. Call logger with "finish" after handler completes
6. Return handler's response and error

**Example:**
\`\`\`go
logs := []string{}
logger := func(ctx context.Context, msg string) {
    logs = append(logs, msg)
}

interceptor := LoggingInterceptor(logger)
handler := func(ctx context.Context, req any) (any, error) {
    return "response", nil
}

resp, err := interceptor(ctx, "request", handler)
// logs = ["start", "finish"]
\`\`\`

**Constraints:**
- Must call logger before and after handler execution
- Must handle nil logger gracefully`,
	initialCode: `package grpcx

import (
	"context"
)

// Handler simulates grpc.UnaryHandler
type Handler func(ctx context.Context, req any) (any, error)

// UnaryServerInterceptor wraps a handler
type UnaryServerInterceptor func(ctx context.Context, req any, next Handler) (any, error)

// TODO: Implement LoggingInterceptor
func LoggingInterceptor(logger func(context.Context, string)) UnaryServerInterceptor {
	// TODO: Implement
}`,
	solutionCode: `package grpcx

import (
	"context"
)

type Handler func(ctx context.Context, req any) (any, error)

type UnaryServerInterceptor func(ctx context.Context, req any, next Handler) (any, error)

func LoggingInterceptor(logger func(context.Context, string)) UnaryServerInterceptor {
	if logger == nil {	// Check if logger is nil
		logger = func(context.Context, string) {}	// Use no-op logger
	}
	return func(ctx context.Context, req any, next Handler) (any, error) {
		logger(ctx, "start")	// Log start of execution
		resp, err := next(ctx, req)	// Execute handler
		logger(ctx, "finish")	// Log finish of execution
		return resp, err	// Return handler result
	}
}`,
			hint1: `Return a function that calls logger("start"), then next(), then logger("finish").`,
			hint2: `Check if logger is nil and replace with a no-op function to avoid nil pointer dereference.`,
			testCode: `package grpcx

import (
	"context"
	"errors"
	"testing"
)

func Test1(t *testing.T) {
	// Test interceptor with valid logger
	logs := []string{}
	logger := func(ctx context.Context, msg string) {
		logs = append(logs, msg)
	}
	interceptor := LoggingInterceptor(logger)
	if interceptor == nil {
		t.Error("interceptor should not be nil")
	}
}

func Test2(t *testing.T) {
	// Test interceptor with nil logger (should not panic)
	interceptor := LoggingInterceptor(nil)
	if interceptor == nil {
		t.Error("interceptor with nil logger should not be nil")
	}
}

func Test3(t *testing.T) {
	// Test logs "start" and "finish" in order
	logs := []string{}
	logger := func(ctx context.Context, msg string) {
		logs = append(logs, msg)
	}
	interceptor := LoggingInterceptor(logger)
	handler := func(ctx context.Context, req any) (any, error) {
		return "response", nil
	}
	interceptor(context.Background(), "request", handler)
	if len(logs) != 2 {
		t.Errorf("expected 2 logs, got %d", len(logs))
	}
	if logs[0] != "start" {
		t.Errorf("first log = %q, want 'start'", logs[0])
	}
	if logs[1] != "finish" {
		t.Errorf("second log = %q, want 'finish'", logs[1])
	}
}

func Test4(t *testing.T) {
	// Test handler response is returned
	interceptor := LoggingInterceptor(nil)
	handler := func(ctx context.Context, req any) (any, error) {
		return "test-response", nil
	}
	resp, err := interceptor(context.Background(), "request", handler)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if resp != "test-response" {
		t.Errorf("response = %v, want 'test-response'", resp)
	}
}

func Test5(t *testing.T) {
	// Test handler error is returned
	expectedErr := errors.New("handler error")
	interceptor := LoggingInterceptor(nil)
	handler := func(ctx context.Context, req any) (any, error) {
		return nil, expectedErr
	}
	resp, err := interceptor(context.Background(), "request", handler)
	if err != expectedErr {
		t.Errorf("error = %v, want %v", err, expectedErr)
	}
	if resp != nil {
		t.Errorf("response = %v, want nil", resp)
	}
}

func Test6(t *testing.T) {
	// Test logs even when handler returns error
	logs := []string{}
	logger := func(ctx context.Context, msg string) {
		logs = append(logs, msg)
	}
	interceptor := LoggingInterceptor(logger)
	handler := func(ctx context.Context, req any) (any, error) {
		return nil, errors.New("error")
	}
	interceptor(context.Background(), "request", handler)
	if len(logs) != 2 {
		t.Errorf("expected 2 logs even on error, got %d", len(logs))
	}
}

func Test7(t *testing.T) {
	// Test context is passed to logger
	var receivedCtx context.Context
	logger := func(ctx context.Context, msg string) {
		receivedCtx = ctx
	}
	interceptor := LoggingInterceptor(logger)
	handler := func(ctx context.Context, req any) (any, error) {
		return nil, nil
	}
	ctx := context.WithValue(context.Background(), "key", "value")
	interceptor(ctx, "request", handler)
	if receivedCtx.Value("key") != "value" {
		t.Error("context was not passed to logger")
	}
}

func Test8(t *testing.T) {
	// Test request is passed to handler
	var receivedReq any
	interceptor := LoggingInterceptor(nil)
	handler := func(ctx context.Context, req any) (any, error) {
		receivedReq = req
		return nil, nil
	}
	interceptor(context.Background(), "test-request", handler)
	if receivedReq != "test-request" {
		t.Errorf("request = %v, want 'test-request'", receivedReq)
	}
}

func Test9(t *testing.T) {
	// Test handler is called exactly once
	callCount := 0
	interceptor := LoggingInterceptor(nil)
	handler := func(ctx context.Context, req any) (any, error) {
		callCount++
		return nil, nil
	}
	interceptor(context.Background(), "request", handler)
	if callCount != 1 {
		t.Errorf("handler called %d times, want 1", callCount)
	}
}

func Test10(t *testing.T) {
	// Test multiple interceptor calls are independent
	logs := []string{}
	logger := func(ctx context.Context, msg string) {
		logs = append(logs, msg)
	}
	interceptor := LoggingInterceptor(logger)
	handler := func(ctx context.Context, req any) (any, error) {
		return nil, nil
	}
	interceptor(context.Background(), "req1", handler)
	interceptor(context.Background(), "req2", handler)
	if len(logs) != 4 {
		t.Errorf("expected 4 logs for 2 calls, got %d", len(logs))
	}
}`,
			whyItMatters: `LoggingInterceptor enables observability in gRPC services by tracking all RPC calls, essential for debugging and monitoring microservices.

**Why Logging Interceptors:**
- **Request Tracking:** Log every RPC call for audit trails
- **Performance Monitoring:** Measure handler execution time
- **Debugging:** See exact call flow in distributed systems
- **Compliance:** Record all service access for regulations

**Production Pattern:**
\`\`\`go
// Enhanced logging with method name and timing
func StructuredLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        start := time.Now()

        // Log request start with metadata
        method := info.FullMethod
        log.Printf("[gRPC] START method=%s", method)

        // Execute handler
        resp, err := next(ctx, req)

        // Log completion with status
        duration := time.Since(start)
        status := "OK"
        if err != nil {
            status = "ERROR"
        }

        log.Printf("[gRPC] FINISH method=%s status=%s duration=%v", method, status, duration)

        return resp, err
    }
}

// JSON structured logging
func JSONLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        start := time.Now()

        logEntry := map[string]interface{}{
            "timestamp": time.Now().UTC().Format(time.RFC3339),
            "method":    info.FullMethod,
            "type":      "start",
        }
        json.NewEncoder(os.Stdout).Encode(logEntry)

        resp, err := next(ctx, req)

        logEntry = map[string]interface{}{
            "timestamp":   time.Now().UTC().Format(time.RFC3339),
            "method":      info.FullMethod,
            "type":        "finish",
            "duration_ms": time.Since(start).Milliseconds(),
            "success":     err == nil,
        }

        if err != nil {
            logEntry["error"] = err.Error()
        }

        json.NewEncoder(os.Stdout).Encode(logEntry)

        return resp, err
    }
}

// Logging with request/response payloads (be careful with sensitive data!)
func VerboseLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        // Log request
        log.Printf("[gRPC] → %s request: %+v", info.FullMethod, req)

        resp, err := next(ctx, req)

        // Log response
        if err != nil {
            log.Printf("[gRPC] ← %s error: %v", info.FullMethod, err)
        } else {
            log.Printf("[gRPC] ← %s response: %+v", info.FullMethod, resp)
        }

        return resp, err
    }
}

// Conditional logging (only log errors)
func ErrorLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        resp, err := next(ctx, req)

        if err != nil {
            log.Printf("[gRPC ERROR] method=%s error=%v", info.FullMethod, err)

            // Send to error tracking system
            sentry.CaptureException(err)
        }

        return resp, err
    }
}

// Logging with metrics integration
func MetricsLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        start := time.Now()

        resp, err := next(ctx, req)

        duration := time.Since(start)

        // Record metrics
        metrics.RecordRPCDuration(info.FullMethod, duration)

        if err != nil {
            metrics.IncrementRPCErrors(info.FullMethod)
            log.Printf("[gRPC] method=%s error=%v duration=%v", info.FullMethod, err, duration)
        } else {
            metrics.IncrementRPCSuccess(info.FullMethod)
            log.Printf("[gRPC] method=%s success duration=%v", info.FullMethod, duration)
        }

        return resp, err
    }
}

// Sampling logging (log only 10% of requests)
func SampledLoggingInterceptor(sampleRate float64) UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        shouldLog := rand.Float64() < sampleRate

        if shouldLog {
            start := time.Now()
            log.Printf("[gRPC SAMPLE] START %s", info.FullMethod)

            resp, err := next(ctx, req)

            duration := time.Since(start)
            log.Printf("[gRPC SAMPLE] FINISH %s duration=%v error=%v", info.FullMethod, duration, err != nil)

            return resp, err
        }

        return next(ctx, req)
    }
}
\`\`\`

**Real-World Benefits:**
- **Distributed Tracing:** Track requests across multiple services
- **Performance Analysis:** Identify slow RPCs
- **Error Investigation:** See which methods fail most
- **Capacity Planning:** Analyze traffic patterns

**Logging Best Practices:**
- **Structured Logs:** Use JSON for machine parsing
- **Include Context:** Method name, duration, status
- **Avoid PII:** Don't log sensitive user data
- **Sample High Traffic:** Log 1-10% of requests for busy services
- **Log Errors:** Always log error responses

**Common Log Formats:**
- **Apache Combined Log:** \`method status duration\`
- **JSON:** \`{"method": "...", "status": "...", "duration": 123}\`
- **Key-Value:** \`method=GetUser status=OK duration=50ms\`

Without LoggingInterceptor, tracking gRPC calls requires adding logging to every handler—duplicate code and missed calls.`,	order: 0,
	translations: {
		ru: {
			title: 'Логирование gRPC вызовов',
			description: `Реализуйте **LoggingInterceptor**, который логирует начало и завершение выполнения gRPC handler используя предоставленную функцию logger.

**Требования:**
1. Создайте функцию \`LoggingInterceptor(logger func(context.Context, string)) UnaryServerInterceptor\`
2. Обработайте nil logger (используйте no-op функцию)
3. Вызовите logger с "start" перед выполнением handler
4. Вызовите handler (next)
5. Вызовите logger с "finish" после завершения handler
6. Верните ответ и ошибку handler

**Пример:**
\`\`\`go
logs := []string{}
logger := func(ctx context.Context, msg string) {
    logs = append(logs, msg)
}

interceptor := LoggingInterceptor(logger)
handler := func(ctx context.Context, req any) (any, error) {
    return "response", nil
}

resp, err := interceptor(ctx, "request", handler)
// logs = ["start", "finish"]
\`\`\`

**Ограничения:**
- Должен вызывать logger до и после выполнения handler
- Должен корректно обрабатывать nil logger`,
			hint1: `Верните функцию которая вызывает logger("start"), затем next(), затем logger("finish").`,
			hint2: `Проверьте если logger nil и замените no-op функцией чтобы избежать nil pointer dereference.`,
			whyItMatters: `LoggingInterceptor обеспечивает observability в gRPC сервисах отслеживая все RPC вызовы, необходимо для отладки и мониторинга микросервисов.

**Почему Logging Interceptors:**
- **Отслеживание запросов:** Логирование каждого RPC вызова для audit trails
- **Мониторинг производительности:** Измерение времени выполнения handler
- **Отладка:** Видение точного call flow в распределённых системах
- **Соответствие требованиям:** Запись всех обращений к сервису для регуляторных требований

**Продакшен паттерн:**
\`\`\`go
// Расширенное логирование с именем метода и временем
func StructuredLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        start := time.Now()

        // Логируем начало запроса с метаданными
        method := info.FullMethod
        log.Printf("[gRPC] START method=%s", method)

        // Выполняем handler
        resp, err := next(ctx, req)

        // Логируем завершение со статусом
        duration := time.Since(start)
        status := "OK"
        if err != nil {
            status = "ERROR"
        }

        log.Printf("[gRPC] FINISH method=%s status=%s duration=%v", method, status, duration)

        return resp, err
    }
}

// JSON структурированное логирование
func JSONLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        start := time.Now()

        logEntry := map[string]interface{}{
            "timestamp": time.Now().UTC().Format(time.RFC3339),
            "method":    info.FullMethod,
            "type":      "start",
        }
        json.NewEncoder(os.Stdout).Encode(logEntry)

        resp, err := next(ctx, req)

        logEntry = map[string]interface{}{
            "timestamp":   time.Now().UTC().Format(time.RFC3339),
            "method":      info.FullMethod,
            "type":        "finish",
            "duration_ms": time.Since(start).Milliseconds(),
            "success":     err == nil,
        }

        if err != nil {
            logEntry["error"] = err.Error()
        }

        json.NewEncoder(os.Stdout).Encode(logEntry)

        return resp, err
    }
}

// Логирование с request/response payload (осторожно с чувствительными данными!)
func VerboseLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        // Логируем запрос
        log.Printf("[gRPC] → %s request: %+v", info.FullMethod, req)

        resp, err := next(ctx, req)

        // Логируем ответ
        if err != nil {
            log.Printf("[gRPC] ← %s error: %v", info.FullMethod, err)
        } else {
            log.Printf("[gRPC] ← %s response: %+v", info.FullMethod, resp)
        }

        return resp, err
    }
}

// Условное логирование (только ошибки)
func ErrorLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        resp, err := next(ctx, req)

        if err != nil {
            log.Printf("[gRPC ERROR] method=%s error=%v", info.FullMethod, err)

            // Отправка в систему отслеживания ошибок
            sentry.CaptureException(err)
        }

        return resp, err
    }
}

// Логирование с интеграцией метрик
func MetricsLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        start := time.Now()

        resp, err := next(ctx, req)

        duration := time.Since(start)

        // Записываем метрики
        metrics.RecordRPCDuration(info.FullMethod, duration)

        if err != nil {
            metrics.IncrementRPCErrors(info.FullMethod)
            log.Printf("[gRPC] method=%s error=%v duration=%v", info.FullMethod, err, duration)
        } else {
            metrics.IncrementRPCSuccess(info.FullMethod)
            log.Printf("[gRPC] method=%s success duration=%v", info.FullMethod, duration)
        }

        return resp, err
    }
}

// Логирование с сэмплированием (логируем только 10% запросов)
func SampledLoggingInterceptor(sampleRate float64) UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        shouldLog := rand.Float64() < sampleRate

        if shouldLog {
            start := time.Now()
            log.Printf("[gRPC SAMPLE] START %s", info.FullMethod)

            resp, err := next(ctx, req)

            duration := time.Since(start)
            log.Printf("[gRPC SAMPLE] FINISH %s duration=%v error=%v", info.FullMethod, duration, err != nil)

            return resp, err
        }

        return next(ctx, req)
    }
}
\`\`\`

**Практические преимущества:**
- **Distributed Tracing:** Отслеживание запросов через множество сервисов
- **Анализ производительности:** Идентификация медленных RPC
- **Исследование ошибок:** Видение какие методы чаще всего падают
- **Планирование мощностей:** Анализ паттернов трафика

**Лучшие практики логирования:**
- **Структурированные логи:** Используйте JSON для машинного парсинга
- **Включайте контекст:** Имя метода, длительность, статус
- **Избегайте PII:** Не логируйте чувствительные пользовательские данные
- **Сэмплируйте высоконагруженные сервисы:** Логируйте 1-10% запросов для загруженных сервисов
- **Логируйте ошибки:** Всегда логируйте ошибочные ответы

**Общие форматы логов:**
- **Apache Combined Log:** \`method status duration\`
- **JSON:** \`{"method": "...", "status": "...", "duration": 123}\`
- **Key-Value:** \`method=GetUser status=OK duration=50ms\`

Без LoggingInterceptor отслеживание gRPC вызовов требует добавления логирования в каждый handler—дублирование кода и пропущенные вызовы.`,
			solutionCode: `package grpcx

import (
	"context"
)

type Handler func(ctx context.Context, req any) (any, error)

type UnaryServerInterceptor func(ctx context.Context, req any, next Handler) (any, error)

func LoggingInterceptor(logger func(context.Context, string)) UnaryServerInterceptor {
	if logger == nil {	// Проверка на nil logger
		logger = func(context.Context, string) {}	// Используем no-op logger
	}
	return func(ctx context.Context, req any, next Handler) (any, error) {
		logger(ctx, "start")	// Логируем начало выполнения
		resp, err := next(ctx, req)	// Выполняем handler
		logger(ctx, "finish")	// Логируем завершение выполнения
		return resp, err	// Возвращаем результат handler
	}
}`
		},
		uz: {
			title: 'gRPC chaqiruvlarini loglash',
			description: `Berilgan logger funksiyasidan foydalanib gRPC handler bajarilishining boshlanishi va tugashini loglaydigan **LoggingInterceptor** ni amalga oshiring.

**Talablar:**
1. \`LoggingInterceptor(logger func(context.Context, string)) UnaryServerInterceptor\` funksiyasini yarating
2. nil loggerni ishlang (no-op funksiyadan foydalaning)
3. Handler bajarilishidan oldin "start" bilan loggerni chaqiring
4. Handlerni (next) chaqiring
5. Handler tugagandan keyin "finish" bilan loggerni chaqiring
6. Handler javob va xatosini qaytaring

**Misol:**
\`\`\`go
logs := []string{}
logger := func(ctx context.Context, msg string) {
    logs = append(logs, msg)
}

interceptor := LoggingInterceptor(logger)
handler := func(ctx context.Context, req any) (any, error) {
    return "response", nil
}

resp, err := interceptor(ctx, "request", handler)
// logs = ["start", "finish"]
\`\`\`

**Cheklovlar:**
- Handler bajarilishidan oldin va keyin loggerni chaqirishi kerak
- nil loggerni to'g'ri ishlashi kerak`,
			hint1: `logger("start"), keyin next(), keyin logger("finish") ni chaqiruvchi funksiyani qaytaring.`,
			hint2: `Logger nil ekanligini tekshiring va nil pointer dereference dan qochish uchun no-op funksiya bilan almashtiring.`,
			whyItMatters: `LoggingInterceptor barcha RPC chaqiruvlarni kuzatish orqali gRPC xizmatlarida observability ni ta'minlaydi, mikroservislarni debug va monitoring uchun zarur.

**Nima uchun Logging Interceptors:**
- **Request kuzatish:** Audit traillar uchun har bir RPC chaqiruvini loglash
- **Performans monitoring:** Handler bajarilish vaqtini o'lchash
- **Debug:** Taqsimlangan tizimlarda aniq call flow ni ko'rish
- **Muvofiqlik:** Tartibga solish talablari uchun barcha xizmat murojaatlarini yozib qolish

**Ishlab chiqarish patterni:**
\`\`\`go
// Metod nomi va vaqt bilan kengaytirilgan loglash
func StructuredLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        start := time.Now()

        // Request boshlanishini metadata bilan loglash
        method := info.FullMethod
        log.Printf("[gRPC] START method=%s", method)

        // Handlerni bajarish
        resp, err := next(ctx, req)

        // Tugashni status bilan loglash
        duration := time.Since(start)
        status := "OK"
        if err != nil {
            status = "ERROR"
        }

        log.Printf("[gRPC] FINISH method=%s status=%s duration=%v", method, status, duration)

        return resp, err
    }
}

// JSON strukturali loglash
func JSONLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        start := time.Now()

        logEntry := map[string]interface{}{
            "timestamp": time.Now().UTC().Format(time.RFC3339),
            "method":    info.FullMethod,
            "type":      "start",
        }
        json.NewEncoder(os.Stdout).Encode(logEntry)

        resp, err := next(ctx, req)

        logEntry = map[string]interface{}{
            "timestamp":   time.Now().UTC().Format(time.RFC3339),
            "method":      info.FullMethod,
            "type":        "finish",
            "duration_ms": time.Since(start).Milliseconds(),
            "success":     err == nil,
        }

        if err != nil {
            logEntry["error"] = err.Error()
        }

        json.NewEncoder(os.Stdout).Encode(logEntry)

        return resp, err
    }
}

// Request/response payload bilan loglash (maxfiy ma'lumotlarga ehtiyot bo'ling!)
func VerboseLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        // Requestni loglash
        log.Printf("[gRPC] → %s request: %+v", info.FullMethod, req)

        resp, err := next(ctx, req)

        // Responseni loglash
        if err != nil {
            log.Printf("[gRPC] ← %s error: %v", info.FullMethod, err)
        } else {
            log.Printf("[gRPC] ← %s response: %+v", info.FullMethod, resp)
        }

        return resp, err
    }
}

// Shartli loglash (faqat xatolar)
func ErrorLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        resp, err := next(ctx, req)

        if err != nil {
            log.Printf("[gRPC ERROR] method=%s error=%v", info.FullMethod, err)

            // Xatolarni kuzatish tizimiga yuborish
            sentry.CaptureException(err)
        }

        return resp, err
    }
}

// Metrikalar integratsiyasi bilan loglash
func MetricsLoggingInterceptor() UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        start := time.Now()

        resp, err := next(ctx, req)

        duration := time.Since(start)

        // Metrikalarni yozish
        metrics.RecordRPCDuration(info.FullMethod, duration)

        if err != nil {
            metrics.IncrementRPCErrors(info.FullMethod)
            log.Printf("[gRPC] method=%s error=%v duration=%v", info.FullMethod, err, duration)
        } else {
            metrics.IncrementRPCSuccess(info.FullMethod)
            log.Printf("[gRPC] method=%s success duration=%v", info.FullMethod, duration)
        }

        return resp, err
    }
}

// Sampling bilan loglash (faqat 10% requestlarni loglash)
func SampledLoggingInterceptor(sampleRate float64) UnaryServerInterceptor {
    return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, next Handler) (any, error) {
        shouldLog := rand.Float64() < sampleRate

        if shouldLog {
            start := time.Now()
            log.Printf("[gRPC SAMPLE] START %s", info.FullMethod)

            resp, err := next(ctx, req)

            duration := time.Since(start)
            log.Printf("[gRPC SAMPLE] FINISH %s duration=%v error=%v", info.FullMethod, duration, err != nil)

            return resp, err
        }

        return next(ctx, req)
    }
}
\`\`\`

**Amaliy foydalari:**
- **Distributed Tracing:** Ko'plab servislar orqali requestlarni kuzatish
- **Performans tahlili:** Sekin RPC larni aniqlash
- **Xatolarni tekshirish:** Qaysi metodlar ko'proq muvaffaqiyatsiz bo'lishini ko'rish
- **Quvvat rejalashtirish:** Trafik patternlarini tahlil qilish

**Loglash eng yaxshi amaliyotlari:**
- **Strukturali loglar:** Mashina parsing uchun JSON dan foydalaning
- **Kontekstni qo'shing:** Metod nomi, davomiyligi, status
- **PII dan qoching:** Maxfiy foydalanuvchi ma'lumotlarini loglashtirmang
- **Yuqori trafikli servislarni sampling qiling:** Band servislar uchun requestlarning 1-10% ini loglang
- **Xatolarni loglang:** Har doim xato javoblarini loglang

**Keng tarqalgan log formatlari:**
- **Apache Combined Log:** \`method status duration\`
- **JSON:** \`{"method": "...", "status": "...", "duration": 123}\`
- **Key-Value:** \`method=GetUser status=OK duration=50ms\`

LoggingInterceptor siz gRPC chaqiruvlarni kuzatish har bir handlerga loglash qo'shishni talab qiladi—kod dublikatsiyasi va o'tkazib yuborilgan chaqiruvlar.`,
			solutionCode: `package grpcx

import (
	"context"
)

type Handler func(ctx context.Context, req any) (any, error)

type UnaryServerInterceptor func(ctx context.Context, req any, next Handler) (any, error)

func LoggingInterceptor(logger func(context.Context, string)) UnaryServerInterceptor {
	if logger == nil {	// Logger nil ekanligini tekshirish
		logger = func(context.Context, string) {}	// No-op logger ishlatamiz
	}
	return func(ctx context.Context, req any, next Handler) (any, error) {
		logger(ctx, "start")	// Bajarilish boshlanishini loglash
		resp, err := next(ctx, req)	// Handlerni bajarish
		logger(ctx, "finish")	// Bajarilish tugashini loglash
		return resp, err	// Handler natijasini qaytarish
	}
}`
		}
	}
};

export default task;
