import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-concurrency-wait-for-signal',
	title: 'Wait For Signal',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'context', 'channels'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **WaitForSignal** that waits for either context cancellation or signal from channel.

**Requirements:**
1. Create function \`WaitForSignal(ctx context.Context, signal <-chan struct{}) error\`
2. Handle nil context (use Background)
3. Wait for either ctx.Done() or signal
4. Return context error if context canceled
5. Return nil if signal received
6. Use select to wait for both

**Example:**
\`\`\`go
signal := make(chan struct{})
ctx := context.Background()

go func() {
    time.Sleep(100 * time.Millisecond)
    close(signal)
}()

err := WaitForSignal(ctx, signal)
// err = nil (signal received)

ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
defer cancel()

signal = make(chan struct{}) // Never closes

err = WaitForSignal(ctx, signal)
// err = context.DeadlineExceeded
\`\`\`

**Constraints:**
- Must use select statement
- Must wait for context or signal
- Must return appropriate error`,
	initialCode: `package concurrency

import (
	"context"
)

// TODO: Implement WaitForSignal
func WaitForSignal($2) error {
	return nil // TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
)

func WaitForSignal(ctx context.Context, signal <-chan struct{}) error {
	if ctx == nil {                                             // Handle nil context
		ctx = context.Background()                          // Use Background as fallback
	}
	select {
	case <-ctx.Done():                                          // Context canceled
		return ctx.Err()                                    // Return context error
	case <-signal:                                              // Signal received
		return nil                                          // Return success
	}
}`,
			hint1: `Use a select statement with two cases: one for ctx.Done() and one for signal channel.`,
			hint2: `Return ctx.Err() if context is canceled, return nil if signal is received.`,
			whyItMatters: `WaitForSignal enables race conditions between context cancellation and external signals, essential for timeout-aware event handling.

**Why Signal Waiting:**
- **Timeout Protection:** Don't wait forever for signals
- **Cancellation:** Stop waiting when context is canceled
- **Event Coordination:** Wait for external events with timeout
- **Resource Safety:** Free resources if signal never arrives

**Production Pattern:**
\`\`\`go
// Wait for service to be ready
func WaitForServiceReady(ctx context.Context, service *Service) error {
    ready := make(chan struct{})

    go func() {
        for !service.IsReady() {
            time.Sleep(100 * time.Millisecond)
        }
        close(ready)
    }()

    return WaitForSignal(ctx, ready)
}

// Wait for database connection
func WaitForDatabase(ctx context.Context, db *sql.DB) error {
    connected := make(chan struct{})

    go func() {
        for {
            if err := db.Ping(); err == nil {
                close(connected)
                return
            }
            time.Sleep(time.Second)
        }
    }()

    return WaitForSignal(ctx, connected)
}

// Wait for file to appear
func WaitForFile(ctx context.Context, path string) error {
    exists := make(chan struct{})

    go func() {
        for {
            if _, err := os.Stat(path); err == nil {
                close(exists)
                return
            }
            time.Sleep(500 * time.Millisecond)
        }
    }()

    return WaitForSignal(ctx, exists)
}

// Wait for external API to respond
func WaitForAPIResponse(ctx context.Context, endpoint string) (*Response, error) {
    response := make(chan *Response)
    done := make(chan struct{})

    go func() {
        defer close(done)
        resp := pollAPI(endpoint)
        response <- resp
    }()

    if err := WaitForSignal(ctx, done); err != nil {
        return nil, err
    }

    return <-response, nil
}

// Wait for user input with timeout
func WaitForInput(ctx context.Context) (string, error) {
    input := make(chan string)
    done := make(chan struct{})

    go func() {
        defer close(done)
        var s string
        fmt.Scanln(&s)
        input <- s
    }()

    if err := WaitForSignal(ctx, done); err != nil {
        return "", err
    }

    return <-input, nil
}

// Wait for health check to pass
func WaitForHealthy(ctx context.Context, checker HealthChecker) error {
    healthy := make(chan struct{})

    go func() {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()

        for range ticker.C {
            if checker.IsHealthy() {
                close(healthy)
                return
            }
        }
    }()

    return WaitForSignal(ctx, healthy)
}

// Wait for cache to warm up
func WaitForCacheWarmup(ctx context.Context, cache *Cache) error {
    warmed := make(chan struct{})

    go func() {
        cache.WarmUp()
        close(warmed)
    }()

    return WaitForSignal(ctx, warmed)
}

// Wait for initialization sequence
func WaitForInit(ctx context.Context) error {
    initDone := make(chan struct{})

    go func() {
        defer close(initDone)
        loadConfig()
        connectDatabase()
        startServices()
    }()

    if err := WaitForSignal(ctx, initDone); err != nil {
        return fmt.Errorf("initialization failed: %w", err)
    }

    return nil
}

// Graceful startup with timeout
func GracefulStartup(timeout time.Duration) error {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()

    // Wait for all services with timeout
    services := []string{"api", "database", "cache", "queue"}

    for _, service := range services {
        ready := startService(service)
        if err := WaitForSignal(ctx, ready); err != nil {
            return fmt.Errorf("%s failed to start: %w", service, err)
        }
    }

    return nil
}
\`\`\`

**Real-World Benefits:**
- **Timeout Safety:** Never wait forever for signals
- **Graceful Degradation:** Handle timeouts appropriately
- **Resource Control:** Free resources on timeout
- **Predictable Behavior:** Operations complete or timeout

**Common Use Cases:**
- **Service Readiness:** Wait for services to start
- **Health Checks:** Wait for system to become healthy
- **File Operations:** Wait for files to appear
- **Database Connections:** Wait for DB to be ready
- **API Polling:** Wait for external API responses
- **User Input:** Get input with timeout

**Error Handling:**
- **Timeout:** context.DeadlineExceeded
- **Cancellation:** context.Canceled
- **Success:** nil (signal received)

Without WaitForSignal, waiting for external events with timeout protection requires complex, error-prone select statements scattered throughout code.`,
	testCode: `package concurrency

import (
	"context"
	"errors"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	signal := make(chan struct{})
	close(signal)
	err := WaitForSignal(context.Background(), signal)
	if err != nil { t.Errorf("expected nil for closed signal, got %v", err) }
}

func Test2(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	signal := make(chan struct{})
	err := WaitForSignal(ctx, signal)
	if !errors.Is(err, context.DeadlineExceeded) { t.Errorf("expected DeadlineExceeded, got %v", err) }
}

func Test3(t *testing.T) {
	signal := make(chan struct{})
	close(signal)
	err := WaitForSignal(nil, signal)
	if err != nil { t.Errorf("expected nil for nil context with closed signal, got %v", err) }
}

func Test4(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	signal := make(chan struct{})
	go func() {
		time.Sleep(50*time.Millisecond)
		close(signal)
	}()
	cancel()
	err := WaitForSignal(ctx, signal)
	if err == nil { t.Error("expected error for canceled context") }
}

func Test5(t *testing.T) {
	signal := make(chan struct{})
	go func() {
		time.Sleep(20*time.Millisecond)
		close(signal)
	}()
	err := WaitForSignal(context.Background(), signal)
	if err != nil { t.Errorf("expected nil when signal received, got %v", err) }
}

func Test6(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	signal := make(chan struct{})
	err := WaitForSignal(ctx, signal)
	if !errors.Is(err, context.Canceled) { t.Errorf("expected Canceled, got %v", err) }
}

func Test7(t *testing.T) {
	signal := make(chan struct{})
	go func() { close(signal) }()
	start := time.Now()
	_ = WaitForSignal(context.Background(), signal)
	if time.Since(start) > 50*time.Millisecond { t.Error("should return quickly after signal") }
}

func Test8(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	signal := make(chan struct{})
	go func() {
		time.Sleep(20*time.Millisecond)
		close(signal)
	}()
	err := WaitForSignal(ctx, signal)
	if err != nil { t.Errorf("expected nil, got %v", err) }
}

func Test9(t *testing.T) {
	signal := make(chan struct{}, 1)
	signal <- struct{}{}
	err := WaitForSignal(context.Background(), signal)
	if err != nil { t.Errorf("expected nil for buffered signal, got %v", err) }
}

func Test10(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	signal := make(chan struct{})
	start := time.Now()
	_ = WaitForSignal(ctx, signal)
	if time.Since(start) > 50*time.Millisecond { t.Error("timeout should work") }
}
`,
	order: 4,
	translations: {
		ru: {
			title: 'Ожидание системного сигнала завершения',
			description: `Реализуйте **WaitForSignal**, который ждёт либо отмены контекста, либо сигнала из канала.

**Требования:**
1. Создайте функцию \`WaitForSignal(ctx context.Context, signal <-chan struct{}) error\`
2. Обработайте nil context (используйте Background)
3. Ждите либо ctx.Done() либо signal
4. Верните ошибку контекста если контекст отменён
5. Верните nil если получен сигнал
6. Используйте select для ожидания обоих

**Пример:**
\`\`\`go
signal := make(chan struct{})
ctx := context.Background()

go func() {
    time.Sleep(100 * time.Millisecond)
    close(signal)
}()

err := WaitForSignal(ctx, signal)
// err = nil (сигнал получен)

ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
defer cancel()

signal = make(chan struct{}) // Никогда не закроется

err = WaitForSignal(ctx, signal)
// err = context.DeadlineExceeded
\`\`\`

**Ограничения:**
- Должен использовать select statement
- Должен ждать контекст или сигнал
- Должен возвращать соответствующую ошибку`,
			hint1: `Используйте select statement с двумя cases: один для ctx.Done() и один для signal канала.`,
			hint2: `Верните ctx.Err() если контекст отменён, верните nil если получен сигнал.`,
			whyItMatters: `WaitForSignal позволяет гонку условий между отменой контекста и внешними сигналами, необходим для обработки событий с учётом таймаутов.

**Почему Signal Waiting:**
- **Защита от таймаута:** Не ждать вечно сигналов
- **Отмена:** Прекратить ожидание при отмене контекста
- **Координация событий:** Ждать внешние события с таймаутом
- **Безопасность ресурсов:** Освободить ресурсы если сигнал не приходит

**Продакшен паттерн:**
\`\`\`go
// Ожидание готовности сервиса
func WaitForServiceReady(ctx context.Context, service *Service) error {
    ready := make(chan struct{})

    go func() {
        for !service.IsReady() {
            time.Sleep(100 * time.Millisecond)
        }
        close(ready)
    }()

    return WaitForSignal(ctx, ready)
}

// Ожидание подключения к БД
func WaitForDatabase(ctx context.Context, db *sql.DB) error {
    connected := make(chan struct{})

    go func() {
        for {
            if err := db.Ping(); err == nil {
                close(connected)
                return
            }
            time.Sleep(time.Second)
        }
    }()

    return WaitForSignal(ctx, connected)
}

// Ожидание появления файла
func WaitForFile(ctx context.Context, path string) error {
    exists := make(chan struct{})

    go func() {
        for {
            if _, err := os.Stat(path); err == nil {
                close(exists)
                return
            }
            time.Sleep(500 * time.Millisecond)
        }
    }()

    return WaitForSignal(ctx, exists)
}

// Ожидание ответа внешнего API
func WaitForAPIResponse(ctx context.Context, endpoint string) (*Response, error) {
    response := make(chan *Response)
    done := make(chan struct{})

    go func() {
        defer close(done)
        resp := pollAPI(endpoint)
        response <- resp
    }()

    if err := WaitForSignal(ctx, done); err != nil {
        return nil, err
    }

    return <-response, nil
}

// Ожидание пользовательского ввода с таймаутом
func WaitForInput(ctx context.Context) (string, error) {
    input := make(chan string)
    done := make(chan struct{})

    go func() {
        defer close(done)
        var s string
        fmt.Scanln(&s)
        input <- s
    }()

    if err := WaitForSignal(ctx, done); err != nil {
        return "", err
    }

    return <-input, nil
}

// Ожидание успешного health check
func WaitForHealthy(ctx context.Context, checker HealthChecker) error {
    healthy := make(chan struct{})

    go func() {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()

        for range ticker.C {
            if checker.IsHealthy() {
                close(healthy)
                return
            }
        }
    }()

    return WaitForSignal(ctx, healthy)
}

// Ожидание прогрева кеша
func WaitForCacheWarmup(ctx context.Context, cache *Cache) error {
    warmed := make(chan struct{})

    go func() {
        cache.WarmUp()
        close(warmed)
    }()

    return WaitForSignal(ctx, warmed)
}

// Ожидание последовательности инициализации
func WaitForInit(ctx context.Context) error {
    initDone := make(chan struct{})

    go func() {
        defer close(initDone)
        loadConfig()
        connectDatabase()
        startServices()
    }()

    if err := WaitForSignal(ctx, initDone); err != nil {
        return fmt.Errorf("initialization failed: %w", err)
    }

    return nil
}

// Graceful запуск с таймаутом
func GracefulStartup(timeout time.Duration) error {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()

    // Ожидание всех сервисов с таймаутом
    services := []string{"api", "database", "cache", "queue"}

    for _, service := range services {
        ready := startService(service)
        if err := WaitForSignal(ctx, ready); err != nil {
            return fmt.Errorf("%s failed to start: %w", service, err)
        }
    }

    return nil
}
\`\`\`

**Практические преимущества:**
- **Безопасность таймаута:** Никогда не ждать вечно сигналов
- **Graceful Degradation:** Обработка таймаутов соответствующим образом
- **Контроль ресурсов:** Освобождение ресурсов при таймауте
- **Предсказуемое поведение:** Операции завершаются или таймаут

**Обычные случаи использования:**
- **Готовность сервиса:** Ожидание запуска сервисов
- **Health Checks:** Ожидание здоровья системы
- **Файловые операции:** Ожидание появления файлов
- **Подключения к БД:** Ожидание готовности БД
- **API Polling:** Ожидание ответов внешних API
- **Пользовательский ввод:** Получение ввода с таймаутом

**Обработка ошибок:**
- **Timeout:** context.DeadlineExceeded
- **Cancellation:** context.Canceled
- **Success:** nil (получен сигнал)

Без WaitForSignal ожидание внешних событий с защитой от таймаута требует сложных, подверженных ошибкам select statements разбросанных по коду.`,
			solutionCode: `package concurrency

import (
	"context"
)

func WaitForSignal(ctx context.Context, signal <-chan struct{}) error {
	if ctx == nil {                                             // Обработка nil контекста
		ctx = context.Background()                          // Используем Background как fallback
	}
	select {
	case <-ctx.Done():                                          // Контекст отменён
		return ctx.Err()                                    // Возвращаем ошибку контекста
	case <-signal:                                              // Сигнал получен
		return nil                                          // Возвращаем успех
	}
}`
		},
		uz: {
			title: 'Tizim signalining tugashini kutish',
			description: `Kontekst bekor qilinishini yoki kanaldan signalni kutadigan **WaitForSignal** ni amalga oshiring.

**Talablar:**
1. \`WaitForSignal(ctx context.Context, signal <-chan struct{}) error\` funksiyasini yarating
2. nil kontekstni ishlang (Background dan foydalaning)
3. ctx.Done() yoki signalni kuting
4. Agar kontekst bekor qilinsa kontekst xatosini qaytaring
5. Agar signal olinsa nil qaytaring
6. Ikkalasini kutish uchun select dan foydalaning

**Misol:**
\`\`\`go
signal := make(chan struct{})
ctx := context.Background()

go func() {
    time.Sleep(100 * time.Millisecond)
    close(signal)
}()

err := WaitForSignal(ctx, signal)
// err = nil (signal olindi)

ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
defer cancel()

signal = make(chan struct{}) // Hech qachon yopilmaydi

err = WaitForSignal(ctx, signal)
// err = context.DeadlineExceeded
\`\`\`

**Cheklovlar:**
- select statementdan foydalanishi kerak
- Kontekst yoki signalni kutishi kerak
- Tegishli xatoni qaytarishi kerak`,
			hint1: `Ikki case bilan select statementdan foydalaning: biri ctx.Done() uchun va biri signal kanali uchun.`,
			hint2: `Agar kontekst bekor qilinsa ctx.Err() ni qaytaring, signal olinsa nil qaytaring.`,
			whyItMatters: `WaitForSignal kontekst bekor qilinishi va tashqi signallar o'rtasida poyga sharoitlarini yoqadi, timeout xabardor hodisalarni qayta ishlash uchun zarur.

**Nima uchun Signal Waiting:**
- **Timeout himoyasi:** Signallarni abadiy kutmaslik
- **Bekor qilish:** Kontekst bekor qilinganda kutishni to'xtatish
- **Hodisalarni muvofiqlashtirish:** Timeout bilan tashqi hodisalarni kutish
- **Resurs xavfsizligi:** Agar signal kelmasa resurslarni ozod qilish

**Ishlab chiqarish patterni:**
\`\`\`go
// Xizmat tayyorligini kutish
func WaitForServiceReady(ctx context.Context, service *Service) error {
    ready := make(chan struct{})

    go func() {
        for !service.IsReady() {
            time.Sleep(100 * time.Millisecond)
        }
        close(ready)
    }()

    return WaitForSignal(ctx, ready)
}

// Ma'lumotlar bazasi ulanishini kutish
func WaitForDatabase(ctx context.Context, db *sql.DB) error {
    connected := make(chan struct{})

    go func() {
        for {
            if err := db.Ping(); err == nil {
                close(connected)
                return
            }
            time.Sleep(time.Second)
        }
    }()

    return WaitForSignal(ctx, connected)
}

// Fayl paydo bo'lishini kutish
func WaitForFile(ctx context.Context, path string) error {
    exists := make(chan struct{})

    go func() {
        for {
            if _, err := os.Stat(path); err == nil {
                close(exists)
                return
            }
            time.Sleep(500 * time.Millisecond)
        }
    }()

    return WaitForSignal(ctx, exists)
}

// Tashqi API javobini kutish
func WaitForAPIResponse(ctx context.Context, endpoint string) (*Response, error) {
    response := make(chan *Response)
    done := make(chan struct{})

    go func() {
        defer close(done)
        resp := pollAPI(endpoint)
        response <- resp
    }()

    if err := WaitForSignal(ctx, done); err != nil {
        return nil, err
    }

    return <-response, nil
}

// Timeout bilan foydalanuvchi kiritishini kutish
func WaitForInput(ctx context.Context) (string, error) {
    input := make(chan string)
    done := make(chan struct{})

    go func() {
        defer close(done)
        var s string
        fmt.Scanln(&s)
        input <- s
    }()

    if err := WaitForSignal(ctx, done); err != nil {
        return "", err
    }

    return <-input, nil
}

// Muvaffaqiyatli health checkni kutish
func WaitForHealthy(ctx context.Context, checker HealthChecker) error {
    healthy := make(chan struct{})

    go func() {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()

        for range ticker.C {
            if checker.IsHealthy() {
                close(healthy)
                return
            }
        }
    }()

    return WaitForSignal(ctx, healthy)
}

// Kesh isitilishini kutish
func WaitForCacheWarmup(ctx context.Context, cache *Cache) error {
    warmed := make(chan struct{})

    go func() {
        cache.WarmUp()
        close(warmed)
    }()

    return WaitForSignal(ctx, warmed)
}

// Initsializatsiya ketma-ketligini kutish
func WaitForInit(ctx context.Context) error {
    initDone := make(chan struct{})

    go func() {
        defer close(initDone)
        loadConfig()
        connectDatabase()
        startServices()
    }()

    if err := WaitForSignal(ctx, initDone); err != nil {
        return fmt.Errorf("initialization failed: %w", err)
    }

    return nil
}

// Timeout bilan Graceful ishga tushirish
func GracefulStartup(timeout time.Duration) error {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()

    // Timeout bilan barcha xizmatlarni kutish
    services := []string{"api", "database", "cache", "queue"}

    for _, service := range services {
        ready := startService(service)
        if err := WaitForSignal(ctx, ready); err != nil {
            return fmt.Errorf("%s failed to start: %w", service, err)
        }
    }

    return nil
}
\`\`\`

**Amaliy foydalari:**
- **Timeout xavfsizligi:** Signallarni hech qachon abadiy kutmaslik
- **Graceful Degradation:** Timeoutlarni tegishli qayta ishlash
- **Resurs nazorati:** Timeoutda resurslarni ozod qilish
- **Bashorat qilinadigan xatti-harakat:** Operatsiyalar tugaydi yoki timeout

**Oddiy foydalanish holatlari:**
- **Xizmat tayyorligi:** Xizmatlar ishga tushishini kutish
- **Health Checks:** Tizim sog'lom bo'lishini kutish
- **Fayl operatsiyalari:** Fayllar paydo bo'lishini kutish
- **DB ulanishlari:** DBning tayyor bo'lishini kutish
- **API Polling:** Tashqi API javoblarini kutish
- **Foydalanuvchi kiritishi:** Timeout bilan kiritish olish

**Xatolarni qayta ishlash:**
- **Timeout:** context.DeadlineExceeded
- **Bekor qilish:** context.Canceled
- **Muvaffaqiyat:** nil (signal olindi)

WaitForSignal bo'lmasa, timeout himoyasi bilan tashqi hodisalarni kutish kod bo'ylab tarqalgan murakkab va xatolarga moyil select statementlarni talab qiladi.`,
			solutionCode: `package concurrency

import (
	"context"
)

func WaitForSignal(ctx context.Context, signal <-chan struct{}) error {
	if ctx == nil {                                             // nil kontekstni ishlash
		ctx = context.Background()                          // Fallback sifatida Background ishlatamiz
	}
	select {
	case <-ctx.Done():                                          // Kontekst bekor qilindi
		return ctx.Err()                                    // Kontekst xatosini qaytaramiz
	case <-signal:                                              // Signal olindi
		return nil                                          // Muvaffaqiyatni qaytaramiz
	}
}`
		}
	}
};

export default task;
