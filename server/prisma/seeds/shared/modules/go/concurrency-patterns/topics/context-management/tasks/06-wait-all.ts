import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-concurrency-wait-all',
	title: 'Wait All',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'context', 'synchronization'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **WaitAll** that waits for all signal channels to close or context to be canceled.

**Requirements:**
1. Create function \`WaitAll(ctx context.Context, signals ...<-chan struct{}) error\`
2. Handle nil context (use Background)
3. Handle empty signals (return nil immediately)
4. Use sync.WaitGroup to wait for all signals
5. Each signal waits in separate goroutine
6. Return context error if context canceled before all signals
7. Return nil if all signals received

**Example:**
\`\`\`go
sig1 := make(chan struct{})
sig2 := make(chan struct{})
sig3 := make(chan struct{})

go func() {
    time.Sleep(100 * time.Millisecond)
    close(sig1)
}()
go func() {
    time.Sleep(150 * time.Millisecond)
    close(sig2)
}()
go func() {
    time.Sleep(200 * time.Millisecond)
    close(sig3)
}()

err := WaitAll(context.Background(), sig1, sig2, sig3)
// err = nil (all signals received)

ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
defer cancel()

err = WaitAll(ctx, sig1, sig2, sig3)
// err = context.DeadlineExceeded (timeout before all signals)
\`\`\`

**Constraints:**
- Must use sync.WaitGroup
- Must wait for all signals in parallel
- Must handle context cancellation`,
	initialCode: `package concurrency

import (
	"context"
	"sync"
)

// TODO: Implement WaitAll
func WaitAll($2) error {
	return nil // TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
	"sync"
)

func WaitAll(ctx context.Context, signals ...<-chan struct{}) error {
	if ctx == nil {                                             // Handle nil context
		ctx = context.Background()                          // Use Background as fallback
	}
	if len(signals) == 0 {                                      // No signals to wait for
		return nil                                          // Return immediately
	}
	done := make(chan struct{})                                 // Channel to signal all complete
	go func() {                                                 // Run waiting in background
		defer close(done)                                   // Close when all signals received
		var wg sync.WaitGroup                               // WaitGroup for all signals
		wg.Add(len(signals))                                // Add all signals to group
		for _, ch := range signals {                        // Iterate over all signals
			ch := ch                                    // Capture loop variable
			go func() {                                 // Wait for each signal in goroutine
				defer wg.Done()                     // Mark this signal as done
				select {
				case <-ctx.Done():              // Context canceled
				case <-ch:                      // Signal received
				}
			}()
		}
		wg.Wait()                                           // Wait for all signals
	}()
	select {
	case <-ctx.Done():                                          // Context canceled
		return ctx.Err()                                    // Return context error
	case <-done:                                                // All signals received
		return ctx.Err()                                    // Return nil or context error
	}
}`,
			hint1: `Use sync.WaitGroup with len(signals) count. Launch a goroutine for each signal that waits for either ctx.Done() or the signal.`,
			hint2: `Create a done channel that closes when WaitGroup finishes. Use select to race between ctx.Done() and done.`,
			whyItMatters: `WaitAll enables waiting for multiple concurrent operations to complete with timeout protection, essential for coordinating parallel tasks.

**Why Wait All:**
- **Parallel Coordination:** Wait for all tasks to finish
- **Timeout Protection:** Don't wait forever if tasks hang
- **Resource Synchronization:** Know when all resources ready
- **Batch Completion:** Ensure entire batch completes

**Production Pattern:**
\`\`\`go
// Wait for multiple services to start
func StartAllServices(ctx context.Context, services []*Service) error {
    signals := make([]<-chan struct{}, len(services))

    for i, svc := range services {
        ready := make(chan struct{})
        signals[i] = ready

        go func(s *Service, done chan struct{}) {
            s.Start()
            close(done)
        }(svc, ready)
    }

    return WaitAll(ctx, signals...)
}

// Wait for multiple API calls to complete
func FetchAllData(ctx context.Context, urls []string) ([]Response, error) {
    responses := make([]Response, len(urls))
    signals := make([]<-chan struct{}, len(urls))

    for i, url := range urls {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, u string, d chan struct{}) {
            defer close(d)
            responses[idx] = fetch(u)
        }(i, url, done)
    }

    if err := WaitAll(ctx, signals...); err != nil {
        return nil, err
    }

    return responses, nil
}

// Wait for all database migrations
func RunAllMigrations(ctx context.Context, migrations []Migration) error {
    signals := make([]<-chan struct{}, len(migrations))

    for i, migration := range migrations {
        done := make(chan struct{})
        signals[i] = done

        go func(m Migration, d chan struct{}) {
            defer close(d)
            m.Apply()
        }(migration, done)
    }

    return WaitAll(ctx, signals...)
}

// Wait for all workers to finish processing
func ProcessBatch(ctx context.Context, items []Item, workers int) error {
    itemChan := make(chan Item, len(items))
    for _, item := range items {
        itemChan <- item
    }
    close(itemChan)

    signals := make([]<-chan struct{}, workers)

    for i := 0; i < workers; i++ {
        done := make(chan struct{})
        signals[i] = done

        go func(d chan struct{}) {
            defer close(d)
            for item := range itemChan {
                process(item)
            }
        }(done)
    }

    return WaitAll(ctx, signals...)
}

// Wait for all files to be processed
func ProcessAllFiles(ctx context.Context, files []string) error {
    signals := make([]<-chan struct{}, len(files))

    for i, file := range files {
        done := make(chan struct{})
        signals[i] = done

        go func(f string, d chan struct{}) {
            defer close(d)
            processFile(f)
        }(file, done)
    }

    return WaitAll(ctx, signals...)
}

// Wait for all cache warmup tasks
func WarmUpAllCaches(ctx context.Context, caches []*Cache) error {
    signals := make([]<-chan struct{}, len(caches))

    for i, cache := range caches {
        done := make(chan struct{})
        signals[i] = done

        go func(c *Cache, d chan struct{}) {
            defer close(d)
            c.WarmUp()
        }(cache, done)
    }

    return WaitAll(ctx, signals...)
}

// Wait for all health checks to pass
func WaitForAllHealthy(ctx context.Context, services []HealthChecker) error {
    signals := make([]<-chan struct{}, len(services))

    for i, svc := range services {
        healthy := make(chan struct{})
        signals[i] = healthy

        go func(s HealthChecker, h chan struct{}) {
            ticker := time.NewTicker(time.Second)
            defer ticker.Stop()
            defer close(h)

            for range ticker.C {
                if s.IsHealthy() {
                    return
                }
            }
        }(svc, healthy)
    }

    return WaitAll(ctx, signals...)
}

// Graceful shutdown all components
func ShutdownAllComponents(ctx context.Context, components []Component) error {
    signals := make([]<-chan struct{}, len(components))

    for i, comp := range components {
        done := make(chan struct{})
        signals[i] = done

        go func(c Component, d chan struct{}) {
            defer close(d)
            c.Shutdown()
        }(comp, done)
    }

    return WaitAll(ctx, signals...)
}

// Parallel test execution
func RunAllTests(ctx context.Context, tests []TestCase) error {
    signals := make([]<-chan struct{}, len(tests))

    for i, test := range tests {
        done := make(chan struct{})
        signals[i] = done

        go func(t TestCase, d chan struct{}) {
            defer close(d)
            t.Run()
        }(test, done)
    }

    return WaitAll(ctx, signals...)
}
\`\`\`

**Real-World Benefits:**
- **Parallel Efficiency:** All tasks run concurrently
- **Complete Visibility:** Know when ALL tasks finish
- **Timeout Control:** Don't wait forever for slow tasks
- **Error Handling:** Can cancel all on first failure

**Common Use Cases:**
- **Service Startup:** Wait for all services to be ready
- **Batch Processing:** Wait for all items to be processed
- **API Fan-out:** Call multiple APIs in parallel
- **Database Operations:** Run multiple queries concurrently
- **File Processing:** Process multiple files in parallel
- **Health Checks:** Wait for all services to be healthy

**vs WaitAny:**
- **WaitAll:** Need ALL operations to complete
- **WaitAny:** Need ANY operation to complete
- Use WaitAll for batch completeness
- Use WaitAny for first-responder pattern

Without WaitAll, coordinating multiple concurrent operations with timeout protection requires complex synchronization code scattered throughout the application.`,
	testCode: `package concurrency

import (
	"context"
	"errors"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	err := WaitAll(context.Background())
	if err != nil { t.Errorf("expected nil for no signals, got %v", err) }
}

func Test2(t *testing.T) {
	sig := make(chan struct{})
	close(sig)
	err := WaitAll(context.Background(), sig)
	if err != nil { t.Errorf("expected nil for single closed signal, got %v", err) }
}

func Test3(t *testing.T) {
	sig1, sig2, sig3 := make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() { close(sig1) }()
	go func() { close(sig2) }()
	go func() { close(sig3) }()
	err := WaitAll(context.Background(), sig1, sig2, sig3)
	if err != nil { t.Errorf("expected nil for all closed signals, got %v", err) }
}

func Test4(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	sig := make(chan struct{})
	err := WaitAll(ctx, sig)
	if !errors.Is(err, context.DeadlineExceeded) { t.Errorf("expected DeadlineExceeded, got %v", err) }
}

func Test5(t *testing.T) {
	sig := make(chan struct{})
	close(sig)
	err := WaitAll(nil, sig)
	if err != nil { t.Errorf("expected nil for nil context with closed signal, got %v", err) }
}

func Test6(t *testing.T) {
	err := WaitAll(nil)
	if err != nil { t.Errorf("expected nil for nil context with no signals, got %v", err) }
}

func Test7(t *testing.T) {
	sig1, sig2 := make(chan struct{}), make(chan struct{})
	go func() { time.Sleep(10*time.Millisecond); close(sig1) }()
	go func() { time.Sleep(20*time.Millisecond); close(sig2) }()
	start := time.Now()
	_ = WaitAll(context.Background(), sig1, sig2)
	if time.Since(start) < 15*time.Millisecond { t.Error("should wait for all signals") }
}

func Test8(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	sig := make(chan struct{})
	err := WaitAll(ctx, sig)
	if !errors.Is(err, context.Canceled) { t.Errorf("expected Canceled, got %v", err) }
}

func Test9(t *testing.T) {
	sig1 := make(chan struct{})
	sig2 := make(chan struct{})
	close(sig1)
	go func() { time.Sleep(30*time.Millisecond); close(sig2) }()
	start := time.Now()
	_ = WaitAll(context.Background(), sig1, sig2)
	if time.Since(start) < 20*time.Millisecond { t.Error("should wait for second signal too") }
}

func Test10(t *testing.T) {
	signals := make([]<-chan struct{}, 50)
	for i := range signals {
		ch := make(chan struct{})
		close(ch)
		signals[i] = ch
	}
	err := WaitAll(context.Background(), signals...)
	if err != nil { t.Errorf("expected nil for many closed signals, got %v", err) }
}
`,
	order: 5,
	translations: {
		ru: {
			title: 'Ожидание завершения всех горутин',
			description: `Реализуйте **WaitAll**, который ждёт закрытия всех сигнальных каналов или отмены контекста.

**Требования:**
1. Создайте функцию \`WaitAll(ctx context.Context, signals ...<-chan struct{}) error\`
2. Обработайте nil context (используйте Background)
3. Обработайте пустые signals (верните nil сразу)
4. Используйте sync.WaitGroup для ожидания всех сигналов
5. Каждый сигнал ждёт в отдельной горутине
6. Верните ошибку контекста если контекст отменён до всех сигналов
7. Верните nil если получены все сигналы

**Пример:**
\`\`\`go
sig1 := make(chan struct{})
sig2 := make(chan struct{})
sig3 := make(chan struct{})

go func() {
    time.Sleep(100 * time.Millisecond)
    close(sig1)
}()
go func() {
    time.Sleep(150 * time.Millisecond)
    close(sig2)
}()
go func() {
    time.Sleep(200 * time.Millisecond)
    close(sig3)
}()

err := WaitAll(context.Background(), sig1, sig2, sig3)
// err = nil (все сигналы получены)

ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
defer cancel()

err = WaitAll(ctx, sig1, sig2, sig3)
// err = context.DeadlineExceeded (таймаут до всех сигналов)
\`\`\`

**Ограничения:**
- Должен использовать sync.WaitGroup
- Должен ждать все сигналы параллельно
- Должен обрабатывать отмену контекста`,
			hint1: `Используйте sync.WaitGroup с len(signals) счётчиком. Запустите горутину для каждого сигнала которая ждёт либо ctx.Done() либо сигнал.`,
			hint2: `Создайте done канал который закрывается когда WaitGroup завершается. Используйте select для гонки между ctx.Done() и done.`,
			whyItMatters: `WaitAll позволяет ждать завершения нескольких конкурентных операций с защитой от таймаута, необходим для координации параллельных задач.

**Почему Wait All:**
- **Параллельная координация:** Ждать завершения всех задач
- **Защита от таймаута:** Не ждать вечно если задачи зависли
- **Синхронизация ресурсов:** Знать когда все ресурсы готовы
- **Завершение batch:** Убедиться что весь batch завершён

**Production Pattern:**
\`\`\`go
// Ожидание запуска нескольких сервисов
func StartAllServices(ctx context.Context, services []*Service) error {
    signals := make([]<-chan struct{}, len(services))

    for i, svc := range services {
        ready := make(chan struct{})
        signals[i] = ready

        go func(s *Service, done chan struct{}) {
            s.Start()
            close(done)
        }(svc, ready)
    }

    return WaitAll(ctx, signals...)
}

// Ожидание завершения нескольких API вызовов
func FetchAllData(ctx context.Context, urls []string) ([]Response, error) {
    responses := make([]Response, len(urls))
    signals := make([]<-chan struct{}, len(urls))

    for i, url := range urls {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, u string, d chan struct{}) {
            defer close(d)
            responses[idx] = fetch(u)
        }(i, url, done)
    }

    if err := WaitAll(ctx, signals...); err != nil {
        return nil, err
    }

    return responses, nil
}

// Ожидание всех миграций БД
func RunAllMigrations(ctx context.Context, migrations []Migration) error {
    signals := make([]<-chan struct{}, len(migrations))

    for i, migration := range migrations {
        done := make(chan struct{})
        signals[i] = done

        go func(m Migration, d chan struct{}) {
            defer close(d)
            m.Apply()
        }(migration, done)
    }

    return WaitAll(ctx, signals...)
}

// Ожидание завершения обработки всеми воркерами
func ProcessBatch(ctx context.Context, items []Item, workers int) error {
    itemChan := make(chan Item, len(items))
    for _, item := range items {
        itemChan <- item
    }
    close(itemChan)

    signals := make([]<-chan struct{}, workers)

    for i := 0; i < workers; i++ {
        done := make(chan struct{})
        signals[i] = done

        go func(d chan struct{}) {
            defer close(d)
            for item := range itemChan {
                process(item)
            }
        }(done)
    }

    return WaitAll(ctx, signals...)
}

// Ожидание обработки всех файлов
func ProcessAllFiles(ctx context.Context, files []string) error {
    signals := make([]<-chan struct{}, len(files))

    for i, file := range files {
        done := make(chan struct{})
        signals[i] = done

        go func(f string, d chan struct{}) {
            defer close(d)
            processFile(f)
        }(file, done)
    }

    return WaitAll(ctx, signals...)
}

// Ожидание прогрева всех кешей
func WarmUpAllCaches(ctx context.Context, caches []*Cache) error {
    signals := make([]<-chan struct{}, len(caches))

    for i, cache := range caches {
        done := make(chan struct{})
        signals[i] = done

        go func(c *Cache, d chan struct{}) {
            defer close(d)
            c.WarmUp()
        }(cache, done)
    }

    return WaitAll(ctx, signals...)
}

// Ожидание прохождения всех health checks
func WaitForAllHealthy(ctx context.Context, services []HealthChecker) error {
    signals := make([]<-chan struct{}, len(services))

    for i, svc := range services {
        healthy := make(chan struct{})
        signals[i] = healthy

        go func(s HealthChecker, h chan struct{}) {
            ticker := time.NewTicker(time.Second)
            defer ticker.Stop()
            defer close(h)

            for range ticker.C {
                if s.IsHealthy() {
                    return
                }
            }
        }(svc, healthy)
    }

    return WaitAll(ctx, signals...)
}

// Graceful shutdown всех компонентов
func ShutdownAllComponents(ctx context.Context, components []Component) error {
    signals := make([]<-chan struct{}, len(components))

    for i, comp := range components {
        done := make(chan struct{})
        signals[i] = done

        go func(c Component, d chan struct{}) {
            defer close(d)
            c.Shutdown()
        }(comp, done)
    }

    return WaitAll(ctx, signals...)
}

// Параллельное выполнение тестов
func RunAllTests(ctx context.Context, tests []TestCase) error {
    signals := make([]<-chan struct{}, len(tests))

    for i, test := range tests {
        done := make(chan struct{})
        signals[i] = done

        go func(t TestCase, d chan struct{}) {
            defer close(d)
            t.Run()
        }(test, done)
    }

    return WaitAll(ctx, signals...)
}
\`\`\`

**Практические преимущества:**
- **Параллельная эффективность:** Все задачи выполняются конкурентно
- **Полная видимость:** Знать когда ВСЕ задачи завершены
- **Контроль таймаутов:** Не ждать вечно медленных задач
- **Обработка ошибок:** Можно отменить все при первой ошибке

**Типичные сценарии использования:**
- **Запуск сервисов:** Ждать готовности всех сервисов
- **Batch обработка:** Ждать обработки всех элементов
- **API Fan-out:** Параллельный вызов нескольких API
- **Операции с БД:** Конкурентное выполнение запросов
- **Обработка файлов:** Параллельная обработка файлов
- **Health Checks:** Ждать здоровья всех сервисов

**WaitAll vs WaitAny:**
- **WaitAll:** Нужно завершение ВСЕХ операций
- **WaitAny:** Нужно завершение ЛЮБОЙ операции
- WaitAll для полноты batch
- WaitAny для паттерна первого ответившего

Без WaitAll координация нескольких конкурентных операций с защитой от таймаута требует сложного кода синхронизации разбросанного по всему приложению.`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

func WaitAll(ctx context.Context, signals ...<-chan struct{}) error {
	if ctx == nil {                                             // Обработка nil контекста
		ctx = context.Background()                          // Используем Background как fallback
	}
	if len(signals) == 0 {                                      // Нет сигналов для ожидания
		return nil                                          // Возвращаемся сразу
	}
	done := make(chan struct{})                                 // Канал для сигнала о завершении всех
	go func() {                                                 // Запускаем ожидание в фоне
		defer close(done)                                   // Закрываем когда все сигналы получены
		var wg sync.WaitGroup                               // WaitGroup для всех сигналов
		wg.Add(len(signals))                                // Добавляем все сигналы в группу
		for _, ch := range signals {                        // Итерируем по всем сигналам
			ch := ch                                    // Захватываем переменную цикла
			go func() {                                 // Ждём каждый сигнал в горутине
				defer wg.Done()                     // Отмечаем этот сигнал как завершённый
				select {
				case <-ctx.Done():              // Контекст отменён
				case <-ch:                      // Сигнал получен
				}
			}()
		}
		wg.Wait()                                           // Ждём все сигналы
	}()
	select {
	case <-ctx.Done():                                          // Контекст отменён
		return ctx.Err()                                    // Возвращаем ошибку контекста
	case <-done:                                                // Все сигналы получены
		return ctx.Err()                                    // Возвращаем nil или ошибку контекста
	}
}`
		},
		uz: {
			title: 'Barcha goroutinalarning tugashini kutish',
			description: `Barcha signal kanallarining yopilishini yoki kontekstning bekor qilinishini kutadigan **WaitAll** ni amalga oshiring.

**Talablar:**
1. \`WaitAll(ctx context.Context, signals ...<-chan struct{}) error\` funksiyasini yarating
2. nil kontekstni ishlang (Background dan foydalaning)
3. Bo'sh signallarni ishlang (darhol nil qaytaring)
4. Barcha signallarni kutish uchun sync.WaitGroup dan foydalaning
5. Har bir signal alohida goroutinada kutadi
6. Agar kontekst barcha signallardan oldin bekor qilinsa kontekst xatosini qaytaring
7. Agar barcha signallar olinsa nil qaytaring

**Misol:**
\`\`\`go
sig1 := make(chan struct{})
sig2 := make(chan struct{})
sig3 := make(chan struct{})

go func() {
    time.Sleep(100 * time.Millisecond)
    close(sig1)
}()
go func() {
    time.Sleep(150 * time.Millisecond)
    close(sig2)
}()
go func() {
    time.Sleep(200 * time.Millisecond)
    close(sig3)
}()

err := WaitAll(context.Background(), sig1, sig2, sig3)
// err = nil (barcha signallar olindi)

ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
defer cancel()

err = WaitAll(ctx, sig1, sig2, sig3)
// err = context.DeadlineExceeded (barcha signallardan oldin timeout)
\`\`\`

**Cheklovlar:**
- sync.WaitGroup dan foydalanishi kerak
- Barcha signallarni parallel kutishi kerak
- Kontekst bekor qilishni ishlashi kerak`,
			hint1: `len(signals) hisoblagichi bilan sync.WaitGroup dan foydalaning. Har bir signal uchun ctx.Done() yoki signalni kutadigan goroutina ishga tushiring.`,
			hint2: `WaitGroup tugaganda yopiladigan done kanali yarating. ctx.Done() va done o'rtasida poyga uchun select dan foydalaning.`,
			whyItMatters: `WaitAll timeout himoyasi bilan bir nechta parallel operatsiyalarning tugashini kutishni yoqadi, parallel vazifalarni muvofiqlashtirish uchun zarur.

**Nima uchun Wait All:**
- **Parallel koordinatsiya:** Barcha vazifalar tugashini kutish
- **Timeout himoyasi:** Agar vazifalar osilib qolsa abadiy kutmaslik
- **Resurslarni sinxronlashtirish:** Barcha resurslar tayyor bo'lganda bilish
- **Batch tugashi:** Butun batch tugallanganligini ta'minlash

**Production patternlar:**
\`\`\`go
// Ko'p xizmatlarni ishga tushirishni kutish
func StartAllServices(ctx context.Context, services []*Service) error {
    signals := make([]<-chan struct{}, len(services))

    for i, svc := range services {
        ready := make(chan struct{})
        signals[i] = ready

        go func(s *Service, done chan struct{}) {
            s.Start()
            close(done)
        }(svc, ready)
    }

    return WaitAll(ctx, signals...)
}

// Ko'p API chaqiruvlarining tugashini kutish
func FetchAllData(ctx context.Context, urls []string) ([]Response, error) {
    responses := make([]Response, len(urls))
    signals := make([]<-chan struct{}, len(urls))

    for i, url := range urls {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, u string, d chan struct{}) {
            defer close(d)
            responses[idx] = fetch(u)
        }(i, url, done)
    }

    if err := WaitAll(ctx, signals...); err != nil {
        return nil, err
    }

    return responses, nil
}

// Barcha ma'lumotlar bazasi migratsiyalarini kutish
func RunAllMigrations(ctx context.Context, migrations []Migration) error {
    signals := make([]<-chan struct{}, len(migrations))

    for i, migration := range migrations {
        done := make(chan struct{})
        signals[i] = done

        go func(m Migration, d chan struct{}) {
            defer close(d)
            m.Apply()
        }(migration, done)
    }

    return WaitAll(ctx, signals...)
}

// Barcha workerlarning qayta ishlashini tugatishini kutish
func ProcessBatch(ctx context.Context, items []Item, workers int) error {
    itemChan := make(chan Item, len(items))
    for _, item := range items {
        itemChan <- item
    }
    close(itemChan)

    signals := make([]<-chan struct{}, workers)

    for i := 0; i < workers; i++ {
        done := make(chan struct{})
        signals[i] = done

        go func(d chan struct{}) {
            defer close(d)
            for item := range itemChan {
                process(item)
            }
        }(done)
    }

    return WaitAll(ctx, signals...)
}

// Barcha fayllarning qayta ishlanishini kutish
func ProcessAllFiles(ctx context.Context, files []string) error {
    signals := make([]<-chan struct{}, len(files))

    for i, file := range files {
        done := make(chan struct{})
        signals[i] = done

        go func(f string, d chan struct{}) {
            defer close(d)
            processFile(f)
        }(file, done)
    }

    return WaitAll(ctx, signals...)
}

// Barcha keshlarni isitishni kutish
func WarmUpAllCaches(ctx context.Context, caches []*Cache) error {
    signals := make([]<-chan struct{}, len(caches))

    for i, cache := range caches {
        done := make(chan struct{})
        signals[i] = done

        go func(c *Cache, d chan struct{}) {
            defer close(d)
            c.WarmUp()
        }(cache, done)
    }

    return WaitAll(ctx, signals...)
}

// Barcha health checklar o'tishini kutish
func WaitForAllHealthy(ctx context.Context, services []HealthChecker) error {
    signals := make([]<-chan struct{}, len(services))

    for i, svc := range services {
        healthy := make(chan struct{})
        signals[i] = healthy

        go func(s HealthChecker, h chan struct{}) {
            ticker := time.NewTicker(time.Second)
            defer ticker.Stop()
            defer close(h)

            for range ticker.C {
                if s.IsHealthy() {
                    return
                }
            }
        }(svc, healthy)
    }

    return WaitAll(ctx, signals...)
}

// Barcha komponentlarni graceful shutdown
func ShutdownAllComponents(ctx context.Context, components []Component) error {
    signals := make([]<-chan struct{}, len(components))

    for i, comp := range components {
        done := make(chan struct{})
        signals[i] = done

        go func(c Component, d chan struct{}) {
            defer close(d)
            c.Shutdown()
        }(comp, done)
    }

    return WaitAll(ctx, signals...)
}

// Parallel test bajarish
func RunAllTests(ctx context.Context, tests []TestCase) error {
    signals := make([]<-chan struct{}, len(tests))

    for i, test := range tests {
        done := make(chan struct{})
        signals[i] = done

        go func(t TestCase, d chan struct{}) {
            defer close(d)
            t.Run()
        }(test, done)
    }

    return WaitAll(ctx, signals...)
}
\`\`\`

**Haqiqiy foydalari:**
- **Parallel samaradorlik:** Barcha vazifalar bir vaqtda ishlaydi
- **To'liq ko'rinish:** BARCHA vazifalar tugaganda bilish
- **Timeout nazorati:** Sekin vazifalarni abadiy kutmaslik
- **Xatolarni qayta ishlash:** Birinchi xatoda barchasini bekor qilish mumkin

**Umumiy foydalanish holatlari:**
- **Xizmatlarni ishga tushirish:** Barcha xizmatlar tayyor bo'lishini kutish
- **Batch qayta ishlash:** Barcha elementlarning qayta ishlanishini kutish
- **API Fan-out:** Ko'p APIlarni parallel chaqirish
- **Ma'lumotlar bazasi operatsiyalari:** Ko'p so'rovlarni parallel bajarish
- **Fayl qayta ishlash:** Ko'p fayllarni parallel qayta ishlash
- **Health Checks:** Barcha xizmatlarning sog'lom bo'lishini kutish

**WaitAll vs WaitAny:**
- **WaitAll:** BARCHA operatsiyalarning tugashi kerak
- **WaitAny:** ISTALGAN operatsiyaning tugashi kerak
- Batch to'liqligi uchun WaitAll dan foydalaning
- Birinchi javob beruvchi pattern uchun WaitAny dan foydalaning

WaitAll bo'lmasa, timeout himoyasi bilan ko'p parallel operatsiyalarni muvofiqlashtirish butun ilova bo'ylab tarqalgan murakkab sinxronlashtirish kodini talab qiladi.`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

func WaitAll(ctx context.Context, signals ...<-chan struct{}) error {
	if ctx == nil {                                             // nil kontekstni ishlash
		ctx = context.Background()                          // Fallback sifatida Background ishlatamiz
	}
	if len(signals) == 0 {                                      // Kutish uchun signallar yo'q
		return nil                                          // Darhol qaytamiz
	}
	done := make(chan struct{})                                 // Hammasi tugaganligi signali uchun kanal
	go func() {                                                 // Fonda kutishni ishga tushiramiz
		defer close(done)                                   // Barcha signallar olinganda yopamiz
		var wg sync.WaitGroup                               // Barcha signallar uchun WaitGroup
		wg.Add(len(signals))                                // Barcha signallarni guruhga qo'shamiz
		for _, ch := range signals {                        // Barcha signallar bo'yicha iteratsiya
			ch := ch                                    // Sikl o'zgaruvchisini ushlash
			go func() {                                 // Har bir signalni goroutinada kutamiz
				defer wg.Done()                     // Bu signalni tugallangan deb belgilaymiz
				select {
				case <-ctx.Done():              // Kontekst bekor qilindi
				case <-ch:                      // Signal olindi
				}
			}()
		}
		wg.Wait()                                           // Barcha signallarni kutamiz
	}()
	select {
	case <-ctx.Done():                                          // Kontekst bekor qilindi
		return ctx.Err()                                    // Kontekst xatosini qaytaramiz
	case <-done:                                                // Barcha signallar olindi
		return ctx.Err()                                    // nil yoki kontekst xatosini qaytaramiz
	}
}`
		}
	}
};

export default task;
