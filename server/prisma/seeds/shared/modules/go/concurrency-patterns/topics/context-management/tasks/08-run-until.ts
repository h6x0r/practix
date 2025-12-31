import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-concurrency-run-until',
	title: 'Run Until',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'context', 'loop'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **RunUntil** that repeatedly executes function f until context is canceled or done signal is received.

**Requirements:**
1. Create function \`RunUntil(ctx context.Context, done <-chan struct{}, f func()) error\`
2. Handle nil context (use Background)
3. Run f repeatedly in a loop
4. Check for ctx.Done() and done channel in select with default
5. Execute f() in default case (non-blocking)
6. Return context error if context canceled
7. Return nil if done signal received

**Example:**
\`\`\`go
counter := 0
done := make(chan struct{})

go func() {
    time.Sleep(100 * time.Millisecond)
    close(done)
}()

err := RunUntil(context.Background(), done, func() {
    counter++
    time.Sleep(10 * time.Millisecond)
})
// err = nil, counter ≈ 10

ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
defer cancel()

counter = 0
err = RunUntil(ctx, make(chan struct{}), func() {
    counter++
    time.Sleep(10 * time.Millisecond)
})
// err = context.DeadlineExceeded, counter ≈ 5
\`\`\`

**Constraints:**
- Must check cancellation in each iteration
- Must use select with default
- Must not block between iterations`,
	initialCode: `package concurrency

import (
	"context"
)

// TODO: Implement RunUntil
func RunUntil(ctx context.Context, done <-chan struct{}, f func()) error {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
)

func RunUntil(ctx context.Context, done <-chan struct{}, f func()) error {
	if ctx == nil {                                             // Handle nil context
		ctx = context.Background()                          // Use Background as fallback
	}
	for {                                                       // Infinite loop
		select {
		case <-ctx.Done():                                  // Context canceled
			return ctx.Err()                            // Return context error
		case <-done:                                        // Done signal received
			return nil                                  // Return success
		default:                                            // No cancellation yet
		}
		f()                                                 // Execute function
	}
}`,
			hint1: `Use an infinite for loop with select statement checking ctx.Done() and done. Use default case to execute f() when no cancellation.`,
			hint2: `The select with default makes it non-blocking. Check cancellation first, then execute f() if no cancel signal.`,
			whyItMatters: `RunUntil provides a controlled loop that can be canceled at any time, essential for background workers and periodic tasks.

**Why Controlled Loops:**
- **Graceful Shutdown:** Stop loops cleanly
- **Resource Control:** Cancel background work
- **Responsive:** Check cancellation frequently
- **Flexibility:** External control via done channel

**Production Pattern:**
\`\`\`go
// Background metrics collector
func CollectMetrics(ctx context.Context, collector *MetricsCollector) error {
    done := make(chan struct{})

    go func() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()
        <-ticker.C
        close(done)
    }()

    return RunUntil(ctx, done, func() {
        metrics := collector.Collect()
        collector.Send(metrics)
        time.Sleep(time.Second)
    })
}

// Process messages from queue
func ProcessQueue(ctx context.Context, queue *Queue) error {
    return RunUntil(ctx, queue.Closed(), func() {
        msg := queue.Poll()
        if msg != nil {
            processMessage(msg)
        } else {
            time.Sleep(100 * time.Millisecond)
        }
    })
}

// Health check loop
func HealthCheckLoop(ctx context.Context, checker HealthChecker) error {
    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        if !checker.Check() {
            log.Println("Health check failed")
        }
        time.Sleep(5 * time.Second)
    })
}

// Cache refresh loop
func RefreshCache(ctx context.Context, cache *Cache, interval time.Duration) error {
    ticker := time.NewTicker(interval)
    defer ticker.Stop()

    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        select {
        case <-ticker.C:
            cache.Refresh()
        default:
            time.Sleep(100 * time.Millisecond)
        }
    })
}

// Log file watcher
func WatchLogFile(ctx context.Context, path string) error {
    file, _ := os.Open(path)
    defer file.Close()

    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        line := readLine(file)
        if line != "" {
            processLogLine(line)
        } else {
            time.Sleep(100 * time.Millisecond)
        }
    })
}

// Connection pool maintainer
func MaintainPool(ctx context.Context, pool *ConnectionPool) error {
    return RunUntil(ctx, pool.Closed(), func() {
        pool.RemoveStale()
        pool.CreateNew()
        time.Sleep(30 * time.Second)
    })
}

// Background data sync
func SyncData(ctx context.Context, syncer *DataSyncer) error {
    done := make(chan struct{})

    go func() {
        <-syncer.CompletedSignal()
        close(done)
    }()

    return RunUntil(ctx, done, func() {
        batch := syncer.GetNextBatch()
        if len(batch) > 0 {
            syncer.Sync(batch)
        } else {
            time.Sleep(time.Second)
        }
    })
}

// Garbage collector
func GarbageCollect(ctx context.Context, threshold time.Duration) error {
    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        removeOldItems(threshold)
        time.Sleep(time.Minute)
    })
}

// Event processor
func ProcessEvents(ctx context.Context, events <-chan Event) error {
    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        select {
        case event := <-events:
            handleEvent(event)
        default:
            time.Sleep(10 * time.Millisecond)
        }
    })
}

// Rate-limited API poller
func PollAPI(ctx context.Context, endpoint string, rate time.Duration) error {
    ticker := time.NewTicker(rate)
    defer ticker.Stop()

    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        select {
        case <-ticker.C:
            response := callAPI(endpoint)
            processResponse(response)
        default:
            time.Sleep(100 * time.Millisecond)
        }
    })
}

// Worker with controllable loop
type Worker struct {
    done chan struct{}
}

func (w *Worker) Run(ctx context.Context) error {
    return RunUntil(ctx, w.done, func() {
        work := getWork()
        processWork(work)
    })
}

func (w *Worker) Stop() {
    close(w.done)
}
\`\`\`

**Real-World Benefits:**
- **Clean Shutdown:** Stop loops immediately on cancel
- **No Zombie Loops:** All loops respect context
- **External Control:** Stop via done channel or context
- **Predictable:** Consistent cancellation pattern

**Common Use Cases:**
- **Background Workers:** Process tasks until stopped
- **Metrics Collection:** Collect until service stops
- **Queue Processing:** Process messages until queue closed
- **Health Checks:** Check health until canceled
- **Cache Refresh:** Refresh periodically until stopped
- **File Watching:** Watch files until done
- **Connection Maintenance:** Maintain pool until closed

**vs Traditional for Loop:**
- **Traditional:** Hard to cancel cleanly
- **RunUntil:** Cancellation built-in
- **RunUntil:** Context-aware
- **RunUntil:** Consistent pattern

**Best Practices:**
- **Check Frequently:** Don't let f() run too long
- **Sleep in f():** Give CPU time to other goroutines
- **Use Ticker:** For periodic work
- **Defer Cleanup:** Always clean up resources

Without RunUntil, every background loop needs custom cancellation logic, leading to inconsistent and error-prone patterns.`,
	testCode: `package concurrency

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	done := make(chan struct{})
	close(done)
	err := RunUntil(context.Background(), done, func() {})
	if err != nil { t.Errorf("expected nil for closed done channel, got %v", err) }
}

func Test2(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	done := make(chan struct{})
	err := RunUntil(ctx, done, func() { time.Sleep(10*time.Millisecond) })
	if !errors.Is(err, context.DeadlineExceeded) { t.Errorf("expected DeadlineExceeded, got %v", err) }
}

func Test3(t *testing.T) {
	done := make(chan struct{})
	close(done)
	err := RunUntil(nil, done, func() {})
	if err != nil { t.Errorf("expected nil for nil context with closed done, got %v", err) }
}

func Test4(t *testing.T) {
	var counter int64
	done := make(chan struct{})
	go func() { time.Sleep(100*time.Millisecond); close(done) }()
	_ = RunUntil(context.Background(), done, func() {
		atomic.AddInt64(&counter, 1)
		time.Sleep(20*time.Millisecond)
	})
	if atomic.LoadInt64(&counter) < 3 { t.Errorf("expected at least 3 iterations, got %d", counter) }
}

func Test5(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	done := make(chan struct{})
	err := RunUntil(ctx, done, func() {})
	if !errors.Is(err, context.Canceled) { t.Errorf("expected Canceled, got %v", err) }
}

func Test6(t *testing.T) {
	var counter int64
	done := make(chan struct{})
	go func() { time.Sleep(50*time.Millisecond); close(done) }()
	_ = RunUntil(context.Background(), done, func() { atomic.AddInt64(&counter, 1) })
	if atomic.LoadInt64(&counter) < 1 { t.Error("expected at least one iteration") }
}

func Test7(t *testing.T) {
	done := make(chan struct{})
	close(done)
	executed := false
	_ = RunUntil(context.Background(), done, func() { executed = true })
	if executed { t.Error("should not execute if done is closed immediately") }
}

func Test8(t *testing.T) {
	start := time.Now()
	done := make(chan struct{})
	go func() { time.Sleep(50*time.Millisecond); close(done) }()
	_ = RunUntil(context.Background(), done, func() { time.Sleep(10*time.Millisecond) })
	elapsed := time.Since(start)
	if elapsed < 40*time.Millisecond { t.Error("should run for at least 40ms") }
}

func Test9(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	done := make(chan struct{})
	go func() { time.Sleep(50*time.Millisecond); close(done) }()
	err := RunUntil(ctx, done, func() { time.Sleep(5*time.Millisecond) })
	if err != nil { t.Errorf("expected nil (done closed), got %v", err) }
}

func Test10(t *testing.T) {
	var counter int64
	done := make(chan struct{})
	go func() { time.Sleep(30*time.Millisecond); close(done) }()
	_ = RunUntil(context.Background(), done, func() { atomic.AddInt64(&counter, 1) })
	if atomic.LoadInt64(&counter) > 1000 { t.Error("function should have yielded control") }
}
`,
	order: 7,
	translations: {
		ru: {
			title: 'Выполнение до условия',
			description: `Реализуйте **RunUntil**, который повторно выполняет функцию f пока контекст не отменён или не получен сигнал done.

**Требования:**
1. Создайте функцию \`RunUntil(ctx context.Context, done <-chan struct{}, f func()) error\`
2. Обработайте nil context (используйте Background)
3. Запускайте f повторно в цикле
4. Проверяйте ctx.Done() и done канал в select с default
5. Выполняйте f() в default case (без блокировки)
6. Верните ошибку контекста если контекст отменён
7. Верните nil если получен сигнал done

**Пример:**
\`\`\`go
counter := 0
done := make(chan struct{})

go func() {
    time.Sleep(100 * time.Millisecond)
    close(done)
}()

err := RunUntil(context.Background(), done, func() {
    counter++
    time.Sleep(10 * time.Millisecond)
})
// err = nil, counter ≈ 10

ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
defer cancel()

counter = 0
err = RunUntil(ctx, make(chan struct{}), func() {
    counter++
    time.Sleep(10 * time.Millisecond)
})
// err = context.DeadlineExceeded, counter ≈ 5
\`\`\`

**Ограничения:**
- Должен проверять отмену в каждой итерации
- Должен использовать select с default
- Не должен блокироваться между итерациями`,
			hint1: `Используйте бесконечный for цикл с select statement проверяющим ctx.Done() и done. Используйте default case для выполнения f() когда нет отмены.`,
			hint2: `Select с default делает его неблокирующим. Проверьте отмену сначала, затем выполните f() если нет сигнала отмены.`,
			whyItMatters: `RunUntil предоставляет контролируемый цикл который можно отменить в любое время, необходим для фоновых workers и периодических задач.

**Почему Controlled Loops критичны:**
- **Graceful Shutdown:** Чистая остановка циклов
- **Контроль ресурсов:** Отмена фоновой работы
- **Отзывчивость:** Частая проверка отмены
- **Гибкость:** Внешний контроль через done канал

**Production паттерны:**
\`\`\`go
// Фоновый сборщик метрик
func CollectMetrics(ctx context.Context, collector *MetricsCollector) error {
    done := make(chan struct{})

    go func() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()
        <-ticker.C
        close(done)
    }()

    return RunUntil(ctx, done, func() {
        metrics := collector.Collect()
        collector.Send(metrics)
        time.Sleep(time.Second)
    })
}

// Обработка сообщений из очереди
func ProcessQueue(ctx context.Context, queue *Queue) error {
    return RunUntil(ctx, queue.Closed(), func() {
        msg := queue.Poll()
        if msg != nil {
            processMessage(msg)
        } else {
            time.Sleep(100 * time.Millisecond)
        }
    })
}

// Health check цикл
func HealthCheckLoop(ctx context.Context, checker HealthChecker) error {
    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        if !checker.Check() {
            log.Println("Health check failed")
        }
        time.Sleep(5 * time.Second)
    })
}

// Обновление кеша
func RefreshCache(ctx context.Context, cache *Cache, interval time.Duration) error {
    ticker := time.NewTicker(interval)
    defer ticker.Stop()

    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        select {
        case <-ticker.C:
            cache.Refresh()
        default:
            time.Sleep(100 * time.Millisecond)
        }
    })
}

// Наблюдение за файлами логов
func WatchLogFile(ctx context.Context, path string) error {
    file, _ := os.Open(path)
    defer file.Close()

    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        line := readLine(file)
        if line != "" {
            processLogLine(line)
        } else {
            time.Sleep(100 * time.Millisecond)
        }
    })
}

// Поддержка пула соединений
func MaintainPool(ctx context.Context, pool *ConnectionPool) error {
    return RunUntil(ctx, pool.Closed(), func() {
        pool.RemoveStale()
        pool.CreateNew()
        time.Sleep(30 * time.Second)
    })
}

// Фоновая синхронизация данных
func SyncData(ctx context.Context, syncer *DataSyncer) error {
    done := make(chan struct{})

    go func() {
        <-syncer.CompletedSignal()
        close(done)
    }()

    return RunUntil(ctx, done, func() {
        batch := syncer.GetNextBatch()
        if len(batch) > 0 {
            syncer.Sync(batch)
        } else {
            time.Sleep(time.Second)
        }
    })
}

// Сборщик мусора
func GarbageCollect(ctx context.Context, threshold time.Duration) error {
    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        removeOldItems(threshold)
        time.Sleep(time.Minute)
    })
}

// Обработчик событий
func ProcessEvents(ctx context.Context, events <-chan Event) error {
    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        select {
        case event := <-events:
            handleEvent(event)
        default:
            time.Sleep(10 * time.Millisecond)
        }
    })
}

// API опрос с ограничением скорости
func PollAPI(ctx context.Context, endpoint string, rate time.Duration) error {
    ticker := time.NewTicker(rate)
    defer ticker.Stop()

    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        select {
        case <-ticker.C:
            response := callAPI(endpoint)
            processResponse(response)
        default:
            time.Sleep(100 * time.Millisecond)
        }
    })
}

// Worker с контролируемым циклом
type Worker struct {
    done chan struct{}
}

func (w *Worker) Run(ctx context.Context) error {
    return RunUntil(ctx, w.done, func() {
        work := getWork()
        processWork(work)
    })
}

func (w *Worker) Stop() {
    close(w.done)
}
\`\`\`

**Реальные преимущества:**
- **Чистое завершение:** Циклы останавливаются немедленно при отмене
- **Нет зомби-циклов:** Все циклы уважают контекст
- **Внешний контроль:** Остановка через done канал или контекст
- **Предсказуемость:** Согласованный паттерн отмены

**Типичные сценарии использования:**
- **Фоновые Workers:** Обработка задач до остановки
- **Сбор метрик:** Сбор до остановки сервиса
- **Обработка очередей:** Обработка сообщений до закрытия очереди
- **Health Checks:** Проверка здоровья до отмены
- **Обновление кеша:** Периодическое обновление до остановки
- **Наблюдение за файлами:** Наблюдение до завершения
- **Поддержка соединений:** Поддержка пула до закрытия

**vs Традиционный for Loop:**
- **Традиционный:** Сложно чисто отменить
- **RunUntil:** Отмена встроена
- **RunUntil:** Context-aware
- **RunUntil:** Согласованный паттерн

**Лучшие практики:**
- **Частые проверки:** Не позволяйте f() выполняться слишком долго
- **Sleep в f():** Давайте процессорное время другим горутинам
- **Используйте Ticker:** Для периодической работы
- **Defer Cleanup:** Всегда очищайте ресурсы

Без RunUntil каждый фоновый цикл нуждается в собственной логике отмены, приводя к несогласованным и подверженным ошибкам паттернам.`,
			solutionCode: `package concurrency

import (
	"context"
)

func RunUntil(ctx context.Context, done <-chan struct{}, f func()) error {
	if ctx == nil {                                             // Обработка nil контекста
		ctx = context.Background()                          // Используем Background как fallback
	}
	for {                                                       // Бесконечный цикл
		select {
		case <-ctx.Done():                                  // Контекст отменён
			return ctx.Err()                            // Возвращаем ошибку контекста
		case <-done:                                        // Получен сигнал done
			return nil                                  // Возвращаем успех
		default:                                            // Отмены пока нет
		}
		f()                                                 // Выполняем функцию
	}
}`
		},
		uz: {
			title: 'Shartgacha bajarish',
			description: `Kontekst bekor qilinmaguncha yoki done signali olinmaguncha f funksiyasini takroriy bajariladigan **RunUntil** ni amalga oshiring.

**Talablar:**
1. \`RunUntil(ctx context.Context, done <-chan struct{}, f func()) error\` funksiyasini yarating
2. nil kontekstni ishlang (Background dan foydalaning)
3. f ni siklda takroriy ishga tushiring
4. select da default bilan ctx.Done() va done kanalini tekshiring
5. f() ni default caseda bajaring (bloklanmasdan)
6. Agar kontekst bekor qilinsa kontekst xatosini qaytaring
7. Agar done signali olinsa nil qaytaring

**Misol:**
\`\`\`go
counter := 0
done := make(chan struct{})

go func() {
    time.Sleep(100 * time.Millisecond)
    close(done)
}()

err := RunUntil(context.Background(), done, func() {
    counter++
    time.Sleep(10 * time.Millisecond)
})
// err = nil, counter ≈ 10

ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
defer cancel()

counter = 0
err = RunUntil(ctx, make(chan struct{}), func() {
    counter++
    time.Sleep(10 * time.Millisecond)
})
// err = context.DeadlineExceeded, counter ≈ 5
\`\`\`

**Cheklovlar:**
- Har bir iteratsiyada bekor qilinishni tekshirishi kerak
- default bilan select dan foydalanishi kerak
- Iteratsiyalar o'rtasida bloklanmasligi kerak`,
			hint1: `ctx.Done() va done ni tekshiradigan select statement bilan cheksiz for siklidan foydalaning. Bekor qilish bo'lmaganda f() ni bajarish uchun default casedan foydalaning.`,
			hint2: `default bilan select uni bloklanmaydigan qiladi. Avval bekor qilinishni tekshiring, keyin bekor qilish signali bo'lmasa f() ni bajaring.`,
			whyItMatters: `RunUntil istalgan vaqtda bekor qilinishi mumkin bo'lgan boshqariladigan siklni ta'minlaydi, background workerlar va davriy vazifalar uchun zarur.

**Nima uchun Controlled Loops muhim:**
- **Graceful Shutdown:** Sikllarni toza to'xtatish
- **Resurslarni boshqarish:** Background ishlarni bekor qilish
- **Tezkor javob:** Bekor qilinishni tez-tez tekshirish
- **Moslashuvchanlik:** done kanali orqali tashqi boshqarish

**Production patternlar:**
\`\`\`go
// Background metrikalarni yig'ish
func CollectMetrics(ctx context.Context, collector *MetricsCollector) error {
    done := make(chan struct{})

    go func() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()
        <-ticker.C
        close(done)
    }()

    return RunUntil(ctx, done, func() {
        metrics := collector.Collect()
        collector.Send(metrics)
        time.Sleep(time.Second)
    })
}

// Navbatdan xabarlarni qayta ishlash
func ProcessQueue(ctx context.Context, queue *Queue) error {
    return RunUntil(ctx, queue.Closed(), func() {
        msg := queue.Poll()
        if msg != nil {
            processMessage(msg)
        } else {
            time.Sleep(100 * time.Millisecond)
        }
    })
}

// Health check sikli
func HealthCheckLoop(ctx context.Context, checker HealthChecker) error {
    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        if !checker.Check() {
            log.Println("Health check muvaffaqiyatsiz")
        }
        time.Sleep(5 * time.Second)
    })
}

// Keshni yangilash
func RefreshCache(ctx context.Context, cache *Cache, interval time.Duration) error {
    ticker := time.NewTicker(interval)
    defer ticker.Stop()

    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        select {
        case <-ticker.C:
            cache.Refresh()
        default:
            time.Sleep(100 * time.Millisecond)
        }
    })
}

// Log fayllarni kuzatish
func WatchLogFile(ctx context.Context, path string) error {
    file, _ := os.Open(path)
    defer file.Close()

    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        line := readLine(file)
        if line != "" {
            processLogLine(line)
        } else {
            time.Sleep(100 * time.Millisecond)
        }
    })
}

// Ulanish poolini saqlash
func MaintainPool(ctx context.Context, pool *ConnectionPool) error {
    return RunUntil(ctx, pool.Closed(), func() {
        pool.RemoveStale()
        pool.CreateNew()
        time.Sleep(30 * time.Second)
    })
}

// Background ma'lumotlarni sinxronlash
func SyncData(ctx context.Context, syncer *DataSyncer) error {
    done := make(chan struct{})

    go func() {
        <-syncer.CompletedSignal()
        close(done)
    }()

    return RunUntil(ctx, done, func() {
        batch := syncer.GetNextBatch()
        if len(batch) > 0 {
            syncer.Sync(batch)
        } else {
            time.Sleep(time.Second)
        }
    })
}

// Axlat yig'ish
func GarbageCollect(ctx context.Context, threshold time.Duration) error {
    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        removeOldItems(threshold)
        time.Sleep(time.Minute)
    })
}

// Hodisalarni qayta ishlash
func ProcessEvents(ctx context.Context, events <-chan Event) error {
    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        select {
        case event := <-events:
            handleEvent(event)
        default:
            time.Sleep(10 * time.Millisecond)
        }
    })
}

// Tezlik cheklangan API so'rovi
func PollAPI(ctx context.Context, endpoint string, rate time.Duration) error {
    ticker := time.NewTicker(rate)
    defer ticker.Stop()

    done := make(chan struct{})

    return RunUntil(ctx, done, func() {
        select {
        case <-ticker.C:
            response := callAPI(endpoint)
            processResponse(response)
        default:
            time.Sleep(100 * time.Millisecond)
        }
    })
}

// Boshqariladigan siklga ega Worker
type Worker struct {
    done chan struct{}
}

func (w *Worker) Run(ctx context.Context) error {
    return RunUntil(ctx, w.done, func() {
        work := getWork()
        processWork(work)
    })
}

func (w *Worker) Stop() {
    close(w.done)
}
\`\`\`

**Haqiqiy foydalari:**
- **Toza tugash:** Sikllar bekor qilganda darhol to'xtaydi
- **Zombi sikllar yo'q:** Barcha sikllar kontekstni hurmat qiladi
- **Tashqi boshqarish:** done kanali yoki kontekst orqali to'xtatish
- **Bashorat qilinadigan:** Izchil bekor qilish patterni

**Umumiy foydalanish stsenariylari:**
- **Background Workerlar:** Vazifalarni to'xtatilguncha qayta ishlash
- **Metrikalarni yig'ish:** Xizmat to'xtatilguncha yig'ish
- **Navbatni qayta ishlash:** Navbat yopilguncha xabarlarni qayta ishlash
- **Health Checks:** Bekor qilinguncha sog'liqni tekshirish
- **Keshni yangilash:** To'xtatilguncha davriy yangilash
- **Fayllarni kuzatish:** Tugaguncha kuzatish
- **Ulanish saqlash:** Yopilguncha poolni saqlash

**vs An'anaviy for Loop:**
- **An'anaviy:** Toza bekor qilish qiyin
- **RunUntil:** Bekor qilish o'rnatilgan
- **RunUntil:** Context-aware
- **RunUntil:** Izchil pattern

**Eng yaxshi amaliyotlar:**
- **Tez-tez tekshirish:** f() ni juda uzoq vaqt ishlatmang
- **f() da Sleep:** Boshqa goroutinalarga CPU vaqtini bering
- **Ticker ishlatish:** Davriy ish uchun
- **Defer Cleanup:** Har doim resurslarni tozalang

RunUntil bo'lmasa, har bir background sikl o'zining bekor qilish mantiqiga muhtoj bo'lib, izchil bo'lmagan va xatolarga moyil patternlarga olib keladi.`,
			solutionCode: `package concurrency

import (
	"context"
)

func RunUntil(ctx context.Context, done <-chan struct{}, f func()) error {
	if ctx == nil {                                             // nil kontekstni ishlash
		ctx = context.Background()                          // Fallback sifatida Background ishlatamiz
	}
	for {                                                       // Cheksiz sikl
		select {
		case <-ctx.Done():                                  // Kontekst bekor qilindi
			return ctx.Err()                            // Kontekst xatosini qaytaramiz
		case <-done:                                        // Done signali olindi
			return nil                                  // Muvaffaqiyatni qaytaramiz
		default:                                            // Hali bekor qilish yo'q
		}
		f()                                                 // Funksiyani bajaramiz
	}
}`
		}
	}
};

export default task;
