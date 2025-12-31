import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-workerpool-run-parallel-bounded',
	title: 'Run Parallel Bounded',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'worker-pool', 'parallel', 'semaphore'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **RunParallelBounded** that executes jobs concurrently but limits the number of concurrent executions using a semaphore pattern.

**Requirements:**
1. Create function \`RunParallelBounded(ctx context.Context, jobs []Job, limit int) error\`
2. Handle nil context (return nil)
3. Handle limit <= 0 (set limit to 1)
4. Use buffered channel as semaphore to limit concurrency
5. Execute jobs concurrently up to limit
6. Use sync.WaitGroup to wait for all jobs
7. Return first error encountered (use sync.Once)
8. Check context cancellation when acquiring semaphore and in goroutines

**Example:**
\`\`\`go
jobs := []Job{
    job1, job2, job3, job4, job5, job6, job7, job8,
}

// Only 3 jobs run concurrently at a time
err := RunParallelBounded(ctx, jobs, 3)

// Timeline:
// t=0:   job1, job2, job3 start (limit reached)
// t=10:  job1 finishes, job4 starts
// t=20:  job2 finishes, job5 starts
// t=30:  job3 finishes, job6 starts
// ...and so on until all complete
\`\`\`

**Constraints:**
- Must use buffered channel as semaphore (size = limit)
- Must respect concurrency limit
- Must wait for all jobs to complete
- Must handle context cancellation when acquiring semaphore`,
	initialCode: `package concurrency

import (
	"context"
	"sync"
)

type Job func(context.Context) error

// TODO: Implement RunParallelBounded
func RunParallelBounded($2) error {
	return nil // TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type Job func(context.Context) error

func RunParallelBounded(ctx context.Context, jobs []Job, limit int) error {
	if ctx == nil {                                                 // Handle nil context
		return nil                                              // Return nil for safety
	}
	if limit <= 0 {                                                 // Handle invalid limit
		limit = 1                                               // Set minimum limit
	}
	var (
		wg       sync.WaitGroup                                 // Wait for all goroutines
		once     sync.Once                                      // Capture first error only
		firstErr error                                          // Store first error
		sem      = make(chan struct{}, limit)                   // Semaphore channel
	)
	recordErr := func(err error) {                                  // Helper to record error
		if err != nil {                                         // Only if error exists
			once.Do(func() { firstErr = err })              // Set first error once
		}
	}
	if err := ctx.Err(); err != nil {                               // Check initial context
		return err                                              // Return if already cancelled
	}
	for _, job := range jobs {                                      // Iterate through jobs
		if job == nil {                                         // Skip nil jobs
			continue                                        // Move to next
		}
		select {                                                // Acquire semaphore or cancel
		case <-ctx.Done():                                      // Context cancelled
			return ctx.Err()                                // Return cancellation error
		case sem <- struct{}{}:                                 // Acquired semaphore slot
		}
		wg.Add(1)                                               // Increment wait group
		go func(job Job) {                                      // Launch goroutine
			defer wg.Done()                                 // Decrement when done
			defer func() { <-sem }()                        // Release semaphore
			select {                                        // Check context
			case <-ctx.Done():                              // Context cancelled
				recordErr(ctx.Err())                    // Record cancellation
				return                                  // Exit goroutine
			default:                                        // Context not cancelled
			}
			recordErr(job(ctx))                             // Execute and record error
		}(job)                                                  // Pass job to avoid closure issue
	}
	wg.Wait()                                                       // Wait for all goroutines
	if firstErr != nil {                                            // Check if error occurred
		return firstErr                                         // Return first error
	}
	return ctx.Err()                                                // Return final context state
}`,
	testCode: `package concurrency

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestRunParallelBounded1(t *testing.T) {
	// Test respects concurrency limit
	ctx := context.Background()
	var active int32
	var maxActive int32
	limit := 3

	jobs := make([]Job, 10)
	for i := 0; i < 10; i++ {
		jobs[i] = func(ctx context.Context) error {
			current := atomic.AddInt32(&active, 1)
			if current > maxActive {
				atomic.StoreInt32(&maxActive, current)
			}
			time.Sleep(20 * time.Millisecond)
			atomic.AddInt32(&active, -1)
			return nil
		}
	}

	err := RunParallelBounded(ctx, jobs, limit)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if maxActive > int32(limit) {
		t.Errorf("expected max %d concurrent jobs, got %d", limit, maxActive)
	}
}

func TestRunParallelBounded2(t *testing.T) {
	// Test with limit of 1 (sequential-like)
	ctx := context.Background()
	var counter int32
	order := []int{}

	jobs := []Job{
		func(ctx context.Context) error {
			order = append(order, 1)
			atomic.AddInt32(&counter, 1)
			time.Sleep(10 * time.Millisecond)
			return nil
		},
		func(ctx context.Context) error {
			order = append(order, 2)
			atomic.AddInt32(&counter, 1)
			time.Sleep(10 * time.Millisecond)
			return nil
		},
		func(ctx context.Context) error {
			order = append(order, 3)
			atomic.AddInt32(&counter, 1)
			time.Sleep(10 * time.Millisecond)
			return nil
		},
	}

	err := RunParallelBounded(ctx, jobs, 1)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if counter != 3 {
		t.Errorf("expected 3 jobs executed, got %d", counter)
	}
}

func TestRunParallelBounded3(t *testing.T) {
	// Test with error
	ctx := context.Background()
	expectedErr := errors.New("job failed")
	var counter int32

	jobs := []Job{
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			return nil
		},
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			return expectedErr
		},
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			return nil
		},
	}

	err := RunParallelBounded(ctx, jobs, 2)
	if err != expectedErr {
		t.Errorf("expected error %v, got %v", expectedErr, err)
	}
}

func TestRunParallelBounded4(t *testing.T) {
	// Test with zero limit (should use 1)
	ctx := context.Background()
	var counter int32

	jobs := []Job{
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			return nil
		},
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			return nil
		},
	}

	err := RunParallelBounded(ctx, jobs, 0)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if counter != 2 {
		t.Errorf("expected 2 jobs executed, got %d", counter)
	}
}

func TestRunParallelBounded5(t *testing.T) {
	// Test with negative limit (should use 1)
	ctx := context.Background()
	var counter int32

	jobs := []Job{
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			return nil
		},
	}

	err := RunParallelBounded(ctx, jobs, -5)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if counter != 1 {
		t.Errorf("expected 1 job executed, got %d", counter)
	}
}

func TestRunParallelBounded6(t *testing.T) {
	// Test skips nil jobs
	ctx := context.Background()
	var counter int32

	jobs := []Job{
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			return nil
		},
		nil,
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			return nil
		},
	}

	err := RunParallelBounded(ctx, jobs, 2)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if counter != 2 {
		t.Errorf("expected 2 jobs executed, got %d", counter)
	}
}

func TestRunParallelBounded7(t *testing.T) {
	// Test with nil context
	jobs := []Job{
		func(ctx context.Context) error {
			return nil
		},
	}

	err := RunParallelBounded(nil, jobs, 5)
	if err != nil {
		t.Errorf("expected no error with nil context, got %v", err)
	}
}

func TestRunParallelBounded8(t *testing.T) {
	// Test context cancellation
	ctx, cancel := context.WithCancel(context.Background())
	var counter int32

	jobs := make([]Job, 10)
	for i := 0; i < 10; i++ {
		jobs[i] = func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			time.Sleep(50 * time.Millisecond)
			return nil
		}
	}

	go func() {
		time.Sleep(25 * time.Millisecond)
		cancel()
	}()

	err := RunParallelBounded(ctx, jobs, 3)
	// Should get cancellation error
	if err == nil {
		t.Errorf("expected context cancelled error")
	}
}

func TestRunParallelBounded9(t *testing.T) {
	// Test empty job slice
	ctx := context.Background()
	jobs := []Job{}

	err := RunParallelBounded(ctx, jobs, 5)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestRunParallelBounded10(t *testing.T) {
	// Test limit larger than job count
	ctx := context.Background()
	var counter int32

	jobs := []Job{
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			return nil
		},
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			return nil
		},
	}

	err := RunParallelBounded(ctx, jobs, 100)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if counter != 2 {
		t.Errorf("expected 2 jobs executed, got %d", counter)
	}
}`,
			hint1: `Create a buffered channel with size=limit: sem := make(chan struct{}, limit). Send to channel to acquire slot, receive from channel to release.`,
			hint2: `Use select to acquire semaphore with context cancellation: select { case <-ctx.Done(): return; case sem <- struct{}{}: }. Remember to release with defer func() { <-sem }().`,
			whyItMatters: `RunParallelBounded provides controlled parallelism, preventing resource exhaustion while maintaining good performance.

**Why Bounded Parallelism:**
- **Resource Protection:** Prevent overwhelming CPU, memory, or network
- **Rate Limiting:** Control load on downstream services
- **Stability:** Avoid crashes from too many goroutines
- **Predictable Performance:** Consistent resource usage

**Production Pattern:**
\`\`\`go
// Bounded API calls to prevent rate limiting
func FetchUserData(ctx context.Context, userIDs []string) error {
    jobs := make([]Job, len(userIDs))
    for i, id := range userIDs {
        userID := id
        jobs[i] = func(ctx context.Context) error {
            return fetchUser(ctx, userID)
        }
    }

    // Only 10 concurrent API calls
    return RunParallelBounded(ctx, jobs, 10)
}

// Bounded database connections
func ProcessRecords(ctx context.Context, recordIDs []string) error {
    jobs := make([]Job, len(recordIDs))
    for i, id := range recordIDs {
        recordID := id
        jobs[i] = func(ctx context.Context) error {
            return processRecord(ctx, recordID)
        }
    }

    // Limit to database connection pool size
    maxDBConns := 20
    return RunParallelBounded(ctx, jobs, maxDBConns)
}

// Bounded file processing
func ProcessLargeFiles(ctx context.Context, files []string) error {
    jobs := make([]Job, len(files))
    for i, file := range files {
        filename := file
        jobs[i] = func(ctx context.Context) error {
            return processLargeFile(ctx, filename)
        }
    }

    // Limit to number of CPU cores
    maxCPU := runtime.NumCPU()
    return RunParallelBounded(ctx, jobs, maxCPU)
}

// Bounded image processing
func ResizeImages(ctx context.Context, images []string) error {
    jobs := make([]Job, len(images))
    for i, img := range images {
        imagePath := img
        jobs[i] = func(ctx context.Context) error {
            // Memory-intensive operation
            return resizeImage(ctx, imagePath)
        }
    }

    // Limit based on available memory
    // Assuming each resize takes ~100MB, with 2GB available
    maxConcurrent := 20
    return RunParallelBounded(ctx, jobs, maxConcurrent)
}

// Bounded network requests
func DownloadFiles(ctx context.Context, urls []string) error {
    jobs := make([]Job, len(urls))
    for i, url := range urls {
        downloadURL := url
        jobs[i] = func(ctx context.Context) error {
            return downloadFile(ctx, downloadURL)
        }
    }

    // Limit concurrent downloads to avoid bandwidth saturation
    return RunParallelBounded(ctx, jobs, 5)
}

// Adaptive concurrency based on system resources
func ProcessWithAdaptiveConcurrency(ctx context.Context, jobs []Job) error {
    // Adjust based on current load
    var limit int
    load := getSystemLoad()

    switch {
    case load < 0.5:
        limit = runtime.NumCPU() * 2 // Low load, more parallelism
    case load < 0.8:
        limit = runtime.NumCPU()     // Medium load, match CPU count
    default:
        limit = runtime.NumCPU() / 2 // High load, reduce parallelism
    }

    if limit < 1 {
        limit = 1
    }

    return RunParallelBounded(ctx, jobs, limit)
}

// Tiered processing with different limits
func ProcessByPriority(ctx context.Context, highPriority, lowPriority []Job) error {
    // Process high priority with more resources
    if err := RunParallelBounded(ctx, highPriority, 20); err != nil {
        return err
    }

    // Process low priority with fewer resources
    return RunParallelBounded(ctx, lowPriority, 5)
}
\`\`\`

**Real-World Benefits:**
- **Prevents Overload:** Won't overwhelm systems with too many concurrent operations
- **Better Stability:** Predictable resource usage
- **Cost Control:** Avoid excessive API rate charges
- **Fair Resource Sharing:** Leave resources for other operations

**Performance vs Resources:**
\`\`\`
Unbounded: 1000 goroutines, 2GB RAM, system crash
Bounded(50): 50 goroutines, 100MB RAM, stable
Bounded(10): 10 goroutines, 20MB RAM, slower but safer
\`\`\`

**Common Limit Values:**
- **CPU-bound:** runtime.NumCPU() or runtime.NumCPU() * 2
- **Memory-bound:** Based on available memory / per-job memory
- **Network-bound:** 5-20 concurrent connections
- **Database:** Match connection pool size (10-100)
- **API calls:** Based on rate limits (often 10-100)

**Semaphore Pattern:**
The buffered channel acts as a semaphore:
- **Send to channel:** Acquire a slot (blocks if full)
- **Receive from channel:** Release a slot
- **Channel size:** Maximum concurrent operations

This pattern is fundamental in concurrent programming for resource management.

**When to Use:**
- API calls with rate limits
- Database operations with connection pools
- CPU or memory-intensive operations
- Any operation that needs controlled concurrency

**Comparison:**
- **RunParallel:** Unlimited concurrency, risk of resource exhaustion
- **RunParallelBounded:** Controlled concurrency, stable and predictable
- **RunSequential:** No concurrency, slowest but simplest`,	order: 3,
	translations: {
		ru: {
			title: 'Параллельное выполнение с лимитом горутин',
			description: `Реализуйте **RunParallelBounded**, который выполняет задачи одновременно, но ограничивает количество одновременных выполнений используя паттерн семафора.

**Требования:**
1. Создайте функцию \`RunParallelBounded(ctx context.Context, jobs []Job, limit int) error\`
2. Обработайте nil context (верните nil)
3. Обработайте limit <= 0 (установите limit в 1)
4. Используйте буферизованный канал как семафор для ограничения конкурентности
5. Выполняйте задачи одновременно до limit
6. Используйте sync.WaitGroup для ожидания всех задач
7. Верните первую встреченную ошибку (используйте sync.Once)
8. Проверяйте отмену контекста при получении семафора и в горутинах

**Пример:**
\`\`\`go
jobs := []Job{
    job1, job2, job3, job4, job5, job6, job7, job8,
}

// Только 3 задачи выполняются одновременно
err := RunParallelBounded(ctx, jobs, 3)

// Временная шкала:
// t=0:   job1, job2, job3 стартуют (достигнут лимит)
// t=10:  job1 завершилась, job4 стартует
// t=20:  job2 завершилась, job5 стартует
// t=30:  job3 завершилась, job6 стартует
// ...и так далее до завершения всех
`,
			hint1: `Создайте буферизованный канал с size=limit: sem := make(chan struct{}, limit). Отправка в канал получает слот, получение из канала освобождает.`,
			hint2: `Используйте select для получения семафора с отменой контекста: select { case <-ctx.Done(): return; case sem <- struct{}{}: }. Не забудьте освободить через defer func() { <-sem }().`,
			whyItMatters: `RunParallelBounded обеспечивает контролируемый параллелизм, предотвращая истощение ресурсов при сохранении хорошей производительности.

**Почему ограниченный параллелизм:**
- **Защита ресурсов:** Предотвращение перегрузки CPU, памяти или сети
- **Ограничение скорости:** Контроль нагрузки на нижестоящие сервисы
- **Стабильность:** Избежание сбоев от слишком большого числа горутин
- **Предсказуемая производительность:** Согласованное использование ресурсов

**Продакшен паттерн:**
\`\`\`go
// Bounded API calls to prevent rate limiting
func FetchUserData(ctx context.Context, userIDs []string) error {
    jobs := make([]Job, len(userIDs))
    for i, id := range userIDs {
        userID := id
        jobs[i] = func(ctx context.Context) error {
            return fetchUser(ctx, userID)
        }
    }

    // Only 10 concurrent API calls
    return RunParallelBounded(ctx, jobs, 10)
}

// Bounded database connections
func ProcessRecords(ctx context.Context, recordIDs []string) error {
    jobs := make([]Job, len(recordIDs))
    for i, id := range recordIDs {
        recordID := id
        jobs[i] = func(ctx context.Context) error {
            return processRecord(ctx, recordID)
        }
    }

    // Limit to database connection pool size
    maxDBConns := 20
    return RunParallelBounded(ctx, jobs, maxDBConns)
}

// Bounded file processing
func ProcessLargeFiles(ctx context.Context, files []string) error {
    jobs := make([]Job, len(files))
    for i, file := range files {
        filename := file
        jobs[i] = func(ctx context.Context) error {
            return processLargeFile(ctx, filename)
        }
    }

    // Limit to number of CPU cores
    maxCPU := runtime.NumCPU()
    return RunParallelBounded(ctx, jobs, maxCPU)
}

// Bounded image processing
func ResizeImages(ctx context.Context, images []string) error {
    jobs := make([]Job, len(images))
    for i, img := range images {
        imagePath := img
        jobs[i] = func(ctx context.Context) error {
            // Memory-intensive operation
            return resizeImage(ctx, imagePath)
        }
    }

    // Limit based on available memory
    // Assuming each resize takes ~100MB, with 2GB available
    maxConcurrent := 20
    return RunParallelBounded(ctx, jobs, maxConcurrent)
}

// Bounded network requests
func DownloadFiles(ctx context.Context, urls []string) error {
    jobs := make([]Job, len(urls))
    for i, url := range urls {
        downloadURL := url
        jobs[i] = func(ctx context.Context) error {
            return downloadFile(ctx, downloadURL)
        }
    }

    // Limit concurrent downloads to avoid bandwidth saturation
    return RunParallelBounded(ctx, jobs, 5)
}

// Adaptive concurrency based on system resources
func ProcessWithAdaptiveConcurrency(ctx context.Context, jobs []Job) error {
    // Adjust based on current load
    var limit int
    load := getSystemLoad()

    switch {
    case load < 0.5:
        limit = runtime.NumCPU() * 2 // Low load, more parallelism
    case load < 0.8:
        limit = runtime.NumCPU()     // Medium load, match CPU count
    default:
        limit = runtime.NumCPU() / 2 // High load, reduce parallelism
    }

    if limit < 1 {
        limit = 1
    }

    return RunParallelBounded(ctx, jobs, limit)
}

// Tiered processing with different limits
func ProcessByPriority(ctx context.Context, highPriority, lowPriority []Job) error {
    // Process high priority with more resources
    if err := RunParallelBounded(ctx, highPriority, 20); err != nil {
        return err
    }

    // Process low priority with fewer resources
    return RunParallelBounded(ctx, lowPriority, 5)
}
\`\`\`

**Практические преимущества:**
- **Предотвращение перегрузки:** Не перегружает системы слишком большим количеством одновременных операций
- **Лучшая стабильность:** Предсказуемое использование ресурсов
- **Контроль затрат:** Избежание избыточных API тарифов
- **Справедливое распределение ресурсов:** Оставляет ресурсы для других операций

**Производительность vs Ресурсы:**
\`\`\`
Неограниченно: 1000 горутин, 2GB RAM, крах системы
Bounded(50): 50 горутин, 100MB RAM, стабильно
Bounded(10): 10 горутин, 20MB RAM, медленнее но безопаснее
\`\`\`

**Общие значения лимитов:**
- **CPU-bound:** runtime.NumCPU() или runtime.NumCPU() * 2
- **Memory-bound:** На основе доступной памяти / памяти на задачу
- **Network-bound:** 5-20 одновременных подключений
- **Database:** Соответствует размеру пула соединений (10-100)
- **API вызовы:** На основе лимитов скорости (часто 10-100)

**Паттерн семафора:**
Буферизованный канал действует как семафор:
- **Отправка в канал:** Получение слота (блокируется если полон)
- **Получение из канала:** Освобождение слота
- **Размер канала:** Максимум одновременных операций

Этот паттерн фундаментален в конкурентном программировании для управления ресурсами.

**Когда использовать:**
- API вызовы с лимитами скорости
- Операции БД с пулами соединений
- CPU или память-интенсивные операции
- Любая операция требующая контролируемой конкурентности

**Сравнение:**
- **RunParallel:** Неограниченная конкурентность, риск истощения ресурсов
- **RunParallelBounded:** Контролируемая конкурентность, стабильно и предсказуемо
- **RunSequential:** Без конкурентности, медленнее но проще всего`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type Job func(context.Context) error

func RunParallelBounded(ctx context.Context, jobs []Job, limit int) error {
	if ctx == nil {                                                 // Обработка nil контекста
		return nil                                              // Возврат nil для безопасности
	}
	if limit <= 0 {                                                 // Обработка неверного лимита
		limit = 1                                               // Установка минимального лимита
	}
	var (
		wg       sync.WaitGroup                                 // Ожидание всех горутин
		once     sync.Once                                      // Захват только первой ошибки
		firstErr error                                          // Хранение первой ошибки
		sem      = make(chan struct{}, limit)                   // Канал-семафор
	)
	recordErr := func(err error) {                                  // Помощник для записи ошибки
		if err != nil {                                         // Только если ошибка существует
			once.Do(func() { firstErr = err })              // Установить первую ошибку один раз
		}
	}
	if err := ctx.Err(); err != nil {                               // Проверка начального контекста
		return err                                              // Возврат если уже отменён
	}
	for _, job := range jobs {                                      // Итерация по задачам
		if job == nil {                                         // Пропуск nil задач
			continue                                        // Переход к следующей
		}
		select {                                                // Получение семафора или отмена
		case <-ctx.Done():                                      // Контекст отменён
			return ctx.Err()                                // Возврат ошибки отмены
		case sem <- struct{}{}:                                 // Получен слот семафора
		}
		wg.Add(1)                                               // Увеличение счётчика wait group
		go func(job Job) {                                      // Запуск горутины
			defer wg.Done()                                 // Уменьшение при завершении
			defer func() { <-sem }()                        // Освобождение семафора
			select {                                        // Проверка контекста
			case <-ctx.Done():                              // Контекст отменён
				recordErr(ctx.Err())                    // Запись отмены
				return                                  // Выход из горутины
			default:                                        // Контекст не отменён
			}
			recordErr(job(ctx))                             // Выполнение и запись ошибки
		}(job)                                                  // Передача job для избежания проблемы замыкания
	}
	wg.Wait()                                                       // Ожидание всех горутин
	if firstErr != nil {                                            // Проверка возникновения ошибки
		return firstErr                                         // Возврат первой ошибки
	}
	return ctx.Err()                                                // Возврат финального состояния контекста
}`
		},
		uz: {
			title: 'Goroutinalar limitli parallel bajarish',
			description: `Vazifalarni bir vaqtda bajaradigan, lekin semafor patternidan foydalanib bir vaqtdagi bajarishlar sonini cheklaydigan **RunParallelBounded** ni amalga oshiring.

**Talablar:**
1. \`RunParallelBounded(ctx context.Context, jobs []Job, limit int) error\` funksiyasini yarating
2. nil kontekstni ishlang (nil qaytaring)
3. limit <= 0 ni ishlang (limitni 1 ga o'rnating)
4. Parallellikni cheklash uchun buferli kanalni semafor sifatida ishlating
5. Vazifalarni limitgacha bir vaqtda bajaring
6. Barcha vazifalarni kutish uchun sync.WaitGroup dan foydalaning
7. Duch kelgan birinchi xatoni qaytaring (sync.Once dan foydalaning)
8. Semafor olishda va goroutinalarda kontekst bekor qilinishini tekshiring

**Misol:**
\`\`\`go
jobs := []Job{
    job1, job2, job3, job4, job5, job6, job7, job8,
}

// Bir vaqtning o'zida faqat 3 ta vazifa bajariladi
err := RunParallelBounded(ctx, jobs, 3)

// Vaqt chizig'i:
// t=0:   job1, job2, job3 boshlanadi (limit to'ldi)
// t=10:  job1 tugadi, job4 boshlanadi
// t=20:  job2 tugadi, job5 boshlanadi
// t=30:  job3 tugadi, job6 boshlanadi
// ...hammasi tugagunga qadar
`,
			hint1: `size=limit bilan buferli kanal yarating: sem := make(chan struct{}, limit). Kanalga yuborish slotni oladi, kanaldan qabul qilish ozod qiladi.`,
			hint2: `Kontekst bekor qilish bilan semaforni olish uchun select dan foydalaning: select { case <-ctx.Done(): return; case sem <- struct{}{}: }. defer func() { <-sem }() orqali ozod qilishni unutmang.`,
			whyItMatters: `RunParallelBounded boshqariladigan parallellikni ta'minlaydi, yaxshi samaradorlikni saqlab qolgan holda resurslar tugashining oldini oladi.

**Nima uchun cheklangan parallellik:**
- **Resurslarni himoya qilish:** CPU, xotira yoki tarmoqni haddan tashqari yuklashning oldini olish
- **Tezlikni cheklash:** Pastki xizmatlardagi yukni nazorat qilish
- **Barqarorlik:** Juda ko'p goroutinalardan kelib chiqadigan nosozliklardan qochish
- **Oldindan aytib beriladigan samaradorlik:** Izchil resurslardan foydalanish

**Ishlab chiqarish patterni:**
\`\`\`go
// Bounded API calls to prevent rate limiting
func FetchUserData(ctx context.Context, userIDs []string) error {
    jobs := make([]Job, len(userIDs))
    for i, id := range userIDs {
        userID := id
        jobs[i] = func(ctx context.Context) error {
            return fetchUser(ctx, userID)
        }
    }

    // Only 10 concurrent API calls
    return RunParallelBounded(ctx, jobs, 10)
}

// Bounded database connections
func ProcessRecords(ctx context.Context, recordIDs []string) error {
    jobs := make([]Job, len(recordIDs))
    for i, id := range recordIDs {
        recordID := id
        jobs[i] = func(ctx context.Context) error {
            return processRecord(ctx, recordID)
        }
    }

    // Limit to database connection pool size
    maxDBConns := 20
    return RunParallelBounded(ctx, jobs, maxDBConns)
}

// Bounded file processing
func ProcessLargeFiles(ctx context.Context, files []string) error {
    jobs := make([]Job, len(files))
    for i, file := range files {
        filename := file
        jobs[i] = func(ctx context.Context) error {
            return processLargeFile(ctx, filename)
        }
    }

    // Limit to number of CPU cores
    maxCPU := runtime.NumCPU()
    return RunParallelBounded(ctx, jobs, maxCPU)
}

// Bounded image processing
func ResizeImages(ctx context.Context, images []string) error {
    jobs := make([]Job, len(images))
    for i, img := range images {
        imagePath := img
        jobs[i] = func(ctx context.Context) error {
            // Memory-intensive operation
            return resizeImage(ctx, imagePath)
        }
    }

    // Limit based on available memory
    // Assuming each resize takes ~100MB, with 2GB available
    maxConcurrent := 20
    return RunParallelBounded(ctx, jobs, maxConcurrent)
}

// Bounded network requests
func DownloadFiles(ctx context.Context, urls []string) error {
    jobs := make([]Job, len(urls))
    for i, url := range urls {
        downloadURL := url
        jobs[i] = func(ctx context.Context) error {
            return downloadFile(ctx, downloadURL)
        }
    }

    // Limit concurrent downloads to avoid bandwidth saturation
    return RunParallelBounded(ctx, jobs, 5)
}

// Adaptive concurrency based on system resources
func ProcessWithAdaptiveConcurrency(ctx context.Context, jobs []Job) error {
    // Adjust based on current load
    var limit int
    load := getSystemLoad()

    switch {
    case load < 0.5:
        limit = runtime.NumCPU() * 2 // Low load, more parallelism
    case load < 0.8:
        limit = runtime.NumCPU()     // Medium load, match CPU count
    default:
        limit = runtime.NumCPU() / 2 // High load, reduce parallelism
    }

    if limit < 1 {
        limit = 1
    }

    return RunParallelBounded(ctx, jobs, limit)
}

// Tiered processing with different limits
func ProcessByPriority(ctx context.Context, highPriority, lowPriority []Job) error {
    // Process high priority with more resources
    if err := RunParallelBounded(ctx, highPriority, 20); err != nil {
        return err
    }

    // Process low priority with fewer resources
    return RunParallelBounded(ctx, lowPriority, 5)
}
\`\`\`

**Amaliy foydalari:**
- **Ortiqcha yuklashning oldini olish:** Juda ko'p bir vaqtdagi operatsiyalar bilan tizimlarni ortiqcha yuklamaydi
- **Yaxshiroq barqarorlik:** Oldindan aytib beriladigan resurslardan foydalanish
- **Xarajatlarni nazorat qilish:** Ortiqcha API to'lovlaridan qochish
- **Resurslarni adolatli taqsimlash:** Boshqa operatsiyalar uchun resurslar qoldiradi

**Samaradorlik vs Resurslar:**
\`\`\`
Cheksiz: 1000 goroutinalar, 2GB RAM, tizim ishdan chiqishi
Bounded(50): 50 goroutinalar, 100MB RAM, barqaror
Bounded(10): 10 goroutinalar, 20MB RAM, sekinroq lekin xavfsizroq
\`\`\`

**Umumiy limit qiymatlari:**
- **CPU-bound:** runtime.NumCPU() yoki runtime.NumCPU() * 2
- **Memory-bound:** Mavjud xotira / vazifa uchun xotira asosida
- **Network-bound:** 5-20 bir vaqtdagi ulanishlar
- **Database:** Ulanish puli hajmiga mos keladi (10-100)
- **API chaqiruvlar:** Tezlik limitlariga qarab (ko'pincha 10-100)

**Semafor patterni:**
Buferli kanal semafor vazifasini bajaradi:
- **Kanalga yuborish:** Slotni olish (to'lgan bo'lsa bloklanadi)
- **Kanaldan qabul qilish:** Slotni ozod qilish
- **Kanal hajmi:** Maksimal bir vaqtdagi operatsiyalar

Bu pattern resurslarni boshqarish uchun parallel dasturlashda asosiy hisoblanadi.

**Qachon ishlatiladi:**
- Tezlik limitlari bilan API chaqiruvlar
- Ulanish pullari bilan ma'lumotlar bazasi operatsiyalari
- CPU yoki xotira intensiv operatsiyalar
- Boshqariladigan parallellikni talab qiladigan har qanday operatsiya

**Taqqoslash:**
- **RunParallel:** Cheklanmagan parallellik, resurslar tugash xavfi
- **RunParallelBounded:** Boshqariladigan parallellik, barqaror va oldindan aytib beriladigan
- **RunSequential:** Parallellik yo'q, sekinroq lekin eng oddiy`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type Job func(context.Context) error

func RunParallelBounded(ctx context.Context, jobs []Job, limit int) error {
	if ctx == nil {                                                 // nil kontekstni ishlash
		return nil                                              // Xavfsizlik uchun nil qaytarish
	}
	if limit <= 0 {                                                 // Noto'g'ri limitni ishlash
		limit = 1                                               // Minimal limitni o'rnatish
	}
	var (
		wg       sync.WaitGroup                                 // Barcha goroutinalarni kutish
		once     sync.Once                                      // Faqat birinchi xatoni ushlash
		firstErr error                                          // Birinchi xatoni saqlash
		sem      = make(chan struct{}, limit)                   // Semafor kanali
	)
	recordErr := func(err error) {                                  // Xatoni yozish yordamchisi
		if err != nil {                                         // Faqat xato mavjud bo'lsa
			once.Do(func() { firstErr = err })              // Birinchi xatoni bir marta o'rnatish
		}
	}
	if err := ctx.Err(); err != nil {                               // Boshlang'ich kontekstni tekshirish
		return err                                              // Agar allaqachon bekor qilingan bo'lsa qaytarish
	}
	for _, job := range jobs {                                      // Vazifalar bo'yicha iteratsiya
		if job == nil {                                         // nil vazifalarni o'tkazib yuborish
			continue                                        // Keyingisiga o'tish
		}
		select {                                                // Semaforni olish yoki bekor qilish
		case <-ctx.Done():                                      // Kontekst bekor qilindi
			return ctx.Err()                                // Bekor qilish xatosini qaytarish
		case sem <- struct{}{}:                                 // Semafor sloti olindi
		}
		wg.Add(1)                                               // Wait group hisoblagichini oshirish
		go func(job Job) {                                      // Goroutinani ishga tushirish
			defer wg.Done()                                 // Tugaganda kamaytirish
			defer func() { <-sem }()                        // Semaforni ozod qilish
			select {                                        // Kontekstni tekshirish
			case <-ctx.Done():                              // Kontekst bekor qilindi
				recordErr(ctx.Err())                    // Bekor qilishni yozish
				return                                  // Goroutinadan chiqish
			default:                                        // Kontekst bekor qilinmagan
			}
			recordErr(job(ctx))                             // Bajarish va xatoni yozish
		}(job)                                                  // Yopilish muammosidan qochish uchun jobni o'tkazish
	}
	wg.Wait()                                                       // Barcha goroutinalarni kutish
	if firstErr != nil {                                            // Xato yuz berganligini tekshirish
		return firstErr                                         // Birinchi xatoni qaytarish
	}
	return ctx.Err()                                                // Kontekstning yakuniy holatini qaytarish
}`
		}
	}
};

export default task;
