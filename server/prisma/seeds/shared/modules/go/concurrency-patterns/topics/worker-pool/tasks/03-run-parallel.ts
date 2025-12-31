import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-workerpool-run-parallel',
	title: 'Run Parallel',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'worker-pool', 'parallel', 'goroutines'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **RunParallel** that executes all jobs concurrently in separate goroutines.

**Requirements:**
1. Create function \`RunParallel(ctx context.Context, jobs []Job) error\`
2. Handle nil context (return nil)
3. Check context before starting jobs
4. Execute all jobs concurrently in goroutines
5. Skip nil jobs
6. Wait for all jobs to complete using sync.WaitGroup
7. Return first error encountered (use sync.Once)
8. Check context cancellation in each goroutine

**Example:**
\`\`\`go
jobs := []Job{
    func(ctx context.Context) error {
        time.Sleep(100 * time.Millisecond)
        fmt.Println("Job 1")
        return nil
    },
    func(ctx context.Context) error {
        time.Sleep(50 * time.Millisecond)
        fmt.Println("Job 2")
        return nil
    },
    func(ctx context.Context) error {
        time.Sleep(75 * time.Millisecond)
        return errors.New("job 3 failed")
    },
}

err := RunParallel(ctx, jobs)
// Output may be: Job 2, Job 3 failed, Job 1 (order varies)
// err = "job 3 failed" (first error captured)
// All jobs run concurrently
\`\`\`

**Constraints:**
- Must use goroutines for concurrent execution
- Must use sync.WaitGroup to wait for all jobs
- Must use sync.Once to capture first error only
- Must check context cancellation`,
	initialCode: `package concurrency

import (
	"context"
	"sync"
)

type Job func(context.Context) error

// TODO: Implement RunParallel
func RunParallel($2) error {
	return nil // TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type Job func(context.Context) error

func RunParallel(ctx context.Context, jobs []Job) error {
	if ctx == nil {                                                 // Handle nil context
		return nil                                              // Return nil for safety
	}
	var (
		wg       sync.WaitGroup                                 // Wait for all goroutines
		once     sync.Once                                      // Capture first error only
		firstErr error                                          // Store first error
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
		wg.Add(1)                                               // Increment wait group
		go func(job Job) {                                      // Launch goroutine
			defer wg.Done()                                 // Decrement when done
			select {                                        // Check context first
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
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestRunParallel1(t *testing.T) {
	// Test all jobs execute in parallel
	ctx := context.Background()
	var counter int32
	jobs := []Job{
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			time.Sleep(10 * time.Millisecond)
			return nil
		},
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			time.Sleep(10 * time.Millisecond)
			return nil
		},
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			time.Sleep(10 * time.Millisecond)
			return nil
		},
	}

	start := time.Now()
	err := RunParallel(ctx, jobs)
	duration := time.Since(start)

	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if counter != 3 {
		t.Errorf("expected 3 jobs executed, got %d", counter)
	}
	// Should complete in ~10ms if parallel, not 30ms
	if duration > 100*time.Millisecond {
		t.Errorf("jobs may not be running in parallel, took %v", duration)
	}
}

func TestRunParallel2(t *testing.T) {
	// Test captures first error
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

	err := RunParallel(ctx, jobs)
	if err != expectedErr {
		t.Errorf("expected error %v, got %v", expectedErr, err)
	}
	// All jobs should still execute
	time.Sleep(50 * time.Millisecond)
	if counter != 3 {
		t.Errorf("expected 3 jobs executed, got %d", counter)
	}
}

func TestRunParallel3(t *testing.T) {
	// Test empty job slice
	ctx := context.Background()
	jobs := []Job{}

	err := RunParallel(ctx, jobs)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestRunParallel4(t *testing.T) {
	// Test with nil context
	jobs := []Job{
		func(ctx context.Context) error {
			return nil
		},
	}

	err := RunParallel(nil, jobs)
	if err != nil {
		t.Errorf("expected no error with nil context, got %v", err)
	}
}

func TestRunParallel5(t *testing.T) {
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
		nil,
	}

	err := RunParallel(ctx, jobs)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if counter != 2 {
		t.Errorf("expected 2 jobs executed, got %d", counter)
	}
}

func TestRunParallel6(t *testing.T) {
	// Test with context cancellation before execution
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	jobs := []Job{
		func(ctx context.Context) error {
			return nil
		},
	}

	err := RunParallel(ctx, jobs)
	if err == nil {
		t.Errorf("expected context cancelled error")
	}
}

func TestRunParallel7(t *testing.T) {
	// Test context cancellation during execution
	ctx, cancel := context.WithCancel(context.Background())
	var counter int32

	jobs := []Job{
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			time.Sleep(50 * time.Millisecond)
			return nil
		},
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			time.Sleep(50 * time.Millisecond)
			return nil
		},
		func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			time.Sleep(50 * time.Millisecond)
			return nil
		},
	}

	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	err := RunParallel(ctx, jobs)
	// Should get either cancellation error or nil depending on timing
	_ = err
}

func TestRunParallel8(t *testing.T) {
	// Test many jobs execute in parallel
	ctx := context.Background()
	var counter int32
	numJobs := 100

	jobs := make([]Job, numJobs)
	for i := 0; i < numJobs; i++ {
		jobs[i] = func(ctx context.Context) error {
			atomic.AddInt32(&counter, 1)
			return nil
		}
	}

	err := RunParallel(ctx, jobs)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if counter != int32(numJobs) {
		t.Errorf("expected %d jobs executed, got %d", numJobs, counter)
	}
}

func TestRunParallel9(t *testing.T) {
	// Test only first error is returned
	ctx := context.Background()
	err1 := errors.New("error 1")
	err2 := errors.New("error 2")
	var mu sync.Mutex
	var errors []error

	jobs := []Job{
		func(ctx context.Context) error {
			time.Sleep(10 * time.Millisecond)
			return err1
		},
		func(ctx context.Context) error {
			time.Sleep(15 * time.Millisecond)
			return err2
		},
		func(ctx context.Context) error {
			return nil
		},
	}

	err := RunParallel(ctx, jobs)
	mu.Lock()
	errors = append(errors, err)
	mu.Unlock()

	// Should get one of the errors
	if err == nil {
		t.Errorf("expected an error")
	}
	if err != err1 && err != err2 {
		t.Errorf("expected err1 or err2, got %v", err)
	}
}

func TestRunParallel10(t *testing.T) {
	// Test single job
	ctx := context.Background()
	executed := false

	jobs := []Job{
		func(ctx context.Context) error {
			executed = true
			return nil
		},
	}

	err := RunParallel(ctx, jobs)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if !executed {
		t.Errorf("expected job to be executed")
	}
}`,
			hint1: `Use sync.WaitGroup to track goroutines. Call wg.Add(1) before launching each goroutine and wg.Done() when it completes.`,
			hint2: `Use sync.Once with a recordErr helper function to capture only the first error. Pass job as parameter to goroutine to avoid closure variable issues.`,
			whyItMatters: `RunParallel dramatically speeds up independent job execution by running them concurrently, essential for high-performance applications.

**Why Parallel Execution:**
- **Speed:** Multiple jobs run simultaneously
- **Throughput:** Process more work in less time
- **Resource Utilization:** Use all available CPU cores
- **Scalability:** Handle large workloads efficiently

**Production Pattern:**
\`\`\`go
// Parallel API calls
func FetchMultipleUsers(ctx context.Context, userIDs []string) (map[string]*User, error) {
    var (
        mu    sync.Mutex
        users = make(map[string]*User)
    )

    jobs := make([]Job, len(userIDs))
    for i, id := range userIDs {
        userID := id
        jobs[i] = func(ctx context.Context) error {
            user, err := fetchUser(ctx, userID)
            if err != nil {
                return err
            }
            mu.Lock()
            users[userID] = user
            mu.Unlock()
            return nil
        }
    }

    if err := RunParallel(ctx, jobs); err != nil {
        return nil, err
    }

    return users, nil
}

// Parallel data validation
func ValidateRecords(ctx context.Context, records []Record) error {
    jobs := make([]Job, len(records))
    for i, record := range records {
        r := record
        jobs[i] = func(ctx context.Context) error {
            return validateRecord(ctx, r)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel file processing
func ProcessFiles(ctx context.Context, files []string) error {
    jobs := make([]Job, len(files))
    for i, file := range files {
        filename := file
        jobs[i] = func(ctx context.Context) error {
            data, err := os.ReadFile(filename)
            if err != nil {
                return err
            }
            return processData(ctx, data)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel cache warming
func WarmCache(ctx context.Context, keys []string) error {
    jobs := make([]Job, len(keys))
    for i, key := range keys {
        k := key
        jobs[i] = func(ctx context.Context) error {
            value, err := fetchFromDB(ctx, k)
            if err != nil {
                return err
            }
            return cache.Set(k, value)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel health checks
func CheckServicesHealth(ctx context.Context, services []Service) error {
    jobs := make([]Job, len(services))
    for i, service := range services {
        s := service
        jobs[i] = func(ctx context.Context) error {
            return s.HealthCheck(ctx)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel notification sending
func SendNotifications(ctx context.Context, notifications []Notification) error {
    jobs := make([]Job, len(notifications))
    for i, notif := range notifications {
        n := notif
        jobs[i] = func(ctx context.Context) error {
            return sendNotification(ctx, n)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel image processing
func ProcessImages(ctx context.Context, images []string) error {
    jobs := make([]Job, len(images))
    for i, img := range images {
        imagePath := img
        jobs[i] = func(ctx context.Context) error {
            return resizeAndOptimize(ctx, imagePath)
        }
    }

    return RunParallel(ctx, jobs)
}
\`\`\`

**Real-World Benefits:**
- **10-100x Faster:** For I/O-bound operations
- **Better Resource Use:** Utilize all CPU cores
- **Responsive Systems:** Don't block on slow operations
- **Scalability:** Handle growing workloads

**Performance Comparison:**
\`\`\`
Sequential (10 jobs × 100ms each): 1000ms
Parallel (10 jobs × 100ms each):    100ms
Speedup: 10x
\`\`\`

**Important Considerations:**
- **Thread Safety:** Use mutexes for shared data
- **Error Handling:** sync.Once captures first error only
- **Context Cancellation:** Check ctx.Done() in each goroutine
- **Closure Variables:** Pass variables as parameters to avoid issues

**When to Use:**
- Independent jobs with no dependencies
- I/O-bound operations (API calls, file I/O, database queries)
- CPU-bound operations that benefit from parallelism
- Multiple health checks or validations

**When NOT to Use:**
- Jobs have dependencies on each other
- Order of execution matters
- Shared resources without proper synchronization
- Need to limit concurrent operations (use RunParallelBounded instead)

The sync.Once pattern ensures you get only the first error, which is usually sufficient for failing fast. All goroutines still complete, but you don't see every error.`,	order: 2,
	translations: {
		ru: {
			title: 'Параллельное выполнение',
			description: `Реализуйте **RunParallel**, который выполняет все задачи одновременно в отдельных горутинах.

**Требования:**
1. Создайте функцию \`RunParallel(ctx context.Context, jobs []Job) error\`
2. Обработайте nil context (верните nil)
3. Проверьте контекст перед запуском задач
4. Выполняйте все задачи одновременно в горутинах
5. Пропускайте nil задачи
6. Дождитесь завершения всех задач используя sync.WaitGroup
7. Верните первую встреченную ошибку (используйте sync.Once)
8. Проверяйте отмену контекста в каждой горутине

**Пример:**
\`\`\`go
jobs := []Job{
    func(ctx context.Context) error {
        time.Sleep(100 * time.Millisecond)
        fmt.Println("Job 1")
        return nil
    },
    func(ctx context.Context) error {
        time.Sleep(50 * time.Millisecond)
        fmt.Println("Job 2")
        return nil
    },
    func(ctx context.Context) error {
        time.Sleep(75 * time.Millisecond)
        return errors.New("job 3 failed")
    },
}

err := RunParallel(ctx, jobs)
// Вывод может быть: Job 2, Job 3 failed, Job 1 (порядок меняется)
// err = "job 3 failed" (первая захваченная ошибка)
// Все задачи выполняются одновременно
`,
			hint1: `Используйте sync.WaitGroup для отслеживания горутин. Вызовите wg.Add(1) перед запуском каждой горутины и wg.Done() когда она завершится.`,
			hint2: `Используйте sync.Once с функцией-помощником recordErr для захвата только первой ошибки. Передавайте job как параметр горутине чтобы избежать проблем с замыканиями.`,
			whyItMatters: `RunParallel значительно ускоряет выполнение независимых задач запуская их одновременно, необходим для высокопроизводительных приложений.

**Почему параллельное выполнение:**
- **Скорость:** Множество задач выполняются одновременно
- **Пропускная способность:** Обработка больше работы за меньшее время
- **Использование ресурсов:** Использование всех доступных ядер CPU
- **Масштабируемость:** Эффективная обработка больших нагрузок

**Продакшен паттерн:**
\`\`\`go
// Parallel API calls
func FetchMultipleUsers(ctx context.Context, userIDs []string) (map[string]*User, error) {
    var (
        mu    sync.Mutex
        users = make(map[string]*User)
    )

    jobs := make([]Job, len(userIDs))
    for i, id := range userIDs {
        userID := id
        jobs[i] = func(ctx context.Context) error {
            user, err := fetchUser(ctx, userID)
            if err != nil {
                return err
            }
            mu.Lock()
            users[userID] = user
            mu.Unlock()
            return nil
        }
    }

    if err := RunParallel(ctx, jobs); err != nil {
        return nil, err
    }

    return users, nil
}

// Parallel data validation
func ValidateRecords(ctx context.Context, records []Record) error {
    jobs := make([]Job, len(records))
    for i, record := range records {
        r := record
        jobs[i] = func(ctx context.Context) error {
            return validateRecord(ctx, r)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel file processing
func ProcessFiles(ctx context.Context, files []string) error {
    jobs := make([]Job, len(files))
    for i, file := range files {
        filename := file
        jobs[i] = func(ctx context.Context) error {
            data, err := os.ReadFile(filename)
            if err != nil {
                return err
            }
            return processData(ctx, data)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel cache warming
func WarmCache(ctx context.Context, keys []string) error {
    jobs := make([]Job, len(keys))
    for i, key := range keys {
        k := key
        jobs[i] = func(ctx context.Context) error {
            value, err := fetchFromDB(ctx, k)
            if err != nil {
                return err
            }
            return cache.Set(k, value)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel health checks
func CheckServicesHealth(ctx context.Context, services []Service) error {
    jobs := make([]Job, len(services))
    for i, service := range services {
        s := service
        jobs[i] = func(ctx context.Context) error {
            return s.HealthCheck(ctx)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel notification sending
func SendNotifications(ctx context.Context, notifications []Notification) error {
    jobs := make([]Job, len(notifications))
    for i, notif := range notifications {
        n := notif
        jobs[i] = func(ctx context.Context) error {
            return sendNotification(ctx, n)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel image processing
func ProcessImages(ctx context.Context, images []string) error {
    jobs := make([]Job, len(images))
    for i, img := range images {
        imagePath := img
        jobs[i] = func(ctx context.Context) error {
            return resizeAndOptimize(ctx, imagePath)
        }
    }

    return RunParallel(ctx, jobs)
}
\`\`\`

**Практические преимущества:**
- **10-100x быстрее:** Для I/O-bound операций
- **Лучшее использование ресурсов:** Использование всех ядер CPU
- **Отзывчивые системы:** Не блокируется на медленных операциях
- **Масштабируемость:** Обработка растущих нагрузок

**Сравнение производительности:**
\`\`\`
Последовательно (10 задач × 100ms каждая): 1000ms
Параллельно (10 задач × 100ms каждая): 100ms
Ускорение: 10x
\`\`\`

**Важные соображения:**
- **Потокобезопасность:** Используйте мьютексы для разделяемых данных
- **Обработка ошибок:** sync.Once захватывает только первую ошибку
- **Отмена контекста:** Проверяйте ctx.Done() в каждой горутине
- **Переменные замыкания:** Передавайте переменные как параметры во избежание проблем

**Когда использовать:**
- Независимые задачи без зависимостей
- I/O-bound операции (API вызовы, файловый I/O, запросы к БД)
- CPU-bound операции которые выигрывают от параллелизма
- Множественные проверки работоспособности или валидации

**Когда НЕ использовать:**
- Задачи имеют зависимости друг от друга
- Порядок выполнения важен
- Разделяемые ресурсы без надлежащей синхронизации
- Нужно ограничить конкурентные операции (используйте RunParallelBounded)

Паттерн sync.Once обеспечивает получение только первой ошибки, что обычно достаточно для быстрого отказа. Все горутины всё равно завершаются, но вы не видите каждую ошибку.`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type Job func(context.Context) error

func RunParallel(ctx context.Context, jobs []Job) error {
	if ctx == nil {                                                 // Обработка nil контекста
		return nil                                              // Возврат nil для безопасности
	}
	var (
		wg       sync.WaitGroup                                 // Ожидание всех горутин
		once     sync.Once                                      // Захват только первой ошибки
		firstErr error                                          // Хранение первой ошибки
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
		wg.Add(1)                                               // Увеличение счётчика wait group
		go func(job Job) {                                      // Запуск горутины
			defer wg.Done()                                 // Уменьшение при завершении
			select {                                        // Проверка контекста сначала
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
			title: 'Parallel bajarish',
			description: `Barcha vazifalarni alohida goroutinalarda bir vaqtda bajaradigan **RunParallel** ni amalga oshiring.

**Talablar:**
1. \`RunParallel(ctx context.Context, jobs []Job) error\` funksiyasini yarating
2. nil kontekstni ishlang (nil qaytaring)
3. Vazifalarni boshlashdan oldin kontekstni tekshiring
4. Barcha vazifalarni goroutinalarda bir vaqtda bajaring
5. nil vazifalarni o'tkazib yuboring
6. sync.WaitGroup dan foydalanib barcha vazifalar tugashini kuting
7. Duch kelgan birinchi xatoni qaytaring (sync.Once dan foydalaning)
8. Har bir goroutinada kontekst bekor qilinishini tekshiring

**Misol:**
\`\`\`go
jobs := []Job{
    func(ctx context.Context) error {
        time.Sleep(100 * time.Millisecond)
        fmt.Println("Job 1")
        return nil
    },
    func(ctx context.Context) error {
        time.Sleep(50 * time.Millisecond)
        fmt.Println("Job 2")
        return nil
    },
    func(ctx context.Context) error {
        time.Sleep(75 * time.Millisecond)
        return errors.New("job 3 failed")
    },
}

err := RunParallel(ctx, jobs)
// Chiqish: Job 2, Job 3 failed, Job 1 bo'lishi mumkin (tartib o'zgaradi)
// err = "job 3 failed" (birinchi ushlangan xato)
// Barcha vazifalar bir vaqtda ishlaydi
`,
			hint1: `Goroutinalarni kuzatish uchun sync.WaitGroup dan foydalaning. Har bir goroutinani ishga tushirishdan oldin wg.Add(1) ni chaqiring va tugaganda wg.Done() ni chaqiring.`,
			hint2: `Faqat birinchi xatoni ushlash uchun recordErr yordamchi funksiyasi bilan sync.Once dan foydalaning. Yopilish muammolaridan qochish uchun jobni goroutinaga parametr sifatida o'tkazing.`,
			whyItMatters: `RunParallel mustaqil vazifalarni bir vaqtda ishga tushirish orqali sezilarli darajada tezlashtiradi, yuqori samaradorlikdagi ilovalar uchun zarur.

**Nima uchun parallel bajarish:**
- **Tezlik:** Ko'plab vazifalar bir vaqtda bajariladi
- **O'tkazuvchanlik:** Kamroq vaqtda ko'proq ishni qayta ishlash
- **Resurslardan foydalanish:** Barcha mavjud CPU yadrolaridan foydalanish
- **Kengaytirilish:** Katta yuklarni samarali boshqarish

**Ishlab chiqarish patterni:**
\`\`\`go
// Parallel API calls
func FetchMultipleUsers(ctx context.Context, userIDs []string) (map[string]*User, error) {
    var (
        mu    sync.Mutex
        users = make(map[string]*User)
    )

    jobs := make([]Job, len(userIDs))
    for i, id := range userIDs {
        userID := id
        jobs[i] = func(ctx context.Context) error {
            user, err := fetchUser(ctx, userID)
            if err != nil {
                return err
            }
            mu.Lock()
            users[userID] = user
            mu.Unlock()
            return nil
        }
    }

    if err := RunParallel(ctx, jobs); err != nil {
        return nil, err
    }

    return users, nil
}

// Parallel data validation
func ValidateRecords(ctx context.Context, records []Record) error {
    jobs := make([]Job, len(records))
    for i, record := range records {
        r := record
        jobs[i] = func(ctx context.Context) error {
            return validateRecord(ctx, r)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel file processing
func ProcessFiles(ctx context.Context, files []string) error {
    jobs := make([]Job, len(files))
    for i, file := range files {
        filename := file
        jobs[i] = func(ctx context.Context) error {
            data, err := os.ReadFile(filename)
            if err != nil {
                return err
            }
            return processData(ctx, data)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel cache warming
func WarmCache(ctx context.Context, keys []string) error {
    jobs := make([]Job, len(keys))
    for i, key := range keys {
        k := key
        jobs[i] = func(ctx context.Context) error {
            value, err := fetchFromDB(ctx, k)
            if err != nil {
                return err
            }
            return cache.Set(k, value)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel health checks
func CheckServicesHealth(ctx context.Context, services []Service) error {
    jobs := make([]Job, len(services))
    for i, service := range services {
        s := service
        jobs[i] = func(ctx context.Context) error {
            return s.HealthCheck(ctx)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel notification sending
func SendNotifications(ctx context.Context, notifications []Notification) error {
    jobs := make([]Job, len(notifications))
    for i, notif := range notifications {
        n := notif
        jobs[i] = func(ctx context.Context) error {
            return sendNotification(ctx, n)
        }
    }

    return RunParallel(ctx, jobs)
}

// Parallel image processing
func ProcessImages(ctx context.Context, images []string) error {
    jobs := make([]Job, len(images))
    for i, img := range images {
        imagePath := img
        jobs[i] = func(ctx context.Context) error {
            return resizeAndOptimize(ctx, imagePath)
        }
    }

    return RunParallel(ctx, jobs)
}
\`\`\`

**Amaliy foydalari:**
- **10-100x tezroq:** I/O-bound operatsiyalar uchun
- **Resurslardan yaxshiroq foydalanish:** Barcha CPU yadrolaridan foydalanish
- **Javob beradigan tizimlar:** Sekin operatsiyalarda bloklanmaydi
- **Kengayish:** O'sib borayotgan yuklarni boshqarish

**Samaradorlikni taqqoslash:**
\`\`\`
Ketma-ket (10 vazifa × 100ms har biri): 1000ms
Parallel (10 vazifa × 100ms har biri): 100ms
Tezlashtirish: 10x
\`\`\`

**Muhim fikrlar:**
- **Potok xavfsizligi:** Umumiy ma'lumotlar uchun mutexlardan foydalaning
- **Xatolarni boshqarish:** sync.Once faqat birinchi xatoni ushlaydi
- **Kontekstni bekor qilish:** Har bir goroutinada ctx.Done() ni tekshiring
- **Yopilish o'zgaruvchilari:** Muammolardan qochish uchun o'zgaruvchilarni parametr sifatida o'tkazing

**Qachon ishlatiladi:**
- Bog'liqliksiz mustaqil vazifalar
- I/O-bound operatsiyalar (API chaqiruvlar, fayl I/O, ma'lumotlar bazasi so'rovlari)
- Parallellikdan foyda oladigan CPU-bound operatsiyalar
- Ko'plab sog'liqni tekshirish yoki tekshiruvlar

**Qachon ishlatilMAYDI:**
- Vazifalar bir-biriga bog'liq
- Bajarish tartibi muhim
- To'g'ri sinxronizatsiya qilinmagan umumiy resurslar
- Parallel operatsiyalarni cheklash kerak (RunParallelBounded dan foydalaning)

sync.Once patterni faqat birinchi xatoni olishni ta'minlaydi, bu odatda tez muvaffaqiyatsizlik uchun etarli. Barcha goroutinalar baribir tugaydi, lekin siz har bir xatoni ko'rmaysiz.`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type Job func(context.Context) error

func RunParallel(ctx context.Context, jobs []Job) error {
	if ctx == nil {                                                 // nil kontekstni ishlash
		return nil                                              // Xavfsizlik uchun nil qaytarish
	}
	var (
		wg       sync.WaitGroup                                 // Barcha goroutinalarni kutish
		once     sync.Once                                      // Faqat birinchi xatoni ushlash
		firstErr error                                          // Birinchi xatoni saqlash
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
		wg.Add(1)                                               // Wait group hisoblagichini oshirish
		go func(job Job) {                                      // Goroutinani ishga tushirish
			defer wg.Done()                                 // Tugaganda kamaytirish
			select {                                        // Avval kontekstni tekshirish
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
