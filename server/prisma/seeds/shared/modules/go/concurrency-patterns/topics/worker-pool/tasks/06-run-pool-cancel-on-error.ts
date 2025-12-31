import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-workerpool-run-pool-cancel-on-error',
	title: 'Run Pool Cancel On Error',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'worker-pool', 'error-handling', 'cancellation'],
	estimatedTime: '35m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **RunPoolCancelOnError** that creates a worker pool that automatically cancels all workers when the first error occurs.

**Requirements:**
1. Create function \`RunPoolCancelOnError(ctx context.Context, jobs <-chan Job, workers int) error\`
2. Handle nil context (return nil)
3. Handle workers <= 0 (set to 1)
4. Create derived context with cancel function
5. Create fixed number of worker goroutines
6. When any job returns error, cancel context immediately (use sync.Once)
7. All workers should exit when context is cancelled
8. Return first error encountered
9. Defer cancel to ensure cleanup

**Difference from RunPool:**
- **RunPool:** Workers continue processing even after error
- **RunPoolCancelOnError:** First error stops all workers immediately

**Example:**
\`\`\`go
jobs := make(chan Job, 10)

go func() {
    jobs <- func(ctx context.Context) error {
        time.Sleep(10 * time.Millisecond)
        return nil // job 1 succeeds
    }
    jobs <- func(ctx context.Context) error {
        return errors.New("job 2 failed") // triggers cancellation
    }
    jobs <- func(ctx context.Context) error {
        // This job likely won't execute due to cancellation
        time.Sleep(100 * time.Millisecond)
        return nil
    }
    close(jobs)
}()

err := RunPoolCancelOnError(ctx, jobs, 3)
// err = "job 2 failed"
// Workers stop processing remaining jobs after first error
\`\`\`

**Constraints:**
- Must create derived context with context.WithCancel
- Must call cancel() when first error occurs
- Must use sync.Once to ensure cancel called once
- Must defer cancel() for cleanup`,
	initialCode: `package concurrency

import (
	"context"
	"sync"
)

type Job func(context.Context) error

// TODO: Implement RunPoolCancelOnError
func RunPoolCancelOnError($2) error {
	return nil // TODO: Implement
}`,
	testCode: `package concurrency

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	err := RunPoolCancelOnError(nil, nil, 1)
	if err != nil {
		t.Errorf("expected nil for nil context, got %v", err)
	}
}

func Test2(t *testing.T) {
	jobs := make(chan Job)
	close(jobs)
	err := RunPoolCancelOnError(context.Background(), jobs, 1)
	if err != nil {
		t.Errorf("expected nil for closed channel, got %v", err)
	}
}

func Test3(t *testing.T) {
	jobs := make(chan Job, 3)
	jobs <- func(ctx context.Context) error { return nil }
	jobs <- func(ctx context.Context) error { return nil }
	jobs <- func(ctx context.Context) error { return nil }
	close(jobs)
	err := RunPoolCancelOnError(context.Background(), jobs, 2)
	if err != nil {
		t.Errorf("expected nil for successful jobs, got %v", err)
	}
}

func Test4(t *testing.T) {
	jobs := make(chan Job, 1)
	expectedErr := errors.New("test error")
	jobs <- func(ctx context.Context) error { return expectedErr }
	close(jobs)
	err := RunPoolCancelOnError(context.Background(), jobs, 1)
	if err != expectedErr {
		t.Errorf("expected %v, got %v", expectedErr, err)
	}
}

func Test5(t *testing.T) {
	jobs := make(chan Job, 5)
	var count atomic.Int32
	expectedErr := errors.New("cancel error")
	jobs <- func(ctx context.Context) error { count.Add(1); time.Sleep(50*time.Millisecond); return nil }
	jobs <- func(ctx context.Context) error { count.Add(1); return expectedErr }
	jobs <- func(ctx context.Context) error { time.Sleep(100*time.Millisecond); count.Add(1); return nil }
	jobs <- func(ctx context.Context) error { time.Sleep(100*time.Millisecond); count.Add(1); return nil }
	jobs <- func(ctx context.Context) error { time.Sleep(100*time.Millisecond); count.Add(1); return nil }
	close(jobs)
	start := time.Now()
	err := RunPoolCancelOnError(context.Background(), jobs, 2)
	elapsed := time.Since(start)
	if err != expectedErr {
		t.Errorf("expected %v, got %v", expectedErr, err)
	}
	if elapsed > 200*time.Millisecond {
		t.Errorf("expected fast cancellation, took %v", elapsed)
	}
}

func Test6(t *testing.T) {
	jobs := make(chan Job, 1)
	jobs <- func(ctx context.Context) error { return nil }
	close(jobs)
	err := RunPoolCancelOnError(context.Background(), jobs, 0)
	if err != nil {
		t.Errorf("expected nil with workers=0 (should default to 1), got %v", err)
	}
}

func Test7(t *testing.T) {
	jobs := make(chan Job, 1)
	jobs <- func(ctx context.Context) error { return nil }
	close(jobs)
	err := RunPoolCancelOnError(context.Background(), jobs, -5)
	if err != nil {
		t.Errorf("expected nil with negative workers, got %v", err)
	}
}

func Test8(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	jobs := make(chan Job, 1)
	jobs <- func(ctx context.Context) error { return nil }
	close(jobs)
	err := RunPoolCancelOnError(ctx, jobs, 1)
	if err != context.Canceled {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func Test9(t *testing.T) {
	jobs := make(chan Job, 2)
	jobs <- nil
	jobs <- func(ctx context.Context) error { return nil }
	close(jobs)
	err := RunPoolCancelOnError(context.Background(), jobs, 1)
	if err != nil {
		t.Errorf("expected nil when skipping nil job, got %v", err)
	}
}

func Test10(t *testing.T) {
	jobs := make(chan Job, 3)
	err1 := errors.New("error1")
	err2 := errors.New("error2")
	jobs <- func(ctx context.Context) error { time.Sleep(10*time.Millisecond); return err1 }
	jobs <- func(ctx context.Context) error { time.Sleep(10*time.Millisecond); return err2 }
	jobs <- func(ctx context.Context) error { return nil }
	close(jobs)
	err := RunPoolCancelOnError(context.Background(), jobs, 3)
	if err != err1 && err != err2 {
		t.Errorf("expected one of the errors, got %v", err)
	}
}
`,
	solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type Job func(context.Context) error

func RunPoolCancelOnError(ctx context.Context, jobs <-chan Job, workers int) error {
	if ctx == nil {                                                 // Handle nil context
		return nil                                              // Return nil for safety
	}
	if workers <= 0 {                                               // Handle invalid workers count
		workers = 1                                             // Set minimum workers
	}
	ctx, cancel := context.WithCancel(ctx)                          // Create cancellable context
	defer cancel()                                                  // Always cancel to free resources
	var (
		wg       sync.WaitGroup                                 // Track all workers
		once     sync.Once                                      // Ensure cancel called once
		firstErr error                                          // Store first error
	)
	worker := func() {                                              // Worker function
		defer wg.Done()                                         // Decrement counter when done
		for {                                                   // Worker loop
			select {                                        // Check context or receive job
			case <-ctx.Done():                              // Context cancelled
				return                                  // Exit worker
			case job, ok := <-jobs:                         // Receive job from channel
				if !ok {                                // Channel closed
					return                          // Exit worker
				}
				if job == nil {                         // Skip nil job
					continue                        // Next iteration
				}
				if err := job(ctx); err != nil {        // Execute job
					once.Do(func() {                // Execute once only
						firstErr = err          // Store first error
						cancel()                // Cancel all workers
					})
				}
			}
		}
	}
	wg.Add(workers)                                                 // Add all workers to wait group
	for i := 0; i < workers; i++ {                                  // Create workers
		go worker()                                             // Launch worker goroutine
	}
	wg.Wait()                                                       // Wait for all workers to finish
	if firstErr != nil {                                            // Check if error occurred
		return firstErr                                         // Return first error
	}
	if err := ctx.Err(); err != nil {                               // Check context state
		return err                                              // Return context error
	}
	return nil                                                      // No errors
}`,
			hint1: `Create derived context: ctx, cancel := context.WithCancel(ctx). Use defer cancel() to ensure cleanup.`,
			hint2: `When job returns error, use once.Do(func() { firstErr = err; cancel() }) to capture error and cancel context once. Workers will exit when ctx.Done() fires.`,
			whyItMatters: `RunPoolCancelOnError implements fail-fast behavior, stopping all work immediately when an error occurs, saving resources and time.

**Why Cancel On Error:**
- **Fail Fast:** Don't waste resources on remaining work
- **Quick Feedback:** Report errors immediately
- **Resource Efficiency:** Stop unnecessary processing
- **Consistency:** All-or-nothing semantics

**Production Pattern:**
\`\`\`go
// Batch data validation - stop on first invalid record
func ValidateBatch(ctx context.Context, records []Record) error {
    jobs := make(chan Job, len(records))

    go func() {
        defer close(jobs)
        for _, record := range records {
            r := record
            select {
            case jobs <- func(ctx context.Context) error {
                return validateRecord(ctx, r)
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    // Stop validation on first error
    return RunPoolCancelOnError(ctx, jobs, 10)
}

// Dependency download - abort if any dependency fails
func DownloadDependencies(ctx context.Context, deps []Dependency) error {
    jobs := make(chan Job, len(deps))

    go func() {
        defer close(jobs)
        for _, dep := range deps {
            d := dep
            select {
            case jobs <- func(ctx context.Context) error {
                return downloadDependency(ctx, d)
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    // If any dependency fails, stop all downloads
    return RunPoolCancelOnError(ctx, jobs, 5)
}

// Service health check - fail fast if critical service down
func CheckCriticalServices(ctx context.Context, services []Service) error {
    jobs := make(chan Job, len(services))

    go func() {
        defer close(jobs)
        for _, svc := range services {
            s := svc
            select {
            case jobs <- func(ctx context.Context) error {
                if err := s.HealthCheck(ctx); err != nil {
                    return fmt.Errorf("critical service %s failed: %w", s.Name, err)
                }
                return nil
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    return RunPoolCancelOnError(ctx, jobs, len(services))
}

// Database transaction - rollback if any operation fails
func ExecuteTransactionSteps(ctx context.Context, steps []DBOperation) error {
    jobs := make(chan Job, len(steps))

    tx, err := db.Begin()
    if err != nil {
        return err
    }

    go func() {
        defer close(jobs)
        for _, step := range steps {
            op := step
            select {
            case jobs <- func(ctx context.Context) error {
                return op.Execute(tx)
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    err = RunPoolCancelOnError(ctx, jobs, 1) // Sequential for transactions
    if err != nil {
        tx.Rollback()
        return err
    }

    return tx.Commit()
}

// File upload with cleanup on error
func UploadFiles(ctx context.Context, files []File) error {
    jobs := make(chan Job, len(files))
    uploaded := make([]string, 0, len(files))
    var mu sync.Mutex

    go func() {
        defer close(jobs)
        for _, file := range files {
            f := file
            select {
            case jobs <- func(ctx context.Context) error {
                url, err := uploadFile(ctx, f)
                if err != nil {
                    return err
                }
                mu.Lock()
                uploaded = append(uploaded, url)
                mu.Unlock()
                return nil
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    err := RunPoolCancelOnError(ctx, jobs, 5)
    if err != nil {
        // Cleanup uploaded files on error
        for _, url := range uploaded {
            deleteFile(url)
        }
        return err
    }

    return nil
}

// API request chain - abort if any request fails
func FetchRelatedData(ctx context.Context, userID string) (*UserData, error) {
    type fetchJob struct {
        name string
        fetch func(context.Context) error
    }

    var (
        profile  *Profile
        settings *Settings
        posts    []Post
        friends  []User
    )

    fetchJobs := []fetchJob{
        {"profile", func(ctx context.Context) error {
            var err error
            profile, err = fetchProfile(ctx, userID)
            return err
        }},
        {"settings", func(ctx context.Context) error {
            var err error
            settings, err = fetchSettings(ctx, userID)
            return err
        }},
        {"posts", func(ctx context.Context) error {
            var err error
            posts, err = fetchPosts(ctx, userID)
            return err
        }},
        {"friends", func(ctx context.Context) error {
            var err error
            friends, err = fetchFriends(ctx, userID)
            return err
        }},
    }

    jobs := make(chan Job, len(fetchJobs))
    go func() {
        defer close(jobs)
        for _, fj := range fetchJobs {
            job := fj
            select {
            case jobs <- func(ctx context.Context) error {
                if err := job.fetch(ctx); err != nil {
                    return fmt.Errorf("%s fetch failed: %w", job.name, err)
                }
                return nil
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    if err := RunPoolCancelOnError(ctx, jobs, 4); err != nil {
        return nil, err
    }

    return &UserData{
        Profile:  profile,
        Settings: settings,
        Posts:    posts,
        Friends:  friends,
    }, nil
}
\`\`\`

**Real-World Benefits:**
- **Resource Savings:** Don't process remaining jobs after error
- **Faster Failures:** Return error immediately
- **Cleaner Semantics:** All-or-nothing behavior
- **Better UX:** Quick error feedback

**When to Use:**
- **Validation:** Stop on first invalid item
- **Dependencies:** All must succeed
- **Critical Operations:** Fail if any step fails
- **Transactional Work:** All-or-nothing semantics
- **Resource Cleanup:** Stop before more resources allocated

**When NOT to Use:**
- **Best Effort:** Want partial results
- **Independent Jobs:** Errors in one shouldn't stop others
- **Logging/Monitoring:** Want to see all errors
- **Batch Processing:** Some failures acceptable

**Comparison:**
\`\`\`
RunPool:
- Job 1: Success
- Job 2: Error (recorded)
- Job 3: Still processes
- Job 4: Still processes
- Returns first error, but all jobs ran

RunPoolCancelOnError:
- Job 1: Success
- Job 2: Error (cancels context)
- Job 3: Skipped (context cancelled)
- Job 4: Skipped (context cancelled)
- Returns first error, remaining jobs cancelled
\`\`\`

**Context Cancellation Propagation:**
When cancel() is called:
1. ctx.Done() channel closes
2. All workers receive on ctx.Done()
3. Workers exit their loops
4. No new jobs are processed
5. WaitGroup completes
6. Function returns with error

This pattern is essential for implementing fail-fast behavior in concurrent systems.`,	order: 5,
	translations: {
		ru: {
			title: 'Пул с отменой при ошибке',
			description: `Реализуйте **RunPoolCancelOnError**, который создаёт пул воркеров, автоматически отменяющий всех воркеров при первой ошибке.

**Требования:**
1. Создайте функцию \`RunPoolCancelOnError(ctx context.Context, jobs <-chan Job, workers int) error\`
2. Обработайте nil context (верните nil)
3. Обработайте workers <= 0 (установите в 1)
4. Создайте производный контекст с функцией отмены
5. Создайте фиксированное количество воркер-горутин
6. Когда любая задача возвращает ошибку, немедленно отмените контекст (используйте sync.Once)
7. Все воркеры должны выйти при отмене контекста
8. Верните первую встреченную ошибку
9. Используйте defer cancel для обеспечения очистки

**Разница с RunPool:**
- **RunPool:** Воркеры продолжают обработку даже после ошибки
- **RunPoolCancelOnError:** Первая ошибка немедленно останавливает всех воркеров

**Пример:**
\`\`\`go
jobs := make(chan Job, 10)

go func() {
    jobs <- func(ctx context.Context) error {
        time.Sleep(10 * time.Millisecond)
        return nil // задача 1 успешна
    }
    jobs <- func(ctx context.Context) error {
        return errors.New("job 2 failed") // вызывает отмену
    }
    jobs <- func(ctx context.Context) error {
        // Эта задача скорее всего не выполнится из-за отмены
        time.Sleep(100 * time.Millisecond)
        return nil
    }
    close(jobs)
}()

err := RunPoolCancelOnError(ctx, jobs, 3)
// err = "job 2 failed"
// Воркеры прекращают обработку оставшихся задач после первой ошибки
`,
			hint1: `Создайте производный контекст: ctx, cancel := context.WithCancel(ctx). Используйте defer cancel() для обеспечения очистки.`,
			hint2: `Когда задача возвращает ошибку, используйте once.Do(func() { firstErr = err; cancel() }) для захвата ошибки и отмены контекста один раз. Воркеры выйдут когда сработает ctx.Done().`,
			whyItMatters: `RunPoolCancelOnError реализует критически важное поведение fail-fast, немедленно останавливая всю работу при ошибке, экономя ресурсы и время.

**Почему это важно:**

**Почему отменять при ошибке:**
- **Быстрый fail:** Не тратить ресурсы на оставшуюся работу
- **Быстрая обратная связь:** Немедленное сообщение об ошибках
- **Эффективность ресурсов:** Остановка ненужной обработки
- **Согласованность:** Семантика всё-или-ничего

**Продакшен паттерн:**
\`\`\`go
// Пакетная валидация данных - остановка при первой невалидной записи
func ValidateBatch(ctx context.Context, records []Record) error {
    jobs := make(chan Job, len(records))

    go func() {
        defer close(jobs)
        for _, record := range records {
            r := record
            select {
            case jobs <- func(ctx context.Context) error {
                return validateRecord(ctx, r)
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    // Остановка валидации при первой ошибке
    return RunPoolCancelOnError(ctx, jobs, 10)
}

// Загрузка зависимостей - прерывание если любая зависимость провалилась
func DownloadDependencies(ctx context.Context, deps []Dependency) error {
    jobs := make(chan Job, len(deps))

    go func() {
        defer close(jobs)
        for _, dep := range deps {
            d := dep
            select {
            case jobs <- func(ctx context.Context) error {
                return downloadDependency(ctx, d)
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    // Если любая зависимость провалилась, остановить все загрузки
    return RunPoolCancelOnError(ctx, jobs, 5)
}

// Проверка здоровья сервиса - быстрый fail если критический сервис упал
func CheckCriticalServices(ctx context.Context, services []Service) error {
    jobs := make(chan Job, len(services))

    go func() {
        defer close(jobs)
        for _, svc := range services {
            s := svc
            select {
            case jobs <- func(ctx context.Context) error {
                if err := s.HealthCheck(ctx); err != nil {
                    return fmt.Errorf("critical service %s failed: %w", s.Name, err)
                }
                return nil
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    return RunPoolCancelOnError(ctx, jobs, len(services))
}

// Транзакция базы данных - откат если любая операция провалилась
func ExecuteTransactionSteps(ctx context.Context, steps []DBOperation) error {
    jobs := make(chan Job, len(steps))

    tx, err := db.Begin()
    if err != nil {
        return err
    }

    go func() {
        defer close(jobs)
        for _, step := range steps {
            op := step
            select {
            case jobs <- func(ctx context.Context) error {
                return op.Execute(tx)
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    err = RunPoolCancelOnError(ctx, jobs, 1) // Последовательно для транзакций
    if err != nil {
        tx.Rollback()
        return err
    }

    return tx.Commit()
}

// Загрузка файлов с очисткой при ошибке
func UploadFiles(ctx context.Context, files []File) error {
    jobs := make(chan Job, len(files))
    uploaded := make([]string, 0, len(files))
    var mu sync.Mutex

    go func() {
        defer close(jobs)
        for _, file := range files {
            f := file
            select {
            case jobs <- func(ctx context.Context) error {
                url, err := uploadFile(ctx, f)
                if err != nil {
                    return err
                }
                mu.Lock()
                uploaded = append(uploaded, url)
                mu.Unlock()
                return nil
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    err := RunPoolCancelOnError(ctx, jobs, 5)
    if err != nil {
        // Очистка загруженных файлов при ошибке
        for _, url := range uploaded {
            deleteFile(url)
        }
        return err
    }

    return nil
}
\`\`\`

**Реальные преимущества:**
- **Экономия ресурсов:** Не обрабатывать оставшиеся задачи после ошибки
- **Быстрые сбои:** Возврат ошибки немедленно
- **Чистая семантика:** Поведение всё-или-ничего
- **Лучший UX:** Быстрая обратная связь об ошибках
- **Экономия времени:** Не ждать завершения остальных задач
- **Защита данных:** Предотвращение частично выполненных операций

**Когда использовать:**
- **Валидация:** Остановка при первом невалидном элементе
- **Зависимости:** Все должны успешно выполниться
- **Критические операции:** Fail если любой шаг провалился
- **Транзакционная работа:** Семантика всё-или-ничего
- **Очистка ресурсов:** Остановка до выделения большего количества ресурсов

**Когда НЕ использовать:**
- **Best Effort:** Нужны частичные результаты
- **Независимые задачи:** Ошибки в одной не должны останавливать другие
- **Логирование/Мониторинг:** Хочется видеть все ошибки
- **Пакетная обработка:** Некоторые сбои приемлемы

**Сравнение:**
\`\`\`
RunPool:
- Задача 1: Успех
- Задача 2: Ошибка (записана)
- Задача 3: Всё равно обрабатывается
- Задача 4: Всё равно обрабатывается
- Возвращает первую ошибку, но все задачи выполнены

RunPoolCancelOnError:
- Задача 1: Успех
- Задача 2: Ошибка (отменяет контекст)
- Задача 3: Пропущена (контекст отменён)
- Задача 4: Пропущена (контекст отменён)
- Возвращает первую ошибку, оставшиеся задачи отменены
\`\`\`

**Распространение отмены контекста:**
Когда вызывается cancel():
1. канал ctx.Done() закрывается
2. Все воркеры получают из ctx.Done()
3. Воркеры выходят из своих циклов
4. Новые задачи не обрабатываются
5. WaitGroup завершается
6. Функция возвращает ошибку

**Реальный кейс: Микросервисная архитектура**
Система с 10 микросервисами, где каждый должен быть доступен для развертывания:
\`\`\`go
func CheckSystemHealth(ctx context.Context, services []Service) error {
    jobs := make(chan Job, len(services))

    go func() {
        defer close(jobs)
        for _, svc := range services {
            s := svc
            select {
            case jobs <- func(ctx context.Context) error {
                if err := s.HealthCheck(ctx); err != nil {
                    return fmt.Errorf("критический сервис %s недоступен: %w", s.Name, err)
                }
                return nil
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    return RunPoolCancelOnError(ctx, jobs, 10)
}

// Результаты:
// - До: Проверяли все 10 сервисов даже если первый упал (30 секунд)
// - После: Останавливаемся при первом упавшем сервисе (3 секунды)
// - Экономия времени: 90% для failed deployments
// - Быстрая обратная связь для DevOps команды
\`\`\`

**Реальный кейс 2: API запрос цепочка**
Получение связанных данных пользователя параллельно:
\`\`\`go
func FetchRelatedData(ctx context.Context, userID string) (*UserData, error) {
    type fetchJob struct {
        name string
        fetch func(context.Context) error
    }

    var (
        profile  *Profile
        settings *Settings
        posts    []Post
        friends  []User
    )

    fetchJobs := []fetchJob{
        {"profile", func(ctx context.Context) error {
            var err error
            profile, err = fetchProfile(ctx, userID)
            return err
        }},
        {"settings", func(ctx context.Context) error {
            var err error
            settings, err = fetchSettings(ctx, userID)
            return err
        }},
        {"posts", func(ctx context.Context) error {
            var err error
            posts, err = fetchPosts(ctx, userID)
            return err
        }},
        {"friends", func(ctx context.Context) error {
            var err error
            friends, err = fetchFriends(ctx, userID)
            return err
        }},
    }

    jobs := make(chan Job, len(fetchJobs))
    go func() {
        defer close(jobs)
        for _, fj := range fetchJobs {
            job := fj
            select {
            case jobs <- func(ctx context.Context) error {
                if err := job.fetch(ctx); err != nil {
                    return fmt.Errorf("%s fetch failed: %w", job.name, err)
                }
                return nil
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    if err := RunPoolCancelOnError(ctx, jobs, 4); err != nil {
        return nil, err
    }

    return &UserData{
        Profile:  profile,
        Settings: settings,
        Posts:    posts,
        Friends:  friends,
    }, nil
}

// Эффект:
// - До: Ждали загрузки всех 4 секций даже если профиль не найден (5 секунд)
// - После: Останавливаемся при первой ошибке (100ms)
// - Улучшение отзывчивости: 5000ms → 100ms
// - Экономия API вызовов: 75% при ошибках
\`\`\`

**Production Best Practices:**
1. Используйте RunPoolCancelOnError для операций где все должны успешно выполниться
2. Всегда проверяйте авторство коммитов перед amend
3. Реализуйте cleanup логику для частично выполненных операций
4. Добавляйте подробное логирование для отладки
5. Используйте тайм ауты для предотвращения бесконечного ожидания
6. Тестируйте сценарии с ошибками в разных точках выполнения
7. Документируйте какие операции используют fail-fast поведение
8. Мониторьте время выполнения до первой ошибки
9. Реализуйте механизмы отката для частично успешных операций

**Реальное влияние на бизнес:**
Компания с CI/CD pipeline:
- **До RunPoolCancelOnError**: Каждая failed сборка занимала 15 мин (проверяли все 20 тестов)
- **После RunPoolCancelOnError**: Failed сборки занимают 2 мин (останавливаемся при первом падении)
- **Экономия**: 13 мин × 50 failed сборок/день = 650 мин/день = 10.8 часов/день
- **Стоимость**: Экономия compute ресурсов $5K/месяц
- **Developer experience**: Быстрая обратная связь улучшила моральный дух команды

Этот паттерн необходим для реализации fail-fast поведения в конкурентных системах. Он гарантирует что ресурсы не тратятся на бесполезную работу и ошибки обнаруживаются быстро, что критично для production систем с высокой нагрузкой.`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type Job func(context.Context) error

func RunPoolCancelOnError(ctx context.Context, jobs <-chan Job, workers int) error {
	if ctx == nil {                                                 // Обработка nil контекста
		return nil                                              // Возврат nil для безопасности
	}
	if workers <= 0 {                                               // Обработка неверного количества воркеров
		workers = 1                                             // Установка минимального количества воркеров
	}
	ctx, cancel := context.WithCancel(ctx)                          // Создание отменяемого контекста
	defer cancel()                                                  // Всегда отменять для освобождения ресурсов
	var (
		wg       sync.WaitGroup                                 // Отслеживание всех воркеров
		once     sync.Once                                      // Обеспечение однократного вызова cancel
		firstErr error                                          // Хранение первой ошибки
	)
	worker := func() {                                              // Функция воркера
		defer wg.Done()                                         // Уменьшение счётчика при завершении
		for {                                                   // Цикл воркера
			select {                                        // Проверка контекста или получение задачи
			case <-ctx.Done():                              // Контекст отменён
				return                                  // Выход из воркера
			case job, ok := <-jobs:                         // Получение задачи из канала
				if !ok {                                // Канал закрыт
					return                          // Выход из воркера
				}
				if job == nil {                         // Пропуск nil задачи
					continue                        // Следующая итерация
				}
				if err := job(ctx); err != nil {        // Выполнение задачи
					once.Do(func() {                // Выполнить только один раз
						firstErr = err          // Сохранить первую ошибку
						cancel()                // Отменить всех воркеров
					})
				}
			}
		}
	}
	wg.Add(workers)                                                 // Добавление всех воркеров в wait group
	for i := 0; i < workers; i++ {                                  // Создание воркеров
		go worker()                                             // Запуск горутины воркера
	}
	wg.Wait()                                                       // Ожидание завершения всех воркеров
	if firstErr != nil {                                            // Проверка возникновения ошибки
		return firstErr                                         // Возврат первой ошибки
	}
	if err := ctx.Err(); err != nil {                               // Проверка состояния контекста
		return err                                              // Возврат ошибки контекста
	}
	return nil                                                      // Без ошибок
}`
		},
		uz: {
			title: 'Xatoda bekor qilinadigan pool',
			description: `Birinchi xato yuz berganda barcha workerlarni avtomatik bekor qiladigan worker pulini yaratadigan **RunPoolCancelOnError** ni amalga oshiring.

**Talablar:**
1. \`RunPoolCancelOnError(ctx context.Context, jobs <-chan Job, workers int) error\` funksiyasini yarating
2. nil kontekstni ishlang (nil qaytaring)
3. workers <= 0 ni ishlang (1 ga o'rnating)
4. Bekor qilish funksiyasi bilan hosil qilingan kontekst yarating
5. Belgilangan miqdorda worker goroutinalarini yarating
6. Biron vazifa xato qaytarganda, kontekstni darhol bekor qiling (sync.Once dan foydalaning)
7. Kontekst bekor qilinganda barcha workerlar chiqishi kerak
8. Duch kelgan birinchi xatoni qaytaring
9. Tozalashni ta'minlash uchun defer cancel dan foydalaning

**RunPool bilan farqi:**
- **RunPool:** Xatodan keyin ham workerlar qayta ishlashni davom ettiradi
- **RunPoolCancelOnError:** Birinchi xato darhol barcha workerlarni to'xtatadi

**Misol:**
\`\`\`go
jobs := make(chan Job, 10)

go func() {
    jobs <- func(ctx context.Context) error {
        time.Sleep(10 * time.Millisecond)
        return nil // vazifa 1 muvaffaqiyatli
    }
    jobs <- func(ctx context.Context) error {
        return errors.New("job 2 failed") // bekor qilishni boshlaydi
    }
    jobs <- func(ctx context.Context) error {
        // Bu vazifa bekor qilish tufayli bajarilmasligi mumkin
        time.Sleep(100 * time.Millisecond)
        return nil
    }
    close(jobs)
}()

err := RunPoolCancelOnError(ctx, jobs, 3)
// err = "job 2 failed"
// Workerlar birinchi xatodan keyin qolgan vazifalarni qayta ishlashni to'xtatadilar
`,
			hint1: `Hosil qilingan kontekst yarating: ctx, cancel := context.WithCancel(ctx). Tozalashni ta'minlash uchun defer cancel() dan foydalaning.`,
			hint2: `Vazifa xato qaytarganda, xatoni ushlash va kontekstni bir marta bekor qilish uchun once.Do(func() { firstErr = err; cancel() }) dan foydalaning. ctx.Done() ishga tushganda workerlar chiqadi.`,
			whyItMatters: `RunPoolCancelOnError muhim fail-fast xatti-harakatini amalga oshiradi, xato yuz berganda barcha ishlarni darhol to'xtatib, resurslar va vaqtni tejaydi.

**Nima uchun bu muhim:**

**Nima uchun xatoda bekor qilish:**
- **Tez fail:** Qolgan ish uchun resurslarni sarflamaslik
- **Tez qaytarilma:** Xatolar haqida darhol xabar berish
- **Resurslar samaradorligi:** Keraksiz qayta ishlashni to'xtatish
- **Barqarorlik:** Hammasi-yoki-hech narsa semantikasi

**Ishlab chiqarish patterni:**
\`\`\`go
// Paket ma'lumotlarni tekshirish - birinchi noto'g'ri yozuvda to'xtatish
func ValidateBatch(ctx context.Context, records []Record) error {
    jobs := make(chan Job, len(records))

    go func() {
        defer close(jobs)
        for _, record := range records {
            r := record
            select {
            case jobs <- func(ctx context.Context) error {
                return validateRecord(ctx, r)
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    // Birinchi xatoda tekshirishni to'xtatish
    return RunPoolCancelOnError(ctx, jobs, 10)
}

// Bog'liqliklarni yuklab olish - agar biron bog'liqlik muvaffaqiyatsiz bo'lsa to'xtatish
func DownloadDependencies(ctx context.Context, deps []Dependency) error {
    jobs := make(chan Job, len(deps))

    go func() {
        defer close(jobs)
        for _, dep := range deps {
            d := dep
            select {
            case jobs <- func(ctx context.Context) error {
                return downloadDependency(ctx, d)
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    // Agar biron bog'liqlik muvaffaqiyatsiz bo'lsa, barcha yuklab olishlarni to'xtatish
    return RunPoolCancelOnError(ctx, jobs, 5)
}

// Xizmat sog'ligini tekshirish - agar muhim xizmat ishlamay qolsa tez fail
func CheckCriticalServices(ctx context.Context, services []Service) error {
    jobs := make(chan Job, len(services))

    go func() {
        defer close(jobs)
        for _, svc := range services {
            s := svc
            select {
            case jobs <- func(ctx context.Context) error {
                if err := s.HealthCheck(ctx); err != nil {
                    return fmt.Errorf("critical service %s failed: %w", s.Name, err)
                }
                return nil
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    return RunPoolCancelOnError(ctx, jobs, len(services))
}
\`\`\`

**Haqiqiy foydalari:**
- **Resurs tejash:** Xatodan keyin qolgan vazifalarni qayta ishlamaslik
- **Tez muvaffaqiyatsizliklar:** Xatoni darhol qaytarish
- **Toza semantika:** Hammasi-yoki-hech narsa xatti-harakati
- **Yaxshiroq UX:** Xatolar haqida tez qaytarilma

**Qachon ishlatiladi:**
- **Tekshirish:** Birinchi noto'g'ri elementda to'xtatish
- **Bog'liqliklar:** Hammasi muvaffaqiyatli bo'lishi kerak
- **Muhim operatsiyalar:** Agar biron qadam muvaffaqiyatsiz bo'lsa fail
- **Tranzaksion ish:** Hammasi-yoki-hech narsa semantikasi
- **Resurslarni tozalash:** Ko'proq resurslar ajratilishidan oldin to'xtatish

**Qachon ISHLATILMASLIGI kerak:**
- **Best Effort:** Qisman natijalar kerak
- **Mustaqil vazifalar:** Birida xato boshqalarni to'xtatmasligi kerak
- **Logging/Monitoring:** Barcha xatolarni ko'rish kerak
- **Paket qayta ishlash:** Ba'zi muvaffaqiyatsizliklar qabul qilinadigan

**Taqqoslash:**
\`\`\`
RunPool:
- Vazifa 1: Muvaffaqiyat
- Vazifa 2: Xato (yozilgan)
- Vazifa 3: Baribir qayta ishlanadi
- Vazifa 4: Baribir qayta ishlanadi
- Birinchi xatoni qaytaradi, lekin barcha vazifalar bajarildi

RunPoolCancelOnError:
- Vazifa 1: Muvaffaqiyat
- Vazifa 2: Xato (kontekstni bekor qiladi)
- Vazifa 3: O'tkazib yuborilgan (kontekst bekor qilindi)
- Vazifa 4: O'tkazib yuborilgan (kontekst bekor qilindi)
- Birinchi xatoni qaytaradi, qolgan vazifalar bekor qilindi
\`\`\`

**Kontekst bekor qilish tarqalishi:**
cancel() chaqirilganda:
1. ctx.Done() kanali yopiladi
2. Barcha workerlar ctx.Done() dan oladi
3. Workerlar o'z tsikllaridan chiqadi
4. Yangi vazifalar qayta ishlanmaydi
5. WaitGroup tugaydi
6. Funksiya xato bilan qaytadi

**Haqiqiy holat: Mikroservis arxitekturasi**
Har biri deploy uchun mavjud bo'lishi kerak bo'lgan 10 ta mikroservisli tizim:
\`\`\`go
func CheckSystemHealth(ctx context.Context, services []Service) error {
    jobs := make(chan Job, len(services))

    go func() {
        defer close(jobs)
        for _, svc := range services {
            s := svc
            select {
            case jobs <- func(ctx context.Context) error {
                if err := s.HealthCheck(ctx); err != nil {
                    return fmt.Errorf("muhim xizmat %s mavjud emas: %w", s.Name, err)
                }
                return nil
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    return RunPoolCancelOnError(ctx, jobs, 10)
}

// Natijalar:
// - Oldin: Birinchisi qulab tushgan bo'lsa ham barcha 10 xizmatni tekshirdik (30 soniya)
// - Keyin: Birinchi nosoz xizmatda to'xtaymiz (3 soniya)
// - Vaqt tejash: Muvaffaqiyatsiz deploymentlar uchun 90%
// - DevOps jamoasi uchun tez qaytarilma
\`\`\`

**Haqiqiy holat 2: API so'rovlar zanjiri**
Foydalanuvchi bog'liq ma'lumotlarini parallel ravishda olish:
\`\`\`go
func FetchRelatedData(ctx context.Context, userID string) (*UserData, error) {
    type fetchJob struct {
        name string
        fetch func(context.Context) error
    }

    var (
        profile  *Profile
        settings *Settings
        posts    []Post
        friends  []User
    )

    fetchJobs := []fetchJob{
        {"profile", func(ctx context.Context) error {
            var err error
            profile, err = fetchProfile(ctx, userID)
            return err
        }},
        {"settings", func(ctx context.Context) error {
            var err error
            settings, err = fetchSettings(ctx, userID)
            return err
        }},
        {"posts", func(ctx context.Context) error {
            var err error
            posts, err = fetchPosts(ctx, userID)
            return err
        }},
        {"friends", func(ctx context.Context) error {
            var err error
            friends, err = fetchFriends(ctx, userID)
            return err
        }},
    }

    jobs := make(chan Job, len(fetchJobs))
    go func() {
        defer close(jobs)
        for _, fj := range fetchJobs {
            job := fj
            select {
            case jobs <- func(ctx context.Context) error {
                if err := job.fetch(ctx); err != nil {
                    return fmt.Errorf("%s olish muvaffaqiyatsiz: %w", job.name, err)
                }
                return nil
            }:
            case <-ctx.Done():
                return
            }
        }
    }()

    if err := RunPoolCancelOnError(ctx, jobs, 4); err != nil {
        return nil, err
    }

    return &UserData{
        Profile:  profile,
        Settings: settings,
        Posts:    posts,
        Friends:  friends,
    }, nil
}

// Ta'sir:
// - Oldin: Profil topilmagan bo'lsa ham barcha 4 bo'limning yuklanishini kutdik (5 soniya)
// - Keyin: Birinchi xatoda to'xtaymiz (100ms)
// - Sezgirlik yaxshilanishi: 5000ms → 100ms
// - API chaqiruvlarini tejash: Xatolarda 75%
\`\`\`

**Production Best Practices:**
1. Hammasi muvaffaqiyatli bo'lishi kerak bo'lgan operatsiyalar uchun RunPoolCancelOnError dan foydalaning
2. Amend qilishdan oldin doim commit muallifligini tekshiring
3. Qisman bajarilgan operatsiyalar uchun cleanup mantiqini amalga oshiring
4. Debugging uchun batafsil logging qo'shing
5. Cheksiz kutishning oldini olish uchun timeoutlardan foydalaning
6. Xatolarni turli nuqtalarda test qilish stsenariylarini sinab ko'ring
7. Qaysi operatsiyalar fail-fast xatti-harakatidan foydalanishini hujjatlang
8. Birinchi xatogacha bajarilish vaqtini monitor qiling
9. Qisman muvaffaqiyatli operatsiyalar uchun orqaga qaytarish mexanizmlarini amalga oshiring

**Biznesga haqiqiy ta'sir:**
CI/CD pipelineli kompaniya:
- **RunPoolCancelOnError dan oldin**: Har bir muvaffaqiyatsiz build 15 daqiqa davom etdi (barcha 20 testni tekshirdik)
- **RunPoolCancelOnError dan keyin**: Muvaffaqiyatsiz buildlar 2 daqiqa davom etadi (birinchi muvaffaqiyatsizlikda to'xtaymiz)
- **Tejamkorlik**: 13 daqiqa × kuniga 50 ta muvaffaqiyatsiz build = kuniga 650 daqiqa = kuniga 10.8 soat
- **Xarajat**: Hisoblash resurslarini oyiga $5K tejash
- **Developer tajribasi**: Tez qaytarilma jamoa ruhini yaxshiladi

Bu pattern parallel tizimlarda fail-fast xatti-harakatini amalga oshirish uchun zarur. U resurslarning befoyda ishga sarflanmasligini va xatolarning tez aniqlanishini kafolatlaydi, bu yuqori yukli production tizimlari uchun muhimdir.`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type Job func(context.Context) error

func RunPoolCancelOnError(ctx context.Context, jobs <-chan Job, workers int) error {
	if ctx == nil {                                                 // nil kontekstni ishlash
		return nil                                              // Xavfsizlik uchun nil qaytarish
	}
	if workers <= 0 {                                               // Noto'g'ri workerlar sonini ishlash
		workers = 1                                             // Minimal workerlar sonini o'rnatish
	}
	ctx, cancel := context.WithCancel(ctx)                          // Bekor qilinadigan kontekst yaratish
	defer cancel()                                                  // Resurslarni ozod qilish uchun doim bekor qilish
	var (
		wg       sync.WaitGroup                                 // Barcha workerlarni kuzatish
		once     sync.Once                                      // cancel bir marta chaqirilishini ta'minlash
		firstErr error                                          // Birinchi xatoni saqlash
	)
	worker := func() {                                              // Worker funksiyasi
		defer wg.Done()                                         // Tugaganda hisoblagichni kamaytirish
		for {                                                   // Worker tsikli
			select {                                        // Kontekstni tekshirish yoki vazifani qabul qilish
			case <-ctx.Done():                              // Kontekst bekor qilindi
				return                                  // Workerdan chiqish
			case job, ok := <-jobs:                         // Kanaldan vazifani qabul qilish
				if !ok {                                // Kanal yopildi
					return                          // Workerdan chiqish
				}
				if job == nil {                         // nil vazifani o'tkazib yuborish
					continue                        // Keyingi iteratsiya
				}
				if err := job(ctx); err != nil {        // Vazifani bajarish
					once.Do(func() {                // Faqat bir marta bajarish
						firstErr = err          // Birinchi xatoni saqlash
						cancel()                // Barcha workerlarni bekor qilish
					})
				}
			}
		}
	}
	wg.Add(workers)                                                 // Barcha workerlarni wait groupga qo'shish
	for i := 0; i < workers; i++ {                                  // Workerlarni yaratish
		go worker()                                             // Worker goroutinasini ishga tushirish
	}
	wg.Wait()                                                       // Barcha workerlar tugashini kutish
	if firstErr != nil {                                            // Xato yuz berganligini tekshirish
		return firstErr                                         // Birinchi xatoni qaytarish
	}
	if err := ctx.Err(); err != nil {                               // Kontekst holatini tekshirish
		return err                                              // Kontekst xatosini qaytarish
	}
	return nil                                                      // Xatolar yo'q
}`
		}
	}
};

export default task;
