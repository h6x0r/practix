import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-workerpool-make-job-queue',
	title: 'Make Job Queue',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'worker-pool', 'channels', 'producer'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **MakeJobQueue** that converts a slice of jobs into a channel, acting as a producer for worker pools.

**Requirements:**
1. Create function \`MakeJobQueue(ctx context.Context, jobs []Job) <-chan Job\`
2. Return read-only channel of jobs
3. If context is already cancelled, return closed channel immediately
4. Launch goroutine to send jobs to channel
5. Close channel after all jobs sent or context cancelled
6. Handle nil context (send all jobs without cancellation check)
7. Check context cancellation when sending each job

**Pattern:**
This is a producer function that feeds jobs to worker pools like RunPool.

**Example:**
\`\`\`go
jobs := []Job{
    func(ctx context.Context) error {
        fmt.Println("Job 1")
        return nil
    },
    func(ctx context.Context) error {
        fmt.Println("Job 2")
        return nil
    },
    func(ctx context.Context) error {
        fmt.Println("Job 3")
        return nil
    },
}

// Convert slice to channel
jobQueue := MakeJobQueue(ctx, jobs)

// Feed to worker pool
err := RunPool(ctx, jobQueue, 2)

// With cancellation
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
defer cancel()

jobQueue := MakeJobQueue(ctx, manyJobs)
// If timeout occurs, queue stops sending and closes
\`\`\`

**Constraints:**
- Must return read-only channel (<-chan Job)
- Must close channel when done or cancelled
- Must launch goroutine for sending (non-blocking)
- Must handle context cancellation gracefully`,
	initialCode: `package concurrency

import (
	"context"
)

type Job func(context.Context) error

// TODO: Implement MakeJobQueue
func MakeJobQueue(ctx context.Context, jobs []Job) <-chan Job {
	// TODO: Implement
}`,
	testCode: `package concurrency

import (
	"context"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	ch := MakeJobQueue(context.Background(), nil)
	count := 0
	for range ch {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 jobs from nil slice, got %d", count)
	}
}

func Test2(t *testing.T) {
	ch := MakeJobQueue(context.Background(), []Job{})
	count := 0
	for range ch {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 jobs from empty slice, got %d", count)
	}
}

func Test3(t *testing.T) {
	jobs := []Job{
		func(ctx context.Context) error { return nil },
	}
	ch := MakeJobQueue(context.Background(), jobs)
	count := 0
	for range ch {
		count++
	}
	if count != 1 {
		t.Errorf("expected 1 job, got %d", count)
	}
}

func Test4(t *testing.T) {
	jobs := []Job{
		func(ctx context.Context) error { return nil },
		func(ctx context.Context) error { return nil },
		func(ctx context.Context) error { return nil },
		func(ctx context.Context) error { return nil },
		func(ctx context.Context) error { return nil },
	}
	ch := MakeJobQueue(context.Background(), jobs)
	count := 0
	for range ch {
		count++
	}
	if count != 5 {
		t.Errorf("expected 5 jobs, got %d", count)
	}
}

func Test5(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	jobs := []Job{
		func(ctx context.Context) error { return nil },
		func(ctx context.Context) error { return nil },
	}
	ch := MakeJobQueue(ctx, jobs)
	count := 0
	for range ch {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 jobs from pre-cancelled context, got %d", count)
	}
}

func Test6(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	jobs := make([]Job, 100)
	for i := 0; i < 100; i++ {
		jobs[i] = func(ctx context.Context) error { return nil }
	}
	ch := MakeJobQueue(ctx, jobs)
	count := 0
	for range ch {
		time.Sleep(5 * time.Millisecond)
		count++
	}
	if count >= 100 {
		t.Errorf("expected context timeout to stop sending, got all %d", count)
	}
}

func Test7(t *testing.T) {
	jobs := []Job{
		func(ctx context.Context) error { return nil },
	}
	ch := MakeJobQueue(nil, jobs)
	count := 0
	for range ch {
		count++
	}
	if count != 1 {
		t.Errorf("expected 1 job with nil context, got %d", count)
	}
}

func Test8(t *testing.T) {
	jobs := []Job{
		func(ctx context.Context) error { return nil },
		func(ctx context.Context) error { return nil },
		func(ctx context.Context) error { return nil },
	}
	ch := MakeJobQueue(nil, jobs)
	count := 0
	for range ch {
		count++
	}
	if count != 3 {
		t.Errorf("expected 3 jobs with nil context, got %d", count)
	}
}

func Test9(t *testing.T) {
	jobs := []Job{
		func(ctx context.Context) error { return nil },
	}
	ch := MakeJobQueue(context.Background(), jobs)
	select {
	case <-ch:
	case <-time.After(100 * time.Millisecond):
		t.Error("expected immediate job delivery, timed out")
	}
}

func Test10(t *testing.T) {
	ch1 := MakeJobQueue(context.Background(), []Job{func(ctx context.Context) error { return nil }})
	ch2 := MakeJobQueue(context.Background(), []Job{func(ctx context.Context) error { return nil }})
	count1, count2 := 0, 0
	for range ch1 {
		count1++
	}
	for range ch2 {
		count2++
	}
	if count1 != 1 || count2 != 1 {
		t.Errorf("expected independent channels, got %d and %d", count1, count2)
	}
}
`,
	solutionCode: `package concurrency

import (
	"context"
)

type Job func(context.Context) error

func MakeJobQueue(ctx context.Context, jobs []Job) <-chan Job {
	out := make(chan Job)                                           // Create output channel
	if ctx != nil && ctx.Err() != nil {                             // Check if context already cancelled
		close(out)                                              // Return closed channel
		return out                                              // Early return
	}
	go func() {                                                     // Launch producer goroutine
		defer close(out)                                        // Always close channel
		if ctx == nil {                                         // Handle nil context
			for _, job := range jobs {                      // Send all jobs
				out <- job                              // Send without checking context
			}
			return                                          // Done sending
		}
		for _, job := range jobs {                              // Iterate through jobs
			select {                                        // Check context or send
			case <-ctx.Done():                              // Context cancelled
				return                                  // Stop sending and close
			case out <- job:                                // Send job to channel
			}
		}
	}()
	return out                                                      // Return read-only channel
}`,
			hint1: `Create channel, check if context already cancelled (return closed channel), then launch goroutine with defer close(out).`,
			hint2: `In goroutine, use select { case <-ctx.Done(): return; case out <- job: } to send jobs with cancellation support. Handle nil context separately.`,
			whyItMatters: `MakeJobQueue decouples job production from consumption, enabling clean separation of concerns in concurrent pipelines.

**Why Job Queue:**
- **Decoupling:** Separate job creation from execution
- **Backpressure:** Channel naturally limits in-flight jobs
- **Cancellation:** Stop sending jobs when context cancelled
- **Composition:** Easily chain with worker pools

**Production Pattern:**
\`\`\`go
// Database record processing
func ProcessRecords(ctx context.Context, db *DB) error {
    records, err := db.FetchRecords()
    if err != nil {
        return err
    }

    jobs := make([]Job, len(records))
    for i, record := range records {
        r := record
        jobs[i] = func(ctx context.Context) error {
            return processRecord(ctx, r)
        }
    }

    jobQueue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, jobQueue, 10)
}

// Multi-stage pipeline
func ProcessDataPipeline(ctx context.Context, data []Data) error {
    // Stage 1: Validation
    validationJobs := make([]Job, len(data))
    for i, d := range data {
        item := d
        validationJobs[i] = func(ctx context.Context) error {
            return validate(ctx, item)
        }
    }

    queue := MakeJobQueue(ctx, validationJobs)
    if err := RunPool(ctx, queue, 5); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }

    // Stage 2: Processing
    processingJobs := make([]Job, len(data))
    for i, d := range data {
        item := d
        processingJobs[i] = func(ctx context.Context) error {
            return process(ctx, item)
        }
    }

    queue = MakeJobQueue(ctx, processingJobs)
    return RunPool(ctx, queue, 3)
}

// Dynamic job generation
func ProcessUserActions(ctx context.Context, userID string) error {
    actions, err := fetchUserActions(userID)
    if err != nil {
        return err
    }

    jobs := make([]Job, 0, len(actions))
    for _, action := range actions {
        a := action

        // Different job types based on action
        switch a.Type {
        case "email":
            jobs = append(jobs, func(ctx context.Context) error {
                return sendEmail(ctx, a)
            })
        case "notification":
            jobs = append(jobs, func(ctx context.Context) error {
                return sendNotification(ctx, a)
            })
        case "update":
            jobs = append(jobs, func(ctx context.Context) error {
                return updateRecord(ctx, a)
            })
        }
    }

    queue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, queue, 5)
}

// Batch API requests
func FetchMultipleResources(ctx context.Context, urls []string) ([]Response, error) {
    var (
        mu        sync.Mutex
        responses []Response
    )

    jobs := make([]Job, len(urls))
    for i, url := range urls {
        resourceURL := url
        jobs[i] = func(ctx context.Context) error {
            resp, err := fetchResource(ctx, resourceURL)
            if err != nil {
                return err
            }
            mu.Lock()
            responses = append(responses, resp)
            mu.Unlock()
            return nil
        }
    }

    queue := MakeJobQueue(ctx, jobs)
    if err := RunPool(ctx, queue, 10); err != nil {
        return nil, err
    }

    return responses, nil
}

// File processing with timeout
func ProcessFilesWithTimeout(ctx context.Context, files []string, timeout time.Duration) error {
    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    jobs := make([]Job, len(files))
    for i, file := range files {
        filename := file
        jobs[i] = func(ctx context.Context) error {
            return processFile(ctx, filename)
        }
    }

    // If timeout occurs, MakeJobQueue stops sending
    queue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, queue, runtime.NumCPU())
}

// Filtered job processing
func ProcessFilteredItems(ctx context.Context, items []Item) error {
    jobs := make([]Job, 0, len(items))

    for _, item := range items {
        i := item

        // Only create jobs for items that pass filter
        if shouldProcess(i) {
            jobs = append(jobs, func(ctx context.Context) error {
                return processItem(ctx, i)
            })
        }
    }

    if len(jobs) == 0 {
        return nil // Nothing to process
    }

    queue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, queue, 5)
}

// Priority-based processing
func ProcessByPriority(ctx context.Context, items []PriorityItem) error {
    // Sort by priority
    sort.Slice(items, func(i, j int) bool {
        return items[i].Priority > items[j].Priority
    })

    jobs := make([]Job, len(items))
    for i, item := range items {
        itm := item
        jobs[i] = func(ctx context.Context) error {
            return processItem(ctx, itm)
        }
    }

    // High priority items processed first
    queue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, queue, 5)
}

// Retry logic with job queue
func ProcessWithRetry(ctx context.Context, jobs []Job, maxRetries int) error {
    for attempt := 0; attempt <= maxRetries; attempt++ {
        if attempt > 0 {
            log.Printf("Retry attempt %d/%d", attempt, maxRetries)
            time.Sleep(time.Second * time.Duration(attempt))
        }

        queue := MakeJobQueue(ctx, jobs)
        err := RunPool(ctx, queue, 5)

        if err == nil {
            return nil
        }

        if attempt == maxRetries {
            return fmt.Errorf("failed after %d attempts: %w", maxRetries+1, err)
        }
    }

    return nil
}
\`\`\`

**Real-World Benefits:**
- **Clean Separation:** Producer and consumer are independent
- **Cancellation:** Automatically stops on context cancel
- **Composability:** Easy to build pipelines
- **Testability:** Easy to test producers and consumers separately

**Pipeline Pattern:**
\`\`\`
Data Source -> MakeJobQueue -> Worker Pool -> Results
   (slice)        (channel)      (processing)   (output)
\`\`\`

**Why Return Read-Only Channel:**
Returning <-chan Job (not chan Job) prevents callers from:
- Sending to the channel (only producer can send)
- Closing the channel (only producer closes)
- Ensures clean ownership and prevents bugs

**Cancellation Behavior:**
\`\`\`go
// With timeout
ctx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
defer cancel()

jobs := make([]Job, 1000) // Many jobs
queue := MakeJobQueue(ctx, jobs)

// If timeout occurs while sending:
// - Producer stops sending remaining jobs
// - Channel closes
// - Worker pool processes what's already sent
// - Clean shutdown
\`\`\`

**When to Use:**
- Converting slices to channels for worker pools
- Need cancellable job production
- Building concurrent pipelines
- Separating job creation from execution

**Pattern Combination:**
\`\`\`go
// Common pattern: Slice -> Queue -> Pool
jobs := generateJobs(data)
queue := MakeJobQueue(ctx, jobs)
err := RunPool(ctx, queue, workers)
\`\`\`

This is a fundamental pattern for building concurrent job processing systems.`,	order: 6,
	translations: {
		ru: {
			title: 'Очередь задач',
			description: `Реализуйте **MakeJobQueue**, который преобразует слайс задач в канал, действуя как продюсер для пулов воркеров.

**Требования:**
1. Создайте функцию \`MakeJobQueue(ctx context.Context, jobs []Job) <-chan Job\`
2. Верните канал только для чтения
3. Если контекст уже отменён, немедленно верните закрытый канал
4. Запустите горутину для отправки задач в канал
5. Закройте канал после отправки всех задач или отмены контекста
6. Обработайте nil context (отправьте все задачи без проверки отмены)
7. Проверяйте отмену контекста при отправке каждой задачи

**Паттерн:**
Это функция-продюсер которая подаёт задачи в пулы воркеров типа RunPool.

**Пример:**
\`\`\`go
jobs := []Job{
    func(ctx context.Context) error {
        fmt.Println("Job 1")
        return nil
    },
    func(ctx context.Context) error {
        fmt.Println("Job 2")
        return nil
    },
    func(ctx context.Context) error {
        fmt.Println("Job 3")
        return nil
    },
}

// Преобразовать слайс в канал
jobQueue := MakeJobQueue(ctx, jobs)

// Подать в пул воркеров
err := RunPool(ctx, jobQueue, 2)

// С отменой
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
defer cancel()

jobQueue := MakeJobQueue(ctx, manyJobs)
// Если происходит таймаут, очередь прекращает отправку и закрывается
`,
			hint1: `Создайте канал, проверьте не отменён ли контекст уже (верните закрытый канал), затем запустите горутину с defer close(out).`,
			hint2: `В горутине используйте select { case <-ctx.Done(): return; case out <- job: } для отправки задач с поддержкой отмены. Обработайте nil context отдельно.`,
			whyItMatters: `MakeJobQueue разделяет производство и потребление задач, обеспечивая чистое разделение обязанностей в конкурентных конвейерах с поддержкой отмены и композиции.

**Почему очередь задач:**
- **Разделение обязанностей:** Создание задач отделено от выполнения
- **Обратное давление:** Канал естественно ограничивает задачи в полёте
- **Поддержка отмены:** Автоматическая остановка при отмене контекста
- **Композиция конвейеров:** Легко объединять с другими компонентами
- **Безопасность типов:** Канал только для чтения предотвращает ошибки

**Продакшен паттерн:**
\`\`\`go
// Обработка записей из базы данных
func ProcessRecords(ctx context.Context, db *DB) error {
    records, err := db.FetchRecords()
    if err != nil {
        return err
    }

    jobs := make([]Job, len(records))
    for i, record := range records {
        r := record  // Захват переменной для замыкания
        jobs[i] = func(ctx context.Context) error {
            return processRecord(ctx, r)
        }
    }

    jobQueue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, jobQueue, 10)
}

// Многоступенчатый конвейер обработки
func ProcessDataPipeline(ctx context.Context, data []Data) error {
    // Этап 1: Валидация
    validationJobs := make([]Job, len(data))
    for i, d := range data {
        item := d
        validationJobs[i] = func(ctx context.Context) error {
            return validate(ctx, item)
        }
    }

    queue := MakeJobQueue(ctx, validationJobs)
    if err := RunPool(ctx, queue, 5); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }

    // Этап 2: Обработка
    processingJobs := make([]Job, len(data))
    for i, d := range data {
        item := d
        processingJobs[i] = func(ctx context.Context) error {
            return process(ctx, item)
        }
    }

    queue = MakeJobQueue(ctx, processingJobs)
    return RunPool(ctx, queue, 3)
}

// Динамическое создание задач по типам
func ProcessUserActions(ctx context.Context, userID string) error {
    actions, err := fetchUserActions(userID)
    if err != nil {
        return err
    }

    jobs := make([]Job, 0, len(actions))
    for _, action := range actions {
        a := action

        // Различные типы задач в зависимости от действия
        switch a.Type {
        case "email":
            jobs = append(jobs, func(ctx context.Context) error {
                return sendEmail(ctx, a)
            })
        case "notification":
            jobs = append(jobs, func(ctx context.Context) error {
                return sendNotification(ctx, a)
            })
        case "update":
            jobs = append(jobs, func(ctx context.Context) error {
                return updateRecord(ctx, a)
            })
        }
    }

    queue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, queue, 5)
}

// Пакетные запросы к API
func FetchMultipleResources(ctx context.Context, urls []string) ([]Response, error) {
    var (
        mu        sync.Mutex
        responses []Response
    )

    jobs := make([]Job, len(urls))
    for i, url := range urls {
        resourceURL := url
        jobs[i] = func(ctx context.Context) error {
            resp, err := fetchResource(ctx, resourceURL)
            if err != nil {
                return err
            }
            mu.Lock()
            responses = append(responses, resp)
            mu.Unlock()
            return nil
        }
    }

    queue := MakeJobQueue(ctx, jobs)
    if err := RunPool(ctx, queue, 10); err != nil {
        return nil, err
    }

    return responses, nil
}

// Обработка файлов с тайм-аутом
func ProcessFilesWithTimeout(ctx context.Context, files []string, timeout time.Duration) error {
    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    jobs := make([]Job, len(files))
    for i, file := range files {
        filename := file
        jobs[i] = func(ctx context.Context) error {
            return processFile(ctx, filename)
        }
    }

    // Если наступает тайм-аут, MakeJobQueue прекращает отправку
    queue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, queue, runtime.NumCPU())
}

// Фильтрованная обработка задач
func ProcessFilteredItems(ctx context.Context, items []Item) error {
    jobs := make([]Job, 0, len(items))

    for _, item := range items {
        i := item

        // Создавать задачи только для элементов прошедших фильтр
        if shouldProcess(i) {
            jobs = append(jobs, func(ctx context.Context) error {
                return processItem(ctx, i)
            })
        }
    }

    if len(jobs) == 0 {
        return nil  // Нечего обрабатывать
    }

    queue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, queue, 5)
}

// Обработка с приоритетами
func ProcessByPriority(ctx context.Context, items []PriorityItem) error {
    // Сортировка по приоритету
    sort.Slice(items, func(i, j int) bool {
        return items[i].Priority > items[j].Priority
    })

    jobs := make([]Job, len(items))
    for i, item := range items {
        itm := item
        jobs[i] = func(ctx context.Context) error {
            return processItem(ctx, itm)
        }
    }

    // Элементы с высоким приоритетом обрабатываются первыми
    queue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, queue, 5)
}

// Логика повторов с очередью задач
func ProcessWithRetry(ctx context.Context, jobs []Job, maxRetries int) error {
    for attempt := 0; attempt <= maxRetries; attempt++ {
        if attempt > 0 {
            log.Printf("Попытка повтора %d/%d", attempt, maxRetries)
            time.Sleep(time.Second * time.Duration(attempt))
        }

        queue := MakeJobQueue(ctx, jobs)
        err := RunPool(ctx, queue, 5)

        if err == nil {
            return nil
        }

        if attempt == maxRetries {
            return fmt.Errorf("failed after %d attempts: %w", maxRetries+1, err)
        }
    }

    return nil
}
\`\`\`

**Практические преимущества:**
- **Чистое разделение:** Продюсер и консьюмер независимы
- **Автоматическая отмена:** Останавливается при отмене контекста
- **Композируемость:** Легко строить конвейеры обработки
- **Тестируемость:** Легко тестировать продюсеры и консьюмеры отдельно
- **Безопасность:** Read-only канал предотвращает ошибки

**Паттерн конвейера:**
\`\`\`
Источник данных -> MakeJobQueue -> Пул воркеров -> Результаты
   (слайс)           (канал)        (обработка)     (вывод)
\`\`\`

**Почему возвращать канал только для чтения:**
Возврат <-chan Job (а не chan Job) предотвращает вызывающий код от:
- Отправки в канал (только продюсер может отправлять)
- Закрытия канала (только продюсер закрывает)
- Обеспечивает чистое владение и предотвращает баги

**Поведение при отмене:**
\`\`\`go
// С тайм-аутом
ctx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
defer cancel()

jobs := make([]Job, 1000)  // Много задач
queue := MakeJobQueue(ctx, jobs)

// Если наступает тайм-аут во время отправки:
// - Продюсер прекращает отправку оставшихся задач
// - Канал закрывается
// - Пул воркеров обрабатывает то что уже отправлено
// - Чистое завершение
\`\`\`

**Когда использовать:**
- Преобразование слайсов в каналы для пулов воркеров
- Нужна отменяемая генерация задач
- Построение конкурентных конвейеров
- Разделение создания задач от выполнения

**Комбинация паттернов:**
\`\`\`go
// Общий паттерн: Слайс -> Очередь -> Пул
jobs := generateJobs(data)
queue := MakeJobQueue(ctx, jobs)
err := RunPool(ctx, queue, workers)
\`\`\`

Это фундаментальный паттерн для построения систем конкурентной обработки задач.`,
			solutionCode: `package concurrency

import (
	"context"
)

type Job func(context.Context) error

func MakeJobQueue(ctx context.Context, jobs []Job) <-chan Job {
	out := make(chan Job)                                           // Создание выходного канала
	if ctx != nil && ctx.Err() != nil {                             // Проверка не отменён ли контекст уже
		close(out)                                              // Возврат закрытого канала
		return out                                              // Ранний возврат
	}
	go func() {                                                     // Запуск горутины-продюсера
		defer close(out)                                        // Всегда закрывать канал
		if ctx == nil {                                         // Обработка nil контекста
			for _, job := range jobs {                      // Отправка всех задач
				out <- job                              // Отправка без проверки контекста
			}
			return                                          // Завершение отправки
		}
		for _, job := range jobs {                              // Итерация по задачам
			select {                                        // Проверка контекста или отправка
			case <-ctx.Done():                              // Контекст отменён
				return                                  // Остановка отправки и закрытие
			case out <- job:                                // Отправка задачи в канал
			}
		}
	}()
	return out                                                      // Возврат канала только для чтения
}`
		},
		uz: {
			title: 'Vazifalar navbati',
			description: `Vazifalar sliceini kanalga aylantiradigan, worker pullari uchun ishlab chiqaruvchi bo'lib xizmat qiladigan **MakeJobQueue** ni amalga oshiring.

**Talablar:**
1. \`MakeJobQueue(ctx context.Context, jobs []Job) <-chan Job\` funksiyasini yarating
2. Faqat o'qish uchun kanalni qaytaring
3. Agar kontekst allaqachon bekor qilingan bo'lsa, darhol yopiq kanalni qaytaring
4. Kanalga vazifalarni yuborish uchun goroutinani ishga tushiring
5. Barcha vazifalar yuborilgandan yoki kontekst bekor qilingandan keyin kanalni yoping
6. nil kontekstni ishlang (bekor qilishni tekshirmasdan barcha vazifalarni yuboring)
7. Har bir vazifani yuborishda kontekst bekor qilinishini tekshiring

**Pattern:**
Bu RunPool kabi worker pullariga vazifalarni taqdim etadigan ishlab chiqaruvchi funksiya.

**Misol:**
\`\`\`go
jobs := []Job{
    func(ctx context.Context) error {
        fmt.Println("Job 1")
        return nil
    },
    func(ctx context.Context) error {
        fmt.Println("Job 2")
        return nil
    },
    func(ctx context.Context) error {
        fmt.Println("Job 3")
        return nil
    },
}

// Sliceni kanalga aylantirish
jobQueue := MakeJobQueue(ctx, jobs)

// Worker puliga berish
err := RunPool(ctx, jobQueue, 2)

// Bekor qilish bilan
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
defer cancel()

jobQueue := MakeJobQueue(ctx, manyJobs)
// Agar timeout yuz bersa, navbat yuborishni to'xtatadi va yopiladi
`,
			hint1: `Kanal yarating, kontekst allaqachon bekor qilinganligini tekshiring (yopiq kanalni qaytaring), keyin defer close(out) bilan goroutinani ishga tushiring.`,
			hint2: `Goroutinada bekor qilish yordamida vazifalarni yuborish uchun select { case <-ctx.Done(): return; case out <- job: } dan foydalaning. nil kontekstni alohida ishlang.`,
			whyItMatters: `MakeJobQueue vazifalarni ishlab chiqarish va iste'mol qilishni ajratadi, bekor qilish va kompozitsiya qo'llab-quvvatlanishida parallel quvurlarda mas'uliyatlarning toza ajratilishini ta'minlaydi.

**Nima uchun vazifalar navbati:**
- **Mas'uliyatlarni ajratish:** Vazifalarni yaratish bajarishdan ajratilgan
- **Orqa bosim:** Kanal parvozda vazifalarni tabiiy ravishda cheklaydi
- **Bekor qilish qo'llab-quvvatlash:** Kontekst bekor qilinganda avtomatik to'xtatish
- **Quvur kompozitsiyasi:** Boshqa komponentlar bilan osongina birlashtirish
- **Tur xavfsizligi:** Faqat o'qish uchun kanal xatolarning oldini oladi

**Ishlab chiqarish patterni:**
\`\`\`go
// Ma'lumotlar bazasidan yozuvlarni qayta ishlash
func ProcessRecords(ctx context.Context, db *DB) error {
    records, err := db.FetchRecords()
    if err != nil {
        return err
    }

    jobs := make([]Job, len(records))
    for i, record := range records {
        r := record  // Yopilish uchun o'zgaruvchini ushlash
        jobs[i] = func(ctx context.Context) error {
            return processRecord(ctx, r)
        }
    }

    jobQueue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, jobQueue, 10)
}

// Ko'p bosqichli qayta ishlash quvuri
func ProcessDataPipeline(ctx context.Context, data []Data) error {
    // Bosqich 1: Tekshirish
    validationJobs := make([]Job, len(data))
    for i, d := range data {
        item := d
        validationJobs[i] = func(ctx context.Context) error {
            return validate(ctx, item)
        }
    }

    queue := MakeJobQueue(ctx, validationJobs)
    if err := RunPool(ctx, queue, 5); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }

    // Bosqich 2: Qayta ishlash
    processingJobs := make([]Job, len(data))
    for i, d := range data {
        item := d
        processingJobs[i] = func(ctx context.Context) error {
            return process(ctx, item)
        }
    }

    queue = MakeJobQueue(ctx, processingJobs)
    return RunPool(ctx, queue, 3)
}

// Turlarga ko'ra dinamik vazifa yaratish
func ProcessUserActions(ctx context.Context, userID string) error {
    actions, err := fetchUserActions(userID)
    if err != nil {
        return err
    }

    jobs := make([]Job, 0, len(actions))
    for _, action := range actions {
        a := action

        // Harakatga qarab turli vazifa turlari
        switch a.Type {
        case "email":
            jobs = append(jobs, func(ctx context.Context) error {
                return sendEmail(ctx, a)
            })
        case "notification":
            jobs = append(jobs, func(ctx context.Context) error {
                return sendNotification(ctx, a)
            })
        case "update":
            jobs = append(jobs, func(ctx context.Context) error {
                return updateRecord(ctx, a)
            })
        }
    }

    queue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, queue, 5)
}

// APIga paket so'rovlar
func FetchMultipleResources(ctx context.Context, urls []string) ([]Response, error) {
    var (
        mu        sync.Mutex
        responses []Response
    )

    jobs := make([]Job, len(urls))
    for i, url := range urls {
        resourceURL := url
        jobs[i] = func(ctx context.Context) error {
            resp, err := fetchResource(ctx, resourceURL)
            if err != nil {
                return err
            }
            mu.Lock()
            responses = append(responses, resp)
            mu.Unlock()
            return nil
        }
    }

    queue := MakeJobQueue(ctx, jobs)
    if err := RunPool(ctx, queue, 10); err != nil {
        return nil, err
    }

    return responses, nil
}

// Timeout bilan fayllarni qayta ishlash
func ProcessFilesWithTimeout(ctx context.Context, files []string, timeout time.Duration) error {
    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    jobs := make([]Job, len(files))
    for i, file := range files {
        filename := file
        jobs[i] = func(ctx context.Context) error {
            return processFile(ctx, filename)
        }
    }

    // Agar timeout yuz bersa, MakeJobQueue yuborishni to'xtatadi
    queue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, queue, runtime.NumCPU())
}

// Filtrlangan vazifalarni qayta ishlash
func ProcessFilteredItems(ctx context.Context, items []Item) error {
    jobs := make([]Job, 0, len(items))

    for _, item := range items {
        i := item

        // Faqat filtrdan o'tgan elementlar uchun vazifalar yaratish
        if shouldProcess(i) {
            jobs = append(jobs, func(ctx context.Context) error {
                return processItem(ctx, i)
            })
        }
    }

    if len(jobs) == 0 {
        return nil  // Qayta ishlash uchun hech narsa yo'q
    }

    queue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, queue, 5)
}

// Ustuvorlik asosida qayta ishlash
func ProcessByPriority(ctx context.Context, items []PriorityItem) error {
    // Ustuvorlik bo'yicha saralash
    sort.Slice(items, func(i, j int) bool {
        return items[i].Priority > items[j].Priority
    })

    jobs := make([]Job, len(items))
    for i, item := range items {
        itm := item
        jobs[i] = func(ctx context.Context) error {
            return processItem(ctx, itm)
        }
    }

    // Yuqori ustuvorlikdagi elementlar birinchi qayta ishlanadi
    queue := MakeJobQueue(ctx, jobs)
    return RunPool(ctx, queue, 5)
}

// Vazifalar navbati bilan qayta urinish mantiqasi
func ProcessWithRetry(ctx context.Context, jobs []Job, maxRetries int) error {
    for attempt := 0; attempt <= maxRetries; attempt++ {
        if attempt > 0 {
            log.Printf("Qayta urinish %d/%d", attempt, maxRetries)
            time.Sleep(time.Second * time.Duration(attempt))
        }

        queue := MakeJobQueue(ctx, jobs)
        err := RunPool(ctx, queue, 5)

        if err == nil {
            return nil
        }

        if attempt == maxRetries {
            return fmt.Errorf("failed after %d attempts: %w", maxRetries+1, err)
        }
    }

    return nil
}
\`\`\`

**Amaliy afzalliklar:**
- **Toza ajratish:** Ishlab chiqaruvchi va iste'molchi mustaqil
- **Avtomatik bekor qilish:** Kontekst bekor qilinganda to'xtatiladi
- **Kompozitsiyalanish:** Qayta ishlash quvurlarini osongina qurish
- **Testlanish:** Ishlab chiqaruvchilar va iste'molchilarni alohida test qilish oson
- **Xavfsizlik:** Faqat o'qish uchun kanal xatolarning oldini oladi

**Quvur patterni:**
\`\`\`
Ma'lumot manbai -> MakeJobQueue -> Worker puli -> Natijalar
   (slice)           (kanal)       (qayta ishlash)  (chiqish)
\`\`\`

**Nima uchun faqat o'qish uchun kanal qaytarish:**
<-chan Job (chan Job emas) qaytarish chaqiruvchi kodga yo'l qo'ymaydi:
- Kanalga yuborish (faqat ishlab chiqaruvchi yuborishi mumkin)
- Kanalni yopish (faqat ishlab chiqaruvchi yopadi)
- Toza egalikni ta'minlaydi va xatolarning oldini oladi

**Bekor qilish harakati:**
\`\`\`go
// Timeout bilan
ctx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
defer cancel()

jobs := make([]Job, 1000)  // Ko'p vazifalar
queue := MakeJobQueue(ctx, jobs)

// Agar yuborish paytida timeout yuz bersa:
// - Ishlab chiqaruvchi qolgan vazifalarni yuborishni to'xtatadi
// - Kanal yopiladi
// - Worker puli allaqachon yuborilgan narsalarni qayta ishlaydi
// - Toza to'xtatish
\`\`\`

**Qachon ishlatish:**
- Slicelarni worker pullari uchun kanallarga aylantirish
- Bekor qilinadigan vazifa ishlab chiqarish kerak
- Parallel quvurlarni qurish
- Vazifalarni yaratishni bajarishdan ajratish

**Patternlar kombinatsiyasi:**
\`\`\`go
// Umumiy pattern: Slice -> Navbat -> Pul
jobs := generateJobs(data)
queue := MakeJobQueue(ctx, jobs)
err := RunPool(ctx, queue, workers)
\`\`\`

Bu parallel vazifalarni qayta ishlash tizimlarini qurish uchun fundamental pattern.`,
			solutionCode: `package concurrency

import (
	"context"
)

type Job func(context.Context) error

func MakeJobQueue(ctx context.Context, jobs []Job) <-chan Job {
	out := make(chan Job)                                           // Chiqish kanalini yaratish
	if ctx != nil && ctx.Err() != nil {                             // Kontekst allaqachon bekor qilinganligini tekshirish
		close(out)                                              // Yopiq kanalni qaytarish
		return out                                              // Erta qaytish
	}
	go func() {                                                     // Ishlab chiqaruvchi goroutinasini ishga tushirish
		defer close(out)                                        // Kanalni doim yopish
		if ctx == nil {                                         // nil kontekstni ishlash
			for _, job := range jobs {                      // Barcha vazifalarni yuborish
				out <- job                              // Kontekstni tekshirmasdan yuborish
			}
			return                                          // Yuborish tugadi
		}
		for _, job := range jobs {                              // Vazifalar bo'yicha iteratsiya
			select {                                        // Kontekstni tekshirish yoki yuborish
			case <-ctx.Done():                              // Kontekst bekor qilindi
				return                                  // Yuborishni to'xtatish va yopish
			case out <- job:                                // Vazifani kanalga yuborish
			}
		}
	}()
	return out                                                      // Faqat o'qish uchun kanalni qaytarish
}`
		}
	}
};

export default task;
