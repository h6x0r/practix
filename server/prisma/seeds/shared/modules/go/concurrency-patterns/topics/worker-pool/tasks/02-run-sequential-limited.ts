import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-workerpool-run-sequential-limited',
	title: 'Run Sequential Limited',
	difficulty: 'easy',	tags: ['go', 'concurrency', 'worker-pool', 'sequential', 'limit'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **RunSequentialLimited** that executes jobs sequentially but stops after processing a limited number of jobs.

**Requirements:**
1. Create function \`RunSequentialLimited(ctx context.Context, jobs []Job, limit int) (int, error)\`
2. Handle nil context (return 0, nil)
3. Handle limit <= 0 (return 0, nil)
4. Execute up to \`limit\` jobs sequentially
5. Count nil jobs as processed (increment counter)
6. Stop after reaching limit
7. Return count of processed jobs and any error
8. Check context cancellation before each job

**Example:**
\`\`\`go
jobs := []Job{
    job1, job2, job3, job4, job5,
}

// Process only first 3 jobs
count, err := RunSequentialLimited(ctx, jobs, 3)
// count = 3, only jobs 1-3 executed

// Process with nil job
jobs := []Job{job1, nil, job3, job4}
count, err := RunSequentialLimited(ctx, jobs, 3)
// count = 3, nil counted as processed

// Error in second job
jobs := []Job{
    func(ctx context.Context) error { return nil },
    func(ctx context.Context) error { return errors.New("failed") },
    func(ctx context.Context) error { return nil },
}
count, err := RunSequentialLimited(ctx, jobs, 5)
// count = 2, err = "failed"
\`\`\`

**Constraints:**
- Must execute jobs sequentially
- Must stop after limit jobs processed
- Must count nil jobs as processed
- Must stop on first error (after incrementing count)`,
	initialCode: `package concurrency

import (
	"context"
)

type Job func(context.Context) error

// TODO: Implement RunSequentialLimited
func RunSequentialLimited(ctx context.Context, jobs []Job, limit int) (int, error) {
	var zero int
	return zero, nil // TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
)

type Job func(context.Context) error

func RunSequentialLimited(ctx context.Context, jobs []Job, limit int) (int, error) {
	if ctx == nil {                                                 // Handle nil context
		return 0, nil                                           // Return zero count
	}
	if limit <= 0 {                                                 // Handle invalid limit
		return 0, nil                                           // Return zero count
	}
	count := 0                                                      // Track processed jobs
	for _, job := range jobs {                                      // Iterate through jobs
		if count >= limit {                                     // Check if limit reached
			break                                           // Stop processing
		}
		if err := ctx.Err(); err != nil {                       // Check context cancellation
			return count, err                               // Return count and error
		}
		if job == nil {                                         // Handle nil job
			count++                                         // Count as processed
			continue                                        // Skip to next
		}
		if err := job(ctx); err != nil {                        // Execute job
			count++                                         // Count even on error
			return count, err                               // Return count and error
		}
		count++                                                 // Increment counter
	}
	return count, ctx.Err()                                         // Return final count and context state
}`,
	testCode: `package concurrency

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestRunSequentialLimited1(t *testing.T) {
	// Test limiting to 3 jobs out of 5
	ctx := context.Background()
	executed := []int{}
	jobs := []Job{
		func(ctx context.Context) error {
			executed = append(executed, 1)
			return nil
		},
		func(ctx context.Context) error {
			executed = append(executed, 2)
			return nil
		},
		func(ctx context.Context) error {
			executed = append(executed, 3)
			return nil
		},
		func(ctx context.Context) error {
			executed = append(executed, 4)
			return nil
		},
		func(ctx context.Context) error {
			executed = append(executed, 5)
			return nil
		},
	}

	count, err := RunSequentialLimited(ctx, jobs, 3)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if count != 3 {
		t.Errorf("expected count 3, got %d", count)
	}
	if len(executed) != 3 {
		t.Errorf("expected 3 jobs executed, got %d", len(executed))
	}
}

func TestRunSequentialLimited2(t *testing.T) {
	// Test error stops execution and increments count
	ctx := context.Background()
	executed := []int{}
	expectedErr := errors.New("job failed")

	jobs := []Job{
		func(ctx context.Context) error {
			executed = append(executed, 1)
			return nil
		},
		func(ctx context.Context) error {
			executed = append(executed, 2)
			return expectedErr
		},
		func(ctx context.Context) error {
			executed = append(executed, 3)
			return nil
		},
	}

	count, err := RunSequentialLimited(ctx, jobs, 5)
	if err != expectedErr {
		t.Errorf("expected error %v, got %v", expectedErr, err)
	}
	if count != 2 {
		t.Errorf("expected count 2, got %d", count)
	}
	if len(executed) != 2 {
		t.Errorf("expected 2 jobs executed, got %d", len(executed))
	}
}

func TestRunSequentialLimited3(t *testing.T) {
	// Test limit of 0 executes nothing
	ctx := context.Background()
	executed := false

	jobs := []Job{
		func(ctx context.Context) error {
			executed = true
			return nil
		},
	}

	count, err := RunSequentialLimited(ctx, jobs, 0)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if count != 0 {
		t.Errorf("expected count 0, got %d", count)
	}
	if executed {
		t.Errorf("expected no jobs executed")
	}
}

func TestRunSequentialLimited4(t *testing.T) {
	// Test negative limit executes nothing
	ctx := context.Background()
	executed := false

	jobs := []Job{
		func(ctx context.Context) error {
			executed = true
			return nil
		},
	}

	count, err := RunSequentialLimited(ctx, jobs, -5)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if count != 0 {
		t.Errorf("expected count 0, got %d", count)
	}
	if executed {
		t.Errorf("expected no jobs executed")
	}
}

func TestRunSequentialLimited5(t *testing.T) {
	// Test nil jobs are counted
	ctx := context.Background()
	executed := []int{}

	jobs := []Job{
		func(ctx context.Context) error {
			executed = append(executed, 1)
			return nil
		},
		nil,
		func(ctx context.Context) error {
			executed = append(executed, 3)
			return nil
		},
	}

	count, err := RunSequentialLimited(ctx, jobs, 5)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if count != 3 {
		t.Errorf("expected count 3, got %d", count)
	}
	if len(executed) != 2 {
		t.Errorf("expected 2 jobs executed (nil skipped), got %d", len(executed))
	}
}

func TestRunSequentialLimited6(t *testing.T) {
	// Test limit larger than jobs count
	ctx := context.Background()
	executed := []int{}

	jobs := []Job{
		func(ctx context.Context) error {
			executed = append(executed, 1)
			return nil
		},
		func(ctx context.Context) error {
			executed = append(executed, 2)
			return nil
		},
	}

	count, err := RunSequentialLimited(ctx, jobs, 100)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if count != 2 {
		t.Errorf("expected count 2, got %d", count)
	}
	if len(executed) != 2 {
		t.Errorf("expected 2 jobs executed, got %d", len(executed))
	}
}

func TestRunSequentialLimited7(t *testing.T) {
	// Test with nil context
	jobs := []Job{
		func(ctx context.Context) error {
			return nil
		},
	}

	count, err := RunSequentialLimited(nil, jobs, 5)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if count != 0 {
		t.Errorf("expected count 0 with nil context, got %d", count)
	}
}

func TestRunSequentialLimited8(t *testing.T) {
	// Test context cancellation
	ctx, cancel := context.WithCancel(context.Background())
	executed := []int{}

	jobs := []Job{
		func(ctx context.Context) error {
			executed = append(executed, 1)
			return nil
		},
		func(ctx context.Context) error {
			cancel() // Cancel after second job
			executed = append(executed, 2)
			return nil
		},
		func(ctx context.Context) error {
			executed = append(executed, 3)
			return nil
		},
	}

	count, err := RunSequentialLimited(ctx, jobs, 5)
	if err == nil {
		t.Errorf("expected context cancelled error")
	}
	if count != 2 {
		t.Errorf("expected count 2, got %d", count)
	}
}

func TestRunSequentialLimited9(t *testing.T) {
	// Test empty job slice
	ctx := context.Background()
	jobs := []Job{}

	count, err := RunSequentialLimited(ctx, jobs, 5)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if count != 0 {
		t.Errorf("expected count 0, got %d", count)
	}
}

func TestRunSequentialLimited10(t *testing.T) {
	// Test limit exactly matches job count
	ctx := context.Background()
	executed := []int{}

	jobs := []Job{
		func(ctx context.Context) error {
			executed = append(executed, 1)
			return nil
		},
		func(ctx context.Context) error {
			executed = append(executed, 2)
			return nil
		},
		func(ctx context.Context) error {
			executed = append(executed, 3)
			return nil
		},
	}

	count, err := RunSequentialLimited(ctx, jobs, 3)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if count != 3 {
		t.Errorf("expected count 3, got %d", count)
	}
	if len(executed) != 3 {
		t.Errorf("expected 3 jobs executed, got %d", len(executed))
	}
}`,
			hint1: `Use a counter variable to track processed jobs. Check if count >= limit before processing each job.`,
			hint2: `Increment count for both successful jobs and nil jobs. Increment count even when a job returns an error before returning.`,
			whyItMatters: `RunSequentialLimited is useful when you want to process only a subset of jobs, useful for rate limiting, testing, or gradual rollouts.

**Why Limit Job Execution:**
- **Rate Limiting:** Process fixed number per interval
- **Testing:** Validate with small batches first
- **Resource Control:** Prevent overwhelming systems
- **Gradual Rollout:** Deploy changes incrementally

**Production Pattern:**
\`\`\`go
// Batch processing with limits
func ProcessUserBatch(ctx context.Context, users []User) error {
    jobs := make([]Job, len(users))
    for i, user := range users {
        u := user
        jobs[i] = func(ctx context.Context) error {
            return processUser(ctx, u)
        }
    }

    // Process 100 users at a time
    batchSize := 100
    processed := 0

    for processed < len(jobs) {
        count, err := RunSequentialLimited(ctx, jobs[processed:], batchSize)
        processed += count
        if err != nil {
            return fmt.Errorf("processed %d/%d: %w", processed, len(jobs), err)
        }
        time.Sleep(time.Second) // Rate limiting between batches
    }

    return nil
}

// Gradual feature rollout
func RolloutFeature(ctx context.Context, targets []string, percentage int) error {
    limit := (len(targets) * percentage) / 100

    jobs := make([]Job, len(targets))
    for i, target := range targets {
        t := target
        jobs[i] = func(ctx context.Context) error {
            return enableFeature(ctx, t)
        }
    }

    count, err := RunSequentialLimited(ctx, jobs, limit)
    log.Printf("Enabled feature for %d/%d targets", count, len(targets))
    return err
}

// Database migration with checkpoints
func MigrateRecords(ctx context.Context, recordIDs []string) error {
    jobs := make([]Job, len(recordIDs))
    for i, id := range recordIDs {
        recordID := id
        jobs[i] = func(ctx context.Context) error {
            return migrateRecord(ctx, recordID)
        }
    }

    checkpointSize := 1000
    totalProcessed := 0

    for totalProcessed < len(jobs) {
        count, err := RunSequentialLimited(ctx, jobs[totalProcessed:], checkpointSize)
        totalProcessed += count

        // Save checkpoint
        saveCheckpoint(totalProcessed)

        if err != nil {
            return fmt.Errorf("migration failed at record %d: %w", totalProcessed, err)
        }
    }

    return nil
}

// API rate limiting
func ProcessAPIRequests(ctx context.Context, requests []Request) error {
    jobs := make([]Job, len(requests))
    for i, req := range requests {
        r := req
        jobs[i] = func(ctx context.Context) error {
            return makeAPICall(ctx, r)
        }
    }

    // API limit: 10 requests per second
    ticker := time.NewTicker(time.Second)
    defer ticker.Stop()

    processed := 0
    for processed < len(jobs) {
        <-ticker.C // Wait for next second

        count, err := RunSequentialLimited(ctx, jobs[processed:], 10)
        processed += count

        if err != nil {
            log.Printf("API error after %d requests: %v", processed, err)
            return err
        }
    }

    return nil
}

// Testing with progressive sample sizes
func TestDataProcessing(ctx context.Context, testData []TestCase) error {
    jobs := make([]Job, len(testData))
    for i, tc := range testData {
        testCase := tc
        jobs[i] = func(ctx context.Context) error {
            return runTest(ctx, testCase)
        }
    }

    // Test with increasing sample sizes
    sampleSizes := []int{10, 50, 100, 500, len(jobs)}

    for _, size := range sampleSizes {
        log.Printf("Testing with %d samples...", size)
        count, err := RunSequentialLimited(ctx, jobs, size)

        if err != nil {
            return fmt.Errorf("test failed at sample size %d (processed %d): %w", size, count, err)
        }

        log.Printf("Passed %d tests", count)
    }

    return nil
}
\`\`\`

**Real-World Benefits:**
- **Controlled Processing:** Prevent overwhelming downstream systems
- **Checkpointing:** Save progress at regular intervals
- **Testing:** Validate with small samples before full run
- **Rate Limiting:** Respect API or resource limits

**When to Use:**
- Batch processing with size limits
- Gradual rollouts or canary deployments
- Rate-limited API calls
- Testing with progressive samples
- Database migrations with checkpoints

**Return Value Design:**
The function returns both count and error, allowing callers to know exactly how many jobs were processed before an error occurred. This is crucial for checkpoint/resume functionality.`,	order: 1,
	translations: {
		ru: {
			title: 'Последовательное выполнение с лимитом',
			description: `Реализуйте **RunSequentialLimited**, который выполняет задачи последовательно, но останавливается после обработки ограниченного количества задач.

**Требования:**
1. Создайте функцию \`RunSequentialLimited(ctx context.Context, jobs []Job, limit int) (int, error)\`
2. Обработайте nil context (верните 0, nil)
3. Обработайте limit <= 0 (верните 0, nil)
4. Выполните до \`limit\` задач последовательно
5. Считайте nil задачи как обработанные (увеличьте счётчик)
6. Остановитесь после достижения лимита
7. Верните количество обработанных задач и любую ошибку
8. Проверяйте отмену контекста перед каждой задачей

**Пример:**
\`\`\`go
jobs := []Job{
    job1, job2, job3, job4, job5,
}

// Обработать только первые 3 задачи
count, err := RunSequentialLimited(ctx, jobs, 3)
// count = 3, выполнены только задачи 1-3

// Обработка с nil задачей
jobs := []Job{job1, nil, job3, job4}
count, err := RunSequentialLimited(ctx, jobs, 3)
// count = 3, nil считается обработанной

// Ошибка во второй задаче
jobs := []Job{
    func(ctx context.Context) error { return nil },
    func(ctx context.Context) error { return errors.New("failed") },
    func(ctx context.Context) error { return nil },
}
count, err := RunSequentialLimited(ctx, jobs, 5)
// count = 2, err = "failed"
`,
			hint1: `Используйте переменную-счётчик для отслеживания обработанных задач. Проверяйте count >= limit перед обработкой каждой задачи.`,
			hint2: `Увеличивайте count для успешных задач и nil задач. Увеличивайте count даже когда задача возвращает ошибку перед возвратом.`,
			whyItMatters: `RunSequentialLimited полезен когда нужно обработать только подмножество задач, полезно для ограничения скорости, тестирования или постепенного развёртывания.

**Почему ограничивать выполнение:**
- **Ограничение скорости:** Обработка фиксированного количества за интервал
- **Тестирование:** Проверка с малыми партиями сначала
- **Контроль ресурсов:** Предотвращение перегрузки систем
- **Постепенное развёртывание:** Инкрементальное развёртывание изменений

**Продакшен паттерн:**
\`\`\`go
// Batch processing with limits
func ProcessUserBatch(ctx context.Context, users []User) error {
    jobs := make([]Job, len(users))
    for i, user := range users {
        u := user
        jobs[i] = func(ctx context.Context) error {
            return processUser(ctx, u)
        }
    }

    // Process 100 users at a time
    batchSize := 100
    processed := 0

    for processed < len(jobs) {
        count, err := RunSequentialLimited(ctx, jobs[processed:], batchSize)
        processed += count
        if err != nil {
            return fmt.Errorf("processed %d/%d: %w", processed, len(jobs), err)
        }
        time.Sleep(time.Second) // Rate limiting between batches
    }

    return nil
}

// Gradual feature rollout
func RolloutFeature(ctx context.Context, targets []string, percentage int) error {
    limit := (len(targets) * percentage) / 100

    jobs := make([]Job, len(targets))
    for i, target := range targets {
        t := target
        jobs[i] = func(ctx context.Context) error {
            return enableFeature(ctx, t)
        }
    }

    count, err := RunSequentialLimited(ctx, jobs, limit)
    log.Printf("Enabled feature for %d/%d targets", count, len(targets))
    return err
}

// Database migration with checkpoints
func MigrateRecords(ctx context.Context, recordIDs []string) error {
    jobs := make([]Job, len(recordIDs))
    for i, id := range recordIDs {
        recordID := id
        jobs[i] = func(ctx context.Context) error {
            return migrateRecord(ctx, recordID)
        }
    }

    checkpointSize := 1000
    totalProcessed := 0

    for totalProcessed < len(jobs) {
        count, err := RunSequentialLimited(ctx, jobs[totalProcessed:], checkpointSize)
        totalProcessed += count

        // Save checkpoint
        saveCheckpoint(totalProcessed)

        if err != nil {
            return fmt.Errorf("migration failed at record %d: %w", totalProcessed, err)
        }
    }

    return nil
}

// API rate limiting
func ProcessAPIRequests(ctx context.Context, requests []Request) error {
    jobs := make([]Job, len(requests))
    for i, req := range requests {
        r := req
        jobs[i] = func(ctx context.Context) error {
            return makeAPICall(ctx, r)
        }
    }

    // API limit: 10 requests per second
    ticker := time.NewTicker(time.Second)
    defer ticker.Stop()

    processed := 0
    for processed < len(jobs) {
        <-ticker.C // Wait for next second

        count, err := RunSequentialLimited(ctx, jobs[processed:], 10)
        processed += count

        if err != nil {
            log.Printf("API error after %d requests: %v", processed, err)
            return err
        }
    }

    return nil
}

// Testing with progressive sample sizes
func TestDataProcessing(ctx context.Context, testData []TestCase) error {
    jobs := make([]Job, len(testData))
    for i, tc := range testData {
        testCase := tc
        jobs[i] = func(ctx context.Context) error {
            return runTest(ctx, testCase)
        }
    }

    // Test with increasing sample sizes
    sampleSizes := []int{10, 50, 100, 500, len(jobs)}

    for _, size := range sampleSizes {
        log.Printf("Testing with %d samples...", size)
        count, err := RunSequentialLimited(ctx, jobs, size)

        if err != nil {
            return fmt.Errorf("test failed at sample size %d (processed %d): %w", size, count, err)
        }

        log.Printf("Passed %d tests", count)
    }

    return nil
}
\`\`\`

**Практические преимущества:**
- **Контролируемая обработка:** Предотвращение перегрузки нижестоящих систем
- **Контрольные точки:** Сохранение прогресса через регулярные интервалы
- **Тестирование:** Проверка с малыми выборками перед полным запуском
- **Ограничение скорости:** Соблюдение ограничений API или ресурсов

**Когда использовать:**
- Пакетная обработка с ограничениями размера
- Постепенные развёртывания или canary deployments
- API вызовы с ограничением скорости
- Тестирование с прогрессивными выборками
- Миграции БД с контрольными точками

**Дизайн возвращаемого значения:**
Функция возвращает и count и error, позволяя вызывающим точно знать сколько задач было обработано до возникновения ошибки. Это критично для функциональности контрольных точек/возобновления.`,
			solutionCode: `package concurrency

import (
	"context"
)

type Job func(context.Context) error

func RunSequentialLimited(ctx context.Context, jobs []Job, limit int) (int, error) {
	if ctx == nil {                                                 // Обработка nil контекста
		return 0, nil                                           // Возврат нулевого счёта
	}
	if limit <= 0 {                                                 // Обработка неверного лимита
		return 0, nil                                           // Возврат нулевого счёта
	}
	count := 0                                                      // Отслеживание обработанных задач
	for _, job := range jobs {                                      // Итерация по задачам
		if count >= limit {                                     // Проверка достижения лимита
			break                                           // Остановка обработки
		}
		if err := ctx.Err(); err != nil {                       // Проверка отмены контекста
			return count, err                               // Возврат счёта и ошибки
		}
		if job == nil {                                         // Обработка nil задачи
			count++                                         // Считать как обработанную
			continue                                        // Пропустить к следующей
		}
		if err := job(ctx); err != nil {                        // Выполнение задачи
			count++                                         // Считать даже при ошибке
			return count, err                               // Возврат счёта и ошибки
		}
		count++                                                 // Увеличение счётчика
	}
	return count, ctx.Err()                                         // Возврат финального счёта и состояния контекста
}`
		},
		uz: {
			title: 'Limitli ketma-ket bajarish',
			description: `Vazifalarni ketma-ket bajaradigan, lekin cheklangan miqdordagi vazifalarni qayta ishlagandan keyin to'xtaydigan **RunSequentialLimited** ni amalga oshiring.

**Talablar:**
1. \`RunSequentialLimited(ctx context.Context, jobs []Job, limit int) (int, error)\` funksiyasini yarating
2. nil kontekstni ishlang (0, nil qaytaring)
3. limit <= 0 ni ishlang (0, nil qaytaring)
4. \`limit\` gacha vazifalarni ketma-ket bajaring
5. nil vazifalarni qayta ishlangan deb hisoblang (hisoblagichni oshiring)
6. Limitga yetgandan keyin to'xtating
7. Qayta ishlangan vazifalar sonini va har qanday xatoni qaytaring
8. Har bir vazifadan oldin kontekst bekor qilinishini tekshiring

**Misol:**
\`\`\`go
jobs := []Job{
    job1, job2, job3, job4, job5,
}

// Faqat birinchi 3 vazifani qayta ishlang
count, err := RunSequentialLimited(ctx, jobs, 3)
// count = 3, faqat 1-3 vazifalar bajarildi

// nil vazifa bilan qayta ishlash
jobs := []Job{job1, nil, job3, job4}
count, err := RunSequentialLimited(ctx, jobs, 3)
// count = 3, nil qayta ishlangan deb hisoblanadi

// Ikkinchi vazifada xato
jobs := []Job{
    func(ctx context.Context) error { return nil },
    func(ctx context.Context) error { return errors.New("failed") },
    func(ctx context.Context) error { return nil },
}
count, err := RunSequentialLimited(ctx, jobs, 5)
// count = 2, err = "failed"
`,
			hint1: `Qayta ishlangan vazifalarni kuzatish uchun hisoblagich o'zgaruvchisidan foydalaning. Har bir vazifani qayta ishlashdan oldin count >= limit ni tekshiring.`,
			hint2: `Muvaffaqiyatli vazifalar va nil vazifalar uchun countni oshiring. Vazifa xato qaytarganda ham qaytarishdan oldin countni oshiring.`,
			whyItMatters: `RunSequentialLimited faqat vazifalarning bir qismini qayta ishlash kerak bo'lganda foydali, tezlikni cheklash, test qilish yoki bosqichma-bosqich joylash uchun foydali.

**Nima uchun bajarishni cheklash:**
- **Tezlikni cheklash:** Interval uchun belgilangan miqdorni qayta ishlash
- **Test qilish:** Avval kichik partiyalar bilan tekshirish
- **Resurslarni nazorat qilish:** Tizimlarni haddan tashqari yuklashning oldini olish
- **Bosqichma-bosqich joylash:** O'zgarishlarni bosqichma-bosqich joylash

**Ishlab chiqarish patterni:**
\`\`\`go
// Batch processing with limits
func ProcessUserBatch(ctx context.Context, users []User) error {
    jobs := make([]Job, len(users))
    for i, user := range users {
        u := user
        jobs[i] = func(ctx context.Context) error {
            return processUser(ctx, u)
        }
    }

    // Process 100 users at a time
    batchSize := 100
    processed := 0

    for processed < len(jobs) {
        count, err := RunSequentialLimited(ctx, jobs[processed:], batchSize)
        processed += count
        if err != nil {
            return fmt.Errorf("processed %d/%d: %w", processed, len(jobs), err)
        }
        time.Sleep(time.Second) // Rate limiting between batches
    }

    return nil
}

// Gradual feature rollout
func RolloutFeature(ctx context.Context, targets []string, percentage int) error {
    limit := (len(targets) * percentage) / 100

    jobs := make([]Job, len(targets))
    for i, target := range targets {
        t := target
        jobs[i] = func(ctx context.Context) error {
            return enableFeature(ctx, t)
        }
    }

    count, err := RunSequentialLimited(ctx, jobs, limit)
    log.Printf("Enabled feature for %d/%d targets", count, len(targets))
    return err
}

// Database migration with checkpoints
func MigrateRecords(ctx context.Context, recordIDs []string) error {
    jobs := make([]Job, len(recordIDs))
    for i, id := range recordIDs {
        recordID := id
        jobs[i] = func(ctx context.Context) error {
            return migrateRecord(ctx, recordID)
        }
    }

    checkpointSize := 1000
    totalProcessed := 0

    for totalProcessed < len(jobs) {
        count, err := RunSequentialLimited(ctx, jobs[totalProcessed:], checkpointSize)
        totalProcessed += count

        // Save checkpoint
        saveCheckpoint(totalProcessed)

        if err != nil {
            return fmt.Errorf("migration failed at record %d: %w", totalProcessed, err)
        }
    }

    return nil
}

// API rate limiting
func ProcessAPIRequests(ctx context.Context, requests []Request) error {
    jobs := make([]Job, len(requests))
    for i, req := range requests {
        r := req
        jobs[i] = func(ctx context.Context) error {
            return makeAPICall(ctx, r)
        }
    }

    // API limit: 10 requests per second
    ticker := time.NewTicker(time.Second)
    defer ticker.Stop()

    processed := 0
    for processed < len(jobs) {
        <-ticker.C // Wait for next second

        count, err := RunSequentialLimited(ctx, jobs[processed:], 10)
        processed += count

        if err != nil {
            log.Printf("API error after %d requests: %v", processed, err)
            return err
        }
    }

    return nil
}

// Testing with progressive sample sizes
func TestDataProcessing(ctx context.Context, testData []TestCase) error {
    jobs := make([]Job, len(testData))
    for i, tc := range testData {
        testCase := tc
        jobs[i] = func(ctx context.Context) error {
            return runTest(ctx, testCase)
        }
    }

    // Test with increasing sample sizes
    sampleSizes := []int{10, 50, 100, 500, len(jobs)}

    for _, size := range sampleSizes {
        log.Printf("Testing with %d samples...", size)
        count, err := RunSequentialLimited(ctx, jobs, size)

        if err != nil {
            return fmt.Errorf("test failed at sample size %d (processed %d): %w", size, count, err)
        }

        log.Printf("Passed %d tests", count)
    }

    return nil
}
\`\`\`

**Amaliy foydalari:**
- **Boshqariladigan qayta ishlash:** Pastki tizimlarni ortiqcha yuklashning oldini olish
- **Nazorat nuqtalari:** Muntazam intervallar orqali jarayonni saqlash
- **Test qilish:** To'liq ishga tushirishdan oldin kichik namunalar bilan tekshirish
- **Tezlikni cheklash:** API yoki resurs cheklovlariga rioya qilish

**Qachon ishlatiladi:**
- O'lcham cheklovlari bilan paketli qayta ishlash
- Bosqichma-bosqich joylashlar yoki canary deployments
- Tezlik cheklangan API chaqiruvlari
- Progressiv namunalar bilan test qilish
- Nazorat nuqtalari bilan DB migratsiyalari

**Qaytariladigan qiymat dizayni:**
Funksiya count va error ni qaytaradi, bu chaqiruvchilarga xato yuz berishidan oldin qancha vazifa qayta ishlanganligini aniq bilish imkonini beradi. Bu nazorat nuqtalari/qayta boshlash funksiyasi uchun juda muhim.`,
			solutionCode: `package concurrency

import (
	"context"
)

type Job func(context.Context) error

func RunSequentialLimited(ctx context.Context, jobs []Job, limit int) (int, error) {
	if ctx == nil {                                                 // nil kontekstni ishlash
		return 0, nil                                           // Nol sonini qaytarish
	}
	if limit <= 0 {                                                 // Noto'g'ri limitni ishlash
		return 0, nil                                           // Nol sonini qaytarish
	}
	count := 0                                                      // Qayta ishlangan vazifalarni kuzatish
	for _, job := range jobs {                                      // Vazifalar bo'yicha iteratsiya
		if count >= limit {                                     // Limitga yetilganligini tekshirish
			break                                           // Qayta ishlashni to'xtatish
		}
		if err := ctx.Err(); err != nil {                       // Kontekst bekor qilinishini tekshirish
			return count, err                               // Soni va xatoni qaytarish
		}
		if job == nil {                                         // nil vazifani ishlash
			count++                                         // Qayta ishlangan deb hisoblash
			continue                                        // Keyingisiga o'tish
		}
		if err := job(ctx); err != nil {                        // Vazifani bajarish
			count++                                         // Xatoda ham hisoblash
			return count, err                               // Soni va xatoni qaytarish
		}
		count++                                                 // Hisoblagichni oshirish
	}
	return count, ctx.Err()                                         // Yakuniy son va kontekst holatini qaytarish
}`
		}
	}
};

export default task;
