import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-workerpool-throttle-job-submission',
	title: 'Throttle Job Submission',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'worker-pool', 'rate-limiting', 'throttling'],
	estimatedTime: '35m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **ThrottleJobSubmission** that rate-limits job submission by ensuring a minimum time interval between consecutive jobs.

**Requirements:**
1. Create function \`ThrottleJobSubmission(ctx context.Context, in <-chan Job, interval time.Duration) <-chan Job\`
2. Return read-only channel of throttled jobs
3. Ensure minimum \`interval\` time between consecutive job emissions
4. Handle nil context (forward jobs without throttling)
5. Handle interval <= 0 (forward jobs immediately without throttling)
6. Track time of last job sent using time.Now()
7. Use time.NewTimer for precise waiting
8. Clean up timer resources properly
9. Close output channel when input closes or context cancels

**Throttling Logic:**
- First job: Send immediately, record time
- Subsequent jobs: Wait for interval since last send, then send

**Example:**
\`\`\`go
in := make(chan Job, 5)

// Send jobs rapidly
go func() {
    for i := 0; i < 5; i++ {
        in <- func(ctx context.Context) error {
            fmt.Printf("Job %d\\n", i)
            return nil
        }
    }
    close(in)
}()

// Throttle to 1 job per 100ms
throttled := ThrottleJobSubmission(ctx, in, 100*time.Millisecond)

start := time.Now()
for job := range throttled {
    fmt.Printf("Received at %v\\n", time.Since(start))
    job(ctx)
}

// Output:
// Received at 0s      (job 0 - immediate)
// Received at 100ms   (job 1 - waited 100ms)
// Received at 200ms   (job 2 - waited 100ms)
// Received at 300ms   (job 3 - waited 100ms)
// Received at 400ms   (job 4 - waited 100ms)
\`\`\`

**Constraints:**
- Must maintain precise interval between emissions
- Must use time.NewTimer (not time.Sleep in select)
- Must clean up timer with timer.Stop() and drain channel if needed
- Must handle context cancellation during wait`,
	initialCode: `package concurrency

import (
	"context"
	"time"
)

type Job func(context.Context) error

// TODO: Implement ThrottleJobSubmission
func ThrottleJobSubmission(ctx context.Context, in <-chan Job, interval time.Duration) <-chan Job {
	// TODO: Implement
}`,
	testCode: `package concurrency

import (
	"context"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	in := make(chan Job)
	close(in)
	out := ThrottleJobSubmission(context.Background(), in, 100*time.Millisecond)
	count := 0
	for range out {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 jobs from closed input, got %d", count)
	}
}

func Test2(t *testing.T) {
	in := make(chan Job, 1)
	in <- func(ctx context.Context) error { return nil }
	close(in)
	out := ThrottleJobSubmission(context.Background(), in, 50*time.Millisecond)
	count := 0
	for range out {
		count++
	}
	if count != 1 {
		t.Errorf("expected 1 job, got %d", count)
	}
}

func Test3(t *testing.T) {
	in := make(chan Job, 3)
	in <- func(ctx context.Context) error { return nil }
	in <- func(ctx context.Context) error { return nil }
	in <- func(ctx context.Context) error { return nil }
	close(in)
	start := time.Now()
	out := ThrottleJobSubmission(context.Background(), in, 50*time.Millisecond)
	count := 0
	for range out {
		count++
	}
	elapsed := time.Since(start)
	if count != 3 {
		t.Errorf("expected 3 jobs, got %d", count)
	}
	if elapsed < 100*time.Millisecond {
		t.Errorf("expected at least 100ms for 3 jobs at 50ms interval, got %v", elapsed)
	}
}

func Test4(t *testing.T) {
	in := make(chan Job, 2)
	in <- func(ctx context.Context) error { return nil }
	in <- func(ctx context.Context) error { return nil }
	close(in)
	start := time.Now()
	out := ThrottleJobSubmission(context.Background(), in, 0)
	count := 0
	for range out {
		count++
	}
	elapsed := time.Since(start)
	if count != 2 {
		t.Errorf("expected 2 jobs, got %d", count)
	}
	if elapsed > 50*time.Millisecond {
		t.Errorf("expected fast with interval=0, got %v", elapsed)
	}
}

func Test5(t *testing.T) {
	in := make(chan Job, 2)
	in <- func(ctx context.Context) error { return nil }
	in <- func(ctx context.Context) error { return nil }
	close(in)
	start := time.Now()
	out := ThrottleJobSubmission(context.Background(), in, -100*time.Millisecond)
	count := 0
	for range out {
		count++
	}
	elapsed := time.Since(start)
	if count != 2 {
		t.Errorf("expected 2 jobs with negative interval, got %d", count)
	}
	if elapsed > 50*time.Millisecond {
		t.Errorf("expected fast with negative interval, got %v", elapsed)
	}
}

func Test6(t *testing.T) {
	in := make(chan Job, 2)
	in <- func(ctx context.Context) error { return nil }
	in <- func(ctx context.Context) error { return nil }
	close(in)
	out := ThrottleJobSubmission(nil, in, 50*time.Millisecond)
	count := 0
	for range out {
		count++
	}
	if count != 2 {
		t.Errorf("expected 2 jobs with nil context, got %d", count)
	}
}

func Test7(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	in := make(chan Job, 5)
	for i := 0; i < 5; i++ {
		in <- func(ctx context.Context) error { return nil }
	}
	close(in)
	out := ThrottleJobSubmission(ctx, in, 100*time.Millisecond)
	count := 0
	for range out {
		count++
		if count == 2 {
			cancel()
		}
	}
	if count >= 5 {
		t.Errorf("expected cancellation to stop throttling, got all %d", count)
	}
}

func Test8(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	in := make(chan Job, 2)
	in <- func(ctx context.Context) error { return nil }
	in <- func(ctx context.Context) error { return nil }
	close(in)
	out := ThrottleJobSubmission(ctx, in, 50*time.Millisecond)
	count := 0
	for range out {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 jobs from pre-cancelled context, got %d", count)
	}
}

func Test9(t *testing.T) {
	in := make(chan Job, 1)
	in <- func(ctx context.Context) error { return nil }
	close(in)
	start := time.Now()
	out := ThrottleJobSubmission(context.Background(), in, 50*time.Millisecond)
	<-out
	elapsed := time.Since(start)
	if elapsed > 30*time.Millisecond {
		t.Errorf("first job should be immediate, took %v", elapsed)
	}
}

func Test10(t *testing.T) {
	in := make(chan Job, 5)
	for i := 0; i < 5; i++ {
		in <- func(ctx context.Context) error { return nil }
	}
	close(in)
	start := time.Now()
	out := ThrottleJobSubmission(context.Background(), in, 30*time.Millisecond)
	var times []time.Duration
	for range out {
		times = append(times, time.Since(start))
	}
	if len(times) != 5 {
		t.Errorf("expected 5 jobs, got %d", len(times))
	}
	for i := 1; i < len(times); i++ {
		diff := times[i] - times[i-1]
		if diff < 20*time.Millisecond {
			t.Errorf("expected ~30ms between jobs, got %v at position %d", diff, i)
		}
	}
}
`,
	solutionCode: `package concurrency

import (
	"context"
	"time"
)

type Job func(context.Context) error

func ThrottleJobSubmission(ctx context.Context, in <-chan Job, interval time.Duration) <-chan Job {
	out := make(chan Job)                                           // Create output channel
	go func() {                                                     // Launch throttler goroutine
		defer close(out)                                        // Always close output channel
		if ctx == nil {                                         // Handle nil context
			for job := range in {                           // Forward all jobs
				out <- job                              // No throttling
			}
			return                                          // Done
		}
		if interval <= 0 {                                      // Handle non-positive interval
			for {                                           // Forward loop
				select {                                // Check context or receive
				case <-ctx.Done():                      // Context cancelled
					return                          // Exit
				case job, ok := <-in:                   // Receive job
					if !ok {                        // Input channel closed
						return                  // Exit
					}
					select {                        // Send or cancel
					case <-ctx.Done():              // Context cancelled
						return                  // Exit
					case out <- job:                // Forward job immediately
					}
				}
			}
		}
		var last time.Time                                      // Track last emission time
		for {                                                   // Throttling loop
			select {                                        // Check context or receive
			case <-ctx.Done():                              // Context cancelled
				return                                  // Exit
			case job, ok := <-in:                           // Receive job
				if !ok {                                // Input channel closed
					return                          // Exit
				}
				if !last.IsZero() {                     // Not first job
					wait := interval - time.Since(last) // Calculate wait time
					if wait > 0 {                   // Need to wait
						timer := time.NewTimer(wait) // Create timer
						select {                // Wait or cancel
						case <-ctx.Done():      // Context cancelled
							if !timer.Stop() { // Try to stop timer
								<-timer.C // Drain if already fired
							}
							return          // Exit
						case <-timer.C:         // Timer fired
						}
					}
				}
				select {                                // Send or cancel
				case <-ctx.Done():                      // Context cancelled
					return                          // Exit
				case out <- job:                        // Send job
					last = time.Now()               // Update last emission time
				}
			}
		}
	}()
	return out                                                      // Return read-only channel
}`,
			hint1: `Use var last time.Time to track last emission. For first job (last.IsZero()), send immediately. For others, calculate wait := interval - time.Since(last).`,
			hint2: `If wait > 0, use timer := time.NewTimer(wait) and select between <-ctx.Done() and <-timer.C. If stopping timer, check timer.Stop() and drain with <-timer.C if false.`,
			whyItMatters: `ThrottleJobSubmission implements rate limiting, essential for protecting APIs, databases, and external services from being overwhelmed.

**Why Throttle Job Submission:**
- **API Rate Limits:** Respect external API quotas
- **Resource Protection:** Prevent overwhelming services
- **Cost Control:** Limit expensive operations
- **Fair Usage:** Distribute load over time

**Production Pattern:**
\`\`\`go
// API rate limiting (e.g., 10 requests per second)
func CallExternalAPI(ctx context.Context, requests []Request) error {
    jobs := make(chan Job, len(requests))

    go func() {
        defer close(jobs)
        for _, req := range requests {
            r := req
            jobs <- func(ctx context.Context) error {
                return makeAPICall(ctx, r)
            }
        }
    }()

    // Throttle to 1 request per 100ms (10/second)
    throttled := ThrottleJobSubmission(ctx, jobs, 100*time.Millisecond)
    return RunPool(ctx, throttled, 1)
}

// Database write throttling
func BulkInsert(ctx context.Context, records []Record) error {
    jobs := make(chan Job, len(records))

    go func() {
        defer close(jobs)
        for _, record := range records {
            r := record
            jobs <- func(ctx context.Context) error {
                return db.Insert(r)
            }
        }
    }()

    // Throttle to 1 insert per 10ms to avoid overwhelming DB
    throttled := ThrottleJobSubmission(ctx, jobs, 10*time.Millisecond)
    return RunPool(ctx, throttled, 5)
}

// Email sending rate limit
func SendBulkEmails(ctx context.Context, emails []Email) error {
    jobs := make(chan Job, len(emails))

    go func() {
        defer close(jobs)
        for _, email := range emails {
            e := email
            jobs <- func(ctx context.Context) error {
                return sendEmail(ctx, e)
            }
        }
    }()

    // Email provider limit: 100/minute = 1 per 600ms
    throttled := ThrottleJobSubmission(ctx, jobs, 600*time.Millisecond)
    return RunPool(ctx, throttled, 1)
}

// Cloud API with burst limits
func UploadFiles(ctx context.Context, files []File) error {
    jobs := make(chan Job, len(files))

    go func() {
        defer close(jobs)
        for _, file := range files {
            f := file
            jobs <- func(ctx context.Context) error {
                return uploadFile(ctx, f)
            }
        }
    }()

    // Cloud storage: 1000 requests/minute = 1 per 60ms
    throttled := ThrottleJobSubmission(ctx, jobs, 60*time.Millisecond)
    return RunPool(ctx, throttled, 10) // Parallel but throttled submission
}

// Webhook notifications
func SendWebhooks(ctx context.Context, webhooks []Webhook) error {
    jobs := make(chan Job, len(webhooks))

    go func() {
        defer close(jobs)
        for _, webhook := range webhooks {
            w := webhook
            jobs <- func(ctx context.Context) error {
                return sendWebhook(ctx, w)
            }
        }
    }()

    // Respect webhook rate limits
    throttled := ThrottleJobSubmission(ctx, jobs, 200*time.Millisecond)
    return RunPool(ctx, throttled, 3)
}

// Crawling with politeness delay
func CrawlWebsite(ctx context.Context, urls []string) error {
    jobs := make(chan Job, len(urls))

    go func() {
        defer close(jobs)
        for _, url := range urls {
            u := url
            jobs <- func(ctx context.Context) error {
                return crawlPage(ctx, u)
            }
        }
    }()

    // Politeness delay: 1 second between requests
    throttled := ThrottleJobSubmission(ctx, jobs, time.Second)
    return RunPool(ctx, throttled, 1)
}

// Multi-tier rate limiting
func ProcessWithTieredRates(ctx context.Context, items []Item) error {
    priorityJobs := make(chan Job, 100)
    normalJobs := make(chan Job, 100)

    // Separate by priority
    go func() {
        defer close(priorityJobs)
        defer close(normalJobs)

        for _, item := range items {
            i := item
            job := func(ctx context.Context) error {
                return processItem(ctx, i)
            }

            if i.Priority == "high" {
                priorityJobs <- job
            } else {
                normalJobs <- job
            }
        }
    }()

    var wg sync.WaitGroup
    var firstErr error
    var once sync.Once

    // High priority: faster rate
    wg.Add(1)
    go func() {
        defer wg.Done()
        throttled := ThrottleJobSubmission(ctx, priorityJobs, 50*time.Millisecond)
        if err := RunPool(ctx, throttled, 5); err != nil {
            once.Do(func() { firstErr = err })
        }
    }()

    // Normal priority: slower rate
    wg.Add(1)
    go func() {
        defer wg.Done()
        throttled := ThrottleJobSubmission(ctx, normalJobs, 200*time.Millisecond)
        if err := RunPool(ctx, throttled, 2); err != nil {
            once.Do(func() { firstErr = err })
        }
    }()

    wg.Wait()
    return firstErr
}

// Dynamic rate adjustment
func ProcessWithDynamicRate(ctx context.Context, jobs <-chan Job) error {
    var currentInterval = 100 * time.Millisecond
    adjustedJobs := make(chan Job)

    go func() {
        defer close(adjustedJobs)
        for job := range jobs {
            // Adjust rate based on system load
            load := getSystemLoad()
            if load > 0.8 {
                currentInterval = 200 * time.Millisecond // Slow down
            } else if load < 0.3 {
                currentInterval = 50 * time.Millisecond // Speed up
            }

            throttled := ThrottleJobSubmission(ctx, makeSingleJobChan(job), currentInterval)
            for j := range throttled {
                adjustedJobs <- j
            }
        }
    }()

    return RunPool(ctx, adjustedJobs, 5)
}
\`\`\`

**Real-World Benefits:**
- **Compliance:** Respect rate limits and quotas
- **Stability:** Prevent overwhelming downstream services
- **Cost Savings:** Avoid overage charges
- **Good Citizenship:** Be a polite API consumer

**Common Rate Limits:**
- **Twitter API:** 300 requests per 15 minutes
- **GitHub API:** 5000 requests per hour
- **Google APIs:** Various (e.g., 100 queries/second)
- **SendGrid:** Various based on plan
- **Stripe:** 100 requests per second

**Calculating Interval:**
\`\`\`go
// Rate limit: R requests per time period T
interval = T / R

// Examples:
// 100 requests/minute: interval = 60s / 100 = 600ms
// 10 requests/second: interval = 1s / 10 = 100ms
// 5000 requests/hour: interval = 3600s / 5000 = 720ms
\`\`\`

**Timer Cleanup Pattern:**
It's crucial to properly clean up timers:
\`\`\`go
timer := time.NewTimer(wait)
select {
case <-ctx.Done():
    if !timer.Stop() {  // Try to stop timer
        <-timer.C       // Drain if already fired
    }
    return
case <-timer.C:
    // Timer fired normally
}
\`\`\`

This prevents goroutine leaks and ensures proper resource management.

**When to Use:**
- Calling external APIs with rate limits
- Database operations that might overwhelm the DB
- Email/SMS sending with provider limits
- Web crawling with politeness delays
- Any operation with throughput constraints

**Pipeline Position:**
ThrottleJobSubmission fits between job production and consumption:
\`\`\`
Producer -> ThrottleJobSubmission -> Worker Pool
\`\`\`

This pattern is essential for building production systems that integrate with rate-limited services.`,	order: 7,
	translations: {
		ru: {
			title: 'Контроль скорости подачи задач в пул',
			description: `Реализуйте **ThrottleJobSubmission**, который ограничивает скорость отправки задач, обеспечивая минимальный временной интервал между последовательными задачами.

**Требования:**
1. Создайте функцию \`ThrottleJobSubmission(ctx context.Context, in <-chan Job, interval time.Duration) <-chan Job\`
2. Верните канал только для чтения с троттлированными задачами
3. Обеспечьте минимальный интервал \`interval\` между последовательными эмиссиями задач
4. Обработайте nil context (перенаправляйте задачи без троттлинга)
5. Обработайте interval <= 0 (перенаправляйте задачи немедленно без троттлинга)
6. Отслеживайте время последней отправленной задачи используя time.Now()
7. Используйте time.NewTimer для точного ожидания
8. Правильно очищайте ресурсы таймера
9. Закройте выходной канал когда входной закрывается или контекст отменяется

**Логика троттлинга:**
- Первая задача: Отправить немедленно, записать время
- Последующие задачи: Ждать interval с момента последней отправки, затем отправить

**Пример:**
\`\`\`go
in := make(chan Job, 5)

// Быстро отправить задачи
go func() {
    for i := 0; i < 5; i++ {
        in <- func(ctx context.Context) error {
            fmt.Printf("Job %d\\n", i)
            return nil
        }
    }
    close(in)
}()

// Троттлинг до 1 задачи на 100мс
throttled := ThrottleJobSubmission(ctx, in, 100*time.Millisecond)

start := time.Now()
for job := range throttled {
    fmt.Printf("Received at %v\\n", time.Since(start))
    job(ctx)
}

// Вывод:
// Received at 0s      (задача 0 - немедленно)
// Received at 100ms   (задача 1 - ждали 100мс)
// Received at 200ms   (задача 2 - ждали 100мс)
// Received at 300ms   (задача 3 - ждали 100мс)
// Received at 400ms   (задача 4 - ждали 100мс)
`,
			hint1: `Используйте var last time.Time для отслеживания последней эмиссии. Для первой задачи (last.IsZero()), отправьте немедленно. Для остальных вычисляйте wait := interval - time.Since(last).`,
			hint2: `Если wait > 0, используйте timer := time.NewTimer(wait) и select между <-ctx.Done() и <-timer.C. При остановке таймера проверьте timer.Stop() и если false, очистите <-timer.C.`,
			whyItMatters: `ThrottleJobSubmission реализует ограничение скорости (rate limiting), необходимое для защиты API, баз данных и внешних сервисов от перегрузки и соблюдения квот.

**Почему троттлить отправку задач:**
- **Лимиты API:** Соблюдение квот внешних API (Twitter, GitHub, Google)
- **Защита ресурсов:** Предотвращение перегрузки downstream сервисов
- **Контроль затрат:** Ограничение дорогостоящих операций и избежание overage
- **Справедливое использование:** Равномерное распределение нагрузки во времени
- **Вежливость:** Быть хорошим гражданином при использовании чужих API

**Продакшен паттерн:**
\`\`\`go
// Ограничение вызовов API (например, 10 запросов в секунду)
func CallExternalAPI(ctx context.Context, requests []Request) error {
    jobs := make(chan Job, len(requests))

    go func() {
        defer close(jobs)
        for _, req := range requests {
            r := req
            jobs <- func(ctx context.Context) error {
                return makeAPICall(ctx, r)
            }
        }
    }()

    // Троттлинг до 1 запроса на 100мс (10/секунду)
    throttled := ThrottleJobSubmission(ctx, jobs, 100*time.Millisecond)
    return RunPool(ctx, throttled, 1)
}

// Троттлинг записи в базу данных
func BulkInsert(ctx context.Context, records []Record) error {
    jobs := make(chan Job, len(records))

    go func() {
        defer close(jobs)
        for _, record := range records {
            r := record
            jobs <- func(ctx context.Context) error {
                return db.Insert(r)
            }
        }
    }()

    // Троттлинг до 1 вставки на 10мс чтобы не перегрузить БД
    throttled := ThrottleJobSubmission(ctx, jobs, 10*time.Millisecond)
    return RunPool(ctx, throttled, 5)
}

// Ограничение скорости отправки email
func SendBulkEmails(ctx context.Context, emails []Email) error {
    jobs := make(chan Job, len(emails))

    go func() {
        defer close(jobs)
        for _, email := range emails {
            e := email
            jobs <- func(ctx context.Context) error {
                return sendEmail(ctx, e)
            }
        }
    }()

    // Лимит email-провайдера: 100/минуту = 1 на 600мс
    throttled := ThrottleJobSubmission(ctx, jobs, 600*time.Millisecond)
    return RunPool(ctx, throttled, 1)
}

// Облачное API с burst лимитами
func UploadFiles(ctx context.Context, files []File) error {
    jobs := make(chan Job, len(files))

    go func() {
        defer close(jobs)
        for _, file := range files {
            f := file
            jobs <- func(ctx context.Context) error {
                return uploadFile(ctx, f)
            }
        }
    }()

    // Облачное хранилище: 1000 запросов/минуту = 1 на 60мс
    throttled := ThrottleJobSubmission(ctx, jobs, 60*time.Millisecond)
    return RunPool(ctx, throttled, 10)  // Параллельно но с троттлингом отправки
}

// Webhook уведомления
func SendWebhooks(ctx context.Context, webhooks []Webhook) error {
    jobs := make(chan Job, len(webhooks))

    go func() {
        defer close(jobs)
        for _, webhook := range webhooks {
            w := webhook
            jobs <- func(ctx context.Context) error {
                return sendWebhook(ctx, w)
            }
        }
    }()

    // Соблюдение лимитов webhook
    throttled := ThrottleJobSubmission(ctx, jobs, 200*time.Millisecond)
    return RunPool(ctx, throttled, 3)
}

// Веб-краулинг с вежливой задержкой
func CrawlWebsite(ctx context.Context, urls []string) error {
    jobs := make(chan Job, len(urls))

    go func() {
        defer close(jobs)
        for _, url := range urls {
            u := url
            jobs <- func(ctx context.Context) error {
                return crawlPage(ctx, u)
            }
        }
    }()

    // Вежливая задержка: 1 секунда между запросами
    throttled := ThrottleJobSubmission(ctx, jobs, time.Second)
    return RunPool(ctx, throttled, 1)
}

// Многоуровневое ограничение скорости
func ProcessWithTieredRates(ctx context.Context, items []Item) error {
    priorityJobs := make(chan Job, 100)
    normalJobs := make(chan Job, 100)

    // Разделение по приоритету
    go func() {
        defer close(priorityJobs)
        defer close(normalJobs)

        for _, item := range items {
            i := item
            job := func(ctx context.Context) error {
                return processItem(ctx, i)
            }

            if i.Priority == "high" {
                priorityJobs <- job
            } else {
                normalJobs <- job
            }
        }
    }()

    var wg sync.WaitGroup
    var firstErr error
    var once sync.Once

    // Высокий приоритет: быстрее
    wg.Add(1)
    go func() {
        defer wg.Done()
        throttled := ThrottleJobSubmission(ctx, priorityJobs, 50*time.Millisecond)
        if err := RunPool(ctx, throttled, 5); err != nil {
            once.Do(func() { firstErr = err })
        }
    }()

    // Нормальный приоритет: медленнее
    wg.Add(1)
    go func() {
        defer wg.Done()
        throttled := ThrottleJobSubmission(ctx, normalJobs, 200*time.Millisecond)
        if err := RunPool(ctx, throttled, 2); err != nil {
            once.Do(func() { firstErr = err })
        }
    }()

    wg.Wait()
    return firstErr
}

// Динамическая корректировка скорости
func ProcessWithDynamicRate(ctx context.Context, jobs <-chan Job) error {
    var currentInterval = 100 * time.Millisecond
    adjustedJobs := make(chan Job)

    go func() {
        defer close(adjustedJobs)
        for job := range jobs {
            // Корректировка скорости на основе нагрузки системы
            load := getSystemLoad()
            if load > 0.8 {
                currentInterval = 200 * time.Millisecond  // Замедление
            } else if load < 0.3 {
                currentInterval = 50 * time.Millisecond  // Ускорение
            }

            throttled := ThrottleJobSubmission(ctx, makeSingleJobChan(job), currentInterval)
            for j := range throttled {
                adjustedJobs <- j
            }
        }
    }()

    return RunPool(ctx, adjustedJobs, 5)
}
\`\`\`

**Практические преимущества:**
- **Соответствие требованиям:** Соблюдение лимитов скорости и квот
- **Стабильность:** Предотвращение перегрузки downstream сервисов
- **Экономия затрат:** Избежание платы за превышение лимитов
- **Хорошее гражданство:** Вежливое использование чужих API
- **Предсказуемость:** Контролируемая и стабильная нагрузка

**Типичные лимиты скорости:**
- **Twitter API:** 300 запросов за 15 минут
- **GitHub API:** 5000 запросов в час
- **Google APIs:** Различные (например, 100 запросов/секунду)
- **SendGrid:** Зависит от плана
- **Stripe:** 100 запросов в секунду

**Расчёт интервала:**
\`\`\`go
// Лимит скорости: R запросов за период времени T
interval = T / R

// Примеры:
// 100 запросов/минута: interval = 60s / 100 = 600ms
// 10 запросов/секунда: interval = 1s / 10 = 100ms
// 5000 запросов/час: interval = 3600s / 5000 = 720ms
\`\`\`

**Паттерн очистки таймера:**
Критически важно правильно очищать таймеры:
\`\`\`go
timer := time.NewTimer(wait)
select {
case <-ctx.Done():
    if !timer.Stop() {  // Попытка остановить таймер
        <-timer.C       // Очистка если уже сработал
    }
    return
case <-timer.C:
    // Таймер сработал нормально
}
\`\`\`

Это предотвращает утечки горутин и обеспечивает правильное управление ресурсами.

**Когда использовать:**
- Вызовы внешних API с лимитами скорости
- Операции с БД которые могут перегрузить её
- Отправка email/SMS с лимитами провайдера
- Веб-краулинг с вежливыми задержками
- Любые операции с ограничениями throughput

**Позиция в конвейере:**
ThrottleJobSubmission вписывается между производством и потреблением задач:
\`\`\`
Продюсер -> ThrottleJobSubmission -> Пул воркеров
\`\`\`

Этот паттерн необходим для построения продакшен систем интегрирующихся с rate-limited сервисами.`,
			solutionCode: `package concurrency

import (
	"context"
	"time"
)

type Job func(context.Context) error

func ThrottleJobSubmission(ctx context.Context, in <-chan Job, interval time.Duration) <-chan Job {
	out := make(chan Job)                                           // Создание выходного канала
	go func() {                                                     // Запуск горутины троттлера
		defer close(out)                                        // Всегда закрывать выходной канал
		if ctx == nil {                                         // Обработка nil контекста
			for job := range in {                           // Перенаправление всех задач
				out <- job                              // Без троттлинга
			}
			return                                          // Завершено
		}
		if interval <= 0 {                                      // Обработка неположительного интервала
			for {                                           // Цикл перенаправления
				select {                                // Проверка контекста или получение
				case <-ctx.Done():                      // Контекст отменён
					return                          // Выход
				case job, ok := <-in:                   // Получение задачи
					if !ok {                        // Входной канал закрыт
						return                  // Выход
					}
					select {                        // Отправка или отмена
					case <-ctx.Done():              // Контекст отменён
						return                  // Выход
					case out <- job:                // Немедленное перенаправление задачи
					}
				}
			}
		}
		var last time.Time                                      // Отслеживание времени последней эмиссии
		for {                                                   // Цикл троттлинга
			select {                                        // Проверка контекста или получение
			case <-ctx.Done():                              // Контекст отменён
				return                                  // Выход
			case job, ok := <-in:                           // Получение задачи
				if !ok {                                // Входной канал закрыт
					return                          // Выход
				}
				if !last.IsZero() {                     // Не первая задача
					wait := interval - time.Since(last) // Расчёт времени ожидания
					if wait > 0 {                   // Нужно подождать
						timer := time.NewTimer(wait) // Создание таймера
						select {                // Ожидание или отмена
						case <-ctx.Done():      // Контекст отменён
							if !timer.Stop() { // Попытка остановить таймер
								<-timer.C // Очистка если уже сработал
							}
							return          // Выход
						case <-timer.C:         // Таймер сработал
						}
					}
				}
				select {                                // Отправка или отмена
				case <-ctx.Done():                      // Контекст отменён
					return                          // Выход
				case out <- job:                        // Отправка задачи
					last = time.Now()               // Обновление времени последней эмиссии
				}
			}
		}
	}()
	return out                                                      // Возврат канала только для чтения
}`
		},
		uz: {
			title: 'Pulga vazifa berish tezligini boshqarish',
			description: `Ketma-ket vazifalar o'rtasida minimal vaqt oralig'ini ta'minlab, vazifalarni yuborish tezligini cheklaydigan **ThrottleJobSubmission** ni amalga oshiring.

**Talablar:**
1. \`ThrottleJobSubmission(ctx context.Context, in <-chan Job, interval time.Duration) <-chan Job\` funksiyasini yarating
2. Throttle qilingan vazifalar bilan faqat o'qish uchun kanalni qaytaring
3. Ketma-ket vazifalar emissiyalari o'rtasida minimal 'interval' ni ta'minlang
4. nil kontekstni ishlang (throttle qilmasdan vazifalarni yo'naltiring)
5. interval <= 0 ni ishlang (vazifalarni darhol throttle qilmasdan yo'naltiring)
6. time.Now() dan foydalanib oxirgi yuborilgan vazifa vaqtini kuzating
7. Aniq kutish uchun time.NewTimer dan foydalaning
8. Timer resurslarini to'g'ri tozalang
9. Kirish yopilganda yoki kontekst bekor qilinganda chiqish kanalini yoping

**Throttle logikasi:**
- Birinchi vazifa: Darhol yuboring, vaqtni yozib oling
- Keyingi vazifalar: Oxirgi yuborishdan boshlab interval kuting, keyin yuboring

**Misol:**
\`\`\`go
in := make(chan Job, 5)

// Vazifalarni tez yuborish
go func() {
    for i := 0; i < 5; i++ {
        in <- func(ctx context.Context) error {
            fmt.Printf("Job %d\\n", i)
            return nil
        }
    }
    close(in)
}()

// 100ms da 1 vazifaga throttle qilish
throttled := ThrottleJobSubmission(ctx, in, 100*time.Millisecond)

start := time.Now()
for job := range throttled {
    fmt.Printf("Received at %v\\n", time.Since(start))
    job(ctx)
}

// Chiqish:
// Received at 0s      (vazifa 0 - darhol)
// Received at 100ms   (vazifa 1 - 100ms kutildi)
// Received at 200ms   (vazifa 2 - 100ms kutildi)
// Received at 300ms   (vazifa 3 - 100ms kutildi)
// Received at 400ms   (vazifa 4 - 100ms kutildi)
`,
			hint1: `Oxirgi emissiyani kuzatish uchun var last time.Time dan foydalaning. Birinchi vazifa uchun (last.IsZero()), darhol yuboring. Qolganlari uchun wait := interval - time.Since(last) ni hisoblang.`,
			hint2: `Agar wait > 0 bo'lsa, timer := time.NewTimer(wait) dan foydalaning va <-ctx.Done() va <-timer.C o'rtasida select qiling. Timerni to'xtatishda timer.Stop() ni tekshiring va agar false bo'lsa <-timer.C bilan tozalang.`,
			whyItMatters: `ThrottleJobSubmission tezlikni cheklashni (rate limiting) amalga oshiradi, API, ma'lumotlar bazalari va tashqi xizmatlarni ortiqcha yuklanishdan himoya qilish va kvotalarga rioya qilish uchun zarur.

**Nima uchun vazifalarni yuborishni throttle qilish:**
- **API limitleri:** Tashqi API kvotalariga rioya qilish (Twitter, GitHub, Google)
- **Resurslarni himoya qilish:** Downstream xizmatlarning ortiqcha yuklanishini oldini olish
- **Xarajatlarni nazorat qilish:** Qimmat operatsiyalarni cheklash va ortiqcha to'lovdan qochish
- **Adolatli foydalanish:** Yukni vaqt bo'yicha bir xil taqsimlash
- **Nazokat:** Boshqalarning APIlaridan foydalanishda yaxshi fuqaro bo'lish

**Ishlab chiqarish patterni:**
\`\`\`go
// API chaqiruvlarini cheklash (masalan, soniyada 10 so'rov)
func CallExternalAPI(ctx context.Context, requests []Request) error {
    jobs := make(chan Job, len(requests))

    go func() {
        defer close(jobs)
        for _, req := range requests {
            r := req
            jobs <- func(ctx context.Context) error {
                return makeAPICall(ctx, r)
            }
        }
    }()

    // 100ms da 1 so'rovga throttle qilish (10/soniya)
    throttled := ThrottleJobSubmission(ctx, jobs, 100*time.Millisecond)
    return RunPool(ctx, throttled, 1)
}

// Ma'lumotlar bazasiga yozishni throttle qilish
func BulkInsert(ctx context.Context, records []Record) error {
    jobs := make(chan Job, len(records))

    go func() {
        defer close(jobs)
        for _, record := range records {
            r := record
            jobs <- func(ctx context.Context) error {
                return db.Insert(r)
            }
        }
    }()

    // DBni ortiqcha yuklamaslik uchun 10ms da 1 insertga throttle qilish
    throttled := ThrottleJobSubmission(ctx, jobs, 10*time.Millisecond)
    return RunPool(ctx, throttled, 5)
}

// Email yuborish tezligini cheklash
func SendBulkEmails(ctx context.Context, emails []Email) error {
    jobs := make(chan Job, len(emails))

    go func() {
        defer close(jobs)
        for _, email := range emails {
            e := email
            jobs <- func(ctx context.Context) error {
                return sendEmail(ctx, e)
            }
        }
    }()

    // Email provayderning limiti: 100/daqiqa = 600ms da 1
    throttled := ThrottleJobSubmission(ctx, jobs, 600*time.Millisecond)
    return RunPool(ctx, throttled, 1)
}

// Burst limitleri bilan bulutli API
func UploadFiles(ctx context.Context, files []File) error {
    jobs := make(chan Job, len(files))

    go func() {
        defer close(jobs)
        for _, file := range files {
            f := file
            jobs <- func(ctx context.Context) error {
                return uploadFile(ctx, f)
            }
        }
    }()

    // Bulutli saqlash: 1000 so'rov/daqiqa = 60ms da 1
    throttled := ThrottleJobSubmission(ctx, jobs, 60*time.Millisecond)
    return RunPool(ctx, throttled, 10)  // Parallel lekin throttle qilingan yuborish
}

// Webhook bildirishnomalari
func SendWebhooks(ctx context.Context, webhooks []Webhook) error {
    jobs := make(chan Job, len(webhooks))

    go func() {
        defer close(jobs)
        for _, webhook := range webhooks {
            w := webhook
            jobs <- func(ctx context.Context) error {
                return sendWebhook(ctx, w)
            }
        }
    }()

    // Webhook limitlariga rioya qilish
    throttled := ThrottleJobSubmission(ctx, jobs, 200*time.Millisecond)
    return RunPool(ctx, throttled, 3)
}

// Nazokat kechikishi bilan veb-crawling
func CrawlWebsite(ctx context.Context, urls []string) error {
    jobs := make(chan Job, len(urls))

    go func() {
        defer close(jobs)
        for _, url := range urls {
            u := url
            jobs <- func(ctx context.Context) error {
                return crawlPage(ctx, u)
            }
        }
    }()

    // Nazokat kechikishi: so'rovlar orasida 1 soniya
    throttled := ThrottleJobSubmission(ctx, jobs, time.Second)
    return RunPool(ctx, throttled, 1)
}

// Ko'p darajali tezlikni cheklash
func ProcessWithTieredRates(ctx context.Context, items []Item) error {
    priorityJobs := make(chan Job, 100)
    normalJobs := make(chan Job, 100)

    // Ustuvorlik bo'yicha ajratish
    go func() {
        defer close(priorityJobs)
        defer close(normalJobs)

        for _, item := range items {
            i := item
            job := func(ctx context.Context) error {
                return processItem(ctx, i)
            }

            if i.Priority == "high" {
                priorityJobs <- job
            } else {
                normalJobs <- job
            }
        }
    }()

    var wg sync.WaitGroup
    var firstErr error
    var once sync.Once

    // Yuqori ustuvorlik: tezroq
    wg.Add(1)
    go func() {
        defer wg.Done()
        throttled := ThrottleJobSubmission(ctx, priorityJobs, 50*time.Millisecond)
        if err := RunPool(ctx, throttled, 5); err != nil {
            once.Do(func() { firstErr = err })
        }
    }()

    // Oddiy ustuvorlik: sekinroq
    wg.Add(1)
    go func() {
        defer wg.Done()
        throttled := ThrottleJobSubmission(ctx, normalJobs, 200*time.Millisecond)
        if err := RunPool(ctx, throttled, 2); err != nil {
            once.Do(func() { firstErr = err })
        }
    }()

    wg.Wait()
    return firstErr
}

// Dinamik tezlik sozlash
func ProcessWithDynamicRate(ctx context.Context, jobs <-chan Job) error {
    var currentInterval = 100 * time.Millisecond
    adjustedJobs := make(chan Job)

    go func() {
        defer close(adjustedJobs)
        for job := range jobs {
            // Tizim yuklamisiga qarab tezlikni sozlash
            load := getSystemLoad()
            if load > 0.8 {
                currentInterval = 200 * time.Millisecond  // Sekinlashtirish
            } else if load < 0.3 {
                currentInterval = 50 * time.Millisecond  // Tezlashtirish
            }

            throttled := ThrottleJobSubmission(ctx, makeSingleJobChan(job), currentInterval)
            for j := range throttled {
                adjustedJobs <- j
            }
        }
    }()

    return RunPool(ctx, adjustedJobs, 5)
}
\`\`\`

**Amaliy afzalliklar:**
- **Talablarga muvofiqlik:** Tezlik limitleri va kvotalariga rioya qilish
- **Barqarorlik:** Downstream xizmatlarning ortiqcha yuklanishini oldini olish
- **Xarajatlarni tejash:** Limitdan oshish to'lovlaridan qochish
- **Yaxshi fuqarolik:** Boshqalarning APIlaridan nazokat bilan foydalanish
- **Bashorat qilinadigan:** Boshqariladigan va barqaror yuk

**Odatiy tezlik limitleri:**
- **Twitter API:** 15 daqiqada 300 so'rov
- **GitHub API:** Soatiga 5000 so'rov
- **Google APIlar:** Turlicha (masalan, soniyada 100 so'rov)
- **SendGrid:** Rejaga bog'liq
- **Stripe:** Soniyada 100 so'rov

**Intervalni hisoblash:**
\`\`\`go
// Tezlik limiti: T vaqt davri uchun R so'rovlar
interval = T / R

// Misollar:
// 100 so'rov/daqiqa: interval = 60s / 100 = 600ms
// 10 so'rov/soniya: interval = 1s / 10 = 100ms
// 5000 so'rov/soat: interval = 3600s / 5000 = 720ms
\`\`\`

**Timer tozalash patterni:**
Timerlarni to'g'ri tozalash juda muhim:
\`\`\`go
timer := time.NewTimer(wait)
select {
case <-ctx.Done():
    if !timer.Stop() {  // Timerni to'xtatishga urinish
        <-timer.C       // Agar allaqachon ishga tushgan bo'lsa tozalash
    }
    return
case <-timer.C:
    // Timer normal ishga tushdi
}
\`\`\`

Bu goroutine oqishlarini oldini oladi va to'g'ri resurs boshqaruvini ta'minlaydi.

**Qachon ishlatish:**
- Tezlik limitleri bilan tashqi APIlarni chaqirish
- DBni ortiqcha yuklashi mumkin bo'lgan operatsiyalar
- Provayderning limitleri bilan email/SMS yuborish
- Nazokat kechikishlari bilan veb-crawling
- Throughput cheklovlari bor har qanday operatsiyalar

**Quvurdagi pozitsiyasi:**
ThrottleJobSubmission vazifalarni ishlab chiqarish va iste'mol qilish o'rtasida joylashadi:
\`\`\`
Ishlab chiqaruvchi -> ThrottleJobSubmission -> Worker puli
\`\`\`

Bu pattern rate-limited xizmatlar bilan integratsiyalashuv ishlab chiqarish tizimlarini qurish uchun zarur.`,
			solutionCode: `package concurrency

import (
	"context"
	"time"
)

type Job func(context.Context) error

func ThrottleJobSubmission(ctx context.Context, in <-chan Job, interval time.Duration) <-chan Job {
	out := make(chan Job)                                           // Chiqish kanalini yaratish
	go func() {                                                     // Throttler goroutinasini ishga tushirish
		defer close(out)                                        // Chiqish kanalini doim yopish
		if ctx == nil {                                         // nil kontekstni ishlash
			for job := range in {                           // Barcha vazifalarni yo'naltirish
				out <- job                              // Throttle qilmasdan
			}
			return                                          // Tugadi
		}
		if interval <= 0 {                                      // Musbat bo'lmagan intervalni ishlash
			for {                                           // Yo'naltirish tsikli
				select {                                // Kontekstni tekshirish yoki qabul qilish
				case <-ctx.Done():                      // Kontekst bekor qilindi
					return                          // Chiqish
				case job, ok := <-in:                   // Vazifani qabul qilish
					if !ok {                        // Kirish kanali yopildi
						return                  // Chiqish
					}
					select {                        // Yuborish yoki bekor qilish
					case <-ctx.Done():              // Kontekst bekor qilindi
						return                  // Chiqish
					case out <- job:                // Vazifani darhol yo'naltirish
					}
				}
			}
		}
		var last time.Time                                      // Oxirgi emissiya vaqtini kuzatish
		for {                                                   // Throttle qilish tsikli
			select {                                        // Kontekstni tekshirish yoki qabul qilish
			case <-ctx.Done():                              // Kontekst bekor qilindi
				return                                  // Chiqish
			case job, ok := <-in:                           // Vazifani qabul qilish
				if !ok {                                // Kirish kanali yopildi
					return                          // Chiqish
				}
				if !last.IsZero() {                     // Birinchi vazifa emas
					wait := interval - time.Since(last) // Kutish vaqtini hisoblash
					if wait > 0 {                   // Kutish kerak
						timer := time.NewTimer(wait) // Timer yaratish
						select {                // Kutish yoki bekor qilish
						case <-ctx.Done():      // Kontekst bekor qilindi
							if !timer.Stop() { // Timerni to'xtatishga urinish
								<-timer.C // Agar allaqachon ishga tushgan bo'lsa tozalash
							}
							return          // Chiqish
						case <-timer.C:         // Timer ishga tushdi
						}
					}
				}
				select {                                // Yuborish yoki bekor qilish
				case <-ctx.Done():                      // Kontekst bekor qilindi
					return                          // Chiqish
				case out <- job:                        // Vazifani yuborish
					last = time.Now()               // Oxirgi emissiya vaqtini yangilash
				}
			}
		}
	}()
	return out                                                      // Faqat o'qish uchun kanalni qaytarish
}`
		}
	}
};

export default task;
