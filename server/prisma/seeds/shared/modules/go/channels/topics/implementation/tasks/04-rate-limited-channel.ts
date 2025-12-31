import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-channels-rate-limiter',
	title: 'Rate-Limited Channel Pattern',
	difficulty: 'hard',
	tags: ['go', 'channels', 'concurrency', 'rate-limiting', 'throttling'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a rate limiter that throttles channel reads using time.Ticker to control processing rate.

**Requirements:**
1. **RateLimitedChannel**: Wrap input channel with rate limiting
2. **Configurable Rate**: Control items per second using time.Ticker
3. **Burst Support**: Allow burst processing with token bucket algorithm
4. **Context Awareness**: Stop on context cancellation
5. **Clean Shutdown**: Close output when input is closed

**Rate Limiter Pattern:**
\`\`\`go
func RateLimitedChannel[T any](
    ctx context.Context,
    in <-chan T,
    ratePerSecond int,
    burst int,
) <-chan T {
    // Create ticker for rate limiting
    // Use token bucket algorithm for burst
    // Forward items at controlled rate
    // Stop on context cancellation
}
\`\`\`

**Token Bucket Algorithm:**
\`\`\`
Bucket capacity = burst
Refill rate = ratePerSecond tokens/second

Every 1/ratePerSecond:
  - Add 1 token to bucket (max: burst)
  - If token available:
      - Consume token
      - Forward item from input to output
  - If no token:
      - Wait for next refill
\`\`\`

**Example Usage:**
\`\`\`go
// Rate limit API requests
func RateLimitAPI(requests <-chan Request) <-chan Request {
    ctx := context.Background()

    // 10 requests per second, burst of 20
    return RateLimitedChannel(ctx, requests, 10, 20)
}

// Process with rate limiting
rateLimited := RateLimitAPI(allRequests)
for req := range rateLimited {
    makeAPICall(req)  // Max 10 calls/sec, burst up to 20
}

// Database write throttling
func ThrottleWrites(records <-chan Record) <-chan Record {
    ctx := context.Background()

    // 100 writes per second, burst of 200
    return RateLimitedChannel(ctx, records, 100, 200)
}
\`\`\`

**Pattern Flow:**
\`\`\`
Input Channel → [Rate Limiter] → Output Channel
                     ↓
                [Token Bucket]
                     ↓
                [time.Ticker]

Fast Input:   ████████████████
Rate Limited: ██  ██  ██  ██  (controlled rate)
\`\`\`

**Real-World Scenarios:**

**1. API Rate Limiting:**
\`\`\`go
// Respect external API limits (100 req/min)
func ProcessUsers(users <-chan User) error {
    ctx := context.Background()

    // Convert to 1.67 req/sec with burst of 5
    rateLimited := RateLimitedChannel(ctx, users, 2, 5)

    for user := range rateLimited {
        if err := externalAPI.UpdateUser(user); err != nil {
            return err
        }
    }
    return nil
}
\`\`\`

**2. Database Connection Throttling:**
\`\`\`go
// Prevent overwhelming database
func BulkInsert(records <-chan Record) error {
    ctx := context.Background()

    // 50 inserts per second, burst of 100
    throttled := RateLimitedChannel(ctx, records, 50, 100)

    for rec := range throttled {
        if err := db.Insert(rec); err != nil {
            return err
        }
    }
    return nil
}
\`\`\`

**3. Log Processing:**
\`\`\`go
// Send logs to remote service (rate limited)
func ForwardLogs(logs <-chan LogEntry) error {
    ctx := context.Background()

    // 1000 logs per second, burst of 2000
    rateLimited := RateLimitedChannel(ctx, logs, 1000, 2000)

    for log := range rateLimited {
        if err := remoteSvc.SendLog(log); err != nil {
            return err
        }
    }
    return nil
}
\`\`\`

**Constraints:**
- Must use time.Ticker for rate control
- Must implement token bucket algorithm for burst
- Must respect context cancellation
- Must not leak goroutines
- Must close output channel when input closes
- Must handle ratePerSecond <= 0 (unlimited rate)
- Must handle burst <= 0 (no burst, only steady rate)`,
	initialCode: `package channelsx

import (
	"context"
	"time"
)

// TODO: Implement RateLimitedChannel
// Use time.Ticker to control rate
// Implement token bucket algorithm for burst support
// Forward items from input at controlled rate
// Stop on context cancellation
func RateLimitedChannel[T any](ctx context.Context, in <-chan T, ratePerSecond int, burst int) <-chan T {
	// TODO: Implement
}`,
	solutionCode: `package channelsx

import (
	"context"
	"time"
)

func RateLimitedChannel[T any](ctx context.Context, in <-chan T, ratePerSecond int, burst int) <-chan T {
	if ctx == nil {
		ctx = context.Background()
	}

	out := make(chan T)

	// handle unlimited rate
	if ratePerSecond <= 0 {
		go func() {
			defer close(out)
			for {
				select {
				case <-ctx.Done():
					return
				case v, ok := <-in:
					if !ok {
						return
					}
					select {
					case <-ctx.Done():
						return
					case out <- v:
					}
				}
			}
		}()
		return out
	}

	// ensure burst is at least 1
	if burst <= 0 {
		burst = 1
	}

	go func() {
		defer close(out)

		// calculate interval between tokens
		interval := time.Second / time.Duration(ratePerSecond)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		// token bucket with initial burst capacity
		tokens := burst

		for {
			select {
			case <-ctx.Done():                      // context canceled
				return

			case <-ticker.C:                        // refill token
				if tokens < burst {
					tokens++                         // add token up to burst limit
				}

			case v, ok := <-in:
				if !ok {                             // input channel closed
					return
				}

				// wait until token available
				for tokens <= 0 {
					select {
					case <-ctx.Done():
						return
					case <-ticker.C:                 // wait for token refill
						tokens++
					}
				}

				// consume token and forward item
				tokens--
				select {
				case <-ctx.Done():
					return
				case out <- v:                       // forward at controlled rate
				}
			}
		}
	}()

	return out                                      // return rate-limited output channel
}`,
		testCode: `package channelsx

import (
	"context"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// RateLimitedChannel returns a channel
	ctx := context.Background()
	input := make(chan int)
	close(input)

	output := RateLimitedChannel(ctx, input, 10, 5)
	if output == nil {
		t.Error("expected non-nil channel")
	}

	for range output {
	}
}

func Test2(t *testing.T) {
	// RateLimitedChannel forwards items
	ctx := context.Background()
	input := make(chan int, 3)
	input <- 1
	input <- 2
	input <- 3
	close(input)

	output := RateLimitedChannel(ctx, input, 1000, 10)

	var results []int
	for v := range output {
		results = append(results, v)
	}

	if len(results) != 3 {
		t.Errorf("expected 3 items, got %d", len(results))
	}
}

func Test3(t *testing.T) {
	// RateLimitedChannel closes output when input closes
	ctx := context.Background()
	input := make(chan int)
	close(input)

	output := RateLimitedChannel(ctx, input, 10, 5)

	select {
	case _, ok := <-output:
		if ok {
			t.Error("expected closed channel")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("channel not closed in time")
	}
}

func Test4(t *testing.T) {
	// RateLimitedChannel stops on context cancellation
	ctx, cancel := context.WithCancel(context.Background())
	input := make(chan int)

	output := RateLimitedChannel(ctx, input, 10, 5)

	cancel()

	select {
	case <-output:
	case <-time.After(100 * time.Millisecond):
		t.Error("channel not closed after context cancel")
	}
}

func Test5(t *testing.T) {
	// RateLimitedChannel handles nil context
	input := make(chan int)
	close(input)

	output := RateLimitedChannel(nil, input, 10, 5)
	if output == nil {
		t.Error("expected non-nil channel")
	}

	for range output {
	}
}

func Test6(t *testing.T) {
	// RateLimitedChannel with unlimited rate (ratePerSecond <= 0)
	ctx := context.Background()
	input := make(chan int, 5)
	for i := 0; i < 5; i++ {
		input <- i
	}
	close(input)

	output := RateLimitedChannel(ctx, input, 0, 5)

	count := 0
	for range output {
		count++
	}

	if count != 5 {
		t.Errorf("expected 5 items, got %d", count)
	}
}

func Test7(t *testing.T) {
	// RateLimitedChannel with burst <= 0 defaults to 1
	ctx := context.Background()
	input := make(chan int, 2)
	input <- 1
	input <- 2
	close(input)

	output := RateLimitedChannel(ctx, input, 100, 0)

	count := 0
	for range output {
		count++
	}

	if count != 2 {
		t.Errorf("expected 2 items, got %d", count)
	}
}

func Test8(t *testing.T) {
	// RateLimitedChannel preserves item order
	ctx := context.Background()
	input := make(chan int, 5)
	for i := 0; i < 5; i++ {
		input <- i
	}
	close(input)

	output := RateLimitedChannel(ctx, input, 1000, 10)

	expected := 0
	for v := range output {
		if v != expected {
			t.Errorf("expected %d, got %d", expected, v)
		}
		expected++
	}
}

func Test9(t *testing.T) {
	// RateLimitedChannel with burst allows initial burst
	ctx := context.Background()
	input := make(chan int, 10)
	for i := 0; i < 10; i++ {
		input <- i
	}
	close(input)

	start := time.Now()
	output := RateLimitedChannel(ctx, input, 1, 10)

	count := 0
	for range output {
		count++
		if count == 5 {
			break
		}
	}

	elapsed := time.Since(start)
	if elapsed > 500*time.Millisecond {
		t.Errorf("initial burst should be fast, took %v", elapsed)
	}
}

func Test10(t *testing.T) {
	// RateLimitedChannel works with different types
	ctx := context.Background()
	input := make(chan string, 2)
	input <- "hello"
	input <- "world"
	close(input)

	output := RateLimitedChannel(ctx, input, 1000, 10)

	var results []string
	for v := range output {
		results = append(results, v)
	}

	if len(results) != 2 {
		t.Errorf("expected 2 items, got %d", len(results))
	}
}
`,
	hint1: `Use time.NewTicker(time.Second / time.Duration(ratePerSecond)) to create a ticker that fires at the desired rate. Use a token counter starting at burst value.`,
	hint2: `In a select statement, handle three cases: ctx.Done(), ticker.C (add token if below burst), and input channel (wait for token, consume it, forward item).`,
	whyItMatters: `Rate limiting is essential for protecting services from overload and respecting external API limits in production systems.

**Why This Matters:**

**1. API Rate Limit Compliance**
Most external APIs have rate limits. Exceeding them causes errors and bans:
\`\`\`go
// Problem: Hitting Stripe API too fast
func ProcessPayments(payments <-chan Payment) error {
    for payment := range payments {
        // Stripe allows 100 req/sec
        resp, err := stripe.CreateCharge(payment)
        if err != nil {
            // ERROR 429: Too Many Requests
            // Your IP is banned for 1 hour!
            return err
        }
    }
}

// Solution: Rate limiting
func ProcessPayments(payments <-chan Payment) error {
    ctx := context.Background()

    // Respect Stripe's 100 req/sec limit with burst of 25
    rateLimited := RateLimitedChannel(ctx, payments, 100, 25)

    for payment := range rateLimited {
        // Guaranteed to stay under rate limit
        resp, err := stripe.CreateCharge(payment)
        if err != nil {
            return err
        }
    }
    return nil
}

// Results:
// - Before: Processing 1000 payments → IP banned after 10 seconds
// - After: Processing 1000 payments → Completes in 10 seconds, no bans
// - Business impact: Zero downtime, 100% payment success rate
\`\`\`

**2. Real Production: SaaS Platform**
Multi-tenant SaaS with 1000 customers calling external APIs:
\`\`\`go
type APIRequest struct {
    CustomerID string
    Endpoint   string
    Payload    interface{}
}

func APIGateway(requests <-chan APIRequest) error {
    ctx := context.Background()

    // External service allows 500 req/sec total
    // Use 450 req/sec with burst of 100 for safety margin
    rateLimited := RateLimitedChannel(ctx, requests, 450, 100)

    for req := range rateLimited {
        resp, err := externalAPI.Call(req.Endpoint, req.Payload)
        if err != nil {
            log.Printf("API error for customer %s: %v", req.CustomerID, err)
            continue
        }

        // Store response
        saveResponse(req.CustomerID, resp)
    }

    return nil
}

// Impact:
// - Before: No rate limiting
//   - Random 429 errors during peak hours
//   - 15% of requests failed
//   - Angry customers, lost revenue
//
// - After: Rate limiting implemented
//   - Zero 429 errors
//   - 100% success rate
//   - Predictable performance
//   - Customer satisfaction: 75% → 98%
//   - Retained $200K annual revenue
\`\`\`

**3. Database Connection Pool Protection**
Preventing database overload:
\`\`\`go
// Problem: Thundering herd on database
func ImportRecords(records <-chan Record) error {
    for rec := range records {
        // 1000 records arrive at once
        // Database has only 20 connections
        // 980 requests wait, timeout, retry → cascade failure
        if err := db.Insert(rec); err != nil {
            return err
        }
    }
}

// Solution: Rate limiting to match database capacity
func ImportRecords(records <-chan Record) error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
    defer cancel()

    // Database can handle 100 inserts/sec comfortably
    // Allow burst of 50 for bursty traffic
    rateLimited := RateLimitedChannel(ctx, records, 100, 50)

    for rec := range rateLimited {
        if err := db.Insert(rec); err != nil {
            return err
        }
    }
    return nil
}

// Results:
// - Before: Database CPU: 95%, connection timeouts, cascading failures
// - After: Database CPU: 60%, zero timeouts, stable
// - Import reliability: 70% → 99.9%
// - Reduced database costs by 40% (no need for over-provisioning)
\`\`\`

**4. Email Service Protection**
Respecting SendGrid's rate limits:
\`\`\`go
type EmailJob struct {
    To      string
    Subject string
    Body    string
}

func SendEmails(jobs <-chan EmailJob) error {
    ctx := context.Background()

    // SendGrid free tier: 100 emails/day = 0.0012 emails/sec
    // Convert to 1 email per 15 seconds
    // But allow burst of 10 for responsive user experience
    rateLimited := RateLimitedChannel(ctx, jobs, 1, 10)

    for job := range rateLimited {
        if err := sendgrid.Send(job.To, job.Subject, job.Body); err != nil {
            log.Printf("Failed to send email to %s: %v", job.To, err)
            continue
        }
    }
    return nil
}

// Impact:
// - Before: Hit daily limit in first hour, rest of day broken
// - After: Emails spread throughout day, never hit limit
// - Upgrade to paid tier delayed by 6 months: $3000 saved
\`\`\`

**5. Webhook Delivery**
Delivering webhooks to customer endpoints:
\`\`\`go
type Webhook struct {
    URL     string
    Payload interface{}
    Retries int
}

func DeliverWebhooks(webhooks <-chan Webhook) error {
    ctx := context.Background()

    // Be nice to customer servers
    // 10 webhooks per second with burst of 20
    rateLimited := RateLimitedChannel(ctx, webhooks, 10, 20)

    for webhook := range rateLimited {
        if err := httpPost(webhook.URL, webhook.Payload); err != nil {
            log.Printf("Webhook delivery failed to %s: %v", webhook.URL, err)

            // Retry with exponential backoff
            if webhook.Retries < 3 {
                webhook.Retries++
                requeue(webhook)
            }
        }
    }
    return nil
}

// Customer feedback:
// - Before: "Your webhooks crashed our server!" (angry customer, lost deal)
// - After: "Webhooks are working great!" (happy customer, $50K/year contract)
\`\`\`

**6. Token Bucket vs Fixed Window**
\`\`\`go
// WRONG: Fixed window (allows bursts at window boundaries)
func FixedWindowRateLimiter(in <-chan Request) <-chan Request {
    out := make(chan Request)
    go func() {
        defer close(out)

        count := 0
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()

        for {
            select {
            case <-ticker.C:
                count = 0  // Reset every second

            case req := <-in:
                count++
                if count <= 10 {
                    out <- req
                }
                // BUG: Can get 20 requests in 1.1s (10 at 0.9s, 10 at 1.1s)
            }
        }
    }()
    return out
}

// RIGHT: Token bucket (smooth rate limiting)
func TokenBucketRateLimiter(in <-chan Request) <-chan Request {
    // Our implementation above
    // Smooth rate limiting with burst support
    // No spikes at window boundaries
}
\`\`\`

**7. Burst Capacity Benefits**
\`\`\`go
// No burst: Poor user experience
rateLimited := RateLimitedChannel(ctx, requests, 10, 1)
// User uploads 5 files → takes 5 seconds (1 per sec)
// User frustrated by delay

// With burst: Good user experience
rateLimited := RateLimitedChannel(ctx, requests, 10, 20)
// User uploads 5 files → completes instantly (burst of 20)
// Files then process at 10/sec
// User happy with responsive UI
\`\`\`

**8. Dynamic Rate Adjustment**
\`\`\`go
// Adjust rate based on system load
func AdaptiveRateLimiting(requests <-chan Request) {
    ctx := context.Background()

    // Monitor system metrics
    currentLoad := getSystemLoad()

    rate := 100  // default
    if currentLoad > 0.8 {
        rate = 50  // reduce rate when system is stressed
    } else if currentLoad < 0.3 {
        rate = 200  // increase rate when system has capacity
    }

    rateLimited := RateLimitedChannel(ctx, requests, rate, rate/2)

    // Process at adaptive rate
    for req := range rateLimited {
        processRequest(req)
    }
}
\`\`\`

**9. Testing Rate Limiter**
\`\`\`go
func TestRateLimiter(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()

    // Create test input
    input := make(chan int, 100)
    for i := 0; i < 100; i++ {
        input <- i
    }
    close(input)

    // Rate limit to 10/sec
    start := time.Now()
    rateLimited := RateLimitedChannel(ctx, input, 10, 1)

    count := 0
    for range rateLimited {
        count++
    }
    elapsed := time.Since(start)

    // Should take ~10 seconds for 100 items at 10/sec
    if elapsed < 9*time.Second || elapsed > 11*time.Second {
        t.Errorf("Expected ~10s, got %v", elapsed)
    }

    if count != 100 {
        t.Errorf("Expected 100 items, got %d", count)
    }
}
\`\`\`

**Real-World Impact:**

**Case Study: Event Processing Platform**
Processing 1M events/hour from IoT devices:

**Before Rate Limiting:**
- Events flood downstream services
- Database connections exhausted
- Cascading failures across microservices
- 30% of events lost
- 3-4 outages per week
- On-call engineer burnout
- Angry customers threatening to leave

**After Rate Limiting:**
- Smooth 280 events/sec (1M/hour) with burst of 500
- Database connections under control
- Zero cascading failures
- 99.99% event delivery
- Zero outages in 6 months
- Happy on-call engineers
- Customer retention: 95% → 99%
- Revenue protected: $2M/year

**Cost Savings:**
- Reduced database capacity by 50%: $60K/year
- Eliminated need for over-provisioned infrastructure: $40K/year
- Reduced incident response time: 200 hours/year saved
- Total savings: $100K/year

**Production Best Practices:**
1. Always add burst capacity for responsive UX
2. Set rate to 90% of API limit (safety margin)
3. Monitor rate limiter metrics (tokens available, items throttled)
4. Add observability for debugging (log when throttling occurs)
5. Use context for graceful shutdown
6. Test with realistic traffic patterns
7. Document rate limits in service SLA
8. Add circuit breaker for upstream protection

Rate limiting is not just about compliance—it's about building reliable, predictable systems that protect both your services and your customers.`,
	order: 3,
	translations: {
		ru: {
			title: 'Канал с ограничением скорости',
			solutionCode: `package channelsx

import (
	"context"
	"time"
)

func RateLimitedChannel[T any](ctx context.Context, in <-chan T, ratePerSecond int, burst int) <-chan T {
	if ctx == nil {
		ctx = context.Background()
	}

	out := make(chan T)

	// обработка неограниченной скорости
	if ratePerSecond <= 0 {
		go func() {
			defer close(out)
			for {
				select {
				case <-ctx.Done():
					return
				case v, ok := <-in:
					if !ok {
						return
					}
					select {
					case <-ctx.Done():
						return
					case out <- v:
					}
				}
			}
		}()
		return out
	}

	// убедиться что burst минимум 1
	if burst <= 0 {
		burst = 1
	}

	go func() {
		defer close(out)

		// вычислить интервал между токенами
		interval := time.Second / time.Duration(ratePerSecond)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		// bucket токенов с начальной burst ёмкостью
		tokens := burst

		for {
			select {
			case <-ctx.Done():                      // контекст отменён
				return

			case <-ticker.C:                        // пополнить токен
				if tokens < burst {
					tokens++                         // добавить токен до burst лимита
				}

			case v, ok := <-in:
				if !ok {                             // входной канал закрыт
					return
				}

				// ждать пока токен доступен
				for tokens <= 0 {
					select {
					case <-ctx.Done():
						return
					case <-ticker.C:                 // ждать пополнения токена
						tokens++
					}
				}

				// потребить токен и переслать элемент
				tokens--
				select {
				case <-ctx.Done():
					return
				case out <- v:                       // переслать с контролируемой скоростью
				}
			}
		}
	}()

	return out                                      // вернуть выход с ограниченной скоростью
}`,
			description: `Реализуйте rate limiter который ограничивает чтение из канала используя time.Ticker для контроля скорости обработки.

**Требования:**
1. **RateLimitedChannel**: Обернуть входной канал с ограничением скорости
2. **Configurable Rate**: Контролировать элементы в секунду используя time.Ticker
3. **Burst Support**: Разрешить burst обработку с алгоритмом token bucket
4. **Context Awareness**: Остановка при отмене контекста
5. **Clean Shutdown**: Закрыть выход когда вход закрыт

**Rate Limiter паттерн:**
\`\`\`go
func RateLimitedChannel[T any](
    ctx context.Context,
    in <-chan T,
    ratePerSecond int,
    burst int,
) <-chan T {
    // Создать ticker для ограничения скорости
    // Использовать token bucket алгоритм для burst
    // Пересылать элементы с контролируемой скоростью
    // Остановка при отмене контекста
}
\`\`\`

**Token Bucket алгоритм:**
\`\`\`
Ёмкость bucket = burst
Скорость пополнения = ratePerSecond токенов/секунду

Каждые 1/ratePerSecond:
  - Добавить 1 токен в bucket (макс: burst)
  - Если токен доступен:
      - Потребить токен
      - Переслать элемент из входа в выход
  - Если нет токена:
      - Ждать следующего пополнения
\`\`\`

**Поток паттерна:**
\`\`\`
Input Channel → [Rate Limiter] → Output Channel
                     ↓
                [Token Bucket]
                     ↓
                [time.Ticker]

Быстрый вход:   ████████████████
С ограничением: ██  ██  ██  ██  (контролируемая скорость)
\`\`\`

**Пример использования:**
\`\`\`go
// Ограничение скорости API запросов
func RateLimitAPI(requests <-chan Request) <-chan Request {
    ctx := context.Background()

    // 10 запросов в секунду, burst 20
    return RateLimitedChannel(ctx, requests, 10, 20)
}

// Обработка с ограничением скорости
rateLimited := RateLimitAPI(allRequests)
for req := range rateLimited {
    makeAPICall(req)  // Макс 10 вызовов/сек, burst до 20
}
\`\`\`

**Ограничения:**
- Должен использовать time.Ticker для контроля скорости
- Должен реализовывать token bucket алгоритм для burst
- Должен уважать отмену контекста
- Не должен утекать горутины
- Должен закрывать выходной канал когда входной закрывается
- Должен обрабатывать ratePerSecond <= 0 (неограниченная скорость)
- Должен обрабатывать burst <= 0 (без burst, только стабильная скорость)`,
			hint1: `Используйте time.NewTicker(time.Second / time.Duration(ratePerSecond)) чтобы создать ticker который срабатывает с нужной скоростью. Используйте счётчик токенов начинающийся со значения burst.`,
			hint2: `В select statement обрабатывайте три случая: ctx.Done(), ticker.C (добавить токен если ниже burst), и входной канал (ждать токен, потребить его, переслать элемент).`,
			whyItMatters: `Rate limiting является критически важным продакшен паттерном для защиты сервисов от перегрузки и соблюдения лимитов внешних API в production системах.

**Почему это важно:**

**1. Соблюдение API лимитов**
Большинство внешних API имеют rate limits. Их превышение вызывает ошибки и баны:
\`\`\`go
// Проблема: Слишком быстрые запросы к Stripe API
func ProcessPayments(payments <-chan Payment) error {
    for payment := range payments {
        // Stripe разрешает 100 req/sec
        resp, err := stripe.CreateCharge(payment)
        if err != nil {
            // ERROR 429: Too Many Requests
            // Ваш IP забанен на 1 час!
            return err
        }
    }
}

// Решение: Rate limiting
func ProcessPayments(payments <-chan Payment) error {
    ctx := context.Background()

    // Соблюдаем лимит Stripe 100 req/sec с burst 25
    rateLimited := RateLimitedChannel(ctx, payments, 100, 25)

    for payment := range rateLimited {
        // Гарантированно не превышаем rate limit
        resp, err := stripe.CreateCharge(payment)
        if err != nil {
            return err
        }
    }
    return nil
}

// Результаты:
// - До: Обработка 1000 платежей → IP забанен через 10 секунд
// - После: Обработка 1000 платежей → Завершается за 10 секунд, без банов
// - Бизнес эффект: Нулевое время простоя, 100% успешных платежей
\`\`\`

**2. Реальный Production сценарий: SaaS платформа**
Multi-tenant SaaS с 1000 клиентами вызывающими внешние API:
\`\`\`go
func APIGateway(requests <-chan APIRequest) error {
    ctx := context.Background()

    // Внешний сервис разрешает 500 req/sec всего
    // Используем 450 req/sec с burst 100 для запаса безопасности
    rateLimited := RateLimitedChannel(ctx, requests, 450, 100)

    for req := range rateLimited {
        resp, err := externalAPI.Call(req.Endpoint, req.Payload)
        if err != nil {
            log.Printf("API ошибка для клиента %s: %v", req.CustomerID, err)
            continue
        }

        saveResponse(req.CustomerID, resp)
    }

    return nil
}

// Эффект:
// - До: Без rate limiting
//   - Случайные 429 ошибки в пиковые часы
//   - 15% запросов падало
//   - Злые клиенты, потерянная выручка
//
// - После: Реализован rate limiting
//   - Ноль 429 ошибок
//   - 100% успешности
//   - Предсказуемая производительность
//   - Удовлетворённость клиентов: 75% → 98%
//   - Сохранено $200K годовой выручки
\`\`\`

**3. Защита пула соединений с базой данных**
Предотвращение перегрузки базы данных:
\`\`\`go
// Проблема: Thundering herd на базе данных
func ImportRecords(records <-chan Record) error {
    for rec := range records {
        // 1000 записей приходят одновременно
        // База имеет только 20 соединений
        // 980 запросов ждут, timeout, retry → каскадный сбой
        if err := db.Insert(rec); err != nil {
            return err
        }
    }
}

// Решение: Rate limiting под ёмкость базы
func ImportRecords(records <-chan Record) error {
    ctx := context.Background()

    // База комфортно обрабатывает 100 вставок/сек
    // Разрешаем burst 50 для bursty трафика
    rateLimited := RateLimitedChannel(ctx, records, 100, 50)

    for rec := range rateLimited {
        if err := db.Insert(rec); err != nil {
            return err
        }
    }
    return nil
}

// Результаты:
// - До: CPU базы: 95%, timeoutы соединений, каскадные сбои
// - После: CPU базы: 60%, ноль timeoutов, стабильно
// - Надёжность импорта: 70% → 99.9%
// - Сокращены затраты на базу на 40% (нет нужды в over-provisioning)
\`\`\`

**4. Защита email сервиса**
Соблюдение лимитов SendGrid в стартапе:
\`\`\`go
func SendEmails(jobs <-chan EmailJob) error {
    ctx := context.Background()

    // SendGrid free tier: 100 emails/день = 0.0012 emails/сек
    // Конвертируем в 1 email за 15 секунд
    // Но разрешаем burst 10 для отзывчивого UX
    rateLimited := RateLimitedChannel(ctx, jobs, 1, 10)

    for job := range rateLimited {
        if err := sendgrid.Send(job.To, job.Subject, job.Body); err != nil {
            log.Printf("Не удалось отправить email на %s: %v", job.To, err)
            continue
        }
    }
    return nil
}

// Эффект:
// - До: Достигли дневного лимита за первый час, остаток дня сломан
// - После: Emailы распределены по дню, никогда не достигаем лимита
// - Переход на платный тариф отложен на 6 месяцев: сэкономлено $3000
\`\`\`

**5. Webhook доставка**
Доставка webhook к серверам клиентов без перегрузки:
\`\`\`go
type Webhook struct {
    URL     string
    Payload interface{}
    Retries int
}

func DeliverWebhooks(webhooks <-chan Webhook) error {
    ctx := context.Background()

    // Быть вежливым к серверам клиентов
    // 10 webhooks в секунду с burst 20
    rateLimited := RateLimitedChannel(ctx, webhooks, 10, 20)

    for webhook := range rateLimited {
        if err := httpPost(webhook.URL, webhook.Payload); err != nil {
            log.Printf("Webhook доставка провалилась на %s: %v", webhook.URL, err)

            // Retry с экспоненциальным backoff
            if webhook.Retries < 3 {
                webhook.Retries++
                requeue(webhook)
            }
        }
    }
    return nil
}

// Отзыв клиента:
// - До: "Ваши webhooks роняли наш сервер!" (злой клиент, потеряна сделка)
// - После: "Webhooks работают отлично!" (счастливый клиент, $50K/год контракт)
\`\`\`

**5. Тестирование Rate Limiter**
\`\`\`go
func TestRateLimiter(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()

    // Создание тестового ввода
    input := make(chan int, 100)
    for i := 0; i < 100; i++ {
        input <- i
    }
    close(input)

    // Ограничение до 10/сек
    start := time.Now()
    rateLimited := RateLimitedChannel(ctx, input, 10, 1)

    count := 0
    for range rateLimited {
        count++
    }
    elapsed := time.Since(start)

    // Должно занять ~10 секунд для 100 элементов при 10/сек
    if elapsed < 9*time.Second || elapsed > 11*time.Second {
        t.Errorf("Ожидалось ~10с, получено %v", elapsed)
    }

    if count != 100 {
        t.Errorf("Ожидалось 100 элементов, получено %d", count)
    }
}
\`\`\`

**6. Динамическая адаптация скорости**
\`\`\`go
// Адаптация скорости на основе нагрузки системы
func AdaptiveRateLimiting(requests <-chan Request) {
    ctx := context.Background()

    // Мониторинг метрик системы
    currentLoad := getSystemLoad()

    rate := 100  // по умолчанию
    if currentLoad > 0.8 {
        rate = 50  // снижение при высокой нагрузке
    } else if currentLoad < 0.3 {
        rate = 200  // увеличение при свободных ресурсах
    }

    rateLimited := RateLimitedChannel(ctx, requests, rate, rate/2)

    // Обработка с адаптивной скоростью
    for req := range rateLimited {
        processRequest(req)
    }
}
\`\`\`

**Production Best Practices:**
1. Всегда добавляйте burst ёмкость для отзывчивого UX
2. Устанавливайте rate на 90% от API лимита (запас безопасности)
3. Мониторьте метрики rate limiter (доступные токены, ограниченные элементы)
4. Добавляйте observability для отладки (логируйте когда происходит throttling)
5. Используйте context для graceful shutdown
6. Тестируйте с реалистичными паттернами трафика
7. Документируйте rate limits в service SLA
8. Рассмотрите использование distributed rate limiting для кластеров
9. Реализуйте механизмы backoff для повторных попыток
10. Добавляйте алерты при приближении к лимитам

**Реальное влияние:**
Event processing платформа обрабатывающая 1M событий/час:
- **До**: События заливают downstream сервисы → 30% событий потеряно → 3-4 outage/неделю
- **После**: Плавные 280 событий/сек с burst 500 → 99.99% доставка → ноль outage за 6 месяцев
- **Результат**: Защищена выручка $2M/год, сэкономлено $100K/год на инфраструктуре
- **Удовлетворенность клиентов**: 70% → 98%
- **Время отклика on-call инженеров**: Сокращено на 90%

Rate limiting это не только о соблюдении правил - это о построении надёжных, предсказуемых систем которые защищают как ваши сервисы так и ваших клиентов. Это фундаментальный инструмент для любого production Go приложения.`
		},
		uz: {
			title: `Tezlik cheklangan kanal`,
			solutionCode: `package channelsx

import (
	"context"
	"time"
)

func RateLimitedChannel[T any](ctx context.Context, in <-chan T, ratePerSecond int, burst int) <-chan T {
	if ctx == nil {
		ctx = context.Background()
	}

	out := make(chan T)

	// cheksiz tezlikni qayta ishlash
	if ratePerSecond <= 0 {
		go func() {
			defer close(out)
			for {
				select {
				case <-ctx.Done():
					return
				case v, ok := <-in:
					if !ok {
						return
					}
					select {
					case <-ctx.Done():
						return
					case out <- v:
					}
				}
			}
		}()
		return out
	}

	// burst kamida 1 ekanligiga ishonch hosil qilish
	if burst <= 0 {
		burst = 1
	}

	go func() {
		defer close(out)

		// tokenlar orasidagi intervalni hisoblash
		interval := time.Second / time.Duration(ratePerSecond)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		// boshlang'ich burst sig'imi bilan token bucket
		tokens := burst

		for {
			select {
			case <-ctx.Done():                      // kontekst bekor qilindi
				return

			case <-ticker.C:                        // tokenni to'ldirish
				if tokens < burst {
					tokens++                         // burst limitigacha token qo'shish
				}

			case v, ok := <-in:
				if !ok {                             // kirish kanali yopilgan
					return
				}

				// token mavjud bo'lguncha kutish
				for tokens <= 0 {
					select {
					case <-ctx.Done():
						return
					case <-ticker.C:                 // token to'ldirilishini kutish
						tokens++
					}
				}

				// tokenni iste'mol qilish va elementni yo'naltirish
				tokens--
				select {
				case <-ctx.Done():
					return
				case out <- v:                       // nazorat qilinadigan tezlikda yo'naltirish
				}
			}
		}
	}()

	return out                                      // tezlik cheklangan chiqish kanalini qaytarish
}`,
			description: `Qayta ishlash tezligini nazorat qilish uchun time.Ticker dan foydalangan holda kanaldan o'qishni cheklaydigan rate limiter ni amalga oshiring.

**Talablar:**
1. **RateLimitedChannel**: Kirish kanalini tezlik cheklash bilan o'rash
2. **Configurable Rate**: time.Ticker dan foydalanib soniyasiga elementlarni nazorat qilish
3. **Burst Support**: Token bucket algoritmi bilan burst qayta ishlashga ruxsat berish
4. **Context Awareness**: Kontekst bekor qilinganda to'xtatish
5. **Clean Shutdown**: Kirish yopilganda chiqishni yopish

**Rate Limiter pattern:**
\`\`\`go
func RateLimitedChannel[T any](
    ctx context.Context,
    in <-chan T,
    ratePerSecond int,
    burst int,
) <-chan T {
    // Tezlik cheklash uchun ticker yaratish
    // Burst uchun token bucket algoritmidan foydalanish
    // Elementlarni nazorat qilinadigan tezlikda yo'naltirish
    // Kontekst bekor qilinganda to'xtatish
}
\`\`\`

**Token Bucket algoritmi:**
\`\`\`
Bucket sig'imi = burst
To'ldirish tezligi = ratePerSecond tokenlar/soniya

Har 1/ratePerSecond da:
  - Bucketga 1 token qo'shish (maks: burst)
  - Agar token mavjud bo'lsa:
      - Tokenni iste'mol qilish
      - Elementni kirishdan chiqishga yo'naltirish
  - Agar token bo'lmasa:
      - Keyingi to'ldirilishni kutish
\`\`\`

**Pattern oqimi:**
\`\`\`
Input Channel → [Rate Limiter] → Output Channel
                     ↓
                [Token Bucket]
                     ↓
                [time.Ticker]

Tez kirish:       ████████████████
Tezlik cheklangan: ██  ██  ██  ██  (nazorat qilinadigan tezlik)
\`\`\`

**Foydalanish misoli:**
\`\`\`go
// API so'rovlarini tezlik cheklash
func RateLimitAPI(requests <-chan Request) <-chan Request {
    ctx := context.Background()

    // Soniyasiga 10 ta so'rov, 20 ta burst
    return RateLimitedChannel(ctx, requests, 10, 20)
}

// Tezlik cheklash bilan qayta ishlash
rateLimited := RateLimitAPI(allRequests)
for req := range rateLimited {
    makeAPICall(req)  // Maks 10 ta chaqiruv/soniya, 20 tagacha burst
}
\`\`\`

**Cheklovlar:**
- Tezlikni nazorat qilish uchun time.Ticker dan foydalanishi kerak
- Burst uchun token bucket algoritmini amalga oshirishi kerak
- Kontekst bekor qilishni hurmat qilishi kerak
- Gorutinlarni sizib chiqarmasligi kerak
- Kirish yopilganda chiqish kanalini yopishi kerak
- ratePerSecond <= 0 ni qayta ishlashi kerak (cheksiz tezlik)
- burst <= 0 ni qayta ishlashi kerak (burst yo'q, faqat barqaror tezlik)`,
			hint1: `Kerakli tezlikda ishlaydigan ticker yaratish uchun time.NewTicker(time.Second / time.Duration(ratePerSecond)) dan foydalaning. Burst qiymatidan boshlanadigan token hisoblagichidan foydalaning.`,
			hint2: `Select statementida uchta holatni qayta ishlang: ctx.Done(), ticker.C (burstdan past bo'lsa token qo'shish), va kirish kanali (token kutish, uni iste'mol qilish, elementni yo'naltirish).`,
			whyItMatters: `Rate limiting ishlab chiqarish tizimlarida xizmatlarni ortiqcha yuklashdan himoya qilish va tashqi API limitlarini hurmat qilish uchun muhim production patterni hisoblanadi.

**Nima uchun bu muhim:**

**1. API rate limit ga rioya qilish**
Ko'pchilik tashqi APIlar rate limitlariga ega. Ularni oshirish xatolarga va banlarga olib keladi:
\`\`\`go
// Muammo: Stripe API ga juda tez so'rovlar
func ProcessPayments(payments <-chan Payment) error {
    for payment := range payments {
        // Stripe soniyasiga 100 req ga ruxsat beradi
        resp, err := stripe.CreateCharge(payment)
        if err != nil {
            // ERROR 429: Too Many Requests
            // Sizning IP 1 soatga banlandi!
            return err
        }
    }
}

// Yechim: Rate limiting
func ProcessPayments(payments <-chan Payment) error {
    ctx := context.Background()

    // Stripe ning 100 req/sec limitiga rioya qilamiz, 25 ta burst bilan
    rateLimited := RateLimitedChannel(ctx, payments, 100, 25)

    for payment := range rateLimited {
        // Rate limit oshmasligiga kafolat
        resp, err := stripe.CreateCharge(payment)
        if err != nil {
            return err
        }
    }
    return nil
}

// Natijalar:
// - Oldin: 1000 ta to'lovni qayta ishlash → 10 soniyadan keyin IP banlandi
// - Keyin: 1000 ta to'lovni qayta ishlash → 10 soniyada tugadi, ban yo'q
// - Biznes ta'siri: Nol to'xtash vaqti, 100% muvaffaqiyatli to'lovlar
\`\`\`

**2. Haqiqiy Production stsenariy: SaaS platformasi**
Tashqi APIlarni chaqiradigan 1000 ta mijozli multi-tenant SaaS:
\`\`\`go
func APIGateway(requests <-chan APIRequest) error {
    ctx := context.Background()

    // Tashqi xizmat jami 500 req/sec ga ruxsat beradi
    // Xavfsizlik zahirasi uchun 450 req/sec, 100 ta burst ishlatamiz
    rateLimited := RateLimitedChannel(ctx, requests, 450, 100)

    for req := range rateLimited {
        resp, err := externalAPI.Call(req.Endpoint, req.Payload)
        if err != nil {
            log.Printf("Mijoz %s uchun API xatosi: %v", req.CustomerID, err)
            continue
        }

        saveResponse(req.CustomerID, resp)
    }

    return nil
}

// Ta'sir:
// - Oldin: Rate limiting yo'q
//   - Eng yuqori soatlarda tasodifiy 429 xatolar
//   - So'rovlarning 15% muvaffaqiyatsiz
//   - G'azablangan mijozlar, yo'qotilgan daromad
//
// - Keyin: Rate limiting amalga oshirildi
//   - Nol 429 xatolar
//   - 100% muvaffaqiyat darajasi
//   - Bashorat qilinadigan ishlash
//   - Mijozlar qoniqishi: 75% → 98%
//   - Yiliga $200K daromad saqlab qolindi
\`\`\`

**3. Ma'lumotlar bazasi ulanish poolini himoya qilish**
Ma'lumotlar bazasining ortiqcha yuklanishini oldini olish:
\`\`\`go
// Muammo: Ma'lumotlar bazasida Thundering herd
func ImportRecords(records <-chan Record) error {
    for rec := range records {
        // 1000 ta yozuv bir vaqtning o'zida keladi
        // Ma'lumotlar bazasida faqat 20 ta ulanish bor
        // 980 ta so'rov kutadi, timeout, retry → kaskadli nosozlik
        if err := db.Insert(rec); err != nil {
            return err
        }
    }
}

// Yechim: Ma'lumotlar bazasi sig'imiga mos rate limiting
func ImportRecords(records <-chan Record) error {
    ctx := context.Background()

    // Ma'lumotlar bazasi soniyasiga 100 ta insertni qulay qayta ishlaydi
    // Burst traffic uchun 50 ta burstga ruxsat beramiz
    rateLimited := RateLimitedChannel(ctx, records, 100, 50)

    for rec := range rateLimited {
        if err := db.Insert(rec); err != nil {
            return err
        }
    }
    return nil
}

// Natijalar:
// - Oldin: Ma'lumotlar bazasi CPU: 95%, ulanish timeoutlari, kaskadli nosozliklar
// - Keyin: Ma'lumotlar bazasi CPU: 60%, nol timeoutlar, barqaror
// - Import ishonchliligi: 70% → 99.9%
// - Ma'lumotlar bazasi xarajatlari 40% ga qisqartirildi (ortiqcha ta'minlash kerak emas)
\`\`\`

**4. Email xizmatini himoya qilish**
Startupda SendGrid limitlariga rioya qilish:
\`\`\`go
func SendEmails(jobs <-chan EmailJob) error {
    ctx := context.Background()

    // SendGrid free tier: Kuniga 100 email = 0.0012 email/soniya
    // Har 15 soniyada 1 emailga aylantiramiz
    // Lekin sezgir UX uchun 10 ta burstga ruxsat beramiz
    rateLimited := RateLimitedChannel(ctx, jobs, 1, 10)

    for job := range rateLimited {
        if err := sendgrid.Send(job.To, job.Subject, job.Body); err != nil {
            log.Printf("%s ga email yuborib bo'lmadi: %v", job.To, err)
            continue
        }
    }
    return nil
}

// Ta'sir:
// - Oldin: Kunlik limitga birinchi soatda yetdi, kunning qolgan qismi buzildi
// - Keyin: Emaillar kun davomida taqsimlandi, limitga hech qachon yetmaydi
// - Pullik tarifga o'tish 6 oy kechiktirildi: $3000 tejaldi
\`\`\`

**5. Webhook yetkazib berish**
Mijozlar serverlarini ortiqcha yuklamasdan webhooklarni yetkazib berish:
\`\`\`go
type Webhook struct {
    URL     string
    Payload interface{}
    Retries int
}

func DeliverWebhooks(webhooks <-chan Webhook) error {
    ctx := context.Background()

    // Mijozlar serverlariga muloyim bo'lish
    // Soniyasiga 10 ta webhook, 20 ta burst bilan
    rateLimited := RateLimitedChannel(ctx, webhooks, 10, 20)

    for webhook := range rateLimited {
        if err := httpPost(webhook.URL, webhook.Payload); err != nil {
            log.Printf("%s ga webhook yetkazib berishda xato: %v", webhook.URL, err)

            // Eksponensial backoff bilan qayta urinish
            if webhook.Retries < 3 {
                webhook.Retries++
                requeue(webhook)
            }
        }
    }
    return nil
}

// Mijoz fikri:
// - Oldin: "Sizning webhooklar bizning serverimizni quladi!" (g'azablangan mijoz, shartnoma yo'qoldi)
// - Keyin: "Webhooklar ajoyib ishlayapti!" (mamnun mijoz, yiliga $50K shartnoma)
\`\`\`

**5. Rate Limiter ni test qilish**
\`\`\`go
func TestRateLimiter(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()

    // Test kirishini yaratish
    input := make(chan int, 100)
    for i := 0; i < 100; i++ {
        input <- i
    }
    close(input)

    // 10/soniyaga cheklash
    start := time.Now()
    rateLimited := RateLimitedChannel(ctx, input, 10, 1)

    count := 0
    for range rateLimited {
        count++
    }
    elapsed := time.Since(start)

    // 10/soniyada 100 element uchun ~10 soniya olishi kerak
    if elapsed < 9*time.Second || elapsed > 11*time.Second {
        t.Errorf("~10s kutilgan, %v olindi", elapsed)
    }

    if count != 100 {
        t.Errorf("100 ta element kutilgan, %d ta olindi", count)
    }
}
\`\`\`

**6. Dinamik tezlik moslashuvi**
\`\`\`go
// Tizim yukiga asoslangan tezlik moslashuvi
func AdaptiveRateLimiting(requests <-chan Request) {
    ctx := context.Background()

    // Tizim metrikalarini monitoring qilish
    currentLoad := getSystemLoad()

    rate := 100  // standart
    if currentLoad > 0.8 {
        rate = 50  // yuqori yukda kamaytiramiz
    } else if currentLoad < 0.3 {
        rate = 200  // bo'sh resurslarda oshiramiz
    }

    rateLimited := RateLimitedChannel(ctx, requests, rate, rate/2)

    // Adaptiv tezlikda qayta ishlash
    for req := range rateLimited {
        processRequest(req)
    }
}
\`\`\`

**Production Best Practices:**
1. Sezgir UX uchun har doim burst sig'imini qo'shing
2. Rate ni API limitining 90% ga o'rnating (xavfsizlik zahirasi)
3. Rate limiter metrikalarini monitor qiling (mavjud tokenlar, cheklangan elementlar)
4. Debugging uchun observability qo'shing (throttling yuz berganda log qiling)
5. Graceful shutdown uchun contextdan foydalaning
6. Realistik traffic patternlari bilan test qiling
7. Service SLA da rate limitlarni hujjatlang
8. Klasterlar uchun distributed rate limiting dan foydalanishni ko'rib chiqing
9. Qayta urinishlar uchun backoff mexanizmlarini amalga oshiring
10. Limitlarga yaqinlashganda ogohlantirish qo'shing

**Haqiqiy ta'sir:**
Soatiga 1M hodisani qayta ishlaydigan event processing platformasi:
- **Oldin**: Hodisalar downstream xizmatlarni to'ldiradi → 30% hodisalar yo'qolgan → haftasiga 3-4 nosozlik
- **Keyin**: 500 ta burst bilan silliq 280 hodisa/soniya → 99.99% yetkazib berish → 6 oyda nol nosozlik
- **Natija**: Yiliga $2M daromad himoya qilindi, infrastructurada yiliga $100K tejaldi
- **Mijozlar qoniqishi**: 70% → 98%
- **On-call muhandislar javob vaqti**: 90% ga qisqartirildi

Rate limiting faqat qoidalarga rioya qilish haqida emas - bu xizmatlaringizni ham, mijozlaringizni ham himoya qiladigan ishonchli, bashorat qilinadigan tizimlarni qurish haqida. Bu har qanday production Go ilovasi uchun fundamental vositadir.`
		}
	}
};

export default task;
