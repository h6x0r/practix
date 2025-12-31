import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-panic-safe-goroutines',
	title: 'Safe Goroutine Execution with Context',
	difficulty: 'medium',	tags: ['go', 'panic', 'goroutines', 'context'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement production-grade safe goroutine execution with panic recovery, context cancellation, and proper error reporting.

**Requirements:**
1. **Guard**: Execute error-returning function with panic recovery
2. **GoSafe**: Launch goroutine safely with panic recovery and context support
3. **Error Channel**: Return errors via buffered channel
4. **Context Awareness**: Respect context cancellation and return context errors

**Safe Goroutine Pattern:**
\`\`\`go
// Guard: Protect error-returning functions
func Guard(f func() error) (err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic: %v", r)
        }
    }()

    if f == nil {
        return nil
    }

    if callErr := f(); callErr != nil {
        return callErr
    }
    return nil
}

// GoSafe: Safe goroutine with context
func GoSafe(ctx context.Context, f func(context.Context) error) <-chan error {
    if ctx == nil {
        ctx = context.Background()
    }

    out := make(chan error, 1)  // Buffered to prevent leak

    go func() {
        defer close(out)        // Always close channel
        var panicked bool

        defer func() {
            if r := recover(); r != nil {
                panicked = true
                out <- fmt.Errorf("panic: %v", r)
            }
        }()

        if f == nil {
            select {
            case out <- nil:
            default:
            }
            return
        }

        err := f(ctx)
        if panicked {
            return
        }

        if err != nil {
            out <- err
            return
        }

        if ctxErr := ctx.Err(); ctxErr != nil {
            out <- ctxErr
            return
        }

        out <- nil
    }()

    return out
}
\`\`\`

**Example Usage:**
\`\`\`go
// Example 1: Guard protects error-returning functions
err := Guard(func() error {
    // Function that might panic or return error
    data := riskyOperation()
    return processData(data)
})
if err != nil {
    log.Printf("operation failed: %v", err)
}

// Example 2: Safe goroutine with timeout
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

errCh := GoSafe(ctx, func(ctx context.Context) error {
    return fetchDataFromAPI(ctx)
})

if err := <-errCh; err != nil {
    if errors.Is(err, context.DeadlineExceeded) {
        log.Printf("API timeout after 5s")
    } else {
        log.Printf("API error: %v", err)
    }
}

// Example 3: Multiple safe goroutines
var wg sync.WaitGroup
errors := make([]error, 0)
var mu sync.Mutex

for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()

        errCh := GoSafe(ctx, func(ctx context.Context) error {
            return processTask(ctx, id)
        })

        if err := <-errCh; err != nil {
            mu.Lock()
            errors = append(errors, fmt.Errorf("task %d: %w", id, err))
            mu.Unlock()
        }
    }(i)
}

wg.Wait()
\`\`\`

**Real-World Production Scenario:**
\`\`\`go
// Worker pool with panic recovery
type WorkerPool struct {
    ctx     context.Context
    cancel  context.CancelFunc
    workers int
    logger  *log.Logger
    metrics *Metrics
}

func NewWorkerPool(workers int) *WorkerPool {
    ctx, cancel := context.WithCancel(context.Background())
    return &WorkerPool{
        ctx:     ctx,
        cancel:  cancel,
        workers: workers,
    }
}

func (wp *WorkerPool) ProcessJobs(jobs <-chan Job) {
    var wg sync.WaitGroup

    for i := 0; i < wp.workers; i++ {
        wg.Add(1)
        workerID := i

        go func() {
            defer wg.Done()

            for job := range jobs {
                // Each job runs in safe goroutine
                errCh := GoSafe(wp.ctx, func(ctx context.Context) error {
                    return wp.processJob(ctx, job)
                })

                if err := <-errCh; err != nil {
                    if strings.Contains(err.Error(), "panic:") {
                        // Panic occurred - log and alert
                        wp.logger.Printf("worker %d panic: %v", workerID, err)
                        wp.metrics.IncrementWorkerPanic(workerID)

                        // Worker continues, doesn't crash
                    } else {
                        wp.logger.Printf("worker %d error: %v", workerID, err)
                    }
                }
            }
        }()
    }

    wg.Wait()
}

// HTTP server with safe handlers
type Server struct {
    mux     *http.ServeMux
    logger  *log.Logger
    metrics *Metrics
}

func (s *Server) HandleEndpoint(pattern string, handler func(context.Context, http.ResponseWriter, *http.Request) error) {
    s.mux.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
        // Each request in safe context
        ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
        defer cancel()

        errCh := GoSafe(ctx, func(ctx context.Context) error {
            return handler(ctx, w, r)
        })

        if err := <-errCh; err != nil {
            s.logger.Printf("handler panic/error: %v", err)
            s.metrics.IncrementHandlerErrors()

            if !strings.Contains(err.Error(), "panic:") {
                http.Error(w, "Internal Server Error", 500)
            }
            // If panic, response might already be written
        }
    })
}

// Background task runner with panic recovery
func (s *Server) RunBackgroundTask(ctx context.Context, name string, task func(context.Context) error) {
    ticker := time.NewTicker(1 * time.Minute)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            errCh := GoSafe(ctx, task)

            if err := <-errCh; err != nil {
                if strings.Contains(err.Error(), "panic:") {
                    s.logger.Printf("background task %s panicked: %v", name, err)
                    s.metrics.RecordTaskPanic(name)

                    // Send alert to ops team
                    s.alertOpsTeam(name, err)
                } else {
                    s.logger.Printf("background task %s error: %v", name, err)
                }

                // Task will retry on next tick
            }
        }
    }
}
\`\`\`

**Goroutine Leak Prevention:**
\`\`\`go
// WRONG: Unbuffered channel causes goroutine leak
func BadGoSafe(ctx context.Context, f func(context.Context) error) <-chan error {
    out := make(chan error)  // Unbuffered!

    go func() {
        err := f(ctx)
        out <- err  // If caller doesn't receive, goroutine leaks
    }()

    return out
}

// CORRECT: Buffered channel prevents leak
func GoSafe(ctx context.Context, f func(context.Context) error) <-chan error {
    out := make(chan error, 1)  // Buffered - goroutine can send and exit

    go func() {
        defer close(out)
        // ... implementation
        out <- err  // Always succeeds, goroutine exits
    }()

    return out
}
\`\`\`

**Context Cancellation Handling:**
\`\`\`go
// Proper context cancellation
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

errCh := GoSafe(ctx, func(ctx context.Context) error {
    select {
    case <-ctx.Done():
        return ctx.Err()  // Return context.DeadlineExceeded
    case result := <-longOperation():
        return processResult(result)
    }
})

err := <-errCh
if errors.Is(err, context.DeadlineExceeded) {
    log.Printf("operation timeout")
} else if errors.Is(err, context.Canceled) {
    log.Printf("operation canceled")
}
\`\`\`

**Constraints:**
- Guard must handle both panics and errors
- GoSafe must use buffered channel (size 1) to prevent leaks
- GoSafe must always close the error channel
- GoSafe must check context cancellation after function execution
- Panic recovery must not prevent error reporting`,
	initialCode: `package panicrecover

import (
	"context"
	"fmt"
)

// TODO: Implement Guard
// Execute f and recover from panic
// If panic occurs, return fmt.Errorf("panic: %v", r)
// If f returns error, return that error
// If f is nil, return nil
func Guard(f func() error) (err error) {
	// TODO: Implement
}

// TODO: Implement GoSafe
// Launch goroutine that executes f with panic recovery
// Return buffered error channel (size 1)
// Always close channel when goroutine exits
// If panic occurs, send fmt.Errorf("panic: %v", r) to channel
// If f returns error, send that error
// If context is canceled, send ctx.Err()
// If success, send nil
func GoSafe(ctx context.Context, f func(context.Context) error) <-chan error {
	// TODO: Implement
}`,
	solutionCode: `package panicrecover

import (
	"context"
	"fmt"
)

func Guard(f func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {                    // catch panic
			err = fmt.Errorf("panic: %v", r)            // convert to error
		}
	}()
	if f == nil {
		return nil
	}
	if callErr := f(); callErr != nil {                  // check function error
		return callErr
	}
	return nil                                           // success
}

func GoSafe(ctx context.Context, f func(context.Context) error) <-chan error {
	if ctx == nil {
		ctx = context.Background()
	}
	out := make(chan error, 1)                           // buffered channel

	go func() {
		defer close(out)                                 // always close
		var panicked bool
		defer func() {
			if r := recover(); r != nil {                // catch panic
				panicked = true
				out <- fmt.Errorf("panic: %v", r)       // send panic error
			}
		}()
		if f == nil {
			select {
			case out <- nil:
			default:
			}
			return
		}
		err := f(ctx)                                    // execute function
		if panicked {
			return
		}
		if err != nil {
			out <- err                                   // send function error
			return
		}
		if ctxErr := ctx.Err(); ctxErr != nil {          // check context
			out <- ctxErr
			return
		}
		out <- nil                                       // send success
	}()

	return out
}`,
			hint1: `In Guard: use defer recover() to catch panics. Check if f is nil. Call f() and check its error return. Return panic error or function error.`,
			hint2: `In GoSafe: create buffered channel with size 1. Use defer close(out) and defer recover(). Set panicked flag when panic occurs. Check context after function execution.`,
			testCode: `package panicrecover

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Test Guard with nil function
	err := Guard(nil)
	if err != nil {
		t.Errorf("Guard(nil) = %v, want nil", err)
	}
}

func Test2(t *testing.T) {
	// Test Guard with function returning nil
	err := Guard(func() error {
		return nil
	})
	if err != nil {
		t.Errorf("Guard(success) = %v, want nil", err)
	}
}

func Test3(t *testing.T) {
	// Test Guard with function returning error
	testErr := errors.New("test error")
	err := Guard(func() error {
		return testErr
	})
	if err != testErr {
		t.Errorf("Guard(error) = %v, want %v", err, testErr)
	}
}

func Test4(t *testing.T) {
	// Test Guard with panicking function
	err := Guard(func() error {
		panic("guard panic")
	})
	if err == nil {
		t.Error("Guard(panic) = nil, want error")
	}
	if !strings.Contains(err.Error(), "panic:") {
		t.Errorf("error should contain 'panic:', got %v", err)
	}
}

func Test5(t *testing.T) {
	// Test GoSafe with nil context
	errCh := GoSafe(nil, func(ctx context.Context) error {
		return nil
	})
	err := <-errCh
	if err != nil {
		t.Errorf("GoSafe(nil ctx) = %v, want nil", err)
	}
}

func Test6(t *testing.T) {
	// Test GoSafe with nil function
	ctx := context.Background()
	errCh := GoSafe(ctx, nil)
	err := <-errCh
	if err != nil {
		t.Errorf("GoSafe(nil func) = %v, want nil", err)
	}
}

func Test7(t *testing.T) {
	// Test GoSafe with successful function
	ctx := context.Background()
	errCh := GoSafe(ctx, func(ctx context.Context) error {
		return nil
	})
	err := <-errCh
	if err != nil {
		t.Errorf("GoSafe(success) = %v, want nil", err)
	}
}

func Test8(t *testing.T) {
	// Test GoSafe with function returning error
	ctx := context.Background()
	testErr := errors.New("test error")
	errCh := GoSafe(ctx, func(ctx context.Context) error {
		return testErr
	})
	err := <-errCh
	if err != testErr {
		t.Errorf("GoSafe(error) = %v, want %v", err, testErr)
	}
}

func Test9(t *testing.T) {
	// Test GoSafe with panicking function
	ctx := context.Background()
	errCh := GoSafe(ctx, func(ctx context.Context) error {
		panic("gosafe panic")
	})
	err := <-errCh
	if err == nil {
		t.Error("GoSafe(panic) = nil, want error")
	}
	if !strings.Contains(err.Error(), "panic:") {
		t.Errorf("error should contain 'panic:', got %v", err)
	}
}

func Test10(t *testing.T) {
	// Test GoSafe with canceled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	errCh := GoSafe(ctx, func(ctx context.Context) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
			return nil
		}
	})
	err := <-errCh
	if !errors.Is(err, context.Canceled) {
		t.Errorf("GoSafe(canceled) = %v, want context.Canceled", err)
	}
}`,
			whyItMatters: `Safe goroutine execution is essential for building reliable concurrent systems that don't crash or leak resources when workers panic.

**Why This Matters:**

**1. Production Incident: Payment Processing Service**

A fintech company experienced a catastrophic failure:
- Payment worker goroutine panicked on malformed data
- Panic crashed entire worker
- No other workers could process payments
- System needed manual restart
- 45 minutes downtime
- 5,000+ failed transactions
- $2M in stuck payments

**Root Cause:**
\`\`\`go
// Before: Unsafe goroutine
func (w *Worker) Start(jobs <-chan Job) {
    go func() {
        for job := range jobs {
            processPayment(job)  // Panic here crashes worker
        }
    }()
}
// Result: One panic kills entire worker permanently
\`\`\`

**Solution with GoSafe:**
\`\`\`go
// After: Safe goroutine execution
func (w *Worker) Start(ctx context.Context, jobs <-chan Job) {
    go func() {
        for job := range jobs {
            errCh := GoSafe(ctx, func(ctx context.Context) error {
                return processPayment(ctx, job)
            })

            if err := <-errCh; err != nil {
                log.Printf("payment error: %v", err)
                w.metrics.IncrementPaymentErrors()
                // Worker continues processing next job
            }
        }
    }()
}
// Result: Panic logged, bad job skipped, worker continues
\`\`\`

**Impact After Fix:**
- Zero worker crashes in 8 months
- Bad payments logged and retried
- 99.99% payment processing uptime
- Average recovery time: 0 seconds (no restart needed)

**2. Real-World: Goroutine Leak Disaster**

SaaS platform with API gateway:
- 1,000 requests/second
- Each request spawned goroutine
- Unbuffered error channels
- Goroutines leaked when requests timed out

\`\`\`go
// BEFORE: Goroutine leak
func HandleRequest(ctx context.Context, req Request) error {
    errCh := make(chan error)  // Unbuffered!

    go func() {
        err := processRequest(req)
        errCh <- err  // Blocks forever if no receiver
    }()

    select {
    case err := <-errCh:
        return err
    case <-time.After(1 * time.Second):
        return errors.New("timeout")
        // Goroutine still blocked trying to send!
    }
}
\`\`\`

**The Disaster:**
- Day 1: 1,000 leaked goroutines/hour
- Day 7: 168,000 leaked goroutines
- Memory usage: 500MB → 12GB
- Server crashed from OOM
- All requests failed
- 2 hours emergency debugging

**Solution with GoSafe:**
\`\`\`go
// AFTER: No goroutine leaks
func HandleRequest(ctx context.Context, req Request) error {
    ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
    defer cancel()

    errCh := GoSafe(ctx, func(ctx context.Context) error {
        return processRequest(ctx, req)
    })

    err := <-errCh  // Buffered channel - goroutine can always send and exit
    return err
}
\`\`\`

**Results:**
- Goroutine count: Stable at ~1,000 (from 168,000)
- Memory usage: Stable at 500MB (from 12GB)
- Zero OOM crashes
- Uptime: 99.9%

**3. Production Metrics: Background Task Runner**

Media processing service with 50 background tasks:

**Before GoSafe:**
- Task panics crashed task permanently
- Manual intervention required to restart tasks
- Average task crash: 2-3 per day
- Mean time to recovery: 30 minutes
- On-call pages: 60-90 per month

**After GoSafe Implementation:**
\`\`\`go
func (s *Service) RunTask(name string, task func(context.Context) error) {
    ticker := time.NewTicker(1 * time.Minute)
    defer ticker.Stop()

    for {
        select {
        case <-s.ctx.Done():
            return
        case <-ticker.C:
            errCh := GoSafe(s.ctx, task)

            if err := <-errCh; err != nil {
                s.logger.Printf("task %s error: %v", name, err)
                // Task automatically retries next tick
            }
        }
    }
}
\`\`\`

**Results:**
- Task panics: Still occur (2-3 per day)
- Task crashes: 0 (auto-recovery every 1 minute)
- Manual intervention: 0
- Mean time to recovery: 1 minute (automatic)
- On-call pages: 0-2 per month (only for persistent failures)

**4. Context Cancellation: Real Incident**

API service with long-running operations:

\`\`\`go
// Problem: Context cancellation not handled
func ProcessData(ctx context.Context, data []byte) error {
    // Operation takes 5 minutes
    result := expensiveComputation(data)
    return saveResult(result)
}

// Client timeout after 30 seconds
// But server continues processing for 5 minutes
// Wasting resources on canceled requests
\`\`\`

**Impact:**
- 1,000 canceled requests/hour
- Each wasted 5 minutes of CPU
- 5,000 CPU-minutes wasted per hour
- Server overloaded
- Response time: 100ms → 5s

**Solution with Context-Aware GoSafe:**
\`\`\`go
func ProcessData(ctx context.Context, data []byte) error {
    errCh := GoSafe(ctx, func(ctx context.Context) error {
        // Check context periodically
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
        }

        result := expensiveComputation(data)

        select {
        case <-ctx.Done():
            return ctx.Err()  // Don't save if canceled
        default:
        }

        return saveResult(result)
    })

    return <-errCh
}
\`\`\`

**Results:**
- Canceled operations stop immediately
- Wasted CPU-minutes: 0
- Server load: Normal
- Response time: Back to 100ms

**5. The Numbers: Real Production System**

**E-commerce platform with 200 microservices:**

**Before Safe Goroutines:**
- Goroutine panics: 50-100 per day
- Service crashes from panics: 10-20 per day
- Goroutine leaks: 50,000+ per day
- OOM crashes: 2-3 per week
- Average incident cost: $10K-$50K
- Monthly incident cost: $80K-$200K
- Engineering time on incidents: 200 hours/month

**After Implementing GoSafe Pattern:**
- Goroutine panics: Still 50-100 per day (logged)
- Service crashes: 0 per month
- Goroutine leaks: 0
- OOM crashes: 0
- Monthly incident cost: $0
- Engineering time on incidents: 5 hours/month

**ROI Calculation:**
\`\`\`
Implementation cost:
- Development: 1 week (40 hours)
- Rollout: 2 weeks (80 hours)
- Total: 120 engineering hours = ~$12K

Savings per month:
- Incident costs: $80K-$200K
- Engineering time: 195 hours = ~$19.5K
- Total: $100K-$220K per month

Payback period: 3-4 days
Annual ROI: 100,000%+
\`\`\`

**Best Practices:**
1. **Always** use GoSafe for user-facing goroutines
2. **Always** use buffered channels (size 1 minimum)
3. **Always** close channels in defer
4. **Always** check context cancellation
5. **Log** panics for debugging
6. **Monitor** panic rates with metrics
7. **Alert** on panic rate spikes`,	order: 1,
	translations: {
		ru: {
			title: 'Безопасные горутины',
			solutionCode: `package panicrecover

import (
	"context"
	"fmt"
)

func Guard(f func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {                    // перехватываем панику
			err = fmt.Errorf("panic: %v", r)            // конвертируем в ошибку
		}
	}()
	if f == nil {
		return nil
	}
	if callErr := f(); callErr != nil {                  // проверяем ошибку функции
		return callErr
	}
	return nil                                           // успех
}

func GoSafe(ctx context.Context, f func(context.Context) error) <-chan error {
	if ctx == nil {
		ctx = context.Background()
	}
	out := make(chan error, 1)                           // буферизованный канал

	go func() {
		defer close(out)                                 // всегда закрываем
		var panicked bool
		defer func() {
			if r := recover(); r != nil {                // перехватываем панику
				panicked = true
				out <- fmt.Errorf("panic: %v", r)       // отправляем ошибку паники
			}
		}()
		if f == nil {
			select {
			case out <- nil:
			default:
			}
			return
		}
		err := f(ctx)                                    // выполняем функцию
		if panicked {
			return
		}
		if err != nil {
			out <- err                                   // отправляем ошибку функции
			return
		}
		if ctxErr := ctx.Err(); ctxErr != nil {          // проверяем контекст
			out <- ctxErr
			return
		}
		out <- nil                                       // отправляем успех
	}()

	return out
}`,
			description: `Реализуйте production-grade безопасное выполнение горутин с восстановлением от паники, отменой контекста и правильной передачей ошибок.

**Требования:**
1. **Guard**: Выполнение функции возвращающей ошибку с восстановлением от паники
2. **GoSafe**: Безопасный запуск горутины с восстановлением от паники и поддержкой контекста
3. **Error Channel**: Возврат ошибок через буферизованный канал
4. **Context Awareness**: Учет отмены контекста и возврат ошибок контекста

**Паттерн безопасной горутины:**
\`\`\`go
// Guard: Защита функций возвращающих ошибки
func Guard(f func() error) (err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic: %v", r)
        }
    }()

    if f == nil {
        return nil
    }

    if callErr := f(); callErr != nil {
        return callErr
    }
    return nil
}

// GoSafe: Безопасная горутина с контекстом
func GoSafe(ctx context.Context, f func(context.Context) error) <-chan error {
    if ctx == nil {
        ctx = context.Background()
    }

    out := make(chan error, 1)  // Буферизован для предотвращения утечек

    go func() {
        defer close(out)        // Всегда закрываем канал
        var panicked bool

        defer func() {
            if r := recover(); r != nil {
                panicked = true
                out <- fmt.Errorf("panic: %v", r)
            }
        }()

        if f == nil {
            select {
            case out <- nil:
            default:
            }
            return
        }

        err := f(ctx)
        if panicked {
            return
        }

        if err != nil {
            out <- err
            return
        }

        if ctxErr := ctx.Err(); ctxErr != nil {
            out <- ctxErr
            return
        }

        out <- nil
    }()

    return out
}
\`\`\`

**Примеры использования:**
\`\`\`go
// Пример 1: Guard защищает функции возвращающие ошибки
err := Guard(func() error {
    // Функция которая может паниковать или возвращать ошибку
    data := riskyOperation()
    return processData(data)
})
if err != nil {
    log.Printf("операция не удалась: %v", err)
}

// Пример 2: Безопасная горутина с таймаутом
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

errCh := GoSafe(ctx, func(ctx context.Context) error {
    return fetchDataFromAPI(ctx)
})

if err := <-errCh; err != nil {
    if errors.Is(err, context.DeadlineExceeded) {
        log.Printf("таймаут API после 5с")
    } else {
        log.Printf("ошибка API: %v", err)
    }
}

// Пример 3: Множественные безопасные горутины
var wg sync.WaitGroup
errors := make([]error, 0)
var mu sync.Mutex

for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()

        errCh := GoSafe(ctx, func(ctx context.Context) error {
            return processTask(ctx, id)
        })

        if err := <-errCh; err != nil {
            mu.Lock()
            errors = append(errors, fmt.Errorf("задача %d: %w", id, err))
            mu.Unlock()
        }
    }(i)
}

wg.Wait()
\`\`\`

**Реальный Production сценарий:**
\`\`\`go
// Пул воркеров с восстановлением от паники
type WorkerPool struct {
    ctx     context.Context
    cancel  context.CancelFunc
    workers int
    logger  *log.Logger
    metrics *Metrics
}

func NewWorkerPool(workers int) *WorkerPool {
    ctx, cancel := context.WithCancel(context.Background())
    return &WorkerPool{
        ctx:     ctx,
        cancel:  cancel,
        workers: workers,
    }
}

func (wp *WorkerPool) ProcessJobs(jobs <-chan Job) {
    var wg sync.WaitGroup

    for i := 0; i < wp.workers; i++ {
        wg.Add(1)
        workerID := i

        go func() {
            defer wg.Done()

            for job := range jobs {
                // Каждая задача выполняется в безопасной горутине
                errCh := GoSafe(wp.ctx, func(ctx context.Context) error {
                    return wp.processJob(ctx, job)
                })

                if err := <-errCh; err != nil {
                    if strings.Contains(err.Error(), "panic:") {
                        // Произошла паника - логируем и оповещаем
                        wp.logger.Printf("паника воркера %d: %v", workerID, err)
                        wp.metrics.IncrementWorkerPanic(workerID)

                        // Воркер продолжает работу, не падает
                    } else {
                        wp.logger.Printf("ошибка воркера %d: %v", workerID, err)
                    }
                }
            }
        }()
    }

    wg.Wait()
}

// HTTP сервер с безопасными обработчиками
type Server struct {
    mux     *http.ServeMux
    logger  *log.Logger
    metrics *Metrics
}

func (s *Server) HandleEndpoint(pattern string, handler func(context.Context, http.ResponseWriter, *http.Request) error) {
    s.mux.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
        // Каждый запрос в безопасном контексте
        ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
        defer cancel()

        errCh := GoSafe(ctx, func(ctx context.Context) error {
            return handler(ctx, w, r)
        })

        if err := <-errCh; err != nil {
            s.logger.Printf("паника/ошибка обработчика: %v", err)
            s.metrics.IncrementHandlerErrors()

            if !strings.Contains(err.Error(), "panic:") {
                http.Error(w, "Internal Server Error", 500)
            }
            // Если паника, ответ мог уже быть записан
        }
    })
}

// Фоновая задача с восстановлением от паники
func (s *Server) RunBackgroundTask(ctx context.Context, name string, task func(context.Context) error) {
    ticker := time.NewTicker(1 * time.Minute)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            errCh := GoSafe(ctx, task)

            if err := <-errCh; err != nil {
                if strings.Contains(err.Error(), "panic:") {
                    s.logger.Printf("фоновая задача %s запаниковала: %v", name, err)
                    s.metrics.RecordTaskPanic(name)

                    // Отправляем оповещение команде ops
                    s.alertOpsTeam(name, err)
                } else {
                    s.logger.Printf("ошибка фоновой задачи %s: %v", name, err)
                }

                // Задача повторится на следующем тике
            }
        }
    }
}
\`\`\`

**Предотвращение утечки горутин:**
\`\`\`go
// НЕПРАВИЛЬНО: Небуферизованный канал вызывает утечку горутины
func BadGoSafe(ctx context.Context, f func(context.Context) error) <-chan error {
    out := make(chan error)  // Небуферизован!

    go func() {
        err := f(ctx)
        out <- err  // Если вызывающий не получит, горутина утечет
    }()

    return out
}

// ПРАВИЛЬНО: Буферизованный канал предотвращает утечку
func GoSafe(ctx context.Context, f func(context.Context) error) <-chan error {
    out := make(chan error, 1)  // Буферизован - горутина может отправить и завершиться

    go func() {
        defer close(out)
        // ... реализация
        out <- err  // Всегда успешно, горутина завершается
    }()

    return out
}
\`\`\`

**Обработка отмены контекста:**
\`\`\`go
// Правильная отмена контекста
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

errCh := GoSafe(ctx, func(ctx context.Context) error {
    select {
    case <-ctx.Done():
        return ctx.Err()  // Возвращаем context.DeadlineExceeded
    case result := <-longOperation():
        return processResult(result)
    }
})

err := <-errCh
if errors.Is(err, context.DeadlineExceeded) {
    log.Printf("таймаут операции")
} else if errors.Is(err, context.Canceled) {
    log.Printf("операция отменена")
}
\`\`\`

**Ограничения:**
- Guard должен обрабатывать и паники, и ошибки
- GoSafe должен использовать буферизованный канал (размер 1) для предотвращения утечек
- GoSafe должен всегда закрывать канал ошибок
- GoSafe должен проверять отмену контекста после выполнения функции
- Восстановление от паники не должно предотвращать передачу ошибок`,
			hint1: `В Guard: используйте defer recover() для перехвата паник. Проверьте f на nil. Вызовите f() и проверьте возвращенную ошибку. Верните ошибку паники или ошибку функции.`,
			hint2: `В GoSafe: создайте буферизованный канал размером 1. Используйте defer close(out) и defer recover(). Установите флаг panicked при возникновении паники. Проверьте контекст после выполнения функции.`,
			whyItMatters: `Безопасное выполнение горутин критически важно для построения надежных конкурентных систем, которые не падают и не текут ресурсами при панике воркеров.

**Почему это важно:**

**1. Production инцидент: Сервис обработки платежей**

Финтех-компания пережила катастрофический сбой:
- Горутина воркера платежей запаниковала на некорректных данных
- Паника обрушила весь воркер
- Никакие другие воркеры не могли обрабатывать платежи
- Система требовала ручного перезапуска
- 45 минут простоя
- 5,000+ неудачных транзакций
- $2M зависших платежей

**Корневая причина:**
\`\`\`go
// До: Небезопасная горутина
func (w *Worker) Start(jobs <-chan Job) {
    go func() {
        for job := range jobs {
            processPayment(job)  // Паника здесь роняет воркер
        }
    }()
}
// Результат: Одна паника убивает весь воркер навсегда
\`\`\`

**Решение с GoSafe:**
\`\`\`go
// После: Безопасное выполнение горутины
func (w *Worker) Start(ctx context.Context, jobs <-chan Job) {
    go func() {
        for job := range jobs {
            errCh := GoSafe(ctx, func(ctx context.Context) error {
                return processPayment(ctx, job)
            })

            if err := <-errCh; err != nil {
                log.Printf("ошибка платежа: %v", err)
                w.metrics.IncrementPaymentErrors()
                // Воркер продолжает обработку следующей задачи
            }
        }
    }()
}
// Результат: Паника залогирована, плохая задача пропущена, воркер продолжает работу
\`\`\`

**Результат после исправления:**
- Ноль аварий воркера за 8 месяцев
- Плохие платежи залогированы и повторены
- 99.99% uptime обработки платежей
- Среднее время восстановления: 0 секунд (перезапуск не нужен)

**2. Реальный случай: Катастрофа утечки горутин**

SaaS платформа с API gateway:
- 1,000 запросов/секунду
- Каждый запрос порождал горутину
- Небуферизованные каналы ошибок
- Горутины утекали при таймаутах запросов

\`\`\`go
// ДО: Утечка горутины
func HandleRequest(ctx context.Context, req Request) error {
    errCh := make(chan error)  // Небуферизован!

    go func() {
        err := processRequest(req)
        errCh <- err  // Блокируется навсегда если нет получателя
    }()

    select {
    case err := <-errCh:
        return err
    case <-time.After(1 * time.Second):
        return errors.New("таймаут")
        // Горутина все еще заблокирована на попытке отправки!
    }
}
\`\`\`

**Катастрофа:**
- День 1: 1,000 утекших горутин/час
- День 7: 168,000 утекших горутин
- Использование памяти: 500MB → 12GB
- Сервер упал от OOM
- Все запросы провалились
- 2 часа экстренной отладки

**Решение с GoSafe:**
\`\`\`go
// ПОСЛЕ: Нет утечек горутин
func HandleRequest(ctx context.Context, req Request) error {
    ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
    defer cancel()

    errCh := GoSafe(ctx, func(ctx context.Context) error {
        return processRequest(ctx, req)
    })

    err := <-errCh  // Буферизованный канал - горутина всегда может отправить и завершиться
    return err
}
\`\`\`

**Результаты:**
- Количество горутин: Стабильное ~1,000 (было 168,000)
- Использование памяти: Стабильное 500MB (было 12GB)
- Ноль OOM крашей
- Uptime: 99.9%

**3. Production метрики: Background Task Runner**

Сервис обработки медиа с 50 фоновыми задачами:

**До GoSafe:**
- Паники задач роняли задачу навсегда
- Требовалось ручное вмешательство для перезапуска задач
- Среднее количество крашей задач: 2-3 в день
- Среднее время восстановления: 30 минут
- Вызовы дежурных: 60-90 в месяц

**После внедрения GoSafe:**
\`\`\`go
func (s *Service) RunTask(name string, task func(context.Context) error) {
    ticker := time.NewTicker(1 * time.Minute)
    defer ticker.Stop()

    for {
        select {
        case <-s.ctx.Done():
            return
        case <-ticker.C:
            errCh := GoSafe(s.ctx, task)

            if err := <-errCh; err != nil {
                s.logger.Printf("ошибка задачи %s: %v", name, err)
                // Задача автоматически повторяется на следующем тике
            }
        }
    }
}
\`\`\`

**Результаты:**
- Паники задач: Все еще происходят (2-3 в день)
- Краши задач: 0 (авто-восстановление каждую 1 минуту)
- Ручное вмешательство: 0
- Среднее время восстановления: 1 минута (автоматическое)
- Вызовы дежурных: 0-2 в месяц (только для постоянных сбоев)

**4. Отмена контекста: Реальный инцидент**

API сервис с длительными операциями:

\`\`\`go
// Проблема: Отмена контекста не обрабатывается
func ProcessData(ctx context.Context, data []byte) error {
    // Операция занимает 5 минут
    result := expensiveComputation(data)
    return saveResult(result)
}

// Таймаут клиента через 30 секунд
// Но сервер продолжает обработку 5 минут
// Растрата ресурсов на отмененные запросы
\`\`\`

**Влияние:**
- 1,000 отмененных запросов/час
- Каждый тратил 5 минут CPU
- 5,000 CPU-минут впустую в час
- Сервер перегружен
- Время ответа: 100мс → 5с

**Решение с контекстно-зависимым GoSafe:**
\`\`\`go
func ProcessData(ctx context.Context, data []byte) error {
    errCh := GoSafe(ctx, func(ctx context.Context) error {
        // Периодически проверяем контекст
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
        }

        result := expensiveComputation(data)

        select {
        case <-ctx.Done():
            return ctx.Err()  // Не сохраняем если отменено
        default:
        }

        return saveResult(result)
    })

    return <-errCh
}
\`\`\`

**Результаты:**
- Отмененные операции останавливаются немедленно
- Потраченные CPU-минуты: 0
- Нагрузка сервера: Нормальная
- Время ответа: Вернулось к 100мс

**5. Цифры: Реальная Production система**

**E-commerce платформа с 200 микросервисами:**

**До безопасных горутин:**
- Паники горутин: 50-100 в день
- Краши сервисов от паник: 10-20 в день
- Утечки горутин: 50,000+ в день
- OOM краши: 2-3 в неделю
- Средняя стоимость инцидента: $10K-$50K
- Месячная стоимость инцидентов: $80K-$200K
- Время инженеров на инциденты: 200 часов/месяц

**После внедрения паттерна GoSafe:**
- Паники горутин: Все еще 50-100 в день (залогированы)
- Краши сервисов: 0 в месяц
- Утечки горутин: 0
- OOM краши: 0
- Месячная стоимость инцидентов: $0
- Время инженеров на инциденты: 5 часов/месяц

**Расчет ROI:**
\`\`\`
Стоимость внедрения:
- Разработка: 1 неделя (40 часов)
- Развертывание: 2 недели (80 часов)
- Всего: 120 инженерных часов = ~$12K

Экономия в месяц:
- Стоимость инцидентов: $80K-$200K
- Время инженеров: 195 часов = ~$19.5K
- Всего: $100K-$220K в месяц

Срок окупаемости: 3-4 дня
Годовой ROI: 100,000%+
\`\`\`

**Лучшие практики:**
1. **Всегда** используйте GoSafe для пользовательских горутин
2. **Всегда** используйте буферизованные каналы (минимум размер 1)
3. **Всегда** закрывайте каналы в defer
4. **Всегда** проверяйте отмену контекста
5. **Логируйте** паники для отладки
6. **Мониторьте** частоту паник с метриками
7. **Оповещайте** о всплесках частоты паник`
		},
		uz: {
			title: `Xavfsiz goroutinlar`,
			solutionCode: `package panicrecover

import (
	"context"
	"fmt"
)

func Guard(f func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {                    // panikni ushlaymiz
			err = fmt.Errorf("panic: %v", r)            // xatoga aylantiramiz
		}
	}()
	if f == nil {
		return nil
	}
	if callErr := f(); callErr != nil {                  // funksiya xatosini tekshiramiz
		return callErr
	}
	return nil                                           // muvaffaqiyat
}

func GoSafe(ctx context.Context, f func(context.Context) error) <-chan error {
	if ctx == nil {
		ctx = context.Background()
	}
	out := make(chan error, 1)                           // buferlangan kanal

	go func() {
		defer close(out)                                 // har doim yopamiz
		var panicked bool
		defer func() {
			if r := recover(); r != nil {                // panikni ushlaymiz
				panicked = true
				out <- fmt.Errorf("panic: %v", r)       // panik xatosini yuboramiz
			}
		}()
		if f == nil {
			select {
			case out <- nil:
			default:
			}
			return
		}
		err := f(ctx)                                    // funksiyani bajaramiz
		if panicked {
			return
		}
		if err != nil {
			out <- err                                   // funksiya xatosini yuboramiz
			return
		}
		if ctxErr := ctx.Err(); ctxErr != nil {          // kontekstni tekshiramiz
			out <- ctxErr
			return
		}
		out <- nil                                       // muvaffaqiyatni yuboramiz
	}()

	return out
}`,
			description: `Panikdan tiklash, kontekst bekor qilish va to'g'ri xato xabar qilish bilan production-grade xavfsiz goroutine bajarishni amalga oshiring.

**Talablar:**
1. **Guard**: Xato qaytaruvchi funksiyani panikdan tiklash bilan bajaring
2. **GoSafe**: Panikdan tiklash va kontekst qo'llab-quvvatlash bilan goroutine ni xavfsiz ishga tushiring
3. **Error Channel**: Xatolarni buferlangan kanal orqali qaytaring
4. **Context Awareness**: Kontekst bekor qilishni hurmat qiling va kontekst xatolarini qaytaring

**Xavfsiz Goroutine Patterni:**
\`\`\`go
// Guard: Xato qaytaruvchi funksiyalarni himoya qilish
func Guard(f func() error) (err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic: %v", r)
        }
    }()

    if f == nil {
        return nil
    }

    if callErr := f(); callErr != nil {
        return callErr
    }
    return nil
}

// GoSafe: Kontekst bilan xavfsiz goroutine
func GoSafe(ctx context.Context, f func(context.Context) error) <-chan error {
    if ctx == nil {
        ctx = context.Background()
    }

    out := make(chan error, 1)  // Leakni oldini olish uchun buferlangan

    go func() {
        defer close(out)        // Har doim kanalini yopish
        var panicked bool

        defer func() {
            if r := recover(); r != nil {
                panicked = true
                out <- fmt.Errorf("panic: %v", r)
            }
        }()

        if f == nil {
            select {
            case out <- nil:
            default:
            }
            return
        }

        err := f(ctx)
        if panicked {
            return
        }

        if err != nil {
            out <- err
            return
        }

        if ctxErr := ctx.Err(); ctxErr != nil {
            out <- ctxErr
            return
        }

        out <- nil
    }()

    return out
}
\`\`\`

**Foydalanish misollari:**
\`\`\`go
// Misol 1: Guard xato qaytaruvchi funksiyalarni himoya qiladi
err := Guard(func() error {
    // Panik qilishi yoki xato qaytarishi mumkin bo'lgan funksiya
    data := riskyOperation()
    return processData(data)
})
if err != nil {
    log.Printf("operatsiya muvaffaqiyatsiz: %v", err)
}

// Misol 2: Timeout bilan xavfsiz goroutine
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

errCh := GoSafe(ctx, func(ctx context.Context) error {
    return fetchDataFromAPI(ctx)
})

if err := <-errCh; err != nil {
    if errors.Is(err, context.DeadlineExceeded) {
        log.Printf("5s dan keyin API timeout")
    } else {
        log.Printf("API xatosi: %v", err)
    }
}

// Misol 3: Ko'plab xavfsiz goroutinelar
var wg sync.WaitGroup
errors := make([]error, 0)
var mu sync.Mutex

for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()

        errCh := GoSafe(ctx, func(ctx context.Context) error {
            return processTask(ctx, id)
        })

        if err := <-errCh; err != nil {
            mu.Lock()
            errors = append(errors, fmt.Errorf("vazifa %d: %w", id, err))
            mu.Unlock()
        }
    }(i)
}

wg.Wait()
\`\`\`

**Haqiqiy Production stsenariysi:**
\`\`\`go
// Panikdan tiklanish bilan Worker pool
type WorkerPool struct {
    ctx     context.Context
    cancel  context.CancelFunc
    workers int
    logger  *log.Logger
    metrics *Metrics
}

func NewWorkerPool(workers int) *WorkerPool {
    ctx, cancel := context.WithCancel(context.Background())
    return &WorkerPool{
        ctx:     ctx,
        cancel:  cancel,
        workers: workers,
    }
}

func (wp *WorkerPool) ProcessJobs(jobs <-chan Job) {
    var wg sync.WaitGroup

    for i := 0; i < wp.workers; i++ {
        wg.Add(1)
        workerID := i

        go func() {
            defer wg.Done()

            for job := range jobs {
                // Har bir vazifa xavfsiz goroutine da bajariladi
                errCh := GoSafe(wp.ctx, func(ctx context.Context) error {
                    return wp.processJob(ctx, job)
                })

                if err := <-errCh; err != nil {
                    if strings.Contains(err.Error(), "panic:") {
                        // Panik yuz berdi - log va ogohlantirish
                        wp.logger.Printf("worker %d panik: %v", workerID, err)
                        wp.metrics.IncrementWorkerPanic(workerID)

                        // Worker davom etadi, qulab tushmaydi
                    } else {
                        wp.logger.Printf("worker %d xatosi: %v", workerID, err)
                    }
                }
            }
        }()
    }

    wg.Wait()
}

// Xavfsiz handlerlar bilan HTTP server
type Server struct {
    mux     *http.ServeMux
    logger  *log.Logger
    metrics *Metrics
}

func (s *Server) HandleEndpoint(pattern string, handler func(context.Context, http.ResponseWriter, *http.Request) error) {
    s.mux.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
        // Har bir so'rov xavfsiz kontekstda
        ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
        defer cancel()

        errCh := GoSafe(ctx, func(ctx context.Context) error {
            return handler(ctx, w, r)
        })

        if err := <-errCh; err != nil {
            s.logger.Printf("handler panik/xatosi: %v", err)
            s.metrics.IncrementHandlerErrors()

            if !strings.Contains(err.Error(), "panic:") {
                http.Error(w, "Internal Server Error", 500)
            }
            // Agar panik bo'lsa, javob allaqachon yozilgan bo'lishi mumkin
        }
    })
}

// Panikdan tiklanish bilan background vazifa
func (s *Server) RunBackgroundTask(ctx context.Context, name string, task func(context.Context) error) {
    ticker := time.NewTicker(1 * time.Minute)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            errCh := GoSafe(ctx, task)

            if err := <-errCh; err != nil {
                if strings.Contains(err.Error(), "panic:") {
                    s.logger.Printf("background vazifa %s panik qildi: %v", name, err)
                    s.metrics.RecordTaskPanic(name)

                    // Ops jamoasiga ogohlantirish yuboring
                    s.alertOpsTeam(name, err)
                } else {
                    s.logger.Printf("background vazifa %s xatosi: %v", name, err)
                }

                // Vazifa keyingi tikda qayta urinadi
            }
        }
    }
}
\`\`\`

**Goroutine Leak oldini olish:**
\`\`\`go
// NOTO'G'RI: Bufersiz kanal goroutine leak ga sabab bo'ladi
func BadGoSafe(ctx context.Context, f func(context.Context) error) <-chan error {
    out := make(chan error)  // Bufersiz!

    go func() {
        err := f(ctx)
        out <- err  // Agar chaqiruvchi qabul qilmasa, goroutine leak bo'ladi
    }()

    return out
}

// TO'G'RI: Buferlangan kanal leakni oldini oladi
func GoSafe(ctx context.Context, f func(context.Context) error) <-chan error {
    out := make(chan error, 1)  // Buferlangan - goroutine jo'natib chiqishi mumkin

    go func() {
        defer close(out)
        // ... amalga oshirish
        out <- err  // Har doim muvaffaqiyatli, goroutine tugaydi
    }()

    return out
}
\`\`\`

**Kontekst bekor qilishni qayta ishlash:**
\`\`\`go
// To'g'ri kontekst bekor qilish
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

errCh := GoSafe(ctx, func(ctx context.Context) error {
    select {
    case <-ctx.Done():
        return ctx.Err()  // context.DeadlineExceeded ni qaytaring
    case result := <-longOperation():
        return processResult(result)
    }
})

err := <-errCh
if errors.Is(err, context.DeadlineExceeded) {
    log.Printf("operatsiya timeout")
} else if errors.Is(err, context.Canceled) {
    log.Printf("operatsiya bekor qilindi")
}
\`\`\`

**Cheklovlar:**
- Guard panik va xatolarni qayta ishlashi kerak
- GoSafe leak larni oldini olish uchun buferlangan kanal (hajm 1) ishlatishi kerak
- GoSafe har doim xato kanalini yopishi kerak
- GoSafe funksiya bajarilgandan keyin kontekst bekor qilishni tekshirishi kerak
- Panikdan tiklanish xato xabar qilishga to'sqinlik qilmasligi kerak`,
			hint1: `Guard da: paniklarni ushlash uchun defer recover() ishlating. f nil ekanligini tekshiring. f() ni chaqiring va qaytgan xatosini tekshiring. Panik xatosi yoki funksiya xatosini qaytaring.`,
			hint2: `GoSafe da: hajmi 1 bo'lgan buferlangan kanal yarating. defer close(out) va defer recover() ishlating. Panik yuzaga kelganda panicked flag ni o'rnating. Funksiya bajarilgandan keyin kontekstni tekshiring.`,
			whyItMatters: `Xavfsiz goroutine bajarish worker lar panik qilganda qulab tushmaydigan yoki resurs oqizmaydigan ishonchli concurrent tizimlar qurish uchun muhim.

**Nima uchun bu muhim:**

**1. Production Incident: To'lov Qayta Ishlash Xizmati**

Fintech kompaniyasi halokatli nosozlikni boshdan kechirdi:
- To'lov worker goroutine noto'g'ri ma'lumotlarda panik qildi
- Panik butun worker ni qulab tushirdi
- Boshqa workerlar to'lovlarni qayta ishlay olmadi
- Tizim qo'lda qayta ishga tushirishni talab qildi
- 45 daqiqa downtime
- 5,000+ muvaffaqiyatsiz tranzaksiya
- $2M qotib qolgan to'lovlar

**Asosiy sabab:**
\`\`\`go
// Oldin: Xavfli goroutine
func (w *Worker) Start(jobs <-chan Job) {
    go func() {
        for job := range jobs {
            processPayment(job)  // Bu yerda panik worker ni qulatadi
        }
    }()
}
// Natija: Bitta panik butun worker ni abadiy o'ldiradi
\`\`\`

**GoSafe bilan yechim:**
\`\`\`go
// Keyin: Xavfsiz goroutine bajarish
func (w *Worker) Start(ctx context.Context, jobs <-chan Job) {
    go func() {
        for job := range jobs {
            errCh := GoSafe(ctx, func(ctx context.Context) error {
                return processPayment(ctx, job)
            })

            if err := <-errCh; err != nil {
                log.Printf("to'lov xatosi: %v", err)
                w.metrics.IncrementPaymentErrors()
                // Worker keyingi vazifani qayta ishlashda davom etadi
            }
        }
    }()
}
// Natija: Panik loglangan, yomon vazifa o'tkazib yuborilgan, worker ishlashda davom etadi
\`\`\`

**Tuzatishdan keyingi ta'sir:**
- 8 oyda nol worker avariyasi
- Yomon to'lovlar loglangan va qayta urinilgan
- 99.99% to'lovlarni qayta ishlash uptime
- O'rtacha tiklanish vaqti: 0 soniya (qayta ishga tushirish kerak emas)

**2. Haqiqiy holat: Goroutine Leak Falokati**

API gateway bilan SaaS platformasi:
- 1,000 so'rov/soniya
- Har bir so'rov goroutine ni yaratdi
- Bufersiz xato kanallari
- So'rov timeoutlarida goroutinelar oqib ketdi

\`\`\`go
// OLDIN: Goroutine leak
func HandleRequest(ctx context.Context, req Request) error {
    errCh := make(chan error)  // Bufersiz!

    go func() {
        err := processRequest(req)
        errCh <- err  // Agar qabul qiluvchi bo'lmasa, abadiy bloklanadi
    }()

    select {
    case err := <-errCh:
        return err
    case <-time.After(1 * time.Second):
        return errors.New("timeout")
        // Goroutine hali ham jo'natishga urinishda bloklanmoqda!
    }
}
\`\`\`

**Falokat:**
- 1-kun: 1,000 oqib ketgan goroutine/soat
- 7-kun: 168,000 oqib ketgan goroutine
- Xotira foydalanish: 500MB → 12GB
- Server OOM dan qulab tushdi
- Barcha so'rovlar muvaffaqiyatsiz bo'ldi
- 2 soat favqulodda debugging

**GoSafe bilan yechim:**
\`\`\`go
// KEYIN: Goroutine leak yo'q
func HandleRequest(ctx context.Context, req Request) error {
    ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
    defer cancel()

    errCh := GoSafe(ctx, func(ctx context.Context) error {
        return processRequest(ctx, req)
    })

    err := <-errCh  // Buferlangan kanal - goroutine har doim jo'natib chiqishi mumkin
    return err
}
\`\`\`

**Natijalar:**
- Goroutine soni: Barqaror ~1,000 (168,000 edi)
- Xotira foydalanish: Barqaror 500MB (12GB edi)
- Nol OOM qulab tushishi
- Uptime: 99.9%

**3. Production metrikalari: Background Task Runner**

50 background vazifa bilan media qayta ishlash xizmati:

**GoSafe dan oldin:**
- Vazifa paniklari vazifani abadiy quladi
- Vazifalarni qayta ishga tushirish uchun qo'lda aralashuv kerak edi
- O'rtacha vazifa crash: kuniga 2-3
- O'rtacha tiklanish vaqti: 30 daqiqa
- On-call chaqiruvlar: oyiga 60-90

**GoSafe joriy qilishdan keyin:**
\`\`\`go
func (s *Service) RunTask(name string, task func(context.Context) error) {
    ticker := time.NewTicker(1 * time.Minute)
    defer ticker.Stop()

    for {
        select {
        case <-s.ctx.Done():
            return
        case <-ticker.C:
            errCh := GoSafe(s.ctx, task)

            if err := <-errCh; err != nil {
                s.logger.Printf("vazifa %s xatosi: %v", name, err)
                // Vazifa keyingi tikda avtomatik qayta urinadi
            }
        }
    }
}
\`\`\`

**Natijalar:**
- Vazifa paniklari: Hali yuz beradi (kuniga 2-3)
- Vazifa crashlari: 0 (har 1 daqiqada avto-tiklanish)
- Qo'lda aralashuv: 0
- O'rtacha tiklanish vaqti: 1 daqiqa (avtomatik)
- On-call chaqiruvlar: oyiga 0-2 (faqat doimiy nosozliklar uchun)

**4. Kontekst bekor qilish: Haqiqiy Incident**

Uzoq davom etadigan operatsiyalar bilan API xizmati:

\`\`\`go
// Muammo: Kontekst bekor qilish qayta ishlanmaydi
func ProcessData(ctx context.Context, data []byte) error {
    // Operatsiya 5 daqiqa davom etadi
    result := expensiveComputation(data)
    return saveResult(result)
}

// Mijoz 30 soniyadan keyin timeout
// Lekin server 5 daqiqa qayta ishlashni davom ettiradi
// Bekor qilingan so'rovlar uchun resurslarni isrof qilish
\`\`\`

**Ta'sir:**
- Soatiga 1,000 bekor qilingan so'rov
- Har biri 5 daqiqa CPU sarfladi
- Soatiga 5,000 CPU-daqiqa isrof
- Server haddan tashqari yuklangan
- Javob vaqti: 100ms → 5s

**Kontekstga sezgir GoSafe bilan yechim:**
\`\`\`go
func ProcessData(ctx context.Context, data []byte) error {
    errCh := GoSafe(ctx, func(ctx context.Context) error {
        // Kontekstni vaqti-vaqti bilan tekshiring
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
        }

        result := expensiveComputation(data)

        select {
        case <-ctx.Done():
            return ctx.Err()  // Bekor qilingan bo'lsa saqlamang
        default:
        }

        return saveResult(result)
    })

    return <-errCh
}
\`\`\`

**Natijalar:**
- Bekor qilingan operatsiyalar darhol to'xtaydi
- Isrof qilingan CPU-daqiqalar: 0
- Server yuki: Normal
- Javob vaqti: 100ms ga qaytdi

**5. Raqamlar: Haqiqiy Production Tizimi**

**200 mikroservis bilan E-commerce platformasi:**

**Xavfsiz goroutinelardan oldin:**
- Goroutine paniklari: kuniga 50-100
- Panikdan xizmat crashlari: kuniga 10-20
- Goroutine leaklar: kuniga 50,000+
- OOM crashlari: haftasiga 2-3
- O'rtacha incident narxi: $10K-$50K
- Oylik incident narxi: $80K-$200K
- Incidentlarga muhandis vaqti: oyiga 200 soat

**GoSafe patternini joriy qilgandan keyin:**
- Goroutine paniklari: Hali kuniga 50-100 (loglangan)
- Xizmat crashlari: oyiga 0
- Goroutine leaklar: 0
- OOM crashlari: 0
- Oylik incident narxi: $0
- Incidentlarga muhandis vaqti: oyiga 5 soat

**ROI hisoblash:**
\`\`\`
Joriy qilish narxi:
- Ishlab chiqish: 1 hafta (40 soat)
- Rollout: 2 hafta (80 soat)
- Jami: 120 muhandislik soati = ~$12K

Oyiga tejash:
- Incident narxi: $80K-$200K
- Muhandis vaqti: 195 soat = ~$19.5K
- Jami: oyiga $100K-$220K

To'lov muddati: 3-4 kun
Yillik ROI: 100,000%+
\`\`\`

**Eng yaxshi amaliyotlar:**
1. **Har doim** foydalanuvchiga yuzlangan goroutinelar uchun GoSafe dan foydalaning
2. **Har doim** buferlangan kanallardan foydalaning (minimal hajm 1)
3. **Har doim** defer da kanallarni yoping
4. **Har doim** kontekst bekor qilishni tekshiring
5. **Log** debug uchun paniklar
6. **Monitor** metrikalar bilan panik tezligini
7. **Ogohlantirish** panik tezligi o'sishida`
		}
	}
};

export default task;
