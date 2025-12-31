import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-panic-to-error',
	title: 'Panic to Error Conversion with Stack Trace',
	difficulty: 'hard',
	tags: ['go', 'panic', 'recover', 'stack-trace', 'debugging'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement advanced panic recovery that converts panics into detailed error values with stack traces for production debugging and monitoring.

**Requirements:**
1. **PanicError**: Custom error type containing panic value and stack trace
2. **CatchPanic**: Execute function and convert panic to PanicError
3. **PanicValue**: Extract original panic value from error
4. **Stack Trace**: Capture stack trace at panic point using runtime/debug

**Panic to Error Pattern:**
\`\`\`go
// Custom error type with stack trace
type PanicError struct {
    Value      interface{}  // Original panic value
    StackTrace string      // Stack trace at panic point
}

func (e *PanicError) Error() string {
    return fmt.Sprintf("panic: %v", e.Value)
}

// Convert panic to structured error
func CatchPanic(f func() error) error {
    var panicErr error

    func() {
        defer func() {
            if r := recover(); r != nil {
                panicErr = &PanicError{
                    Value:      r,
                    StackTrace: string(debug.Stack()),
                }
            }
        }()

        if f != nil {
            if err := f(); err != nil {
                panicErr = err
            }
        }
    }()

    return panicErr
}

// Check if error is from panic
func IsPanicError(err error) bool {
    if err == nil {
        return false
    }
    _, ok := err.(*PanicError)
    return ok
}

// Extract panic value from error
func PanicValue(err error) interface{} {
    if pe, ok := err.(*PanicError); ok {
        return pe.Value
    }
    return nil
}
\`\`\`

**Example Usage:**
\`\`\`go
// Example 1: Capturing panic details for logging
func ProcessUserData(data []byte) error {
    err := CatchPanic(func() error {
        // Code that might panic
        result := json.Unmarshal(data, &obj)
        return processResult(result)
    })

    if err != nil {
        if IsPanicError(err) {
            // Panic occurred - log with stack trace
            pe := err.(*PanicError)
            log.Printf("PANIC: %v\n%s", pe.Value, pe.StackTrace)

            // Send to error tracking service
            sentry.CaptureException(err)

            // Return generic error to user
            return errors.New("data processing failed")
        }
        // Regular error - return as-is
        return err
    }
    return nil
}

// Example 2: API handler with detailed error reporting
func (s *Server) HandleAPI(w http.ResponseWriter, r *http.Request) {
    err := CatchPanic(func() error {
        return s.processRequest(r)
    })

    if err != nil {
        if pe, ok := err.(*PanicError); ok {
            // Panic - log stack trace and return 500
            s.logger.Printf("Handler panic: %v\n%s", pe.Value, pe.StackTrace)
            s.metrics.IncrementPanicCounter("api_handler")

            http.Error(w, "Internal Server Error", 500)
            return
        }

        // Regular error - return 400
        http.Error(w, err.Error(), 400)
    }
}

// Example 3: Database operation with panic recovery
func (db *Database) QuerySafely(query string) ([]Row, error) {
    var rows []Row

    err := CatchPanic(func() error {
        var err error
        rows, err = db.Query(query)
        return err
    })

    if IsPanicError(err) {
        // Database panic - critical error
        db.logger.Printf("Database panic on query: %s", query)

        pe := err.(*PanicError)
        db.alertOps("Database Panic", string(pe.StackTrace))

        return nil, fmt.Errorf("database error: %w", err)
    }

    return rows, err
}
\`\`\`

**Real-World Production Scenario:**
\`\`\`go
// Panic tracking and alerting system
type PanicTracker struct {
    mu       sync.Mutex
    panics   map[string]int  // panic signature -> count
    logger   *log.Logger
    alerting *AlertService
    sentry   *sentry.Client
}

func (pt *PanicTracker) TrackPanic(component string, err error) {
    if !IsPanicError(err) {
        return
    }

    pe := err.(*PanicError)
    signature := fmt.Sprintf("%s:%v", component, pe.Value)

    pt.mu.Lock()
    pt.panics[signature]++
    count := pt.panics[signature]
    pt.mu.Unlock()

    // Log with full stack trace
    pt.logger.Printf(
        "PANIC [%s] (count: %d): %v\n%s",
        component,
        count,
        pe.Value,
        pe.StackTrace,
    )

    // Send to Sentry with grouping
    pt.sentry.CaptureException(err, map[string]string{
        "component": component,
        "signature": signature,
    })

    // Alert on repeated panics
    if count >= 10 {
        pt.alerting.SendAlert(fmt.Sprintf(
            "Repeated panic in %s: %v (count: %d)",
            component,
            pe.Value,
            count,
        ))
    }
}

// Middleware with panic tracking
func (pt *PanicTracker) PanicMiddleware(handler http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        err := CatchPanic(func() error {
            handler.ServeHTTP(w, r)
            return nil
        })

        if err != nil {
            pt.TrackPanic("http_handler", err)
            http.Error(w, "Internal Server Error", 500)
        }
    })
}

// Background job with panic recovery and reporting
type JobProcessor struct {
    tracker *PanicTracker
    metrics *Metrics
}

func (jp *JobProcessor) ProcessJob(job Job) error {
    startTime := time.Now()

    err := CatchPanic(func() error {
        return job.Execute()
    })

    duration := time.Since(startTime)
    jp.metrics.RecordJobDuration(job.Type, duration)

    if IsPanicError(err) {
        // Track panic statistics
        jp.tracker.TrackPanic(job.Type, err)
        jp.metrics.IncrementJobPanic(job.Type)

        pe := err.(*PanicError)

        // Store in dead letter queue with details
        jp.deadLetterQueue.Add(DeadLetter{
            Job:        job,
            PanicValue: pe.Value,
            StackTrace: pe.StackTrace,
            Timestamp:  time.Now(),
        })

        return fmt.Errorf("job panicked: %w", err)
    }

    if err != nil {
        jp.metrics.IncrementJobError(job.Type)
        return err
    }

    jp.metrics.IncrementJobSuccess(job.Type)
    return nil
}
\`\`\`

**Stack Trace Analysis:**
\`\`\`go
// Parse stack trace for debugging
func AnalyzeStackTrace(err error) *StackInfo {
    if !IsPanicError(err) {
        return nil
    }

    pe := err.(*PanicError)
    lines := strings.Split(pe.StackTrace, "\n")

    info := &StackInfo{
        PanicValue: pe.Value,
    }

    // Find panic origin (first non-runtime frame)
    for i, line := range lines {
        if strings.Contains(line, ".go:") {
            if !strings.Contains(line, "/runtime/") {
                // Found application code
                info.OriginFile = extractFilePath(line)
                info.OriginLine = extractLineNumber(line)

                if i > 0 {
                    info.OriginFunc = strings.TrimSpace(lines[i-1])
                }
                break
            }
        }
    }

    return info
}

// Automated panic classification
func ClassifyPanic(err error) string {
    val := PanicValue(err)
    if val == nil {
        return "unknown"
    }

    switch v := val.(type) {
    case *runtime.TypeAssertionError:
        return "type_assertion"
    case string:
        if strings.Contains(v, "nil pointer") {
            return "nil_pointer"
        }
        if strings.Contains(v, "index out of range") {
            return "index_out_of_bounds"
        }
        if strings.Contains(v, "slice bounds") {
            return "slice_bounds"
        }
        if strings.Contains(v, "divide by zero") {
            return "division_by_zero"
        }
    }

    return "runtime_error"
}
\`\`\`

**Testing and Debugging:**
\`\`\`go
// Test helper for panic scenarios
func TestPanicRecovery(t *testing.T) {
    tests := []struct {
        name      string
        fn        func() error
        wantPanic bool
        panicVal  interface{}
    }{
        {
            name: "nil pointer panic",
            fn: func() error {
                var p *int
                _ = *p
                return nil
            },
            wantPanic: true,
        },
        {
            name: "regular error",
            fn: func() error {
                return errors.New("regular error")
            },
            wantPanic: false,
        },
        {
            name: "success",
            fn: func() error {
                return nil
            },
            wantPanic: false,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := CatchPanic(tt.fn)

            if tt.wantPanic && !IsPanicError(err) {
                t.Error("expected panic error, got none")
            }

            if !tt.wantPanic && IsPanicError(err) {
                t.Errorf("unexpected panic: %v", err)
            }

            if IsPanicError(err) {
                pe := err.(*PanicError)
                if pe.StackTrace == "" {
                    t.Error("missing stack trace")
                }
            }
        })
    }
}
\`\`\`

**Constraints:**
- PanicError must implement error interface
- CatchPanic must capture stack trace using debug.Stack()
- IsPanicError must safely check error type
- PanicValue must return nil for non-panic errors
- Stack trace must be captured at panic point, not recovery point`,
	initialCode: `package panicrecover

import (
	"fmt"
	"runtime/debug"
)

// TODO: Define PanicError struct
// Must contain Value interface{} and StackTrace string
// Must implement Error() method returning "panic: %v" formatted string
type PanicError struct {
	// TODO: Add fields
}

// TODO: Implement Error method for PanicError
func (e *PanicError) Error() string {
	return "" // TODO: Implement
}

// TODO: Implement CatchPanic
// Execute f and recover from panics
// If panic occurs, return &PanicError with value and debug.Stack()
// If f returns error, return that error
// If success, return nil
func CatchPanic(f func() error) error {
	// TODO: Implement
}

// TODO: Implement IsPanicError
// Check if err is a *PanicError using type assertion
// Return false if err is nil
func IsPanicError(err error) bool {
	return false // TODO: Implement
}

// TODO: Implement PanicValue
// Extract panic value from PanicError
// Return nil if not a PanicError
func PanicValue(err error) interface{} {
	// TODO: Implement
}`,
	solutionCode: `package panicrecover

import (
	"fmt"
	"runtime/debug"
)

type PanicError struct {
	Value      interface{}                              // original panic value
	StackTrace string                                   // stack trace at panic
}

func (e *PanicError) Error() string {
	return fmt.Sprintf("panic: %v", e.Value)            // format as error
}

func CatchPanic(f func() error) error {
	var panicErr error                                   // capture error

	func() {                                             // anonymous function for defer
		defer func() {
			if r := recover(); r != nil {                // catch panic
				panicErr = &PanicError{
					Value:      r,                       // store panic value
					StackTrace: string(debug.Stack()),   // capture stack trace
				}
			}
		}()

		if f != nil {
			if err := f(); err != nil {                  // execute and check error
				panicErr = err
			}
		}
	}()

	return panicErr                                      // return captured error
}

func IsPanicError(err error) bool {
	if err == nil {
		return false                                     // nil is not panic error
	}
	_, ok := err.(*PanicError)                           // type assertion
	return ok
}

func PanicValue(err error) interface{} {
	if pe, ok := err.(*PanicError); ok {                 // extract if PanicError
		return pe.Value
	}
	return nil                                           // not a panic error
}`,
	hint1: `Define PanicError struct with Value and StackTrace fields. Implement Error() method returning fmt.Sprintf("panic: %v", e.Value).`,
	hint2: `In CatchPanic: use nested anonymous function with defer recover(). On panic, create &PanicError{Value: r, StackTrace: string(debug.Stack())}. Check f() return error separately.`,
	testCode: `package panicrecover

import (
	"errors"
	"strings"
	"testing"
)

func Test1(t *testing.T) {
	// Test PanicError Error method
	pe := &PanicError{Value: "test panic", StackTrace: "stack..."}
	if pe.Error() != "panic: test panic" {
		t.Errorf("PanicError.Error() = %q, want %q", pe.Error(), "panic: test panic")
	}
}

func Test2(t *testing.T) {
	// Test CatchPanic with nil function
	err := CatchPanic(nil)
	if err != nil {
		t.Errorf("CatchPanic(nil) = %v, want nil", err)
	}
}

func Test3(t *testing.T) {
	// Test CatchPanic with successful function
	err := CatchPanic(func() error {
		return nil
	})
	if err != nil {
		t.Errorf("CatchPanic(success) = %v, want nil", err)
	}
}

func Test4(t *testing.T) {
	// Test CatchPanic with function returning error
	testErr := errors.New("test error")
	err := CatchPanic(func() error {
		return testErr
	})
	if err != testErr {
		t.Errorf("CatchPanic(error) = %v, want %v", err, testErr)
	}
}

func Test5(t *testing.T) {
	// Test CatchPanic with panicking function
	err := CatchPanic(func() error {
		panic("catch panic test")
	})
	if err == nil {
		t.Error("CatchPanic(panic) = nil, want error")
	}
	pe, ok := err.(*PanicError)
	if !ok {
		t.Errorf("CatchPanic(panic) type = %T, want *PanicError", err)
	}
	if pe.Value != "catch panic test" {
		t.Errorf("PanicError.Value = %v, want %q", pe.Value, "catch panic test")
	}
	if pe.StackTrace == "" {
		t.Error("PanicError.StackTrace is empty, want stack trace")
	}
}

func Test6(t *testing.T) {
	// Test IsPanicError with nil
	if IsPanicError(nil) {
		t.Error("IsPanicError(nil) = true, want false")
	}
}

func Test7(t *testing.T) {
	// Test IsPanicError with regular error
	regularErr := errors.New("regular error")
	if IsPanicError(regularErr) {
		t.Error("IsPanicError(regular) = true, want false")
	}
}

func Test8(t *testing.T) {
	// Test IsPanicError with PanicError
	pe := &PanicError{Value: "test"}
	if !IsPanicError(pe) {
		t.Error("IsPanicError(PanicError) = false, want true")
	}
}

func Test9(t *testing.T) {
	// Test PanicValue with non-panic error
	regularErr := errors.New("regular error")
	val := PanicValue(regularErr)
	if val != nil {
		t.Errorf("PanicValue(regular) = %v, want nil", val)
	}
}

func Test10(t *testing.T) {
	// Test PanicValue with PanicError
	pe := &PanicError{Value: 42, StackTrace: "stack"}
	val := PanicValue(pe)
	if val != 42 {
		t.Errorf("PanicValue(PanicError) = %v, want 42", val)
	}
}`,
	whyItMatters: `Converting panics to structured errors with stack traces is essential for debugging production issues and building observable systems that provide actionable insights.

**Why This Matters:**

**1. Production Incident: Mystery Crash Investigation**

An e-commerce platform experienced intermittent crashes:
- Service crashed 10-15 times per day
- Logs showed only "panic recovered" messages
- No stack traces captured
- Engineers couldn't identify root cause
- 3 weeks of investigation yielded nothing
- Cost: 100+ engineering hours

**Root Cause:**
\`\`\`go
// Before: No stack trace captured
func ProcessOrder(order Order) error {
    defer func() {
        if r := recover() {
            log.Printf("panic: %v", r)  // Just the panic value
            // Where did it panic? Unknown!
        }
    }()

    processPayment(order)
    updateInventory(order)
    sendEmail(order)
    return nil
}
// Result: Know panic occurred, but not WHERE in code
\`\`\`

**Solution with PanicError:**
\`\`\`go
// After: Full stack trace captured
err := CatchPanic(func() error {
    processPayment(order)
    updateInventory(order)
    sendEmail(order)
    return nil
})

if pe, ok := err.(*PanicError); ok {
    // Log full stack trace
    log.Printf("Panic: %v\n%s", pe.Value, pe.StackTrace)

    // Output:
    // panic: nil pointer dereference
    // goroutine 42 [running]:
    // main.updateInventory(...)
    //     /app/inventory.go:123
    // main.ProcessOrder(...)
    //     /app/order.go:45
}
// Result: Identified exact line causing panic in 5 minutes
\`\`\`

**Impact After Fix:**
- Bug found in inventory.go line 123 (nil map access)
- Fixed in 30 minutes
- Zero crashes after fix
- Investigation time: 3 weeks → 5 minutes

**2. Real-World: Sentry Integration**

SaaS platform with 1M+ users:

**Before PanicError:**
- Panics logged but not sent to error tracking
- Errors grouped by generic "panic recovered" message
- No stack traces in Sentry
- Unable to prioritize critical panics

**After PanicError Implementation:**
\`\`\`go
func (s *Server) HandleRequest(r Request) error {
    err := CatchPanic(func() error {
        return s.process(r)
    })

    if IsPanicError(err) {
        // Send to Sentry with stack trace
        pe := err.(*PanicError)
        sentry.CaptureException(err, map[string]interface{}{
            "panic_value": pe.Value,
            "stack_trace": pe.StackTrace,
            "user_id":     r.UserID,
            "endpoint":    r.Endpoint,
        })

        // Classify panic
        category := classifyPanic(pe)
        metrics.IncrementPanic(category)

        return errors.New("request failed")
    }

    return err
}
\`\`\`

**Results:**
- All panics visible in Sentry dashboard
- Panics grouped by stack trace (not generic message)
- Top 5 panic types identified immediately
- Prioritized fixes by frequency and impact

**Panic Statistics (First Week):**
- nil pointer dereference: 450 (40%) → Fixed in 2 days
- index out of bounds: 280 (25%) → Fixed in 1 day
- type assertion: 220 (19%) → Fixed in 3 days
- map concurrent access: 180 (16%) → Fixed in 1 day

**Impact:**
- Panics reduced by 90% in 1 week
- Mean time to fix: 2 days (from unknown)
- Customer complaints: 50/week → 5/week

**3. Production Metrics: Panic Analysis**

Financial services API (100M requests/day):

**Before PanicError:**
- 100-200 panics per day
- Engineers manually grep logs for stack traces
- Average debug time per panic: 2-4 hours
- No panic trending or classification

**After PanicError with Analysis:**
\`\`\`go
// Automated panic classification
type PanicStats struct {
    Category      string
    Count         int
    FirstSeen     time.Time
    LastSeen      time.Time
    AffectedUsers int
    ExampleTrace  string
}

func (pt *PanicTracker) AnalyzePanic(err error) {
    pe := err.(*PanicError)

    // Classify panic type
    category := ClassifyPanic(err)

    // Extract origin location
    info := AnalyzeStackTrace(err)

    pt.mu.Lock()
    stats := pt.stats[category]
    stats.Count++
    stats.LastSeen = time.Now()

    if stats.FirstSeen.IsZero() {
        stats.FirstSeen = time.Now()
        stats.ExampleTrace = pe.StackTrace
    }

    pt.stats[category] = stats
    pt.mu.Unlock()

    // Alert on new panic types
    if stats.Count == 1 {
        pt.alerting.Send(fmt.Sprintf(
            "NEW PANIC TYPE: %s\n%s\nLocation: %s:%d",
            category,
            pe.Value,
            info.OriginFile,
            info.OriginLine,
        ))
    }
}
\`\`\`

**Results:**
- Automatic panic classification and trending
- Identify new panic types immediately (alerts within 1 minute)
- Track panic "hotspots" in codebase
- Debug time: 2-4 hours → 15-30 minutes (avg)

**Panic Dashboard Metrics:**
\`\`\`
Top 5 Panic Categories (Last 30 Days):
1. nil_pointer: 1,234 occurrences (65% in cache.go:89)
2. index_out_of_bounds: 456 (40% in parser.go:234)
3. type_assertion: 234 (80% in handler.go:123)
4. division_by_zero: 89 (95% in metrics.go:456)
5. slice_bounds: 45 (100% in validator.go:78)

Action: Fixed top 3 → 90% panic reduction
\`\`\`

**4. Dead Letter Queue with Panic Details**

Message processing system handling 1M messages/day:

\`\`\`go
// Before: Lost panic details
if err := processMessage(msg); err != nil {
    deadLetterQueue.Add(msg)  // Just the message
    // No info about WHY it failed
}

// After: Preserve panic details
err := CatchPanic(func() error {
    return processMessage(msg)
})

if IsPanicError(err) {
    pe := err.(*PanicError)

    // Store message with full panic context
    deadLetterQueue.Add(DeadLetter{
        Message:    msg,
        PanicValue: pe.Value,
        StackTrace: pe.StackTrace,
        Timestamp:  time.Now(),
    })

    // Engineers can replay with exact context
}
\`\`\`

**Impact:**
- Failed messages include WHY they failed
- Replay failures with fix applied
- Debug time per message: 30 min → 5 min
- Recovery rate: 60% → 95%

**5. The Numbers: Real Production System**

**API Gateway (500M requests/day):**

**Before PanicError:**
- Panic debugging time: 100-200 hours/month
- Unresolved panics: 20-30
- Customer impact: High (unexplained failures)
- Engineering cost: $20K-$40K/month

**After PanicError Implementation:**
- Panic debugging time: 10-20 hours/month
- Unresolved panics: 0-2
- Customer impact: Low (fast fixes)
- Engineering cost: $2K-$4K/month

**ROI Calculation:**
\`\`\`
Implementation: 2 weeks (80 hours) = ~$8K

Savings per month:
- Engineering time: 80-180 hours = $8K-$18K
- Faster fixes: Reduced customer churn = $10K-$20K
- Total: $18K-$38K per month

Payback period: 2 weeks
Annual ROI: 500-1000%
\`\`\`

**Best Practices:**
1. **Always capture stack traces** with debug.Stack()
2. **Send to error tracking** (Sentry, Datadog, etc.)
3. **Classify panic types** for trending
4. **Alert on new panic types** immediately
5. **Include context** (user ID, request ID, etc.)
6. **Store in dead letter queue** for replay
7. **Build panic dashboard** for visibility`,
	order: 2,
	translations: {
		ru: {
			title: 'Преобразование panic в ошибку',
			solutionCode: `package panicrecover

import (
	"fmt"
	"runtime/debug"
)

type PanicError struct {
	Value      interface{}                              // оригинальное значение паники
	StackTrace string                                   // стек вызовов при панике
}

func (e *PanicError) Error() string {
	return fmt.Sprintf("panic: %v", e.Value)            // форматируем как ошибку
}

func CatchPanic(f func() error) error {
	var panicErr error                                   // захватываем ошибку

	func() {                                             // анонимная функция для defer
		defer func() {
			if r := recover(); r != nil {                // перехватываем панику
				panicErr = &PanicError{
					Value:      r,                       // сохраняем значение паники
					StackTrace: string(debug.Stack()),   // захватываем стек вызовов
				}
			}
		}()

		if f != nil {
			if err := f(); err != nil {                  // выполняем и проверяем ошибку
				panicErr = err
			}
		}
	}()

	return panicErr                                      // возвращаем захваченную ошибку
}

func IsPanicError(err error) bool {
	if err == nil {
		return false                                     // nil не ошибка паники
	}
	_, ok := err.(*PanicError)                           // приведение типа
	return ok
}

func PanicValue(err error) interface{} {
	if pe, ok := err.(*PanicError); ok {                 // извлекаем если PanicError
		return pe.Value
	}
	return nil                                           // не ошибка паники
}`,
			description: `Реализуйте продвинутое восстановление от паники, которое конвертирует панику в детализированные ошибки со стеком вызовов для production отладки и мониторинга.

**Требования:**
1. **PanicError**: Пользовательский тип ошибки содержащий значение паники и стек вызовов
2. **CatchPanic**: Выполнение функции и конвертация паники в PanicError
3. **PanicValue**: Извлечение оригинального значения паники из ошибки
4. **Stack Trace**: Захват стека вызовов в точке паники используя runtime/debug

**Паттерн конвертации panic в error:**
\`\`\`go
// Пользовательский тип ошибки со стеком вызовов
type PanicError struct {
    Value      interface{}  // Оригинальное значение паники
    StackTrace string      // Стек вызовов в точке паники
}

func (e *PanicError) Error() string {
    return fmt.Sprintf("panic: %v", e.Value)
}

// Конвертация паники в структурированную ошибку
func CatchPanic(f func() error) error {
    var panicErr error

    func() {
        defer func() {
            if r := recover(); r != nil {
                panicErr = &PanicError{
                    Value:      r,
                    StackTrace: string(debug.Stack()),
                }
            }
        }()

        if f != nil {
            if err := f(); err != nil {
                panicErr = err
            }
        }
    }()

    return panicErr
}

// Проверка является ли ошибка результатом паники
func IsPanicError(err error) bool {
    if err == nil {
        return false
    }
    _, ok := err.(*PanicError)
    return ok
}

// Извлечение значения паники из ошибки
func PanicValue(err error) interface{} {
    if pe, ok := err.(*PanicError); ok {
        return pe.Value
    }
    return nil
}
\`\`\`

**Примеры использования:**
\`\`\`go
// Пример 1: Захват деталей паники для логирования
func ProcessUserData(data []byte) error {
    err := CatchPanic(func() error {
        // Код который может паниковать
        result := json.Unmarshal(data, &obj)
        return processResult(result)
    })

    if err != nil {
        if IsPanicError(err) {
            // Паника произошла - логируем со стеком вызовов
            pe := err.(*PanicError)
            log.Printf("ПАНИКА: %v\n%s", pe.Value, pe.StackTrace)

            // Отправляем в сервис отслеживания ошибок
            sentry.CaptureException(err)

            // Возвращаем общую ошибку пользователю
            return errors.New("обработка данных не удалась")
        }
        // Обычная ошибка - возвращаем как есть
        return err
    }
    return nil
}

// Пример 2: API обработчик с детальным отчетом об ошибках
func (s *Server) HandleAPI(w http.ResponseWriter, r *http.Request) {
    err := CatchPanic(func() error {
        return s.processRequest(r)
    })

    if err != nil {
        if pe, ok := err.(*PanicError); ok {
            // Паника - логируем стек вызовов и возвращаем 500
            s.logger.Printf("Паника обработчика: %v\n%s", pe.Value, pe.StackTrace)
            s.metrics.IncrementPanicCounter("api_handler")

            http.Error(w, "Internal Server Error", 500)
            return
        }

        // Обычная ошибка - возвращаем 400
        http.Error(w, err.Error(), 400)
    }
}

// Пример 3: Операция с БД с восстановлением от паники
func (db *Database) QuerySafely(query string) ([]Row, error) {
    var rows []Row

    err := CatchPanic(func() error {
        var err error
        rows, err = db.Query(query)
        return err
    })

    if IsPanicError(err) {
        // Паника БД - критическая ошибка
        db.logger.Printf("Паника базы данных на запросе: %s", query)

        pe := err.(*PanicError)
        db.alertOps("Паника БД", string(pe.StackTrace))

        return nil, fmt.Errorf("ошибка базы данных: %w", err)
    }

    return rows, err
}
\`\`\`

**Реальный Production сценарий:**
\`\`\`go
// Система отслеживания и оповещения о паниках
type PanicTracker struct {
    mu       sync.Mutex
    panics   map[string]int  // сигнатура паники -> счетчик
    logger   *log.Logger
    alerting *AlertService
    sentry   *sentry.Client
}

func (pt *PanicTracker) TrackPanic(component string, err error) {
    if !IsPanicError(err) {
        return
    }

    pe := err.(*PanicError)
    signature := fmt.Sprintf("%s:%v", component, pe.Value)

    pt.mu.Lock()
    pt.panics[signature]++
    count := pt.panics[signature]
    pt.mu.Unlock()

    // Логируем с полным стеком вызовов
    pt.logger.Printf(
        "ПАНИКА [%s] (счетчик: %d): %v\n%s",
        component,
        count,
        pe.Value,
        pe.StackTrace,
    )

    // Отправляем в Sentry с группировкой
    pt.sentry.CaptureException(err, map[string]string{
        "component": component,
        "signature": signature,
    })

    // Оповещаем при повторяющихся паниках
    if count >= 10 {
        pt.alerting.SendAlert(fmt.Sprintf(
            "Повторяющаяся паника в %s: %v (счетчик: %d)",
            component,
            pe.Value,
            count,
        ))
    }
}

// Middleware с отслеживанием паники
func (pt *PanicTracker) PanicMiddleware(handler http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        err := CatchPanic(func() error {
            handler.ServeHTTP(w, r)
            return nil
        })

        if err != nil {
            pt.TrackPanic("http_handler", err)
            http.Error(w, "Internal Server Error", 500)
        }
    })
}

// Фоновая задача с восстановлением от паники и отчетами
type JobProcessor struct {
    tracker *PanicTracker
    metrics *Metrics
}

func (jp *JobProcessor) ProcessJob(job Job) error {
    startTime := time.Now()

    err := CatchPanic(func() error {
        return job.Execute()
    })

    duration := time.Since(startTime)
    jp.metrics.RecordJobDuration(job.Type, duration)

    if IsPanicError(err) {
        // Отслеживаем статистику паник
        jp.tracker.TrackPanic(job.Type, err)
        jp.metrics.IncrementJobPanic(job.Type)

        pe := err.(*PanicError)

        // Сохраняем в очередь мертвых писем с деталями
        jp.deadLetterQueue.Add(DeadLetter{
            Job:        job,
            PanicValue: pe.Value,
            StackTrace: pe.StackTrace,
            Timestamp:  time.Now(),
        })

        return fmt.Errorf("задача запаниковала: %w", err)
    }

    if err != nil {
        jp.metrics.IncrementJobError(job.Type)
        return err
    }

    jp.metrics.IncrementJobSuccess(job.Type)
    return nil
}
\`\`\`

**Анализ стека вызовов:**
\`\`\`go
// Парсинг стека вызовов для отладки
func AnalyzeStackTrace(err error) *StackInfo {
    if !IsPanicError(err) {
        return nil
    }

    pe := err.(*PanicError)
    lines := strings.Split(pe.StackTrace, "\n")

    info := &StackInfo{
        PanicValue: pe.Value,
    }

    // Находим источник паники (первый не-runtime фрейм)
    for i, line := range lines {
        if strings.Contains(line, ".go:") {
            if !strings.Contains(line, "/runtime/") {
                // Найден код приложения
                info.OriginFile = extractFilePath(line)
                info.OriginLine = extractLineNumber(line)

                if i > 0 {
                    info.OriginFunc = strings.TrimSpace(lines[i-1])
                }
                break
            }
        }
    }

    return info
}

// Автоматическая классификация паники
func ClassifyPanic(err error) string {
    val := PanicValue(err)
    if val == nil {
        return "unknown"
    }

    switch v := val.(type) {
    case *runtime.TypeAssertionError:
        return "type_assertion"
    case string:
        if strings.Contains(v, "nil pointer") {
            return "nil_pointer"
        }
        if strings.Contains(v, "index out of range") {
            return "index_out_of_bounds"
        }
        if strings.Contains(v, "slice bounds") {
            return "slice_bounds"
        }
        if strings.Contains(v, "divide by zero") {
            return "division_by_zero"
        }
    }

    return "runtime_error"
}
\`\`\`

**Тестирование и отладка:**
\`\`\`go
// Тестовый помощник для сценариев паники
func TestPanicRecovery(t *testing.T) {
    tests := []struct {
        name      string
        fn        func() error
        wantPanic bool
        panicVal  interface{}
    }{
        {
            name: "nil pointer panic",
            fn: func() error {
                var p *int
                _ = *p
                return nil
            },
            wantPanic: true,
        },
        {
            name: "обычная ошибка",
            fn: func() error {
                return errors.New("обычная ошибка")
            },
            wantPanic: false,
        },
        {
            name: "успех",
            fn: func() error {
                return nil
            },
            wantPanic: false,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := CatchPanic(tt.fn)

            if tt.wantPanic && !IsPanicError(err) {
                t.Error("ожидалась ошибка паники, не получена")
            }

            if !tt.wantPanic && IsPanicError(err) {
                t.Errorf("неожиданная паника: %v", err)
            }

            if IsPanicError(err) {
                pe := err.(*PanicError)
                if pe.StackTrace == "" {
                    t.Error("отсутствует стек вызовов")
                }
            }
        })
    }
}
\`\`\`

**Ограничения:**
- PanicError должен реализовывать интерфейс error
- CatchPanic должен захватывать стек вызовов используя debug.Stack()
- IsPanicError должен безопасно проверять тип ошибки
- PanicValue должен возвращать nil для не-паник ошибок
- Стек вызовов должен захватываться в точке паники, а не восстановления`,
			hint1: `Определите структуру PanicError с полями Value и StackTrace. Реализуйте метод Error() возвращающий fmt.Sprintf("panic: %v", e.Value).`,
			hint2: `В CatchPanic: используйте вложенную анонимную функцию с defer recover(). При панике создайте &PanicError{Value: r, StackTrace: string(debug.Stack())}. Проверьте ошибку возврата f() отдельно.`,
			whyItMatters: `Конвертация паник в структурированные ошибки со стеком вызовов критически важна для отладки production проблем и построения наблюдаемых систем, которые предоставляют действенные инсайты.

**Почему это важно:**

**1. Production инцидент: Расследование загадочного краша**

E-commerce платформа испытывала периодические краши:
- Сервис падал 10-15 раз в день
- Логи показывали только сообщения "panic recovered"
- Стеки вызовов не захватывались
- Инженеры не могли определить корневую причину
- 3 недели расследования не дали результатов
- Стоимость: 100+ инженерных часов

**Корневая причина:**
\`\`\`go
// До: Стек вызовов не захвачен
func ProcessOrder(order Order) error {
    defer func() {
        if r := recover() {
            log.Printf("паника: %v", r)  // Только значение паники
            // ГДЕ это произошло? Неизвестно!
        }
    }()

    processPayment(order)
    updateInventory(order)
    sendEmail(order)
    return nil
}
// Результат: Знаем что паника произошла, но не ГДЕ в коде
\`\`\`

**Решение с PanicError:**
\`\`\`go
// После: Захвачен полный стек вызовов
err := CatchPanic(func() error {
    processPayment(order)
    updateInventory(order)
    sendEmail(order)
    return nil
})

if pe, ok := err.(*PanicError); ok {
    // Логируем полный стек вызовов
    log.Printf("Паника: %v\n%s", pe.Value, pe.StackTrace)

    // Вывод:
    // паника: разыменование nil указателя
    // горутина 42 [running]:
    // main.updateInventory(...)
    //     /app/inventory.go:123
    // main.ProcessOrder(...)
    //     /app/order.go:45
}
// Результат: Идентифицирована точная строка вызывающая панику за 5 минут
\`\`\`

**Результат после исправления:**
- Баг найден в inventory.go строка 123 (доступ к nil map)
- Исправлен за 30 минут
- Ноль крашей после исправления
- Время расследования: 3 недели → 5 минут

**2. Реальный случай: Интеграция Sentry**

SaaS платформа с 1M+ пользователей:

**До PanicError:**
- Паники логировались, но не отправлялись в систему отслеживания ошибок
- Ошибки группировались по общему сообщению "panic recovered"
- Нет стеков вызовов в Sentry
- Невозможно приоритизировать критические паники

**После внедрения PanicError:**
\`\`\`go
func (s *Server) HandleRequest(r Request) error {
    err := CatchPanic(func() error {
        return s.process(r)
    })

    if IsPanicError(err) {
        // Отправляем в Sentry со стеком вызовов
        pe := err.(*PanicError)
        sentry.CaptureException(err, map[string]interface{}{
            "panic_value": pe.Value,
            "stack_trace": pe.StackTrace,
            "user_id":     r.UserID,
            "endpoint":    r.Endpoint,
        })

        // Классифицируем панику
        category := classifyPanic(pe)
        metrics.IncrementPanic(category)

        return errors.New("запрос не удался")
    }

    return err
}
\`\`\`

**Результаты:**
- Все паники видимы в панели Sentry
- Паники группируются по стеку вызовов (не общее сообщение)
- Топ 5 типов паник идентифицированы немедленно
- Приоритизированы исправления по частоте и влиянию

**Статистика паник (Первая неделя):**
- разыменование nil указателя: 450 (40%) → Исправлено за 2 дня
- выход за границы индекса: 280 (25%) → Исправлено за 1 день
- приведение типа: 220 (19%) → Исправлено за 3 дня
- конкурентный доступ к map: 180 (16%) → Исправлено за 1 день

**Влияние:**
- Паники снижены на 90% за 1 неделю
- Среднее время исправления: 2 дня (было неизвестно)
- Жалобы клиентов: 50/неделю → 5/неделю

**3. Production метрики: Анализ паники**

Финансовый API сервис (100M запросов/день):

**До PanicError:**
- 100-200 паник в день
- Инженеры вручную просматривали логи для стеков вызовов
- Среднее время отладки на панику: 2-4 часа
- Нет трендов или классификации паник

**После PanicError с анализом:**
\`\`\`go
// Автоматическая классификация паники
type PanicStats struct {
    Category      string
    Count         int
    FirstSeen     time.Time
    LastSeen      time.Time
    AffectedUsers int
    ExampleTrace  string
}

func (pt *PanicTracker) AnalyzePanic(err error) {
    pe := err.(*PanicError)

    // Классифицируем тип паники
    category := ClassifyPanic(err)

    // Извлекаем местоположение источника
    info := AnalyzeStackTrace(err)

    pt.mu.Lock()
    stats := pt.stats[category]
    stats.Count++
    stats.LastSeen = time.Now()

    if stats.FirstSeen.IsZero() {
        stats.FirstSeen = time.Now()
        stats.ExampleTrace = pe.StackTrace
    }

    pt.stats[category] = stats
    pt.mu.Unlock()

    // Оповещаем о новых типах паник
    if stats.Count == 1 {
        pt.alerting.Send(fmt.Sprintf(
            "НОВЫЙ ТИП ПАНИКИ: %s\n%s\nМестоположение: %s:%d",
            category,
            pe.Value,
            info.OriginFile,
            info.OriginLine,
        ))
    }
}
\`\`\`

**Результаты:**
- Автоматическая классификация и тренды паник
- Идентификация новых типов паник немедленно (оповещения в течение 1 минуты)
- Отслеживание "горячих точек" паник в кодовой базе
- Время отладки: 2-4 часа → 15-30 минут (среднее)

**Метрики панели паник:**
\`\`\`
Топ 5 категорий паник (Последние 30 дней):
1. nil_pointer: 1,234 случаев (65% в cache.go:89)
2. index_out_of_bounds: 456 (40% в parser.go:234)
3. type_assertion: 234 (80% в handler.go:123)
4. division_by_zero: 89 (95% в metrics.go:456)
5. slice_bounds: 45 (100% в validator.go:78)

Действие: Исправлен топ 3 → 90% снижение паник
\`\`\`

**4. Dead Letter Queue с деталями паники**

Система обработки сообщений обрабатывает 1M сообщений/день:

\`\`\`go
// До: Детали паники потеряны
if err := processMessage(msg); err != nil {
    deadLetterQueue.Add(msg)  // Только сообщение
    // Нет информации о том ПОЧЕМУ это не удалось
}

// После: Сохранены детали паники
err := CatchPanic(func() error {
    return processMessage(msg)
})

if IsPanicError(err) {
    pe := err.(*PanicError)

    // Сохраняем сообщение с полным контекстом паники
    deadLetterQueue.Add(DeadLetter{
        Message:    msg,
        PanicValue: pe.Value,
        StackTrace: pe.StackTrace,
        Timestamp:  time.Now(),
    })

    // Инженеры могут воспроизвести с точным контекстом
}
\`\`\`

**Влияние:**
- Неудачные сообщения включают ПОЧЕМУ они не удались
- Воспроизведение сбоев с примененным исправлением
- Время отладки на сообщение: 30 мин → 5 мин
- Процент восстановления: 60% → 95%

**5. Цифры: Реальная Production система**

**API Gateway (500M запросов/день):**

**До PanicError:**
- Время отладки паник: 100-200 часов/месяц
- Нерешенные паники: 20-30
- Влияние на клиентов: Высокое (необъяснимые сбои)
- Стоимость инженерии: $20K-$40K/месяц

**После внедрения PanicError:**
- Время отладки паник: 10-20 часов/месяц
- Нерешенные паники: 0-2
- Влияние на клиентов: Низкое (быстрые исправления)
- Стоимость инженерии: $2K-$4K/месяц

**Расчет ROI:**
\`\`\`
Внедрение: 2 недели (80 часов) = ~$8K

Экономия в месяц:
- Время инженеров: 80-180 часов = $8K-$18K
- Быстрые исправления: Снижение оттока клиентов = $10K-$20K
- Всего: $18K-$38K в месяц

Срок окупаемости: 2 недели
Годовой ROI: 500-1000%
\`\`\`

**Лучшие практики:**
1. **Всегда захватывайте стеки вызовов** с debug.Stack()
2. **Отправляйте в систему отслеживания ошибок** (Sentry, Datadog и т.д.)
3. **Классифицируйте типы паник** для трендов
4. **Оповещайте о новых типах паник** немедленно
5. **Включайте контекст** (user ID, request ID и т.д.)
6. **Сохраняйте в dead letter queue** для воспроизведения
7. **Создайте панель паник** для видимости`
		},
		uz: {
			title: `Panicni xatoga aylantirish`,
			solutionCode: `package panicrecover

import (
	"fmt"
	"runtime/debug"
)

type PanicError struct {
	Value      interface{}                              // asl panik qiymati
	StackTrace string                                   // panikdagi stack trace
}

func (e *PanicError) Error() string {
	return fmt.Sprintf("panic: %v", e.Value)            // xato sifatida formatlaymiz
}

func CatchPanic(f func() error) error {
	var panicErr error                                   // xatoni ushlaymiz

	func() {                                             // defer uchun anonim funksiya
		defer func() {
			if r := recover(); r != nil {                // panikni ushlaymiz
				panicErr = &PanicError{
					Value:      r,                       // panik qiymatini saqlaymiz
					StackTrace: string(debug.Stack()),   // stack trace ni ushlaymiz
				}
			}
		}()

		if f != nil {
			if err := f(); err != nil {                  // bajaramiz va xatoni tekshiramiz
				panicErr = err
			}
		}
	}()

	return panicErr                                      // ushlangan xatoni qaytaramiz
}

func IsPanicError(err error) bool {
	if err == nil {
		return false                                     // nil panik xatosi emas
	}
	_, ok := err.(*PanicError)                           // type assertion
	return ok
}

func PanicValue(err error) interface{} {
	if pe, ok := err.(*PanicError); ok {                 // PanicError bo'lsa chiqarib olamiz
		return pe.Value
	}
	return nil                                           // panik xatosi emas
}`,
			description: `Production debugging va monitoring uchun stack trace bilan paniklarni batafsil xato qiymatlariga aylantiradigan ilg'or panikdan tiklanishni amalga oshiring.

**Talablar:**
1. **PanicError**: Panik qiymatini va stack trace ni o'z ichiga olgan custom xato turi
2. **CatchPanic**: Funksiyani bajaring va panikni PanicError ga aylantiring
3. **PanicValue**: Xatodan asl panik qiymatini chiqarib oling
4. **Stack Trace**: runtime/debug dan foydalanib panik nuqtasida stack trace ni ushlang

**Panic dan Error ga aylantirish patterni:**
\`\`\`go
// Stack trace bilan custom xato turi
type PanicError struct {
    Value      interface{}  // Asl panik qiymati
    StackTrace string      // Panik nuqtasida stack trace
}

func (e *PanicError) Error() string {
    return fmt.Sprintf("panic: %v", e.Value)
}

// Panikni strukturalangan xatoga aylantirish
func CatchPanic(f func() error) error {
    var panicErr error

    func() {
        defer func() {
            if r := recover(); r != nil {
                panicErr = &PanicError{
                    Value:      r,
                    StackTrace: string(debug.Stack()),
                }
            }
        }()

        if f != nil {
            if err := f(); err != nil {
                panicErr = err
            }
        }
    }()

    return panicErr
}

// Xato panikdan ekanligini tekshirish
func IsPanicError(err error) bool {
    if err == nil {
        return false
    }
    _, ok := err.(*PanicError)
    return ok
}

// Xatodan panik qiymatini chiqarish
func PanicValue(err error) interface{} {
    if pe, ok := err.(*PanicError); ok {
        return pe.Value
    }
    return nil
}
\`\`\`

**Foydalanish misollari:**
\`\`\`go
// Misol 1: Logging uchun panik tafsilotlarini ushlash
func ProcessUserData(data []byte) error {
    err := CatchPanic(func() error {
        // Panik qilishi mumkin bo'lgan kod
        result := json.Unmarshal(data, &obj)
        return processResult(result)
    })

    if err != nil {
        if IsPanicError(err) {
            // Panik yuz berdi - stack trace bilan log qiling
            pe := err.(*PanicError)
            log.Printf("PANIK: %v\n%s", pe.Value, pe.StackTrace)

            // Xato kuzatuv xizmatiga yuboring
            sentry.CaptureException(err)

            // Foydalanuvchiga umumiy xato qaytaring
            return errors.New("ma'lumotlarni qayta ishlash muvaffaqiyatsiz")
        }
        // Oddiy xato - boricha qaytaring
        return err
    }
    return nil
}

// Misol 2: Batafsil xato hisoboti bilan API handler
func (s *Server) HandleAPI(w http.ResponseWriter, r *http.Request) {
    err := CatchPanic(func() error {
        return s.processRequest(r)
    })

    if err != nil {
        if pe, ok := err.(*PanicError); ok {
            // Panik - stack trace ni log qiling va 500 qaytaring
            s.logger.Printf("Handler panik: %v\n%s", pe.Value, pe.StackTrace)
            s.metrics.IncrementPanicCounter("api_handler")

            http.Error(w, "Internal Server Error", 500)
            return
        }

        // Oddiy xato - 400 qaytaring
        http.Error(w, err.Error(), 400)
    }
}

// Misol 3: Panikdan tiklanish bilan database operatsiya
func (db *Database) QuerySafely(query string) ([]Row, error) {
    var rows []Row

    err := CatchPanic(func() error {
        var err error
        rows, err = db.Query(query)
        return err
    })

    if IsPanicError(err) {
        // Database panik - jiddiy xato
        db.logger.Printf("So'rov bo'yicha database panik: %s", query)

        pe := err.(*PanicError)
        db.alertOps("Database Panik", string(pe.StackTrace))

        return nil, fmt.Errorf("database xatosi: %w", err)
    }

    return rows, err
}
\`\`\`

**Haqiqiy Production stsenariysi:**
\`\`\`go
// Panik kuzatuv va ogohlantirish tizimi
type PanicTracker struct {
    mu       sync.Mutex
    panics   map[string]int  // panik imzo -> hisoblagich
    logger   *log.Logger
    alerting *AlertService
    sentry   *sentry.Client
}

func (pt *PanicTracker) TrackPanic(component string, err error) {
    if !IsPanicError(err) {
        return
    }

    pe := err.(*PanicError)
    signature := fmt.Sprintf("%s:%v", component, pe.Value)

    pt.mu.Lock()
    pt.panics[signature]++
    count := pt.panics[signature]
    pt.mu.Unlock()

    // To'liq stack trace bilan log
    pt.logger.Printf(
        "PANIK [%s] (soni: %d): %v\n%s",
        component,
        count,
        pe.Value,
        pe.StackTrace,
    )

    // Guruhlanish bilan Sentry ga yuboring
    pt.sentry.CaptureException(err, map[string]string{
        "component": component,
        "signature": signature,
    })

    // Takrorlanuvchi paniklarni ogohlantirishq
    if count >= 10 {
        pt.alerting.SendAlert(fmt.Sprintf(
            "%s da takrorlanuvchi panik: %v (soni: %d)",
            component,
            pe.Value,
            count,
        ))
    }
}

// Panik kuzatuv bilan middleware
func (pt *PanicTracker) PanicMiddleware(handler http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        err := CatchPanic(func() error {
            handler.ServeHTTP(w, r)
            return nil
        })

        if err != nil {
            pt.TrackPanic("http_handler", err)
            http.Error(w, "Internal Server Error", 500)
        }
    })
}

// Panikdan tiklanish va hisobot bilan background vazifa
type JobProcessor struct {
    tracker *PanicTracker
    metrics *Metrics
}

func (jp *JobProcessor) ProcessJob(job Job) error {
    startTime := time.Now()

    err := CatchPanic(func() error {
        return job.Execute()
    })

    duration := time.Since(startTime)
    jp.metrics.RecordJobDuration(job.Type, duration)

    if IsPanicError(err) {
        // Panik statistikasini kuzatish
        jp.tracker.TrackPanic(job.Type, err)
        jp.metrics.IncrementJobPanic(job.Type)

        pe := err.(*PanicError)

        // Tafsilotlar bilan dead letter queue ga saqlash
        jp.deadLetterQueue.Add(DeadLetter{
            Job:        job,
            PanicValue: pe.Value,
            StackTrace: pe.StackTrace,
            Timestamp:  time.Now(),
        })

        return fmt.Errorf("vazifa panik qildi: %w", err)
    }

    if err != nil {
        jp.metrics.IncrementJobError(job.Type)
        return err
    }

    jp.metrics.IncrementJobSuccess(job.Type)
    return nil
}
\`\`\`

**Stack Trace tahlili:**
\`\`\`go
// Debug uchun stack trace ni tahlil qilish
func AnalyzeStackTrace(err error) *StackInfo {
    if !IsPanicError(err) {
        return nil
    }

    pe := err.(*PanicError)
    lines := strings.Split(pe.StackTrace, "\n")

    info := &StackInfo{
        PanicValue: pe.Value,
    }

    // Panik manbasini topish (birinchi runtime bo'lmagan frame)
    for i, line := range lines {
        if strings.Contains(line, ".go:") {
            if !strings.Contains(line, "/runtime/") {
                // Ilova kodi topildi
                info.OriginFile = extractFilePath(line)
                info.OriginLine = extractLineNumber(line)

                if i > 0 {
                    info.OriginFunc = strings.TrimSpace(lines[i-1])
                }
                break
            }
        }
    }

    return info
}

// Avtomatik panik klassifikatsiyasi
func ClassifyPanic(err error) string {
    val := PanicValue(err)
    if val == nil {
        return "unknown"
    }

    switch v := val.(type) {
    case *runtime.TypeAssertionError:
        return "type_assertion"
    case string:
        if strings.Contains(v, "nil pointer") {
            return "nil_pointer"
        }
        if strings.Contains(v, "index out of range") {
            return "index_out_of_bounds"
        }
        if strings.Contains(v, "slice bounds") {
            return "slice_bounds"
        }
        if strings.Contains(v, "divide by zero") {
            return "division_by_zero"
        }
    }

    return "runtime_error"
}
\`\`\`

**Test va debugging:**
\`\`\`go
// Panik stsenariylari uchun test yordamchisi
func TestPanicRecovery(t *testing.T) {
    tests := []struct {
        name      string
        fn        func() error
        wantPanic bool
        panicVal  interface{}
    }{
        {
            name: "nil pointer panik",
            fn: func() error {
                var p *int
                _ = *p
                return nil
            },
            wantPanic: true,
        },
        {
            name: "oddiy xato",
            fn: func() error {
                return errors.New("oddiy xato")
            },
            wantPanic: false,
        },
        {
            name: "muvaffaqiyat",
            fn: func() error {
                return nil
            },
            wantPanic: false,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := CatchPanic(tt.fn)

            if tt.wantPanic && !IsPanicError(err) {
                t.Error("panik xatosi kutilgan, olinmadi")
            }

            if !tt.wantPanic && IsPanicError(err) {
                t.Errorf("kutilmagan panik: %v", err)
            }

            if IsPanicError(err) {
                pe := err.(*PanicError)
                if pe.StackTrace == "" {
                    t.Error("stack trace yo'q")
                }
            }
        })
    }
}
\`\`\`

**Cheklovlar:**
- PanicError error interface ni amalga oshirishi kerak
- CatchPanic debug.Stack() dan foydalanib stack trace ni ushlashi kerak
- IsPanicError xato turini xavfsiz tekshirishi kerak
- PanicValue panik bo'lmagan xatolar uchun nil qaytarishi kerak
- Stack trace tiklanish nuqtasida emas, panik nuqtasida ushlanishi kerak`,
			hint1: `Value va StackTrace maydonlari bilan PanicError struct ni aniqlang. fmt.Sprintf("panic: %v", e.Value) qaytaradigan Error() metodini amalga oshiring.`,
			hint2: `CatchPanic da: defer recover() bilan ichki anonim funksiyadan foydalaning. Panikda &PanicError{Value: r, StackTrace: string(debug.Stack())} yarating. f() qaytgan xatosini alohida tekshiring.`,
			whyItMatters: `Stack trace bilan paniklarni strukturalangan xatolarga aylantirish production muammolarini tuzatish va harakatchan tizimlar qurish uchun muhim.

**Nima uchun bu muhim:**

**1. Production Incident: Sirli Crash Tekshiruvi**

E-commerce platformasi vaqti-vaqti bilan crashlarni boshdan kechirdi:
- Xizmat kuniga 10-15 marta qulab tushdi
- Loglar faqat "panic recovered" xabarlarini ko'rsatdi
- Stack tracelar ushlanmagan
- Muhandislar asosiy sababni aniqlay olmadi
- 3 haftalik tekshiruv natija bermadi
- Xarajat: 100+ muhandislik soati

**Asosiy sabab:**
\`\`\`go
// Oldin: Stack trace ushlanmagan
func ProcessOrder(order Order) error {
    defer func() {
        if r := recover() {
            log.Printf("panik: %v", r)  // Faqat panik qiymati
            // Bu QAYERDA yuz berdi? Noma'lum!
        }
    }()

    processPayment(order)
    updateInventory(order)
    sendEmail(order)
    return nil
}
// Natija: Panik yuz berganini bilamiz, lekin kodda QAYERDA - bilmaymiz
\`\`\`

**PanicError bilan yechim:**
\`\`\`go
// Keyin: To'liq stack trace ushlangan
err := CatchPanic(func() error {
    processPayment(order)
    updateInventory(order)
    sendEmail(order)
    return nil
})

if pe, ok := err.(*PanicError); ok {
    // To'liq stack trace ni log qiling
    log.Printf("Panik: %v\n%s", pe.Value, pe.StackTrace)

    // Natija:
    // panik: nil pointer dereference
    // goroutine 42 [running]:
    // main.updateInventory(...)
    //     /app/inventory.go:123
    // main.ProcessOrder(...)
    //     /app/order.go:45
}
// Natija: Panikka sabab bo'lgan aniq qator 5 daqiqada aniqlandi
\`\`\`

**Tuzatishdan keyingi ta'sir:**
- Bug inventory.go 123-qatorda topildi (nil map ga kirish)
- 30 daqiqada tuzatildi
- Tuzatishdan keyin nol crash
- Tekshiruv vaqti: 3 hafta → 5 daqiqa

**2. Haqiqiy holat: Sentry Integratsiyasi**

1M+ foydalanuvchili SaaS platformasi:

**PanicError dan oldin:**
- Paniklar loglangan, lekin xato kuzatuv tizimiga yuborilmagan
- Xatolar umumiy "panic recovered" xabari bo'yicha guruhlangan
- Sentry da stack trace yo'q
- Muhim paniklarni ustuvor qilish mumkin emas

**PanicError joriy qilgandan keyin:**
\`\`\`go
func (s *Server) HandleRequest(r Request) error {
    err := CatchPanic(func() error {
        return s.process(r)
    })

    if IsPanicError(err) {
        // Stack trace bilan Sentry ga yuboring
        pe := err.(*PanicError)
        sentry.CaptureException(err, map[string]interface{}{
            "panic_value": pe.Value,
            "stack_trace": pe.StackTrace,
            "user_id":     r.UserID,
            "endpoint":    r.Endpoint,
        })

        // Panikni klassifikatsiya qilish
        category := classifyPanic(pe)
        metrics.IncrementPanic(category)

        return errors.New("so'rov muvaffaqiyatsiz")
    }

    return err
}
\`\`\`

**Natijalar:**
- Barcha paniklar Sentry panelida ko'rinadi
- Paniklar stack trace bo'yicha guruhlangan (umumiy xabar emas)
- Top 5 panik turlari darhol aniqlangan
- Chastota va ta'sir bo'yicha tuzatishlar ustuvor qilingan

**Panik statistikasi (Birinchi hafta):**
- nil pointer dereference: 450 (40%) → 2 kunda tuzatildi
- index out of bounds: 280 (25%) → 1 kunda tuzatildi
- type assertion: 220 (19%) → 3 kunda tuzatildi
- map concurrent access: 180 (16%) → 1 kunda tuzatildi

**Ta'sir:**
- Paniklar 1 haftada 90% kamaydi
- O'rtacha tuzatish vaqti: 2 kun (noma'lum edi)
- Mijoz shikoyatlari: haftasiga 50 → haftasiga 5

**3. Production metrikalari: Panik tahlili**

Moliyaviy API xizmati (kuniga 100M so'rov):

**PanicError dan oldin:**
- Kuniga 100-200 panik
- Muhandislar stack trace uchun loglarni qo'lda ko'rib chiqishdi
- Panik uchun o'rtacha debugging vaqti: 2-4 soat
- Panik tendentsiyasi yoki klassifikatsiyasi yo'q

**Tahlil bilan PanicError dan keyin:**
\`\`\`go
// Avtomatik panik klassifikatsiyasi
type PanicStats struct {
    Category      string
    Count         int
    FirstSeen     time.Time
    LastSeen      time.Time
    AffectedUsers int
    ExampleTrace  string
}

func (pt *PanicTracker) AnalyzePanic(err error) {
    pe := err.(*PanicError)

    // Panik turini klassifikatsiya qilish
    category := ClassifyPanic(err)

    // Manba joylashuvini chiqarish
    info := AnalyzeStackTrace(err)

    pt.mu.Lock()
    stats := pt.stats[category]
    stats.Count++
    stats.LastSeen = time.Now()

    if stats.FirstSeen.IsZero() {
        stats.FirstSeen = time.Now()
        stats.ExampleTrace = pe.StackTrace
    }

    pt.stats[category] = stats
    pt.mu.Unlock()

    // Yangi panik turlari haqida ogohlantirish
    if stats.Count == 1 {
        pt.alerting.Send(fmt.Sprintf(
            "YANGI PANIK TURI: %s\n%s\nJoylashuv: %s:%d",
            category,
            pe.Value,
            info.OriginFile,
            info.OriginLine,
        ))
    }
}
\`\`\`

**Natijalar:**
- Avtomatik panik klassifikatsiyasi va tendentsiyalari
- Yangi panik turlarini darhol aniqlash (1 daqiqa ichida ogohlantirishlar)
- Kod bazasida panik "hot spot"larini kuzatish
- Debugging vaqti: 2-4 soat → 15-30 daqiqa (o'rtacha)

**Panik paneli metrikalari:**
\`\`\`
Top 5 Panik Kategoriyalari (So'nggi 30 kun):
1. nil_pointer: 1,234 hodisa (65% cache.go:89 da)
2. index_out_of_bounds: 456 (40% parser.go:234 da)
3. type_assertion: 234 (80% handler.go:123 da)
4. division_by_zero: 89 (95% metrics.go:456 da)
5. slice_bounds: 45 (100% validator.go:78 da)

Harakat: Top 3 tuzatildi → 90% panik kamayishi
\`\`\`

**4. Panik tafsilotlari bilan Dead Letter Queue**

Kuniga 1M xabarni qayta ishlaydigan xabar qayta ishlash tizimi:

\`\`\`go
// Oldin: Panik tafsilotlari yo'qolgan
if err := processMessage(msg); err != nil {
    deadLetterQueue.Add(msg)  // Faqat xabar
    // NIMA UCHUN muvaffaqiyatsiz bo'lganligi haqida ma'lumot yo'q
}

// Keyin: Panik tafsilotlari saqlangan
err := CatchPanic(func() error {
    return processMessage(msg)
})

if IsPanicError(err) {
    pe := err.(*PanicError)

    // To'liq panik konteksti bilan xabarni saqlash
    deadLetterQueue.Add(DeadLetter{
        Message:    msg,
        PanicValue: pe.Value,
        StackTrace: pe.StackTrace,
        Timestamp:  time.Now(),
    })

    // Muhandislar aniq kontekst bilan qayta ishlay olishadi
}
\`\`\`

**Ta'sir:**
- Muvaffaqiyatsiz xabarlar NIMA UCHUN muvaffaqiyatsiz bo'lganini o'z ichiga oladi
- Tuzatish qo'llanilgan holda nosozliklarni qayta ishlab chiqarish
- Xabar uchun debugging vaqti: 30 daqiqa → 5 daqiqa
- Tiklash foizi: 60% → 95%

**5. Raqamlar: Haqiqiy Production Tizimi**

**API Gateway (kuniga 500M so'rov):**

**PanicError dan oldin:**
- Panik debugging vaqti: oyiga 100-200 soat
- Hal qilinmagan paniklar: 20-30
- Mijozlarga ta'siri: Yuqori (tushuntirilmagan nosozliklar)
- Muhandislik xarajati: oyiga $20K-$40K

**PanicError joriy qilgandan keyin:**
- Panik debugging vaqti: oyiga 10-20 soat
- Hal qilinmagan paniklar: 0-2
- Mijozlarga ta'siri: Past (tez tuzatishlar)
- Muhandislik xarajati: oyiga $2K-$4K

**ROI hisoblash:**
\`\`\`
Joriy qilish: 2 hafta (80 soat) = ~$8K

Oylik tejamkorlik:
- Muhandis vaqti: 80-180 soat = $8K-$18K
- Tez tuzatishlar: Mijoz oqishi kamayishi = $10K-$20K
- Jami: oyiga $18K-$38K

To'lov muddati: 2 hafta
Yillik ROI: 500-1000%
\`\`\`

**Eng yaxshi amaliyotlar:**
1. **Har doim stack tracelarni ushlang** debug.Stack() bilan
2. **Xato kuzatuv tizimiga yuboring** (Sentry, Datadog, va boshqalar)
3. **Panik turlarini klassifikatsiya qiling** tendentsiyalar uchun
4. **Yangi panik turlari haqida darhol ogohlantiring**
5. **Kontekstni qo'shing** (user ID, request ID, va boshqalar)
6. **Dead letter queue ga saqlang** qayta ishlab chiqarish uchun
7. **Panik panelini yarating** ko'rinish uchun`
		}
	}
};

export default task;
