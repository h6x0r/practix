import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-retry-on-panic',
	title: 'Retry with Panic Recovery and Exponential Backoff',
	difficulty: 'hard',
	tags: ['go', 'panic', 'retry', 'resilience', 'backoff'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement production-grade retry mechanism with panic recovery, exponential backoff, and configurable retry policies for building resilient systems.

**Requirements:**
1. **RetryConfig**: Configuration for retry behavior (max attempts, backoff)
2. **RetryWithPanic**: Execute function with retry on both errors and panics
3. **Exponential Backoff**: Implement exponential backoff with jitter
4. **Attempt Tracking**: Track attempts, panics, and errors for observability

**Retry with Panic Pattern:**
\`\`\`go
// Retry configuration
type RetryConfig struct {
    MaxAttempts   int           // Maximum retry attempts
    InitialDelay  time.Duration // Initial backoff delay
    MaxDelay      time.Duration // Maximum backoff delay
    Multiplier    float64       // Backoff multiplier (e.g., 2.0)
    RetryOnPanic  bool          // Whether to retry on panic
}

// Retry result with observability
type RetryResult struct {
    Attempts     int           // Total attempts made
    PanicCount   int           // Number of panics
    ErrorCount   int           // Number of errors
    Success      bool          // Whether eventually succeeded
    LastError    error         // Last error/panic if failed
    Duration     time.Duration // Total retry duration
}

// Default config with exponential backoff
func DefaultRetryConfig() RetryConfig {
    return RetryConfig{
        MaxAttempts:  3,
        InitialDelay: 100 * time.Millisecond,
        MaxDelay:     5 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }
}

// Retry function with panic recovery
func RetryWithPanic(config RetryConfig, f func() error) *RetryResult {
    result := &RetryResult{
        Success: false,
    }

    startTime := time.Now()
    delay := config.InitialDelay

    for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
        result.Attempts = attempt

        // Execute with panic recovery
        err := executeSafely(f)

        if err == nil {
            // Success
            result.Success = true
            result.Duration = time.Since(startTime)
            return result
        }

        // Track error type
        if isPanic(err) {
            result.PanicCount++
            if !config.RetryOnPanic {
                result.LastError = err
                result.Duration = time.Since(startTime)
                return result
            }
        } else {
            result.ErrorCount++
        }

        result.LastError = err

        // Don't sleep after last attempt
        if attempt < config.MaxAttempts {
            // Exponential backoff with jitter
            jitter := time.Duration(rand.Float64() * float64(delay) * 0.3)
            sleepTime := delay + jitter

            if sleepTime > config.MaxDelay {
                sleepTime = config.MaxDelay
            }

            time.Sleep(sleepTime)

            // Increase delay for next attempt
            delay = time.Duration(float64(delay) * config.Multiplier)
        }
    }

    result.Duration = time.Since(startTime)
    return result
}

// Execute function safely with panic recovery
func executeSafely(f func() error) error {
    var panicErr error

    func() {
        defer func() {
            if r := recover() {
                panicErr = fmt.Errorf("panic: %v", r)
            }
        }()

        if f != nil {
            panicErr = f()
        }
    }()

    return panicErr
}

// Check if error is from panic
func isPanic(err error) bool {
    return err != nil && strings.HasPrefix(err.Error(), "panic:")
}
\`\`\`

**Example Usage:**
\`\`\`go
// Example 1: Retry external API call with panic recovery
func FetchUserData(userID string) (*User, error) {
    config := RetryConfig{
        MaxAttempts:  5,
        InitialDelay: 100 * time.Millisecond,
        MaxDelay:     5 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    var user *User

    result := RetryWithPanic(config, func() error {
        // External API might panic or return error
        var err error
        user, err = externalAPI.GetUser(userID)
        return err
    })

    if !result.Success {
        log.Printf(
            "Failed after %d attempts (%d panics, %d errors) in %v: %v",
            result.Attempts,
            result.PanicCount,
            result.ErrorCount,
            result.Duration,
            result.LastError,
        )
        return nil, result.LastError
    }

    log.Printf("Success after %d attempts in %v", result.Attempts, result.Duration)
    return user, nil
}

// Example 2: Database operation with retry
func (db *Database) InsertWithRetry(data Record) error {
    config := DefaultRetryConfig()
    config.MaxAttempts = 3

    result := RetryWithPanic(config, func() error {
        return db.Insert(data)
    })

    if result.PanicCount > 0 {
        // Database panics - critical issue
        db.alertOps("Database panic during insert", result.LastError)
    }

    if !result.Success {
        return fmt.Errorf("insert failed: %w", result.LastError)
    }

    return nil
}

// Example 3: Message processing with selective retry
func ProcessMessage(msg Message) error {
    config := RetryConfig{
        MaxAttempts:  3,
        InitialDelay: 50 * time.Millisecond,
        MaxDelay:     1 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,  // Retry panics
    }

    result := RetryWithPanic(config, func() error {
        return parseAndProcess(msg)
    })

    if !result.Success {
        // Move to dead letter queue
        if result.PanicCount > 0 {
            deadLetterQueue.AddWithReason(msg, "repeated panics", result.LastError)
        } else {
            deadLetterQueue.AddWithReason(msg, "processing error", result.LastError)
        }
        return result.LastError
    }

    return nil
}
\`\`\`

**Real-World Production Scenario:**
\`\`\`go
// Production API client with retry and observability
type APIClient struct {
    baseURL    string
    httpClient *http.Client
    metrics    *Metrics
    logger     *log.Logger
}

func (c *APIClient) CallWithRetry(endpoint string, payload interface{}) (*Response, error) {
    config := RetryConfig{
        MaxAttempts:  5,
        InitialDelay: 200 * time.Millisecond,
        MaxDelay:     10 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    var response *Response

    result := RetryWithPanic(config, func() error {
        req, err := c.buildRequest(endpoint, payload)
        if err != nil {
            return err
        }

        resp, err := c.httpClient.Do(req)
        if err != nil {
            return err
        }
        defer resp.Body.Close()

        if resp.StatusCode >= 500 {
            // Retry on server errors
            return fmt.Errorf("server error: %d", resp.StatusCode)
        }

        if resp.StatusCode >= 400 {
            // Don't retry client errors
            return &NonRetryableError{
                StatusCode: resp.StatusCode,
                Message:    "client error",
            }
        }

        response, err = parseResponse(resp)
        return err
    })

    // Record metrics
    c.metrics.RecordAPICall(endpoint, result)

    if result.PanicCount > 0 {
        // API client panicked - investigate
        c.logger.Printf(
            "API panic on %s: %d panics in %d attempts",
            endpoint,
            result.PanicCount,
            result.Attempts,
        )
        c.metrics.IncrementAPIPanic(endpoint)
    }

    if !result.Success {
        c.logger.Printf(
            "API call failed: %s after %d attempts (%v)",
            endpoint,
            result.Attempts,
            result.Duration,
        )
        return nil, result.LastError
    }

    c.logger.Printf(
        "API success: %s in %d attempts (%v)",
        endpoint,
        result.Attempts,
        result.Duration,
    )

    return response, nil
}

// Worker pool with retry on panic
type WorkerPool struct {
    workers    int
    retryConfig RetryConfig
    metrics     *Metrics
}

func (wp *WorkerPool) ProcessJobs(jobs <-chan Job) {
    var wg sync.WaitGroup

    for i := 0; i < wp.workers; i++ {
        wg.Add(1)
        workerID := i

        go func() {
            defer wg.Done()

            for job := range jobs {
                result := RetryWithPanic(wp.retryConfig, func() error {
                    return job.Execute()
                })

                // Record metrics per worker
                wp.metrics.RecordJobResult(workerID, job.Type, result)

                if result.PanicCount > 0 {
                    log.Printf(
                        "Worker %d: Job %s panicked %d times",
                        workerID,
                        job.ID,
                        result.PanicCount,
                    )
                }

                if !result.Success {
                    log.Printf(
                        "Worker %d: Job %s failed after %d attempts: %v",
                        workerID,
                        job.ID,
                        result.Attempts,
                        result.LastError,
                    )

                    // Move to failed jobs queue
                    wp.handleFailedJob(job, result)
                }
            }
        }()
    }

    wg.Wait()
}

// Circuit breaker integration
type CircuitBreaker struct {
    maxFailures   int
    failures      int
    lastFailTime  time.Time
    state         string  // "closed", "open", "half-open"
    retryConfig   RetryConfig
}

func (cb *CircuitBreaker) Call(f func() error) error {
    if cb.state == "open" {
        // Circuit is open - fail fast
        if time.Since(cb.lastFailTime) < 30*time.Second {
            return errors.New("circuit breaker open")
        }
        // Try to close circuit
        cb.state = "half-open"
    }

    result := RetryWithPanic(cb.retryConfig, f)

    if result.Success {
        // Success - reset failures
        cb.failures = 0
        cb.state = "closed"
        return nil
    }

    // Failure - increment counter
    cb.failures++
    cb.lastFailTime = time.Now()

    if cb.failures >= cb.maxFailures {
        cb.state = "open"
        log.Printf("Circuit breaker opened after %d failures", cb.failures)
    }

    return result.LastError
}
\`\`\`

**Advanced Retry Strategies:**
\`\`\`go
// Conditional retry based on error type
type RetryPolicy func(attempt int, err error) bool

func RetryWithPolicy(
    config RetryConfig,
    policy RetryPolicy,
    f func() error,
) *RetryResult {
    result := &RetryResult{}
    startTime := time.Now()
    delay := config.InitialDelay

    for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
        result.Attempts = attempt
        err := executeSafely(f)

        if err == nil {
            result.Success = true
            result.Duration = time.Since(startTime)
            return result
        }

        // Check if should retry
        if !policy(attempt, err) {
            result.LastError = err
            result.Duration = time.Since(startTime)
            return result
        }

        result.LastError = err

        if attempt < config.MaxAttempts {
            time.Sleep(delay)
            delay = time.Duration(float64(delay) * config.Multiplier)
            if delay > config.MaxDelay {
                delay = config.MaxDelay
            }
        }
    }

    result.Duration = time.Since(startTime)
    return result
}

// Example retry policies
func RetryOnTransientErrors(attempt int, err error) bool {
    // Don't retry on panics
    if isPanic(err) {
        return false
    }

    // Retry on specific errors
    if strings.Contains(err.Error(), "timeout") ||
       strings.Contains(err.Error(), "connection refused") ||
       strings.Contains(err.Error(), "temporary failure") {
        return true
    }

    return false
}

func RetryOnPanicOnly(attempt int, err error) bool {
    return isPanic(err)
}
\`\`\`

**Observability and Monitoring:**
\`\`\`go
// Retry metrics collector
type RetryMetrics struct {
    mu              sync.Mutex
    totalRetries    int64
    successfulRetries int64
    failedRetries   int64
    totalPanics     int64
    retryDurations  []time.Duration
}

func (rm *RetryMetrics) Record(result *RetryResult) {
    rm.mu.Lock()
    defer rm.mu.Unlock()

    rm.totalRetries++
    rm.totalPanics += int64(result.PanicCount)

    if result.Success {
        rm.successfulRetries++
    } else {
        rm.failedRetries++
    }

    rm.retryDurations = append(rm.retryDurations, result.Duration)
}

func (rm *RetryMetrics) Report() {
    rm.mu.Lock()
    defer rm.mu.Unlock()

    successRate := float64(rm.successfulRetries) / float64(rm.totalRetries) * 100
    panicRate := float64(rm.totalPanics) / float64(rm.totalRetries) * 100

    avgDuration := time.Duration(0)
    if len(rm.retryDurations) > 0 {
        var total time.Duration
        for _, d := range rm.retryDurations {
            total += d
        }
        avgDuration = total / time.Duration(len(rm.retryDurations))
    }

    log.Printf(\`Retry Metrics:
  Total Retries: %d
  Success Rate: %.2f%%
  Failed Retries: %d
  Total Panics: %d
  Panic Rate: %.2f%%
  Avg Duration: %v
\`, rm.totalRetries, successRate, rm.failedRetries, rm.totalPanics, panicRate, avgDuration)
}
\`\`\`

**Constraints:**
- RetryConfig must have all required fields
- RetryWithPanic must respect MaxAttempts limit
- Exponential backoff must include jitter to avoid thundering herd
- Must not sleep after last failed attempt
- RetryResult must track all attempts, panics, and errors
- Must distinguish between panics and regular errors`,
	initialCode: `package panicrecover

import (
	"fmt"
	"math/rand"
	"strings"
	"time"
)

// TODO: Define RetryConfig struct
// MaxAttempts int, InitialDelay time.Duration, MaxDelay time.Duration
// Multiplier float64, RetryOnPanic bool
type RetryConfig struct {
	// TODO: Add fields
}

// TODO: Define RetryResult struct
// Attempts int, PanicCount int, ErrorCount int, Success bool
// LastError error, Duration time.Duration
type RetryResult struct {
	// TODO: Add fields
}

// TODO: Implement DefaultRetryConfig
// Return RetryConfig with: MaxAttempts=3, InitialDelay=100ms
// MaxDelay=5s, Multiplier=2.0, RetryOnPanic=true
func DefaultRetryConfig() RetryConfig {
	// TODO: Implement
}

// TODO: Implement RetryWithPanic
// Execute f up to config.MaxAttempts times
// Recover from panics if config.RetryOnPanic is true
// Use exponential backoff: delay *= Multiplier (with jitter)
// Track attempts, panics, errors in RetryResult
// Return result with Success=true if any attempt succeeds
func RetryWithPanic(config RetryConfig, f func() error) *RetryResult {
	// TODO: Implement
}

// TODO: Implement executeSafely helper
// Execute f with panic recovery
// Return error if panic occurs (format: "panic: %v")
// Return f() error if no panic
func executeSafely(f func() error) error {
	// TODO: Implement
}

// TODO: Implement isPanic helper
// Check if error message starts with "panic:"
// Return false if err is nil
func isPanic(err error) bool {
	return false // TODO: Implement
}`,
	solutionCode: `package panicrecover

import (
	"fmt"
	"math/rand"
	"strings"
	"time"
)

type RetryConfig struct {
	MaxAttempts   int           // max retry attempts
	InitialDelay  time.Duration // initial backoff delay
	MaxDelay      time.Duration // max backoff delay
	Multiplier    float64       // backoff multiplier
	RetryOnPanic  bool          // retry on panic
}

type RetryResult struct {
	Attempts     int           // total attempts
	PanicCount   int           // panic count
	ErrorCount   int           // error count
	Success      bool          // success flag
	LastError    error         // last error
	Duration     time.Duration // total duration
}

func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:  3,                       // retry 3 times
		InitialDelay: 100 * time.Millisecond, // start with 100ms
		MaxDelay:     5 * time.Second,         // cap at 5s
		Multiplier:   2.0,                     // double each time
		RetryOnPanic: true,                    // retry panics
	}
}

func RetryWithPanic(config RetryConfig, f func() error) *RetryResult {
	result := &RetryResult{
		Success: false,                        // start as failed
	}

	startTime := time.Now()                    // track duration
	delay := config.InitialDelay               // current backoff delay

	for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
		result.Attempts = attempt              // track attempt number

		err := executeSafely(f)                // execute with panic recovery

		if err == nil {
			result.Success = true              // success
			result.Duration = time.Since(startTime)
			return result
		}

		if isPanic(err) {                      // check if panic
			result.PanicCount++
			if !config.RetryOnPanic {          // don't retry panics
				result.LastError = err
				result.Duration = time.Since(startTime)
				return result
			}
		} else {
			result.ErrorCount++                // regular error
		}

		result.LastError = err

		if attempt < config.MaxAttempts {      // don't sleep after last attempt
			jitter := time.Duration(rand.Float64() * float64(delay) * 0.3) // 30% jitter
			sleepTime := delay + jitter

			if sleepTime > config.MaxDelay {   // cap delay
				sleepTime = config.MaxDelay
			}

			time.Sleep(sleepTime)              // backoff

			delay = time.Duration(float64(delay) * config.Multiplier) // increase delay
		}
	}

	result.Duration = time.Since(startTime)
	return result
}

func executeSafely(f func() error) error {
	var panicErr error                         // capture error

	func() {                                   // anonymous function
		defer func() {
			if r := recover(); r != nil {      // catch panic
				panicErr = fmt.Errorf("panic: %v", r)
			}
		}()

		if f != nil {
			panicErr = f()                     // execute function
		}
	}()

	return panicErr                            // return error or nil
}

func isPanic(err error) bool {
	return err != nil && strings.HasPrefix(err.Error(), "panic:") // check prefix
}`,
	hint1: `Define structs with all required fields. DefaultRetryConfig returns RetryConfig{MaxAttempts: 3, InitialDelay: 100*time.Millisecond, MaxDelay: 5*time.Second, Multiplier: 2.0, RetryOnPanic: true}.`,
	hint2: `In RetryWithPanic: loop from 1 to MaxAttempts. Call executeSafely(f). If err==nil, set Success=true and return. Track panics/errors. Sleep with jitter before next attempt (not after last). In executeSafely: use defer recover to catch panics.`,
	testCode: `package panicrecover

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Test DefaultRetryConfig
	cfg := DefaultRetryConfig()
	if cfg.MaxAttempts != 3 {
		t.Errorf("DefaultRetryConfig().MaxAttempts = %v, want 3", cfg.MaxAttempts)
	}
	if cfg.InitialDelay != 100*time.Millisecond {
		t.Errorf("DefaultRetryConfig().InitialDelay = %v, want 100ms", cfg.InitialDelay)
	}
}

func Test2(t *testing.T) {
	// Test RetryWithPanic with nil function
	cfg := RetryConfig{MaxAttempts: 1, InitialDelay: time.Millisecond, MaxDelay: time.Second, Multiplier: 1.5, RetryOnPanic: true}
	result := RetryWithPanic(cfg, nil)
	if !result.Success {
		t.Error("RetryWithPanic(nil) should succeed")
	}
}

func Test3(t *testing.T) {
	// Test RetryWithPanic with immediately successful function
	cfg := DefaultRetryConfig()
	result := RetryWithPanic(cfg, func() error {
		return nil
	})
	if !result.Success {
		t.Error("RetryWithPanic(success) should succeed")
	}
	if result.Attempts != 1 {
		t.Errorf("Attempts = %v, want 1", result.Attempts)
	}
}

func Test4(t *testing.T) {
	// Test RetryWithPanic with always failing function
	cfg := RetryConfig{MaxAttempts: 3, InitialDelay: time.Millisecond, MaxDelay: 10 * time.Millisecond, Multiplier: 2.0, RetryOnPanic: true}
	result := RetryWithPanic(cfg, func() error {
		return errors.New("always fails")
	})
	if result.Success {
		t.Error("RetryWithPanic(always fails) should fail")
	}
	if result.Attempts != 3 {
		t.Errorf("Attempts = %v, want 3", result.Attempts)
	}
	if result.ErrorCount != 3 {
		t.Errorf("ErrorCount = %v, want 3", result.ErrorCount)
	}
}

func Test5(t *testing.T) {
	// Test RetryWithPanic with panicking function
	cfg := RetryConfig{MaxAttempts: 2, InitialDelay: time.Millisecond, MaxDelay: 10 * time.Millisecond, Multiplier: 2.0, RetryOnPanic: true}
	result := RetryWithPanic(cfg, func() error {
		panic("test panic")
	})
	if result.Success {
		t.Error("RetryWithPanic(panic) should fail")
	}
	if result.PanicCount != 2 {
		t.Errorf("PanicCount = %v, want 2", result.PanicCount)
	}
}

func Test6(t *testing.T) {
	// Test RetryWithPanic with RetryOnPanic=false
	cfg := RetryConfig{MaxAttempts: 3, InitialDelay: time.Millisecond, MaxDelay: 10 * time.Millisecond, Multiplier: 2.0, RetryOnPanic: false}
	result := RetryWithPanic(cfg, func() error {
		panic("no retry panic")
	})
	if result.Attempts != 1 {
		t.Errorf("Attempts = %v, want 1 (no retry on panic)", result.Attempts)
	}
	if result.PanicCount != 1 {
		t.Errorf("PanicCount = %v, want 1", result.PanicCount)
	}
}

func Test7(t *testing.T) {
	// Test RetryWithPanic with function that succeeds on second attempt
	attempt := 0
	cfg := RetryConfig{MaxAttempts: 3, InitialDelay: time.Millisecond, MaxDelay: 10 * time.Millisecond, Multiplier: 2.0, RetryOnPanic: true}
	result := RetryWithPanic(cfg, func() error {
		attempt++
		if attempt < 2 {
			return errors.New("temporary error")
		}
		return nil
	})
	if !result.Success {
		t.Error("Should succeed on second attempt")
	}
	if result.Attempts != 2 {
		t.Errorf("Attempts = %v, want 2", result.Attempts)
	}
}

func Test8(t *testing.T) {
	// Test executeSafely with successful function
	err := executeSafely(func() error {
		return nil
	})
	if err != nil {
		t.Errorf("executeSafely(success) = %v, want nil", err)
	}
}

func Test9(t *testing.T) {
	// Test executeSafely with panicking function
	err := executeSafely(func() error {
		panic("safe panic")
	})
	if err == nil {
		t.Error("executeSafely(panic) = nil, want error")
	}
	if !strings.Contains(err.Error(), "panic:") {
		t.Errorf("error should contain 'panic:', got %v", err)
	}
}

func Test10(t *testing.T) {
	// Test isPanic function
	panicErr := errors.New("panic: something went wrong")
	regularErr := errors.New("regular error")
	if !isPanic(panicErr) {
		t.Error("isPanic(panic error) = false, want true")
	}
	if isPanic(regularErr) {
		t.Error("isPanic(regular error) = true, want false")
	}
	if isPanic(nil) {
		t.Error("isPanic(nil) = true, want false")
	}
}`,
	whyItMatters: `Retry with panic recovery is critical for building resilient systems that can automatically recover from transient failures and unexpected panics without human intervention.

**Why This Matters:**

**1. Production Incident: Payment Gateway Outage**

A fintech company processed 100K+ daily transactions:
- Payment gateway API occasionally panicked (3rd party SDK bug)
- Single panic failed transaction permanently
- No retry mechanism in place
- Customer transaction lost
- Manual refund required ($50 per incident)
- 50-100 panics per day
- Daily cost: $2.5K-$5K in refunds
- Monthly cost: $75K-$150K

**Root Cause:**
\`\`\`go
// Before: No retry on panic
func ProcessPayment(payment Payment) error {
    // Payment gateway SDK panics on network hiccup
    result := gateway.Charge(payment)
    return processResult(result)
}
// Result: Panic = permanent transaction failure
\`\`\`

**Solution with RetryWithPanic:**
\`\`\`go
// After: Retry with panic recovery
func ProcessPayment(payment Payment) error {
    config := RetryConfig{
        MaxAttempts:  5,
        InitialDelay: 100 * time.Millisecond,
        MaxDelay:     5 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    var result *PaymentResult

    retryResult := RetryWithPanic(config, func() error {
        result = gateway.Charge(payment)
        return processResult(result)
    })

    if !retryResult.Success {
        log.Printf(
            "Payment failed after %d attempts (%d panics): %v",
            retryResult.Attempts,
            retryResult.PanicCount,
            retryResult.LastError,
        )
        return retryResult.LastError
    }

    if retryResult.Attempts > 1 {
        log.Printf(
            "Payment succeeded on attempt %d (recovered from %d panics)",
            retryResult.Attempts,
            retryResult.PanicCount,
        )
    }

    return nil
}
\`\`\`

**Impact After Fix:**
- Panics per day: Still 50-100 (SDK bug not fixed)
- Failed transactions: 95% reduction (95-100 recovered by retry)
- Refund cost: $75K-$150K/month → $3.75K-$7.5K/month
- Annual savings: $855K-$1.7M
- Customer satisfaction: +40%

**2. Real-World: External API Reliability**

SaaS platform integrating with 20+ external APIs:

**Before RetryWithPanic:**
- External APIs panic occasionally (malformed responses, nil pointers)
- Any panic = feature broken for user
- Users report "feature not working"
- Support team investigates each incident
- Average resolution time: 2-4 hours (waiting for API fix)

**Problem:**
\`\`\`go
// Unreliable external API
func FetchUserProfile(userID string) (*Profile, error) {
    // API SDK panics on unexpected response format
    profile := externalAPI.GetProfile(userID)
    return profile, nil
}
// Result: One malformed response breaks feature
\`\`\`

**Solution:**
\`\`\`go
func FetchUserProfile(userID string) (*Profile, error) {
    config := DefaultRetryConfig()
    config.MaxAttempts = 5

    var profile *Profile

    result := RetryWithPanic(config, func() error {
        var err error
        profile, err = externalAPI.GetProfile(userID)
        return err
    })

    if !result.Success {
        metrics.IncrementAPIFailure("profile_fetch")

        if result.PanicCount > 0 {
            // API panicked - alert but don't block user
            alerting.Send(fmt.Sprintf(
                "Profile API panicked %d times for user %s",
                result.PanicCount,
                userID,
            ))
        }

        return nil, result.LastError
    }

    if result.PanicCount > 0 {
        log.Printf(
            "Recovered from %d panics fetching profile %s",
            result.PanicCount,
            userID,
        )
        metrics.IncrementAPIRetrySuccess("profile_fetch")
    }

    return profile, nil
}
\`\`\`

**Results:**
- API panics: 200-300 per day (external issue)
- Feature availability: 85% → 99%
- Successful recoveries: 90%+ (panics resolved by retry)
- Support tickets: 100/week → 10/week
- User satisfaction: +50%

**3. Production Metrics: Database Deadlock Recovery**

E-commerce platform with high transaction volume:

**Before Retry:**
- Database deadlocks cause transaction panics
- 50-100 deadlocks per hour during peak
- Each deadlock = failed order
- Users see "checkout failed" error
- Cart abandonment rate: 15%

**After RetryWithPanic Implementation:**
\`\`\`go
func (db *Database) ExecuteTransaction(tx func() error) error {
    config := RetryConfig{
        MaxAttempts:  5,
        InitialDelay: 50 * time.Millisecond,
        MaxDelay:     1 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    result := RetryWithPanic(config, tx)

    if result.PanicCount > 0 {
        // Database panic (deadlock or constraint violation)
        metrics.RecordDBPanic(result.PanicCount)
    }

    if !result.Success {
        log.Printf(
            "Transaction failed after %d attempts (%d panics, %d errors)",
            result.Attempts,
            result.PanicCount,
            result.ErrorCount,
        )
        return result.LastError
    }

    if result.Attempts > 1 {
        log.Printf(
            "Transaction succeeded after %d attempts (recovered from %d panics)",
            result.Attempts,
            result.PanicCount,
        )
        metrics.RecordDBRetrySuccess()
    }

    return nil
}
\`\`\`

**Results:**
- Deadlocks per hour: Still 50-100 (database contention)
- Transaction success rate: 75% → 98%
- Recovered by retry: 92%
- Failed orders: 75% reduction
- Cart abandonment: 15% → 8%
- Revenue impact: +$500K per month

**4. Worker Pool Resilience**

Media processing service with 100 workers:

**Statistics Before Retry:**
- Worker panic rate: 1-2% of jobs
- Panic = worker crashes permanently
- Lost workers accumulate over time
- Start with 100 workers → 50 workers after 24 hours
- Processing capacity degrades 50%
- Manual restart required daily

**Implementation:**
\`\`\`go
func (wp *WorkerPool) ProcessWithRetry(job Job) error {
    config := RetryConfig{
        MaxAttempts:  3,
        InitialDelay: 100 * time.Millisecond,
        MaxDelay:     2 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    result := RetryWithPanic(config, func() error {
        return job.Execute()
    })

    wp.metrics.RecordJobResult(job.Type, result)

    if result.PanicCount > 0 {
        log.Printf(
            "Job %s: Recovered from %d panics, attempt %d/%d",
            job.ID,
            result.PanicCount,
            result.Attempts,
            config.MaxAttempts,
        )
    }

    if !result.Success {
        if result.PanicCount >= 3 {
            // Persistent panic - bad job data
            wp.deadLetterQueue.Add(job, "repeated panics")
        }
        return result.LastError
    }

    return nil
}
\`\`\`

**Results:**
- Job panic rate: Still 1-2% (data quality issue)
- Worker crashes: 100% → 0% (retry handles panics)
- Worker count stability: 100 workers 24/7
- Processing capacity: Stable at 100%
- Manual restarts: Daily → Never
- Job success rate: 85% → 97%

**5. The Numbers: Real Production System**

**Microservices Architecture (50 services, 500M requests/day):**

**Before Retry with Panic Recovery:**
- Service panics: 1,000-2,000 per day
- Failed requests from panics: 1,000-2,000 per day
- Retry implemented manually (inconsistent)
- Some services retry, some don't
- Average retry attempts: 1.5
- Panic recovery rate: 40%

**After Standardized RetryWithPanic:**
- Service panics: Still 1,000-2,000 per day (underlying bugs)
- Failed requests: 100-200 per day
- Panic recovery rate: 90%
- Average retry attempts: 2.3
- Request success rate: 99.96% → 99.99%

**Business Impact:**
\`\`\`
Before:
- Failed requests: 2,000/day
- User-facing failures: 1,500/day (75%)
- Support tickets: 100/day
- Lost revenue: $10K/day
- Monthly cost: $300K

After:
- Failed requests: 200/day (90% recovered by retry)
- User-facing failures: 50/day (25%)
- Support tickets: 5/day
- Lost revenue: $500/day
- Monthly cost: $15K

Savings: $285K per month = $3.42M per year

Implementation cost: 1 week = ~$10K
ROI: 34,200% first year
\`\`\`

**Best Practices:**
1. **Always use exponential backoff** with jitter
2. **Configure max attempts** based on operation type (3-5 typical)
3. **Track retry metrics** for observability
4. **Alert on high panic rates** (>5%)
5. **Don't retry forever** - fail eventually
6. **Add jitter** to prevent thundering herd
7. **Distinguish panic vs error** for better debugging
8. **Log retry statistics** for each operation`,
	order: 3,
	translations: {
		ru: {
			title: 'Повтор при panic',
			solutionCode: `package panicrecover

import (
	"fmt"
	"math/rand"
	"strings"
	"time"
)

type RetryConfig struct {
	MaxAttempts   int           // макс попыток
	InitialDelay  time.Duration // начальная задержка
	MaxDelay      time.Duration // макс задержка
	Multiplier    float64       // множитель задержки
	RetryOnPanic  bool          // повторять при панике
}

type RetryResult struct {
	Attempts     int           // всего попыток
	PanicCount   int           // количество паник
	ErrorCount   int           // количество ошибок
	Success      bool          // флаг успеха
	LastError    error         // последняя ошибка
	Duration     time.Duration // общая длительность
}

func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:  3,                       // 3 попытки
		InitialDelay: 100 * time.Millisecond, // начать с 100мс
		MaxDelay:     5 * time.Second,         // макс 5с
		Multiplier:   2.0,                     // удваивать каждый раз
		RetryOnPanic: true,                    // повторять панику
	}
}

func RetryWithPanic(config RetryConfig, f func() error) *RetryResult {
	result := &RetryResult{
		Success: false,                        // начать как неудача
	}

	startTime := time.Now()                    // отслеживать длительность
	delay := config.InitialDelay               // текущая задержка

	for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
		result.Attempts = attempt              // номер попытки

		err := executeSafely(f)                // выполнить с восстановлением

		if err == nil {
			result.Success = true              // успех
			result.Duration = time.Since(startTime)
			return result
		}

		if isPanic(err) {                      // проверить паника ли
			result.PanicCount++
			if !config.RetryOnPanic {          // не повторять панику
				result.LastError = err
				result.Duration = time.Since(startTime)
				return result
			}
		} else {
			result.ErrorCount++                // обычная ошибка
		}

		result.LastError = err

		if attempt < config.MaxAttempts {      // не спать после последней попытки
			jitter := time.Duration(rand.Float64() * float64(delay) * 0.3) // 30% jitter
			sleepTime := delay + jitter

			if sleepTime > config.MaxDelay {   // ограничить задержку
				sleepTime = config.MaxDelay
			}

			time.Sleep(sleepTime)              // задержка

			delay = time.Duration(float64(delay) * config.Multiplier) // увеличить задержку
		}
	}

	result.Duration = time.Since(startTime)
	return result
}

func executeSafely(f func() error) error {
	var panicErr error                         // захватить ошибку

	func() {                                   // анонимная функция
		defer func() {
			if r := recover(); r != nil {      // перехватить панику
				panicErr = fmt.Errorf("panic: %v", r)
			}
		}()

		if f != nil {
			panicErr = f()                     // выполнить функцию
		}
	}()

	return panicErr                            // вернуть ошибку или nil
}

func isPanic(err error) bool {
	return err != nil && strings.HasPrefix(err.Error(), "panic:") // проверить префикс
}`,
			description: `Реализуйте production-grade механизм повторов с восстановлением от паники, экспоненциальной задержкой и настраиваемыми политиками повторов для построения устойчивых систем.

**Требования:**
1. **RetryConfig**: Конфигурация поведения повторов (макс попытки, задержка)
2. **RetryWithPanic**: Выполнение функции с повтором при ошибках и паниках
3. **Exponential Backoff**: Реализация экспоненциальной задержки с jitter
4. **Attempt Tracking**: Отслеживание попыток, паник и ошибок для наблюдаемости

**Паттерн повтора с восстановлением от паники:**
\`\`\`go
// Конфигурация повторов
type RetryConfig struct {
    MaxAttempts   int           // Максимальное количество попыток
    InitialDelay  time.Duration // Начальная задержка
    MaxDelay      time.Duration // Максимальная задержка
    Multiplier    float64       // Множитель задержки (напр., 2.0)
    RetryOnPanic  bool          // Повторять при панике
}

// Результат повтора для наблюдаемости
type RetryResult struct {
    Attempts     int           // Всего попыток
    PanicCount   int           // Количество паник
    ErrorCount   int           // Количество ошибок
    Success      bool          // Успех в итоге
    LastError    error         // Последняя ошибка/паника
    Duration     time.Duration // Общее время повторов
}

// Функция повтора с восстановлением от паники
func RetryWithPanic(config RetryConfig, f func() error) *RetryResult {
    result := &RetryResult{Success: false}
    startTime := time.Now()
    delay := config.InitialDelay

    for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
        result.Attempts = attempt
        err := executeSafely(f)

        if err == nil {
            result.Success = true
            result.Duration = time.Since(startTime)
            return result
        }

        if isPanic(err) {
            result.PanicCount++
            if !config.RetryOnPanic {
                result.LastError = err
                result.Duration = time.Since(startTime)
                return result
            }
        } else {
            result.ErrorCount++
        }

        result.LastError = err

        if attempt < config.MaxAttempts {
            jitter := time.Duration(rand.Float64() * float64(delay) * 0.3)
            sleepTime := delay + jitter
            if sleepTime > config.MaxDelay {
                sleepTime = config.MaxDelay
            }
            time.Sleep(sleepTime)
            delay = time.Duration(float64(delay) * config.Multiplier)
        }
    }

    result.Duration = time.Since(startTime)
    return result
}
\`\`\`

**Примеры использования:**
\`\`\`go
// Пример 1: Повтор вызова внешнего API
func FetchUserData(userID string) (*User, error) {
    config := RetryConfig{
        MaxAttempts:  5,
        InitialDelay: 100 * time.Millisecond,
        MaxDelay:     5 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    var user *User
    result := RetryWithPanic(config, func() error {
        var err error
        user, err = externalAPI.GetUser(userID)
        return err
    })

    if !result.Success {
        log.Printf("Не удалось после %d попыток (%d паник): %v",
            result.Attempts, result.PanicCount, result.LastError)
        return nil, result.LastError
    }
    return user, nil
}

// Пример 2: Операция с базой данных с повтором
func (db *Database) InsertWithRetry(data Record) error {
    config := DefaultRetryConfig()
    result := RetryWithPanic(config, func() error {
        return db.Insert(data)
    })

    if result.PanicCount > 0 {
        db.alertOps("Паника базы данных при вставке", result.LastError)
    }

    if !result.Success {
        return fmt.Errorf("вставка не удалась: %w", result.LastError)
    }
    return nil
}

// Пример 3: Обработка сообщений с выборочным повтором
func ProcessMessage(msg Message) error {
    config := RetryConfig{
        MaxAttempts:  3,
        InitialDelay: 50 * time.Millisecond,
        MaxDelay:     1 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    result := RetryWithPanic(config, func() error {
        return parseAndProcess(msg)
    })

    if !result.Success {
        if result.PanicCount > 0 {
            deadLetterQueue.AddWithReason(msg, "повторяющиеся паники", result.LastError)
        } else {
            deadLetterQueue.AddWithReason(msg, "ошибка обработки", result.LastError)
        }
        return result.LastError
    }
    return nil
}
\`\`\`

**Реальный production сценарий:**
\`\`\`go
// Production API клиент с повтором и наблюдаемостью
type APIClient struct {
    baseURL    string
    httpClient *http.Client
    metrics    *Metrics
    logger     *log.Logger
}

func (c *APIClient) CallWithRetry(endpoint string, payload interface{}) (*Response, error) {
    config := RetryConfig{
        MaxAttempts:  5,
        InitialDelay: 200 * time.Millisecond,
        MaxDelay:     10 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    var response *Response
    result := RetryWithPanic(config, func() error {
        req, err := c.buildRequest(endpoint, payload)
        if err != nil {
            return err
        }

        resp, err := c.httpClient.Do(req)
        if err != nil {
            return err
        }
        defer resp.Body.Close()

        if resp.StatusCode >= 500 {
            return fmt.Errorf("ошибка сервера: %d", resp.StatusCode)
        }
        if resp.StatusCode >= 400 {
            return &NonRetryableError{StatusCode: resp.StatusCode}
        }

        response, err = parseResponse(resp)
        return err
    })

    c.metrics.RecordAPICall(endpoint, result)

    if result.PanicCount > 0 {
        c.logger.Printf("API паника на %s: %d паник за %d попыток",
            endpoint, result.PanicCount, result.Attempts)
        c.metrics.IncrementAPIPanic(endpoint)
    }

    if !result.Success {
        return nil, result.LastError
    }
    return response, nil
}

// Worker pool с повтором при панике
type WorkerPool struct {
    workers     int
    retryConfig RetryConfig
    metrics     *Metrics
}

func (wp *WorkerPool) ProcessJobs(jobs <-chan Job) {
    var wg sync.WaitGroup
    for i := 0; i < wp.workers; i++ {
        wg.Add(1)
        workerID := i
        go func() {
            defer wg.Done()
            for job := range jobs {
                result := RetryWithPanic(wp.retryConfig, func() error {
                    return job.Execute()
                })

                wp.metrics.RecordJobResult(workerID, job.Type, result)

                if result.PanicCount > 0 {
                    log.Printf("Worker %d: Job %s паниковал %d раз",
                        workerID, job.ID, result.PanicCount)
                }

                if !result.Success {
                    wp.handleFailedJob(job, result)
                }
            }
        }()
    }
    wg.Wait()
}
\`\`\`

**Ограничения:**
- RetryConfig должен иметь все обязательные поля
- RetryWithPanic должен соблюдать лимит MaxAttempts
- Экспоненциальная задержка должна включать jitter для предотвращения thundering herd
- Не спать после последней неудачной попытки
- RetryResult должен отслеживать все попытки, паники и ошибки
- Необходимо различать паники и обычные ошибки`,
			hint1: `Определите структуры со всеми необходимыми полями. DefaultRetryConfig возвращает RetryConfig{MaxAttempts: 3, InitialDelay: 100*time.Millisecond, MaxDelay: 5*time.Second, Multiplier: 2.0, RetryOnPanic: true}.`,
			hint2: `В RetryWithPanic: цикл от 1 до MaxAttempts. Вызовите executeSafely(f). Если err==nil, установите Success=true и верните. Отслеживайте паники/ошибки. Спите с jitter перед следующей попыткой (не после последней). В executeSafely: используйте defer recover для перехвата паник.`,
			whyItMatters: `Повтор с восстановлением от паники критически важен для построения устойчивых систем, которые могут автоматически восстанавливаться от временных сбоев и неожиданных паник без вмешательства человека. Это основа production-ready микросервисной архитектуры.

**Почему это важно:**

**1. Production инцидент: Сбой платежного шлюза**

Fintech компания обрабатывала 100K+ транзакций в день:
- API платежного шлюза периодически паниковал (баг SDK третьей стороны)
- Одна паника - транзакция потеряна навсегда
- Механизм повтора отсутствовал
- Транзакция клиента потеряна
- Требовался ручной возврат ($50 за инцидент)
- 50-100 паник в день
- Ежедневные потери: $2.5K-$5K в возвратах
- Ежемесячные потери: $75K-$150K

**Корневая причина:**
\`\`\`go
// До: Без повтора при панике
func ProcessPayment(payment Payment) error {
    // SDK платежного шлюза паникует при сетевом сбое
    result := gateway.Charge(payment)
    return processResult(result)
}
// Результат: Паника = постоянный сбой транзакции
\`\`\`

**Решение с RetryWithPanic:**
\`\`\`go
// После: Повтор с восстановлением от паники
func ProcessPayment(payment Payment) error {
    config := RetryConfig{
        MaxAttempts:  5,
        InitialDelay: 100 * time.Millisecond,
        MaxDelay:     5 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    var result *PaymentResult
    retryResult := RetryWithPanic(config, func() error {
        result = gateway.Charge(payment)
        return processResult(result)
    })

    if !retryResult.Success {
        log.Printf("Платеж не удался после %d попыток (%d паник): %v",
            retryResult.Attempts, retryResult.PanicCount, retryResult.LastError)
        return retryResult.LastError
    }

    if retryResult.Attempts > 1 {
        log.Printf("Платеж успешен на попытке %d (восстановлено из %d паник)",
            retryResult.Attempts, retryResult.PanicCount)
    }
    return nil
}
\`\`\`

**Результат после исправления:**
- Паник в день: всё ещё 50-100 (баг SDK не исправлен)
- Неудачные транзакции: снижение на 95% (95-100 восстановлены повтором)
- Затраты на возвраты: $75K-$150K/месяц → $3.75K-$7.5K/месяц
- Годовая экономия: $855K-$1.7M
- Удовлетворенность клиентов: +40%

**2. Реальный случай: Надежность внешних API**

SaaS платформа интегрируется с 20+ внешними API:

**До RetryWithPanic:**
- Внешние API иногда паникуют (некорректные ответы, nil pointers)
- Любая паника = функция сломана для пользователя
- Пользователи сообщают "функция не работает"
- Команда поддержки расследует каждый инцидент
- Среднее время решения: 2-4 часа (ожидание исправления API)

**Проблема:**
\`\`\`go
// Ненадёжный внешний API
func FetchUserProfile(userID string) (*Profile, error) {
    // SDK API паникует при неожиданном формате ответа
    profile := externalAPI.GetProfile(userID)
    return profile, nil
}
// Результат: Один некорректный ответ ломает функцию
\`\`\`

**Решение:**
\`\`\`go
func FetchUserProfile(userID string) (*Profile, error) {
    config := DefaultRetryConfig()
    config.MaxAttempts = 5

    var profile *Profile
    result := RetryWithPanic(config, func() error {
        var err error
        profile, err = externalAPI.GetProfile(userID)
        return err
    })

    if !result.Success {
        metrics.IncrementAPIFailure("profile_fetch")

        if result.PanicCount > 0 {
            // API запаниковал - алерт, но не блокируем пользователя
            alerting.Send(fmt.Sprintf(
                "Profile API паниковал %d раз для пользователя %s",
                result.PanicCount, userID,
            ))
        }
        return nil, result.LastError
    }

    if result.PanicCount > 0 {
        log.Printf("Восстановлено из %d паник при получении профиля %s",
            result.PanicCount, userID)
        metrics.IncrementAPIRetrySuccess("profile_fetch")
    }

    return profile, nil
}
\`\`\`

**Результаты:**
- API паники: 200-300 в день (внешняя проблема)
- Доступность функций: 85% → 99%
- Успешные восстановления: 90%+ (паники решены повтором)
- Тикеты поддержки: 100/неделя → 10/неделя
- Удовлетворенность пользователей: +50%

**3. Production метрики: Восстановление от deadlock БД**

E-commerce платформа с высоким объёмом транзакций:

**До Retry:**
- Database deadlock вызывают паники транзакций
- 50-100 deadlock в час в пиковое время
- Каждый deadlock = неудачный заказ
- Пользователи видят ошибку "checkout failed"
- Процент брошенных корзин: 15%

**После внедрения RetryWithPanic:**
\`\`\`go
func (db *Database) ExecuteTransaction(tx func() error) error {
    config := RetryConfig{
        MaxAttempts:  5,
        InitialDelay: 50 * time.Millisecond,
        MaxDelay:     1 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    result := RetryWithPanic(config, tx)

    if result.PanicCount > 0 {
        // Паника БД (deadlock или constraint violation)
        metrics.RecordDBPanic(result.PanicCount)
    }

    if !result.Success {
        log.Printf(
            "Транзакция не удалась после %d попыток (%d паник, %d ошибок)",
            result.Attempts, result.PanicCount, result.ErrorCount,
        )
        return result.LastError
    }

    if result.Attempts > 1 {
        log.Printf(
            "Транзакция успешна после %d попыток (восстановлено из %d паник)",
            result.Attempts, result.PanicCount,
        )
        metrics.RecordDBRetrySuccess()
    }

    return nil
}
\`\`\`

**Результаты:**
- Deadlock в час: всё ещё 50-100 (конфликт БД)
- Успех транзакций: 75% → 98%
- Восстановлено повтором: 92%
- Неудачные заказы: сокращение на 75%
- Брошенные корзины: 15% → 8%
- Влияние на выручку: +$500K в месяц

**4. Устойчивость Worker Pool**

Сервис обработки медиа со 100 воркерами:

**Статистика до Retry:**
- Процент паник воркеров: 1-2% задач
- Паника = воркер падает навсегда
- Потерянные воркеры накапливаются со временем
- Начало: 100 воркеров → 50 воркеров через 24 часа
- Мощность обработки деградирует на 50%
- Требуется ежедневный ручной перезапуск

**Реализация:**
\`\`\`go
func (wp *WorkerPool) ProcessWithRetry(job Job) error {
    config := RetryConfig{
        MaxAttempts:  3,
        InitialDelay: 100 * time.Millisecond,
        MaxDelay:     2 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    result := RetryWithPanic(config, func() error {
        return job.Execute()
    })

    wp.metrics.RecordJobResult(job.Type, result)

    if result.PanicCount > 0 {
        log.Printf(
            "Job %s: Восстановлено из %d паник, попытка %d/%d",
            job.ID, result.PanicCount, result.Attempts, config.MaxAttempts,
        )
    }

    if !result.Success {
        if result.PanicCount >= 3 {
            // Постоянная паника - плохие данные задачи
            wp.deadLetterQueue.Add(job, "repeated panics")
        }
        return result.LastError
    }

    return nil
}
\`\`\`

**Результаты:**
- Процент паник задач: всё ещё 1-2% (проблема качества данных)
- Падения воркеров: 100% → 0% (retry обрабатывает паники)
- Стабильность количества воркеров: 100 воркеров 24/7
- Мощность обработки: Стабильно 100%
- Ручные перезапуски: Ежедневно → Никогда
- Успешность задач: 85% → 97%

**5. Цифры: Реальная Production система**

**Микросервисная архитектура (50 сервисов, 500M запросов/день):**

**До Retry с восстановлением от паники:**
- Паники сервисов: 1,000-2,000 в день
- Неудачные запросы из-за паник: 1,000-2,000 в день
- Retry реализован вручную (непоследовательно)
- Некоторые сервисы повторяют, некоторые нет
- Среднее количество попыток: 1.5
- Процент восстановления от паник: 40%

**После стандартизации RetryWithPanic:**
- Паники сервисов: всё ещё 1,000-2,000 в день (базовые баги)
- Неудачные запросы: 100-200 в день
- Процент восстановления от паник: 90%
- Среднее количество попыток: 2.3
- Успешность запросов: 99.96% → 99.99%

**Бизнес-влияние:**
\`\`\`
До:
- Неудачные запросы: 2,000/день
- Видимые пользователям сбои: 1,500/день (75%)
- Тикеты поддержки: 100/день
- Потерянная выручка: $10K/день
- Месячные потери: $300K

После:
- Неудачные запросы: 200/день (90% восстановлены retry)
- Видимые пользователям сбои: 50/день (25%)
- Тикеты поддержки: 5/день
- Потерянная выручка: $500/день
- Месячные потери: $15K

Экономия: $285K в месяц = $3.42M в год

Стоимость реализации: 1 неделя = ~$10K
ROI: 34,200% за первый год
\`\`\`

**6. Интеграция с Circuit Breaker**

Для критических систем комбинируйте retry с circuit breaker:
\`\`\`go
type CircuitBreaker struct {
    maxFailures  int
    failures     int
    lastFailTime time.Time
    state        string // "closed", "open", "half-open"
    retryConfig  RetryConfig
}

func (cb *CircuitBreaker) Call(f func() error) error {
    if cb.state == "open" {
        // Circuit открыт - fail fast
        if time.Since(cb.lastFailTime) < 30*time.Second {
            return errors.New("circuit breaker open")
        }
        // Попробовать закрыть circuit
        cb.state = "half-open"
    }

    result := RetryWithPanic(cb.retryConfig, f)

    if result.Success {
        // Успех - сбросить счётчик сбоев
        cb.failures = 0
        cb.state = "closed"
        return nil
    }

    // Сбой - увеличить счётчик
    cb.failures++
    cb.lastFailTime = time.Now()

    if cb.failures >= cb.maxFailures {
        cb.state = "open"
        log.Printf("Circuit breaker открыт после %d сбоев", cb.failures)
    }

    return result.LastError
}
\`\`\`

**7. Наблюдаемость и мониторинг**

Production системы должны отслеживать retry метрики:
\`\`\`go
type RetryMetrics struct {
    mu                 sync.Mutex
    totalRetries       int64
    successfulRetries  int64
    failedRetries      int64
    totalPanics        int64
    retryDurations     []time.Duration
}

func (rm *RetryMetrics) Record(result *RetryResult) {
    rm.mu.Lock()
    defer rm.mu.Unlock()

    rm.totalRetries++
    rm.totalPanics += int64(result.PanicCount)

    if result.Success {
        rm.successfulRetries++
    } else {
        rm.failedRetries++
    }

    rm.retryDurations = append(rm.retryDurations, result.Duration)
}

func (rm *RetryMetrics) Report() {
    rm.mu.Lock()
    defer rm.mu.Unlock()

    successRate := float64(rm.successfulRetries) / float64(rm.totalRetries) * 100
    panicRate := float64(rm.totalPanics) / float64(rm.totalRetries) * 100

    avgDuration := time.Duration(0)
    if len(rm.retryDurations) > 0 {
        var total time.Duration
        for _, d := range rm.retryDurations {
            total += d
        }
        avgDuration = total / time.Duration(len(rm.retryDurations))
    }

    log.Printf(\`Retry метрики:
  Всего повторов: %d
  Процент успеха: %.2f%%
  Неудачные повторы: %d
  Всего паник: %d
  Процент паник: %.2f%%
  Средняя длительность: %v
\`, rm.totalRetries, successRate, rm.failedRetries, rm.totalPanics, panicRate, avgDuration)
}
\`\`\`

**Ключевые метрики для алертинга:**
- **Panic rate >5%**: Расследовать базовую проблему
- **Retry success rate <50%**: Конфигурация может быть недостаточной
- **Average attempts >3**: Сервис может быть перегружен
- **Failed retries spike**: Возможный полный сбой downstream сервиса

**Лучшие практики:**
1. **Всегда используйте экспоненциальную задержку** с jitter (30% типично)
2. **Настраивайте max attempts** в зависимости от типа операции (3-5 обычно)
3. **Отслеживайте метрики повторов** для наблюдаемости (успех, паники, длительность)
4. **Алертите при высоком проценте паник** (>5% требует расследования)
5. **Не повторяйте вечно** - в конце концов отказывайте (fail gracefully)
6. **Добавляйте jitter** для предотвращения thundering herd эффекта
7. **Различайте паники и ошибки** для лучшей отладки (разные код-пути)
8. **Логируйте статистику повторов** для каждой операции (помогает в диагностике)
9. **Используйте context для таймаутов** - не полагайтесь только на retry
10. **Тестируйте сценарии паники** - симулируйте сбои в тестах

Retry с восстановлением от паники - это не просто оптимизация, это необходимость для production систем. Правильная реализация может сэкономить миллионы и предотвратить критические сбои.`
		},
		uz: {
			title: `Panicda qayta urinish`,
			solutionCode: `package panicrecover

import (
	"fmt"
	"math/rand"
	"strings"
	"time"
)

type RetryConfig struct {
	MaxAttempts   int           // maksimal urinishlar
	InitialDelay  time.Duration // boshlang'ich kechikish
	MaxDelay      time.Duration // maksimal kechikish
	Multiplier    float64       // kechikish ko'paytiruvchisi
	RetryOnPanic  bool          // panikda qayta urinish
}

type RetryResult struct {
	Attempts     int           // jami urinishlar
	PanicCount   int           // paniklar soni
	ErrorCount   int           // xatolar soni
	Success      bool          // muvaffaqiyat belgisi
	LastError    error         // oxirgi xato
	Duration     time.Duration // umumiy davomiyligi
}

func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:  3,                       // 3 urinish
		InitialDelay: 100 * time.Millisecond, // 100ms dan boshlash
		MaxDelay:     5 * time.Second,         // 5s gacha
		Multiplier:   2.0,                     // har safar ikki barobar
		RetryOnPanic: true,                    // panikni qayta urinish
	}
}

func RetryWithPanic(config RetryConfig, f func() error) *RetryResult {
	result := &RetryResult{
		Success: false,                        // muvaffaqiyatsiz boshlanadi
	}

	startTime := time.Now()                    // davomiylikni kuzatish
	delay := config.InitialDelay               // joriy kechikish

	for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
		result.Attempts = attempt              // urinish raqami

		err := executeSafely(f)                // tiklanish bilan bajarish

		if err == nil {
			result.Success = true              // muvaffaqiyat
			result.Duration = time.Since(startTime)
			return result
		}

		if isPanic(err) {                      // panik ekanligini tekshirish
			result.PanicCount++
			if !config.RetryOnPanic {          // panikni qayta urinmaslik
				result.LastError = err
				result.Duration = time.Since(startTime)
				return result
			}
		} else {
			result.ErrorCount++                // oddiy xato
		}

		result.LastError = err

		if attempt < config.MaxAttempts {      // oxirgi urinishdan keyin uxlamaslik
			jitter := time.Duration(rand.Float64() * float64(delay) * 0.3) // 30% jitter
			sleepTime := delay + jitter

			if sleepTime > config.MaxDelay {   // kechikishni cheklash
				sleepTime = config.MaxDelay
			}

			time.Sleep(sleepTime)              // kutish

			delay = time.Duration(float64(delay) * config.Multiplier) // kechikishni oshirish
		}
	}

	result.Duration = time.Since(startTime)
	return result
}

func executeSafely(f func() error) error {
	var panicErr error                         // xatoni ushlash

	func() {                                   // anonim funksiya
		defer func() {
			if r := recover(); r != nil {      // panikni ushlash
				panicErr = fmt.Errorf("panic: %v", r)
			}
		}()

		if f != nil {
			panicErr = f()                     // funksiyani bajarish
		}
	}()

	return panicErr                            // xato yoki nil qaytarish
}

func isPanic(err error) bool {
	return err != nil && strings.HasPrefix(err.Error(), "panic:") // prefiksni tekshirish
}`,
			description: `Barqaror tizimlar qurish uchun panikdan tiklanish, eksponensial backoff va sozlanishi mumkin retry siyosatlari bilan production-grade retry mexanizmini amalga oshiring.

**Talablar:**
1. **RetryConfig**: Retry xatti-harakati uchun konfiguratsiya (maksimal urinishlar, backoff)
2. **RetryWithPanic**: Xato va paniklarda qayta urinish bilan funksiyani bajaring
3. **Exponential Backoff**: Jitter bilan eksponensial backoff ni amalga oshiring
4. **Attempt Tracking**: Kuzatiluvchanlik uchun urinishlar, paniklar va xatolarni kuzating

**Panikdan tiklanish bilan retry pattern:**
\`\`\`go
// Retry konfiguratsiyasi
type RetryConfig struct {
    MaxAttempts   int           // Maksimal urinishlar soni
    InitialDelay  time.Duration // Boshlang'ich kechikish
    MaxDelay      time.Duration // Maksimal kechikish
    Multiplier    float64       // Kechikish ko'paytiruvchisi (masalan, 2.0)
    RetryOnPanic  bool          // Panikda qayta urinish
}

// Kuzatiluvchanlik uchun retry natijasi
type RetryResult struct {
    Attempts     int           // Jami urinishlar
    PanicCount   int           // Paniklar soni
    ErrorCount   int           // Xatolar soni
    Success      bool          // Oxirida muvaffaqiyat
    LastError    error         // Oxirgi xato/panik
    Duration     time.Duration // Umumiy retry vaqti
}

// Panikdan tiklanish bilan retry funksiyasi
func RetryWithPanic(config RetryConfig, f func() error) *RetryResult {
    result := &RetryResult{Success: false}
    startTime := time.Now()
    delay := config.InitialDelay

    for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
        result.Attempts = attempt
        err := executeSafely(f)

        if err == nil {
            result.Success = true
            result.Duration = time.Since(startTime)
            return result
        }

        if isPanic(err) {
            result.PanicCount++
            if !config.RetryOnPanic {
                result.LastError = err
                result.Duration = time.Since(startTime)
                return result
            }
        } else {
            result.ErrorCount++
        }

        result.LastError = err

        if attempt < config.MaxAttempts {
            jitter := time.Duration(rand.Float64() * float64(delay) * 0.3)
            sleepTime := delay + jitter
            if sleepTime > config.MaxDelay {
                sleepTime = config.MaxDelay
            }
            time.Sleep(sleepTime)
            delay = time.Duration(float64(delay) * config.Multiplier)
        }
    }

    result.Duration = time.Since(startTime)
    return result
}
\`\`\`

**Foydalanish misollari:**
\`\`\`go
// Misol 1: Tashqi API chaqiruvini qayta urinish
func FetchUserData(userID string) (*User, error) {
    config := RetryConfig{
        MaxAttempts:  5,
        InitialDelay: 100 * time.Millisecond,
        MaxDelay:     5 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    var user *User
    result := RetryWithPanic(config, func() error {
        var err error
        user, err = externalAPI.GetUser(userID)
        return err
    })

    if !result.Success {
        log.Printf("%d urinishdan keyin muvaffaqiyatsiz (%d panik): %v",
            result.Attempts, result.PanicCount, result.LastError)
        return nil, result.LastError
    }
    return user, nil
}

// Misol 2: Ma'lumotlar bazasi operatsiyasi retry bilan
func (db *Database) InsertWithRetry(data Record) error {
    config := DefaultRetryConfig()
    result := RetryWithPanic(config, func() error {
        return db.Insert(data)
    })

    if result.PanicCount > 0 {
        db.alertOps("Insert da ma'lumotlar bazasi paniki", result.LastError)
    }

    if !result.Success {
        return fmt.Errorf("insert muvaffaqiyatsiz: %w", result.LastError)
    }
    return nil
}

// Misol 3: Tanlab qayta urinish bilan xabarni qayta ishlash
func ProcessMessage(msg Message) error {
    config := RetryConfig{
        MaxAttempts:  3,
        InitialDelay: 50 * time.Millisecond,
        MaxDelay:     1 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    result := RetryWithPanic(config, func() error {
        return parseAndProcess(msg)
    })

    if !result.Success {
        if result.PanicCount > 0 {
            deadLetterQueue.AddWithReason(msg, "takroriy paniklar", result.LastError)
        } else {
            deadLetterQueue.AddWithReason(msg, "qayta ishlash xatosi", result.LastError)
        }
        return result.LastError
    }
    return nil
}
\`\`\`

**Haqiqiy production stsenariy:**
\`\`\`go
// Production API mijoz retry va kuzatiluvchanlik bilan
type APIClient struct {
    baseURL    string
    httpClient *http.Client
    metrics    *Metrics
    logger     *log.Logger
}

func (c *APIClient) CallWithRetry(endpoint string, payload interface{}) (*Response, error) {
    config := RetryConfig{
        MaxAttempts:  5,
        InitialDelay: 200 * time.Millisecond,
        MaxDelay:     10 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    var response *Response
    result := RetryWithPanic(config, func() error {
        req, err := c.buildRequest(endpoint, payload)
        if err != nil {
            return err
        }

        resp, err := c.httpClient.Do(req)
        if err != nil {
            return err
        }
        defer resp.Body.Close()

        if resp.StatusCode >= 500 {
            return fmt.Errorf("server xatosi: %d", resp.StatusCode)
        }
        if resp.StatusCode >= 400 {
            return &NonRetryableError{StatusCode: resp.StatusCode}
        }

        response, err = parseResponse(resp)
        return err
    })

    c.metrics.RecordAPICall(endpoint, result)

    if result.PanicCount > 0 {
        c.logger.Printf("%s da API panik: %d urinishda %d panik",
            endpoint, result.Attempts, result.PanicCount)
        c.metrics.IncrementAPIPanic(endpoint)
    }

    if !result.Success {
        return nil, result.LastError
    }
    return response, nil
}

// Panikda qayta urinish bilan Worker pool
type WorkerPool struct {
    workers     int
    retryConfig RetryConfig
    metrics     *Metrics
}

func (wp *WorkerPool) ProcessJobs(jobs <-chan Job) {
    var wg sync.WaitGroup
    for i := 0; i < wp.workers; i++ {
        wg.Add(1)
        workerID := i
        go func() {
            defer wg.Done()
            for job := range jobs {
                result := RetryWithPanic(wp.retryConfig, func() error {
                    return job.Execute()
                })

                wp.metrics.RecordJobResult(workerID, job.Type, result)

                if result.PanicCount > 0 {
                    log.Printf("Worker %d: Job %s %d marta panik qildi",
                        workerID, job.ID, result.PanicCount)
                }

                if !result.Success {
                    wp.handleFailedJob(job, result)
                }
            }
        }()
    }
    wg.Wait()
}
\`\`\`

**Cheklovlar:**
- RetryConfig barcha kerakli maydonlarga ega bo'lishi kerak
- RetryWithPanic MaxAttempts chegarasiga rioya qilishi kerak
- Eksponensial backoff thundering herd dan qochish uchun jitter ni o'z ichiga olishi kerak
- Oxirgi muvaffaqiyatsiz urinishdan keyin uxlamaslik kerak
- RetryResult barcha urinishlar, paniklar va xatolarni kuzatishi kerak
- Paniklar va oddiy xatolarni farqlash kerak`,
			hint1: `Barcha kerakli maydonlar bilan struct larni aniqlang. DefaultRetryConfig RetryConfig{MaxAttempts: 3, InitialDelay: 100*time.Millisecond, MaxDelay: 5*time.Second, Multiplier: 2.0, RetryOnPanic: true} qaytaradi.`,
			hint2: `RetryWithPanic da: 1 dan MaxAttempts gacha tsikl. executeSafely(f) ni chaqiring. err==nil bo'lsa, Success=true o'rnating va qaytaring. Panik/xatolarni kuzating. Keyingi urinishdan oldin jitter bilan uxlang (oxirgisidan keyin emas). executeSafely da: paniklarni ushlash uchun defer recover ishlating.`,
			whyItMatters: `Panikdan tiklanish bilan retry — insonning aralashuvisiz vaqtinchalik nosozliklar va kutilmagan paniklardan avtomatik tiklanishi mumkin bo'lgan barqaror tizimlar qurish uchun muhim. Bu production-ready mikroservis arxitekturasi uchun zarurdir.

**Nima uchun bu muhim:**

**1. Production incident: To'lov gateway nosozligi**

Fintech kompaniyasi kuniga 100K+ tranzaksiya qayta ishladi:
- To'lov gateway API vaqti-vaqti bilan panik qildi (3-chi tomon SDK bug)
- Bitta panik = tranzaksiya abadiy yo'qoldi
- Retry mexanizmi yo'q edi
- Mijoz tranzaksiyasi yo'qoldi
- Qo'lda qaytarish talab qilindi ($50 har bir hodisa uchun)
- Kuniga 50-100 panik
- Kunlik yo'qotish: $2.5K-$5K
- Oylik yo'qotish: $75K-$150K

**Asosiy sabab:**
\`\`\`go
// Oldin: Panikda retry yo'q
func ProcessPayment(payment Payment) error {
    // To'lov gateway SDK tarmoq nosozligida panik qiladi
    result := gateway.Charge(payment)
    return processResult(result)
}
// Natija: Panik = doimiy tranzaksiya muvaffaqiyatsizligi
\`\`\`

**RetryWithPanic bilan yechim:**
\`\`\`go
// Keyin: Panikdan tiklanish bilan retry
func ProcessPayment(payment Payment) error {
    config := RetryConfig{
        MaxAttempts:  5,
        InitialDelay: 100 * time.Millisecond,
        MaxDelay:     5 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    var result *PaymentResult
    retryResult := RetryWithPanic(config, func() error {
        result = gateway.Charge(payment)
        return processResult(result)
    })

    if !retryResult.Success {
        log.Printf("%d urinishdan keyin to'lov muvaffaqiyatsiz (%d panik): %v",
            retryResult.Attempts, retryResult.PanicCount, retryResult.LastError)
        return retryResult.LastError
    }

    if retryResult.Attempts > 1 {
        log.Printf("To'lov %d-urinishda muvaffaqiyatli (%d panikdan tiklandi)",
            retryResult.Attempts, retryResult.PanicCount)
    }
    return nil
}
\`\`\`

**Tuzatishdan keyingi natija:**
- Kunlik paniklar: hali ham 50-100 (SDK bug tuzatilmagan)
- Muvaffaqiyatsiz tranzaksiyalar: 95% kamayish (95-100 retry orqali tiklandi)
- Qaytarish xarajatlari: $75K-$150K/oy → $3.75K-$7.5K/oy
- Yillik tejamkor: $855K-$1.7M
- Mijoz qoniqishi: +40%

**2. Haqiqiy holat: Tashqi API ishonchliligi**

20+ tashqi API bilan integratsiyalashgan SaaS platformasi:

**RetryWithPanic dan oldin:**
- Tashqi API lar ba'zan panik qiladi (noto'g'ri javoblar, nil pointerlar)
- Har qanday panik = foydalanuvchi uchun funksiya buzilgan
- Foydalanuvchilar "funksiya ishlamayapti" deb xabar beradi
- Qo'llab-quvvatlash jamoasi har bir hodisani tekshiradi
- O'rtacha hal qilish vaqti: 2-4 soat (API tuzatilishini kutish)

**Muammo:**
\`\`\`go
// Ishonchsiz tashqi API
func FetchUserProfile(userID string) (*Profile, error) {
    // API SDK kutilmagan javob formatida panik qiladi
    profile := externalAPI.GetProfile(userID)
    return profile, nil
}
// Natija: Bitta noto'g'ri javob funksiyani buzadi
\`\`\`

**Yechim:**
\`\`\`go
func FetchUserProfile(userID string) (*Profile, error) {
    config := DefaultRetryConfig()
    config.MaxAttempts = 5

    var profile *Profile
    result := RetryWithPanic(config, func() error {
        var err error
        profile, err = externalAPI.GetProfile(userID)
        return err
    })

    if !result.Success {
        metrics.IncrementAPIFailure("profile_fetch")

        if result.PanicCount > 0 {
            // API panik qildi - ogohlantiring lekin foydalanuvchini bloklamang
            alerting.Send(fmt.Sprintf(
                "Profile API %d marta panik qildi foydalanuvchi %s uchun",
                result.PanicCount, userID,
            ))
        }
        return nil, result.LastError
    }

    if result.PanicCount > 0 {
        log.Printf("%d panikdan tiklanildi profil %s ni olishda",
            result.PanicCount, userID)
        metrics.IncrementAPIRetrySuccess("profile_fetch")
    }

    return profile, nil
}
\`\`\`

**Natijalar:**
- API paniklar: kuniga 200-300 (tashqi muammo)
- Funksiya mavjudligi: 85% → 99%
- Muvaffaqiyatli tiklanishlar: 90%+ (paniklar retry orqali hal qilindi)
- Qo'llab-quvvatlash tiketlari: 100/hafta → 10/hafta
- Foydalanuvchi qoniqishi: +50%

**3. Production metrikalar: Ma'lumotlar bazasi deadlock tiklanishi**

Yuqori tranzaksiya hajmi bo'lgan e-commerce platformasi:

**Retry dan oldin:**
- Database deadlock lar tranzaksiya paniklariga olib keladi
- Peak vaqtda soatiga 50-100 deadlock
- Har bir deadlock = muvaffaqiyatsiz buyurtma
- Foydalanuvchilar "checkout failed" xatosini ko'radi
- Savatni tashlab ketish darajasi: 15%

**RetryWithPanic joriy etilgandan keyin:**
\`\`\`go
func (db *Database) ExecuteTransaction(tx func() error) error {
    config := RetryConfig{
        MaxAttempts:  5,
        InitialDelay: 50 * time.Millisecond,
        MaxDelay:     1 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    result := RetryWithPanic(config, tx)

    if result.PanicCount > 0 {
        // Database panik (deadlock yoki constraint violation)
        metrics.RecordDBPanic(result.PanicCount)
    }

    if !result.Success {
        log.Printf(
            "Tranzaksiya %d urinishdan keyin muvaffaqiyatsiz (%d panik, %d xato)",
            result.Attempts, result.PanicCount, result.ErrorCount,
        )
        return result.LastError
    }

    if result.Attempts > 1 {
        log.Printf(
            "Tranzaksiya %d urinishdan keyin muvaffaqiyatli (%d panikdan tiklandi)",
            result.Attempts, result.PanicCount,
        )
        metrics.RecordDBRetrySuccess()
    }

    return nil
}
\`\`\`

**Natijalar:**
- Soatiga deadlock: hali ham 50-100 (database raqobat)
- Tranzaksiya muvaffaqiyati: 75% → 98%
- Retry orqali tiklangan: 92%
- Muvaffaqiyatsiz buyurtmalar: 75% kamayish
- Savatni tashlab ketish: 15% → 8%
- Daromadga ta'sir: oyiga +$500K

**4. Worker Pool barqarorligi**

100 worker bilan media qayta ishlash xizmati:

**Retry dan oldin statistika:**
- Worker panik darajasi: vazifalarning 1-2%
- Panik = worker abadiy qulab tushadi
- Yo'qolgan workerlar vaqt o'tishi bilan to'planadi
- Boshlash: 100 worker → 24 soatdan keyin 50 worker
- Qayta ishlash quvvati 50% pasayadi
- Kunlik qo'lda qayta ishga tushirish talab qilinadi

**Joriy etish:**
\`\`\`go
func (wp *WorkerPool) ProcessWithRetry(job Job) error {
    config := RetryConfig{
        MaxAttempts:  3,
        InitialDelay: 100 * time.Millisecond,
        MaxDelay:     2 * time.Second,
        Multiplier:   2.0,
        RetryOnPanic: true,
    }

    result := RetryWithPanic(config, func() error {
        return job.Execute()
    })

    wp.metrics.RecordJobResult(job.Type, result)

    if result.PanicCount > 0 {
        log.Printf(
            "Vazifa %s: %d panikdan tiklandi, urinish %d/%d",
            job.ID, result.PanicCount, result.Attempts, config.MaxAttempts,
        )
    }

    if !result.Success {
        if result.PanicCount >= 3 {
            // Doimiy panik - yomon vazifa ma'lumotlari
            wp.deadLetterQueue.Add(job, "takroriy paniklar")
        }
        return result.LastError
    }

    return nil
}
\`\`\`

**Natijalar:**
- Vazifa panik darajasi: hali ham 1-2% (ma'lumot sifati muammosi)
- Worker qulab tushishlari: 100% → 0% (retry panikalarni boshqaradi)
- Worker soni barqarorligi: 24/7 100 worker
- Qayta ishlash quvvati: Barqaror 100%
- Qo'lda qayta ishga tushirish: Kundalik → Hech qachon
- Vazifa muvaffaqiyati: 85% → 97%

**5. Raqamlar: Haqiqiy Production tizimi**

**Mikroservis arxitekturasi (50 servis, kuniga 500M so'rov):**

**Panikdan tiklanish bilan Retry dan oldin:**
- Servis paniklari: kuniga 1,000-2,000
- Paniklar tufayli muvaffaqiyatsiz so'rovlar: kuniga 1,000-2,000
- Retry qo'lda amalga oshirilgan (izchil emas)
- Ba'zi servislar retry qiladi, ba'zilari yo'q
- O'rtacha retry urinishlari: 1.5
- Panikdan tiklanish darajasi: 40%

**RetryWithPanic standartlashtirilgandan keyin:**
- Servis paniklari: hali ham kuniga 1,000-2,000 (asosiy buglar)
- Muvaffaqiyatsiz so'rovlar: kuniga 100-200
- Panikdan tiklanish darajasi: 90%
- O'rtacha retry urinishlari: 2.3
- So'rov muvaffaqiyati: 99.96% → 99.99%

**Biznesga ta'siri:**
\`\`\`
Oldin:
- Muvaffaqiyatsiz so'rovlar: 2,000/kun
- Foydalanuvchiga ko'rinadigan nosozliklar: 1,500/kun (75%)
- Qo'llab-quvvatlash tiketlari: 100/kun
- Yo'qolgan daromad: $10K/kun
- Oylik xarajat: $300K

Keyin:
- Muvaffaqiyatsiz so'rovlar: 200/kun (90% retry orqali tiklandi)
- Foydalanuvchiga ko'rinadigan nosozliklar: 50/kun (25%)
- Qo'llab-quvvatlash tiketlari: 5/kun
- Yo'qolgan daromad: $500/kun
- Oylik xarajat: $15K

Tejamkorlik: oyiga $285K = yiliga $3.42M

Joriy etish xarajati: 1 hafta = ~$10K
ROI: Birinchi yilda 34,200%
\`\`\`

**6. Circuit Breaker bilan integratsiya**

Kritik tizimlar uchun retry ni circuit breaker bilan birlashtiring:
\`\`\`go
type CircuitBreaker struct {
    maxFailures  int
    failures     int
    lastFailTime time.Time
    state        string // "closed", "open", "half-open"
    retryConfig  RetryConfig
}

func (cb *CircuitBreaker) Call(f func() error) error {
    if cb.state == "open" {
        // Circuit ochiq - tez fail qiling
        if time.Since(cb.lastFailTime) < 30*time.Second {
            return errors.New("circuit breaker ochiq")
        }
        // Circuit ni yopishga harakat qiling
        cb.state = "half-open"
    }

    result := RetryWithPanic(cb.retryConfig, f)

    if result.Success {
        // Muvaffaqiyat - nosozliklar hisobini tiklash
        cb.failures = 0
        cb.state = "closed"
        return nil
    }

    // Nosozlik - hisobni oshirish
    cb.failures++
    cb.lastFailTime = time.Now()

    if cb.failures >= cb.maxFailures {
        cb.state = "open"
        log.Printf("%d nosozlikdan keyin circuit breaker ochildi", cb.failures)
    }

    return result.LastError
}
\`\`\`

**7. Kuzatiluvchanlik va monitoring**

Production tizimlari retry metrikalarini kuzatishi kerak:
\`\`\`go
type RetryMetrics struct {
    mu                 sync.Mutex
    totalRetries       int64
    successfulRetries  int64
    failedRetries      int64
    totalPanics        int64
    retryDurations     []time.Duration
}

func (rm *RetryMetrics) Record(result *RetryResult) {
    rm.mu.Lock()
    defer rm.mu.Unlock()

    rm.totalRetries++
    rm.totalPanics += int64(result.PanicCount)

    if result.Success {
        rm.successfulRetries++
    } else {
        rm.failedRetries++
    }

    rm.retryDurations = append(rm.retryDurations, result.Duration)
}

func (rm *RetryMetrics) Report() {
    rm.mu.Lock()
    defer rm.mu.Unlock()

    successRate := float64(rm.successfulRetries) / float64(rm.totalRetries) * 100
    panicRate := float64(rm.totalPanics) / float64(rm.totalRetries) * 100

    avgDuration := time.Duration(0)
    if len(rm.retryDurations) > 0 {
        var total time.Duration
        for _, d := range rm.retryDurations {
            total += d
        }
        avgDuration = total / time.Duration(len(rm.retryDurations))
    }

    log.Printf(\`Retry metrikalari:
  Jami retrylar: %d
  Muvaffaqiyat darajasi: %.2f%%
  Muvaffaqiyatsiz retrylar: %d
  Jami paniklar: %d
  Panik darajasi: %.2f%%
  O'rtacha davomiyligi: %v
\`, rm.totalRetries, successRate, rm.failedRetries, rm.totalPanics, panicRate, avgDuration)
}
\`\`\`

**Ogohlantirish uchun asosiy metrikalar:**
- **Panik darajasi >5%:** Asosiy muammoni tekshiring
- **Retry muvaffaqiyat darajasi <50%:** Konfiguratsiya yetarli bo'lmasligi mumkin
- **O'rtacha urinishlar >3:** Servis ortiqcha yuklangan bo'lishi mumkin
- **Muvaffaqiyatsiz retrylar ko'payishi:** Downstream servisning to'liq nosozligi mumkin

**Eng yaxshi amaliyotlar:**

1. **Har doim eksponensial backoff ishlatinig** jitter bilan (odatda 30%)
2. **Max attempts ni sozlang** operatsiya turiga qarab (odatda 3-5)
3. **Retry metrikalarini kuzating** kuzatiluvchanlik uchun (muvaffaqiyat, paniklar, davomiyligi)
4. **Yuqori panik darajasida ogohlantiring** (>5% tekshirish talab qiladi)
5. **Abadiy retry qilmang** - oxir-oqibat muvaffaqiyatsiz bo'ling (gracefully fail)
6. **Jitter qo'shing** thundering herd effektini oldini olish uchun
7. **Panik va xatolarni farqlang** yaxshiroq debugging uchun (turli kod yo'llari)
8. **Retry statistikasini loging qiling** har bir operatsiya uchun (diagnostikaga yordam beradi)
9. **Timeout uchun context ishlatinig** - faqat retry ga tayanmang
10. **Panik stsenariylarini test qiling** - testlarda nosozliklarni simulyatsiya qiling

**Xulosa:**

Panikdan tiklanish bilan retry — bu shunchaki optimallashtirish emas, balki production tizimlari uchun zarurlikdir. To'g'ri amalga oshirilsa, millionlab tejash mumkin va kritik nosozliklarning oldini olish mumkin. Bu mikroservislar arxitekturasida barqarorlik va ishonchlilikni ta'minlashning asosiy komponentidir va har bir production Go tizimida bo'lishi kerak.`
		}
	}
};

export default task;
