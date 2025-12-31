import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-circuit-breaker',
	title: 'Circuit Breaker',
	difficulty: 'medium',
	tags: ['go', 'ml', 'resilience', 'circuit-breaker'],
	estimatedTime: '30m',
	isPremium: true,
	order: 3,
	description: `# Circuit Breaker

Implement circuit breaker pattern for ML inference.

## Task

Build a circuit breaker that:
- Opens after consecutive failures
- Enters half-open state for testing
- Closes when requests succeed
- Provides fallback responses

## Example

\`\`\`go
cb := NewCircuitBreaker(threshold: 5, timeout: 30*time.Second)
result, err := cb.Execute(func() ([]float32, error) {
    return model.Predict(input)
})
\`\`\``,

	initialCode: `package main

import (
	"fmt"
	"time"
)

// CircuitState represents circuit breaker state
type CircuitState int

const (
	StateClosed CircuitState = iota
	StateOpen
	StateHalfOpen
)

// CircuitBreaker implements the circuit breaker pattern
type CircuitBreaker struct {
	// Your fields here
}

// NewCircuitBreaker creates a circuit breaker
func NewCircuitBreaker(threshold int, timeout time.Duration) *CircuitBreaker {
	// Your code here
	return nil
}

// Execute runs function with circuit breaker protection
func (cb *CircuitBreaker) Execute(fn func() ([]float32, error)) ([]float32, error) {
	// Your code here
	return nil, nil
}

// State returns current circuit state
func (cb *CircuitBreaker) State() CircuitState {
	// Your code here
	return StateClosed
}

func main() {
	fmt.Println("Circuit Breaker")
}`,

	solutionCode: `package main

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// CircuitState represents circuit breaker state
type CircuitState int32

const (
	StateClosed CircuitState = iota
	StateOpen
	StateHalfOpen
)

func (s CircuitState) String() string {
	switch s {
	case StateClosed:
		return "CLOSED"
	case StateOpen:
		return "OPEN"
	case StateHalfOpen:
		return "HALF-OPEN"
	default:
		return "UNKNOWN"
	}
}

// CircuitBreakerConfig holds configuration
type CircuitBreakerConfig struct {
	FailureThreshold    int
	SuccessThreshold    int
	Timeout             time.Duration
	HalfOpenMaxRequests int
}

// DefaultCircuitBreakerConfig returns sensible defaults
func DefaultCircuitBreakerConfig() CircuitBreakerConfig {
	return CircuitBreakerConfig{
		FailureThreshold:    5,
		SuccessThreshold:    3,
		Timeout:             30 * time.Second,
		HalfOpenMaxRequests: 3,
	}
}

// CircuitBreaker implements the circuit breaker pattern
type CircuitBreaker struct {
	config CircuitBreakerConfig

	state            int32
	failures         int64
	successes        int64
	halfOpenRequests int64
	lastStateChange  time.Time

	fallback func() ([]float32, error)
	mu       sync.RWMutex

	// Metrics
	totalRequests   int64
	totalSuccesses  int64
	totalFailures   int64
	totalRejected   int64
}

// NewCircuitBreaker creates a circuit breaker
func NewCircuitBreaker(threshold int, timeout time.Duration) *CircuitBreaker {
	config := DefaultCircuitBreakerConfig()
	config.FailureThreshold = threshold
	config.Timeout = timeout

	return &CircuitBreaker{
		config:          config,
		lastStateChange: time.Now(),
	}
}

// SetConfig updates configuration
func (cb *CircuitBreaker) SetConfig(config CircuitBreakerConfig) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.config = config
}

// SetFallback sets the fallback function
func (cb *CircuitBreaker) SetFallback(fn func() ([]float32, error)) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.fallback = fn
}

// Execute runs function with circuit breaker protection
func (cb *CircuitBreaker) Execute(fn func() ([]float32, error)) ([]float32, error) {
	atomic.AddInt64(&cb.totalRequests, 1)

	state := cb.State()

	switch state {
	case StateOpen:
		// Check if timeout has passed
		if cb.shouldAttemptReset() {
			cb.transitionTo(StateHalfOpen)
		} else {
			atomic.AddInt64(&cb.totalRejected, 1)
			return cb.doFallback()
		}

	case StateHalfOpen:
		// Limit concurrent requests in half-open
		current := atomic.AddInt64(&cb.halfOpenRequests, 1)
		defer atomic.AddInt64(&cb.halfOpenRequests, -1)

		cb.mu.RLock()
		max := cb.config.HalfOpenMaxRequests
		cb.mu.RUnlock()

		if int(current) > max {
			atomic.AddInt64(&cb.totalRejected, 1)
			return cb.doFallback()
		}
	}

	// Execute the function
	result, err := fn()

	if err != nil {
		cb.recordFailure()
		return result, err
	}

	cb.recordSuccess()
	return result, nil
}

// State returns current circuit state
func (cb *CircuitBreaker) State() CircuitState {
	return CircuitState(atomic.LoadInt32(&cb.state))
}

// shouldAttemptReset checks if we should try to reset
func (cb *CircuitBreaker) shouldAttemptReset() bool {
	cb.mu.RLock()
	timeout := cb.config.Timeout
	lastChange := cb.lastStateChange
	cb.mu.RUnlock()

	return time.Since(lastChange) >= timeout
}

// recordSuccess records a successful request
func (cb *CircuitBreaker) recordSuccess() {
	atomic.AddInt64(&cb.totalSuccesses, 1)

	state := cb.State()

	if state == StateClosed {
		// Reset failure count on success
		atomic.StoreInt64(&cb.failures, 0)
		return
	}

	if state == StateHalfOpen {
		successes := atomic.AddInt64(&cb.successes, 1)

		cb.mu.RLock()
		threshold := cb.config.SuccessThreshold
		cb.mu.RUnlock()

		if int(successes) >= threshold {
			cb.transitionTo(StateClosed)
		}
	}
}

// recordFailure records a failed request
func (cb *CircuitBreaker) recordFailure() {
	atomic.AddInt64(&cb.totalFailures, 1)

	state := cb.State()

	if state == StateHalfOpen {
		cb.transitionTo(StateOpen)
		return
	}

	if state == StateClosed {
		failures := atomic.AddInt64(&cb.failures, 1)

		cb.mu.RLock()
		threshold := cb.config.FailureThreshold
		cb.mu.RUnlock()

		if int(failures) >= threshold {
			cb.transitionTo(StateOpen)
		}
	}
}

// transitionTo transitions to a new state
func (cb *CircuitBreaker) transitionTo(newState CircuitState) {
	oldState := cb.State()
	if oldState == newState {
		return
	}

	cb.mu.Lock()
	atomic.StoreInt32(&cb.state, int32(newState))
	cb.lastStateChange = time.Now()
	atomic.StoreInt64(&cb.failures, 0)
	atomic.StoreInt64(&cb.successes, 0)
	cb.mu.Unlock()

	fmt.Printf("Circuit breaker: %s -> %s\\n", oldState, newState)
}

// doFallback executes the fallback function
func (cb *CircuitBreaker) doFallback() ([]float32, error) {
	cb.mu.RLock()
	fallback := cb.fallback
	cb.mu.RUnlock()

	if fallback != nil {
		return fallback()
	}

	return nil, errors.New("circuit breaker open")
}

// Reset manually resets the circuit breaker
func (cb *CircuitBreaker) Reset() {
	cb.transitionTo(StateClosed)
}

// Stats returns circuit breaker statistics
type CircuitBreakerStats struct {
	State           CircuitState
	TotalRequests   int64
	TotalSuccesses  int64
	TotalFailures   int64
	TotalRejected   int64
	SuccessRate     float64
}

func (cb *CircuitBreaker) Stats() CircuitBreakerStats {
	stats := CircuitBreakerStats{
		State:          cb.State(),
		TotalRequests:  atomic.LoadInt64(&cb.totalRequests),
		TotalSuccesses: atomic.LoadInt64(&cb.totalSuccesses),
		TotalFailures:  atomic.LoadInt64(&cb.totalFailures),
		TotalRejected:  atomic.LoadInt64(&cb.totalRejected),
	}

	if stats.TotalRequests > 0 {
		stats.SuccessRate = float64(stats.TotalSuccesses) / float64(stats.TotalRequests)
	}

	return stats
}

// InferenceCircuitBreaker wraps model inference
type InferenceCircuitBreaker struct {
	cb    *CircuitBreaker
	model interface {
		Predict([]float32) []float32
	}
}

func NewInferenceCircuitBreaker(model interface{ Predict([]float32) []float32 }, config CircuitBreakerConfig) *InferenceCircuitBreaker {
	cb := NewCircuitBreaker(config.FailureThreshold, config.Timeout)
	cb.SetConfig(config)

	return &InferenceCircuitBreaker{
		cb:    cb,
		model: model,
	}
}

func (icb *InferenceCircuitBreaker) Predict(input []float32) ([]float32, error) {
	return icb.cb.Execute(func() ([]float32, error) {
		result := icb.model.Predict(input)
		if result == nil {
			return nil, errors.New("prediction failed")
		}
		return result, nil
	})
}

func main() {
	cb := NewCircuitBreaker(3, 5*time.Second)

	// Set fallback
	cb.SetFallback(func() ([]float32, error) {
		return []float32{0.5, 0.5}, nil // Default prediction
	})

	// Simulate requests
	for i := 0; i < 10; i++ {
		result, err := cb.Execute(func() ([]float32, error) {
			if i < 5 {
				return nil, errors.New("service unavailable")
			}
			return []float32{0.8, 0.2}, nil
		})

		fmt.Printf("Request %d: result=%v, err=%v, state=%s\\n",
			i, result, err, cb.State())

		time.Sleep(100 * time.Millisecond)
	}

	fmt.Printf("\\nStats: %+v\\n", cb.Stats())
}`,

	testCode: `package main

import (
	"errors"
	"testing"
	"time"
)

func TestCircuitBreakerClosed(t *testing.T) {
	cb := NewCircuitBreaker(3, time.Second)

	result, err := cb.Execute(func() ([]float32, error) {
		return []float32{0.8, 0.2}, nil
	})

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result")
	}
	if cb.State() != StateClosed {
		t.Error("Should remain closed on success")
	}
}

func TestCircuitBreakerOpens(t *testing.T) {
	cb := NewCircuitBreaker(3, time.Second)

	// Fail 3 times
	for i := 0; i < 3; i++ {
		cb.Execute(func() ([]float32, error) {
			return nil, errors.New("failure")
		})
	}

	if cb.State() != StateOpen {
		t.Error("Should be open after failures")
	}
}

func TestCircuitBreakerRejectsWhenOpen(t *testing.T) {
	cb := NewCircuitBreaker(1, time.Second)

	// Fail to open
	cb.Execute(func() ([]float32, error) {
		return nil, errors.New("failure")
	})

	// Should be rejected
	_, err := cb.Execute(func() ([]float32, error) {
		return []float32{0.8, 0.2}, nil
	})

	if err == nil {
		t.Error("Should reject when open")
	}

	stats := cb.Stats()
	if stats.TotalRejected != 1 {
		t.Errorf("Expected 1 rejected, got %d", stats.TotalRejected)
	}
}

func TestCircuitBreakerFallback(t *testing.T) {
	cb := NewCircuitBreaker(1, time.Second)
	cb.SetFallback(func() ([]float32, error) {
		return []float32{0.5, 0.5}, nil
	})

	// Fail to open
	cb.Execute(func() ([]float32, error) {
		return nil, errors.New("failure")
	})

	// Should use fallback
	result, err := cb.Execute(func() ([]float32, error) {
		return []float32{0.8, 0.2}, nil
	})

	if err != nil {
		t.Error("Fallback should not error")
	}
	if result[0] != 0.5 {
		t.Error("Should use fallback result")
	}
}

func TestCircuitBreakerHalfOpen(t *testing.T) {
	cb := NewCircuitBreaker(1, 10*time.Millisecond)

	// Fail to open
	cb.Execute(func() ([]float32, error) {
		return nil, errors.New("failure")
	})

	// Wait for timeout
	time.Sleep(20 * time.Millisecond)

	// Should transition to half-open
	cb.Execute(func() ([]float32, error) {
		return []float32{0.8, 0.2}, nil
	})

	state := cb.State()
	if state != StateHalfOpen && state != StateClosed {
		t.Errorf("Should be half-open or closed, got %s", state)
	}
}

func TestCircuitBreakerReset(t *testing.T) {
	cb := NewCircuitBreaker(1, time.Second)

	// Fail to open
	cb.Execute(func() ([]float32, error) {
		return nil, errors.New("failure")
	})

	if cb.State() != StateOpen {
		t.Fatal("Should be open")
	}

	cb.Reset()

	if cb.State() != StateClosed {
		t.Error("Should be closed after reset")
	}
}

func TestCircuitBreakerStats(t *testing.T) {
	cb := NewCircuitBreaker(5, time.Second)

	// Some successes
	for i := 0; i < 3; i++ {
		cb.Execute(func() ([]float32, error) {
			return []float32{0.8, 0.2}, nil
		})
	}

	// Some failures
	for i := 0; i < 2; i++ {
		cb.Execute(func() ([]float32, error) {
			return nil, errors.New("failure")
		})
	}

	stats := cb.Stats()

	if stats.TotalRequests != 5 {
		t.Errorf("Expected 5 requests, got %d", stats.TotalRequests)
	}
	if stats.TotalSuccesses != 3 {
		t.Errorf("Expected 3 successes, got %d", stats.TotalSuccesses)
	}
	if stats.TotalFailures != 2 {
		t.Errorf("Expected 2 failures, got %d", stats.TotalFailures)
	}
}

func TestCircuitStateString(t *testing.T) {
	if StateClosed.String() != "CLOSED" {
		t.Errorf("Expected CLOSED, got %s", StateClosed.String())
	}
	if StateOpen.String() != "OPEN" {
		t.Errorf("Expected OPEN, got %s", StateOpen.String())
	}
	if StateHalfOpen.String() != "HALF-OPEN" {
		t.Errorf("Expected HALF-OPEN, got %s", StateHalfOpen.String())
	}
}

func TestSetConfig(t *testing.T) {
	cb := NewCircuitBreaker(5, time.Second)

	config := CircuitBreakerConfig{
		FailureThreshold:    2,
		SuccessThreshold:    1,
		Timeout:             500 * time.Millisecond,
		HalfOpenMaxRequests: 1,
	}
	cb.SetConfig(config)

	// Fail twice to open (new threshold is 2)
	cb.Execute(func() ([]float32, error) {
		return nil, errors.New("fail")
	})
	cb.Execute(func() ([]float32, error) {
		return nil, errors.New("fail")
	})

	if cb.State() != StateOpen {
		t.Error("Should be open after 2 failures with new config")
	}
}

type mockModel struct{}

func (m *mockModel) Predict(input []float32) []float32 {
	return []float32{0.9, 0.1}
}

func TestInferenceCircuitBreaker(t *testing.T) {
	model := &mockModel{}
	config := DefaultCircuitBreakerConfig()

	icb := NewInferenceCircuitBreaker(model, config)

	result, err := icb.Predict([]float32{1, 2, 3})
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	if result[0] != 0.9 {
		t.Errorf("Expected 0.9, got %f", result[0])
	}
}`,

	hint1: 'Use atomic operations for thread-safe state transitions',
	hint2: 'Implement half-open state to allow gradual recovery',

	whyItMatters: `Circuit breakers prevent cascade failures:

- **Fail fast**: Stop sending requests to failing services
- **Resource protection**: Prevent resource exhaustion
- **Graceful degradation**: Provide fallback responses
- **Auto-recovery**: Gradually restore service after recovery

Circuit breakers are essential for resilient ML systems.`,

	translations: {
		ru: {
			title: 'Circuit Breaker',
			description: `# Circuit Breaker

Реализуйте паттерн circuit breaker для ML инференса.

## Задача

Создайте circuit breaker:
- Открытие после последовательных сбоев
- Переход в half-open состояние для тестирования
- Закрытие при успешных запросах
- Предоставление fallback ответов

## Пример

\`\`\`go
cb := NewCircuitBreaker(threshold: 5, timeout: 30*time.Second)
result, err := cb.Execute(func() ([]float32, error) {
    return model.Predict(input)
})
\`\`\``,
			hint1: 'Используйте атомарные операции для потокобезопасных переходов состояний',
			hint2: 'Реализуйте half-open состояние для постепенного восстановления',
			whyItMatters: `Circuit breaker предотвращает каскадные сбои:

- **Быстрый отказ**: Прекращение отправки запросов к сбойным сервисам
- **Защита ресурсов**: Предотвращение исчерпания ресурсов
- **Graceful degradation**: Предоставление fallback ответов
- **Авто-восстановление**: Постепенное восстановление сервиса`,
		},
		uz: {
			title: 'Circuit Breaker',
			description: `# Circuit Breaker

ML inference uchun circuit breaker patternini amalga oshiring.

## Topshiriq

Circuit breaker yarating:
- Ketma-ket nosozliklardan keyin ochish
- Test qilish uchun half-open holatga o'tish
- Muvaffaqiyatli so'rovlarda yopish
- Fallback javoblarini taqdim etish

## Misol

\`\`\`go
cb := NewCircuitBreaker(threshold: 5, timeout: 30*time.Second)
result, err := cb.Execute(func() ([]float32, error) {
    return model.Predict(input)
})
\`\`\``,
			hint1: "Thread-safe holat o'tishlari uchun atomik operatsiyalardan foydalaning",
			hint2: "Asta-sekin tiklash uchun half-open holatni amalga oshiring",
			whyItMatters: `Circuit breaker kaskad nosozliklarni oldini oladi:

- **Tez muvaffaqiyatsizlik**: Nosoz xizmatlarga so'rovlar yuborishni to'xtatish
- **Resurs himoyasi**: Resurslarning tugashini oldini olish
- **Yumshoq degradatsiya**: Fallback javoblarini taqdim etish
- **Avto-tiklash**: Tiklanishdan keyin xizmatni asta-sekin tiklash`,
		},
	},
};

export default task;
