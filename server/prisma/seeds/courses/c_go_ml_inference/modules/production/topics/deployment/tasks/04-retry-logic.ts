import { Task } from '../../../../../../../types';

const task: Task = {
  slug: 'goml-retry-logic',
  title: 'Retry Logic for ML Inference',
  difficulty: 'medium',
  tags: ['go', 'ml', 'retry', 'resilience', 'backoff'],
  estimatedTime: '25m',
  isPremium: false,
  order: 4,

  description: `
## Retry Logic for ML Inference

Implement retry logic with exponential backoff for ML inference requests to handle transient failures.

### Requirements

1. **Retryer** - Main retry component:
   - \`NewRetryer(config RetryConfig)\` - Create with configuration
   - \`Do(ctx context.Context, fn func() error) error\` - Execute with retries
   - \`DoWithResult[T](ctx context.Context, fn func() (T, error)) (T, error)\` - Execute with result

2. **RetryConfig** - Configuration options:
   - \`MaxAttempts int\` - Maximum retry attempts
   - \`InitialDelay time.Duration\` - Initial backoff delay
   - \`MaxDelay time.Duration\` - Maximum backoff delay
   - \`Multiplier float64\` - Backoff multiplier (e.g., 2.0 for exponential)
   - \`Jitter float64\` - Random jitter factor (0.0-1.0)
   - \`RetryableErrors []error\` - Errors that should trigger retry

3. **Backoff Strategies**:
   - Exponential: delay = initial * multiplier^attempt
   - With jitter: delay = delay * (1 + random(-jitter, jitter))
   - Capped: min(delay, maxDelay)

4. **Context Support**:
   - Respect context cancellation
   - Stop retrying if context deadline exceeded

### Example

\`\`\`go
retryer := NewRetryer(RetryConfig{
    MaxAttempts:  3,
    InitialDelay: 100 * time.Millisecond,
    MaxDelay:     5 * time.Second,
    Multiplier:   2.0,
    Jitter:       0.1,
})

result, err := retryer.DoWithResult(ctx, func() ([]float32, error) {
    return model.Predict(features)
})
\`\`\`
`,

  initialCode: `package retry

import (
	"context"
	"errors"
	"time"
)

type RetryConfig struct {
	MaxAttempts     int
	InitialDelay    time.Duration
	MaxDelay        time.Duration
	Multiplier      float64
	Jitter          float64
}

type Retryer struct {
}

func NewRetryer(config RetryConfig) *Retryer {
	return nil
}

func (r *Retryer) Do(ctx context.Context, fn func() error) error {
	return nil
}

func DoWithResult[T any](r *Retryer, ctx context.Context, fn func() (T, error)) (T, error) {
	var zero T
	return zero, nil
}

func (r *Retryer) calculateDelay(attempt int) time.Duration {
	return 0
}

func (r *Retryer) isRetryable(err error) bool {
	return false
}

func sleep(ctx context.Context, d time.Duration) error {
	return nil
}`,

  solutionCode: `package retry

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"time"
)

// RetryConfig configures retry behavior
type RetryConfig struct {
	MaxAttempts     int
	InitialDelay    time.Duration
	MaxDelay        time.Duration
	Multiplier      float64
	Jitter          float64
	RetryableErrors []error
}

// Retryer handles retry logic with backoff
type Retryer struct {
	config RetryConfig
}

// NewRetryer creates a new retryer
func NewRetryer(config RetryConfig) *Retryer {
	if config.MaxAttempts <= 0 {
		config.MaxAttempts = 3
	}
	if config.InitialDelay <= 0 {
		config.InitialDelay = 100 * time.Millisecond
	}
	if config.MaxDelay <= 0 {
		config.MaxDelay = 30 * time.Second
	}
	if config.Multiplier <= 0 {
		config.Multiplier = 2.0
	}
	return &Retryer{config: config}
}

// Do executes a function with retry logic
func (r *Retryer) Do(ctx context.Context, fn func() error) error {
	var lastErr error

	for attempt := 0; attempt < r.config.MaxAttempts; attempt++ {
		// Check context before attempt
		if ctx.Err() != nil {
			return ctx.Err()
		}

		err := fn()
		if err == nil {
			return nil
		}

		lastErr = err

		// Check if error is retryable
		if !r.isRetryable(err) {
			return err
		}

		// Don't sleep after last attempt
		if attempt < r.config.MaxAttempts-1 {
			delay := r.calculateDelay(attempt)
			if err := sleep(ctx, delay); err != nil {
				return err
			}
		}
	}

	return lastErr
}

// DoWithResult executes a function that returns a result with retry logic
func DoWithResult[T any](r *Retryer, ctx context.Context, fn func() (T, error)) (T, error) {
	var result T
	var lastErr error

	for attempt := 0; attempt < r.config.MaxAttempts; attempt++ {
		if ctx.Err() != nil {
			return result, ctx.Err()
		}

		res, err := fn()
		if err == nil {
			return res, nil
		}

		lastErr = err

		if !r.isRetryable(err) {
			return result, err
		}

		if attempt < r.config.MaxAttempts-1 {
			delay := r.calculateDelay(attempt)
			if err := sleep(ctx, delay); err != nil {
				return result, err
			}
		}
	}

	return result, lastErr
}

// calculateDelay calculates the backoff delay for an attempt
func (r *Retryer) calculateDelay(attempt int) time.Duration {
	// Exponential backoff
	delay := float64(r.config.InitialDelay) * math.Pow(r.config.Multiplier, float64(attempt))

	// Apply jitter
	if r.config.Jitter > 0 {
		jitterRange := delay * r.config.Jitter
		jitter := (rand.Float64()*2 - 1) * jitterRange
		delay += jitter
	}

	// Cap at max delay
	if delay > float64(r.config.MaxDelay) {
		delay = float64(r.config.MaxDelay)
	}

	// Ensure non-negative
	if delay < 0 {
		delay = 0
	}

	return time.Duration(delay)
}

// isRetryable checks if an error should trigger a retry
func (r *Retryer) isRetryable(err error) bool {
	if err == nil {
		return false
	}

	// If no specific errors defined, retry all
	if len(r.config.RetryableErrors) == 0 {
		return true
	}

	for _, retryableErr := range r.config.RetryableErrors {
		if errors.Is(err, retryableErr) {
			return true
		}
	}

	return false
}

// sleep waits for duration or until context is cancelled
func sleep(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// RetryableError wraps an error as retryable
type RetryableError struct {
	Err error
}

func (e *RetryableError) Error() string {
	return e.Err.Error()
}

func (e *RetryableError) Unwrap() error {
	return e.Err
}

// ErrTemporary is a common retryable error
var ErrTemporary = errors.New("temporary error")

// IsTemporary returns a retryable temporary error
func IsTemporary(err error) error {
	return &RetryableError{Err: err}
}
`,

  testCode: `package retry

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestNewRetryer(t *testing.T) {
	r := NewRetryer(RetryConfig{
		MaxAttempts:  3,
		InitialDelay: 100 * time.Millisecond,
	})

	if r == nil {
		t.Fatal("Expected non-nil retryer")
	}
}

func TestDoSuccess(t *testing.T) {
	r := NewRetryer(RetryConfig{MaxAttempts: 3})

	attempts := 0
	err := r.Do(context.Background(), func() error {
		attempts++
		return nil
	})

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if attempts != 1 {
		t.Errorf("Expected 1 attempt, got %d", attempts)
	}
}

func TestDoRetry(t *testing.T) {
	r := NewRetryer(RetryConfig{
		MaxAttempts:  3,
		InitialDelay: time.Millisecond,
	})

	attempts := 0
	err := r.Do(context.Background(), func() error {
		attempts++
		if attempts < 3 {
			return errors.New("transient error")
		}
		return nil
	})

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if attempts != 3 {
		t.Errorf("Expected 3 attempts, got %d", attempts)
	}
}

func TestDoMaxAttempts(t *testing.T) {
	r := NewRetryer(RetryConfig{
		MaxAttempts:  3,
		InitialDelay: time.Millisecond,
	})

	attempts := 0
	testErr := errors.New("permanent error")
	err := r.Do(context.Background(), func() error {
		attempts++
		return testErr
	})

	if err == nil {
		t.Error("Expected error")
	}
	if attempts != 3 {
		t.Errorf("Expected 3 attempts, got %d", attempts)
	}
}

func TestDoWithResultSuccess(t *testing.T) {
	r := NewRetryer(RetryConfig{MaxAttempts: 3})

	result, err := DoWithResult(r, context.Background(), func() (int, error) {
		return 42, nil
	})

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result != 42 {
		t.Errorf("Expected 42, got %d", result)
	}
}

func TestDoWithResultRetry(t *testing.T) {
	r := NewRetryer(RetryConfig{
		MaxAttempts:  3,
		InitialDelay: time.Millisecond,
	})

	attempts := 0
	result, err := DoWithResult(r, context.Background(), func() (string, error) {
		attempts++
		if attempts < 2 {
			return "", errors.New("transient")
		}
		return "success", nil
	})

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result != "success" {
		t.Errorf("Expected 'success', got '%s'", result)
	}
}

func TestContextCancellation(t *testing.T) {
	r := NewRetryer(RetryConfig{
		MaxAttempts:  10,
		InitialDelay: time.Second,
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	err := r.Do(ctx, func() error {
		return errors.New("error")
	})

	if !errors.Is(err, context.Canceled) {
		t.Errorf("Expected context.Canceled, got %v", err)
	}
}

func TestNonRetryableError(t *testing.T) {
	errPermanent := errors.New("permanent")
	errRetryable := errors.New("retryable")

	r := NewRetryer(RetryConfig{
		MaxAttempts:     3,
		InitialDelay:    time.Millisecond,
		RetryableErrors: []error{errRetryable},
	})

	attempts := 0
	err := r.Do(context.Background(), func() error {
		attempts++
		return errPermanent
	})

	if attempts != 1 {
		t.Errorf("Expected 1 attempt for non-retryable, got %d", attempts)
	}
	if !errors.Is(err, errPermanent) {
		t.Errorf("Expected permanent error, got %v", err)
	}
}

func TestCalculateDelay(t *testing.T) {
	r := NewRetryer(RetryConfig{
		MaxAttempts:  5,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     1 * time.Second,
		Multiplier:   2.0,
		Jitter:       0,
	})

	expected := []time.Duration{
		100 * time.Millisecond,  // 100 * 2^0
		200 * time.Millisecond,  // 100 * 2^1
		400 * time.Millisecond,  // 100 * 2^2
		800 * time.Millisecond,  // 100 * 2^3
		1000 * time.Millisecond, // capped at max
	}

	for i, exp := range expected {
		delay := r.calculateDelay(i)
		if delay != exp {
			t.Errorf("Attempt %d: expected %v, got %v", i, exp, delay)
		}
	}
}

func TestJitter(t *testing.T) {
	r := NewRetryer(RetryConfig{
		InitialDelay: 100 * time.Millisecond,
		Multiplier:   1.0,
		Jitter:       0.5,
		MaxDelay:     time.Second,
	})

	// With 50% jitter, delays should vary
	delays := make(map[time.Duration]bool)
	for i := 0; i < 20; i++ {
		delay := r.calculateDelay(0)
		delays[delay] = true
	}

	// Should have some variation
	if len(delays) < 2 {
		t.Error("Jitter should produce varying delays")
	}
}
`,

  hint1: `Use math.Pow for exponential backoff calculation. For attempt 0, delay should be InitialDelay. For attempt n, delay = InitialDelay * Multiplier^n.`,

  hint2: `Use time.Timer with select and context.Done() for interruptible sleep. This ensures quick response to context cancellation.`,

  whyItMatters: `Retry logic is essential for handling transient failures in ML inference. Network issues, GPU memory pressure, or temporary model unavailability can cause intermittent failures. Exponential backoff with jitter prevents thundering herd problems when multiple clients retry simultaneously.`,

  translations: {
    ru: {
      title: 'Логика Повторных Попыток для ML Инференса',
      description: `
## Логика Повторных Попыток для ML Инференса

Реализуйте логику повторных попыток с экспоненциальной задержкой для ML-инференс запросов для обработки временных сбоев.

### Требования

1. **Retryer** - Основной компонент повторов:
   - \`NewRetryer(config RetryConfig)\` - Создание с конфигурацией
   - \`Do(ctx context.Context, fn func() error) error\` - Выполнение с повторами
   - \`DoWithResult[T](ctx context.Context, fn func() (T, error)) (T, error)\` - Выполнение с результатом

2. **RetryConfig** - Параметры конфигурации:
   - \`MaxAttempts int\` - Максимум попыток
   - \`InitialDelay time.Duration\` - Начальная задержка
   - \`MaxDelay time.Duration\` - Максимальная задержка
   - \`Multiplier float64\` - Множитель задержки
   - \`Jitter float64\` - Фактор случайности (0.0-1.0)
   - \`RetryableErrors []error\` - Ошибки для повтора

3. **Стратегии задержки**:
   - Экспоненциальная: delay = initial * multiplier^attempt
   - С jitter: delay = delay * (1 + random(-jitter, jitter))
   - Ограниченная: min(delay, maxDelay)

4. **Поддержка контекста**:
   - Уважение отмены контекста
   - Прекращение повторов при истечении deadline

### Пример

\`\`\`go
retryer := NewRetryer(RetryConfig{
    MaxAttempts:  3,
    InitialDelay: 100 * time.Millisecond,
    MaxDelay:     5 * time.Second,
    Multiplier:   2.0,
    Jitter:       0.1,
})

result, err := retryer.DoWithResult(ctx, func() ([]float32, error) {
    return model.Predict(features)
})
\`\`\`
`,
      hint1: 'Используйте math.Pow для вычисления экспоненциальной задержки. Для попытки 0 задержка равна InitialDelay. Для попытки n: delay = InitialDelay * Multiplier^n.',
      hint2: 'Используйте time.Timer с select и context.Done() для прерываемого ожидания. Это обеспечивает быстрый ответ на отмену контекста.',
      whyItMatters: 'Логика повторов необходима для обработки временных сбоев в ML-инференсе. Сетевые проблемы, нехватка памяти GPU или временная недоступность модели могут вызвать кратковременные сбои. Экспоненциальная задержка с jitter предотвращает проблему thundering herd, когда множество клиентов повторяют запросы одновременно.',
    },
    uz: {
      title: 'ML Inference uchun Retry Logic',
      description: `
## ML Inference uchun Retry Logic

Vaqtinchalik nosozliklarni boshqarish uchun eksponensial backoff bilan ML inference so'rovlari uchun retry logikasini amalga oshiring.

### Talablar

1. **Retryer** - Asosiy retry komponenti:
   - \`NewRetryer(config RetryConfig)\` - Konfiguratsiya bilan yaratish
   - \`Do(ctx context.Context, fn func() error) error\` - Retrylar bilan bajarish
   - \`DoWithResult[T](ctx context.Context, fn func() (T, error)) (T, error)\` - Natija bilan bajarish

2. **RetryConfig** - Konfiguratsiya parametrlari:
   - \`MaxAttempts int\` - Maksimal urinishlar
   - \`InitialDelay time.Duration\` - Boshlang'ich kechikish
   - \`MaxDelay time.Duration\` - Maksimal kechikish
   - \`Multiplier float64\` - Backoff ko'paytiruvchi
   - \`Jitter float64\` - Tasodifiy jitter omili (0.0-1.0)
   - \`RetryableErrors []error\` - Retryni ishga tushiruvchi xatolar

3. **Backoff strategiyalari**:
   - Eksponensial: delay = initial * multiplier^attempt
   - Jitter bilan: delay = delay * (1 + random(-jitter, jitter))
   - Chegaralangan: min(delay, maxDelay)

4. **Context qo'llab-quvvatlash**:
   - Context bekor qilishni hurmat qilish
   - Context deadline tugaganda retrylarni to'xtatish

### Misol

\`\`\`go
retryer := NewRetryer(RetryConfig{
    MaxAttempts:  3,
    InitialDelay: 100 * time.Millisecond,
    MaxDelay:     5 * time.Second,
    Multiplier:   2.0,
    Jitter:       0.1,
})

result, err := retryer.DoWithResult(ctx, func() ([]float32, error) {
    return model.Predict(features)
})
\`\`\`
`,
      hint1: "Eksponensial backoff hisoblash uchun math.Pow dan foydalaning. 0-urinish uchun kechikish InitialDelay ga teng. n-urinish uchun: delay = InitialDelay * Multiplier^n.",
      hint2: "To'xtatilishi mumkin bo'lgan kutish uchun select va context.Done() bilan time.Timer dan foydalaning. Bu context bekor qilishga tez javob berishni ta'minlaydi.",
      whyItMatters: "Retry logic ML inferencedagi vaqtinchalik nosozliklarni boshqarish uchun muhim. Tarmoq muammolari, GPU xotira bosimi yoki vaqtinchalik model mavjud emasligi vaqti-vaqti bilan nosozliklarga sabab bo'lishi mumkin. Jitter bilan eksponensial backoff bir vaqtda ko'p mijozlar retry qilganda thundering herd muammosini oldini oladi.",
    },
  },
};

export default task;
