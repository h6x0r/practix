import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-retry-until',
	title: 'Conditional Retry with Predicate',
	difficulty: 'medium',	tags: ['go', 'retry', 'conditional', 'predicate'],
	estimatedTime: '35m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **RetryUntil** function that retries an operation only when a predicate function returns true.

**Requirements:**
1. Retry operation up to \`n\` times
2. Call \`shouldRetry(err)\` after each failure
3. Stop retrying if \`shouldRetry(err)\` returns false
4. Apply exponential backoff between retries
5. Respect context cancellation
6. Return immediately on success or when predicate rejects error

**Function Signature:**
\`\`\`go
func RetryUntil(
    ctx context.Context,
    n int,
    base time.Duration,
    op Op,
    shouldRetry func(error) bool,
) error
\`\`\`

**Retry Logic:**
\`\`\`
Attempt 1 → NetworkError → shouldRetry(err)=true → wait & retry
Attempt 2 → AuthError → shouldRetry(err)=false → return immediately
\`\`\`

**Example:**
\`\`\`go
// Retry only on temporary network errors
isRetryable := func(err error) bool {
    var netErr *net.OpError
    return errors.As(err, &netErr) && netErr.Temporary()
}

err := RetryUntil(ctx, 5, 100*time.Millisecond, fetchData, isRetryable)
// Retries network errors, stops on auth/validation errors
\`\`\`

**Use Cases:**
\`\`\`go
// 1. Retry only 5xx server errors, not 4xx client errors
shouldRetry := func(err error) bool {
    var httpErr *HTTPError
    if errors.As(err, &httpErr) {
        return httpErr.Code >= 500
    }
    return false
}

// 2. Retry only temporary database errors
shouldRetry := func(err error) bool {
    return err == sql.ErrConnDone || err == driver.ErrBadConn
}

// 3. Retry only rate limit errors
shouldRetry := func(err error) bool {
    return errors.Is(err, ErrRateLimited)
}
\`\`\`

**Edge Cases:**
- \`shouldRetry\` is nil → default to always retry
- Predicate returns false on first error → return immediately
- Context canceled → return ctx.Err()

**Constraints:**
- Check predicate before sleeping
- Don't retry non-retryable errors
- Handle nil predicate gracefully (default: always retry)`,
	initialCode: `package retryx

import (
	"context"
	"time"
)

// TODO: Implement RetryUntil function
// Retry operation while shouldRetry returns true
func RetryUntil(
	ctx context.Context,
	n int,
	base time.Duration,
	op Op,
	shouldRetry func(error) bool,
) error {
	// TODO: Implement
}`,
	solutionCode: `package retryx

import (
	"context"
	"time"
)

func RetryUntil(
	ctx context.Context,
	n int,
	base time.Duration,
	op Op,
	shouldRetry func(error) bool,
) error {
	if shouldRetry == nil {	// Default predicate: always retry
		shouldRetry = func(error) bool { return true }
	}
	var lastErr error	// Track last error
	for attempt := 0; attempt < n; attempt++ {	// Up to n attempts
		if ctx.Err() != nil {	// Context canceled
			return ctx.Err()
		}
		if err := op(ctx); err != nil {	// Operation failed
			lastErr = err	// Remember error
			if !shouldRetry(err) {	// Predicate rejects error
				return err	// Stop retrying, return immediately
			}
			if err := SleepContext(ctx, Backoff(attempt, base)); err != nil {	// Wait before retry
				return err	// Context canceled during sleep
			}
			continue	// Try again
		}
		return nil	// Success
	}
	return lastErr	// All retries exhausted
}`,
	testCode: `package retryx

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestRetryUntilSuccessOnFirstAttempt(t *testing.T) {
	ctx := context.Background()
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		return nil
	}
	shouldRetry := func(err error) bool { return true }
	err := RetryUntil(ctx, 5, 100*time.Millisecond, op, shouldRetry)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if callCount != 1 {
		t.Errorf("expected 1 call, got %d", callCount)
	}
}

func TestRetryUntilNonRetryableError(t *testing.T) {
	ctx := context.Background()
	nonRetryableErr := errors.New("non-retryable")
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		return nonRetryableErr
	}
	shouldRetry := func(err error) bool { return false }
	err := RetryUntil(ctx, 5, 100*time.Millisecond, op, shouldRetry)
	if err != nonRetryableErr {
		t.Errorf("expected %v, got %v", nonRetryableErr, err)
	}
	if callCount != 1 {
		t.Errorf("expected 1 call (no retries), got %d", callCount)
	}
}

func TestRetryUntilRetryableError(t *testing.T) {
	ctx := context.Background()
	retryableErr := errors.New("retryable")
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		if callCount < 3 {
			return retryableErr
		}
		return nil
	}
	shouldRetry := func(err error) bool { return err == retryableErr }
	err := RetryUntil(ctx, 5, 10*time.Millisecond, op, shouldRetry)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if callCount != 3 {
		t.Errorf("expected 3 calls, got %d", callCount)
	}
}

func TestRetryUntilNilPredicate(t *testing.T) {
	ctx := context.Background()
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		if callCount < 3 {
			return errors.New("temporary")
		}
		return nil
	}
	err := RetryUntil(ctx, 5, 10*time.Millisecond, op, nil)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if callCount != 3 {
		t.Errorf("expected 3 calls, got %d", callCount)
	}
}

func TestRetryUntilAllRetriesFail(t *testing.T) {
	ctx := context.Background()
	expectedErr := errors.New("persistent")
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		return expectedErr
	}
	shouldRetry := func(err error) bool { return true }
	err := RetryUntil(ctx, 3, 10*time.Millisecond, op, shouldRetry)
	if err != expectedErr {
		t.Errorf("expected %v, got %v", expectedErr, err)
	}
	if callCount != 3 {
		t.Errorf("expected 3 calls, got %d", callCount)
	}
}

func TestRetryUntilContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	op := func(ctx context.Context) error {
		return errors.New("error")
	}
	shouldRetry := func(err error) bool { return true }
	err := RetryUntil(ctx, 5, 100*time.Millisecond, op, shouldRetry)
	if err != context.Canceled {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestRetryUntilPredicateChangesDecision(t *testing.T) {
	ctx := context.Background()
	retryableErr := errors.New("retryable")
	nonRetryableErr := errors.New("non-retryable")
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		if callCount == 1 {
			return retryableErr
		}
		return nonRetryableErr
	}
	shouldRetry := func(err error) bool {
		return err == retryableErr
	}
	err := RetryUntil(ctx, 5, 10*time.Millisecond, op, shouldRetry)
	if err != nonRetryableErr {
		t.Errorf("expected %v, got %v", nonRetryableErr, err)
	}
	if callCount != 2 {
		t.Errorf("expected 2 calls, got %d", callCount)
	}
}

func TestRetryUntilWithZeroAttempts(t *testing.T) {
	ctx := context.Background()
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		return errors.New("should not be called")
	}
	shouldRetry := func(err error) bool { return true }
	err := RetryUntil(ctx, 0, 100*time.Millisecond, op, shouldRetry)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if callCount != 0 {
		t.Errorf("expected 0 calls, got %d", callCount)
	}
}

func TestRetryUntilPredicateReturnsFalseImmediately(t *testing.T) {
	ctx := context.Background()
	start := time.Now()
	op := func(ctx context.Context) error {
		return errors.New("fail")
	}
	shouldRetry := func(err error) bool { return false }
	RetryUntil(ctx, 5, 100*time.Millisecond, op, shouldRetry)
	elapsed := time.Since(start)
	// Should return immediately without sleeping
	if elapsed > 50*time.Millisecond {
		t.Errorf("expected immediate return, got %v delay", elapsed)
	}
}

func TestRetryUntilContextCanceledDuringRetry(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		return errors.New("temporary")
	}
	shouldRetry := func(err error) bool { return true }
	err := RetryUntil(ctx, 10, 100*time.Millisecond, op, shouldRetry)
	if err == nil {
		t.Errorf("expected error, got nil")
	}
	if callCount > 3 {
		t.Errorf("expected fewer calls due to timeout, got %d", callCount)
	}
}`,
			hint1: `Call shouldRetry(err) after each failure to decide whether to continue.`,
			hint2: `Return immediately when shouldRetry returns false, don't wait.`,
			whyItMatters: `Conditional retry logic prevents wasting time retrying unrecoverable errors.

**The Problem with Blind Retries:**
\`\`\`go
// Bad: Retries everything, even permanent failures
func fetchUser(id int) error {
    for i := 0; i < 3; i++ {
        resp, err := http.Get(fmt.Sprintf("/users/%d", id))
        if err == nil {
            return nil
        }
        time.Sleep(time.Second)  // Wasteful!
    }
    return err
}

// If user doesn't exist (404), this wastes 2 seconds
// retrying an error that will never succeed!
\`\`\`

**Smart Conditional Retry:**
\`\`\`go
isRetryable := func(err error) bool {
    var httpErr *HTTPError
    if errors.As(err, &httpErr) {
        // Retry server errors (500-599)
        // Don't retry client errors (400-499)
        return httpErr.Code >= 500
    }
    return true  // Retry network errors
}

err := RetryUntil(ctx, 3, 100*time.Millisecond, fetchUser, isRetryable)
// Returns immediately on 404, retries on 503
\`\`\`

**Error Classification:**

**Retryable Errors (Temporary):**
- 500 Internal Server Error → Server bug, might recover
- 502 Bad Gateway → Upstream service down, retry
- 503 Service Unavailable → Server overloaded, retry with backoff
- 504 Gateway Timeout → Slow response, retry
- Connection refused → Service restarting, retry
- Timeout → Network congestion, retry
- Rate limit exceeded → Wait and retry

**Non-Retryable Errors (Permanent):**
- 400 Bad Request → Invalid input, won't change on retry
- 401 Unauthorized → Invalid credentials
- 403 Forbidden → No permission
- 404 Not Found → Resource doesn't exist
- 422 Unprocessable Entity → Validation failed
- Context canceled → User canceled request
- Invalid argument → Programming error

**Production Example (AWS SDK):**
\`\`\`go
// AWS SDK's retry logic
isRetryable := func(err error) bool {
    // Retry throttling errors with exponential backoff
    if errors.Is(err, ErrThrottling) {
        return true
    }
    // Retry transient network errors
    if errors.Is(err, ErrNetworkError) {
        return true
    }
    // Don't retry validation errors
    if errors.Is(err, ErrValidation) {
        return false
    }
    return false
}
\`\`\`

**Real-World Impact:**
\`\`\`
Scenario: API call fails with 401 Unauthorized

Without predicate:
  Retry 1: 401 + 100ms wait = WASTED
  Retry 2: 401 + 200ms wait = WASTED
  Retry 3: 401 + 400ms wait = WASTED
  Total: 700ms wasted, user sees slow error

With predicate:
  Attempt 1: 401 → shouldRetry=false → return immediately
  Total: <10ms, instant feedback to user
\`\`\`

**Cost Savings:**
- **Bandwidth:** Don't retry requests doomed to fail
- **Latency:** Fail fast on permanent errors
- **Resources:** Free up connections/goroutines faster
- **User Experience:** Show meaningful errors quickly

**Debugging Benefit:**
The predicate function is a central place to log retry decisions:
\`\`\`go
shouldRetry := func(err error) bool {
    retry := isTransientError(err)
    log.Printf("Error: %v, Retrying: %v", err, retry)
    return retry
}
\``,	order: 2,
	translations: {
		ru: {
			title: 'Повтор до успеха',
			solutionCode: `package retryx

import (
	"context"
	"time"
)

func RetryUntil(
	ctx context.Context,
	n int,
	base time.Duration,
	op Op,
	shouldRetry func(error) bool,
) error {
	if shouldRetry == nil {	// Предикат по умолчанию: всегда retry
		shouldRetry = func(error) bool { return true }
	}
	var lastErr error	// Отслеживаем последнюю ошибку
	for attempt := 0; attempt < n; attempt++ {	// До n попыток
		if ctx.Err() != nil {	// Контекст отменён
			return ctx.Err()
		}
		if err := op(ctx); err != nil {	// Операция упала
			lastErr = err	// Запоминаем ошибку
			if !shouldRetry(err) {	// Предикат отклонил ошибку
				return err	// Прекращаем retry, возвращаем сразу
			}
			if err := SleepContext(ctx, Backoff(attempt, base)); err != nil {	// Ждём перед retry
				return err	// Контекст отменён во время сна
			}
			continue	// Пробуем ещё раз
		}
		return nil	// Успех
	}
	return lastErr	// Все retry исчерпаны
}`,
			description: `Реализуйте функцию **RetryUntil**, которая повторяет операцию только когда функция-предикат возвращает true.

**Требования:**
1. Повторите операцию до \`n\` раз
2. Вызовите \`shouldRetry(err)\` после каждой ошибки
3. Остановите retry если \`shouldRetry(err)\` вернул false
4. Примените экспоненциальную задержку между retry
5. Учитывайте отмену контекста
6. Верните результат сразу при успехе или когда предикат отклонил ошибку

**Сигнатура функции:**
\`\`\`go
func RetryUntil(
    ctx context.Context,
    n int,
    base time.Duration,
    op Op,
    shouldRetry func(error) bool,
) error
\`\`\`

**Логика Retry:**
\`\`\`
Попытка 1 → NetworkError → shouldRetry(err)=true → ждем & retry
Попытка 2 → AuthError → shouldRetry(err)=false → возврат сразу
\`\`\`

**Пример:**
\`\`\`go
// Retry только временных сетевых ошибок
isRetryable := func(err error) bool {
    var netErr *net.OpError
    return errors.As(err, &netErr) && netErr.Temporary()
}

err := RetryUntil(ctx, 5, 100*time.Millisecond, fetchData, isRetryable)
// Повторяет сетевые ошибки, останавливается на auth/validation ошибках
\`\`\`

**Use Cases:**
\`\`\`go
// 1. Retry только 5xx server errors, не 4xx client errors
shouldRetry := func(err error) bool {
    var httpErr *HTTPError
    if errors.As(err, &httpErr) {
        return httpErr.Code >= 500
    }
    return false
}

// 2. Retry только временные database errors
shouldRetry := func(err error) bool {
    return err == sql.ErrConnDone || err == driver.ErrBadConn
}
`,
			hint1: `Вызовите shouldRetry(err) после каждой ошибки чтобы решить продолжать ли.`,
			hint2: `Возвращайте сразу когда shouldRetry вернул false, не ждите.`,
			whyItMatters: `Условная retry логика предотвращает трату времени на повтор неисправимых ошибок.

**Проблема слепых Retry:**
\`\`\`go
// Плохо: Retry всего, даже постоянных сбоев
func fetchUser(id int) error {
    for i := 0; i < 3; i++ {
        resp, err := http.Get(fmt.Sprintf("/users/%d", id))
        if err == nil {
            return nil
        }
        time.Sleep(time.Second)  // Расточительно!
    }
    return err
}

// Если пользователь не существует (404), тратит 2 секунды
// повторяя ошибку которая никогда не успешна!
\`\`\`

**Умный условный Retry:**
\`\`\`go
isRetryable := func(err error) bool {
    var httpErr *HTTPError
    if errors.As(err, &httpErr) {
        // Retry server errors (500-599)
        // Не retry client errors (400-499)
        return httpErr.Code >= 500
    }
    return true  // Retry сетевых ошибок
}

err := RetryUntil(ctx, 3, 100*time.Millisecond, fetchUser, isRetryable)
// Возвращается сразу на 404, повторяет на 503
\`\`\`

**Классификация ошибок:**

**Retry-able ошибки (Временные):**
- 500 Internal Server Error → Баг сервера, может восстановиться
- 502 Bad Gateway → Upstream сервис недоступен, retry
- 503 Service Unavailable → Сервер перегружен, retry с backoff
- 504 Gateway Timeout → Медленный ответ, retry
- Connection refused → Сервис перезапускается, retry
- Timeout → Перегрузка сети, retry
- Rate limit exceeded → Подождать и retry

**Не retry-able ошибки (Постоянные):**
- 400 Bad Request → Некорректный ввод, не изменится при retry
- 401 Unauthorized → Некорректные credentials
- 403 Forbidden → Нет разрешения
- 404 Not Found → Ресурс не существует
- 422 Unprocessable Entity → Валидация провалилась
- Context canceled → Пользователь отменил запрос
- Invalid argument → Ошибка программирования

**Продакшен пример (AWS SDK):**
\`\`\`go
// AWS SDK retry логика
isRetryable := func(err error) bool {
    // Retry throttling errors с exponential backoff
    if errors.Is(err, ErrThrottling) {
        return true
    }
    // Retry transient сетевых ошибок
    if errors.Is(err, ErrNetworkError) {
        return true
    }
    // Не retry validation errors
    if errors.Is(err, ErrValidation) {
        return false
    }
    return false
}
\`\`\`

**Реальное влияние:**
\`\`\`
Сценарий: API вызов падает с 401 Unauthorized

Без предиката:
  Retry 1: 401 + 100ms ожидание = ПОТРАЧЕНО ВПУСТУЮ
  Retry 2: 401 + 200ms ожидание = ПОТРАЧЕНО ВПУСТУЮ
  Retry 3: 401 + 400ms ожидание = ПОТРАЧЕНО ВПУСТУЮ
  Всего: 700мс потрачено, пользователь видит медленную ошибку

С предикатом:
  Попытка 1: 401 → shouldRetry=false → возврат сразу
  Всего: <10мс, мгновенная обратная связь пользователю
\`\`\`

**Экономия ресурсов:**
- **Bandwidth:** Не retry запросы обреченные на провал
- **Latency:** Быстрый fail на постоянных ошибках
- **Resources:** Быстрее освобождаем connections/goroutines
- **User Experience:** Быстро показываем осмысленные ошибки

**Преимущество для отладки:**
Функция предиката - центральное место для логирования retry решений:
\`\`\`go
shouldRetry := func(err error) bool {
    retry := isTransientError(err)
    log.Printf("Ошибка: %v, Retry: %v", err, retry)
    return retry
}
\`\`\``
		},
		uz: {
			title: `Muvaffaqiyatgacha qayta urinish`,
			solutionCode: `package retryx

import (
	"context"
	"time"
)

func RetryUntil(
	ctx context.Context,
	n int,
	base time.Duration,
	op Op,
	shouldRetry func(error) bool,
) error {
	if shouldRetry == nil {	// Standart predikat: har doim retry
		shouldRetry = func(error) bool { return true }
	}
	var lastErr error	// Oxirgi xatoni kuzatamiz
	for attempt := 0; attempt < n; attempt++ {	// n martagacha urinish
		if ctx.Err() != nil {	// Kontekst bekor qilindi
			return ctx.Err()
		}
		if err := op(ctx); err != nil {	// Operatsiya muvaffaqiyatsiz
			lastErr = err	// Xatoni eslab qolamiz
			if !shouldRetry(err) {	// Predikat xatoni rad etdi
				return err	// Retry ni to'xtatamiz, darhol qaytamiz
			}
			if err := SleepContext(ctx, Backoff(attempt, base)); err != nil {	// Retrydan oldin kutamiz
				return err	// Uxlash paytida kontekst bekor qilindi
			}
			continue	// Yana urinib ko'ramiz
		}
		return nil	// Muvaffaqiyat
	}
	return lastErr	// Barcha retrylar tugadi
}`,
			description: `Faqat predikat funksiya true qaytarganda operatsiyani qayta urinadigan **RetryUntil** funksiyasini amalga oshiring.

**Talablar:**
1. Operatsiyani \`n\` martagacha qayta urining
2. Har bir muvaffaqiyatsizlikdan keyin \`shouldRetry(err)\` ni chaqiring
3. Agar \`shouldRetry(err)\` false qaytarsa retry ni to'xtating
4. Retrylar orasida eksponensial backoff qo'llang
5. Kontekst bekor qilishni hurmat qiling
6. Muvaffaqiyatda yoki predikat xatoni rad etganda darhol qaytaring

**Funksiya imzosi:**
\`\`\`go
func RetryUntil(
    ctx context.Context,
    n int,
    base time.Duration,
    op Op,
    shouldRetry func(error) bool,
) error
\`\`\`

**Retry mantiqi:**
\`\`\`
Urinish 1 → NetworkError → shouldRetry(err)=true → kutish & retry
Urinish 2 → AuthError → shouldRetry(err)=false → darhol qaytish
\`\`\`

**Edge Case lar:**
- \`shouldRetry\` nil → standart sifatida har doim retry
- Birinchi xatoda predikat false qaytardi → darhol qaytish
- Kontekst bekor qilindi → ctx.Err() qaytaring

**Cheklovlar:**
- Uxlashdan oldin predikatni tekshiring
- Retry qilib bo'lmaydigan xatolarni retry qilmang
- nil predikatni to'g'ri qayta ishlang (standart: har doim retry)`,
			hint1: `Davom etish kerakligini hal qilish uchun har bir muvaffaqiyatsizlikdan keyin shouldRetry(err) ni chaqiring.`,
			hint2: `shouldRetry false qaytarganda darhol qaytaring, kutmang.`,
			whyItMatters: `Shartli retry mantiqi tiklab bo'lmaydigan xatolarni retry qilishga vaqt sarflashni oldini oladi.

**Ko'r Retrylar bilan Muammo:**
\`\`\`go
// Yomon: Hamma narsani retry qiladi, doimiy nosozliklarni ham
func fetchUser(id int) error {
    for i := 0; i < 3; i++ {
        resp, err := http.Get(fmt.Sprintf("/users/%d", id))
        if err == nil {
            return nil
        }
        time.Sleep(time.Second)  // Isrofgarchilik!
    }
    return err
}

// Agar foydalanuvchi mavjud bo'lmasa (404), 2 soniya sarflaydi
// hech qachon muvaffaqiyatli bo'lmaydigan xatoni retry qilib!
\`\`\`

**Aqlli Shartli Retry:**
\`\`\`go
isRetryable := func(err error) bool {
    var httpErr *HTTPError
    if errors.As(err, &httpErr) {
        // Server xatolarini retry qilish (500-599)
        // Mijoz xatolarini retry qilmaslik (400-499)
        return httpErr.Code >= 500
    }
    return true  // Tarmoq xatolarini retry qilish
}

err := RetryUntil(ctx, 3, 100*time.Millisecond, fetchUser, isRetryable)
// 404 da darhol qaytadi, 503 da retry qiladi
\`\`\`

**Xatolarni Tasniflash:**

**Retry qilish mumkin bo'lgan xatolar (Vaqtinchalik):**
- 500 Internal Server Error → Server xatosi, tiklanishi mumkin
- 502 Bad Gateway → Upstream xizmat mavjud emas, retry
- 503 Service Unavailable → Server ortiqcha yuklangan, backoff bilan retry
- 504 Gateway Timeout → Sekin javob, retry
- Connection refused → Xizmat qayta ishga tushmoqda, retry
- Timeout → Tarmoq yuklanishi, retry
- Rate limit exceeded → Kutish va retry

**Retry qilib bo'lmaydigan xatolar (Doimiy):**
- 400 Bad Request → Noto'g'ri kirish, retry da o'zgarmaydi
- 401 Unauthorized → Noto'g'ri credentials
- 403 Forbidden → Ruxsat yo'q
- 404 Not Found → Resurs mavjud emas
- 422 Unprocessable Entity → Validatsiya muvaffaqiyatsiz
- Context canceled → Foydalanuvchi so'rovni bekor qildi
- Invalid argument → Dasturlash xatosi

**Ishlab chiqarish misoli (AWS SDK):**
\`\`\`go
// AWS SDK retry mantiqi
isRetryable := func(err error) bool {
    // Throttling xatolarini eksponensial backoff bilan retry qilish
    if errors.Is(err, ErrThrottling) {
        return true
    }
    // Vaqtinchalik tarmoq xatolarini retry qilish
    if errors.Is(err, ErrNetworkError) {
        return true
    }
    // Validatsiya xatolarini retry qilmaslik
    if errors.Is(err, ErrValidation) {
        return false
    }
    return false
}
\`\`\`

**Haqiqiy dunyo ta'siri:**
\`\`\`
Stsenariy: API chaqiruvi 401 Unauthorized bilan muvaffaqiyatsiz

Predikatsiz:
  Retry 1: 401 + 100ms kutish = BEHUDA SARFLANDI
  Retry 2: 401 + 200ms kutish = BEHUDA SARFLANDI
  Retry 3: 401 + 400ms kutish = BEHUDA SARFLANDI
  Jami: 700ms sarflandi, foydalanuvchi sekin xatoni ko'radi

Predikat bilan:
  Urinish 1: 401 → shouldRetry=false → darhol qaytish
  Jami: <10ms, foydalanuvchiga zudlik bilan javob
\`\`\`

**Resurs Tejash:**
- **Bandwidth:** Muvaffaqiyatsizlikka mahkum so'rovlarni retry qilmaslik
- **Latency:** Doimiy xatolarda tez fail qilish
- **Resources:** Connectionlar/goroutinelarni tezroq bo'shatish
- **User Experience:** Ma'noli xatolarni tez ko'rsatish

**Debug Foydalari:**
Predikat funksiyasi retry qarorlarini log qilish uchun markaziy joy:
\`\`\`go
shouldRetry := func(err error) bool {
    retry := isTransientError(err)
    log.Printf("Xato: %v, Retry: %v", err, retry)
    return retry
}
\`\`\``
		}
	}
};

export default task;
