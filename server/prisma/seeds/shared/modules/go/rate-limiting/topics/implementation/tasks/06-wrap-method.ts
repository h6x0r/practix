import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-ratelimit-wrap-method',
	title: 'Wrap Method',
	difficulty: 'medium',	tags: ['go', 'rate-limiting', 'higher-order-function', 'decorator'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement the **Wrap** method that decorates functions with automatic rate limiting.

**Requirements:**
1. Create method \`Wrap(fn func(ctx context.Context) error) func(ctx context.Context) error\`
2. Handle nil function:
   2.1. Return nil (makes it easy to detect programming errors)
   2.2. Don't return no-op function
3. Return wrapped function that:
   3.1. First calls \`Allow(ctx)\` to wait for token
   3.2. If rate limiter returns error, return that error immediately
   3.3. If token acquired, call original function \`fn(ctx)\`
   3.4. Return result of original function
4. Preserve original context throughout the call chain

**Example:**
\`\`\`go
limiter := New(10, 2)  // 10 RPS, burst 2

// Original function
processData := func(ctx context.Context) error {
    fmt.Println("Processing...")
    return nil
}

// Wrap with rate limiting
rateLimitedProcess := limiter.Wrap(processData)

// Call wrapped function - automatically rate limited
err := rateLimitedProcess(ctx)
// First waits for token, then executes processData

// Nil function handling
wrapped := limiter.Wrap(nil)  // Returns nil
if wrapped == nil {
    panic("forgot to pass function")  // Easy to detect
}
\`\`\`

**Constraints:**
- Must return nil for nil function (not no-op)
- Must call Allow before executing fn
- Must propagate context correctly`,
	initialCode: `package ratelimit

import (
	"context"
)

type Limiter struct {
	mu           sync.Mutex
	interval     time.Duration
	burst        int
	reservations []time.Time
}

// TODO: Implement Wrap method
func (l *Limiter) Wrap(fn func(ctx context.Context) error) func(ctx context.Context) error {
	// TODO: Implement
}`,
	solutionCode: `package ratelimit

import (
	"context"
)

type Limiter struct {
	mu           sync.Mutex
	interval     time.Duration
	burst        int
	reservations []time.Time
}

func (l *Limiter) Wrap(fn func(ctx context.Context) error) func(ctx context.Context) error {
	if fn == nil {                                       // Handle nil function: return nil for easy error detection
		return nil                                       // Caller can check if wrapped == nil
	}
	return func(ctx context.Context) error {             // Return wrapped function
		if err := l.Allow(ctx); err != nil {             // First acquire rate limit token
			return err                                   // Rate limit error (timeout/cancelled)
		}
		return fn(ctx)                                   // Execute original function with same context
	}
}`,
	testCode: `package ratelimit

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestWrapWithNilFunction(t *testing.T) {
	l := &Limiter{}
	result := l.Wrap(nil)
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestWrapWithValidFunction(t *testing.T) {
	l := &Limiter{interval: 100 * time.Millisecond, burst: 1}
	called := false
	fn := func(ctx context.Context) error {
		called = true
		return nil
	}
	wrapped := l.Wrap(fn)
	if wrapped == nil {
		t.Errorf("expected wrapped function, got nil")
	}
}

func TestWrapExecutesFunction(t *testing.T) {
	l := &Limiter{interval: 100 * time.Millisecond, burst: 1}
	callCount := 0
	fn := func(ctx context.Context) error {
		callCount++
		return nil
	}
	wrapped := l.Wrap(fn)
	wrapped(context.Background())
	if callCount != 1 {
		t.Errorf("expected callCount 1, got %d", callCount)
	}
}

func TestWrapPreservesContext(t *testing.T) {
	l := &Limiter{interval: 100 * time.Millisecond, burst: 1}
	type key string
	testKey := key("test")
	testValue := "value"
	var receivedValue string
	fn := func(ctx context.Context) error {
		receivedValue = ctx.Value(testKey).(string)
		return nil
	}
	wrapped := l.Wrap(fn)
	ctx := context.WithValue(context.Background(), testKey, testValue)
	wrapped(ctx)
	if receivedValue != testValue {
		t.Errorf("expected %s, got %s", testValue, receivedValue)
	}
}

func TestWrapReturnsOriginalError(t *testing.T) {
	l := &Limiter{interval: 100 * time.Millisecond, burst: 1}
	expectedErr := errors.New("test error")
	fn := func(ctx context.Context) error {
		return expectedErr
	}
	wrapped := l.Wrap(fn)
	err := wrapped(context.Background())
	if err != expectedErr {
		t.Errorf("expected %v, got %v", expectedErr, err)
	}
}

func TestWrapCallsAllowBeforeFunction(t *testing.T) {
	l := &Limiter{interval: 100 * time.Millisecond, burst: 1}
	allowCalled := false
	functionCalled := false
	fn := func(ctx context.Context) error {
		functionCalled = true
		return nil
	}
	wrapped := l.Wrap(fn)
	wrapped(context.Background())
	allowCalled = true
	if !allowCalled || !functionCalled {
		t.Errorf("expected both Allow and function to be called")
	}
}

func TestWrapWithCancelledContext(t *testing.T) {
	l := &Limiter{interval: 100 * time.Millisecond, burst: 1}
	fn := func(ctx context.Context) error {
		return nil
	}
	wrapped := l.Wrap(fn)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := wrapped(ctx)
	if err == nil {
		t.Errorf("expected error from cancelled context, got nil")
	}
}

func TestWrapMultipleCalls(t *testing.T) {
	l := &Limiter{interval: 100 * time.Millisecond, burst: 1}
	callCount := 0
	fn := func(ctx context.Context) error {
		callCount++
		return nil
	}
	wrapped := l.Wrap(fn)
	wrapped(context.Background())
	wrapped(context.Background())
	if callCount != 2 {
		t.Errorf("expected callCount 2, got %d", callCount)
	}
}

func TestWrapDoesNotCallFunctionOnAllowError(t *testing.T) {
	l := &Limiter{interval: 100 * time.Millisecond, burst: 0}
	functionCalled := false
	fn := func(ctx context.Context) error {
		functionCalled = true
		return nil
	}
	wrapped := l.Wrap(fn)
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()
	wrapped(ctx)
	time.Sleep(10 * time.Millisecond)
	if functionCalled {
		t.Errorf("expected function not to be called when Allow fails")
	}
}

func TestWrapFunctionReturnsNil(t *testing.T) {
	l := &Limiter{interval: 100 * time.Millisecond, burst: 1}
	fn := func(ctx context.Context) error {
		return nil
	}
	wrapped := l.Wrap(fn)
	err := wrapped(context.Background())
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
}`,
			hint1: `Check if fn is nil first, return nil (not a no-op function). This makes nil function easier to detect.`,
			hint2: `Return anonymous function that calls l.Allow(ctx) first, then fn(ctx) if no error.`,
			whyItMatters: `Wrap method enables declarative rate limiting, making it easy to add throttling to any function without modifying its logic.

**Why Function Wrapping:**
- **Separation of Concerns:** Rate limiting logic separated from business logic
- **Composability:** Chain multiple wrappers (rate limit + retry + circuit breaker)
- **Declarative:** Rate limiting behavior defined at function creation, not at call site
- **Reusability:** Wrap once, use many times without repeating rate limit code

**Real-World Patterns:**

**HTTP Handler Wrapping:**
\`\`\`go
limiter := New(100, 10)

// Original handler
handleUser := func(ctx context.Context) error {
    user := getUserFromDB(ctx)
    return json.NewEncoder(w).Encode(user)
}

// Wrap with rate limiting
rateLimitedHandler := limiter.Wrap(handleUser)

// Use in HTTP server
http.HandleFunc("/user", func(w http.ResponseWriter, r *http.Request) {
    if err := rateLimitedHandler(r.Context()); err != nil {
        http.Error(w, "Rate limit exceeded", 429)
        return
    }
})
\`\`\`

**External API Client:**
\`\`\`go
type APIClient struct {
    limiter        *Limiter
    fetchTweets    func(ctx context.Context) error
    postTweet      func(ctx context.Context) error
}

func NewAPIClient() *APIClient {
    limiter := New(50, 5)  // Twitter API: 50 req/sec

    return &APIClient{
        limiter: limiter,
        // Wrap all API methods with rate limiting
        fetchTweets: limiter.Wrap(func(ctx context.Context) error {
            return callTwitterAPI(ctx, "/tweets")
        }),
        postTweet: limiter.Wrap(func(ctx context.Context) error {
            return callTwitterAPI(ctx, "/tweet")
        }),
    }
}

// All methods automatically rate limited
func (c *APIClient) GetTweets(ctx context.Context) error {
    return c.fetchTweets(ctx)  // Rate limited automatically
}
\`\`\`

**Background Job Processing:**
\`\`\`go
type JobProcessor struct {
    limiter     *Limiter
    processJob  func(ctx context.Context) error
}

func NewJobProcessor() *JobProcessor {
    limiter := New(10, 2)  // 10 jobs/sec, burst 2

    return &JobProcessor{
        limiter: limiter,
        processJob: limiter.Wrap(func(ctx context.Context) error {
            // Actual job processing logic
            return processJobImpl(ctx)
        }),
    }
}

func (p *JobProcessor) Process(ctx context.Context, jobs []Job) error {
    for _, job := range jobs {
        // Rate limiting handled by wrapper
        if err := p.processJob(ctx); err != nil {
            return err
        }
    }
    return nil
}
\`\`\`

**Database Operations:**
\`\`\`go
type Database struct {
    limiter *Limiter
    query   func(ctx context.Context) error
    insert  func(ctx context.Context) error
}

func NewDatabase(db *sql.DB) *Database {
    limiter := New(1000, 50)  // 1000 queries/sec, burst 50

    return &Database{
        limiter: limiter,
        // Wrap all DB operations
        query: limiter.Wrap(func(ctx context.Context) error {
            return db.QueryContext(ctx, "SELECT ...")
        }),
        insert: limiter.Wrap(func(ctx context.Context) error {
            return db.ExecContext(ctx, "INSERT ...")
        }),
    }
}
\`\`\`

**Composition Pattern (Multiple Wrappers):**
\`\`\`go
// Combine rate limiting, retry, and circuit breaker
func buildResilientClient() func(ctx context.Context) error {
    limiter := New(100, 10)

    // Original function
    callAPI := func(ctx context.Context) error {
        return httpClient.Do(req)
    }

    // Wrap with rate limiting
    rateLimited := limiter.Wrap(callAPI)

    // Wrap with retry logic
    withRetry := retryWrapper(rateLimited, 3)

    // Wrap with circuit breaker
    resilient := circuitBreakerWrapper(withRetry)

    return resilient
}

// Usage: All protections applied transparently
err := resilientClient(ctx)
\`\`\`

**Nil Function Handling Pattern:**
\`\`\`go
// CORRECT: Return nil for easy detection
func (l *Limiter) Wrap(fn func(ctx context.Context) error) func(ctx context.Context) error {
    if fn == nil {
        return nil  // Caller can detect: if wrapped == nil { panic(...) }
    }
    return func(ctx context.Context) error {
        if err := l.Allow(ctx); err != nil {
            return err
        }
        return fn(ctx)
    }
}

// Usage: Easy to detect programming errors
wrapped := limiter.Wrap(nil)
if wrapped == nil {
    panic("forgot to pass function")  // Fails fast at startup
}

// WRONG: Return no-op function
func (l *Limiter) Wrap(fn func(ctx context.Context) error) func(ctx context.Context) error {
    if fn == nil {
        return func(ctx context.Context) error { return nil }  // Silent failure
    }
    // ...
}

// Usage: Silent bug, hard to debug
wrapped := limiter.Wrap(nil)  // Returns no-op, no error
wrapped(ctx)                   // Silently does nothing, bug hidden
\`\`\`

**Testing Pattern:**
\`\`\`go
func TestRateLimitedOperation(t *testing.T) {
    limiter := New(10, 1)
    callCount := 0

    // Wrap test function
    operation := limiter.Wrap(func(ctx context.Context) error {
        callCount++
        return nil
    })

    // First call: immediate
    start := time.Now()
    operation(context.Background())
    assert.Less(t, time.Since(start), 10*time.Millisecond)

    // Second call: rate limited (waits ~100ms at 10 RPS)
    start = time.Now()
    operation(context.Background())
    assert.Greater(t, time.Since(start), 90*time.Millisecond)
    assert.Equal(t, 2, callCount)
}
\`\`\`

**Key Concepts:**
- **Decorator Pattern:** Wrap function with additional behavior (rate limiting)
- **Higher-Order Function:** Function that takes and returns functions
- **Closure:** Wrapped function captures limiter and original fn
- **Nil Handling:** Return nil (not no-op) for early error detection

**Benefits:**
- **Clean Code:** Rate limiting defined once at wrapper creation
- **Type Safety:** Same function signature in and out
- **Testability:** Easy to test wrapped vs unwrapped functions
- **Flexibility:** Can unwrap or swap wrappers as needed

**Common Use Cases:**
- **API Clients:** Wrap all API methods with rate limiting
- **Database Pools:** Wrap query/exec methods to throttle
- **Background Workers:** Wrap job processing functions
- **Event Handlers:** Wrap event processing with rate limits

**Performance:**
- **Zero Overhead:** Wrapper only adds one function call
- **No Allocation:** After initial wrapper creation (closure allocated once)
- **Efficient:** Rate limiting logic only runs when function is called

Wrap method makes rate limiting elegant and composable, following Go's philosophy of simple, composable abstractions.`,	order: 5,
	translations: {
		ru: {
			title: 'Обёртка функции с ограничением скорости',
			solutionCode: `package ratelimit

import (
	"context"
)

type Limiter struct {
	mu           sync.Mutex
	interval     time.Duration
	burst        int
	reservations []time.Time
}

func (l *Limiter) Wrap(fn func(ctx context.Context) error) func(ctx context.Context) error {
	if fn == nil {                                       // Обработка nil функции: возвращаем nil для простого обнаружения ошибок
		return nil                                       // Вызывающий может проверить if wrapped == nil
	}
	return func(ctx context.Context) error {             // Возвращаем обёрнутую функцию
		if err := l.Allow(ctx); err != nil {             // Сначала получаем токен rate limit
			return err                                   // Ошибка rate limit (таймаут/отменён)
		}
		return fn(ctx)                                   // Выполняем оригинальную функцию с тем же контекстом
	}
}`,
			description: `Реализуйте метод **Wrap**, который декорирует функции автоматическим rate limiting.

**Требования:**
1. Создайте метод \`Wrap(fn func(ctx context.Context) error) func(ctx context.Context) error\`
2. Обработайте nil функцию:
   2.1. Верните nil (упрощает обнаружение ошибок программирования)
   2.2. Не возвращайте no-op функцию
3. Верните обёрнутую функцию которая:
   3.1. Сначала вызывает \`Allow(ctx)\` для ожидания токена
   3.2. Если rate limiter возвращает ошибку, немедленно вернуть эту ошибку
   3.3. Если токен получен, вызвать оригинальную функцию \`fn(ctx)\`
   3.4. Вернуть результат оригинальной функции
4. Сохраните оригинальный контекст через всю цепочку вызовов

**Пример:**
\`\`\`go
limiter := New(10, 2)  // 10 RPS, burst 2

// Оригинальная функция
processData := func(ctx context.Context) error {
    fmt.Println("Processing...")
    return nil
}

// Обернуть с rate limiting
rateLimitedProcess := limiter.Wrap(processData)

// Вызов обёрнутой функции - автоматически rate limited
err := rateLimitedProcess(ctx)
// Сначала ждёт токен, затем выполняет processData

// Обработка nil функции
wrapped := limiter.Wrap(nil)  // Возвращает nil
if wrapped == nil {
    panic("forgot to pass function")  // Легко обнаружить
}
\`\`\`

**Ограничения:**
- Должен возвращать nil для nil функции (не no-op)
- Должен вызывать Allow перед выполнением fn
- Должен корректно передавать контекст`,
			hint1: `Сначала проверьте fn на nil, верните nil (не no-op функцию). Это упрощает обнаружение nil функций.`,
			hint2: `Верните анонимную функцию которая сначала вызывает l.Allow(ctx), затем fn(ctx) если нет ошибки.`,
			whyItMatters: `Метод Wrap обеспечивает декларативное rate limiting, упрощая добавление throttling к любой функции без изменения её логики.

**Почему обёртывание функций:**
- **Separation of Concerns:** Логика rate limiting отделена от бизнес-логики
- **Composability:** Цепочка нескольких обёрток (rate limit + retry + circuit breaker)
- **Declarative:** Поведение rate limiting определено при создании функции, не в месте вызова
- **Reusability:** Обернули один раз, используем много раз без повторения кода

**Реальные паттерны:**

**HTTP Handler обёртывание:**
\`\`\`go
limiter := New(100, 10)

// Оригинальный handler
handleUser := func(ctx context.Context) error {
    user := getUserFromDB(ctx)
    return json.NewEncoder(w).Encode(user)
}

// Обернуть с rate limiting
rateLimitedHandler := limiter.Wrap(handleUser)

// Использование в HTTP сервере
http.HandleFunc("/user", func(w http.ResponseWriter, r *http.Request) {
    if err := rateLimitedHandler(r.Context()); err != nil {
        http.Error(w, "Rate limit exceeded", 429)
        return
    }
})
\`\`\`

**External API Client:**
\`\`\`go
type APIClient struct {
    limiter        *Limiter
    fetchTweets    func(ctx context.Context) error
    postTweet      func(ctx context.Context) error
}

func NewAPIClient() *APIClient {
    limiter := New(50, 5)  // Twitter API: 50 req/sec

    return &APIClient{
        limiter: limiter,
        // Обернуть все API методы с rate limiting
        fetchTweets: limiter.Wrap(func(ctx context.Context) error {
            return callTwitterAPI(ctx, "/tweets")
        }),
        postTweet: limiter.Wrap(func(ctx context.Context) error {
            return callTwitterAPI(ctx, "/tweet")
        }),
    }
}

// Все методы автоматически rate limited
func (c *APIClient) GetTweets(ctx context.Context) error {
    return c.fetchTweets(ctx)  // Rate limited автоматически
}
\`\`\`

**Background обработка задач:**
\`\`\`go
type JobProcessor struct {
    limiter     *Limiter
    processJob  func(ctx context.Context) error
}

func NewJobProcessor() *JobProcessor {
    limiter := New(10, 2)  // 10 jobs/sec, burst 2

    return &JobProcessor{
        limiter: limiter,
        processJob: limiter.Wrap(func(ctx context.Context) error {
            // Фактическая логика обработки задачи
            return processJobImpl(ctx)
        }),
    }
}

func (p *JobProcessor) Process(ctx context.Context, jobs []Job) error {
    for _, job := range jobs {
        // Rate limiting обрабатывается обёрткой
        if err := p.processJob(ctx); err != nil {
            return err
        }
    }
    return nil
}
\`\`\`

**Операции с базой данных:**
\`\`\`go
type Database struct {
    limiter *Limiter
    query   func(ctx context.Context) error
    insert  func(ctx context.Context) error
}

func NewDatabase(db *sql.DB) *Database {
    limiter := New(1000, 50)  // 1000 queries/sec, burst 50

    return &Database{
        limiter: limiter,
        // Обернуть все операции с БД
        query: limiter.Wrap(func(ctx context.Context) error {
            return db.QueryContext(ctx, "SELECT ...")
        }),
        insert: limiter.Wrap(func(ctx context.Context) error {
            return db.ExecContext(ctx, "INSERT ...")
        }),
    }
}
\`\`\`

**Паттерн композиции (несколько обёрток):**
\`\`\`go
// Объединить rate limiting, retry и circuit breaker
func buildResilientClient() func(ctx context.Context) error {
    limiter := New(100, 10)

    // Оригинальная функция
    callAPI := func(ctx context.Context) error {
        return httpClient.Do(req)
    }

    // Обернуть с rate limiting
    rateLimited := limiter.Wrap(callAPI)

    // Обернуть с логикой retry
    withRetry := retryWrapper(rateLimited, 3)

    // Обернуть с circuit breaker
    resilient := circuitBreakerWrapper(withRetry)

    return resilient
}

// Использование: Все защиты применены прозрачно
err := resilientClient(ctx)
\`\`\`

**Паттерн обработки nil функции:**
\`\`\`go
// ПРАВИЛЬНО: Вернуть nil для лёгкого обнаружения
func (l *Limiter) Wrap(fn func(ctx context.Context) error) func(ctx context.Context) error {
    if fn == nil {
        return nil  // Вызывающий может обнаружить: if wrapped == nil { panic(...) }
    }
    return func(ctx context.Context) error {
        if err := l.Allow(ctx); err != nil {
            return err
        }
        return fn(ctx)
    }
}

// Использование: Легко обнаружить ошибки программирования
wrapped := limiter.Wrap(nil)
if wrapped == nil {
    panic("forgot to pass function")  // Падает быстро при запуске
}

// НЕПРАВИЛЬНО: Вернуть no-op функцию
func (l *Limiter) Wrap(fn func(ctx context.Context) error) func(ctx context.Context) error {
    if fn == nil {
        return func(ctx context.Context) error { return nil }  // Тихий сбой
    }
    // ...
}

// Использование: Тихий баг, сложно отлаживать
wrapped := limiter.Wrap(nil)  // Возвращает no-op, без ошибки
wrapped(ctx)                   // Тихо ничего не делает, баг скрыт
\`\`\`

**Паттерн тестирования:**
\`\`\`go
func TestRateLimitedOperation(t *testing.T) {
    limiter := New(10, 1)
    callCount := 0

    // Обернуть тестовую функцию
    operation := limiter.Wrap(func(ctx context.Context) error {
        callCount++
        return nil
    })

    // Первый вызов: немедленно
    start := time.Now()
    operation(context.Background())
    assert.Less(t, time.Since(start), 10*time.Millisecond)

    // Второй вызов: rate limited (ждёт ~100ms при 10 RPS)
    start = time.Now()
    operation(context.Background())
    assert.Greater(t, time.Since(start), 90*time.Millisecond)
    assert.Equal(t, 2, callCount)
}
\`\`\`

**Ключевые концепции:**
- **Паттерн декоратора:** Обернуть функцию с дополнительным поведением (rate limiting)
- **Higher-Order функция:** Функция которая принимает и возвращает функции
- **Замыкание:** Обёрнутая функция захватывает limiter и оригинальный fn
- **Обработка Nil:** Вернуть nil (не no-op) для раннего обнаружения ошибок

**Преимущества:**
- **Чистый код:** Rate limiting определён один раз при создании обёртки
- **Type Safety:** Та же сигнатура функции на входе и выходе
- **Testability:** Легко тестировать обёрнутые vs необёрнутые функции
- **Гибкость:** Можно разворачивать или менять обёртки по необходимости

**Общие случаи использования:**
- **API Clients:** Оберните все API методы с rate limiting
- **Database Pools:** Оберните query/exec методы для throttle
- **Background Workers:** Оберните функции обработки задач
- **Event Handlers:** Оберните обработку событий с rate limits

**Производительность:**
- **Zero Overhead:** Обёртка добавляет только один вызов функции
- **No Allocation:** После начального создания обёртки (closure выделяется один раз)
- **Эффективно:** Логика rate limiting запускается только при вызове функции

Метод Wrap делает rate limiting элегантным и композируемым, следуя философии Go простых, композируемых абстракций.`
		},
		uz: {
			title: `Tezlik cheklovi bilan funksiyani o'rash`,
			solutionCode: `package ratelimit

import (
	"context"
)

type Limiter struct {
	mu           sync.Mutex
	interval     time.Duration
	burst        int
	reservations []time.Time
}

func (l *Limiter) Wrap(fn func(ctx context.Context) error) func(ctx context.Context) error {
	if fn == nil {                                       // nil funksiyani qayta ishlash: xatolarni oson aniqlash uchun nil qaytaramiz
		return nil                                       // Chaqiruvchi if wrapped == nil tekshirishi mumkin
	}
	return func(ctx context.Context) error {             // O'ralgan funksiya qaytaramiz
		if err := l.Allow(ctx); err != nil {             // Avval rate limit token olamiz
			return err                                   // Rate limit xatosi (timeout/bekor qilindi)
		}
		return fn(ctx)                                   // Asl funksiyani xuddi shu kontekst bilan bajaramiz
	}
}`,
			description: `Funksiyalarni avtomatik rate limiting bilan bezaydigan **Wrap** metodini amalga oshiring.

**Talablar:**
1. \`Wrap(fn func(ctx context.Context) error) func(ctx context.Context) error\` metodini yarating
2. nil funksiyani qayta ishlang:
   2.1. nil qaytaring (dasturlash xatolarini aniqlashni osonlashtiradi)
   2.2. no-op funksiya qaytarmang
3. O'ralgan funksiya qaytaring:
   3.1. Avval token uchun \`Allow(ctx)\` chaqiring
   3.2. Rate limiter xato qaytarsa, darhol shu xatoni qaytaring
   3.3. Token olinsa, asl funksiya \`fn(ctx)\` ni chaqiring
   3.4. Asl funksiya natijasini qaytaring
4. Chaqiruv zanjiri davomida asl kontekstni saqlang

**Misol:**
\`\`\`go
limiter := New(10, 2)  // 10 RPS, burst 2

// Asl funksiya
processData := func(ctx context.Context) error {
    fmt.Println("Processing...")
    return nil
}

// Rate limiting bilan o'rash
rateLimitedProcess := limiter.Wrap(processData)

// O'ralgan funksiyani chaqirish - avtomatik rate limited
err := rateLimitedProcess(ctx)
// Avval token kutadi, keyin processData ni bajaradi

// nil funksiyani qayta ishlash
wrapped := limiter.Wrap(nil)  // nil qaytaradi
if wrapped == nil {
    panic("forgot to pass function")  // Aniqlash oson
}
\`\`\`

**Cheklovlar:**
- nil funksiya uchun nil qaytarishi kerak (no-op emas)
- fn ni bajarishdan oldin Allow chaqirishi kerak
- Kontekstni to'g'ri tarqatishi kerak`,
			hint1: `Avval fn ni nil uchun tekshiring, nil qaytaring (no-op funksiya emas). Bu nil funksiyalarni aniqlashni osonlashtiradi.`,
			hint2: `Avval l.Allow(ctx), keyin xato bo'lmasa fn(ctx) chaqiradigan anonim funksiya qaytaring.`,
			whyItMatters: `Wrap metodi deklarativ rate limiting ni ta'minlaydi, har qanday funksiyaga uning mantiqini o'zgartirmasdan throttling qo'shishni osonlashtiradi.

**Nima uchun funksiya o'rash:**
- **Concern larni ajratish:** Rate limiting mantiqi biznes mantiqidan ajratilgan
- **Composability:** Bir nechta o'ramlar zanjiri (rate limit + retry + circuit breaker)
- **Deklarativ:** Rate limiting xatti-harakati funksiya yaratilganda belgilanadi, chaqiruv joyida emas
- **Qayta foydalanish:** Bir marta o'rang, rate limit kodini takrorlamasdan ko'p marta foydalaning

**Haqiqiy patternlar:**

**HTTP Handler o'rash:**
\`\`\`go
limiter := New(100, 10)

// Asl handler
handleUser := func(ctx context.Context) error {
    user := getUserFromDB(ctx)
    return json.NewEncoder(w).Encode(user)
}

// Rate limiting bilan o'rash
rateLimitedHandler := limiter.Wrap(handleUser)

// HTTP serverda foydalanish
http.HandleFunc("/user", func(w http.ResponseWriter, r *http.Request) {
    if err := rateLimitedHandler(r.Context()); err != nil {
        http.Error(w, "Rate limit exceeded", 429)
        return
    }
})
\`\`\`

**External API Client:**
\`\`\`go
type APIClient struct {
    limiter        *Limiter
    fetchTweets    func(ctx context.Context) error
    postTweet      func(ctx context.Context) error
}

func NewAPIClient() *APIClient {
    limiter := New(50, 5)  // Twitter API: 50 req/sec

    return &APIClient{
        limiter: limiter,
        // Barcha API metodlarini rate limiting bilan o'rash
        fetchTweets: limiter.Wrap(func(ctx context.Context) error {
            return callTwitterAPI(ctx, "/tweets")
        }),
        postTweet: limiter.Wrap(func(ctx context.Context) error {
            return callTwitterAPI(ctx, "/tweet")
        }),
    }
}

// Barcha metodlar avtomatik rate limited
func (c *APIClient) GetTweets(ctx context.Context) error {
    return c.fetchTweets(ctx)  // Avtomatik rate limited
}
\`\`\`

**Background vazifalarni qayta ishlash:**
\`\`\`go
type JobProcessor struct {
    limiter     *Limiter
    processJob  func(ctx context.Context) error
}

func NewJobProcessor() *JobProcessor {
    limiter := New(10, 2)  // 10 jobs/sec, burst 2

    return &JobProcessor{
        limiter: limiter,
        processJob: limiter.Wrap(func(ctx context.Context) error {
            // Haqiqiy vazifani qayta ishlash mantiqi
            return processJobImpl(ctx)
        }),
    }
}

func (p *JobProcessor) Process(ctx context.Context, jobs []Job) error {
    for _, job := range jobs {
        // Rate limiting o'ram tomonidan boshqariladi
        if err := p.processJob(ctx); err != nil {
            return err
        }
    }
    return nil
}
\`\`\`

**Ma'lumotlar bazasi operatsiyalari:**
\`\`\`go
type Database struct {
    limiter *Limiter
    query   func(ctx context.Context) error
    insert  func(ctx context.Context) error
}

func NewDatabase(db *sql.DB) *Database {
    limiter := New(1000, 50)  // 1000 queries/sec, burst 50

    return &Database{
        limiter: limiter,
        // Barcha DB operatsiyalarini o'rash
        query: limiter.Wrap(func(ctx context.Context) error {
            return db.QueryContext(ctx, "SELECT ...")
        }),
        insert: limiter.Wrap(func(ctx context.Context) error {
            return db.ExecContext(ctx, "INSERT ...")
        }),
    }
}
\`\`\`

**Kompozitsiya patterni (bir nechta o'ramlar):**
\`\`\`go
// Rate limiting, retry va circuit breaker ni birlashtirish
func buildResilientClient() func(ctx context.Context) error {
    limiter := New(100, 10)

    // Asl funksiya
    callAPI := func(ctx context.Context) error {
        return httpClient.Do(req)
    }

    // Rate limiting bilan o'rash
    rateLimited := limiter.Wrap(callAPI)

    // Retry mantiqi bilan o'rash
    withRetry := retryWrapper(rateLimited, 3)

    // Circuit breaker bilan o'rash
    resilient := circuitBreakerWrapper(withRetry)

    return resilient
}

// Foydalanish: Barcha himoyalar shaffof qo'llaniladi
err := resilientClient(ctx)
\`\`\`

**nil funksiyani qayta ishlash patterni:**
\`\`\`go
// TO'G'RI: Oson aniqlash uchun nil qaytarish
func (l *Limiter) Wrap(fn func(ctx context.Context) error) func(ctx context.Context) error {
    if fn == nil {
        return nil  // Chaqiruvchi aniqlashi mumkin: if wrapped == nil { panic(...) }
    }
    return func(ctx context.Context) error {
        if err := l.Allow(ctx); err != nil {
            return err
        }
        return fn(ctx)
    }
}

// Foydalanish: Dasturlash xatolarini aniqlash oson
wrapped := limiter.Wrap(nil)
if wrapped == nil {
    panic("forgot to pass function")  // Ishga tushirishda tezda muvaffaqiyatsiz
}

// NOTO'G'RI: no-op funksiya qaytarish
func (l *Limiter) Wrap(fn func(ctx context.Context) error) func(ctx context.Context) error {
    if fn == nil {
        return func(ctx context.Context) error { return nil }  // Sokin muvaffaqiyatsizlik
    }
    // ...
}

// Foydalanish: Sokin bug, debug qilish qiyin
wrapped := limiter.Wrap(nil)  // no-op ni qaytaradi, xato yo'q
wrapped(ctx)                   // Sokin hech narsa qilmaydi, bug yashiringan
\`\`\`

**Sinov patterni:**
\`\`\`go
func TestRateLimitedOperation(t *testing.T) {
    limiter := New(10, 1)
    callCount := 0

    // Test funksiyasini o'rash
    operation := limiter.Wrap(func(ctx context.Context) error {
        callCount++
        return nil
    })

    // Birinchi chaqiruv: darhol
    start := time.Now()
    operation(context.Background())
    assert.Less(t, time.Since(start), 10*time.Millisecond)

    // Ikkinchi chaqiruv: rate limited (10 RPS da ~100ms kutadi)
    start = time.Now()
    operation(context.Background())
    assert.Greater(t, time.Since(start), 90*time.Millisecond)
    assert.Equal(t, 2, callCount)
}
\`\`\`

**Asosiy tushunchalar:**
- **Dekorator patterni:** Qo'shimcha xatti-harakat (rate limiting) bilan funksiyani o'rash
- **Higher-Order funksiya:** Funksiyalarni qabul qiladigan va qaytaradigan funksiya
- **Closure:** O'ralgan funksiya limiter va asl fn ni ushlab qoladi
- **Nil qayta ishlash:** Erta xatolarni aniqlash uchun nil (no-op emas) qaytarish

**Afzalliklar:**
- **Toza kod:** Rate limiting o'ram yaratilganda bir marta belgilanadi
- **Type Safety:** Kirishda va chiqishda bir xil funksiya imzosi
- **Testability:** O'ralgan vs o'ralmagan funksiyalarni osongina sinash
- **Moslashuvchanlik:** Kerak bo'lganda o'ramlarni ochish yoki almashtirish mumkin

**Umumiy foydalanish holatlari:**
- **API Clientlar:** Barcha API metodlarini rate limiting bilan o'rang
- **Database Poollar:** Throttle uchun query/exec metodlarini o'rang
- **Background Workerlar:** Vazifalarni qayta ishlash funksiyalarini o'rang
- **Event Handlerlar:** Rate limitlar bilan event qayta ishlashni o'rang

**Performans:**
- **Zero Overhead:** O'ram faqat bitta funksiya chaqiruvini qo'shadi
- **No Allocation:** Dastlabki o'ram yaratilgandan keyin (closure bir marta ajratiladi)
- **Samarali:** Rate limiting mantiqi faqat funksiya chaqirilganda ishga tushadi

Wrap metodi rate limiting ni nafis va kompozitsion qiladi, Go ning oddiy, kompozitsion abstraktsiyalar falsafasiga amal qiladi.`
		}
	}
};

export default task;
