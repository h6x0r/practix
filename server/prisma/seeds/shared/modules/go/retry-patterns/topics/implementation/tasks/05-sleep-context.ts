import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-retry-sleep-context',
	title: 'Context-Aware Sleep',
	difficulty: 'medium',	tags: ['go', 'context', 'sleep', 'timer'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **SleepContext** function that sleeps for a duration while respecting context cancellation.

**Requirements:**
1. Sleep for duration \`d\`
2. Return immediately if context is canceled
3. Return \`ctx.Err()\` if canceled during sleep
4. Return \`nil\` if sleep completed successfully
5. Handle d <= 0 by returning immediately
6. Properly clean up timer resources

**Why Not time.Sleep?**
\`\`\`go
// ❌ BAD: Ignores context cancellation
time.Sleep(5 * time.Second)  // Blocks for full 5s even if ctx canceled

// ✅ GOOD: Respects context
SleepContext(ctx, 5*time.Second)  // Returns immediately on ctx.Done()
\`\`\`

**Timer Resource Management:**
\`\`\`go
// Must clean up timer to prevent goroutine leak
timer := time.NewTimer(d)
defer func() {
    if !timer.Stop() {  // Try to stop timer
        <-timer.C       // Drain channel if already fired
    }
}()
\`\`\`

**Example:**
\`\`\`go
// Sleep for 1 second unless canceled
ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
defer cancel()

err := SleepContext(ctx, 1*time.Second)
// Returns after 500ms with ctx.Err() (timeout)

// Sleep for 0 duration
err := SleepContext(ctx, 0)
// Returns immediately with nil
\`\`\`

**Select Statement Pattern:**
\`\`\`go
select {
case <-ctx.Done():
    // Context canceled before timer fired
    return ctx.Err()
case <-timer.C:
    // Timer fired successfully
    return nil
}
\`\`\`

**Use Cases:**
\`\`\`go
// 1. Rate limiting with cancellation
func rateLimit(ctx context.Context) error {
    return SleepContext(ctx, time.Second/requestsPerSecond)
}

// 2. Polling with timeout
for {
    if done := checkCondition(); done {
        break
    }
    if err := SleepContext(ctx, 100*time.Millisecond); err != nil {
        return err  // Context canceled
    }
}

// 3. Retry with backoff (used by Do function)
if err := SleepContext(ctx, Backoff(attempt, base)); err != nil {
    return err  // Stop retrying if context canceled
}
\`\`\`

**Edge Cases:**
- d <= 0 → return nil immediately (no sleep)
- Context already canceled → return ctx.Err() immediately
- Timer fires before context canceled → return nil
- Context canceled during sleep → clean up timer and return ctx.Err()

**Timer Cleanup Importance:**
\`\`\`go
// ❌ BAD: Timer goroutine leaks
timer := time.NewTimer(d)
select {
case <-ctx.Done():
    return ctx.Err()  // Timer still running!
case <-timer.C:
    return nil
}

// ✅ GOOD: Timer cleaned up
timer := time.NewTimer(d)
defer func() {
    if !timer.Stop() {
        <-timer.C  // Prevent goroutine leak
    }
}()
select {
case <-ctx.Done():
    return ctx.Err()  // Timer will be stopped by defer
case <-timer.C:
    return nil
}
\`\`\`

**Constraints:**
- Use time.NewTimer, not time.After (better resource control)
- Clean up timer in all code paths
- Handle zero/negative durations gracefully`,
	initialCode: `package retryx

import (
	"context"
	"time"
)

// TODO: Implement SleepContext function
// Sleep for duration d while respecting context cancellation
func SleepContext($2) error {
	return nil // TODO: Implement
}`,
	solutionCode: `package retryx

import (
	"context"
	"time"
)

func SleepContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {	// No sleep needed for non-positive durations
		return nil
	}
	timer := time.NewTimer(d)	// Create timer for requested duration
	defer func() {
		if !timer.Stop() {	// Stop timer and check if already fired
			<-timer.C	// Drain channel to prevent goroutine leak
		}
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()	// Context canceled, return error
	case <-timer.C:
		return nil	// Timer fired successfully
	}
}`,
	testCode: `package retryx

import (
	"context"
	"testing"
	"time"
)

func TestSleepContextCompletesSuccessfully(t *testing.T) {
	ctx := context.Background()
	start := time.Now()
	err := SleepContext(ctx, 50*time.Millisecond)
	elapsed := time.Since(start)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if elapsed < 40*time.Millisecond {
		t.Errorf("expected sleep of ~50ms, got %v", elapsed)
	}
}

func TestSleepContextWithCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := SleepContext(ctx, 1*time.Second)
	if err != context.Canceled {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestSleepContextCanceledDuringSleep(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	start := time.Now()
	err := SleepContext(ctx, 1*time.Second)
	elapsed := time.Since(start)
	if err == nil {
		t.Errorf("expected error, got nil")
	}
	if elapsed > 100*time.Millisecond {
		t.Errorf("expected early return due to timeout, got %v", elapsed)
	}
}

func TestSleepContextWithZeroDuration(t *testing.T) {
	ctx := context.Background()
	start := time.Now()
	err := SleepContext(ctx, 0)
	elapsed := time.Since(start)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if elapsed > 10*time.Millisecond {
		t.Errorf("expected immediate return, got %v delay", elapsed)
	}
}

func TestSleepContextWithNegativeDuration(t *testing.T) {
	ctx := context.Background()
	start := time.Now()
	err := SleepContext(ctx, -100*time.Millisecond)
	elapsed := time.Since(start)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if elapsed > 10*time.Millisecond {
		t.Errorf("expected immediate return, got %v delay", elapsed)
	}
}

func TestSleepContextMultipleCalls(t *testing.T) {
	ctx := context.Background()
	for i := 0; i < 3; i++ {
		err := SleepContext(ctx, 10*time.Millisecond)
		if err != nil {
			t.Errorf("expected nil on iteration %d, got %v", i, err)
		}
	}
}

func TestSleepContextShortDuration(t *testing.T) {
	ctx := context.Background()
	start := time.Now()
	err := SleepContext(ctx, 1*time.Millisecond)
	elapsed := time.Since(start)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if elapsed > 20*time.Millisecond {
		t.Errorf("expected quick completion, got %v", elapsed)
	}
}

func TestSleepContextWithDeadline(t *testing.T) {
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(30*time.Millisecond))
	defer cancel()
	start := time.Now()
	err := SleepContext(ctx, 1*time.Second)
	elapsed := time.Since(start)
	if err == nil {
		t.Errorf("expected error, got nil")
	}
	if elapsed > 100*time.Millisecond {
		t.Errorf("expected early return due to deadline, got %v", elapsed)
	}
}

func TestSleepContextReturnsNilAfterCompletion(t *testing.T) {
	ctx := context.Background()
	err := SleepContext(ctx, 20*time.Millisecond)
	if err != nil {
		t.Errorf("expected nil after successful sleep, got %v", err)
	}
}

func TestSleepContextImmediateCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(1 * time.Millisecond)
		cancel()
	}()
	start := time.Now()
	err := SleepContext(ctx, 1*time.Second)
	elapsed := time.Since(start)
	if err == nil {
		t.Errorf("expected error, got nil")
	}
	if elapsed > 50*time.Millisecond {
		t.Errorf("expected quick cancellation, got %v", elapsed)
	}
}`,
			hint1: `Use time.NewTimer(d) and select on both ctx.Done() and timer.C channels.`,
			hint2: `In defer, call timer.Stop() and drain timer.C if Stop() returns false.`,
			whyItMatters: `Context-aware sleep is essential for building responsive, cancellable operations in Go.

**The Problem with time.Sleep:**
\`\`\`go
// User clicks "Cancel" button in UI
ctx, cancel := context.WithCancel(context.Background())

go func() {
    for i := 0; i < 10; i++ {
        doWork()
        time.Sleep(1 * time.Second)  // ❌ Ignores cancellation!
    }
}()

// User cancels after 2 seconds
time.Sleep(2 * time.Second)
cancel()

// Worker continues for 8 more seconds! 😱
// User thinks request was canceled but it's still running
\`\`\`

**With SleepContext:**
\`\`\`go
go func() {
    for i := 0; i < 10; i++ {
        doWork()
        if err := SleepContext(ctx, 1*time.Second); err != nil {
            return  // ✅ Stops immediately on cancel
        }
    }
}()

// User cancels after 2 seconds
time.Sleep(2 * time.Second)
cancel()

// Worker stops within milliseconds! 🎉
\`\`\`

**Production Impact:**

**1. Request Timeout Compliance:**
\`\`\`go
// API handler with 5-second timeout
ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
defer cancel()

// Without SleepContext:
for _, item := range items {
    process(item)
    time.Sleep(100 * time.Millisecond)  // Total: 1000 items * 100ms = 100s
}
// Timeout violated by 95 seconds!

// With SleepContext:
for _, item := range items {
    process(item)
    if err := SleepContext(ctx, 100*time.Millisecond); err != nil {
        return err  // Stops at 5 seconds
    }
}
// Respects timeout, returns error immediately
\`\`\`

**2. Graceful Shutdown:**
\`\`\`go
// Background worker
func worker(ctx context.Context) {
    for {
        task := fetchTask()
        processTask(task)

        // Wait 1 minute before next task
        if err := SleepContext(ctx, 1*time.Minute); err != nil {
            log.Info("Shutdown signal received, stopping worker")
            return  // Clean shutdown within milliseconds
        }
    }
}

// Main
ctx, cancel := context.WithCancel(context.Background())
go worker(ctx)

// On SIGTERM/SIGINT:
cancel()  // Worker stops immediately, not after 1 minute
\`\`\`

**3. Resource Leak Prevention:**
\`\`\`go
// ❌ BAD: Goroutine leak
for i := 0; i < 1000; i++ {
    go func() {
        time.Sleep(1 * time.Hour)  // 1000 goroutines sleep for 1 hour!
    }()
}
// Memory usage: 1000 goroutines * 2KB = 2MB wasted

// ✅ GOOD: Clean cancellation
ctx, cancel := context.WithCancel(context.Background())
for i := 0; i < 1000; i++ {
    go func() {
        SleepContext(ctx, 1*time.Hour)  // All return immediately on cancel
    }()
}
cancel()  // All goroutines stop, memory freed
\`\`\`

**4. Timer Resource Management:**
\`\`\`go
// time.After creates timer that can't be stopped
select {
case <-ctx.Done():
    return ctx.Err()
case <-time.After(1 * time.Hour):  // ❌ Timer goroutine runs for 1 hour
    return nil
}

// time.NewTimer allows cleanup
timer := time.NewTimer(1 * time.Hour)
defer timer.Stop()  // ✅ Timer stopped, goroutine released
select {
case <-ctx.Done():
    return ctx.Err()
case <-timer.C:
    return nil
}
\`\`\`

**Real-World Examples:**

**Kubernetes Controller:**
\`\`\`go
// Reconciliation loop
for {
    if err := reconcile(); err != nil {
        log.Error("Reconciliation failed", "error", err)
    }
    // Wait before next reconciliation
    SleepContext(ctx, 30*time.Second)
}
// Stops immediately on pod termination
\`\`\`

**Database Connection Pool:**
\`\`\`go
// Health check loop
for {
    if err := db.Ping(); err != nil {
        metrics.RecordDBDown()
    }
    SleepContext(ctx, 10*time.Second)
}
// Stops on application shutdown
\`\`\`

**HTTP Client Retry:**
\`\`\`go
// Already implemented in Do() function!
if err := SleepContext(ctx, Backoff(attempt, base)); err != nil {
    return err  // Stop retrying if user canceled request
}
\`\`\`

**Performance Note:**
SleepContext is as efficient as time.Sleep (both use OS timers), but adds context awareness with negligible overhead (one select statement).

**Testing Benefit:**
\`\`\`go
// Easy to test with short timeout
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
defer cancel()

err := SleepContext(ctx, 1*time.Hour)
// Test completes in 10ms instead of 1 hour!
assert.Equal(t, context.DeadlineExceeded, err)
\``,	order: 4,
	translations: {
		ru: {
			title: 'Ожидание с контекстом',
			solutionCode: `package retryx

import (
	"context"
	"time"
)

func SleepContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {	// Сон не нужен для неположительных длительностей
		return nil
	}
	timer := time.NewTimer(d)	// Создаём таймер на запрошенную длительность
	defer func() {
		if !timer.Stop() {	// Останавливаем таймер и проверяем сработал ли он
			<-timer.C	// Освобождаем канал для предотвращения утечки goroutine
		}
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()	// Контекст отменён, возвращаем ошибку
	case <-timer.C:
		return nil	// Таймер сработал успешно
	}
}`,
			description: `Реализуйте функцию **SleepContext**, которая спит заданную длительность с учетом отмены контекста.

**Требования:**
1. Спите длительность \`d\`
2. Возвращайте сразу если контекст отменен
3. Возвращайте \`ctx.Err()\` если отменено во время сна
4. Возвращайте \`nil\` если сон завершен успешно
5. Обработайте d <= 0 возвращая сразу
6. Корректно освободите ресурсы таймера

**Почему не time.Sleep?**
\`\`\`go
// ❌ ПЛОХО: Игнорирует отмену контекста
time.Sleep(5 * time.Second)  // Блокирует на 5с даже если ctx отменен

// ✅ ХОРОШО: Учитывает контекст
SleepContext(ctx, 5*time.Second)  // Возвращается сразу при ctx.Done()
\`\`\`

**Управление ресурсами таймера:**
\`\`\`go
// Нужно очистить таймер чтобы предотвратить утечку goroutine
timer := time.NewTimer(d)
defer func() {
    if !timer.Stop() {  // Попытка остановить таймер
        <-timer.C       // Освободить канал если уже сработал
    }
}()
\`\`\`

**Пример:**
\`\`\`go
// Спать 1 секунду если не отменено
ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
defer cancel()

err := SleepContext(ctx, 1*time.Second)
// Возвращается через 500мс с ctx.Err() (timeout)

// Сон 0 длительности
err := SleepContext(ctx, 0)
// Возвращается сразу с nil
\`\`\`

**Паттерн Select:**
\`\`\`go
select {
case <-ctx.Done():
    // Контекст отменен до срабатывания таймера
    return ctx.Err()
case <-timer.C:
    // Таймер сработал успешно
    return nil
}
\`\`\`

**Use Cases:**
\`\`\`go
// 1. Rate limiting с отменой
func rateLimit(ctx context.Context) error {
    return SleepContext(ctx, time.Second/requestsPerSecond)
}

// 2. Polling с timeout
for {
    if done := checkCondition(); done {
        break
    }
    if err := SleepContext(ctx, 100*time.Millisecond); err != nil {
        return err  // Контекст отменен
    }
}

// 3. Retry с backoff (используется функцией Do)
if err := SleepContext(ctx, Backoff(attempt, base)); err != nil {
    return err  // Остановить retry если контекст отменен
}
`,
			hint1: `Используйте time.NewTimer(d) и select на каналах ctx.Done() и timer.C.`,
			hint2: `В defer вызовите timer.Stop() и освободите timer.C если Stop() вернул false.`,
			whyItMatters: `Context-aware sleep важен для построения отзывчивых, отменяемых операций в Go.

**Проблема с time.Sleep:**
\`\`\`go
// Пользователь нажимает кнопку "Отмена" в UI
ctx, cancel := context.WithCancel(context.Background())

go func() {
    for i := 0; i < 10; i++ {
        doWork()
        time.Sleep(1 * time.Second)  // ❌ Игнорирует отмену!
    }
}()

// Пользователь отменяет через 2 секунды
time.Sleep(2 * time.Second)
cancel()

// Worker продолжает еще 8 секунд! 😱
// Пользователь думает запрос отменен, но он все еще выполняется
\`\`\`

**С SleepContext:**
\`\`\`go
go func() {
    for i := 0; i < 10; i++ {
        doWork()
        if err := SleepContext(ctx, 1*time.Second); err != nil {
            return  // ✅ Останавливается сразу при отмене
        }
    }
}()

// Пользователь отменяет через 2 секунды
time.Sleep(2 * time.Second)
cancel()

// Worker останавливается в течение миллисекунд! 🎉
\`\`\`

**Production влияние:**

**1. Соблюдение Request Timeout:**
\`\`\`go
// API handler с 5-секундным timeout
ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
defer cancel()

// Без SleepContext:
for _, item := range items {
    process(item)
    time.Sleep(100 * time.Millisecond)  // Всего: 1000 items * 100ms = 100s
}
// Timeout нарушен на 95 секунд!

// С SleepContext:
for _, item := range items {
    process(item)
    if err := SleepContext(ctx, 100*time.Millisecond); err != nil {
        return err  // Останавливается на 5 секундах
    }
}
// Учитывает timeout, возвращает ошибку сразу
\`\`\`

**2. Graceful Shutdown:**
\`\`\`go
// Background worker
func worker(ctx context.Context) {
    for {
        task := fetchTask()
        processTask(task)

        // Ждать 1 минуту перед следующей задачей
        if err := SleepContext(ctx, 1*time.Minute); err != nil {
            log.Info("Получен сигнал остановки, останавливаем worker")
            return  // Чистая остановка в течение миллисекунд
        }
    }
}

// Main
ctx, cancel := context.WithCancel(context.Background())
go worker(ctx)

// При SIGTERM/SIGINT:
cancel()  // Worker останавливается сразу, не через 1 минуту
\`\`\`

**3. Предотвращение утечки ресурсов:**
\`\`\`go
// ❌ ПЛОХО: Утечка goroutine
for i := 0; i < 1000; i++ {
    go func() {
        time.Sleep(1 * time.Hour)  // 1000 goroutines спят 1 час!
    }()
}
// Использование памяти: 1000 goroutines * 2KB = 2MB потрачено впустую

// ✅ ХОРОШО: Чистая отмена
ctx, cancel := context.WithCancel(context.Background())
for i := 0; i < 1000; i++ {
    go func() {
        SleepContext(ctx, 1*time.Hour)  // Все возвращаются сразу при отмене
    }()
}
cancel()  // Все goroutines останавливаются, память освобождена
\`\`\`

**4. Управление ресурсами Timer:**
\`\`\`go
// time.After создает таймер который нельзя остановить
select {
case <-ctx.Done():
    return ctx.Err()
case <-time.After(1 * time.Hour):  // ❌ Goroutine таймера работает 1 час
    return nil
}

// time.NewTimer позволяет cleanup
timer := time.NewTimer(1 * time.Hour)
defer timer.Stop()  // ✅ Таймер остановлен, goroutine освобождена
select {
case <-ctx.Done():
    return ctx.Err()
case <-timer.C:
    return nil
}
\`\`\`

**Реальные примеры:**

**Kubernetes Controller:**
\`\`\`go
// Reconciliation loop
for {
    if err := reconcile(); err != nil {
        log.Error("Reconciliation failed", "error", err)
    }
    // Ждать перед следующей reconciliation
    SleepContext(ctx, 30*time.Second)
}
// Останавливается сразу при завершении pod
\`\`\`

**Database Connection Pool:**
\`\`\`go
// Health check loop
for {
    if err := db.Ping(); err != nil {
        metrics.RecordDBDown()
    }
    SleepContext(ctx, 10*time.Second)
}
// Останавливается при остановке приложения
\`\`\`

**HTTP Client Retry:**
\`\`\`go
// Уже реализовано в функции Do()!
if err := SleepContext(ctx, Backoff(attempt, base)); err != nil {
    return err  // Останавливает retry если пользователь отменил запрос
}
\`\`\`

**Замечание по производительности:**
SleepContext так же эффективен как time.Sleep (оба используют OS таймеры), но добавляет context awareness с незначительными накладными расходами (одна select инструкция).

**Преимущество для тестирования:**
\`\`\`go
// Легко тестировать с коротким timeout
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
defer cancel()

err := SleepContext(ctx, 1*time.Hour)
// Тест завершается за 10мс вместо 1 часа!
assert.Equal(t, context.DeadlineExceeded, err)
\`\`\``
		},
		uz: {
			title: `Kontekst bilan kutish`,
			solutionCode: `package retryx

import (
	"context"
	"time"
)

func SleepContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {	// Ijobiy bo'lmagan davomiyliklar uchun uxlash kerak emas
		return nil
	}
	timer := time.NewTimer(d)	// So'ralgan davomiylik uchun taymer yaratamiz
	defer func() {
		if !timer.Stop() {	// Taymerni to'xtatamiz va allaqachon ishga tushganini tekshiramiz
			<-timer.C	// Goroutine oqishini oldini olish uchun kanalni bo'shatamiz
		}
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()	// Kontekst bekor qilindi, xato qaytaramiz
	case <-timer.C:
		return nil	// Taymer muvaffaqiyatli ishga tushdi
	}
}`,
			description: `Kontekst bekor qilishni hurmat qilib ma'lum davomiylik uxlaydigan **SleepContext** funksiyasini amalga oshiring.

**Talablar:**
1. \`d\` davomiylik uxlang
2. Kontekst bekor qilinsa darhol qaytaring
3. Uxlash paytida bekor qilinsa \`ctx.Err()\` qaytaring
4. Uxlash muvaffaqiyatli tugasa \`nil\` qaytaring
5. d <= 0 uchun darhol qaytarish orqali qayta ishlang
6. Taymer resurslarini to'g'ri tozalang

**Nega time.Sleep emas?**
\`\`\`go
// ❌ YOMON: Kontekst bekor qilishni e'tiborsiz qoldiradi
time.Sleep(5 * time.Second)  // ctx bekor qilinsa ham to'liq 5s bloklaydi

// ✅ YAXSHI: Kontekstni hurmat qiladi
SleepContext(ctx, 5*time.Second)  // ctx.Done() da darhol qaytadi
\`\`\`

**Taymer resurslarini boshqarish:**
\`\`\`go
// Goroutine oqishini oldini olish uchun taymerni tozalash kerak
timer := time.NewTimer(d)
defer func() {
    if !timer.Stop() {  // Taymerni to'xtatishga urinish
        <-timer.C       // Agar allaqachon ishga tushgan bo'lsa kanalni bo'shatish
    }
}()
\`\`\`

**Select bayonoti patterni:**
\`\`\`go
select {
case <-ctx.Done():
    // Taymer ishga tushishidan oldin kontekst bekor qilindi
    return ctx.Err()
case <-timer.C:
    // Taymer muvaffaqiyatli ishga tushdi
    return nil
}
\`\`\`

**Cheklovlar:**
- time.After emas, time.NewTimer ishlating (yaxshiroq resurs boshqaruvi)
- Barcha kod yo'llarida taymerni tozalang
- Nol/salbiy davomiyliklarni to'g'ri qayta ishlang`,
			hint1: `time.NewTimer(d) ishlating va ctx.Done() va timer.C kanallarida select qiling.`,
			hint2: `defer da timer.Stop() ni chaqiring va Stop() false qaytarsa timer.C ni bo'shating.`,
			whyItMatters: `Kontekstni hisobga olgan sleep Go da javob beradigan, bekor qilinadigan operatsiyalarni qurish uchun muhimdir.

**time.Sleep bilan muammo:**
\`\`\`go
// Foydalanuvchi UI da "Bekor qilish" tugmasini bosadi
ctx, cancel := context.WithCancel(context.Background())

go func() {
    for i := 0; i < 10; i++ {
        doWork()
        time.Sleep(1 * time.Second)  // ❌ Bekor qilishni e'tiborsiz qoldiradi!
    }
}()

// Foydalanuvchi 2 soniyadan keyin bekor qiladi
time.Sleep(2 * time.Second)
cancel()

// Worker yana 8 soniya davom etadi! 😱
// Foydalanuvchi so'rov bekor qilindi deb o'ylaydi, lekin u hali ham bajarilmoqda
\`\`\`

**SleepContext bilan:**
\`\`\`go
go func() {
    for i := 0; i < 10; i++ {
        doWork()
        if err := SleepContext(ctx, 1*time.Second); err != nil {
            return  // ✅ Bekor qilishda darhol to'xtaydi
        }
    }
}()

// Foydalanuvchi 2 soniyadan keyin bekor qiladi
time.Sleep(2 * time.Second)
cancel()

// Worker millisekundlar ichida to'xtaydi! 🎉
\`\`\`

**Production ta'siri:**

**1. Request Timeout ga rioya qilish:**
\`\`\`go
// 5 soniyalik timeout bilan API handler
ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
defer cancel()

// SleepContext siz:
for _, item := range items {
    process(item)
    time.Sleep(100 * time.Millisecond)  // Jami: 1000 ta element * 100ms = 100s
}
// Timeout 95 soniyaga buzildi!

// SleepContext bilan:
for _, item := range items {
    process(item)
    if err := SleepContext(ctx, 100*time.Millisecond); err != nil {
        return err  // 5 soniyada to'xtaydi
    }
}
// Timeout ni hurmat qiladi, darhol xato qaytaradi
\`\`\`

**2. Graceful Shutdown:**
\`\`\`go
// Background worker
func worker(ctx context.Context) {
    for {
        task := fetchTask()
        processTask(task)

        // Keyingi vazifadan oldin 1 daqiqa kutish
        if err := SleepContext(ctx, 1*time.Minute); err != nil {
            log.Info("To'xtatish signali qabul qilindi, workerni to'xtatish")
            return  // Millisekundlar ichida toza to'xtatish
        }
    }
}

// Main
ctx, cancel := context.WithCancel(context.Background())
go worker(ctx)

// SIGTERM/SIGINT da:
cancel()  // Worker darhol to'xtaydi, 1 daqiqadan keyin emas
\`\`\`

**3. Resurs oqishini oldini olish:**
\`\`\`go
// ❌ YOMON: Goroutine oqishi
for i := 0; i < 1000; i++ {
    go func() {
        time.Sleep(1 * time.Hour)  // 1000 goroutine 1 soat uxlaydi!
    }()
}
// Xotira foydalanishi: 1000 goroutine * 2KB = 2MB behuda sarflandi

// ✅ YAXSHI: Toza bekor qilish
ctx, cancel := context.WithCancel(context.Background())
for i := 0; i < 1000; i++ {
    go func() {
        SleepContext(ctx, 1*time.Hour)  // Hammasi bekor qilishda darhol qaytadi
    }()
}
cancel()  // Barcha goroutinelar to'xtaydi, xotira bo'shatiladi
\`\`\`

**4. Taymer resurslarini boshqarish:**
\`\`\`go
// time.After to'xtatib bo'lmaydigan taymer yaratadi
select {
case <-ctx.Done():
    return ctx.Err()
case <-time.After(1 * time.Hour):  // ❌ Taymer goroutine 1 soat ishlaydi
    return nil
}

// time.NewTimer tozalashga imkon beradi
timer := time.NewTimer(1 * time.Hour)
defer timer.Stop()  // ✅ Taymer to'xtatildi, goroutine bo'shatildi
select {
case <-ctx.Done():
    return ctx.Err()
case <-timer.C:
    return nil
}
\`\`\`

**Haqiqiy misollar:**

**Kubernetes Controller:**
\`\`\`go
// Reconciliation loop
for {
    if err := reconcile(); err != nil {
        log.Error("Reconciliation failed", "error", err)
    }
    // Keyingi reconciliation dan oldin kutish
    SleepContext(ctx, 30*time.Second)
}
// Pod tugaganda darhol to'xtaydi
\`\`\`

**Database Connection Pool:**
\`\`\`go
// Health check loop
for {
    if err := db.Ping(); err != nil {
        metrics.RecordDBDown()
    }
    SleepContext(ctx, 10*time.Second)
}
// Dastur to'xtaganda to'xtaydi
\`\`\`

**HTTP Client Retry:**
\`\`\`go
// Do() funksiyasida allaqachon amalga oshirilgan!
if err := SleepContext(ctx, Backoff(attempt, base)); err != nil {
    return err  // Foydalanuvchi so'rovni bekor qilsa retry ni to'xtatadi
}
\`\`\`

**Samaradorlik eslatmasi:**
SleepContext time.Sleep kabi samarali (ikkalasi ham OS taymerlaridan foydalanadi), lekin ahamiyatsiz qo'shimcha xarajatlar bilan kontekst xabardorligini qo'shadi (bitta select bayonoti).

**Test qilish foydalari:**
\`\`\`go
// Qisqa timeout bilan osongina test qilish
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
defer cancel()

err := SleepContext(ctx, 1*time.Hour)
// Test 1 soat o'rniga 10ms da tugaydi!
assert.Equal(t, context.DeadlineExceeded, err)
\`\`\``
		}
	}
};

export default task;
