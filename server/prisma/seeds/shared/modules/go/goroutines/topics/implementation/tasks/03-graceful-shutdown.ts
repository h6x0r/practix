import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-goroutines-graceful-shutdown',
	title: 'Graceful Shutdown Coordinator',
	difficulty: 'medium',
	tags: ['go', 'goroutines', 'context', 'shutdown'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a shutdown coordinator that waits for all goroutines to complete gracefully before exiting.

**Requirements:**
1. **ShutdownCoordinator**: Track multiple goroutines and wait for completion
2. **Done Channel**: Provide a channel that closes when all goroutines finish
3. **Context Propagation**: Pass context to all tracked goroutines
4. **Timeout Support**: Return error if shutdown exceeds timeout

**Implementation Pattern:**
\`\`\`go
type ShutdownCoordinator struct {
    wg  sync.WaitGroup
    ctx context.Context
}

func NewShutdownCoordinator(ctx context.Context) *ShutdownCoordinator {
    return &ShutdownCoordinator{ctx: ctx}
}

func (sc *ShutdownCoordinator) Go(fn func(context.Context)) {
    sc.wg.Add(1)
    go func() {
        defer sc.wg.Done()
        fn(sc.ctx)
    }()
}

func (sc *ShutdownCoordinator) Wait(timeout time.Duration) error {
    done := make(chan struct{})
    go func() {
        sc.wg.Wait()
        close(done)
    }()

    select {
    case <-done:
        return nil
    case <-time.After(timeout):
        return fmt.Errorf("shutdown timeout exceeded")
    }
}
\`\`\`

**Example Usage:**
\`\`\`go
// Start multiple workers with graceful shutdown
ctx, cancel := context.WithCancel(context.Background())
sc := NewShutdownCoordinator(ctx)

// Start workers
for i := 0; i < 5; i++ {
    sc.Go(func(ctx context.Context) {
        processJobs(ctx) // respects context cancellation
    })
}

// Initiate shutdown
cancel()

// Wait for all workers to finish (10 second timeout)
if err := sc.Wait(10 * time.Second); err != nil {
    log.Printf("Forced shutdown: %v", err)
}
\`\`\`

**Constraints:**
- Must use sync.WaitGroup internally
- Must support adding goroutines dynamically
- Must respect timeout in Wait method
- Must propagate context to all goroutines`,
	initialCode: `package goroutinesx

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type ShutdownCoordinator struct {
	// TODO: Add fields for tracking goroutines
}

// TODO: Implement NewShutdownCoordinator
// Initialize coordinator with context
func NewShutdownCoordinator(ctx context.Context) *ShutdownCoordinator {
	// TODO: Implement
}

// TODO: Implement Go
// Add goroutine to tracking and execute with context
func (sc *ShutdownCoordinator) Go(fn func(context.Context)) {
	// TODO: Implement
}

// TODO: Implement Wait
// Wait for all goroutines to complete or timeout
// Return error if timeout exceeded
func (sc *ShutdownCoordinator) Wait(timeout time.Duration) error {
	return nil // TODO: Implement
}`,
	solutionCode: `package goroutinesx

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type ShutdownCoordinator struct {
	wg  sync.WaitGroup
	ctx context.Context
}

func NewShutdownCoordinator(ctx context.Context) *ShutdownCoordinator {
	if ctx == nil {
		ctx = context.Background()	// default to background context
	}
	return &ShutdownCoordinator{
		ctx: ctx,
	}
}

func (sc *ShutdownCoordinator) Go(fn func(context.Context)) {
	sc.wg.Add(1)	// increment wait group counter
	go func() {
		defer sc.wg.Done()	// decrement counter when done
		fn(sc.ctx)	// execute function with coordinator context
	}()
}

func (sc *ShutdownCoordinator) Wait(timeout time.Duration) error {
	done := make(chan struct{})	// channel to signal completion

	go func() {
		sc.wg.Wait()	// wait for all goroutines
		close(done)	// signal completion
	}()

	select {
	case <-done:	// all goroutines completed
		return nil
	case <-time.After(timeout):	// timeout exceeded
		return fmt.Errorf("shutdown timeout exceeded: %v", timeout)
	}
}`,
	testCode: `package goroutinesx

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// NewShutdownCoordinator creates coordinator
	ctx := context.Background()
	sc := NewShutdownCoordinator(ctx)
	if sc == nil {
		t.Error("expected non-nil coordinator")
	}
}

func Test2(t *testing.T) {
	// Wait returns nil immediately with no goroutines
	ctx := context.Background()
	sc := NewShutdownCoordinator(ctx)

	err := sc.Wait(1*time.Second)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test3(t *testing.T) {
	// Go executes function
	ctx := context.Background()
	sc := NewShutdownCoordinator(ctx)

	var executed int32
	sc.Go(func(context.Context) {
		atomic.StoreInt32(&executed, 1)
	})

	sc.Wait(1*time.Second)

	if atomic.LoadInt32(&executed) != 1 {
		t.Error("expected function to execute")
	}
}

func Test4(t *testing.T) {
	// Wait waits for all goroutines
	ctx := context.Background()
	sc := NewShutdownCoordinator(ctx)

	var count int32
	for i := 0; i < 5; i++ {
		sc.Go(func(context.Context) {
			time.Sleep(50*time.Millisecond)
			atomic.AddInt32(&count, 1)
		})
	}

	sc.Wait(1*time.Second)

	if atomic.LoadInt32(&count) != 5 {
		t.Errorf("expected 5, got %d", atomic.LoadInt32(&count))
	}
}

func Test5(t *testing.T) {
	// Go passes context to function
	ctx, cancel := context.WithCancel(context.Background())
	sc := NewShutdownCoordinator(ctx)

	var gotContext bool
	sc.Go(func(ctx context.Context) {
		if ctx != nil {
			gotContext = true
		}
	})

	cancel()
	sc.Wait(1*time.Second)

	if !gotContext {
		t.Error("expected context to be passed")
	}
}

func Test6(t *testing.T) {
	// Wait returns error on timeout
	ctx := context.Background()
	sc := NewShutdownCoordinator(ctx)

	sc.Go(func(context.Context) {
		time.Sleep(500*time.Millisecond) // Long running task
	})

	err := sc.Wait(50*time.Millisecond)
	if err == nil {
		t.Error("expected timeout error")
	}
}

func Test7(t *testing.T) {
	// Multiple Go calls work correctly
	ctx := context.Background()
	sc := NewShutdownCoordinator(ctx)

	var results []int32
	for i := int32(0); i < 3; i++ {
		val := i
		sc.Go(func(context.Context) {
			results = append(results, val)
		})
	}

	sc.Wait(1*time.Second)

	if len(results) != 3 {
		t.Errorf("expected 3 results, got %d", len(results))
	}
}

func Test8(t *testing.T) {
	// Wait returns nil when all complete before timeout
	ctx := context.Background()
	sc := NewShutdownCoordinator(ctx)

	sc.Go(func(context.Context) {
		time.Sleep(10*time.Millisecond)
	})

	err := sc.Wait(1*time.Second)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test9(t *testing.T) {
	// NewShutdownCoordinator handles nil context
	sc := NewShutdownCoordinator(nil)
	if sc == nil {
		t.Error("expected non-nil coordinator with nil context")
	}

	sc.Go(func(ctx context.Context) {
		// Should work without panic
	})

	sc.Wait(1*time.Second)
}

func Test10(t *testing.T) {
	// Context cancellation propagates to goroutines
	ctx, cancel := context.WithCancel(context.Background())
	sc := NewShutdownCoordinator(ctx)

	var cancelled bool
	sc.Go(func(ctx context.Context) {
		select {
		case <-ctx.Done():
			cancelled = true
		case <-time.After(1*time.Second):
		}
	})

	cancel()
	sc.Wait(500*time.Millisecond)

	if !cancelled {
		t.Error("expected context cancellation")
	}
}
`,
	hint1: `Use sync.WaitGroup to track goroutines. In Go(), call wg.Add(1) before launching goroutine and wg.Done() in defer.`,
	hint2: `In Wait(), create a done channel and launch goroutine that calls wg.Wait() then closes the channel. Use select with time.After() for timeout.`,
	whyItMatters: `Graceful shutdown is critical for production services. It ensures in-flight requests complete, connections close cleanly, and data is not lost. This pattern prevents dropped requests and data corruption during deployments.`,
	order: 2,
	translations: {
		ru: {
			title: 'Корректное завершение',
			solutionCode: `package goroutinesx

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type ShutdownCoordinator struct {
	wg  sync.WaitGroup
	ctx context.Context
}

func NewShutdownCoordinator(ctx context.Context) *ShutdownCoordinator {
	if ctx == nil {
		ctx = context.Background()	// по умолчанию используем background контекст
	}
	return &ShutdownCoordinator{
		ctx: ctx,
	}
}

func (sc *ShutdownCoordinator) Go(fn func(context.Context)) {
	sc.wg.Add(1)	// увеличиваем счётчик wait group
	go func() {
		defer sc.wg.Done()	// уменьшаем счётчик при завершении
		fn(sc.ctx)	// выполняем функцию с контекстом координатора
	}()
}

func (sc *ShutdownCoordinator) Wait(timeout time.Duration) error {
	done := make(chan struct{})	// канал для сигнала о завершении

	go func() {
		sc.wg.Wait()	// ждём все горутины
		close(done)	// сигнализируем о завершении
	}()

	select {
	case <-done:	// все горутины завершены
		return nil
	case <-time.After(timeout):	// превышен таймаут
		return fmt.Errorf("shutdown timeout exceeded: %v", timeout)
	}
}`,
			description: `Реализуйте координатор завершения который ждёт изящного завершения всех горутин перед выходом.

**Требования:**
1. **ShutdownCoordinator**: Отслеживайте несколько горутин и ждите их завершения
2. **Done канал**: Предоставьте канал который закрывается когда все горутины завершены
3. **Распространение контекста**: Передайте контекст всем отслеживаемым горутинам
4. **Поддержка таймаута**: Верните ошибку если завершение превышает таймаут

**Паттерн реализации:**
\`\`\`go
type ShutdownCoordinator struct {
    wg  sync.WaitGroup
    ctx context.Context
}

func NewShutdownCoordinator(ctx context.Context) *ShutdownCoordinator {
    return &ShutdownCoordinator{ctx: ctx}
}

func (sc *ShutdownCoordinator) Go(fn func(context.Context)) {
    sc.wg.Add(1)
    go func() {
        defer sc.wg.Done()
        fn(sc.ctx)
    }()
}

func (sc *ShutdownCoordinator) Wait(timeout time.Duration) error {
    done := make(chan struct{})
    go func() {
        sc.wg.Wait()
        close(done)
    }()

    select {
    case <-done:
        return nil
    case <-time.After(timeout):
        return fmt.Errorf("shutdown timeout exceeded")
    }
}
\`\`\`

**Пример использования:**
\`\`\`go
// Запуск нескольких воркеров с изящным завершением
ctx, cancel := context.WithCancel(context.Background())
sc := NewShutdownCoordinator(ctx)

// Запуск воркеров
for i := 0; i < 5; i++ {
    sc.Go(func(ctx context.Context) {
        processJobs(ctx) // уважает отмену контекста
    })
}

// Инициация завершения
cancel()

// Ожидание завершения всех воркеров (таймаут 10 секунд)
if err := sc.Wait(10 * time.Second); err != nil {
    log.Printf("Forced shutdown: %v", err)
}
\`\`\`

**Ограничения:**
- Должен использовать sync.WaitGroup внутри
- Должен поддерживать динамическое добавление горутин
- Должен уважать таймаут в методе Wait
- Должен распространять контекст на все горутины`,
			hint1: `Используйте sync.WaitGroup для отслеживания горутин. В Go() вызовите wg.Add(1) перед запуском горутины и wg.Done() в defer.`,
			hint2: `В Wait() создайте канал done и запустите горутину которая вызывает wg.Wait() затем закрывает канал. Используйте select с time.After() для таймаута.`,
			whyItMatters: `Изящное завершение критично для production сервисов. Оно гарантирует завершение обрабатываемых запросов, чистое закрытие соединений и отсутствие потери данных. Этот паттерн предотвращает потерянные запросы и повреждение данных при развёртываниях.`
		},
		uz: {
			title: `Graceful shutdown`,
			solutionCode: `package goroutinesx

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type ShutdownCoordinator struct {
	wg  sync.WaitGroup
	ctx context.Context
}

func NewShutdownCoordinator(ctx context.Context) *ShutdownCoordinator {
	if ctx == nil {
		ctx = context.Background()	// default background kontekstdan foydalanamiz
	}
	return &ShutdownCoordinator{
		ctx: ctx,
	}
}

func (sc *ShutdownCoordinator) Go(fn func(context.Context)) {
	sc.wg.Add(1)	// wait group hisoblagichini oshiramiz
	go func() {
		defer sc.wg.Done()	// tugatilganda hisoblagichni kamaytiramiz
		fn(sc.ctx)	// koordinator konteksti bilan funktsiyani bajaramiz
	}()
}

func (sc *ShutdownCoordinator) Wait(timeout time.Duration) error {
	done := make(chan struct{})	// tugatish signali uchun kanal

	go func() {
		sc.wg.Wait()	// barcha goroutinelarni kutamiz
		close(done)	// tugatish haqida signal beramiz
	}()

	select {
	case <-done:	// barcha goroutinelar tugadi
		return nil
	case <-time.After(timeout):	// timeout oshib ketdi
		return fmt.Errorf("shutdown timeout exceeded: %v", timeout)
	}
}`,
			description: `Chiqishdan oldin barcha goroutinelarning nozik tugashini kutadigan tugatish koordinatorini amalga oshiring.

**Talablar:**
1. **ShutdownCoordinator**: Bir nechta goroutinelarni kuzatish va tugashini kutish
2. **Done kanali**: Barcha goroutinelar tugaganda yopiladigan kanal taqdim etish
3. **Kontekst tarqatish**: Barcha kuzatilgan goroutinelarga kontekstni uzatish
4. **Timeout qo'llab-quvvatlash**: Tugatish timeoutdan oshsa xato qaytarish

**Amalga oshirish patterni:**
\`\`\`go
type ShutdownCoordinator struct {
    wg  sync.WaitGroup
    ctx context.Context
}

func NewShutdownCoordinator(ctx context.Context) *ShutdownCoordinator {
    return &ShutdownCoordinator{ctx: ctx}
}

func (sc *ShutdownCoordinator) Go(fn func(context.Context)) {
    sc.wg.Add(1)
    go func() {
        defer sc.wg.Done()
        fn(sc.ctx)
    }()
}

func (sc *ShutdownCoordinator) Wait(timeout time.Duration) error {
    done := make(chan struct{})
    go func() {
        sc.wg.Wait()
        close(done)
    }()

    select {
    case <-done:
        return nil
    case <-time.After(timeout):
        return fmt.Errorf("shutdown timeout exceeded")
    }
}
\`\`\`

**Cheklovlar:**
- Ichida sync.WaitGroup ishlatish kerak
- Goroutinelarni dinamik qo'shishni qo'llab-quvvatlash kerak
- Wait metodida timeoutga rioya qilish kerak
- Barcha goroutinelarga kontekstni tarqatish kerak`,
			hint1: `Goroutinelarni kuzatish uchun sync.WaitGroup ishlating. Go() da goroutine ishga tushirishdan oldin wg.Add(1) ni chaqiring va defer da wg.Done() ni chaqiring.`,
			hint2: `Wait() da done kanalini yarating va wg.Wait() ni chaqiruvchi goroutine ishga tushiring, keyin kanalni yoping. Timeout uchun time.After() bilan select ishlating.`,
			whyItMatters: `Nozik tugatish production xizmatlar uchun muhimdir. U jarayonda turgan so'rovlarning tugashini, ulanishlarning tozalanib yopilishini va ma'lumotlar yo'qotilmasligini ta'minlaydi. Bu pattern deployment paytida so'rovlar va ma'lumotlarning yo'qolishini oldini oladi.`
		}
	}
};

export default task;
