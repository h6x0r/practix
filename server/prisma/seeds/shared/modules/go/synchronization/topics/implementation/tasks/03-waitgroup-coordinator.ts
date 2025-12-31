import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-sync-waitgroup-coordinator',
	title: 'WaitGroup Coordinator for Parallel Tasks',
	difficulty: 'easy',
	tags: ['go', 'sync', 'waitgroup', 'goroutines'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a task coordinator using sync.WaitGroup to manage parallel goroutines.

**Requirements:**
1. **TaskCoordinator**: Manage multiple parallel workers
2. **AddWorker**: Start goroutine and track with WaitGroup
3. **WaitAll**: Block until all workers complete
4. **Error Handling**: Collect errors from workers safely

**Implementation Pattern:**
\`\`\`go
type TaskCoordinator struct {
    wg     sync.WaitGroup
    mu     sync.Mutex
    errors []error
}

func (tc *TaskCoordinator) AddWorker(fn func() error) {
    tc.wg.Add(1)
    go func() {
        defer tc.wg.Done()
        if err := fn(); err != nil {
            tc.mu.Lock()
            tc.errors = append(tc.errors, err)
            tc.mu.Unlock()
        }
    }()
}

func (tc *TaskCoordinator) WaitAll() []error {
    tc.wg.Wait()
    return tc.errors
}
\`\`\`

**Example Usage:**
\`\`\`go
coordinator := NewTaskCoordinator()

// Start parallel workers
for i := 0; i < 10; i++ {
    id := i
    coordinator.AddWorker(func() error {
        return processTask(id)
    })
}

// Wait for all to complete
errors := coordinator.WaitAll()
if len(errors) > 0 {
    fmt.Printf("Tasks completed with %d errors\n", len(errors))
}
\`\`\`

**Constraints:**
- Must use sync.WaitGroup
- AddWorker must call Add(1) before goroutine
- Goroutine must defer Done()
- Error collection must be thread-safe`,
	initialCode: `package syncx

import "sync"

type TaskCoordinator struct {
	wg     sync.WaitGroup
	mu     sync.Mutex
	errors []error
}

func NewTaskCoordinator() *TaskCoordinator {
	return &TaskCoordinator{
		errors: make([]error, 0),
	}
}

// TODO: Implement AddWorker
// Add(1) to WaitGroup, start goroutine with defer Done()
// Execute fn and collect error if any
func (tc *TaskCoordinator) AddWorker(fn func() error) {
	// TODO: Implement
}

// TODO: Implement WaitAll
// Wait for all goroutines, return collected errors
func (tc *TaskCoordinator) WaitAll() []error {
	// TODO: Implement
}`,
	solutionCode: `package syncx

import "sync"

type TaskCoordinator struct {
	wg     sync.WaitGroup
	mu     sync.Mutex
	errors []error
}

func NewTaskCoordinator() *TaskCoordinator {
	return &TaskCoordinator{
		errors: make([]error, 0),
	}
}

func (tc *TaskCoordinator) AddWorker(fn func() error) {
	tc.wg.Add(1)	// increment counter before goroutine
	go func() {
		defer tc.wg.Done()	// decrement counter when goroutine completes
		if err := fn(); err != nil {	// execute worker function
			tc.mu.Lock()	// protect shared errors slice
			tc.errors = append(tc.errors, err)
			tc.mu.Unlock()
		}
	}()
}

func (tc *TaskCoordinator) WaitAll() []error {
	tc.wg.Wait()	// block until all Done() calls match Add() calls
	return tc.errors	// safe to return after all goroutines finish
}`,
	testCode: `package syncx

import (
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestTaskCoordinatorNoWorkers(t *testing.T) {
	tc := NewTaskCoordinator()
	errs := tc.WaitAll()
	if len(errs) != 0 {
		t.Errorf("expected 0 errors, got %d", len(errs))
	}
}

func TestTaskCoordinatorSingleWorkerSuccess(t *testing.T) {
	tc := NewTaskCoordinator()
	executed := false
	tc.AddWorker(func() error {
		executed = true
		return nil
	})
	errs := tc.WaitAll()
	if !executed {
		t.Errorf("worker was not executed")
	}
	if len(errs) != 0 {
		t.Errorf("expected 0 errors, got %d", len(errs))
	}
}

func TestTaskCoordinatorSingleWorkerError(t *testing.T) {
	tc := NewTaskCoordinator()
	expectedErr := errors.New("worker error")
	tc.AddWorker(func() error {
		return expectedErr
	})
	errs := tc.WaitAll()
	if len(errs) != 1 {
		t.Errorf("expected 1 error, got %d", len(errs))
	}
	if errs[0] != expectedErr {
		t.Errorf("expected %v, got %v", expectedErr, errs[0])
	}
}

func TestTaskCoordinatorMultipleWorkersSuccess(t *testing.T) {
	tc := NewTaskCoordinator()
	var counter int32
	for i := 0; i < 10; i++ {
		tc.AddWorker(func() error {
			atomic.AddInt32(&counter, 1)
			return nil
		})
	}
	errs := tc.WaitAll()
	if counter != 10 {
		t.Errorf("expected counter=10, got %d", counter)
	}
	if len(errs) != 0 {
		t.Errorf("expected 0 errors, got %d", len(errs))
	}
}

func TestTaskCoordinatorMultipleWorkersWithErrors(t *testing.T) {
	tc := NewTaskCoordinator()
	for i := 0; i < 5; i++ {
		tc.AddWorker(func() error {
			return errors.New("error")
		})
	}
	errs := tc.WaitAll()
	if len(errs) != 5 {
		t.Errorf("expected 5 errors, got %d", len(errs))
	}
}

func TestTaskCoordinatorMixedSuccessAndErrors(t *testing.T) {
	tc := NewTaskCoordinator()
	tc.AddWorker(func() error { return nil })
	tc.AddWorker(func() error { return errors.New("error1") })
	tc.AddWorker(func() error { return nil })
	tc.AddWorker(func() error { return errors.New("error2") })
	errs := tc.WaitAll()
	if len(errs) != 2 {
		t.Errorf("expected 2 errors, got %d", len(errs))
	}
}

func TestTaskCoordinatorWorkerDelay(t *testing.T) {
	tc := NewTaskCoordinator()
	start := time.Now()
	tc.AddWorker(func() error {
		time.Sleep(50 * time.Millisecond)
		return nil
	})
	tc.WaitAll()
	elapsed := time.Since(start)
	if elapsed < 40*time.Millisecond {
		t.Errorf("expected to wait at least 50ms, got %v", elapsed)
	}
}

func TestTaskCoordinatorConcurrentErrorCollection(t *testing.T) {
	tc := NewTaskCoordinator()
	n := 100
	for i := 0; i < n; i++ {
		tc.AddWorker(func() error {
			return errors.New("concurrent error")
		})
	}
	errs := tc.WaitAll()
	if len(errs) != n {
		t.Errorf("expected %d errors, got %d", n, len(errs))
	}
}

func TestTaskCoordinatorReuse(t *testing.T) {
	tc := NewTaskCoordinator()
	tc.AddWorker(func() error { return nil })
	errs1 := tc.WaitAll()
	if len(errs1) != 0 {
		t.Errorf("expected 0 errors on first use, got %d", len(errs1))
	}

	tc.AddWorker(func() error { return errors.New("error") })
	errs2 := tc.WaitAll()
	if len(errs2) != 1 {
		t.Errorf("expected 1 error on second use, got %d", len(errs2))
	}
}

func TestTaskCoordinatorAllWorkersComplete(t *testing.T) {
	tc := NewTaskCoordinator()
	var completed int32
	workers := 50
	for i := 0; i < workers; i++ {
		tc.AddWorker(func() error {
			time.Sleep(10 * time.Millisecond)
			atomic.AddInt32(&completed, 1)
			return nil
		})
	}
	tc.WaitAll()
	if completed != int32(workers) {
		t.Errorf("expected %d workers to complete, got %d", workers, completed)
	}
}`,
	hint1: `In AddWorker: Call tc.wg.Add(1), then start goroutine with defer tc.wg.Done(). Execute fn() and collect error with mutex protection.`,
	hint2: `In WaitAll: Simply call tc.wg.Wait() to block until all workers complete, then return tc.errors.`,
	whyItMatters: `WaitGroup is the standard way to wait for multiple goroutines to complete in Go. It's essential for parallel processing, fan-out/fan-in patterns, and coordinating concurrent work. Understanding WaitGroup prevents common bugs like premature program termination or goroutine leaks.

**Production Pattern:**

\`\`\`go
type TaskCoordinator struct {
    wg     sync.WaitGroup
    mu     sync.Mutex
    errors []error
}

func (tc *TaskCoordinator) AddWorker(fn func() error) {
    tc.wg.Add(1)                   // increment counter before goroutine
    go func() {
        defer tc.wg.Done()          // decrement counter when goroutine completes
        if err := fn(); err != nil { // execute worker function
            tc.mu.Lock()             // protect shared errors slice
            tc.errors = append(tc.errors, err)
            tc.mu.Unlock()
        }
    }()
}

func (tc *TaskCoordinator) WaitAll() []error {
    tc.wg.Wait()                   // block until all Done() calls match Add() calls
    return tc.errors               // safe to return after all goroutines finish
}
\`\`\`

**Practical Benefits:**
- Standard way to coordinate multiple goroutines
- Prevents premature program termination
- Critical for parallel processing and fan-out/fan-in patterns
- Allows safe collection of results and errors
- Understanding prevents goroutine leaks`,
	order: 2,
	translations: {
		ru: {
			title: 'Координатор WaitGroup',
			solutionCode: `package syncx

import "sync"

type TaskCoordinator struct {
	wg     sync.WaitGroup
	mu     sync.Mutex
	errors []error
}

func NewTaskCoordinator() *TaskCoordinator {
	return &TaskCoordinator{
		errors: make([]error, 0),
	}
}

func (tc *TaskCoordinator) AddWorker(fn func() error) {
	tc.wg.Add(1)	// увеличиваем счетчик перед горутиной
	go func() {
		defer tc.wg.Done()	// уменьшаем счетчик когда горутина завершается
		if err := fn(); err != nil {	// выполняем функцию worker
			tc.mu.Lock()	// защищаем общий срез ошибок
			tc.errors = append(tc.errors, err)
			tc.mu.Unlock()
		}
	}()
}

func (tc *TaskCoordinator) WaitAll() []error {
	tc.wg.Wait()	// блокируемся пока все Done() не совпадут с Add()
	return tc.errors	// безопасно возвращать после завершения всех горутин
}`,
			description: `Реализуйте координатор задач используя sync.WaitGroup для управления параллельными горутинами.

**Требования:**
1. **TaskCoordinator**: Управление несколькими параллельными workers
2. **AddWorker**: Запуск горутины и отслеживание через WaitGroup
3. **WaitAll**: Блокировка до завершения всех workers
4. **Обработка ошибок**: Безопасный сбор ошибок от workers

**Паттерн реализации:**
\`\`\`go
type TaskCoordinator struct {
    wg     sync.WaitGroup
    mu     sync.Mutex
    errors []error
}

func (tc *TaskCoordinator) AddWorker(fn func() error) {
    tc.wg.Add(1)
    go func() {
        defer tc.wg.Done()
        if err := fn(); err != nil {
            tc.mu.Lock()
            tc.errors = append(tc.errors, err)
            tc.mu.Unlock()
        }
    }()
}
\`\`\`

**Ограничения:**
- Должен использовать sync.WaitGroup
- AddWorker должен вызывать Add(1) перед горутиной
- Горутина должна использовать defer Done()
- Сбор ошибок должен быть потокобезопасным`,
			hint1: `В AddWorker: Вызовите tc.wg.Add(1), затем запустите горутину с defer tc.wg.Done(). Выполните fn() и соберите ошибку с защитой mutex.`,
			hint2: `В WaitAll: Просто вызовите tc.wg.Wait() для блокировки до завершения всех workers, затем верните tc.errors.`,
			whyItMatters: `WaitGroup - стандартный способ ожидания завершения множества горутин в Go. Это критично для параллельной обработки, fan-out/fan-in паттернов и координации конкурентной работы. Понимание WaitGroup предотвращает частые баги как преждевременное завершение программы или утечки горутин.

**Почему важно:**

**Продакшен паттерн:**
\`\`\`go
type TaskCoordinator struct {
    wg     sync.WaitGroup
    mu     sync.Mutex
    errors []error
}

func (tc *TaskCoordinator) AddWorker(fn func() error) {
    tc.wg.Add(1)                   // увеличиваем счетчик перед горутиной
    go func() {
        defer tc.wg.Done()          // уменьшаем счетчик когда горутина завершается
        if err := fn(); err != nil { // выполняем функцию worker
            tc.mu.Lock()             // защищаем общий срез ошибок
            tc.errors = append(tc.errors, err)
            tc.mu.Unlock()
        }
    }()
}

func (tc *TaskCoordinator) WaitAll() []error {
    tc.wg.Wait()                   // блокируемся пока все Done() не совпадут с Add()
    return tc.errors               // безопасно возвращать после завершения всех горутин
}
\`\`\`

**Практические преимущества:**
- Стандартный способ координации множества горутин
- Предотвращает преждевременное завершение программы
- Критично для параллельной обработки и fan-out/fan-in паттернов
- Позволяет безопасный сбор результатов и ошибок
- Понимание предотвращает утечки горутин`
		},
		uz: {
			title: `WaitGroup koordinatori`,
			solutionCode: `package syncx

import "sync"

type TaskCoordinator struct {
	wg     sync.WaitGroup
	mu     sync.Mutex
	errors []error
}

func NewTaskCoordinator() *TaskCoordinator {
	return &TaskCoordinator{
		errors: make([]error, 0),
	}
}

func (tc *TaskCoordinator) AddWorker(fn func() error) {
	tc.wg.Add(1)	// goroutine dan oldin hisoblagichni oshiramiz
	go func() {
		defer tc.wg.Done()	// goroutine tugaganda hisoblagichni kamaytiramiz
		if err := fn(); err != nil {	// worker funksiyasini bajaramiz
			tc.mu.Lock()	// umumiy xatolar slice ni himoyalaymiz
			tc.errors = append(tc.errors, err)
			tc.mu.Unlock()
		}
	}()
}

func (tc *TaskCoordinator) WaitAll() []error {
	tc.wg.Wait()	// barcha Done() lar Add() ga mos kelguncha bloklaymiz
	return tc.errors	// barcha goroutine lar tugagandan keyin xavfsiz qaytaramiz
}`,
			description: `Parallel goroutine larni boshqarish uchun sync.WaitGroup ishlatib vazifa koordinatorini amalga oshiring.

**Talablar:**
1. **TaskCoordinator**: Bir nechta parallel worker larni boshqarish
2. **AddWorker**: Goroutine ni boshlash va WaitGroup bilan kuzatish
3. **WaitAll**: Barcha worker lar tugaguncha bloklash
4. **Xato Boshqaruvi**: Worker lardan xatolarni xavfsiz yig'ish

**Amalga oshirish patterni:**
\`\`\`go
type TaskCoordinator struct {
    wg     sync.WaitGroup
    mu     sync.Mutex
    errors []error
}

func (tc *TaskCoordinator) AddWorker(fn func() error) {
    tc.wg.Add(1)
    go func() {
        defer tc.wg.Done()
        if err := fn(); err != nil {
            tc.mu.Lock()
            tc.errors = append(tc.errors, err)
            tc.mu.Unlock()
        }
    }()
}

func (tc *TaskCoordinator) WaitAll() []error {
    tc.wg.Wait()
    return tc.errors
}
\`\`\`

**Misol ishlatish:**
\`\`\`go
coordinator := NewTaskCoordinator()

// Parallel worker larni boshlash
for i := 0; i < 10; i++ {
    id := i
    coordinator.AddWorker(func() error {
        return processTask(id)
    })
}

// Hammasi tugashini kutish
errors := coordinator.WaitAll()
if len(errors) > 0 {
    fmt.Printf("Vazifalar %d xato bilan tugadi\n", len(errors))
}
\`\`\`

**Cheklovlar:**
- sync.WaitGroup ishlatish kerak
- AddWorker goroutine dan oldin Add(1) chaqirishi kerak
- Goroutine defer Done() ishlatishi kerak
- Xatolarni yig'ish thread-safe bo'lishi kerak`,
			hint1: `AddWorker da: tc.wg.Add(1) chaqiring, keyin defer tc.wg.Done() bilan goroutine boshlang. fn() ni bajarib mutex himoyasi bilan xatoni yig'ing.`,
			hint2: `WaitAll da: Barcha worker lar tugaguncha bloklash uchun oddiy tc.wg.Wait() chaqiring, keyin tc.errors ni qaytaring.`,
			whyItMatters: `WaitGroup Go da bir nechta goroutine larning tugashini kutishning standart usuli. Bu parallel ishlov berish, fan-out/fan-in patternlar va parallel ishni muvofiqlashtirish uchun zarurdir. WaitGroup ni tushunish erta dastur tugashi yoki goroutine sizib ketishi kabi keng tarqalgan xatolarning oldini oladi.

**Nima uchun bu muhim:**

**Ishlab chiqarish patterni:**
\`\`\`go
type TaskCoordinator struct {
    wg     sync.WaitGroup
    mu     sync.Mutex
    errors []error
}

func (tc *TaskCoordinator) AddWorker(fn func() error) {
    tc.wg.Add(1)                   // goroutine dan oldin hisoblagichni oshiramiz
    go func() {
        defer tc.wg.Done()          // goroutine tugaganda hisoblagichni kamaytiramiz
        if err := fn(); err != nil { // worker funksiyasini bajaramiz
            tc.mu.Lock()             // umumiy xatolar slice ni himoyalaymiz
            tc.errors = append(tc.errors, err)
            tc.mu.Unlock()
        }
    }()
}

func (tc *TaskCoordinator) WaitAll() []error {
    tc.wg.Wait()                   // barcha Done() lar Add() ga mos kelguncha bloklaymiz
    return tc.errors               // barcha goroutine lar tugagandan keyin xavfsiz qaytaramiz
}
\`\`\`

**Amaliy foydalari:**
- Bir nechta goroutine larni muvofiqlashtirish uchun standart usul
- Dasturning erta tugashining oldini oladi
- Parallel ishlov berish va fan-out/fan-in patternlar uchun muhim
- Natijalar va xatolarni xavfsiz yig'ishga imkon beradi
- Tushunish goroutine sizib ketishini oldini oladi`
		}
	}
};

export default task;
