import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-sync-semaphore',
	title: 'Semaphore for Resource Limiting',
	difficulty: 'medium',
	tags: ['go', 'sync', 'semaphore', 'channels', 'concurrency'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a semaphore using buffered channels to limit concurrent access to resources.

**Requirements:**
1. **Semaphore**: Control max concurrent operations
2. **Acquire**: Block until permit available, then take permit
3. **Release**: Return permit, allowing waiting goroutine to proceed
4. **Resource Pool**: Demonstrate limiting concurrent database connections

**Implementation Pattern:**
\`\`\`go
type Semaphore struct {
    permits chan struct{}
}

func NewSemaphore(maxConcurrent int) *Semaphore {
    return &Semaphore{
        permits: make(chan struct{}, maxConcurrent),
    }
}

func (s *Semaphore) Acquire() {
    s.permits <- struct{}{}  // Block if channel full
}

func (s *Semaphore) Release() {
    <-s.permits  // Free slot for waiting goroutine
}
\`\`\`

**Example Usage:**
\`\`\`go
// Limit to 3 concurrent database connections
sem := NewSemaphore(3)

var wg sync.WaitGroup
for i := 0; i < 100; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()

        sem.Acquire()              // Block if 3 connections active
        defer sem.Release()         // Always release permit

        queryDatabase(id)          // Max 3 concurrent
    }(i)
}
wg.Wait()
\`\`\`

**Practical Applications:**
- Rate limiting API requests
- Limiting concurrent database connections
- Controlling worker pool size
- Preventing resource exhaustion

**Constraints:**
- Must use buffered channel for permits
- Acquire must block when limit reached
- Always defer Release after Acquire
- Channel size determines max concurrency`,
	initialCode: `package syncx

type Semaphore struct {
	permits chan struct{}
}

// TODO: Implement NewSemaphore
// Create buffered channel with maxConcurrent capacity
func NewSemaphore(maxConcurrent int) *Semaphore {
	// TODO: Implement
}

// TODO: Implement Acquire
// Send empty struct to channel (blocks if full)
func (s *Semaphore) Acquire() {
	// TODO: Implement
}

// TODO: Implement Release
// Receive from channel to free slot
func (s *Semaphore) Release() {
	// TODO: Implement
}

// Helper: Execute function with semaphore protection
func (s *Semaphore) Execute(fn func()) {
	s.Acquire()
	defer s.Release()
	fn()
}`,
	solutionCode: `package syncx

type Semaphore struct {
	permits chan struct{}
}

func NewSemaphore(maxConcurrent int) *Semaphore {
	return &Semaphore{
		permits: make(chan struct{}, maxConcurrent),	// buffered channel = semaphore
	}
}

func (s *Semaphore) Acquire() {
	s.permits <- struct{}{}	// send blocks when channel full (all permits taken)
}

func (s *Semaphore) Release() {
	<-s.permits	// receive frees slot (returns permit to pool)
}

func (s *Semaphore) Execute(fn func()) {
	s.Acquire()	// acquire permit (blocks if limit reached)
	defer s.Release()	// always return permit
	fn()	// execute protected operation
}`,
	testCode: `package syncx

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestSemaphoreAcquireRelease(t *testing.T) {
	sem := NewSemaphore(1)
	sem.Acquire()
	sem.Release()
}

func TestSemaphoreSinglePermit(t *testing.T) {
	sem := NewSemaphore(1)
	var counter int32
	var wg sync.WaitGroup

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem.Acquire()
			current := atomic.AddInt32(&counter, 1)
			if current > 1 {
				t.Errorf("expected max 1 concurrent, got %d", current)
			}
			time.Sleep(10 * time.Millisecond)
			atomic.AddInt32(&counter, -1)
			sem.Release()
		}()
	}

	wg.Wait()
}

func TestSemaphoreMultiplePermits(t *testing.T) {
	sem := NewSemaphore(3)
	var counter int32
	var wg sync.WaitGroup

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem.Acquire()
			current := atomic.AddInt32(&counter, 1)
			if current > 3 {
				t.Errorf("expected max 3 concurrent, got %d", current)
			}
			time.Sleep(10 * time.Millisecond)
			atomic.AddInt32(&counter, -1)
			sem.Release()
		}()
	}

	wg.Wait()
}

func TestSemaphoreExecute(t *testing.T) {
	sem := NewSemaphore(2)
	var counter int32
	var wg sync.WaitGroup

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem.Execute(func() {
				current := atomic.AddInt32(&counter, 1)
				if current > 2 {
					t.Errorf("expected max 2 concurrent, got %d", current)
				}
				time.Sleep(10 * time.Millisecond)
				atomic.AddInt32(&counter, -1)
			})
		}()
	}

	wg.Wait()
}

func TestSemaphoreZeroPermits(t *testing.T) {
	sem := NewSemaphore(0)
	done := make(chan bool)

	go func() {
		sem.Acquire()
		done <- true
	}()

	select {
	case <-done:
		t.Errorf("should not acquire with 0 permits")
	case <-time.After(50 * time.Millisecond):
		// Expected to timeout
	}
}

func TestSemaphoreBlocksWhenFull(t *testing.T) {
	sem := NewSemaphore(1)
	sem.Acquire()

	acquired := false
	go func() {
		sem.Acquire()
		acquired = true
		sem.Release()
	}()

	time.Sleep(50 * time.Millisecond)
	if acquired {
		t.Errorf("should not have acquired while semaphore full")
	}

	sem.Release()
	time.Sleep(50 * time.Millisecond)
	if !acquired {
		t.Errorf("should have acquired after release")
	}
}

func TestSemaphoreFairness(t *testing.T) {
	sem := NewSemaphore(1)
	var order []int
	var mu sync.Mutex
	var wg sync.WaitGroup

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			sem.Acquire()
			mu.Lock()
			order = append(order, id)
			mu.Unlock()
			time.Sleep(10 * time.Millisecond)
			sem.Release()
		}(i)
	}

	wg.Wait()
	if len(order) != 5 {
		t.Errorf("expected 5 items in order, got %d", len(order))
	}
}

func TestSemaphoreReleaseWithoutAcquire(t *testing.T) {
	sem := NewSemaphore(1)
	sem.Release() // Should not panic
}

func TestSemaphoreMultipleReleases(t *testing.T) {
	sem := NewSemaphore(1)
	sem.Acquire()
	sem.Release()
	sem.Release() // Extra release
}

func TestSemaphoreLargePermitCount(t *testing.T) {
	sem := NewSemaphore(100)
	var wg sync.WaitGroup
	var counter int32

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem.Acquire()
			atomic.AddInt32(&counter, 1)
			sem.Release()
		}()
	}

	wg.Wait()
	if counter != 100 {
		t.Errorf("expected counter=100, got %d", counter)
	}
}`,
	hint1: `NewSemaphore: Create buffered channel with make(chan struct{}, maxConcurrent). The buffer size is the key!`,
	hint2: `Acquire sends to channel: s.permits <- struct{}{}. Release receives: <-s.permits. Sending blocks when buffer full.`,
	whyItMatters: `Semaphores are crucial for controlling concurrent resource access in production systems. They prevent resource exhaustion (too many DB connections, API requests), enable rate limiting, and protect bounded resources. Unlike mutexes which provide exclusive access, semaphores allow N concurrent operations, making them ideal for connection pools, worker pools, and rate limiters.

**Production Pattern:**

\`\`\`go
type Semaphore struct {
    permits chan struct{}
}

func NewSemaphore(maxConcurrent int) *Semaphore {
    return &Semaphore{
        permits: make(chan struct{}, maxConcurrent), // buffered channel = semaphore
    }
}

func (s *Semaphore) Acquire() {
    s.permits <- struct{}{} // send blocks when channel full (all permits taken)
}

func (s *Semaphore) Release() {
    <-s.permits // receive frees slot (returns permit to pool)
}

func (s *Semaphore) Execute(fn func()) {
    s.Acquire()             // acquire permit (blocks if limit reached)
    defer s.Release()       // always return permit
    fn()                    // execute protected operation
}
\`\`\`

**Practical Benefits:**
- Control maximum number of concurrent operations
- Prevents resource exhaustion (DB connections, API requests)
- Ideal for connection pools, worker pools and rate limiters
- Unlike Mutex (1 operation), semaphores allow N concurrent operations
- Critical for protecting bounded resources in production systems`,
	order: 3,
	translations: {
		ru: {
			title: 'Паттерн семафор',
			solutionCode: `package syncx

type Semaphore struct {
	permits chan struct{}
}

func NewSemaphore(maxConcurrent int) *Semaphore {
	return &Semaphore{
		permits: make(chan struct{}, maxConcurrent),	// буферизованный канал = семафор
	}
}

func (s *Semaphore) Acquire() {
	s.permits <- struct{}{}	// отправка блокируется когда канал полон (все разрешения взяты)
}

func (s *Semaphore) Release() {
	<-s.permits	// получение освобождает слот (возвращает разрешение в пул)
}

func (s *Semaphore) Execute(fn func()) {
	s.Acquire()	// получаем разрешение (блокируется если лимит достигнут)
	defer s.Release()	// всегда возвращаем разрешение
	fn()	// выполняем защищенную операцию
}`,
			description: `Реализуйте семафор используя буферизованные каналы для ограничения конкурентного доступа к ресурсам.

**Требования:**
1. **Semaphore**: Контроль максимального количества конкурентных операций
2. **Acquire**: Блокировка до доступности разрешения, затем взятие разрешения
3. **Release**: Возврат разрешения, позволяя ждущей горутине продолжить
4. **Resource Pool**: Демонстрация ограничения конкурентных подключений к БД

**Паттерн реализации:**
\`\`\`go
type Semaphore struct {
    permits chan struct{}
}

func NewSemaphore(maxConcurrent int) *Semaphore {
    return &Semaphore{
        permits: make(chan struct{}, maxConcurrent),
    }
}

func (s *Semaphore) Acquire() {
    s.permits <- struct{}{}  // Блокировка если канал полон
}

func (s *Semaphore) Release() {
    <-s.permits  // Освобождаем слот для ждущей горутины
}
\`\`\`

**Практические применения:**
- Ограничение скорости API запросов
- Ограничение конкурентных подключений к БД
- Контроль размера пула workers
- Предотвращение истощения ресурсов

**Ограничения:**
- Должен использовать буферизованный канал для разрешений
- Acquire должен блокироваться при достижении лимита
- Всегда defer Release после Acquire
- Размер канала определяет максимальную конкурентность`,
			hint1: `NewSemaphore: Создайте буферизованный канал с make(chan struct{}, maxConcurrent). Размер буфера - это ключ!`,
			hint2: `Acquire отправляет в канал: s.permits <- struct{}{}. Release получает: <-s.permits. Отправка блокируется когда буфер полон.`,
			whyItMatters: `Семафоры критичны для контроля конкурентного доступа к ресурсам в production системах. Они предотвращают истощение ресурсов (слишком много подключений к БД, API запросов), обеспечивают rate limiting и защищают ограниченные ресурсы. В отличие от mutex которые обеспечивают эксклюзивный доступ, семафоры позволяют N конкурентных операций, что делает их идеальными для connection pools, worker pools и rate limiters.

**Почему важно:**

**Продакшен паттерн:**
\`\`\`go
type Semaphore struct {
    permits chan struct{}
}

func NewSemaphore(maxConcurrent int) *Semaphore {
    return &Semaphore{
        permits: make(chan struct{}, maxConcurrent), // буферизованный канал = семафор
    }
}

func (s *Semaphore) Acquire() {
    s.permits <- struct{}{} // отправка блокируется когда канал полон (все разрешения взяты)
}

func (s *Semaphore) Release() {
    <-s.permits // получение освобождает слот (возвращает разрешение в пул)
}

func (s *Semaphore) Execute(fn func()) {
    s.Acquire()             // получаем разрешение (блокируется если лимит достигнут)
    defer s.Release()       // всегда возвращаем разрешение
    fn()                    // выполняем защищенную операцию
}
\`\`\`

**Практические преимущества:**
- Контроль максимального количества конкурентных операций
- Предотвращает истощение ресурсов (DB connections, API requests)
- Идеально для connection pools, worker pools и rate limiters
- В отличие от Mutex (1 операция), семафоры позволяют N конкурентных операций
- Критично для защиты ограниченных ресурсов в production системах`
		},
		uz: {
			title: `Semafor patterni`,
			solutionCode: `package syncx

type Semaphore struct {
	permits chan struct{}
}

func NewSemaphore(maxConcurrent int) *Semaphore {
	return &Semaphore{
		permits: make(chan struct{}, maxConcurrent),	// buferli kanal = semafor
	}
}

func (s *Semaphore) Acquire() {
	s.permits <- struct{}{}	// kanal to'lganda yuborish bloklanadi (barcha ruxsatlar olingan)
}

func (s *Semaphore) Release() {
	<-s.permits	// qabul qilish slot ni bo'shatadi (ruxsatni poolga qaytaradi)
}

func (s *Semaphore) Execute(fn func()) {
	s.Acquire()	// ruxsat olamiz (limit yetganda bloklanadi)
	defer s.Release()	// har doim ruxsatni qaytaramiz
	fn()	// himoyalangan operatsiyani bajaramiz
}`,
			description: `Resurslarga parallel kirishni cheklash uchun buferli kanallardan foydalanib semafor amalga oshiring.

**Talablar:**
1. **Semaphore**: Maksimal parallel operatsiyalarni boshqarish
2. **Acquire**: Ruxsat mavjud bo'lguncha bloklash, keyin ruxsat olish
3. **Release**: Ruxsatni qaytarish, kutayotgan goroutine ga davom etishga ruxsat berish
4. **Resource Pool**: Parallel ma'lumotlar bazasi ulanishlarini cheklashni ko'rsatish

**Amalga oshirish patterni:**
\`\`\`go
type Semaphore struct {
    permits chan struct{}
}

func NewSemaphore(maxConcurrent int) *Semaphore {
    return &Semaphore{
        permits: make(chan struct{}, maxConcurrent),
    }
}

func (s *Semaphore) Acquire() {
    s.permits <- struct{}{}  // Kanal to'lsa bloklash
}

func (s *Semaphore) Release() {
    <-s.permits  // Kutayotgan goroutine uchun slot bo'shatish
}
\`\`\`

**Misol ishlatish:**
\`\`\`go
// 3 ta parallel ma'lumotlar bazasi ulanishiga cheklash
sem := NewSemaphore(3)

var wg sync.WaitGroup
for i := 0; i < 100; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()

        sem.Acquire()              // 3 ta ulanish faol bo'lsa bloklash
        defer sem.Release()         // Har doim ruxsatni qaytarish

        queryDatabase(id)          // Maksimal 3 ta parallel
    }(i)
}
wg.Wait()
\`\`\`

**Amaliy qo'llanmalar:**
- API so'rovlarini tezlik cheklash
- Parallel ma'lumotlar bazasi ulanishlarini cheklash
- Worker pool hajmini boshqarish
- Resurs tugashining oldini olish

**Cheklovlar:**
- Ruxsatlar uchun buferli kanal ishlatish kerak
- Acquire limit yetganda bloklashi kerak
- Acquire dan keyin har doim defer Release
- Kanal hajmi maksimal parallellikni belgilaydi`,
			hint1: `NewSemaphore: make(chan struct{}, maxConcurrent) bilan buferli kanal yarating. Bufer hajmi kalitdir!`,
			hint2: `Acquire kanalga yuboradi: s.permits <- struct{}{}. Release qabul qiladi: <-s.permits. Bufer to'lganda yuborish bloklanadi.`,
			whyItMatters: `Semaforlar production tizimlarda parallel resurs kirishini boshqarish uchun juda muhimdir. Ular resurs tugashining oldini oladi (juda ko'p DB ulanishlari, API so'rovlar), tezlik cheklashni ta'minlaydi va chegaralangan resurslarni himoya qiladi. Eksklyuziv kirish ta'minlaydigan mutex dan farqli o'laroq, semaforlar N ta parallel operatsiyaga ruxsat beradi, bu ularni connection pool, worker pool va rate limiterlar uchun ideal qiladi.

**Nima uchun bu muhim:**

**Ishlab chiqarish patterni:**
\`\`\`go
type Semaphore struct {
    permits chan struct{}
}

func NewSemaphore(maxConcurrent int) *Semaphore {
    return &Semaphore{
        permits: make(chan struct{}, maxConcurrent), // buferli kanal = semafor
    }
}

func (s *Semaphore) Acquire() {
    s.permits <- struct{}{} // kanal to'lganda yuborish bloklanadi (barcha ruxsatlar olingan)
}

func (s *Semaphore) Release() {
    <-s.permits // qabul qilish slot ni bo'shatadi (ruxsatni poolga qaytaradi)
}

func (s *Semaphore) Execute(fn func()) {
    s.Acquire()             // ruxsat olamiz (limit yetganda bloklanadi)
    defer s.Release()       // har doim ruxsatni qaytaramiz
    fn()                    // himoyalangan operatsiyani bajaramiz
}
\`\`\`

**Amaliy foydalari:**
- Maksimal parallel operatsiyalarni boshqarish
- Resurs tugashini oldini oladi (DB connections, API requests)
- Connection pools, worker pools va rate limiters uchun ideal
- Mutex dan farqli (1 operatsiya), semaforlar N ta parallel operatsiyaga ruxsat beradi
- Production tizimlarda chegaralangan resurslarni himoya qilish uchun muhim`
		}
	}
};

export default task;
