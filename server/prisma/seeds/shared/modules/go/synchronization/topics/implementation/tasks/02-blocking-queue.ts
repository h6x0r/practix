import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-sync-blocking-queue',
	title: 'Blocking Queue with Condition Variable',
	difficulty: 'medium',	tags: ['go', 'sync', 'cond', 'queue'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement a blocking queue using sync.Cond that blocks Pop when empty.

**Requirements:**
1. **Queue**: Generic queue with slice backing
2. **Push**: Add item and signal waiting consumers
3. **Pop**: Block until item available, then remove and return
4. **Condition Variable**: Use sync.Cond for blocking

**Implementation Pattern:**
\`\`\`go
type Queue[T any] struct {
    mu sync.Mutex
    cv *sync.Cond
    q  []T
}

func NewQueue[T any]() *Queue[T] {
    q := &Queue[T]{}
    q.cv = sync.NewCond(&q.mu)
    return q
}

func (q *Queue[T]) Push(v T) {
    q.mu.Lock()
    q.q = append(q.q, v)
    q.mu.Unlock()
    q.cv.Signal()  // Wake one waiting goroutine
}

func (q *Queue[T]) Pop() T {
    q.mu.Lock()
    defer q.mu.Unlock()
    for len(q.q) == 0 {
        q.cv.Wait()  // Block until signaled
    }
    v := q.q[0]
    q.q = q.q[1:]
    return v
}
\`\`\`

**Example Usage:**
\`\`\`go
queue := NewQueue[int]()

// Consumer goroutine
go func() {
    for i := 0; i < 10; i++ {
        val := queue.Pop()  // Blocks until item available
        process(val)
    }
}()

// Producer goroutine
for i := 0; i < 10; i++ {
    queue.Push(i)  // Signals waiting consumer
}
\`\`\`

**Constraints:**
- Must use sync.Cond
- Pop must block when queue is empty
- Push must signal waiting goroutines
- Must hold mutex during Wait`,
	initialCode: `package syncx

import "sync"

type Queue[T any] struct {
	mu sync.Mutex
	cv *sync.Cond
	q  []T
}

func NewQueue[T any]() *Queue[T] {
	q := &Queue[T]{}
	q.cv = sync.NewCond(&q.mu)
	return q
}

// TODO: Implement Push
// Lock, append to queue, unlock, signal
func (q *Queue[T]) Push(v T) {
	// TODO: Implement
}

// TODO: Implement Pop
// Lock, wait while empty, pop first item, unlock
func (q *Queue[T]) Pop() T {
	// TODO: Implement
}`,
	solutionCode: `package syncx

import "sync"

type Queue[T any] struct {
	mu sync.Mutex
	cv *sync.Cond
	q  []T
}

func NewQueue[T any]() *Queue[T] {
	q := &Queue[T]{}
	q.cv = sync.NewCond(&q.mu)
	return q
}

func (q *Queue[T]) Push(v T) {
	q.mu.Lock()	// acquire lock
	q.q = append(q.q, v)	// add item
	q.mu.Unlock()	// release lock
	q.cv.Signal()	// wake one waiting goroutine
}

func (q *Queue[T]) Pop() T {
	q.mu.Lock()	// acquire lock
	defer q.mu.Unlock()	// ensure unlock
	for len(q.q) == 0 {	// wait while empty
		q.cv.Wait()	// block until signaled (releases lock while waiting)
	}
	v := q.q[0]	// get first item
	q.q = q.q[1:]	// remove from queue
	return v	// return item
}`,
	testCode: `package syncx

import (
	"sync"
	"testing"
	"time"
)

func TestQueuePushPop(t *testing.T) {
	q := NewQueue[int]()
	q.Push(42)
	result := q.Pop()
	if result != 42 {
		t.Errorf("expected 42, got %d", result)
	}
}

func TestQueueMultiplePushPop(t *testing.T) {
	q := NewQueue[int]()
	q.Push(1)
	q.Push(2)
	q.Push(3)
	if q.Pop() != 1 {
		t.Errorf("expected 1")
	}
	if q.Pop() != 2 {
		t.Errorf("expected 2")
	}
	if q.Pop() != 3 {
		t.Errorf("expected 3")
	}
}

func TestQueueBlockingPop(t *testing.T) {
	q := NewQueue[int]()
	done := make(chan bool)

	go func() {
		result := q.Pop()
		if result != 100 {
			t.Errorf("expected 100, got %d", result)
		}
		done <- true
	}()

	time.Sleep(50 * time.Millisecond)
	q.Push(100)
	<-done
}

func TestQueueFIFOOrder(t *testing.T) {
	q := NewQueue[string]()
	q.Push("first")
	q.Push("second")
	q.Push("third")

	if q.Pop() != "first" {
		t.Errorf("expected first")
	}
	if q.Pop() != "second" {
		t.Errorf("expected second")
	}
	if q.Pop() != "third" {
		t.Errorf("expected third")
	}
}

func TestQueueConcurrentPushPop(t *testing.T) {
	q := NewQueue[int]()
	var wg sync.WaitGroup
	n := 100

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(val int) {
			defer wg.Done()
			q.Push(val)
		}(i)
	}

	results := make([]int, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = q.Pop()
		}(i)
	}

	wg.Wait()
	if len(results) != n {
		t.Errorf("expected %d results, got %d", n, len(results))
	}
}

func TestQueueMultipleProducersConsumers(t *testing.T) {
	q := NewQueue[int]()
	var wg sync.WaitGroup
	producers := 5
	consumers := 5
	itemsPerProducer := 20

	for i := 0; i < producers; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < itemsPerProducer; j++ {
				q.Push(id*100 + j)
			}
		}(i)
	}

	consumed := make([]int, 0)
	var mu sync.Mutex
	for i := 0; i < consumers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < itemsPerProducer; j++ {
				val := q.Pop()
				mu.Lock()
				consumed = append(consumed, val)
				mu.Unlock()
			}
		}()
	}

	wg.Wait()
	expected := producers * itemsPerProducer
	if len(consumed) != expected {
		t.Errorf("expected %d items consumed, got %d", expected, len(consumed))
	}
}

func TestQueueEmptyInitially(t *testing.T) {
	q := NewQueue[int]()
	done := make(chan bool)

	go func() {
		time.Sleep(20 * time.Millisecond)
		q.Push(1)
		done <- true
	}()

	result := q.Pop()
	<-done
	if result != 1 {
		t.Errorf("expected 1, got %d", result)
	}
}

func TestQueueStringType(t *testing.T) {
	q := NewQueue[string]()
	q.Push("hello")
	q.Push("world")
	if q.Pop() != "hello" {
		t.Errorf("expected hello")
	}
	if q.Pop() != "world" {
		t.Errorf("expected world")
	}
}

func TestQueueStructType(t *testing.T) {
	type Data struct {
		ID   int
		Name string
	}
	q := NewQueue[Data]()
	q.Push(Data{ID: 1, Name: "test"})
	result := q.Pop()
	if result.ID != 1 || result.Name != "test" {
		t.Errorf("expected {1, test}, got {%d, %s}", result.ID, result.Name)
	}
}

func TestQueueSequentialPushPop(t *testing.T) {
	q := NewQueue[int]()
	for i := 0; i < 10; i++ {
		q.Push(i)
		result := q.Pop()
		if result != i {
			t.Errorf("expected %d, got %d", i, result)
		}
	}
}`,
			hint1: `In Push: Lock, append, Unlock, then Signal. In Pop: Lock with defer, loop while empty calling Wait, then pop first element.`,
			hint2: `Use for loop with Wait: for len(q.q) == 0 { q.cv.Wait() }. Wait releases lock while blocking.`,
			whyItMatters: `Blocking queues are essential for producer-consumer patterns in concurrent systems. sync.Cond provides efficient blocking and waking mechanisms, avoiding busy-waiting and reducing CPU usage. This pattern is used in job queues, rate limiters, and bounded buffer implementations.

**Production Pattern:**

\`\`\`go
type Queue[T any] struct {
    mu sync.Mutex
    cv *sync.Cond
    q  []T
}

func (q *Queue[T]) Push(v T) {
    q.mu.Lock()                    // acquire lock
    q.q = append(q.q, v)           // add item
    q.mu.Unlock()                  // release lock
    q.cv.Signal()                  // wake one waiting goroutine
}

func (q *Queue[T]) Pop() T {
    q.mu.Lock()                    // acquire lock
    defer q.mu.Unlock()            // ensure unlock
    for len(q.q) == 0 {            // wait while empty
        q.cv.Wait()                 // block until signaled (releases lock while waiting)
    }
    v := q.q[0]                    // get first item
    q.q = q.q[1:]                  // remove from queue
    return v                       // return item
}
\`\`\`

**Practical Benefits:**
- Efficient blocking and waking mechanisms
- Avoids busy-waiting and reduces CPU usage
- Ideal for job queues, rate limiters and bounded buffer implementations
- sync.Cond allows multiple waiting goroutines
- Critical for producer-consumer patterns in concurrent systems`,	order: 1,
	translations: {
		ru: {
			title: 'Блокирующая очередь',
			solutionCode: `package syncx

import "sync"

type Queue[T any] struct {
	mu sync.Mutex
	cv *sync.Cond
	q  []T
}

func NewQueue[T any]() *Queue[T] {
	q := &Queue[T]{}
	q.cv = sync.NewCond(&q.mu)
	return q
}

func (q *Queue[T]) Push(v T) {
	q.mu.Lock()	// захватываем блокировку
	q.q = append(q.q, v)	// добавляем элемент
	q.mu.Unlock()	// освобождаем блокировку
	q.cv.Signal()	// пробуждаем одну ждущую горутину
}

func (q *Queue[T]) Pop() T {
	q.mu.Lock()	// захватываем блокировку
	defer q.mu.Unlock()	// гарантируем разблокировку
	for len(q.q) == 0 {	// ждем пока пусто
		q.cv.Wait()	// блокируемся до сигнала (освобождает lock во время ожидания)
	}
	v := q.q[0]	// получаем первый элемент
	q.q = q.q[1:]	// удаляем из очереди
	return v	// возвращаем элемент
}`,
			description: `Реализуйте блокирующую очередь используя sync.Cond которая блокирует Pop когда пуста.

**Требования:**
1. **Queue**: Обобщенная очередь с поддержкой slice
2. **Push**: Добавить элемент и сигнализировать ожидающим потребителям
3. **Pop**: Блокировать до появления элемента, затем удалить и вернуть
4. **Условная переменная**: Использовать sync.Cond для блокировки

**Паттерн реализации:**
\`\`\`go
type Queue[T any] struct {
    mu sync.Mutex
    cv *sync.Cond
    q  []T
}

func NewQueue[T any]() *Queue[T] {
    q := &Queue[T]{}
    q.cv = sync.NewCond(&q.mu)
    return q
}

func (q *Queue[T]) Push(v T) {
    q.mu.Lock()
    q.q = append(q.q, v)
    q.mu.Unlock()
    q.cv.Signal()  // Разбудить одну ожидающую горутину
}

func (q *Queue[T]) Pop() T {
    q.mu.Lock()
    defer q.mu.Unlock()
    for len(q.q) == 0 {
        q.cv.Wait()  // Блокировать до сигнала
    }
    v := q.q[0]
    q.q = q.q[1:]
    return v
}
\`\`\`

**Пример использования:**
\`\`\`go
queue := NewQueue[int]()

// Горутина-потребитель
go func() {
    for i := 0; i < 10; i++ {
        val := queue.Pop()  // Блокируется до появления элемента
        process(val)
    }
}()

// Горутина-производитель
for i := 0; i < 10; i++ {
    queue.Push(i)  // Сигнализирует ожидающему потребителю
}
\`\`\`

**Ограничения:**
- Необходимо использовать sync.Cond
- Pop должен блокировать когда очередь пуста
- Push должен сигнализировать ожидающим горутинам
- Необходимо удерживать мьютекс во время Wait`,
			hint1: `В Push: Lock, append, Unlock, затем Signal. В Pop: Lock с defer, цикл пока пусто вызывая Wait, затем pop первого элемента.`,
			hint2: `Используйте for loop с Wait: for len(q.q) == 0 { q.cv.Wait() }. Wait освобождает lock во время блокировки.`,
			whyItMatters: `Блокирующие очереди критичны для producer-consumer паттернов в конкурентных системах. sync.Cond обеспечивает эффективные механизмы блокировки и пробуждения.

**Почему важно:**

**Продакшен паттерн:**
\`\`\`go
type Queue[T any] struct {
    mu sync.Mutex
    cv *sync.Cond
    q  []T
}

func (q *Queue[T]) Push(v T) {
    q.mu.Lock()                    // захватываем блокировку
    q.q = append(q.q, v)           // добавляем элемент
    q.mu.Unlock()                  // освобождаем блокировку
    q.cv.Signal()                  // пробуждаем одну ждущую горутину
}

func (q *Queue[T]) Pop() T {
    q.mu.Lock()                    // захватываем блокировку
    defer q.mu.Unlock()            // гарантируем разблокировку
    for len(q.q) == 0 {            // ждем пока пусто
        q.cv.Wait()                 // блокируемся до сигнала (освобождает lock во время ожидания)
    }
    v := q.q[0]                    // получаем первый элемент
    q.q = q.q[1:]                  // удаляем из очереди
    return v                       // возвращаем элемент
}
\`\`\`

**Практические преимущества:**
- Эффективная блокировка и пробуждение механизмы
- Избегает busy-waiting и снижает использование CPU
- Идеально для job queues, rate limiters и bounded buffer реализаций
- sync.Cond позволяет несколько ждущих горутин
- Критично для producer-consumer паттернов в конкурентных системах`
		},
		uz: {
			title: `Blokirovka qiluvchi navbat`,
			solutionCode: `package syncx

import "sync"

type Queue[T any] struct {
	mu sync.Mutex
	cv *sync.Cond
	q  []T
}

func NewQueue[T any]() *Queue[T] {
	q := &Queue[T]{}
	q.cv = sync.NewCond(&q.mu)
	return q
}

func (q *Queue[T]) Push(v T) {
	q.mu.Lock()	// qulfni olamiz
	q.q = append(q.q, v)	// element qo'shamiz
	q.mu.Unlock()	// qulfni bo'shatamiz
	q.cv.Signal()	// bitta kutayotgan goroutine ni uyg'otamiz
}

func (q *Queue[T]) Pop() T {
	q.mu.Lock()	// qulfni olamiz
	defer q.mu.Unlock()	// qulfni ochishni kafolatlaymiz
	for len(q.q) == 0 {	// bo'sh bo'lganida kutamiz
		q.cv.Wait()	// signal bo'lguncha bloklaymiz (kutish paytida lock ni bo'shatadi)
	}
	v := q.q[0]	// birinchi elementni olamiz
	q.q = q.q[1:]	// navbatdan olib tashlaymiz
	return v	// elementni qaytaramiz
}`,
			description: `Bo'sh bo'lganda Pop ni bloklaydigan sync.Cond ishlatib bloklash navbatini amalga oshiring.

**Talablar:**
1. **Queue**: Slice bilan qo'llab-quvvatlanadigan generik navbat
2. **Push**: Element qo'shish va kutayotgan iste'molchilarni signal qilish
3. **Pop**: Element mavjud bo'lguncha bloklash, keyin olib tashlash va qaytarish
4. **Condition Variable**: Bloklash uchun sync.Cond ishlatish

**Amalga oshirish patterni:**
\`\`\`go
type Queue[T any] struct {
    mu sync.Mutex
    cv *sync.Cond
    q  []T
}

func (q *Queue[T]) Push(v T) {
    q.mu.Lock()
    q.q = append(q.q, v)
    q.mu.Unlock()
    q.cv.Signal()  // Bitta kutayotgan goroutine ni uyg'otish
}

func (q *Queue[T]) Pop() T {
    q.mu.Lock()
    defer q.mu.Unlock()
    for len(q.q) == 0 {
        q.cv.Wait()  // Signal bo'lguncha bloklash
    }
    v := q.q[0]
    q.q = q.q[1:]
    return v
}
\`\`\`

**Cheklovlar:**
- sync.Cond ishlatish kerak
- Pop navbat bo'sh bo'lganda bloklashi kerak
- Push kutayotgan goroutine larni signal qilishi kerak
- Wait paytida mutex ushlab turish kerak`,
			hint1: `Push da: Lock, append, Unlock, keyin Signal. Pop da: defer bilan Lock, bo'sh bo'lganida Wait chaqirib loop, keyin birinchi elementni pop qilish.`,
			hint2: `Wait bilan for loop ishlating: for len(q.q) == 0 { q.cv.Wait() }. Wait bloklash paytida lock ni bo'shatadi.`,
			whyItMatters: `Bloklash navbatlari parallel tizimlarda producer-consumer patternlar uchun muhimdir. sync.Cond samarali bloklash va uyg'otish mexanizmlarini ta'minlaydi, busy-waiting dan qochadi va CPU ishlatishini kamaytiradi. Bu pattern ish navbatlari, rate limiterlar va chegaralangan bufer amalga oshirishlarida ishlatiladi.

**Nima uchun bu muhim:**

**Ishlab chiqarish patterni:**
\`\`\`go
type Queue[T any] struct {
    mu sync.Mutex
    cv *sync.Cond
    q  []T
}

func (q *Queue[T]) Push(v T) {
    q.mu.Lock()                    // qulfni olamiz
    q.q = append(q.q, v)           // element qo'shamiz
    q.mu.Unlock()                  // qulfni bo'shatamiz
    q.cv.Signal()                  // bitta kutayotgan goroutine ni uyg'otamiz
}

func (q *Queue[T]) Pop() T {
    q.mu.Lock()                    // qulfni olamiz
    defer q.mu.Unlock()            // qulfni ochishni kafolatlaymiz
    for len(q.q) == 0 {            // bo'sh bo'lganida kutamiz
        q.cv.Wait()                 // signal bo'lguncha bloklaymiz (kutish paytida lock ni bo'shatadi)
    }
    v := q.q[0]                    // birinchi elementni olamiz
    q.q = q.q[1:]                  // navbatdan olib tashlaymiz
    return v                       // elementni qaytaramiz
}
\`\`\`

**Amaliy foydalari:**
- Samarali bloklash va uyg'otish mexanizmlari
- Busy-waiting dan qochadi va CPU ishlatishini kamaytiradi
- Job queues, rate limiters va bounded buffer amalga oshirishlar uchun ideal
- sync.Cond bir nechta kutayotgan goroutine larga ruxsat beradi
- Parallel tizimlarda producer-consumer patternlar uchun muhim`
		}
	}
};

export default task;
