import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-sync-safe-counter',
	title: 'Thread-Safe Counter with RWMutex',
	difficulty: 'easy',	tags: ['go', 'sync', 'mutex', 'concurrency'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement a thread-safe counter using sync.RWMutex for concurrent access.

**Requirements:**
1. **SafeCounter**: Store map[string]int with RWMutex protection
2. **Inc(key)**: Atomically increment counter for key
3. **Get(key)**: Safely read counter value
4. **Thread Safety**: Prevent data races with proper locking

**Implementation Pattern:**
\`\`\`go
type SafeCounter struct {
    mu sync.RWMutex
    m  map[string]int
}

func (s *SafeCounter) Inc(k string) {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.m[k]++
}

func (s *SafeCounter) Get(k string) int {
    s.mu.RLock()
    defer s.mu.RUnlock()
    return s.m[k]
}
\`\`\`

**Example Usage:**
\`\`\`go
counter := NewSafeCounter()

// Concurrent increments
var wg sync.WaitGroup
for i := 0; i < 1000; i++ {
    wg.Add(1)
    go func() {
        defer wg.Done()
        counter.Inc("requests")
    }()
}
wg.Wait()

fmt.Println(counter.Get("requests")) // 1000
\`\`\`

**Constraints:**
- Must use sync.RWMutex
- Inc must use Lock (write operation)
- Get must use RLock (read operation)
- Always defer unlock`,
	initialCode: `package syncx

import "sync"

type SafeCounter struct {
	mu sync.RWMutex
	m  map[string]int
}

func NewSafeCounter() *SafeCounter {
	return &SafeCounter{
		m: make(map[string]int),
	}
}

// TODO: Implement Inc
// Lock mutex, increment map value, unlock
func (s *SafeCounter) Inc(k string) {
	// TODO: Implement
}

// TODO: Implement Get
// RLock mutex, read value, RUnlock
func (s *SafeCounter) Get(k string) int {
	return 0 // TODO: Implement
}`,
	solutionCode: `package syncx

import "sync"

type SafeCounter struct {
	mu sync.RWMutex
	m  map[string]int
}

func NewSafeCounter() *SafeCounter {
	return &SafeCounter{
		m: make(map[string]int),
	}
}

func (s *SafeCounter) Inc(k string) {
	s.mu.Lock()	// acquire write lock
	defer s.mu.Unlock()	// ensure unlock on return
	s.m[k]++	// safely increment
}

func (s *SafeCounter) Get(k string) int {
	s.mu.RLock()	// acquire read lock (allows concurrent readers)
	defer s.mu.RUnlock()	// ensure unlock on return
	return s.m[k]	// safely read value
}`,
	testCode: `package syncx

import (
	"sync"
	"testing"
)

func TestSafeCounterIncSingleKey(t *testing.T) {
	counter := NewSafeCounter()
	counter.Inc("test")
	result := counter.Get("test")
	if result != 1 {
		t.Errorf("expected 1, got %d", result)
	}
}

func TestSafeCounterIncMultipleTimes(t *testing.T) {
	counter := NewSafeCounter()
	counter.Inc("test")
	counter.Inc("test")
	counter.Inc("test")
	result := counter.Get("test")
	if result != 3 {
		t.Errorf("expected 3, got %d", result)
	}
}

func TestSafeCounterMultipleKeys(t *testing.T) {
	counter := NewSafeCounter()
	counter.Inc("key1")
	counter.Inc("key2")
	counter.Inc("key1")
	if counter.Get("key1") != 2 {
		t.Errorf("expected key1=2, got %d", counter.Get("key1"))
	}
	if counter.Get("key2") != 1 {
		t.Errorf("expected key2=1, got %d", counter.Get("key2"))
	}
}

func TestSafeCounterGetNonExistentKey(t *testing.T) {
	counter := NewSafeCounter()
	result := counter.Get("nonexistent")
	if result != 0 {
		t.Errorf("expected 0 for nonexistent key, got %d", result)
	}
}

func TestSafeCounterConcurrentInc(t *testing.T) {
	counter := NewSafeCounter()
	var wg sync.WaitGroup
	n := 1000
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			counter.Inc("concurrent")
		}()
	}
	wg.Wait()
	result := counter.Get("concurrent")
	if result != n {
		t.Errorf("expected %d, got %d", n, result)
	}
}

func TestSafeCounterConcurrentIncMultipleKeys(t *testing.T) {
	counter := NewSafeCounter()
	var wg sync.WaitGroup
	keys := []string{"a", "b", "c"}
	incPerKey := 100
	for _, key := range keys {
		for i := 0; i < incPerKey; i++ {
			wg.Add(1)
			k := key
			go func() {
				defer wg.Done()
				counter.Inc(k)
			}()
		}
	}
	wg.Wait()
	for _, key := range keys {
		result := counter.Get(key)
		if result != incPerKey {
			t.Errorf("expected %d for key %s, got %d", incPerKey, key, result)
		}
	}
}

func TestSafeCounterConcurrentReadWrite(t *testing.T) {
	counter := NewSafeCounter()
	var wg sync.WaitGroup
	writes := 500
	reads := 500

	for i := 0; i < writes; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			counter.Inc("mixed")
		}()
	}

	for i := 0; i < reads; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			counter.Get("mixed")
		}()
	}

	wg.Wait()
	result := counter.Get("mixed")
	if result != writes {
		t.Errorf("expected %d, got %d", writes, result)
	}
}

func TestSafeCounterZeroValue(t *testing.T) {
	counter := NewSafeCounter()
	result := counter.Get("zero")
	if result != 0 {
		t.Errorf("expected 0, got %d", result)
	}
}

func TestSafeCounterIncAfterGet(t *testing.T) {
	counter := NewSafeCounter()
	counter.Inc("test")
	val1 := counter.Get("test")
	counter.Inc("test")
	val2 := counter.Get("test")
	if val1 != 1 || val2 != 2 {
		t.Errorf("expected val1=1, val2=2, got val1=%d, val2=%d", val1, val2)
	}
}

func TestSafeCounterMultipleGoroutinesSameKey(t *testing.T) {
	counter := NewSafeCounter()
	var wg sync.WaitGroup
	goroutines := 100
	incrementsPerGoroutine := 10

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < incrementsPerGoroutine; j++ {
				counter.Inc("shared")
			}
		}()
	}

	wg.Wait()
	expected := goroutines * incrementsPerGoroutine
	result := counter.Get("shared")
	if result != expected {
		t.Errorf("expected %d, got %d", expected, result)
	}
}`,
			hint1: `In Inc, use s.mu.Lock() and defer s.mu.Unlock() before modifying s.m[k].`,
			hint2: `In Get, use s.mu.RLock() and defer s.mu.RUnlock() before reading s.m[k].`,
			whyItMatters: `Thread-safe data structures are fundamental to concurrent Go programming. RWMutex provides better performance than regular Mutex for read-heavy workloads, allowing multiple concurrent readers while ensuring exclusive write access.

**Production Pattern:**

\`\`\`go
type SafeCounter struct {
    mu sync.RWMutex
    m  map[string]int
}

func (s *SafeCounter) Inc(k string) {
    s.mu.Lock()                    // acquire write lock
    defer s.mu.Unlock()            // ensure unlock on return
    s.m[k]++                       // safely increment
}

func (s *SafeCounter) Get(k string) int {
    s.mu.RLock()                   // acquire read lock (allows concurrent readers)
    defer s.mu.RUnlock()           // ensure unlock on return
    return s.m[k]                  // safely read value
}
\`\`\`

**Practical Benefits:**
- Multiple goroutines can safely read simultaneously
- Exclusive lock only for writes
- Better performance for read-heavy workloads than regular Mutex
- Prevents data races and incorrect behavior
- Critical for high-performance concurrent systems`,	order: 0,
	translations: {
		ru: {
			title: 'Потокобезопасный счётчик',
			solutionCode: `package syncx

import "sync"

type SafeCounter struct {
	mu sync.RWMutex
	m  map[string]int
}

func NewSafeCounter() *SafeCounter {
	return &SafeCounter{
		m: make(map[string]int),
	}
}

func (s *SafeCounter) Inc(k string) {
	s.mu.Lock()	// захватываем блокировку записи
	defer s.mu.Unlock()	// гарантируем разблокировку при возврате
	s.m[k]++	// безопасно инкрементируем
}

func (s *SafeCounter) Get(k string) int {
	s.mu.RLock()	// захватываем блокировку чтения (позволяет конкурентных читателей)
	defer s.mu.RUnlock()	// гарантируем разблокировку при возврате
	return s.m[k]	// безопасно читаем значение
}`,
			description: `Реализуйте потокобезопасный счетчик используя sync.RWMutex для конкурентного доступа.

**Требования:**
1. **SafeCounter**: Хранение map[string]int с защитой RWMutex
2. **Inc(key)**: Атомарное увеличение счетчика для ключа
3. **Get(key)**: Безопасное чтение значения счетчика
4. **Потокобезопасность**: Предотвращение гонок данных с помощью правильной блокировки

**Паттерн реализации:**
\`\`\`go
type SafeCounter struct {
    mu sync.RWMutex
    m  map[string]int
}

func (s *SafeCounter) Inc(k string) {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.m[k]++
}

func (s *SafeCounter) Get(k string) int {
    s.mu.RLock()
    defer s.mu.RUnlock()
    return s.m[k]
}
\`\`\`

**Пример использования:**
\`\`\`go
counter := NewSafeCounter()

// Конкурентные увеличения
var wg sync.WaitGroup
for i := 0; i < 1000; i++ {
    wg.Add(1)
    go func() {
        defer wg.Done()
        counter.Inc("requests")
    }()
}
wg.Wait()

fmt.Println(counter.Get("requests")) // 1000
\`\`\`

**Ограничения:**
- Необходимо использовать sync.RWMutex
- Inc должен использовать Lock (операция записи)
- Get должен использовать RLock (операция чтения)
- Всегда использовать defer unlock`,
			hint1: `В Inc используйте s.mu.Lock() и defer s.mu.Unlock() перед изменением s.m[k].`,
			hint2: `В Get используйте s.mu.RLock() и defer s.mu.RUnlock() перед чтением s.m[k].`,
			whyItMatters: `Потокобезопасные структуры данных фундаментальны для конкурентного Go программирования. RWMutex обеспечивает лучшую производительность чем обычный Mutex для read-heavy нагрузок.

**Почему важно:**

**Продакшен паттерн:**
\`\`\`go
type SafeCounter struct {
    mu sync.RWMutex
    m  map[string]int
}

func (s *SafeCounter) Inc(k string) {
    s.mu.Lock()                    // захватываем блокировку записи
    defer s.mu.Unlock()            // гарантируем разблокировку при возврате
    s.m[k]++                       // безопасно инкрементируем
}

func (s *SafeCounter) Get(k string) int {
    s.mu.RLock()                   // захватываем блокировку чтения (позволяет конкурентных читателей)
    defer s.mu.RUnlock()           // гарантируем разблокировку при возврате
    return s.m[k]                  // безопасно читаем значение
}
\`\`\`

**Практические преимущества:**
- Множественные горутины могут безопасно читать одновременно
- Эксклюзивная блокировка только для записи
- Лучшая производительность для read-heavy workloads чем обычный Mutex
- Предотвращает data races и некорректное поведение
- Критично для высокопроизводительных конкурентных систем`
		},
		uz: {
			title: `Thread-safe hisoblagich`,
			solutionCode: `package syncx

import "sync"

type SafeCounter struct {
	mu sync.RWMutex
	m  map[string]int
}

func NewSafeCounter() *SafeCounter {
	return &SafeCounter{
		m: make(map[string]int),
	}
}

func (s *SafeCounter) Inc(k string) {
	s.mu.Lock()	// yozish qulfini olamiz
	defer s.mu.Unlock()	// qaytishda qulfni ochishni kafolatlaymiz
	s.m[k]++	// xavfsiz oshiramiz
}

func (s *SafeCounter) Get(k string) int {
	s.mu.RLock()	// o'qish qulfini olamiz (parallel o'quvchilarga ruxsat beradi)
	defer s.mu.RUnlock()	// qaytishda qulfni ochishni kafolatlaymiz
	return s.m[k]	// qiymatni xavfsiz o'qiymiz
}`,
			description: `Parallel kirish uchun sync.RWMutex ishlatib thread-safe hisoblagichni amalga oshiring.

**Talablar:**
1. **SafeCounter**: RWMutex himoyasi bilan map[string]int saqlash
2. **Inc(key)**: Kalit uchun hisoblagichni atomik oshirish
3. **Get(key)**: Hisoblagich qiymatini xavfsiz o'qish
4. **Thread Xavfsizligi**: To'g'ri qulflash bilan data race larni oldini olish

**Amalga oshirish patterni:**
\`\`\`go
type SafeCounter struct {
    mu sync.RWMutex
    m  map[string]int
}

func (s *SafeCounter) Inc(k string) {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.m[k]++
}

func (s *SafeCounter) Get(k string) int {
    s.mu.RLock()
    defer s.mu.RUnlock()
    return s.m[k]
}
\`\`\`

**Cheklovlar:**
- sync.RWMutex ishlatish kerak
- Inc Lock ishlatishi kerak (yozish operatsiyasi)
- Get RLock ishlatishi kerak (o'qish operatsiyasi)
- Har doim defer unlock`,
			hint1: `Inc da s.m[k] ni o'zgartirishdan oldin s.mu.Lock() va defer s.mu.Unlock() ishlating.`,
			hint2: `Get da s.m[k] ni o'qishdan oldin s.mu.RLock() va defer s.mu.RUnlock() ishlating.`,
			whyItMatters: `Thread-safe ma'lumotlar strukturalari parallel Go dasturlashning asosidir. RWMutex oddiy Mutex dan ko'ra o'qishga og'ir yuklamalar uchun yaxshiroq ishlash ta'minlaydi, bir vaqtda bir nechta o'quvchilarga ruxsat beradi va eksklyuziv yozish kirishini ta'minlaydi.

**Nima uchun bu muhim:**

**Ishlab chiqarish patterni:**
\`\`\`go
type SafeCounter struct {
    mu sync.RWMutex
    m  map[string]int
}

func (s *SafeCounter) Inc(k string) {
    s.mu.Lock()                    // yozish qulfini olamiz
    defer s.mu.Unlock()            // qaytishda qulfni ochishni kafolatlaymiz
    s.m[k]++                       // xavfsiz oshiramiz
}

func (s *SafeCounter) Get(k string) int {
    s.mu.RLock()                   // o'qish qulfini olamiz (parallel o'quvchilarga ruxsat beradi)
    defer s.mu.RUnlock()           // qaytishda qulfni ochishni kafolatlaymiz
    return s.m[k]                  // qiymatni xavfsiz o'qiymiz
}
\`\`\`

**Amaliy foydalari:**
- Bir nechta goroutine lar bir vaqtda xavfsiz o'qishi mumkin
- Yozish uchun faqat eksklyuziv qulflash
- O'qishga og'ir yuklamalar uchun oddiy Mutex dan yaxshiroq ishlash
- Data race va noto'g'ri xatti-harakatni oldini oladi
- Yuqori ishlashli parallel tizimlar uchun muhim`
		}
	}
};

export default task;
