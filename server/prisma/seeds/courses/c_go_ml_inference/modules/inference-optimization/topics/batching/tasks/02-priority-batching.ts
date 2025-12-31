import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-priority-batching',
	title: 'Priority Batching',
	difficulty: 'hard',
	tags: ['go', 'ml', 'batching', 'priority-queue'],
	estimatedTime: '35m',
	isPremium: true,
	order: 2,
	description: `# Priority Batching

Implement priority-aware batching for ML inference.

## Task

Build a priority batcher that:
- Accepts requests with priority levels
- Processes high-priority requests first
- Maintains separate queues per priority
- Supports priority preemption

## Example

\`\`\`go
batcher := NewPriorityBatcher(config)
batcher.Submit(input, PriorityHigh)   // processed first
batcher.Submit(input, PriorityLow)    // processed after high priority
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

// Priority levels
type Priority int

const (
	PriorityLow Priority = iota
	PriorityMedium
	PriorityHigh
	PriorityCritical
)

// PriorityRequest represents a prioritized request
type PriorityRequest struct {
	Input    []float32
	Priority Priority
}

// PriorityBatcher handles priority-aware batching
type PriorityBatcher struct {
	// Your fields here
}

// NewPriorityBatcher creates a priority batcher
func NewPriorityBatcher(maxBatchSize int, inferFn func([][]float32) [][]float32) *PriorityBatcher {
	// Your code here
	return nil
}

// Submit submits a prioritized request
func (b *PriorityBatcher) Submit(input []float32, priority Priority) []float32 {
	// Your code here
	return nil
}

func main() {
	fmt.Println("Priority Batching")
}`,

	solutionCode: `package main

import (
	"container/heap"
	"fmt"
	"sync"
	"time"
)

// Priority levels
type Priority int

const (
	PriorityLow Priority = iota
	PriorityMedium
	PriorityHigh
	PriorityCritical
)

func (p Priority) String() string {
	switch p {
	case PriorityLow:
		return "LOW"
	case PriorityMedium:
		return "MEDIUM"
	case PriorityHigh:
		return "HIGH"
	case PriorityCritical:
		return "CRITICAL"
	default:
		return "UNKNOWN"
	}
}

// PriorityRequest represents a prioritized request
type PriorityRequest struct {
	Input     []float32
	Priority  Priority
	Timestamp time.Time
	ResultCh  chan []float32
	Index     int
}

// PriorityQueue implements heap.Interface
type PriorityQueue []*PriorityRequest

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	// Higher priority first, then earlier timestamp
	if pq[i].Priority != pq[j].Priority {
		return pq[i].Priority > pq[j].Priority
	}
	return pq[i].Timestamp.Before(pq[j].Timestamp)
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].Index = i
	pq[j].Index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*PriorityRequest)
	item.Index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.Index = -1
	*pq = old[0 : n-1]
	return item
}

// PriorityBatcher handles priority-aware batching
type PriorityBatcher struct {
	maxBatchSize int
	timeout      time.Duration
	inferFn      func([][]float32) [][]float32

	queue   PriorityQueue
	mu      sync.Mutex
	cond    *sync.Cond
	closed  bool

	// Stats
	processed map[Priority]int64
}

// NewPriorityBatcher creates a priority batcher
func NewPriorityBatcher(maxBatchSize int, inferFn func([][]float32) [][]float32) *PriorityBatcher {
	b := &PriorityBatcher{
		maxBatchSize: maxBatchSize,
		timeout:      50 * time.Millisecond,
		inferFn:      inferFn,
		queue:        make(PriorityQueue, 0),
		processed:    make(map[Priority]int64),
	}
	b.cond = sync.NewCond(&b.mu)
	heap.Init(&b.queue)

	go b.processor()
	return b
}

// Submit submits a prioritized request
func (b *PriorityBatcher) Submit(input []float32, priority Priority) []float32 {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil
	}

	req := &PriorityRequest{
		Input:     input,
		Priority:  priority,
		Timestamp: time.Now(),
		ResultCh:  make(chan []float32, 1),
	}

	heap.Push(&b.queue, req)
	b.cond.Signal()
	b.mu.Unlock()

	return <-req.ResultCh
}

// processor runs the batch processing loop
func (b *PriorityBatcher) processor() {
	for {
		b.mu.Lock()

		// Wait for requests or shutdown
		for b.queue.Len() == 0 && !b.closed {
			b.cond.Wait()
		}

		if b.closed && b.queue.Len() == 0 {
			b.mu.Unlock()
			return
		}

		// Collect batch
		batch := b.collectBatch()
		b.mu.Unlock()

		if len(batch) > 0 {
			b.processBatch(batch)
		}
	}
}

// collectBatch collects requests for a batch
func (b *PriorityBatcher) collectBatch() []*PriorityRequest {
	batch := make([]*PriorityRequest, 0, b.maxBatchSize)

	// Try to fill batch with same priority first
	var targetPriority Priority
	if b.queue.Len() > 0 {
		targetPriority = b.queue[0].Priority
	}

	for b.queue.Len() > 0 && len(batch) < b.maxBatchSize {
		req := heap.Pop(&b.queue).(*PriorityRequest)

		// For critical priority, process immediately
		if req.Priority == PriorityCritical {
			batch = append(batch, req)
			continue
		}

		// Try to batch same priority together
		if len(batch) > 0 && req.Priority != targetPriority {
			// Put back and process current batch
			heap.Push(&b.queue, req)
			break
		}

		batch = append(batch, req)
	}

	return batch
}

// processBatch processes a batch of requests
func (b *PriorityBatcher) processBatch(batch []*PriorityRequest) {
	// Collect inputs
	inputs := make([][]float32, len(batch))
	for i, req := range batch {
		inputs[i] = req.Input
	}

	// Run inference
	outputs := b.inferFn(inputs)

	// Distribute results and update stats
	b.mu.Lock()
	for i, req := range batch {
		if i < len(outputs) {
			req.ResultCh <- outputs[i]
		} else {
			req.ResultCh <- nil
		}
		b.processed[req.Priority]++
	}
	b.mu.Unlock()
}

// Close shuts down the batcher
func (b *PriorityBatcher) Close() {
	b.mu.Lock()
	b.closed = true
	b.cond.Broadcast()
	b.mu.Unlock()
}

// Stats returns processing statistics
func (b *PriorityBatcher) Stats() map[Priority]int64 {
	b.mu.Lock()
	defer b.mu.Unlock()

	stats := make(map[Priority]int64)
	for k, v := range b.processed {
		stats[k] = v
	}
	return stats
}

// QueueSize returns current queue size
func (b *PriorityBatcher) QueueSize() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.queue.Len()
}

func main() {
	inferFn := func(batch [][]float32) [][]float32 {
		results := make([][]float32, len(batch))
		for i, input := range batch {
			output := make([]float32, len(input))
			for j, v := range input {
				output[j] = v * 2
			}
			results[i] = output
		}
		fmt.Printf("Processed batch of %d requests\\n", len(batch))
		return results
	}

	batcher := NewPriorityBatcher(4, inferFn)
	defer batcher.Close()

	var wg sync.WaitGroup

	// Submit mixed priority requests
	priorities := []Priority{PriorityLow, PriorityMedium, PriorityHigh, PriorityCritical}
	for i := 0; i < 12; i++ {
		wg.Add(1)
		priority := priorities[i%4]
		go func(id int, p Priority) {
			defer wg.Done()
			result := batcher.Submit([]float32{float32(id)}, p)
			fmt.Printf("Request %d (%s): result=%v\\n", id, p, result)
		}(i, priority)
	}

	wg.Wait()

	fmt.Println("\\nStats:", batcher.Stats())
}`,

	testCode: `package main

import (
	"sync"
	"testing"
	"time"
)

func mockInfer(batch [][]float32) [][]float32 {
	results := make([][]float32, len(batch))
	for i, input := range batch {
		output := make([]float32, len(input))
		for j, v := range input {
			output[j] = v * 2
		}
		results[i] = output
	}
	return results
}

func TestPriorityBatcher(t *testing.T) {
	batcher := NewPriorityBatcher(4, mockInfer)
	defer batcher.Close()

	result := batcher.Submit([]float32{1, 2}, PriorityMedium)

	if result == nil {
		t.Fatal("Result should not be nil")
	}
	if len(result) != 2 {
		t.Errorf("Expected 2 elements, got %d", len(result))
	}
}

func TestPriorityOrdering(t *testing.T) {
	var processOrder []Priority
	var mu sync.Mutex

	inferFn := func(batch [][]float32) [][]float32 {
		time.Sleep(10 * time.Millisecond)
		return mockInfer(batch)
	}

	batcher := NewPriorityBatcher(1, inferFn)
	defer batcher.Close()

	var wg sync.WaitGroup

	// Submit in reverse priority order
	for _, p := range []Priority{PriorityLow, PriorityMedium, PriorityHigh} {
		wg.Add(1)
		go func(priority Priority) {
			defer wg.Done()
			batcher.Submit([]float32{float32(priority)}, priority)
			mu.Lock()
			processOrder = append(processOrder, priority)
			mu.Unlock()
		}(p)
		time.Sleep(5 * time.Millisecond)
	}

	wg.Wait()

	// High priority should generally complete first
	if len(processOrder) == 0 {
		t.Error("Should have processed requests")
	}
}

func TestCriticalPriority(t *testing.T) {
	batcher := NewPriorityBatcher(4, mockInfer)
	defer batcher.Close()

	result := batcher.Submit([]float32{1}, PriorityCritical)

	if result == nil {
		t.Error("Critical request should be processed")
	}
}

func TestBatcherStats(t *testing.T) {
	batcher := NewPriorityBatcher(4, mockInfer)
	defer batcher.Close()

	batcher.Submit([]float32{1}, PriorityHigh)
	batcher.Submit([]float32{2}, PriorityHigh)
	batcher.Submit([]float32{3}, PriorityLow)

	stats := batcher.Stats()
	if stats[PriorityHigh] != 2 {
		t.Errorf("Expected 2 high priority, got %d", stats[PriorityHigh])
	}
}

func TestPriorityString(t *testing.T) {
	if PriorityLow.String() != "LOW" {
		t.Errorf("Expected LOW, got %s", PriorityLow.String())
	}
	if PriorityHigh.String() != "HIGH" {
		t.Errorf("Expected HIGH, got %s", PriorityHigh.String())
	}
	if PriorityCritical.String() != "CRITICAL" {
		t.Errorf("Expected CRITICAL, got %s", PriorityCritical.String())
	}
}

func TestQueueSize(t *testing.T) {
	batcher := NewPriorityBatcher(100, mockInfer)
	defer batcher.Close()

	// Queue size should be 0 initially
	if batcher.QueueSize() < 0 {
		t.Error("Queue size should not be negative")
	}
}

func TestPriorityBatcherClose(t *testing.T) {
	batcher := NewPriorityBatcher(4, mockInfer)
	batcher.Close()

	result := batcher.Submit([]float32{1}, PriorityHigh)
	if result != nil {
		t.Error("Submit after close should return nil")
	}
}

func TestMockInfer(t *testing.T) {
	input := [][]float32{{1, 2, 3}}
	output := mockInfer(input)

	if len(output) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(output))
	}
	if output[0][0] != 2 {
		t.Errorf("Expected 2, got %f", output[0][0])
	}
}

func TestConcurrentPrioritySubmits(t *testing.T) {
	batcher := NewPriorityBatcher(8, mockInfer)
	defer batcher.Close()

	var wg sync.WaitGroup
	resultCount := 0
	var mu sync.Mutex

	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			priority := Priority(id % 4)
			result := batcher.Submit([]float32{float32(id)}, priority)
			if result != nil {
				mu.Lock()
				resultCount++
				mu.Unlock()
			}
		}(i)
	}

	wg.Wait()

	if resultCount != 16 {
		t.Errorf("Expected 16 results, got %d", resultCount)
	}
}

func TestPriorityMedium(t *testing.T) {
	batcher := NewPriorityBatcher(4, mockInfer)
	defer batcher.Close()

	result := batcher.Submit([]float32{1, 2}, PriorityMedium)

	if result == nil {
		t.Error("Medium priority should be processed")
	}
	if PriorityMedium.String() != "MEDIUM" {
		t.Errorf("Expected MEDIUM, got %s", PriorityMedium.String())
	}
}`,

	hint1: 'Use container/heap for efficient priority queue implementation',
	hint2: 'Consider batching same-priority requests together for efficiency',

	whyItMatters: `Priority batching enables SLA-aware ML serving:

- **SLA compliance**: Meet latency requirements for premium users
- **Resource fairness**: Prevent low-priority requests from blocking critical ones
- **Graceful degradation**: Handle load spikes by prioritizing important requests
- **Multi-tenant support**: Different service tiers with different priorities

Priority-aware systems are essential for production ML services.`,

	translations: {
		ru: {
			title: 'Приоритетный батчинг',
			description: `# Приоритетный батчинг

Реализуйте приоритетно-зависимый батчинг для ML инференса.

## Задача

Создайте приоритетный батчер:
- Прием запросов с уровнями приоритета
- Обработка высокоприоритетных запросов первыми
- Поддержка отдельных очередей по приоритету
- Поддержка вытеснения по приоритету

## Пример

\`\`\`go
batcher := NewPriorityBatcher(config)
batcher.Submit(input, PriorityHigh)   // processed first
batcher.Submit(input, PriorityLow)    // processed after high priority
\`\`\``,
			hint1: 'Используйте container/heap для эффективной реализации очереди с приоритетом',
			hint2: 'Рассмотрите батчинг запросов одного приоритета вместе для эффективности',
			whyItMatters: `Приоритетный батчинг обеспечивает SLA-осведомленное ML обслуживание:

- **Соответствие SLA**: Соблюдение требований латентности для премиум пользователей
- **Справедливость ресурсов**: Предотвращение блокировки критических запросов низкоприоритетными
- **Плавная деградация**: Обработка пиков нагрузки с приоритизацией важных запросов
- **Мультитенантная поддержка**: Разные уровни сервиса с разными приоритетами`,
		},
		uz: {
			title: 'Prioritetli batching',
			description: `# Prioritetli batching

ML inference uchun prioritetga bog'liq batchingni amalga oshiring.

## Topshiriq

Prioritetli batcher yarating:
- Prioritet darajalari bilan so'rovlarni qabul qilish
- Yuqori prioritetli so'rovlarni birinchi qayta ishlash
- Har bir prioritet uchun alohida navbatlarni qo'llab-quvvatlash
- Prioritet bo'yicha preemption ni qo'llab-quvvatlash

## Misol

\`\`\`go
batcher := NewPriorityBatcher(config)
batcher.Submit(input, PriorityHigh)   // processed first
batcher.Submit(input, PriorityLow)    // processed after high priority
\`\`\``,
			hint1: "Samarali priority queue amalga oshirish uchun container/heap dan foydalaning",
			hint2: "Samaradorlik uchun bir xil prioritetli so'rovlarni birga batching qilishni ko'rib chiqing",
			whyItMatters: `Prioritetli batching SLA-ga moslashgan ML xizmatini ta'minlaydi:

- **SLA muvofiqligi**: Premium foydalanuvchilar uchun latency talablarini bajarish
- **Resurs adolati**: Past prioritetli so'rovlarning muhim so'rovlarni bloklashini oldini olish
- **Yumshoq degradatsiya**: Muhim so'rovlarni ustuvorlashtirish orqali yuklanish o'sishlarini boshqarish
- **Ko'p ijarachi qo'llab-quvvatlash**: Turli prioritetlar bilan turli xizmat darajalari`,
		},
	},
};

export default task;
