import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-dynamic-batching',
	title: 'Dynamic Batching',
	difficulty: 'medium',
	tags: ['go', 'ml', 'batching', 'optimization'],
	estimatedTime: '30m',
	isPremium: true,
	order: 1,
	description: `# Dynamic Batching

Implement dynamic request batching for ML inference optimization.

## Task

Build a dynamic batcher that:
- Collects inference requests over a time window
- Batches requests up to a maximum size
- Triggers inference when batch is full or timeout expires
- Returns results to individual callers

## Example

\`\`\`go
batcher := NewDynamicBatcher(maxBatchSize: 32, timeout: 10*time.Millisecond)
result := batcher.Submit(input) // blocks until batch processes
\`\`\``,

	initialCode: `package main

import (
	"fmt"
	"sync"
	"time"
)

// InferenceRequest represents a single inference request
type InferenceRequest struct {
	ID    int
	Input []float32
}

// InferenceResult represents inference output
type InferenceResult struct {
	ID     int
	Output []float32
}

// DynamicBatcher batches inference requests
type DynamicBatcher struct {
	// Your fields here
}

// NewDynamicBatcher creates a dynamic batcher
func NewDynamicBatcher(maxBatchSize int, timeout time.Duration, inferFn func([][]float32) [][]float32) *DynamicBatcher {
	// Your code here
	return nil
}

// Submit submits a request and waits for result
func (b *DynamicBatcher) Submit(input []float32) []float32 {
	// Your code here
	return nil
}

// Close shuts down the batcher
func (b *DynamicBatcher) Close() {
	// Your code here
}

func main() {
	fmt.Println("Dynamic Batching")
}`,

	solutionCode: `package main

import (
	"fmt"
	"sync"
	"time"
)

// InferenceRequest represents a single inference request
type InferenceRequest struct {
	ID    int
	Input []float32
}

// InferenceResult represents inference output
type InferenceResult struct {
	ID     int
	Output []float32
}

// pendingRequest holds request and result channel
type pendingRequest struct {
	input  []float32
	result chan []float32
}

// DynamicBatcher batches inference requests
type DynamicBatcher struct {
	maxBatchSize int
	timeout      time.Duration
	inferFn      func([][]float32) [][]float32

	pending  []pendingRequest
	mu       sync.Mutex
	timer    *time.Timer
	closed   bool
	closeCh  chan struct{}
}

// NewDynamicBatcher creates a dynamic batcher
func NewDynamicBatcher(maxBatchSize int, timeout time.Duration, inferFn func([][]float32) [][]float32) *DynamicBatcher {
	b := &DynamicBatcher{
		maxBatchSize: maxBatchSize,
		timeout:      timeout,
		inferFn:      inferFn,
		pending:      make([]pendingRequest, 0, maxBatchSize),
		closeCh:      make(chan struct{}),
	}
	return b
}

// Submit submits a request and waits for result
func (b *DynamicBatcher) Submit(input []float32) []float32 {
	b.mu.Lock()

	if b.closed {
		b.mu.Unlock()
		return nil
	}

	resultCh := make(chan []float32, 1)
	b.pending = append(b.pending, pendingRequest{
		input:  input,
		result: resultCh,
	})

	// Start timer on first request
	if len(b.pending) == 1 {
		b.timer = time.AfterFunc(b.timeout, func() {
			b.processBatch()
		})
	}

	// Process immediately if batch is full
	if len(b.pending) >= b.maxBatchSize {
		if b.timer != nil {
			b.timer.Stop()
		}
		b.processBatchLocked()
		b.mu.Unlock()
	} else {
		b.mu.Unlock()
	}

	// Wait for result
	return <-resultCh
}

// processBatch processes the current batch (with lock acquisition)
func (b *DynamicBatcher) processBatch() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.processBatchLocked()
}

// processBatchLocked processes batch (caller must hold lock)
func (b *DynamicBatcher) processBatchLocked() {
	if len(b.pending) == 0 {
		return
	}

	// Collect inputs
	batch := make([][]float32, len(b.pending))
	for i, req := range b.pending {
		batch[i] = req.input
	}

	// Run inference
	outputs := b.inferFn(batch)

	// Distribute results
	for i, req := range b.pending {
		if i < len(outputs) {
			req.result <- outputs[i]
		} else {
			req.result <- nil
		}
	}

	// Clear pending
	b.pending = b.pending[:0]
}

// Close shuts down the batcher
func (b *DynamicBatcher) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return
	}

	b.closed = true
	if b.timer != nil {
		b.timer.Stop()
	}

	// Process remaining requests
	b.processBatchLocked()
	close(b.closeCh)
}

// Stats returns batcher statistics
type BatcherStats struct {
	TotalRequests   int64
	TotalBatches    int64
	AvgBatchSize    float64
}

// AdaptiveBatcher adjusts batch size based on load
type AdaptiveBatcher struct {
	*DynamicBatcher
	minBatchSize int
	stats        BatcherStats
	mu           sync.Mutex
}

// NewAdaptiveBatcher creates an adaptive batcher
func NewAdaptiveBatcher(minSize, maxSize int, timeout time.Duration, inferFn func([][]float32) [][]float32) *AdaptiveBatcher {
	return &AdaptiveBatcher{
		DynamicBatcher: NewDynamicBatcher(maxSize, timeout, inferFn),
		minBatchSize:   minSize,
	}
}

func main() {
	// Mock inference function
	inferFn := func(batch [][]float32) [][]float32 {
		results := make([][]float32, len(batch))
		for i, input := range batch {
			// Simulate model: multiply by 2
			output := make([]float32, len(input))
			for j, v := range input {
				output[j] = v * 2
			}
			results[i] = output
		}
		fmt.Printf("Processed batch of %d requests\\n", len(batch))
		return results
	}

	batcher := NewDynamicBatcher(4, 50*time.Millisecond, inferFn)
	defer batcher.Close()

	// Submit requests concurrently
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			input := []float32{float32(id), float32(id * 2)}
			result := batcher.Submit(input)
			fmt.Printf("Request %d: input=%v, output=%v\\n", id, input, result)
		}(i)
	}

	wg.Wait()
}`,

	testCode: `package main

import (
	"sync"
	"testing"
	"time"
)

func mockInferFn(batch [][]float32) [][]float32 {
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

func TestDynamicBatcher(t *testing.T) {
	batcher := NewDynamicBatcher(4, 100*time.Millisecond, mockInferFn)
	defer batcher.Close()

	result := batcher.Submit([]float32{1, 2, 3})

	if result == nil {
		t.Fatal("Result should not be nil")
	}
	if len(result) != 3 {
		t.Errorf("Expected 3 elements, got %d", len(result))
	}
}

func TestBatchingMultipleRequests(t *testing.T) {
	batchSizes := make([]int, 0)
	var mu sync.Mutex

	inferFn := func(batch [][]float32) [][]float32 {
		mu.Lock()
		batchSizes = append(batchSizes, len(batch))
		mu.Unlock()
		return mockInferFn(batch)
	}

	batcher := NewDynamicBatcher(4, 50*time.Millisecond, inferFn)
	defer batcher.Close()

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			batcher.Submit([]float32{float32(id)})
		}(i)
	}
	wg.Wait()

	// Should have batched requests
	if len(batchSizes) == 0 {
		t.Error("Should have processed batches")
	}
}

func TestBatchTimeout(t *testing.T) {
	processed := make(chan struct{})

	inferFn := func(batch [][]float32) [][]float32 {
		close(processed)
		return mockInferFn(batch)
	}

	batcher := NewDynamicBatcher(100, 20*time.Millisecond, inferFn)
	defer batcher.Close()

	go batcher.Submit([]float32{1})

	select {
	case <-processed:
		// Success - batch processed on timeout
	case <-time.After(100 * time.Millisecond):
		t.Error("Batch should have been processed on timeout")
	}
}

func TestBatcherClose(t *testing.T) {
	batcher := NewDynamicBatcher(4, 100*time.Millisecond, mockInferFn)
	batcher.Close()

	result := batcher.Submit([]float32{1})
	if result != nil {
		t.Error("Submit after close should return nil")
	}
}

func TestBatchFullTrigger(t *testing.T) {
	batchProcessed := make(chan int, 10)

	inferFn := func(batch [][]float32) [][]float32 {
		batchProcessed <- len(batch)
		return mockInferFn(batch)
	}

	batcher := NewDynamicBatcher(2, 1*time.Second, inferFn)
	defer batcher.Close()

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		batcher.Submit([]float32{1})
	}()
	go func() {
		defer wg.Done()
		batcher.Submit([]float32{2})
	}()

	wg.Wait()

	// Check that batch was processed with 2 items
	select {
	case size := <-batchProcessed:
		if size != 2 {
			t.Errorf("Expected batch size 2, got %d", size)
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("Batch should be processed when full")
	}
}

func TestMockInferFn(t *testing.T) {
	input := [][]float32{{1, 2, 3}}
	output := mockInferFn(input)

	if len(output) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(output))
	}
	if output[0][0] != 2 {
		t.Errorf("Expected 2, got %f", output[0][0])
	}
}

func TestNewAdaptiveBatcher(t *testing.T) {
	batcher := NewAdaptiveBatcher(2, 8, 100*time.Millisecond, mockInferFn)
	defer batcher.Close()

	if batcher.minBatchSize != 2 {
		t.Errorf("Expected minBatchSize 2, got %d", batcher.minBatchSize)
	}
}

func TestConcurrentSubmits(t *testing.T) {
	batcher := NewDynamicBatcher(16, 50*time.Millisecond, mockInferFn)
	defer batcher.Close()

	var wg sync.WaitGroup
	results := make([][]float32, 20)

	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			results[id] = batcher.Submit([]float32{float32(id)})
		}(i)
	}

	wg.Wait()

	for i, r := range results {
		if r == nil {
			t.Errorf("Result %d should not be nil", i)
		}
	}
}

func TestDoubleClose(t *testing.T) {
	batcher := NewDynamicBatcher(4, 100*time.Millisecond, mockInferFn)
	batcher.Close()
	batcher.Close() // Should not panic

	// If we get here without panic, test passes
}

func TestPendingRequestStructure(t *testing.T) {
	// Test that pendingRequest struct can be created
	resultCh := make(chan []float32, 1)
	pr := pendingRequest{
		input:  []float32{1, 2, 3},
		result: resultCh,
	}

	if pr.input[0] != 1 || len(pr.input) != 3 {
		t.Error("pendingRequest input not set correctly")
	}
}`,

	hint1: 'Use channels to return results to individual callers',
	hint2: 'time.AfterFunc can trigger batch processing on timeout',

	whyItMatters: `Dynamic batching is essential for efficient ML inference:

- **GPU utilization**: Batching maximizes GPU throughput
- **Latency vs throughput**: Balance individual latency with overall throughput
- **Resource efficiency**: Process more requests with same resources
- **Cost optimization**: Reduce infrastructure costs in production

Proper batching can improve throughput 10x or more.`,

	translations: {
		ru: {
			title: 'Динамический батчинг',
			description: `# Динамический батчинг

Реализуйте динамический батчинг запросов для оптимизации ML инференса.

## Задача

Создайте динамический батчер:
- Сбор запросов инференса за временное окно
- Батчинг запросов до максимального размера
- Запуск инференса при заполнении батча или истечении таймаута
- Возврат результатов отдельным вызывающим

## Пример

\`\`\`go
batcher := NewDynamicBatcher(maxBatchSize: 32, timeout: 10*time.Millisecond)
result := batcher.Submit(input) // blocks until batch processes
\`\`\``,
			hint1: 'Используйте каналы для возврата результатов отдельным вызывающим',
			hint2: 'time.AfterFunc может запускать обработку батча по таймауту',
			whyItMatters: `Динамический батчинг необходим для эффективного ML инференса:

- **Утилизация GPU**: Батчинг максимизирует пропускную способность GPU
- **Латентность vs пропускная способность**: Баланс между латентностью и общей пропускной способностью
- **Эффективность ресурсов**: Обработка большего количества запросов с теми же ресурсами
- **Оптимизация стоимости**: Снижение затрат на инфраструктуру в продакшене`,
		},
		uz: {
			title: 'Dinamik batching',
			description: `# Dinamik batching

ML inference optimizatsiyasi uchun dinamik so'rovlar batchingini amalga oshiring.

## Topshiriq

Dinamik batcher yarating:
- Vaqt oynasi davomida inference so'rovlarini yig'ish
- So'rovlarni maksimal o'lchamgacha batching qilish
- Batch to'lganda yoki timeout tugaganda inference ni ishga tushirish
- Natijalarni alohida chaqiruvchilarga qaytarish

## Misol

\`\`\`go
batcher := NewDynamicBatcher(maxBatchSize: 32, timeout: 10*time.Millisecond)
result := batcher.Submit(input) // blocks until batch processes
\`\`\``,
			hint1: "Natijalarni alohida chaqiruvchilarga qaytarish uchun kanallardan foydalaning",
			hint2: 'time.AfterFunc timeout bo\'yicha batch qayta ishlashni ishga tushirishi mumkin',
			whyItMatters: `Dinamik batching samarali ML inference uchun zarur:

- **GPU utilizatsiyasi**: Batching GPU throughput ni maksimallashtiradi
- **Latency vs throughput**: Individual latency va umumiy throughput orasidagi muvozanat
- **Resurs samaradorligi**: Xuddi shu resurslar bilan ko'proq so'rovlarni qayta ishlash
- **Xarajatlarni optimallashtirish**: Production da infratuzilma xarajatlarini kamaytirish`,
		},
	},
};

export default task;
