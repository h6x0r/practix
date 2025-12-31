import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-adaptive-batching',
	title: 'Adaptive Batching',
	difficulty: 'hard',
	tags: ['go', 'ml', 'batching', 'adaptive'],
	estimatedTime: '40m',
	isPremium: true,
	order: 3,
	description: `# Adaptive Batching

Implement adaptive batching that adjusts parameters based on load.

## Task

Build an adaptive batcher that:
- Monitors request rate and latency
- Adjusts batch size dynamically
- Adjusts timeout based on queue depth
- Maintains target latency SLA

## Example

\`\`\`go
batcher := NewAdaptiveBatcher(config)
batcher.SetTargetLatency(50 * time.Millisecond)
// Automatically adjusts batch size and timeout
\`\`\``,

	initialCode: `package main

import (
	"fmt"
	"time"
)

// AdaptiveConfig holds adaptive batcher configuration
type AdaptiveConfig struct {
	MinBatchSize   int
	MaxBatchSize   int
	MinTimeout     time.Duration
	MaxTimeout     time.Duration
	TargetLatency  time.Duration
}

// AdaptiveBatcher adjusts batching parameters dynamically
type AdaptiveBatcher struct {
	// Your fields here
}

// NewAdaptiveBatcher creates an adaptive batcher
func NewAdaptiveBatcher(config AdaptiveConfig, inferFn func([][]float32) [][]float32) *AdaptiveBatcher {
	// Your code here
	return nil
}

// Submit submits a request
func (b *AdaptiveBatcher) Submit(input []float32) []float32 {
	// Your code here
	return nil
}

// GetCurrentConfig returns current adaptive parameters
func (b *AdaptiveBatcher) GetCurrentConfig() (batchSize int, timeout time.Duration) {
	// Your code here
	return 0, 0
}

func main() {
	fmt.Println("Adaptive Batching")
}`,

	solutionCode: `package main

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// AdaptiveConfig holds adaptive batcher configuration
type AdaptiveConfig struct {
	MinBatchSize   int
	MaxBatchSize   int
	MinTimeout     time.Duration
	MaxTimeout     time.Duration
	TargetLatency  time.Duration
	AdjustInterval time.Duration
}

// DefaultAdaptiveConfig returns sensible defaults
func DefaultAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{
		MinBatchSize:   1,
		MaxBatchSize:   64,
		MinTimeout:     5 * time.Millisecond,
		MaxTimeout:     100 * time.Millisecond,
		TargetLatency:  50 * time.Millisecond,
		AdjustInterval: 1 * time.Second,
	}
}

// LatencyTracker tracks request latencies
type LatencyTracker struct {
	samples    []time.Duration
	maxSamples int
	mu         sync.Mutex
}

// NewLatencyTracker creates a latency tracker
func NewLatencyTracker(maxSamples int) *LatencyTracker {
	return &LatencyTracker{
		samples:    make([]time.Duration, 0, maxSamples),
		maxSamples: maxSamples,
	}
}

// Record records a latency sample
func (t *LatencyTracker) Record(d time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(t.samples) >= t.maxSamples {
		t.samples = t.samples[1:]
	}
	t.samples = append(t.samples, d)
}

// P99 returns the 99th percentile latency
func (t *LatencyTracker) P99() time.Duration {
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(t.samples) == 0 {
		return 0
	}

	// Simple P99 calculation
	sorted := make([]time.Duration, len(t.samples))
	copy(sorted, t.samples)

	// Bubble sort for simplicity
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	idx := int(float64(len(sorted)) * 0.99)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// Average returns average latency
func (t *LatencyTracker) Average() time.Duration {
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(t.samples) == 0 {
		return 0
	}

	var sum time.Duration
	for _, s := range t.samples {
		sum += s
	}
	return sum / time.Duration(len(t.samples))
}

// pendingReq holds pending request info
type pendingReq struct {
	input    []float32
	resultCh chan []float32
	submitted time.Time
}

// AdaptiveBatcher adjusts batching parameters dynamically
type AdaptiveBatcher struct {
	config       AdaptiveConfig
	inferFn      func([][]float32) [][]float32

	currentBatchSize int
	currentTimeout   time.Duration

	pending      []pendingReq
	latencyTracker *LatencyTracker
	requestRate  float64
	lastAdjust   time.Time

	mu       sync.Mutex
	timer    *time.Timer
	closed   bool
	closeCh  chan struct{}
}

// NewAdaptiveBatcher creates an adaptive batcher
func NewAdaptiveBatcher(config AdaptiveConfig, inferFn func([][]float32) [][]float32) *AdaptiveBatcher {
	b := &AdaptiveBatcher{
		config:           config,
		inferFn:          inferFn,
		currentBatchSize: (config.MinBatchSize + config.MaxBatchSize) / 2,
		currentTimeout:   (config.MinTimeout + config.MaxTimeout) / 2,
		pending:          make([]pendingReq, 0),
		latencyTracker:   NewLatencyTracker(1000),
		lastAdjust:       time.Now(),
		closeCh:          make(chan struct{}),
	}

	go b.adjustLoop()
	return b
}

// Submit submits a request
func (b *AdaptiveBatcher) Submit(input []float32) []float32 {
	b.mu.Lock()

	if b.closed {
		b.mu.Unlock()
		return nil
	}

	resultCh := make(chan []float32, 1)
	b.pending = append(b.pending, pendingReq{
		input:     input,
		resultCh:  resultCh,
		submitted: time.Now(),
	})

	// Start timer on first request
	if len(b.pending) == 1 {
		b.timer = time.AfterFunc(b.currentTimeout, func() {
			b.processBatch()
		})
	}

	// Process if batch is full
	if len(b.pending) >= b.currentBatchSize {
		if b.timer != nil {
			b.timer.Stop()
		}
		b.processBatchLocked()
		b.mu.Unlock()
	} else {
		b.mu.Unlock()
	}

	return <-resultCh
}

// processBatch processes with lock acquisition
func (b *AdaptiveBatcher) processBatch() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.processBatchLocked()
}

// processBatchLocked processes batch (caller holds lock)
func (b *AdaptiveBatcher) processBatchLocked() {
	if len(b.pending) == 0 {
		return
	}

	batch := make([][]float32, len(b.pending))
	for i, req := range b.pending {
		batch[i] = req.input
	}

	// Run inference
	outputs := b.inferFn(batch)

	// Distribute results and record latencies
	now := time.Now()
	for i, req := range b.pending {
		latency := now.Sub(req.submitted)
		b.latencyTracker.Record(latency)

		if i < len(outputs) {
			req.resultCh <- outputs[i]
		} else {
			req.resultCh <- nil
		}
	}

	b.pending = b.pending[:0]
}

// adjustLoop periodically adjusts parameters
func (b *AdaptiveBatcher) adjustLoop() {
	ticker := time.NewTicker(b.config.AdjustInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			b.adjust()
		case <-b.closeCh:
			return
		}
	}
}

// adjust adapts batch size and timeout
func (b *AdaptiveBatcher) adjust() {
	b.mu.Lock()
	defer b.mu.Unlock()

	avgLatency := b.latencyTracker.Average()
	p99Latency := b.latencyTracker.P99()

	// If latency is too high, reduce batch size and timeout
	if p99Latency > b.config.TargetLatency {
		// Reduce batch size
		reduction := float64(p99Latency) / float64(b.config.TargetLatency)
		newSize := int(float64(b.currentBatchSize) / math.Sqrt(reduction))
		if newSize < b.config.MinBatchSize {
			newSize = b.config.MinBatchSize
		}
		b.currentBatchSize = newSize

		// Reduce timeout
		newTimeout := time.Duration(float64(b.currentTimeout) / math.Sqrt(reduction))
		if newTimeout < b.config.MinTimeout {
			newTimeout = b.config.MinTimeout
		}
		b.currentTimeout = newTimeout
	} else if avgLatency < b.config.TargetLatency/2 {
		// Latency is low, can increase batch size for throughput
		headroom := float64(b.config.TargetLatency) / float64(avgLatency+1)
		newSize := int(float64(b.currentBatchSize) * math.Sqrt(headroom))
		if newSize > b.config.MaxBatchSize {
			newSize = b.config.MaxBatchSize
		}
		b.currentBatchSize = newSize

		// Can also increase timeout
		newTimeout := time.Duration(float64(b.currentTimeout) * 1.1)
		if newTimeout > b.config.MaxTimeout {
			newTimeout = b.config.MaxTimeout
		}
		b.currentTimeout = newTimeout
	}

	b.lastAdjust = time.Now()
}

// GetCurrentConfig returns current adaptive parameters
func (b *AdaptiveBatcher) GetCurrentConfig() (batchSize int, timeout time.Duration) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.currentBatchSize, b.currentTimeout
}

// GetLatencyStats returns latency statistics
func (b *AdaptiveBatcher) GetLatencyStats() (avg, p99 time.Duration) {
	return b.latencyTracker.Average(), b.latencyTracker.P99()
}

// Close shuts down the batcher
func (b *AdaptiveBatcher) Close() {
	b.mu.Lock()
	b.closed = true
	if b.timer != nil {
		b.timer.Stop()
	}
	b.processBatchLocked()
	b.mu.Unlock()
	close(b.closeCh)
}

func main() {
	config := DefaultAdaptiveConfig()
	config.TargetLatency = 30 * time.Millisecond

	inferFn := func(batch [][]float32) [][]float32 {
		// Simulate variable inference time based on batch size
		time.Sleep(time.Duration(len(batch)) * 2 * time.Millisecond)

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

	batcher := NewAdaptiveBatcher(config, inferFn)
	defer batcher.Close()

	// Simulate variable load
	var wg sync.WaitGroup
	for round := 0; round < 3; round++ {
		numRequests := (round + 1) * 20
		fmt.Printf("\\nRound %d: %d requests\\n", round+1, numRequests)

		for i := 0; i < numRequests; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				batcher.Submit([]float32{1, 2, 3})
			}()
		}
		wg.Wait()

		batchSize, timeout := batcher.GetCurrentConfig()
		avg, p99 := batcher.GetLatencyStats()
		fmt.Printf("Config: batchSize=%d, timeout=%v\\n", batchSize, timeout)
		fmt.Printf("Latency: avg=%v, p99=%v\\n", avg, p99)

		time.Sleep(1500 * time.Millisecond) // Let adjustment happen
	}
}`,

	testCode: `package main

import (
	"sync"
	"testing"
	"time"
)

func TestAdaptiveBatcher(t *testing.T) {
	config := DefaultAdaptiveConfig()
	inferFn := func(batch [][]float32) [][]float32 {
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

	batcher := NewAdaptiveBatcher(config, inferFn)
	defer batcher.Close()

	result := batcher.Submit([]float32{1, 2})

	if result == nil {
		t.Fatal("Result should not be nil")
	}
}

func TestAdaptiveAdjustment(t *testing.T) {
	config := DefaultAdaptiveConfig()
	config.AdjustInterval = 100 * time.Millisecond
	config.TargetLatency = 20 * time.Millisecond

	inferFn := func(batch [][]float32) [][]float32 {
		// Slow inference to trigger adjustment
		time.Sleep(50 * time.Millisecond)
		results := make([][]float32, len(batch))
		for i := range batch {
			results[i] = []float32{1}
		}
		return results
	}

	batcher := NewAdaptiveBatcher(config, inferFn)
	defer batcher.Close()

	initialBatchSize, _ := batcher.GetCurrentConfig()

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			batcher.Submit([]float32{1})
		}()
	}
	wg.Wait()

	time.Sleep(200 * time.Millisecond)

	newBatchSize, _ := batcher.GetCurrentConfig()

	// Batch size should adjust (decrease due to high latency)
	if newBatchSize >= initialBatchSize && initialBatchSize > config.MinBatchSize {
		t.Logf("Batch size: initial=%d, new=%d", initialBatchSize, newBatchSize)
	}
}

func TestLatencyTracker(t *testing.T) {
	tracker := NewLatencyTracker(100)

	for i := 0; i < 100; i++ {
		tracker.Record(time.Duration(i) * time.Millisecond)
	}

	avg := tracker.Average()
	p99 := tracker.P99()

	if avg < 40*time.Millisecond || avg > 60*time.Millisecond {
		t.Errorf("Average should be around 50ms, got %v", avg)
	}

	if p99 < 90*time.Millisecond {
		t.Errorf("P99 should be around 99ms, got %v", p99)
	}
}

func TestGetCurrentConfig(t *testing.T) {
	config := DefaultAdaptiveConfig()
	batcher := NewAdaptiveBatcher(config, func(b [][]float32) [][]float32 {
		return b
	})
	defer batcher.Close()

	batchSize, timeout := batcher.GetCurrentConfig()

	if batchSize < config.MinBatchSize || batchSize > config.MaxBatchSize {
		t.Errorf("Batch size out of bounds: %d", batchSize)
	}
	if timeout < config.MinTimeout || timeout > config.MaxTimeout {
		t.Errorf("Timeout out of bounds: %v", timeout)
	}
}

func TestDefaultAdaptiveConfig(t *testing.T) {
	config := DefaultAdaptiveConfig()

	if config.MinBatchSize != 1 {
		t.Errorf("Expected MinBatchSize 1, got %d", config.MinBatchSize)
	}
	if config.MaxBatchSize != 64 {
		t.Errorf("Expected MaxBatchSize 64, got %d", config.MaxBatchSize)
	}
	if config.TargetLatency != 50*time.Millisecond {
		t.Errorf("Expected TargetLatency 50ms, got %v", config.TargetLatency)
	}
}

func TestLatencyTrackerEmpty(t *testing.T) {
	tracker := NewLatencyTracker(100)

	avg := tracker.Average()
	p99 := tracker.P99()

	if avg != 0 {
		t.Errorf("Empty tracker average should be 0, got %v", avg)
	}
	if p99 != 0 {
		t.Errorf("Empty tracker P99 should be 0, got %v", p99)
	}
}

func TestLatencyTrackerOverflow(t *testing.T) {
	tracker := NewLatencyTracker(5)

	for i := 0; i < 10; i++ {
		tracker.Record(time.Duration(i+1) * time.Millisecond)
	}

	// Should only keep last 5 samples (6-10ms)
	avg := tracker.Average()
	if avg < 7*time.Millisecond || avg > 9*time.Millisecond {
		t.Errorf("Average should be around 8ms, got %v", avg)
	}
}

func TestAdaptiveBatcherClose(t *testing.T) {
	config := DefaultAdaptiveConfig()
	batcher := NewAdaptiveBatcher(config, func(b [][]float32) [][]float32 {
		return b
	})

	batcher.Close()

	// Submit after close should return nil
	result := batcher.Submit([]float32{1, 2, 3})
	if result != nil {
		t.Error("Submit after close should return nil")
	}
}

func TestGetLatencyStats(t *testing.T) {
	config := DefaultAdaptiveConfig()
	batcher := NewAdaptiveBatcher(config, func(b [][]float32) [][]float32 {
		time.Sleep(5 * time.Millisecond)
		return b
	})
	defer batcher.Close()

	batcher.Submit([]float32{1})

	avg, p99 := batcher.GetLatencyStats()

	if avg == 0 {
		t.Error("Average latency should be recorded after submit")
	}
	if p99 == 0 {
		t.Error("P99 latency should be recorded after submit")
	}
}

func TestConcurrentAdaptiveSubmits(t *testing.T) {
	config := DefaultAdaptiveConfig()
	batcher := NewAdaptiveBatcher(config, func(b [][]float32) [][]float32 {
		results := make([][]float32, len(b))
		for i := range b {
			results[i] = []float32{1}
		}
		return results
	})
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
}`,

	hint1: 'Track P99 latency to make informed adjustments',
	hint2: 'Use exponential moving average for smoother adjustments',

	whyItMatters: `Adaptive batching maintains performance under varying load:

- **Self-tuning**: No manual parameter adjustment needed
- **Load adaptation**: Responds to traffic patterns automatically
- **SLA maintenance**: Keeps latency within target bounds
- **Resource efficiency**: Maximizes throughput while meeting latency goals

Adaptive systems are crucial for production ML services.`,

	translations: {
		ru: {
			title: 'Адаптивный батчинг',
			description: `# Адаптивный батчинг

Реализуйте адаптивный батчинг с автоматической настройкой параметров.

## Задача

Создайте адаптивный батчер:
- Мониторинг частоты запросов и латентности
- Динамическая настройка размера батча
- Настройка таймаута на основе глубины очереди
- Поддержание целевого SLA по латентности

## Пример

\`\`\`go
batcher := NewAdaptiveBatcher(config)
batcher.SetTargetLatency(50 * time.Millisecond)
// Automatically adjusts batch size and timeout
\`\`\``,
			hint1: 'Отслеживайте P99 латентность для обоснованных настроек',
			hint2: 'Используйте экспоненциальное скользящее среднее для более плавных настроек',
			whyItMatters: `Адаптивный батчинг поддерживает производительность при переменной нагрузке:

- **Самонастройка**: Не требуется ручная настройка параметров
- **Адаптация к нагрузке**: Автоматическая реакция на паттерны трафика
- **Поддержание SLA**: Удержание латентности в целевых границах
- **Эффективность ресурсов**: Максимизация пропускной способности при соблюдении целей латентности`,
		},
		uz: {
			title: 'Adaptiv batching',
			description: `# Adaptiv batching

Parametrlarni yukga qarab avtomatik sozlaydigan adaptiv batchingni amalga oshiring.

## Topshiriq

Adaptiv batcher yarating:
- So'rov tezligi va latency ni monitoring qilish
- Batch o'lchamini dinamik sozlash
- Navbat chuqurligi asosida timeout ni sozlash
- Maqsadli latency SLA ni saqlash

## Misol

\`\`\`go
batcher := NewAdaptiveBatcher(config)
batcher.SetTargetLatency(50 * time.Millisecond)
// Automatically adjusts batch size and timeout
\`\`\``,
			hint1: "Asosli sozlashlar uchun P99 latency ni kuzatib boring",
			hint2: "Yumshoqroq sozlashlar uchun eksponensial harakatlanuvchi o'rtacha foydalaning",
			whyItMatters: `Adaptiv batching o'zgaruvchan yuk ostida ishlashni saqlaydi:

- **O'z-o'zini sozlash**: Qo'lda parametr sozlash kerak emas
- **Yukga moslashish**: Trafik patternlariga avtomatik javob beradi
- **SLA ni saqlash**: Latency ni maqsadli chegaralarda saqlaydi
- **Resurs samaradorligi**: Latency maqsadlariga rioya qilgan holda throughput ni maksimallashtiradi`,
		},
	},
};

export default task;
