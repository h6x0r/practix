import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-throughput-metrics',
	title: 'Throughput Metrics',
	difficulty: 'medium',
	tags: ['go', 'ml', 'metrics', 'throughput'],
	estimatedTime: '25m',
	isPremium: true,
	order: 2,
	description: `# Throughput Metrics

Track inference throughput and request rates.

## Task

Implement throughput metrics that:
- Count requests per second
- Track successful vs failed requests
- Monitor batch sizes
- Calculate throughput over time windows

## Example

\`\`\`go
metrics := NewThroughputMetrics()
metrics.RecordRequest("model-v1", true)
fmt.Println(metrics.GetRPS("model-v1")) // requests per second
\`\`\``,

	initialCode: `package main

import (
	"fmt"
	"time"
)

// ThroughputMetrics tracks request throughput
type ThroughputMetrics struct {
	// Your fields here
}

// NewThroughputMetrics creates throughput metrics
func NewThroughputMetrics() *ThroughputMetrics {
	// Your code here
	return nil
}

// RecordRequest records a request
func (m *ThroughputMetrics) RecordRequest(model string, success bool) {
	// Your code here
}

// GetRPS returns requests per second
func (m *ThroughputMetrics) GetRPS(model string) float64 {
	// Your code here
	return 0
}

func main() {
	fmt.Println("Throughput Metrics")
}`,

	solutionCode: `package main

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// SlidingWindow tracks events over a time window
type SlidingWindow struct {
	windowSize   time.Duration
	bucketSize   time.Duration
	buckets      []int64
	bucketCount  int
	currentIdx   int
	lastRotation time.Time
	total        int64
	mu           sync.Mutex
}

// NewSlidingWindow creates a sliding window
func NewSlidingWindow(windowSize, bucketSize time.Duration) *SlidingWindow {
	bucketCount := int(windowSize / bucketSize)
	if bucketCount < 1 {
		bucketCount = 1
	}

	return &SlidingWindow{
		windowSize:   windowSize,
		bucketSize:   bucketSize,
		buckets:      make([]int64, bucketCount),
		bucketCount:  bucketCount,
		lastRotation: time.Now(),
	}
}

// Increment adds to the current bucket
func (sw *SlidingWindow) Increment(n int64) {
	sw.mu.Lock()
	defer sw.mu.Unlock()

	sw.rotate()
	sw.buckets[sw.currentIdx] += n
	sw.total += n
}

// rotate rotates buckets based on elapsed time
func (sw *SlidingWindow) rotate() {
	now := time.Now()
	elapsed := now.Sub(sw.lastRotation)
	rotations := int(elapsed / sw.bucketSize)

	if rotations == 0 {
		return
	}

	if rotations >= sw.bucketCount {
		// Clear all buckets
		for i := range sw.buckets {
			sw.buckets[i] = 0
		}
		sw.total = 0
	} else {
		// Rotate and clear old buckets
		for i := 0; i < rotations; i++ {
			sw.currentIdx = (sw.currentIdx + 1) % sw.bucketCount
			sw.total -= sw.buckets[sw.currentIdx]
			sw.buckets[sw.currentIdx] = 0
		}
	}

	sw.lastRotation = now
}

// Sum returns the sum of all buckets
func (sw *SlidingWindow) Sum() int64 {
	sw.mu.Lock()
	defer sw.mu.Unlock()

	sw.rotate()
	return sw.total
}

// Rate returns events per second
func (sw *SlidingWindow) Rate() float64 {
	sum := sw.Sum()
	return float64(sum) / sw.windowSize.Seconds()
}

// ModelThroughput holds throughput metrics for a model
type ModelThroughput struct {
	successWindow *SlidingWindow
	failureWindow *SlidingWindow
	batchSizes    []int
	maxBatchHist  int
	totalRequests int64
	totalErrors   int64
	mu            sync.Mutex
}

// ThroughputMetrics tracks request throughput
type ThroughputMetrics struct {
	models     map[string]*ModelThroughput
	windowSize time.Duration
	mu         sync.RWMutex
}

// NewThroughputMetrics creates throughput metrics
func NewThroughputMetrics() *ThroughputMetrics {
	return &ThroughputMetrics{
		models:     make(map[string]*ModelThroughput),
		windowSize: time.Minute,
	}
}

// SetWindowSize sets the measurement window
func (m *ThroughputMetrics) SetWindowSize(d time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.windowSize = d
}

// getOrCreateModel gets or creates model metrics
func (m *ThroughputMetrics) getOrCreateModel(model string) *ModelThroughput {
	m.mu.RLock()
	mt, exists := m.models[model]
	m.mu.RUnlock()

	if exists {
		return mt
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if mt, exists = m.models[model]; exists {
		return mt
	}

	bucketSize := m.windowSize / 60
	if bucketSize < time.Second {
		bucketSize = time.Second
	}

	mt = &ModelThroughput{
		successWindow: NewSlidingWindow(m.windowSize, bucketSize),
		failureWindow: NewSlidingWindow(m.windowSize, bucketSize),
		batchSizes:    make([]int, 0, 1000),
		maxBatchHist:  1000,
	}
	m.models[model] = mt
	return mt
}

// RecordRequest records a request
func (m *ThroughputMetrics) RecordRequest(model string, success bool) {
	mt := m.getOrCreateModel(model)

	if success {
		mt.successWindow.Increment(1)
		atomic.AddInt64(&mt.totalRequests, 1)
	} else {
		mt.failureWindow.Increment(1)
		atomic.AddInt64(&mt.totalErrors, 1)
	}
}

// RecordBatch records a batch request
func (m *ThroughputMetrics) RecordBatch(model string, batchSize int, success bool) {
	mt := m.getOrCreateModel(model)

	if success {
		mt.successWindow.Increment(int64(batchSize))
	} else {
		mt.failureWindow.Increment(int64(batchSize))
	}

	mt.mu.Lock()
	if len(mt.batchSizes) < mt.maxBatchHist {
		mt.batchSizes = append(mt.batchSizes, batchSize)
	}
	mt.mu.Unlock()

	atomic.AddInt64(&mt.totalRequests, int64(batchSize))
}

// GetRPS returns requests per second
func (m *ThroughputMetrics) GetRPS(model string) float64 {
	m.mu.RLock()
	mt, exists := m.models[model]
	m.mu.RUnlock()

	if !exists {
		return 0
	}

	return mt.successWindow.Rate() + mt.failureWindow.Rate()
}

// GetSuccessRate returns success rate (0-1)
func (m *ThroughputMetrics) GetSuccessRate(model string) float64 {
	m.mu.RLock()
	mt, exists := m.models[model]
	m.mu.RUnlock()

	if !exists {
		return 0
	}

	success := mt.successWindow.Sum()
	failure := mt.failureWindow.Sum()
	total := success + failure

	if total == 0 {
		return 1.0
	}

	return float64(success) / float64(total)
}

// GetStats returns throughput statistics
type ThroughputStats struct {
	RPS            float64
	SuccessRate    float64
	TotalRequests  int64
	TotalErrors    int64
	AvgBatchSize   float64
}

func (m *ThroughputMetrics) GetStats(model string) ThroughputStats {
	m.mu.RLock()
	mt, exists := m.models[model]
	m.mu.RUnlock()

	if !exists {
		return ThroughputStats{SuccessRate: 1.0}
	}

	stats := ThroughputStats{
		RPS:           m.GetRPS(model),
		SuccessRate:   m.GetSuccessRate(model),
		TotalRequests: atomic.LoadInt64(&mt.totalRequests),
		TotalErrors:   atomic.LoadInt64(&mt.totalErrors),
	}

	mt.mu.Lock()
	if len(mt.batchSizes) > 0 {
		var sum int
		for _, size := range mt.batchSizes {
			sum += size
		}
		stats.AvgBatchSize = float64(sum) / float64(len(mt.batchSizes))
	}
	mt.mu.Unlock()

	return stats
}

// EWMARate calculates exponentially weighted moving average rate
type EWMARate struct {
	rate      float64
	alpha     float64
	lastTick  time.Time
	count     int64
	mu        sync.Mutex
	tickInterval time.Duration
}

func NewEWMARate(halfLife time.Duration) *EWMARate {
	tickInterval := time.Second
	alpha := 1.0 - math.Exp(-tickInterval.Seconds()/halfLife.Seconds())

	e := &EWMARate{
		alpha:        alpha,
		lastTick:     time.Now(),
		tickInterval: tickInterval,
	}

	go e.tickLoop()
	return e
}

func (e *EWMARate) tickLoop() {
	ticker := time.NewTicker(e.tickInterval)
	for range ticker.C {
		e.tick()
	}
}

func (e *EWMARate) tick() {
	e.mu.Lock()
	defer e.mu.Unlock()

	instantRate := float64(e.count) / e.tickInterval.Seconds()
	e.rate = e.rate + e.alpha*(instantRate-e.rate)
	e.count = 0
	e.lastTick = time.Now()
}

func (e *EWMARate) Increment() {
	e.mu.Lock()
	e.count++
	e.mu.Unlock()
}

func (e *EWMARate) Rate() float64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.rate
}

func main() {
	metrics := NewThroughputMetrics()

	// Simulate requests
	for i := 0; i < 100; i++ {
		success := i%10 != 0 // 90% success rate
		metrics.RecordRequest("model-v1", success)
	}

	// Simulate batch requests
	for i := 0; i < 50; i++ {
		batchSize := 16 + i%32
		metrics.RecordBatch("model-v1", batchSize, true)
	}

	fmt.Printf("RPS: %.2f\\n", metrics.GetRPS("model-v1"))
	fmt.Printf("Success Rate: %.2f%%\\n", metrics.GetSuccessRate("model-v1")*100)

	stats := metrics.GetStats("model-v1")
	fmt.Printf("Stats: %+v\\n", stats)
}`,

	testCode: `package main

import (
	"testing"
	"time"
)

func TestSlidingWindow(t *testing.T) {
	sw := NewSlidingWindow(time.Second, 100*time.Millisecond)

	sw.Increment(5)
	sw.Increment(3)

	sum := sw.Sum()
	if sum != 8 {
		t.Errorf("Expected sum 8, got %d", sum)
	}
}

func TestSlidingWindowRate(t *testing.T) {
	sw := NewSlidingWindow(time.Second, 100*time.Millisecond)

	sw.Increment(100)

	rate := sw.Rate()
	if rate < 99 || rate > 101 {
		t.Errorf("Expected rate ~100, got %f", rate)
	}
}

func TestThroughputMetrics(t *testing.T) {
	metrics := NewThroughputMetrics()

	metrics.RecordRequest("model1", true)
	metrics.RecordRequest("model1", true)
	metrics.RecordRequest("model1", false)

	successRate := metrics.GetSuccessRate("model1")
	expected := 2.0 / 3.0

	if successRate < expected-0.01 || successRate > expected+0.01 {
		t.Errorf("Expected success rate ~0.67, got %f", successRate)
	}
}

func TestThroughputRPS(t *testing.T) {
	metrics := NewThroughputMetrics()

	for i := 0; i < 100; i++ {
		metrics.RecordRequest("model1", true)
	}

	rps := metrics.GetRPS("model1")
	if rps == 0 {
		t.Error("RPS should be greater than 0")
	}
}

func TestRecordBatch(t *testing.T) {
	metrics := NewThroughputMetrics()

	metrics.RecordBatch("model1", 32, true)
	metrics.RecordBatch("model1", 16, true)

	stats := metrics.GetStats("model1")

	if stats.TotalRequests != 48 {
		t.Errorf("Expected 48 requests, got %d", stats.TotalRequests)
	}

	if stats.AvgBatchSize != 24 {
		t.Errorf("Expected avg batch size 24, got %f", stats.AvgBatchSize)
	}
}

func TestNonexistentModel(t *testing.T) {
	metrics := NewThroughputMetrics()

	rps := metrics.GetRPS("nonexistent")
	if rps != 0 {
		t.Errorf("Expected 0 for nonexistent model, got %f", rps)
	}

	successRate := metrics.GetSuccessRate("nonexistent")
	if successRate != 0 {
		t.Errorf("Expected 0 for nonexistent model, got %f", successRate)
	}
}

func TestThroughputStats(t *testing.T) {
	metrics := NewThroughputMetrics()

	for i := 0; i < 50; i++ {
		metrics.RecordRequest("model1", true)
	}
	for i := 0; i < 10; i++ {
		metrics.RecordRequest("model1", false)
	}

	stats := metrics.GetStats("model1")

	if stats.TotalRequests != 50 {
		t.Errorf("Expected 50 requests, got %d", stats.TotalRequests)
	}

	if stats.TotalErrors != 10 {
		t.Errorf("Expected 10 errors, got %d", stats.TotalErrors)
	}
}

func TestSetWindowSize(t *testing.T) {
	metrics := NewThroughputMetrics()
	metrics.SetWindowSize(30 * time.Second)

	metrics.RecordRequest("model1", true)
	rps := metrics.GetRPS("model1")

	if rps == 0 {
		t.Error("RPS should be greater than 0 after window size change")
	}
}

func TestEWMARate(t *testing.T) {
	ewma := NewEWMARate(5 * time.Second)

	for i := 0; i < 10; i++ {
		ewma.Increment()
	}

	// Rate may not update immediately due to tick interval
	rate := ewma.Rate()
	if rate < 0 {
		t.Error("Rate should be non-negative")
	}
}

func TestMultipleModels(t *testing.T) {
	metrics := NewThroughputMetrics()

	metrics.RecordRequest("model1", true)
	metrics.RecordRequest("model2", true)
	metrics.RecordRequest("model3", false)

	stats1 := metrics.GetStats("model1")
	stats2 := metrics.GetStats("model2")
	stats3 := metrics.GetStats("model3")

	if stats1.TotalRequests != 1 {
		t.Errorf("Expected 1 request for model1, got %d", stats1.TotalRequests)
	}
	if stats2.TotalRequests != 1 {
		t.Errorf("Expected 1 request for model2, got %d", stats2.TotalRequests)
	}
	if stats3.TotalErrors != 1 {
		t.Errorf("Expected 1 error for model3, got %d", stats3.TotalErrors)
	}
}`,

	hint1: 'Use sliding windows for accurate rate calculation',
	hint2: 'Track both instantaneous and moving average rates',

	whyItMatters: `Throughput metrics reveal system capacity:

- **Capacity planning**: Understand max sustainable load
- **Anomaly detection**: Spot traffic drops or spikes
- **Scaling decisions**: Know when to add more instances
- **Cost optimization**: Right-size your infrastructure

Accurate throughput tracking is essential for ML service operations.`,

	translations: {
		ru: {
			title: 'Метрики пропускной способности',
			description: `# Метрики пропускной способности

Отслеживайте пропускную способность и частоту запросов инференса.

## Задача

Реализуйте метрики пропускной способности:
- Подсчет запросов в секунду
- Отслеживание успешных и неудачных запросов
- Мониторинг размеров батчей
- Расчет пропускной способности по временным окнам

## Пример

\`\`\`go
metrics := NewThroughputMetrics()
metrics.RecordRequest("model-v1", true)
fmt.Println(metrics.GetRPS("model-v1")) // requests per second
\`\`\``,
			hint1: 'Используйте скользящие окна для точного расчета скорости',
			hint2: 'Отслеживайте как мгновенную так и скользящую среднюю скорость',
			whyItMatters: `Метрики пропускной способности показывают емкость системы:

- **Планирование емкости**: Понимание максимальной устойчивой нагрузки
- **Обнаружение аномалий**: Выявление падений или всплесков трафика
- **Решения по масштабированию**: Когда добавлять инстансы
- **Оптимизация затрат**: Правильный размер инфраструктуры`,
		},
		uz: {
			title: "O'tkazish qobiliyati metrikalari",
			description: `# O'tkazish qobiliyati metrikalari

Inference throughput va so'rov tezliklarini kuzatib boring.

## Topshiriq

Throughput metrikalarini amalga oshiring:
- Sekundiga so'rovlarni hisoblash
- Muvaffaqiyatli va muvaffaqiyatsiz so'rovlarni kuzatish
- Batch o'lchamlarini monitoring qilish
- Vaqt oynalari bo'yicha throughput ni hisoblash

## Misol

\`\`\`go
metrics := NewThroughputMetrics()
metrics.RecordRequest("model-v1", true)
fmt.Println(metrics.GetRPS("model-v1")) // requests per second
\`\`\``,
			hint1: "Aniq tezlik hisoblash uchun sirpanuvchi oynalardan foydalaning",
			hint2: "Ham lahzali ham harakatlanuvchi o'rtacha tezliklarni kuzatib boring",
			whyItMatters: `Throughput metrikalari tizim sig'imini ko'rsatadi:

- **Sig'imni rejalashtirish**: Maksimal barqaror yukni tushunish
- **Anomaliyalarni aniqlash**: Trafik tushishi yoki o'sishlarini aniqlash
- **Masshtablash qarorlari**: Qachon ko'proq instance qo'shish kerakligini bilish
- **Xarajatlarni optimallashtirish**: Infratuzilmangizni to'g'ri o'lchamda qilish`,
		},
	},
};

export default task;
