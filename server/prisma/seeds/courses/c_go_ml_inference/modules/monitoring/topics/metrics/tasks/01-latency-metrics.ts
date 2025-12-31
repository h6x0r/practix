import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-latency-metrics',
	title: 'Latency Metrics',
	difficulty: 'medium',
	tags: ['go', 'ml', 'metrics', 'prometheus'],
	estimatedTime: '25m',
	isPremium: true,
	order: 1,
	description: `# Latency Metrics

Implement latency metrics collection for ML inference.

## Task

Build metrics collection that:
- Tracks inference latency distribution (histogram)
- Tracks p50, p90, p99 latencies
- Exposes Prometheus-compatible metrics
- Supports per-model metrics

## Example

\`\`\`go
metrics := NewInferenceMetrics()
metrics.RecordLatency("model-v1", 15*time.Millisecond)
// GET /metrics -> inference_latency_seconds{model="model-v1"} histogram
\`\`\``,

	initialCode: `package main

import (
	"fmt"
	"time"
)

// InferenceMetrics collects inference metrics
type InferenceMetrics struct {
	// Your fields here
}

// NewInferenceMetrics creates inference metrics
func NewInferenceMetrics() *InferenceMetrics {
	// Your code here
	return nil
}

// RecordLatency records inference latency
func (m *InferenceMetrics) RecordLatency(model string, latency time.Duration) {
	// Your code here
}

// GetPercentile returns the Nth percentile latency
func (m *InferenceMetrics) GetPercentile(model string, percentile float64) time.Duration {
	// Your code here
	return 0
}

func main() {
	fmt.Println("Latency Metrics")
}`,

	solutionCode: `package main

import (
	"fmt"
	"math"
	"net/http"
	"sort"
	"sync"
	"time"
)

// LatencyBucket represents a histogram bucket
type LatencyBucket struct {
	UpperBound float64
	Count      int64
}

// LatencyHistogram tracks latency distribution
type LatencyHistogram struct {
	buckets    []LatencyBucket
	sum        float64
	count      int64
	samples    []float64
	maxSamples int
	mu         sync.RWMutex
}

// NewLatencyHistogram creates a histogram with default buckets
func NewLatencyHistogram() *LatencyHistogram {
	// Default buckets in seconds
	bounds := []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0}
	buckets := make([]LatencyBucket, len(bounds)+1)
	for i, bound := range bounds {
		buckets[i] = LatencyBucket{UpperBound: bound}
	}
	buckets[len(bounds)] = LatencyBucket{UpperBound: math.Inf(1)}

	return &LatencyHistogram{
		buckets:    buckets,
		samples:    make([]float64, 0, 10000),
		maxSamples: 10000,
	}
}

// Observe records a latency value
func (h *LatencyHistogram) Observe(seconds float64) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.sum += seconds
	h.count++

	// Update buckets
	for i := range h.buckets {
		if seconds <= h.buckets[i].UpperBound {
			h.buckets[i].Count++
			break
		}
	}

	// Keep samples for percentile calculation
	if len(h.samples) < h.maxSamples {
		h.samples = append(h.samples, seconds)
	}
}

// Percentile returns the Nth percentile
func (h *LatencyHistogram) Percentile(p float64) float64 {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if len(h.samples) == 0 {
		return 0
	}

	sorted := make([]float64, len(h.samples))
	copy(sorted, h.samples)
	sort.Float64s(sorted)

	idx := int(float64(len(sorted)) * p / 100.0)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// Stats returns histogram statistics
func (h *LatencyHistogram) Stats() (count int64, sum, avg float64) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	count = h.count
	sum = h.sum
	if count > 0 {
		avg = sum / float64(count)
	}
	return
}

// ModelMetrics holds metrics for a single model
type ModelMetrics struct {
	latency      *LatencyHistogram
	requestCount int64
	errorCount   int64
	mu           sync.RWMutex
}

// InferenceMetrics collects inference metrics
type InferenceMetrics struct {
	models map[string]*ModelMetrics
	mu     sync.RWMutex
}

// NewInferenceMetrics creates inference metrics
func NewInferenceMetrics() *InferenceMetrics {
	return &InferenceMetrics{
		models: make(map[string]*ModelMetrics),
	}
}

// getOrCreateModel gets or creates model metrics
func (m *InferenceMetrics) getOrCreateModel(model string) *ModelMetrics {
	m.mu.RLock()
	mm, exists := m.models[model]
	m.mu.RUnlock()

	if exists {
		return mm
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Double check after acquiring write lock
	if mm, exists = m.models[model]; exists {
		return mm
	}

	mm = &ModelMetrics{
		latency: NewLatencyHistogram(),
	}
	m.models[model] = mm
	return mm
}

// RecordLatency records inference latency
func (m *InferenceMetrics) RecordLatency(model string, latency time.Duration) {
	mm := m.getOrCreateModel(model)
	mm.mu.Lock()
	mm.latency.Observe(latency.Seconds())
	mm.requestCount++
	mm.mu.Unlock()
}

// RecordError records an error
func (m *InferenceMetrics) RecordError(model string) {
	mm := m.getOrCreateModel(model)
	mm.mu.Lock()
	mm.errorCount++
	mm.mu.Unlock()
}

// GetPercentile returns the Nth percentile latency
func (m *InferenceMetrics) GetPercentile(model string, percentile float64) time.Duration {
	m.mu.RLock()
	mm, exists := m.models[model]
	m.mu.RUnlock()

	if !exists {
		return 0
	}

	seconds := mm.latency.Percentile(percentile)
	return time.Duration(seconds * float64(time.Second))
}

// GetStats returns model statistics
func (m *InferenceMetrics) GetStats(model string) (requests, errors int64, avgLatency time.Duration) {
	m.mu.RLock()
	mm, exists := m.models[model]
	m.mu.RUnlock()

	if !exists {
		return
	}

	mm.mu.RLock()
	defer mm.mu.RUnlock()

	requests = mm.requestCount
	errors = mm.errorCount
	_, _, avg := mm.latency.Stats()
	avgLatency = time.Duration(avg * float64(time.Second))
	return
}

// PrometheusHandler returns metrics in Prometheus format
func (m *InferenceMetrics) PrometheusHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.mu.RLock()
		defer m.mu.RUnlock()

		w.Header().Set("Content-Type", "text/plain")

		for modelName, mm := range m.models {
			mm.mu.RLock()

			count, sum, _ := mm.latency.Stats()

			// Request count
			fmt.Fprintf(w, "inference_requests_total{model=\"%s\"} %d\\n", modelName, mm.requestCount)

			// Error count
			fmt.Fprintf(w, "inference_errors_total{model=\"%s\"} %d\\n", modelName, mm.errorCount)

			// Latency histogram
			fmt.Fprintf(w, "inference_latency_seconds_count{model=\"%s\"} %d\\n", modelName, count)
			fmt.Fprintf(w, "inference_latency_seconds_sum{model=\"%s\"} %f\\n", modelName, sum)

			// Percentiles
			fmt.Fprintf(w, "inference_latency_seconds{model=\"%s\",quantile=\"0.5\"} %f\\n",
				modelName, mm.latency.Percentile(50))
			fmt.Fprintf(w, "inference_latency_seconds{model=\"%s\",quantile=\"0.9\"} %f\\n",
				modelName, mm.latency.Percentile(90))
			fmt.Fprintf(w, "inference_latency_seconds{model=\"%s\",quantile=\"0.99\"} %f\\n",
				modelName, mm.latency.Percentile(99))

			mm.mu.RUnlock()
		}
	})
}

func main() {
	metrics := NewInferenceMetrics()

	// Simulate some inference requests
	for i := 0; i < 100; i++ {
		latency := time.Duration(10+i%50) * time.Millisecond
		metrics.RecordLatency("model-v1", latency)

		if i%10 == 0 {
			metrics.RecordError("model-v1")
		}
	}

	fmt.Printf("P50: %v\\n", metrics.GetPercentile("model-v1", 50))
	fmt.Printf("P90: %v\\n", metrics.GetPercentile("model-v1", 90))
	fmt.Printf("P99: %v\\n", metrics.GetPercentile("model-v1", 99))

	requests, errors, avgLatency := metrics.GetStats("model-v1")
	fmt.Printf("Requests: %d, Errors: %d, Avg Latency: %v\\n", requests, errors, avgLatency)

	// Start metrics server
	http.Handle("/metrics", metrics.PrometheusHandler())
	fmt.Println("\\nMetrics server on :9090/metrics")
	// http.ListenAndServe(":9090", nil)
}`,

	testCode: `package main

import (
	"testing"
	"time"
)

func TestLatencyHistogram(t *testing.T) {
	h := NewLatencyHistogram()

	h.Observe(0.010) // 10ms
	h.Observe(0.020) // 20ms
	h.Observe(0.030) // 30ms

	count, sum, avg := h.Stats()

	if count != 3 {
		t.Errorf("Expected count 3, got %d", count)
	}

	if sum < 0.059 || sum > 0.061 {
		t.Errorf("Expected sum ~0.06, got %f", sum)
	}

	if avg < 0.019 || avg > 0.021 {
		t.Errorf("Expected avg ~0.02, got %f", avg)
	}
}

func TestLatencyPercentile(t *testing.T) {
	h := NewLatencyHistogram()

	for i := 1; i <= 100; i++ {
		h.Observe(float64(i) / 1000) // 1-100ms
	}

	p50 := h.Percentile(50)
	p90 := h.Percentile(90)
	p99 := h.Percentile(99)

	if p50 < 0.049 || p50 > 0.051 {
		t.Errorf("P50 should be ~0.05, got %f", p50)
	}

	if p90 < 0.089 || p90 > 0.091 {
		t.Errorf("P90 should be ~0.09, got %f", p90)
	}

	if p99 < 0.098 || p99 > 0.100 {
		t.Errorf("P99 should be ~0.099, got %f", p99)
	}
}

func TestInferenceMetrics(t *testing.T) {
	metrics := NewInferenceMetrics()

	metrics.RecordLatency("model1", 10*time.Millisecond)
	metrics.RecordLatency("model1", 20*time.Millisecond)
	metrics.RecordLatency("model2", 30*time.Millisecond)

	requests1, _, _ := metrics.GetStats("model1")
	requests2, _, _ := metrics.GetStats("model2")

	if requests1 != 2 {
		t.Errorf("Expected 2 requests for model1, got %d", requests1)
	}

	if requests2 != 1 {
		t.Errorf("Expected 1 request for model2, got %d", requests2)
	}
}

func TestRecordError(t *testing.T) {
	metrics := NewInferenceMetrics()

	metrics.RecordLatency("model1", 10*time.Millisecond)
	metrics.RecordError("model1")
	metrics.RecordError("model1")

	requests, errors, _ := metrics.GetStats("model1")

	if requests != 1 {
		t.Errorf("Expected 1 request, got %d", requests)
	}

	if errors != 2 {
		t.Errorf("Expected 2 errors, got %d", errors)
	}
}

func TestGetPercentile(t *testing.T) {
	metrics := NewInferenceMetrics()

	for i := 1; i <= 100; i++ {
		metrics.RecordLatency("model1", time.Duration(i)*time.Millisecond)
	}

	p50 := metrics.GetPercentile("model1", 50)
	if p50 < 49*time.Millisecond || p50 > 51*time.Millisecond {
		t.Errorf("P50 should be ~50ms, got %v", p50)
	}
}

func TestNonexistentModel(t *testing.T) {
	metrics := NewInferenceMetrics()

	p50 := metrics.GetPercentile("nonexistent", 50)
	if p50 != 0 {
		t.Errorf("Expected 0 for nonexistent model, got %v", p50)
	}

	requests, errors, _ := metrics.GetStats("nonexistent")
	if requests != 0 || errors != 0 {
		t.Error("Expected zero stats for nonexistent model")
	}
}

func TestEmptyHistogram(t *testing.T) {
	h := NewLatencyHistogram()

	count, sum, avg := h.Stats()
	if count != 0 || sum != 0 || avg != 0 {
		t.Error("Empty histogram should have zero stats")
	}

	p50 := h.Percentile(50)
	if p50 != 0 {
		t.Errorf("Empty histogram percentile should be 0, got %f", p50)
	}
}

func TestMultipleModels(t *testing.T) {
	metrics := NewInferenceMetrics()

	metrics.RecordLatency("model-a", 10*time.Millisecond)
	metrics.RecordLatency("model-b", 20*time.Millisecond)
	metrics.RecordLatency("model-c", 30*time.Millisecond)

	reqA, _, _ := metrics.GetStats("model-a")
	reqB, _, _ := metrics.GetStats("model-b")
	reqC, _, _ := metrics.GetStats("model-c")

	if reqA != 1 || reqB != 1 || reqC != 1 {
		t.Error("Each model should have 1 request")
	}
}

func TestGetStats(t *testing.T) {
	metrics := NewInferenceMetrics()

	metrics.RecordLatency("model1", 100*time.Millisecond)
	metrics.RecordLatency("model1", 200*time.Millisecond)
	metrics.RecordError("model1")

	requests, errors, avgLatency := metrics.GetStats("model1")

	if requests != 2 {
		t.Errorf("Expected 2 requests, got %d", requests)
	}
	if errors != 1 {
		t.Errorf("Expected 1 error, got %d", errors)
	}
	if avgLatency < 140*time.Millisecond || avgLatency > 160*time.Millisecond {
		t.Errorf("Expected avg ~150ms, got %v", avgLatency)
	}
}

func TestHistogramBuckets(t *testing.T) {
	h := NewLatencyHistogram()

	if len(h.buckets) == 0 {
		t.Fatal("Histogram should have buckets")
	}

	// Observe values in different buckets
	h.Observe(0.001) // 1ms
	h.Observe(0.100) // 100ms
	h.Observe(1.000) // 1s

	count, _, _ := h.Stats()
	if count != 3 {
		t.Errorf("Expected 3 observations, got %d", count)
	}
}`,

	hint1: 'Use histograms for latency distributions, not just averages',
	hint2: 'Keep a sample of values for accurate percentile calculation',

	whyItMatters: `Latency metrics are critical for ML service monitoring:

- **SLA tracking**: Verify you meet latency commitments
- **Performance regression**: Detect slowdowns early
- **Capacity planning**: Understand performance under load
- **Troubleshooting**: Identify latency spikes and their causes

Histograms capture the full latency distribution, not just averages.`,

	translations: {
		ru: {
			title: 'Метрики латентности',
			description: `# Метрики латентности

Реализуйте сбор метрик латентности для ML инференса.

## Задача

Создайте сбор метрик:
- Отслеживание распределения латентности (гистограмма)
- Отслеживание p50, p90, p99 латентности
- Экспорт Prometheus-совместимых метрик
- Поддержка метрик по модели

## Пример

\`\`\`go
metrics := NewInferenceMetrics()
metrics.RecordLatency("model-v1", 15*time.Millisecond)
// GET /metrics -> inference_latency_seconds{model="model-v1"} histogram
\`\`\``,
			hint1: 'Используйте гистограммы для распределений латентности, а не только средние',
			hint2: 'Храните выборку значений для точного расчета перцентилей',
			whyItMatters: `Метрики латентности критичны для мониторинга ML сервисов:

- **Отслеживание SLA**: Проверка соблюдения обязательств по латентности
- **Регрессия производительности**: Раннее обнаружение замедлений
- **Планирование емкости**: Понимание производительности под нагрузкой
- **Устранение неполадок**: Выявление скачков латентности и их причин`,
		},
		uz: {
			title: 'Latency metrikalari',
			description: `# Latency metrikalari

ML inference uchun latency metrikalarini yig'ishni amalga oshiring.

## Topshiriq

Metrikalar yig'ishni yarating:
- Latency taqsimotini kuzatish (gistogramma)
- p50, p90, p99 latencylarni kuzatish
- Prometheus-mos metrikalarni eksport qilish
- Har bir model uchun metrikalarni qo'llab-quvvatlash

## Misol

\`\`\`go
metrics := NewInferenceMetrics()
metrics.RecordLatency("model-v1", 15*time.Millisecond)
// GET /metrics -> inference_latency_seconds{model="model-v1"} histogram
\`\`\``,
			hint1: "Latency taqsimotlari uchun gistogrammalardan foydalaning, faqat o'rtachalardan emas",
			hint2: "Aniq percentile hisoblash uchun qiymatlar namunasini saqlang",
			whyItMatters: `Latency metrikalari ML xizmat monitoringi uchun muhim:

- **SLA kuzatish**: Latency majburiyatlariga rioya qilishni tekshirish
- **Ishlash regressiyasi**: Sekinlashishlarni erta aniqlash
- **Sig'imni rejalashtirish**: Yuk ostida ishlashni tushunish
- **Muammolarni bartaraf etish**: Latency o'sishlarini va ularning sabablarini aniqlash`,
		},
	},
};

export default task;
