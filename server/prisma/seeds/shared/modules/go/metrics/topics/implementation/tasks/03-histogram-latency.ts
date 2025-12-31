import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-metrics-histogram-latency',
	title: 'Histogram for Request Latency Tracking',
	difficulty: 'medium',
	tags: ['go', 'metrics', 'histogram', 'performance'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a histogram to track request latency distribution and calculate percentiles.

**Requirements:**
1. **Histogram Type**: Store buckets map[float64]int64 and sum/count for tracking
2. **Observe(value)**: Record observation in appropriate bucket and update sum/count
3. **Percentile(p)**: Calculate percentile (e.g., p50, p95, p99) from distribution
4. **RenderHistogram**: Format histogram as Prometheus metric with buckets and quantiles

**Histogram Pattern:**
\`\`\`go
type Histogram struct {
    mu      sync.RWMutex
    buckets map[float64]int64  // bucket upper bound -> count
    sum     float64            // sum of all observations
    count   int64              // total observations
}

// Record observation
func (h *Histogram) Observe(value float64) {
    // Increment appropriate bucket
    // Add to sum and count
}

// Calculate percentile
func (h *Histogram) Percentile(p float64) float64 {
    // p = 0.50 for median, 0.95 for p95, 0.99 for p99
    // Return estimated value at percentile
}
\`\`\`

**Key Concepts:**
- Histogram buckets define ranges (0.1, 0.5, 1.0, 5.0, 10.0 seconds)
- Each observation increments all buckets >= value (cumulative)
- Percentiles estimate distribution (p50=median, p95, p99)
- Sum/count track total for average calculation
- Thread-safe with RWMutex for concurrent observations

**Example Usage:**
\`\`\`go
var requestLatency = NewHistogram([]float64{0.1, 0.5, 1.0, 5.0, 10.0})

func HandleRequest(w http.ResponseWriter, r *http.Request) {
    start := time.Now()

    // Process request...
    processRequest(w, r)

    // Record latency
    duration := time.Since(start).Seconds()
    requestLatency.Observe(duration)
}

// Metrics endpoint
func MetricsHandler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprint(w, "# HELP http_request_duration_seconds Request latency\n")
    fmt.Fprint(w, "# TYPE http_request_duration_seconds histogram\n")
    fmt.Fprint(w, RenderHistogram("http_request_duration_seconds", requestLatency))
    // Output:
    // http_request_duration_seconds_bucket{le="0.1"} 150
    // http_request_duration_seconds_bucket{le="0.5"} 280
    // http_request_duration_seconds_bucket{le="1.0"} 295
    // http_request_duration_seconds_bucket{le="+Inf"} 300
    // http_request_duration_seconds_sum 245.3
    // http_request_duration_seconds_count 300
}

// Check SLA compliance
func CheckSLA() {
    p95 := requestLatency.Percentile(0.95)
    p99 := requestLatency.Percentile(0.99)

    if p95 > 1.0 {
        alert("P95 latency above 1 second: ", p95)
    }
    if p99 > 5.0 {
        alert("P99 latency above 5 seconds: ", p99)
    }
}
\`\`\`

**Prometheus Format:**
\`\`\`go
// Histogram output includes:
// 1. Buckets with cumulative counts
http_request_duration_seconds_bucket{le="0.1"} 150
http_request_duration_seconds_bucket{le="0.5"} 280
http_request_duration_seconds_bucket{le="+Inf"} 300

// 2. Sum of all observations
http_request_duration_seconds_sum 245.3

// 3. Total count
http_request_duration_seconds_count 300

// From this, Prometheus can calculate:
// - Average: sum / count = 0.818 seconds
// - Rate: rate(count[5m]) = requests per second
// - Percentiles: histogram_quantile(0.95, ...)
\`\`\`

**Constraints:**
- Must use sync.RWMutex for thread safety
- Buckets are cumulative (each bucket includes all smaller values)
- Always include +Inf bucket for all observations
- Percentile must interpolate between buckets
- Return 0 for percentile if no observations
- RenderHistogram must format with _bucket{le="..."}, _sum, _count suffixes`,
	initialCode: `package metricsx

import (
	"fmt"
	"sort"
	"sync"
)

// TODO: Implement Histogram type
// Store buckets (upper bounds), sum, count
type Histogram struct {
	mu      sync.RWMutex
	buckets []float64         // sorted bucket upper bounds
	counts  map[float64]int64 // bucket upper bound -> cumulative count
	sum     float64           // sum of all observations
	count   int64             // total observations
}

// TODO: Implement NewHistogram
// Initialize histogram with bucket boundaries
// Add +Inf bucket automatically
func NewHistogram(buckets []float64) *Histogram {
	// TODO: Implement
}

// TODO: Implement Observe
// Record value in appropriate buckets (cumulative)
// Update sum and count
func (h *Histogram) Observe(value float64) {
	// TODO: Implement
}

// TODO: Implement Percentile
// Calculate percentile (0.50 = median, 0.95 = p95, 0.99 = p99)
// Interpolate between buckets for estimate
// Return 0 if no observations
func (h *Histogram) Percentile(p float64) float64 {
	// TODO: Implement
}

// TODO: Implement RenderHistogram
// Format as Prometheus histogram with:
// - Buckets: name_bucket{le="value"} count
// - Sum: name_sum value
// - Count: name_count value
func RenderHistogram(name string, h *Histogram) string {
	return "" // TODO: Implement
}`,
	solutionCode: `package metricsx

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

type Histogram struct {
	mu      sync.RWMutex
	buckets []float64	// sorted bucket upper bounds including +Inf
	counts  map[float64]int64	// bucket upper bound -> cumulative count
	sum     float64	// sum of all observations
	count   int64	// total observations
}

func NewHistogram(buckets []float64) *Histogram {
	sorted := make([]float64, len(buckets)+1)	// allocate space for buckets plus +Inf
	copy(sorted, buckets)	// copy provided buckets
	sorted[len(buckets)] = math.Inf(1)	// add +Inf bucket to catch all values
	sort.Float64s(sorted)	// ensure buckets are sorted for proper operation

	counts := make(map[float64]int64)	// initialize counts map
	for _, bucket := range sorted {	// pre-initialize all buckets to zero
		counts[bucket] = 0
	}

	return &Histogram{
		buckets: sorted,
		counts:  counts,
	}
}

func (h *Histogram) Observe(value float64) {
	h.mu.Lock()	// acquire write lock for modifications
	defer h.mu.Unlock()

	for _, bucket := range h.buckets {	// iterate through all buckets
		if value <= bucket {	// observation fits in this bucket
			h.counts[bucket]++	// increment this bucket (cumulative counting)
		}
	}

	h.sum += value	// add to sum for average calculation
	h.count++	// increment total observation count
}

func (h *Histogram) Percentile(p float64) float64 {
	h.mu.RLock()	// acquire read lock for safe access
	defer h.mu.RUnlock()

	if h.count == 0 {	// no data to calculate percentile from
		return 0
	}

	targetCount := float64(h.count) * p	// calculate target observation position
	var cumCount int64	// track cumulative count across buckets

	for i, bucket := range h.buckets {	// iterate buckets to find percentile
		cumCount = h.counts[bucket]	// get cumulative count at this bucket
		if float64(cumCount) >= targetCount {	// percentile falls in this bucket
			if i == 0 {	// first bucket, use bucket as estimate
				return bucket
			}
			prevBucket := h.buckets[i-1]	// get previous bucket for interpolation
			prevCount := h.counts[prevBucket]	// get previous cumulative count

			// Linear interpolation between buckets
			if cumCount == prevCount {	// avoid division by zero
				return bucket
			}

			ratio := (targetCount - float64(prevCount)) / float64(cumCount-prevCount)	// calculate position within bucket
			return prevBucket + ratio*(bucket-prevBucket)	// interpolate value between bucket boundaries
		}
	}

	return h.buckets[len(h.buckets)-1]	// fallback to last bucket
}

func RenderHistogram(name string, h *Histogram) string {
	if h == nil {	// handle nil histogram gracefully
		return ""
	}

	h.mu.RLock()	// acquire read lock for rendering
	defer h.mu.RUnlock()

	var result string

	// Render buckets with cumulative counts
	for _, bucket := range h.buckets {
		leValue := "+Inf"	// default label for infinity bucket
		if !math.IsInf(bucket, 1) {	// not infinity, use numeric value
			leValue = fmt.Sprintf("%g", bucket)
		}
		count := h.counts[bucket]	// get cumulative count for bucket
		result += fmt.Sprintf("%s_bucket{le=\"%s\"} %d\n", name, leValue, count)	// format prometheus bucket line
	}

	// Render sum and count
	result += fmt.Sprintf("%s_sum %g\n", name, h.sum)	// total sum of all observations
	result += fmt.Sprintf("%s_count %d\n", name, h.count)	// total number of observations

	return result
}`,
	testCode: `package metricsx

import (
	"math"
	"strings"
	"sync"
	"testing"
)

func Test1(t *testing.T) {
	// NewHistogram creates histogram with +Inf bucket
	h := NewHistogram([]float64{0.1, 0.5, 1.0})
	if len(h.buckets) != 4 {
		t.Errorf("expected 4 buckets (including +Inf), got %d", len(h.buckets))
	}
	if !math.IsInf(h.buckets[len(h.buckets)-1], 1) {
		t.Error("last bucket should be +Inf")
	}
}

func Test2(t *testing.T) {
	// Observe increments appropriate buckets (cumulative)
	h := NewHistogram([]float64{0.1, 0.5, 1.0})
	h.Observe(0.3)
	if h.counts[0.5] != 1 {
		t.Errorf("expected bucket 0.5 to have count 1, got %d", h.counts[0.5])
	}
	if h.counts[1.0] != 1 {
		t.Errorf("expected bucket 1.0 to have count 1, got %d", h.counts[1.0])
	}
	if h.counts[0.1] != 0 {
		t.Errorf("expected bucket 0.1 to have count 0, got %d", h.counts[0.1])
	}
}

func Test3(t *testing.T) {
	// Observe updates sum and count
	h := NewHistogram([]float64{1.0, 5.0})
	h.Observe(2.5)
	h.Observe(3.5)
	if h.sum != 6.0 {
		t.Errorf("expected sum 6.0, got %g", h.sum)
	}
	if h.count != 2 {
		t.Errorf("expected count 2, got %d", h.count)
	}
}

func Test4(t *testing.T) {
	// Percentile returns 0 for empty histogram
	h := NewHistogram([]float64{1.0, 5.0})
	p50 := h.Percentile(0.50)
	if p50 != 0 {
		t.Errorf("expected 0 for empty histogram, got %g", p50)
	}
}

func Test5(t *testing.T) {
	// Percentile calculates correctly
	h := NewHistogram([]float64{0.1, 0.5, 1.0, 5.0})
	// Add observations to buckets
	for i := 0; i < 50; i++ {
		h.Observe(0.05) // goes to 0.1 bucket
	}
	for i := 0; i < 50; i++ {
		h.Observe(0.8) // goes to 1.0 bucket
	}
	// P50 should be around 0.1 (median is at 50th observation)
	p50 := h.Percentile(0.50)
	if p50 > 0.2 {
		t.Errorf("expected p50 around 0.1, got %g", p50)
	}
}

func Test6(t *testing.T) {
	// RenderHistogram returns empty for nil
	result := RenderHistogram("test", nil)
	if result != "" {
		t.Errorf("expected empty string for nil, got '%s'", result)
	}
}

func Test7(t *testing.T) {
	// RenderHistogram includes _bucket, _sum, _count
	h := NewHistogram([]float64{0.5, 1.0})
	h.Observe(0.3)
	result := RenderHistogram("latency", h)
	if !strings.Contains(result, "latency_bucket") {
		t.Error("missing _bucket in output")
	}
	if !strings.Contains(result, "latency_sum") {
		t.Error("missing _sum in output")
	}
	if !strings.Contains(result, "latency_count") {
		t.Error("missing _count in output")
	}
}

func Test8(t *testing.T) {
	// RenderHistogram includes +Inf bucket
	h := NewHistogram([]float64{1.0})
	h.Observe(0.5)
	result := RenderHistogram("test", h)
	if !strings.Contains(result, \`le="+Inf"\`) {
		t.Error("missing +Inf bucket")
	}
}

func Test9(t *testing.T) {
	// Concurrent observations are thread-safe
	h := NewHistogram([]float64{1.0, 5.0, 10.0})
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			h.Observe(2.5)
		}()
	}
	wg.Wait()
	if h.count != 100 {
		t.Errorf("expected 100 observations, got %d", h.count)
	}
}

func Test10(t *testing.T) {
	// Buckets are sorted
	h := NewHistogram([]float64{5.0, 0.1, 1.0})
	prev := 0.0
	for _, b := range h.buckets {
		if b < prev {
			t.Errorf("buckets not sorted: %v", h.buckets)
			break
		}
		prev = b
	}
}
`,
	hint1: `For Observe, iterate through all buckets and increment those where value <= bucket (cumulative). Update sum and count.`,
	hint2: `For Percentile, calculate targetCount = count * p, then find the bucket where cumulative count >= targetCount. Interpolate between buckets.`,
	whyItMatters: `Histograms are essential for understanding latency distribution and performance characteristics in production systems.

**Why This Matters:**

**1. Why Not Just Average?**
Averages hide critical performance issues:
\`\`\`go
// API endpoint response times
Requests: 100
Average: 150ms ← Looks good!

// But looking at distribution:
P50 (median):  50ms   ← Fast for most users
P95:          200ms   ← Still acceptable
P99:         5000ms   ← 1% of users suffer!

// The slow 1% ruins user experience
// But average masks the problem
\`\`\`

**2. Real Production Scenario**
E-commerce checkout endpoint monitoring:
\`\`\`go
var checkoutLatency = NewHistogram([]float64{
    0.1,   // 100ms - excellent
    0.5,   // 500ms - good
    1.0,   // 1 second - acceptable
    2.0,   // 2 seconds - slow
    5.0,   // 5 seconds - very slow
    10.0,  // 10 seconds - timeout risk
})

func HandleCheckout(w http.ResponseWriter, r *http.Request) {
    start := time.Now()

    // Process payment
    if err := processPayment(r); err != nil {
        http.Error(w, "Payment failed", 500)
        return
    }

    // Record latency
    duration := time.Since(start).Seconds()
    checkoutLatency.Observe(duration)

    w.WriteHeader(200)
}

// Alert on SLA violations
func MonitorSLA() {
    ticker := time.NewTicker(1 * time.Minute)
    for range ticker.C {
        p50 := checkoutLatency.Percentile(0.50)
        p95 := checkoutLatency.Percentile(0.95)
        p99 := checkoutLatency.Percentile(0.99)

        log.Printf("Checkout latency - P50: %.2fs, P95: %.2fs, P99: %.2fs", p50, p95, p99)

        // SLA: 95% of requests under 1 second
        if p95 > 1.0 {
            alertPagerDuty("SLA violation: P95 latency %.2fs", p95)
        }

        // Critical: 99% under 5 seconds
        if p99 > 5.0 {
            alertPagerDuty("CRITICAL: P99 latency %.2fs", p99)
        }
    }
}
\`\`\`

**3. Discovering Performance Bottlenecks**
Histogram reveals the problem:
\`\`\`go
// Before optimization:
P50: 0.3s  ← Most requests OK
P95: 2.5s  ← 5% too slow
P99: 8.0s  ← 1% terrible

// Investigate: What makes 5% slow?
// Check histogram buckets:
// 0-0.5s:  85% ← Fast path
// 0.5-1s:  10% ← Normal path
// 1-5s:     4% ← Cache misses?
// 5-10s:    1% ← Database timeouts!

// Fix: Add caching + connection pooling

// After optimization:
P50: 0.2s  ← Improved
P95: 0.4s  ← Much better!
P99: 1.2s  ← Acceptable now
\`\`\`

**4. Understanding Load Patterns**
\`\`\`go
// Normal traffic pattern:
http_request_duration_seconds_bucket{le="0.1"} 850   ← 85% fast
http_request_duration_seconds_bucket{le="0.5"} 950   ← 95% acceptable
http_request_duration_seconds_bucket{le="1.0"} 990   ← 99% under 1s
http_request_duration_seconds_bucket{le="+Inf"} 1000

// During traffic spike:
http_request_duration_seconds_bucket{le="0.1"} 300   ← Only 30% fast now
http_request_duration_seconds_bucket{le="0.5"} 600   ← 60% acceptable
http_request_duration_seconds_bucket{le="1.0"} 800   ← 80% under 1s
http_request_duration_seconds_bucket{le="5.0"} 950   ← 5% very slow
http_request_duration_seconds_bucket{le="+Inf"} 1000

// Diagnosis: System overloaded, needs scaling
\`\`\`

**5. Percentiles for SLA Contracts**
Many companies define SLAs using percentiles:
\`\`\`go
// Netflix SLA example:
// "99% of API requests complete in under 300ms"
//  Translation: P99 < 300ms

// Monitoring code:
func CheckNetflixSLA() bool {
    p99 := apiLatency.Percentile(0.99)
    return p99 < 0.3  // 300ms in seconds
}

// Stripe payment API SLA:
// "95% of API calls complete in under 500ms"
func CheckStripeSLA() bool {
    p95 := paymentLatency.Percentile(0.95)
    return p95 < 0.5
}
\`\`\`

**6. Bucket Design Matters**
Choose buckets based on your SLAs:
\`\`\`go
// Web API - user experience focused
[]float64{0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0}
// Buckets match user perception thresholds

// Background job processing
[]float64{1, 5, 10, 30, 60, 300, 600}
// Longer timescales for batch operations

// Database queries
[]float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0}
// Fine granularity for query optimization
\`\`\`

**7. Prometheus Queries on Histograms**
\`\`\`promql
# Calculate P95 latency over last 5 minutes
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Alert if P99 > 1 second
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1

# Compare P50 vs P95 (detect outliers)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
/
histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))
# If ratio > 5, you have outlier problem
\`\`\`

**8. Real-World Impact**
Ride-sharing company tracking pickup time:
- **Before histograms**: "Average pickup time: 6 minutes"
- **After histograms**:
  - P50: 4 minutes ← Most pickups fine
  - P95: 15 minutes ← 5% waiting too long
  - P99: 35 minutes ← 1% terrible experience
- **Action**: Discovered certain neighborhoods had poor driver coverage
- **Result**:
  - Adjusted driver incentives for those areas
  - P95 dropped to 8 minutes
  - Customer satisfaction +20%
  - Churn rate -15%

**Production Best Practices:**
1. Always track latency with histograms, not just averages
2. Choose buckets based on SLA boundaries
3. Monitor P50, P95, P99 percentiles
4. Alert on percentile violations, not averages
5. Use histograms for any metric with distribution (latency, size, duration)
6. Keep bucket count reasonable (5-10 buckets typically sufficient)
7. Include +Inf bucket to capture all observations

Histograms reveal the full story of your system's performance, not just the average case.`,
	order: 2,
	translations: {
		ru: {
			title: 'Гистограмма задержки запросов',
			solutionCode: `package metricsx

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

type Histogram struct {
	mu      sync.RWMutex
	buckets []float64         // отсортированные верхние границы включая +Inf
	counts  map[float64]int64 // верхняя граница -> кумулятивный счет
	sum     float64           // сумма всех наблюдений
	count   int64             // общее количество наблюдений
}

func NewHistogram(buckets []float64) *Histogram {
	sorted := make([]float64, len(buckets)+1)           // выделяем место для корзин плюс +Inf
	copy(sorted, buckets)                               // копируем предоставленные корзины
	sorted[len(buckets)] = math.Inf(1)                  // добавляем +Inf корзину для захвата всех значений
	sort.Float64s(sorted)                               // гарантируем сортировку корзин для правильной работы

	counts := make(map[float64]int64)                   // инициализируем карту счетчиков
	for _, bucket := range sorted {                      // предварительно инициализируем все корзины нулями
		counts[bucket] = 0
	}

	return &Histogram{
		buckets: sorted,
		counts:  counts,
	}
}

func (h *Histogram) Observe(value float64) {
	h.mu.Lock()                                         // захватываем блокировку записи для модификаций
	defer h.mu.Unlock()

	for _, bucket := range h.buckets {                   // итерируем по всем корзинам
		if value <= bucket {                              // наблюдение подходит в эту корзину
			h.counts[bucket]++                             // инкрементируем эту корзину (кумулятивный подсчет)
		}
	}

	h.sum += value                                       // добавляем к сумме для расчета среднего
	h.count++                                            // инкрементируем общее количество наблюдений
}

func (h *Histogram) Percentile(p float64) float64 {
	h.mu.RLock()                                        // захватываем блокировку чтения для безопасного доступа
	defer h.mu.RUnlock()

	if h.count == 0 {                                    // нет данных для расчета персентиля
		return 0
	}

	targetCount := float64(h.count) * p                  // вычисляем целевую позицию наблюдения
	var cumCount int64                                   // отслеживаем кумулятивный счет по корзинам

	for i, bucket := range h.buckets {                   // итерируем корзины для поиска персентиля
		cumCount = h.counts[bucket]                       // получаем кумулятивный счет в этой корзине
		if float64(cumCount) >= targetCount {             // персентиль попадает в эту корзину
			if i == 0 {                                    // первая корзина, используем корзину как оценку
				return bucket
			}
			prevBucket := h.buckets[i-1]                  // получаем предыдущую корзину для интерполяции
			prevCount := h.counts[prevBucket]             // получаем предыдущий кумулятивный счет

			// Линейная интерполяция между корзинами
			if cumCount == prevCount {                     // избегаем деления на ноль
				return bucket
			}

			ratio := (targetCount - float64(prevCount)) / float64(cumCount-prevCount)  // вычисляем позицию внутри корзины
			return prevBucket + ratio*(bucket-prevBucket)  // интерполируем значение между границами корзин
		}
	}

	return h.buckets[len(h.buckets)-1]                   // запасной вариант - последняя корзина
}

func RenderHistogram(name string, h *Histogram) string {
	if h == nil {                                        // обрабатываем nil гистограмму gracefully
		return ""
	}

	h.mu.RLock()                                        // захватываем блокировку чтения для рендеринга
	defer h.mu.RUnlock()

	var result string

	// Рендерим корзины с кумулятивными счетчиками
	for _, bucket := range h.buckets {
		leValue := "+Inf"                                 // метка по умолчанию для бесконечной корзины
		if !math.IsInf(bucket, 1) {                       // не бесконечность, используем числовое значение
			leValue = fmt.Sprintf("%g", bucket)
		}
		count := h.counts[bucket]                         // получаем кумулятивный счет для корзины
		result += fmt.Sprintf("%s_bucket{le=\"%s\"} %d\n", name, leValue, count)  // форматируем строку prometheus корзины
	}

	// Рендерим sum и count
	result += fmt.Sprintf("%s_sum %g\n", name, h.sum)    // общая сумма всех наблюдений
	result += fmt.Sprintf("%s_count %d\n", name, h.count) // общее количество наблюдений

	return result
}`,
			description: `Реализуйте гистограмму для отслеживания распределения задержки запросов и расчета перцентилей.

**Требования:**
1. **Histogram Type**: Хранить buckets map[float64]int64 и sum/count для отслеживания
2. **Observe(value)**: Записать наблюдение в соответствующую корзину и обновить sum/count
3. **Percentile(p)**: Вычислить перцентиль (например, p50, p95, p99) из распределения
4. **RenderHistogram**: Отформатировать гистограмму как Prometheus метрику с корзинами и квантилями

**Ключевые концепции:**
- Корзины гистограммы определяют диапазоны (0.1, 0.5, 1.0, 5.0, 10.0 секунд)
- Каждое наблюдение инкрементирует все корзины >= значения (кумулятивно)
- Перцентили оценивают распределение (p50=медиана, p95, p99)
- Sum/count отслеживают общую сумму для расчета среднего
- Потокобезопасно с RWMutex для конкурентных наблюдений

**Ограничения:**
- Должен использовать sync.RWMutex для потокобезопасности
- Корзины кумулятивные (каждая корзина включает все меньшие значения)
- Всегда включать +Inf корзину для всех наблюдений
- Percentile должен интерполировать между корзинами
- Возвращать 0 для перцентиля если нет наблюдений
- RenderHistogram должен форматировать с суффиксами _bucket{le="..."}, _sum, _count`,
			hint1: `Для Observe итерируйте по всем корзинам и инкрементируйте те, где value <= bucket (кумулятивно). Обновите sum и count.`,
			hint2: `Для Percentile вычислите targetCount = count * p, затем найдите корзину где кумулятивный счет >= targetCount. Интерполируйте между корзинами.`,
			whyItMatters: `Гистограммы необходимы для понимания распределения задержки и характеристик производительности в production системах.

**Почему важно:**

**1. Почему не просто среднее?**
Среднее скрывает критические проблемы производительности:
\`\`\`go
// API endpoint время ответа
Запросов: 100
Среднее: 150ms ← Выглядит хорошо!

// Но смотря на распределение:
P50 (медиана):  50ms   ← Быстро для большинства пользователей
P95:          200ms   ← Все еще приемлемо
P99:         5000ms   ← 1% пользователей страдают!

// Медленный 1% портит user experience
// Но среднее маскирует проблему
\`\`\`

**2. Real Production сценарий**
Мониторинг E-commerce checkout endpoint:
\`\`\`go
var checkoutLatency = NewHistogram([]float64{
    0.1,   // 100ms - отлично
    0.5,   // 500ms - хорошо
    1.0,   // 1 секунда - приемлемо
    2.0,   // 2 секунды - медленно
    5.0,   // 5 секунд - очень медленно
    10.0,  // 10 секунд - риск timeout
})

func HandleCheckout(w http.ResponseWriter, r *http.Request) {
    start := time.Now()

    // Обрабатываем платеж
    if err := processPayment(r); err != nil {
        http.Error(w, "Payment failed", 500)
        return
    }

    // Записываем задержку
    duration := time.Since(start).Seconds()
    checkoutLatency.Observe(duration)

    w.WriteHeader(200)
}

// Алерт при нарушении SLA
func MonitorSLA() {
    ticker := time.NewTicker(1 * time.Minute)
    for range ticker.C {
        p50 := checkoutLatency.Percentile(0.50)
        p95 := checkoutLatency.Percentile(0.95)
        p99 := checkoutLatency.Percentile(0.99)

        log.Printf("Checkout latency - P50: %.2fs, P95: %.2fs, P99: %.2fs", p50, p95, p99)

        // SLA: 95% запросов под 1 секунду
        if p95 > 1.0 {
            alertPagerDuty("SLA нарушение: P95 latency %.2fs", p95)
        }

        // Критично: 99% под 5 секунд
        if p99 > 5.0 {
            alertPagerDuty("CRITICAL: P99 latency %.2fs", p99)
        }
    }
}
\`\`\`

**3. Обнаружение узких мест производительности**
Гистограмма раскрывает проблему:
\`\`\`go
// До оптимизации:
P50: 0.3s  ← Большинство запросов OK
P95: 2.5s  ← 5% слишком медленно
P99: 8.0s  ← 1% ужасно

// Исследуем: Что делает 5% медленными?
// Проверяем корзины гистограммы:
// 0-0.5s:  85% ← Быстрый путь
// 0.5-1s:  10% ← Нормальный путь
// 1-5s:     4% ← Cache misses?
// 5-10s:    1% ← Database timeouts!

// Исправление: Добавляем кеширование + connection pooling

// После оптимизации:
P50: 0.2s  ← Улучшено
P95: 0.4s  ← Намного лучше!
P99: 1.2s  ← Теперь приемлемо
\`\`\`

**4. Понимание паттернов нагрузки**
\`\`\`go
// Нормальный паттерн трафика:
http_request_duration_seconds_bucket{le="0.1"} 850   ← 85% быстро
http_request_duration_seconds_bucket{le="0.5"} 950   ← 95% приемлемо
http_request_duration_seconds_bucket{le="1.0"} 990   ← 99% под 1s
http_request_duration_seconds_bucket{le="+Inf"} 1000

// Во время спайка трафика:
http_request_duration_seconds_bucket{le="0.1"} 300   ← Только 30% быстро сейчас
http_request_duration_seconds_bucket{le="0.5"} 600   ← 60% приемлемо
http_request_duration_seconds_bucket{le="1.0"} 800   ← 80% под 1s
http_request_duration_seconds_bucket{le="5.0"} 950   ← 5% очень медленно
http_request_duration_seconds_bucket{le="+Inf"} 1000

// Диагноз: Система перегружена, нужно масштабирование
\`\`\`

**5. Перцентили для SLA контрактов**
Многие компании определяют SLA используя перцентили:
\`\`\`go
// Пример Netflix SLA:
// "99% API запросов завершаются менее чем за 300ms"
//  Перевод: P99 < 300ms

// Код мониторинга:
func CheckNetflixSLA() bool {
    p99 := apiLatency.Percentile(0.99)
    return p99 < 0.3  // 300ms в секундах
}

// Stripe payment API SLA:
// "95% API вызовов завершаются менее чем за 500ms"
func CheckStripeSLA() bool {
    p95 := paymentLatency.Percentile(0.95)
    return p95 < 0.5
}
\`\`\`

**6. Дизайн корзин важен**
Выбирайте корзины на основе ваших SLA:
\`\`\`go
// Web API - фокус на user experience
[]float64{0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0}
// Корзины соответствуют порогам восприятия пользователя

// Background job обработка
[]float64{1, 5, 10, 30, 60, 300, 600}
// Более длинные временные шкалы для batch операций

// Database запросы
[]float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0}
// Точная гранулярность для оптимизации запросов
\`\`\`

**7. Prometheus запросы на гистограммах**
\`\`\`promql
# Вычислить P95 latency за последние 5 минут
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Алерт если P99 > 1 секунды
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1

# Сравнить P50 vs P95 (обнаружить выбросы)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
/
histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))
# Если соотношение > 5, у вас проблема с выбросами
\`\`\`

**8. Real-World влияние**
Компания ride-sharing отслеживает время подачи:
- **До гистограмм**: "Среднее время подачи: 6 минут"
- **После гистограмм**:
  - P50: 4 минуты ← Большинство подач в порядке
  - P95: 15 минут ← 5% ждут слишком долго
  - P99: 35 минут ← 1% ужасный опыт
- **Действие**: Обнаружили определенные районы с плохим покрытием водителями
- **Результат**:
  - Скорректировали стимулы водителей для этих районов
  - P95 упал до 8 минут
  - Удовлетворенность клиентов +20%
  - Churn rate -15%

**Production лучшие практики:**
1. Всегда отслеживайте задержку гистограммами, не только средним
2. Выбирайте корзины на основе границ SLA
3. Мониторьте перцентили P50, P95, P99
4. Алерты на нарушения перцентилей, не средних
5. Используйте гистограммы для любых метрик с распределением (latency, size, duration)
6. Держите количество корзин разумным (5-10 корзин обычно достаточно)
7. Включайте корзину +Inf для захвата всех наблюдений

Гистограммы раскрывают полную историю производительности вашей системы, не только средний случай.`
		},
		uz: {
			title: `Request kechikish histogrammasi`,
			solutionCode: `package metricsx

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

type Histogram struct {
	mu      sync.RWMutex
	buckets []float64         // +Inf ni o'z ichiga olgan saralangan yuqori chegaralar
	counts  map[float64]int64 // yuqori chegara -> kumulyativ hisob
	sum     float64           // barcha kuzatishlarning yig'indisi
	count   int64             // jami kuzatishlar
}

func NewHistogram(buckets []float64) *Histogram {
	sorted := make([]float64, len(buckets)+1)           // bucket'lar va +Inf uchun joy ajratamiz
	copy(sorted, buckets)                               // taqdim etilgan bucket'larni nusxalaymiz
	sorted[len(buckets)] = math.Inf(1)                  // barcha qiymatlarni ushlab qolish uchun +Inf bucket qo'shamiz
	sort.Float64s(sorted)                               // to'g'ri ishlash uchun bucket'larni saralanganligini ta'minlaymiz

	counts := make(map[float64]int64)                   // hisoblagichlar xaritasini initsializatsiya qilamiz
	for _, bucket := range sorted {                      // barcha bucket'larni nolga oldindan initsializatsiya qilamiz
		counts[bucket] = 0
	}

	return &Histogram{
		buckets: sorted,
		counts:  counts,
	}
}

func (h *Histogram) Observe(value float64) {
	h.mu.Lock()                                         // o'zgartirishlar uchun yozish qulfini olamiz
	defer h.mu.Unlock()

	for _, bucket := range h.buckets {                   // barcha bucket'lar bo'ylab takrorlaymiz
		if value <= bucket {                              // kuzatish bu bucket'ga mos keladi
			h.counts[bucket]++                             // bu bucket'ni oshiramiz (kumulyativ hisoblash)
		}
	}

	h.sum += value                                       // o'rtacha hisoblash uchun yig'indiga qo'shamiz
	h.count++                                            // jami kuzatishlar sonini oshiramiz
}

func (h *Histogram) Percentile(p float64) float64 {
	h.mu.RLock()                                        // xavfsiz kirish uchun o'qish qulfini olamiz
	defer h.mu.RUnlock()

	if h.count == 0 {                                    // persentil hisoblash uchun ma'lumot yo'q
		return 0
	}

	targetCount := float64(h.count) * p                  // maqsadli kuzatish pozitsiyasini hisoblaymiz
	var cumCount int64                                   // bucket'lar bo'ylab kumulyativ hisobni kuzatamiz

	for i, bucket := range h.buckets {                   // persentilni topish uchun bucket'larni takrorlaymiz
		cumCount = h.counts[bucket]                       // bu bucket'dagi kumulyativ hisobni olamiz
		if float64(cumCount) >= targetCount {             // persentil bu bucket'ga tushadi
			if i == 0 {                                    // birinchi bucket, bucket'ni baholash sifatida ishlatamiz
				return bucket
			}
			prevBucket := h.buckets[i-1]                  // interpolatsiya uchun oldingi bucket'ni olamiz
			prevCount := h.counts[prevBucket]             // oldingi kumulyativ hisobni olamiz

			// Bucket'lar orasida chiziqli interpolatsiya
			if cumCount == prevCount {                     // nolga bo'lishdan qochamiz
				return bucket
			}

			ratio := (targetCount - float64(prevCount)) / float64(cumCount-prevCount)  // bucket ichidagi pozitsiyani hisoblaymiz
			return prevBucket + ratio*(bucket-prevBucket)  // bucket chegaralari orasida qiymatni interpolatsiya qilamiz
		}
	}

	return h.buckets[len(h.buckets)-1]                   // zaxira variant - oxirgi bucket
}

func RenderHistogram(name string, h *Histogram) string {
	if h == nil {                                        // nil histogramni gracefully qayta ishlaymiz
		return ""
	}

	h.mu.RLock()                                        // render qilish uchun o'qish qulfini olamiz
	defer h.mu.RUnlock()

	var result string

	// Kumulyativ hisoblar bilan bucket'larni render qilamiz
	for _, bucket := range h.buckets {
		leValue := "+Inf"                                 // cheksiz bucket uchun standart yorliq
		if !math.IsInf(bucket, 1) {                       // cheksizlik emas, raqamli qiymatni ishlatamiz
			leValue = fmt.Sprintf("%g", bucket)
		}
		count := h.counts[bucket]                         // bucket uchun kumulyativ hisobni olamiz
		result += fmt.Sprintf("%s_bucket{le=\"%s\"} %d\n", name, leValue, count)  // prometheus bucket qatorini formatlaymiz
	}

	// Sum va count ni render qilamiz
	result += fmt.Sprintf("%s_sum %g\n", name, h.sum)    // barcha kuzatishlarning umumiy yig'indisi
	result += fmt.Sprintf("%s_count %d\n", name, h.count) // jami kuzatishlar soni

	return result
}`,
			description: `So'rov kechikishi taqsimotini kuzatish va persentillarni hisoblash uchun histogram amalga oshiring.

**Talablar:**
1. **Histogram Turi**: Kuzatish uchun buckets map[float64]int64 va sum/count saqlash
2. **Observe(value)**: Tegishli bucket'da kuzatishni yozish va sum/count ni yangilash
3. **Percentile(p)**: Taqsimotdan persentilni hisoblash (masalan, p50, p95, p99)
4. **RenderHistogram**: Histogram'ni bucket'lar va kvantilar bilan Prometheus metrikasi sifatida formatlash

**Asosiy tushunchalar:**
- Histogram bucket'lari diapazonlarni belgilaydi (0.1, 0.5, 1.0, 5.0, 10.0 soniya)
- Har bir kuzatish barcha bucket'larni >= qiymatga oshiradi (kumulyativ)
- Persentillar taqsimotni baholaydi (p50=median, p95, p99)
- Sum/count o'rtacha hisoblash uchun umumiy yig'indini kuzatadi
- Parallel kuzatishlar uchun RWMutex bilan thread-safe

**Cheklovlar:**
- Thread xavfsizligi uchun sync.RWMutex ishlatish kerak
- Bucket'lar kumulyativ (har bir bucket barcha kichikroq qiymatlarni o'z ichiga oladi)
- Barcha kuzatishlar uchun har doim +Inf bucket qo'shing
- Percentile bucket'lar orasida interpolatsiya qilishi kerak
- Agar kuzatishlar bo'lmasa persentil uchun 0 qaytaring
- RenderHistogram _bucket{le="..."}, _sum, _count qo'shimchalari bilan formatlashi kerak`,
			hint1: `Observe uchun barcha bucket'larni takrorlang va value <= bucket bo'lgan joyda oshiring (kumulyativ). sum va count ni yangilang.`,
			hint2: `Percentile uchun targetCount = count * p ni hisoblang, keyin kumulyativ hisob >= targetCount bo'lgan bucket'ni toping. Bucket'lar orasida interpolatsiya qiling.`,
			whyItMatters: `Histogramlar production tizimlarda kechikish taqsimoti va ishlash xususiyatlarini tushunish uchun muhimdir.

**Nima uchun bu muhim:**

**1. Nega faqat o'rtacha emas?**
O'rtacha muhim ishlash muammolarini yashiradi:
\`\`\`go
// API endpoint javob vaqti
So'rovlar: 100
O'rtacha: 150ms ← Yaxshi ko'rinadi!

// Ammo taqsimotga qarab:
P50 (median):  50ms   ← Ko'pchilik foydalanuvchilar uchun tez
P95:          200ms   ← Hali ham qabul qilinadigan
P99:         5000ms   ← 1% foydalanuvchilar azob chekadi!

// Sekin 1% foydalanuvchi tajribasini buzadi
// Ammo o'rtacha muammoni yashiradi
\`\`\`

**2. Haqiqiy Production stsenariysi**
E-commerce checkout endpoint monitoringi:
\`\`\`go
var checkoutLatency = NewHistogram([]float64{
    0.1,   // 100ms - a'lo
    0.5,   // 500ms - yaxshi
    1.0,   // 1 soniya - qabul qilinadigan
    2.0,   // 2 soniya - sekin
    5.0,   // 5 soniya - juda sekin
    10.0,  // 10 soniya - timeout xavfi
})

func HandleCheckout(w http.ResponseWriter, r *http.Request) {
    start := time.Now()

    // To'lovni qayta ishlaymiz
    if err := processPayment(r); err != nil {
        http.Error(w, "Payment failed", 500)
        return
    }

    // Kechikishni yozamiz
    duration := time.Since(start).Seconds()
    checkoutLatency.Observe(duration)

    w.WriteHeader(200)
}

// SLA buzilganda alert
func MonitorSLA() {
    ticker := time.NewTicker(1 * time.Minute)
    for range ticker.C {
        p50 := checkoutLatency.Percentile(0.50)
        p95 := checkoutLatency.Percentile(0.95)
        p99 := checkoutLatency.Percentile(0.99)

        log.Printf("Checkout latency - P50: %.2fs, P95: %.2fs, P99: %.2fs", p50, p95, p99)

        // SLA: 95% so'rovlar 1 soniya ostida
        if p95 > 1.0 {
            alertPagerDuty("SLA buzilishi: P95 latency %.2fs", p95)
        }

        // Kritik: 99% 5 soniya ostida
        if p99 > 5.0 {
            alertPagerDuty("CRITICAL: P99 latency %.2fs", p99)
        }
    }
}
\`\`\`

**3. Ishlash tor joylarini aniqlash**
Histogram muammoni ochib beradi:
\`\`\`go
// Optimizatsiyadan oldin:
P50: 0.3s  ← Ko'pchilik so'rovlar OK
P95: 2.5s  ← 5% juda sekin
P99: 8.0s  ← 1% dahshatli

// Tekshiramiz: 5% ni nima sekin qiladi?
// Histogram bucketlarini tekshiramiz:
// 0-0.5s:  85% ← Tez yo'l
// 0.5-1s:  10% ← Oddiy yo'l
// 1-5s:     4% ← Cache misslar?
// 5-10s:    1% ← Database timeoutlar!

// Tuzatish: Keshlash + connection pooling qo'shamiz

// Optimizatsiyadan keyin:
P50: 0.2s  ← Yaxshilandi
P95: 0.4s  ← Ancha yaxshi!
P99: 1.2s  ← Endi qabul qilinadigan
\`\`\`

**4. Yuk patternlarini tushunish**
\`\`\`go
// Oddiy trafik patterni:
http_request_duration_seconds_bucket{le="0.1"} 850   ← 85% tez
http_request_duration_seconds_bucket{le="0.5"} 950   ← 95% qabul qilinadigan
http_request_duration_seconds_bucket{le="1.0"} 990   ← 99% 1s ostida
http_request_duration_seconds_bucket{le="+Inf"} 1000

// Trafik portlashi paytida:
http_request_duration_seconds_bucket{le="0.1"} 300   ← Faqat 30% tez hozir
http_request_duration_seconds_bucket{le="0.5"} 600   ← 60% qabul qilinadigan
http_request_duration_seconds_bucket{le="1.0"} 800   ← 80% 1s ostida
http_request_duration_seconds_bucket{le="5.0"} 950   ← 5% juda sekin
http_request_duration_seconds_bucket{le="+Inf"} 1000

// Diagnoz: Tizim ortiqcha yuklangan, masshtablash kerak
\`\`\`

**5. SLA shartnomalar uchun persentillar**
Ko'p kompaniyalar persentillardan foydalanib SLA ni belgilaydilar:
\`\`\`go
// Netflix SLA misoli:
// "99% API so'rovlari 300ms dan kam vaqtda tugaydi"
//  Tarjima: P99 < 300ms

// Monitoring kodi:
func CheckNetflixSLA() bool {
    p99 := apiLatency.Percentile(0.99)
    return p99 < 0.3  // soniyalarda 300ms
}

// Stripe to'lov API SLA:
// "95% API chaqiruvlari 500ms dan kam vaqtda tugaydi"
func CheckStripeSLA() bool {
    p95 := paymentLatency.Percentile(0.95)
    return p95 < 0.5
}
\`\`\`

**6. Bucket dizayni muhim**
SLA asosida bucketlarni tanlang:
\`\`\`go
// Web API - foydalanuvchi tajribasiga fokus
[]float64{0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0}
// Bucketlar foydalanuvchi idrok chegaralariga mos keladi

// Background job qayta ishlash
[]float64{1, 5, 10, 30, 60, 300, 600}
// Batch operatsiyalar uchun uzunroq vaqt shkalasi

// Database so'rovlari
[]float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0}
// So'rovlarni optimallash uchun aniq granulyarlik
\`\`\`

**7. Histogramlarda Prometheus so'rovlari**
\`\`\`promql
# Oxirgi 5 daqiqadagi P95 latency ni hisoblash
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Agar P99 > 1 soniya bo'lsa alert
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1

# P50 vs P95 ni solishtirish (outlierlarni aniqlash)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
/
histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))
# Agar nisbat > 5 bo'lsa, outlier muammosi bor
\`\`\`

**8. Real-World ta'sir**
Ride-sharing kompaniya yetib kelish vaqtini kuzatadi:
- **Histogramlardan oldin**: "O'rtacha yetib kelish vaqti: 6 daqiqa"
- **Histogramlardan keyin**:
  - P50: 4 daqiqa ← Ko'pchilik yetib kelishlar yaxshi
  - P95: 15 daqiqa ← 5% juda uzoq kutadi
  - P99: 35 daqiqa ← 1% dahshatli tajriba
- **Harakat**: Ma'lum hududlarda haydovchilar yetishmasligi aniqlandi
- **Natija**:
  - O'sha hududlar uchun haydovchi stimullarini o'zgartirdilar
  - P95 8 daqiqaga tushdi
  - Mijoz qoniqishi +20%
  - Churn rate -15%

**Production eng yaxshi amaliyotlar:**
1. Har doim kechikishni histogramlar bilan kuzating, faqat o'rtacha bilan emas
2. SLA chegaralari asosida bucketlarni tanlang
3. P50, P95, P99 persentillarini monitoring qiling
4. Persentil buzilishlarida alertlar, o'rtachada emas
5. Taqsimotga ega har qanday metrika uchun histogramlardan foydalaning (latency, size, duration)
6. Bucket sonini oqilona ushlang (5-10 bucket odatda yetarli)
7. Barcha kuzatishlarni qamrab olish uchun +Inf bucket ni qo'shing

Histogramlar tizimingiz ishlashining to'liq tarixini ochib beradi, faqat o'rtacha holatni emas.`
		}
	}
};

export default task;
