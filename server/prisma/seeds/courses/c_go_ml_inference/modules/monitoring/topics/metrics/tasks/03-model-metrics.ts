import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-model-metrics',
	title: 'Model Quality Metrics',
	difficulty: 'hard',
	tags: ['go', 'ml', 'metrics', 'monitoring'],
	estimatedTime: '35m',
	isPremium: true,
	order: 3,
	description: `# Model Quality Metrics

Track model prediction quality metrics in production.

## Task

Implement metrics that:
- Track prediction confidence distribution
- Monitor prediction drift
- Log prediction samples for analysis
- Alert on quality degradation

## Example

\`\`\`go
monitor := NewModelMonitor("model-v1")
monitor.RecordPrediction(prediction, confidence)
if monitor.DetectDrift() {
    alert.Send("Model drift detected")
}
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

// ModelMonitor monitors model quality
type ModelMonitor struct {
	// Your fields here
}

// NewModelMonitor creates a model monitor
func NewModelMonitor(modelID string) *ModelMonitor {
	// Your code here
	return nil
}

// RecordPrediction records a prediction with confidence
func (m *ModelMonitor) RecordPrediction(prediction []float32, confidence float32) {
	// Your code here
}

// DetectDrift checks for prediction distribution drift
func (m *ModelMonitor) DetectDrift() bool {
	// Your code here
	return false
}

// GetConfidenceStats returns confidence statistics
func (m *ModelMonitor) GetConfidenceStats() (avg, min, max float32) {
	// Your code here
	return 0, 0, 0
}

func main() {
	fmt.Println("Model Quality Metrics")
}`,

	solutionCode: `package main

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// PredictionSample stores a prediction for analysis
type PredictionSample struct {
	Prediction []float32
	Confidence float32
	Timestamp  time.Time
	Class      int
}

// DistributionTracker tracks class distribution
type DistributionTracker struct {
	counts      map[int]int64
	total       int64
	windowSize  int64
	windowStart time.Time
	history     []map[int]float64
	maxHistory  int
	mu          sync.RWMutex
}

// NewDistributionTracker creates a distribution tracker
func NewDistributionTracker(windowSize int64, maxHistory int) *DistributionTracker {
	return &DistributionTracker{
		counts:      make(map[int]int64),
		windowSize:  windowSize,
		windowStart: time.Now(),
		history:     make([]map[int]float64, 0, maxHistory),
		maxHistory:  maxHistory,
	}
}

// Record records a class
func (dt *DistributionTracker) Record(class int) {
	dt.mu.Lock()
	defer dt.mu.Unlock()

	dt.counts[class]++
	dt.total++

	// Check if window is complete
	if dt.total >= dt.windowSize {
		dt.saveWindow()
	}
}

// saveWindow saves current distribution to history
func (dt *DistributionTracker) saveWindow() {
	dist := make(map[int]float64)
	for class, count := range dt.counts {
		dist[class] = float64(count) / float64(dt.total)
	}

	if len(dt.history) >= dt.maxHistory {
		dt.history = dt.history[1:]
	}
	dt.history = append(dt.history, dist)

	// Reset window
	dt.counts = make(map[int]int64)
	dt.total = 0
	dt.windowStart = time.Now()
}

// GetDistribution returns current distribution
func (dt *DistributionTracker) GetDistribution() map[int]float64 {
	dt.mu.RLock()
	defer dt.mu.RUnlock()

	dist := make(map[int]float64)
	if dt.total == 0 {
		return dist
	}

	for class, count := range dt.counts {
		dist[class] = float64(count) / float64(dt.total)
	}
	return dist
}

// KLDivergence calculates KL divergence between distributions
func KLDivergence(p, q map[int]float64) float64 {
	var kl float64
	for class, pVal := range p {
		qVal, exists := q[class]
		if !exists || qVal == 0 {
			qVal = 0.0001 // Smoothing
		}
		if pVal > 0 {
			kl += pVal * math.Log(pVal/qVal)
		}
	}
	return kl
}

// ConfidenceTracker tracks confidence scores
type ConfidenceTracker struct {
	sum     float64
	sumSq   float64
	min     float32
	max     float32
	count   int64
	buckets []int64
	mu      sync.RWMutex
}

// NewConfidenceTracker creates a confidence tracker
func NewConfidenceTracker() *ConfidenceTracker {
	return &ConfidenceTracker{
		min:     1.0,
		max:     0.0,
		buckets: make([]int64, 10), // 0-0.1, 0.1-0.2, ..., 0.9-1.0
	}
}

// Record records a confidence value
func (ct *ConfidenceTracker) Record(confidence float32) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	ct.sum += float64(confidence)
	ct.sumSq += float64(confidence) * float64(confidence)
	ct.count++

	if confidence < ct.min {
		ct.min = confidence
	}
	if confidence > ct.max {
		ct.max = confidence
	}

	// Update histogram
	bucket := int(confidence * 10)
	if bucket >= 10 {
		bucket = 9
	}
	ct.buckets[bucket]++
}

// Stats returns confidence statistics
func (ct *ConfidenceTracker) Stats() (avg, stddev, min, max float32) {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	if ct.count == 0 {
		return 0, 0, 0, 0
	}

	avg = float32(ct.sum / float64(ct.count))
	variance := ct.sumSq/float64(ct.count) - float64(avg)*float64(avg)
	if variance > 0 {
		stddev = float32(math.Sqrt(variance))
	}

	return avg, stddev, ct.min, ct.max
}

// ModelMonitor monitors model quality
type ModelMonitor struct {
	modelID        string
	distribution   *DistributionTracker
	confidence     *ConfidenceTracker
	baselineHist   map[int]float64
	driftThreshold float64
	samples        []PredictionSample
	maxSamples     int
	mu             sync.RWMutex
}

// NewModelMonitor creates a model monitor
func NewModelMonitor(modelID string) *ModelMonitor {
	return &ModelMonitor{
		modelID:        modelID,
		distribution:   NewDistributionTracker(1000, 10),
		confidence:     NewConfidenceTracker(),
		driftThreshold: 0.1,
		samples:        make([]PredictionSample, 0, 1000),
		maxSamples:     1000,
	}
}

// SetBaseline sets the baseline distribution
func (m *ModelMonitor) SetBaseline(baseline map[int]float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.baselineHist = baseline
}

// SetDriftThreshold sets the drift detection threshold
func (m *ModelMonitor) SetDriftThreshold(threshold float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.driftThreshold = threshold
}

// RecordPrediction records a prediction with confidence
func (m *ModelMonitor) RecordPrediction(prediction []float32, confidence float32) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Find predicted class (argmax)
	class := 0
	maxVal := prediction[0]
	for i, v := range prediction {
		if v > maxVal {
			maxVal = v
			class = i
		}
	}

	m.distribution.Record(class)
	m.confidence.Record(confidence)

	// Store sample
	if len(m.samples) < m.maxSamples {
		sample := PredictionSample{
			Prediction: make([]float32, len(prediction)),
			Confidence: confidence,
			Timestamp:  time.Now(),
			Class:      class,
		}
		copy(sample.Prediction, prediction)
		m.samples = append(m.samples, sample)
	}
}

// DetectDrift checks for prediction distribution drift
func (m *ModelMonitor) DetectDrift() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.baselineHist == nil {
		return false
	}

	current := m.distribution.GetDistribution()
	if len(current) == 0 {
		return false
	}

	kl := KLDivergence(current, m.baselineHist)
	return kl > m.driftThreshold
}

// GetDriftScore returns the current drift score
func (m *ModelMonitor) GetDriftScore() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.baselineHist == nil {
		return 0
	}

	current := m.distribution.GetDistribution()
	if len(current) == 0 {
		return 0
	}

	return KLDivergence(current, m.baselineHist)
}

// GetConfidenceStats returns confidence statistics
func (m *ModelMonitor) GetConfidenceStats() (avg, min, max float32) {
	avgVal, _, minVal, maxVal := m.confidence.Stats()
	return avgVal, minVal, maxVal
}

// GetDistribution returns current class distribution
func (m *ModelMonitor) GetDistribution() map[int]float64 {
	return m.distribution.GetDistribution()
}

// GetRecentSamples returns recent prediction samples
func (m *ModelMonitor) GetRecentSamples(n int) []PredictionSample {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if n > len(m.samples) {
		n = len(m.samples)
	}

	result := make([]PredictionSample, n)
	copy(result, m.samples[len(m.samples)-n:])
	return result
}

// MonitoringReport generates a monitoring report
type MonitoringReport struct {
	ModelID         string
	Timestamp       time.Time
	ConfidenceAvg   float32
	ConfidenceMin   float32
	ConfidenceMax   float32
	DriftScore      float64
	DriftDetected   bool
	ClassDistribution map[int]float64
}

func (m *ModelMonitor) GenerateReport() MonitoringReport {
	avg, min, max := m.GetConfidenceStats()

	return MonitoringReport{
		ModelID:           m.modelID,
		Timestamp:         time.Now(),
		ConfidenceAvg:     avg,
		ConfidenceMin:     min,
		ConfidenceMax:     max,
		DriftScore:        m.GetDriftScore(),
		DriftDetected:     m.DetectDrift(),
		ClassDistribution: m.GetDistribution(),
	}
}

func main() {
	monitor := NewModelMonitor("image-classifier-v1")

	// Set baseline distribution (from training data)
	baseline := map[int]float64{
		0: 0.3,  // cat
		1: 0.4,  // dog
		2: 0.3,  // bird
	}
	monitor.SetBaseline(baseline)
	monitor.SetDriftThreshold(0.15)

	// Simulate predictions
	for i := 0; i < 1000; i++ {
		// Simulated prediction with some drift
		var prediction []float32
		var confidence float32

		if i < 500 {
			// Normal distribution
			prediction = []float32{0.3, 0.5, 0.2}
			confidence = 0.8 + float32(i%10)/100
		} else {
			// Drifted distribution (more cats)
			prediction = []float32{0.6, 0.3, 0.1}
			confidence = 0.7 + float32(i%10)/100
		}

		monitor.RecordPrediction(prediction, confidence)
	}

	report := monitor.GenerateReport()
	fmt.Printf("Model: %s\\n", report.ModelID)
	fmt.Printf("Confidence: avg=%.2f, min=%.2f, max=%.2f\\n",
		report.ConfidenceAvg, report.ConfidenceMin, report.ConfidenceMax)
	fmt.Printf("Drift Score: %.4f\\n", report.DriftScore)
	fmt.Printf("Drift Detected: %v\\n", report.DriftDetected)
	fmt.Printf("Distribution: %v\\n", report.ClassDistribution)
}`,

	testCode: `package main

import (
	"testing"
)

func TestDistributionTracker(t *testing.T) {
	dt := NewDistributionTracker(100, 10)

	for i := 0; i < 50; i++ {
		dt.Record(0)
	}
	for i := 0; i < 50; i++ {
		dt.Record(1)
	}

	dist := dt.GetDistribution()

	if dist[0] < 0.49 || dist[0] > 0.51 {
		t.Errorf("Expected class 0 at 0.5, got %f", dist[0])
	}
}

func TestConfidenceTracker(t *testing.T) {
	ct := NewConfidenceTracker()

	ct.Record(0.8)
	ct.Record(0.9)
	ct.Record(1.0)

	avg, _, min, max := ct.Stats()

	if avg < 0.89 || avg > 0.91 {
		t.Errorf("Expected avg ~0.9, got %f", avg)
	}
	if min != 0.8 {
		t.Errorf("Expected min 0.8, got %f", min)
	}
	if max != 1.0 {
		t.Errorf("Expected max 1.0, got %f", max)
	}
}

func TestModelMonitor(t *testing.T) {
	monitor := NewModelMonitor("test-model")

	prediction := []float32{0.7, 0.2, 0.1}
	monitor.RecordPrediction(prediction, 0.85)

	avg, min, max := monitor.GetConfidenceStats()

	if avg != 0.85 {
		t.Errorf("Expected avg 0.85, got %f", avg)
	}
	if min != 0.85 || max != 0.85 {
		t.Error("Min and max should both be 0.85")
	}
}

func TestDriftDetection(t *testing.T) {
	monitor := NewModelMonitor("test-model")

	baseline := map[int]float64{
		0: 0.5,
		1: 0.5,
	}
	monitor.SetBaseline(baseline)
	monitor.SetDriftThreshold(0.1)

	// Record predictions matching baseline
	for i := 0; i < 100; i++ {
		if i%2 == 0 {
			monitor.RecordPrediction([]float32{0.9, 0.1}, 0.9)
		} else {
			monitor.RecordPrediction([]float32{0.1, 0.9}, 0.9)
		}
	}

	if monitor.DetectDrift() {
		t.Error("Should not detect drift when matching baseline")
	}
}

func TestDriftDetectionWithDrift(t *testing.T) {
	monitor := NewModelMonitor("test-model")

	baseline := map[int]float64{
		0: 0.5,
		1: 0.5,
	}
	monitor.SetBaseline(baseline)
	monitor.SetDriftThreshold(0.1)

	// Record predictions with significant drift
	for i := 0; i < 100; i++ {
		monitor.RecordPrediction([]float32{0.9, 0.1}, 0.9) // All class 0
	}

	if !monitor.DetectDrift() {
		t.Error("Should detect drift when distribution differs")
	}
}

func TestKLDivergence(t *testing.T) {
	p := map[int]float64{0: 0.5, 1: 0.5}
	q := map[int]float64{0: 0.5, 1: 0.5}

	kl := KLDivergence(p, q)
	if kl > 0.001 {
		t.Errorf("Same distributions should have KL ~0, got %f", kl)
	}
}

func TestGetRecentSamples(t *testing.T) {
	monitor := NewModelMonitor("test-model")

	for i := 0; i < 10; i++ {
		monitor.RecordPrediction([]float32{float32(i) / 10, 1 - float32(i)/10}, 0.9)
	}

	samples := monitor.GetRecentSamples(5)
	if len(samples) != 5 {
		t.Errorf("Expected 5 samples, got %d", len(samples))
	}
}

func TestGetDistribution(t *testing.T) {
	monitor := NewModelMonitor("test-model")

	for i := 0; i < 10; i++ {
		monitor.RecordPrediction([]float32{0.9, 0.1}, 0.9) // All class 0
	}

	dist := monitor.GetDistribution()
	if dist[0] != 1.0 {
		t.Errorf("Expected all class 0, got %v", dist)
	}
}

func TestGenerateReport(t *testing.T) {
	monitor := NewModelMonitor("report-test")

	for i := 0; i < 10; i++ {
		monitor.RecordPrediction([]float32{0.8, 0.2}, 0.85)
	}

	report := monitor.GenerateReport()
	if report.ModelID != "report-test" {
		t.Errorf("Expected ModelID 'report-test', got %s", report.ModelID)
	}
	if report.ConfidenceAvg != 0.85 {
		t.Errorf("Expected ConfidenceAvg 0.85, got %f", report.ConfidenceAvg)
	}
}

func TestGetDriftScore(t *testing.T) {
	monitor := NewModelMonitor("drift-test")

	baseline := map[int]float64{0: 0.5, 1: 0.5}
	monitor.SetBaseline(baseline)

	for i := 0; i < 100; i++ {
		monitor.RecordPrediction([]float32{0.9, 0.1}, 0.9)
	}

	score := monitor.GetDriftScore()
	if score <= 0 {
		t.Error("Drift score should be positive when distribution differs")
	}
}`,

	hint1: 'Use KL divergence to detect distribution drift',
	hint2: 'Track confidence distributions, not just averages',

	whyItMatters: `Model quality monitoring prevents silent failures:

- **Drift detection**: Catch when model behavior changes
- **Confidence tracking**: Identify uncertain predictions
- **Data quality**: Detect input distribution changes
- **Continuous validation**: Monitor model performance over time

Proactive monitoring is essential for reliable ML systems.`,

	translations: {
		ru: {
			title: 'Метрики качества модели',
			description: `# Метрики качества модели

Отслеживайте метрики качества предсказаний модели в продакшене.

## Задача

Реализуйте метрики:
- Отслеживание распределения уверенности предсказаний
- Мониторинг дрейфа предсказаний
- Логирование образцов предсказаний для анализа
- Оповещение о деградации качества

## Пример

\`\`\`go
monitor := NewModelMonitor("model-v1")
monitor.RecordPrediction(prediction, confidence)
if monitor.DetectDrift() {
    alert.Send("Model drift detected")
}
\`\`\``,
			hint1: 'Используйте KL дивергенцию для обнаружения дрейфа распределения',
			hint2: 'Отслеживайте распределения уверенности, а не только средние',
			whyItMatters: `Мониторинг качества модели предотвращает тихие сбои:

- **Обнаружение дрейфа**: Отлов изменений поведения модели
- **Отслеживание уверенности**: Выявление неопределенных предсказаний
- **Качество данных**: Обнаружение изменений входного распределения
- **Непрерывная валидация**: Мониторинг производительности модели во времени`,
		},
		uz: {
			title: 'Model sifat metrikalari',
			description: `# Model sifat metrikalari

Production da model bashorat sifati metrikalarini kuzatib boring.

## Topshiriq

Metrikalarni amalga oshiring:
- Bashorat ishonch taqsimotini kuzatish
- Bashorat driftini monitoring qilish
- Tahlil uchun bashorat namunalarini loglash
- Sifat yomonlashganda ogohlantirish

## Misol

\`\`\`go
monitor := NewModelMonitor("model-v1")
monitor.RecordPrediction(prediction, confidence)
if monitor.DetectDrift() {
    alert.Send("Model drift detected")
}
\`\`\``,
			hint1: "Taqsimot driftini aniqlash uchun KL divergence dan foydalaning",
			hint2: "Faqat o'rtachalarni emas, ishonch taqsimotlarini kuzatib boring",
			whyItMatters: `Model sifatini monitoring qilish jimjit nosozliklarni oldini oladi:

- **Drift aniqlash**: Model xulqi o'zgarganda ushlash
- **Ishonchni kuzatish**: Noaniq bashoratlarni aniqlash
- **Ma'lumotlar sifati**: Kirish taqsimoti o'zgarishlarini aniqlash
- **Doimiy tekshirish**: Vaqt o'tishi bilan model ishlashini monitoring qilish`,
		},
	},
};

export default task;
