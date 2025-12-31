import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-ab-testing',
	title: 'A/B Testing Models',
	difficulty: 'medium',
	tags: ['go', 'ml', 'ab-testing', 'experimentation'],
	estimatedTime: '30m',
	isPremium: true,
	order: 2,
	description: `# A/B Testing Models

Implement A/B testing for ML model comparison.

## Task

Build an A/B testing system that:
- Routes traffic to different model versions
- Tracks metrics per variant
- Supports gradual rollouts
- Enables quick rollback

## Example

\`\`\`go
router := NewABRouter(controlModel, treatmentModel, 0.1) // 10% to treatment
result := router.Route(userID, input)
router.RecordOutcome(userID, success)
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

// ABRouter routes requests between model variants
type ABRouter struct {
	// Your fields here
}

// NewABRouter creates an A/B router
func NewABRouter(control, treatment Model, treatmentRatio float64) *ABRouter {
	// Your code here
	return nil
}

// Route routes a request to appropriate variant
func (r *ABRouter) Route(userID string, input []float32) ([]float32, string) {
	// Your code here
	return nil, ""
}

// RecordOutcome records the outcome for analysis
func (r *ABRouter) RecordOutcome(userID string, success bool) {
	// Your code here
}

// Model interface
type Model interface {
	Predict(input []float32) []float32
	Name() string
}

func main() {
	fmt.Println("A/B Testing Models")
}`,

	solutionCode: `package main

import (
	"fmt"
	"hash/fnv"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// Model interface for A/B testing
type Model interface {
	Predict(input []float32) []float32
	Name() string
}

// SimpleModel implements Model
type SimpleModel struct {
	name string
	version string
}

func NewSimpleModel(name, version string) *SimpleModel {
	return &SimpleModel{name: name, version: version}
}

func (m *SimpleModel) Predict(input []float32) []float32 {
	return []float32{0.8, 0.2}
}

func (m *SimpleModel) Name() string {
	return m.name
}

// VariantMetrics tracks metrics for a variant
type VariantMetrics struct {
	requests   int64
	successes  int64
	latencySum int64
	errors     int64
}

// ABConfig holds A/B test configuration
type ABConfig struct {
	TreatmentRatio float64
	StickyRouting  bool
	MinSampleSize  int
	ConfidenceLevel float64
}

// DefaultABConfig returns default configuration
func DefaultABConfig() ABConfig {
	return ABConfig{
		TreatmentRatio:  0.1,
		StickyRouting:   true,
		MinSampleSize:   100,
		ConfidenceLevel: 0.95,
	}
}

// ABRouter routes requests between model variants
type ABRouter struct {
	control       Model
	treatment     Model
	config        ABConfig

	controlMetrics   VariantMetrics
	treatmentMetrics VariantMetrics

	userAssignments map[string]string
	mu              sync.RWMutex

	active bool
}

// NewABRouter creates an A/B router
func NewABRouter(control, treatment Model, treatmentRatio float64) *ABRouter {
	config := DefaultABConfig()
	config.TreatmentRatio = treatmentRatio

	return &ABRouter{
		control:         control,
		treatment:       treatment,
		config:          config,
		userAssignments: make(map[string]string),
		active:          true,
	}
}

// SetConfig updates configuration
func (r *ABRouter) SetConfig(config ABConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.config = config
}

// Route routes a request to appropriate variant
func (r *ABRouter) Route(userID string, input []float32) ([]float32, string) {
	if !r.active {
		return r.control.Predict(input), r.control.Name()
	}

	variant := r.getVariant(userID)

	var prediction []float32
	var modelName string

	start := time.Now()

	if variant == "treatment" {
		prediction = r.treatment.Predict(input)
		modelName = r.treatment.Name()
		atomic.AddInt64(&r.treatmentMetrics.requests, 1)
		atomic.AddInt64(&r.treatmentMetrics.latencySum, time.Since(start).Milliseconds())
	} else {
		prediction = r.control.Predict(input)
		modelName = r.control.Name()
		atomic.AddInt64(&r.controlMetrics.requests, 1)
		atomic.AddInt64(&r.controlMetrics.latencySum, time.Since(start).Milliseconds())
	}

	return prediction, modelName
}

// getVariant determines which variant for a user
func (r *ABRouter) getVariant(userID string) string {
	if r.config.StickyRouting {
		r.mu.RLock()
		variant, exists := r.userAssignments[userID]
		r.mu.RUnlock()

		if exists {
			return variant
		}
	}

	// Deterministic assignment based on user ID
	var variant string
	if r.hashToRatio(userID) < r.config.TreatmentRatio {
		variant = "treatment"
	} else {
		variant = "control"
	}

	if r.config.StickyRouting {
		r.mu.Lock()
		r.userAssignments[userID] = variant
		r.mu.Unlock()
	}

	return variant
}

// hashToRatio converts user ID to a ratio [0, 1)
func (r *ABRouter) hashToRatio(userID string) float64 {
	h := fnv.New64a()
	h.Write([]byte(userID))
	hash := h.Sum64()
	return float64(hash) / float64(^uint64(0))
}

// RecordOutcome records the outcome for analysis
func (r *ABRouter) RecordOutcome(userID string, success bool) {
	variant := r.getVariant(userID)

	if variant == "treatment" {
		if success {
			atomic.AddInt64(&r.treatmentMetrics.successes, 1)
		}
	} else {
		if success {
			atomic.AddInt64(&r.controlMetrics.successes, 1)
		}
	}
}

// RecordError records an error
func (r *ABRouter) RecordError(userID string) {
	variant := r.getVariant(userID)

	if variant == "treatment" {
		atomic.AddInt64(&r.treatmentMetrics.errors, 1)
	} else {
		atomic.AddInt64(&r.controlMetrics.errors, 1)
	}
}

// ABStats holds A/B test statistics
type ABStats struct {
	ControlRequests    int64
	ControlSuccessRate float64
	ControlAvgLatency  float64
	TreatmentRequests    int64
	TreatmentSuccessRate float64
	TreatmentAvgLatency  float64
	Uplift             float64
	Significant        bool
}

// GetStats returns current A/B test statistics
func (r *ABRouter) GetStats() ABStats {
	stats := ABStats{
		ControlRequests:   atomic.LoadInt64(&r.controlMetrics.requests),
		TreatmentRequests: atomic.LoadInt64(&r.treatmentMetrics.requests),
	}

	controlSucc := atomic.LoadInt64(&r.controlMetrics.successes)
	treatSucc := atomic.LoadInt64(&r.treatmentMetrics.successes)
	controlLatency := atomic.LoadInt64(&r.controlMetrics.latencySum)
	treatLatency := atomic.LoadInt64(&r.treatmentMetrics.latencySum)

	if stats.ControlRequests > 0 {
		stats.ControlSuccessRate = float64(controlSucc) / float64(stats.ControlRequests)
		stats.ControlAvgLatency = float64(controlLatency) / float64(stats.ControlRequests)
	}

	if stats.TreatmentRequests > 0 {
		stats.TreatmentSuccessRate = float64(treatSucc) / float64(stats.TreatmentRequests)
		stats.TreatmentAvgLatency = float64(treatLatency) / float64(stats.TreatmentRequests)
	}

	if stats.ControlSuccessRate > 0 {
		stats.Uplift = (stats.TreatmentSuccessRate - stats.ControlSuccessRate) / stats.ControlSuccessRate * 100
	}

	// Simple significance check
	minSamples := int64(r.config.MinSampleSize)
	stats.Significant = stats.ControlRequests >= minSamples && stats.TreatmentRequests >= minSamples

	return stats
}

// Stop stops the A/B test
func (r *ABRouter) Stop() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.active = false
}

// SetTreatmentRatio updates traffic ratio
func (r *ABRouter) SetTreatmentRatio(ratio float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.config.TreatmentRatio = ratio
}

// Rollback switches all traffic to control
func (r *ABRouter) Rollback() {
	r.SetTreatmentRatio(0)
}

// Graduate switches all traffic to treatment
func (r *ABRouter) Graduate() {
	r.SetTreatmentRatio(1.0)
}

// GradualRollout implements gradual rollout
type GradualRollout struct {
	*ABRouter
	stages     []float64
	currentStage int
	checkInterval time.Duration
	minRequests  int64
}

func NewGradualRollout(control, treatment Model, stages []float64) *GradualRollout {
	return &GradualRollout{
		ABRouter:      NewABRouter(control, treatment, stages[0]),
		stages:        stages,
		currentStage:  0,
		checkInterval: 5 * time.Minute,
		minRequests:   100,
	}
}

// Advance advances to next rollout stage
func (g *GradualRollout) Advance() bool {
	if g.currentStage >= len(g.stages)-1 {
		return false
	}

	g.currentStage++
	g.SetTreatmentRatio(g.stages[g.currentStage])
	return true
}

// GetCurrentStage returns current rollout percentage
func (g *GradualRollout) GetCurrentStage() float64 {
	return g.stages[g.currentStage]
}

func main() {
	control := NewSimpleModel("model-v1", "1.0")
	treatment := NewSimpleModel("model-v2", "2.0")

	router := NewABRouter(control, treatment, 0.2)

	// Simulate traffic
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < 1000; i++ {
		userID := fmt.Sprintf("user-%d", rand.Intn(100))
		_, variant := router.Route(userID, []float32{1, 2, 3})

		// Simulate outcome (treatment slightly better)
		success := rand.Float64() < 0.8
		if variant == treatment.Name() {
			success = rand.Float64() < 0.85
		}
		router.RecordOutcome(userID, success)
	}

	stats := router.GetStats()
	fmt.Printf("Control: %d requests, %.2f%% success rate\\n",
		stats.ControlRequests, stats.ControlSuccessRate*100)
	fmt.Printf("Treatment: %d requests, %.2f%% success rate\\n",
		stats.TreatmentRequests, stats.TreatmentSuccessRate*100)
	fmt.Printf("Uplift: %.2f%%\\n", stats.Uplift)
	fmt.Printf("Significant: %v\\n", stats.Significant)
}`,

	testCode: `package main

import (
	"fmt"
	"testing"
)

func TestABRouter(t *testing.T) {
	control := NewSimpleModel("control", "1.0")
	treatment := NewSimpleModel("treatment", "2.0")

	router := NewABRouter(control, treatment, 0.5)

	pred, variant := router.Route("user-1", []float32{1, 2, 3})

	if pred == nil {
		t.Error("Prediction should not be nil")
	}
	if variant == "" {
		t.Error("Variant should be set")
	}
}

func TestStickyRouting(t *testing.T) {
	control := NewSimpleModel("control", "1.0")
	treatment := NewSimpleModel("treatment", "2.0")

	router := NewABRouter(control, treatment, 0.5)

	_, variant1 := router.Route("user-1", []float32{1})
	_, variant2 := router.Route("user-1", []float32{2})

	if variant1 != variant2 {
		t.Error("Same user should get same variant (sticky routing)")
	}
}

func TestTrafficSplit(t *testing.T) {
	control := NewSimpleModel("control", "1.0")
	treatment := NewSimpleModel("treatment", "2.0")

	router := NewABRouter(control, treatment, 0.3)

	// Route many users
	treatmentCount := 0
	for i := 0; i < 1000; i++ {
		userID := fmt.Sprintf("user-%d", i)
		_, variant := router.Route(userID, []float32{1})
		if variant == treatment.Name() {
			treatmentCount++
		}
	}

	// Should be roughly 30% treatment
	ratio := float64(treatmentCount) / 1000.0
	if ratio < 0.2 || ratio > 0.4 {
		t.Errorf("Expected ~30%% treatment, got %.1f%%", ratio*100)
	}
}

func TestRecordOutcome(t *testing.T) {
	control := NewSimpleModel("control", "1.0")
	treatment := NewSimpleModel("treatment", "2.0")

	router := NewABRouter(control, treatment, 0.5)

	router.Route("user-1", []float32{1})
	router.RecordOutcome("user-1", true)

	stats := router.GetStats()
	totalRequests := stats.ControlRequests + stats.TreatmentRequests

	if totalRequests != 1 {
		t.Errorf("Expected 1 request, got %d", totalRequests)
	}
}

func TestRollback(t *testing.T) {
	control := NewSimpleModel("control", "1.0")
	treatment := NewSimpleModel("treatment", "2.0")

	router := NewABRouter(control, treatment, 0.5)
	router.Rollback()

	// All traffic should go to control
	for i := 0; i < 100; i++ {
		_, variant := router.Route(fmt.Sprintf("user-%d", i), []float32{1})
		if variant != control.Name() {
			t.Error("After rollback, all traffic should go to control")
		}
	}
}

func TestGradualRollout(t *testing.T) {
	control := NewSimpleModel("control", "1.0")
	treatment := NewSimpleModel("treatment", "2.0")

	stages := []float64{0.1, 0.25, 0.5, 1.0}
	rollout := NewGradualRollout(control, treatment, stages)

	if rollout.GetCurrentStage() != 0.1 {
		t.Errorf("Expected stage 0.1, got %f", rollout.GetCurrentStage())
	}

	rollout.Advance()
	if rollout.GetCurrentStage() != 0.25 {
		t.Errorf("Expected stage 0.25, got %f", rollout.GetCurrentStage())
	}
}

func TestStats(t *testing.T) {
	control := NewSimpleModel("control", "1.0")
	treatment := NewSimpleModel("treatment", "2.0")

	router := NewABRouter(control, treatment, 0.5)

	for i := 0; i < 100; i++ {
		router.Route(fmt.Sprintf("user-%d", i), []float32{1})
		router.RecordOutcome(fmt.Sprintf("user-%d", i), i%2 == 0)
	}

	stats := router.GetStats()

	if stats.ControlRequests+stats.TreatmentRequests != 100 {
		t.Error("Total requests should be 100")
	}
}

func TestGraduate(t *testing.T) {
	control := NewSimpleModel("control", "1.0")
	treatment := NewSimpleModel("treatment", "2.0")

	router := NewABRouter(control, treatment, 0.5)
	router.Graduate()

	// All traffic should go to treatment
	for i := 0; i < 100; i++ {
		_, variant := router.Route(fmt.Sprintf("new-user-%d", i), []float32{1})
		if variant != treatment.Name() {
			t.Error("After graduate, all traffic should go to treatment")
		}
	}
}

func TestStop(t *testing.T) {
	control := NewSimpleModel("control", "1.0")
	treatment := NewSimpleModel("treatment", "2.0")

	router := NewABRouter(control, treatment, 0.5)
	router.Stop()

	// All traffic should go to control when stopped
	for i := 0; i < 100; i++ {
		_, variant := router.Route(fmt.Sprintf("stop-user-%d", i), []float32{1})
		if variant != control.Name() {
			t.Error("After stop, all traffic should go to control")
		}
	}
}

func TestRecordError(t *testing.T) {
	control := NewSimpleModel("control", "1.0")
	treatment := NewSimpleModel("treatment", "2.0")

	router := NewABRouter(control, treatment, 0.5)
	router.Route("user-1", []float32{1})
	router.RecordError("user-1")

	// Should not panic and error should be recorded
	stats := router.GetStats()
	if stats.ControlRequests+stats.TreatmentRequests != 1 {
		t.Error("Expected 1 request recorded")
	}
}`,

	hint1: 'Use consistent hashing for sticky user assignment',
	hint2: 'Track metrics atomically for concurrent access',

	whyItMatters: `A/B testing enables data-driven model improvements:

- **Safe deployment**: Test new models on small traffic
- **Statistical rigor**: Make decisions based on data
- **Quick rollback**: Revert if issues arise
- **Continuous improvement**: Iterate based on results

A/B testing is essential for production ML.`,

	translations: {
		ru: {
			title: 'A/B тестирование моделей',
			description: `# A/B тестирование моделей

Реализуйте A/B тестирование для сравнения ML моделей.

## Задача

Создайте систему A/B тестирования:
- Маршрутизация трафика между версиями моделей
- Отслеживание метрик по вариантам
- Поддержка постепенных раскаток
- Быстрый откат

## Пример

\`\`\`go
router := NewABRouter(controlModel, treatmentModel, 0.1) // 10% to treatment
result := router.Route(userID, input)
router.RecordOutcome(userID, success)
\`\`\``,
			hint1: 'Используйте консистентное хеширование для постоянного назначения пользователей',
			hint2: 'Отслеживайте метрики атомарно для конкурентного доступа',
			whyItMatters: `A/B тестирование обеспечивает улучшения моделей на основе данных:

- **Безопасный деплой**: Тестирование новых моделей на малом трафике
- **Статистическая строгость**: Принятие решений на основе данных
- **Быстрый откат**: Откат при возникновении проблем
- **Непрерывное улучшение**: Итерации на основе результатов`,
		},
		uz: {
			title: 'A/B testing modellari',
			description: `# A/B testing modellari

ML modellarini solishtirish uchun A/B testing ni amalga oshiring.

## Topshiriq

A/B testing tizimini yarating:
- Trafikni turli model versiyalari orasida yo'naltirish
- Har bir variant uchun metrikalarni kuzatish
- Asta-sekin rolloutlarni qo'llab-quvvatlash
- Tez rollback ni ta'minlash

## Misol

\`\`\`go
router := NewABRouter(controlModel, treatmentModel, 0.1) // 10% to treatment
result := router.Route(userID, input)
router.RecordOutcome(userID, success)
\`\`\``,
			hint1: "Doimiy foydalanuvchi tayinlash uchun izchil hashing dan foydalaning",
			hint2: "Bir vaqtda kirish uchun metrikalarni atomik ravishda kuzatib boring",
			whyItMatters: `A/B testing ma'lumotlarga asoslangan model yaxshilanishlarini ta'minlaydi:

- **Xavfsiz deploy**: Yangi modellarni kichik trafikda test qilish
- **Statistik qat'iylik**: Ma'lumotlarga asoslangan qarorlar qabul qilish
- **Tez rollback**: Muammolar paydo bo'lganda qaytarish
- **Doimiy yaxshilash**: Natijalarga asoslangan iteratsiyalar`,
		},
	},
};

export default task;
