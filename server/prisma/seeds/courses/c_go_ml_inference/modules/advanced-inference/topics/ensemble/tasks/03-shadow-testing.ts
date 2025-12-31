import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-shadow-testing',
	title: 'Shadow Testing',
	difficulty: 'medium',
	tags: ['go', 'ml', 'shadow', 'testing'],
	estimatedTime: '25m',
	isPremium: true,
	order: 3,
	description: `# Shadow Testing

Implement shadow testing for safe model evaluation.

## Task

Build a shadow testing system that:
- Runs shadow model alongside primary
- Compares predictions without affecting users
- Logs disagreements for analysis
- Measures latency impact

## Example

\`\`\`go
shadow := NewShadowRunner(primaryModel, shadowModel)
result := shadow.Run(input)
// Returns primary prediction, logs shadow for comparison
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

// ShadowRunner runs shadow model testing
type ShadowRunner struct {
	// Your fields here
}

// NewShadowRunner creates a shadow runner
func NewShadowRunner(primary, shadow Model) *ShadowRunner {
	// Your code here
	return nil
}

// Run executes both models, returns primary result
func (s *ShadowRunner) Run(input []float32) []float32 {
	// Your code here
	return nil
}

// GetDisagreements returns prediction disagreements
func (s *ShadowRunner) GetDisagreements() []Disagreement {
	// Your code here
	return nil
}

// Model interface
type Model interface {
	Predict(input []float32) []float32
	Name() string
}

// Disagreement records a prediction mismatch
type Disagreement struct {
	Input           []float32
	PrimaryResult   []float32
	ShadowResult    []float32
}

func main() {
	fmt.Println("Shadow Testing")
}`,

	solutionCode: `package main

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// Model interface for shadow testing
type Model interface {
	Predict(input []float32) []float32
	Name() string
}

// SimpleModel implements Model
type SimpleModel struct {
	name string
	bias float32
}

func NewSimpleModel(name string, bias float32) *SimpleModel {
	return &SimpleModel{name: name, bias: bias}
}

func (m *SimpleModel) Predict(input []float32) []float32 {
	output := make([]float32, 2)
	var sum float32
	for _, v := range input {
		sum += v
	}
	output[0] = 0.5 + m.bias + sum/float32(len(input)+1)/10
	output[1] = 1.0 - output[0]
	return output
}

func (m *SimpleModel) Name() string {
	return m.name
}

// Disagreement records a prediction mismatch
type Disagreement struct {
	Timestamp       time.Time
	Input           []float32
	PrimaryResult   []float32
	ShadowResult    []float32
	PrimaryClass    int
	ShadowClass     int
	ConfidenceDiff  float32
}

// ShadowStats tracks shadow testing statistics
type ShadowStats struct {
	TotalRequests     int64
	Disagreements     int64
	PrimaryLatencySum int64
	ShadowLatencySum  int64
	ShadowErrors      int64
}

// ShadowConfig configures shadow testing
type ShadowConfig struct {
	DisagreementThreshold float32
	MaxDisagreements      int
	SampleRate            float64
	Async                 bool
}

// DefaultShadowConfig returns default configuration
func DefaultShadowConfig() ShadowConfig {
	return ShadowConfig{
		DisagreementThreshold: 0.1,
		MaxDisagreements:      1000,
		SampleRate:            1.0,
		Async:                 true,
	}
}

// ShadowRunner runs shadow model testing
type ShadowRunner struct {
	primary       Model
	shadow        Model
	config        ShadowConfig
	stats         ShadowStats
	disagreements []Disagreement
	mu            sync.RWMutex
}

// NewShadowRunner creates a shadow runner
func NewShadowRunner(primary, shadow Model) *ShadowRunner {
	return &ShadowRunner{
		primary:       primary,
		shadow:        shadow,
		config:        DefaultShadowConfig(),
		disagreements: make([]Disagreement, 0),
	}
}

// SetConfig updates configuration
func (s *ShadowRunner) SetConfig(config ShadowConfig) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.config = config
}

// Run executes both models, returns primary result
func (s *ShadowRunner) Run(input []float32) []float32 {
	atomic.AddInt64(&s.stats.TotalRequests, 1)

	// Always run primary synchronously
	primaryStart := time.Now()
	primaryResult := s.primary.Predict(input)
	primaryLatency := time.Since(primaryStart).Milliseconds()
	atomic.AddInt64(&s.stats.PrimaryLatencySum, primaryLatency)

	// Run shadow based on sample rate and config
	if s.shouldRunShadow() {
		if s.config.Async {
			go s.runShadow(input, primaryResult)
		} else {
			s.runShadow(input, primaryResult)
		}
	}

	return primaryResult
}

// shouldRunShadow determines if shadow should run
func (s *ShadowRunner) shouldRunShadow() bool {
	if s.config.SampleRate >= 1.0 {
		return true
	}
	// Simple random sampling
	return float64(time.Now().UnixNano()%1000)/1000.0 < s.config.SampleRate
}

// runShadow executes shadow model and compares
func (s *ShadowRunner) runShadow(input, primaryResult []float32) {
	shadowStart := time.Now()
	shadowResult := s.shadow.Predict(input)
	shadowLatency := time.Since(shadowStart).Milliseconds()
	atomic.AddInt64(&s.stats.ShadowLatencySum, shadowLatency)

	if shadowResult == nil {
		atomic.AddInt64(&s.stats.ShadowErrors, 1)
		return
	}

	// Check for disagreement
	if s.isDisagreement(primaryResult, shadowResult) {
		s.recordDisagreement(input, primaryResult, shadowResult)
	}
}

// isDisagreement checks if predictions differ significantly
func (s *ShadowRunner) isDisagreement(primary, shadow []float32) bool {
	// Check argmax disagreement
	primaryClass := argmax(primary)
	shadowClass := argmax(shadow)

	if primaryClass != shadowClass {
		return true
	}

	// Check confidence difference
	for i := range primary {
		diff := math.Abs(float64(primary[i] - shadow[i]))
		if diff > float64(s.config.DisagreementThreshold) {
			return true
		}
	}

	return false
}

// argmax returns index of maximum value
func argmax(arr []float32) int {
	maxIdx := 0
	maxVal := arr[0]
	for i, v := range arr {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// recordDisagreement records a disagreement
func (s *ShadowRunner) recordDisagreement(input, primary, shadow []float32) {
	atomic.AddInt64(&s.stats.Disagreements, 1)

	disagreement := Disagreement{
		Timestamp:      time.Now(),
		Input:          make([]float32, len(input)),
		PrimaryResult:  make([]float32, len(primary)),
		ShadowResult:   make([]float32, len(shadow)),
		PrimaryClass:   argmax(primary),
		ShadowClass:    argmax(shadow),
	}
	copy(disagreement.Input, input)
	copy(disagreement.PrimaryResult, primary)
	copy(disagreement.ShadowResult, shadow)

	s.mu.Lock()
	if len(s.disagreements) < s.config.MaxDisagreements {
		s.disagreements = append(s.disagreements, disagreement)
	}
	s.mu.Unlock()
}

// GetDisagreements returns recorded disagreements
func (s *ShadowRunner) GetDisagreements() []Disagreement {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]Disagreement, len(s.disagreements))
	copy(result, s.disagreements)
	return result
}

// GetStats returns shadow testing statistics
func (s *ShadowRunner) GetStats() ShadowStats {
	return ShadowStats{
		TotalRequests:     atomic.LoadInt64(&s.stats.TotalRequests),
		Disagreements:     atomic.LoadInt64(&s.stats.Disagreements),
		PrimaryLatencySum: atomic.LoadInt64(&s.stats.PrimaryLatencySum),
		ShadowLatencySum:  atomic.LoadInt64(&s.stats.ShadowLatencySum),
		ShadowErrors:      atomic.LoadInt64(&s.stats.ShadowErrors),
	}
}

// DisagreementRate returns the disagreement rate
func (s *ShadowRunner) DisagreementRate() float64 {
	total := atomic.LoadInt64(&s.stats.TotalRequests)
	if total == 0 {
		return 0
	}
	return float64(atomic.LoadInt64(&s.stats.Disagreements)) / float64(total)
}

// LatencyOverhead returns shadow latency overhead
func (s *ShadowRunner) LatencyOverhead() float64 {
	total := atomic.LoadInt64(&s.stats.TotalRequests)
	if total == 0 {
		return 0
	}
	primary := float64(atomic.LoadInt64(&s.stats.PrimaryLatencySum)) / float64(total)
	shadow := float64(atomic.LoadInt64(&s.stats.ShadowLatencySum)) / float64(total)
	return shadow - primary
}

// ClearDisagreements clears recorded disagreements
func (s *ShadowRunner) ClearDisagreements() {
	s.mu.Lock()
	s.disagreements = s.disagreements[:0]
	s.mu.Unlock()
}

// Report generates a shadow testing report
type ShadowReport struct {
	TotalRequests     int64
	DisagreementRate  float64
	AvgPrimaryLatency float64
	AvgShadowLatency  float64
	ShadowErrors      int64
	TopDisagreements  []Disagreement
}

func (s *ShadowRunner) GenerateReport() ShadowReport {
	stats := s.GetStats()

	report := ShadowReport{
		TotalRequests:    stats.TotalRequests,
		DisagreementRate: s.DisagreementRate(),
		ShadowErrors:     stats.ShadowErrors,
	}

	if stats.TotalRequests > 0 {
		report.AvgPrimaryLatency = float64(stats.PrimaryLatencySum) / float64(stats.TotalRequests)
		report.AvgShadowLatency = float64(stats.ShadowLatencySum) / float64(stats.TotalRequests)
	}

	disagreements := s.GetDisagreements()
	if len(disagreements) > 10 {
		report.TopDisagreements = disagreements[:10]
	} else {
		report.TopDisagreements = disagreements
	}

	return report
}

func main() {
	primary := NewSimpleModel("primary-v1", 0.0)
	shadow := NewSimpleModel("shadow-v2", 0.15) // Different bias to cause disagreements

	runner := NewShadowRunner(primary, shadow)
	runner.SetConfig(ShadowConfig{
		DisagreementThreshold: 0.1,
		MaxDisagreements:      100,
		SampleRate:            1.0,
		Async:                 false, // Sync for demo
	})

	// Run some predictions
	for i := 0; i < 100; i++ {
		input := []float32{float32(i) / 100, float32(i%10) / 10}
		runner.Run(input)
	}

	// Wait for async operations
	time.Sleep(100 * time.Millisecond)

	report := runner.GenerateReport()
	fmt.Printf("Total Requests: %d\\n", report.TotalRequests)
	fmt.Printf("Disagreement Rate: %.2f%%\\n", report.DisagreementRate*100)
	fmt.Printf("Avg Primary Latency: %.2fms\\n", report.AvgPrimaryLatency)
	fmt.Printf("Avg Shadow Latency: %.2fms\\n", report.AvgShadowLatency)
	fmt.Printf("Top Disagreements: %d\\n", len(report.TopDisagreements))
}`,

	testCode: `package main

import (
	"testing"
	"time"
)

func TestShadowRunner(t *testing.T) {
	primary := NewSimpleModel("primary", 0.0)
	shadow := NewSimpleModel("shadow", 0.0)

	runner := NewShadowRunner(primary, shadow)
	result := runner.Run([]float32{1, 2, 3})

	if result == nil {
		t.Error("Result should not be nil")
	}
}

func TestDisagreementDetection(t *testing.T) {
	primary := NewSimpleModel("primary", 0.0)
	shadow := NewSimpleModel("shadow", 0.5) // Large bias to cause disagreement

	runner := NewShadowRunner(primary, shadow)
	runner.SetConfig(ShadowConfig{
		DisagreementThreshold: 0.1,
		MaxDisagreements:      100,
		SampleRate:            1.0,
		Async:                 false,
	})

	runner.Run([]float32{1, 2, 3})

	disagreements := runner.GetDisagreements()
	if len(disagreements) == 0 {
		t.Error("Should detect disagreement with different bias")
	}
}

func TestNoDisagreement(t *testing.T) {
	primary := NewSimpleModel("primary", 0.0)
	shadow := NewSimpleModel("shadow", 0.0) // Same as primary

	runner := NewShadowRunner(primary, shadow)
	runner.SetConfig(ShadowConfig{
		DisagreementThreshold: 0.1,
		MaxDisagreements:      100,
		SampleRate:            1.0,
		Async:                 false,
	})

	runner.Run([]float32{1, 2, 3})

	disagreements := runner.GetDisagreements()
	if len(disagreements) != 0 {
		t.Error("Should not detect disagreement with same model")
	}
}

func TestStats(t *testing.T) {
	primary := NewSimpleModel("primary", 0.0)
	shadow := NewSimpleModel("shadow", 0.0)

	runner := NewShadowRunner(primary, shadow)
	runner.SetConfig(ShadowConfig{
		SampleRate: 1.0,
		Async:      false,
	})

	for i := 0; i < 10; i++ {
		runner.Run([]float32{1, 2, 3})
	}

	stats := runner.GetStats()
	if stats.TotalRequests != 10 {
		t.Errorf("Expected 10 requests, got %d", stats.TotalRequests)
	}
}

func TestAsyncExecution(t *testing.T) {
	primary := NewSimpleModel("primary", 0.0)
	shadow := NewSimpleModel("shadow", 0.3)

	runner := NewShadowRunner(primary, shadow)
	runner.SetConfig(ShadowConfig{
		DisagreementThreshold: 0.1,
		SampleRate:            1.0,
		Async:                 true,
	})

	runner.Run([]float32{1, 2, 3})

	// Give async operation time to complete
	time.Sleep(50 * time.Millisecond)

	disagreements := runner.GetDisagreements()
	if len(disagreements) == 0 {
		t.Error("Should detect disagreement async")
	}
}

func TestSampleRate(t *testing.T) {
	primary := NewSimpleModel("primary", 0.0)
	shadow := NewSimpleModel("shadow", 0.5)

	runner := NewShadowRunner(primary, shadow)
	runner.SetConfig(ShadowConfig{
		DisagreementThreshold: 0.1,
		SampleRate:            0.0, // No sampling
		Async:                 false,
	})

	for i := 0; i < 100; i++ {
		runner.Run([]float32{1, 2, 3})
	}

	stats := runner.GetStats()
	if stats.ShadowLatencySum != 0 {
		t.Error("Shadow should not run with 0% sample rate")
	}
}

func TestArgmax(t *testing.T) {
	arr := []float32{0.1, 0.7, 0.2}
	idx := argmax(arr)
	if idx != 1 {
		t.Errorf("Expected argmax 1, got %d", idx)
	}
}

func TestDisagreementRate(t *testing.T) {
	primary := NewSimpleModel("primary", 0.0)
	shadow := NewSimpleModel("shadow", 0.5)

	runner := NewShadowRunner(primary, shadow)
	runner.SetConfig(ShadowConfig{
		DisagreementThreshold: 0.1,
		SampleRate:            1.0,
		Async:                 false,
	})

	for i := 0; i < 10; i++ {
		runner.Run([]float32{1, 2, 3})
	}

	rate := runner.DisagreementRate()
	if rate < 0 || rate > 1 {
		t.Errorf("Disagreement rate should be between 0 and 1, got %f", rate)
	}
}

func TestClearDisagreements(t *testing.T) {
	primary := NewSimpleModel("primary", 0.0)
	shadow := NewSimpleModel("shadow", 0.5)

	runner := NewShadowRunner(primary, shadow)
	runner.SetConfig(ShadowConfig{
		DisagreementThreshold: 0.1,
		SampleRate:            1.0,
		Async:                 false,
	})

	runner.Run([]float32{1, 2, 3})
	runner.ClearDisagreements()

	disagreements := runner.GetDisagreements()
	if len(disagreements) != 0 {
		t.Error("Disagreements should be empty after clear")
	}
}

func TestGenerateReport(t *testing.T) {
	primary := NewSimpleModel("primary", 0.0)
	shadow := NewSimpleModel("shadow", 0.3)

	runner := NewShadowRunner(primary, shadow)
	runner.SetConfig(ShadowConfig{
		DisagreementThreshold: 0.1,
		SampleRate:            1.0,
		Async:                 false,
	})

	for i := 0; i < 5; i++ {
		runner.Run([]float32{float32(i), 2, 3})
	}

	report := runner.GenerateReport()
	if report.TotalRequests != 5 {
		t.Errorf("Expected 5 requests, got %d", report.TotalRequests)
	}
}`,

	hint1: 'Run shadow model asynchronously to avoid latency impact',
	hint2: 'Compare both class predictions and confidence values',

	whyItMatters: `Shadow testing enables safe model evaluation:

- **Risk-free testing**: Evaluate without affecting users
- **Real traffic**: Test with actual production data
- **Behavior comparison**: Identify prediction differences
- **Regression detection**: Catch issues before deployment

Shadow testing is essential for safe ML deployments.`,

	translations: {
		ru: {
			title: 'Теневое тестирование',
			description: `# Теневое тестирование

Реализуйте теневое тестирование для безопасной оценки моделей.

## Задача

Создайте систему теневого тестирования:
- Запуск теневой модели параллельно с основной
- Сравнение предсказаний без влияния на пользователей
- Логирование расхождений для анализа
- Измерение влияния на латентность

## Пример

\`\`\`go
shadow := NewShadowRunner(primaryModel, shadowModel)
result := shadow.Run(input)
// Returns primary prediction, logs shadow for comparison
\`\`\``,
			hint1: 'Запускайте теневую модель асинхронно чтобы избежать влияния на латентность',
			hint2: 'Сравнивайте как предсказания классов так и значения уверенности',
			whyItMatters: `Теневое тестирование обеспечивает безопасную оценку моделей:

- **Тестирование без риска**: Оценка без влияния на пользователей
- **Реальный трафик**: Тестирование на реальных production данных
- **Сравнение поведения**: Выявление различий в предсказаниях
- **Обнаружение регрессий**: Отлов проблем до деплоя`,
		},
		uz: {
			title: 'Shadow testing',
			description: `# Shadow testing

Xavfsiz model baholash uchun shadow testing ni amalga oshiring.

## Topshiriq

Shadow testing tizimini yarating:
- Asosiy model bilan birga shadow modelni ishga tushirish
- Foydalanuvchilarga ta'sir qilmasdan bashoratlarni solishtirish
- Tahlil uchun kelishmovchiliklarni loglash
- Latency ta'sirini o'lchash

## Misol

\`\`\`go
shadow := NewShadowRunner(primaryModel, shadowModel)
result := shadow.Run(input)
// Returns primary prediction, logs shadow for comparison
\`\`\``,
			hint1: "Latency ta'sirini oldini olish uchun shadow modelni asinxron ishga tushiring",
			hint2: "Ham klass bashoratlari ham ishonch qiymatlarini solishtiring",
			whyItMatters: `Shadow testing xavfsiz model baholashni ta'minlaydi:

- **Xavfsiz testing**: Foydalanuvchilarga ta'sir qilmasdan baholash
- **Haqiqiy trafik**: Haqiqiy production ma'lumotlari bilan test qilish
- **Xulq-atvor taqqoslash**: Bashorat farqlarini aniqlash
- **Regressiyalarni aniqlash**: Deploy dan oldin muammolarni ushlash`,
		},
	},
};

export default task;
