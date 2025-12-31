import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-model-ensemble',
	title: 'Model Ensemble',
	difficulty: 'hard',
	tags: ['go', 'ml', 'ensemble', 'inference'],
	estimatedTime: '35m',
	isPremium: true,
	order: 1,
	description: `# Model Ensemble

Implement ensemble inference combining multiple models.

## Task

Build an ensemble system that:
- Runs multiple models in parallel
- Combines predictions using voting/averaging
- Supports weighted ensembles
- Handles model failures gracefully

## Example

\`\`\`go
ensemble := NewEnsemble(models, weights)
prediction := ensemble.Predict(input)
// Combines predictions from all models
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

// Model interface for ensemble members
type Model interface {
	Predict(input []float32) []float32
	Name() string
}

// Ensemble combines multiple models
type Ensemble struct {
	// Your fields here
}

// NewEnsemble creates an ensemble
func NewEnsemble(models []Model, weights []float64) *Ensemble {
	// Your code here
	return nil
}

// Predict runs ensemble inference
func (e *Ensemble) Predict(input []float32) []float32 {
	// Your code here
	return nil
}

func main() {
	fmt.Println("Model Ensemble")
}`,

	solutionCode: `package main

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// Model interface for ensemble members
type Model interface {
	Predict(input []float32) []float32
	Name() string
}

// SimpleModel implements Model for testing
type SimpleModel struct {
	name   string
	bias   float32
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

// EnsembleConfig holds ensemble configuration
type EnsembleConfig struct {
	Strategy       string  // "average", "weighted", "voting"
	Timeout        time.Duration
	MinModels      int     // Minimum models required for valid prediction
	ConfidenceThreshold float64
}

// DefaultEnsembleConfig returns default configuration
func DefaultEnsembleConfig() EnsembleConfig {
	return EnsembleConfig{
		Strategy:  "weighted",
		Timeout:   5 * time.Second,
		MinModels: 1,
		ConfidenceThreshold: 0.0,
	}
}

// ModelResult holds a single model's prediction
type ModelResult struct {
	ModelName  string
	Prediction []float32
	Error      error
	Latency    time.Duration
}

// Ensemble combines multiple models
type Ensemble struct {
	models  []Model
	weights []float64
	config  EnsembleConfig
}

// NewEnsemble creates an ensemble
func NewEnsemble(models []Model, weights []float64) *Ensemble {
	// Normalize weights if provided
	if len(weights) == 0 {
		weights = make([]float64, len(models))
		for i := range weights {
			weights[i] = 1.0 / float64(len(models))
		}
	} else {
		var sum float64
		for _, w := range weights {
			sum += w
		}
		for i := range weights {
			weights[i] /= sum
		}
	}

	return &Ensemble{
		models:  models,
		weights: weights,
		config:  DefaultEnsembleConfig(),
	}
}

// SetConfig sets ensemble configuration
func (e *Ensemble) SetConfig(config EnsembleConfig) {
	e.config = config
}

// Predict runs ensemble inference
func (e *Ensemble) Predict(input []float32) []float32 {
	ctx, cancel := context.WithTimeout(context.Background(), e.config.Timeout)
	defer cancel()

	return e.PredictWithContext(ctx, input)
}

// PredictWithContext runs ensemble with context
func (e *Ensemble) PredictWithContext(ctx context.Context, input []float32) []float32 {
	results := e.runModels(ctx, input)

	switch e.config.Strategy {
	case "voting":
		return e.votingCombine(results)
	case "average":
		return e.averageCombine(results)
	default:
		return e.weightedCombine(results)
	}
}

// runModels runs all models in parallel
func (e *Ensemble) runModels(ctx context.Context, input []float32) []ModelResult {
	results := make([]ModelResult, len(e.models))
	var wg sync.WaitGroup

	for i, model := range e.models {
		wg.Add(1)
		go func(idx int, m Model) {
			defer wg.Done()

			start := time.Now()
			result := ModelResult{
				ModelName: m.Name(),
			}

			done := make(chan struct{})
			go func() {
				defer close(done)
				result.Prediction = m.Predict(input)
			}()

			select {
			case <-done:
				result.Latency = time.Since(start)
			case <-ctx.Done():
				result.Error = ctx.Err()
			}

			results[idx] = result
		}(i, model)
	}

	wg.Wait()
	return results
}

// weightedCombine combines predictions with weights
func (e *Ensemble) weightedCombine(results []ModelResult) []float32 {
	var validResults []ModelResult
	var validWeights []float64

	for i, r := range results {
		if r.Error == nil && r.Prediction != nil {
			validResults = append(validResults, r)
			validWeights = append(validWeights, e.weights[i])
		}
	}

	if len(validResults) < e.config.MinModels {
		return nil
	}

	// Renormalize weights
	var weightSum float64
	for _, w := range validWeights {
		weightSum += w
	}

	outputLen := len(validResults[0].Prediction)
	output := make([]float32, outputLen)

	for i, r := range validResults {
		weight := validWeights[i] / weightSum
		for j, v := range r.Prediction {
			output[j] += v * float32(weight)
		}
	}

	return output
}

// averageCombine takes simple average
func (e *Ensemble) averageCombine(results []ModelResult) []float32 {
	var validResults []ModelResult

	for _, r := range results {
		if r.Error == nil && r.Prediction != nil {
			validResults = append(validResults, r)
		}
	}

	if len(validResults) < e.config.MinModels {
		return nil
	}

	outputLen := len(validResults[0].Prediction)
	output := make([]float32, outputLen)

	for _, r := range validResults {
		for j, v := range r.Prediction {
			output[j] += v
		}
	}

	n := float32(len(validResults))
	for i := range output {
		output[i] /= n
	}

	return output
}

// votingCombine uses majority voting
func (e *Ensemble) votingCombine(results []ModelResult) []float32 {
	votes := make(map[int]float64)

	for i, r := range results {
		if r.Error == nil && r.Prediction != nil {
			// Find argmax
			maxIdx := 0
			maxVal := r.Prediction[0]
			for j, v := range r.Prediction {
				if v > maxVal {
					maxVal = v
					maxIdx = j
				}
			}
			votes[maxIdx] += e.weights[i]
		}
	}

	if len(votes) == 0 {
		return nil
	}

	// Find winning class
	winningClass := 0
	maxVotes := 0.0
	for class, v := range votes {
		if v > maxVotes {
			maxVotes = v
			winningClass = class
		}
	}

	// Return one-hot prediction
	outputLen := len(results[0].Prediction)
	output := make([]float32, outputLen)
	output[winningClass] = 1.0

	return output
}

// GetModelStats returns individual model statistics
func (e *Ensemble) GetModelStats(results []ModelResult) map[string]time.Duration {
	stats := make(map[string]time.Duration)
	for _, r := range results {
		stats[r.ModelName] = r.Latency
	}
	return stats
}

// StackingEnsemble uses a meta-model to combine predictions
type StackingEnsemble struct {
	baseModels []Model
	metaModel  Model
}

func NewStackingEnsemble(baseModels []Model, metaModel Model) *StackingEnsemble {
	return &StackingEnsemble{
		baseModels: baseModels,
		metaModel:  metaModel,
	}
}

func (s *StackingEnsemble) Predict(input []float32) []float32 {
	// Get base predictions
	var metaInput []float32
	for _, model := range s.baseModels {
		pred := model.Predict(input)
		metaInput = append(metaInput, pred...)
	}

	// Meta-model makes final prediction
	return s.metaModel.Predict(metaInput)
}

// CascadeEnsemble runs models in sequence
type CascadeEnsemble struct {
	models    []Model
	threshold float64
}

func NewCascadeEnsemble(models []Model, threshold float64) *CascadeEnsemble {
	return &CascadeEnsemble{
		models:    models,
		threshold: threshold,
	}
}

func (c *CascadeEnsemble) Predict(input []float32) []float32 {
	for _, model := range c.models {
		pred := model.Predict(input)

		// Find max confidence
		maxConf := float64(0)
		for _, v := range pred {
			if float64(v) > maxConf {
				maxConf = float64(v)
			}
		}

		if maxConf >= c.threshold {
			return pred
		}
	}

	// Return last model's prediction if no threshold met
	return c.models[len(c.models)-1].Predict(input)
}

// CalibratePredictions calibrates predictions using temperature scaling
func CalibratePredictions(predictions []float32, temperature float64) []float32 {
	output := make([]float32, len(predictions))
	var sum float64

	for i, p := range predictions {
		scaled := math.Exp(float64(p) / temperature)
		output[i] = float32(scaled)
		sum += scaled
	}

	for i := range output {
		output[i] /= float32(sum)
	}

	return output
}

func main() {
	// Create ensemble members
	models := []Model{
		NewSimpleModel("model-a", 0.1),
		NewSimpleModel("model-b", -0.05),
		NewSimpleModel("model-c", 0.05),
	}
	weights := []float64{0.5, 0.3, 0.2}

	ensemble := NewEnsemble(models, weights)

	input := []float32{1.0, 2.0, 3.0}

	// Weighted ensemble
	pred := ensemble.Predict(input)
	fmt.Printf("Weighted: %v\\n", pred)

	// Average ensemble
	ensemble.SetConfig(EnsembleConfig{Strategy: "average", MinModels: 1, Timeout: 5 * time.Second})
	pred = ensemble.Predict(input)
	fmt.Printf("Average: %v\\n", pred)

	// Voting ensemble
	ensemble.SetConfig(EnsembleConfig{Strategy: "voting", MinModels: 1, Timeout: 5 * time.Second})
	pred = ensemble.Predict(input)
	fmt.Printf("Voting: %v\\n", pred)

	// Cascade ensemble
	cascade := NewCascadeEnsemble(models, 0.7)
	pred = cascade.Predict(input)
	fmt.Printf("Cascade: %v\\n", pred)
}`,

	testCode: `package main

import (
	"context"
	"testing"
	"time"
)

func TestEnsemble(t *testing.T) {
	models := []Model{
		NewSimpleModel("a", 0.1),
		NewSimpleModel("b", 0.0),
	}

	ensemble := NewEnsemble(models, nil)
	pred := ensemble.Predict([]float32{1, 2, 3})

	if pred == nil {
		t.Fatal("Prediction should not be nil")
	}
	if len(pred) != 2 {
		t.Errorf("Expected 2 outputs, got %d", len(pred))
	}
}

func TestWeightedEnsemble(t *testing.T) {
	models := []Model{
		NewSimpleModel("a", 0.2),
		NewSimpleModel("b", 0.0),
	}
	weights := []float64{0.8, 0.2}

	ensemble := NewEnsemble(models, weights)
	pred := ensemble.Predict([]float32{1, 2, 3})

	if pred == nil {
		t.Fatal("Prediction should not be nil")
	}
}

func TestVotingEnsemble(t *testing.T) {
	models := []Model{
		NewSimpleModel("a", 0.3),
		NewSimpleModel("b", 0.3),
		NewSimpleModel("c", -0.3),
	}

	ensemble := NewEnsemble(models, nil)
	ensemble.SetConfig(EnsembleConfig{
		Strategy:  "voting",
		MinModels: 1,
		Timeout:   5 * time.Second,
	})

	pred := ensemble.Predict([]float32{1, 2, 3})

	if pred == nil {
		t.Fatal("Prediction should not be nil")
	}

	// Should be one-hot
	var sum float32
	for _, v := range pred {
		sum += v
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("Voting output should sum to 1, got %f", sum)
	}
}

func TestEnsembleTimeout(t *testing.T) {
	models := []Model{
		NewSimpleModel("a", 0.1),
	}

	ensemble := NewEnsemble(models, nil)

	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()

	// Should still work with short timeout for fast models
	pred := ensemble.PredictWithContext(ctx, []float32{1, 2, 3})
	if pred == nil {
		t.Log("Prediction timed out (acceptable)")
	}
}

func TestCascadeEnsemble(t *testing.T) {
	models := []Model{
		NewSimpleModel("fast", 0.0),
		NewSimpleModel("accurate", 0.3),
	}

	cascade := NewCascadeEnsemble(models, 0.9)
	pred := cascade.Predict([]float32{1, 2, 3})

	if pred == nil {
		t.Fatal("Prediction should not be nil")
	}
}

func TestCalibratePredictions(t *testing.T) {
	predictions := []float32{0.9, 0.1}
	calibrated := CalibratePredictions(predictions, 2.0)

	// Higher temperature should make distribution more uniform
	diff := calibrated[0] - calibrated[1]
	originalDiff := predictions[0] - predictions[1]

	if diff >= originalDiff {
		t.Error("Temperature scaling should reduce confidence gap")
	}
}

func TestStackingEnsemble(t *testing.T) {
	baseModels := []Model{
		NewSimpleModel("a", 0.1),
		NewSimpleModel("b", 0.0),
	}
	metaModel := NewSimpleModel("meta", 0.0)

	stacking := NewStackingEnsemble(baseModels, metaModel)
	pred := stacking.Predict([]float32{1, 2, 3})

	if pred == nil {
		t.Fatal("Prediction should not be nil")
	}
}

func TestAverageEnsemble(t *testing.T) {
	models := []Model{
		NewSimpleModel("a", 0.2),
		NewSimpleModel("b", 0.0),
	}

	ensemble := NewEnsemble(models, nil)
	ensemble.SetConfig(EnsembleConfig{
		Strategy:  "average",
		MinModels: 1,
		Timeout:   5 * time.Second,
	})

	pred := ensemble.Predict([]float32{1, 2, 3})
	if pred == nil {
		t.Fatal("Prediction should not be nil")
	}
	if len(pred) != 2 {
		t.Errorf("Expected 2 outputs, got %d", len(pred))
	}
}

func TestSetConfig(t *testing.T) {
	models := []Model{
		NewSimpleModel("a", 0.1),
	}
	ensemble := NewEnsemble(models, nil)

	config := EnsembleConfig{
		Strategy:  "voting",
		MinModels: 2,
		Timeout:   10 * time.Second,
	}
	ensemble.SetConfig(config)

	// Verify config was set by running prediction
	pred := ensemble.Predict([]float32{1, 2, 3})
	// MinModels=2 but only 1 model, should return nil
	if pred != nil {
		t.Error("Should return nil when MinModels not met")
	}
}

func TestGetModelStats(t *testing.T) {
	models := []Model{
		NewSimpleModel("model-a", 0.1),
		NewSimpleModel("model-b", 0.0),
	}
	ensemble := NewEnsemble(models, nil)

	results := []ModelResult{
		{ModelName: "model-a", Latency: 100 * time.Millisecond},
		{ModelName: "model-b", Latency: 50 * time.Millisecond},
	}

	stats := ensemble.GetModelStats(results)
	if len(stats) != 2 {
		t.Fatalf("Expected 2 stats, got %d", len(stats))
	}
	if stats["model-a"] != 100*time.Millisecond {
		t.Error("Wrong latency for model-a")
	}
}`,

	hint1: 'Run models in parallel using goroutines for lower latency',
	hint2: 'Normalize weights after removing failed models',

	whyItMatters: `Model ensembles improve prediction quality:

- **Accuracy**: Combine diverse models for better predictions
- **Robustness**: Handle individual model failures
- **Calibration**: Better uncertainty estimates
- **Flexibility**: Different combination strategies for different needs

Ensembles are widely used in production ML systems.`,

	translations: {
		ru: {
			title: 'Ансамбль моделей',
			description: `# Ансамбль моделей

Реализуйте ансамблевый инференс объединяющий несколько моделей.

## Задача

Создайте ансамблевую систему:
- Запуск нескольких моделей параллельно
- Объединение предсказаний голосованием/усреднением
- Поддержка взвешенных ансамблей
- Изящная обработка сбоев моделей

## Пример

\`\`\`go
ensemble := NewEnsemble(models, weights)
prediction := ensemble.Predict(input)
// Combines predictions from all models
\`\`\``,
			hint1: 'Запускайте модели параллельно используя goroutines для низкой латентности',
			hint2: 'Нормализуйте веса после удаления сбойных моделей',
			whyItMatters: `Ансамбли моделей улучшают качество предсказаний:

- **Точность**: Объединение разнообразных моделей для лучших предсказаний
- **Надежность**: Обработка сбоев отдельных моделей
- **Калибровка**: Лучшие оценки неопределенности
- **Гибкость**: Разные стратегии комбинирования для разных задач`,
		},
		uz: {
			title: 'Model ensemble',
			description: `# Model ensemble

Bir nechta modellarni birlashtiradigan ensemble inference ni amalga oshiring.

## Topshiriq

Ensemble tizimini yarating:
- Bir nechta modellarni parallel ishga tushirish
- Ovoz berish/o'rtacha olish orqali bashoratlarni birlashtirish
- Vaznli ensemblarni qo'llab-quvvatlash
- Model nosozliklarini yumshoq qayta ishlash

## Misol

\`\`\`go
ensemble := NewEnsemble(models, weights)
prediction := ensemble.Predict(input)
// Combines predictions from all models
\`\`\``,
			hint1: "Past latency uchun goroutinlar yordamida modellarni parallel ishga tushiring",
			hint2: "Muvaffaqiyatsiz modellarni olib tashlagandan keyin vaznlarni normallang",
			whyItMatters: `Model ensemblelari bashorat sifatini yaxshilaydi:

- **Aniqlik**: Yaxshiroq bashoratlar uchun turli modellarni birlashtirish
- **Mustahkamlik**: Individual model nosozliklarini qayta ishlash
- **Kalibrlash**: Yaxshiroq noaniqlik baholashlari
- **Moslashuvchanlik**: Turli ehtiyojlar uchun turli birlashtirish strategiyalari`,
		},
	},
};

export default task;
