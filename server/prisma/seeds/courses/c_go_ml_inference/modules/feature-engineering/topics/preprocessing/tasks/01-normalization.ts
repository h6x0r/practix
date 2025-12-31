import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-normalization',
	title: 'Feature Normalization',
	difficulty: 'easy',
	tags: ['go', 'ml', 'normalization', 'preprocessing'],
	estimatedTime: '20m',
	isPremium: false,
	order: 1,
	description: `# Feature Normalization

Implement feature normalization techniques for ML inference.

## Task

Build normalizers that:
- Implement min-max scaling
- Implement z-score standardization
- Handle batch normalization
- Support inverse transforms

## Example

\`\`\`go
normalizer := NewMinMaxNormalizer(0, 1)
normalizer.Fit(trainingData)
normalized := normalizer.Transform(data)
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

type MinMaxNormalizer struct {
	// Your fields here
}

type StandardNormalizer struct {
	// Your fields here
}

func NewMinMaxNormalizer(min, max float64) *MinMaxNormalizer {
	return nil
}

func (n *MinMaxNormalizer) Fit(data []float64) {
}

func (n *MinMaxNormalizer) Transform(data []float64) []float64 {
	return nil
}

func (n *MinMaxNormalizer) InverseTransform(data []float64) []float64 {
	return nil
}

func main() {
}`,

	solutionCode: `package main

import (
	"fmt"
	"math"
)

// MinMaxNormalizer scales features to [min, max] range
type MinMaxNormalizer struct {
	featureMin float64
	featureMax float64
	targetMin  float64
	targetMax  float64
	fitted     bool
}

// NewMinMaxNormalizer creates a min-max normalizer
func NewMinMaxNormalizer(min, max float64) *MinMaxNormalizer {
	return &MinMaxNormalizer{
		targetMin: min,
		targetMax: max,
		fitted:    false,
	}
}

// Fit calculates normalization parameters from training data
func (n *MinMaxNormalizer) Fit(data []float64) {
	if len(data) == 0 {
		return
	}

	n.featureMin = data[0]
	n.featureMax = data[0]

	for _, v := range data {
		if v < n.featureMin {
			n.featureMin = v
		}
		if v > n.featureMax {
			n.featureMax = v
		}
	}

	n.fitted = true
}

// Transform normalizes data to target range
func (n *MinMaxNormalizer) Transform(data []float64) []float64 {
	if !n.fitted || n.featureMax == n.featureMin {
		return data
	}

	result := make([]float64, len(data))
	scale := (n.targetMax - n.targetMin) / (n.featureMax - n.featureMin)

	for i, v := range data {
		result[i] = (v-n.featureMin)*scale + n.targetMin
	}

	return result
}

// InverseTransform denormalizes data back to original range
func (n *MinMaxNormalizer) InverseTransform(data []float64) []float64 {
	if !n.fitted || n.targetMax == n.targetMin {
		return data
	}

	result := make([]float64, len(data))
	scale := (n.featureMax - n.featureMin) / (n.targetMax - n.targetMin)

	for i, v := range data {
		result[i] = (v-n.targetMin)*scale + n.featureMin
	}

	return result
}

// FitTransform fits and transforms in one step
func (n *MinMaxNormalizer) FitTransform(data []float64) []float64 {
	n.Fit(data)
	return n.Transform(data)
}

// StandardNormalizer implements z-score normalization
type StandardNormalizer struct {
	mean   float64
	std    float64
	fitted bool
}

// NewStandardNormalizer creates a z-score normalizer
func NewStandardNormalizer() *StandardNormalizer {
	return &StandardNormalizer{fitted: false}
}

// Fit calculates mean and std from training data
func (n *StandardNormalizer) Fit(data []float64) {
	if len(data) == 0 {
		return
	}

	// Calculate mean
	var sum float64
	for _, v := range data {
		sum += v
	}
	n.mean = sum / float64(len(data))

	// Calculate std
	var variance float64
	for _, v := range data {
		diff := v - n.mean
		variance += diff * diff
	}
	n.std = math.Sqrt(variance / float64(len(data)))

	if n.std == 0 {
		n.std = 1 // Prevent division by zero
	}

	n.fitted = true
}

// Transform standardizes data to zero mean, unit variance
func (n *StandardNormalizer) Transform(data []float64) []float64 {
	if !n.fitted {
		return data
	}

	result := make([]float64, len(data))
	for i, v := range data {
		result[i] = (v - n.mean) / n.std
	}

	return result
}

// InverseTransform converts back to original scale
func (n *StandardNormalizer) InverseTransform(data []float64) []float64 {
	if !n.fitted {
		return data
	}

	result := make([]float64, len(data))
	for i, v := range data {
		result[i] = v*n.std + n.mean
	}

	return result
}

// FitTransform fits and transforms in one step
func (n *StandardNormalizer) FitTransform(data []float64) []float64 {
	n.Fit(data)
	return n.Transform(data)
}

// GetParams returns normalization parameters
func (n *StandardNormalizer) GetParams() (mean, std float64) {
	return n.mean, n.std
}

// BatchNormalizer normalizes batches of features
type BatchNormalizer struct {
	normalizers []*StandardNormalizer
	numFeatures int
}

// NewBatchNormalizer creates a batch normalizer for n features
func NewBatchNormalizer(numFeatures int) *BatchNormalizer {
	normalizers := make([]*StandardNormalizer, numFeatures)
	for i := 0; i < numFeatures; i++ {
		normalizers[i] = NewStandardNormalizer()
	}
	return &BatchNormalizer{
		normalizers: normalizers,
		numFeatures: numFeatures,
	}
}

// Fit fits all feature normalizers
func (b *BatchNormalizer) Fit(data [][]float64) {
	for i := 0; i < b.numFeatures; i++ {
		featureData := make([]float64, len(data))
		for j, row := range data {
			if i < len(row) {
				featureData[j] = row[i]
			}
		}
		b.normalizers[i].Fit(featureData)
	}
}

// Transform normalizes a batch
func (b *BatchNormalizer) Transform(data [][]float64) [][]float64 {
	result := make([][]float64, len(data))
	for i, row := range data {
		result[i] = make([]float64, len(row))
		for j, v := range row {
			if j < b.numFeatures {
				result[i][j] = (v - b.normalizers[j].mean) / b.normalizers[j].std
			}
		}
	}
	return result
}

func main() {
	// Min-max normalization
	data := []float64{10, 20, 30, 40, 50}

	minmax := NewMinMaxNormalizer(0, 1)
	normalized := minmax.FitTransform(data)
	fmt.Println("Original:", data)
	fmt.Println("Normalized (0-1):", normalized)
	fmt.Println("Inverse:", minmax.InverseTransform(normalized))

	// Z-score normalization
	std := NewStandardNormalizer()
	standardized := std.FitTransform(data)
	fmt.Println("Standardized:", standardized)
	mean, stdv := std.GetParams()
	fmt.Printf("Mean: %.2f, Std: %.2f\\n", mean, stdv)
}`,

	testCode: `package main

import (
	"math"
	"testing"
)

func TestMinMaxNormalizer(t *testing.T) {
	normalizer := NewMinMaxNormalizer(0, 1)
	data := []float64{10, 20, 30, 40, 50}

	normalized := normalizer.FitTransform(data)

	if normalized[0] != 0 {
		t.Errorf("Min should be 0, got %f", normalized[0])
	}
	if normalized[4] != 1 {
		t.Errorf("Max should be 1, got %f", normalized[4])
	}
}

func TestMinMaxInverse(t *testing.T) {
	normalizer := NewMinMaxNormalizer(0, 1)
	original := []float64{10, 20, 30, 40, 50}

	normalized := normalizer.FitTransform(original)
	restored := normalizer.InverseTransform(normalized)

	for i := range original {
		if math.Abs(original[i]-restored[i]) > 0.001 {
			t.Errorf("Inverse failed at %d: %f != %f", i, original[i], restored[i])
		}
	}
}

func TestStandardNormalizer(t *testing.T) {
	normalizer := NewStandardNormalizer()
	data := []float64{1, 2, 3, 4, 5}

	standardized := normalizer.FitTransform(data)

	// Check mean is ~0
	var sum float64
	for _, v := range standardized {
		sum += v
	}
	mean := sum / float64(len(standardized))
	if math.Abs(mean) > 0.001 {
		t.Errorf("Mean should be ~0, got %f", mean)
	}
}

func TestStandardInverse(t *testing.T) {
	normalizer := NewStandardNormalizer()
	original := []float64{1, 2, 3, 4, 5}

	standardized := normalizer.FitTransform(original)
	restored := normalizer.InverseTransform(standardized)

	for i := range original {
		if math.Abs(original[i]-restored[i]) > 0.001 {
			t.Errorf("Inverse failed at %d: %f != %f", i, original[i], restored[i])
		}
	}
}

func TestBatchNormalizer(t *testing.T) {
	batch := NewBatchNormalizer(2)
	data := [][]float64{
		{1, 100},
		{2, 200},
		{3, 300},
	}

	batch.Fit(data)
	normalized := batch.Transform(data)

	if len(normalized) != 3 {
		t.Errorf("Expected 3 rows, got %d", len(normalized))
	}
}

func TestMinMaxCustomRange(t *testing.T) {
	normalizer := NewMinMaxNormalizer(-1, 1)
	data := []float64{0, 50, 100}

	normalized := normalizer.FitTransform(data)

	if normalized[0] != -1 {
		t.Errorf("Min should be -1, got %f", normalized[0])
	}
	if normalized[2] != 1 {
		t.Errorf("Max should be 1, got %f", normalized[2])
	}
	if math.Abs(normalized[1]-0) > 0.001 {
		t.Errorf("Middle should be ~0, got %f", normalized[1])
	}
}

func TestStandardGetParams(t *testing.T) {
	normalizer := NewStandardNormalizer()
	data := []float64{2, 4, 6, 8, 10}

	normalizer.Fit(data)
	mean, std := normalizer.GetParams()

	if math.Abs(mean-6) > 0.001 {
		t.Errorf("Mean should be 6, got %f", mean)
	}
	if std <= 0 {
		t.Error("Std should be positive")
	}
}

func TestMinMaxNotFitted(t *testing.T) {
	normalizer := NewMinMaxNormalizer(0, 1)
	data := []float64{1, 2, 3}

	// Transform without Fit should return original
	result := normalizer.Transform(data)

	for i := range data {
		if result[i] != data[i] {
			t.Error("Unfitted transform should return original data")
		}
	}
}

func TestStandardNotFitted(t *testing.T) {
	normalizer := NewStandardNormalizer()
	data := []float64{1, 2, 3}

	// Transform without Fit should return original
	result := normalizer.Transform(data)

	for i := range data {
		if result[i] != data[i] {
			t.Error("Unfitted transform should return original data")
		}
	}
}

func TestBatchNormalizerColumns(t *testing.T) {
	batch := NewBatchNormalizer(3)
	data := [][]float64{
		{1, 10, 100},
		{2, 20, 200},
		{3, 30, 300},
	}

	batch.Fit(data)
	normalized := batch.Transform(data)

	if len(normalized[0]) != 3 {
		t.Errorf("Expected 3 columns, got %d", len(normalized[0]))
	}

	// Check second column is also normalized
	var sum float64
	for _, row := range normalized {
		sum += row[1]
	}
	if math.Abs(sum) > 0.001 {
		t.Errorf("Column mean should be ~0, got %f", sum/3)
	}
}`,

	hint1: 'Min-max scaling: (x - min) / (max - min) * (target_max - target_min) + target_min',
	hint2: 'Z-score: (x - mean) / std',

	whyItMatters: `Normalization is essential for ML inference:

- **Model expectation**: Models trained on normalized data expect normalized input
- **Numerical stability**: Prevents overflow/underflow in computations
- **Consistent results**: Ensures reproducible predictions
- **Performance**: Some optimizations require normalized inputs

Using the same normalization at inference as training is critical.`,

	translations: {
		ru: {
			title: 'Нормализация признаков',
			description: `# Нормализация признаков

Реализуйте техники нормализации признаков для ML инференса.

## Задача

Создайте нормализаторы:
- Реализация min-max масштабирования
- Реализация z-score стандартизации
- Обработка batch нормализации
- Поддержка обратных преобразований

## Пример

\`\`\`go
normalizer := NewMinMaxNormalizer(0, 1)
normalizer.Fit(trainingData)
normalized := normalizer.Transform(data)
\`\`\``,
			hint1: 'Min-max: (x - min) / (max - min) * (target_max - target_min) + target_min',
			hint2: 'Z-score: (x - mean) / std',
			whyItMatters: `Нормализация необходима для ML инференса:

- **Ожидания модели**: Модели обученные на нормализованных данных ожидают нормализованный вход
- **Численная стабильность**: Предотвращение переполнения в вычислениях
- **Консистентные результаты**: Обеспечение воспроизводимых предсказаний
- **Производительность**: Некоторые оптимизации требуют нормализованных входов`,
		},
		uz: {
			title: 'Feature normalizatsiyasi',
			description: `# Feature normalizatsiyasi

ML inference uchun feature normalizatsiya texnikalarini amalga oshiring.

## Topshiriq

Normalizatorlarni yarating:
- Min-max masshtablashni amalga oshirish
- Z-score standartlashtirishni amalga oshirish
- Batch normalizatsiyani qayta ishlash
- Teskari o'zgartirishlarni qo'llab-quvvatlash

## Misol

\`\`\`go
normalizer := NewMinMaxNormalizer(0, 1)
normalizer.Fit(trainingData)
normalized := normalizer.Transform(data)
\`\`\``,
			hint1: 'Min-max: (x - min) / (max - min) * (target_max - target_min) + target_min',
			hint2: 'Z-score: (x - mean) / std',
			whyItMatters: `Normalizatsiya ML inference uchun zarur:

- **Model kutishi**: Normallashtirilgan ma'lumotlarda o'qitilgan modellar normallashtirilgan kirishni kutadi
- **Raqamli barqarorlik**: Hisoblashlarda overflow/underflow ni oldini olish
- **Izchil natijalar**: Qayta takrorlanadigan bashoratlarni ta'minlash
- **Ishlash**: Ba'zi optimallashtirishlar normallashtirilgan kirishlarni talab qiladi`,
		},
	},
};

export default task;
