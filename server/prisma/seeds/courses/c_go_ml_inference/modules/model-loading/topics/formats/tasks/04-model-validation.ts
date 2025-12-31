import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-model-validation',
	title: 'Model Validation',
	difficulty: 'medium',
	tags: ['go', 'ml', 'validation', 'testing'],
	estimatedTime: '25m',
	isPremium: false,
	order: 4,
	description: `# Model Validation

Implement model validation and sanity checks before deployment.

## Task

Build model validation that:
- Verifies model input/output shapes
- Tests with sample inputs
- Compares outputs against baselines
- Checks for NaN/Inf values

## Example

\`\`\`go
validator := NewModelValidator(model)
result := validator.Validate(testCases)
if !result.Passed {
    log.Println("Validation failed:", result.Errors)
}
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

// ModelValidator validates ML models
type ModelValidator struct {
	// Your fields here
}

// ValidationResult holds validation results
type ValidationResult struct {
	Passed   bool
	Errors   []string
	Warnings []string
}

// TestCase defines a validation test case
type TestCase struct {
	Input    []float32
	Expected []float32
	Epsilon  float32
}

// NewModelValidator creates a validator
func NewModelValidator(model interface{}) *ModelValidator {
	// Your code here
	return nil
}

// Validate runs all validation checks
func (v *ModelValidator) Validate(testCases []TestCase) ValidationResult {
	// Your code here
	return ValidationResult{}
}

// CheckShapes validates input/output shapes
func (v *ModelValidator) CheckShapes() error {
	// Your code here
	return nil
}

func main() {
	fmt.Println("Model Validation")
}`,

	solutionCode: `package main

import (
	"fmt"
	"math"
)

// Predictor interface for model inference
type Predictor interface {
	Predict(input []float32) ([]float32, error)
	GetInputShape() []int64
	GetOutputShape() []int64
}

// TestCase defines a validation test case
type TestCase struct {
	Name     string
	Input    []float32
	Expected []float32
	Epsilon  float32
}

// ValidationResult holds validation results
type ValidationResult struct {
	Passed      bool
	Errors      []string
	Warnings    []string
	NumTests    int
	NumPassed   int
	NumFailed   int
	MaxError    float64
	AvgLatency  float64
}

// ModelValidator validates ML models
type ModelValidator struct {
	model           Predictor
	verbose         bool
	checkNaN        bool
	checkInf        bool
	maxOutputValue  float64
}

// NewModelValidator creates a validator
func NewModelValidator(model Predictor) *ModelValidator {
	return &ModelValidator{
		model:          model,
		verbose:        true,
		checkNaN:       true,
		checkInf:       true,
		maxOutputValue: 1e10,
	}
}

// SetVerbose enables verbose output
func (v *ModelValidator) SetVerbose(verbose bool) {
	v.verbose = verbose
}

// CheckShapes validates input/output shapes
func (v *ModelValidator) CheckShapes() error {
	inputShape := v.model.GetInputShape()
	outputShape := v.model.GetOutputShape()

	if len(inputShape) == 0 {
		return fmt.Errorf("input shape is empty")
	}
	if len(outputShape) == 0 {
		return fmt.Errorf("output shape is empty")
	}

	for i, dim := range inputShape {
		if dim <= 0 {
			return fmt.Errorf("invalid input dimension at index %d: %d", i, dim)
		}
	}

	for i, dim := range outputShape {
		if dim <= 0 {
			return fmt.Errorf("invalid output dimension at index %d: %d", i, dim)
		}
	}

	return nil
}

// CheckNumericalStability tests for NaN/Inf
func (v *ModelValidator) CheckNumericalStability(input []float32) error {
	output, err := v.model.Predict(input)
	if err != nil {
		return fmt.Errorf("prediction failed: %w", err)
	}

	for i, val := range output {
		if v.checkNaN && math.IsNaN(float64(val)) {
			return fmt.Errorf("NaN detected at output index %d", i)
		}
		if v.checkInf && math.IsInf(float64(val), 0) {
			return fmt.Errorf("Inf detected at output index %d", i)
		}
		if math.Abs(float64(val)) > v.maxOutputValue {
			return fmt.Errorf("output value too large at index %d: %f", i, val)
		}
	}

	return nil
}

// RunTestCase runs a single test case
func (v *ModelValidator) RunTestCase(tc TestCase) (bool, float64, error) {
	output, err := v.model.Predict(tc.Input)
	if err != nil {
		return false, 0, err
	}

	if len(output) != len(tc.Expected) {
		return false, 0, fmt.Errorf("output length mismatch: got %d, expected %d",
			len(output), len(tc.Expected))
	}

	var maxError float64
	for i, expected := range tc.Expected {
		diff := math.Abs(float64(output[i] - expected))
		if diff > maxError {
			maxError = diff
		}
		if diff > float64(tc.Epsilon) {
			return false, maxError, fmt.Errorf("value mismatch at index %d: got %f, expected %f (diff: %f)",
				i, output[i], expected, diff)
		}
	}

	return true, maxError, nil
}

// Validate runs all validation checks
func (v *ModelValidator) Validate(testCases []TestCase) ValidationResult {
	result := ValidationResult{
		Passed:    true,
		NumTests:  len(testCases),
		MaxError:  0,
	}

	// Check shapes
	if err := v.CheckShapes(); err != nil {
		result.Passed = false
		result.Errors = append(result.Errors, "Shape check: "+err.Error())
		return result
	}

	if v.verbose {
		fmt.Println("Shape validation passed")
	}

	// Run test cases
	for _, tc := range testCases {
		// Check numerical stability
		if err := v.CheckNumericalStability(tc.Input); err != nil {
			result.Warnings = append(result.Warnings,
				fmt.Sprintf("Test '%s': %v", tc.Name, err))
		}

		// Run test
		passed, maxErr, err := v.RunTestCase(tc)
		if err != nil {
			result.Passed = false
			result.NumFailed++
			result.Errors = append(result.Errors,
				fmt.Sprintf("Test '%s': %v", tc.Name, err))
		} else if passed {
			result.NumPassed++
		}

		if maxErr > result.MaxError {
			result.MaxError = maxErr
		}

		if v.verbose {
			status := "PASS"
			if !passed {
				status = "FAIL"
			}
			fmt.Printf("Test '%s': %s (max error: %.6f)\\n", tc.Name, status, maxErr)
		}
	}

	result.Passed = result.NumFailed == 0 && len(result.Errors) == 0
	return result
}

// GenerateRandomInput creates random test input
func (v *ModelValidator) GenerateRandomInput() []float32 {
	shape := v.model.GetInputShape()
	size := 1
	for _, dim := range shape {
		size *= int(dim)
	}

	input := make([]float32, size)
	for i := range input {
		input[i] = float32(i) * 0.1
	}
	return input
}

// Summary returns validation summary string
func (r *ValidationResult) Summary() string {
	return fmt.Sprintf("Validation: %d/%d passed, max error: %.6f",
		r.NumPassed, r.NumTests, r.MaxError)
}

// Simple mock model for testing
type MockModel struct {
	inputShape  []int64
	outputShape []int64
}

func (m *MockModel) Predict(input []float32) ([]float32, error) {
	// Simple pass-through
	output := make([]float32, 2)
	output[0] = input[0] * 0.5
	output[1] = input[1] * 0.5
	return output, nil
}

func (m *MockModel) GetInputShape() []int64  { return m.inputShape }
func (m *MockModel) GetOutputShape() []int64 { return m.outputShape }

func main() {
	model := &MockModel{
		inputShape:  []int64{1, 4},
		outputShape: []int64{1, 2},
	}

	validator := NewModelValidator(model)

	testCases := []TestCase{
		{
			Name:     "basic",
			Input:    []float32{1.0, 2.0, 3.0, 4.0},
			Expected: []float32{0.5, 1.0},
			Epsilon:  0.01,
		},
	}

	result := validator.Validate(testCases)
	fmt.Println(result.Summary())
}`,

	testCode: `package main

import (
	"testing"
)

func TestNewModelValidator(t *testing.T) {
	model := &MockModel{
		inputShape:  []int64{1, 4},
		outputShape: []int64{1, 2},
	}
	validator := NewModelValidator(model)

	if validator == nil {
		t.Fatal("Validator is nil")
	}
}

func TestCheckShapes(t *testing.T) {
	model := &MockModel{
		inputShape:  []int64{1, 4},
		outputShape: []int64{1, 2},
	}
	validator := NewModelValidator(model)

	err := validator.CheckShapes()
	if err != nil {
		t.Fatalf("Shape check failed: %v", err)
	}
}

func TestValidatePass(t *testing.T) {
	model := &MockModel{
		inputShape:  []int64{1, 4},
		outputShape: []int64{1, 2},
	}
	validator := NewModelValidator(model)
	validator.SetVerbose(false)

	testCases := []TestCase{
		{
			Name:     "test1",
			Input:    []float32{1.0, 2.0, 3.0, 4.0},
			Expected: []float32{0.5, 1.0},
			Epsilon:  0.01,
		},
	}

	result := validator.Validate(testCases)
	if !result.Passed {
		t.Fatalf("Validation should pass: %v", result.Errors)
	}
}

func TestValidateFail(t *testing.T) {
	model := &MockModel{
		inputShape:  []int64{1, 4},
		outputShape: []int64{1, 2},
	}
	validator := NewModelValidator(model)
	validator.SetVerbose(false)

	testCases := []TestCase{
		{
			Name:     "wrong_expected",
			Input:    []float32{1.0, 2.0, 3.0, 4.0},
			Expected: []float32{999.0, 999.0}, // Wrong values
			Epsilon:  0.01,
		},
	}

	result := validator.Validate(testCases)
	if result.Passed {
		t.Fatal("Validation should fail")
	}
}

func TestGenerateRandomInput(t *testing.T) {
	model := &MockModel{
		inputShape:  []int64{1, 4},
		outputShape: []int64{1, 2},
	}
	validator := NewModelValidator(model)

	input := validator.GenerateRandomInput()
	if len(input) != 4 {
		t.Fatalf("Expected 4 inputs, got %d", len(input))
	}
}

func TestResultSummary(t *testing.T) {
	result := ValidationResult{NumPassed: 5, NumTests: 5, MaxError: 0.001}
	summary := result.Summary()
	if summary == "" {
		t.Fatal("Summary is empty")
	}
}

func TestCheckNumericalStability(t *testing.T) {
	model := &MockModel{
		inputShape:  []int64{1, 4},
		outputShape: []int64{1, 2},
	}
	validator := NewModelValidator(model)

	err := validator.CheckNumericalStability([]float32{1.0, 2.0, 3.0, 4.0})
	if err != nil {
		t.Fatalf("Numerical stability check failed: %v", err)
	}
}

func TestRunTestCase(t *testing.T) {
	model := &MockModel{
		inputShape:  []int64{1, 4},
		outputShape: []int64{1, 2},
	}
	validator := NewModelValidator(model)

	tc := TestCase{
		Name:     "test",
		Input:    []float32{1.0, 2.0, 3.0, 4.0},
		Expected: []float32{0.5, 1.0},
		Epsilon:  0.01,
	}

	passed, _, err := validator.RunTestCase(tc)
	if err != nil || !passed {
		t.Fatalf("Test case should pass")
	}
}

func TestSetVerbose(t *testing.T) {
	model := &MockModel{
		inputShape:  []int64{1, 4},
		outputShape: []int64{1, 2},
	}
	validator := NewModelValidator(model)
	validator.SetVerbose(false)
	// Should not panic
}

func TestValidationResultFields(t *testing.T) {
	model := &MockModel{
		inputShape:  []int64{1, 4},
		outputShape: []int64{1, 2},
	}
	validator := NewModelValidator(model)
	validator.SetVerbose(false)

	testCases := []TestCase{
		{Name: "t1", Input: []float32{1.0, 2.0, 3.0, 4.0}, Expected: []float32{0.5, 1.0}, Epsilon: 0.01},
	}

	result := validator.Validate(testCases)
	if result.NumTests != 1 {
		t.Fatalf("Expected 1 test, got %d", result.NumTests)
	}
}`,

	hint1: 'Compare outputs with expected values using epsilon tolerance for floating-point comparison',
	hint2: 'Check for NaN and Inf values that indicate numerical instability',

	whyItMatters: `Model validation prevents production failures:

- **Correctness**: Verify model produces expected outputs
- **Stability**: Detect numerical issues before deployment
- **Regression**: Catch model degradation after updates
- **Confidence**: Deploy with assurance that model works

Validation is essential for reliable ML systems.`,

	translations: {
		ru: {
			title: 'Валидация модели',
			description: `# Валидация модели

Реализуйте валидацию модели и проверки перед деплоем.

## Задача

Создайте валидацию модели:
- Проверка входных/выходных размерностей
- Тестирование с примерами входных данных
- Сравнение выходов с базовыми значениями
- Проверка на NaN/Inf значения

## Пример

\`\`\`go
validator := NewModelValidator(model)
result := validator.Validate(testCases)
if !result.Passed {
    log.Println("Validation failed:", result.Errors)
}
\`\`\``,
			hint1: 'Сравнивайте выходы с ожидаемыми значениями используя epsilon-толерантность',
			hint2: 'Проверяйте на NaN и Inf значения, которые указывают на численную нестабильность',
			whyItMatters: `Валидация модели предотвращает сбои в продакшене:

- **Корректность**: Проверка что модель дает ожидаемые выходы
- **Стабильность**: Обнаружение численных проблем до деплоя
- **Регрессия**: Отлов деградации модели после обновлений
- **Уверенность**: Деплой с гарантией работоспособности`,
		},
		uz: {
			title: 'Model validatsiyasi',
			description: `# Model validatsiyasi

Deploydan oldin model validatsiyasi va tekshiruvlarni amalga oshiring.

## Topshiriq

Model validatsiyasini yarating:
- Kirish/chiqish shakllarini tekshirish
- Namuna kirishlar bilan test qilish
- Chiqishlarni bazaviy qiymatlar bilan solishtirish
- NaN/Inf qiymatlarni tekshirish

## Misol

\`\`\`go
validator := NewModelValidator(model)
result := validator.Validate(testCases)
if !result.Passed {
    log.Println("Validation failed:", result.Errors)
}
\`\`\``,
			hint1: "Chiqishlarni kutilgan qiymatlar bilan epsilon toleransi yordamida solishtiring",
			hint2: "Raqamli beqarorlikni ko'rsatadigan NaN va Inf qiymatlarni tekshiring",
			whyItMatters: `Model validatsiyasi ishlab chiqarish nosozliklarini oldini oladi:

- **To'g'rilik**: Model kutilgan chiqishlarni berishini tasdiqlash
- **Barqarorlik**: Deploydan oldin raqamli muammolarni aniqlash
- **Regressiya**: Yangilanishlardan keyin model degradatsiyasini ushlash
- **Ishonch**: Model ishlashiga kafolat bilan deploy qilish`,
		},
	},
};

export default task;
