import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-onnx-runtime',
	title: 'ONNX Runtime Inference',
	difficulty: 'medium',
	tags: ['go', 'ml', 'onnx', 'inference'],
	estimatedTime: '30m',
	isPremium: false,
	order: 1,
	description: `# ONNX Runtime Inference

Load and run ONNX models in Go using ONNX Runtime.

## Task

Implement ONNX model inference:
- Load ONNX model from file
- Create input tensors
- Run inference session
- Extract output tensors

## Example

\`\`\`go
model, _ := NewONNXModel("model.onnx")
input := []float32{1.0, 2.0, 3.0, 4.0}
output := model.Predict(input)
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

// ONNXModel represents an ONNX model wrapper
type ONNXModel struct {
	// Your fields here
}

// NewONNXModel loads an ONNX model from file
func NewONNXModel(path string) (*ONNXModel, error) {
	// Your code here
	return nil, nil
}

// Predict runs inference on input data
func (m *ONNXModel) Predict(input []float32) ([]float32, error) {
	// Your code here
	return nil, nil
}

// GetInputShape returns the expected input shape
func (m *ONNXModel) GetInputShape() []int64 {
	// Your code here
	return nil
}

// Close releases model resources
func (m *ONNXModel) Close() error {
	// Your code here
	return nil
}

func main() {
	fmt.Println("ONNX Runtime Inference")
}`,

	solutionCode: `package main

import (
	"fmt"
	"math"
)

// Tensor represents a multi-dimensional array
type Tensor struct {
	Data  []float32
	Shape []int64
}

// NewTensor creates a new tensor with given shape
func NewTensor(data []float32, shape []int64) *Tensor {
	return &Tensor{Data: data, Shape: shape}
}

// ONNXModel represents a simulated ONNX model wrapper
type ONNXModel struct {
	path       string
	inputShape []int64
	weights    [][]float32
	biases     []float32
}

// NewONNXModel loads an ONNX model from file
func NewONNXModel(path string) (*ONNXModel, error) {
	// Simulate loading a simple neural network model
	model := &ONNXModel{
		path:       path,
		inputShape: []int64{1, 4}, // batch_size x features
		// Simulated weights for a simple linear layer
		weights: [][]float32{
			{0.5, -0.3, 0.2, 0.1},
			{-0.2, 0.4, -0.1, 0.3},
		},
		biases: []float32{0.1, -0.1},
	}
	return model, nil
}

// Predict runs inference on input data
func (m *ONNXModel) Predict(input []float32) ([]float32, error) {
	if int64(len(input)) != m.inputShape[1] {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d",
			m.inputShape[1], len(input))
	}

	// Simple linear layer: output = input @ weights.T + bias
	output := make([]float32, len(m.weights))
	for i, w := range m.weights {
		sum := m.biases[i]
		for j, val := range input {
			sum += val * w[j]
		}
		// Apply ReLU activation
		output[i] = float32(math.Max(float64(sum), 0))
	}

	return output, nil
}

// PredictBatch runs batch inference
func (m *ONNXModel) PredictBatch(inputs [][]float32) ([][]float32, error) {
	outputs := make([][]float32, len(inputs))
	for i, input := range inputs {
		out, err := m.Predict(input)
		if err != nil {
			return nil, err
		}
		outputs[i] = out
	}
	return outputs, nil
}

// GetInputShape returns the expected input shape
func (m *ONNXModel) GetInputShape() []int64 {
	return m.inputShape
}

// GetOutputShape returns the output shape
func (m *ONNXModel) GetOutputShape() []int64 {
	return []int64{1, int64(len(m.weights))}
}

// GetModelPath returns the model file path
func (m *ONNXModel) GetModelPath() string {
	return m.path
}

// Softmax applies softmax to outputs
func Softmax(logits []float32) []float32 {
	// Find max for numerical stability
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute exp and sum
	exps := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		exps[i] = float32(math.Exp(float64(v - maxVal)))
		sum += exps[i]
	}

	// Normalize
	for i := range exps {
		exps[i] /= sum
	}
	return exps
}

// Close releases model resources
func (m *ONNXModel) Close() error {
	m.weights = nil
	m.biases = nil
	return nil
}

func main() {
	// Load model
	model, err := NewONNXModel("model.onnx")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer model.Close()

	// Print model info
	fmt.Println("Input shape:", model.GetInputShape())
	fmt.Println("Output shape:", model.GetOutputShape())

	// Run inference
	input := []float32{1.0, 2.0, 3.0, 4.0}
	output, err := model.Predict(input)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	fmt.Println("Input:", input)
	fmt.Println("Output:", output)
	fmt.Println("Softmax:", Softmax(output))
}`,

	testCode: `package main

import (
	"testing"
)

func TestNewONNXModel(t *testing.T) {
	model, err := NewONNXModel("test.onnx")
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer model.Close()

	if model == nil {
		t.Fatal("Model is nil")
	}
}

func TestPredict(t *testing.T) {
	model, _ := NewONNXModel("test.onnx")
	defer model.Close()

	input := []float32{1.0, 2.0, 3.0, 4.0}
	output, err := model.Predict(input)

	if err != nil {
		t.Fatalf("Prediction failed: %v", err)
	}
	if len(output) == 0 {
		t.Fatal("Empty output")
	}
}

func TestGetInputShape(t *testing.T) {
	model, _ := NewONNXModel("test.onnx")
	defer model.Close()

	shape := model.GetInputShape()
	if len(shape) != 2 {
		t.Fatalf("Expected 2D shape, got %d dimensions", len(shape))
	}
}

func TestPredictBatch(t *testing.T) {
	model, _ := NewONNXModel("test.onnx")
	defer model.Close()

	inputs := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{0.5, 1.5, 2.5, 3.5},
	}

	outputs, err := model.PredictBatch(inputs)
	if err != nil {
		t.Fatalf("Batch prediction failed: %v", err)
	}
	if len(outputs) != 2 {
		t.Fatalf("Expected 2 outputs, got %d", len(outputs))
	}
}

func TestSoftmax(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0}
	probs := Softmax(logits)

	var sum float32
	for _, p := range probs {
		sum += p
		if p < 0 || p > 1 {
			t.Fatalf("Invalid probability: %f", p)
		}
	}

	if sum < 0.99 || sum > 1.01 {
		t.Fatalf("Probabilities don't sum to 1: %f", sum)
	}
}

func TestGetOutputShape(t *testing.T) {
	model, _ := NewONNXModel("test.onnx")
	defer model.Close()

	shape := model.GetOutputShape()
	if len(shape) != 2 {
		t.Fatalf("Expected 2D output shape, got %d dimensions", len(shape))
	}
}

func TestGetModelPath(t *testing.T) {
	model, _ := NewONNXModel("test.onnx")
	defer model.Close()

	if model.GetModelPath() != "test.onnx" {
		t.Fatalf("Expected path test.onnx, got %s", model.GetModelPath())
	}
}

func TestPredictWrongInputSize(t *testing.T) {
	model, _ := NewONNXModel("test.onnx")
	defer model.Close()

	input := []float32{1.0, 2.0} // Wrong size
	_, err := model.Predict(input)
	if err == nil {
		t.Fatal("Expected error for wrong input size")
	}
}

func TestClose(t *testing.T) {
	model, _ := NewONNXModel("test.onnx")
	err := model.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}
}

func TestSoftmaxOrder(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0}
	probs := Softmax(logits)

	if probs[2] <= probs[1] || probs[1] <= probs[0] {
		t.Fatal("Softmax should preserve order")
	}
}`,

	hint1: 'ONNX Runtime uses sessions to run inference - create once, reuse for each prediction',
	hint2: 'Input/output tensors must match the expected shapes defined in the model',

	whyItMatters: `ONNX is the standard for ML model interoperability:

- **Cross-framework**: Train in PyTorch/TensorFlow, deploy anywhere
- **Optimized**: Hardware-specific optimizations (CPU, GPU, edge)
- **Production-ready**: Used by major tech companies
- **Go ecosystem**: go-onnxruntime provides native bindings

ONNX enables using pre-trained models in Go services.`,

	translations: {
		ru: {
			title: 'Инференс ONNX Runtime',
			description: `# Инференс ONNX Runtime

Загрузка и запуск ONNX моделей в Go с использованием ONNX Runtime.

## Задача

Реализуйте инференс ONNX модели:
- Загрузка ONNX модели из файла
- Создание входных тензоров
- Запуск сессии инференса
- Извлечение выходных тензоров

## Пример

\`\`\`go
model, _ := NewONNXModel("model.onnx")
input := []float32{1.0, 2.0, 3.0, 4.0}
output := model.Predict(input)
\`\`\``,
			hint1: 'ONNX Runtime использует сессии для инференса - создайте один раз, используйте повторно',
			hint2: 'Входные/выходные тензоры должны соответствовать формам, определенным в модели',
			whyItMatters: `ONNX - стандарт интероперабельности ML моделей:

- **Кросс-фреймворк**: Обучайте в PyTorch/TensorFlow, деплойте везде
- **Оптимизировано**: Аппаратно-специфичные оптимизации
- **Production-ready**: Используется крупными компаниями
- **Go экосистема**: go-onnxruntime предоставляет нативные биндинги`,
		},
		uz: {
			title: 'ONNX Runtime Inference',
			description: `# ONNX Runtime Inference

ONNX Runtime yordamida Go da ONNX modellarni yuklash va ishga tushirish.

## Topshiriq

ONNX model inference ni amalga oshiring:
- ONNX modelni fayldan yuklash
- Kirish tensorlarini yaratish
- Inference sessiyasini ishga tushirish
- Chiqish tensorlarini ajratib olish

## Misol

\`\`\`go
model, _ := NewONNXModel("model.onnx")
input := []float32{1.0, 2.0, 3.0, 4.0}
output := model.Predict(input)
\`\`\``,
			hint1: "ONNX Runtime inference uchun sessiyalardan foydalanadi - bir marta yarating, qayta foydalaning",
			hint2: "Kirish/chiqish tensorlari modelda belgilangan shakllariga mos kelishi kerak",
			whyItMatters: `ONNX ML model interoperabelligi uchun standart:

- **Kross-framework**: PyTorch/TensorFlow da o'qiting, hamma joyda deploy qiling
- **Optimallashtirilgan**: Hardware-specific optimallashtirishlar
- **Production-ready**: Yirik texnologik kompaniyalar tomonidan ishlatiladi
- **Go ekotizimi**: go-onnxruntime nativ bindinglarni taqdim etadi`,
		},
	},
};

export default task;
