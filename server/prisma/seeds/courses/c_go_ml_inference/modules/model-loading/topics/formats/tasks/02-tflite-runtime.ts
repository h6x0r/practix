import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-tflite-runtime',
	title: 'TensorFlow Lite Inference',
	difficulty: 'medium',
	tags: ['go', 'ml', 'tflite', 'inference', 'edge'],
	estimatedTime: '30m',
	isPremium: false,
	order: 2,
	description: `# TensorFlow Lite Inference

Run TensorFlow Lite models in Go for edge deployment.

## Task

Implement TFLite model inference:
- Load TFLite model from flatbuffer
- Allocate tensors
- Set input data
- Invoke interpreter and read output

## Example

\`\`\`go
model, _ := NewTFLiteModel("model.tflite")
input := []float32{1.0, 2.0, 3.0}
output := model.Invoke(input)
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

// TFLiteModel represents a TensorFlow Lite model
type TFLiteModel struct {
	// Your fields here
}

// NewTFLiteModel loads a TFLite model
func NewTFLiteModel(path string) (*TFLiteModel, error) {
	// Your code here
	return nil, nil
}

// Invoke runs inference
func (m *TFLiteModel) Invoke(input []float32) ([]float32, error) {
	// Your code here
	return nil, nil
}

// GetInputDetails returns input tensor info
func (m *TFLiteModel) GetInputDetails() TensorInfo {
	// Your code here
	return TensorInfo{}
}

// Close releases resources
func (m *TFLiteModel) Close() error {
	// Your code here
	return nil
}

type TensorInfo struct {
	Name  string
	Shape []int
	Type  string
}

func main() {
	fmt.Println("TFLite Inference")
}`,

	solutionCode: `package main

import (
	"fmt"
	"math"
)

// TensorInfo holds tensor metadata
type TensorInfo struct {
	Name  string
	Shape []int
	Type  string
	Index int
}

// TFLiteModel represents a simulated TensorFlow Lite model
type TFLiteModel struct {
	path         string
	inputInfo    TensorInfo
	outputInfo   TensorInfo
	weights      [][]float32
	biases       []float32
	isQuantized  bool
	inputScale   float32
	inputZero    int32
	outputScale  float32
	outputZero   int32
}

// NewTFLiteModel loads a TFLite model
func NewTFLiteModel(path string) (*TFLiteModel, error) {
	model := &TFLiteModel{
		path: path,
		inputInfo: TensorInfo{
			Name:  "input",
			Shape: []int{1, 3},
			Type:  "float32",
			Index: 0,
		},
		outputInfo: TensorInfo{
			Name:  "output",
			Shape: []int{1, 2},
			Type:  "float32",
			Index: 0,
		},
		// Simulated model weights
		weights: [][]float32{
			{0.5, 0.3, -0.2},
			{-0.1, 0.4, 0.6},
		},
		biases:      []float32{0.1, -0.05},
		isQuantized: false,
		inputScale:  1.0,
		inputZero:   0,
		outputScale: 1.0,
		outputZero:  0,
	}
	return model, nil
}

// NewQuantizedTFLiteModel loads a quantized model
func NewQuantizedTFLiteModel(path string) (*TFLiteModel, error) {
	model, _ := NewTFLiteModel(path)
	model.isQuantized = true
	model.inputInfo.Type = "uint8"
	model.outputInfo.Type = "uint8"
	model.inputScale = 0.00784314
	model.inputZero = 127
	model.outputScale = 0.00390625
	model.outputZero = 0
	return model, nil
}

// Invoke runs inference
func (m *TFLiteModel) Invoke(input []float32) ([]float32, error) {
	expectedSize := 1
	for _, dim := range m.inputInfo.Shape {
		expectedSize *= dim
	}

	if len(input) != expectedSize {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d",
			expectedSize, len(input))
	}

	// Simple linear layer computation
	output := make([]float32, len(m.weights))
	for i, w := range m.weights {
		sum := m.biases[i]
		for j, val := range input {
			sum += val * w[j]
		}
		// Sigmoid activation
		output[i] = float32(1.0 / (1.0 + math.Exp(-float64(sum))))
	}

	return output, nil
}

// InvokeQuantized runs quantized inference
func (m *TFLiteModel) InvokeQuantized(input []uint8) ([]uint8, error) {
	// Dequantize input
	floatInput := make([]float32, len(input))
	for i, v := range input {
		floatInput[i] = (float32(v) - float32(m.inputZero)) * m.inputScale
	}

	// Run inference
	floatOutput, err := m.Invoke(floatInput)
	if err != nil {
		return nil, err
	}

	// Quantize output
	output := make([]uint8, len(floatOutput))
	for i, v := range floatOutput {
		quantized := int32(v/m.outputScale) + m.outputZero
		if quantized < 0 {
			quantized = 0
		} else if quantized > 255 {
			quantized = 255
		}
		output[i] = uint8(quantized)
	}

	return output, nil
}

// GetInputDetails returns input tensor info
func (m *TFLiteModel) GetInputDetails() TensorInfo {
	return m.inputInfo
}

// GetOutputDetails returns output tensor info
func (m *TFLiteModel) GetOutputDetails() TensorInfo {
	return m.outputInfo
}

// IsQuantized returns whether model uses quantization
func (m *TFLiteModel) IsQuantized() bool {
	return m.isQuantized
}

// GetModelPath returns model file path
func (m *TFLiteModel) GetModelPath() string {
	return m.path
}

// Resize adjusts input dimensions (for dynamic shapes)
func (m *TFLiteModel) Resize(newShape []int) error {
	if len(newShape) != len(m.inputInfo.Shape) {
		return fmt.Errorf("shape dimension mismatch")
	}
	m.inputInfo.Shape = newShape
	return nil
}

// Close releases resources
func (m *TFLiteModel) Close() error {
	m.weights = nil
	m.biases = nil
	return nil
}

func main() {
	// Load model
	model, err := NewTFLiteModel("model.tflite")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer model.Close()

	// Print model info
	fmt.Println("Input:", model.GetInputDetails())
	fmt.Println("Output:", model.GetOutputDetails())

	// Run inference
	input := []float32{1.0, 2.0, 3.0}
	output, err := model.Invoke(input)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	fmt.Println("Input:", input)
	fmt.Println("Output:", output)
}`,

	testCode: `package main

import (
	"testing"
)

func TestNewTFLiteModel(t *testing.T) {
	model, err := NewTFLiteModel("test.tflite")
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	if model == nil {
		t.Fatal("Model is nil")
	}
}

func TestInvoke(t *testing.T) {
	model, _ := NewTFLiteModel("test.tflite")
	defer model.Close()

	input := []float32{1.0, 2.0, 3.0}
	output, err := model.Invoke(input)

	if err != nil {
		t.Fatalf("Invoke failed: %v", err)
	}
	if len(output) != 2 {
		t.Fatalf("Expected 2 outputs, got %d", len(output))
	}

	// Check sigmoid outputs are in [0, 1]
	for i, v := range output {
		if v < 0 || v > 1 {
			t.Fatalf("Output %d out of sigmoid range: %f", i, v)
		}
	}
}

func TestGetInputDetails(t *testing.T) {
	model, _ := NewTFLiteModel("test.tflite")
	defer model.Close()

	info := model.GetInputDetails()
	if info.Name == "" {
		t.Fatal("Input name is empty")
	}
	if len(info.Shape) == 0 {
		t.Fatal("Input shape is empty")
	}
}

func TestQuantizedModel(t *testing.T) {
	model, _ := NewQuantizedTFLiteModel("test_quant.tflite")
	defer model.Close()

	if !model.IsQuantized() {
		t.Fatal("Model should be quantized")
	}

	input := []uint8{128, 200, 50}
	output, err := model.InvokeQuantized(input)
	if err != nil {
		t.Fatalf("Quantized invoke failed: %v", err)
	}
	if len(output) == 0 {
		t.Fatal("Empty output")
	}
}

func TestResize(t *testing.T) {
	model, _ := NewTFLiteModel("test.tflite")
	defer model.Close()

	newShape := []int{2, 3}
	err := model.Resize(newShape)
	if err != nil {
		t.Fatalf("Resize failed: %v", err)
	}

	info := model.GetInputDetails()
	if info.Shape[0] != 2 {
		t.Fatalf("Expected batch 2, got %d", info.Shape[0])
	}
}

func TestGetOutputDetails(t *testing.T) {
	model, _ := NewTFLiteModel("test.tflite")
	defer model.Close()

	info := model.GetOutputDetails()
	if info.Name == "" {
		t.Fatal("Output name is empty")
	}
}

func TestGetModelPath(t *testing.T) {
	model, _ := NewTFLiteModel("test.tflite")
	defer model.Close()

	if model.GetModelPath() != "test.tflite" {
		t.Fatalf("Expected test.tflite, got %s", model.GetModelPath())
	}
}

func TestInvokeWrongSize(t *testing.T) {
	model, _ := NewTFLiteModel("test.tflite")
	defer model.Close()

	input := []float32{1.0} // Wrong size
	_, err := model.Invoke(input)
	if err == nil {
		t.Fatal("Expected error for wrong input size")
	}
}

func TestClose(t *testing.T) {
	model, _ := NewTFLiteModel("test.tflite")
	err := model.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}
}

func TestIsQuantizedFalse(t *testing.T) {
	model, _ := NewTFLiteModel("test.tflite")
	defer model.Close()

	if model.IsQuantized() {
		t.Fatal("Non-quantized model should return false")
	}
}`,

	hint1: 'TFLite uses flatbuffer format - load model bytes and create interpreter',
	hint2: 'For quantized models, dequantize inputs and quantize outputs using scale/zero-point',

	whyItMatters: `TensorFlow Lite is designed for edge and mobile:

- **Small footprint**: Minimal binary size for embedded systems
- **Quantization**: INT8 inference for faster execution
- **Hardware acceleration**: Delegates for GPU, DSP, NPU
- **Go support**: gocv and tflite-go bindings available

TFLite enables ML on resource-constrained devices.`,

	translations: {
		ru: {
			title: 'Инференс TensorFlow Lite',
			description: `# Инференс TensorFlow Lite

Запуск TensorFlow Lite моделей в Go для edge деплоя.

## Задача

Реализуйте TFLite инференс:
- Загрузка TFLite модели из flatbuffer
- Аллокация тензоров
- Установка входных данных
- Вызов интерпретатора и чтение выхода

## Пример

\`\`\`go
model, _ := NewTFLiteModel("model.tflite")
input := []float32{1.0, 2.0, 3.0}
output := model.Invoke(input)
\`\`\``,
			hint1: 'TFLite использует формат flatbuffer - загрузите байты модели и создайте интерпретатор',
			hint2: 'Для квантизированных моделей деквантизируйте входы и квантизируйте выходы',
			whyItMatters: `TensorFlow Lite разработан для edge и мобильных устройств:

- **Малый размер**: Минимальный бинарник для встраиваемых систем
- **Квантизация**: INT8 инференс для быстрого выполнения
- **Аппаратное ускорение**: Делегаты для GPU, DSP, NPU
- **Go поддержка**: Доступны биндинги gocv и tflite-go`,
		},
		uz: {
			title: 'TensorFlow Lite Inference',
			description: `# TensorFlow Lite Inference

Edge deployment uchun Go da TensorFlow Lite modellarni ishga tushirish.

## Topshiriq

TFLite inference ni amalga oshiring:
- TFLite modelni flatbufferdan yuklash
- Tensorlarni ajratish
- Kirish ma'lumotlarini o'rnatish
- Interpretatorni chaqirish va chiqishni o'qish

## Misol

\`\`\`go
model, _ := NewTFLiteModel("model.tflite")
input := []float32{1.0, 2.0, 3.0}
output := model.Invoke(input)
\`\`\``,
			hint1: "TFLite flatbuffer formatidan foydalanadi - model baytlarini yuklang va interpreter yarating",
			hint2: "Kvantlangan modellar uchun kirishlarni dekvantlang va chiqishlarni kvantlang",
			whyItMatters: `TensorFlow Lite edge va mobil uchun yaratilgan:

- **Kichik hajm**: O'rnatilgan tizimlar uchun minimal binary hajmi
- **Kvantlash**: Tez bajarish uchun INT8 inference
- **Hardware tezlashtirish**: GPU, DSP, NPU uchun delegatlar
- **Go qo'llab-quvvatlash**: gocv va tflite-go bindinglar mavjud`,
		},
	},
};

export default task;
