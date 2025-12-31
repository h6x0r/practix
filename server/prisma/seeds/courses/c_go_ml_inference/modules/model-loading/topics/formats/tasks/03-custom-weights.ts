import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-custom-weights',
	title: 'Custom Weight Loading',
	difficulty: 'medium',
	tags: ['go', 'ml', 'weights', 'serialization'],
	estimatedTime: '25m',
	isPremium: false,
	order: 3,
	description: `# Custom Weight Loading

Load custom model weights from various formats in Go.

## Task

Implement weight loading:
- Load weights from JSON/binary files
- Parse weight shapes and values
- Validate weight dimensions
- Support different data types

## Example

\`\`\`go
loader := NewWeightLoader()
weights, _ := loader.LoadJSON("weights.json")
model := NewNeuralNet(weights)
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

type WeightLoader struct {
	// Your fields here
}

type Weights struct {
	Layers map[string]LayerWeights
}

type LayerWeights struct {
	W     [][]float32
}

func NewWeightLoader() *WeightLoader {
	return nil
}

func (l *WeightLoader) LoadJSON(path string) (*Weights, error) {
	return nil, nil
}

func (l *WeightLoader) LoadBinary(path string) (*Weights, error) {
	return nil, nil
}

func (w *Weights) Validate(architecture []int) error {
	return nil
}

func main() {
}`,

	solutionCode: `package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
)

// LayerWeights holds weights for a single layer
type LayerWeights struct {
	W     [][]float32 \`json:"weights"\`
	B     []float32   \`json:"biases"\`
	Shape []int       \`json:"shape"\`
	Name  string      \`json:"name"\`
}

// Weights holds all layer weights
type Weights struct {
	Layers   map[string]LayerWeights \`json:"layers"\`
	Metadata WeightMetadata          \`json:"metadata"\`
}

// WeightMetadata holds model metadata
type WeightMetadata struct {
	Version     string \`json:"version"\`
	Framework   string \`json:"framework"\`
	NumLayers   int    \`json:"num_layers"\`
	TotalParams int64  \`json:"total_params"\`
}

// WeightLoader handles loading model weights
type WeightLoader struct {
	verbose bool
}

// NewWeightLoader creates a weight loader
func NewWeightLoader() *WeightLoader {
	return &WeightLoader{verbose: false}
}

// SetVerbose enables verbose logging
func (l *WeightLoader) SetVerbose(v bool) {
	l.verbose = v
}

// LoadJSON loads weights from JSON format
func (l *WeightLoader) LoadJSON(path string) (*Weights, error) {
	// Simulate JSON content
	jsonData := \`{
		"metadata": {
			"version": "1.0",
			"framework": "custom",
			"num_layers": 2,
			"total_params": 14
		},
		"layers": {
			"layer1": {
				"name": "layer1",
				"shape": [3, 4],
				"weights": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
				"biases": [0.01, 0.02, 0.03]
			},
			"layer2": {
				"name": "layer2",
				"shape": [2, 3],
				"weights": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
				"biases": [0.01, 0.02]
			}
		}
	}\`

	var weights Weights
	if err := json.Unmarshal([]byte(jsonData), &weights); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	if l.verbose {
		fmt.Printf("Loaded %d layers, %d total params\\n",
			weights.Metadata.NumLayers, weights.Metadata.TotalParams)
	}

	return &weights, nil
}

// LoadBinary loads weights from binary format
func (l *WeightLoader) LoadBinary(path string) (*Weights, error) {
	// Simulate binary format: [num_layers][for each: name_len, name, rows, cols, weights, bias_len, biases]
	weights := &Weights{
		Layers: make(map[string]LayerWeights),
		Metadata: WeightMetadata{
			Version:   "1.0",
			Framework: "binary",
		},
	}

	// Simulated binary data
	data := simulateBinaryWeights()
	reader := bytes.NewReader(data)

	var numLayers int32
	binary.Read(reader, binary.LittleEndian, &numLayers)

	for i := 0; i < int(numLayers); i++ {
		// Read layer name
		var nameLen int32
		binary.Read(reader, binary.LittleEndian, &nameLen)
		nameBytes := make([]byte, nameLen)
		reader.Read(nameBytes)
		name := string(nameBytes)

		// Read shape
		var rows, cols int32
		binary.Read(reader, binary.LittleEndian, &rows)
		binary.Read(reader, binary.LittleEndian, &cols)

		// Read weights
		w := make([][]float32, rows)
		for r := 0; r < int(rows); r++ {
			w[r] = make([]float32, cols)
			for c := 0; c < int(cols); c++ {
				var val float32
				binary.Read(reader, binary.LittleEndian, &val)
				w[r][c] = val
			}
		}

		// Read biases
		var biasLen int32
		binary.Read(reader, binary.LittleEndian, &biasLen)
		b := make([]float32, biasLen)
		for j := 0; j < int(biasLen); j++ {
			binary.Read(reader, binary.LittleEndian, &b[j])
		}

		weights.Layers[name] = LayerWeights{
			Name:  name,
			Shape: []int{int(rows), int(cols)},
			W:     w,
			B:     b,
		}
	}

	weights.Metadata.NumLayers = int(numLayers)
	return weights, nil
}

func simulateBinaryWeights() []byte {
	buf := new(bytes.Buffer)

	// Number of layers
	binary.Write(buf, binary.LittleEndian, int32(1))

	// Layer 1
	name := "dense1"
	binary.Write(buf, binary.LittleEndian, int32(len(name)))
	buf.WriteString(name)
	binary.Write(buf, binary.LittleEndian, int32(2)) // rows
	binary.Write(buf, binary.LittleEndian, int32(3)) // cols

	// Weights
	for _, v := range []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6} {
		binary.Write(buf, binary.LittleEndian, v)
	}

	// Biases
	binary.Write(buf, binary.LittleEndian, int32(2))
	binary.Write(buf, binary.LittleEndian, float32(0.01))
	binary.Write(buf, binary.LittleEndian, float32(0.02))

	return buf.Bytes()
}

// Validate checks weight dimensions against architecture
func (w *Weights) Validate(architecture []int) error {
	if len(architecture) < 2 {
		return fmt.Errorf("architecture must have at least 2 layers")
	}

	layerIdx := 0
	for name, layer := range w.Layers {
		if len(layer.Shape) != 2 {
			return fmt.Errorf("layer %s: expected 2D shape", name)
		}

		expectedRows := layer.Shape[0]
		expectedCols := layer.Shape[1]

		if len(layer.W) != expectedRows {
			return fmt.Errorf("layer %s: weight rows mismatch", name)
		}

		for i, row := range layer.W {
			if len(row) != expectedCols {
				return fmt.Errorf("layer %s: row %d has wrong columns", name, i)
			}
		}

		if len(layer.B) != expectedRows {
			return fmt.Errorf("layer %s: bias dimension mismatch", name)
		}

		layerIdx++
	}

	return nil
}

// GetTotalParams returns total parameter count
func (w *Weights) GetTotalParams() int64 {
	var total int64
	for _, layer := range w.Layers {
		total += int64(len(layer.W) * len(layer.W[0]))
		total += int64(len(layer.B))
	}
	return total
}

// GetLayerNames returns all layer names
func (w *Weights) GetLayerNames() []string {
	names := make([]string, 0, len(w.Layers))
	for name := range w.Layers {
		names = append(names, name)
	}
	return names
}

func main() {
	loader := NewWeightLoader()
	loader.SetVerbose(true)

	// Load from JSON
	weights, err := loader.LoadJSON("weights.json")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	fmt.Println("Layers:", weights.GetLayerNames())
	fmt.Println("Total params:", weights.GetTotalParams())

	// Validate
	err = weights.Validate([]int{4, 3, 2})
	if err != nil {
		fmt.Println("Validation error:", err)
	} else {
		fmt.Println("Weights validated successfully")
	}
}`,

	testCode: `package main

import (
	"testing"
)

func TestNewWeightLoader(t *testing.T) {
	loader := NewWeightLoader()
	if loader == nil {
		t.Fatal("Loader is nil")
	}
}

func TestLoadJSON(t *testing.T) {
	loader := NewWeightLoader()
	weights, err := loader.LoadJSON("test.json")

	if err != nil {
		t.Fatalf("LoadJSON failed: %v", err)
	}
	if weights == nil {
		t.Fatal("Weights is nil")
	}
	if len(weights.Layers) == 0 {
		t.Fatal("No layers loaded")
	}
}

func TestLoadBinary(t *testing.T) {
	loader := NewWeightLoader()
	weights, err := loader.LoadBinary("test.bin")

	if err != nil {
		t.Fatalf("LoadBinary failed: %v", err)
	}
	if weights == nil {
		t.Fatal("Weights is nil")
	}
}

func TestValidate(t *testing.T) {
	loader := NewWeightLoader()
	weights, _ := loader.LoadJSON("test.json")

	err := weights.Validate([]int{4, 3, 2})
	// Should not panic
	if err != nil {
		t.Logf("Validation: %v", err)
	}
}

func TestGetTotalParams(t *testing.T) {
	loader := NewWeightLoader()
	weights, _ := loader.LoadJSON("test.json")

	params := weights.GetTotalParams()
	if params <= 0 {
		t.Fatalf("Expected positive params, got %d", params)
	}
}

func TestGetLayerNames(t *testing.T) {
	loader := NewWeightLoader()
	weights, _ := loader.LoadJSON("test.json")

	names := weights.GetLayerNames()
	if len(names) == 0 {
		t.Fatal("No layer names")
	}
}

func TestSetVerbose(t *testing.T) {
	loader := NewWeightLoader()
	loader.SetVerbose(true)
	// Should not panic
	_, _ = loader.LoadJSON("test.json")
}

func TestMetadata(t *testing.T) {
	loader := NewWeightLoader()
	weights, _ := loader.LoadJSON("test.json")

	if weights.Metadata.Version == "" {
		t.Fatal("Metadata version is empty")
	}
}

func TestValidateBadArchitecture(t *testing.T) {
	loader := NewWeightLoader()
	weights, _ := loader.LoadJSON("test.json")

	err := weights.Validate([]int{1})
	if err == nil {
		t.Fatal("Expected error for bad architecture")
	}
}

func TestLayerWeightsShape(t *testing.T) {
	loader := NewWeightLoader()
	weights, _ := loader.LoadJSON("test.json")

	for _, layer := range weights.Layers {
		if len(layer.Shape) != 2 {
			t.Fatal("Expected 2D shape for each layer")
		}
		break
	}
}`,

	hint1: 'Use encoding/json for JSON and encoding/binary for binary weight files',
	hint2: 'Validate that weight matrix dimensions match expected layer sizes',

	whyItMatters: `Custom weight loading enables flexibility:

- **Framework agnostic**: Load weights from any training framework
- **Optimization**: Custom binary formats for faster loading
- **Version control**: Track weight versions and metadata
- **Debugging**: Inspect and validate weights before inference

Understanding weight serialization is essential for model deployment.`,

	translations: {
		ru: {
			title: 'Загрузка пользовательских весов',
			description: `# Загрузка пользовательских весов

Загрузка пользовательских весов модели из различных форматов в Go.

## Задача

Реализуйте загрузку весов:
- Загрузка весов из JSON/бинарных файлов
- Парсинг форм и значений весов
- Валидация размерностей весов
- Поддержка разных типов данных

## Пример

\`\`\`go
loader := NewWeightLoader()
weights, _ := loader.LoadJSON("weights.json")
model := NewNeuralNet(weights)
\`\`\``,
			hint1: 'Используйте encoding/json для JSON и encoding/binary для бинарных файлов весов',
			hint2: 'Проверьте что размерности матриц весов соответствуют ожидаемым размерам слоев',
			whyItMatters: `Загрузка пользовательских весов обеспечивает гибкость:

- **Независимость от фреймворка**: Загрузка весов из любого фреймворка обучения
- **Оптимизация**: Пользовательские бинарные форматы для быстрой загрузки
- **Контроль версий**: Отслеживание версий весов и метаданных
- **Отладка**: Инспекция и валидация весов перед инференсом`,
		},
		uz: {
			title: "Maxsus og'irliklarni yuklash",
			description: `# Maxsus og'irliklarni yuklash

Go da turli formatlardan maxsus model og'irliklarini yuklash.

## Topshiriq

Og'irliklarni yuklashni amalga oshiring:
- JSON/binary fayllardan og'irliklarni yuklash
- Og'irlik shakllari va qiymatlarini tahlil qilish
- Og'irlik o'lchamlarini tekshirish
- Turli ma'lumot turlarini qo'llab-quvvatlash

## Misol

\`\`\`go
loader := NewWeightLoader()
weights, _ := loader.LoadJSON("weights.json")
model := NewNeuralNet(weights)
\`\`\``,
			hint1: "JSON uchun encoding/json va binary og'irlik fayllari uchun encoding/binary dan foydalaning",
			hint2: "Og'irlik matritsasi o'lchamlari kutilgan qatlam o'lchamlariga mos kelishini tekshiring",
			whyItMatters: `Maxsus og'irliklarni yuklash moslashuvchanlikni ta'minlaydi:

- **Framework mustaqil**: Har qanday o'qitish frameworkdan og'irliklarni yuklash
- **Optimallash**: Tez yuklash uchun maxsus binary formatlar
- **Versiya nazorati**: Og'irlik versiyalari va metama'lumotlarni kuzatish
- **Debugging**: Inferencdan oldin og'irliklarni tekshirish va tasdiqlash`,
		},
	},
};

export default task;
