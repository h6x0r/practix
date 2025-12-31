import { Task } from '../../../../../../../types';

const task: Task = {
  slug: 'goml-image-preprocessing',
  title: 'Image Preprocessing Pipeline',
  difficulty: 'medium',
  tags: ['go', 'ml', 'image', 'preprocessing', 'normalization'],
  estimatedTime: '30m',
  isPremium: false,
  order: 4,

  description: `
## Image Preprocessing Pipeline

Build an image preprocessing pipeline for ML inference that handles resizing, normalization, and format conversion.

### Requirements

1. **ImagePreprocessor** - Main preprocessing component:
   - \`NewImagePreprocessor(config ImageConfig)\` - Create with target dimensions
   - \`Preprocess(img image.Image) ([]float32, error)\` - Convert image to tensor
   - \`PreprocessBatch(images []image.Image) ([][]float32, error)\` - Batch processing

2. **ImageConfig** - Configuration structure:
   - \`Width, Height int\` - Target dimensions
   - \`Channels int\` - Number of channels (1 for grayscale, 3 for RGB)
   - \`Mean []float32\` - Per-channel mean for normalization
   - \`Std []float32\` - Per-channel std for normalization
   - \`ScaleToUnit bool\` - Scale to [0,1] before normalization

3. **Operations**:
   - Resize to target dimensions using bilinear interpolation
   - Convert to float32 tensor
   - Apply normalization: \`(pixel - mean) / std\`
   - Support both HWC and CHW layouts

### Example

\`\`\`go
config := ImageConfig{
    Width:       224,
    Height:      224,
    Channels:    3,
    Mean:        []float32{0.485, 0.456, 0.406},
    Std:         []float32{0.229, 0.224, 0.225},
    ScaleToUnit: true,
}

preprocessor := NewImagePreprocessor(config)
tensor, err := preprocessor.Preprocess(img)
// tensor is []float32 of length 224*224*3 = 150528
\`\`\`
`,

  initialCode: `package imagepreprocess

import (
	"image"
	"image/color"
)

type ImageConfig struct {
	Width       int
	Height      int
	Channels    int
	ScaleToUnit bool
	Layout      string // "HWC" or "CHW"
}

type ImagePreprocessor struct {
}

func NewImagePreprocessor(config ImageConfig) *ImagePreprocessor {
	return nil
}

func (p *ImagePreprocessor) Preprocess(img image.Image) ([]float32, error) {
	return nil, nil
}

func (p *ImagePreprocessor) PreprocessBatch(images []image.Image) ([][]float32, error) {
	return nil, nil
}

func (p *ImagePreprocessor) resize(img image.Image) image.Image {
	return nil
}

func bilinearInterpolation(img image.Image, x, y float64) color.RGBA {
	return color.RGBA{}
}`,

  solutionCode: `package imagepreprocess

import (
	"errors"
	"image"
	"image/color"
	"math"
)

// ImageConfig configures the preprocessing pipeline
type ImageConfig struct {
	Width       int
	Height      int
	Channels    int
	Mean        []float32
	Std         []float32
	ScaleToUnit bool
	Layout      string // "HWC" or "CHW"
}

// ImagePreprocessor handles image preprocessing for ML models
type ImagePreprocessor struct {
	config ImageConfig
}

// NewImagePreprocessor creates a new image preprocessor
func NewImagePreprocessor(config ImageConfig) *ImagePreprocessor {
	if config.Layout == "" {
		config.Layout = "CHW"
	}
	if config.Channels == 0 {
		config.Channels = 3
	}
	return &ImagePreprocessor{config: config}
}

// Preprocess converts an image to a normalized float32 tensor
func (p *ImagePreprocessor) Preprocess(img image.Image) ([]float32, error) {
	if img == nil {
		return nil, errors.New("image is nil")
	}

	// Resize image
	resized := p.resize(img)

	// Convert to tensor
	size := p.config.Width * p.config.Height * p.config.Channels
	tensor := make([]float32, size)

	bounds := resized.Bounds()
	for y := 0; y < p.config.Height; y++ {
		for x := 0; x < p.config.Width; x++ {
			r, g, b, _ := resized.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()

			// Convert to 0-255 range
			channels := []float32{
				float32(r >> 8),
				float32(g >> 8),
				float32(b >> 8),
			}

			// Scale to unit if needed
			if p.config.ScaleToUnit {
				for i := range channels {
					channels[i] /= 255.0
				}
			}

			// Apply normalization
			for c := 0; c < p.config.Channels && c < 3; c++ {
				val := channels[c]
				if len(p.config.Mean) > c {
					val -= p.config.Mean[c]
				}
				if len(p.config.Std) > c && p.config.Std[c] != 0 {
					val /= p.config.Std[c]
				}

				// Store based on layout
				var idx int
				if p.config.Layout == "CHW" {
					idx = c*p.config.Height*p.config.Width + y*p.config.Width + x
				} else {
					idx = y*p.config.Width*p.config.Channels + x*p.config.Channels + c
				}
				tensor[idx] = val
			}
		}
	}

	return tensor, nil
}

// PreprocessBatch processes multiple images
func (p *ImagePreprocessor) PreprocessBatch(images []image.Image) ([][]float32, error) {
	results := make([][]float32, len(images))
	for i, img := range images {
		tensor, err := p.Preprocess(img)
		if err != nil {
			return nil, err
		}
		results[i] = tensor
	}
	return results, nil
}

// resize resizes an image using bilinear interpolation
func (p *ImagePreprocessor) resize(img image.Image) image.Image {
	srcBounds := img.Bounds()
	srcW := srcBounds.Dx()
	srcH := srcBounds.Dy()

	dst := image.NewRGBA(image.Rect(0, 0, p.config.Width, p.config.Height))

	xRatio := float64(srcW) / float64(p.config.Width)
	yRatio := float64(srcH) / float64(p.config.Height)

	for y := 0; y < p.config.Height; y++ {
		for x := 0; x < p.config.Width; x++ {
			srcX := float64(x) * xRatio
			srcY := float64(y) * yRatio
			c := bilinearInterpolation(img, srcX+float64(srcBounds.Min.X), srcY+float64(srcBounds.Min.Y))
			dst.Set(x, y, c)
		}
	}

	return dst
}

// bilinearInterpolation performs bilinear interpolation for a pixel
func bilinearInterpolation(img image.Image, x, y float64) color.RGBA {
	bounds := img.Bounds()

	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := x0 + 1
	y1 := y0 + 1

	// Clamp to bounds
	x0 = clamp(x0, bounds.Min.X, bounds.Max.X-1)
	x1 = clamp(x1, bounds.Min.X, bounds.Max.X-1)
	y0 = clamp(y0, bounds.Min.Y, bounds.Max.Y-1)
	y1 = clamp(y1, bounds.Min.Y, bounds.Max.Y-1)

	// Get four corners
	c00 := img.At(x0, y0)
	c10 := img.At(x1, y0)
	c01 := img.At(x0, y1)
	c11 := img.At(x1, y1)

	// Calculate weights
	xWeight := x - float64(x0)
	yWeight := y - float64(y0)

	// Interpolate
	r00, g00, b00, a00 := c00.RGBA()
	r10, g10, b10, a10 := c10.RGBA()
	r01, g01, b01, a01 := c01.RGBA()
	r11, g11, b11, a11 := c11.RGBA()

	r := bilinearValue(float64(r00), float64(r10), float64(r01), float64(r11), xWeight, yWeight)
	g := bilinearValue(float64(g00), float64(g10), float64(g01), float64(g11), xWeight, yWeight)
	b := bilinearValue(float64(b00), float64(b10), float64(b01), float64(b11), xWeight, yWeight)
	a := bilinearValue(float64(a00), float64(a10), float64(a01), float64(a11), xWeight, yWeight)

	return color.RGBA{
		R: uint8(r >> 8),
		G: uint8(g >> 8),
		B: uint8(b >> 8),
		A: uint8(a >> 8),
	}
}

func bilinearValue(v00, v10, v01, v11, xw, yw float64) uint32 {
	top := v00*(1-xw) + v10*xw
	bottom := v01*(1-xw) + v11*xw
	return uint32(top*(1-yw) + bottom*yw)
}

func clamp(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
`,

  testCode: `package imagepreprocess

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func createTestImage(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{
				R: uint8(x * 255 / w),
				G: uint8(y * 255 / h),
				B: 128,
				A: 255,
			})
		}
	}
	return img
}

func TestNewImagePreprocessor(t *testing.T) {
	config := ImageConfig{
		Width:    224,
		Height:   224,
		Channels: 3,
	}

	p := NewImagePreprocessor(config)
	if p == nil {
		t.Fatal("Expected non-nil preprocessor")
	}
	if p.config.Layout != "CHW" {
		t.Errorf("Expected default layout CHW, got %s", p.config.Layout)
	}
}

func TestPreprocess(t *testing.T) {
	config := ImageConfig{
		Width:       4,
		Height:      4,
		Channels:    3,
		ScaleToUnit: true,
		Layout:      "CHW",
	}

	p := NewImagePreprocessor(config)
	img := createTestImage(8, 8)

	tensor, err := p.Preprocess(img)
	if err != nil {
		t.Fatalf("Preprocess failed: %v", err)
	}

	expectedLen := 4 * 4 * 3
	if len(tensor) != expectedLen {
		t.Errorf("Expected tensor length %d, got %d", expectedLen, len(tensor))
	}

	// Check values are in expected range (0-1 for scaled)
	for i, v := range tensor {
		if v < 0 || v > 1 {
			t.Errorf("Tensor[%d] = %f, expected in [0,1]", i, v)
			break
		}
	}
}

func TestPreprocessWithNormalization(t *testing.T) {
	config := ImageConfig{
		Width:       2,
		Height:      2,
		Channels:    3,
		Mean:        []float32{0.5, 0.5, 0.5},
		Std:         []float32{0.5, 0.5, 0.5},
		ScaleToUnit: true,
		Layout:      "CHW",
	}

	p := NewImagePreprocessor(config)

	// Create uniform gray image
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			img.Set(x, y, color.RGBA{128, 128, 128, 255})
		}
	}

	tensor, err := p.Preprocess(img)
	if err != nil {
		t.Fatalf("Preprocess failed: %v", err)
	}

	// After normalization: (0.5 - 0.5) / 0.5 = 0
	for i, v := range tensor {
		if math.Abs(float64(v)) > 0.1 {
			t.Errorf("Tensor[%d] = %f, expected ~0", i, v)
		}
	}
}

func TestPreprocessBatch(t *testing.T) {
	config := ImageConfig{
		Width:    4,
		Height:   4,
		Channels: 3,
	}

	p := NewImagePreprocessor(config)

	images := []image.Image{
		createTestImage(8, 8),
		createTestImage(16, 16),
		createTestImage(4, 4),
	}

	results, err := p.PreprocessBatch(images)
	if err != nil {
		t.Fatalf("PreprocessBatch failed: %v", err)
	}

	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}

	expectedLen := 4 * 4 * 3
	for i, tensor := range results {
		if len(tensor) != expectedLen {
			t.Errorf("Result[%d] length = %d, expected %d", i, len(tensor), expectedLen)
		}
	}
}

func TestPreprocessNilImage(t *testing.T) {
	config := ImageConfig{Width: 4, Height: 4, Channels: 3}
	p := NewImagePreprocessor(config)

	_, err := p.Preprocess(nil)
	if err == nil {
		t.Error("Expected error for nil image")
	}
}

func TestHWCLayout(t *testing.T) {
	config := ImageConfig{
		Width:    2,
		Height:   2,
		Channels: 3,
		Layout:   "HWC",
	}

	p := NewImagePreprocessor(config)
	img := createTestImage(4, 4)

	tensor, err := p.Preprocess(img)
	if err != nil {
		t.Fatalf("Preprocess failed: %v", err)
	}

	// HWC layout: [H][W][C]
	expectedLen := 2 * 2 * 3
	if len(tensor) != expectedLen {
		t.Errorf("Expected tensor length %d, got %d", expectedLen, len(tensor))
	}
}

func TestGrayscaleConfig(t *testing.T) {
	config := ImageConfig{
		Width:    4,
		Height:   4,
		Channels: 1,
		Layout:   "CHW",
	}

	p := NewImagePreprocessor(config)
	if p.config.Channels != 1 {
		t.Errorf("Expected 1 channel, got %d", p.config.Channels)
	}
}

func TestBilinearInterpolation(t *testing.T) {
	img := createTestImage(10, 10)

	// Test interpolation at integer coordinates
	c := bilinearInterpolation(img, 5.0, 5.0)
	if c.A != 255 {
		t.Errorf("Expected alpha 255, got %d", c.A)
	}

	// Test interpolation at fractional coordinates
	c2 := bilinearInterpolation(img, 5.5, 5.5)
	if c2.A != 255 {
		t.Errorf("Expected alpha 255, got %d", c2.A)
	}
}

func TestResizeUpscale(t *testing.T) {
	config := ImageConfig{
		Width:    16,
		Height:   16,
		Channels: 3,
	}

	p := NewImagePreprocessor(config)
	smallImg := createTestImage(4, 4)

	resized := p.resize(smallImg)
	bounds := resized.Bounds()

	if bounds.Dx() != 16 || bounds.Dy() != 16 {
		t.Errorf("Expected 16x16, got %dx%d", bounds.Dx(), bounds.Dy())
	}
}

func TestEmptyBatch(t *testing.T) {
	config := ImageConfig{
		Width:    4,
		Height:   4,
		Channels: 3,
	}

	p := NewImagePreprocessor(config)
	results, err := p.PreprocessBatch([]image.Image{})

	if err != nil {
		t.Fatalf("Empty batch should not error: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Expected 0 results, got %d", len(results))
	}
}
`,

  hint1: `Use bilinear interpolation for resizing: sample four neighboring pixels and weight them by distance to get smooth results.`,

  hint2: `For CHW layout, index = channel * height * width + y * width + x. For HWC layout, index = y * width * channels + x * channels + channel.`,

  whyItMatters: `Image preprocessing is critical for ML model accuracy. Consistent resizing, normalization, and format conversion ensure that production inference matches training conditions. Using ImageNet normalization values (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) is standard for pretrained vision models.`,

  translations: {
    ru: {
      title: 'Пайплайн Предобработки Изображений',
      description: `
## Пайплайн Предобработки Изображений

Создайте пайплайн предобработки изображений для ML-инференса с изменением размера, нормализацией и преобразованием формата.

### Требования

1. **ImagePreprocessor** - Основной компонент предобработки:
   - \`NewImagePreprocessor(config ImageConfig)\` - Создание с целевыми размерами
   - \`Preprocess(img image.Image) ([]float32, error)\` - Преобразование изображения в тензор
   - \`PreprocessBatch(images []image.Image) ([][]float32, error)\` - Пакетная обработка

2. **ImageConfig** - Структура конфигурации:
   - \`Width, Height int\` - Целевые размеры
   - \`Channels int\` - Количество каналов (1 для grayscale, 3 для RGB)
   - \`Mean []float32\` - Среднее по каналам для нормализации
   - \`Std []float32\` - Стандартное отклонение по каналам
   - \`ScaleToUnit bool\` - Масштабирование в [0,1] перед нормализацией

3. **Операции**:
   - Изменение размера с билинейной интерполяцией
   - Преобразование в float32 тензор
   - Применение нормализации: \`(pixel - mean) / std\`
   - Поддержка HWC и CHW layout

### Пример

\`\`\`go
config := ImageConfig{
    Width:       224,
    Height:      224,
    Channels:    3,
    Mean:        []float32{0.485, 0.456, 0.406},
    Std:         []float32{0.229, 0.224, 0.225},
    ScaleToUnit: true,
}

preprocessor := NewImagePreprocessor(config)
tensor, err := preprocessor.Preprocess(img)
// tensor is []float32 of length 224*224*3 = 150528
\`\`\`
`,
      hint1: 'Используйте билинейную интерполяцию для изменения размера: выборка четырёх соседних пикселей и взвешивание по расстоянию для получения гладких результатов.',
      hint2: 'Для CHW layout: index = channel * height * width + y * width + x. Для HWC layout: index = y * width * channels + x * channels + channel.',
      whyItMatters: 'Предобработка изображений критически важна для точности ML-моделей. Согласованное изменение размера, нормализация и преобразование формата гарантируют соответствие продакшн-инференса условиям обучения. Использование значений нормализации ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) является стандартом для предобученных моделей компьютерного зрения.',
    },
    uz: {
      title: 'Tasvirlarni Oldindan Qayta Ishlash Pipeline',
      description: `
## Tasvirlarni Oldindan Qayta Ishlash Pipeline

ML inference uchun o'lchamini o'zgartirish, normalizatsiya va format konvertatsiyasi bilan tasvirlarni oldindan qayta ishlash pipeline yarating.

### Talablar

1. **ImagePreprocessor** - Asosiy qayta ishlash komponenti:
   - \`NewImagePreprocessor(config ImageConfig)\` - Maqsad o'lchamlari bilan yaratish
   - \`Preprocess(img image.Image) ([]float32, error)\` - Tasvirni tensorga aylantirish
   - \`PreprocessBatch(images []image.Image) ([][]float32, error)\` - Batch qayta ishlash

2. **ImageConfig** - Konfiguratsiya strukturasi:
   - \`Width, Height int\` - Maqsad o'lchamlari
   - \`Channels int\` - Kanallar soni (1 grayscale, 3 RGB uchun)
   - \`Mean []float32\` - Normalizatsiya uchun kanal bo'yicha o'rtacha
   - \`Std []float32\` - Kanal bo'yicha standart og'ish
   - \`ScaleToUnit bool\` - Normalizatsiyadan oldin [0,1] ga masshtablash

3. **Operatsiyalar**:
   - Bilinear interpolyatsiya bilan o'lchamini o'zgartirish
   - float32 tensorga konvertatsiya
   - Normalizatsiyani qo'llash: \`(pixel - mean) / std\`
   - HWC va CHW layoutlarni qo'llab-quvvatlash

### Misol

\`\`\`go
config := ImageConfig{
    Width:       224,
    Height:      224,
    Channels:    3,
    Mean:        []float32{0.485, 0.456, 0.406},
    Std:         []float32{0.229, 0.224, 0.225},
    ScaleToUnit: true,
}

preprocessor := NewImagePreprocessor(config)
tensor, err := preprocessor.Preprocess(img)
// tensor []float32 uzunligi 224*224*3 = 150528
\`\`\`
`,
      hint1: "O'lchamini o'zgartirish uchun bilinear interpolyatsiyadan foydalaning: to'rtta qo'shni pikselni tanlash va masofa bo'yicha tortish orqali silliq natijalar olish.",
      hint2: "CHW layout uchun: index = channel * height * width + y * width + x. HWC layout uchun: index = y * width * channels + x * channels + channel.",
      whyItMatters: "Tasvirlarni oldindan qayta ishlash ML model aniqligi uchun juda muhim. Bir xil o'lcham o'zgartirish, normalizatsiya va format konvertatsiyasi production inference o'qitish shartlariga mos kelishini ta'minlaydi. ImageNet normalizatsiya qiymatlari (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])dan foydalanish oldindan o'qitilgan ko'rish modellari uchun standart hisoblanadi.",
    },
  },
};

export default task;
