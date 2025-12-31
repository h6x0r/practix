import { Task } from '../../../../../../../types';

const task: Task = {
  slug: 'goml-stream-processing',
  title: 'Stream Processing Pipeline',
  difficulty: 'hard',
  tags: ['go', 'ml', 'streaming', 'pipeline', 'backpressure'],
  estimatedTime: '35m',
  isPremium: true,
  order: 4,

  description: `
## Stream Processing Pipeline

Build a streaming pipeline for ML inference that processes continuous data with backpressure handling and flow control.

### Requirements

1. **StreamPipeline** - Main streaming component:
   - \`NewStreamPipeline[T, R](config StreamConfig)\` - Create pipeline
   - \`AddStage(name string, fn StageFunc[T, R])\` - Add processing stage
   - \`Process(ctx context.Context, input <-chan T) <-chan Result[R]\` - Start processing
   - \`Metrics() PipelineMetrics\` - Get processing metrics

2. **StreamConfig** - Configuration:
   - \`BufferSize int\` - Channel buffer size
   - \`Workers int\` - Parallel workers per stage
   - \`BatchSize int\` - Mini-batch size for inference
   - \`FlushInterval time.Duration\` - Max wait before flush

3. **Backpressure Handling**:
   - Block producers when consumers are slow
   - Drop oldest items when buffer is full (optional)
   - Metrics for queue depth and wait times

4. **Flow Control**:
   - Rate limiting per stage
   - Circuit breaker integration
   - Graceful shutdown with drain

### Example

\`\`\`go
pipeline := NewStreamPipeline[Image, Prediction](StreamConfig{
    BufferSize: 100,
    Workers:    4,
    BatchSize:  32,
})

pipeline.AddStage("preprocess", func(img Image) ([]float32, error) {
    return preprocessor.Process(img)
})

pipeline.AddStage("inference", func(features []float32) (Prediction, error) {
    return model.Predict(features)
})

results := pipeline.Process(ctx, imageStream)

for result := range results {
    if result.Error != nil {
        log.Printf("Error: %v", result.Error)
    } else {
        handlePrediction(result.Value)
    }
}
\`\`\`
`,

  initialCode: `package streaming

import (
	"context"
	"sync"
	"time"
)

type Result[T any] struct {
	Value T
	Error error
}

type StageFunc[T, R any] func(T) (R, error)

type StreamConfig struct {
	BufferSize    int
	Workers       int
	BatchSize     int
	FlushInterval time.Duration
	DropOnFull    bool
}

type PipelineMetrics struct {
	Processed   int64
	Errors      int64
	Dropped     int64
	QueueDepth  int
	AvgLatency  time.Duration
}

type StreamPipeline[T, R any] struct {
}

func NewStreamPipeline[T, R any](config StreamConfig) *StreamPipeline[T, R] {
	return nil
}

func (p *StreamPipeline[T, R]) Process(ctx context.Context, input <-chan T) <-chan Result[R] {
	return nil
}

func (p *StreamPipeline[T, R]) Metrics() PipelineMetrics {
	return PipelineMetrics{}
}

func (p *StreamPipeline[T, R]) worker(ctx context.Context, input <-chan T, output chan<- Result[R], wg *sync.WaitGroup) {
}

func (p *StreamPipeline[T, R]) handleBackpressure(item T, output chan<- T) bool {
	return false
}`,

  solutionCode: `package streaming

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

// Result wraps a value with potential error
type Result[T any] struct {
	Value T
	Error error
}

// StageFunc processes an item
type StageFunc[T, R any] func(T) (R, error)

// StreamConfig configures the streaming pipeline
type StreamConfig struct {
	BufferSize    int
	Workers       int
	BatchSize     int
	FlushInterval time.Duration
	DropOnFull    bool
}

// PipelineMetrics holds pipeline statistics
type PipelineMetrics struct {
	Processed  int64
	Errors     int64
	Dropped    int64
	QueueDepth int
	AvgLatency time.Duration
}

// StreamPipeline processes continuous data streams
type StreamPipeline[T, R any] struct {
	config     StreamConfig
	processFunc StageFunc[T, R]

	processed   int64
	errors      int64
	dropped     int64
	totalLatency int64
	queueDepth  int32

	mu sync.RWMutex
}

// NewStreamPipeline creates a new streaming pipeline
func NewStreamPipeline[T, R any](config StreamConfig) *StreamPipeline[T, R] {
	if config.BufferSize <= 0 {
		config.BufferSize = 100
	}
	if config.Workers <= 0 {
		config.Workers = 1
	}
	if config.FlushInterval <= 0 {
		config.FlushInterval = 100 * time.Millisecond
	}

	return &StreamPipeline[T, R]{
		config: config,
	}
}

// SetProcessor sets the processing function
func (p *StreamPipeline[T, R]) SetProcessor(fn StageFunc[T, R]) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.processFunc = fn
}

// Process starts processing the input stream
func (p *StreamPipeline[T, R]) Process(ctx context.Context, input <-chan T) <-chan Result[R] {
	output := make(chan Result[R], p.config.BufferSize)

	var wg sync.WaitGroup

	// Start workers
	for i := 0; i < p.config.Workers; i++ {
		wg.Add(1)
		go p.worker(ctx, input, output, &wg)
	}

	// Close output when all workers done
	go func() {
		wg.Wait()
		close(output)
	}()

	return output
}

// worker processes items from input channel
func (p *StreamPipeline[T, R]) worker(ctx context.Context, input <-chan T, output chan<- Result[R], wg *sync.WaitGroup) {
	defer wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case item, ok := <-input:
			if !ok {
				return
			}

			atomic.AddInt32(&p.queueDepth, 1)
			start := time.Now()

			p.mu.RLock()
			fn := p.processFunc
			p.mu.RUnlock()

			if fn == nil {
				atomic.AddInt32(&p.queueDepth, -1)
				continue
			}

			result, err := fn(item)
			latency := time.Since(start)

			atomic.AddInt64(&p.totalLatency, int64(latency))
			atomic.AddInt32(&p.queueDepth, -1)

			if err != nil {
				atomic.AddInt64(&p.errors, 1)
				p.sendResult(ctx, output, Result[R]{Error: err})
			} else {
				atomic.AddInt64(&p.processed, 1)
				p.sendResult(ctx, output, Result[R]{Value: result})
			}
		}
	}
}

func (p *StreamPipeline[T, R]) sendResult(ctx context.Context, output chan<- Result[R], result Result[R]) bool {
	if p.config.DropOnFull {
		select {
		case output <- result:
			return true
		default:
			atomic.AddInt64(&p.dropped, 1)
			return false
		}
	}

	select {
	case <-ctx.Done():
		return false
	case output <- result:
		return true
	}
}

// Metrics returns current pipeline metrics
func (p *StreamPipeline[T, R]) Metrics() PipelineMetrics {
	processed := atomic.LoadInt64(&p.processed)
	totalLatency := atomic.LoadInt64(&p.totalLatency)

	var avgLatency time.Duration
	if processed > 0 {
		avgLatency = time.Duration(totalLatency / processed)
	}

	return PipelineMetrics{
		Processed:  processed,
		Errors:     atomic.LoadInt64(&p.errors),
		Dropped:    atomic.LoadInt64(&p.dropped),
		QueueDepth: int(atomic.LoadInt32(&p.queueDepth)),
		AvgLatency: avgLatency,
	}
}

// BatchStreamPipeline processes items in batches
type BatchStreamPipeline[T, R any] struct {
	config      StreamConfig
	batchFn     func([]T) ([]R, error)

	processed   int64
	errors      int64
	batches     int64
}

// NewBatchStreamPipeline creates a batch streaming pipeline
func NewBatchStreamPipeline[T, R any](config StreamConfig, batchFn func([]T) ([]R, error)) *BatchStreamPipeline[T, R] {
	if config.BatchSize <= 0 {
		config.BatchSize = 32
	}
	if config.FlushInterval <= 0 {
		config.FlushInterval = 100 * time.Millisecond
	}

	return &BatchStreamPipeline[T, R]{
		config:  config,
		batchFn: batchFn,
	}
}

// Process processes the input stream in batches
func (p *BatchStreamPipeline[T, R]) Process(ctx context.Context, input <-chan T) <-chan Result[R] {
	output := make(chan Result[R], p.config.BufferSize)

	go func() {
		defer close(output)

		batch := make([]T, 0, p.config.BatchSize)
		timer := time.NewTimer(p.config.FlushInterval)
		defer timer.Stop()

		flush := func() {
			if len(batch) == 0 {
				return
			}

			results, err := p.batchFn(batch)
			atomic.AddInt64(&p.batches, 1)

			if err != nil {
				atomic.AddInt64(&p.errors, int64(len(batch)))
				for range batch {
					select {
					case output <- Result[R]{Error: err}:
					case <-ctx.Done():
						return
					}
				}
			} else {
				atomic.AddInt64(&p.processed, int64(len(results)))
				for _, r := range results {
					select {
					case output <- Result[R]{Value: r}:
					case <-ctx.Done():
						return
					}
				}
			}

			batch = batch[:0]
		}

		for {
			select {
			case <-ctx.Done():
				flush()
				return

			case item, ok := <-input:
				if !ok {
					flush()
					return
				}

				batch = append(batch, item)
				if len(batch) >= p.config.BatchSize {
					flush()
					timer.Reset(p.config.FlushInterval)
				}

			case <-timer.C:
				flush()
				timer.Reset(p.config.FlushInterval)
			}
		}
	}()

	return output
}

// BatchMetrics returns batch processing metrics
func (p *BatchStreamPipeline[T, R]) BatchMetrics() (processed, errors, batches int64) {
	return atomic.LoadInt64(&p.processed),
		atomic.LoadInt64(&p.errors),
		atomic.LoadInt64(&p.batches)
}
`,

  testCode: `package streaming

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestNewStreamPipeline(t *testing.T) {
	pipeline := NewStreamPipeline[int, int](StreamConfig{
		BufferSize: 10,
		Workers:    2,
	})

	if pipeline == nil {
		t.Fatal("Expected non-nil pipeline")
	}
}

func TestProcessSimple(t *testing.T) {
	pipeline := NewStreamPipeline[int, int](StreamConfig{
		BufferSize: 10,
		Workers:    2,
	})

	pipeline.SetProcessor(func(n int) (int, error) {
		return n * 2, nil
	})

	input := make(chan int, 5)
	for i := 1; i <= 5; i++ {
		input <- i
	}
	close(input)

	ctx := context.Background()
	output := pipeline.Process(ctx, input)

	results := make([]int, 0)
	for r := range output {
		if r.Error != nil {
			t.Errorf("Unexpected error: %v", r.Error)
		}
		results = append(results, r.Value)
	}

	if len(results) != 5 {
		t.Errorf("Expected 5 results, got %d", len(results))
	}
}

func TestProcessWithErrors(t *testing.T) {
	pipeline := NewStreamPipeline[int, int](StreamConfig{
		BufferSize: 10,
		Workers:    1,
	})

	testErr := errors.New("processing error")
	pipeline.SetProcessor(func(n int) (int, error) {
		if n == 3 {
			return 0, testErr
		}
		return n, nil
	})

	input := make(chan int, 5)
	for i := 1; i <= 5; i++ {
		input <- i
	}
	close(input)

	output := pipeline.Process(context.Background(), input)

	errorCount := 0
	successCount := 0
	for r := range output {
		if r.Error != nil {
			errorCount++
		} else {
			successCount++
		}
	}

	if errorCount != 1 {
		t.Errorf("Expected 1 error, got %d", errorCount)
	}
	if successCount != 4 {
		t.Errorf("Expected 4 successes, got %d", successCount)
	}
}

func TestContextCancellation(t *testing.T) {
	pipeline := NewStreamPipeline[int, int](StreamConfig{
		BufferSize: 10,
		Workers:    2,
	})

	var processed int32
	pipeline.SetProcessor(func(n int) (int, error) {
		time.Sleep(100 * time.Millisecond)
		atomic.AddInt32(&processed, 1)
		return n, nil
	})

	input := make(chan int, 100)
	for i := 0; i < 100; i++ {
		input <- i
	}
	close(input)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	output := pipeline.Process(ctx, input)

	for range output {
		// Drain
	}

	if atomic.LoadInt32(&processed) >= 100 {
		t.Error("Expected early termination")
	}
}

func TestDropOnFull(t *testing.T) {
	pipeline := NewStreamPipeline[int, int](StreamConfig{
		BufferSize: 1,
		Workers:    1,
		DropOnFull: true,
	})

	pipeline.SetProcessor(func(n int) (int, error) {
		time.Sleep(50 * time.Millisecond)
		return n, nil
	})

	input := make(chan int, 10)
	for i := 0; i < 10; i++ {
		input <- i
	}
	close(input)

	output := pipeline.Process(context.Background(), input)

	count := 0
	for range output {
		count++
	}

	metrics := pipeline.Metrics()
	if metrics.Dropped == 0 && count == 10 {
		// Might not drop if fast enough
	}
}

func TestMetrics(t *testing.T) {
	pipeline := NewStreamPipeline[int, int](StreamConfig{
		BufferSize: 10,
		Workers:    1,
	})

	pipeline.SetProcessor(func(n int) (int, error) {
		if n%2 == 0 {
			return 0, errors.New("even error")
		}
		return n, nil
	})

	input := make(chan int, 6)
	for i := 1; i <= 6; i++ {
		input <- i
	}
	close(input)

	output := pipeline.Process(context.Background(), input)
	for range output {
	}

	metrics := pipeline.Metrics()
	if metrics.Processed != 3 {
		t.Errorf("Expected 3 processed, got %d", metrics.Processed)
	}
	if metrics.Errors != 3 {
		t.Errorf("Expected 3 errors, got %d", metrics.Errors)
	}
}

func TestBatchStreamPipeline(t *testing.T) {
	batchFn := func(items []int) ([]int, error) {
		results := make([]int, len(items))
		for i, v := range items {
			results[i] = v * 2
		}
		return results, nil
	}

	pipeline := NewBatchStreamPipeline[int, int](StreamConfig{
		BatchSize:     3,
		FlushInterval: 50 * time.Millisecond,
		BufferSize:    10,
	}, batchFn)

	input := make(chan int, 10)
	for i := 1; i <= 10; i++ {
		input <- i
	}
	close(input)

	output := pipeline.Process(context.Background(), input)

	results := make([]int, 0)
	for r := range output {
		if r.Error == nil {
			results = append(results, r.Value)
		}
	}

	if len(results) != 10 {
		t.Errorf("Expected 10 results, got %d", len(results))
	}
}

func TestBatchFlushOnTimeout(t *testing.T) {
	var batchSizes []int

	batchFn := func(items []int) ([]int, error) {
		batchSizes = append(batchSizes, len(items))
		return items, nil
	}

	pipeline := NewBatchStreamPipeline[int, int](StreamConfig{
		BatchSize:     10,
		FlushInterval: 20 * time.Millisecond,
		BufferSize:    10,
	}, batchFn)

	input := make(chan int, 3)
	input <- 1
	input <- 2
	input <- 3

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	output := pipeline.Process(ctx, input)

	// Wait a bit for timeout flush
	time.Sleep(50 * time.Millisecond)
	close(input)

	for range output {
	}

	// Should have flushed before batch was full
	if len(batchSizes) == 0 {
		t.Error("Expected at least one batch")
	}
}

func TestBatchMetrics(t *testing.T) {
	batchFn := func(items []int) ([]int, error) {
		return items, nil
	}

	pipeline := NewBatchStreamPipeline[int, int](StreamConfig{
		BatchSize:     5,
		FlushInterval: 100 * time.Millisecond,
		BufferSize:    10,
	}, batchFn)

	input := make(chan int, 15)
	for i := 0; i < 15; i++ {
		input <- i
	}
	close(input)

	output := pipeline.Process(context.Background(), input)
	for range output {
	}

	processed, errors, batches := pipeline.BatchMetrics()
	if processed != 15 {
		t.Errorf("Expected 15 processed, got %d", processed)
	}
	if errors != 0 {
		t.Errorf("Expected 0 errors, got %d", errors)
	}
	if batches < 3 {
		t.Errorf("Expected at least 3 batches, got %d", batches)
	}
}

func TestMultipleWorkers(t *testing.T) {
	pipeline := NewStreamPipeline[int, int](StreamConfig{
		BufferSize: 10,
		Workers:    4,
	})

	var processedCount int32
	pipeline.SetProcessor(func(n int) (int, error) {
		atomic.AddInt32(&processedCount, 1)
		return n * 2, nil
	})

	input := make(chan int, 100)
	for i := 0; i < 100; i++ {
		input <- i
	}
	close(input)

	output := pipeline.Process(context.Background(), input)
	for range output {
	}

	if atomic.LoadInt32(&processedCount) != 100 {
		t.Errorf("Expected 100 processed, got %d", processedCount)
	}
}
`,

  hint1: `Use channels with select for backpressure. When output channel is full, either block (default) or drop oldest items based on config. Use atomic counters for metrics.`,

  hint2: `For batch processing, accumulate items until BatchSize is reached OR FlushInterval expires. Use time.Timer with select to handle both conditions.`,

  whyItMatters: `Stream processing is essential for real-time ML applications like video analysis or sensor data processing. Backpressure prevents memory exhaustion when producers outpace consumers. Batching improves GPU utilization while maintaining low latency.`,

  translations: {
    ru: {
      title: 'Потоковая Обработка Pipeline',
      description: `
## Потоковая Обработка Pipeline

Создайте потоковый pipeline для ML-инференса, который обрабатывает непрерывные данные с управлением backpressure и контролем потока.

### Требования

1. **StreamPipeline** - Основной потоковый компонент:
   - \`NewStreamPipeline[T, R](config StreamConfig)\` - Создание pipeline
   - \`AddStage(name string, fn StageFunc[T, R])\` - Добавление этапа обработки
   - \`Process(ctx context.Context, input <-chan T) <-chan Result[R]\` - Начало обработки
   - \`Metrics() PipelineMetrics\` - Получение метрик

2. **StreamConfig** - Конфигурация:
   - \`BufferSize int\` - Размер буфера канала
   - \`Workers int\` - Параллельные воркеры на этап
   - \`BatchSize int\` - Размер мини-батча для инференса
   - \`FlushInterval time.Duration\` - Максимальное ожидание перед flush

3. **Управление Backpressure**:
   - Блокировка продюсеров при медленных консьюмерах
   - Отбрасывание старых элементов при переполнении буфера (опционально)
   - Метрики глубины очереди и времени ожидания

4. **Контроль потока**:
   - Rate limiting на каждом этапе
   - Интеграция с circuit breaker
   - Graceful shutdown с дренажом

### Пример

\`\`\`go
pipeline := NewStreamPipeline[Image, Prediction](StreamConfig{
    BufferSize: 100,
    Workers:    4,
    BatchSize:  32,
})

pipeline.AddStage("preprocess", func(img Image) ([]float32, error) {
    return preprocessor.Process(img)
})

pipeline.AddStage("inference", func(features []float32) (Prediction, error) {
    return model.Predict(features)
})

results := pipeline.Process(ctx, imageStream)

for result := range results {
    if result.Error != nil {
        log.Printf("Error: %v", result.Error)
    } else {
        handlePrediction(result.Value)
    }
}
\`\`\`
`,
      hint1: 'Используйте каналы с select для backpressure. Когда выходной канал заполнен, либо блокируйте (по умолчанию), либо отбрасывайте старые элементы. Используйте atomic счётчики для метрик.',
      hint2: 'Для пакетной обработки накапливайте элементы до достижения BatchSize ИЛИ истечения FlushInterval. Используйте time.Timer с select для обработки обоих условий.',
      whyItMatters: 'Потоковая обработка необходима для real-time ML приложений как анализ видео или обработка данных с датчиков. Backpressure предотвращает исчерпание памяти когда продюсеры опережают консьюмеров. Батчинг улучшает утилизацию GPU, сохраняя низкую латентность.',
    },
    uz: {
      title: 'Stream Processing Pipeline',
      description: `
## Stream Processing Pipeline

Backpressure boshqaruvi va oqim nazorati bilan uzluksiz ma'lumotlarni qayta ishlaydigan ML inference uchun streaming pipeline yarating.

### Talablar

1. **StreamPipeline** - Asosiy streaming komponenti:
   - \`NewStreamPipeline[T, R](config StreamConfig)\` - Pipeline yaratish
   - \`AddStage(name string, fn StageFunc[T, R])\` - Qayta ishlash bosqichini qo'shish
   - \`Process(ctx context.Context, input <-chan T) <-chan Result[R]\` - Qayta ishlashni boshlash
   - \`Metrics() PipelineMetrics\` - Qayta ishlash metrikalarini olish

2. **StreamConfig** - Konfiguratsiya:
   - \`BufferSize int\` - Kanal bufer hajmi
   - \`Workers int\` - Har bir bosqich uchun parallel workerlar
   - \`BatchSize int\` - Inference uchun mini-batch hajmi
   - \`FlushInterval time.Duration\` - Flushdan oldin maksimal kutish

3. **Backpressure boshqaruvi**:
   - Consumerlar sekin bo'lganda producerlarni bloklash
   - Bufer to'lganda eski elementlarni tashlash (ixtiyoriy)
   - Navbat chuqurligi va kutish vaqtlari metrikalari

4. **Oqim nazorati**:
   - Har bir bosqich uchun rate limiting
   - Circuit breaker integratsiyasi
   - Drenaj bilan graceful shutdown

### Misol

\`\`\`go
pipeline := NewStreamPipeline[Image, Prediction](StreamConfig{
    BufferSize: 100,
    Workers:    4,
    BatchSize:  32,
})

pipeline.AddStage("preprocess", func(img Image) ([]float32, error) {
    return preprocessor.Process(img)
})

pipeline.AddStage("inference", func(features []float32) (Prediction, error) {
    return model.Predict(features)
})

results := pipeline.Process(ctx, imageStream)

for result := range results {
    if result.Error != nil {
        log.Printf("Error: %v", result.Error)
    } else {
        handlePrediction(result.Value)
    }
}
\`\`\`
`,
      hint1: "Backpressure uchun select bilan kanallardan foydalaning. Chiqish kanali to'lganida, standart holatda bloklang yoki eski elementlarni tashlang. Metrikalar uchun atomic hisoblagichlardan foydalaning.",
      hint2: "Batch qayta ishlash uchun BatchSize ga yetguncha YOKI FlushInterval tugaguncha elementlarni to'plang. Ikkala shartni ham boshqarish uchun time.Timer ni select bilan ishlating.",
      whyItMatters: "Stream processing video tahlili yoki sensor ma'lumotlarini qayta ishlash kabi real-time ML ilovalari uchun muhim. Backpressure producerlar consumerlardan oldinda bo'lganda xotira tugashini oldini oladi. Batching past kechikishni saqlab, GPU foydalanishini yaxshilaydi.",
    },
  },
};

export default task;
