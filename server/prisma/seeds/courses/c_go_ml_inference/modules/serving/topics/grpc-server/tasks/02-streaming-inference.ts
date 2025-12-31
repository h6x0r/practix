import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-streaming-inference',
	title: 'Streaming Inference',
	difficulty: 'hard',
	tags: ['go', 'ml', 'grpc', 'streaming'],
	estimatedTime: '40m',
	isPremium: true,
	order: 2,
	description: `# Streaming Inference

Implement streaming inference for real-time ML predictions.

## Task

Build streaming inference that:
- Supports server-side streaming
- Supports client-side streaming
- Supports bidirectional streaming
- Handles backpressure properly

## Example

\`\`\`go
// Client streams inputs, server streams predictions
stream, _ := client.StreamPredict(ctx)
for _, input := range inputs {
    stream.Send(&PredictRequest{Input: input})
}
stream.CloseSend()
for {
    resp, err := stream.Recv()
    if err == io.EOF { break }
    // handle prediction
}
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

// StreamProcessor handles streaming inference
type StreamProcessor struct {
	// Your fields here
}

// NewStreamProcessor creates a stream processor
func NewStreamProcessor(model Model) *StreamProcessor {
	// Your code here
	return nil
}

// ProcessServerStream handles server-side streaming
func (p *StreamProcessor) ProcessServerStream(inputs [][]float32, sendFn func([]float32) error) error {
	// Your code here
	return nil
}

// ProcessClientStream handles client-side streaming
func (p *StreamProcessor) ProcessClientStream(recvFn func() ([]float32, error)) ([]float32, error) {
	// Your code here
	return nil, nil
}

// Model interface
type Model interface {
	Predict(input []float32) []float32
}

func main() {
	fmt.Println("Streaming Inference")
}`,

	solutionCode: `package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"sync"
	"time"
)

// Model interface for inference
type Model interface {
	Predict(input []float32) []float32
}

// SimpleModel for testing
type SimpleModel struct{}

func (m *SimpleModel) Predict(input []float32) []float32 {
	output := make([]float32, 2)
	var sum float32
	for _, v := range input {
		sum += v
	}
	output[0] = 1.0 / (1.0 + sum/float32(len(input)+1))
	output[1] = 1.0 - output[0]
	return output
}

// StreamConfig holds streaming configuration
type StreamConfig struct {
	MaxBufferSize   int
	ProcessTimeout  time.Duration
	EnableBatching  bool
	BatchSize       int
	BatchTimeout    time.Duration
}

func DefaultStreamConfig() StreamConfig {
	return StreamConfig{
		MaxBufferSize:  1000,
		ProcessTimeout: 30 * time.Second,
		EnableBatching: true,
		BatchSize:      32,
		BatchTimeout:   10 * time.Millisecond,
	}
}

// StreamProcessor handles streaming inference
type StreamProcessor struct {
	model  Model
	config StreamConfig
}

// NewStreamProcessor creates a stream processor
func NewStreamProcessor(model Model) *StreamProcessor {
	return &StreamProcessor{
		model:  model,
		config: DefaultStreamConfig(),
	}
}

// SetConfig sets the stream configuration
func (p *StreamProcessor) SetConfig(config StreamConfig) {
	p.config = config
}

// ProcessServerStream handles server-side streaming
// Server receives single request, streams multiple responses
func (p *StreamProcessor) ProcessServerStream(inputs [][]float32, sendFn func([]float32) error) error {
	for _, input := range inputs {
		prediction := p.model.Predict(input)
		if err := sendFn(prediction); err != nil {
			return fmt.Errorf("failed to send: %w", err)
		}
	}
	return nil
}

// ProcessClientStream handles client-side streaming
// Client streams multiple inputs, server returns aggregated response
func (p *StreamProcessor) ProcessClientStream(recvFn func() ([]float32, error)) ([]float32, error) {
	var allPredictions [][]float32

	for {
		input, err := recvFn()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to receive: %w", err)
		}

		prediction := p.model.Predict(input)
		allPredictions = append(allPredictions, prediction)
	}

	// Aggregate predictions (average)
	if len(allPredictions) == 0 {
		return nil, nil
	}

	result := make([]float32, len(allPredictions[0]))
	for _, pred := range allPredictions {
		for i, v := range pred {
			result[i] += v
		}
	}
	for i := range result {
		result[i] /= float32(len(allPredictions))
	}

	return result, nil
}

// BidirectionalStream handles bidirectional streaming
type BidirectionalStream struct {
	processor  *StreamProcessor
	inputChan  chan []float32
	outputChan chan []float32
	errorChan  chan error
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
}

// NewBidirectionalStream creates a bidirectional stream
func NewBidirectionalStream(processor *StreamProcessor, ctx context.Context) *BidirectionalStream {
	ctx, cancel := context.WithCancel(ctx)
	bs := &BidirectionalStream{
		processor:  processor,
		inputChan:  make(chan []float32, processor.config.MaxBufferSize),
		outputChan: make(chan []float32, processor.config.MaxBufferSize),
		errorChan:  make(chan error, 1),
		ctx:        ctx,
		cancel:     cancel,
	}

	bs.wg.Add(1)
	go bs.processLoop()

	return bs
}

// Send sends input to the stream
func (bs *BidirectionalStream) Send(input []float32) error {
	select {
	case bs.inputChan <- input:
		return nil
	case <-bs.ctx.Done():
		return bs.ctx.Err()
	default:
		return errors.New("buffer full - backpressure")
	}
}

// Recv receives output from the stream
func (bs *BidirectionalStream) Recv() ([]float32, error) {
	select {
	case output := <-bs.outputChan:
		return output, nil
	case err := <-bs.errorChan:
		return nil, err
	case <-bs.ctx.Done():
		return nil, bs.ctx.Err()
	}
}

// Close closes the stream
func (bs *BidirectionalStream) Close() error {
	close(bs.inputChan)
	bs.wg.Wait()
	bs.cancel()
	return nil
}

// processLoop processes inputs and produces outputs
func (bs *BidirectionalStream) processLoop() {
	defer bs.wg.Done()
	defer close(bs.outputChan)

	if bs.processor.config.EnableBatching {
		bs.processBatched()
	} else {
		bs.processOneByOne()
	}
}

// processOneByOne processes inputs one at a time
func (bs *BidirectionalStream) processOneByOne() {
	for input := range bs.inputChan {
		output := bs.processor.model.Predict(input)
		select {
		case bs.outputChan <- output:
		case <-bs.ctx.Done():
			return
		}
	}
}

// processBatched processes inputs in batches
func (bs *BidirectionalStream) processBatched() {
	batch := make([][]float32, 0, bs.processor.config.BatchSize)
	timer := time.NewTimer(bs.processor.config.BatchTimeout)
	defer timer.Stop()

	flush := func() {
		if len(batch) == 0 {
			return
		}

		for _, input := range batch {
			output := bs.processor.model.Predict(input)
			select {
			case bs.outputChan <- output:
			case <-bs.ctx.Done():
				return
			}
		}
		batch = batch[:0]
	}

	for {
		select {
		case input, ok := <-bs.inputChan:
			if !ok {
				flush()
				return
			}

			batch = append(batch, input)
			if len(batch) >= bs.processor.config.BatchSize {
				flush()
				timer.Reset(bs.processor.config.BatchTimeout)
			}

		case <-timer.C:
			flush()
			timer.Reset(bs.processor.config.BatchTimeout)

		case <-bs.ctx.Done():
			return
		}
	}
}

// BackpressureController manages flow control
type BackpressureController struct {
	maxInFlight int
	inFlight    int
	mu          sync.Mutex
	cond        *sync.Cond
}

func NewBackpressureController(maxInFlight int) *BackpressureController {
	bc := &BackpressureController{
		maxInFlight: maxInFlight,
	}
	bc.cond = sync.NewCond(&bc.mu)
	return bc
}

func (bc *BackpressureController) Acquire() {
	bc.mu.Lock()
	for bc.inFlight >= bc.maxInFlight {
		bc.cond.Wait()
	}
	bc.inFlight++
	bc.mu.Unlock()
}

func (bc *BackpressureController) Release() {
	bc.mu.Lock()
	bc.inFlight--
	bc.cond.Signal()
	bc.mu.Unlock()
}

func main() {
	model := &SimpleModel{}
	processor := NewStreamProcessor(model)

	// Server-side streaming example
	inputs := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}

	fmt.Println("Server-side streaming:")
	err := processor.ProcessServerStream(inputs, func(output []float32) error {
		fmt.Printf("  Prediction: %v\\n", output)
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}

	// Bidirectional streaming example
	fmt.Println("\\nBidirectional streaming:")
	ctx := context.Background()
	stream := NewBidirectionalStream(processor, ctx)

	// Send inputs
	go func() {
		for _, input := range inputs {
			stream.Send(input)
		}
		stream.Close()
	}()

	// Receive outputs
	for {
		output, err := stream.Recv()
		if err != nil {
			break
		}
		fmt.Printf("  Prediction: %v\\n", output)
	}
}`,

	testCode: `package main

import (
	"context"
	"io"
	"testing"
	"time"
)

func TestProcessServerStream(t *testing.T) {
	model := &SimpleModel{}
	processor := NewStreamProcessor(model)

	inputs := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
	}

	var outputs [][]float32
	err := processor.ProcessServerStream(inputs, func(output []float32) error {
		outputs = append(outputs, output)
		return nil
	})

	if err != nil {
		t.Fatalf("ProcessServerStream failed: %v", err)
	}

	if len(outputs) != 2 {
		t.Errorf("Expected 2 outputs, got %d", len(outputs))
	}
}

func TestProcessClientStream(t *testing.T) {
	model := &SimpleModel{}
	processor := NewStreamProcessor(model)

	inputs := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
	}
	idx := 0

	result, err := processor.ProcessClientStream(func() ([]float32, error) {
		if idx >= len(inputs) {
			return nil, io.EOF
		}
		input := inputs[idx]
		idx++
		return input, nil
	})

	if err != nil {
		t.Fatalf("ProcessClientStream failed: %v", err)
	}

	if result == nil {
		t.Error("Result should not be nil")
	}
}

func TestBidirectionalStream(t *testing.T) {
	model := &SimpleModel{}
	processor := NewStreamProcessor(model)

	config := DefaultStreamConfig()
	config.EnableBatching = false
	processor.SetConfig(config)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream := NewBidirectionalStream(processor, ctx)

	// Send inputs
	inputs := [][]float32{{1, 2, 3}, {4, 5, 6}}
	go func() {
		for _, input := range inputs {
			stream.Send(input)
		}
		stream.Close()
	}()

	// Receive outputs
	outputCount := 0
	for {
		_, err := stream.Recv()
		if err != nil {
			break
		}
		outputCount++
	}

	if outputCount != 2 {
		t.Errorf("Expected 2 outputs, got %d", outputCount)
	}
}

func TestBackpressure(t *testing.T) {
	model := &SimpleModel{}
	processor := NewStreamProcessor(model)

	config := DefaultStreamConfig()
	config.MaxBufferSize = 2
	processor.SetConfig(config)

	ctx := context.Background()
	stream := NewBidirectionalStream(processor, ctx)
	defer stream.Close()

	// Fill buffer
	stream.Send([]float32{1})
	stream.Send([]float32{2})

	// Third send should return backpressure error
	err := stream.Send([]float32{3})
	if err == nil {
		t.Error("Expected backpressure error")
	}
}

func TestBackpressureController(t *testing.T) {
	bc := NewBackpressureController(2)

	bc.Acquire()
	bc.Acquire()

	done := make(chan struct{})
	go func() {
		bc.Acquire() // Should block
		close(done)
	}()

	select {
	case <-done:
		t.Error("Acquire should block when at limit")
	case <-time.After(50 * time.Millisecond):
		// Expected
	}

	bc.Release()

	select {
	case <-done:
		// Expected
	case <-time.After(100 * time.Millisecond):
		t.Error("Acquire should unblock after release")
	}
}

func TestSimpleModel(t *testing.T) {
	model := &SimpleModel{}
	input := []float32{1, 2, 3}

	output := model.Predict(input)

	if len(output) != 2 {
		t.Errorf("Expected 2 outputs, got %d", len(output))
	}
	if output[0]+output[1] < 0.99 || output[0]+output[1] > 1.01 {
		t.Error("Outputs should sum to ~1")
	}
}

func TestDefaultStreamConfig(t *testing.T) {
	config := DefaultStreamConfig()

	if config.MaxBufferSize <= 0 {
		t.Error("MaxBufferSize should be positive")
	}
	if config.BatchSize <= 0 {
		t.Error("BatchSize should be positive")
	}
}

func TestSetConfig(t *testing.T) {
	model := &SimpleModel{}
	processor := NewStreamProcessor(model)

	config := StreamConfig{
		MaxBufferSize: 500,
		BatchSize:     64,
	}
	processor.SetConfig(config)

	if processor.config.MaxBufferSize != 500 {
		t.Error("Config not set correctly")
	}
}

func TestProcessClientStreamEmpty(t *testing.T) {
	model := &SimpleModel{}
	processor := NewStreamProcessor(model)

	result, err := processor.ProcessClientStream(func() ([]float32, error) {
		return nil, io.EOF
	})

	if err != nil {
		t.Fatalf("Should not error on empty stream: %v", err)
	}
	if result != nil {
		t.Error("Empty stream should return nil result")
	}
}

func TestBidirectionalStreamBatching(t *testing.T) {
	model := &SimpleModel{}
	processor := NewStreamProcessor(model)

	config := DefaultStreamConfig()
	config.EnableBatching = true
	config.BatchSize = 2
	config.BatchTimeout = 100 * time.Millisecond
	processor.SetConfig(config)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream := NewBidirectionalStream(processor, ctx)

	go func() {
		stream.Send([]float32{1})
		stream.Send([]float32{2})
		stream.Send([]float32{3})
		stream.Close()
	}()

	outputCount := 0
	for {
		_, err := stream.Recv()
		if err != nil {
			break
		}
		outputCount++
	}

	if outputCount != 3 {
		t.Errorf("Expected 3 outputs, got %d", outputCount)
	}
}`,

	hint1: 'Use channels for non-blocking communication in streaming',
	hint2: 'Implement batching to improve throughput in streaming scenarios',

	whyItMatters: `Streaming inference enables real-time ML applications:

- **Real-time processing**: Process data as it arrives
- **Memory efficiency**: No need to buffer entire dataset
- **Responsiveness**: Return predictions as soon as available
- **Backpressure**: Handle varying client speeds gracefully

Streaming is essential for video, audio, and time-series ML.`,

	translations: {
		ru: {
			title: 'Потоковый инференс',
			description: `# Потоковый инференс

Реализуйте потоковый инференс для real-time ML предсказаний.

## Задача

Создайте потоковый инференс:
- Поддержка серверного стриминга
- Поддержка клиентского стриминга
- Поддержка двунаправленного стриминга
- Правильная обработка backpressure

## Пример

\`\`\`go
// Client streams inputs, server streams predictions
stream, _ := client.StreamPredict(ctx)
for _, input := range inputs {
    stream.Send(&PredictRequest{Input: input})
}
stream.CloseSend()
for {
    resp, err := stream.Recv()
    if err == io.EOF { break }
    // handle prediction
}
\`\`\``,
			hint1: 'Используйте каналы для неблокирующей коммуникации в стриминге',
			hint2: 'Реализуйте батчинг для улучшения пропускной способности в потоковых сценариях',
			whyItMatters: `Потоковый инференс обеспечивает real-time ML приложения:

- **Real-time обработка**: Обработка данных по мере поступления
- **Эффективность памяти**: Не нужно буферизовать весь датасет
- **Отзывчивость**: Возврат предсказаний как только они готовы
- **Backpressure**: Изящная обработка разной скорости клиентов`,
		},
		uz: {
			title: 'Oqimli inference',
			description: `# Oqimli inference

Real-time ML bashoratlari uchun oqimli inference ni amalga oshiring.

## Topshiriq

Oqimli inference yarating:
- Server-side streaming ni qo'llab-quvvatlash
- Client-side streaming ni qo'llab-quvvatlash
- Bidirectional streaming ni qo'llab-quvvatlash
- Backpressure ni to'g'ri qayta ishlash

## Misol

\`\`\`go
// Client streams inputs, server streams predictions
stream, _ := client.StreamPredict(ctx)
for _, input := range inputs {
    stream.Send(&PredictRequest{Input: input})
}
stream.CloseSend()
for {
    resp, err := stream.Recv()
    if err == io.EOF { break }
    // handle prediction
}
\`\`\``,
			hint1: "Streaming da bloklanmaydigan aloqa uchun kanallardan foydalaning",
			hint2: "Streaming stsenariylarda throughput ni yaxshilash uchun batching ni amalga oshiring",
			whyItMatters: `Oqimli inference real-time ML ilovalarini ta'minlaydi:

- **Real-time qayta ishlash**: Ma'lumotlarni kelganda qayta ishlash
- **Xotira samaradorligi**: Butun datasetni buferlashtirish kerak emas
- **Javob berish**: Bashoratlarni tayyor bo'lganda qaytarish
- **Backpressure**: Turli mijoz tezliklarini yumshoq boshqarish`,
		},
	},
};

export default task;
