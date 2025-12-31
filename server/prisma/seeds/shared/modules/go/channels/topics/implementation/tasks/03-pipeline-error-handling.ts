import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-channels-pipeline-error',
	title: 'Pipeline with Error Handling',
	difficulty: 'medium',
	tags: ['go', 'channels', 'concurrency', 'pipeline', 'error-handling'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a multi-stage pipeline that chains channel processing stages with proper error handling and propagation.

**Requirements:**
1. **Pipeline**: Chain multiple processing stages together
2. **Error Channel**: Return separate error channel for error propagation
3. **Stage Function**: Each stage transforms input to output
4. **Context Awareness**: Stop all stages on context cancellation or first error

**Pipeline Pattern:**
\`\`\`go
type Stage[In, Out any] func(context.Context, <-chan In) (<-chan Out, <-chan error)

func Pipeline[T any](
    ctx context.Context,
    in <-chan T,
    stages ...Stage[T, T],
) (<-chan T, <-chan error) {
    // Chain stages together
    // Merge all error channels
    // Return final output and unified errors
}
\`\`\`

**Key Concepts:**
- Pipeline chains stages where output of one becomes input of next
- Each stage runs concurrently in its own goroutine
- Errors from any stage propagate through error channel
- Context cancellation stops entire pipeline
- Non-blocking error collection prevents deadlocks

**Example Usage:**
\`\`\`go
// Data processing pipeline
func ProcessData(input <-chan string) (<-chan string, <-chan error) {
    ctx := context.Background()

    // Stage 1: Validate
    validate := func(ctx context.Context, in <-chan string) (<-chan string, <-chan error) {
        out := make(chan string)
        errs := make(chan error)
        go func() {
            defer close(out)
            defer close(errs)
            for v := range in {
                if v == "" {
                    errs <- errors.New("empty value")
                    continue
                }
                select {
                case <-ctx.Done():
                    return
                case out <- v:
                }
            }
        }()
        return out, errs
    }

    // Stage 2: Transform
    transform := func(ctx context.Context, in <-chan string) (<-chan string, <-chan error) {
        out := make(chan string)
        errs := make(chan error)
        go func() {
            defer close(out)
            defer close(errs)
            for v := range in {
                transformed := strings.ToUpper(v)
                select {
                case <-ctx.Done():
                    return
                case out <- transformed:
                }
            }
        }()
        return out, errs
    }

    // Stage 3: Enrich
    enrich := func(ctx context.Context, in <-chan string) (<-chan string, <-chan error) {
        out := make(chan string)
        errs := make(chan error)
        go func() {
            defer close(out)
            defer close(errs)
            for v := range in {
                enriched := fmt.Sprintf("PROCESSED: %s", v)
                select {
                case <-ctx.Done():
                    return
                case out <- enriched:
                }
            }
        }()
        return out, errs
    }

    // Chain all stages
    return Pipeline(ctx, input, validate, transform, enrich)
}

// Usage
input := make(chan string)
go func() {
    input <- "hello"
    input <- "world"
    close(input)
}()

output, errs := ProcessData(input)

// Process results and errors
go func() {
    for err := range errs {
        log.Printf("Pipeline error: %v", err)
    }
}()

for result := range output {
    fmt.Println(result) // "PROCESSED: HELLO", "PROCESSED: WORLD"
}
\`\`\`

**Pattern Flow:**
\`\`\`
Input → Stage1 → Stage2 → Stage3 → Output
         ↓        ↓        ↓
         Error    Error    Error
                  ↓
         Merged Error Channel
\`\`\`

**Real-World Scenarios:**

**1. Data ETL Pipeline:**
\`\`\`go
// Extract → Transform → Load
func ETLPipeline(records <-chan RawRecord) (<-chan ProcessedRecord, <-chan error) {
    ctx := context.Background()

    extract := func(ctx context.Context, in <-chan RawRecord) (<-chan RawRecord, <-chan error) {
        // Validate and parse raw records
    }

    transform := func(ctx context.Context, in <-chan RawRecord) (<-chan ProcessedRecord, <-chan error) {
        // Clean, normalize, enrich data
    }

    load := func(ctx context.Context, in <-chan ProcessedRecord) (<-chan ProcessedRecord, <-chan error) {
        // Write to database, handle duplicates
    }

    return Pipeline(ctx, records, extract, transform, load)
}
\`\`\`

**2. Image Processing:**
\`\`\`go
func ImagePipeline(images <-chan Image) (<-chan Image, <-chan error) {
    ctx := context.Background()

    resize := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        // Resize to standard dimensions
    }

    compress := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        // Apply compression
    }

    watermark := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        // Add watermark
    }

    return Pipeline(ctx, images, resize, compress, watermark)
}
\`\`\`

**Constraints:**
- Must handle variable number of stages
- Must merge all error channels without blocking
- Must propagate context cancellation to all stages
- Must close all channels properly to prevent leaks
- Must handle nil context (use Background)`,
	initialCode: `package channelsx

import (
	"context"
)

type Stage[In, Out any] func(context.Context, <-chan In) (<-chan Out, <-chan error)

// TODO: Implement Pipeline
// Chain stages: output of stage N becomes input of stage N+1
// Collect errors from all stages into single error channel
// Stop all stages on context cancellation
func Pipeline[T any](ctx context.Context, in <-chan T, stages ...Stage[T, T]) (<-chan T, <-chan error) {
	// TODO: Implement
}`,
	solutionCode: `package channelsx

import (
	"context"
	"sync"
)

type Stage[In, Out any] func(context.Context, <-chan In) (<-chan Out, <-chan error)

func Pipeline[T any](ctx context.Context, in <-chan T, stages ...Stage[T, T]) (<-chan T, <-chan error) {
	if ctx == nil {
		ctx = context.Background()
	}

	// collect error channels from all stages
	var errorChannels []<-chan error

	// chain stages: output of stage N becomes input of stage N+1
	current := in
	for _, stage := range stages {
		out, errs := stage(ctx, current)    // execute stage
		errorChannels = append(errorChannels, errs)
		current = out                        // output becomes input for next stage
	}

	// merge all error channels into one
	mergedErrors := mergeErrors(ctx, errorChannels...)

	return current, mergedErrors            // return final output and merged errors
}

// mergeErrors combines multiple error channels into single channel
func mergeErrors(ctx context.Context, errChannels ...<-chan error) <-chan error {
	out := make(chan error)

	var wg sync.WaitGroup

	forward := func(errCh <-chan error) {
		defer wg.Done()
		for {
			select {
			case <-ctx.Done():              // context canceled
				return
			case err, ok := <-errCh:
				if !ok {                     // error channel closed
					return
				}
				select {
				case <-ctx.Done():           // check again before sending
					return
				case out <- err:             // forward error to output
				}
			}
		}
	}

	// launch goroutine for each error channel
	for _, errCh := range errChannels {
		wg.Add(1)
		go forward(errCh)
	}

	// close output when all error channels are done
	go func() {
		wg.Wait()
		close(out)
	}()

	return out
}`,
		testCode: `package channelsx

import (
	"context"
	"errors"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Pipeline with no stages returns input
	ctx := context.Background()
	input := make(chan int, 1)
	input <- 42
	close(input)

	output, errs := Pipeline(ctx, input)

	result := <-output
	if result != 42 {
		t.Errorf("expected 42, got %d", result)
	}

	for range errs {
	}
}

func Test2(t *testing.T) {
	// Pipeline with one stage processes values
	ctx := context.Background()
	input := make(chan int, 2)
	input <- 1
	input <- 2
	close(input)

	double := func(ctx context.Context, in <-chan int) (<-chan int, <-chan error) {
		out := make(chan int)
		errs := make(chan error)
		go func() {
			defer close(out)
			defer close(errs)
			for v := range in {
				out <- v * 2
			}
		}()
		return out, errs
	}

	output, errs := Pipeline(ctx, input, double)

	var results []int
	for v := range output {
		results = append(results, v)
	}
	for range errs {
	}

	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
}

func Test3(t *testing.T) {
	// Pipeline propagates errors from stage
	ctx := context.Background()
	input := make(chan int, 1)
	input <- 1
	close(input)

	errorStage := func(ctx context.Context, in <-chan int) (<-chan int, <-chan error) {
		out := make(chan int)
		errs := make(chan error, 1)
		go func() {
			defer close(out)
			defer close(errs)
			for range in {
				errs <- errors.New("stage error")
			}
		}()
		return out, errs
	}

	output, errs := Pipeline(ctx, input, errorStage)

	for range output {
	}

	var errList []error
	for err := range errs {
		errList = append(errList, err)
	}

	if len(errList) != 1 {
		t.Errorf("expected 1 error, got %d", len(errList))
	}
}

func Test4(t *testing.T) {
	// Pipeline handles nil context
	input := make(chan int)
	close(input)

	output, errs := Pipeline(nil, input)

	for range output {
	}
	for range errs {
	}
}

func Test5(t *testing.T) {
	// Pipeline stops on context cancellation
	ctx, cancel := context.WithCancel(context.Background())

	input := make(chan int)

	slowStage := func(ctx context.Context, in <-chan int) (<-chan int, <-chan error) {
		out := make(chan int)
		errs := make(chan error)
		go func() {
			defer close(out)
			defer close(errs)
			for {
				select {
				case <-ctx.Done():
					return
				case v, ok := <-in:
					if !ok {
						return
					}
					out <- v
				}
			}
		}()
		return out, errs
	}

	output, errs := Pipeline(ctx, input, slowStage)

	cancel()

	go func() {
		for range output {
		}
	}()

	for range errs {
	}
}

func Test6(t *testing.T) {
	// Pipeline chains multiple stages
	ctx := context.Background()
	input := make(chan int, 1)
	input <- 5
	close(input)

	addOne := func(ctx context.Context, in <-chan int) (<-chan int, <-chan error) {
		out := make(chan int)
		errs := make(chan error)
		go func() {
			defer close(out)
			defer close(errs)
			for v := range in {
				out <- v + 1
			}
		}()
		return out, errs
	}

	output, errs := Pipeline(ctx, input, addOne, addOne, addOne)

	result := <-output
	if result != 8 {
		t.Errorf("expected 8, got %d", result)
	}

	for range errs {
	}
}

func Test7(t *testing.T) {
	// Pipeline merges errors from multiple stages
	ctx := context.Background()
	input := make(chan int, 1)
	input <- 1
	close(input)

	errorStage := func(ctx context.Context, in <-chan int) (<-chan int, <-chan error) {
		out := make(chan int)
		errs := make(chan error, 1)
		go func() {
			defer close(out)
			defer close(errs)
			for v := range in {
				errs <- errors.New("error")
				out <- v
			}
		}()
		return out, errs
	}

	output, errs := Pipeline(ctx, input, errorStage, errorStage)

	for range output {
	}

	errCount := 0
	for range errs {
		errCount++
	}

	if errCount != 2 {
		t.Errorf("expected 2 errors, got %d", errCount)
	}
}

func Test8(t *testing.T) {
	// Pipeline closes output when input closes
	ctx := context.Background()
	input := make(chan int)
	close(input)

	passThrough := func(ctx context.Context, in <-chan int) (<-chan int, <-chan error) {
		out := make(chan int)
		errs := make(chan error)
		go func() {
			defer close(out)
			defer close(errs)
			for v := range in {
				out <- v
			}
		}()
		return out, errs
	}

	output, errs := Pipeline(ctx, input, passThrough)

	select {
	case _, ok := <-output:
		if ok {
			t.Error("expected closed channel")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("channel not closed in time")
	}

	for range errs {
	}
}

func Test9(t *testing.T) {
	// Pipeline processes many items
	ctx := context.Background()
	input := make(chan int, 100)
	for i := 0; i < 100; i++ {
		input <- i
	}
	close(input)

	passThrough := func(ctx context.Context, in <-chan int) (<-chan int, <-chan error) {
		out := make(chan int)
		errs := make(chan error)
		go func() {
			defer close(out)
			defer close(errs)
			for v := range in {
				out <- v
			}
		}()
		return out, errs
	}

	output, errs := Pipeline(ctx, input, passThrough)

	count := 0
	for range output {
		count++
	}

	if count != 100 {
		t.Errorf("expected 100, got %d", count)
	}

	for range errs {
	}
}

func Test10(t *testing.T) {
	// Pipeline returns both channels
	ctx := context.Background()
	input := make(chan int)
	close(input)

	output, errs := Pipeline(ctx, input)

	if output == nil {
		t.Error("expected non-nil output channel")
	}
	if errs == nil {
		t.Error("expected non-nil error channel")
	}

	for range output {
	}
	for range errs {
	}
}
`,
	hint1: `Chain stages by passing the output channel of one stage as the input channel to the next stage.`,
	hint2: `Collect all error channels from stages and use a fan-in pattern to merge them into a single error channel.`,
	whyItMatters: `Pipeline pattern with error handling is critical for building robust data processing systems in production Go.

**Why This Matters:**

**1. Composable Data Processing**
Break complex processing into simple, testable stages:
\`\`\`go
// Before: Monolithic processing (hard to test, modify)
func ProcessRecord(rec Record) (Result, error) {
    // 500 lines of validation, transformation, enrichment, storage
    // One bug breaks everything
    // Can't reuse components
}

// After: Pipeline with stages (modular, testable)
func ProcessRecord(rec Record) (Result, error) {
    input := make(chan Record, 1)
    input <- rec
    close(input)

    ctx := context.Background()
    output, errs := Pipeline(ctx, input,
        validateStage,     // 50 lines, tested independently
        transformStage,    // 50 lines, tested independently
        enrichStage,       // 50 lines, tested independently
        storeStage,        // 50 lines, tested independently
    )

    // Each stage is simple, reusable, testable
}
\`\`\`

**2. Real Production: Log Processing Pipeline**
Processing 10M logs/day at a fintech company:
\`\`\`go
type LogEntry struct {
    Timestamp time.Time
    Level     string
    Service   string
    Message   string
}

func LogPipeline(logs <-chan LogEntry) (<-chan LogEntry, <-chan error) {
    ctx := context.Background()

    // Stage 1: Parse and validate
    parse := func(ctx context.Context, in <-chan LogEntry) (<-chan LogEntry, <-chan error) {
        out := make(chan LogEntry)
        errs := make(chan error, 100)  // buffered for non-blocking errors

        go func() {
            defer close(out)
            defer close(errs)

            for log := range in {
                // Validate timestamp
                if log.Timestamp.IsZero() {
                    errs <- fmt.Errorf("invalid timestamp for service %s", log.Service)
                    continue
                }

                // Normalize level
                log.Level = strings.ToUpper(log.Level)

                select {
                case <-ctx.Done():
                    return
                case out <- log:
                }
            }
        }()
        return out, errs
    }

    // Stage 2: Enrich with metadata
    enrich := func(ctx context.Context, in <-chan LogEntry) (<-chan LogEntry, <-chan error) {
        out := make(chan LogEntry)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for log := range in {
                // Add service metadata from cache
                metadata, err := getServiceMetadata(log.Service)
                if err != nil {
                    errs <- fmt.Errorf("failed to enrich service %s: %w", log.Service, err)
                    continue
                }

                log.Message = fmt.Sprintf("[%s:%s] %s", metadata.Region, metadata.Version, log.Message)

                select {
                case <-ctx.Done():
                    return
                case out <- log:
                }
            }
        }()
        return out, errs
    }

    // Stage 3: Filter and alert
    alert := func(ctx context.Context, in <-chan LogEntry) (<-chan LogEntry, <-chan error) {
        out := make(chan LogEntry)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for log := range in {
                // Alert on critical errors
                if log.Level == "ERROR" || log.Level == "FATAL" {
                    if err := sendAlert(log); err != nil {
                        errs <- fmt.Errorf("failed to send alert: %w", err)
                    }
                }

                select {
                case <-ctx.Done():
                    return
                case out <- log:
                }
            }
        }()
        return out, errs
    }

    return Pipeline(ctx, logs, parse, enrich, alert)
}

// Processing results
output, errors := LogPipeline(logStream)

// Error monitoring goroutine
go func() {
    errorCount := 0
    for err := range errors {
        errorCount++
        log.Printf("Pipeline error: %v", err)

        // Alert if error rate is high
        if errorCount > 100 {
            alertOps("High error rate in log pipeline")
            errorCount = 0
        }
    }
}()

// Processed logs
for log := range output {
    indexToElasticsearch(log)
}

// Results:
// - Before: Monolithic parser crashed on bad data (45 min downtime)
// - After: Errors isolated per stage, invalid logs skipped
// - Availability: 99.9% → 99.99%
// - Cost savings: $50K/year (reduced incident response time)
\`\`\`

**3. E-commerce Order Processing**
Processing 50K orders/day:
\`\`\`go
func OrderPipeline(orders <-chan Order) (<-chan Order, <-chan error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
    defer cancel()

    // Stage 1: Validate payment
    validatePayment := func(ctx context.Context, in <-chan Order) (<-chan Order, <-chan error) {
        out := make(chan Order)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for order := range in {
                if err := verifyPayment(order.PaymentID); err != nil {
                    errs <- fmt.Errorf("payment validation failed for order %s: %w", order.ID, err)
                    continue
                }

                select {
                case <-ctx.Done():
                    return
                case out <- order:
                }
            }
        }()
        return out, errs
    }

    // Stage 2: Check inventory
    checkInventory := func(ctx context.Context, in <-chan Order) (<-chan Order, <-chan error) {
        out := make(chan Order)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for order := range in {
                available, err := checkStock(order.Items)
                if err != nil {
                    errs <- fmt.Errorf("inventory check failed for order %s: %w", order.ID, err)
                    continue
                }
                if !available {
                    errs <- fmt.Errorf("insufficient stock for order %s", order.ID)
                    refundPayment(order.PaymentID)
                    continue
                }

                select {
                case <-ctx.Done():
                    return
                case out <- order:
                }
            }
        }()
        return out, errs
    }

    // Stage 3: Create shipment
    createShipment := func(ctx context.Context, in <-chan Order) (<-chan Order, <-chan error) {
        out := make(chan Order)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for order := range in {
                shipmentID, err := createShipping(order)
                if err != nil {
                    errs <- fmt.Errorf("shipment creation failed for order %s: %w", order.ID, err)
                    continue
                }

                order.ShipmentID = shipmentID

                select {
                case <-ctx.Done():
                    return
                case out <- order:
                }
            }
        }()
        return out, errs
    }

    return Pipeline(ctx, orders, validatePayment, checkInventory, createShipment)
}

// Impact:
// - Before: Single failure in payment validation blocked all orders
// - After: Errors isolated, valid orders continue processing
// - Order processing rate: 80% → 98%
// - Revenue recovered: $500K/year from reduced processing failures
\`\`\`

**4. Image Processing Service**
Processing 1M images/month:
\`\`\`go
func ImageProcessingPipeline(images <-chan Image) (<-chan Image, <-chan error) {
    ctx := context.Background()

    // Stage 1: Resize
    resize := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        out := make(chan Image)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for img := range in {
                resized, err := resizeImage(img, 1920, 1080)
                if err != nil {
                    errs <- fmt.Errorf("resize failed for image %s: %w", img.ID, err)
                    continue
                }

                select {
                case <-ctx.Done():
                    return
                case out <- resized:
                }
            }
        }()
        return out, errs
    }

    // Stage 2: Compress
    compress := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        out := make(chan Image)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for img := range in {
                compressed, err := compressImage(img, 80)
                if err != nil {
                    errs <- fmt.Errorf("compress failed for image %s: %w", img.ID, err)
                    continue
                }

                select {
                case <-ctx.Done():
                    return
                case out <- compressed:
                }
            }
        }()
        return out, errs
    }

    // Stage 3: Upload to CDN
    upload := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        out := make(chan Image)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for img := range in {
                url, err := uploadToCDN(img)
                if err != nil {
                    errs <- fmt.Errorf("upload failed for image %s: %w", img.ID, err)
                    continue
                }

                img.URL = url

                select {
                case <-ctx.Done():
                    return
                case out <- img:
                }
            }
        }()
        return out, errs
    }

    return Pipeline(ctx, images, resize, compress, upload)
}

// Results:
// - Processing time: 200ms → 50ms per image (concurrent stages)
// - Error visibility: Real-time monitoring of each stage
// - Retry logic: Can retry individual stages without reprocessing
\`\`\`

**5. Why Separate Error Channel is Critical**
\`\`\`go
// WRONG - Mixing data and errors in same channel
type Result struct {
    Data  string
    Error error
}

func BrokenPipeline(in <-chan string) <-chan Result {
    // Complicated to handle, can't fan-in easily
    // Consumer must check every result for error
}

// RIGHT - Separate channels
func CorrectPipeline(in <-chan string) (<-chan string, <-chan error) {
    // Clean separation of concerns
    // Can monitor errors independently
    // Multiple consumers can read same data channel
}

// Usage
output, errs := CorrectPipeline(input)

// Dedicated error monitoring
go func() {
    for err := range errs {
        metrics.IncrementCounter("pipeline_errors")
        logger.Error(err)
    }
}()

// Clean data processing
for data := range output {
    process(data)  // No error checking needed
}
\`\`\`

**6. Testing Pipelines**
\`\`\`go
func TestPipeline(t *testing.T) {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // Create test stages
    stage1 := func(ctx context.Context, in <-chan int) (<-chan int, <-chan error) {
        out := make(chan int)
        errs := make(chan error)
        go func() {
            defer close(out)
            defer close(errs)
            for v := range in {
                if v < 0 {
                    errs <- fmt.Errorf("negative value: %d", v)
                    continue
                }
                out <- v * 2
            }
        }()
        return out, errs
    }

    // Test input
    input := make(chan int, 3)
    input <- 1
    input <- -1  // Should error
    input <- 2
    close(input)

    // Run pipeline
    output, errors := Pipeline(ctx, input, stage1)

    // Collect results
    var results []int
    for v := range output {
        results = append(results, v)
    }

    // Collect errors
    var errs []error
    for err := range errors {
        errs = append(errs, err)
    }

    // Verify
    if len(results) != 2 {
        t.Errorf("expected 2 results, got %d", len(results))
    }
    if len(errs) != 1 {
        t.Errorf("expected 1 error, got %d", len(errs))
    }
}
\`\`\`

**Production Best Practices:**
1. Use buffered error channels to prevent blocking stages
2. Monitor error rates for each stage independently
3. Add context timeouts to prevent infinite pipeline runs
4. Test each stage in isolation before integration
5. Use metrics to track throughput of each stage
6. Implement retry logic for transient errors
7. Add circuit breakers for external dependencies

**Real-World Impact:**
Data processing company:
- **Before**: Monolithic ETL pipeline (30 min to process 1M records)
- **After**: 5-stage pipeline (8 min to process 1M records, 3.75x faster)
- **Error handling**: Before: One bad record crashed entire job → After: Bad records skipped, logged separately
- **Monitoring**: Real-time visibility into each stage's performance
- **Result**: Saved 22 min per run × 24 runs/day = 8.8 hours/day saved
- **Cost**: Reduced compute costs by $10K/month

Pipeline pattern transforms complex processing into simple, composable, testable components. Master it for building robust production systems.`,
	order: 2,
	translations: {
		ru: {
			title: 'Обработка ошибок в pipeline',
			solutionCode: `package channelsx

import (
	"context"
	"sync"
)

type Stage[In, Out any] func(context.Context, <-chan In) (<-chan Out, <-chan error)

func Pipeline[T any](ctx context.Context, in <-chan T, stages ...Stage[T, T]) (<-chan T, <-chan error) {
	if ctx == nil {
		ctx = context.Background()
	}

	// собираем каналы ошибок от всех стадий
	var errorChannels []<-chan error

	// цепляем стадии: выход стадии N становится входом стадии N+1
	current := in
	for _, stage := range stages {
		out, errs := stage(ctx, current)    // выполняем стадию
		errorChannels = append(errorChannels, errs)
		current = out                        // выход становится входом для следующей стадии
	}

	// объединяем все каналы ошибок в один
	mergedErrors := mergeErrors(ctx, errorChannels...)

	return current, mergedErrors            // возвращаем финальный выход и объединённые ошибки
}

// mergeErrors объединяет несколько каналов ошибок в один канал
func mergeErrors(ctx context.Context, errChannels ...<-chan error) <-chan error {
	out := make(chan error)

	var wg sync.WaitGroup

	forward := func(errCh <-chan error) {
		defer wg.Done()
		for {
			select {
			case <-ctx.Done():              // контекст отменён
				return
			case err, ok := <-errCh:
				if !ok {                     // канал ошибок закрыт
					return
				}
				select {
				case <-ctx.Done():           // проверяем снова перед отправкой
					return
				case out <- err:             // пересылаем ошибку в выход
				}
			}
		}
	}

	// запускаем горутину для каждого канала ошибок
	for _, errCh := range errChannels {
		wg.Add(1)
		go forward(errCh)
	}

	// закрываем выход когда все каналы ошибок завершены
	go func() {
		wg.Wait()
		close(out)
	}()

	return out
}`,
			description: `Реализуйте многостадийный pipeline который связывает стадии обработки каналов с правильной обработкой и распространением ошибок.

**Требования:**
1. **Pipeline**: Связать несколько стадий обработки вместе
2. **Error Channel**: Вернуть отдельный канал ошибок для распространения ошибок
3. **Stage Function**: Каждая стадия преобразует вход в выход
4. **Context Awareness**: Остановить все стадии при отмене контекста или первой ошибке

**Pipeline паттерн:**
\`\`\`go
type Stage[In, Out any] func(context.Context, <-chan In) (<-chan Out, <-chan error)

func Pipeline[T any](
    ctx context.Context,
    in <-chan T,
    stages ...Stage[T, T],
) (<-chan T, <-chan error) {
    // Связать стадии вместе
    // Объединить все каналы ошибок
    // Вернуть финальный выход и объединённые ошибки
}
\`\`\`

**Ключевые концепции:**
- Pipeline связывает стадии где выход одной становится входом следующей
- Каждая стадия работает конкурентно в своей горутине
- Ошибки из любой стадии распространяются через канал ошибок
- Отмена контекста останавливает весь pipeline
- Неблокирующий сбор ошибок предотвращает deadlockи

**Пример использования:**
\`\`\`go
// Pipeline обработки данных
func ProcessData(input <-chan string) (<-chan string, <-chan error) {
    ctx := context.Background()

    // Стадия 1: Валидация
    validate := func(ctx context.Context, in <-chan string) (<-chan string, <-chan error) {
        out := make(chan string)
        errs := make(chan error)
        go func() {
            defer close(out)
            defer close(errs)
            for v := range in {
                if v == "" {
                    errs <- errors.New("пустое значение")
                    continue
                }
                select {
                case <-ctx.Done():
                    return
                case out <- v:
                }
            }
        }()
        return out, errs
    }

    // Стадия 2: Трансформация
    transform := func(ctx context.Context, in <-chan string) (<-chan string, <-chan error) {
        out := make(chan string)
        errs := make(chan error)
        go func() {
            defer close(out)
            defer close(errs)
            for v := range in {
                transformed := strings.ToUpper(v)
                select {
                case <-ctx.Done():
                    return
                case out <- transformed:
                }
            }
        }()
        return out, errs
    }

    return Pipeline(ctx, input, validate, transform)
}
\`\`\`

**Реальные сценарии:**

**1. ETL Pipeline обработки данных:**
\`\`\`go
func ETLPipeline(records <-chan RawRecord) (<-chan ProcessedRecord, <-chan error) {
    ctx := context.Background()

    extract := func(ctx context.Context, in <-chan RawRecord) (<-chan RawRecord, <-chan error) {
        // Валидация и парсинг сырых записей
    }

    transform := func(ctx context.Context, in <-chan RawRecord) (<-chan ProcessedRecord, <-chan error) {
        // Очистка, нормализация, обогащение данных
    }

    load := func(ctx context.Context, in <-chan ProcessedRecord) (<-chan ProcessedRecord, <-chan error) {
        // Запись в БД, обработка дубликатов
    }

    return Pipeline(ctx, records, extract, transform, load)
}
\`\`\`

**2. Pipeline обработки изображений:**
\`\`\`go
func ImagePipeline(images <-chan Image) (<-chan Image, <-chan error) {
    ctx := context.Background()

    resize := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        // Изменение размера до стандартных размеров
    }

    compress := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        // Применение сжатия
    }

    watermark := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        // Добавление водяного знака
    }

    return Pipeline(ctx, images, resize, compress, watermark)
}
\`\`\`

**Ограничения:**
- Должен обрабатывать переменное количество стадий
- Должен объединять все каналы ошибок без блокировки
- Должен распространять отмену контекста на все стадии
- Должен закрывать все каналы правильно для предотвращения утечек
- Должен обрабатывать nil context (использовать Background)`,
			hint1: `Связывайте стадии передавая выходной канал одной стадии как входной канал следующей стадии.`,
			hint2: `Соберите все каналы ошибок от стадий и используйте fan-in паттерн чтобы объединить их в один канал ошибок.`,
			whyItMatters: `Паттерн Pipeline с обработкой ошибок критичен для построения надёжных систем обработки данных в production Go.

**Почему это важно:**

**1. Композитная обработка данных**
Разбейте сложную обработку на простые, тестируемые стадии:
\`\`\`go
// До: Монолитная обработка (сложно тестировать, изменять)
func ProcessRecord(rec Record) (Result, error) {
    // 500 строк валидации, трансформации, обогащения, хранения
    // Один баг ломает всё
    // Нельзя переиспользовать компоненты
}

// После: Pipeline со стадиями (модульный, тестируемый)
func ProcessRecord(rec Record) (Result, error) {
    input := make(chan Record, 1)
    input <- rec
    close(input)

    ctx := context.Background()
    output, errs := Pipeline(ctx, input,
        validateStage,     // 50 строк, тестируется независимо
        transformStage,    // 50 строк, тестируется независимо
        enrichStage,       // 50 строк, тестируется независимо
        storeStage,        // 50 строк, тестируется независимо
    )

    // Каждая стадия проста, переиспользуема, тестируема
}
\`\`\`

**2. Реальный Production сценарий: Обработка логов**
Обработка 10M логов/день в финтех компании:
\`\`\`go
type LogEntry struct {
    Timestamp time.Time
    Level     string
    Service   string
    Message   string
}

func LogPipeline(logs <-chan LogEntry) (<-chan LogEntry, <-chan error) {
    ctx := context.Background()

    // Стадия 1: Парсинг и валидация
    parse := func(ctx context.Context, in <-chan LogEntry) (<-chan LogEntry, <-chan error) {
        out := make(chan LogEntry)
        errs := make(chan error, 100)  // буферизован для неблокирующих ошибок

        go func() {
            defer close(out)
            defer close(errs)

            for log := range in {
                // Валидация timestamp
                if log.Timestamp.IsZero() {
                    errs <- fmt.Errorf("неверный timestamp для сервиса %s", log.Service)
                    continue
                }

                // Нормализация уровня
                log.Level = strings.ToUpper(log.Level)

                select {
                case <-ctx.Done():
                    return
                case out <- log:
                }
            }
        }()
        return out, errs
    }

    // Стадия 2: Обогащение метаданными
    enrich := func(ctx context.Context, in <-chan LogEntry) (<-chan LogEntry, <-chan error) {
        out := make(chan LogEntry)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for log := range in {
                // Добавляем метаданные сервиса из кэша
                metadata, err := getServiceMetadata(log.Service)
                if err != nil {
                    errs <- fmt.Errorf("не удалось обогатить сервис %s: %w", log.Service, err)
                    continue
                }

                log.Message = fmt.Sprintf("[%s:%s] %s", metadata.Region, metadata.Version, log.Message)

                select {
                case <-ctx.Done():
                    return
                case out <- log:
                }
            }
        }()
        return out, errs
    }

    // Стадия 3: Фильтрация и алертинг
    alert := func(ctx context.Context, in <-chan LogEntry) (<-chan LogEntry, <-chan error) {
        out := make(chan LogEntry)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for log := range in {
                // Алерт на критические ошибки
                if log.Level == "ERROR" || log.Level == "FATAL" {
                    if err := sendAlert(log); err != nil {
                        errs <- fmt.Errorf("не удалось отправить алерт: %w", err)
                    }
                }

                select {
                case <-ctx.Done():
                    return
                case out <- log:
                }
            }
        }()
        return out, errs
    }

    return Pipeline(ctx, logs, parse, enrich, alert)
}

// Обработка результатов
output, errors := LogPipeline(logStream)

// Горутина мониторинга ошибок
go func() {
    errorCount := 0
    for err := range errors {
        errorCount++
        log.Printf("Ошибка pipeline: %v", err)

        // Алерт если частота ошибок высокая
        if errorCount > 100 {
            alertOps("Высокая частота ошибок в pipeline логов")
            errorCount = 0
        }
    }
}()

// Обработанные логи
for log := range output {
    indexToElasticsearch(log)
}

// Результаты:
// - До: Монолитный парсер падал на плохих данных (45 мин простоя)
// - После: Ошибки изолированы по стадиям, невалидные логи пропускаются
// - Доступность: 99.9% → 99.99%
// - Экономия: $50K/год (сокращено время реагирования на инциденты)
\`\`\`

**3. E-commerce обработка заказов**
Обработка 50K заказов/день с критической бизнес-логикой:
\`\`\`go
func OrderPipeline(orders <-chan Order) (<-chan Order, <-chan error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
    defer cancel()

    // Стадия 1: Валидация платежа
    validatePayment := func(ctx context.Context, in <-chan Order) (<-chan Order, <-chan error) {
        out := make(chan Order)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for order := range in {
                if err := verifyPayment(order.PaymentID); err != nil {
                    errs <- fmt.Errorf("валидация платежа провалилась для заказа %s: %w", order.ID, err)
                    continue
                }

                select {
                case <-ctx.Done():
                    return
                case out <- order:
                }
            }
        }()
        return out, errs
    }

    // Стадия 2: Проверка инвентаря
    checkInventory := func(ctx context.Context, in <-chan Order) (<-chan Order, <-chan error) {
        out := make(chan Order)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for order := range in {
                available, err := checkStock(order.Items)
                if err != nil {
                    errs <- fmt.Errorf("проверка инвентаря провалилась для заказа %s: %w", order.ID, err)
                    continue
                }
                if !available {
                    errs <- fmt.Errorf("недостаточно товара для заказа %s", order.ID)
                    refundPayment(order.PaymentID)
                    continue
                }

                select {
                case <-ctx.Done():
                    return
                case out <- order:
                }
            }
        }()
        return out, errs
    }

    // Стадия 3: Создание отправки
    createShipment := func(ctx context.Context, in <-chan Order) (<-chan Order, <-chan error) {
        out := make(chan Order)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for order := range in {
                shipmentID, err := createShipping(order)
                if err != nil {
                    errs <- fmt.Errorf("создание отправки провалилось для заказа %s: %w", order.ID, err)
                    continue
                }

                order.ShipmentID = shipmentID

                select {
                case <-ctx.Done():
                    return
                case out <- order:
                }
            }
        }()
        return out, errs
    }

    return Pipeline(ctx, orders, validatePayment, checkInventory, createShipment)
}

// Эффект:
// - До: Одна ошибка в валидации платежа блокировала все заказы
// - После: Ошибки изолированы, валидные заказы продолжают обработку
// - Процент обработки заказов: 80% → 98%
// - Восстановленная выручка: $500K/год от сокращения сбоев обработки
\`\`\`

**4. Сервис обработки изображений**
Обработка 1M изображений/месяц с автоматическим retry:
\`\`\`go
func ImageProcessingPipeline(images <-chan Image) (<-chan Image, <-chan error) {
    ctx := context.Background()

    // Стадия 1: Изменение размера
    resize := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        out := make(chan Image)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for img := range in {
                resized, err := resizeImage(img, 1920, 1080)
                if err != nil {
                    errs <- fmt.Errorf("ошибка изменения размера изображения %s: %w", img.ID, err)
                    continue
                }

                select {
                case <-ctx.Done():
                    return
                case out <- resized:
                }
            }
        }()
        return out, errs
    }

    // Стадия 2: Сжатие
    compress := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        out := make(chan Image)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for img := range in {
                compressed, err := compressImage(img, 80)
                if err != nil {
                    errs <- fmt.Errorf("ошибка сжатия изображения %s: %w", img.ID, err)
                    continue
                }

                select {
                case <-ctx.Done():
                    return
                case out <- compressed:
                }
            }
        }()
        return out, errs
    }

    // Стадия 3: Загрузка на CDN
    upload := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        out := make(chan Image)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for img := range in {
                url, err := uploadToCDN(img)
                if err != nil {
                    errs <- fmt.Errorf("ошибка загрузки изображения %s: %w", img.ID, err)
                    continue
                }

                img.URL = url

                select {
                case <-ctx.Done():
                    return
                case out <- img:
                }
            }
        }()
        return out, errs
    }

    return Pipeline(ctx, images, resize, compress, upload)
}

// Результаты:
// - Время обработки: 200ms → 50ms на изображение (конкурентные стадии)
// - Видимость ошибок: Мониторинг каждой стадии в реальном времени
// - Логика retry: Можно повторить отдельные стадии без переобработки
\`\`\`

**5. Почему отдельный канал ошибок критичен**
\`\`\`go
// НЕПРАВИЛЬНО - Смешивание данных и ошибок в одном канале
type Result struct {
    Data  string
    Error error
}

func BrokenPipeline(in <-chan string) <-chan Result {
    // Сложно обрабатывать, нельзя легко объединять
    // Потребитель должен проверять каждый результат на ошибку
    // Нет чистого разделения между успехом и провалом
}

// ПРАВИЛЬНО - Отдельные каналы
func CorrectPipeline(in <-chan string) (<-chan string, <-chan error) {
    // Чистое разделение ответственности
    // Можно мониторить ошибки независимо
    // Несколько потребителей могут читать тот же канал данных
}

// Использование
output, errs := CorrectPipeline(input)

// Выделенный мониторинг ошибок
go func() {
    for err := range errs {
        metrics.IncrementCounter("pipeline_errors")
        logger.Error(err)
        alertIfCritical(err)
    }
}()

// Чистая обработка данных
for data := range output {
    process(data)  // Нет нужды проверять ошибки
}
\`\`\`

**6. Тестирование Pipeline**
\`\`\`go
func TestPipeline(t *testing.T) {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // Создание тестовых стадий
    stage1 := func(ctx context.Context, in <-chan int) (<-chan int, <-chan error) {
        out := make(chan int)
        errs := make(chan error)
        go func() {
            defer close(out)
            defer close(errs)
            for v := range in {
                if v < 0 {
                    errs <- fmt.Errorf("отрицательное значение: %d", v)
                    continue
                }
                out <- v * 2
            }
        }()
        return out, errs
    }

    // Тестовый ввод
    input := make(chan int, 3)
    input <- 1
    input <- -1  // Должна вызвать ошибку
    input <- 2
    close(input)

    // Запуск pipeline
    output, errors := Pipeline(ctx, input, stage1)

    // Сбор результатов
    var results []int
    for v := range output {
        results = append(results, v)
    }

    // Сбор ошибок
    var errs []error
    for err := range errors {
        errs = append(errs, err)
    }

    // Проверка
    if len(results) != 2 {
        t.Errorf("ожидалось 2 результата, получено %d", len(results))
    }
    if len(errs) != 1 {
        t.Errorf("ожидалась 1 ошибка, получено %d", len(errs))
    }
}
\`\`\`

**Production Best Practices:**
1. Используйте буферизованные каналы ошибок для предотвращения блокировки стадий
2. Мониторьте частоту ошибок для каждой стадии независимо
3. Добавляйте таймауты контекста для предотвращения бесконечного запуска pipeline
4. Тестируйте каждую стадию изолированно перед интеграцией
5. Используйте метрики для отслеживания пропускной способности каждой стадии
6. Реализуйте логику retry для временных ошибок
7. Добавьте circuit breakers для внешних зависимостей
8. Логируйте метаданные стадий для отладки проблем производительности
9. Используйте трассировку для отслеживания элементов через весь pipeline
10. Реализуйте graceful shutdown для корректной остановки всех стадий

**Реальное влияние:**
Компания обработки данных:
- **До**: Монолитный ETL pipeline (30 мин на обработку 1M записей)
- **После**: 5-стадийный pipeline (8 мин на обработку 1M записей, 3.75x быстрее)
- **Обработка ошибок**: До: Одна плохая запись роняла всю задачу → После: Плохие записи пропускаются, логируются отдельно
- **Мониторинг**: Видимость производительности каждой стадии в реальном времени
- **Результат**: Сэкономлено 22 мин за запуск × 24 запуска/день = 8.8 часов/день
- **Стоимость**: Сокращение затрат на вычисления на $10K/месяц
- **Масштабируемость**: Теперь можно добавлять новые стадии без переписывания всего кода
- **Обслуживаемость**: Время на исправление багов сократилось с 3 дней до 4 часов

Pipeline паттерн превращает сложную обработку в простые, композитные, тестируемые компоненты. Освойте его для построения надёжных production систем. Это фундаментальный паттерн для любого Go-разработчика работающего с конкурентной обработкой данных.`
		},
		uz: {
			title: `Pipelineda xatolarni ishlash`,
			solutionCode: `package channelsx

import (
	"context"
	"sync"
)

type Stage[In, Out any] func(context.Context, <-chan In) (<-chan Out, <-chan error)

func Pipeline[T any](ctx context.Context, in <-chan T, stages ...Stage[T, T]) (<-chan T, <-chan error) {
	if ctx == nil {
		ctx = context.Background()
	}

	// barcha bosqichlardan xato kanallarini to'playmiz
	var errorChannels []<-chan error

	// bosqichlarni zanjirlash: N bosqichning chiqishi N+1 bosqichning kirishi bo'ladi
	current := in
	for _, stage := range stages {
		out, errs := stage(ctx, current)    // bosqichni bajaramiz
		errorChannels = append(errorChannels, errs)
		current = out                        // chiqish keyingi bosqich uchun kirish bo'ladi
	}

	// barcha xato kanallarini bittaga birlashtiramiz
	mergedErrors := mergeErrors(ctx, errorChannels...)

	return current, mergedErrors            // yakuniy chiqish va birlashtirilgan xatolarni qaytaramiz
}

// mergeErrors bir nechta xato kanallarini bitta kanalga birlashtiradi
func mergeErrors(ctx context.Context, errChannels ...<-chan error) <-chan error {
	out := make(chan error)

	var wg sync.WaitGroup

	forward := func(errCh <-chan error) {
		defer wg.Done()
		for {
			select {
			case <-ctx.Done():              // kontekst bekor qilindi
				return
			case err, ok := <-errCh:
				if !ok {                     // xato kanali yopilgan
					return
				}
				select {
				case <-ctx.Done():           // yuborishdan oldin yana tekshiramiz
					return
				case out <- err:             // xatoni chiqishga yo'naltiramiz
				}
			}
		}
	}

	// har bir xato kanali uchun gorutin ishga tushiramiz
	for _, errCh := range errChannels {
		wg.Add(1)
		go forward(errCh)
	}

	// barcha xato kanallari tugaganda chiqishni yopamiz
	go func() {
		wg.Wait()
		close(out)
	}()

	return out
}`,
			description: `Ko'p bosqichli pipeline ni amalga oshiring, u kanal qayta ishlash bosqichlarini to'g'ri xatolarni qayta ishlash va tarqatish bilan bog'laydi.

**Talablar:**
1. **Pipeline**: Bir nechta qayta ishlash bosqichlarini birga bog'lash
2. **Error Channel**: Xatolarni tarqatish uchun alohida xato kanalini qaytarish
3. **Stage Function**: Har bir bosqich kirishni chiqishga o'zgartiradi
4. **Context Awareness**: Kontekst bekor qilinganda yoki birinchi xatoda barcha bosqichlarni to'xtatish

**Pipeline pattern:**
\`\`\`go
type Stage[In, Out any] func(context.Context, <-chan In) (<-chan Out, <-chan error)

func Pipeline[T any](
    ctx context.Context,
    in <-chan T,
    stages ...Stage[T, T],
) (<-chan T, <-chan error) {
    // Bosqichlarni birga bog'lash
    // Barcha xato kanallarini birlashtirish
    // Yakuniy chiqish va birlashtirilgan xatolarni qaytarish
}
\`\`\`

**Asosiy tushunchalar:**
- Pipeline bosqichlarni bog'laydi, bunda birining chiqishi keyingisining kirishi bo'ladi
- Har bir bosqich o'z gorutinida parallel ishlaydi
- Istalgan bosqichdan xatolar xato kanali orqali tarqaladi
- Kontekst bekor qilish butun pipelineni to'xtatadi
- Blokirovka qilmaydigan xatolarni to'plash deadlocklarning oldini oladi

**Foydalanish misoli:**
\`\`\`go
// Ma'lumotlarni qayta ishlash pipelinei
func ProcessData(input <-chan string) (<-chan string, <-chan error) {
    ctx := context.Background()

    // Bosqich 1: Validatsiya
    validate := func(ctx context.Context, in <-chan string) (<-chan string, <-chan error) {
        out := make(chan string)
        errs := make(chan error)
        go func() {
            defer close(out)
            defer close(errs)
            for v := range in {
                if v == "" {
                    errs <- errors.New("bo'sh qiymat")
                    continue
                }
                select {
                case <-ctx.Done():
                    return
                case out <- v:
                }
            }
        }()
        return out, errs
    }

    // Bosqich 2: Transformatsiya
    transform := func(ctx context.Context, in <-chan string) (<-chan string, <-chan error) {
        out := make(chan string)
        errs := make(chan error)
        go func() {
            defer close(out)
            defer close(errs)
            for v := range in {
                transformed := strings.ToUpper(v)
                select {
                case <-ctx.Done():
                    return
                case out <- transformed:
                }
            }
        }()
        return out, errs
    }

    return Pipeline(ctx, input, validate, transform)
}
\`\`\`

**Haqiqiy stsenariylar:**

**1. ETL Pipeline ma'lumotlarni qayta ishlash:**
\`\`\`go
func ETLPipeline(records <-chan RawRecord) (<-chan ProcessedRecord, <-chan error) {
    ctx := context.Background()

    extract := func(ctx context.Context, in <-chan RawRecord) (<-chan RawRecord, <-chan error) {
        // Xom yozuvlarni validatsiya qilish va parsing
    }

    transform := func(ctx context.Context, in <-chan RawRecord) (<-chan ProcessedRecord, <-chan error) {
        // Ma'lumotlarni tozalash, normalizatsiya qilish, boyitish
    }

    load := func(ctx context.Context, in <-chan ProcessedRecord) (<-chan ProcessedRecord, <-chan error) {
        // Ma'lumotlar bazasiga yozish, duplikatlarni qayta ishlash
    }

    return Pipeline(ctx, records, extract, transform, load)
}
\`\`\`

**2. Rasmlarni qayta ishlash pipelinei:**
\`\`\`go
func ImagePipeline(images <-chan Image) (<-chan Image, <-chan error) {
    ctx := context.Background()

    resize := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        // Standart o'lchamlarga o'zgartirish
    }

    compress := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        // Siqishni qo'llash
    }

    watermark := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        // Watermark qo'shish
    }

    return Pipeline(ctx, images, resize, compress, watermark)
}
\`\`\`

**Cheklovlar:**
- O'zgaruvchan miqdordagi bosqichlarni qayta ishlashi kerak
- Barcha xato kanallarini blokirovka qilmasdan birlashtirishish kerak
- Kontekst bekor qilishni barcha bosqichlarga tarqatishi kerak
- Oqish oldini olish uchun barcha kanallarni to'g'ri yopishi kerak
- nil kontekstni qayta ishlashi kerak (Background ishlatish)`,
			hint1: `Bir bosqichning chiqish kanalini keyingi bosqichning kirish kanali sifatida o'tkazib, bosqichlarni zanjirlang.`,
			hint2: `Bosqichlardan barcha xato kanallarini to'plang va ularni bitta xato kanaliga birlashtirish uchun fan-in patterndan foydalaning.`,
			whyItMatters: `Xatolarni qayta ishlash bilan Pipeline pattern production Go da ishonchli ma'lumotlarni qayta ishlash tizimlarini qurish uchun muhim fundamental patterndir.

**Nima uchun bu muhim:**

**1. Kompozit ma'lumotlarni qayta ishlash**
Murakkab qayta ishlashni oddiy, test qilinadigan bosqichlarga bo'ling:
\`\`\`go
// Oldin: Monolit qayta ishlash (test qilish, o'zgartirish qiyin)
func ProcessRecord(rec Record) (Result, error) {
    // 500 qator validatsiya, transformatsiya, boyitish, saqlash
    // Bitta xato hammani buzadi
    // Komponentlarni qayta ishlatib bo'lmaydi
    // Debug qilish juda qiyin
}

// Keyin: Bosqichlar bilan Pipeline (modulli, test qilinadigan)
func ProcessRecord(rec Record) (Result, error) {
    input := make(chan Record, 1)
    input <- rec
    close(input)

    ctx := context.Background()
    output, errs := Pipeline(ctx, input,
        validateStage,     // 50 qator, mustaqil test qilinadi
        transformStage,    // 50 qator, mustaqil test qilinadi
        enrichStage,       // 50 qator, mustaqil test qilinadi
        storeStage,        // 50 qator, mustaqil test qilinadi
    )

    // Har bir bosqich oddiy, qayta foydalaniladigan, test qilinadigan
    // Xato qayerda yuz berganini osongina topish mumkin
}
\`\`\`

**2. Haqiqiy Production stsenariy: Loglarni qayta ishlash**
Fintech kompaniyasida kuniga 10M logni qayta ishlash:
\`\`\`go
type LogEntry struct {
    Timestamp time.Time
    Level     string
    Service   string
    Message   string
}

func LogPipeline(logs <-chan LogEntry) (<-chan LogEntry, <-chan error) {
    ctx := context.Background()

    // Bosqich 1: Parsing va validatsiya
    parse := func(ctx context.Context, in <-chan LogEntry) (<-chan LogEntry, <-chan error) {
        out := make(chan LogEntry)
        errs := make(chan error, 100)  // blokirovka qilmaydigan xatolar uchun buferli

        go func() {
            defer close(out)
            defer close(errs)

            for log := range in {
                // Timestamp validatsiyasi
                if log.Timestamp.IsZero() {
                    errs <- fmt.Errorf("%s xizmati uchun noto'g'ri timestamp", log.Service)
                    continue
                }

                // Daraja normalizatsiyasi
                log.Level = strings.ToUpper(log.Level)

                select {
                case <-ctx.Done():
                    return
                case out <- log:
                }
            }
        }()
        return out, errs
    }

    // Bosqich 2: Metama'lumotlar bilan boyitish
    enrich := func(ctx context.Context, in <-chan LogEntry) (<-chan LogEntry, <-chan error) {
        out := make(chan LogEntry)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for log := range in {
                // Keshdan xizmat metama'lumotlarini qo'shamiz
                metadata, err := getServiceMetadata(log.Service)
                if err != nil {
                    errs <- fmt.Errorf("%s xizmatini boyitib bo'lmadi: %w", log.Service, err)
                    continue
                }

                log.Message = fmt.Sprintf("[%s:%s] %s", metadata.Region, metadata.Version, log.Message)

                select {
                case <-ctx.Done():
                    return
                case out <- log:
                }
            }
        }()
        return out, errs
    }

    // Bosqich 3: Filtrlash va ogohlantirish
    alert := func(ctx context.Context, in <-chan LogEntry) (<-chan LogEntry, <-chan error) {
        out := make(chan LogEntry)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for log := range in {
                // Kritik xatolar uchun ogohlantirish
                if log.Level == "ERROR" || log.Level == "FATAL" {
                    if err := sendAlert(log); err != nil {
                        errs <- fmt.Errorf("ogohlantirish yuborib bo'lmadi: %w", err)
                    }
                }

                select {
                case <-ctx.Done():
                    return
                case out <- log:
                }
            }
        }()
        return out, errs
    }

    return Pipeline(ctx, logs, parse, enrich, alert)
}

// Natijalarni qayta ishlash
output, errors := LogPipeline(logStream)

// Xatolarni monitoring qiluvchi gorutin
go func() {
    errorCount := 0
    for err := range errors {
        errorCount++
        log.Printf("Pipeline xatosi: %v", err)

        // Xato chastotasi yuqori bo'lsa ogohlantirish
        if errorCount > 100 {
            alertOps("Loglar pipelineda yuqori xato chastotasi")
            errorCount = 0
        }
    }
}()

// Qayta ishlangan loglar
for log := range output {
    indexToElasticsearch(log)
}

// Natijalar:
// - Oldin: Monolit parser yomon ma'lumotlarda qulab tushdi (45 daqiqa to'xtash)
// - Keyin: Xatolar bosqichlar bo'yicha izolyatsiya qilingan, noto'g'ri loglar o'tkazib yuboriladi
// - Mavjudlik: 99.9% → 99.99%
// - Tejamkorlik: Yiliga $50K (hodisalarga javob berish vaqti qisqartirildi)
\`\`\`

**3. E-commerce buyurtmalarni qayta ishlash**
Kuniga 50K buyurtmani muhim biznes mantiq bilan qayta ishlash:
\`\`\`go
func OrderPipeline(orders <-chan Order) (<-chan Order, <-chan error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
    defer cancel()

    // Bosqich 1: To'lovni validatsiya qilish
    validatePayment := func(ctx context.Context, in <-chan Order) (<-chan Order, <-chan error) {
        out := make(chan Order)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for order := range in {
                if err := verifyPayment(order.PaymentID); err != nil {
                    errs <- fmt.Errorf("%s buyurtma uchun to'lov validatsiyasi muvaffaqiyatsiz: %w", order.ID, err)
                    continue
                }

                select {
                case <-ctx.Done():
                    return
                case out <- order:
                }
            }
        }()
        return out, errs
    }

    // Bosqich 2: Inventarni tekshirish
    checkInventory := func(ctx context.Context, in <-chan Order) (<-chan Order, <-chan error) {
        out := make(chan Order)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for order := range in {
                available, err := checkStock(order.Items)
                if err != nil {
                    errs <- fmt.Errorf("%s buyurtma uchun inventar tekshiruvi muvaffaqiyatsiz: %w", order.ID, err)
                    continue
                }
                if !available {
                    errs <- fmt.Errorf("%s buyurtma uchun yetarli mahsulot yo'q", order.ID)
                    refundPayment(order.PaymentID)
                    continue
                }

                select {
                case <-ctx.Done():
                    return
                case out <- order:
                }
            }
        }()
        return out, errs
    }

    // Bosqich 3: Yetkazib berishni yaratish
    createShipment := func(ctx context.Context, in <-chan Order) (<-chan Order, <-chan error) {
        out := make(chan Order)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for order := range in {
                shipmentID, err := createShipping(order)
                if err != nil {
                    errs <- fmt.Errorf("%s buyurtma uchun yetkazib berish yaratish muvaffaqiyatsiz: %w", order.ID, err)
                    continue
                }

                order.ShipmentID = shipmentID

                select {
                case <-ctx.Done():
                    return
                case out <- order:
                }
            }
        }()
        return out, errs
    }

    return Pipeline(ctx, orders, validatePayment, checkInventory, createShipment)
}

// Ta'sir:
// - Oldin: To'lov validatsiyasidagi bitta xato barcha buyurtmalarni blokladi
// - Keyin: Xatolar izolyatsiya qilingan, to'g'ri buyurtmalar qayta ishlashda davom etadi
// - Buyurtmalarni qayta ishlash foizi: 80% → 98%
// - Qaytarilgan daromad: Yiliga $500K qayta ishlash xatolarining kamayishidan
\`\`\`

**4. Rasmlarni qayta ishlash xizmati**
Oyiga 1M rasmni avtomatik retry bilan qayta ishlash:
\`\`\`go
func ImageProcessingPipeline(images <-chan Image) (<-chan Image, <-chan error) {
    ctx := context.Background()

    // Bosqich 1: O'lchamini o'zgartirish
    resize := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        out := make(chan Image)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for img := range in {
                resized, err := resizeImage(img, 1920, 1080)
                if err != nil {
                    errs <- fmt.Errorf("%s rasmi o'lchamini o'zgartirishda xato: %w", img.ID, err)
                    continue
                }

                select {
                case <-ctx.Done():
                    return
                case out <- resized:
                }
            }
        }()
        return out, errs
    }

    // Bosqich 2: Siqish
    compress := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        out := make(chan Image)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for img := range in {
                compressed, err := compressImage(img, 80)
                if err != nil {
                    errs <- fmt.Errorf("%s rasmini siqishda xato: %w", img.ID, err)
                    continue
                }

                select {
                case <-ctx.Done():
                    return
                case out <- compressed:
                }
            }
        }()
        return out, errs
    }

    // Bosqich 3: CDN ga yuklash
    upload := func(ctx context.Context, in <-chan Image) (<-chan Image, <-chan error) {
        out := make(chan Image)
        errs := make(chan error, 100)

        go func() {
            defer close(out)
            defer close(errs)

            for img := range in {
                url, err := uploadToCDN(img)
                if err != nil {
                    errs <- fmt.Errorf("%s rasmini yuklashda xato: %w", img.ID, err)
                    continue
                }

                img.URL = url

                select {
                case <-ctx.Done():
                    return
                case out <- img:
                }
            }
        }()
        return out, errs
    }

    return Pipeline(ctx, images, resize, compress, upload)
}

// Natijalar:
// - Qayta ishlash vaqti: 200ms → 50ms har bir rasm uchun (parallel bosqichlar)
// - Xatolarni ko'rish: Har bir bosqichni real vaqtda monitoring qilish
// - Retry mantiq: Alohida bosqichlarni qayta ishlamasdan takrorlash mumkin
\`\`\`

**5. Alohida xato kanali nima uchun muhim**
\`\`\`go
// NOTO'G'RI - Ma'lumotlar va xatolarni bir kanalda aralashtirish
type Result struct {
    Data  string
    Error error
}

func BrokenPipeline(in <-chan string) <-chan Result {
    // Qayta ishlash qiyin, osonlikcha birlashtirib bo'lmaydi
    // Iste'molchi har bir natijani xatoga tekshirishi kerak
    // Muvaffaqiyat va muvaffaqiyatsizlik o'rtasida toza ajratish yo'q
}

// TO'G'RI - Alohida kanallar
func CorrectPipeline(in <-chan string) (<-chan string, <-chan error) {
    // Mas'uliyatning toza ajratilishi
    // Xatolarni mustaqil monitor qilish mumkin
    // Bir nechta iste'molchilar bir xil ma'lumot kanalini o'qishi mumkin
}

// Foydalanish
output, errs := CorrectPipeline(input)

// Maxsus xatolarni monitoring qilish
go func() {
    for err := range errs {
        metrics.IncrementCounter("pipeline_errors")
        logger.Error(err)
        alertIfCritical(err)
    }
}()

// Toza ma'lumotlarni qayta ishlash
for data := range output {
    process(data)  // Xatolarni tekshirish shart emas
}
\`\`\`

**6. Pipeline ni test qilish**
\`\`\`go
func TestPipeline(t *testing.T) {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // Test bosqichlarini yaratish
    stage1 := func(ctx context.Context, in <-chan int) (<-chan int, <-chan error) {
        out := make(chan int)
        errs := make(chan error)
        go func() {
            defer close(out)
            defer close(errs)
            for v := range in {
                if v < 0 {
                    errs <- fmt.Errorf("manfiy qiymat: %d", v)
                    continue
                }
                out <- v * 2
            }
        }()
        return out, errs
    }

    // Test kirish
    input := make(chan int, 3)
    input <- 1
    input <- -1  // Xato chiqarishi kerak
    input <- 2
    close(input)

    // Pipeline ni ishga tushirish
    output, errors := Pipeline(ctx, input, stage1)

    // Natijalarni yig'ish
    var results []int
    for v := range output {
        results = append(results, v)
    }

    // Xatolarni yig'ish
    var errs []error
    for err := range errors {
        errs = append(errs, err)
    }

    // Tekshirish
    if len(results) != 2 {
        t.Errorf("2 ta natija kutilgan, %d ta olindi", len(results))
    }
    if len(errs) != 1 {
        t.Errorf("1 ta xato kutilgan, %d ta olindi", len(errs))
    }
}
\`\`\`

**Production Best Practices:**
1. Bosqichlarni blokirovka qilishning oldini olish uchun buferli xato kanallaridan foydalaning
2. Har bir bosqich uchun xato chastotasini mustaqil monitor qiling
3. Cheksiz pipeline ishlashining oldini olish uchun kontekst timeoutlarini qo'shing
4. Integratsiyadan oldin har bir bosqichni alohida test qiling
5. Har bir bosqichning o'tkazish qobiliyatini kuzatish uchun metrikalardan foydalaning
6. Vaqtinchalik xatolar uchun retry mantiqini amalga oshiring
7. Tashqi bog'liqliklar uchun circuit breakerlar qo'shing
8. Ishlash muammolarini tuzatish uchun bosqich metama'lumotlarini logging qiling
9. Elementlarni butun pipeline bo'ylab kuzatish uchun tracingdan foydalaning
10. Barcha bosqichlarni to'g'ri to'xtatish uchun graceful shutdown ni amalga oshiring

**Haqiqiy ta'sir:**
Ma'lumotlarni qayta ishlash kompaniyasi:
- **Oldin**: Monolit ETL pipeline (1M yozuvni qayta ishlash uchun 30 daqiqa)
- **Keyin**: 5 bosqichli pipeline (1M yozuvni qayta ishlash uchun 8 daqiqa, 3.75x tezroq)
- **Xatolarni qayta ishlash**: Oldin: Bitta yomon yozuv butun ishni qulatdi → Keyin: Yomon yozuvlar o'tkazib yuboriladi, alohida loglanadi
- **Monitoring**: Har bir bosqichning ishlashini real vaqtda ko'rish
- **Natija**: Har bir ishga 22 daqiqa tejaldi × kuniga 24 ta ish = kuniga 8.8 soat tejaldi
- **Xarajat**: Hisoblash xarajatlari oyiga $10K ga qisqartirildi
- **Masshtablash**: Endi butun kodni qayta yozmasdan yangi bosqichlarni qo'shish mumkin
- **Xizmat ko'rsatish**: Xatolarni tuzatish vaqti 3 kundan 4 soatga qisqardi

Pipeline pattern murakkab qayta ishlashni oddiy, kompozit, test qilinadigan komponentlarga aylantiradi. Ishonchli production tizimlarni qurish uchun uni o'zlashtiring. Bu parallel ma'lumotlarni qayta ishlash bilan ishlaydigan har qanday Go dasturchisi uchun fundamental patterndir.`
		}
	}
};

export default task;
