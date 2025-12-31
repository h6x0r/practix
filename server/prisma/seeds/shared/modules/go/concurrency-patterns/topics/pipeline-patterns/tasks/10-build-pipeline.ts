import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-pipeline-build-pipeline',
	title: 'BuildPipeline',
	difficulty: 'hard',	tags: ['go', 'concurrency', 'pipeline', 'composition', 'stage'],
	estimatedTime: '40m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **BuildPipeline** that chains multiple Stage functions together to build a complete data processing pipeline.

**Type Definition:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Requirements:**
1. Create function \`BuildPipeline(ctx context.Context, in <-chan int, stages ...Stage) <-chan int\`
2. Handle nil context (default to Background)
3. Chain stages sequentially: output of stage N becomes input of stage N+1
4. Skip nil stages in the stages list
5. If no stages provided, return input channel unchanged
6. Return final output channel from last stage
7. All stages receive the same context for cancellation propagation

**Example:**
\`\`\`go
ctx := context.Background()

// Build pipeline with multiple stages
square := SquareStage(2)
multiply := MultiplyStage(3)
filter := FilterStage(func(n int) bool { return n > 20 })
take := TakeStage(5)

source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
result := BuildPipeline(ctx, source, square, multiply, filter, take)

for v := range result {
    fmt.Println(v)
}
// Output: 27 48 75 108 147 (first 5 values > 20 after square*3)

// No stages (passthrough)
source = Gen(1, 2, 3)
result = BuildPipeline(ctx, source)
for v := range result {
    fmt.Println(v)
}
// Output: 1 2 3

// With nil stages (skipped)
source = Gen(1, 2, 3)
result = BuildPipeline(ctx, source, square, nil, multiply, nil)
for v := range result {
    fmt.Println(v)
}
// Output: 3 12 27 (nil stages skipped)

// With cancellation
ctx, cancel := context.WithCancel(context.Background())
source = GenWithContext(ctx, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
result = BuildPipeline(ctx, source, square, multiply)

count := 0
for v := range result {
    fmt.Println(v)
    count++
    if count == 3 {
        cancel() // Cancel pipeline
    }
}
// Output: First 3 values (3 12 27)
\`\`\`

**Constraints:**
- Must handle nil context (use Background)
- Must skip nil stages
- Must chain stages sequentially
- Must propagate context to all stages`,
	initialCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

// TODO: Implement BuildPipeline
func BuildPipeline(ctx context.Context, in <-chan int, stages ...Stage) <-chan int {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

func BuildPipeline(ctx context.Context, in <-chan int, stages ...Stage) <-chan int {
	if ctx == nil {                                             // Handle nil context
		ctx = context.Background()                          // Default to Background
	}
	out := in                                                   // Start with input
	for _, stage := range stages {                              // Iterate over stages
		if stage == nil {                                   // Skip nil stages
			continue
		}
		out = stage(ctx, out)                               // Chain stage
	}
	return out                                                  // Return final output
}`,
	testCode: `package concurrency

import (
	"context"
	"testing"
	"time"
)

// Helper stage functions for testing
func doubleStage(ctx context.Context, in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for v := range in {
			select {
			case <-ctx.Done():
				return
			case out <- v * 2:
			}
		}
	}()
	return out
}

func addTenStage(ctx context.Context, in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for v := range in {
			select {
			case <-ctx.Done():
				return
			case out <- v + 10:
			}
		}
	}()
	return out
}

func TestBuildPipeline1(t *testing.T) {
	// Test with single stage
	ctx := context.Background()
	in := make(chan int)

	go func() {
		for i := 1; i <= 3; i++ {
			in <- i
		}
		close(in)
	}()

	out := BuildPipeline(ctx, in, doubleStage)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	expected := []int{2, 4, 6}
	if len(result) != len(expected) {
		t.Errorf("expected %d values, got %d", len(expected), len(result))
	}
	for i := range result {
		if result[i] != expected[i] {
			t.Errorf("expected %d, got %d at position %d", expected[i], result[i], i)
		}
	}
}

func TestBuildPipeline2(t *testing.T) {
	// Test with multiple stages
	ctx := context.Background()
	in := make(chan int)

	go func() {
		for i := 1; i <= 3; i++ {
			in <- i
		}
		close(in)
	}()

	out := BuildPipeline(ctx, in, doubleStage, addTenStage)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	expected := []int{12, 14, 16} // (1*2)+10, (2*2)+10, (3*2)+10
	if len(result) != len(expected) {
		t.Errorf("expected %d values, got %d", len(expected), len(result))
	}
	for i := range result {
		if result[i] != expected[i] {
			t.Errorf("expected %d, got %d at position %d", expected[i], result[i], i)
		}
	}
}

func TestBuildPipeline3(t *testing.T) {
	// Test with no stages
	ctx := context.Background()
	in := make(chan int)

	go func() {
		for i := 1; i <= 3; i++ {
			in <- i
		}
		close(in)
	}()

	out := BuildPipeline(ctx, in)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	expected := []int{1, 2, 3}
	if len(result) != len(expected) {
		t.Errorf("expected %d values, got %d", len(expected), len(result))
	}
}

func TestBuildPipeline4(t *testing.T) {
	// Test with nil stages (should skip)
	ctx := context.Background()
	in := make(chan int)

	go func() {
		for i := 1; i <= 3; i++ {
			in <- i
		}
		close(in)
	}()

	out := BuildPipeline(ctx, in, nil, doubleStage, nil, addTenStage, nil)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	expected := []int{12, 14, 16}
	if len(result) != len(expected) {
		t.Errorf("expected %d values, got %d", len(expected), len(result))
	}
}

func TestBuildPipeline5(t *testing.T) {
	// Test with nil context
	in := make(chan int)

	go func() {
		for i := 1; i <= 3; i++ {
			in <- i
		}
		close(in)
	}()

	out := BuildPipeline(nil, in, doubleStage)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	expected := []int{2, 4, 6}
	if len(result) != len(expected) {
		t.Errorf("expected %d values, got %d", len(expected), len(result))
	}
}

func TestBuildPipeline6(t *testing.T) {
	// Test with context cancellation
	ctx, cancel := context.WithCancel(context.Background())
	in := make(chan int)

	go func() {
		for i := 1; i <= 100; i++ {
			select {
			case in <- i:
			case <-time.After(10 * time.Millisecond):
				return
			}
		}
		close(in)
	}()

	out := BuildPipeline(ctx, in, doubleStage, addTenStage)
	cancel() // Cancel immediately

	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	// Should stop early due to cancellation
	if len(result) > 100 {
		t.Errorf("expected at most 100 values due to cancellation, got %d", len(result))
	}
}

func TestBuildPipeline7(t *testing.T) {
	// Test empty input channel
	ctx := context.Background()
	in := make(chan int)
	close(in)

	out := BuildPipeline(ctx, in, doubleStage, addTenStage)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 0 {
		t.Errorf("expected 0 values, got %d", len(result))
	}
}

func TestBuildPipeline8(t *testing.T) {
	// Test three stages
	ctx := context.Background()
	in := make(chan int)

	tripleStage := func(ctx context.Context, in <-chan int) <-chan int {
		out := make(chan int)
		go func() {
			defer close(out)
			for v := range in {
				select {
				case <-ctx.Done():
					return
				case out <- v * 3:
				}
			}
		}()
		return out
	}

	go func() {
		in <- 2
		close(in)
	}()

	out := BuildPipeline(ctx, in, doubleStage, tripleStage, addTenStage)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	expected := []int{22} // (2*2*3)+10 = 22
	if len(result) != 1 || result[0] != expected[0] {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func TestBuildPipeline9(t *testing.T) {
	// Test all nil stages
	ctx := context.Background()
	in := make(chan int)

	go func() {
		for i := 1; i <= 3; i++ {
			in <- i
		}
		close(in)
	}()

	out := BuildPipeline(ctx, in, nil, nil, nil)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	expected := []int{1, 2, 3}
	if len(result) != len(expected) {
		t.Errorf("expected %d values, got %d", len(expected), len(result))
	}
}

func TestBuildPipeline10(t *testing.T) {
	// Test stages applied in correct order
	ctx := context.Background()
	in := make(chan int)

	go func() {
		in <- 5
		close(in)
	}()

	// Order matters: (5+10)*2 = 30, not (5*2)+10 = 20
	out := BuildPipeline(ctx, in, addTenStage, doubleStage)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	expected := 30 // (5+10)*2
	if len(result) != 1 || result[0] != expected {
		t.Errorf("expected %d, got %v", expected, result)
	}
}`,
			hint1: `Check if ctx is nil and set it to context.Background(). Initialize out := in, then loop through stages.`,
			hint2: `For each non-nil stage, apply it: out = stage(ctx, out). This chains stages by using the output of one as input to the next.`,
			whyItMatters: `BuildPipeline demonstrates functional composition of pipeline stages, enabling declarative construction of complex data processing workflows from simple, reusable components.

**Why Pipeline Composition:**
- **Declarative:** Describe what to do, not how
- **Reusable:** Compose same stages in different orders
- **Maintainable:** Easy to add, remove, or reorder stages
- **Testable:** Test stages individually and composed
- **Flexible:** Build different pipelines for different needs

**Production Pattern:**
\`\`\`go
// Data processing pipeline
func ProcessUserData(ctx context.Context, userIDs <-chan int) <-chan ProcessedUser {
    stages := []Stage{
        ValidateStage(),
        EnrichFromDBStage(db),
        FilterActiveStage(),
        TransformStage(),
        DeduplicateStage(),
    }

    processed := BuildPipeline(ctx, userIDs, stages...)
    return convertToProcessedUser(processed)
}

// ETL pipeline
func ETLPipeline(ctx context.Context) error {
    // Extract
    source := extractFromSource(ctx)

    // Transform
    stages := []Stage{
        CleanDataStage(),
        ValidateStage(),
        NormalizeStage(),
        EnrichStage(),
        AggregateStage(),
    }

    transformed := BuildPipeline(ctx, source, stages...)

    // Load
    return loadToDestination(ctx, transformed)
}

// Configurable pipeline
type PipelineConfig struct {
    EnableValidation  bool
    EnableEnrichment bool
    EnableCaching    bool
    Workers          int
}

func ConfigurablePipeline(ctx context.Context, in <-chan int, cfg PipelineConfig) <-chan int {
    var stages []Stage

    if cfg.EnableValidation {
        stages = append(stages, ValidateStage())
    }

    stages = append(stages, ProcessStage(cfg.Workers))

    if cfg.EnableEnrichment {
        stages = append(stages, EnrichStage())
    }

    if cfg.EnableCaching {
        stages = append(stages, CacheStage())
    }

    return BuildPipeline(ctx, in, stages...)
}

// A/B testing pipeline
func ABTestPipeline(ctx context.Context, in <-chan int, variant string) <-chan int {
    baseStages := []Stage{
        ValidateStage(),
        NormalizeStage(),
    }

    var experimentalStages []Stage
    switch variant {
    case "A":
        experimentalStages = []Stage{
            ProcessingStrategyA(),
            OptimizationA(),
        }
    case "B":
        experimentalStages = []Stage{
            ProcessingStrategyB(),
            OptimizationB(),
        }
    default:
        experimentalStages = []Stage{
            DefaultProcessing(),
        }
    }

    finalStages := []Stage{
        PostProcessStage(),
        MetricsStage(),
    }

    allStages := append(baseStages, experimentalStages...)
    allStages = append(allStages, finalStages...)

    return BuildPipeline(ctx, in, allStages...)
}

// Pipeline factory
type PipelineBuilder struct {
    stages []Stage
}

func NewPipelineBuilder() *PipelineBuilder {
    return &PipelineBuilder{
        stages: make([]Stage, 0),
    }
}

func (pb *PipelineBuilder) AddStage(stage Stage) *PipelineBuilder {
    pb.stages = append(pb.stages, stage)
    return pb
}

func (pb *PipelineBuilder) AddSquare(workers int) *PipelineBuilder {
    return pb.AddStage(SquareStage(workers))
}

func (pb *PipelineBuilder) AddMultiply(factor int) *PipelineBuilder {
    return pb.AddStage(MultiplyStage(factor))
}

func (pb *PipelineBuilder) AddFilter(predicate func(int) bool) *PipelineBuilder {
    return pb.AddStage(FilterStage(predicate))
}

func (pb *PipelineBuilder) AddTake(n int) *PipelineBuilder {
    return pb.AddStage(TakeStage(n))
}

func (pb *PipelineBuilder) Build(ctx context.Context, in <-chan int) <-chan int {
    return BuildPipeline(ctx, in, pb.stages...)
}

// Fluent API usage
func FluentExample() {
    ctx := context.Background()
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    result := NewPipelineBuilder().
        AddSquare(2).
        AddMultiply(2).
        AddFilter(func(n int) bool { return n > 20 }).
        AddTake(5).
        Build(ctx, source)

    for v := range result {
        fmt.Println(v)
    }
}

// Pipeline with metrics
func MonitoredPipeline(ctx context.Context, in <-chan int) <-chan int {
    stages := []Stage{
        MetricsStage("input"),
        ProcessStage(4),
        MetricsStage("processed"),
        FilterStage(validationPredicate),
        MetricsStage("filtered"),
        TransformStage(),
        MetricsStage("output"),
    }

    return BuildPipeline(ctx, in, stages...)
}

func MetricsStage(name string) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            count := 0
            start := time.Now()

            for {
                select {
                case <-ctx.Done():
                    logMetrics(name, count, time.Since(start))
                    return
                case v, ok := <-in:
                    if !ok {
                        logMetrics(name, count, time.Since(start))
                        return
                    }
                    count++
                    select {
                    case <-ctx.Done():
                        logMetrics(name, count, time.Since(start))
                        return
                    case out <- v:
                    }
                }
            }
        }()
        return out
    }
}

// Conditional pipeline execution
func ConditionalPipeline(ctx context.Context, in <-chan int, conditions map[string]bool) <-chan int {
    var stages []Stage

    // Always validate
    stages = append(stages, ValidateStage())

    // Conditional stages
    if conditions["cache"] {
        stages = append(stages, CacheCheckStage())
    }

    if conditions["enrich"] {
        stages = append(stages, EnrichStage())
    }

    if conditions["dedupe"] {
        stages = append(stages, DeduplicateStage())
    }

    // Always transform
    stages = append(stages, TransformStage())

    return BuildPipeline(ctx, in, stages...)
}

// Pipeline with error handling
func RobustPipeline(ctx context.Context, in <-chan int) (<-chan int, <-chan error) {
    errChan := make(chan error, 1)

    stages := []Stage{
        ErrorHandlingStage(errChan),
        ProcessStage(4),
        ErrorHandlingStage(errChan),
        TransformStage(),
    }

    out := BuildPipeline(ctx, in, stages...)
    return out, errChan
}

// Parallel pipelines with fan-out/fan-in
func ParallelPipelines(ctx context.Context, in <-chan int, parallelism int) <-chan int {
    // Create multiple identical pipelines
    var outputs []<-chan int

    for i := 0; i < parallelism; i++ {
        pipeline := BuildPipeline(ctx, in,
            ProcessStage(1),
            TransformStage(),
        )
        outputs = append(outputs, pipeline)
    }

    // Merge results
    return FanIn(ctx, outputs...)
}

// Complete real-world example
func ProcessOrders(ctx context.Context, orderIDs <-chan int) (int, error) {
    // Build processing pipeline
    pipeline := NewPipelineBuilder().
        AddStage(FetchOrderStage(db)).
        AddStage(ValidateOrderStage()).
        AddStage(EnrichWithCustomerStage(db)).
        AddStage(CalculateTotalsStage()).
        AddStage(ApplyDiscountsStage()).
        AddStage(SaveOrderStage(db)).
        Build(ctx, orderIDs)

    // Process and collect results
    processedCount := Count(pipeline)
    return processedCount, nil
}
\`\`\`

**Real-World Benefits:**
- **Rapid Development:** Build complex pipelines quickly
- **Code Reuse:** Same stages in multiple pipelines
- **Easy Testing:** Test individual stages and combinations
- **Maintainability:** Clear pipeline structure and flow
- **Flexibility:** Easily modify pipelines for different scenarios

**Pipeline Patterns:**
- **Linear:** Sequential stage chain (most common)
- **Conditional:** Include stages based on conditions
- **Configurable:** Build from configuration
- **Parallel:** Multiple pipelines running concurrently
- **Branching:** Split into multiple pipelines at a point
- **Rejoining:** Merge multiple pipelines with FanIn

**Common Pipeline Structures:**
1. **ETL:** Extract → Transform → Load
2. **Validation:** Validate → Filter → Process
3. **Enrichment:** Fetch → Enrich → Transform
4. **Processing:** Clean → Validate → Process → Store
5. **Analytics:** Collect → Aggregate → Analyze → Report

**Best Practices:**
- **Stage Order:** Put cheap filters early, expensive operations late
- **Error Handling:** Include error handling stages
- **Metrics:** Add metric stages for monitoring
- **Caching:** Add cache stages for frequently accessed data
- **Cancellation:** Always pass context through all stages
- **Testing:** Test stages individually before composing

**Performance Tuning:**
- **Bottleneck Identification:** Add metrics at each stage
- **Worker Adjustment:** Increase workers for slow stages
- **Stage Ordering:** Filter early to reduce downstream work
- **Parallelism:** Use parallel pipelines for CPU-bound work
- **Buffering:** Add buffering between stages if needed

Without BuildPipeline, you'd manually chain stages with nested function calls, making pipelines hard to read, maintain, and modify.`,	order: 9,
	translations: {
		ru: {
			title: 'Построение pipeline',
			description: `Реализуйте **BuildPipeline**, который связывает несколько функций Stage вместе для построения полного конвейера обработки данных.

**Определение типа:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Требования:**
1. Создайте функцию \`BuildPipeline(ctx context.Context, in <-chan int, stages ...Stage) <-chan int\`
2. Обработайте nil контекст (по умолчанию Background)
3. Связывайте стадии последовательно: выход стадии N становится входом стадии N+1
4. Пропускайте nil стадии в списке stages
5. Если стадии не предоставлены, верните входной канал без изменений
6. Верните финальный выходной канал последней стадии
7. Все стадии получают один и тот же контекст для распространения отмены

**Пример:**
\`\`\`go
ctx := context.Background()

square := SquareStage(2)
multiply := MultiplyStage(3)
filter := FilterStage(func(n int) bool { return n > 20 })
take := TakeStage(5)

source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
result := BuildPipeline(ctx, source, square, multiply, filter, take)

for v := range result {
    fmt.Println(v)
}
// Вывод: 27 48 75 108 147
\`\`\`

**Ограничения:**
- Должен обрабатывать nil контекст (использовать Background)
- Должен пропускать nil стадии
- Должен связывать стадии последовательно
- Должен распространять контекст на все стадии`,
			hint1: `Проверьте ctx на nil и установите в context.Background(). Инициализируйте out := in, затем цикл по stages.`,
			hint2: `Для каждой не-nil стадии примените её: out = stage(ctx, out). Это связывает стадии используя выход одной как вход следующей.`,
			whyItMatters: `BuildPipeline демонстрирует функциональную композицию стадий pipeline, обеспечивая декларативное построение сложных рабочих процессов обработки данных из простых переиспользуемых компонентов.

**Почему Pipeline Composition важна:**
- **Декларативность:** Описывать что делать, а не как
- **Переиспользуемость:** Комбинирование простых стадий в сложные workflows
- **Читаемость:** Чистое, линейное выражение логики обработки
- **Тестируемость:** Тестирование каждой стадии независимо

**Production паттерны:**
\`\`\`go
// Обработка данных пользователей
func ProcessUserData(ctx context.Context, userIDs <-chan int) <-chan ProcessedUser {
    stages := []Stage{
        ValidateStage(),
        EnrichFromDBStage(db),
        FilterActiveStage(),
        TransformStage(),
        DeduplicateStage(),
    }

    processed := BuildPipeline(ctx, userIDs, stages...)
    return convertToProcessedUser(processed)
}

// ETL pipeline
func ETLPipeline(ctx context.Context) error {
    // Extract
    source := extractFromSource(ctx)

    // Transform
    stages := []Stage{
        CleanDataStage(),
        ValidateStage(),
        NormalizeStage(),
        EnrichStage(),
        AggregateStage(),
    }

    transformed := BuildPipeline(ctx, source, stages...)

    // Load
    return loadToDestination(ctx, transformed)
}

// Настраиваемый pipeline
type PipelineConfig struct {
    EnableValidation  bool
    EnableEnrichment bool
    EnableCaching    bool
    Workers          int
}

func ConfigurablePipeline(ctx context.Context, in <-chan int, cfg PipelineConfig) <-chan int {
    var stages []Stage

    if cfg.EnableValidation {
        stages = append(stages, ValidateStage())
    }

    stages = append(stages, ProcessStage(cfg.Workers))

    if cfg.EnableEnrichment {
        stages = append(stages, EnrichStage())
    }

    if cfg.EnableCaching {
        stages = append(stages, CacheStage())
    }

    return BuildPipeline(ctx, in, stages...)
}

// A/B тестирование pipeline
func ABTestPipeline(ctx context.Context, in <-chan int, variant string) <-chan int {
    baseStages := []Stage{
        ValidateStage(),
        NormalizeStage(),
    }

    var experimentalStages []Stage
    switch variant {
    case "A":
        experimentalStages = []Stage{
            ProcessingStrategyA(),
            OptimizationA(),
        }
    case "B":
        experimentalStages = []Stage{
            ProcessingStrategyB(),
            OptimizationB(),
        }
    default:
        experimentalStages = []Stage{
            DefaultProcessing(),
        }
    }

    finalStages := []Stage{
        PostProcessStage(),
        MetricsStage(),
    }

    allStages := append(baseStages, experimentalStages...)
    allStages = append(allStages, finalStages...)

    return BuildPipeline(ctx, in, allStages...)
}

// Фабрика pipeline
type PipelineBuilder struct {
    stages []Stage
}

func NewPipelineBuilder() *PipelineBuilder {
    return &PipelineBuilder{
        stages: make([]Stage, 0),
    }
}

func (pb *PipelineBuilder) AddStage(stage Stage) *PipelineBuilder {
    pb.stages = append(pb.stages, stage)
    return pb
}

func (pb *PipelineBuilder) AddSquare(workers int) *PipelineBuilder {
    return pb.AddStage(SquareStage(workers))
}

func (pb *PipelineBuilder) AddMultiply(factor int) *PipelineBuilder {
    return pb.AddStage(MultiplyStage(factor))
}

func (pb *PipelineBuilder) AddFilter(predicate func(int) bool) *PipelineBuilder {
    return pb.AddStage(FilterStage(predicate))
}

func (pb *PipelineBuilder) AddTake(n int) *PipelineBuilder {
    return pb.AddStage(TakeStage(n))
}

func (pb *PipelineBuilder) Build(ctx context.Context, in <-chan int) <-chan int {
    return BuildPipeline(ctx, in, pb.stages...)
}

// Использование Fluent API
func FluentExample() {
    ctx := context.Background()
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    result := NewPipelineBuilder().
        AddSquare(2).
        AddMultiply(2).
        AddFilter(func(n int) bool { return n > 20 }).
        AddTake(5).
        Build(ctx, source)

    for v := range result {
        fmt.Println(v)
    }
}

// Pipeline с метриками
func MonitoredPipeline(ctx context.Context, in <-chan int) <-chan int {
    stages := []Stage{
        MetricsStage("input"),
        ProcessStage(4),
        MetricsStage("processed"),
        FilterStage(validationPredicate),
        MetricsStage("filtered"),
        TransformStage(),
        MetricsStage("output"),
    }

    return BuildPipeline(ctx, in, stages...)
}

func MetricsStage(name string) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            count := 0
            start := time.Now()

            for {
                select {
                case <-ctx.Done():
                    logMetrics(name, count, time.Since(start))
                    return
                case v, ok := <-in:
                    if !ok {
                        logMetrics(name, count, time.Since(start))
                        return
                    }
                    count++
                    select {
                    case <-ctx.Done():
                        logMetrics(name, count, time.Since(start))
                        return
                    case out <- v:
                    }
                }
            }
        }()
        return out
    }
}

// Условное выполнение pipeline
func ConditionalPipeline(ctx context.Context, in <-chan int, conditions map[string]bool) <-chan int {
    var stages []Stage

    // Всегда валидировать
    stages = append(stages, ValidateStage())

    // Условные стадии
    if conditions["cache"] {
        stages = append(stages, CacheCheckStage())
    }

    if conditions["enrich"] {
        stages = append(stages, EnrichStage())
    }

    if conditions["dedupe"] {
        stages = append(stages, DeduplicateStage())
    }

    // Всегда трансформировать
    stages = append(stages, TransformStage())

    return BuildPipeline(ctx, in, stages...)
}

// Pipeline с обработкой ошибок
func RobustPipeline(ctx context.Context, in <-chan int) (<-chan int, <-chan error) {
    errChan := make(chan error, 1)

    stages := []Stage{
        ErrorHandlingStage(errChan),
        ProcessStage(4),
        ErrorHandlingStage(errChan),
        TransformStage(),
    }

    out := BuildPipeline(ctx, in, stages...)
    return out, errChan
}

// Параллельные pipeline с fan-out/fan-in
func ParallelPipelines(ctx context.Context, in <-chan int, parallelism int) <-chan int {
    // Создаём несколько идентичных pipeline
    var outputs []<-chan int

    for i := 0; i < parallelism; i++ {
        pipeline := BuildPipeline(ctx, in,
            ProcessStage(1),
            TransformStage(),
        )
        outputs = append(outputs, pipeline)
    }

    // Объединяем результаты
    return FanIn(ctx, outputs...)
}

// Полный реальный пример
func ProcessOrders(ctx context.Context, orderIDs <-chan int) (int, error) {
    // Строим pipeline обработки
    pipeline := NewPipelineBuilder().
        AddStage(FetchOrderStage(db)).
        AddStage(ValidateOrderStage()).
        AddStage(EnrichWithCustomerStage(db)).
        AddStage(CalculateTotalsStage()).
        AddStage(ApplyDiscountsStage()).
        AddStage(SaveOrderStage(db)).
        Build(ctx, orderIDs)

    // Обрабатываем и собираем результаты
    processedCount := Count(pipeline)
    return processedCount, nil
}
\`\`\`

**Реальные преимущества:**
- **Быстрая разработка:** Построение сложных pipeline быстро
- **Переиспользование кода:** Одни и те же стадии в разных pipeline
- **Лёгкое тестирование:** Тестирование отдельных стадий и комбинаций
- **Поддерживаемость:** Чёткая структура и поток pipeline
- **Гибкость:** Лёгкое изменение pipeline для разных сценариев

**Паттерны Pipeline:**
- **Линейный:** Последовательная цепочка стадий (наиболее распространённый)
- **Условный:** Включение стадий на основе условий
- **Настраиваемый:** Построение из конфигурации
- **Параллельный:** Множественные pipeline работают одновременно
- **Ветвящийся:** Разделение на несколько pipeline в точке
- **Объединяющийся:** Слияние нескольких pipeline с FanIn

**Распространённые структуры Pipeline:**
1. **ETL:** Extract → Transform → Load
2. **Валидация:** Validate → Filter → Process
3. **Обогащение:** Fetch → Enrich → Transform
4. **Обработка:** Clean → Validate → Process → Store
5. **Аналитика:** Collect → Aggregate → Analyze → Report

**Лучшие практики:**
- **Порядок стадий:** Дешёвые фильтры рано, дорогие операции поздно
- **Обработка ошибок:** Включайте стадии обработки ошибок
- **Метрики:** Добавляйте стадии метрик для мониторинга
- **Кэширование:** Добавляйте стадии кэша для часто используемых данных
- **Отмена:** Всегда передавайте контекст через все стадии
- **Тестирование:** Тестируйте стадии по отдельности перед композицией

**Настройка производительности:**
- **Определение узких мест:** Добавляйте метрики на каждой стадии
- **Настройка воркеров:** Увеличивайте воркеров для медленных стадий
- **Порядок стадий:** Фильтруйте рано для уменьшения последующей работы
- **Параллелизм:** Используйте параллельные pipeline для CPU-связанной работы
- **Буферизация:** Добавляйте буферизацию между стадиями при необходимости

Без BuildPipeline вам пришлось бы вручную связывать стадии вложенными вызовами функций, что делает pipeline трудночитаемыми, поддерживаемыми и изменяемыми.`,
			solutionCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

func BuildPipeline(ctx context.Context, in <-chan int, stages ...Stage) <-chan int {
	if ctx == nil {                                             // Обрабатываем nil контекст
		ctx = context.Background()                          // По умолчанию Background
	}
	out := in                                                   // Начинаем с входа
	for _, stage := range stages {                              // Итерируемся по стадиям
		if stage == nil {                                   // Пропускаем nil стадии
			continue
		}
		out = stage(ctx, out)                               // Связываем стадию
	}
	return out                                                  // Возвращаем финальный выход
}`
		},
		uz: {
			title: 'Pipeline yaratish',
			description: `To'liq ma'lumotlarni qayta ishlash pipelineni qurish uchun bir nechta Stage funksiyalarini zanjirlaydigan **BuildPipeline** ni amalga oshiring.

**Tur ta'rifi:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Talablar:**
1. \`BuildPipeline(ctx context.Context, in <-chan int, stages ...Stage) <-chan int\` funksiyasini yarating
2. nil kontekstni ishlang (standart Background)
3. Bosqichlarni ketma-ket zanjir qiling: N bosqichining chiqishi N+1 bosqichining kirishi bo'ladi
4. stages ro'yxatidagi nil bosqichlarni o'tkazib yuboring
5. Agar bosqichlar taqdim etilmagan bo'lsa, kirish kanalini o'zgarmagan holda qaytaring
6. Oxirgi bosqichdan yakuniy chiqish kanalini qaytaring
7. Barcha bosqichlar bekor qilishni tarqatish uchun bir xil kontekstni oladi

**Misol:**
\`\`\`go
ctx := context.Background()

square := SquareStage(2)
multiply := MultiplyStage(3)
filter := FilterStage(func(n int) bool { return n > 20 })
take := TakeStage(5)

source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
result := BuildPipeline(ctx, source, square, multiply, filter, take)

for v := range result {
    fmt.Println(v)
}
// Natija: 27 48 75 108 147
\`\`\`

**Cheklovlar:**
- nil kontekstni ishlashi kerak (Background dan foydalaning)
- nil bosqichlarni o'tkazib yuborishi kerak
- Bosqichlarni ketma-ket zanjir qilishi kerak
- Kontekstni barcha bosqichlarga tarqatishi kerak`,
			hint1: `ctx ni nil ga tekshiring va context.Background() ga o'rnating. out := in ni ishga tushiring, keyin stages bo'yicha tsikl.`,
			hint2: `Har bir nil bo'lmagan bosqich uchun uni qo'llang: out = stage(ctx, out). Bu birining chiqishini keyingisining kirishi sifatida ishlatib bosqichlarni zanjirlaydi.`,
			whyItMatters: `BuildPipeline pipeline bosqichlarining funksional kompozitsiyasini namoyish etadi, oddiy, qayta ishlatilishi mumkin komponentlardan murakkab ma'lumotlarni qayta ishlash ish oqimlarini deklarativ qurishni ta'minlaydi.

**Nima uchun Pipeline Composition muhim:**
- **Deklarativlik:** Nima qilishni tasvirlash, qanday emas
- **Qayta ishlatish:** Oddiy bosqichlarni murakkab workflowlarga birlashtirish
- **O'qilishi:** Qayta ishlash mantiqini toza, chiziqli ifodalash
- **Testlanishi:** Har bir bosqichni mustaqil test qilish

**Production patternlar:**
\`\`\`go
// Foydalanuvchi ma'lumotlarini qayta ishlash
func ProcessUserData(ctx context.Context, userIDs <-chan int) <-chan ProcessedUser {
    stages := []Stage{
        ValidateStage(),
        EnrichFromDBStage(db),
        FilterActiveStage(),
        TransformStage(),
        DeduplicateStage(),
    }

    processed := BuildPipeline(ctx, userIDs, stages...)
    return convertToProcessedUser(processed)
}

// ETL pipeline
func ETLPipeline(ctx context.Context) error {
    // Extract
    source := extractFromSource(ctx)

    // Transform
    stages := []Stage{
        CleanDataStage(),
        ValidateStage(),
        NormalizeStage(),
        EnrichStage(),
        AggregateStage(),
    }

    transformed := BuildPipeline(ctx, source, stages...)

    // Load
    return loadToDestination(ctx, transformed)
}

// Sozlanishi mumkin pipeline
type PipelineConfig struct {
    EnableValidation  bool
    EnableEnrichment bool
    EnableCaching    bool
    Workers          int
}

func ConfigurablePipeline(ctx context.Context, in <-chan int, cfg PipelineConfig) <-chan int {
    var stages []Stage

    if cfg.EnableValidation {
        stages = append(stages, ValidateStage())
    }

    stages = append(stages, ProcessStage(cfg.Workers))

    if cfg.EnableEnrichment {
        stages = append(stages, EnrichStage())
    }

    if cfg.EnableCaching {
        stages = append(stages, CacheStage())
    }

    return BuildPipeline(ctx, in, stages...)
}

// A/B testing pipeline
func ABTestPipeline(ctx context.Context, in <-chan int, variant string) <-chan int {
    baseStages := []Stage{
        ValidateStage(),
        NormalizeStage(),
    }

    var experimentalStages []Stage
    switch variant {
    case "A":
        experimentalStages = []Stage{
            ProcessingStrategyA(),
            OptimizationA(),
        }
    case "B":
        experimentalStages = []Stage{
            ProcessingStrategyB(),
            OptimizationB(),
        }
    default:
        experimentalStages = []Stage{
            DefaultProcessing(),
        }
    }

    finalStages := []Stage{
        PostProcessStage(),
        MetricsStage(),
    }

    allStages := append(baseStages, experimentalStages...)
    allStages = append(allStages, finalStages...)

    return BuildPipeline(ctx, in, allStages...)
}

// Pipeline fabrikasi
type PipelineBuilder struct {
    stages []Stage
}

func NewPipelineBuilder() *PipelineBuilder {
    return &PipelineBuilder{
        stages: make([]Stage, 0),
    }
}

func (pb *PipelineBuilder) AddStage(stage Stage) *PipelineBuilder {
    pb.stages = append(pb.stages, stage)
    return pb
}

func (pb *PipelineBuilder) AddSquare(workers int) *PipelineBuilder {
    return pb.AddStage(SquareStage(workers))
}

func (pb *PipelineBuilder) AddMultiply(factor int) *PipelineBuilder {
    return pb.AddStage(MultiplyStage(factor))
}

func (pb *PipelineBuilder) AddFilter(predicate func(int) bool) *PipelineBuilder {
    return pb.AddStage(FilterStage(predicate))
}

func (pb *PipelineBuilder) AddTake(n int) *PipelineBuilder {
    return pb.AddStage(TakeStage(n))
}

func (pb *PipelineBuilder) Build(ctx context.Context, in <-chan int) <-chan int {
    return BuildPipeline(ctx, in, pb.stages...)
}

// Fluent API dan foydalanish
func FluentExample() {
    ctx := context.Background()
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    result := NewPipelineBuilder().
        AddSquare(2).
        AddMultiply(2).
        AddFilter(func(n int) bool { return n > 20 }).
        AddTake(5).
        Build(ctx, source)

    for v := range result {
        fmt.Println(v)
    }
}

// Metrikalar bilan pipeline
func MonitoredPipeline(ctx context.Context, in <-chan int) <-chan int {
    stages := []Stage{
        MetricsStage("input"),
        ProcessStage(4),
        MetricsStage("processed"),
        FilterStage(validationPredicate),
        MetricsStage("filtered"),
        TransformStage(),
        MetricsStage("output"),
    }

    return BuildPipeline(ctx, in, stages...)
}

func MetricsStage(name string) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            count := 0
            start := time.Now()

            for {
                select {
                case <-ctx.Done():
                    logMetrics(name, count, time.Since(start))
                    return
                case v, ok := <-in:
                    if !ok {
                        logMetrics(name, count, time.Since(start))
                        return
                    }
                    count++
                    select {
                    case <-ctx.Done():
                        logMetrics(name, count, time.Since(start))
                        return
                    case out <- v:
                    }
                }
            }
        }()
        return out
    }
}

// Shartli pipeline bajarish
func ConditionalPipeline(ctx context.Context, in <-chan int, conditions map[string]bool) <-chan int {
    var stages []Stage

    // Har doim tekshirish
    stages = append(stages, ValidateStage())

    // Shartli bosqichlar
    if conditions["cache"] {
        stages = append(stages, CacheCheckStage())
    }

    if conditions["enrich"] {
        stages = append(stages, EnrichStage())
    }

    if conditions["dedupe"] {
        stages = append(stages, DeduplicateStage())
    }

    // Har doim transformatsiya qilish
    stages = append(stages, TransformStage())

    return BuildPipeline(ctx, in, stages...)
}

// Xatolarni qayta ishlash bilan pipeline
func RobustPipeline(ctx context.Context, in <-chan int) (<-chan int, <-chan error) {
    errChan := make(chan error, 1)

    stages := []Stage{
        ErrorHandlingStage(errChan),
        ProcessStage(4),
        ErrorHandlingStage(errChan),
        TransformStage(),
    }

    out := BuildPipeline(ctx, in, stages...)
    return out, errChan
}

// Fan-out/fan-in bilan parallel pipelinelar
func ParallelPipelines(ctx context.Context, in <-chan int, parallelism int) <-chan int {
    // Bir nechta identik pipelinelar yaratamiz
    var outputs []<-chan int

    for i := 0; i < parallelism; i++ {
        pipeline := BuildPipeline(ctx, in,
            ProcessStage(1),
            TransformStage(),
        )
        outputs = append(outputs, pipeline)
    }

    // Natijalarni birlashtiramiz
    return FanIn(ctx, outputs...)
}

// To'liq real misol
func ProcessOrders(ctx context.Context, orderIDs <-chan int) (int, error) {
    // Qayta ishlash pipelineni quramiz
    pipeline := NewPipelineBuilder().
        AddStage(FetchOrderStage(db)).
        AddStage(ValidateOrderStage()).
        AddStage(EnrichWithCustomerStage(db)).
        AddStage(CalculateTotalsStage()).
        AddStage(ApplyDiscountsStage()).
        AddStage(SaveOrderStage(db)).
        Build(ctx, orderIDs)

    // Natijalarni qayta ishlaymiz va yig'amiz
    processedCount := Count(pipeline)
    return processedCount, nil
}
\`\`\`

**Haqiqiy foydalari:**
- **Tez rivojlantirish:** Murakkab pipelinelarni tez qurish
- **Kodni qayta ishlatish:** Bir xil bosqichlar turli pipelinelarda
- **Oson sinov:** Alohida bosqichlar va kombinatsiyalarni test qilish
- **Saqlash mumkinligi:** Aniq pipeline tuzilmasi va oqimi
- **Moslashuvchanlik:** Turli stsenariylar uchun pipelinelarni osongina o'zgartirish

**Pipeline patternlari:**
- **Chiziqli:** Ketma-ket bosqichlar zanjiri (eng keng tarqalgan)
- **Shartli:** Shartlarga asoslangan bosqichlarni kiritish
- **Sozlanishi mumkin:** Konfiguratsiyadan qurish
- **Parallel:** Bir vaqtning o'zida bir nechta pipelinelar ishlaydi
- **Tarvaqaylanuvchi:** Nuqtada bir nechta pipelinelarga bo'linish
- **Birlashuvchi:** FanIn bilan bir nechta pipelinelarni birlashtirish

**Umumiy Pipeline tuzilmalari:**
1. **ETL:** Extract → Transform → Load
2. **Tekshirish:** Validate → Filter → Process
3. **Boyitish:** Fetch → Enrich → Transform
4. **Qayta ishlash:** Clean → Validate → Process → Store
5. **Tahliliyot:** Collect → Aggregate → Analyze → Report

**Eng yaxshi amaliyotlar:**
- **Bosqichlar tartibi:** Arzon filtrlar erta, qimmat operatsiyalar kech
- **Xatolarni qayta ishlash:** Xatolarni qayta ishlash bosqichlarini kiriting
- **Metrikalar:** Monitoring uchun metrik bosqichlarni qo'shing
- **Keshlash:** Tez-tez foydalaniladigan ma'lumotlar uchun kesh bosqichlarini qo'shing
- **Bekor qilish:** Har doim barcha bosqichlar orqali kontekstni uzating
- **Sinov:** Kompozitsiyadan oldin bosqichlarni alohida test qiling

**Unumdorlikni sozlash:**
- **Tor joylarni aniqlash:** Har bir bosqichda metrikalarni qo'shing
- **Workerlarni sozlash:** Sekin bosqichlar uchun workerlarni oshiring
- **Bosqichlar tartibi:** Keyingi ishni kamaytirish uchun erta filtrlang
- **Parallellik:** CPU bilan bog'liq ish uchun parallel pipelinelardan foydalaning
- **Buferlash:** Kerak bo'lganda bosqichlar orasida buferlashni qo'shing

BuildPipeline bo'lmasa, siz bosqichlarni qo'lda joylashtirilgan funksiya chaqiruvlari bilan bog'lashingiz kerak bo'ladi, bu pipelinelarni o'qish, saqlash va o'zgartirishni qiyinlashtiradi.`,
			solutionCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

func BuildPipeline(ctx context.Context, in <-chan int, stages ...Stage) <-chan int {
	if ctx == nil {                                             // nil kontekstni ishlaymiz
		ctx = context.Background()                          // Standart Background
	}
	out := in                                                   // Kirish bilan boshlaymiz
	for _, stage := range stages {                              // Bosqichlar bo'ylab iteratsiya qilamiz
		if stage == nil {                                   // nil bosqichlarni o'tkazib yuboramiz
			continue
		}
		out = stage(ctx, out)                               // Bosqichni zanjirlаymiz
	}
	return out                                                  // Yakuniy chiqishni qaytaramiz
}`
		}
	}
};

export default task;
