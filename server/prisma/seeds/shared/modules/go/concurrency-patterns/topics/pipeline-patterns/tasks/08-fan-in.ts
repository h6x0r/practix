import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-pipeline-fan-in',
	title: 'FanIn',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'pipeline', 'fan-in', 'merge'],
	estimatedTime: '35m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **FanIn** that merges multiple input channels into a single output channel with context cancellation.

**Requirements:**
1. Create function \`FanIn(ctx context.Context, ins ...<-chan int) <-chan int\`
2. Skip nil channels in the input list
3. Create output channel (unbuffered)
4. Launch one goroutine per input channel to forward values
5. Each forwarder respects context cancellation (nested select)
6. Use sync.WaitGroup to track all forwarders
7. Close output when all forwarders finish
8. Return output channel immediately

**Example:**
\`\`\`go
ctx := context.Background()

ch1 := Gen(1, 2, 3)
ch2 := Gen(10, 20, 30)
ch3 := Gen(100, 200, 300)

merged := FanIn(ctx, ch1, ch2, ch3)

for v := range merged {
    fmt.Println(v)
}
// Output: 1 10 100 2 20 200 3 30 300 (order may vary due to concurrency)

// With nil channels
ch1 = Gen(1, 2)
merged = FanIn(ctx, ch1, nil, Gen(10, 20), nil)
for v := range merged {
    fmt.Println(v)
}
// Output: 1 2 10 20 (nil channels skipped)

// With cancellation
ctx, cancel := context.WithCancel(context.Background())
ch1 = GenWithContext(ctx, 1, 2, 3, 4, 5)
ch2 = GenWithContext(ctx, 10, 20, 30, 40, 50)
merged = FanIn(ctx, ch1, ch2)

count := 0
for v := range merged {
    fmt.Println(v)
    count++
    if count == 5 {
        cancel() // Cancel after 5 values
    }
}
// Output: First 5 values (then stops)
\`\`\`

**Constraints:**
- Must skip nil channels
- Must use one goroutine per input channel
- Must use nested select for cancellation
- Must use sync.WaitGroup for coordination`,
	initialCode: `package concurrency

import (
	"context"
	"sync"
)

// TODO: Implement FanIn
func FanIn(ctx context.Context, ins ...<-chan int) <-chan int {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
	"sync"
)

func FanIn(ctx context.Context, ins ...<-chan int) <-chan int {
	out := make(chan int)                                       // Create output channel
	var wg sync.WaitGroup                                       // WaitGroup for forwarders
	forward := func(in <-chan int) {                            // Forwarder function
		defer wg.Done()                                     // Mark done
		for {                                               // Infinite loop
			select {
			case <-ctx.Done():                          // Context cancelled
				return                              // Exit forwarder
			case v, ok := <-in:                         // Read from input
				if !ok {                            // Channel closed
					return                      // Exit forwarder
				}
				select {
				case <-ctx.Done():                  // Check before send
					return                      // Exit forwarder
				case out <- v:                      // Forward value
				}
			}
		}
	}
	for _, in := range ins {                                    // Iterate over inputs
		if in == nil {                                      // Skip nil channels
			continue
		}
		wg.Add(1)                                           // Add to WaitGroup
		go forward(in)                                      // Launch forwarder
	}
	go func() {                                                 // Closer goroutine
		wg.Wait()                                           // Wait for all forwarders
		close(out)                                          // Close output
	}()
	return out                                                  // Return immediately
}`,
	testCode: `package concurrency

import (
	"context"
	"sort"
	"testing"
	"time"
)

func TestFanIn1(t *testing.T) {
	// Test merging two channels
	ctx := context.Background()
	ch1 := make(chan int)
	ch2 := make(chan int)

	go func() {
		for i := 1; i <= 3; i++ {
			ch1 <- i
		}
		close(ch1)
	}()

	go func() {
		for i := 4; i <= 6; i++ {
			ch2 <- i
		}
		close(ch2)
	}()

	out := FanIn(ctx, ch1, ch2)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	sort.Ints(result)
	if len(result) != 6 {
		t.Errorf("expected 6 values, got %d", len(result))
	}
	for i := 0; i < 6; i++ {
		if result[i] != i+1 {
			t.Errorf("expected %d, got %d", i+1, result[i])
		}
	}
}

func TestFanIn2(t *testing.T) {
	// Test with single channel
	ctx := context.Background()
	ch := make(chan int)

	go func() {
		for i := 1; i <= 5; i++ {
			ch <- i
		}
		close(ch)
	}()

	out := FanIn(ctx, ch)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 5 {
		t.Errorf("expected 5 values, got %d", len(result))
	}
}

func TestFanIn3(t *testing.T) {
	// Test with no channels (empty variadic)
	ctx := context.Background()
	out := FanIn(ctx)

	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 0 {
		t.Errorf("expected 0 values, got %d", len(result))
	}
}

func TestFanIn4(t *testing.T) {
	// Test with nil channels (should skip)
	ctx := context.Background()
	ch1 := make(chan int)
	var ch2 chan int = nil
	ch3 := make(chan int)

	go func() {
		ch1 <- 1
		ch1 <- 2
		close(ch1)
	}()

	go func() {
		ch3 <- 3
		ch3 <- 4
		close(ch3)
	}()

	out := FanIn(ctx, ch1, ch2, ch3)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 4 {
		t.Errorf("expected 4 values, got %d", len(result))
	}
}

func TestFanIn5(t *testing.T) {
	// Test with context cancellation
	ctx, cancel := context.WithCancel(context.Background())
	ch1 := make(chan int)
	ch2 := make(chan int)

	go func() {
		for i := 1; i <= 100; i++ {
			select {
			case ch1 <- i:
			case <-time.After(10 * time.Millisecond):
				return
			}
		}
		close(ch1)
	}()

	go func() {
		for i := 101; i <= 200; i++ {
			select {
			case ch2 <- i:
			case <-time.After(10 * time.Millisecond):
				return
			}
		}
		close(ch2)
	}()

	out := FanIn(ctx, ch1, ch2)
	cancel() // Cancel immediately

	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	// Should stop early due to cancellation
	if len(result) > 200 {
		t.Errorf("expected at most 200 values, got %d", len(result))
	}
}

func TestFanIn6(t *testing.T) {
	// Test with multiple channels (3+)
	ctx := context.Background()
	ch1 := make(chan int)
	ch2 := make(chan int)
	ch3 := make(chan int)
	ch4 := make(chan int)

	go func() {
		ch1 <- 1
		close(ch1)
	}()

	go func() {
		ch2 <- 2
		close(ch2)
	}()

	go func() {
		ch3 <- 3
		close(ch3)
	}()

	go func() {
		ch4 <- 4
		close(ch4)
	}()

	out := FanIn(ctx, ch1, ch2, ch3, ch4)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 4 {
		t.Errorf("expected 4 values, got %d", len(result))
	}
}

func TestFanIn7(t *testing.T) {
	// Test with already closed channels
	ctx := context.Background()
	ch1 := make(chan int)
	ch2 := make(chan int)
	close(ch1)
	close(ch2)

	out := FanIn(ctx, ch1, ch2)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 0 {
		t.Errorf("expected 0 values, got %d", len(result))
	}
}

func TestFanIn8(t *testing.T) {
	// Test with different speeds
	ctx := context.Background()
	fast := make(chan int)
	slow := make(chan int)

	go func() {
		for i := 1; i <= 10; i++ {
			fast <- i
		}
		close(fast)
	}()

	go func() {
		for i := 11; i <= 15; i++ {
			time.Sleep(5 * time.Millisecond)
			slow <- i
		}
		close(slow)
	}()

	out := FanIn(ctx, fast, slow)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 15 {
		t.Errorf("expected 15 values, got %d", len(result))
	}
}

func TestFanIn9(t *testing.T) {
	// Test output channel closes when all inputs close
	ctx := context.Background()
	ch1 := make(chan int)
	ch2 := make(chan int)

	go func() {
		ch1 <- 1
		close(ch1)
	}()

	go func() {
		ch2 <- 2
		close(ch2)
	}()

	out := FanIn(ctx, ch1, ch2)
	count := 0
	for range out {
		count++
	}

	if count != 2 {
		t.Errorf("expected 2 values, got %d", count)
	}
	// Channel should be closed after ranging
}

func TestFanIn10(t *testing.T) {
	// Test with buffered channels
	ctx := context.Background()
	ch1 := make(chan int, 5)
	ch2 := make(chan int, 5)

	for i := 1; i <= 5; i++ {
		ch1 <- i
	}
	close(ch1)

	for i := 6; i <= 10; i++ {
		ch2 <- i
	}
	close(ch2)

	out := FanIn(ctx, ch1, ch2)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	sort.Ints(result)
	if len(result) != 10 {
		t.Errorf("expected 10 values, got %d", len(result))
	}
	for i := 0; i < 10; i++ {
		if result[i] != i+1 {
			t.Errorf("expected %d, got %d", i+1, result[i])
		}
	}
}`,
			hint1: `Create a forward function that takes an input channel and uses nested select to forward values to the output while respecting ctx.Done().`,
			hint2: `Iterate over ins, skip nil channels, wg.Add(1) for each valid channel, and launch go forward(in). A separate goroutine should wg.Wait() then close(out).`,
			whyItMatters: `FanIn (merge) enables combining multiple data sources into a single stream, essential for aggregating parallel processing results or merging event streams.

**Why Fan-In:**
- **Aggregation:** Combine results from parallel workers
- **Multiplexing:** Merge multiple event sources
- **Load Balancing:** Distribute work across sources
- **Pipeline Composition:** Join split pipelines back together

**Production Pattern:**
\`\`\`go
// Merge search results from multiple databases
func SearchAllDatabases(ctx context.Context, query string) <-chan Result {
    db1Results := searchDB1(ctx, query)
    db2Results := searchDB2(ctx, query)
    db3Results := searchDB3(ctx, query)

    return FanIn(ctx, db1Results, db2Results, db3Results)
}

// Merge logs from multiple services
func MergeLogs(ctx context.Context, services []string) <-chan LogEntry {
    var channels []<-chan LogEntry

    for _, service := range services {
        ch := streamLogs(ctx, service)
        channels = append(channels, ch)
    }

    return FanIn(ctx, channels...)
}

// Parallel API calls with merge
func FetchFromMultipleAPIs(ctx context.Context, endpoints []string) <-chan APIResponse {
    var channels []<-chan APIResponse

    for _, endpoint := range endpoints {
        ch := fetchFromAPI(ctx, endpoint)
        channels = append(channels, ch)
    }

    return FanIn(ctx, channels...)
}

// Worker pool with fan-out/fan-in
func ProcessWithWorkerPool(ctx context.Context, items <-chan Item, workers int) <-chan Result {
    // Fan-out: distribute work to workers
    var workerChannels []<-chan Result

    for i := 0; i < workers; i++ {
        workerCh := processWorker(ctx, items)
        workerChannels = append(workerChannels, workerCh)
    }

    // Fan-in: merge results from all workers
    return FanIn(ctx, workerChannels...)
}

func processWorker(ctx context.Context, items <-chan Item) <-chan Result {
    out := make(chan Result)
    go func() {
        defer close(out)
        for {
            select {
            case <-ctx.Done():
                return
            case item, ok := <-items:
                if !ok {
                    return
                }
                result := processItem(item)
                select {
                case <-ctx.Done():
                    return
                case out <- result:
                }
            }
        }
    }()
    return out
}

// Event stream multiplexing
func MergeEventStreams(ctx context.Context) <-chan Event {
    clicks := streamClickEvents(ctx)
    pageViews := streamPageViewEvents(ctx)
    purchases := streamPurchaseEvents(ctx)

    return FanIn(ctx, clicks, pageViews, purchases)
}

// Distributed scraping
func ScrapeWebsites(ctx context.Context, urls []string) <-chan ScrapedData {
    var scrapers []<-chan ScrapedData

    for _, url := range urls {
        scraper := scrapeWebsite(ctx, url)
        scrapers = append(scrapers, scraper)
    }

    return FanIn(ctx, scrapers...)
}

// Priority-based fan-in
func FanInWithPriority(ctx context.Context, high, normal, low <-chan int) <-chan int {
    out := make(chan int)

    go func() {
        defer close(out)
        for {
            select {
            case <-ctx.Done():
                return
            case v, ok := <-high:
                if !ok {
                    high = nil
                    if normal == nil && low == nil {
                        return
                    }
                    continue
                }
                select {
                case <-ctx.Done():
                    return
                case out <- v:
                }
            default:
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-high:
                    if !ok {
                        high = nil
                        continue
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                    }
                case v, ok := <-normal:
                    if !ok {
                        normal = nil
                        continue
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                    }
                case v, ok := <-low:
                    if !ok {
                        low = nil
                        continue
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                    }
                }
            }
        }
    }()
    return out
}

// Ordered fan-in (preserve order by timestamp)
func FanInOrdered(ctx context.Context, ins ...<-chan TimestampedValue) <-chan int {
    out := make(chan int)

    go func() {
        defer close(out)
        pq := &PriorityQueue{}
        heap.Init(pq)

        // Initial pull from all channels
        for _, in := range ins {
            if in == nil {
                continue
            }
            select {
            case v, ok := <-in:
                if ok {
                    heap.Push(pq, v)
                }
            default:
            }
        }

        // Process in timestamp order
        for pq.Len() > 0 {
            item := heap.Pop(pq).(TimestampedValue)
            select {
            case <-ctx.Done():
                return
            case out <- item.Value:
            }
        }
    }()
    return out
}

// Complete fan-out/fan-in example
func CompleteExample() {
    ctx := context.Background()

    // Source
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    // Fan-out: split to 3 workers
    worker1 := SquareStage(1)(ctx, source)
    worker2 := SquareStage(1)(ctx, source)
    worker3 := SquareStage(1)(ctx, source)

    // Fan-in: merge results
    merged := FanIn(ctx, worker1, worker2, worker3)

    // Further processing
    multiplied := MultiplyStage(2)(ctx, merged)

    for v := range multiplied {
        fmt.Println(v)
    }
}

// Rate-limited fan-in
func FanInRateLimited(ctx context.Context, limit time.Duration, ins ...<-chan int) <-chan int {
    merged := FanIn(ctx, ins...)
    out := make(chan int)

    go func() {
        defer close(out)
        ticker := time.NewTicker(limit)
        defer ticker.Stop()

        for v := range merged {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                select {
                case <-ctx.Done():
                    return
                case out <- v:
                }
            }
        }
    }()
    return out
}
\`\`\`

**Real-World Benefits:**
- **Parallel Processing:** Combine results from concurrent operations
- **Multi-Source:** Aggregate data from multiple sources
- **Load Distribution:** Balance load across multiple producers
- **Event Processing:** Merge event streams from different sources

**Common Use Cases:**
- **Distributed Search:** Merge results from multiple search nodes
- **Log Aggregation:** Combine logs from multiple services
- **Parallel Scraping:** Merge data from multiple scrapers
- **Worker Pools:** Collect results from parallel workers
- **Event Streams:** Multiplex events from different sources
- **API Aggregation:** Merge responses from multiple APIs

**Fan-In Patterns:**
- **Simple Merge:** All channels equal priority (FanIn)
- **Priority Merge:** Some channels have priority
- **Ordered Merge:** Maintain order by timestamp/sequence
- **Rate-Limited:** Control output rate
- **Buffered:** Add buffering for burst handling

**Performance Considerations:**
- **Goroutines:** One per input channel (O(N) goroutines)
- **Contention:** Output channel can be bottleneck
- **Backpressure:** Slow consumer blocks all producers
- **Buffering:** Consider buffered output for better throughput

**Fan-Out/Fan-In Pattern:**
1. **Single Source** → Fan-out to multiple workers
2. **Parallel Processing** by workers
3. **Fan-in** to collect results
4. **Continue Pipeline** with merged data

Without FanIn, merging multiple channels requires complex select statements or sequential processing, making parallel aggregation difficult and error-prone.`,	order: 7,
	translations: {
		ru: {
			title: 'Объединение каналов',
			description: `Реализуйте **FanIn**, который объединяет несколько входных каналов в один выходной канал с отменой контекста.

**Требования:**
1. Создайте функцию \`FanIn(ctx context.Context, ins ...<-chan int) <-chan int\`
2. Пропускайте nil каналы в списке входов
3. Создайте выходной канал (небуферизованный)
4. Запустите одну горутину на входной канал для пересылки значений
5. Каждый пересыльщик учитывает отмену контекста (вложенный select)
6. Используйте sync.WaitGroup для отслеживания всех пересыльщиков
7. Закройте выход когда все пересыльщики закончат
8. Верните выходной канал немедленно

**Пример:**
\`\`\`go
ctx := context.Background()

ch1 := Gen(1, 2, 3)
ch2 := Gen(10, 20, 30)
ch3 := Gen(100, 200, 300)

merged := FanIn(ctx, ch1, ch2, ch3)

for v := range merged {
    fmt.Println(v)
}
// Вывод: 1 10 100 2 20 200 3 30 300 (порядок может меняться)
\`\`\`

**Ограничения:**
- Должен пропускать nil каналы
- Должен использовать одну горутину на входной канал
- Должен использовать вложенный select для отмены
- Должен использовать sync.WaitGroup для координации`,
			hint1: `Создайте функцию forward которая принимает входной канал и использует вложенный select для пересылки значений в выход учитывая ctx.Done().`,
			hint2: `Итерируйте по ins, пропускайте nil каналы, wg.Add(1) для каждого валидного канала, и запускайте go forward(in). Отдельная горутина должна wg.Wait() затем close(out).`,
			whyItMatters: `FanIn (слияние) обеспечивает объединение нескольких источников данных в один поток, необходимо для агрегации результатов параллельной обработки или слияния потоков событий.

**Почему Fan-In важен:**
- **Агрегация:** Объединение результатов от параллельных workers
- **Мультиплексирование:** Слияние множества источников событий
- **Load Balancing:** Распределение работы по источникам
- **Композиция Pipeline:** Соединение разделённых pipeline обратно

**Production паттерны:**
\`\`\`go
// Слияние результатов поиска из нескольких БД
func SearchAllDatabases(ctx context.Context, query string) <-chan Result {
    db1Results := searchDB1(ctx, query)
    db2Results := searchDB2(ctx, query)
    db3Results := searchDB3(ctx, query)

    return FanIn(ctx, db1Results, db2Results, db3Results)
}

// Слияние логов от нескольких сервисов
func MergeLogs(ctx context.Context, services []string) <-chan LogEntry {
    var channels []<-chan LogEntry

    for _, service := range services {
        ch := streamLogs(ctx, service)
        channels = append(channels, ch)
    }

    return FanIn(ctx, channels...)
}

// Worker pool с fan-out/fan-in
func ProcessWithWorkerPool(ctx context.Context, items <-chan Item, workers int) <-chan Result {
    // Fan-out: распределение работы по workers
    var workerChannels []<-chan Result

    for i := 0; i < workers; i++ {
        workerCh := processWorker(ctx, items)
        workerChannels = append(workerChannels, workerCh)
    }

    // Fan-in: слияние результатов от всех workers
    return FanIn(ctx, workerChannels...)
}

// Параллельные API вызовы со слиянием
func FetchFromMultipleAPIs(ctx context.Context, endpoints []string) <-chan APIResponse {
    var channels []<-chan APIResponse

    for _, endpoint := range endpoints {
        ch := fetchFromAPI(ctx, endpoint)
        channels = append(channels, ch)
    }

    return FanIn(ctx, channels...)
}

// Мультиплексирование потоков событий
func MergeEventStreams(ctx context.Context) <-chan Event {
    clicks := streamClickEvents(ctx)
    pageViews := streamPageViewEvents(ctx)
    purchases := streamPurchaseEvents(ctx)

    return FanIn(ctx, clicks, pageViews, purchases)
}

// Распределённый scraping
func ScrapeWebsites(ctx context.Context, urls []string) <-chan ScrapedData {
    var scrapers []<-chan ScrapedData

    for _, url := range urls {
        scraper := scrapeWebsite(ctx, url)
        scrapers = append(scrapers, scraper)
    }

    return FanIn(ctx, scrapers...)
}

// Fan-in с приоритетами
func FanInWithPriority(ctx context.Context, high, normal, low <-chan int) <-chan int {
    out := make(chan int)

    go func() {
        defer close(out)
        for {
            select {
            case <-ctx.Done():
                return
            case v, ok := <-high:
                if !ok {
                    high = nil
                    if normal == nil && low == nil {
                        return
                    }
                    continue
                }
                select {
                case <-ctx.Done():
                    return
                case out <- v:
                }
            default:
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-high:
                    if !ok {
                        high = nil
                        continue
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                    }
                case v, ok := <-normal:
                    if !ok {
                        normal = nil
                        continue
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                    }
                case v, ok := <-low:
                    if !ok {
                        low = nil
                        continue
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                    }
                }
            }
        }
    }()
    return out
}

// Полный пример fan-out/fan-in
func CompleteExample() {
    ctx := context.Background()

    // Источник
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    // Fan-out: разделение на 3 workers
    worker1 := SquareStage(1)(ctx, source)
    worker2 := SquareStage(1)(ctx, source)
    worker3 := SquareStage(1)(ctx, source)

    // Fan-in: слияние результатов
    merged := FanIn(ctx, worker1, worker2, worker3)

    // Дальнейшая обработка
    multiplied := MultiplyStage(2)(ctx, merged)

    for v := range multiplied {
        fmt.Println(v)
    }
}

// Fan-in с ограничением скорости
func FanInRateLimited(ctx context.Context, limit time.Duration, ins ...<-chan int) <-chan int {
    merged := FanIn(ctx, ins...)
    out := make(chan int)

    go func() {
        defer close(out)
        ticker := time.NewTicker(limit)
        defer ticker.Stop()

        for v := range merged {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                select {
                case <-ctx.Done():
                    return
                case out <- v:
                }
            }
        }
    }()
    return out
}
\`\`\`

**Реальные преимущества:**
- **Параллельная обработка:** Объединение результатов от конкурентных операций
- **Множественные источники:** Агрегация данных из нескольких источников
- **Распределение нагрузки:** Балансировка нагрузки по нескольким producers
- **Обработка событий:** Слияние потоков событий из разных источников

**Типичные сценарии использования:**
- **Распределённый поиск:** Слияние результатов от нескольких search узлов
- **Агрегация логов:** Объединение логов от нескольких сервисов
- **Параллельный scraping:** Слияние данных от нескольких scrapers
- **Worker Pools:** Сбор результатов от параллельных workers
- **Потоки событий:** Мультиплексирование событий из разных источников
- **Агрегация API:** Слияние ответов от нескольких APIs

**Паттерны Fan-In:**
- **Простое слияние:** Все каналы с равным приоритетом (FanIn)
- **Приоритетное слияние:** Некоторые каналы имеют приоритет
- **Упорядоченное слияние:** Сохранение порядка по timestamp/последовательности
- **Rate-Limited:** Контроль скорости вывода
- **Буферизованное:** Добавление буферизации для обработки всплесков

**Соображения производительности:**
- **Горутины:** Одна на входной канал (O(N) горутин)
- **Конкуренция:** Выходной канал может быть бутылочным горлышком
- **Backpressure:** Медленный consumer блокирует всех producers
- **Буферизация:** Рассмотрите буферизованный выход для лучшей пропускной способности

**Паттерн Fan-Out/Fan-In:**
1. **Единый источник** → Fan-out на несколько workers
2. **Параллельная обработка** workers
3. **Fan-in** для сбора результатов
4. **Продолжение Pipeline** со слитыми данными

Без FanIn слияние нескольких каналов требует сложных select statements или последовательной обработки, делая параллельную агрегацию сложной и подверженной ошибкам.`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

func FanIn(ctx context.Context, ins ...<-chan int) <-chan int {
	out := make(chan int)                                       // Создаём выходной канал
	var wg sync.WaitGroup                                       // WaitGroup для пересыльщиков
	forward := func(in <-chan int) {                            // Функция пересыльщика
		defer wg.Done()                                     // Отмечаем завершение
		for {                                               // Бесконечный цикл
			select {
			case <-ctx.Done():                          // Контекст отменён
				return                              // Выходим из пересыльщика
			case v, ok := <-in:                         // Читаем из входа
				if !ok {                            // Канал закрыт
					return                      // Выходим из пересыльщика
				}
				select {
				case <-ctx.Done():                  // Проверяем перед отправкой
					return                      // Выходим из пересыльщика
				case out <- v:                      // Пересылаем значение
				}
			}
		}
	}
	for _, in := range ins {                                    // Итерируемся по входам
		if in == nil {                                      // Пропускаем nil каналы
			continue
		}
		wg.Add(1)                                           // Добавляем в WaitGroup
		go forward(in)                                      // Запускаем пересыльщик
	}
	go func() {                                                 // Горутина закрывателя
		wg.Wait()                                           // Ждём всех пересыльщиков
		close(out)                                          // Закрываем выход
	}()
	return out                                                  // Возвращаем немедленно
}`
		},
		uz: {
			title: 'Kanallarni birlashtirish',
			description: `Kontekst bekor qilish bilan bir nechta kirish kanallarini bitta chiqish kanaliga birlashtiruvchi **FanIn** ni amalga oshiring.

**Talablar:**
1. \`FanIn(ctx context.Context, ins ...<-chan int) <-chan int\` funksiyasini yarating
2. Kirish ro'yxatidagi nil kanallarni o'tkazib yuboring
3. Chiqish kanalini yarating (buferlanmagan)
4. Qiymatlarni uzatish uchun har bir kirish kanali uchun bitta goroutine ishga tushiring
5. Har bir uzatuvchi kontekst bekor qilinishini hurmat qiladi (ichki select)
6. Barcha uzatuvchilarni kuzatish uchun sync.WaitGroup dan foydalaning
7. Barcha uzatuvchilar tugaganda chiqishni yoping
8. Chiqish kanalini darhol qaytaring

**Misol:**
\`\`\`go
ctx := context.Background()

ch1 := Gen(1, 2, 3)
ch2 := Gen(10, 20, 30)
ch3 := Gen(100, 200, 300)

merged := FanIn(ctx, ch1, ch2, ch3)

for v := range merged {
    fmt.Println(v)
}
// Natija: 1 10 100 2 20 200 3 30 300 (tartib o'zgarishi mumkin)
\`\`\`

**Cheklovlar:**
- nil kanallarni o'tkazib yuborishi kerak
- Har bir kirish kanali uchun bitta goroutinedan foydalanishi kerak
- Bekor qilish uchun ichki selectdan foydalanishi kerak
- Muvofiqlashtirish uchun sync.WaitGroup dan foydalanishi kerak`,
			hint1: `Kirish kanalini qabul qiladigan va ctx.Done() ni hisobga olgan holda chiqishga qiymatlarni uzatish uchun ichki selectdan foydalanadigan forward funksiyasini yarating.`,
			hint2: `ins bo'ylab takrorlang, nil kanallarni o'tkazib yuboring, har bir yaroqli kanal uchun wg.Add(1), va go forward(in) ishga tushiring. Alohida goroutine wg.Wait() keyin close(out) qilishi kerak.`,
			whyItMatters: `FanIn (merge) bir nechta ma'lumot manbalarini bitta oqimga birlashtirishni ta'minlaydi, parallel qayta ishlash natijalarini agregatsiya qilish yoki hodisa oqimlarini birlashtirish uchun zarur.

**Nima uchun Fan-In muhim:**
- **Agregatsiya:** Parallel workerlardan natijalarni birlashtirish
- **Multipleksing:** Ko'p hodisa manbalarini birlashtirish
- **Load Balancing:** Manbalar bo'yicha ishni taqsimlash
- **Pipeline kompozitsiya:** Ajratilgan pipelinelarni qaytarib birlashtirish

**Production patternlar:**
\`\`\`go
// Ko'p DBdan qidiruv natijalarini birlashtirish
func SearchAllDatabases(ctx context.Context, query string) <-chan Result {
    db1Results := searchDB1(ctx, query)
    db2Results := searchDB2(ctx, query)
    db3Results := searchDB3(ctx, query)

    return FanIn(ctx, db1Results, db2Results, db3Results)
}

// Ko'p xizmatlardan loglarni birlashtirish
func MergeLogs(ctx context.Context, services []string) <-chan LogEntry {
    var channels []<-chan LogEntry

    for _, service := range services {
        ch := streamLogs(ctx, service)
        channels = append(channels, ch)
    }

    return FanIn(ctx, channels...)
}

// Fan-out/fan-in bilan worker pool
func ProcessWithWorkerPool(ctx context.Context, items <-chan Item, workers int) <-chan Result {
    // Fan-out: ishni workerlarga taqsimlash
    var workerChannels []<-chan Result

    for i := 0; i < workers; i++ {
        workerCh := processWorker(ctx, items)
        workerChannels = append(workerChannels, workerCh)
    }

    // Fan-in: barcha workerlardan natijalarni birlashtirish
    return FanIn(ctx, workerChannels...)
}

// Bir nechta APIlardan parallel chaqiruvlar
func FetchFromMultipleAPIs(ctx context.Context, endpoints []string) <-chan APIResponse {
    var channels []<-chan APIResponse

    for _, endpoint := range endpoints {
        ch := fetchFromAPI(ctx, endpoint)
        channels = append(channels, ch)
    }

    return FanIn(ctx, channels...)
}

// Hodisa oqimlarini multipleksing qilish
func MergeEventStreams(ctx context.Context) <-chan Event {
    clicks := streamClickEvents(ctx)
    pageViews := streamPageViewEvents(ctx)
    purchases := streamPurchaseEvents(ctx)

    return FanIn(ctx, clicks, pageViews, purchases)
}

// Taqsimlangan scraping
func ScrapeWebsites(ctx context.Context, urls []string) <-chan ScrapedData {
    var scrapers []<-chan ScrapedData

    for _, url := range urls {
        scraper := scrapeWebsite(ctx, url)
        scrapers = append(scrapers, scraper)
    }

    return FanIn(ctx, scrapers...)
}

// Ustuvorlik bilan fan-in
func FanInWithPriority(ctx context.Context, high, normal, low <-chan int) <-chan int {
    out := make(chan int)

    go func() {
        defer close(out)
        for {
            select {
            case <-ctx.Done():
                return
            case v, ok := <-high:
                if !ok {
                    high = nil
                    if normal == nil && low == nil {
                        return
                    }
                    continue
                }
                select {
                case <-ctx.Done():
                    return
                case out <- v:
                }
            default:
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-high:
                    if !ok {
                        high = nil
                        continue
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                    }
                case v, ok := <-normal:
                    if !ok {
                        normal = nil
                        continue
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                    }
                case v, ok := <-low:
                    if !ok {
                        low = nil
                        continue
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                    }
                }
            }
        }
    }()
    return out
}

// To'liq fan-out/fan-in misoli
func CompleteExample() {
    ctx := context.Background()

    // Manba
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    // Fan-out: 3 ta workerga ajratish
    worker1 := SquareStage(1)(ctx, source)
    worker2 := SquareStage(1)(ctx, source)
    worker3 := SquareStage(1)(ctx, source)

    // Fan-in: natijalarni birlashtirish
    merged := FanIn(ctx, worker1, worker2, worker3)

    // Keyingi qayta ishlash
    multiplied := MultiplyStage(2)(ctx, merged)

    for v := range multiplied {
        fmt.Println(v)
    }
}

// Tezlik cheklangan fan-in
func FanInRateLimited(ctx context.Context, limit time.Duration, ins ...<-chan int) <-chan int {
    merged := FanIn(ctx, ins...)
    out := make(chan int)

    go func() {
        defer close(out)
        ticker := time.NewTicker(limit)
        defer ticker.Stop()

        for v := range merged {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                select {
                case <-ctx.Done():
                    return
                case out <- v:
                }
            }
        }
    }()
    return out
}
\`\`\`

**Haqiqiy foydalari:**
- **Parallel qayta ishlash:** Konkurent operatsiyalardan natijalarni birlashtirish
- **Ko'p manbalar:** Bir nechta manbalardan ma'lumotlarni agregatsiya qilish
- **Yuk taqsimi:** Bir nechta producerlar bo'ylab yukni balanslash
- **Hodisalarni qayta ishlash:** Turli manbalardan hodisa oqimlarini birlashtirish

**Umumiy foydalanish stsenariylari:**
- **Taqsimlangan qidiruv:** Bir nechta qidiruv tugunlaridan natijalarni birlashtirish
- **Loglarni agregatsiya:** Bir nechta xizmatlardan loglarni birlashtirish
- **Parallel scraping:** Bir nechta scraperlardan ma'lumotlarni birlashtirish
- **Worker Pools:** Parallel workerlardan natijalarni yig'ish
- **Hodisa oqimlari:** Turli manbalardan hodisalarni multipleksing qilish
- **API agregatsiya:** Bir nechta APIlardan javoblarni birlashtirish

**Fan-In patternlari:**
- **Oddiy merge:** Barcha kanallar teng ustuvorlik bilan (FanIn)
- **Ustuvor merge:** Ba'zi kanallar ustuvorlikka ega
- **Tartiblangan merge:** Timestamp/ketma-ketlik bo'yicha tartibni saqlash
- **Rate-Limited:** Chiqish tezligini boshqarish
- **Buferli:** Portlashlarni qayta ishlash uchun buferlash qo'shish

**Samaradorlik mulohazalari:**
- **Goroutinelar:** Har bir kirish kanali uchun bitta (O(N) goroutinelar)
- **Raqobat:** Chiqish kanali bo'g'in bo'lishi mumkin
- **Backpressure:** Sekin consumer barcha producerlarni bloklaydi
- **Buferlash:** Yaxshiroq o'tkazuvchanlik uchun buferli chiqishni ko'rib chiqing

**Fan-Out/Fan-In patterni:**
1. **Yagona manba** → Bir nechta workerlarga fan-out
2. **Parallel qayta ishlash** workerlar tomonidan
3. **Fan-in** natijalarni yig'ish uchun
4. **Pipelineni davom ettirish** birlashtirilgan ma'lumotlar bilan

FanIn bo'lmasa, bir nechta kanallarni birlashtirish murakkab select statementlar yoki ketma-ket qayta ishlashni talab qiladi, parallel agregatsiyani murakkab va xatolarga moyil qiladi.`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

func FanIn(ctx context.Context, ins ...<-chan int) <-chan int {
	out := make(chan int)                                       // Chiqish kanalini yaratamiz
	var wg sync.WaitGroup                                       // Uzatuvchilar uchun WaitGroup
	forward := func(in <-chan int) {                            // Uzatuvchi funksiyasi
		defer wg.Done()                                     // Tugaganini belgilaymiz
		for {                                               // Cheksiz tsikl
			select {
			case <-ctx.Done():                          // Kontekst bekor qilindi
				return                              // Uzatuvchidan chiqamiz
			case v, ok := <-in:                         // Kirishdan o'qiymiz
				if !ok {                            // Kanal yopilgan
					return                      // Uzatuvchidan chiqamiz
				}
				select {
				case <-ctx.Done():                  // Yuborishdan oldin tekshiramiz
					return                      // Uzatuvchidan chiqamiz
				case out <- v:                      // Qiymatni uzatamiz
				}
			}
		}
	}
	for _, in := range ins {                                    // Kirishlar bo'ylab iteratsiya qilamiz
		if in == nil {                                      // nil kanallarni o'tkazib yuboramiz
			continue
		}
		wg.Add(1)                                           // WaitGroup ga qo'shamiz
		go forward(in)                                      // Uzatuvchini ishga tushiramiz
	}
	go func() {                                                 // Yopuvchi goroutine
		wg.Wait()                                           // Barcha uzatuvchilarni kutamiz
		close(out)                                          // Chiqishni yopamiz
	}()
	return out                                                  // Darhol qaytaramiz
}`
		}
	}
};

export default task;
