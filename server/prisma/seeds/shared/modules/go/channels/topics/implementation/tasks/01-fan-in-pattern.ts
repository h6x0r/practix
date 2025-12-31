import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-channels-fan-in',
	title: 'Fan-In Channel Pattern',
	difficulty: 'medium',	tags: ['go', 'channels', 'concurrency', 'patterns'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement the fan-in pattern to merge multiple input channels into a single output channel.

**Requirements:**
1. **FanIn**: Merge multiple channels into one output channel
2. **Context Awareness**: Stop when context is canceled
3. **Goroutine Management**: Launch goroutine per input channel
4. **Clean Shutdown**: Close output channel when all inputs are closed

**Fan-In Pattern:**
\`\`\`go
func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
    out := make(chan T)

    var wg sync.WaitGroup
    for _, in := range ins {
        wg.Add(1)
        go func(ch <-chan T) {
            defer wg.Done()
            // Forward values from ch to out
            // Stop on ctx.Done() or channel close
        }(in)
    }

    go func() {
        wg.Wait()      // Wait for all forwarders
        close(out)     // Close output when done
    }()

    return out
}
\`\`\`

**Key Concepts:**
- Fan-in merges multiple sources into single destination
- Each input gets dedicated goroutine for forwarding
- sync.WaitGroup tracks when all inputs are exhausted
- Context cancellation stops forwarding immediately
- Output channel closed only after all inputs closed

**Example Usage:**
\`\`\`go
// Merging results from multiple workers
func ProcessParallel(items []Item) <-chan Result {
    // Split work across 3 channels
    ch1 := processChunk(items[0:100])
    ch2 := processChunk(items[100:200])
    ch3 := processChunk(items[200:300])

    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // Merge all results into single channel
    results := FanIn(ctx, ch1, ch2, ch3)

    for result := range results {
        // Process merged results
        fmt.Println(result)
    }
}

// Aggregating logs from multiple services
func AggregateLogs(services []string) {
    ctx := context.Background()

    var logChannels []<-chan LogEntry
    for _, svc := range services {
        logChannels = append(logChannels, streamLogs(svc))
    }

    // Single channel with all logs
    allLogs := FanIn(ctx, logChannels...)

    for log := range allLogs {
        saveToDatabase(log)
    }
}
\`\`\`

**Pattern Flow:**
\`\`\`
Input 1 → Goroutine 1 ┐
Input 2 → Goroutine 2 ├→ Output Channel
Input 3 → Goroutine 3 ┘

// All inputs merged into single stream
\`\`\`

**Constraints:**
- Must handle variable number of input channels
- Must respect context cancellation
- Must not leak goroutines
- Must close output channel exactly once
- Must handle nil context (use Background)`,
	initialCode: `package channelsx

import (
	"context"
	"sync"
)

// TODO: Implement FanIn
// Merge multiple input channels into one output
// Launch goroutine for each input channel
// Use WaitGroup to track completion
// Close output when all inputs are done
func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
	// TODO: Implement
}`,
	solutionCode: `package channelsx

import (
	"context"
	"sync"
)

func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
	out := make(chan T)                          // create merged output channel

	var wg sync.WaitGroup
	forward := func(in <-chan T) {
		defer wg.Done()                           // signal completion when done

		for {
			select {
			case <-ctx.Done():                     // context canceled, stop forwarding
				return
			case v, ok := <-in:
				if !ok {                            // input channel closed
					return
				}
				select {
				case <-ctx.Done():                  // check again before sending
					return
				case out <- v:                      // forward value to output
				}
			}
		}
	}

	for _, in := range ins {
		wg.Add(1)
		go forward(in)                            // launch forwarder per input
	}

	go func() {
		wg.Wait()                                  // wait for all forwarders to finish
		close(out)                                 // close output channel once
	}()

	return out                                    // return read-only channel
}`,
	testCode: `package channelsx

import (
	"context"
	"testing"
	"time"
)

func TestFanIn_SingleChannel(t *testing.T) {
	ctx := context.Background()
	ch := make(chan int, 2)
	ch <- 1
	ch <- 2
	close(ch)

	out := FanIn(ctx, ch)
	results := []int{}
	for v := range out {
		results = append(results, v)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 values, got %d", len(results))
	}
}

func TestFanIn_MultipleChannels(t *testing.T) {
	ctx := context.Background()
	ch1 := make(chan int, 1)
	ch2 := make(chan int, 1)
	ch3 := make(chan int, 1)
	ch1 <- 1
	ch2 <- 2
	ch3 <- 3
	close(ch1)
	close(ch2)
	close(ch3)

	out := FanIn(ctx, ch1, ch2, ch3)
	count := 0
	for range out {
		count++
	}
	if count != 3 {
		t.Errorf("expected 3 values, got %d", count)
	}
}

func TestFanIn_EmptyChannels(t *testing.T) {
	ctx := context.Background()
	ch1 := make(chan int)
	ch2 := make(chan int)
	close(ch1)
	close(ch2)

	out := FanIn(ctx, ch1, ch2)
	count := 0
	for range out {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 values, got %d", count)
	}
}

func TestFanIn_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	ch := make(chan int)

	out := FanIn(ctx, ch)
	cancel()

	time.Sleep(50 * time.Millisecond)
	select {
	case _, ok := <-out:
		if ok {
			t.Error("expected channel to be closed after context cancellation")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("timeout waiting for channel to close")
	}
}

func TestFanIn_NoChannels(t *testing.T) {
	ctx := context.Background()
	out := FanIn(ctx, []<-chan int{}...)

	select {
	case _, ok := <-out:
		if ok {
			t.Error("expected channel to be immediately closed")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("timeout waiting for channel to close")
	}
}

func TestFanIn_LargeVolume(t *testing.T) {
	ctx := context.Background()
	ch1 := make(chan int, 50)
	ch2 := make(chan int, 50)

	for i := 0; i < 50; i++ {
		ch1 <- i
		ch2 <- i + 100
	}
	close(ch1)
	close(ch2)

	out := FanIn(ctx, ch1, ch2)
	count := 0
	for range out {
		count++
	}
	if count != 100 {
		t.Errorf("expected 100 values, got %d", count)
	}
}

func TestFanIn_OutputCloses(t *testing.T) {
	ctx := context.Background()
	ch := make(chan int, 1)
	ch <- 42
	close(ch)

	out := FanIn(ctx, ch)
	<-out

	_, ok := <-out
	if ok {
		t.Error("expected output channel to be closed")
	}
}

func TestFanIn_ContextTimeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	ch := make(chan int)
	out := FanIn(ctx, ch)

	time.Sleep(100 * time.Millisecond)
	select {
	case _, ok := <-out:
		if ok {
			t.Error("expected channel to be closed after timeout")
		}
	default:
	}
}

func TestFanIn_MixedSpeeds(t *testing.T) {
	ctx := context.Background()
	fast := make(chan int, 10)
	slow := make(chan int, 1)

	for i := 0; i < 10; i++ {
		fast <- i
	}
	slow <- 100
	close(fast)
	close(slow)

	out := FanIn(ctx, fast, slow)
	count := 0
	for range out {
		count++
	}
	if count != 11 {
		t.Errorf("expected 11 values, got %d", count)
	}
}

func TestFanIn_StringType(t *testing.T) {
	ctx := context.Background()
	ch1 := make(chan string, 2)
	ch2 := make(chan string, 2)
	ch1 <- "hello"
	ch1 <- "world"
	ch2 <- "foo"
	ch2 <- "bar"
	close(ch1)
	close(ch2)

	out := FanIn(ctx, ch1, ch2)
	count := 0
	for range out {
		count++
	}
	if count != 4 {
		t.Errorf("expected 4 values, got %d", count)
	}
}
`,
			hint1: `Create a goroutine for each input channel that forwards values to the output channel using select with ctx.Done().`,
			hint2: `Use sync.WaitGroup to track all goroutines. Launch a cleanup goroutine that waits and closes the output channel.`,
			whyItMatters: `The fan-in pattern is essential for aggregating concurrent operations in production Go systems.

**Why This Matters:**

**1. Aggregating Parallel Work**
When you split work across multiple goroutines, fan-in merges results:
\`\`\`go
// Problem: Processing 1M records takes too long
func ProcessRecords(records []Record) []Result {
    var results []Result
    for _, rec := range records {
        results = append(results, process(rec))
    }
    return results
}
// Time: 100 seconds (sequential)

// Solution: Parallel processing with fan-in
func ProcessRecordsParallel(records []Record) []Result {
    numWorkers := runtime.NumCPU()
    chunkSize := len(records) / numWorkers

    var channels []<-chan Result
    for i := 0; i < numWorkers; i++ {
        start := i * chunkSize
        end := start + chunkSize
        if i == numWorkers-1 {
            end = len(records)
        }

        ch := processChunk(records[start:end])
        channels = append(channels, ch)
    }

    ctx := context.Background()
    merged := FanIn(ctx, channels...)

    var results []Result
    for result := range merged {
        results = append(results, result)
    }
    return results
}
// Time: 12 seconds on 8-core CPU (8.3x faster!)
\`\`\`

**2. Real Production Scenario: Log Aggregation**
Microservices architecture with 20 services:
\`\`\`go
// Collect logs from all services in real-time

type LogEntry struct {
    Service   string
    Timestamp time.Time
    Level     string
    Message   string
}

func AggregateServiceLogs(serviceURLs []string) {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    var logChannels []<-chan LogEntry
    for _, url := range serviceURLs {
        // Each service streams logs via channel
        logCh := streamLogsFromService(url)
        logChannels = append(logChannels, logCh)
    }

    // Single channel with all logs
    allLogs := FanIn(ctx, logChannels...)

    // Process unified log stream
    for log := range allLogs {
        // Send to Elasticsearch, CloudWatch, etc.
        indexLog(log)

        // Alert on errors
        if log.Level == "ERROR" {
            alertOps(log)
        }
    }
}

// Before fan-in: Had to poll each service individually
// After fan-in: Real-time unified log stream
// Result: Mean time to detect errors: 5 minutes → 2 seconds
\`\`\`

**3. Distributed Search**
Search across multiple data sources simultaneously:
\`\`\`go
func SearchAll(query string) []SearchResult {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()

    // Launch searches in parallel
    googleResults := searchGoogle(ctx, query)
    bingResults := searchBing(ctx, query)
    internalResults := searchInternal(ctx, query)

    // Merge all results as they arrive
    allResults := FanIn(ctx, googleResults, bingResults, internalResults)

    var results []SearchResult
    for result := range allResults {
        results = append(results, result)

        // Stop early if we have enough
        if len(results) >= 20 {
            cancel() // Stop other searches
            break
        }
    }

    return results
}

// Sequential: 6 seconds (2s per source)
// Fan-in parallel: 2 seconds (all at once)
// 3x faster response time!
\`\`\`

**4. Financial Data Aggregation**
Collecting market data from multiple exchanges:
\`\`\`go
type PriceUpdate struct {
    Exchange string
    Symbol   string
    Price    float64
    Volume   int64
}

func MonitorPrices(exchanges []Exchange, symbol string) {
    ctx := context.Background()

    var priceChannels []<-chan PriceUpdate
    for _, exchange := range exchanges {
        // Each exchange streams price updates
        priceCh := exchange.StreamPrices(symbol)
        priceChannels = append(priceChannels, priceCh)
    }

    // Unified price feed from all exchanges
    allPrices := FanIn(ctx, priceChannels...)

    bestBid := 0.0
    bestAsk := math.MaxFloat64

    for update := range allPrices {
        // Update best bid/ask across all exchanges
        if update.Price > bestBid {
            bestBid = update.Price
        }
        if update.Price < bestAsk {
            bestAsk = update.Price
        }

        // Arbitrage opportunity?
        if bestAsk < bestBid {
            executeArbitrage(bestBid, bestAsk)
        }
    }
}

// Before: Checked exchanges sequentially (missed opportunities)
// After: Real-time unified feed (caught arbitrage in milliseconds)
// Impact: $50K additional revenue per month
\`\`\`

**5. Graceful Shutdown with Context**
\`\`\`go
func ProcessWithTimeout(sources []<-chan Data) {
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    merged := FanIn(ctx, sources...)

    for data := range merged {
        process(data)
    }
    // Automatically stops after 30 seconds
    // No goroutine leaks!
}

// Context cancellation propagates to all forwarders
// All goroutines clean up properly
// No resource leaks even with hundreds of channels
\`\`\`

**6. Why WaitGroup is Critical**
\`\`\`go
// WRONG - closes output too early
func BrokenFanIn[T any](ins ...<-chan T) <-chan T {
    out := make(chan T)
    for _, in := range ins {
        go func(ch <-chan T) {
            for v := range ch {
                out <- v
            }
        }(in)
    }
    close(out) // BUG! Closes before goroutines finish
    return out
}

// RIGHT - waits for all goroutines
func CorrectFanIn[T any](ins ...<-chan T) <-chan T {
    out := make(chan T)
    var wg sync.WaitGroup

    for _, in := range ins {
        wg.Add(1)
        go func(ch <-chan T) {
            defer wg.Done()
            for v := range ch {
                out <- v
            }
        }(in)
    }

    go func() {
        wg.Wait()     // Wait for ALL forwarders
        close(out)    // Now safe to close
    }()

    return out
}
\`\`\`

**7. Testing Fan-In**
\`\`\`go
func TestFanIn(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), time.Second)
    defer cancel()

    // Create test channels
    ch1 := make(chan int, 2)
    ch2 := make(chan int, 2)

    ch1 <- 1
    ch1 <- 2
    ch2 <- 3
    close(ch1)
    close(ch2)

    // Merge channels
    out := FanIn(ctx, ch1, ch2)

    // Collect results
    var got []int
    for v := range out {
        got = append(got, v)
    }

    // Verify all values received
    if len(got) != 3 {
        t.Errorf("expected 3 values, got %d", len(got))
    }
}
\`\`\`

**Real-World Impact:**
IoT platform collecting sensor data:
- **Before**: Polled 10,000 sensors sequentially
  - Total collection time: 50 minutes
  - Stale data by the time it's processed
  - Missed critical alerts

- **After**: Fan-in pattern with parallel collection
  - Total collection time: 30 seconds (100x faster!)
  - Real-time data processing
  - Alerts fire within seconds
  - Prevented 3 equipment failures worth $200K

**Production Best Practices:**
1. Always use context for cancellation
2. Use WaitGroup to coordinate shutdown
3. Close output channel in separate goroutine
4. Check ctx.Done() before sending to output
5. Handle nil context gracefully
6. Test with -race flag for data races
7. Monitor goroutine counts in production

Fan-in is a fundamental pattern for building scalable, concurrent Go applications. Master it, and you'll be able to aggregate work from any number of sources efficiently.`,	order: 0,
	translations: {
		ru: {
			title: 'Паттерн Fan-In',
			solutionCode: `package channelsx

import (
	"context"
	"sync"
)

func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
	out := make(chan T)                          // создаём объединённый выходной канал

	var wg sync.WaitGroup
	forward := func(in <-chan T) {
		defer wg.Done()                           // сигнализируем завершение когда готово

		for {
			select {
			case <-ctx.Done():                     // контекст отменён, прекращаем пересылку
				return
			case v, ok := <-in:
				if !ok {                            // входной канал закрыт
					return
				}
				select {
				case <-ctx.Done():                  // проверяем снова перед отправкой
					return
				case out <- v:                      // пересылаем значение в выход
				}
			}
		}
	}

	for _, in := range ins {
		wg.Add(1)
		go forward(in)                            // запускаем forwarder для каждого входа
	}

	go func() {
		wg.Wait()                                  // ждём завершения всех forwarderов
		close(out)                                 // закрываем выходной канал один раз
	}()

	return out                                    // возвращаем канал только для чтения
}`,
			description: `Реализуйте fan-in паттерн для объединения нескольких входных каналов в один выходной.

**Требования:**
1. **FanIn**: Объединить несколько каналов в один выходной канал
2. **Context Awareness**: Остановка при отмене контекста
3. **Goroutine Management**: Запустить горутину для каждого входного канала
4. **Clean Shutdown**: Закрыть выходной канал когда все входы закрыты

**Fan-In паттерн:**
\`\`\`go
func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
    out := make(chan T)

    var wg sync.WaitGroup
    for _, in := range ins {
        wg.Add(1)
        go func(ch <-chan T) {
            defer wg.Done()
            // Пересылка значений из ch в out
            // Остановка при ctx.Done() или закрытии канала
        }(in)
    }

    go func() {
        wg.Wait()      // Ожидание всех forwarderов
        close(out)     // Закрытие выхода когда готово
    }()

    return out
}
\`\`\`

**Поток паттерна:**
\`\`\`
Input 1 → Goroutine 1 ┐
Input 2 → Goroutine 2 ├→ Output Channel
Input 3 → Goroutine 3 ┘

// Все входы объединены в единый поток
\`\`\`

**Ключевые концепции:**
- Fan-in объединяет множество источников в одно назначение
- Каждый вход получает выделенную горутину для пересылки
- sync.WaitGroup отслеживает когда все входы исчерпаны
- Отмена контекста останавливает пересылку немедленно
- Выходной канал закрывается только после закрытия всех входов

**Пример использования:**
\`\`\`go
// Объединение результатов от нескольких workerов
func ProcessParallel(items []Item) <-chan Result {
    ch1 := processChunk(items[0:100])
    ch2 := processChunk(items[100:200])
    ch3 := processChunk(items[200:300])

    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // Объединяем все результаты в один канал
    results := FanIn(ctx, ch1, ch2, ch3)

    for result := range results {
        fmt.Println(result)
    }
}
\`\`\`

**Ограничения:**
- Должен обрабатывать переменное число входных каналов
- Должен уважать отмену контекста
- Не должен утекать горутины
- Должен закрывать выходной канал ровно один раз
- Должен обрабатывать nil context (использовать Background)`,
			hint1: `Создайте горутину для каждого входного канала, которая пересылает значения в выходной канал используя select с ctx.Done().`,
			hint2: `Используйте sync.WaitGroup для отслеживания всех горутин. Запустите cleanup горутину, которая ждёт и закрывает выходной канал.`,
			whyItMatters: `Продакшен паттерн Fan-in критичен для агрегации конкурентных операций в production Go системах.

**Практические преимущества:**

**1. Агрегация параллельной работы**
Когда вы разделяете работу на множество горутин, fan-in объединяет результаты:
\`\`\`go
// Проблема: Обработка 1M записей занимает слишком много времени
func ProcessRecords(records []Record) []Result {
    var results []Result
    for _, rec := range records {
        results = append(results, process(rec))
    }
    return results
}
// Время: 100 секунд (последовательная обработка)

// Решение: Параллельная обработка с fan-in
func ProcessRecordsParallel(records []Record) []Result {
    numWorkers := runtime.NumCPU()
    chunkSize := len(records) / numWorkers

    var channels []<-chan Result
    for i := 0; i < numWorkers; i++ {
        start := i * chunkSize
        end := start + chunkSize
        if i == numWorkers-1 {
            end = len(records)
        }

        ch := processChunk(records[start:end])
        channels = append(channels, ch)
    }

    ctx := context.Background()
    merged := FanIn(ctx, channels...)

    var results []Result
    for result := range merged {
        results = append(results, result)
    }
    return results
}
// Время: 12 секунд на 8-ядерном CPU (8.3x быстрее!)
\`\`\`

**2. Реальный Production сценарий: Агрегация логов**
Микросервисная архитектура с 20 сервисами:
\`\`\`go
// Сбор логов со всех сервисов в реальном времени

type LogEntry struct {
    Service   string
    Timestamp time.Time
    Level     string
    Message   string
}

func AggregateServiceLogs(serviceURLs []string) {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    var logChannels []<-chan LogEntry
    for _, url := range serviceURLs {
        // Каждый сервис стримит логи через канал
        logCh := streamLogsFromService(url)
        logChannels = append(logChannels, logCh)
    }

    // Единый канал со всеми логами
    allLogs := FanIn(ctx, logChannels...)

    // Обработка объединённого потока логов
    for log := range allLogs {
        // Отправка в Elasticsearch, CloudWatch, и т.д.
        indexLog(log)

        // Алерт при ошибках
        if log.Level == "ERROR" {
            alertOps(log)
        }
    }
}

// До fan-in: Нужно было опрашивать каждый сервис отдельно
// После fan-in: Объединённый поток логов в реальном времени
// Результат: Среднее время обнаружения ошибок: 5 минут → 2 секунды
\`\`\`

**3. Распределённый поиск**
Поиск по нескольким источникам данных одновременно:
\`\`\`go
func SearchAll(query string) []SearchResult {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()

    // Запуск поиска параллельно
    googleResults := searchGoogle(ctx, query)
    bingResults := searchBing(ctx, query)
    internalResults := searchInternal(ctx, query)

    // Объединение всех результатов по мере поступления
    allResults := FanIn(ctx, googleResults, bingResults, internalResults)

    var results []SearchResult
    for result := range allResults {
        results = append(results, result)

        // Остановка раньше если достаточно результатов
        if len(results) >= 20 {
            cancel() // Остановка других поисков
            break
        }
    }

    return results
}

// Последовательно: 6 секунд (2с на источник)
// Fan-in параллельно: 2 секунды (все одновременно)
// 3x быстрее время ответа!
\`\`\`

**4. Агрегация финансовых данных**
Сбор рыночных данных с нескольких бирж:
\`\`\`go
type PriceUpdate struct {
    Exchange string
    Symbol   string
    Price    float64
    Volume   int64
}

func MonitorPrices(exchanges []Exchange, symbol string) {
    ctx := context.Background()

    var priceChannels []<-chan PriceUpdate
    for _, exchange := range exchanges {
        // Каждая биржа стримит обновления цен
        priceCh := exchange.StreamPrices(symbol)
        priceChannels = append(priceChannels, priceCh)
    }

    // Объединённый поток цен со всех бирж
    allPrices := FanIn(ctx, priceChannels...)

    bestBid := 0.0
    bestAsk := math.MaxFloat64

    for update := range allPrices {
        // Обновление лучшего bid/ask по всем биржам
        if update.Price > bestBid {
            bestBid = update.Price
        }
        if update.Price < bestAsk {
            bestAsk = update.Price
        }

        // Возможность арбитража?
        if bestAsk < bestBid {
            executeArbitrage(bestBid, bestAsk)
        }
    }
}

// До: Проверка бирж последовательно (упущенные возможности)
// После: Объединённый поток в реальном времени (арбитраж за миллисекунды)
// Эффект: $50K дополнительной выручки в месяц
\`\`\`

**5. Graceful Shutdown с Context**
\`\`\`go
func ProcessWithTimeout(sources []<-chan Data) {
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    merged := FanIn(ctx, sources...)

    for data := range merged {
        process(data)
    }
    // Автоматическая остановка через 30 секунд
    // Без утечек горутин!
}

// Отмена контекста распространяется на все forwarders
// Все горутины правильно завершаются
// Нет утечек ресурсов даже с сотнями каналов
\`\`\`

**6. Почему WaitGroup критичен**
\`\`\`go
// НЕПРАВИЛЬНО - закрывает output слишком рано
func BrokenFanIn[T any](ins ...<-chan T) <-chan T {
    out := make(chan T)
    for _, in := range ins {
        go func(ch <-chan T) {
            for v := range ch {
                out <- v
            }
        }(in)
    }
    close(out) // БАГ! Закрывается до завершения горутин
    return out
}

// ПРАВИЛЬНО - ждёт все горутины
func CorrectFanIn[T any](ins ...<-chan T) <-chan T {
    out := make(chan T)
    var wg sync.WaitGroup

    for _, in := range ins {
        wg.Add(1)
        go func(ch <-chan T) {
            defer wg.Done()
            for v := range ch {
                out <- v
            }
        }(in)
    }

    go func() {
        wg.Wait()     // Ждём ВСЕ forwarders
        close(out)    // Теперь безопасно закрывать
    }()

    return out
}
\`\`\`

**7. Тестирование Fan-In**
\`\`\`go
func TestFanIn(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), time.Second)
    defer cancel()

    // Создание тестовых каналов
    ch1 := make(chan int, 2)
    ch2 := make(chan int, 2)

    ch1 <- 1
    ch1 <- 2
    ch2 <- 3
    close(ch1)
    close(ch2)

    // Объединение каналов
    out := FanIn(ctx, ch1, ch2)

    // Сбор результатов
    var got []int
    for v := range out {
        got = append(got, v)
    }

    // Проверка что все значения получены
    if len(got) != 3 {
        t.Errorf("ожидалось 3 значения, получено %d", len(got))
    }
}
\`\`\`

**Реальное влияние:**
IoT платформа собирает данные с сенсоров:
- **До**: Опрос 10,000 сенсоров последовательно
  - Общее время сбора: 50 минут
  - Устаревшие данные к моменту обработки
  - Упущенные критические алерты

- **После**: Fan-in паттерн с параллельным сбором
  - Общее время сбора: 30 секунд (100x быстрее!)
  - Обработка данных в реальном времени
  - Алерты срабатывают за секунды
  - Предотвращено 3 поломки оборудования стоимостью $200K

**Production Best Practices:**
1. Всегда используйте context для отмены
2. Используйте WaitGroup для координации shutdown
3. Закрывайте output канал в отдельной горутине
4. Проверяйте ctx.Done() перед отправкой в output
5. Обрабатывайте nil context gracefully
6. Тестируйте с флагом -race для обнаружения data races
7. Мониторьте количество горутин в production

Fan-in — фундаментальный паттерн для построения масштабируемых конкурентных Go приложений. Освойте его, и вы сможете эффективно агрегировать работу из любого количества источников.`
		},
		uz: {
			title: `Fan-In patterni`,
			solutionCode: `package channelsx

import (
	"context"
	"sync"
)

func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
	out := make(chan T)                          // birlashtirilgan chiqish kanalini yaratamiz

	var wg sync.WaitGroup
	forward := func(in <-chan T) {
		defer wg.Done()                           // tayyor bo'lganda tugashni signallash

		for {
			select {
			case <-ctx.Done():                     // kontekst bekor qilindi, yo'naltirishni to'xtatamiz
				return
			case v, ok := <-in:
				if !ok {                            // kirish kanali yopilgan
					return
				}
				select {
				case <-ctx.Done():                  // yuborishdan oldin yana tekshiramiz
					return
				case out <- v:                      // qiymatni chiqishga yo'naltiramiz
				}
			}
		}
	}

	for _, in := range ins {
		wg.Add(1)
		go forward(in)                            // har bir kirish uchun forwarder ishga tushiramiz
	}

	go func() {
		wg.Wait()                                  // barcha forwarderlar tugashini kutamiz
		close(out)                                 // chiqish kanalini bir marta yopamiz
	}()

	return out                                    // faqat o'qish uchun kanalni qaytaramiz
}`,
			description: `Bir nechta kirish kanallarini bitta chiqish kanaliga birlashtirish uchun fan-in patternini amalga oshiring.

**Talablar:**
1. **FanIn**: Bir nechta kanallarni bitta chiqish kanaliga birlashtirish
2. **Context Awareness**: Kontekst bekor qilinganda to'xtatish
3. **Goroutine Management**: Har bir kirish kanali uchun gorutin ishga tushirish
4. **Clean Shutdown**: Barcha kirishlar yopilganda chiqish kanalini yopish

**Fan-In pattern:**
\`\`\`go
func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
    out := make(chan T)

    var wg sync.WaitGroup
    for _, in := range ins {
        wg.Add(1)
        go func(ch <-chan T) {
            defer wg.Done()
            // Qiymatlarni ch dan out ga yo'naltirish
            // ctx.Done() yoki kanal yopilganda to'xtatish
        }(in)
    }

    go func() {
        wg.Wait()      // Barcha forwarderlarni kutish
        close(out)     // Tayyor bo'lganda chiqishni yopish
    }()

    return out
}
\`\`\`

**Pattern oqimi:**
\`\`\`
Input 1 → Goroutine 1 ┐
Input 2 → Goroutine 2 ├→ Output Channel
Input 3 → Goroutine 3 ┘

// Barcha kirishlar yagona oqimga birlashtirilgan
\`\`\`

**Asosiy tushunchalar:**
- Fan-in ko'p manbalarni bitta manzilga birlashtiradi
- Har bir kirish yo'naltirish uchun ajratilgan gorutinni oladi
- sync.WaitGroup barcha kirishlar tugaganini kuzatadi
- Kontekst bekor qilish yo'naltirishni darhol to'xtatadi
- Chiqish kanali faqat barcha kirishlar yopilgandan keyin yopiladi

**Foydalanish misoli:**
\`\`\`go
// Bir nechta workerlardan natijalarni birlashtirish
func ProcessParallel(items []Item) <-chan Result {
    ch1 := processChunk(items[0:100])
    ch2 := processChunk(items[100:200])
    ch3 := processChunk(items[200:300])

    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // Barcha natijalarni bitta kanalga birlashtiramiz
    results := FanIn(ctx, ch1, ch2, ch3)

    for result := range results {
        fmt.Println(result)
    }
}
\`\`\`

**Cheklovlar:**
- O'zgaruvchan miqdordagi kirish kanallarini qayta ishlashi kerak
- Kontekst bekor qilishni hurmat qilishi kerak
- Gorutinlarni sizib chiqarmasligi kerak
- Chiqish kanalini aynan bir marta yopishi kerak
- nil kontekstni qayta ishlashi kerak (Background ishlatish)`,
			hint1: `Har bir kirish kanali uchun gorutin yarating, u qiymatlarni ctx.Done() bilan select ishlatib chiqish kanaliga yo'naltiradi.`,
			hint2: `Barcha gorutinlarni kuzatish uchun sync.WaitGroup dan foydalaning. Kutadigan va chiqish kanalini yopadigan cleanup gorutinni ishga tushiring.`,
			whyItMatters: `Ishlab chiqarish patterni Fan-in production Go tizimlarida parallel operatsiyalarni agregatsiya qilish uchun muhimdir.

**Amaliy foydalari:**

**1. Parallel ishni agregatsiya qilish**
Ishni ko'p gorutinlarga bo'lganingizda, fan-in natijalarni birlashtiradi:
\`\`\`go
// Muammo: 1M yozuvni qayta ishlash juda ko'p vaqt oladi
func ProcessRecords(records []Record) []Result {
    var results []Result
    for _, rec := range records {
        results = append(results, process(rec))
    }
    return results
}
// Vaqt: 100 soniya (ketma-ket qayta ishlash)

// Yechim: Fan-in bilan parallel qayta ishlash
func ProcessRecordsParallel(records []Record) []Result {
    numWorkers := runtime.NumCPU()
    chunkSize := len(records) / numWorkers

    var channels []<-chan Result
    for i := 0; i < numWorkers; i++ {
        start := i * chunkSize
        end := start + chunkSize
        if i == numWorkers-1 {
            end = len(records)
        }

        ch := processChunk(records[start:end])
        channels = append(channels, ch)
    }

    ctx := context.Background()
    merged := FanIn(ctx, channels...)

    var results []Result
    for result := range merged {
        results = append(results, result)
    }
    return results
}
// Vaqt: 8 yadroli CPU da 12 soniya (8.3x tezroq!)
\`\`\`

**2. Haqiqiy Production stsenariy: Log agregatsiyasi**
20 ta xizmatli mikroxizmat arxitekturasi:
\`\`\`go
// Barcha xizmatlardan real vaqtda loglarni to'plash

type LogEntry struct {
    Service   string
    Timestamp time.Time
    Level     string
    Message   string
}

func AggregateServiceLogs(serviceURLs []string) {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    var logChannels []<-chan LogEntry
    for _, url := range serviceURLs {
        // Har bir xizmat kanal orqali loglarni strim qiladi
        logCh := streamLogsFromService(url)
        logChannels = append(logChannels, logCh)
    }

    // Barcha loglar bilan yagona kanal
    allLogs := FanIn(ctx, logChannels...)

    // Birlashtirilgan log oqimini qayta ishlash
    for log := range allLogs {
        // Elasticsearch, CloudWatch va boshqalarga yuborish
        indexLog(log)

        // Xatolarda ogohlantirish
        if log.Level == "ERROR" {
            alertOps(log)
        }
    }
}

// Fan-in dan oldin: Har bir xizmatni alohida so'rash kerak edi
// Fan-in dan keyin: Real vaqtda birlashtirilgan log oqimi
// Natija: Xatolarni aniqlash o'rtacha vaqti: 5 daqiqa → 2 soniya
\`\`\`

**3. Taqsimlangan qidiruv**
Bir vaqtning o'zida bir nechta ma'lumot manbalaridan qidirish:
\`\`\`go
func SearchAll(query string) []SearchResult {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()

    // Qidiruvni parallel ravishda boshlash
    googleResults := searchGoogle(ctx, query)
    bingResults := searchBing(ctx, query)
    internalResults := searchInternal(ctx, query)

    // Barcha natijalarni kelishi bilan birlashtirish
    allResults := FanIn(ctx, googleResults, bingResults, internalResults)

    var results []SearchResult
    for result := range allResults {
        results = append(results, result)

        // Etarli natijalar bo'lsa ertaroq to'xtatish
        if len(results) >= 20 {
            cancel() // Boshqa qidiruvlarni to'xtatish
            break
        }
    }

    return results
}

// Ketma-ket: 6 soniya (har bir manba uchun 2s)
// Fan-in parallel: 2 soniya (barchasi bir vaqtda)
// 3x tezroq javob vaqti!
\`\`\`

**4. Moliyaviy ma'lumotlarni agregatsiya qilish**
Bir nechta birjalardan bozor ma'lumotlarini to'plash:
\`\`\`go
type PriceUpdate struct {
    Exchange string
    Symbol   string
    Price    float64
    Volume   int64
}

func MonitorPrices(exchanges []Exchange, symbol string) {
    ctx := context.Background()

    var priceChannels []<-chan PriceUpdate
    for _, exchange := range exchanges {
        // Har bir birja narx yangilanishlarini strim qiladi
        priceCh := exchange.StreamPrices(symbol)
        priceChannels = append(priceChannels, priceCh)
    }

    // Barcha birjalardan birlashtirilgan narx oqimi
    allPrices := FanIn(ctx, priceChannels...)

    bestBid := 0.0
    bestAsk := math.MaxFloat64

    for update := range allPrices {
        // Barcha birjalar bo'yicha eng yaxshi bid/ask ni yangilash
        if update.Price > bestBid {
            bestBid = update.Price
        }
        if update.Price < bestAsk {
            bestAsk = update.Price
        }

        // Arbitraj imkoniyati?
        if bestAsk < bestBid {
            executeArbitrage(bestBid, bestAsk)
        }
    }
}

// Oldin: Birjalarni ketma-ket tekshirish (o'tkazib yuborilgan imkoniyatlar)
// Keyin: Real vaqtda birlashtirilgan oqim (millisekundlarda arbitraj)
// Ta'sir: Oyiga $50K qo'shimcha daromad
\`\`\`

**5. Context bilan Graceful Shutdown**
\`\`\`go
func ProcessWithTimeout(sources []<-chan Data) {
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    merged := FanIn(ctx, sources...)

    for data := range merged {
        process(data)
    }
    // 30 soniyadan keyin avtomatik to'xtatish
    // Gorutin sizib chiqishlari yo'q!
}

// Kontekst bekor qilish barcha forwarderlarga tarqaladi
// Barcha gorutinlar to'g'ri tugaydi
// Yuzlab kanallar bo'lsa ham resurs sizib chiqishlari yo'q
\`\`\`

**6. WaitGroup nima uchun muhim**
\`\`\`go
// NOTO'G'RI - outputni juda erta yopadi
func BrokenFanIn[T any](ins ...<-chan T) <-chan T {
    out := make(chan T)
    for _, in := range ins {
        go func(ch <-chan T) {
            for v := range ch {
                out <- v
            }
        }(in)
    }
    close(out) // BUG! Gorutinlar tugashidan oldin yopiladi
    return out
}

// TO'G'RI - barcha gorutinlarni kutadi
func CorrectFanIn[T any](ins ...<-chan T) <-chan T {
    out := make(chan T)
    var wg sync.WaitGroup

    for _, in := range ins {
        wg.Add(1)
        go func(ch <-chan T) {
            defer wg.Done()
            for v := range ch {
                out <- v
            }
        }(in)
    }

    go func() {
        wg.Wait()     // BARCHA forwarderlarnini kutish
        close(out)    // Endi yopish xavfsiz
    }()

    return out
}
\`\`\`

**7. Fan-In ni test qilish**
\`\`\`go
func TestFanIn(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), time.Second)
    defer cancel()

    // Test kanallarini yaratish
    ch1 := make(chan int, 2)
    ch2 := make(chan int, 2)

    ch1 <- 1
    ch1 <- 2
    ch2 <- 3
    close(ch1)
    close(ch2)

    // Kanallarni birlashtirish
    out := FanIn(ctx, ch1, ch2)

    // Natijalarni to'plash
    var got []int
    for v := range out {
        got = append(got, v)
    }

    // Barcha qiymatlar olinganini tekshirish
    if len(got) != 3 {
        t.Errorf("3 ta qiymat kutilgan edi, %d ta olindi", len(got))
    }
}
\`\`\`

**Haqiqiy ta'sir:**
IoT platformasi sensorlardan ma'lumot to'playdi:
- **Oldin**: 10,000 sensorni ketma-ket so'rash
  - Umumiy to'plash vaqti: 50 daqiqa
  - Qayta ishlash vaqtida eskirgan ma'lumotlar
  - O'tkazib yuborilgan kritik ogohlantirishlar

- **Keyin**: Parallel to'plash bilan Fan-in pattern
  - Umumiy to'plash vaqti: 30 soniya (100x tezroq!)
  - Real vaqtda ma'lumotlarni qayta ishlash
  - Ogohlantirishlar soniyalarda ishga tushadi
  - $200K qiymatidagi 3 ta uskunaning buzilishi oldini olindi

**Production Best Practices:**
1. Bekor qilish uchun har doim context dan foydalaning
2. Shutdown ni koordinatsiya qilish uchun WaitGroup dan foydalaning
3. Output kanalini alohida gorutinda yoping
4. Outputga yuborishdan oldin ctx.Done() ni tekshiring
5. nil kontekstni to'g'ri qayta ishlang
6. Data race larni aniqlash uchun -race flagi bilan test qiling
7. Productionda gorutinlar sonini monitor qiling

Fan-in — kengaytiriladigan parallel Go ilovalarini qurish uchun fundamental pattern. Uni o'zlashtiring va istalgan miqdordagi manbalardan ishni samarali agregatsiya qilishingiz mumkin bo'ladi.`
		}
	}
};

export default task;
