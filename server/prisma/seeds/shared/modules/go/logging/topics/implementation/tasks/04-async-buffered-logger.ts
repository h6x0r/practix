import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-logging-async-buffered',
	title: 'Async Non-Blocking Logger with Buffered Channel',
	difficulty: 'hard',
	tags: ['go', 'logging', 'async', 'channels', 'performance'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a high-performance async logger using buffered channels to prevent logging from blocking critical code paths.

**Requirements:**
1. **LogEntry**: Struct holding timestamp, level, and message
2. **Start**: Launch background goroutine that processes log entries from buffered channel
3. **Stop**: Gracefully shut down logger, flushing all pending entries
4. **AsyncLog**: Non-blocking function that sends log entry to channel
5. **workerLoop**: Background goroutine that receives from channel and writes logs

**Async Logging Pattern:**
\`\`\`go
// Traditional synchronous logging
func ProcessRequest(r *Request) {
    log.Print("Processing...") // BLOCKS until disk write completes (1-10ms)
    result := doWork(r)         // Work waits for log
    log.Print("Done")           // BLOCKS again
}

// Async logging
func ProcessRequest(r *Request) {
    AsyncLog("Processing...") // Returns immediately (~100ns)
    result := doWork(r)        // Work starts instantly
    AsyncLog("Done")           // Returns immediately
    // Background goroutine handles actual disk writes
}
\`\`\`

**Why Async Logging:**
- **Non-Blocking**: Main code doesn't wait for disk I/O
- **Batching**: Background worker can batch writes for efficiency
- **Performance**: 100x-1000x faster than synchronous logging
- **Throughput**: Handle 100K+ logs/second without slowing down app

**Key Concepts:**
- Buffered channel acts as queue (\`make(chan LogEntry, 1000)\`)
- Producer (AsyncLog) never blocks unless buffer full
- Consumer (workerLoop) processes entries in background
- Graceful shutdown ensures no log loss

**Implementation Flow:**
\`\`\`go
1. Start() creates buffered channel and launches workerLoop goroutine
2. AsyncLog() creates LogEntry and sends to channel (non-blocking)
3. workerLoop() receives entries and writes to log.Print()
4. Stop() closes channel and waits for workerLoop to finish
\`\`\`

**Example Production Usage:**
\`\`\`go
func main() {
    Start(1000) // Buffer size: 1000 entries
    defer Stop() // Ensure graceful shutdown

    http.HandleFunc("/api/users", handleUsers)
    log.Fatal(http.ListenAndServe(":8080", nil))
}

func handleUsers(w http.ResponseWriter, r *http.Request) {
    AsyncLog("INFO", "Request started: %s", r.URL.Path)

    users := fetchUsers() // Critical work
    AsyncLog("DEBUG", "Fetched %d users", len(users))

    json.NewEncoder(w).Encode(users)
    AsyncLog("INFO", "Response sent")

    // All 3 AsyncLog calls took ~300ns total
    // Actual disk writes happen in background
    // Request latency: ZERO logging overhead!
}
\`\`\`

**Constraints:**
- Use buffered channel with configurable size (passed to Start)
- LogEntry must have: time (time.Time), level (string), message (string)
- Start goroutine with \`go workerLoop()\`
- Stop must: close channel, wait for goroutine to finish processing
- Use \`time.Now()\` to timestamp each entry
- Format output: \`[timestamp] [level] message\``,
	initialCode: `package loggingx

import (
	"log"
	"time"
)

type LogEntry struct {
	Time    time.Time
	Level   string
	Message string
}

var (
	logChan chan LogEntry
	done    chan struct{} // Signal when worker finished
)

// TODO: Implement Start
// 1. Create buffered channel: make(chan LogEntry, bufferSize)
// 2. Create done channel: make(chan struct{})
// 3. Launch background worker: go workerLoop()
func Start(bufferSize int) {
	// TODO: Implement
}

// TODO: Implement Stop
// 1. Close logChan to signal worker to finish
// 2. Wait for worker completion: <-done
func Stop() {
	// TODO: Implement
}

// TODO: Implement AsyncLog
// 1. Create LogEntry with time.Now(), level, formatted message
// 2. Use fmt.Sprintf to format message with args
// 3. Send to channel (non-blocking - use select with default case)
// 4. If channel full, drop or handle error (for simplicity: drop)
func AsyncLog(level string, format string, args ...interface{}) {
	// TODO: Implement
}

// TODO: Implement workerLoop
// 1. Range over logChan (automatically stops when closed)
// 2. For each entry, format and print:
//    log.Printf("[%s] [%s] %s", timestamp, level, message)
// 3. When channel closed and drained, signal done: close(done)
func workerLoop() {
	// TODO: Implement
}`,
	solutionCode: `package loggingx

import (
	"fmt"
	"log"
	"time"
)

type LogEntry struct {
	Time    time.Time
	Level   string
	Message string
}

var (
	logChan chan LogEntry
	done    chan struct{}
)

func Start(bufferSize int) {
	logChan = make(chan LogEntry, bufferSize) // buffered channel allows producers to continue without waiting
	done = make(chan struct{})                // coordination channel for graceful shutdown
	go workerLoop()                           // launch background goroutine to process log entries
}

func Stop() {
	close(logChan) // signal worker no more entries coming - worker will drain and exit
	<-done         // block until worker signals completion - ensures no log loss
}

func AsyncLog(level string, format string, args ...interface{}) {
	entry := LogEntry{
		Time:    time.Now(),                   // capture timestamp immediately for accurate event timing
		Level:   level,                        // store severity level for filtering
		Message: fmt.Sprintf(format, args...), // format message once in caller goroutine
	}

	select {
	case logChan <- entry: // attempt non-blocking send to channel
		// success - entry queued for background processing
	default:
		// channel full - drop entry to avoid blocking caller
		// production systems might increment dropped_logs metric here
	}
}

func workerLoop() {
	for entry := range logChan { // range exits when channel closed and drained
		log.Printf("[%s] [%s] %s",
			entry.Time.Format("2006-01-02 15:04:05.000"), // format timestamp for human readability
			entry.Level,                                    // include severity level
			entry.Message)                                  // log formatted message
	}
	close(done) // signal Stop() that all entries processed and worker exiting
}`,
	testCode: `package loggingx

import (
	"testing"
	"time"
)

func Test1(t *testing.T) {
	Start(10)
	AsyncLog("INFO", "test message")
	Stop()
}

func Test2(t *testing.T) {
	Start(100)
	for i := 0; i < 50; i++ {
		AsyncLog("DEBUG", "message %d", i)
	}
	Stop()
}

func Test3(t *testing.T) {
	Start(5)
	for i := 0; i < 10; i++ {
		AsyncLog("INFO", "overflow %d", i)
	}
	Stop()
}

func Test4(t *testing.T) {
	Start(10)
	AsyncLog("ERROR", "error message")
	AsyncLog("WARN", "warn message")
	AsyncLog("INFO", "info message")
	Stop()
}

func Test5(t *testing.T) {
	Start(10)
	Stop()
}

func Test6(t *testing.T) {
	Start(10)
	AsyncLog("INFO", "with args %s %d", "str", 42)
	Stop()
}

func Test7(t *testing.T) {
	Start(100)
	for i := 0; i < 100; i++ {
		AsyncLog("DEBUG", "batch %d", i)
	}
	time.Sleep(10 * time.Millisecond)
	Stop()
}

func Test8(t *testing.T) {
	Start(1)
	AsyncLog("INFO", "first")
	AsyncLog("INFO", "second")
	AsyncLog("INFO", "third")
	Stop()
}

func Test9(t *testing.T) {
	Start(10)
	AsyncLog("FATAL", "fatal message %v", "critical")
	Stop()
}

func Test10(t *testing.T) {
	Start(50)
	done := make(chan bool)
	go func() {
		for i := 0; i < 25; i++ {
			AsyncLog("INFO", "goroutine %d", i)
		}
		done <- true
	}()
	for i := 0; i < 25; i++ {
		AsyncLog("DEBUG", "main %d", i)
	}
	<-done
	Stop()
}
`,
	hint1: `In Start, create buffered channel with make(chan LogEntry, bufferSize) and launch goroutine with go workerLoop().`,
	hint2: `In workerLoop, use for entry := range logChan to process entries until channel closed. Format timestamp with entry.Time.Format("2006-01-02 15:04:05").`,
	whyItMatters: `Async logging is critical for high-performance systems where logging overhead can become a bottleneck affecting user experience.

**Why This Matters:**

**1. The Synchronous Logging Problem**
\`\`\`go
// Synchronous logging (traditional approach)
func handleRequest(w http.ResponseWriter, r *http.Request) {
    log.Print("Request started")     // BLOCKS for disk I/O: 1-5ms

    user := authenticateUser(r)      // Waits for log
    log.Print("User authenticated")  // BLOCKS again: 1-5ms

    data := fetchData(user)          // Waits for log
    log.Print("Data fetched")        // BLOCKS again: 1-5ms

    json.NewEncoder(w).Encode(data)
}

// Total logging overhead: 3-15ms PER REQUEST
// At 1000 requests/second: 3-15 SECONDS of CPU time wasted!
\`\`\`

**The Problem:**
- Disk writes are SLOW (1-10ms each)
- Each log.Print() blocks until write completes
- Your critical business logic waits for logging
- High-traffic services can't afford this overhead

**2. Async Logging Solution**
\`\`\`go
// Async logging
func handleRequest(w http.ResponseWriter, r *http.Request) {
    AsyncLog("INFO", "Request started")     // Returns in ~100ns

    user := authenticateUser(r)             // Starts immediately
    AsyncLog("INFO", "User authenticated")  // Returns in ~100ns

    data := fetchData(user)                 // Starts immediately
    AsyncLog("INFO", "Data fetched")        // Returns in ~100ns

    json.NewEncoder(w).Encode(data)
}

// Total logging overhead: ~300ns (0.0003ms)
// 50,000x FASTER than synchronous!
\`\`\`

**How it Works:**
1. AsyncLog puts entry in buffered channel (instant)
2. Background goroutine reads from channel
3. Background goroutine writes to disk
4. Main goroutine continues immediately

**3. Real Performance Numbers**

**E-commerce API handling 10,000 requests/second:**

**Before Async Logging (Synchronous):**
\`\`\`
Average request latency: 45ms
5 log statements per request
5ms logging overhead per request
11% of latency is JUST LOGGING!

Server capacity: 10,000 req/s (CPU maxed out)
Logging CPU usage: 40% (!)
\`\`\`

**After Async Logging:**
\`\`\`
Average request latency: 40ms (-11%)
5 log statements per request
0.005ms logging overhead per request
0.01% of latency is logging

Server capacity: 15,000 req/s (+50%)
Logging CPU usage: 5% (background goroutine)
\`\`\`

**Improvements:**
- Request latency: -11%
- Throughput: +50%
- Logging CPU: 40% → 5% (-88%)

**4. Buffered Channel Pattern**
\`\`\`go
// Without buffer (blocking)
logChan := make(chan LogEntry)
logChan <- entry // BLOCKS if worker slow

// With buffer (non-blocking)
logChan := make(chan LogEntry, 1000)
logChan <- entry // Instant unless 1000 entries already queued

// Buffer acts as shock absorber:
// - Traffic spikes: buffer absorbs burst
// - Smooth processing: worker processes at steady rate
\`\`\`

**Buffer Size Selection:**
- Too small (10): Fills up during traffic spikes, logs dropped
- Too large (100K): Memory waste, slow shutdown
- Recommended: 1000-10000 entries
  - 1 LogEntry ≈ 100 bytes
  - 10000 entries = 1MB memory (acceptable)

**5. Graceful Shutdown Pattern**
\`\`\`go
func main() {
    Start(1000)
    defer Stop() // CRITICAL: Ensures logs flushed before exit

    // ... server code ...
}

// Without Stop():
// - Program exits
// - Buffer has 500 pending entries
// - 500 logs LOST!

// With Stop():
// 1. Close channel (no new entries)
// 2. Worker processes remaining 500 entries
// 3. Worker signals done
// 4. Program exits safely
// - ZERO logs lost!
\`\`\`

**6. Real-World Production Scenario**

**Scenario:** Payment processing service

**Traffic Pattern:**
- Normal: 1000 payments/second
- Black Friday: 50,000 payments/second (50x spike!)
- Each payment: 10 log statements

**Without Async Logging:**
\`\`\`
Black Friday morning:
- 50,000 req/s × 10 logs × 2ms per log
- = 1,000,000ms of logging per second
- = Server needs 1000 CPU cores just for logging!
- IMPOSSIBLE! Server crashes!
\`\`\`

**With Async Logging:**
\`\`\`
Black Friday morning:
- 50,000 req/s × 10 logs × 0.0001ms per log
- = 50ms of logging overhead per second
- = 0.1% of one CPU core
- Server handles it easily!

Background worker:
- Processes 500,000 log entries/second
- Uses 1 CPU core at 30% utilization
- Batches disk writes for efficiency
- NO performance impact on payment processing!
\`\`\`

**Result:**
- Payment processing: ZERO downtime
- All logs captured: 500K entries/second
- User experience: Unchanged
- Revenue: $5M/hour protected

**7. Trade-offs and Considerations**

**Advantages:**
- 100-1000x faster than synchronous logging
- Non-blocking: doesn't slow down critical code
- Natural batching: more efficient disk I/O
- Scales to millions of logs/second

**Disadvantages:**
- Logs slightly delayed (milliseconds)
- Can lose logs if buffer overflows
- Requires graceful shutdown
- More complex than synchronous

**When to Use Async Logging:**
- ✅ High-traffic services (>100 req/s)
- ✅ Low-latency requirements (<100ms)
- ✅ Many log statements per request
- ✅ Production systems with strict SLAs

**When NOT to Use:**
- ❌ Debugging during development (logs might not appear before crash)
- ❌ Critical security logs (use sync to guarantee write)
- ❌ Low-traffic services (<10 req/s) - overhead not worth complexity

**8. Advanced: Monitoring**
\`\`\`go
var (
    droppedLogs atomic.Int64
)

func AsyncLog(level string, format string, args ...interface{}) {
    entry := LogEntry{...}

    select {
    case logChan <- entry:
        // Logged successfully
    default:
        droppedLogs.Add(1) // Track drops for monitoring

        // Alert if drops exceed threshold
        if droppedLogs.Load() > 10000 {
            // Send alert: "Logging buffer full - increase size!"
        }
    }
}
\`\`\`

**Production Monitoring:**
- Buffer utilization: len(logChan) / cap(logChan)
- Dropped logs count
- Worker goroutine health
- Logs processed per second

**The Bottom Line:**
Async logging transforms logging from a performance bottleneck into a zero-overhead observability tool. For high-traffic services, it's the difference between meeting SLAs and falling over during peak traffic.`,
	order: 3,
	translations: {
		ru: {
			title: 'Асинхронный буферизованный логгер',
			solutionCode: `package loggingx

import (
	"fmt"
	"log"
	"time"
)

type LogEntry struct {
	Time    time.Time
	Level   string
	Message string
}

var (
	logChan chan LogEntry
	done    chan struct{}
)

func Start(bufferSize int) {
	logChan = make(chan LogEntry, bufferSize) // buffered channel позволяет producers продолжать без ожидания
	done = make(chan struct{})                // coordination channel для graceful shutdown
	go workerLoop()                           // запускаем background goroutine для обработки log entries
}

func Stop() {
	close(logChan) // сигналим worker что больше нет entries - worker завершит обработку и выйдет
	<-done         // блокируемся пока worker не сигнализирует завершение - гарантирует отсутствие потери логов
}

func AsyncLog(level string, format string, args ...interface{}) {
	entry := LogEntry{
		Time:    time.Now(),                   // захватываем timestamp немедленно для точного времени события
		Level:   level,                        // сохраняем уровень severity для фильтрации
		Message: fmt.Sprintf(format, args...), // форматируем сообщение один раз в caller goroutine
	}

	select {
	case logChan <- entry: // попытка non-blocking отправки в channel
		// успех - entry поставлен в очередь для background обработки
	default:
		// channel полон - отбрасываем entry чтобы не блокировать caller
		// production системы могут инкрементить dropped_logs метрику здесь
	}
}

func workerLoop() {
	for entry := range logChan { // range выходит когда channel закрыт и опустошён
		log.Printf("[%s] [%s] %s",
			entry.Time.Format("2006-01-02 15:04:05.000"), // форматируем timestamp для human readability
			entry.Level,                                    // включаем severity level
			entry.Message)                                  // логируем отформатированное сообщение
	}
	close(done) // сигналим Stop() что все entries обработаны и worker завершается
}`,
			description: `Реализуйте высокопроизводительный async logger используя buffered channels чтобы предотвратить блокировку критичных code paths.

**Требования:**
1. **LogEntry**: Struct с timestamp, level, и message
2. **Start**: Запустить background goroutine обрабатывающую log entries из buffered channel
3. **Stop**: Graceful shutdown logger, flushing всех pending entries
4. **AsyncLog**: Non-blocking функция отправляющая log entry в channel
5. **workerLoop**: Background goroutine получающая из channel и пишущая логи

**Паттерн Async Logging:**
\`\`\`go
// Традиционный synchronous logging
func ProcessRequest(r *Request) {
    log.Print("Processing...") // БЛОКИРУЕТСЯ пока disk write не завершится (1-10ms)
    result := doWork(r)         // Работа ждёт лог
}

// Async logging
func ProcessRequest(r *Request) {
    AsyncLog("Processing...") // Возвращается немедленно (~100ns)
    result := doWork(r)        // Работа начинается мгновенно
    // Background goroutine обрабатывает disk writes
}
\`\`\`

**Почему Async Logging:**
- **Non-Blocking**: Основной код не ждёт disk I/O
- **Batching**: Background worker может batch писать для эффективности
- **Производительность**: 100x-1000x быстрее чем synchronous logging
- **Throughput**: Обрабатывать 100K+ логов/секунду без замедления app

**Ключевые концепции:**
- Buffered channel как очередь (\`make(chan LogEntry, 1000)\`)
- Producer (AsyncLog) никогда не блокируется пока buffer не полон
- Consumer (workerLoop) обрабатывает entries в background
- Graceful shutdown гарантирует отсутствие потери логов

**Ограничения:**
- Использовать buffered channel с настраиваемым размером
- LogEntry должен иметь: time (time.Time), level (string), message (string)
- Start goroutine через \`go workerLoop()\`
- Stop должен: закрыть channel, подождать завершения goroutine
- Использовать \`time.Now()\` для timestamp каждого entry
- Формат вывода: \`[timestamp] [level] message\``,
			hint1: `В Start создайте buffered channel через make(chan LogEntry, bufferSize) и запустите goroutine через go workerLoop().`,
			hint2: `В workerLoop используйте for entry := range logChan для обработки entries пока channel не закрыт. Форматируйте timestamp через entry.Time.Format("2006-01-02 15:04:05").`,
			whyItMatters: `Async logging критичен для высокопроизводительных систем где logging overhead может стать bottleneck влияющим на user experience.

**Почему важно:**

**1. Проблема Synchronous Logging**
\`\`\`go
// Synchronous logging (традиционный подход)
func handleRequest(w http.ResponseWriter, r *http.Request) {
    log.Print("Request started")     // БЛОКИРУЕТСЯ на disk I/O: 1-5ms
    user := authenticateUser(r)      // Ждет лог
    log.Print("User authenticated")  // БЛОКИРУЕТСЯ снова: 1-5ms
    data := fetchData(user)          // Ждет лог
    log.Print("Data fetched")        // БЛОКИРУЕТСЯ снова: 1-5ms
    json.NewEncoder(w).Encode(data)
}
// Общий overhead логирования: 3-15ms НА ЗАПРОС
// При 1000 запросов/секунду: 3-15 СЕКУНД CPU времени впустую!
\`\`\`

**Проблема:**
- Disk writes МЕДЛЕННЫЕ (1-10ms каждая)
- Каждый log.Print() блокируется пока write не завершится
- Критичная бизнес-логика ждет логирование
- Высоконагруженные сервисы не могут позволить этот overhead

**2. Решение Async Logging**
\`\`\`go
// Async logging
func handleRequest(w http.ResponseWriter, r *http.Request) {
    AsyncLog("INFO", "Request started")     // Возвращается за ~100ns
    user := authenticateUser(r)             // Начинается немедленно
    AsyncLog("INFO", "User authenticated")  // Возвращается за ~100ns
    data := fetchData(user)                 // Начинается немедленно
    AsyncLog("INFO", "Data fetched")        // Возвращается за ~100ns
    json.NewEncoder(w).Encode(data)
}
// Общий overhead логирования: ~300ns (0.0003ms)
// В 50,000x БЫСТРЕЕ чем synchronous!
\`\`\`

**Как работает:**
1. AsyncLog кладет entry в buffered channel (мгновенно)
2. Background goroutine читает из channel
3. Background goroutine пишет на disk
4. Main goroutine продолжается немедленно

**3. Real числа производительности**

**E-commerce API обрабатывает 10,000 запросов/секунду:**

**До Async Logging (Synchronous):**
\`\`\`
Средняя latency запроса: 45ms
5 log statements на запрос
5ms overhead логирования на запрос
11% latency это ПРОСТО ЛОГИРОВАНИЕ!

Производительность сервера: 10,000 req/s (CPU на максимуме)
CPU использование логированием: 40% (!)
\`\`\`

**После Async Logging:**
\`\`\`
Средняя latency запроса: 40ms (-11%)
5 log statements на запрос
0.005ms overhead логирования на запрос
0.01% latency это логирование

Производительность сервера: 15,000 req/s (+50%)
CPU использование логированием: 5% (background goroutine)
\`\`\`

**Улучшения:**
- Latency запроса: -11%
- Throughput: +50%
- CPU логирования: 40% → 5% (-88%)

**4. Паттерн Buffered Channel**
\`\`\`go
// Без буфера (блокирование)
logChan := make(chan LogEntry)
logChan <- entry // БЛОКИРУЕТСЯ если worker медленный

// С буфером (non-blocking)
logChan := make(chan LogEntry, 1000)
logChan <- entry // Мгновенно пока не 1000 entries в очереди

// Буфер действует как shock absorber:
// - Спайки трафика: буфер поглощает burst
// - Плавная обработка: worker обрабатывает steady rate
\`\`\`

**Выбор размера буфера:**
- Слишком маленький (10): Заполняется при спайках трафика, логи отбрасываются
- Слишком большой (100K): Трата памяти, медленный shutdown
- Рекомендуется: 1000-10000 entries
  - 1 LogEntry ≈ 100 bytes
  - 10000 entries = 1MB памяти (приемлемо)

**5. Паттерн Graceful Shutdown**
\`\`\`go
func main() {
    Start(1000)
    defer Stop() // КРИТИЧНО: Гарантирует flush логов перед выходом
    // ... код сервера ...
}

// Без Stop():
// - Программа завершается
// - Буфер имеет 500 pending entries
// - 500 логов ПОТЕРЯНО!

// С Stop():
// 1. Закрыть channel (новых entries нет)
// 2. Worker обрабатывает оставшиеся 500 entries
// 3. Worker сигнализирует done
// 4. Программа завершается безопасно
// - НОЛЬ логов потеряно!
\`\`\`

**6. Real-World Production Сценарий**

**Сценарий:** Сервис обработки платежей

**Паттерн трафика:**
- Обычно: 1000 платежей/секунду
- Black Friday: 50,000 платежей/секунду (спайк 50x!)
- Каждый платеж: 10 log statements

**Без Async Logging:**
\`\`\`
Black Friday утро:
- 50,000 req/s × 10 logs × 2ms на log
- = 1,000,000ms логирования в секунду
- = Серверу нужно 1000 CPU cores только для логирования!
- НЕВОЗМОЖНО! Сервер крашится!
\`\`\`

**С Async Logging:**
\`\`\`
Black Friday утро:
- 50,000 req/s × 10 logs × 0.0001ms на log
- = 50ms overhead логирования в секунду
- = 0.1% одного CPU core
- Сервер справляется легко!

Background worker:
- Обрабатывает 500,000 log entries/секунду
- Использует 1 CPU core на 30% утилизации
- Батчит disk writes для эффективности
- НОЛЬ влияния производительности на обработку платежей!
\`\`\`

**Результат:**
- Обработка платежей: НОЛЬ downtime
- Все логи захвачены: 500K entries/секунду
- User experience: Без изменений
- Revenue: $5M/час защищен

**7. Trade-offs и соображения**

**Преимущества:**
- 100-1000x быстрее чем synchronous logging
- Non-blocking: не замедляет критичный код
- Естественный batching: более эффективный disk I/O
- Масштабируется до миллионов логов/секунду

**Недостатки:**
- Логи слегка задержаны (миллисекунды)
- Можно потерять логи если буфер переполняется
- Требует graceful shutdown
- Сложнее чем synchronous

**Когда использовать Async Logging:**
- ✅ Высоконагруженные сервисы (>100 req/s)
- ✅ Требования низкой latency (<100ms)
- ✅ Много log statements на запрос
- ✅ Production системы со строгими SLA

**Когда НЕ использовать:**
- ❌ Отладка во время разработки (логи могут не появиться до краша)
- ❌ Критичные security логи (используйте sync для гарантии записи)
- ❌ Низконагруженные сервисы (<10 req/s) - overhead не стоит сложности

**8. Advanced: Мониторинг**
\`\`\`go
var (
    droppedLogs atomic.Int64
)

func AsyncLog(level string, format string, args ...interface{}) {
    entry := LogEntry{...}
    select {
    case logChan <- entry:
        // Залогировано успешно
    default:
        droppedLogs.Add(1) // Отслеживать drops для мониторинга
        // Алерт если drops превышают порог
        if droppedLogs.Load() > 10000 {
            // Отправить алерт: "Logging buffer full - increase size!"
        }
    }
}
\`\`\`

**Production Мониторинг:**
- Утилизация буфера: len(logChan) / cap(logChan)
- Счетчик отброшенных логов
- Здоровье worker goroutine
- Логов обработано в секунду

**Итог:**
Async logging трансформирует логирование из performance bottleneck в zero-overhead observability инструмент. Для высоконагруженных сервисов это разница между выполнением SLA и падением во время пиковой нагрузки.`
		},
		uz: {
			title: `Async buferlangan logger`,
			solutionCode: `package loggingx

import (
	"fmt"
	"log"
	"time"
)

type LogEntry struct {
	Time    time.Time
	Level   string
	Message string
}

var (
	logChan chan LogEntry
	done    chan struct{}
)

func Start(bufferSize int) {
	logChan = make(chan LogEntry, bufferSize) // buffered channel producerlar kutmasdan davom etishiga ruxsat beradi
	done = make(chan struct{})                // graceful shutdown uchun koordinatsiya kanali
	go workerLoop()                           // log yozuvlarni qayta ishlash uchun background goroutine ni ishga tushiramiz
}

func Stop() {
	close(logChan) // workerga boshqa yozuvlar kelmayotganini signal qilamiz - worker qoldiqlarni qayta ishlab chiqadi
	<-done         // worker tugashini signallagunicha bloklaymiz - log yo'qotilmasligini ta'minlaydi
}

func AsyncLog(level string, format string, args ...interface{}) {
	entry := LogEntry{
		Time:    time.Now(),                   // aniq hodisa vaqti uchun timestampni darhol tutib olamiz
		Level:   level,                        // filtrlash uchun severity darajasini saqlaymiz
		Message: fmt.Sprintf(format, args...), // caller goroutine da xabarni bir marta formatlaymiz
	}

	select {
	case logChan <- entry: // channelga non-blocking yuborishga urinish
		// muvaffaqiyat - yozuv background qayta ishlash uchun navbatga qo'yildi
	default:
		// channel to'liq - callerni bloklamaslik uchun yozuvni tashlaymiz
		// production tizimlari bu yerda dropped_logs metrikasini oshirishi mumkin
	}
}

func workerLoop() {
	for entry := range logChan { // range channel yopilganda va bo'shatilganda chiqadi
		log.Printf("[%s] [%s] %s",
			entry.Time.Format("2006-01-02 15:04:05.000"), // inson o'qishi uchun timestampni formatlaymiz
			entry.Level,                                    // severity darajasini qo'shamiz
			entry.Message)                                  // formatlangan xabarni log qilamiz
	}
	close(done) // Stop() ga barcha yozuvlar qayta ishlangani va worker chiqayotganini signal qilamiz
}`,
			description: `Logging kritik kod yo'llarini bloklamasligini ta'minlash uchun buffered channellardan foydalanib yuqori samarali async loggerni amalga oshiring.

**Talablar:**
1. **LogEntry**: timestamp, level va message bilan struct
2. **Start**: Buffered channeldan log yozuvlarini qayta ishlaydigan background goroutineni ishga tushirish
3. **Stop**: Barcha kutilayotgan yozuvlarni flushing qilish, loggerni graceful o'chirish
4. **AsyncLog**: Channelga log yozuvini yuboradigan non-blocking funksiya
5. **workerLoop**: Channeldan qabul qiladigan va loglarni yozadigan background goroutine

**Async Logging Pattern:**
\`\`\`go
// An'anaviy synchronous logging
func ProcessRequest(r *Request) {
    log.Print("Processing...") // disk yozish tugagunicha BLOKLAYDI (1-10ms)
    result := doWork(r)         // Ish log ni kutadi
}

// Async logging
func ProcessRequest(r *Request) {
    AsyncLog("Processing...") // Darhol qaytadi (~100ns)
    result := doWork(r)        // Ish darhol boshlanadi
    // Background goroutine disk yozishlarni boshqaradi
}
\`\`\`

**Nima uchun Async Logging:**
- **Non-Blocking**: Asosiy kod disk I/O ni kutmaydi
- **Batching**: Background worker samaradorlik uchun batch yozishi mumkin
- **Samaradorlik**: Synchronous loggingdan 100x-1000x tezroq
- **Throughput**: Ilovani sekinlashtirmasdan 100K+ log/soniya qayta ishlash

**Asosiy tushunchalar:**
- Buffered channel navbat sifatida ishlaydi (\`make(chan LogEntry, 1000)\`)
- Producer (AsyncLog) buffer to'lmagunicha hech qachon bloklanmaydi
- Consumer (workerLoop) background da yozuvlarni qayta ishlaydi
- Graceful shutdown log yo'qotilmasligini ta'minlaydi

**Cheklovlar:**
- Sozlanishi mumkin o'lchamli buffered channel dan foydalaning
- LogEntry da bo'lishi kerak: time (time.Time), level (string), message (string)
- \`go workerLoop()\` orqali goroutineni ishga tushiring
- Stop: channelni yoping, goroutine tugashini kuting
- Har bir yozuv uchun timestamp uchun \`time.Now()\` dan foydalaning
- Chiqish formati: \`[timestamp] [level] message\``,
			hint1: `Start da make(chan LogEntry, bufferSize) bilan buffered channel yarating va go workerLoop() bilan goroutineni ishga tushiring.`,
			hint2: `workerLoop da channel yopilgunicha yozuvlarni qayta ishlash uchun for entry := range logChan dan foydalaning. entry.Time.Format("2006-01-02 15:04:05") bilan timestampni formatlang.`,
			whyItMatters: `Async logging logging overhead foydalanuvchi tajribasiga ta'sir qiluvchi bottleneck bo'lishi mumkin bo'lgan yuqori samarali tizimlar uchun muhimdir.

**Nima uchun bu muhim:**

**1. Synchronous Logging muammosi**
\`\`\`go
// Synchronous logging (an'anaviy yondashuv)
func handleRequest(w http.ResponseWriter, r *http.Request) {
    log.Print("Request started")     // Disk I/O uchun BLOKLAYDI: 1-5ms
    user := authenticateUser(r)      // Log ni kutadi
    log.Print("User authenticated")  // Yana BLOKLAYDI: 1-5ms
    data := fetchData(user)          // Log ni kutadi
    log.Print("Data fetched")        // Yana BLOKLAYDI: 1-5ms
    json.NewEncoder(w).Encode(data)
}
// Jami logging overhead: SO'ROV UCHUN 3-15ms
// 1000 so'rov/soniyada: 3-15 SONIYA CPU vaqti isrof!
\`\`\`

**Muammo:**
- Disk yozishlar SEKIN (har biri 1-10ms)
- Har bir log.Print() yozish tugagunicha bloklanadi
- Muhim biznes mantiqingiz logging ni kutadi
- Yuqori trafik xizmatlari bu overhead ni to'lay olmaydi

**2. Async Logging yechimi**
\`\`\`go
// Async logging
func handleRequest(w http.ResponseWriter, r *http.Request) {
    AsyncLog("INFO", "Request started")     // ~100ns da qaytadi
    user := authenticateUser(r)             // Darhol boshlanadi
    AsyncLog("INFO", "User authenticated")  // ~100ns da qaytadi
    data := fetchData(user)                 // Darhol boshlanadi
    AsyncLog("INFO", "Data fetched")        // ~100ns da qaytadi
    json.NewEncoder(w).Encode(data)
}
// Jami logging overhead: ~300ns (0.0003ms)
// Synchronousdan 50,000x TEZROQ!
\`\`\`

**Qanday ishlaydi:**
1. AsyncLog yozuvni buffered channelga qo'yadi (bir zumda)
2. Background goroutine channeldan o'qiydi
3. Background goroutine diskka yozadi
4. Main goroutine darhol davom etadi

**3. Haqiqiy samaradorlik raqamlari**

**E-commerce API soniyasiga 10,000 so'rovni qayta ishlaydi:**

**Async Logging dan oldin (Synchronous):**
\`\`\`
O'rtacha so'rov kechikishi: 45ms
So'rov uchun 5 log statement
So'rov uchun 5ms logging overhead
Kechikishning 11% FAQAT LOGGING!

Server sig'imi: 10,000 req/s (CPU maksimalda)
Logging CPU foydalanish: 40% (!)
\`\`\`

**Async Logging dan keyin:**
\`\`\`
O'rtacha so'rov kechikishi: 40ms (-11%)
So'rov uchun 5 log statement
So'rov uchun 0.005ms logging overhead
Kechikishning 0.01% logging

Server sig'imi: 15,000 req/s (+50%)
Logging CPU foydalanish: 5% (background goroutine)
\`\`\`

**Yaxshilanishlar:**
- So'rov kechikishi: -11%
- Throughput: +50%
- Logging CPU: 40% → 5% (-88%)

**4. Buffered Channel Pattern**
\`\`\`go
// Buffersiz (bloklanadi)
logChan := make(chan LogEntry)
logChan <- entry // Worker sekin bo'lsa BLOKLAYDI

// Buffer bilan (bloklanmaydi)
logChan := make(chan LogEntry, 1000)
logChan <- entry // 1000 yozuv navbatda bo'lmagunicha bir zumda

// Buffer zarba yutuvchi sifatida ishlaydi:
// - Trafik portlashlari: buffer portlashni yutadi
// - Silliq qayta ishlash: worker barqaror tezlikda qayta ishlaydi
\`\`\`

**Buffer o'lchamini tanlash:**
- Juda kichik (10): Trafik portlashlarida to'ladi, loglar tashlanadi
- Juda katta (100K): Xotira isrofi, sekin o'chirish
- Tavsiya etilgan: 1000-10000 yozuv
  - 1 LogEntry ≈ 100 bayt
  - 10000 yozuv = 1MB xotira (qabul qilinadigan)

**5. Graceful Shutdown Pattern**
\`\`\`go
func main() {
    Start(1000)
    defer Stop() // MUHIM: Chiqishdan oldin loglarni flush qilishni ta'minlaydi
    // ... server kodi ...
}

// Stop() siz:
// - Dastur chiqadi
// - Bufferda 500 kutilayotgan yozuv bor
// - 500 log YO'QOLDI!

// Stop() bilan:
// 1. Channelni yoping (yangi yozuvlar yo'q)
// 2. Worker qolgan 500 yozuvni qayta ishlaydi
// 3. Worker done signalini beradi
// 4. Dastur xavfsiz chiqadi
// - NOLGA teng loglar yo'qoldi!
\`\`\`

**6. Real-World Production stsenariy**

**Stsenariy:** To'lov qayta ishlash xizmati

**Trafik Pattern:**
- Oddiy: 1000 to'lov/soniya
- Black Friday: 50,000 to'lov/soniya (50x portlash!)
- Har bir to'lov: 10 log statement

**Async Logging siz:**
\`\`\`
Black Friday ertalabi:
- 50,000 req/s × 10 log × 2ms har bir log uchun
- = Soniyada 1,000,000ms logging
- = Serverga faqat logging uchun 1000 CPU yadrolari kerak!
- IMKONSIZ! Server crashlanadi!
\`\`\`

**Async Logging bilan:**
\`\`\`
Black Friday ertalabi:
- 50,000 req/s × 10 log × 0.0001ms har bir log uchun
- = Soniyada 50ms logging overhead
- = Bitta CPU yadroning 0.1%
- Server osonlikcha boshqaradi!

Background worker:
- Soniyasiga 500,000 log yozuvini qayta ishlaydi
- 30% foydalanishda 1 CPU yadroni ishlatadi
- Samaradorlik uchun disk yozishlarni batching qiladi
- To'lov qayta ishlashga samaradorlik TA'SIRI YO'Q!
\`\`\`

**Natija:**
- To'lov qayta ishlash: NOLGA teng downtime
- Barcha loglar tutildi: soniyasiga 500K yozuv
- Foydalanuvchi tajribasi: O'zgarishsiz
- Daromad: soatiga $5M himoyalangan

**7. Afzalliklar va kamchiliklar**

**Afzalliklar:**
- Synchronous loggingdan 100-1000x tezroq
- Bloklanmaydi: muhim kodni sekinlashtirmaydi
- Tabiiy batching: samaraliroq disk I/O
- Millionlab log/soniyaga masshtablanadi

**Kamchiliklar:**
- Loglar biroz kechiktiriladi (millisekund)
- Buffer toshsa loglarni yo'qotishi mumkin
- Graceful shutdown talab qiladi
- Synchronousdan murakkabroq

**Qachon Async Logging dan foydalanish:**
- ✅ Yuqori trafik xizmatlari (>100 req/s)
- ✅ Past kechikish talablari (<100ms)
- ✅ So'rov uchun ko'plab log statementlar
- ✅ Qattiq SLA bilan production tizimlari

**Qachon FOYDALANMASLIK:**
- ❌ Development paytida debugging (crash dan oldin loglar paydo bo'lmasligi mumkin)
- ❌ Muhim xavfsizlik loglari (yozishni kafolatlash uchun sync dan foydalaning)
- ❌ Past trafik xizmatlari (<10 req/s) - overhead murakkablikka arzimaydi

**Xulosa:**
Async logging logging ni samaradorlik muammosidan xavfsiz real vaqtda boshqarish mumkin bo'lgan nol overhead kuzatish vositasiga aylantiradi. Yuqori trafik xizmatlari uchun bu SLA larni bajarish va eng yuqori trafik paytida yiqilib tushish o'rtasidagi farqdir.`
		}
	}
};

export default task;
