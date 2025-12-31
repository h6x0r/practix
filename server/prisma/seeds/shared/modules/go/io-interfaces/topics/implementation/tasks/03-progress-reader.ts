import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-interfaces-progress-reader',
	title: 'Implement Progress Tracking Reader',
	difficulty: 'easy',
	tags: ['go', 'interfaces', 'io', 'reader', 'progress'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a **ProgressReader** that wraps any \`io.Reader\` and reports progress through a callback function.

**Requirements:**
1. Wrap an existing \`io.Reader\` without modifying it
2. Track total bytes read across all Read() calls
3. Call progress callback after each successful read
4. Pass through all read operations transparently
5. Preserve original error behavior (including \`io.EOF\`)
6. Support concurrent reads safely
7. Allow callback to be nil (no-op when nil)
8. Report both bytes read in current operation and cumulative total

**Type Definitions:**
\`\`\`go
// ProgressCallback receives bytes read in this operation and total bytes read
type ProgressCallback func(current int64, total int64)

type ProgressReader struct {
    r        io.Reader
    total    int64
    callback ProgressCallback
    mu       sync.Mutex
}

func NewProgressReader(r io.Reader, callback ProgressCallback) *ProgressReader

func (pr *ProgressReader) Read(p []byte) (n int, err error)
\`\`\`

**Example Usage:**
\`\`\`go
file, _ := os.Open("large-file.dat") // 1GB file
defer file.Close()

// Track download progress
progress := NewProgressReader(file, func(current, total int64) {
    percentage := float64(total) / float64(fileSize) * 100
    fmt.Printf("\rProgress: %.2f%% (%d bytes)", percentage, total)
})

var dst bytes.Buffer
io.Copy(&dst, progress)
// Output:
// Progress: 10.50% (110100480 bytes)
// Progress: 25.30% (265289728 bytes)
// ... updates continuously ...
// Progress: 100.00% (1048576000 bytes)
\`\`\`

**Key Concepts:**
- **Decorator pattern**: Wrap existing reader without changing it
- **Callback functions**: Report progress to caller
- **State tracking**: Maintain cumulative byte count
- **Thread safety**: Protect shared state with mutex
- **Transparent wrapping**: Behave exactly like underlying reader

**Implementation Strategy:**
1. Store wrapped reader, callback, and total counter
2. Protect total counter with sync.Mutex for concurrent safety
3. In Read(): call underlying reader's Read()
4. If bytes read (n > 0), update total and invoke callback
5. Return original n and err from underlying Read()
6. Check callback != nil before invoking
7. Lock mutex before updating/reading total

**Constraints:**
- Must not change behavior of wrapped reader
- Must be thread-safe for concurrent reads
- Callback must be called after successful read (n > 0)
- Must work with any io.Reader implementation
- Total must accurately reflect cumulative bytes read`,
	initialCode: `package interfaces

import (
	"io"
	"sync"
)

// ProgressCallback receives current operation bytes and total bytes read
type ProgressCallback func(current int64, total int64)

type ProgressReader struct {
	// TODO: Add fields (reader, total, callback, mutex)
}

// NewProgressReader creates a new progress tracking reader
func NewProgressReader(r io.Reader, callback ProgressCallback) *ProgressReader {
	// TODO: Initialize ProgressReader
	// TODO: Implement
}

// Read implements io.Reader interface
func (pr *ProgressReader) Read(p []byte) (n int, err error) {
	// TODO: Implement progress tracking read
	// Hint: Read from underlying reader, update total, call callback
	// TODO: Implement
}`,
	solutionCode: `package interfaces

import (
	"io"
	"sync"
)

// ProgressCallback receives current operation bytes and total bytes read
type ProgressCallback func(current int64, total int64)

type ProgressReader struct {
	r        io.Reader
	total    int64
	callback ProgressCallback
	mu       sync.Mutex
}

// NewProgressReader creates a new progress tracking reader
func NewProgressReader(r io.Reader, callback ProgressCallback) *ProgressReader {
	return &ProgressReader{
		r:        r,
		callback: callback,
		total:    0,
	}
}

// Read implements io.Reader interface
func (pr *ProgressReader) Read(p []byte) (n int, err error) {
	// Read from underlying reader
	n, err = pr.r.Read(p)

	// Update total and report progress if we read any bytes
	if n > 0 {
		pr.mu.Lock()
		pr.total += int64(n)
		total := pr.total
		pr.mu.Unlock()

		// Call callback if provided
		if pr.callback != nil {
			pr.callback(int64(n), total)
		}
	}

	return n, err
}`,
	testCode: `package interfaces

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
)

func Test1ProgressReaderBasicRead(t *testing.T) {
	src := strings.NewReader("hello world")
	var currentBytes, totalBytes int64
	pr := NewProgressReader(src, func(current, total int64) {
		currentBytes = current
		totalBytes = total
	})
	buf := make([]byte, 5)
	n, err := pr.Read(buf)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if n != 5 {
		t.Errorf("expected 5 bytes, got %d", n)
	}
	if currentBytes != 5 || totalBytes != 5 {
		t.Errorf("expected current=5, total=5, got current=%d, total=%d", currentBytes, totalBytes)
	}
}

func Test2ProgressReaderCumulativeTotal(t *testing.T) {
	src := strings.NewReader("hello world")
	var totals []int64
	pr := NewProgressReader(src, func(current, total int64) {
		totals = append(totals, total)
	})
	buf := make([]byte, 5)
	pr.Read(buf)
	pr.Read(buf)
	if len(totals) != 2 {
		t.Errorf("expected 2 callbacks, got %d", len(totals))
	}
	if totals[0] != 5 || totals[1] != 10 {
		t.Errorf("expected totals [5, 10], got %v", totals)
	}
}

func Test3ProgressReaderNilCallback(t *testing.T) {
	src := strings.NewReader("hello")
	pr := NewProgressReader(src, nil)
	buf := make([]byte, 10)
	n, err := pr.Read(buf)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if n != 5 {
		t.Errorf("expected 5 bytes, got %d", n)
	}
}

func Test4ProgressReaderEOF(t *testing.T) {
	src := strings.NewReader("abc")
	callbackCalled := false
	pr := NewProgressReader(src, func(current, total int64) {
		callbackCalled = true
	})
	buf := make([]byte, 10)
	n, err := pr.Read(buf)
	if n != 3 || err != nil {
		t.Errorf("expected n=3, err=nil, got n=%d, err=%v", n, err)
	}
	n, err = pr.Read(buf)
	if n != 0 || err != io.EOF {
		t.Errorf("expected n=0, err=EOF, got n=%d, err=%v", n, err)
	}
}

func Test5ProgressReaderErrorPropagation(t *testing.T) {
	errReader := &errorReader{err: errors.New("read error")}
	pr := NewProgressReader(errReader, func(current, total int64) {})
	buf := make([]byte, 10)
	_, err := pr.Read(buf)
	if err == nil || err.Error() != "read error" {
		t.Errorf("expected read error, got %v", err)
	}
}

type errorReader struct {
	err error
}

func (er *errorReader) Read(p []byte) (int, error) {
	return 0, er.err
}

func Test6ProgressReaderConcurrentReads(t *testing.T) {
	data := bytes.Repeat([]byte("a"), 10000)
	src := bytes.NewReader(data)
	var mu sync.Mutex
	var total int64
	pr := NewProgressReader(src, func(current, totalRead int64) {
		mu.Lock()
		total = totalRead
		mu.Unlock()
	})
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			buf := make([]byte, 100)
			for {
				_, err := pr.Read(buf)
				if err == io.EOF {
					break
				}
			}
		}()
	}
	wg.Wait()
	mu.Lock()
	finalTotal := total
	mu.Unlock()
	if finalTotal != 10000 {
		t.Errorf("expected total 10000, got %d", finalTotal)
	}
}

func Test7ProgressReaderCurrentBytes(t *testing.T) {
	src := strings.NewReader("hello world")
	var current int64
	pr := NewProgressReader(src, func(c, total int64) {
		current = c
	})
	buf := make([]byte, 5)
	pr.Read(buf)
	if current != 5 {
		t.Errorf("expected current=5, got %d", current)
	}
	buf = make([]byte, 3)
	pr.Read(buf)
	if current != 3 {
		t.Errorf("expected current=3, got %d", current)
	}
}

func Test8ProgressReaderEmptyRead(t *testing.T) {
	src := strings.NewReader("hello")
	callCount := 0
	pr := NewProgressReader(src, func(current, total int64) {
		callCount++
	})
	buf := make([]byte, 0)
	pr.Read(buf)
	if callCount != 0 {
		t.Error("callback should not be called for empty read")
	}
}

func Test9ProgressReaderLargeData(t *testing.T) {
	data := bytes.Repeat([]byte("x"), 1000000)
	src := bytes.NewReader(data)
	var finalTotal int64
	pr := NewProgressReader(src, func(current, total int64) {
		finalTotal = total
	})
	io.Copy(io.Discard, pr)
	if finalTotal != 1000000 {
		t.Errorf("expected total 1000000, got %d", finalTotal)
	}
}

func Test10ProgressReaderWrapping(t *testing.T) {
	src := strings.NewReader("test data")
	pr1 := NewProgressReader(src, func(current, total int64) {})
	pr2 := NewProgressReader(pr1, func(current, total int64) {})
	buf := make([]byte, 10)
	n, err := pr2.Read(buf)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if n != 9 {
		t.Errorf("expected 9 bytes, got %d", n)
	}
	if string(buf[:n]) != "test data" {
		t.Errorf("expected 'test data', got '%s'", string(buf[:n]))
	}
}`,
	hint1: `Store the wrapped reader, callback function, total counter, and a mutex. In Read(), first call the underlying reader's Read() method, then update the total if n > 0.`,
	hint2: `Lock the mutex before updating the total counter, store the new total in a local variable, then unlock. This ensures thread-safe access. Call the callback after unlocking to avoid holding the lock during callback execution.`,
	whyItMatters: `Progress tracking is essential for user experience in applications that perform long-running I/O operations.

**Why This Matters:**

**1. User Experience**
Without progress indicators, users don't know if:
- Operation is still running
- Application is frozen
- How long to wait
- Whether to cancel

Progress feedback keeps users informed and improves satisfaction.

**2. Real-World Usage**

**File Download with Progress Bar:**
\`\`\`go
func DownloadFile(url, destination string) error {
	// Get file size
    resp, _ := http.Head(url)
    size, _ := strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)

	// Download with progress
    resp, _ = http.Get(url)
    defer resp.Body.Close()

    file, _ := os.Create(destination)
    defer file.Close()

	// Wrap response body with progress tracking
    progress := NewProgressReader(resp.Body, func(current, total int64) {
        percent := float64(total) / float64(size) * 100
        bar := strings.Repeat("=", int(percent/2))
        fmt.Printf("\r[%-50s] %.1f%%", bar, percent)
    })

    io.Copy(file, progress)
    fmt.Println("\nDownload complete!")
    return nil
}
\`\`\`

**3. Production Pattern: Upload with Timeout**

\`\`\`go
func UploadWithProgress(data io.Reader, size int64) error {
    progress := NewProgressReader(data, func(current, total int64) {
        log.Printf("Uploaded %d/%d bytes (%.1f%%)",
            total, size, float64(total)/float64(size)*100)
    })

	// Upload with progress tracking
    req, _ := http.NewRequest("POST", "https://api.example.com/upload", progress)
    req.ContentLength = size

    client := &http.Client{Timeout: 5 * time.Minute}
    resp, err := client.Do(req)
    if err != nil {
        return fmt.Errorf("upload failed: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != 200 {
        return fmt.Errorf("upload failed with status: %d", resp.StatusCode)
    }

    return nil
}
\`\`\`

**4. Real Incident: Silent Timeout**

A backup service had this code:
\`\`\`go
// BAD - no progress feedback
func BackupDatabase(db *sql.DB) error {
    file, _ := os.Create("backup.sql")
    defer file.Close()

	// Takes 2 hours, no feedback
    return db.Backup(file)
}
\`\`\`

Problems:
- Users thought it was frozen and killed process
- No way to estimate completion time
- No monitoring for stuck operations
- Support tickets: "Is backup working?"

Fix with progress tracking:
\`\`\`go
// GOOD - progress feedback and monitoring
func BackupDatabase(db *sql.DB) error {
    file, _ := os.Create("backup.sql")
    defer file.Close()

    pipeReader, pipeWriter := io.Pipe()

	// Backup to pipe in background
    go func() {
        db.Backup(pipeWriter)
        pipeWriter.Close()
    }()

	// Track progress from pipe to file
    lastUpdate := time.Now()
    progress := NewProgressReader(pipeReader, func(current, total int64) {
	// Update every 10 seconds
        if time.Since(lastUpdate) > 10*time.Second {
            log.Printf("Backup progress: %d MB written", total/1024/1024)
            lastUpdate = time.Now()
        }
    })

    _, err := io.Copy(file, progress)
    return err
}
\`\`\`

Result:
- Users see progress in logs
- Can estimate completion time
- Can detect stuck operations (no progress for 5 minutes = alert)
- Support tickets reduced by 80%

**5. Advanced Pattern: Multi-File Progress**

\`\`\`go
type AggregateProgress struct {
    files     int
    completed int
    totalBytes int64
    currentBytes int64
    mu sync.Mutex
}

func (ap *AggregateProgress) NewFileReader(r io.Reader) io.Reader {
    return NewProgressReader(r, func(current, total int64) {
        ap.mu.Lock()
        ap.currentBytes += current
        progress := float64(ap.currentBytes) / float64(ap.totalBytes) * 100
        ap.mu.Unlock()

        fmt.Printf("\rFile %d/%d: %.1f%% complete",
            ap.completed+1, ap.files, progress)
    })
}

func DownloadMultipleFiles(urls []string) error {
	// Calculate total size
    var totalSize int64
    for _, url := range urls {
        resp, _ := http.Head(url)
        size, _ := strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)
        totalSize += size
    }

    progress := &AggregateProgress{
        files: len(urls),
        totalBytes: totalSize,
    }

	// Download all files with aggregate progress
    for _, url := range urls {
        resp, _ := http.Get(url)
        reader := progress.NewFileReader(resp.Body)

        filename := path.Base(url)
        file, _ := os.Create(filename)
        io.Copy(file, reader)

        file.Close()
        resp.Body.Close()

        progress.mu.Lock()
        progress.completed++
        progress.mu.Unlock()
    }

    return nil
}
\`\`\`

**6. Decorator Pattern Benefits**

The decorator pattern used here is powerful:

\`\`\`go
// Stack multiple behaviors
file, _ := os.Open("data.txt")

// Add logging
logged := NewLoggingReader(file)

// Add progress
progressed := NewProgressReader(logged, progressCallback)

// Add rate limiting
limited := NewRateLimitedReader(progressed, 1024*1024) // 1MB/s

// Use combined reader
io.Copy(destination, limited)
// Now you have: file -> logged -> progress -> rate limit -> destination
\`\`\`

Each wrapper adds one behavior without modifying others.

**7. Testing Made Easy**

\`\`\`go
func TestProgressReader(t *testing.T) {
    data := bytes.NewReader([]byte("hello world"))

    var callCount int
    var lastTotal int64

    progress := NewProgressReader(data, func(current, total int64) {
        callCount++
        lastTotal = total

        if current <= 0 {
            t.Error("current should be > 0")
        }
        if total < current {
            t.Error("total should be >= current")
        }
    })

    buf := make([]byte, 5)

	// First read
    n, _ := progress.Read(buf)
    if n != 5 || callCount != 1 || lastTotal != 5 {
        t.Errorf("unexpected: n=%d, calls=%d, total=%d", n, callCount, lastTotal)
    }

	// Second read
    n, _ = progress.Read(buf)
    if n != 5 || callCount != 2 || lastTotal != 10 {
        t.Errorf("unexpected: n=%d, calls=%d, total=%d", n, callCount, lastTotal)
    }
}
\`\`\`

**8. Performance Considerations**

\`\`\`go
// BAD - callback on every byte is slow
io.Copy(dst, NewProgressReader(src, func(c, t int64) {
    fmt.Printf("\r%d bytes", t) // Called millions of times!
}))

// GOOD - throttle callback updates
type ThrottledProgress struct {
    lastUpdate time.Time
    interval   time.Duration
}

func (tp *ThrottledProgress) Callback(current, total int64) {
    if time.Since(tp.lastUpdate) < tp.interval {
        return // Skip update
    }
    tp.lastUpdate = time.Now()
    fmt.Printf("\r%d bytes", total)
}

progress := NewProgressReader(src, throttled.Callback)
io.Copy(dst, progress)
\`\`\`

**9. Monitoring and Alerting**

\`\`\`go
func MonitoredCopy(dst io.Writer, src io.Reader, expectedSize int64) error {
    startTime := time.Now()
    lastProgress := int64(0)
    stuckTime := time.Now()

    progress := NewProgressReader(src, func(current, total int64) {
	// Check if stuck (no progress for 30 seconds)
        if total == lastProgress {
            if time.Since(stuckTime) > 30*time.Second {
                log.Printf("WARNING: No progress for 30s at %d bytes", total)
            }
        } else {
            stuckTime = time.Now()
            lastProgress = total
        }

	// Estimate completion time
        elapsed := time.Since(startTime)
        if total > 0 {
            rate := float64(total) / elapsed.Seconds()
            remaining := float64(expectedSize-total) / rate
            log.Printf("ETA: %.0f seconds (%.1f MB/s)", remaining, rate/1024/1024)
        }
    })

    return io.Copy(dst, progress)
}
\`\`\`

**10. Common Patterns**

**Pattern 1: Progress Bar UI**
\`\`\`go
type ProgressBar struct {
    total int64
    width int
}

func (pb *ProgressBar) Callback(current, total int64) {
    percent := float64(total) / float64(pb.total) * 100
    filled := int(percent) * pb.width / 100
    bar := strings.Repeat("█", filled) + strings.Repeat("░", pb.width-filled)
    fmt.Printf("\r[%s] %.1f%%", bar, percent)
}
\`\`\`

**Pattern 2: Webhook Notifications**
\`\`\`go
func WebhookProgress(url string, total int64) ProgressCallback {
    return func(current, total int64) {
        if total%1024*1024*100 == 0 { // Every 100MB
            json := fmt.Sprintf(\`{"current": %d, "total": %d}\`, current, total)
            http.Post(url, "application/json", strings.NewReader(json))
        }
    }
}
\`\`\`

**Pattern 3: Database Progress Log**
\`\`\`go
func DatabaseProgress(db *sql.DB, jobID int) ProgressCallback {
    return func(current, total int64) {
        db.Exec("UPDATE jobs SET bytes_transferred = ? WHERE id = ?", total, jobID)
    }
}
\`\`\`

**Key Takeaways:**
- Progress tracking dramatically improves UX
- Decorator pattern enables composable behaviors
- Thread safety is critical for concurrent operations
- Throttle callbacks to avoid performance impact
- Enable monitoring and alerting for production
- Simple to implement, huge user satisfaction impact
- Works with any io.Reader transparently`,
	order: 2,
	translations: {
		ru: {
			title: 'Реализация Reader с отслеживанием прогресса',
			description: `Реализуйте **ProgressReader**, который оборачивает любой \`io.Reader\` и сообщает о прогрессе через callback-функцию.

**Требования:**
1. Обернуть существующий \`io.Reader\` без его модификации
2. Отслеживать общее количество прочитанных байт через все вызовы Read()
3. Вызывать callback прогресса после каждого успешного чтения
4. Прозрачно передавать все операции чтения
5. Сохранять исходное поведение ошибок (включая \`io.EOF\`)
6. Безопасно поддерживать параллельные чтения
7. Разрешать callback быть nil (no-op когда nil)
8. Сообщать как байты прочитанные в текущей операции, так и кумулятивную сумму

**Определение типов:**
\`\`\`go
// ProgressCallback получает байты прочитанные в этой операции и общее количество прочитанных байт
type ProgressCallback func(current int64, total int64)

type ProgressReader struct {
    r        io.Reader
    total    int64
    callback ProgressCallback
    mu       sync.Mutex
}

func NewProgressReader(r io.Reader, callback ProgressCallback) *ProgressReader

func (pr *ProgressReader) Read(p []byte) (n int, err error)
\`\`\`

**Пример использования:**
\`\`\`go
file, _ := os.Open("large-file.dat") // файл 1GB
defer file.Close()

// Отслеживание прогресса загрузки
progress := NewProgressReader(file, func(current, total int64) {
    percentage := float64(total) / float64(fileSize) * 100
    fmt.Printf("\\rПрогресс: %.2f%% (%d байт)", percentage, total)
})

var dst bytes.Buffer
io.Copy(&dst, progress)
// Вывод:
// Прогресс: 10.50% (110100480 байт)
// Прогресс: 25.30% (265289728 байт)
// ... обновляется постоянно ...
// Прогресс: 100.00% (1048576000 байт)
\`\`\`

**Ключевые концепции:**
- **Паттерн декоратор**: Оборачивание существующего reader без его изменения
- **Callback-функции**: Сообщение о прогрессе вызывающему коду
- **Отслеживание состояния**: Поддержка кумулятивного счетчика байт
- **Потокобезопасность**: Защита общего состояния мьютексом
- **Прозрачное оборачивание**: Ведет себя точно как базовый reader

**Стратегия реализации:**
1. Хранить обернутый reader, callback и общий счетчик
2. Защитить общий счетчик с помощью sync.Mutex для параллельной безопасности
3. В Read(): вызвать Read() базового reader
4. Если прочитаны байты (n > 0), обновить total и вызвать callback
5. Вернуть исходные n и err из базового Read()
6. Проверить callback != nil перед вызовом
7. Заблокировать мьютекс перед обновлением/чтением total

**Ограничения:**
- Не должна изменять поведение обернутого reader
- Должна быть потокобезопасной для параллельных чтений
- Callback должен вызываться после успешного чтения (n > 0)
- Должна работать с любой реализацией io.Reader
- Total должен точно отражать кумулятивные прочитанные байты`,
			hint1: `Храните обернутый reader, callback-функцию, счетчик total и мьютекс. В Read() сначала вызовите метод Read() базового reader, затем обновите total если n > 0.`,
			hint2: `Заблокируйте мьютекс перед обновлением счетчика total, сохраните новое значение total в локальную переменную, затем разблокируйте. Это обеспечивает потокобезопасный доступ. Вызывайте callback после разблокировки, чтобы не держать блокировку во время выполнения callback.`,
			whyItMatters: `Отслеживание прогресса необходимо для пользовательского опыта в приложениях, выполняющих долгие I/O операции.

**Почему это важно:**

**1. Пользовательский опыт**
Без индикаторов прогресса пользователи не знают:
- Работает ли еще операция
- Зависло ли приложение
- Сколько ждать
- Стоит ли отменять

Обратная связь по прогрессу держит пользователей информированными и улучшает удовлетворенность.

**2. Практическое использование**

**Загрузка файла с прогресс-баром:**
\`\`\`go
func DownloadFile(url, destination string) error {
	// Получить размер файла
    resp, _ := http.Head(url)
    size, _ := strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)

	// Загрузка с прогрессом
    resp, _ = http.Get(url)
    defer resp.Body.Close()

    file, _ := os.Create(destination)
    defer file.Close()

	// Обернуть response body отслеживанием прогресса
    progress := NewProgressReader(resp.Body, func(current, total int64) {
        percent := float64(total) / float64(size) * 100
        bar := strings.Repeat("=", int(percent/2))
        fmt.Printf("\\r[%-50s] %.1f%%", bar, percent)
    })

    io.Copy(file, progress)
    fmt.Println("\\nЗагрузка завершена!")
    return nil
}
\`\`\`

**3. Продакшен паттерн: Загрузка с таймаутом**

\`\`\`go
func UploadWithProgress(data io.Reader, size int64) error {
    progress := NewProgressReader(data, func(current, total int64) {
        log.Printf("Загружено %d/%d байт (%.1f%%)",
            total, size, float64(total)/float64(size)*100)
    })

	// Загрузка с отслеживанием прогресса
    req, _ := http.NewRequest("POST", "https://api.example.com/upload", progress)
    req.ContentLength = size

    client := &http.Client{Timeout: 5 * time.Minute}
    resp, err := client.Do(req)
    if err != nil {
        return fmt.Errorf("загрузка не удалась: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != 200 {
        return fmt.Errorf("загрузка не удалась со статусом: %d", resp.StatusCode)
    }

    return nil
}
\`\`\`

**4. Реальный инцидент: Тихий таймаут**

Сервис резервного копирования имел этот код:
\`\`\`go
// ПЛОХО - нет обратной связи по прогрессу
func BackupDatabase(db *sql.DB) error {
    file, _ := os.Create("backup.sql")
    defer file.Close()

	// Занимает 2 часа, нет обратной связи
    return db.Backup(file)
}
\`\`\`

Проблемы:
- Пользователи думали что процесс завис и убивали его
- Нет способа оценить время завершения
- Нет мониторинга застрявших операций
- Тикеты в поддержку: "Работает ли резервное копирование?"

Исправление с отслеживанием прогресса:
\`\`\`go
// ХОРОШО - обратная связь по прогрессу и мониторинг
func BackupDatabase(db *sql.DB) error {
    file, _ := os.Create("backup.sql")
    defer file.Close()

    pipeReader, pipeWriter := io.Pipe()

	// Резервное копирование в pipe в фоне
    go func() {
        db.Backup(pipeWriter)
        pipeWriter.Close()
    }()

	// Отслеживание прогресса из pipe в файл
    lastUpdate := time.Now()
    progress := NewProgressReader(pipeReader, func(current, total int64) {
	// Обновление каждые 10 секунд
        if time.Since(lastUpdate) > 10*time.Second {
            log.Printf("Прогресс резервного копирования: %d MB записано", total/1024/1024)
            lastUpdate = time.Now()
        }
    })

    _, err := io.Copy(file, progress)
    return err
}
\`\`\`

Результат:
- Пользователи видят прогресс в логах
- Можно оценить время завершения
- Можно обнаружить застрявшие операции (нет прогресса 5 минут = алерт)
- Тикетов в поддержку стало на 80% меньше

**5. Продвинутый паттерн: Прогресс нескольких файлов**

\`\`\`go
type AggregateProgress struct {
    files     int
    completed int
    totalBytes int64
    currentBytes int64
    mu sync.Mutex
}

func (ap *AggregateProgress) NewFileReader(r io.Reader) io.Reader {
    return NewProgressReader(r, func(current, total int64) {
        ap.mu.Lock()
        ap.currentBytes += current
        progress := float64(ap.currentBytes) / float64(ap.totalBytes) * 100
        ap.mu.Unlock()

        fmt.Printf("\\rФайл %d/%d: %.1f%% завершено",
            ap.completed+1, ap.files, progress)
    })
}

func DownloadMultipleFiles(urls []string) error {
	// Вычислить общий размер
    var totalSize int64
    for _, url := range urls {
        resp, _ := http.Head(url)
        size, _ := strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)
        totalSize += size
    }

    progress := &AggregateProgress{
        files: len(urls),
        totalBytes: totalSize,
    }

	// Загрузить все файлы с агрегированным прогрессом
    for _, url := range urls {
        resp, _ := http.Get(url)
        reader := progress.NewFileReader(resp.Body)

        filename := path.Base(url)
        file, _ := os.Create(filename)
        io.Copy(file, reader)

        file.Close()
        resp.Body.Close()

        progress.mu.Lock()
        progress.completed++
        progress.mu.Unlock()
    }

    return nil
}
\`\`\`

**6. Преимущества паттерна декоратор**

Паттерн декоратор здесь мощный:

\`\`\`go
// Наслоение нескольких поведений
file, _ := os.Open("data.txt")

// Добавить логирование
logged := NewLoggingReader(file)

// Добавить прогресс
progressed := NewProgressReader(logged, progressCallback)

// Добавить ограничение скорости
limited := NewRateLimitedReader(progressed, 1024*1024) // 1MB/s

// Использовать комбинированный reader
io.Copy(destination, limited)
// Теперь у вас: file -> logged -> progress -> rate limit -> destination
\`\`\`

Каждая обертка добавляет одно поведение без изменения других.

**7. Упрощенное тестирование**

\`\`\`go
func TestProgressReader(t *testing.T) {
    data := bytes.NewReader([]byte("hello world"))

    var callCount int
    var lastTotal int64

    progress := NewProgressReader(data, func(current, total int64) {
        callCount++
        lastTotal = total

        if current <= 0 {
            t.Error("current должен быть > 0")
        }
        if total < current {
            t.Error("total должен быть >= current")
        }
    })

    buf := make([]byte, 5)

	// Первое чтение
    n, _ := progress.Read(buf)
    if n != 5 || callCount != 1 || lastTotal != 5 {
        t.Errorf("неожиданно: n=%d, calls=%d, total=%d", n, callCount, lastTotal)
    }

	// Второе чтение
    n, _ = progress.Read(buf)
    if n != 5 || callCount != 2 || lastTotal != 10 {
        t.Errorf("неожиданно: n=%d, calls=%d, total=%d", n, callCount, lastTotal)
    }
}
\`\`\`

**8. Соображения производительности**

\`\`\`go
// ПЛОХО - callback на каждый байт медленный
io.Copy(dst, NewProgressReader(src, func(c, t int64) {
    fmt.Printf("\\r%d байт", t) // Вызывается миллионы раз!
}))

// ХОРОШО - ограничить обновления callback
type ThrottledProgress struct {
    lastUpdate time.Time
    interval   time.Duration
}

func (tp *ThrottledProgress) Callback(current, total int64) {
    if time.Since(tp.lastUpdate) < tp.interval {
        return // Пропустить обновление
    }
    tp.lastUpdate = time.Now()
    fmt.Printf("\\r%d байт", total)
}

progress := NewProgressReader(src, throttled.Callback)
io.Copy(dst, progress)
\`\`\`

**9. Мониторинг и оповещения**

\`\`\`go
func MonitoredCopy(dst io.Writer, src io.Reader, expectedSize int64) error {
    startTime := time.Now()
    lastProgress := int64(0)
    stuckTime := time.Now()

    progress := NewProgressReader(src, func(current, total int64) {
	// Проверить застревание (нет прогресса 30 секунд)
        if total == lastProgress {
            if time.Since(stuckTime) > 30*time.Second {
                log.Printf("ПРЕДУПРЕЖДЕНИЕ: Нет прогресса 30с на %d байтах", total)
            }
        } else {
            stuckTime = time.Now()
            lastProgress = total
        }

	// Оценить время завершения
        elapsed := time.Since(startTime)
        if total > 0 {
            rate := float64(total) / elapsed.Seconds()
            remaining := float64(expectedSize-total) / rate
            log.Printf("ETA: %.0f секунд (%.1f MB/s)", remaining, rate/1024/1024)
        }
    })

    return io.Copy(dst, progress)
}
\`\`\`

**10. Общие паттерны**

**Паттерн 1: Прогресс-бар UI**
\`\`\`go
type ProgressBar struct {
    total int64
    width int
}

func (pb *ProgressBar) Callback(current, total int64) {
    percent := float64(total) / float64(pb.total) * 100
    filled := int(percent) * pb.width / 100
    bar := strings.Repeat("█", filled) + strings.Repeat("░", pb.width-filled)
    fmt.Printf("\\r[%s] %.1f%%", bar, percent)
}
\`\`\`

**Паттерн 2: Webhook уведомления**
\`\`\`go
func WebhookProgress(url string, total int64) ProgressCallback {
    return func(current, total int64) {
        if total%1024*1024*100 == 0 { // Каждые 100MB
            json := fmt.Sprintf(\`{"current": %d, "total": %d}\`, current, total)
            http.Post(url, "application/json", strings.NewReader(json))
        }
    }
}
\`\`\`

**Паттерн 3: Лог прогресса в базе данных**
\`\`\`go
func DatabaseProgress(db *sql.DB, jobID int) ProgressCallback {
    return func(current, total int64) {
        db.Exec("UPDATE jobs SET bytes_transferred = ? WHERE id = ?", total, jobID)
    }
}
\`\`\`

**Ключевые выводы:**
- Отслеживание прогресса драматически улучшает UX
- Паттерн декоратор позволяет композицию поведений
- Потокобезопасность критична для параллельных операций
- Ограничивайте callback'и чтобы избежать влияния на производительность
- Включает мониторинг и оповещения для продакшена
- Просто реализовать, огромное влияние на удовлетворенность пользователей
- Работает с любым io.Reader прозрачно`,
			solutionCode: `package interfaces

import (
	"io"
	"sync"
)

// ProgressCallback получает байты текущей операции и общее количество прочитанных байт
type ProgressCallback func(current int64, total int64)

type ProgressReader struct {
	r        io.Reader
	total    int64
	callback ProgressCallback
	mu       sync.Mutex
}

// NewProgressReader создает новый reader с отслеживанием прогресса
func NewProgressReader(r io.Reader, callback ProgressCallback) *ProgressReader {
	return &ProgressReader{
		r:        r,
		callback: callback,
		total:    0,
	}
}

// Read реализует интерфейс io.Reader
func (pr *ProgressReader) Read(p []byte) (n int, err error) {
	// Читаем из базового reader
	n, err = pr.r.Read(p)

	// Обновляем total и сообщаем о прогрессе если прочитали байты
	if n > 0 {
		pr.mu.Lock()
		pr.total += int64(n)
		total := pr.total
		pr.mu.Unlock()

		// Вызываем callback если предоставлен
		if pr.callback != nil {
			pr.callback(int64(n), total)
		}
	}

	return n, err
}`
		},
		uz: {
			title: `Jarayon kuzatish bilan Reader ni amalga oshirish`,
			description: `Har qanday \`io.Reader\` ni o'rab, callback funksiyasi orqali jarayonni hisobot qiluvchi **ProgressReader** ni amalga oshiring.

**Talablar:**
1. Mavjud \`io.Reader\` ni o'zgartirmasdan o'rash
2. Barcha Read() chaqiruvlar orqali jami o'qilgan baytlarni kuzatish
3. Har bir muvaffaqiyatli o'qishdan keyin jarayon callback ni chaqirish
4. Barcha o'qish operatsiyalarini shaffof tarzda o'tkazish
5. Asl xato xatti-harakatini saqlash (\`io.EOF\` ni o'z ichiga olgan holda)
6. Parallel o'qishlarni xavfsiz qo'llab-quvvatlash
7. Callback nil bo'lishiga ruxsat berish (nil bo'lganda hech narsa qilmaslik)
8. Joriy operatsiyada o'qilgan baytlar va kumulyativ jami haqida hisobot berish

**Tur ta'riflari:**
\`\`\`go
// ProgressCallback ushbu operatsiyada o'qilgan baytlar va jami o'qilgan baytlarni oladi
type ProgressCallback func(current int64, total int64)

type ProgressReader struct {
    r        io.Reader
    total    int64
    callback ProgressCallback
    mu       sync.Mutex
}

func NewProgressReader(r io.Reader, callback ProgressCallback) *ProgressReader

func (pr *ProgressReader) Read(p []byte) (n int, err error)
\`\`\`

**Foydalanish misoli:**
\`\`\`go
file, _ := os.Open("large-file.dat") // 1GB fayl
defer file.Close()

// Yuklash jarayonini kuzatish
progress := NewProgressReader(file, func(current, total int64) {
    percentage := float64(total) / float64(fileSize) * 100
    fmt.Printf("\\rJarayon: %.2f%% (%d bayt)", percentage, total)
})

var dst bytes.Buffer
io.Copy(&dst, progress)
// Chiqish:
// Jarayon: 10.50% (110100480 bayt)
// Jarayon: 25.30% (265289728 bayt)
// ... doimiy yangilanadi ...
// Jarayon: 100.00% (1048576000 bayt)
\`\`\`

**Asosiy tushunchalar:**
- **Dekorator patterni**: Mavjud reader ni o'zgartirmasdan o'rash
- **Callback funksiyalari**: Chaqiruvchi kodga jarayon haqida xabar berish
- **Holat kuzatish**: Kumulyativ bayt hisoblagichini saqlash
- **Potok xavfsizligi**: Umumiy holatni mutex bilan himoyalash
- **Shaffof o'rash**: Asosiy reader kabi xuddi shunday harakat qilish

**Amalga oshirish strategiyasi:**
1. O'ralgan reader, callback va umumiy hisoblagichni saqlash
2. Parallel xavfsizlik uchun sync.Mutex bilan umumiy hisoblagichni himoyalash
3. Read() da: asosiy reader ning Read() ni chaqirish
4. Agar baytlar o'qilgan bo'lsa (n > 0), total ni yangilash va callback ni chaqirish
5. Asosiy Read() dan asl n va err ni qaytarish
6. Chaqirishdan oldin callback != nil ni tekshirish
7. Total ni yangilash/o'qishdan oldin mutex ni bloklash

**Cheklovlar:**
- O'ralgan reader ning xatti-harakatini o'zgartirmasligi kerak
- Parallel o'qishlar uchun potok-xavfsiz bo'lishi kerak
- Callback muvaffaqiyatli o'qishdan keyin chaqirilishi kerak (n > 0)
- Har qanday io.Reader amalga oshirish bilan ishlashi kerak
- Total kumulyativ o'qilgan baytlarni aniq aks ettirishi kerak`,
			hint1: `O'ralgan reader, callback funksiyasi, total hisoblagich va mutex ni saqlang. Read() da avval asosiy reader ning Read() metodini chaqiring, keyin agar n > 0 bo'lsa, total ni yangilang.`,
			hint2: `Total hisoblagichni yangilashdan oldin mutex ni bloklang, yangi total qiymatini lokal o'zgaruvchiga saqlang, keyin blokdan chiqing. Bu potok-xavfsiz kirishni ta'minlaydi. Callback bajarilayotganda blokni ushlab turmaslik uchun blokdan chiqqandan keyin callback ni chaqiring.`,
			whyItMatters: `Jarayon kuzatish uzoq davom etadigan I/O operatsiyalarini bajaradigan ilovalarda foydalanuvchi tajribasi uchun zarur.

**Nima uchun bu muhim:**

**1. Foydalanuvchi tajribasi**
Jarayon ko'rsatkichlari bo'lmasa, foydalanuvchilar bilmaydi:
- Operatsiya hali ham ishlayaptimi
- Ilova muzlab qoldimi
- Qancha kutish kerak
- Bekor qilish kerakmi

Jarayon haqida fikr-mulohaza foydalanuvchilarni xabardor qiladi va qoniqishni yaxshilaydi.

**2. Amaliy foydalanish**

**Jarayon chizig'i bilan fayl yuklash:**
\`\`\`go
func DownloadFile(url, destination string) error {
	// Fayl hajmini olish
    resp, _ := http.Head(url)
    size, _ := strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)

	// Jarayon bilan yuklash
    resp, _ = http.Get(url)
    defer resp.Body.Close()

    file, _ := os.Create(destination)
    defer file.Close()

	// Response body ni jarayon kuzatish bilan o'rash
    progress := NewProgressReader(resp.Body, func(current, total int64) {
        percent := float64(total) / float64(size) * 100
        bar := strings.Repeat("=", int(percent/2))
        fmt.Printf("\\r[%-50s] %.1f%%", bar, percent)
    })

    io.Copy(file, progress)
    fmt.Println("\\nYuklash tugallandi!")
    return nil
}
\`\`\`

**3. Production hodisa: Jim timeout**

Zahira nusxa olish xizmati bu kodga ega edi:
\`\`\`go
// YOMON - jarayon haqida fikr-mulohaza yo'q
func BackupDatabase(db *sql.DB) error {
    file, _ := os.Create("backup.sql")
    defer file.Close()

	// 2 soat davom etadi, fikr-mulohaza yo'q
    return db.Backup(file)
}
\`\`\`

Muammolar:
- Foydalanuvchilar jarayon to'xtab qolgan deb o'ylab, uni o'ldirdilar
- Tugash vaqtini baholash imkoni yo'q
- To'xtab qolgan operatsiyalarni monitoring qilish yo'q
- Qo'llab-quvvatlash chiptalar i: "Zahira nusxa olish ishlayaptimi?"

Jarayon kuzatish bilan tuzatish:
\`\`\`go
// YAXSHI - jarayon haqida fikr-mulohaza va monitoring
func BackupDatabase(db *sql.DB) error {
    file, _ := os.Create("backup.sql")
    defer file.Close()

    pipeReader, pipeWriter := io.Pipe()

	// Fonda pipe ga zahira nusxa olish
    go func() {
        db.Backup(pipeWriter)
        pipeWriter.Close()
    }()

	// Pipe dan faylga jarayonni kuzatish
    lastUpdate := time.Now()
    progress := NewProgressReader(pipeReader, func(current, total int64) {
	// Har 10 soniyada yangilash
        if time.Since(lastUpdate) > 10*time.Second {
            log.Printf("Zahira nusxa olish jarayoni: %d MB yozildi", total/1024/1024)
            lastUpdate = time.Now()
        }
    })

    _, err := io.Copy(file, progress)
    return err
}
\`\`\`

Natija:
- Foydalanuvchilar log larda jarayonni ko'rishadi
- Tugash vaqtini baholash mumkin
- To'xtab qolgan operatsiyalarni aniqlash mumkin (5 daqiqa jarayon yo'q = ogohlantirish)
- Qo'llab-quvvatlash chiptalar i 80% kamaydi

**4. Ilg'or pattern: Bir nechta fayl jarayoni**

\`\`\`go
type AggregateProgress struct {
    files     int
    completed int
    totalBytes int64
    currentBytes int64
    mu sync.Mutex
}

func (ap *AggregateProgress) NewFileReader(r io.Reader) io.Reader {
    return NewProgressReader(r, func(current, total int64) {
        ap.mu.Lock()
        ap.currentBytes += current
        progress := float64(ap.currentBytes) / float64(ap.totalBytes) * 100
        ap.mu.Unlock()

        fmt.Printf("\\rFayl %d/%d: %.1f%% tugallandi",
            ap.completed+1, ap.files, progress)
    })
}

func DownloadMultipleFiles(urls []string) error {
	// Umumiy hajmni hisoblash
    var totalSize int64
    for _, url := range urls {
        resp, _ := http.Head(url)
        size, _ := strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)
        totalSize += size
    }

    progress := &AggregateProgress{
        files: len(urls),
        totalBytes: totalSize,
    }

	// Barcha fayllarni agregat jarayon bilan yuklash
    for _, url := range urls {
        resp, _ := http.Get(url)
        reader := progress.NewFileReader(resp.Body)

        filename := path.Base(url)
        file, _ := os.Create(filename)
        io.Copy(file, reader)

        file.Close()
        resp.Body.Close()

        progress.mu.Lock()
        progress.completed++
        progress.mu.Unlock()
    }

    return nil
}
\`\`\`

**5. Dekorator patternining afzalliklari**

Dekorator patterni bu yerda kuchli:

\`\`\`go
// Bir nechta xatti-harakatlarni qatlash
file, _ := os.Open("data.txt")

// Logging qo'shish
logged := NewLoggingReader(file)

// Jarayon qo'shish
progressed := NewProgressReader(logged, progressCallback)

// Tezlikni cheklashni qo'shish
limited := NewRateLimitedReader(progressed, 1024*1024) // 1MB/s

// Birlashtirilgan reader dan foydalanish
io.Copy(destination, limited)
// Endi sizda: file -> logged -> progress -> rate limit -> destination
\`\`\`

Har bir o'rash boshqalarini o'zgartirmasdan bitta xatti-harakatni qo'shadi.

**6. Oson test qilish**

\`\`\`go
func TestProgressReader(t *testing.T) {
    data := bytes.NewReader([]byte("hello world"))

    var callCount int
    var lastTotal int64

    progress := NewProgressReader(data, func(current, total int64) {
        callCount++
        lastTotal = total

        if current <= 0 {
            t.Error("current > 0 bo'lishi kerak")
        }
        if total < current {
            t.Error("total >= current bo'lishi kerak")
        }
    })

    buf := make([]byte, 5)

	// Birinchi o'qish
    n, _ := progress.Read(buf)
    if n != 5 || callCount != 1 || lastTotal != 5 {
        t.Errorf("kutilmagan: n=%d, calls=%d, total=%d", n, callCount, lastTotal)
    }

	// Ikkinchi o'qish
    n, _ = progress.Read(buf)
    if n != 5 || callCount != 2 || lastTotal != 10 {
        t.Errorf("kutilmagan: n=%d, calls=%d, total=%d", n, callCount, lastTotal)
    }
}
\`\`\`

**7. Unumdorlik fikrlari**

\`\`\`go
// YOMON - har bir baytda callback sekin
io.Copy(dst, NewProgressReader(src, func(c, t int64) {
    fmt.Printf("\\r%d bayt", t) // Millionlab marta chaqiriladi!
}))

// YAXSHI - callback yangilanishlarini cheklash
type ThrottledProgress struct {
    lastUpdate time.Time
    interval   time.Duration
}

func (tp *ThrottledProgress) Callback(current, total int64) {
    if time.Since(tp.lastUpdate) < tp.interval {
        return // Yangilanishni o'tkazib yuborish
    }
    tp.lastUpdate = time.Now()
    fmt.Printf("\\r%d bayt", total)
}

progress := NewProgressReader(src, throttled.Callback)
io.Copy(dst, progress)
\`\`\`

**8. Monitoring va ogohlantirishlar**

\`\`\`go
func MonitoredCopy(dst io.Writer, src io.Reader, expectedSize int64) error {
    startTime := time.Now()
    lastProgress := int64(0)
    stuckTime := time.Now()

    progress := NewProgressReader(src, func(current, total int64) {
	// To'xtab qolganligini tekshirish (30 soniya jarayon yo'q)
        if total == lastProgress {
            if time.Since(stuckTime) > 30*time.Second {
                log.Printf("OGOHLANTIRISH: %d baytda 30 soniya jarayon yo'q", total)
            }
        } else {
            stuckTime = time.Now()
            lastProgress = total
        }

	// Tugash vaqtini baholash
        elapsed := time.Since(startTime)
        if total > 0 {
            rate := float64(total) / elapsed.Seconds()
            remaining := float64(expectedSize-total) / rate
            log.Printf("ETA: %.0f soniya (%.1f MB/s)", remaining, rate/1024/1024)
        }
    })

    return io.Copy(dst, progress)
}
\`\`\`

**9. Umumiy patternlar**

**Pattern 1: Jarayon chizig'i UI**
\`\`\`go
type ProgressBar struct {
    total int64
    width int
}

func (pb *ProgressBar) Callback(current, total int64) {
    percent := float64(total) / float64(pb.total) * 100
    filled := int(percent) * pb.width / 100
    bar := strings.Repeat("█", filled) + strings.Repeat("░", pb.width-filled)
    fmt.Printf("\\r[%s] %.1f%%", bar, percent)
}
\`\`\`

**Pattern 2: Webhook bildirishnomalar**
\`\`\`go
func WebhookProgress(url string, total int64) ProgressCallback {
    return func(current, total int64) {
        if total%1024*1024*100 == 0 { // Har 100MB
            json := fmt.Sprintf(\`{"current": %d, "total": %d}\`, current, total)
            http.Post(url, "application/json", strings.NewReader(json))
        }
    }
}
\`\`\`

**Pattern 3: Ma'lumotlar bazasi jarayon logi**
\`\`\`go
func DatabaseProgress(db *sql.DB, jobID int) ProgressCallback {
    return func(current, total int64) {
        db.Exec("UPDATE jobs SET bytes_transferred = ? WHERE id = ?", total, jobID)
    }
}
\`\`\`

**Asosiy xulosalar:**
- Jarayon kuzatish UX ni sezilarli yaxshilaydi
- Dekorator patterni xatti-harakatlar kompozitsiyasini ta'minlaydi
- Parallel operatsiyalar uchun potok xavfsizligi muhim
- Ishlashga ta'sir qilmaslik uchun callback larni cheklang
- Ishlab chiqarish uchun monitoring va ogohlantirishni yoqish
- Amalga oshirish oddiy, foydalanuvchi qoniqishiga katta ta'sir
- Har qanday io.Reader bilan shaffof ishlaydi`,
			solutionCode: `package interfaces

import (
	"io"
	"sync"
)

// ProgressCallback joriy operatsiya baytlari va jami o'qilgan baytlarni oladi
type ProgressCallback func(current int64, total int64)

type ProgressReader struct {
	r        io.Reader
	total    int64
	callback ProgressCallback
	mu       sync.Mutex
}

// NewProgressReader jarayon kuzatish bilan yangi reader yaratadi
func NewProgressReader(r io.Reader, callback ProgressCallback) *ProgressReader {
	return &ProgressReader{
		r:        r,
		callback: callback,
		total:    0,
	}
}

// Read io.Reader interfeysini amalga oshiradi
func (pr *ProgressReader) Read(p []byte) (n int, err error) {
	// Asosiy reader dan o'qish
	n, err = pr.r.Read(p)

	// Agar bayt o'qigan bo'lsak, total ni yangilash va jarayon haqida xabar berish
	if n > 0 {
		pr.mu.Lock()
		pr.total += int64(n)
		total := pr.total
		pr.mu.Unlock()

		// Agar callback berilgan bo'lsa, chaqirish
		if pr.callback != nil {
			pr.callback(int64(n), total)
		}
	}

	return n, err
}`
		}
	}
};

export default task;
