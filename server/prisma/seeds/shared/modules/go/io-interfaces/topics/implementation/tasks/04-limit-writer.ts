import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-interfaces-limit-writer',
	title: 'Implement Limited Writer with Byte Quota',
	difficulty: 'easy',
	tags: ['go', 'interfaces', 'io', 'writer', 'limits'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a **LimitWriter** that wraps any \`io.Writer\` and stops accepting writes after reaching a specified byte limit.

**Requirements:**
1. Wrap an existing \`io.Writer\` without modifying it
2. Accept a maximum byte limit during construction
3. Track bytes written across all Write() calls
4. Allow writes until limit is reached
5. Return \`ErrLimitExceeded\` when trying to write beyond limit
6. Support partial writes when limit is reached mid-write
7. Thread-safe for concurrent writes
8. Provide method to check remaining quota

**Type Definitions:**
\`\`\`go
var ErrLimitExceeded = errors.New("write limit exceeded")

type LimitWriter struct {
    w       io.Writer
    limit   int64
    written int64
    mu      sync.Mutex
}

func NewLimitWriter(w io.Writer, limit int64) *LimitWriter

func (lw *LimitWriter) Write(p []byte) (n int, err error)

func (lw *LimitWriter) Remaining() int64
\`\`\`

**Example Usage:**
\`\`\`go
var buf bytes.Buffer

// Allow only 10 bytes to be written
limited := NewLimitWriter(&buf, 10)

// Write 5 bytes - succeeds
n, err := limited.Write([]byte("hello"))
// n == 5, err == nil, buf contains "hello"

// Write 5 more bytes - succeeds exactly
n, err = limited.Write([]byte("world"))
// n == 5, err == nil, buf contains "helloworld"

// Try to write more - fails
n, err = limited.Write([]byte("!!!"))
// n == 0, err == ErrLimitExceeded, buf still "helloworld"

// Check remaining quota
remaining := limited.Remaining()
// remaining == 0
\`\`\`

**Partial Write Example:**
\`\`\`go
var buf bytes.Buffer
limited := NewLimitWriter(&buf, 10)

// Write 7 bytes
limited.Write([]byte("golang!"))
// buf contains "golang!", 3 bytes remaining

// Try to write 5 bytes but only 3 fit
n, err := limited.Write([]byte("12345"))
// n == 3, err == ErrLimitExceeded
// buf contains "golang!123" (partial write)
\`\`\`

**Key Concepts:**
- **Resource limiting**: Enforce quota on write operations
- **Partial writes**: Write as much as possible before limit
- **State tracking**: Maintain cumulative byte count
- **Thread safety**: Protect shared state with mutex
- **Error signaling**: Clear error when limit exceeded

**Implementation Strategy:**
1. Store wrapped writer, limit, and written counter
2. Protect written counter with sync.Mutex for thread safety
3. In Write(): check if limit already exceeded
4. Calculate available bytes: available = limit - written
5. If available <= 0, return ErrLimitExceeded immediately
6. If len(p) <= available, write all bytes normally
7. If len(p) > available, write only available bytes (partial)
8. Update written counter and return appropriate error
9. Remaining() returns limit - written (thread-safe)

**Constraints:**
- Must not change behavior of wrapped writer
- Must be thread-safe for concurrent writes
- Must support partial writes when limit is reached
- ErrLimitExceeded must be returned when over limit
- Remaining() must return accurate quota at any time`,
	initialCode: `package interfaces

import (
	"errors"
	"io"
	"sync"
)

var ErrLimitExceeded = errors.New("write limit exceeded")

type LimitWriter struct {
	// TODO: Add fields (writer, limit, written counter, mutex)
}

// NewLimitWriter creates a new limited writer
func NewLimitWriter(w io.Writer, limit int64) *LimitWriter {
	// TODO: Initialize LimitWriter
	// TODO: Implement
}

// Write implements io.Writer interface with limit enforcement
func (lw *LimitWriter) Write(p []byte) (n int, err error) {
	// TODO: Implement limited write
	// Hint: Check available space, write partial if needed
	// TODO: Implement
}

// Remaining returns the number of bytes that can still be written
func (lw *LimitWriter) Remaining() int64 {
	// TODO: Return remaining quota (thread-safe)
	// TODO: Implement
}`,
	solutionCode: `package interfaces

import (
	"errors"
	"io"
	"sync"
)

var ErrLimitExceeded = errors.New("write limit exceeded")

type LimitWriter struct {
	w       io.Writer
	limit   int64
	written int64
	mu      sync.Mutex
}

// NewLimitWriter creates a new limited writer
func NewLimitWriter(w io.Writer, limit int64) *LimitWriter {
	return &LimitWriter{
		w:       w,
		limit:   limit,
		written: 0,
	}
}

// Write implements io.Writer interface with limit enforcement
func (lw *LimitWriter) Write(p []byte) (n int, err error) {
	lw.mu.Lock()
	defer lw.mu.Unlock()

	// Calculate available space
	available := lw.limit - lw.written

	// If no space available, reject immediately
	if available <= 0 {
		return 0, ErrLimitExceeded
	}

	// Determine how much to write
	toWrite := int64(len(p))
	if toWrite > available {
		// Partial write - only write what fits
		toWrite = available
	}

	// Write to underlying writer
	n, err = lw.w.Write(p[:toWrite])
	lw.written += int64(n)

	// If we hit the limit, return ErrLimitExceeded
	if lw.written >= lw.limit && err == nil {
		err = ErrLimitExceeded
	}

	return n, err
}

// Remaining returns the number of bytes that can still be written
func (lw *LimitWriter) Remaining() int64 {
	lw.mu.Lock()
	defer lw.mu.Unlock()

	remaining := lw.limit - lw.written
	if remaining < 0 {
		return 0
	}
	return remaining
}`,
	testCode: `package interfaces

import (
	"bytes"
	"errors"
	"sync"
	"testing"
)

func Test1(t *testing.T) {
	// Basic write within limit
	var buf bytes.Buffer
	lw := NewLimitWriter(&buf, 10)
	n, err := lw.Write([]byte("hello"))
	if n != 5 || err != nil {
		t.Errorf("expected n=5, err=nil, got n=%d, err=%v", n, err)
	}
	if buf.String() != "hello" {
		t.Errorf("expected 'hello', got '%s'", buf.String())
	}
}

func Test2(t *testing.T) {
	// Write exactly to limit
	var buf bytes.Buffer
	lw := NewLimitWriter(&buf, 10)
	n, err := lw.Write([]byte("0123456789"))
	if n != 10 {
		t.Errorf("expected n=10, got n=%d", n)
	}
	if err != ErrLimitExceeded {
		t.Errorf("expected ErrLimitExceeded, got %v", err)
	}
	if buf.String() != "0123456789" {
		t.Errorf("expected '0123456789', got '%s'", buf.String())
	}
}

func Test3(t *testing.T) {
	// Partial write when limit reached mid-write
	var buf bytes.Buffer
	lw := NewLimitWriter(&buf, 5)
	n, err := lw.Write([]byte("0123456789"))
	if n != 5 {
		t.Errorf("expected n=5, got n=%d", n)
	}
	if err != ErrLimitExceeded {
		t.Errorf("expected ErrLimitExceeded, got %v", err)
	}
	if buf.String() != "01234" {
		t.Errorf("expected '01234', got '%s'", buf.String())
	}
}

func Test4(t *testing.T) {
	// Write after limit exceeded
	var buf bytes.Buffer
	lw := NewLimitWriter(&buf, 5)
	lw.Write([]byte("12345"))
	n, err := lw.Write([]byte("extra"))
	if n != 0 {
		t.Errorf("expected n=0, got n=%d", n)
	}
	if err != ErrLimitExceeded {
		t.Errorf("expected ErrLimitExceeded, got %v", err)
	}
}

func Test5(t *testing.T) {
	// Multiple writes accumulating
	var buf bytes.Buffer
	lw := NewLimitWriter(&buf, 10)
	lw.Write([]byte("abc"))
	lw.Write([]byte("def"))
	n, err := lw.Write([]byte("ghijk"))
	if n != 4 {
		t.Errorf("expected n=4, got n=%d", n)
	}
	if err != ErrLimitExceeded {
		t.Errorf("expected ErrLimitExceeded, got %v", err)
	}
	if buf.String() != "abcdefghij" {
		t.Errorf("expected 'abcdefghij', got '%s'", buf.String())
	}
}

func Test6(t *testing.T) {
	// Remaining() accuracy
	var buf bytes.Buffer
	lw := NewLimitWriter(&buf, 10)
	if lw.Remaining() != 10 {
		t.Errorf("expected 10 remaining, got %d", lw.Remaining())
	}
	lw.Write([]byte("abc"))
	if lw.Remaining() != 7 {
		t.Errorf("expected 7 remaining, got %d", lw.Remaining())
	}
	lw.Write([]byte("1234567"))
	if lw.Remaining() != 0 {
		t.Errorf("expected 0 remaining, got %d", lw.Remaining())
	}
}

func Test7(t *testing.T) {
	// Empty write
	var buf bytes.Buffer
	lw := NewLimitWriter(&buf, 10)
	n, err := lw.Write([]byte{})
	if n != 0 || err != nil {
		t.Errorf("expected n=0, err=nil, got n=%d, err=%v", n, err)
	}
	if lw.Remaining() != 10 {
		t.Errorf("expected 10 remaining, got %d", lw.Remaining())
	}
}

func Test8(t *testing.T) {
	// Zero limit edge case
	var buf bytes.Buffer
	lw := NewLimitWriter(&buf, 0)
	n, err := lw.Write([]byte("data"))
	if n != 0 {
		t.Errorf("expected n=0, got n=%d", n)
	}
	if err != ErrLimitExceeded {
		t.Errorf("expected ErrLimitExceeded, got %v", err)
	}
	if lw.Remaining() != 0 {
		t.Errorf("expected 0 remaining, got %d", lw.Remaining())
	}
}

func Test9(t *testing.T) {
	// Large write
	var buf bytes.Buffer
	lw := NewLimitWriter(&buf, 1000)
	data := make([]byte, 500)
	for i := range data {
		data[i] = 'x'
	}
	n, err := lw.Write(data)
	if n != 500 || err != nil {
		t.Errorf("expected n=500, err=nil, got n=%d, err=%v", n, err)
	}
	if lw.Remaining() != 500 {
		t.Errorf("expected 500 remaining, got %d", lw.Remaining())
	}
}

func Test10(t *testing.T) {
	// Concurrent writes (thread safety)
	var buf bytes.Buffer
	lw := NewLimitWriter(&buf, 100)
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			lw.Write([]byte("test"))
		}()
	}
	wg.Wait()
	// After 10 writes of 4 bytes each = 40 bytes
	if lw.Remaining() != 60 {
		t.Errorf("expected 60 remaining after concurrent writes, got %d", lw.Remaining())
	}
	if len(buf.String()) != 40 {
		t.Errorf("expected 40 bytes written, got %d", len(buf.String()))
	}
}
`,
	hint1: `In Write(), first lock the mutex and calculate available = limit - written. If available <= 0, return ErrLimitExceeded immediately. Otherwise, determine toWrite = min(len(p), available).`,
	hint2: `After writing to the underlying writer, update the written counter. If written >= limit, set err to ErrLimitExceeded before returning. This ensures the caller knows the limit was reached even on partial writes.`,
	whyItMatters: `Write limits are critical for preventing resource exhaustion, enforcing quotas, and protecting systems from unbounded writes.

**Why This Matters:**

**1. Resource Protection**
Unbounded writes can cause:
- Disk space exhaustion
- Memory exhaustion
- Database quota violations
- Network bandwidth abuse
- Cost overruns (cloud storage)

LimitWriter provides a safety mechanism.

**2. Real-World Usage**

**HTTP Upload Size Limit:**
\`\`\`go
func HandleUpload(w http.ResponseWriter, r *http.Request) {
    maxSize := int64(10 * 1024 * 1024) // 10MB limit

	// Create file with size limit
    file, _ := os.Create("upload.dat")
    defer file.Close()

    limited := NewLimitWriter(file, maxSize)

	// Copy request body with limit enforcement
    _, err := io.Copy(limited, r.Body)
    if err == ErrLimitExceeded {
        file.Close()
        os.Remove("upload.dat") // Clean up
        http.Error(w, "File size exceeds 10MB limit", 413)
        return
    }

    if err != nil {
        http.Error(w, "Upload failed", 500)
        return
    }

    fmt.Fprintf(w, "Upload successful")
}
\`\`\`

**3. Production Incident: Disk Space Attack**

A logging service had this code:
\`\`\`go
// BAD - no write limits
func LogRequest(r *http.Request) {
    logFile, _ := os.OpenFile("requests.log", os.O_APPEND|os.O_CREATE, 0644)
    defer logFile.Close()

	// Attacker sends huge request bodies
    io.Copy(logFile, r.Body) // Writes unlimited data!
}
\`\`\`

Attack scenario:
- Attacker sends 1000 requests with 1GB bodies each
- Fills 1TB disk in minutes
- Service crashes (no disk space)
- All services on same machine fail
- Recovery takes hours

Fix with LimitWriter:
\`\`\`go
// GOOD - enforce per-request write limit
func LogRequest(r *http.Request) {
    logFile, _ := os.OpenFile("requests.log", os.O_APPEND|os.O_CREATE, 0644)
    defer logFile.Close()

	// Limit each request log to 1MB
    limited := NewLimitWriter(logFile, 1024*1024)

    n, err := io.Copy(limited, r.Body)
    if err == ErrLimitExceeded {
        fmt.Fprintf(logFile, "\n[TRUNCATED: %d bytes]\n", n)
    }
}
\`\`\`

Result:
- Maximum 1MB per request logged
- Disk space protected
- Attack mitigated
- Service remains stable

**4. Per-User Quota Enforcement**

\`\`\`go
type UserQuota struct {
    userID string
    limit  int64
    used   int64
    mu     sync.Mutex
}

func (uq *UserQuota) NewWriter(w io.Writer) *LimitWriter {
    uq.mu.Lock()
    remaining := uq.limit - uq.used
    uq.mu.Unlock()

    return NewLimitWriter(w, remaining)
}

func (uq *UserQuota) UpdateUsage(written int64) {
    uq.mu.Lock()
    uq.used += written
    uq.mu.Unlock()
}

func HandleUserUpload(userID string, w http.ResponseWriter, r *http.Request) {
	// Get user's quota
    quota := GetUserQuota(userID) // 100MB per user

	// Create limited writer
    file, _ := os.Create(fmt.Sprintf("uploads/%s/file.dat", userID))
    defer file.Close()

    limited := quota.NewWriter(file)

    n, err := io.Copy(limited, r.Body)

	// Update user's quota
    quota.UpdateUsage(n)

    if err == ErrLimitExceeded {
        http.Error(w, fmt.Sprintf("Quota exceeded. Used: %d MB", quota.used/1024/1024), 507)
        return
    }

    fmt.Fprintf(w, "Uploaded %d bytes. Remaining quota: %d MB",
        n, quota.NewWriter(nil).Remaining()/1024/1024)
}
\`\`\`

**5. Response Size Limiting**

**Prevent Oversized API Responses:**
\`\`\`go
func HandleAPIRequest(w http.ResponseWriter, r *http.Request) {
	// Limit response to 5MB
    limited := NewLimitWriter(w, 5*1024*1024)

	// Query database (potentially large result)
    rows, _ := db.Query("SELECT * FROM large_table WHERE user_id = ?", userID)
    defer rows.Close()

    encoder := json.NewEncoder(limited)

    for rows.Next() {
        var record Record
        rows.Scan(&record)

        err := encoder.Encode(record)
        if err == ErrLimitExceeded {
	// Response too large, return error
            w.WriteHeader(413)
            w.Write([]byte(\`{"error": "Response too large"}\`))
            return
        }
    }
}
\`\`\`

**6. Log Rotation with Size Limits**

\`\`\`go
type SizeRotatingWriter struct {
    filename    string
    maxSize     int64
    currentFile *os.File
    currentSize int64
    mu          sync.Mutex
}

func (srw *SizeRotatingWriter) Write(p []byte) (n int, err error) {
    srw.mu.Lock()
    defer srw.mu.Unlock()

	// Check if we need to rotate
    if srw.currentSize+int64(len(p)) > srw.maxSize {
        srw.rotate()
    }

	// Write with limit enforcement
    limited := NewLimitWriter(srw.currentFile, srw.maxSize-srw.currentSize)
    n, err = limited.Write(p)
    srw.currentSize += int64(n)

    if err == ErrLimitExceeded {
        srw.rotate()
	// Try again with new file
        return srw.currentFile.Write(p)
    }

    return n, err
}

func (srw *SizeRotatingWriter) rotate() {
    if srw.currentFile != nil {
        srw.currentFile.Close()
    }

	// Create new log file
    timestamp := time.Now().Format("20060102-150405")
    newFile := fmt.Sprintf("%s.%s", srw.filename, timestamp)
    srw.currentFile, _ = os.Create(newFile)
    srw.currentSize = 0
}
\`\`\`

**7. Testing with Limits**

\`\`\`go
func TestLimitWriter_ExactLimit(t *testing.T) {
    var buf bytes.Buffer
    limited := NewLimitWriter(&buf, 10)

	// Write exactly to limit
    n, err := limited.Write([]byte("0123456789"))
    if n != 10 || err != ErrLimitExceeded {
        t.Errorf("expected n=10, err=ErrLimitExceeded, got n=%d, err=%v", n, err)
    }

    if buf.String() != "0123456789" {
        t.Errorf("expected '0123456789', got '%s'", buf.String())
    }

	// No remaining quota
    if limited.Remaining() != 0 {
        t.Errorf("expected 0 remaining, got %d", limited.Remaining())
    }
}

func TestLimitWriter_PartialWrite(t *testing.T) {
    var buf bytes.Buffer
    limited := NewLimitWriter(&buf, 5)

	// Try to write 10 bytes, only 5 fit
    n, err := limited.Write([]byte("0123456789"))
    if n != 5 || err != ErrLimitExceeded {
        t.Errorf("expected n=5, err=ErrLimitExceeded, got n=%d, err=%v", n, err)
    }

    if buf.String() != "01234" {
        t.Errorf("expected '01234', got '%s'", buf.String())
    }
}

func TestLimitWriter_MultipleWrites(t *testing.T) {
    var buf bytes.Buffer
    limited := NewLimitWriter(&buf, 10)

	// First write
    n, err := limited.Write([]byte("abc"))
    if n != 3 || err != nil {
        t.Errorf("write 1: expected n=3, err=nil, got n=%d, err=%v", n, err)
    }

	// Second write
    n, err = limited.Write([]byte("defg"))
    if n != 4 || err != nil {
        t.Errorf("write 2: expected n=4, err=nil, got n=%d, err=%v", n, err)
    }

	// Third write (partial)
    n, err = limited.Write([]byte("hijkl"))
    if n != 3 || err != ErrLimitExceeded {
        t.Errorf("write 3: expected n=3, err=ErrLimitExceeded, got n=%d, err=%v", n, err)
    }

    if buf.String() != "abcdefghij" {
        t.Errorf("expected 'abcdefghij', got '%s'", buf.String())
    }
}
\`\`\`

**8. Cloud Storage Cost Control**

\`\`\`go
func UploadToS3WithQuota(bucket, key string, data io.Reader, quota int64) error {
	// Create temporary file with limit
    tmpFile, _ := os.CreateTemp("", "upload-*")
    defer os.Remove(tmpFile.Name())
    defer tmpFile.Close()

	// Write to temp file with limit
    limited := NewLimitWriter(tmpFile, quota)
    written, err := io.Copy(limited, data)

    if err == ErrLimitExceeded {
        return fmt.Errorf("upload exceeds quota of %d bytes", quota)
    }

    if err != nil {
        return fmt.Errorf("upload failed: %w", err)
    }

	// Seek back to start for S3 upload
    tmpFile.Seek(0, 0)

	// Upload to S3
    uploader := s3manager.NewUploader(sess)
    _, err = uploader.Upload(&s3manager.UploadInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
        Body:   tmpFile,
    })

    log.Printf("Uploaded %d bytes to S3 (quota: %d)", written, quota)
    return err
}
\`\`\`

**9. Rate Limiting with Token Bucket**

\`\`\`go
type TokenBucket struct {
    tokens    int64
    refillRate int64 // bytes per second
    lastRefill time.Time
    mu sync.Mutex
}

func (tb *TokenBucket) NewWriter(w io.Writer) io.Writer {
    tb.mu.Lock()
    tb.refill()
    available := tb.tokens
    tb.mu.Unlock()

    return NewLimitWriter(w, available)
}

func (tb *TokenBucket) refill() {
    now := time.Now()
    elapsed := now.Sub(tb.lastRefill).Seconds()
    newTokens := int64(elapsed * float64(tb.refillRate))

    tb.tokens += newTokens
    tb.lastRefill = now
}

// Usage: limit write rate to 1MB/s
bucket := &TokenBucket{
    tokens: 1024 * 1024,
    refillRate: 1024 * 1024,
    lastRefill: time.Now(),
}

limitedWriter := bucket.NewWriter(file)
io.Copy(limitedWriter, data) // Max 1MB/s
\`\`\`

**10. Composite Limit Enforcement**

\`\`\`go
type MultiLimitWriter struct {
    writers []*LimitWriter
}

func (mlw *MultiLimitWriter) Write(p []byte) (n int, err error) {
	// Write to all limited writers
	// Fail if any limit is exceeded
    for i, lw := range mlw.writers {
        n, err = lw.Write(p)
        if err != nil {
            return n, fmt.Errorf("limit %d exceeded: %w", i, err)
        }
    }
    return n, nil
}

// Enforce multiple limits simultaneously
func NewMultiLimitWriter(w io.Writer, perWriteLimit, totalLimit, dailyLimit int64) io.Writer {
    return &MultiLimitWriter{
        writers: []*LimitWriter{
            NewLimitWriter(w, perWriteLimit),	// Per-operation limit
            NewLimitWriter(w, totalLimit),	// Total session limit
            NewLimitWriter(w, dailyLimit),	// Daily quota limit
        },
    }
}
\`\`\`

**Key Takeaways:**
- LimitWriter prevents resource exhaustion attacks
- Essential for enforcing quotas and cost control
- Supports partial writes for graceful degradation
- Thread-safe by design for concurrent operations
- Simple to implement, critical for production systems
- Composable with other io.Writer wrappers
- Enables fine-grained resource control
- Protects both disk space and memory`,
	order: 3,
	translations: {
		ru: {
			title: 'Реализация Writer с ограничением по байтам',
			description: `Реализуйте **LimitWriter**, который оборачивает любой \`io.Writer\` и прекращает принимать записи после достижения указанного лимита байт.

**Требования:**
1. Обернуть существующий \`io.Writer\` без его модификации
2. Принять максимальный лимит байт при создании
3. Отслеживать записанные байты через все вызовы Write()
4. Разрешать запись до достижения лимита
5. Возвращать \`ErrLimitExceeded\` при попытке записи за пределами лимита
6. Поддерживать частичные записи когда лимит достигается в середине записи
7. Потокобезопасность для параллельных записей
8. Предоставить метод для проверки оставшейся квоты

**Определение типов:**
\`\`\`go
var ErrLimitExceeded = errors.New("write limit exceeded")

type LimitWriter struct {
    w       io.Writer
    limit   int64
    written int64
    mu      sync.Mutex
}

func NewLimitWriter(w io.Writer, limit int64) *LimitWriter

func (lw *LimitWriter) Write(p []byte) (n int, err error)

func (lw *LimitWriter) Remaining() int64
\`\`\`

**Пример использования:**
\`\`\`go
var buf bytes.Buffer

// Разрешить записать только 10 байт
limited := NewLimitWriter(&buf, 10)

// Запись 5 байт - успех
n, err := limited.Write([]byte("hello"))
// n == 5, err == nil, buf содержит "hello"

// Запись еще 5 байт - успех точно
n, err = limited.Write([]byte("world"))
// n == 5, err == nil, buf содержит "helloworld"

// Попытка записать больше - неудача
n, err = limited.Write([]byte("!!!"))
// n == 0, err == ErrLimitExceeded, buf все еще "helloworld"

// Проверить оставшуюся квоту
remaining := limited.Remaining()
// remaining == 0
\`\`\`

**Пример частичной записи:**
\`\`\`go
var buf bytes.Buffer
limited := NewLimitWriter(&buf, 10)

// Записать 7 байт
limited.Write([]byte("golang!"))
// buf содержит "golang!", осталось 3 байта

// Попытка записать 5 байт, но только 3 помещаются
n, err := limited.Write([]byte("12345"))
// n == 3, err == ErrLimitExceeded
// buf содержит "golang!123" (частичная запись)
\`\`\`

**Ключевые концепции:**
- **Ограничение ресурсов**: Применение квоты на операции записи
- **Частичные записи**: Запись максимально возможного до лимита
- **Отслеживание состояния**: Поддержка кумулятивного счетчика байт
- **Потокобезопасность**: Защита общего состояния мьютексом
- **Сигнализация ошибок**: Четкая ошибка при превышении лимита

**Стратегия реализации:**
1. Хранить обернутый writer, лимит и счетчик записанных байт
2. Защитить счетчик записанных байт с помощью sync.Mutex для потокобезопасности
3. В Write(): проверить не превышен ли уже лимит
4. Вычислить доступные байты: available = limit - written
5. Если available <= 0, вернуть ErrLimitExceeded немедленно
6. Если len(p) <= available, записать все байты нормально
7. Если len(p) > available, записать только available байт (частично)
8. Обновить счетчик written и вернуть соответствующую ошибку
9. Remaining() возвращает limit - written (потокобезопасно)

**Ограничения:**
- Не должна изменять поведение обернутого writer
- Должна быть потокобезопасной для параллельных записей
- Должна поддерживать частичные записи при достижении лимита
- ErrLimitExceeded должен возвращаться при превышении лимита
- Remaining() должен возвращать точную квоту в любое время`,
			hint1: `В Write() сначала заблокируйте мьютекс и вычислите available = limit - written. Если available <= 0, вернуть ErrLimitExceeded немедленно. Иначе определите toWrite = min(len(p), available).`,
			hint2: `После записи в базовый writer обновите счетчик written. Если written >= limit, установите err в ErrLimitExceeded перед возвратом. Это гарантирует что вызывающий код знает о достижении лимита даже при частичных записях.`,
			whyItMatters: `Ограничения записи критичны для предотвращения истощения ресурсов, применения квот и защиты систем от неограниченных записей.

**Почему это важно:**

**1. Защита ресурсов**
Неограниченные записи могут вызвать:
- Истощение дискового пространства
- Истощение памяти
- Нарушение квот базы данных
- Злоупотребление пропускной способностью сети
- Превышение стоимости (облачное хранилище)

LimitWriter обеспечивает механизм защиты.

**2. Реальное использование**

**Ограничение размера HTTP загрузки:**
\`\`\`go
func HandleUpload(w http.ResponseWriter, r *http.Request) {
    maxSize := int64(10 * 1024 * 1024) // лимит 10MB

	// Создать файл с ограничением размера
    file, _ := os.Create("upload.dat")
    defer file.Close()

    limited := NewLimitWriter(file, maxSize)

	// Копировать тело запроса с применением лимита
    _, err := io.Copy(limited, r.Body)
    if err == ErrLimitExceeded {
        file.Close()
        os.Remove("upload.dat") // Очистка
        http.Error(w, "Размер файла превышает лимит 10MB", 413)
        return
    }

    if err != nil {
        http.Error(w, "Загрузка не удалась", 500)
        return
    }

    fmt.Fprintf(w, "Загрузка успешна")
}
\`\`\`

**3. Продакшен инцидент: Атака на дисковое пространство**

Сервис логирования имел этот код:
\`\`\`go
// ПЛОХО - нет ограничений записи
func LogRequest(r *http.Request) {
    logFile, _ := os.OpenFile("requests.log", os.O_APPEND|os.O_CREATE, 0644)
    defer logFile.Close()

	// Атакующий отправляет огромные тела запросов
    io.Copy(logFile, r.Body) // Записывает неограниченные данные!
}
\`\`\`

Сценарий атаки:
- Атакующий отправляет 1000 запросов с телами по 1GB каждый
- Заполняет 1TB диска за минуты
- Сервис падает (нет места на диске)
- Все сервисы на той же машине падают
- Восстановление занимает часы

Исправление с LimitWriter:
\`\`\`go
// ХОРОШО - применить ограничение записи на запрос
func LogRequest(r *http.Request) {
    logFile, _ := os.OpenFile("requests.log", os.O_APPEND|os.O_CREATE, 0644)
    defer logFile.Close()

	// Ограничить каждый лог запроса до 1MB
    limited := NewLimitWriter(logFile, 1024*1024)

    n, err := io.Copy(limited, r.Body)
    if err == ErrLimitExceeded {
        fmt.Fprintf(logFile, "\\n[ОБРЕЗАНО: %d байт]\\n", n)
    }
}
\`\`\`

Результат:
- Максимум 1MB на запрос залогировано
- Дисковое пространство защищено
- Атака предотвращена
- Сервис остается стабильным

**4. Применение квот на пользователя**

\`\`\`go
type UserQuota struct {
    userID string
    limit  int64
    used   int64
    mu     sync.Mutex
}

func (uq *UserQuota) NewWriter(w io.Writer) *LimitWriter {
    uq.mu.Lock()
    remaining := uq.limit - uq.used
    uq.mu.Unlock()

    return NewLimitWriter(w, remaining)
}

func (uq *UserQuota) UpdateUsage(written int64) {
    uq.mu.Lock()
    uq.used += written
    uq.mu.Unlock()
}

func HandleUserUpload(userID string, w http.ResponseWriter, r *http.Request) {
	// Получить квоту пользователя
    quota := GetUserQuota(userID) // 100MB на пользователя

	// Создать ограниченный writer
    file, _ := os.Create(fmt.Sprintf("uploads/%s/file.dat", userID))
    defer file.Close()

    limited := quota.NewWriter(file)

    n, err := io.Copy(limited, r.Body)

	// Обновить квоту пользователя
    quota.UpdateUsage(n)

    if err == ErrLimitExceeded {
        http.Error(w, fmt.Sprintf("Квота превышена. Использовано: %d MB", quota.used/1024/1024), 507)
        return
    }

    fmt.Fprintf(w, "Загружено %d байт. Осталось квоты: %d MB",
        n, quota.NewWriter(nil).Remaining()/1024/1024)
}
\`\`\`

**5. Ограничение размера ответа**

**Предотвращение слишком больших API ответов:**
\`\`\`go
func HandleAPIRequest(w http.ResponseWriter, r *http.Request) {
	// Ограничить ответ до 5MB
    limited := NewLimitWriter(w, 5*1024*1024)

	// Запрос к базе данных (потенциально большой результат)
    rows, _ := db.Query("SELECT * FROM large_table WHERE user_id = ?", userID)
    defer rows.Close()

    encoder := json.NewEncoder(limited)

    for rows.Next() {
        var record Record
        rows.Scan(&record)

        err := encoder.Encode(record)
        if err == ErrLimitExceeded {
	// Ответ слишком большой, вернуть ошибку
            w.WriteHeader(413)
            w.Write([]byte(\`{"error": "Response too large"}\`))
            return
        }
    }
}
\`\`\`

**6. Ротация логов с ограничениями по размеру**

\`\`\`go
type SizeRotatingWriter struct {
    filename    string
    maxSize     int64
    currentFile *os.File
    currentSize int64
    mu          sync.Mutex
}

func (srw *SizeRotatingWriter) Write(p []byte) (n int, err error) {
    srw.mu.Lock()
    defer srw.mu.Unlock()

	// Проверить нужна ли ротация
    if srw.currentSize+int64(len(p)) > srw.maxSize {
        srw.rotate()
    }

	// Записать с применением лимита
    limited := NewLimitWriter(srw.currentFile, srw.maxSize-srw.currentSize)
    n, err = limited.Write(p)
    srw.currentSize += int64(n)

    if err == ErrLimitExceeded {
        srw.rotate()
	// Попробовать снова с новым файлом
        return srw.currentFile.Write(p)
    }

    return n, err
}

func (srw *SizeRotatingWriter) rotate() {
    if srw.currentFile != nil {
        srw.currentFile.Close()
    }

	// Создать новый файл лога
    timestamp := time.Now().Format("20060102-150405")
    newFile := fmt.Sprintf("%s.%s", srw.filename, timestamp)
    srw.currentFile, _ = os.Create(newFile)
    srw.currentSize = 0
}
\`\`\`

**7. Тестирование с лимитами**

\`\`\`go
func TestLimitWriter_ExactLimit(t *testing.T) {
    var buf bytes.Buffer
    limited := NewLimitWriter(&buf, 10)

	// Записать точно до лимита
    n, err := limited.Write([]byte("0123456789"))
    if n != 10 || err != ErrLimitExceeded {
        t.Errorf("ожидалось n=10, err=ErrLimitExceeded, получено n=%d, err=%v", n, err)
    }

    if buf.String() != "0123456789" {
        t.Errorf("ожидалось '0123456789', получено '%s'", buf.String())
    }

	// Нет оставшейся квоты
    if limited.Remaining() != 0 {
        t.Errorf("ожидалось 0 remaining, получено %d", limited.Remaining())
    }
}

func TestLimitWriter_PartialWrite(t *testing.T) {
    var buf bytes.Buffer
    limited := NewLimitWriter(&buf, 5)

	// Попытка записать 10 байт, только 5 помещается
    n, err := limited.Write([]byte("0123456789"))
    if n != 5 || err != ErrLimitExceeded {
        t.Errorf("ожидалось n=5, err=ErrLimitExceeded, получено n=%d, err=%v", n, err)
    }

    if buf.String() != "01234" {
        t.Errorf("ожидалось '01234', получено '%s'", buf.String())
    }
}

func TestLimitWriter_MultipleWrites(t *testing.T) {
    var buf bytes.Buffer
    limited := NewLimitWriter(&buf, 10)

	// Первая запись
    n, err := limited.Write([]byte("abc"))
    if n != 3 || err != nil {
        t.Errorf("запись 1: ожидалось n=3, err=nil, получено n=%d, err=%v", n, err)
    }

	// Вторая запись
    n, err = limited.Write([]byte("defg"))
    if n != 4 || err != nil {
        t.Errorf("запись 2: ожидалось n=4, err=nil, получено n=%d, err=%v", n, err)
    }

	// Третья запись (частичная)
    n, err = limited.Write([]byte("hijkl"))
    if n != 3 || err != ErrLimitExceeded {
        t.Errorf("запись 3: ожидалось n=3, err=ErrLimitExceeded, получено n=%d, err=%v", n, err)
    }

    if buf.String() != "abcdefghij" {
        t.Errorf("ожидалось 'abcdefghij', получено '%s'", buf.String())
    }
}
\`\`\`

**8. Контроль расходов облачного хранилища**

\`\`\`go
func UploadToS3WithQuota(bucket, key string, data io.Reader, quota int64) error {
	// Создать временный файл с лимитом
    tmpFile, _ := os.CreateTemp("", "upload-*")
    defer os.Remove(tmpFile.Name())
    defer tmpFile.Close()

	// Записать во временный файл с лимитом
    limited := NewLimitWriter(tmpFile, quota)
    written, err := io.Copy(limited, data)

    if err == ErrLimitExceeded {
        return fmt.Errorf("загрузка превышает квоту %d байт", quota)
    }

    if err != nil {
        return fmt.Errorf("загрузка не удалась: %w", err)
    }

	// Вернуться к началу для загрузки в S3
    tmpFile.Seek(0, 0)

	// Загрузить в S3
    uploader := s3manager.NewUploader(sess)
    _, err = uploader.Upload(&s3manager.UploadInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
        Body:   tmpFile,
    })

    log.Printf("Загружено %d байт в S3 (квота: %d)", written, quota)
    return err
}
\`\`\`

**9. Ограничение скорости с Token Bucket**

\`\`\`go
type TokenBucket struct {
    tokens    int64
    refillRate int64 // байт в секунду
    lastRefill time.Time
    mu sync.Mutex
}

func (tb *TokenBucket) NewWriter(w io.Writer) io.Writer {
    tb.mu.Lock()
    tb.refill()
    available := tb.tokens
    tb.mu.Unlock()

    return NewLimitWriter(w, available)
}

func (tb *TokenBucket) refill() {
    now := time.Now()
    elapsed := now.Sub(tb.lastRefill).Seconds()
    newTokens := int64(elapsed * float64(tb.refillRate))

    tb.tokens += newTokens
    tb.lastRefill = now
}

// Использование: ограничить скорость записи до 1MB/s
bucket := &TokenBucket{
    tokens: 1024 * 1024,
    refillRate: 1024 * 1024,
    lastRefill: time.Now(),
}

limitedWriter := bucket.NewWriter(file)
io.Copy(limitedWriter, data) // Максимум 1MB/s
\`\`\`

**10. Составное применение лимитов**

\`\`\`go
type MultiLimitWriter struct {
    writers []*LimitWriter
}

func (mlw *MultiLimitWriter) Write(p []byte) (n int, err error) {
	// Записать во все ограниченные writers
	// Отказать если какой-либо лимит превышен
    for i, lw := range mlw.writers {
        n, err = lw.Write(p)
        if err != nil {
            return n, fmt.Errorf("лимит %d превышен: %w", i, err)
        }
    }
    return n, nil
}

// Применить несколько лимитов одновременно
func NewMultiLimitWriter(w io.Writer, perWriteLimit, totalLimit, dailyLimit int64) io.Writer {
    return &MultiLimitWriter{
        writers: []*LimitWriter{
            NewLimitWriter(w, perWriteLimit),	// Лимит на операцию
            NewLimitWriter(w, totalLimit),	// Лимит на сессию
            NewLimitWriter(w, dailyLimit),	// Дневная квота
        },
    }
}
\`\`\`

**Ключевые выводы:**
- LimitWriter предотвращает атаки истощения ресурсов
- Необходим для применения квот и контроля стоимости
- Поддерживает частичные записи для плавной деградации
- Потокобезопасен по дизайну для параллельных операций
- Просто реализовать, критичен для продакшен систем
- Компонуется с другими io.Writer обертками
- Обеспечивает детальный контроль ресурсов
- Защищает как дисковое пространство так и память`,
			solutionCode: `package interfaces

import (
	"errors"
	"io"
	"sync"
)

var ErrLimitExceeded = errors.New("write limit exceeded")

type LimitWriter struct {
	w       io.Writer
	limit   int64
	written int64
	mu      sync.Mutex
}

// NewLimitWriter создает новый writer с ограничением
func NewLimitWriter(w io.Writer, limit int64) *LimitWriter {
	return &LimitWriter{
		w:       w,
		limit:   limit,
		written: 0,
	}
}

// Write реализует интерфейс io.Writer с применением лимита
func (lw *LimitWriter) Write(p []byte) (n int, err error) {
	lw.mu.Lock()
	defer lw.mu.Unlock()

	// Вычислить доступное пространство
	available := lw.limit - lw.written

	// Если места нет, отклонить немедленно
	if available <= 0 {
		return 0, ErrLimitExceeded
	}

	// Определить сколько записать
	toWrite := int64(len(p))
	if toWrite > available {
		// Частичная запись - записать только что помещается
		toWrite = available
	}

	// Записать в базовый writer
	n, err = lw.w.Write(p[:toWrite])
	lw.written += int64(n)

	// Если достигнут лимит, вернуть ErrLimitExceeded
	if lw.written >= lw.limit && err == nil {
		err = ErrLimitExceeded
	}

	return n, err
}

// Remaining возвращает количество байт которые можно еще записать
func (lw *LimitWriter) Remaining() int64 {
	lw.mu.Lock()
	defer lw.mu.Unlock()

	remaining := lw.limit - lw.written
	if remaining < 0 {
		return 0
	}
	return remaining
}`
		},
		uz: {
			title: `Bayt kvotasi bilan cheklangan Writer ni amalga oshirish`,
			description: `Har qanday \`io.Writer\` ni o'rab, belgilangan bayt limitiga yetgandan keyin yozishni to'xtatuvchi **LimitWriter** ni amalga oshiring.

**Talablar:**
1. Mavjud \`io.Writer\` ni o'zgartirmasdan o'rash
2. Yaratish vaqtida maksimal bayt limitini qabul qilish
3. Barcha Write() chaqiruvlar orqali yozilgan baytlarni kuzatish
4. Limit yetgunga qadar yozishga ruxsat berish
5. Limitdan oshib yozishga urinishda \`ErrLimitExceeded\` qaytarish
6. Yozish o'rtasida limit yetganda qisman yozishni qo'llab-quvvatlash
7. Parallel yozishlar uchun potok-xavfsiz
8. Qolgan kvotani tekshirish uchun metod taqdim etish

**Tur ta'riflari:**
\`\`\`go
var ErrLimitExceeded = errors.New("write limit exceeded")

type LimitWriter struct {
    w       io.Writer
    limit   int64
    written int64
    mu      sync.Mutex
}

func NewLimitWriter(w io.Writer, limit int64) *LimitWriter

func (lw *LimitWriter) Write(p []byte) (n int, err error)

func (lw *LimitWriter) Remaining() int64
\`\`\`

**Foydalanish misoli:**
\`\`\`go
var buf bytes.Buffer

// Faqat 10 bayt yozishga ruxsat berish
limited := NewLimitWriter(&buf, 10)

// 5 bayt yozish - muvaffaqiyat
n, err := limited.Write([]byte("hello"))
// n == 5, err == nil, buf "hello" ni o'z ichiga oladi

// Yana 5 bayt yozish - aniq muvaffaqiyat
n, err = limited.Write([]byte("world"))
// n == 5, err == nil, buf "helloworld" ni o'z ichiga oladi

// Ko'proq yozishga urinish - muvaffaqiyatsiz
n, err = limited.Write([]byte("!!!"))
// n == 0, err == ErrLimitExceeded, buf hali ham "helloworld"

// Qolgan kvotani tekshirish
remaining := limited.Remaining()
// remaining == 0
\`\`\`

**Qisman yozish misoli:**
\`\`\`go
var buf bytes.Buffer
limited := NewLimitWriter(&buf, 10)

// 7 bayt yozish
limited.Write([]byte("golang!"))
// buf "golang!" ni o'z ichiga oladi, 3 bayt qoldi

// 5 bayt yozishga urinish, lekin faqat 3 tasi sig'adi
n, err := limited.Write([]byte("12345"))
// n == 3, err == ErrLimitExceeded
// buf "golang!123" ni o'z ichiga oladi (qisman yozish)
\`\`\`

**Asosiy tushunchalar:**
- **Resurs cheklash**: Yozish operatsiyalarida kvota qo'llash
- **Qisman yozish**: Limitgacha iloji boricha ko'proq yozish
- **Holat kuzatish**: Kumulyativ bayt hisoblagichini saqlash
- **Potok xavfsizligi**: Umumiy holatni mutex bilan himoyalash
- **Xato signallash**: Limit oshganda aniq xato

**Amalga oshirish strategiyasi:**
1. O'ralgan writer, limit va yozilgan hisoblagichni saqlash
2. Potok xavfsizligi uchun sync.Mutex bilan yozilgan hisoblagichni himoyalash
3. Write() da: limit allaqachon oshganligini tekshirish
4. Mavjud baytlarni hisoblash: available = limit - written
5. Agar available <= 0 bo'lsa, darhol ErrLimitExceeded ni qaytarish
6. Agar len(p) <= available bo'lsa, barcha baytlarni odatdagidek yozish
7. Agar len(p) > available bo'lsa, faqat available bayt yozish (qisman)
8. Written hisoblagichni yangilash va tegishli xatoni qaytarish
9. Remaining() limit - written ni qaytaradi (potok-xavfsiz)

**Cheklovlar:**
- O'ralgan writer ning xatti-harakatini o'zgartirmasligi kerak
- Parallel yozishlar uchun potok-xavfsiz bo'lishi kerak
- Limitga yetganda qisman yozishni qo'llab-quvvatlashi kerak
- Limit oshganda ErrLimitExceeded qaytarilishi kerak
- Remaining() istalgan vaqtda aniq kvotani qaytarishi kerak`,
			hint1: `Write() da avval mutex ni bloklang va available = limit - written ni hisoblang. Agar available <= 0 bo'lsa, darhol ErrLimitExceeded ni qaytaring. Aks holda toWrite = min(len(p), available) ni aniqlang.`,
			hint2: `Asosiy writer ga yozgandan keyin written hisoblagichni yangilang. Agar written >= limit bo'lsa, qaytarishdan oldin err ni ErrLimitExceeded ga o'rnating. Bu chaqiruvchi kod limitga yetilganini qisman yozishlarda ham bilishini ta'minlaydi.`,
			whyItMatters: `Yozish cheklari resurslar tugashini oldini olish, kvotalarni qo'llash va tizimlarni cheksiz yozishlardan himoya qilish uchun muhim.

**Nima uchun bu muhim:**

**1. Resurs himoyasi**
Cheksiz yozishlar quyidagilarga olib kelishi mumkin:
- Disk maydoni tugashi
- Xotira tugashi
- Ma'lumotlar bazasi kvotasi buzilishi
- Tarmoq o'tkazuvchanligini suiiste'mol qilish
- Xarajatlarning oshib ketishi (bulutli saqlash)

LimitWriter himoya mexanizmini ta'minlaydi.

**2. Haqiqiy dunyodagi foydalanish**

**HTTP yuklash hajmini cheklash:**
\`\`\`go
func HandleUpload(w http.ResponseWriter, r *http.Request) {
    maxSize := int64(10 * 1024 * 1024) // 10MB limit

	// Hajm cheklovi bilan fayl yaratish
    file, _ := os.Create("upload.dat")
    defer file.Close()

    limited := NewLimitWriter(file, maxSize)

	// So'rov tanasini limit qo'llagan holda nusxalash
    _, err := io.Copy(limited, r.Body)
    if err == ErrLimitExceeded {
        file.Close()
        os.Remove("upload.dat") // Tozalash
        http.Error(w, "Fayl hajmi 10MB limitdan oshib ketdi", 413)
        return
    }

    if err != nil {
        http.Error(w, "Yuklash muvaffaqiyatsiz", 500)
        return
    }

    fmt.Fprintf(w, "Yuklash muvaffaqiyatli")
}
\`\`\`

**3. Production hodisasi: Disk maydoni hujumi**

Logging xizmati bu kodga ega edi:
\`\`\`go
// YOMON - yozish cheklari yo'q
func LogRequest(r *http.Request) {
    logFile, _ := os.OpenFile("requests.log", os.O_APPEND|os.O_CREATE, 0644)
    defer logFile.Close()

	// Hujumchi katta so'rov tanalarini yuboradi
    io.Copy(logFile, r.Body) // Cheksiz ma'lumot yozadi!
}
\`\`\`

Hujum stsenariyasi:
- Hujumchi har biri 1GB tanali 1000 so'rov yuboradi
- Daqiqalar ichida 1TB diskni to'ldiradi
- Xizmat ishdan chiqadi (disk maydoni yo'q)
- Xuddi shu mashinadagi barcha xizmatlar ishdan chiqadi
- Tiklash soatlar davom etadi

LimitWriter bilan tuzatish:
\`\`\`go
// YAXSHI - har bir so'rovga yozish cheklovini qo'llash
func LogRequest(r *http.Request) {
    logFile, _ := os.OpenFile("requests.log", os.O_APPEND|os.O_CREATE, 0644)
    defer logFile.Close()

	// Har bir so'rov logini 1MB ga cheklash
    limited := NewLimitWriter(logFile, 1024*1024)

    n, err := io.Copy(limited, r.Body)
    if err == ErrLimitExceeded {
        fmt.Fprintf(logFile, "\\n[QIRQILGAN: %d bayt]\\n", n)
    }
}
\`\`\`

Natija:
- So'rov uchun maksimal 1MB log
- Disk maydoni himoyalangan
- Hujum oldini olindi
- Xizmat barqaror qoladi

**4. Foydalanuvchi uchun kvota qo'llash**

\`\`\`go
type UserQuota struct {
    userID string
    limit  int64
    used   int64
    mu     sync.Mutex
}

func (uq *UserQuota) NewWriter(w io.Writer) *LimitWriter {
    uq.mu.Lock()
    remaining := uq.limit - uq.used
    uq.mu.Unlock()

    return NewLimitWriter(w, remaining)
}

func (uq *UserQuota) UpdateUsage(written int64) {
    uq.mu.Lock()
    uq.used += written
    uq.mu.Unlock()
}

func HandleUserUpload(userID string, w http.ResponseWriter, r *http.Request) {
	// Foydalanuvchining kvotasini olish
    quota := GetUserQuota(userID) // Foydalanuvchi uchun 100MB

	// Cheklangan writer yaratish
    file, _ := os.Create(fmt.Sprintf("uploads/%s/file.dat", userID))
    defer file.Close()

    limited := quota.NewWriter(file)

    n, err := io.Copy(limited, r.Body)

	// Foydalanuvchi kvotasini yangilash
    quota.UpdateUsage(n)

    if err == ErrLimitExceeded {
        http.Error(w, fmt.Sprintf("Kvota oshib ketdi. Ishlatilgan: %d MB", quota.used/1024/1024), 507)
        return
    }

    fmt.Fprintf(w, "%d bayt yuklandi. Qolgan kvota: %d MB",
        n, quota.NewWriter(nil).Remaining()/1024/1024)
}
\`\`\`

**5. Javob hajmini cheklash**

**Juda katta API javoblarining oldini olish:**
\`\`\`go
func HandleAPIRequest(w http.ResponseWriter, r *http.Request) {
	// Javobni 5MB ga cheklash
    limited := NewLimitWriter(w, 5*1024*1024)

	// Ma'lumotlar bazasiga so'rov (potensial katta natija)
    rows, _ := db.Query("SELECT * FROM large_table WHERE user_id = ?", userID)
    defer rows.Close()

    encoder := json.NewEncoder(limited)

    for rows.Next() {
        var record Record
        rows.Scan(&record)

        err := encoder.Encode(record)
        if err == ErrLimitExceeded {
	// Javob juda katta, xato qaytarish
            w.WriteHeader(413)
            w.Write([]byte(\`{"error": "Response too large"}\`))
            return
        }
    }
}
\`\`\`

**6. Hajm cheklovi bilan log aylanishi**

\`\`\`go
type SizeRotatingWriter struct {
    filename    string
    maxSize     int64
    currentFile *os.File
    currentSize int64
    mu          sync.Mutex
}

func (srw *SizeRotatingWriter) Write(p []byte) (n int, err error) {
    srw.mu.Lock()
    defer srw.mu.Unlock()

	// Aylanish kerakligini tekshirish
    if srw.currentSize+int64(len(p)) > srw.maxSize {
        srw.rotate()
    }

	// Limit qo'llanishi bilan yozish
    limited := NewLimitWriter(srw.currentFile, srw.maxSize-srw.currentSize)
    n, err = limited.Write(p)
    srw.currentSize += int64(n)

    if err == ErrLimitExceeded {
        srw.rotate()
	// Yangi fayl bilan qayta urinish
        return srw.currentFile.Write(p)
    }

    return n, err
}

func (srw *SizeRotatingWriter) rotate() {
    if srw.currentFile != nil {
        srw.currentFile.Close()
    }

	// Yangi log faylini yaratish
    timestamp := time.Now().Format("20060102-150405")
    newFile := fmt.Sprintf("%s.%s", srw.filename, timestamp)
    srw.currentFile, _ = os.Create(newFile)
    srw.currentSize = 0
}
\`\`\`

**7. Cheklovlar bilan test qilish**

\`\`\`go
func TestLimitWriter_ExactLimit(t *testing.T) {
    var buf bytes.Buffer
    limited := NewLimitWriter(&buf, 10)

	// Limitgacha aniq yozish
    n, err := limited.Write([]byte("0123456789"))
    if n != 10 || err != ErrLimitExceeded {
        t.Errorf("kutilgan n=10, err=ErrLimitExceeded, olindi n=%d, err=%v", n, err)
    }

    if buf.String() != "0123456789" {
        t.Errorf("kutilgan '0123456789', olindi '%s'", buf.String())
    }

	// Qolgan kvota yo'q
    if limited.Remaining() != 0 {
        t.Errorf("kutilgan 0 remaining, olindi %d", limited.Remaining())
    }
}

func TestLimitWriter_PartialWrite(t *testing.T) {
    var buf bytes.Buffer
    limited := NewLimitWriter(&buf, 5)

	// 10 bayt yozishga urinish, faqat 5 tasi sig'adi
    n, err := limited.Write([]byte("0123456789"))
    if n != 5 || err != ErrLimitExceeded {
        t.Errorf("kutilgan n=5, err=ErrLimitExceeded, olindi n=%d, err=%v", n, err)
    }

    if buf.String() != "01234" {
        t.Errorf("kutilgan '01234', olindi '%s'", buf.String())
    }
}

func TestLimitWriter_MultipleWrites(t *testing.T) {
    var buf bytes.Buffer
    limited := NewLimitWriter(&buf, 10)

	// Birinchi yozish
    n, err := limited.Write([]byte("abc"))
    if n != 3 || err != nil {
        t.Errorf("yozish 1: kutilgan n=3, err=nil, olindi n=%d, err=%v", n, err)
    }

	// Ikkinchi yozish
    n, err = limited.Write([]byte("defg"))
    if n != 4 || err != nil {
        t.Errorf("yozish 2: kutilgan n=4, err=nil, olindi n=%d, err=%v", n, err)
    }

	// Uchinchi yozish (qisman)
    n, err = limited.Write([]byte("hijkl"))
    if n != 3 || err != ErrLimitExceeded {
        t.Errorf("yozish 3: kutilgan n=3, err=ErrLimitExceeded, olindi n=%d, err=%v", n, err)
    }

    if buf.String() != "abcdefghij" {
        t.Errorf("kutilgan 'abcdefghij', olindi '%s'", buf.String())
    }
}
\`\`\`

**8. Bulutli saqlash xarajatlarini nazorat qilish**

\`\`\`go
func UploadToS3WithQuota(bucket, key string, data io.Reader, quota int64) error {
	// Limit bilan vaqtinchalik fayl yaratish
    tmpFile, _ := os.CreateTemp("", "upload-*")
    defer os.Remove(tmpFile.Name())
    defer tmpFile.Close()

	// Vaqtinchalik faylga limit bilan yozish
    limited := NewLimitWriter(tmpFile, quota)
    written, err := io.Copy(limited, data)

    if err == ErrLimitExceeded {
        return fmt.Errorf("yuklash %d bayt kvotasidan oshib ketdi", quota)
    }

    if err != nil {
        return fmt.Errorf("yuklash muvaffaqiyatsiz: %w", err)
    }

	// S3 ga yuklash uchun boshiga qaytish
    tmpFile.Seek(0, 0)

	// S3 ga yuklash
    uploader := s3manager.NewUploader(sess)
    _, err = uploader.Upload(&s3manager.UploadInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
        Body:   tmpFile,
    })

    log.Printf("S3 ga %d bayt yuklandi (kvota: %d)", written, quota)
    return err
}
\`\`\`

**9. Token Bucket bilan tezlikni cheklash**

\`\`\`go
type TokenBucket struct {
    tokens    int64
    refillRate int64 // bayt/soniya
    lastRefill time.Time
    mu sync.Mutex
}

func (tb *TokenBucket) NewWriter(w io.Writer) io.Writer {
    tb.mu.Lock()
    tb.refill()
    available := tb.tokens
    tb.mu.Unlock()

    return NewLimitWriter(w, available)
}

func (tb *TokenBucket) refill() {
    now := time.Now()
    elapsed := now.Sub(tb.lastRefill).Seconds()
    newTokens := int64(elapsed * float64(tb.refillRate))

    tb.tokens += newTokens
    tb.lastRefill = now
}

// Foydalanish: yozish tezligini 1MB/s ga cheklash
bucket := &TokenBucket{
    tokens: 1024 * 1024,
    refillRate: 1024 * 1024,
    lastRefill: time.Now(),
}

limitedWriter := bucket.NewWriter(file)
io.Copy(limitedWriter, data) // Maksimal 1MB/s
\`\`\`

**10. Ko'p limitlarni qo'llash**

\`\`\`go
type MultiLimitWriter struct {
    writers []*LimitWriter
}

func (mlw *MultiLimitWriter) Write(p []byte) (n int, err error) {
	// Barcha cheklangan writer larga yozish
	// Biron bir limit oshsa muvaffaqiyatsiz
    for i, lw := range mlw.writers {
        n, err = lw.Write(p)
        if err != nil {
            return n, fmt.Errorf("limit %d oshib ketdi: %w", i, err)
        }
    }
    return n, nil
}

// Bir vaqtning o'zida bir nechta limitlarni qo'llash
func NewMultiLimitWriter(w io.Writer, perWriteLimit, totalLimit, dailyLimit int64) io.Writer {
    return &MultiLimitWriter{
        writers: []*LimitWriter{
            NewLimitWriter(w, perWriteLimit),	// Operatsiya uchun limit
            NewLimitWriter(w, totalLimit),	// Jami sessiya limiti
            NewLimitWriter(w, dailyLimit),	// Kunlik kvota limiti
        },
    }
}
\`\`\`

**Asosiy xulosalar:**
- LimitWriter resurslar tugashi hujumlarining oldini oladi
- Kvotalarni qo'llash va xarajatlarni nazorat qilish uchun zarur
- Yumshoq degradatsiya uchun qisman yozishni qo'llab-quvvatlaydi
- Parallel operatsiyalar uchun dizayn bo'yicha potok-xavfsiz
- Amalga oshirish oddiy, ishlab chiqarish tizimlari uchun muhim
- Boshqa io.Writer o'ramlar bilan kompozitsiyalanadi
- Nozik resurs nazoratini ta'minlaydi
- Disk maydoni ham xotirani ham himoya qiladi`,
			solutionCode: `package interfaces

import (
	"errors"
	"io"
	"sync"
)

var ErrLimitExceeded = errors.New("write limit exceeded")

type LimitWriter struct {
	w       io.Writer
	limit   int64
	written int64
	mu      sync.Mutex
}

// NewLimitWriter cheklangan yangi writer yaratadi
func NewLimitWriter(w io.Writer, limit int64) *LimitWriter {
	return &LimitWriter{
		w:       w,
		limit:   limit,
		written: 0,
	}
}

// Write limit qo'llanishi bilan io.Writer interfeysini amalga oshiradi
func (lw *LimitWriter) Write(p []byte) (n int, err error) {
	lw.mu.Lock()
	defer lw.mu.Unlock()

	// Mavjud joyni hisoblash
	available := lw.limit - lw.written

	// Agar joy bo'lmasa, darhol rad etish
	if available <= 0 {
		return 0, ErrLimitExceeded
	}

	// Qancha yozishni aniqlash
	toWrite := int64(len(p))
	if toWrite > available {
		// Qisman yozish - faqat sig'adiganini yozish
		toWrite = available
	}

	// Asosiy writer ga yozish
	n, err = lw.w.Write(p[:toWrite])
	lw.written += int64(n)

	// Agar limitga yetilgan bo'lsa, ErrLimitExceeded ni qaytarish
	if lw.written >= lw.limit && err == nil {
		err = ErrLimitExceeded
	}

	return n, err
}

// Remaining hali yozilishi mumkin bo'lgan baytlar sonini qaytaradi
func (lw *LimitWriter) Remaining() int64 {
	lw.mu.Lock()
	defer lw.mu.Unlock()

	remaining := lw.limit - lw.written
	if remaining < 0 {
		return 0
	}
	return remaining
}`
		}
	}
};

export default task;
