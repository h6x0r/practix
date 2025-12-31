import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-interfaces-copyn',
	title: 'Implement CopyN with io.Reader/Writer Interfaces',
	difficulty: 'easy',
	tags: ['go', 'interfaces', 'io', 'reader', 'writer'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement **CopyN(dst, src, n)** with the semantics of \`io.CopyN\`, but without using it directly.

**Requirements:**
1. Copy exactly \`n\` bytes from \`src\` (io.Reader) to \`dst\` (io.Writer)
2. Return error if \`n < 0\` (invalid parameter)
3. Handle \`n == 0\` gracefully (return 0, nil)
4. Read data in chunks using an intermediate buffer
5. Track the number of bytes written accurately
6. Handle \`io.EOF\` correctly when source ends
7. Return \`io.ErrShortWrite\` if writer writes fewer bytes than reader read

**Function Signature:**
\`\`\`go
func CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error)
\`\`\`

**Example Usage:**
\`\`\`go
src := strings.NewReader("hello world")
var dst bytes.Buffer

// Copy exactly 5 bytes
written, err := CopyN(&dst, src, 5)
// written == 5, dst.String() == "hello", err == nil

// Try to copy more than available
src2 := strings.NewReader("abc")
var dst2 bytes.Buffer
written, err = CopyN(&dst2, src2, 5)
// written == 3, err == io.EOF (source exhausted)
\`\`\`

**Key Concepts:**
- **io.Reader interface**: Read(p []byte) (n int, err error)
- **io.Writer interface**: Write(p []byte) (n int, err error)
- **Buffered reading**: Use intermediate buffer (e.g., 32KB)
- **Partial reads**: Reader may return fewer bytes than requested
- **Short writes**: Writer writing less than provided is an error

**Implementation Strategy:**
1. Validate \`n\` parameter (must be >= 0)
2. Create a buffer for chunked reading (32KB is efficient)
3. Loop until exactly \`n\` bytes are copied
4. Calculate bytes remaining to copy
5. Read from source (limited to buffer size or remaining bytes)
6. Write all read bytes to destination
7. Check for short writes (nw != nr)
8. Handle \`io.EOF\` correctly

**Constraints:**
- Do not use \`io.CopyN\` or \`io.Copy\` from standard library
- Buffer size should be 32KB for efficiency
- Return error immediately if \`n < 0\`
- Track bytes written accurately
- Handle all edge cases: EOF, short write, destination errors`,
	initialCode: `package interfaces

import (
	"fmt"
	"io"
)

// TODO: Implement CopyN
// Copy exactly n bytes from src to dst
// Hint: Create 32KB buffer, loop until n bytes copied
// Handle: n < 0 (error), n == 0 (return), io.EOF, short writes
func CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error) {
	// TODO: Implement
}`,
	solutionCode: `package interfaces

import (
	"fmt"
	"io"
)

func CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error) {
	// Validate n parameter
	if n < 0 {
		return 0, fmt.Errorf("invalid length param")
	}
	if n == 0 {
		return 0, nil
	}

	// Create buffer for efficient chunked reading
	buf := make([]byte, 32*1024)

	// Loop until we've copied exactly n bytes
	for written < n {
		// Calculate how many bytes we still need to copy
		toRead := n - written
		if toRead > int64(len(buf)) {
			toRead = int64(len(buf))
		}

		// Read from source (up to toRead bytes)
		nr, er := src.Read(buf[:toRead])

		// Write any bytes we successfully read
		if nr > 0 {
			nw, ew := dst.Write(buf[:nr])
			if nw > 0 {
				written += int64(nw)
			}

			// Handle write errors
			if ew != nil {
				return written, ew
			}

			// Check for short write (writer wrote less than we gave it)
			if nw != nr {
				return written, io.ErrShortWrite
			}
		}

		// Handle read errors
		if er != nil {
			// EOF after writing exactly n bytes is success
			if er == io.EOF && written == n {
				break
			}
			// EOF before n bytes means source exhausted
			if er == io.EOF {
				return written, io.EOF
			}
			// Other errors
			return written, er
		}
	}

	return written, nil
}`,
	testCode: `package interfaces

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"
)

func Test1CopyNValidInput(t *testing.T) {
	src := strings.NewReader("hello world")
	var dst bytes.Buffer
	written, err := CopyN(&dst, src, 5)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if written != 5 {
		t.Errorf("expected 5 bytes written, got %d", written)
	}
	if dst.String() != "hello" {
		t.Errorf("expected 'hello', got '%s'", dst.String())
	}
}

func Test2CopyNNegativeN(t *testing.T) {
	src := strings.NewReader("hello")
	var dst bytes.Buffer
	written, err := CopyN(&dst, src, -1)
	if err == nil {
		t.Error("expected error for negative n")
	}
	if written != 0 {
		t.Errorf("expected 0 bytes written, got %d", written)
	}
}

func Test3CopyNZeroN(t *testing.T) {
	src := strings.NewReader("hello")
	var dst bytes.Buffer
	written, err := CopyN(&dst, src, 0)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if written != 0 {
		t.Errorf("expected 0 bytes written, got %d", written)
	}
	if dst.Len() != 0 {
		t.Error("expected empty destination buffer")
	}
}

func Test4CopyNSourceExhausted(t *testing.T) {
	src := strings.NewReader("abc")
	var dst bytes.Buffer
	written, err := CopyN(&dst, src, 10)
	if err != io.EOF {
		t.Errorf("expected io.EOF, got %v", err)
	}
	if written != 3 {
		t.Errorf("expected 3 bytes written, got %d", written)
	}
	if dst.String() != "abc" {
		t.Errorf("expected 'abc', got '%s'", dst.String())
	}
}

func Test5CopyNExactMatch(t *testing.T) {
	src := strings.NewReader("exact")
	var dst bytes.Buffer
	written, err := CopyN(&dst, src, 5)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if written != 5 {
		t.Errorf("expected 5 bytes written, got %d", written)
	}
	if dst.String() != "exact" {
		t.Errorf("expected 'exact', got '%s'", dst.String())
	}
}

func Test6CopyNLargeData(t *testing.T) {
	data := strings.Repeat("a", 100000)
	src := strings.NewReader(data)
	var dst bytes.Buffer
	written, err := CopyN(&dst, src, 100000)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if written != 100000 {
		t.Errorf("expected 100000 bytes written, got %d", written)
	}
}

type errorWriter struct{}

func (errorWriter) Write(p []byte) (int, error) {
	return 0, errors.New("write error")
}

func Test7CopyNWriteError(t *testing.T) {
	src := strings.NewReader("hello")
	var dst errorWriter
	written, err := CopyN(&dst, src, 5)
	if err == nil {
		t.Error("expected write error")
	}
	if written != 0 {
		t.Errorf("expected 0 bytes written, got %d", written)
	}
}

type errorReader struct{}

func (errorReader) Read(p []byte) (int, error) {
	return 0, errors.New("read error")
}

func Test8CopyNReadError(t *testing.T) {
	var src errorReader
	var dst bytes.Buffer
	written, err := CopyN(&dst, &src, 5)
	if err == nil {
		t.Error("expected read error")
	}
	if written != 0 {
		t.Errorf("expected 0 bytes written, got %d", written)
	}
}

type shortWriter struct {
	buf bytes.Buffer
}

func (sw *shortWriter) Write(p []byte) (int, error) {
	if len(p) > 0 {
		return sw.buf.Write(p[:len(p)/2])
	}
	return 0, nil
}

func Test9CopyNShortWrite(t *testing.T) {
	src := strings.NewReader("hello world")
	dst := &shortWriter{}
	written, err := CopyN(dst, src, 10)
	if err != io.ErrShortWrite {
		t.Errorf("expected io.ErrShortWrite, got %v", err)
	}
	if written == 0 {
		t.Error("expected some bytes written before short write error")
	}
}

func Test10CopyNEmptySource(t *testing.T) {
	src := strings.NewReader("")
	var dst bytes.Buffer
	written, err := CopyN(&dst, src, 5)
	if err != io.EOF {
		t.Errorf("expected io.EOF, got %v", err)
	}
	if written != 0 {
		t.Errorf("expected 0 bytes written, got %d", written)
	}
}`,
	hint1: `Create a 32KB buffer and loop while written < n. Calculate toRead = min(n - written, buffer size). Read into buf[:toRead], then write all read bytes to dst.`,
	hint2: `Short write detection: After dst.Write(buf[:nr]), check if nw != nr. If writer wrote fewer bytes than you provided, return io.ErrShortWrite immediately.`,
	whyItMatters: `Understanding io.Reader and io.Writer interfaces is fundamental to Go programming and enables powerful composition patterns.

**Why This Matters:**

**1. The Power of Interfaces**
Go's \`io.Reader\` and \`io.Writer\` are the most widely used interfaces in the standard library:

\`\`\`go
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}
\`\`\`

These simple interfaces enable incredible flexibility. Your \`CopyN\` works with:
- Files: \`os.File\` implements both Reader and Writer
- Network: \`net.Conn\` implements both
- Buffers: \`bytes.Buffer\`, \`strings.Reader\`
- Compression: \`gzip.Writer\`, \`gzip.Reader\`
- Encryption: \`crypto/cipher\` streams
- HTTP: \`http.Request.Body\`, \`http.ResponseWriter\`

**2. Real-World Usage**

**HTTP File Upload with Size Limit:**
\`\`\`go
func HandleUpload(w http.ResponseWriter, r *http.Request) {
	// Limit upload to 10MB
    file, _ := os.Create("upload.dat")
    defer file.Close()

	// Copy exactly 10MB from HTTP body to file
    written, err := io.CopyN(file, r.Body, 10*1024*1024)
    if err != nil && err != io.EOF {
        http.Error(w, "Upload failed", 500)
        return
    }

    fmt.Fprintf(w, "Received %d bytes", written)
}
\`\`\`

Without CopyN, you'd need manual buffer management and could accidentally accept files larger than intended.

**3. Production Incident: Unbounded File Upload**

A file-sharing service had this code:
\`\`\`go
// BAD - no size limit
io.Copy(file, r.Body) // Accepts unlimited size!
\`\`\`

Attack scenario:
- Attacker uploads 100GB file
- Fills entire disk
- Service crashes
- All users affected

Fix with CopyN:
\`\`\`go
// GOOD - enforce 50MB limit
maxSize := int64(50 * 1024 * 1024)
written, err := io.CopyN(file, r.Body, maxSize)
if err != nil && err != io.EOF {
	// Handle error
}
// Check if client tried to send more
if _, err := r.Body.Read(make([]byte, 1)); err != io.EOF {
    return errors.New("file exceeds size limit")
}
\`\`\`

**4. Interface Composition Patterns**

**Logging Reader:**
\`\`\`go
type LoggingReader struct {
    r io.Reader
    bytesRead int64
}

func (lr *LoggingReader) Read(p []byte) (n int, err error) {
    n, err = lr.r.Read(p)
    lr.bytesRead += int64(n)
    log.Printf("Read %d bytes (total: %d)", n, lr.bytesRead)
    return n, err
}
\`\`\`

Now wrap any reader:
\`\`\`go
file, _ := os.Open("data.txt")
loggingReader := &LoggingReader{r: file}

// CopyN works transparently with logging
io.CopyN(os.Stdout, loggingReader, 1024)
// Output:
// Read 1024 bytes (total: 1024)
\`\`\`

**5. Why Handle Short Writes?**

Most writers always write all bytes or return error. But some don't:
\`\`\`go
type RateLimitedWriter struct {
    w io.Writer
    remaining int
}

func (rl *RateLimitedWriter) Write(p []byte) (int, error) {
    if len(p) > rl.remaining {
	// Can only write remaining bytes
        n, err := rl.w.Write(p[:rl.remaining])
        rl.remaining -= n
        return n, err // Short write!
    }
    n, err := rl.w.Write(p)
    rl.remaining -= n
    return n, err
}
\`\`\`

If you don't check for short writes, you silently lose data!

**6. EOF Handling**

\`io.EOF\` is not always an error:
\`\`\`go
// Copy 5 bytes from 10-byte source
src := strings.NewReader("0123456789")
written, err := CopyN(dst, src, 5)
// written == 5, err == nil (success!)

// Copy 10 bytes from 5-byte source
src := strings.NewReader("01234")
written, err := CopyN(dst, src, 10)
// written == 5, err == io.EOF (source exhausted)
\`\`\`

**7. Buffer Size Matters**

\`\`\`go
// TOO SMALL - system call overhead
buf := make([]byte, 1) // 1MB copy = 1 million syscalls

// OPTIMAL - balance memory and syscalls
buf := make([]byte, 32*1024) // 1MB copy = ~32 syscalls

// TOO LARGE - wastes memory
buf := make([]byte, 10*1024*1024) // 10MB buffer rarely needed
\`\`\`

32KB is Go's sweet spot: efficient syscalls without excessive memory.

**8. Production Patterns**

**Download Progress Tracking:**
\`\`\`go
type ProgressReader struct {
    r io.Reader
    total int64
    callback func(written int64)
}

func (pr *ProgressReader) Read(p []byte) (int, error) {
    n, err := pr.r.Read(p)
    pr.total += int64(n)
    pr.callback(pr.total)
    return n, err
}

// Track download progress
resp, _ := http.Get("https://example.com/large.zip")
progress := &ProgressReader{
    r: resp.Body,
    callback: func(written int64) {
        fmt.Printf("\rDownloaded: %d bytes", written)
    },
}

io.CopyN(file, progress, expectedSize)
\`\`\`

**9. Testing with Interfaces**

Interfaces make testing easy:
\`\`\`go
type ErrorReader struct{}
func (ErrorReader) Read(p []byte) (int, error) {
    return 0, errors.New("network timeout")
}

func TestCopyN_ReadError(t *testing.T) {
    var dst bytes.Buffer
    _, err := CopyN(&dst, ErrorReader{}, 100)
    if err == nil {
        t.Fatal("expected error")
    }
}
\`\`\`

No need for real files or networks in tests!

**10. Standard Library Examples**

These stdlib functions use the same pattern:
- \`io.Copy(dst, src)\` - copy until EOF
- \`io.CopyN(dst, src, n)\` - copy exactly n bytes
- \`io.CopyBuffer(dst, src, buf)\` - copy with custom buffer
- \`io.ReadFull(r, buf)\` - read exactly len(buf) bytes
- \`io.ReadAtLeast(r, buf, min)\` - read at least min bytes

Understanding \`CopyN\` teaches you how all of these work internally.

**Key Takeaways:**
- Interfaces enable powerful composition
- Small interfaces are more flexible
- Always validate \`n\` parameter
- Check for short writes
- Handle EOF correctly
- Use efficient buffer sizes (32KB)
- Interfaces make testing easy`,
	order: 0,
	translations: {
		ru: {
			title: 'Реализация CopyN с интерфейсами io.Reader/Writer',
			description: `Реализуйте **CopyN(dst, src, n)** с семантикой \`io.CopyN\`, но без прямого использования этой функции.

**Требования:**
1. Скопировать ровно \`n\` байт из \`src\` (io.Reader) в \`dst\` (io.Writer)
2. Вернуть ошибку если \`n < 0\` (невалидный параметр)
3. Обработать \`n == 0\` корректно (вернуть 0, nil)
4. Читать данные порциями используя промежуточный буфер
5. Точно отслеживать количество записанных байт
6. Корректно обработать \`io.EOF\` когда источник закончился
7. Вернуть \`io.ErrShortWrite\` если writer записал меньше байт чем reader прочитал

**Сигнатура функции:**
\`\`\`go
func CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error)
\`\`\`

**Пример использования:**
\`\`\`go
src := strings.NewReader("hello world")
var dst bytes.Buffer

// Скопировать ровно 5 байт
written, err := CopyN(&dst, src, 5)
// written == 5, dst.String() == "hello", err == nil

// Попытка скопировать больше чем доступно
src2 := strings.NewReader("abc")
var dst2 bytes.Buffer
written, err = CopyN(&dst2, src2, 5)
// written == 3, err == io.EOF (источник исчерпан)
\`\`\`

**Ключевые концепции:**
- **Интерфейс io.Reader**: Read(p []byte) (n int, err error)
- **Интерфейс io.Writer**: Write(p []byte) (n int, err error)
- **Буферизованное чтение**: Используйте промежуточный буфер (например, 32KB)
- **Частичное чтение**: Reader может вернуть меньше байт чем запрошено
- **Короткая запись**: Writer, записывающий меньше чем предоставлено — это ошибка

**Стратегия реализации:**
1. Валидировать параметр \`n\` (должен быть >= 0)
2. Создать буфер для порционного чтения (32KB — эффективный размер)
3. Выполнять цикл пока не скопировано ровно \`n\` байт
4. Вычислить оставшееся количество байт для копирования
5. Прочитать из источника (ограничено размером буфера или оставшимися байтами)
6. Записать все прочитанные байты в назначение
7. Проверить короткую запись (nw != nr)
8. Корректно обработать \`io.EOF\`

**Ограничения:**
- Не использовать \`io.CopyN\` или \`io.Copy\` из стандартной библиотеки
- Размер буфера должен быть 32KB для эффективности
- Вернуть ошибку немедленно если \`n < 0\`
- Точно отслеживать записанные байты
- Обработать все граничные случаи: EOF, короткая запись, ошибки назначения`,
			hint1: `Создайте буфер 32KB и организуйте цикл пока written < n. Вычислите toRead = min(n - written, размер буфера). Прочитайте в buf[:toRead], затем запишите все прочитанные байты в dst.`,
			hint2: `Обнаружение короткой записи: После dst.Write(buf[:nr]) проверьте if nw != nr. Если writer записал меньше байт чем вы предоставили, немедленно верните io.ErrShortWrite.`,
			whyItMatters: `Понимание интерфейсов io.Reader и io.Writer фундаментально для программирования на Go и позволяет использовать мощные паттерны композиции.

**Почему это важно:**

**1. Сила интерфейсов**
\`io.Reader\` и \`io.Writer\` — наиболее используемые интерфейсы в стандартной библиотеке Go:

\`\`\`go
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}
\`\`\`

Эти простые интерфейсы обеспечивают невероятную гибкость. Ваш \`CopyN\` работает с:
- Файлами: \`os.File\` реализует оба интерфейса Reader и Writer
- Сетью: \`net.Conn\` реализует оба интерфейса
- Буферами: \`bytes.Buffer\`, \`strings.Reader\`
- Сжатием: \`gzip.Writer\`, \`gzip.Reader\`
- Шифрованием: потоки \`crypto/cipher\`
- HTTP: \`http.Request.Body\`, \`http.ResponseWriter\`

**2. Практическое использование**

**HTTP загрузка файла с ограничением размера:**
\`\`\`go
func HandleUpload(w http.ResponseWriter, r *http.Request) {
	// Ограничение загрузки до 10MB
    file, _ := os.Create("upload.dat")
    defer file.Close()

	// Скопировать ровно 10MB из HTTP body в файл
    written, err := io.CopyN(file, r.Body, 10*1024*1024)
    if err != nil && err != io.EOF {
        http.Error(w, "Ошибка загрузки", 500)
        return
    }

    fmt.Fprintf(w, "Получено %d байт", written)
}
\`\`\`

Без CopyN потребовался бы ручной менеджмент буфера, и можно было бы случайно принять файлы больше допустимого размера.

**3. Инцидент в продакшене: Неограниченная загрузка файла**

Сервис обмена файлами имел следующий код:
\`\`\`go
// ПЛОХО - нет ограничения размера
io.Copy(file, r.Body) // Принимает неограниченный размер!
\`\`\`

Сценарий атаки:
- Атакующий загружает файл 100GB
- Заполняется весь диск
- Сервис падает
- Все пользователи затронуты

Исправление с CopyN:
\`\`\`go
// ХОРОШО - ограничение 50MB
maxSize := int64(50 * 1024 * 1024)
written, err := io.CopyN(file, r.Body, maxSize)
if err != nil && err != io.EOF {
	// Обработка ошибки
}
// Проверка, не пытался ли клиент отправить больше
if _, err := r.Body.Read(make([]byte, 1)); err != io.EOF {
    return errors.New("файл превышает лимит размера")
}
\`\`\`

**4. Паттерны композиции интерфейсов**

**Логирующий Reader:**
\`\`\`go
type LoggingReader struct {
    r io.Reader
    bytesRead int64
}

func (lr *LoggingReader) Read(p []byte) (n int, err error) {
    n, err = lr.r.Read(p)
    lr.bytesRead += int64(n)
    log.Printf("Прочитано %d байт (всего: %d)", n, lr.bytesRead)
    return n, err
}
\`\`\`

Теперь оберните любой reader, и CopyN будет прозрачно работать с логированием:
\`\`\`go
file, _ := os.Open("data.txt")
loggingReader := &LoggingReader{r: file}

// CopyN прозрачно работает с логированием
io.CopyN(os.Stdout, loggingReader, 1024)
// Вывод:
// Прочитано 1024 байт (всего: 1024)
\`\`\`

**5. Почему важно обрабатывать короткие записи?**

Большинство writers всегда записывают все байты или возвращают ошибку. Но некоторые нет:
\`\`\`go
type RateLimitedWriter struct {
    w io.Writer
    remaining int
}

func (rl *RateLimitedWriter) Write(p []byte) (int, error) {
    if len(p) > rl.remaining {
	// Можем записать только оставшиеся байты
        n, err := rl.w.Write(p[:rl.remaining])
        rl.remaining -= n
        return n, err // Короткая запись!
    }
    n, err := rl.w.Write(p)
    rl.remaining -= n
    return n, err
}
\`\`\`

Если не проверять короткие записи, данные молча теряются!

**6. Обработка EOF**

\`io.EOF\` — не всегда ошибка:
\`\`\`go
// Копирование 5 байт из 10-байтового источника
src := strings.NewReader("0123456789")
written, err := CopyN(dst, src, 5)
// written == 5, err == nil (успех!)

// Копирование 10 байт из 5-байтового источника
src := strings.NewReader("01234")
written, err := CopyN(dst, src, 10)
// written == 5, err == io.EOF (источник исчерпан)
\`\`\`

**7. Размер буфера важен**

\`\`\`go
// СЛИШКОМ МАЛЕНЬКИЙ - накладные расходы на системные вызовы
buf := make([]byte, 1) // копирование 1MB = 1 миллион syscalls

// ОПТИМАЛЬНО - баланс памяти и syscalls
buf := make([]byte, 32*1024) // копирование 1MB = ~32 syscalls

// СЛИШКОМ БОЛЬШОЙ - расход памяти
buf := make([]byte, 10*1024*1024) // буфер 10MB редко нужен
\`\`\`

32KB — оптимальная точка Go: эффективные системные вызовы без излишнего потребления памяти.

**8. Паттерны для продакшена**

**Отслеживание прогресса загрузки:**
\`\`\`go
type ProgressReader struct {
    r io.Reader
    total int64
    callback func(written int64)
}

func (pr *ProgressReader) Read(p []byte) (int, error) {
    n, err := pr.r.Read(p)
    pr.total += int64(n)
    pr.callback(pr.total)
    return n, err
}

// Отслеживание прогресса загрузки
resp, _ := http.Get("https://example.com/large.zip")
progress := &ProgressReader{
    r: resp.Body,
    callback: func(written int64) {
        fmt.Printf("\\rЗагружено: %d байт", written)
    },
}

io.CopyN(file, progress, expectedSize)
\`\`\`

**9. Тестирование с интерфейсами**

Интерфейсы делают тестирование простым:
\`\`\`go
type ErrorReader struct{}
func (ErrorReader) Read(p []byte) (int, error) {
    return 0, errors.New("сетевой таймаут")
}

func TestCopyN_ReadError(t *testing.T) {
    var dst bytes.Buffer
    _, err := CopyN(&dst, ErrorReader{}, 100)
    if err == nil {
        t.Fatal("ожидалась ошибка")
    }
}
\`\`\`

Не нужны реальные файлы или сеть в тестах!

**10. Примеры из стандартной библиотеки**

Эти функции стандартной библиотеки используют тот же паттерн:
- \`io.Copy(dst, src)\` — копировать до EOF
- \`io.CopyN(dst, src, n)\` — копировать ровно n байт
- \`io.CopyBuffer(dst, src, buf)\` — копировать с кастомным буфером
- \`io.ReadFull(r, buf)\` — прочитать ровно len(buf) байт
- \`io.ReadAtLeast(r, buf, min)\` — прочитать минимум min байт

Понимание \`CopyN\` учит вас, как все эти функции работают внутри.

**Ключевые выводы:**
- Интерфейсы обеспечивают мощную композицию
- Маленькие интерфейсы более гибкие
- Всегда валидируйте параметр \`n\`
- Проверяйте короткие записи
- Корректно обрабатывайте EOF
- Используйте эффективные размеры буфера (32KB)
- Интерфейсы делают тестирование простым`,
			solutionCode: `package interfaces

import (
	"fmt"
	"io"
)

func CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error) {
	// Валидируем параметр n
	if n < 0 {
		return 0, fmt.Errorf("invalid length param")
	}
	if n == 0 {
		return 0, nil
	}

	// Создаём буфер для эффективного порционного чтения
	buf := make([]byte, 32*1024)

	// Выполняем цикл пока не скопировали ровно n байт
	for written < n {
		// Вычисляем сколько байт ещё нужно скопировать
		toRead := n - written
		if toRead > int64(len(buf)) {
			toRead = int64(len(buf))
		}

		// Читаем из источника (до toRead байт)
		nr, er := src.Read(buf[:toRead])

		// Записываем любые байты которые успешно прочитали
		if nr > 0 {
			nw, ew := dst.Write(buf[:nr])
			if nw > 0 {
				written += int64(nw)
			}

			// Обрабатываем ошибки записи
			if ew != nil {
				return written, ew
			}

			// Проверяем короткую запись (writer записал меньше чем мы дали)
			if nw != nr {
				return written, io.ErrShortWrite
			}
		}

		// Обрабатываем ошибки чтения
		if er != nil {
			// EOF после записи ровно n байт — успех
			if er == io.EOF && written == n {
				break
			}
			// EOF до n байт означает источник исчерпан
			if er == io.EOF {
				return written, io.EOF
			}
			// Другие ошибки
			return written, er
		}
	}

	return written, nil
}`
		},
		uz: {
			title: `io.Reader/Writer interfeyslari bilan CopyN ni amalga oshirish`,
			description: `**CopyN(dst, src, n)** funksiyasini \`io.CopyN\` semantikasi bilan amalga oshiring, lekin uni to'g'ridan-to'g'ri ishlatmasdan.

**Talablar:**
1. \`src\` (io.Reader) dan \`dst\` (io.Writer) ga aynan \`n\` bayt nusxalash
2. Agar \`n < 0\` bo'lsa, xato qaytarish (noto'g'ri parametr)
3. \`n == 0\` holatini to'g'ri qayta ishlash (0, nil qaytarish)
4. Ma'lumotlarni oraliq bufer yordamida bo'laklarda o'qish
5. Yozilgan baytlar sonini aniq kuzatish
6. Manba tugaganda \`io.EOF\` ni to'g'ri qayta ishlash
7. Agar writer reader o'qiganidan kam bayt yozsa, \`io.ErrShortWrite\` qaytarish

**Funksiya imzosi:**
\`\`\`go
func CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error)
\`\`\`

**Foydalanish misoli:**
\`\`\`go
src := strings.NewReader("hello world")
var dst bytes.Buffer

// Aynan 5 bayt nusxalash
written, err := CopyN(&dst, src, 5)
// written == 5, dst.String() == "hello", err == nil

// Mavjud bo'lganidan ko'proq nusxalashga urinish
src2 := strings.NewReader("abc")
var dst2 bytes.Buffer
written, err = CopyN(&dst2, src2, 5)
// written == 3, err == io.EOF (manba tugadi)
\`\`\`

**Asosiy tushunchalar:**
- **io.Reader interfeysi**: Read(p []byte) (n int, err error)
- **io.Writer interfeysi**: Write(p []byte) (n int, err error)
- **Buferli o'qish**: Oraliq buferdan foydalaning (masalan, 32KB)
- **Qisman o'qish**: Reader so'ralganidan kam bayt qaytarishi mumkin
- **Qisqa yozish**: Berilganidan kamroq yozuvchi Writer — bu xato

**Amalga oshirish strategiyasi:**
1. \`n\` parametrini tekshirish (>= 0 bo'lishi kerak)
2. Bo'lakli o'qish uchun bufer yaratish (32KB samarali o'lcham)
3. Aynan \`n\` bayt nusxalanguncha sikl bajarish
4. Nusxalash uchun qolgan baytlar sonini hisoblash
5. Manbadan o'qish (bufer o'lchami yoki qolgan baytlar bilan cheklangan)
6. Barcha o'qilgan baytlarni maqsadga yozish
7. Qisqa yozishni tekshirish (nw != nr)
8. \`io.EOF\` ni to'g'ri qayta ishlash

**Cheklovlar:**
- Standart kutubxonadan \`io.CopyN\` yoki \`io.Copy\` dan foydalanmang
- Samaradorlik uchun bufer o'lchami 32KB bo'lishi kerak
- Agar \`n < 0\` bo'lsa, darhol xato qaytaring
- Yozilgan baytlarni aniq kuzating
- Barcha chegara holatlarini qayta ishlang: EOF, qisqa yozish, maqsad xatolari`,
			hint1: `32KB bufer yarating va written < n bo'lguncha sikl tashkil qiling. toRead = min(n - written, bufer o'lchami) ni hisoblang. buf[:toRead] ga o'qing, so'ngra barcha o'qilgan baytlarni dst ga yozing.`,
			hint2: `Qisqa yozishni aniqlash: dst.Write(buf[:nr]) dan keyin if nw != nr ni tekshiring. Agar writer siz berganingizdan kam bayt yozgan bo'lsa, darhol io.ErrShortWrite qaytaring.`,
			whyItMatters: `io.Reader va io.Writer interfeyslarini tushunish Go dasturlash uchun asosiy bo'lib, kuchli kompozitsiya patternlaridan foydalanish imkonini beradi.

**Nima uchun bu muhim:**

**1. Interfeyslarning kuchi**
\`io.Reader\` va \`io.Writer\` — Go standart kutubxonasidagi eng ko'p ishlatiladigan interfeyslar:

\`\`\`go
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}
\`\`\`

Ushbu oddiy interfeyslar ajoyib moslashuvchanlikni ta'minlaydi. Sizning \`CopyN\` quyidagilar bilan ishlaydi:
- Fayllar: \`os.File\` ikkala Reader va Writer interfeyslarini amalga oshiradi
- Tarmoq: \`net.Conn\` ikkala interfeysni amalga oshiradi
- Buferlar: \`bytes.Buffer\`, \`strings.Reader\`
- Siqish: \`gzip.Writer\`, \`gzip.Reader\`
- Shifrlash: \`crypto/cipher\` oqimlari
- HTTP: \`http.Request.Body\`, \`http.ResponseWriter\`

**2. Amaliy foydalanish**

**Hajm cheklovi bilan HTTP fayl yuklash:**
\`\`\`go
func HandleUpload(w http.ResponseWriter, r *http.Request) {
	// Yuklashni 10MB bilan cheklash
    file, _ := os.Create("upload.dat")
    defer file.Close()

	// HTTP body dan faylga aynan 10MB nusxalash
    written, err := io.CopyN(file, r.Body, 10*1024*1024)
    if err != nil && err != io.EOF {
        http.Error(w, "Yuklash muvaffaqiyatsiz", 500)
        return
    }

    fmt.Fprintf(w, "%d bayt qabul qilindi", written)
}
\`\`\`

CopyN siz qo'lda bufer boshqaruvi kerak bo'lardi va tasodifan belgilangandan katta fayllarni qabul qilish mumkin edi.

**3. Ishlab chiqarishdagi hodisa: Cheklanmagan fayl yuklash**

Fayl almashish xizmati quyidagi kodga ega edi:
\`\`\`go
// YOMON - hajm cheklovi yo'q
io.Copy(file, r.Body) // Cheksiz hajmni qabul qiladi!
\`\`\`

Hujum stsenariysi:
- Hujumchi 100GB fayl yuklaydi
- Butun disk to'ladi
- Xizmat ishdan chiqadi
- Barcha foydalanuvchilar ta'sirlanadi

CopyN bilan tuzatish:
\`\`\`go
// YAXSHI - 50MB cheklov
maxSize := int64(50 * 1024 * 1024)
written, err := io.CopyN(file, r.Body, maxSize)
if err != nil && err != io.EOF {
	// Xatoni qayta ishlash
}
// Mijoz ko'proq yuborishga harakat qilganini tekshirish
if _, err := r.Body.Read(make([]byte, 1)); err != io.EOF {
    return errors.New("fayl hajm chegarasidan oshib ketdi")
}
\`\`\`

**4. Interfeys kompozitsiya patternlari**

**Jurnal yozuvchi Reader:**
\`\`\`go
type LoggingReader struct {
    r io.Reader
    bytesRead int64
}

func (lr *LoggingReader) Read(p []byte) (n int, err error) {
    n, err = lr.r.Read(p)
    lr.bytesRead += int64(n)
    log.Printf("%d bayt o'qildi (jami: %d)", n, lr.bytesRead)
    return n, err
}
\`\`\`

Endi istalgan reader ni o'rang va CopyN jurnal yozish bilan shaffof ishlaydi:
\`\`\`go
file, _ := os.Open("data.txt")
loggingReader := &LoggingReader{r: file}

// CopyN jurnal yozish bilan shaffof ishlaydi
io.CopyN(os.Stdout, loggingReader, 1024)
// Chiqish:
// 1024 bayt o'qildi (jami: 1024)
\`\`\`

**5. Nima uchun qisqa yozishlarni qayta ishlash kerak?**

Ko'pchilik writer lar har doim barcha baytlarni yozadi yoki xato qaytaradi. Lekin ba'zilari bunday emas:
\`\`\`go
type RateLimitedWriter struct {
    w io.Writer
    remaining int
}

func (rl *RateLimitedWriter) Write(p []byte) (int, error) {
    if len(p) > rl.remaining {
	// Faqat qolgan baytlarni yozish mumkin
        n, err := rl.w.Write(p[:rl.remaining])
        rl.remaining -= n
        return n, err // Qisqa yozish!
    }
    n, err := rl.w.Write(p)
    rl.remaining -= n
    return n, err
}
\`\`\`

Agar qisqa yozishlarni tekshirmasangiz, ma'lumotlar jimgina yo'qoladi!

**6. EOF ni qayta ishlash**

\`io.EOF\` har doim ham xato emas:
\`\`\`go
// 10 baytli manbadan 5 bayt nusxalash
src := strings.NewReader("0123456789")
written, err := CopyN(dst, src, 5)
// written == 5, err == nil (muvaffaqiyat!)

// 5 baytli manbadan 10 bayt nusxalash
src := strings.NewReader("01234")
written, err := CopyN(dst, src, 10)
// written == 5, err == io.EOF (manba tugadi)
\`\`\`

**7. Bufer o'lchami muhim**

\`\`\`go
// JUDA KICHIK - tizim chaqiruvlari uchun qo'shimcha xarajatlar
buf := make([]byte, 1) // 1MB nusxalash = 1 million syscall

// OPTIMAL - xotira va syscall lar muvozanati
buf := make([]byte, 32*1024) // 1MB nusxalash = ~32 syscall

// JUDA KATTA - xotira isrofi
buf := make([]byte, 10*1024*1024) // 10MB bufer kamdan-kam kerak
\`\`\`

32KB — Go ning optimal nuqtasi: samarali tizim chaqiruvlari ortiqcha xotira sarflamasdan.

**8. Ishlab chiqarish patternlari**

**Yuklash jarayonini kuzatish:**
\`\`\`go
type ProgressReader struct {
    r io.Reader
    total int64
    callback func(written int64)
}

func (pr *ProgressReader) Read(p []byte) (int, error) {
    n, err := pr.r.Read(p)
    pr.total += int64(n)
    pr.callback(pr.total)
    return n, err
}

// Yuklash jarayonini kuzatish
resp, _ := http.Get("https://example.com/large.zip")
progress := &ProgressReader{
    r: resp.Body,
    callback: func(written int64) {
        fmt.Printf("\\rYuklandi: %d bayt", written)
    },
}

io.CopyN(file, progress, expectedSize)
\`\`\`

**9. Interfeyslar bilan testlash**

Interfeyslar testlashni osonlashtiradi:
\`\`\`go
type ErrorReader struct{}
func (ErrorReader) Read(p []byte) (int, error) {
    return 0, errors.New("tarmoq taymaut")
}

func TestCopyN_ReadError(t *testing.T) {
    var dst bytes.Buffer
    _, err := CopyN(&dst, ErrorReader{}, 100)
    if err == nil {
        t.Fatal("xato kutilgan edi")
    }
}
\`\`\`

Testlarda haqiqiy fayllar yoki tarmoq kerak emas!

**10. Standart kutubxona misollari**

Ushbu standart kutubxona funksiyalari xuddi shu patterndan foydalanadi:
- \`io.Copy(dst, src)\` — EOF gacha nusxalash
- \`io.CopyN(dst, src, n)\` — aynan n bayt nusxalash
- \`io.CopyBuffer(dst, src, buf)\` — maxsus bufer bilan nusxalash
- \`io.ReadFull(r, buf)\` — aynan len(buf) bayt o'qish
- \`io.ReadAtLeast(r, buf, min)\` — kamida min bayt o'qish

\`CopyN\` ni tushunish sizga bu funksiyalarning barchasi ichkarida qanday ishlashini o'rgatadi.

**Asosiy xulosalar:**
- Interfeyslar kuchli kompozitsiyani ta'minlaydi
- Kichik interfeyslar yanada moslashuvchan
- Har doim \`n\` parametrini tekshiring
- Qisqa yozishlarni tekshiring
- EOF ni to'g'ri qayta ishlang
- Samarali bufer o'lchamlaridan foydalaning (32KB)
- Interfeyslar testlashni osonlashtiradi`,
			solutionCode: `package interfaces

import (
	"fmt"
	"io"
)

func CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error) {
	// n parametrini tekshirish
	if n < 0 {
		return 0, fmt.Errorf("invalid length param")
	}
	if n == 0 {
		return 0, nil
	}

	// Samarali bo'lakli o'qish uchun bufer yaratish
	buf := make([]byte, 32*1024)

	// Aynan n bayt nusxalanguncha sikl bajarish
	for written < n {
		// Qancha bayt nusxalash kerakligini hisoblash
		toRead := n - written
		if toRead > int64(len(buf)) {
			toRead = int64(len(buf))
		}

		// Manbadan o'qish (toRead baytgacha)
		nr, er := src.Read(buf[:toRead])

		// Muvaffaqiyatli o'qilgan har qanday baytni yozish
		if nr > 0 {
			nw, ew := dst.Write(buf[:nr])
			if nw > 0 {
				written += int64(nw)
			}

			// Yozish xatolarini qayta ishlash
			if ew != nil {
				return written, ew
			}

			// Qisqa yozishni tekshirish (writer biz berganidan kamroq yozdi)
			if nw != nr {
				return written, io.ErrShortWrite
			}
		}

		// O'qish xatolarini qayta ishlash
		if er != nil {
			// Aynan n bayt yozgandan keyin EOF — muvaffaqiyat
			if er == io.EOF && written == n {
				break
			}
			// n baytdan oldin EOF manba tugadi degani
			if er == io.EOF {
				return written, io.EOF
			}
			// Boshqa xatolar
			return written, er
		}
	}

	return written, nil
}`
		}
	}
};

export default task;
