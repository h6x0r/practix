import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-fundamentals-copy-n',
	title: 'Copy N Bytes Between Streams',
	difficulty: 'medium',	tags: ['go', 'io', 'interfaces', 'buffers'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **CopyN** that copies exactly n bytes from src to dst, similar to io.CopyN but without using it.

**Requirements:**
1. Create function \`CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error)\`
2. Validate that n >= 0 (return error if n < 0)
3. Return early if n == 0
4. Read from src in chunks using an intermediate buffer (32KB)
5. Write read data to dst
6. Track total bytes written
7. Stop when exactly n bytes are written
8. Handle io.EOF correctly (success if we've written n bytes, error otherwise)
9. Handle io.ErrShortWrite when bytes written < bytes read

**Example:**
\`\`\`go
// Copy 100 bytes from reader to writer
written, err := CopyN(writer, reader, 100)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Copied %d bytes\\n", written) // Output: Copied 100 bytes

// Invalid length parameter
written, err := CopyN(writer, reader, -5)
// err = fmt.Errorf("invalid length param")

// Handle EOF when not enough data
written, err := CopyN(writer, reader, 1000)
if written < 1000 {
    // err = io.EOF (not enough data available)
}
\`\`\`

**Constraints:**
- Must use intermediate buffer (32KB recommended)
- Must handle io.EOF and io.ErrShortWrite correctly
- Must validate n >= 0
- Must return early if n == 0
- Must track written bytes accurately`,
	initialCode: `package interfaces

import (
	"io"
)

// TODO: Implement CopyN
func CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error) {
	// TODO: Implement
}`,
	solutionCode: `package interfaces

import (
	"fmt"
	"io"
)

func CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error) {
	if n < 0 {                                                  // Check for negative length
		return 0, fmt.Errorf("invalid length param")          // Return error if invalid
	}
	if n == 0 {                                                 // Optimization for zero bytes
		return 0, nil                                          // Nothing to copy, return early
	}
	buf := make([]byte, 32*1024)                                // Create 32KB buffer for chunked reading
	for written < n {                                           // Loop until we've written n bytes
		toRead := n - written                                 // Calculate remaining bytes to copy
		if toRead > int64(len(buf)) {                         // Prevent buffer overflow
			toRead = int64(len(buf))                          // Cap at buffer size
		}
		nr, er := src.Read(buf[:toRead])                       // Read up to toRead bytes from source
		if nr > 0 {                                            // Only write if we read something
			nw, ew := dst.Write(buf[:nr])                     // Write nr bytes to destination
			if nw > 0 {                                       // Track successful writes
				written += int64(nw)                          // Accumulate bytes written
			}
			if ew != nil {                                    // Check for write errors immediately
				return written, ew                            // Return write error with bytes written so far
			}
			if nw != nr {                                     // Verify all bytes were written
				return written, io.ErrShortWrite              // Return error if short write occurred
			}
		}
		if er != nil {                                         // Check for read errors
			if er == io.EOF && written == n {                // Success case: EOF reached exactly when n bytes copied
				break                                         // Normal completion
			}
			if er == io.EOF {                                 // EOF occurred but we haven't written n bytes
				return written, io.EOF                        // Not enough data in source
			}
			return written, er                                // Return any other read error
		}
	}
	return written, nil                                        // Success: copied exactly n bytes
}`,
	testCode: `package interfaces

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

func Test1(t *testing.T) {
	// Basic copy
	src := strings.NewReader("Hello World")
	dst := &bytes.Buffer{}
	n, err := CopyN(dst, src, 5)
	if err != nil || n != 5 || dst.String() != "Hello" {
		t.Errorf("expected 'Hello', got n=%d, err=%v, str=%s", n, err, dst.String())
	}
}

func Test2(t *testing.T) {
	// Copy zero bytes
	src := strings.NewReader("Hello")
	dst := &bytes.Buffer{}
	n, err := CopyN(dst, src, 0)
	if err != nil || n != 0 {
		t.Errorf("expected 0 bytes, got n=%d, err=%v", n, err)
	}
}

func Test3(t *testing.T) {
	// Negative length
	src := strings.NewReader("Hello")
	dst := &bytes.Buffer{}
	_, err := CopyN(dst, src, -1)
	if err == nil {
		t.Error("expected error for negative length")
	}
}

func Test4(t *testing.T) {
	// Copy exact length of source
	src := strings.NewReader("exact")
	dst := &bytes.Buffer{}
	n, err := CopyN(dst, src, 5)
	if err != nil || n != 5 || dst.String() != "exact" {
		t.Errorf("expected 'exact', got n=%d, err=%v, str=%s", n, err, dst.String())
	}
}

func Test5(t *testing.T) {
	// Source has less data (EOF)
	src := strings.NewReader("abc")
	dst := &bytes.Buffer{}
	n, err := CopyN(dst, src, 10)
	if err != io.EOF || n != 3 {
		t.Errorf("expected EOF with 3 bytes, got n=%d, err=%v", n, err)
	}
}

func Test6(t *testing.T) {
	// Large copy
	src := strings.NewReader(strings.Repeat("x", 100000))
	dst := &bytes.Buffer{}
	n, err := CopyN(dst, src, 50000)
	if err != nil || n != 50000 {
		t.Errorf("expected 50000 bytes, got n=%d, err=%v", n, err)
	}
}

func Test7(t *testing.T) {
	// Copy all data
	data := "test data here"
	src := strings.NewReader(data)
	dst := &bytes.Buffer{}
	n, err := CopyN(dst, src, int64(len(data)))
	if err != nil || n != int64(len(data)) || dst.String() != data {
		t.Errorf("expected full copy, got n=%d, err=%v", n, err)
	}
}

func Test8(t *testing.T) {
	// Empty source
	src := strings.NewReader("")
	dst := &bytes.Buffer{}
	n, err := CopyN(dst, src, 5)
	if err != io.EOF || n != 0 {
		t.Errorf("expected EOF with 0 bytes, got n=%d, err=%v", n, err)
	}
}

func Test9(t *testing.T) {
	// Single byte
	src := strings.NewReader("x")
	dst := &bytes.Buffer{}
	n, err := CopyN(dst, src, 1)
	if err != nil || n != 1 || dst.String() != "x" {
		t.Errorf("expected 'x', got n=%d, err=%v, str=%s", n, err, dst.String())
	}
}

func Test10(t *testing.T) {
	// Copy multiple times
	src := strings.NewReader("abcdefghij")
	dst := &bytes.Buffer{}
	n1, _ := CopyN(dst, src, 3)
	n2, _ := CopyN(dst, src, 3)
	if n1 != 3 || n2 != 3 || dst.String() != "abcdef" {
		t.Errorf("expected 'abcdef', got n1=%d, n2=%d, str=%s", n1, n2, dst.String())
	}
}`,
	hint1: `Validate n >= 0 first, then create a buffer and loop reading from src and writing to dst until exactly n bytes are written.`,
			hint2: `Track bytes written carefully. When you read nr bytes, verify that all nr bytes were actually written (nw == nr). Handle EOF specially: it's OK if you've written exactly n bytes, but error otherwise.`,
			whyItMatters: `CopyN is fundamental for bounded I/O operations, essential in file transfers, network protocols, and resource-constrained scenarios where you need precise control over how much data is transferred.

**Why Bounded Copying:**
- **Resource Protection:** Prevent reading more data than needed into memory
- **Network Safety:** Enforce maximum data transfer per operation
- **Protocol Compliance:** Many protocols require exact byte counts
- **Memory Efficiency:** Control memory usage in large data transfers

**Production Patterns:**

\`\`\`go
// File upload with size limit
func UploadFile(dst io.Writer, src io.Reader, maxSize int64) error {
    written, err := CopyN(dst, src, maxSize)
    if err == io.EOF {
        return fmt.Errorf("incomplete upload: expected %d bytes, got %d", maxSize, written)
    }
    if err != nil {
        return fmt.Errorf("upload error: %w", err)
    }
    return nil
}

// HTTP response body reading with limit
func SafeReadResponse(resp *http.Response, maxBytes int64) ([]byte, error) {
    var buf bytes.Buffer
    _, err := CopyN(&buf, resp.Body, maxBytes)
    if err != nil && err != io.EOF {
        return nil, err
    }
    return buf.Bytes(), nil
}

// Database backup with progress tracking
func BackupDatabase(dst io.Writer, src *sql.Rows, chunkSize int64) error {
    totalWritten := int64(0)
    for {
        written, err := CopyN(dst, src, chunkSize)
        totalWritten += written

        if err == io.EOF {
            fmt.Printf("Backup complete: %d bytes\\n", totalWritten)
            break
        }
        if err != nil {
            return fmt.Errorf("backup failed after %d bytes: %w", totalWritten, err)
        }
    }
    return nil
}

// Buffered network message handling
func ProcessNetworkMessage(conn io.ReadWriter, msgSize int64) error {
    var buf bytes.Buffer
    _, err := CopyN(&buf, conn, msgSize)

    if err != nil && err != io.EOF {
        return fmt.Errorf("failed to read message: %w", err)
    }

    if int64(buf.Len()) != msgSize {
        return fmt.Errorf("incomplete message: expected %d bytes, got %d", msgSize, buf.Len())
    }

    return ProcessMessage(buf.Bytes())
}

// Rate-limited copying with monitoring
func MonitoredCopy(dst io.Writer, src io.Reader, n int64) (int64, error) {
    ticker := time.NewTicker(100 * time.Millisecond)
    defer ticker.Stop()

    written, err := CopyN(dst, src, n)

    select {
    case <-ticker.C:
        rate := float64(written) / 100.0  // bytes per millisecond
        fmt.Printf("Copy rate: %.2f bytes/ms\\n", rate)
    default:
    }

    return written, err
}

// Secure file transfer with verification
func SecureFileCopy(dst io.Writer, src io.Reader, expectedSize int64, expectedHash string) error {
    h := sha256.New()
    multiWriter := io.MultiWriter(dst, h)

    written, err := CopyN(multiWriter, src, expectedSize)

    if err != nil && err != io.EOF {
        return fmt.Errorf("copy error: %w", err)
    }

    if written != expectedSize {
        return fmt.Errorf("size mismatch: expected %d, got %d", expectedSize, written)
    }

    actualHash := fmt.Sprintf("%x", h.Sum(nil))
    if actualHash != expectedHash {
        return fmt.Errorf("hash mismatch")
    }

    return nil
}

// Connection pooling with bounded reads
func BoundedRead(pool *ConnectionPool, msgSize int64) ([]byte, error) {
    conn, err := pool.Get()
    if err != nil {
        return nil, err
    }
    defer pool.Return(conn)

    var buf bytes.Buffer
    _, err = CopyN(&buf, conn, msgSize)

    if err != nil && err != io.EOF {
        return nil, err
    }

    return buf.Bytes(), nil
}
\`\`\`

**Real-World Scenarios:**
- **HTTP Chunked Transfer Encoding:** Reading exactly n bytes per chunk
- **S3/Cloud Upload:** Managing upload progress and chunk boundaries
- **Log Rotation:** Copying log files with size limits
- **Network Protocols:** Reading fixed-size headers and payloads
- **Database Backups:** Streaming large datasets with flow control
- **Rate Limiting:** Enforcing byte-per-second limits with bounded operations

**Common Pitfalls:**
- Not validating n < 0
- Forgetting to handle io.ErrShortWrite (Write returned < bytes Read)
- Not properly distinguishing between EOF (expected when n bytes read) vs EOF (error when n bytes not reached)
- Creating buffer that's too small or too large for the use case

Without proper CopyN implementation, you risk reading too much data (memory issues), not handling partial writes (data loss), and not correctly interpreting EOF conditions (protocol violations).`,	order: 0,
	translations: {
		ru: {
			title: 'Копирование N байтов между потоками',
			description: `Реализуйте **CopyN**, который копирует ровно n байт из src в dst, подобно io.CopyN, но без его использования.

**Требования:**
1. Создайте функцию \`CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error)\`
2. Валидируйте что n >= 0 (верните ошибку если n < 0)
3. Вернитесь ранний если n == 0
4. Читайте из src порциями используя промежуточный буфер (32KB)
5. Пишите прочитанные данные в dst
6. Отслеживайте общее количество скопированных байт
7. Остановитесь когда скопировано ровно n байт
8. Корректно обрабатывайте io.EOF (успех если скопировано n байт, ошибка иначе)
9. Обрабатывайте io.ErrShortWrite когда байт записано < байт прочитано

**Пример:**
\`\`\`go
// Копировать 100 байт из reader в writer
written, err := CopyN(writer, reader, 100)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Скопировано %d байт\\n", written) // Output: Скопировано 100 байт

// Неверный параметр длины
written, err := CopyN(writer, reader, -5)
// err = fmt.Errorf("invalid length param")

// Обработка EOF когда недостаточно данных
written, err := CopyN(writer, reader, 1000)
if written < 1000 {
    // err = io.EOF (недостаточно данных)
}
\`\`\`

**Ограничения:**
- Должен использовать промежуточный буфер (рекомендуется 32KB)
- Должен корректно обрабатывать io.EOF и io.ErrShortWrite
- Должен валидировать n >= 0
- Должен вернуться ранний если n == 0
- Должен точно отслеживать скопированные байты`,
			hint1: `Сначала валидируйте n >= 0, затем создайте буфер и циклом читайте из src и пишите в dst до скопирования ровно n байт.`,
			hint2: `Тщательно отслеживайте скопированные байты. Когда вы читаете nr байт, проверьте что все nr байт были записаны (nw == nr). Обрабатывайте EOF особо: это OK если скопировано ровно n байт, но ошибка иначе.`,
			whyItMatters: `CopyN фундаментален для ограниченных I/O операций, критичен для передачи файлов, сетевых протоколов и сценариев с ограниченными ресурсами где вам нужен точный контроль над объёмом передаваемых данных.

**Почему ограниченное копирование:**
- **Защита ресурсов:** Предотвращение чтения больше данных чем требуется в памяти
- **Безопасность сети:** Обеспечение максимальной передачи данных за операцию
- **Соответствие протоколам:** Многие протоколы требуют точное количество байт
- **Эффективность памяти:** Контроль использования памяти при передаче больших данных

**Продакшен паттерны:**

\`\`\`go
// Загрузка файла с ограничением размера
func UploadFile(dst io.Writer, src io.Reader, maxSize int64) error {
    written, err := CopyN(dst, src, maxSize)
    if err == io.EOF {
        return fmt.Errorf("incomplete upload: expected %d bytes, got %d", maxSize, written)
    }
    if err != nil {
        return fmt.Errorf("upload error: %w", err)
    }
    return nil
}

// Чтение HTTP ответа с ограничением
func SafeReadResponse(resp *http.Response, maxBytes int64) ([]byte, error) {
    var buf bytes.Buffer
    _, err := CopyN(&buf, resp.Body, maxBytes)
    if err != nil && err != io.EOF {
        return nil, err
    }
    return buf.Bytes(), nil
}

// Резервное копирование БД с отслеживанием прогресса
func BackupDatabase(dst io.Writer, src *sql.Rows, chunkSize int64) error {
    totalWritten := int64(0)
    for {
        written, err := CopyN(dst, src, chunkSize)
        totalWritten += written

        if err == io.EOF {
            fmt.Printf("Backup complete: %d bytes\\n", totalWritten)
            break
        }
        if err != nil {
            return fmt.Errorf("backup failed after %d bytes: %w", totalWritten, err)
        }
    }
    return nil
}

// Буферизованная обработка сетевых сообщений
func ProcessNetworkMessage(conn io.ReadWriter, msgSize int64) error {
    var buf bytes.Buffer
    _, err := CopyN(&buf, conn, msgSize)

    if err != nil && err != io.EOF {
        return fmt.Errorf("failed to read message: %w", err)
    }

    if int64(buf.Len()) != msgSize {
        return fmt.Errorf("incomplete message: expected %d bytes, got %d", msgSize, buf.Len())
    }

    return ProcessMessage(buf.Bytes())
}

// Копирование с ограничением скорости и мониторингом
func MonitoredCopy(dst io.Writer, src io.Reader, n int64) (int64, error) {
    ticker := time.NewTicker(100 * time.Millisecond)
    defer ticker.Stop()

    written, err := CopyN(dst, src, n)

    select {
    case <-ticker.C:
        rate := float64(written) / 100.0  // байты в миллисекунду
        fmt.Printf("Copy rate: %.2f bytes/ms\\n", rate)
    default:
    }

    return written, err
}

// Безопасное копирование файла с проверкой
func SecureFileCopy(dst io.Writer, src io.Reader, expectedSize int64, expectedHash string) error {
    h := sha256.New()
    multiWriter := io.MultiWriter(dst, h)

    written, err := CopyN(multiWriter, src, expectedSize)

    if err != nil && err != io.EOF {
        return fmt.Errorf("copy error: %w", err)
    }

    if written != expectedSize {
        return fmt.Errorf("size mismatch: expected %d, got %d", expectedSize, written)
    }

    actualHash := fmt.Sprintf("%x", h.Sum(nil))
    if actualHash != expectedHash {
        return fmt.Errorf("hash mismatch")
    }

    return nil
}

// Пул соединений с ограниченными чтениями
func BoundedRead(pool *ConnectionPool, msgSize int64) ([]byte, error) {
    conn, err := pool.Get()
    if err != nil {
        return nil, err
    }
    defer pool.Return(conn)

    var buf bytes.Buffer
    _, err = CopyN(&buf, conn, msgSize)

    if err != nil && err != io.EOF {
        return nil, err
    }

    return buf.Bytes(), nil
}
\`\`\`

**Практические сценарии:**
- **HTTP Chunked Transfer Encoding:** Чтение ровно n байт на чанк
- **S3/Облачная загрузка:** Управление прогрессом загрузки и границами чанков
- **Ротация логов:** Копирование файлов логов с ограничениями размера
- **Сетевые протоколы:** Чтение заголовков и полезных данных фиксированного размера
- **Резервное копирование БД:** Потоковая передача больших наборов данных с контролем потока
- **Ограничение скорости:** Обеспечение лимитов байт-в-секунду с ограниченными операциями

**Частые ошибки:**
- Не валидировать n < 0
- Забывать обрабатывать io.ErrShortWrite (Write вернул < байт Read)
- Не различать правильно между EOF (ожидается когда n байт прочитано) и EOF (ошибка когда n байт не достигнуто)
- Создание буфера слишком маленького или слишком большого для случая использования

Без правильной реализации CopyN, вы рискуете прочитать слишком много данных (проблемы с памятью), не обработать частичные записи (потеря данных), и неправильно интерпретировать условия EOF (нарушения протокола).`,
			solutionCode: `package interfaces

import (
	"fmt"
	"io"
)

func CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error) {
	if n < 0 {                                                  // Проверить отрицательную длину
		return 0, fmt.Errorf("invalid length param")          // Вернуть ошибку если неверная
	}
	if n == 0 {                                                 // Оптимизация для нуля байт
		return 0, nil                                          // Нечего копировать, вернуться рано
	}
	buf := make([]byte, 32*1024)                                // Создать 32KB буфер для chunked чтения
	for written < n {                                           // Цикл пока не скопировано n байт
		toRead := n - written                                 // Вычислить оставшиеся байты для копирования
		if toRead > int64(len(buf)) {                         // Предотвратить переполнение буфера
			toRead = int64(len(buf))                          // Ограничить размером буфера
		}
		nr, er := src.Read(buf[:toRead])                       // Прочитать до toRead байт из источника
		if nr > 0 {                                            // Писать только если что-то прочитано
			nw, ew := dst.Write(buf[:nr])                     // Записать nr байт в назначение
			if nw > 0 {                                       // Отследить успешные записи
				written += int64(nw)                          // Накопить записанные байты
			}
			if ew != nil {                                    // Проверить ошибки записи немедленно
				return written, ew                            // Вернуть ошибку записи с байтами записанными до сих пор
			}
			if nw != nr {                                     // Проверить все байты были записаны
				return written, io.ErrShortWrite              // Вернуть ошибку если короткая запись произошла
			}
		}
		if er != nil {                                         // Проверить ошибки чтения
			if er == io.EOF && written == n {                // Случай успеха: EOF достигнут ровно когда n байт скопировано
				break                                         // Нормальное завершение
			}
			if er == io.EOF {                                 // EOF произошёл но мы не записали n байт
				return written, io.EOF                        // Недостаточно данных в источнике
			}
			return written, er                                // Вернуть любую другую ошибку чтения
		}
	}
	return written, nil                                        // Успех: скопировано ровно n байт
}`
		},
		uz: {
			title: 'Oqimlar orasida N baytni nusxalash',
			description: `src dan dst ga ровно n baytni ko'chiradigan **CopyN**ni amalga oshiring, io.CopyN ga o'xshash lekin uni ishlatmasdan.

**Talablar:**
1. \`CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error)\` funksiyasini yarating
2. n >= 0 ekanligini tekshiring (agar n < 0 bo'lsa xato qaytaring)
3. Agar n == 0 bo'lsa erta qayting
4. Oraliq bufer (32KB) yordamida src dan portalab o'qing
5. O'qilgan ma'lumotlarni dst ga yozing
6. Ko'chirilingan baytlarning umumiy sonini kuzatib boring
7. Ровно n bayt ko'chirilganda to'xtang
8. io.EOF ni to'g'ri ishlang (muvaffaqiyat agar n bayt ko'chirilsa, boshqacha xato)
9. io.ErrShortWrite ni ishlang yozilgan bayt < o'qilgan bayt bo'lganda

**Misol:**
\`\`\`go
// 100 baytni reader dan writer ga ko'chiring
written, err := CopyN(writer, reader, 100)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Ko'chirildi %d bayt\\n", written) // Output: Ko'chirildi 100 bayt

// Noto'g'ri uzunlik parametri
written, err := CopyN(writer, reader, -5)
// err = fmt.Errorf("invalid length param")

// EOF ni ishlang yetarli ma'lumot yo'q bo'lganda
written, err := CopyN(writer, reader, 1000)
if written < 1000 {
    // err = io.EOF (yetarli ma'lumot yo'q)
}
\`\`\`

**Cheklovlar:**
- Oraliq bufer (32KB tavsiya) ishlatishi kerak
- io.EOF va io.ErrShortWrite ni to'g'ri ishlashi kerak
- n >= 0 ni tekshirishi kerak
- Agar n == 0 bo'lsa erta qaytishi kerak
- Ko'chirilgan baytlarni aniq kuzatishi kerak`,
			hint1: `Avval n >= 0 ni tekshiring, keyin bufer yarating va src dan o'qish va dst ga yozishni tsiklda aynan n bayt ko'chirilgunga qadar takrorlang.`,
			hint2: `Ko'chirilgan baytlarni ehtiyotkorlikka o'qing. nr baytni o'qiganingizda, barcha nr bayt yozilganligini tekshiring (nw == nr). EOF ni alohida ishlang: agar aynan n bayt ko'chirilsa OK, boshqa xolda xato.`,
			whyItMatters: `CopyN cheklangan I/O operatsiyalari uchun fundamental, fayl o'tkazmalarida, tarmoq protokolllarida va taqsimlangan resurslar bo'lgan stsenariylarda juda muhim, bu yerda siz uzatiladigan ma'lumotlar hajmiga aniq nazorat qilishingiz kerak.

**Nega cheklangan ko'chirish:**
- **Resurslarni himoya qilish:** O'qilgan ma'lumotlarning hajmini xotiraga kerak bo'lganidan ortiq emasligiga ishonch hosil qiling
- **Tarmoq xavfsizligi:** Operatsiya uchun maksimal ma'lumot o'tkazishni ta'minlang
- **Protokol muvofiqlik:** Ko'plab protokollar aniq bayt sonini talab qiladi
- **Xotira samaradorligi:** Katta ma'lumotlar o'tkazganda xotira ishlatilishini nazorat qiling

**Production Patterns:**

\`\`\`go
// O'lcham chegarasi bilan fayl yuklash
func UploadFile(dst io.Writer, src io.Reader, maxSize int64) error {
    written, err := CopyN(dst, src, maxSize)
    if err == io.EOF {
        return fmt.Errorf("incomplete upload: expected %d bytes, got %d", maxSize, written)
    }
    if err != nil {
        return fmt.Errorf("upload error: %w", err)
    }
    return nil
}

// Cheklangan HTTP response bodyni o'qish
func SafeReadResponse(resp *http.Response, maxBytes int64) ([]byte, error) {
    var buf bytes.Buffer
    _, err := CopyN(&buf, resp.Body, maxBytes)
    if err != nil && err != io.EOF {
        return nil, err
    }
    return buf.Bytes(), nil
}

// Progress kuzatuvi bilan database zaxira nusxasi
func BackupDatabase(dst io.Writer, src *sql.Rows, chunkSize int64) error {
    totalWritten := int64(0)
    for {
        written, err := CopyN(dst, src, chunkSize)
        totalWritten += written

        if err == io.EOF {
            fmt.Printf("Backup complete: %d bytes\\n", totalWritten)
            break
        }
        if err != nil {
            return fmt.Errorf("backup failed after %d bytes: %w", totalWritten, err)
        }
    }
    return nil
}

// Buferli tarmoq xabarlarini qayta ishlash
func ProcessNetworkMessage(conn io.ReadWriter, msgSize int64) error {
    var buf bytes.Buffer
    _, err := CopyN(&buf, conn, msgSize)

    if err != nil && err != io.EOF {
        return fmt.Errorf("failed to read message: %w", err)
    }

    if int64(buf.Len()) != msgSize {
        return fmt.Errorf("incomplete message: expected %d bytes, got %d", msgSize, buf.Len())
    }

    return ProcessMessage(buf.Bytes())
}

// Monitoring bilan tezlik cheklangan ko'chirish
func MonitoredCopy(dst io.Writer, src io.Reader, n int64) (int64, error) {
    ticker := time.NewTicker(100 * time.Millisecond)
    defer ticker.Stop()

    written, err := CopyN(dst, src, n)

    select {
    case <-ticker.C:
        rate := float64(written) / 100.0  // bayt har millisekund
        fmt.Printf("Copy rate: %.2f bytes/ms\\n", rate)
    default:
    }

    return written, err
}

// Tekshirish bilan xavfsiz fayl ko'chirish
func SecureFileCopy(dst io.Writer, src io.Reader, expectedSize int64, expectedHash string) error {
    h := sha256.New()
    multiWriter := io.MultiWriter(dst, h)

    written, err := CopyN(multiWriter, src, expectedSize)

    if err != nil && err != io.EOF {
        return fmt.Errorf("copy error: %w", err)
    }

    if written != expectedSize {
        return fmt.Errorf("size mismatch: expected %d, got %d", expectedSize, written)
    }

    actualHash := fmt.Sprintf("%x", h.Sum(nil))
    if actualHash != expectedHash {
        return fmt.Errorf("hash mismatch")
    }

    return nil
}

// Cheklangan o'qishlar bilan connection pooling
func BoundedRead(pool *ConnectionPool, msgSize int64) ([]byte, error) {
    conn, err := pool.Get()
    if err != nil {
        return nil, err
    }
    defer pool.Return(conn)

    var buf bytes.Buffer
    _, err = CopyN(&buf, conn, msgSize)

    if err != nil && err != io.EOF {
        return nil, err
    }

    return buf.Bytes(), nil
}
\`\`\`

**Haqiqiy stsenariylar:**
- **HTTP Chunked Transfer Encoding:** Har chunk uchun aniq n bayt o'qish
- **S3/Cloud yuklash:** Yuklash jarayoni va chunk chegaralarini boshqarish
- **Log rotatsiyasi:** O'lcham cheklovlari bilan log fayllarini ko'chirish
- **Tarmoq protokollar:** Belgilangan o'lchamdagi sarlavhalar va payloadlarni o'qish
- **Database zaxira nusxalari:** Katta ma'lumot to'plamlarini oqim nazorati bilan streaming
- **Tezlik cheklash:** Cheklangan operatsiyalar bilan bayt-sekund limitlarini ta'minlash

**Umumiy xatolar:**
- n < 0 ni tekshirmaslik
- io.ErrShortWrite ni qayta ishlashni unutish (Write Read baytlaridan kamroq qaytardi)
- EOF (n bayt o'qilganda kutilgan) va EOF (n bayt erishilmaganda xato) orasidagi farqni to'g'ri ajratmaslik
- Foydalanish holati uchun juda kichik yoki juda katta bufer yaratish

To'g'ri CopyN amalga oshirilmasdan, siz juda ko'p ma'lumot o'qish (xotira muammolari), qisman yozishlarni qayta ishlash (ma'lumot yo'qotish) va EOF shartlarini noto'g'ri talqin qilish (protokol buzilishlari) xavfi ostida qolasiz.`,
			solutionCode: `package interfaces

import (
	"fmt"
	"io"
)

func CopyN(dst io.Writer, src io.Reader, n int64) (written int64, err error) {
	if n < 0 {                                                  // Manfiy uzunlikni tekshirish
		return 0, fmt.Errorf("invalid length param")          // Noto'g'ri bo'lsa xato qaytarish
	}
	if n == 0 {                                                 // Nol baytlar uchun optimallashtirish
		return 0, nil                                          // Ko'chirish uchun hech narsa yo'q, erta qaytish
	}
	buf := make([]byte, 32*1024)                                // Chunked o'qish uchun 32KB bufer yaratish
	for written < n {                                           // n bayt yozilgunga qadar tsikl
		toRead := n - written                                 // Ko'chirish uchun qolgan baytlarni hisoblash
		if toRead > int64(len(buf)) {                         // Bufer to'lib ketishining oldini olish
			toRead = int64(len(buf))                          // Bufer o'lchamida chegaralash
		}
		nr, er := src.Read(buf[:toRead])                       // Manbadan toRead baytgacha o'qish
		if nr > 0 {                                            // Faqat biror narsa o'qilgan bo'lsa yozish
			nw, ew := dst.Write(buf[:nr])                     // Maqsadga nr bayt yozish
			if nw > 0 {                                       // Muvaffaqiyatli yozishlarni kuzatish
				written += int64(nw)                          // Yozilgan baytlarni to'plash
			}
			if ew != nil {                                    // Yozish xatolarini darhol tekshirish
				return written, ew                            // Hozirgacha yozilgan baytlar bilan yozish xatosini qaytarish
			}
			if nw != nr {                                     // Barcha baytlar yozilganligini tekshirish
				return written, io.ErrShortWrite              // Qisqa yozish yuz bergan bo'lsa xato qaytarish
			}
		}
		if er != nil {                                         // O'qish xatolarini tekshirish
			if er == io.EOF && written == n {                // Muvaffaqiyat holati: EOF n bayt ko'chirilganda aniq erishilgan
				break                                         // Oddiy tugash
			}
			if er == io.EOF {                                 // EOF yuz berdi lekin biz n bayt yozmagandik
				return written, io.EOF                        // Manbada yetarli ma'lumot yo'q
			}
			return written, er                                // Boshqa har qanday o'qish xatosini qaytarish
		}
	}
	return written, nil                                        // Muvaffaqiyat: aniq n bayt ko'chirilgan
}`
		}
	}
};

export default task;
