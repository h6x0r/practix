import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-fundamentals-limited-reader',
	title: 'LimitReader: Cap Reading at N Bytes',
	difficulty: 'medium',
	tags: ['go', 'io', 'interfaces', 'security'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement **LimitReader** that returns a Reader that reads from r but stops with EOF after n bytes, similar to io.LimitReader but without using it.

**Requirements:**
1. Create function \`LimitReader(r io.Reader, n int64) io.Reader\`
2. Return a reader that implements io.Reader interface
3. Track total bytes read from the underlying reader
4. Once n bytes have been read, return (0, io.EOF) for subsequent reads
5. Handle reads that would exceed the limit (return partial data)
6. Preserve errors from the underlying reader
7. Handle the case when underlying reader returns EOF before limit
8. Ensure subsequent reads after limit return EOF consistently

**Example:**
\`\`\`go
// Read maximum 100 bytes from file
file, _ := os.Open("large.txt")
limited := LimitReader(file, 100)

data, _ := io.ReadAll(limited)
fmt.Printf("Read %d bytes\\n", len(data)) // Output: Read 100 bytes (max)

// Protect against reading too much data
resp, _ := http.Get("https://api.example.com/data")
limited := LimitReader(resp.Body, 1024*1024) // Max 1MB
body, _ := io.ReadAll(limited)
// Will read at most 1MB, protecting memory

// Read file chunks with limit
file, _ := os.Open("data.bin")
chunk1 := LimitReader(file, 1024)
chunk2 := LimitReader(file, 1024)

io.Copy(dst1, chunk1) // Read first 1KB
io.Copy(dst2, chunk2) // Read next 1KB
\`\`\`

**Constraints:**
- Must not use io.LimitReader
- Must implement io.Reader interface
- Must track bytes read accurately
- Must handle partial reads correctly (when read buffer larger than remaining limit)
- Must return EOF when limit reached
- Must preserve underlying reader errors`,
	initialCode: `package interfaces

import (
	"io"
)

// TODO: Implement LimitReader
func LimitReader(r io.Reader, n int64) io.Reader {
	// TODO: Implement
}`,
	solutionCode: `package interfaces

import (
	"io"
)

// limitedReader implements io.Reader with a maximum byte limit
type limitedReader struct {
	r io.Reader                                         // Underlying reader
	n int64                                             // Remaining bytes to read
}

// Read implements io.Reader interface
func (l *limitedReader) Read(p []byte) (n int, err error) {
	if l.n <= 0 {                                       // Check if limit reached
		return 0, io.EOF                                // Return EOF when limit hit
	}
	if int64(len(p)) > l.n {                            // Read buffer larger than remaining limit
		p = p[0:l.n]                                    // Truncate to remaining bytes
	}
	n, err = l.r.Read(p)                                // Read from underlying reader
	l.n -= int64(n)                                     // Decrement remaining bytes
	return                                              // Return read result
}

func LimitReader(r io.Reader, n int64) io.Reader {
	return &limitedReader{r: r, n: n}                   // Return limited reader
}`,
	testCode: `package interfaces

import (
	"io"
	"strings"
	"testing"
)

func Test1(t *testing.T) {
	// Test basic limit reading
	data := "Hello, World!"
	r := LimitReader(strings.NewReader(data), 5)
	result, err := io.ReadAll(r)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expected := "Hello"
	if string(result) != expected {
		t.Errorf("expected %q, got %q", expected, string(result))
	}
}

func Test2(t *testing.T) {
	// Test reading entire content when limit is larger
	data := "Test"
	r := LimitReader(strings.NewReader(data), 100)
	result, err := io.ReadAll(r)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if string(result) != data {
		t.Errorf("expected %q, got %q", data, string(result))
	}
}

func Test3(t *testing.T) {
	// Test zero limit
	data := "Hello"
	r := LimitReader(strings.NewReader(data), 0)
	result, err := io.ReadAll(r)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(result) != 0 {
		t.Errorf("expected empty result, got %d bytes", len(result))
	}
}

func Test4(t *testing.T) {
	// Test EOF is returned after limit
	data := "Hello, World!"
	r := LimitReader(strings.NewReader(data), 5)
	buf := make([]byte, 10)
	n, err := r.Read(buf)
	if err != nil {
		t.Errorf("unexpected error on first read: %v", err)
	}
	if n != 5 {
		t.Errorf("expected to read 5 bytes, got %d", n)
	}
	// Second read should return EOF
	n, err = r.Read(buf)
	if err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 bytes, got %d", n)
	}
}

func Test5(t *testing.T) {
	// Test multiple small reads
	data := "0123456789"
	r := LimitReader(strings.NewReader(data), 7)
	buf := make([]byte, 3)

	n, err := r.Read(buf)
	if err != nil || n != 3 || string(buf[:n]) != "012" {
		t.Errorf("first read failed: n=%d, err=%v, data=%q", n, err, string(buf[:n]))
	}

	n, err = r.Read(buf)
	if err != nil || n != 3 || string(buf[:n]) != "345" {
		t.Errorf("second read failed: n=%d, err=%v, data=%q", n, err, string(buf[:n]))
	}

	n, err = r.Read(buf)
	if err != nil || n != 1 || string(buf[:n]) != "6" {
		t.Errorf("third read failed: n=%d, err=%v, data=%q", n, err, string(buf[:n]))
	}
}

func Test6(t *testing.T) {
	// Test empty reader
	r := LimitReader(strings.NewReader(""), 10)
	result, err := io.ReadAll(r)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(result) != 0 {
		t.Errorf("expected empty result, got %d bytes", len(result))
	}
}

func Test7(t *testing.T) {
	// Test large buffer read with limit
	data := "Hello, World!"
	r := LimitReader(strings.NewReader(data), 5)
	buf := make([]byte, 100)
	n, err := r.Read(buf)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if n != 5 {
		t.Errorf("expected to read 5 bytes, got %d", n)
	}
	if string(buf[:n]) != "Hello" {
		t.Errorf("expected %q, got %q", "Hello", string(buf[:n]))
	}
}

func Test8(t *testing.T) {
	// Test single byte limit
	data := "ABC"
	r := LimitReader(strings.NewReader(data), 1)
	result, err := io.ReadAll(r)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expected := "A"
	if string(result) != expected {
		t.Errorf("expected %q, got %q", expected, string(result))
	}
}

func Test9(t *testing.T) {
	// Test exact limit match
	data := "12345"
	r := LimitReader(strings.NewReader(data), 5)
	result, err := io.ReadAll(r)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if string(result) != data {
		t.Errorf("expected %q, got %q", data, string(result))
	}
}

func Test10(t *testing.T) {
	// Test consecutive EOF calls
	data := "Test"
	r := LimitReader(strings.NewReader(data), 2)
	io.ReadAll(r) // Consume all limited data

	buf := make([]byte, 10)
	n, err := r.Read(buf)
	if err != io.EOF || n != 0 {
		t.Errorf("expected EOF on first call, got n=%d, err=%v", n, err)
	}

	n, err = r.Read(buf)
	if err != io.EOF || n != 0 {
		t.Errorf("expected EOF on second call, got n=%d, err=%v", n, err)
	}
}`,
	hint1: `Create a struct that holds the reader and a counter (n) for remaining bytes. In Read(), check if limit is reached before reading.`,
	hint2: `If the read buffer p is larger than remaining bytes (l.n), truncate it to p[0:l.n] before reading. After reading, decrement l.n by the number of bytes actually read. Return EOF when l.n <= 0.`,
	whyItMatters: `LimitReader is essential for security and resource management, preventing denial-of-service attacks and memory exhaustion from reading unlimited data.

**Why Limit Reads:**
- **Security:** Prevent DoS attacks from malicious large payloads
- **Memory Safety:** Avoid OOM errors from reading unbounded data
- **Resource Control:** Enforce quotas on data consumption
- **Protocol Compliance:** Many protocols specify maximum message sizes
- **Cost Management:** Limit data transfer costs in cloud environments

**Production Patterns:**

\`\`\`go
// HTTP request body size limiting (prevent DoS)
func SafeHTTPHandler(w http.ResponseWriter, r *http.Request) {
    const maxBodySize = 1024 * 1024 // 1MB limit
    r.Body = io.NopCloser(LimitReader(r.Body, maxBodySize))

    body, err := io.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "Request too large", http.StatusRequestEntityTooLarge)
        return
    }

    ProcessRequest(body)
}

// File upload with size restrictions
func UploadFile(src io.Reader, maxSize int64) error {
    limited := LimitReader(src, maxSize)

    uploaded, err := io.Copy(storage, limited)
    if uploaded == maxSize {
        // Check if there's more data (file too large)
        buf := make([]byte, 1)
        n, _ := src.Read(buf)
        if n > 0 {
            return fmt.Errorf("file exceeds maximum size of %d bytes", maxSize)
        }
    }

    return err
}

// API rate limiting by bytes
func RateLimitedRead(r io.Reader, bytesPerSecond int64) io.Reader {
    ticker := time.NewTicker(time.Second)
    return &rateLimitedReader{
        reader: r,
        limit:  bytesPerSecond,
        ticker: ticker,
    }
}

// Stream processing with memory bounds
func ProcessStream(src io.Reader, chunkSize int64) error {
    for {
        chunk := LimitReader(src, chunkSize)
        data, err := io.ReadAll(chunk)

        if len(data) > 0 {
            if err := ProcessChunk(data); err != nil {
                return err
            }
        }

        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }
    }
    return nil
}

// Database query result limiting
func LimitedQuery(db *sql.DB, query string, maxBytes int64) ([]byte, error) {
    rows, err := db.Query(query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    limited := LimitReader(NewRowsReader(rows), maxBytes)
    return io.ReadAll(limited)
}

// WebSocket message size enforcement
func HandleWebSocket(conn *websocket.Conn, maxMessageSize int64) {
    for {
        _, reader, err := conn.NextReader()
        if err != nil {
            break
        }

        limited := LimitReader(reader, maxMessageSize)
        message, err := io.ReadAll(limited)

        if err != nil {
            conn.WriteMessage(websocket.CloseMessage, []byte("Message too large"))
            break
        }

        HandleMessage(message)
    }
}

// S3/Cloud storage download with quota
func DownloadWithQuota(bucket, key string, quota int64) ([]byte, error) {
    obj, err := s3Client.GetObject(&s3.GetObjectInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
    })
    if err != nil {
        return nil, err
    }
    defer obj.Body.Close()

    limited := LimitReader(obj.Body, quota)
    return io.ReadAll(limited)
}

// Multipart form parsing with limits
func ParseMultipartForm(r *http.Request, maxMemory, maxFileSize int64) error {
    reader, err := r.MultipartReader()
    if err != nil {
        return err
    }

    for {
        part, err := reader.NextPart()
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }

        limited := LimitReader(part, maxFileSize)
        data, err := io.ReadAll(limited)

        if len(data) == int(maxFileSize) {
            return fmt.Errorf("file %s exceeds size limit", part.FileName())
        }

        SaveFile(part.FileName(), data)
    }

    return nil
}

// Network protocol message parsing
func ReadProtocolMessage(conn net.Conn, maxSize int64) ([]byte, error) {
    // Read header (4 bytes for message length)
    header := make([]byte, 4)
    _, err := io.ReadFull(conn, header)
    if err != nil {
        return nil, err
    }

    msgLen := binary.BigEndian.Uint32(header)
    if int64(msgLen) > maxSize {
        return nil, fmt.Errorf("message size %d exceeds limit %d", msgLen, maxSize)
    }

    limited := LimitReader(conn, int64(msgLen))
    return io.ReadAll(limited)
}
\`\`\`

**Real-World Scenarios:**
- **API Gateways:** Enforce maximum request body sizes
- **File Upload Services:** Prevent users from uploading files that are too large
- **Streaming Services:** Control memory usage when processing large streams
- **Database Operations:** Limit result set sizes to prevent memory issues
- **Message Queues:** Enforce maximum message sizes
- **Log Processing:** Read log files in bounded chunks

**Common Pitfalls:**
- Not truncating the read buffer when it exceeds remaining limit
- Forgetting to decrement the counter after each read
- Not handling the case when underlying reader returns EOF early
- Not returning EOF consistently after limit is reached
- Off-by-one errors in limit checking

Without LimitReader, applications are vulnerable to resource exhaustion attacks where malicious actors send extremely large payloads, causing servers to run out of memory or disk space.`,
	order: 3,
	translations: {
		ru: {
			title: 'LimitReader: ограничение чтения до N байт',
			description: `Реализуйте **LimitReader**, который возвращает Reader, читающий из r, но останавливающийся с EOF после n байт, подобно io.LimitReader, но без его использования.

**Требования:**
1. Создайте функцию \`LimitReader(r io.Reader, n int64) io.Reader\`
2. Верните reader, реализующий интерфейс io.Reader
3. Отслеживайте общее количество прочитанных байт из базового reader
4. После прочтения n байт возвращайте (0, io.EOF) для последующих чтений
5. Обрабатывайте чтения, которые превысили бы лимит (возвращайте частичные данные)
6. Сохраните ошибки из базового reader
7. Обрабатывайте случай когда базовый reader возвращает EOF до лимита
8. Убедитесь что последующие чтения после лимита возвращают EOF последовательно

**Пример:**
\`\`\`go
// Прочитать максимум 100 байт из файла
file, _ := os.Open("large.txt")
limited := LimitReader(file, 100)

data, _ := io.ReadAll(limited)
fmt.Printf("Прочитано %d байт\\n", len(data)) // Output: Прочитано 100 байт (макс)

// Защититься от чтения слишком большого объема данных
resp, _ := http.Get("https://api.example.com/data")
limited := LimitReader(resp.Body, 1024*1024) // Макс 1MB
body, _ := io.ReadAll(limited)
// Прочитает максимум 1MB, защищая память

// Читать куски файла с лимитом
file, _ := os.Open("data.bin")
chunk1 := LimitReader(file, 1024)
chunk2 := LimitReader(file, 1024)

io.Copy(dst1, chunk1) // Прочитать первые 1KB
io.Copy(dst2, chunk2) // Прочитать следующие 1KB
\`\`\`

**Ограничения:**
- Не должен использовать io.LimitReader
- Должен реализовать интерфейс io.Reader
- Должен точно отслеживать прочитанные байты
- Должен корректно обрабатывать частичные чтения (когда буфер чтения больше оставшегося лимита)
- Должен возвращать EOF когда лимит достигнут
- Должен сохранять ошибки базового reader`,
			hint1: `Создайте struct, содержащий reader и счётчик (n) для оставшихся байт. В Read() проверьте достигнут ли лимит перед чтением.`,
			hint2: `Если буфер чтения p больше оставшихся байт (l.n), обрежьте его до p[0:l.n] перед чтением. После чтения уменьшите l.n на количество фактически прочитанных байт. Возвращайте EOF когда l.n <= 0.`,
			whyItMatters: `LimitReader критичен для безопасности и управления ресурсами, предотвращая DoS атаки и исчерпание памяти от чтения неограниченных данных.

**Почему ограничивать чтения:**
- **Безопасность:** Предотвратить DoS атаки от вредоносных больших payload
- **Безопасность памяти:** Избежать OOM ошибок от чтения неограниченных данных
- **Контроль ресурсов:** Обеспечить квоты на потребление данных
- **Соответствие протоколам:** Многие протоколы задают максимальные размеры сообщений
- **Управление затратами:** Ограничить стоимость передачи данных в облачных средах

**Production паттерны:**

\`\`\`go
// Ограничение размера тела HTTP запроса (защита от DoS)
func SafeHTTPHandler(w http.ResponseWriter, r *http.Request) {
    const maxBodySize = 1024 * 1024 // Лимит 1MB
    r.Body = io.NopCloser(LimitReader(r.Body, maxBodySize))

    body, err := io.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "Запрос слишком большой", http.StatusRequestEntityTooLarge)
        return
    }

    ProcessRequest(body)
}

// Загрузка файла с ограничением размера
func UploadFile(src io.Reader, maxSize int64) error {
    limited := LimitReader(src, maxSize)

    uploaded, err := io.Copy(storage, limited)
    if uploaded == maxSize {
        // Проверить есть ли еще данные (файл слишком большой)
        buf := make([]byte, 1)
        n, _ := src.Read(buf)
        if n > 0 {
            return fmt.Errorf("файл превышает максимальный размер %d байт", maxSize)
        }
    }

    return err
}

// API rate limiting по байтам
func RateLimitedRead(r io.Reader, bytesPerSecond int64) io.Reader {
    ticker := time.NewTicker(time.Second)
    return &rateLimitedReader{
        reader: r,
        limit:  bytesPerSecond,
        ticker: ticker,
    }
}

// Обработка потоков с ограничением памяти
func ProcessStream(src io.Reader, chunkSize int64) error {
    for {
        chunk := LimitReader(src, chunkSize)
        data, err := io.ReadAll(chunk)

        if len(data) > 0 {
            if err := ProcessChunk(data); err != nil {
                return err
            }
        }

        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }
    }
    return nil
}

// Ограничение результатов запросов к БД
func LimitedQuery(db *sql.DB, query string, maxBytes int64) ([]byte, error) {
    rows, err := db.Query(query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    limited := LimitReader(NewRowsReader(rows), maxBytes)
    return io.ReadAll(limited)
}

// Контроль размера WebSocket сообщений
func HandleWebSocket(conn *websocket.Conn, maxMessageSize int64) {
    for {
        _, reader, err := conn.NextReader()
        if err != nil {
            break
        }

        limited := LimitReader(reader, maxMessageSize)
        message, err := io.ReadAll(limited)

        if err != nil {
            conn.WriteMessage(websocket.CloseMessage, []byte("Сообщение слишком большое"))
            break
        }

        HandleMessage(message)
    }
}

// Загрузка из S3/облачного хранилища с квотой
func DownloadWithQuota(bucket, key string, quota int64) ([]byte, error) {
    obj, err := s3Client.GetObject(&s3.GetObjectInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
    })
    if err != nil {
        return nil, err
    }
    defer obj.Body.Close()

    limited := LimitReader(obj.Body, quota)
    return io.ReadAll(limited)
}

// Парсинг multipart form с ограничениями
func ParseMultipartForm(r *http.Request, maxMemory, maxFileSize int64) error {
    reader, err := r.MultipartReader()
    if err != nil {
        return err
    }

    for {
        part, err := reader.NextPart()
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }

        limited := LimitReader(part, maxFileSize)
        data, err := io.ReadAll(limited)

        if len(data) == int(maxFileSize) {
            return fmt.Errorf("файл %s превышает ограничение размера", part.FileName())
        }

        SaveFile(part.FileName(), data)
    }

    return nil
}

// Парсинг сетевых протокольных сообщений
func ReadProtocolMessage(conn net.Conn, maxSize int64) ([]byte, error) {
    // Читать заголовок (4 байта для длины сообщения)
    header := make([]byte, 4)
    _, err := io.ReadFull(conn, header)
    if err != nil {
        return nil, err
    }

    msgLen := binary.BigEndian.Uint32(header)
    if int64(msgLen) > maxSize {
        return nil, fmt.Errorf("размер сообщения %d превышает лимит %d", msgLen, maxSize)
    }

    limited := LimitReader(conn, int64(msgLen))
    return io.ReadAll(limited)
}
\`\`\`

**Реальные сценарии:**
- **API Gateway:** Контроль максимального размера тела запроса
- **Сервисы загрузки файлов:** Предотвращение загрузки слишком больших файлов
- **Стриминг сервисы:** Контроль использования памяти при обработке больших потоков
- **Операции с БД:** Ограничение размеров результатов для предотвращения проблем с памятью
- **Очереди сообщений:** Контроль максимального размера сообщений
- **Обработка логов:** Чтение файлов логов ограниченными порциями

**Распространённые ошибки:**
- Не обрезать буфер чтения, когда он превышает оставшийся лимит
- Забыть уменьшить счётчик после каждого чтения
- Не обработать случай, когда базовый reader возвращает EOF раньше
- Не возвращать EOF последовательно после достижения лимита
- Ошибки на единицу при проверке лимита

Без LimitReader приложения уязвимы к атакам исчерпания ресурсов, где злоумышленники отправляют экстремально большие payload, заставляя серверы исчерпывать память или дисковое пространство.`,
			solutionCode: `package interfaces

import (
	"io"
)

// limitedReader реализует io.Reader с максимальным лимитом байт
type limitedReader struct {
	r io.Reader                                         // Базовый reader
	n int64                                             // Оставшиеся байты для чтения
}

// Read реализует интерфейс io.Reader
func (l *limitedReader) Read(p []byte) (n int, err error) {
	if l.n <= 0 {                                       // Проверить достигнут ли лимит
		return 0, io.EOF                                // Вернуть EOF когда лимит достигнут
	}
	if int64(len(p)) > l.n {                            // Буфер чтения больше оставшегося лимита
		p = p[0:l.n]                                    // Обрезать до оставшихся байт
	}
	n, err = l.r.Read(p)                                // Читать из базового reader
	l.n -= int64(n)                                     // Уменьшить оставшиеся байты
	return                                              // Вернуть результат чтения
}

func LimitReader(r io.Reader, n int64) io.Reader {
	return &limitedReader{r: r, n: n}                   // Вернуть limited reader
}`
		},
		uz: {
			title: 'LimitReader: N baytgacha o\'qishni cheklash',
			description: `r dan o'qiydigan lekin n baytdan keyin EOF bilan to'xtaydigan Reader qaytaradigan **LimitReader**ni amalga oshiring, io.LimitReader ga o'xshash lekin uni ishlatmasdan.

**Talablar:**
1. \`LimitReader(r io.Reader, n int64) io.Reader\` funksiyasini yarating
2. io.Reader interfeysini amalga oshiradigan reader qaytaring
3. Asosiy reader dan o'qilgan umumiy baytlarni kuzatib boring
4. n bayt o'qilgandan keyin keyingi o'qishlar uchun (0, io.EOF) qaytaring
5. Limitdan oshib ketadigan o'qishlarni ishlang (qisman ma'lumotlarni qaytaring)
6. Asosiy reader dan xatolarni saqlang
7. Asosiy reader limitdan oldin EOF qaytarganda holatni ishlang
8. Limitdan keyin keyingi o'qishlar doimo EOF qaytarishiga ishonch hosil qiling

**Misol:**
\`\`\`go
// Fayldan maksimum 100 bayt o'qish
file, _ := os.Open("large.txt")
limited := LimitReader(file, 100)

data, _ := io.ReadAll(limited)
fmt.Printf("O'qildi %d bayt\\n", len(data)) // Output: O'qildi 100 bayt (maks)

// Juda ko'p ma'lumot o'qishdan himoya
resp, _ := http.Get("https://api.example.com/data")
limited := LimitReader(resp.Body, 1024*1024) // Maks 1MB
body, _ := io.ReadAll(limited)
// Ko'pi bilan 1MB o'qiladi, xotirani himoya qiladi

// Limit bilan fayl qismlarini o'qish
file, _ := os.Open("data.bin")
chunk1 := LimitReader(file, 1024)
chunk2 := LimitReader(file, 1024)

io.Copy(dst1, chunk1) // Birinchi 1KB o'qish
io.Copy(dst2, chunk2) // Keyingi 1KB o'qish
\`\`\`

**Cheklovlar:**
- io.LimitReader ishlatmasligi kerak
- io.Reader interfeysini amalga oshirishi kerak
- O'qilgan baytlarni aniq kuzatishi kerak
- Qisman o'qishlarni to'g'ri ishlashi kerak (o'qish buferi qolgan limitdan katta bo'lganda)
- Limit yetib borganda EOF qaytarishi kerak
- Asosiy reader xatolarini saqlaishi kerak`,
			hint1: `Reader va qolgan baytlar uchun hisoblagich (n) saqlaydigan struct yarating. Read() da o'qishdan oldin limit yetib borgan yoki yo'qligini tekshiring.`,
			hint2: `Agar o'qish buferi p qolgan baytlardan (l.n) katta bo'lsa, o'qishdan oldin uni p[0:l.n] ga qirqing. O'qishdan keyin l.n ni haqiqatda o'qilgan baytlar soniga kamaytiring. l.n <= 0 bo'lganda EOF qaytaring.`,
			whyItMatters: `LimitReader xavfsizlik va resurslarni boshqarish uchun muhim, DoS hujumlarini va cheksiz ma'lumotlarni o'qishdan xotira tugashini oldini oladi.

**Nega o'qishni cheklash:**
- **Xavfsizlik:** Zararli katta payloadlardan DoS hujumlarini oldini olish
- **Xotira xavfsizligi:** Cheksiz ma'lumotlarni o'qishdan OOM xatolaridan qochish
- **Resurslarni boshqarish:** Ma'lumot iste'moli kvotalarini ta'minlash
- **Protokol muvofiqlik:** Ko'plab protokollar maksimal xabar o'lchamlarini belgilaydi
- **Xarajatlarni boshqarish:** Bulut muhitlarida ma'lumot uzatish xarajatlarini cheklash

**Production patternlari:**

\`\`\`go
// HTTP so'rov tanasi o'lchamini cheklash (DoS dan himoya)
func SafeHTTPHandler(w http.ResponseWriter, r *http.Request) {
    const maxBodySize = 1024 * 1024 // 1MB limit
    r.Body = io.NopCloser(LimitReader(r.Body, maxBodySize))

    body, err := io.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "So'rov juda katta", http.StatusRequestEntityTooLarge)
        return
    }

    ProcessRequest(body)
}

// O'lcham cheklovi bilan fayl yuklash
func UploadFile(src io.Reader, maxSize int64) error {
    limited := LimitReader(src, maxSize)

    uploaded, err := io.Copy(storage, limited)
    if uploaded == maxSize {
        // Yana ma'lumot bor yoki yo'qligini tekshirish (fayl juda katta)
        buf := make([]byte, 1)
        n, _ := src.Read(buf)
        if n > 0 {
            return fmt.Errorf("fayl maksimal o'lcham %d baytdan oshib ketdi", maxSize)
        }
    }

    return err
}

// Bayt bo'yicha API rate limiting
func RateLimitedRead(r io.Reader, bytesPerSecond int64) io.Reader {
    ticker := time.NewTicker(time.Second)
    return &rateLimitedReader{
        reader: r,
        limit:  bytesPerSecond,
        ticker: ticker,
    }
}

// Xotira chegaralari bilan oqimlarni qayta ishlash
func ProcessStream(src io.Reader, chunkSize int64) error {
    for {
        chunk := LimitReader(src, chunkSize)
        data, err := io.ReadAll(chunk)

        if len(data) > 0 {
            if err := ProcessChunk(data); err != nil {
                return err
            }
        }

        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }
    }
    return nil
}

// Ma'lumotlar bazasi so'rov natijalarini cheklash
func LimitedQuery(db *sql.DB, query string, maxBytes int64) ([]byte, error) {
    rows, err := db.Query(query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    limited := LimitReader(NewRowsReader(rows), maxBytes)
    return io.ReadAll(limited)
}

// WebSocket xabar o'lchamini nazorat qilish
func HandleWebSocket(conn *websocket.Conn, maxMessageSize int64) {
    for {
        _, reader, err := conn.NextReader()
        if err != nil {
            break
        }

        limited := LimitReader(reader, maxMessageSize)
        message, err := io.ReadAll(limited)

        if err != nil {
            conn.WriteMessage(websocket.CloseMessage, []byte("Xabar juda katta"))
            break
        }

        HandleMessage(message)
    }
}

// Kvota bilan S3/bulut xotirasidan yuklab olish
func DownloadWithQuota(bucket, key string, quota int64) ([]byte, error) {
    obj, err := s3Client.GetObject(&s3.GetObjectInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
    })
    if err != nil {
        return nil, err
    }
    defer obj.Body.Close()

    limited := LimitReader(obj.Body, quota)
    return io.ReadAll(limited)
}

// Cheklovlar bilan multipart form parsing
func ParseMultipartForm(r *http.Request, maxMemory, maxFileSize int64) error {
    reader, err := r.MultipartReader()
    if err != nil {
        return err
    }

    for {
        part, err := reader.NextPart()
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }

        limited := LimitReader(part, maxFileSize)
        data, err := io.ReadAll(limited)

        if len(data) == int(maxFileSize) {
            return fmt.Errorf("fayl %s o'lcham limitidan oshib ketdi", part.FileName())
        }

        SaveFile(part.FileName(), data)
    }

    return nil
}

// Tarmoq protokol xabarlarini parsing
func ReadProtocolMessage(conn net.Conn, maxSize int64) ([]byte, error) {
    // Sarlavhani o'qish (xabar uzunligi uchun 4 bayt)
    header := make([]byte, 4)
    _, err := io.ReadFull(conn, header)
    if err != nil {
        return nil, err
    }

    msgLen := binary.BigEndian.Uint32(header)
    if int64(msgLen) > maxSize {
        return nil, fmt.Errorf("xabar o'lchami %d limitdan %d oshib ketdi", msgLen, maxSize)
    }

    limited := LimitReader(conn, int64(msgLen))
    return io.ReadAll(limited)
}
\`\`\`

**Haqiqiy stsenariylar:**
- **API Gateway:** Maksimal so'rov tanasi o'lchamini nazorat qilish
- **Fayl yuklash xizmatlari:** Juda katta fayllar yuklanishining oldini olish
- **Streaming xizmatlar:** Katta oqimlarni qayta ishlashda xotira ishlatilishini nazorat qilish
- **DB operatsiyalari:** Xotira muammolarini oldini olish uchun natija o'lchamlarini cheklash
- **Xabar navbatlari:** Maksimal xabar o'lchamini nazorat qilish
- **Log qayta ishlash:** Log fayllarini cheklangan qismlarda o'qish

**Keng tarqalgan xatolar:**
- O'qish buferini qolgan limitdan oshganda qirqmaslik
- Har bir o'qishdan keyin hisoblagichni kamaytirishni unutish
- Asosiy reader ertaroq EOF qaytarganda holatni ishlamaslik
- Limitga yetgandan keyin EOF ni izchil qaytarmaslik
- Limitni tekshirishda birga xatoliklar

LimitReader bo'lmasa, ilovalar resurslarni tugash hujumlariga zaif bo'ladi, bu yerda hujumchilar juda katta payloadlar yuboradi va serverlarning xotirasi yoki disk maydoni tugashiga olib keladi.`,
			solutionCode: `package interfaces

import (
	"io"
)

// limitedReader maksimal bayt limiti bilan io.Reader ni amalga oshiradi
type limitedReader struct {
	r io.Reader                                         // Asosiy reader
	n int64                                             // O'qish uchun qolgan baytlar
}

// Read io.Reader interfeysini amalga oshiradi
func (l *limitedReader) Read(p []byte) (n int, err error) {
	if l.n <= 0 {                                       // Limit yetib borgan yoki yo'qligini tekshirish
		return 0, io.EOF                                // Limit yetganda EOF qaytarish
	}
	if int64(len(p)) > l.n {                            // O'qish buferi qolgan limitdan katta
		p = p[0:l.n]                                    // Qolgan baytlarga qirqish
	}
	n, err = l.r.Read(p)                                // Asosiy reader dan o'qish
	l.n -= int64(n)                                     // Qolgan baytlarni kamaytirish
	return                                              // O'qish natijasini qaytarish
}

func LimitReader(r io.Reader, n int64) io.Reader {
	return &limitedReader{r: r, n: n}                   // Limited reader ni qaytarish
}`
		}
	}
};

export default task;
