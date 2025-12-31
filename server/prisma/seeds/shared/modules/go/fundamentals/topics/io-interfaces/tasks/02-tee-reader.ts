import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-fundamentals-tee-reader',
	title: 'TeeReader: Read and Copy Data',
	difficulty: 'medium',
	tags: ['go', 'io', 'interfaces', 'streams'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement **TeeReader** that returns a Reader that writes to w what it reads from r, similar to io.TeeReader but without using it.

**Requirements:**
1. Create function \`TeeReader(r io.Reader, w io.Writer) io.Reader\`
2. Return a custom reader that implements io.Reader interface
3. On each Read() call, read from the original reader r
4. Write the read data to writer w before returning
5. Handle partial writes correctly (if writer can't write all bytes)
6. Return the data read along with any errors
7. Preserve read errors from the source reader
8. Preserve write errors from the destination writer

**Example:**
\`\`\`go
// Capture data while reading
var buf bytes.Buffer
reader := TeeReader(file, &buf)

// Read from file, automatically copied to buf
data := make([]byte, 100)
n, err := reader.Read(data)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Read %d bytes\\n", n)
fmt.Printf("Buffer contains: %s\\n", buf.String())

// Logging API responses while reading
logFile, _ := os.Create("api.log")
tee := TeeReader(resp.Body, logFile)
body, _ := io.ReadAll(tee) // Body read AND logged

// Hashing file while reading
hash := sha256.New()
teeReader := TeeReader(file, hash)
io.Copy(dst, teeReader) // File copied AND hashed
\`\`\`

**Constraints:**
- Must not use io.TeeReader
- Must implement io.Reader interface
- Must handle read and write errors correctly
- Must write exactly what was successfully read
- Return original read data even if write fails`,
	initialCode: `package interfaces

import (
	"io"
)

// TODO: Implement TeeReader
func TeeReader(r io.Reader, w io.Writer) io.Reader {
	// TODO: Implement
}`,
	solutionCode: `package interfaces

import (
	"io"
)

// teeReader implements io.Reader and writes to w what it reads from r
type teeReader struct {
	r io.Reader                                         // Source reader
	w io.Writer                                         // Destination writer
}

// Read implements io.Reader interface
func (t *teeReader) Read(p []byte) (n int, err error) {
	n, err = t.r.Read(p)                                // Read from source reader
	if n > 0 {                                          // Only write if we read something
		if n, err := t.w.Write(p[:n]); err != nil {    // Write exactly what we read
			return n, err                               // Return write error if occurred
		}
	}
	return                                              // Return read result (n, err)
}

func TeeReader(r io.Reader, w io.Writer) io.Reader {
	return &teeReader{r: r, w: w}                       // Return custom tee reader
}`,
	testCode: `package interfaces

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

func Test1(t *testing.T) {
	// Basic tee read
	src := strings.NewReader("Hello World")
	var buf bytes.Buffer
	tee := TeeReader(src, &buf)
	data, _ := io.ReadAll(tee)
	if string(data) != "Hello World" || buf.String() != "Hello World" {
		t.Errorf("expected 'Hello World', got data=%s, buf=%s", data, buf.String())
	}
}

func Test2(t *testing.T) {
	// Empty reader
	src := strings.NewReader("")
	var buf bytes.Buffer
	tee := TeeReader(src, &buf)
	data, _ := io.ReadAll(tee)
	if len(data) != 0 || buf.Len() != 0 {
		t.Errorf("expected empty, got data=%s, buf=%s", data, buf.String())
	}
}

func Test3(t *testing.T) {
	// Partial read
	src := strings.NewReader("Hello World")
	var buf bytes.Buffer
	tee := TeeReader(src, &buf)
	p := make([]byte, 5)
	n, _ := tee.Read(p)
	if n != 5 || string(p) != "Hello" || buf.String() != "Hello" {
		t.Errorf("expected 'Hello', got n=%d, p=%s, buf=%s", n, p, buf.String())
	}
}

func Test4(t *testing.T) {
	// Multiple reads
	src := strings.NewReader("abcdef")
	var buf bytes.Buffer
	tee := TeeReader(src, &buf)
	p := make([]byte, 2)
	tee.Read(p)
	tee.Read(p)
	tee.Read(p)
	if buf.String() != "abcdef" {
		t.Errorf("expected 'abcdef', got buf=%s", buf.String())
	}
}

func Test5(t *testing.T) {
	// Single byte
	src := strings.NewReader("x")
	var buf bytes.Buffer
	tee := TeeReader(src, &buf)
	data, _ := io.ReadAll(tee)
	if string(data) != "x" || buf.String() != "x" {
		t.Errorf("expected 'x', got data=%s, buf=%s", data, buf.String())
	}
}

func Test6(t *testing.T) {
	// Large data
	large := strings.Repeat("abc", 10000)
	src := strings.NewReader(large)
	var buf bytes.Buffer
	tee := TeeReader(src, &buf)
	data, _ := io.ReadAll(tee)
	if string(data) != large || buf.String() != large {
		t.Errorf("large data mismatch")
	}
}

func Test7(t *testing.T) {
	// Read returns n bytes
	src := strings.NewReader("test")
	var buf bytes.Buffer
	tee := TeeReader(src, &buf)
	p := make([]byte, 10)
	n, _ := tee.Read(p)
	if n != 4 {
		t.Errorf("expected n=4, got n=%d", n)
	}
}

func Test8(t *testing.T) {
	// Check EOF
	src := strings.NewReader("hi")
	var buf bytes.Buffer
	tee := TeeReader(src, &buf)
	io.ReadAll(tee)
	_, err := tee.Read(make([]byte, 1))
	if err != io.EOF {
		t.Errorf("expected EOF, got err=%v", err)
	}
}

func Test9(t *testing.T) {
	// Binary data
	binary := []byte{0, 1, 2, 255, 254, 253}
	src := bytes.NewReader(binary)
	var buf bytes.Buffer
	tee := TeeReader(src, &buf)
	data, _ := io.ReadAll(tee)
	if !bytes.Equal(data, binary) || !bytes.Equal(buf.Bytes(), binary) {
		t.Errorf("binary data mismatch")
	}
}

func Test10(t *testing.T) {
	// Chained tee readers
	src := strings.NewReader("chain")
	var buf1, buf2 bytes.Buffer
	tee1 := TeeReader(src, &buf1)
	tee2 := TeeReader(tee1, &buf2)
	io.ReadAll(tee2)
	if buf1.String() != "chain" || buf2.String() != "chain" {
		t.Errorf("expected both buffers 'chain', got buf1=%s, buf2=%s", buf1.String(), buf2.String())
	}
}`,
	hint1: `Create a custom struct that holds both the reader and writer, then implement the Read() method that reads from r and writes to w.`,
	hint2: `In your Read() implementation, first call r.Read(p), then if n > 0, call w.Write(p[:n]). Make sure to write only what was successfully read and handle both read and write errors.`,
	whyItMatters: `TeeReader is essential for observability and monitoring in production systems, enabling you to inspect data streams without modifying the original flow.

**Why Stream Duplication:**
- **Debugging:** Capture request/response data without affecting the main flow
- **Logging:** Record all network traffic for audit trails
- **Hashing:** Compute checksums while processing data
- **Metrics:** Track bytes transferred in real-time
- **Testing:** Verify data integrity during transfers

**Production Patterns:**

\`\`\`go
// HTTP request/response logging
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        var reqLog bytes.Buffer
        teeBody := TeeReader(r.Body, &reqLog)
        r.Body = io.NopCloser(teeBody)

        next.ServeHTTP(w, r)

        log.Printf("Request body: %s", reqLog.String())
    })
}

// File upload with progress and hash verification
func UploadWithVerification(src io.Reader, dst io.Writer) (string, error) {
    hash := sha256.New()
    counter := &ByteCounter{}

    // Chain: src -> hash -> counter -> dst
    tee1 := TeeReader(src, hash)
    tee2 := TeeReader(tee1, counter)

    _, err := io.Copy(dst, tee2)
    if err != nil {
        return "", err
    }

    return fmt.Sprintf("%x", hash.Sum(nil)), nil
}

// S3 upload with local backup
func UploadToS3WithBackup(data io.Reader, bucket, key string) error {
    backup, _ := os.Create("backup/" + key)
    defer backup.Close()

    tee := TeeReader(data, backup)

    return s3.Upload(bucket, key, tee) // Upload to S3 AND save locally
}

// Database query result logging
func LogQuery(rows *sql.Rows) *sql.Rows {
    var queryLog bytes.Buffer
    tee := TeeReader(rows, &queryLog)

    go func() {
        time.Sleep(100 * time.Millisecond)
        log.Printf("Query results: %s", queryLog.String())
    }()

    return tee
}

// Network packet capture
func CaptureTraffic(conn net.Conn, pcap io.Writer) net.Conn {
    return &struct {
        io.Reader
        io.Writer
        net.Conn
    }{
        Reader: TeeReader(conn, pcap),
        Writer: conn,
        Conn:   conn,
    }
}

// API response caching
func CachedAPICall(url string, cache io.Writer) ([]byte, error) {
    resp, err := http.Get(url)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    tee := TeeReader(resp.Body, cache)
    return io.ReadAll(tee) // Read response AND cache it
}

// Multi-destination streaming
func StreamToMultiple(src io.Reader, destinations ...io.Writer) error {
    reader := src
    for _, dst := range destinations {
        reader = TeeReader(reader, dst)
    }

    _, err := io.Copy(io.Discard, reader)
    return err
}
\`\`\`

**Real-World Scenarios:**
- **API Gateways:** Log all requests/responses for debugging
- **CDN/Proxy:** Cache content while serving it
- **Data Pipelines:** Monitor data flowing through ETL processes
- **File Transfers:** Compute checksums during upload/download
- **Security:** Record all network traffic for intrusion detection
- **Metrics:** Track bandwidth usage in real-time

**Common Pitfalls:**
- Not checking if n > 0 before writing (might write garbage)
- Writing p instead of p[:n] (writing more than was read)
- Not handling write errors properly
- Ignoring that Write might not write all bytes

Without TeeReader, you'd need to buffer entire streams in memory or make multiple passes over data, leading to memory issues and performance degradation.`,
	order: 1,
	translations: {
		ru: {
			title: 'TeeReader: чтение и копирование данных',
			description: `Реализуйте **TeeReader**, который возвращает Reader, записывающий в w то, что читает из r, подобно io.TeeReader, но без его использования.

**Требования:**
1. Создайте функцию \`TeeReader(r io.Reader, w io.Writer) io.Reader\`
2. Верните кастомный reader, реализующий интерфейс io.Reader
3. При каждом вызове Read() читайте из оригинального reader r
4. Пишите прочитанные данные в writer w перед возвратом
5. Корректно обрабатывайте частичные записи (если writer не может записать все байты)
6. Верните прочитанные данные вместе с любыми ошибками
7. Сохраните ошибки чтения из исходного reader
8. Сохраните ошибки записи из целевого writer

**Пример:**
\`\`\`go
// Захват данных во время чтения
var buf bytes.Buffer
reader := TeeReader(file, &buf)

// Чтение из файла, автоматически копируется в buf
data := make([]byte, 100)
n, err := reader.Read(data)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Прочитано %d байт\\n", n)
fmt.Printf("Буфер содержит: %s\\n", buf.String())

// Логирование API ответов во время чтения
logFile, _ := os.Create("api.log")
tee := TeeReader(resp.Body, logFile)
body, _ := io.ReadAll(tee) // Body прочитан И залогирован

// Хеширование файла во время чтения
hash := sha256.New()
teeReader := TeeReader(file, hash)
io.Copy(dst, teeReader) // Файл скопирован И хеширован
\`\`\`

**Ограничения:**
- Не должен использовать io.TeeReader
- Должен реализовать интерфейс io.Reader
- Должен корректно обрабатывать ошибки чтения и записи
- Должен писать ровно то, что было успешно прочитано
- Вернуть оригинальные прочитанные данные даже если запись не удалась`,
			hint1: `Создайте кастомный struct, содержащий reader и writer, затем реализуйте метод Read(), который читает из r и пишет в w.`,
			hint2: `В вашей реализации Read(), сначала вызовите r.Read(p), затем если n > 0, вызовите w.Write(p[:n]). Убедитесь что пишете только то что было успешно прочитано и обрабатывайте ошибки чтения и записи.`,
			whyItMatters: `TeeReader критичен для наблюдаемости и мониторинга в production системах, позволяя инспектировать потоки данных без изменения основного потока.

**Почему дублирование потоков:**
- **Отладка:** Захват данных запросов/ответов без влияния на основной поток
- **Логирование:** Запись всего сетевого трафика для аудита
- **Хеширование:** Вычисление контрольных сумм во время обработки данных
- **Метрики:** Отслеживание переданных байт в реальном времени
- **Тестирование:** Проверка целостности данных во время передачи

**Production паттерны:**

\`\`\`go
// Логирование HTTP запросов/ответов
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        var reqLog bytes.Buffer
        teeBody := TeeReader(r.Body, &reqLog)
        r.Body = io.NopCloser(teeBody)

        next.ServeHTTP(w, r)

        log.Printf("Тело запроса: %s", reqLog.String())
    })
}

// Загрузка файла с прогрессом и проверкой хеша
func UploadWithVerification(src io.Reader, dst io.Writer) (string, error) {
    hash := sha256.New()
    counter := &ByteCounter{}

    // Цепочка: src -> hash -> counter -> dst
    tee1 := TeeReader(src, hash)
    tee2 := TeeReader(tee1, counter)

    _, err := io.Copy(dst, tee2)
    if err != nil {
        return "", err
    }

    return fmt.Sprintf("%x", hash.Sum(nil)), nil
}

// Загрузка в S3 с локальной резервной копией
func UploadToS3WithBackup(data io.Reader, bucket, key string) error {
    backup, _ := os.Create("backup/" + key)
    defer backup.Close()

    tee := TeeReader(data, backup)

    return s3.Upload(bucket, key, tee) // Загрузить в S3 И сохранить локально
}

// Логирование результатов запросов к БД
func LogQuery(rows *sql.Rows) *sql.Rows {
    var queryLog bytes.Buffer
    tee := TeeReader(rows, &queryLog)

    go func() {
        time.Sleep(100 * time.Millisecond)
        log.Printf("Результаты запроса: %s", queryLog.String())
    }()

    return tee
}

// Захват сетевых пакетов
func CaptureTraffic(conn net.Conn, pcap io.Writer) net.Conn {
    return &struct {
        io.Reader
        io.Writer
        net.Conn
    }{
        Reader: TeeReader(conn, pcap),
        Writer: conn,
        Conn:   conn,
    }
}

// Кеширование API ответов
func CachedAPICall(url string, cache io.Writer) ([]byte, error) {
    resp, err := http.Get(url)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    tee := TeeReader(resp.Body, cache)
    return io.ReadAll(tee) // Прочитать ответ И закешировать его
}

// Стриминг в несколько мест назначения
func StreamToMultiple(src io.Reader, destinations ...io.Writer) error {
    reader := src
    for _, dst := range destinations {
        reader = TeeReader(reader, dst)
    }

    _, err := io.Copy(io.Discard, reader)
    return err
}
\`\`\`

**Реальные сценарии:**
- **API Gateway:** Логирование всех запросов/ответов для отладки
- **CDN/Прокси:** Кеширование контента во время отдачи
- **Конвейеры данных:** Мониторинг данных проходящих через ETL процессы
- **Передача файлов:** Вычисление контрольных сумм во время загрузки/скачивания
- **Безопасность:** Запись всего сетевого трафика для обнаружения вторжений
- **Метрики:** Отслеживание использования пропускной способности в реальном времени

**Распространённые ошибки:**
- Не проверять n > 0 перед записью (может писать мусор)
- Писать p вместо p[:n] (писать больше чем прочитано)
- Не обрабатывать ошибки записи корректно
- Игнорировать что Write может не записать все байты

Без TeeReader вам пришлось бы буферизовать целые потоки в памяти или делать несколько проходов по данным, что приводит к проблемам с памятью и деградации производительности.`,
			solutionCode: `package interfaces

import (
	"io"
)

// teeReader реализует io.Reader и пишет в w то что читает из r
type teeReader struct {
	r io.Reader                                         // Исходный reader
	w io.Writer                                         // Целевой writer
}

// Read реализует интерфейс io.Reader
func (t *teeReader) Read(p []byte) (n int, err error) {
	n, err = t.r.Read(p)                                // Читать из исходного reader
	if n > 0 {                                          // Писать только если что-то прочитано
		if n, err := t.w.Write(p[:n]); err != nil {    // Писать ровно то что прочитано
			return n, err                               // Вернуть ошибку записи если произошла
		}
	}
	return                                              // Вернуть результат чтения (n, err)
}

func TeeReader(r io.Reader, w io.Writer) io.Reader {
	return &teeReader{r: r, w: w}                       // Вернуть кастомный tee reader
}`
		},
		uz: {
			title: 'TeeReader: ma\'lumotlarni o\'qish va nusxalash',
			description: `r dan o'qiganini w ga yozadigan Reader qaytaradigan **TeeReader**ni amalga oshiring, io.TeeReader ga o'xshash lekin uni ishlatmasdan.

**Talablar:**
1. \`TeeReader(r io.Reader, w io.Writer) io.Reader\` funksiyasini yarating
2. io.Reader interfeysini amalga oshiradigan maxsus reader qaytaring
3. Har bir Read() chaqiruvida asl reader r dan o'qing
4. O'qilgan ma'lumotlarni qaytarishdan oldin writer w ga yozing
5. Qisman yozishlarni to'g'ri ishlang (agar writer barcha baytlarni yoza olmasa)
6. O'qilgan ma'lumotlarni har qanday xatolar bilan qaytaring
7. Manba reader dan o'qish xatolarini saqlang
8. Maqsad writer dan yozish xatolarini saqlang

**Misol:**
\`\`\`go
// O'qish paytida ma'lumotlarni yozib olish
var buf bytes.Buffer
reader := TeeReader(file, &buf)

// Fayldan o'qish, avtomatik ravishda buf ga ko'chiriladi
data := make([]byte, 100)
n, err := reader.Read(data)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("O'qildi %d bayt\\n", n)
fmt.Printf("Bufer saqlaydi: %s\\n", buf.String())

// O'qish paytida API javoblarni loglash
logFile, _ := os.Create("api.log")
tee := TeeReader(resp.Body, logFile)
body, _ := io.ReadAll(tee) // Body o'qilgan VA logga yozilgan

// O'qish paytida faylni xeshlash
hash := sha256.New()
teeReader := TeeReader(file, hash)
io.Copy(dst, teeReader) // Fayl ko'chirilgan VA xeshlangan
\`\`\`

**Cheklovlar:**
- io.TeeReader ishlatmasligi kerak
- io.Reader interfeysini amalga oshirishi kerak
- O'qish va yozish xatolarini to'g'ri ishlashi kerak
- Muvaffaqiyatli o'qilgan narsani aniq yozishi kerak
- Yozish muvaffaqiyatsiz bo'lsa ham asl o'qilgan ma'lumotlarni qaytarish`,
			hint1: `Reader va writer ni saqlaydigan maxsus struct yarating, keyin r dan o'qiydigan va w ga yozadigan Read() metodini amalga oshiring.`,
			hint2: `Read() amalga oshirishda, avval r.Read(p) ni chaqiring, keyin agar n > 0 bo'lsa, w.Write(p[:n]) ni chaqiring. Faqat muvaffaqiyatli o'qilgan narsani yozganingizga ishonch hosil qiling va o'qish va yozish xatolarini ishlang.`,
			whyItMatters: `TeeReader production tizimlarda kuzatuvchanlik va monitoring uchun muhim, asosiy oqimni o'zgartirmasdan ma'lumot oqimlarini tekshirish imkonini beradi.

**Nega oqim dublikatlash:**
- **Debugging:** Asosiy oqimga ta'sir qilmasdan so'rov/javob ma'lumotlarini yozib olish
- **Loglash:** Audit izi uchun barcha tarmoq trafigini yozib olish
- **Xeshlash:** Ma'lumotlarni qayta ishlash vaqtida nazorat yig'indisini hisoblash
- **Metrikalar:** Real vaqtda o'tkazilgan baytlarni kuzatish
- **Testlash:** O'tkazishlar vaqtida ma'lumotlar yaxlitligini tekshirish

**Production patternlari:**

\`\`\`go
// HTTP so'rov/javoblarni loglash
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        var reqLog bytes.Buffer
        teeBody := TeeReader(r.Body, &reqLog)
        r.Body = io.NopCloser(teeBody)

        next.ServeHTTP(w, r)

        log.Printf("So'rov tanasi: %s", reqLog.String())
    })
}

// Progress va hash tekshiruv bilan fayl yuklash
func UploadWithVerification(src io.Reader, dst io.Writer) (string, error) {
    hash := sha256.New()
    counter := &ByteCounter{}

    // Zanjir: src -> hash -> counter -> dst
    tee1 := TeeReader(src, hash)
    tee2 := TeeReader(tee1, counter)

    _, err := io.Copy(dst, tee2)
    if err != nil {
        return "", err
    }

    return fmt.Sprintf("%x", hash.Sum(nil)), nil
}

// Lokal zaxira nusxa bilan S3 ga yuklash
func UploadToS3WithBackup(data io.Reader, bucket, key string) error {
    backup, _ := os.Create("backup/" + key)
    defer backup.Close()

    tee := TeeReader(data, backup)

    return s3.Upload(bucket, key, tee) // S3 ga yuklash VA lokal saqlash
}

// Ma'lumotlar bazasi so'rov natijalarini loglash
func LogQuery(rows *sql.Rows) *sql.Rows {
    var queryLog bytes.Buffer
    tee := TeeReader(rows, &queryLog)

    go func() {
        time.Sleep(100 * time.Millisecond)
        log.Printf("So'rov natijalari: %s", queryLog.String())
    }()

    return tee
}

// Tarmoq paketlarini yozib olish
func CaptureTraffic(conn net.Conn, pcap io.Writer) net.Conn {
    return &struct {
        io.Reader
        io.Writer
        net.Conn
    }{
        Reader: TeeReader(conn, pcap),
        Writer: conn,
        Conn:   conn,
    }
}

// API javoblarni keshlash
func CachedAPICall(url string, cache io.Writer) ([]byte, error) {
    resp, err := http.Get(url)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    tee := TeeReader(resp.Body, cache)
    return io.ReadAll(tee) // Javobni o'qish VA uni keshlash
}

// Bir nechta manzillarga streaming
func StreamToMultiple(src io.Reader, destinations ...io.Writer) error {
    reader := src
    for _, dst := range destinations {
        reader = TeeReader(reader, dst)
    }

    _, err := io.Copy(io.Discard, reader)
    return err
}
\`\`\`

**Haqiqiy stsenariylar:**
- **API Gateway:** Debugging uchun barcha so'rov/javoblarni loglash
- **CDN/Proksi:** Kontentni xizmat qilish vaqtida keshlash
- **Ma'lumot konveyerlari:** ETL jarayonlari orqali o'tadigan ma'lumotlarni monitoring qilish
- **Fayl o'tkazmalari:** Yuklash/yuklab olish vaqtida nazorat yig'indilarini hisoblash
- **Xavfsizlik:** Kirib kelishlarni aniqlash uchun barcha tarmoq trafigini yozib olish
- **Metrikalar:** Real vaqtda tarmoq kengligi ishlatilishini kuzatish

**Keng tarqalgan xatolar:**
- Yozishdan oldin n > 0 ni tekshirmaslik (axlat yozilishi mumkin)
- p[:n] o'rniga p ni yozish (o'qilgandan ko'ra ko'proq yozish)
- Yozish xatolarini to'g'ri ishlamaslik
- Write barcha baytlarni yozmasligi mumkinligini e'tiborsiz qoldirish

TeeReader bo'lmasa, siz butun oqimlarni xotirada buferlashtirish yoki ma'lumotlar bo'ylab bir necha marta o'tishingiz kerak bo'ladi, bu xotira muammolari va ish unumdorligini pasayishiga olib keladi.`,
			solutionCode: `package interfaces

import (
	"io"
)

// teeReader io.Reader ni amalga oshiradi va r dan o'qiganini w ga yozadi
type teeReader struct {
	r io.Reader                                         // Manba reader
	w io.Writer                                         // Maqsad writer
}

// Read io.Reader interfeysini amalga oshiradi
func (t *teeReader) Read(p []byte) (n int, err error) {
	n, err = t.r.Read(p)                                // Manba reader dan o'qish
	if n > 0 {                                          // Faqat biror narsa o'qilgan bo'lsa yozish
		if n, err := t.w.Write(p[:n]); err != nil {    // O'qilgan narsani aniq yozish
			return n, err                               // Yozish xatosi yuz bergan bo'lsa qaytarish
		}
	}
	return                                              // O'qish natijasini qaytarish (n, err)
}

func TeeReader(r io.Reader, w io.Writer) io.Reader {
	return &teeReader{r: r, w: w}                       // Maxsus tee reader ni qaytarish
}`
		}
	}
};

export default task;
