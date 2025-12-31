import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-fundamentals-multi-writer',
	title: 'MultiWriter: Write to Multiple Destinations',
	difficulty: 'medium',
	tags: ['go', 'io', 'interfaces', 'concurrency'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement **MultiWriter** that creates a writer that duplicates its writes to all the provided writers, similar to io.MultiWriter but without using it.

**Requirements:**
1. Create function \`MultiWriter(writers ...io.Writer) io.Writer\`
2. Return a writer that implements io.Writer interface
3. Accept variable number of writers (variadic parameter)
4. On Write() call, write to ALL provided writers
5. Return early if no writers are provided (write to nothing)
6. If any write fails, stop and return the error
7. Return the number of bytes written and any error
8. Ensure all writers receive exactly the same data

**Example:**
\`\`\`go
// Write to file and stdout simultaneously
file, _ := os.Create("output.txt")
multi := MultiWriter(file, os.Stdout)

n, err := multi.Write([]byte("Hello World"))
// "Hello World" written to BOTH file and stdout

// Write to multiple log files
accessLog, _ := os.Create("access.log")
errorLog, _ := os.Create("error.log")
auditLog, _ := os.Create("audit.log")

logger := MultiWriter(accessLog, errorLog, auditLog)
logger.Write([]byte("User logged in"))
// Message written to ALL three log files

// Backup to multiple destinations
primary := s3.NewWriter("primary/backup.tar")
secondary := s3.NewWriter("secondary/backup.tar")
local, _ := os.Create("local/backup.tar")

backup := MultiWriter(primary, secondary, local)
io.Copy(backup, dataSource) // Data written to all 3 destinations
\`\`\`

**Constraints:**
- Must not use io.MultiWriter
- Must implement io.Writer interface
- Must handle zero writers gracefully
- Must write to writers in order
- Stop on first error and return it
- All successful writes must be to the same byte slice`,
	initialCode: `package interfaces

import (
	"io"
)

// TODO: Implement MultiWriter
func MultiWriter(writers ...io.Writer) io.Writer {
	// TODO: Implement
}`,
	solutionCode: `package interfaces

import (
	"io"
)

// multiWriter implements io.Writer and writes to multiple writers
type multiWriter struct {
	writers []io.Writer                                 // Slice of destination writers
}

// Write implements io.Writer interface
func (m *multiWriter) Write(p []byte) (n int, err error) {
	for _, w := range m.writers {                       // Iterate through all writers
		n, err = w.Write(p)                             // Write to current writer
		if err != nil {                                 // Check for write errors
			return                                      // Return immediately on error
		}
		if n != len(p) {                                // Verify all bytes were written
			err = io.ErrShortWrite                      // Set short write error
			return                                      // Return with error
		}
	}
	return len(p), nil                                  // All writes successful
}

func MultiWriter(writers ...io.Writer) io.Writer {
	w := make([]io.Writer, 0, len(writers))             // Create slice with capacity
	for _, writer := range writers {                    // Filter out nil writers
		if writer != nil {                              // Only include non-nil writers
			w = append(w, writer)                       // Add to our slice
		}
	}
	return &multiWriter{writers: w}                     // Return custom multi writer
}`,
	testCode: `package interfaces

import (
	"bytes"
	"testing"
)

func Test1(t *testing.T) {
	// Write to two buffers
	var buf1, buf2 bytes.Buffer
	multi := MultiWriter(&buf1, &buf2)
	n, err := multi.Write([]byte("Hello"))
	if err != nil || n != 5 || buf1.String() != "Hello" || buf2.String() != "Hello" {
		t.Errorf("expected 'Hello' in both, got err=%v, n=%d, buf1=%s, buf2=%s", err, n, buf1.String(), buf2.String())
	}
}

func Test2(t *testing.T) {
	// Write to three buffers
	var buf1, buf2, buf3 bytes.Buffer
	multi := MultiWriter(&buf1, &buf2, &buf3)
	multi.Write([]byte("test"))
	if buf1.String() != "test" || buf2.String() != "test" || buf3.String() != "test" {
		t.Errorf("expected 'test' in all buffers")
	}
}

func Test3(t *testing.T) {
	// Empty writers
	multi := MultiWriter()
	n, err := multi.Write([]byte("data"))
	if err != nil || n != 4 {
		t.Errorf("expected no error, got err=%v, n=%d", err, n)
	}
}

func Test4(t *testing.T) {
	// Single writer
	var buf bytes.Buffer
	multi := MultiWriter(&buf)
	multi.Write([]byte("single"))
	if buf.String() != "single" {
		t.Errorf("expected 'single', got %s", buf.String())
	}
}

func Test5(t *testing.T) {
	// Empty write
	var buf bytes.Buffer
	multi := MultiWriter(&buf)
	n, err := multi.Write([]byte{})
	if err != nil || n != 0 {
		t.Errorf("expected 0 bytes, got n=%d, err=%v", n, err)
	}
}

func Test6(t *testing.T) {
	// Multiple writes
	var buf1, buf2 bytes.Buffer
	multi := MultiWriter(&buf1, &buf2)
	multi.Write([]byte("a"))
	multi.Write([]byte("b"))
	multi.Write([]byte("c"))
	if buf1.String() != "abc" || buf2.String() != "abc" {
		t.Errorf("expected 'abc', got buf1=%s, buf2=%s", buf1.String(), buf2.String())
	}
}

func Test7(t *testing.T) {
	// Large write
	var buf1, buf2 bytes.Buffer
	multi := MultiWriter(&buf1, &buf2)
	large := make([]byte, 100000)
	for i := range large {
		large[i] = 'x'
	}
	n, err := multi.Write(large)
	if err != nil || n != 100000 {
		t.Errorf("expected 100000 bytes, got n=%d, err=%v", n, err)
	}
}

func Test8(t *testing.T) {
	// Binary data
	var buf1, buf2 bytes.Buffer
	multi := MultiWriter(&buf1, &buf2)
	binary := []byte{0, 255, 1, 254}
	multi.Write(binary)
	if !bytes.Equal(buf1.Bytes(), binary) || !bytes.Equal(buf2.Bytes(), binary) {
		t.Errorf("binary data mismatch")
	}
}

func Test9(t *testing.T) {
	// Nil writers filtered
	var buf bytes.Buffer
	multi := MultiWriter(nil, &buf, nil)
	multi.Write([]byte("test"))
	if buf.String() != "test" {
		t.Errorf("expected 'test', got %s", buf.String())
	}
}

func Test10(t *testing.T) {
	// Returns correct byte count
	var buf1, buf2 bytes.Buffer
	multi := MultiWriter(&buf1, &buf2)
	data := []byte("twelve chars")
	n, _ := multi.Write(data)
	if n != len(data) {
		t.Errorf("expected %d, got %d", len(data), n)
	}
}`,
	hint1: `Create a custom struct that holds a slice of writers, then implement Write() that iterates through each writer and calls Write() on each one.`,
	hint2: `In your Write() implementation, loop through all writers and write p to each. Check for errors after each write and return immediately if any error occurs. Make sure to verify that n == len(p) for each write.`,
	whyItMatters: `MultiWriter is crucial for data redundancy, logging, and monitoring in distributed systems where you need to ensure data reaches multiple destinations.

**Why Multiple Destinations:**
- **Redundancy:** Write to primary and backup simultaneously
- **Observability:** Send data to both storage and monitoring systems
- **Compliance:** Write to multiple audit logs for regulatory requirements
- **Performance:** Distribute load across multiple storage backends
- **Reliability:** Ensure data survives single-point failures

**Production Patterns:**

\`\`\`go
// Redundant logging system
func SetupLogging() io.Writer {
    stdout := os.Stdout
    file, _ := os.OpenFile("app.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
    syslog, _ := syslog.New(syslog.LOG_INFO, "myapp")

    return MultiWriter(stdout, file, syslog) // Log to 3 destinations
}

// Database backup with multiple destinations
func BackupDatabase(db *sql.DB) error {
    s3Primary := s3.NewWriter("us-east-1", "backup.sql")
    s3Secondary := s3.NewWriter("eu-west-1", "backup.sql")
    localDisk, _ := os.Create("/backup/db.sql")

    backup := MultiWriter(s3Primary, s3Secondary, localDisk)

    rows, _ := db.Query("SELECT * FROM users")
    return StreamResults(rows, backup) // Written to all 3 locations
}

// Real-time data replication
func ReplicateData(source io.Reader, replicas []string) error {
    writers := make([]io.Writer, len(replicas))
    for i, replica := range replicas {
        conn, err := net.Dial("tcp", replica)
        if err != nil {
            return err
        }
        defer conn.Close()
        writers[i] = conn
    }

    multi := MultiWriter(writers...)
    _, err := io.Copy(multi, source) // Data sent to all replicas
    return err
}

// Metrics and logging combined
func InstrumentedWriter(dst io.Writer, metrics MetricsCollector) io.Writer {
    metricWriter := &MetricWriter{collector: metrics}
    return MultiWriter(dst, metricWriter) // Write data AND collect metrics
}

// File upload with hash verification
func UploadWithHash(src io.Reader, dst io.Writer) (string, error) {
    hash := sha256.New()
    multi := MultiWriter(dst, hash)

    _, err := io.Copy(multi, src)
    if err != nil {
        return "", err
    }

    return fmt.Sprintf("%x", hash.Sum(nil)), nil
}

// CDN edge caching
func CacheToEdges(content io.Reader, edges []Edge) error {
    writers := make([]io.Writer, len(edges))
    for i, edge := range edges {
        writers[i] = edge.Writer()
    }

    multi := MultiWriter(writers...)
    _, err := io.Copy(multi, content) // Content pushed to all edge nodes
    return err
}

// Audit logging with encryption
func SecureAuditLog(message []byte) error {
    plainLog, _ := os.OpenFile("audit.log", os.O_APPEND|os.O_WRONLY, 0600)
    encryptedLog := EncryptWriter(plainLog)
    remoteAudit := RemoteAuditService()

    multi := MultiWriter(plainLog, encryptedLog, remoteAudit)
    _, err := multi.Write(message) // Written to all audit destinations
    return err
}

// Split writes for testing
func TestMultiWriter(t *testing.T) {
    var buf1, buf2, buf3 bytes.Buffer
    multi := MultiWriter(&buf1, &buf2, &buf3)

    data := []byte("test data")
    n, err := multi.Write(data)

    assert.NoError(t, err)
    assert.Equal(t, len(data), n)
    assert.Equal(t, data, buf1.Bytes())
    assert.Equal(t, data, buf2.Bytes())
    assert.Equal(t, data, buf3.Bytes())
}
\`\`\`

**Real-World Scenarios:**
- **Log Aggregation:** Write logs to local disk, syslog, and cloud simultaneously
- **Backup Systems:** Write backups to multiple geographic locations
- **Streaming Platforms:** Distribute video streams to multiple CDN edges
- **Database Replication:** Write to primary and replica databases
- **Monitoring:** Send metrics to multiple monitoring systems
- **Compliance:** Maintain multiple audit trails for regulations

**Common Pitfalls:**
- Not checking for nil writers
- Continuing to write after first error (should stop immediately)
- Not verifying all bytes were written (n == len(p))
- Returning wrong byte count when error occurs
- Not handling empty writer list

Without MultiWriter, you'd need to manually loop through destinations and handle errors, leading to code duplication and potential inconsistencies in error handling.`,
	order: 2,
	translations: {
		ru: {
			title: 'MultiWriter: запись в несколько назначений',
			description: `Реализуйте **MultiWriter**, который создаёт writer, дублирующий свои записи во все предоставленные writers, подобно io.MultiWriter, но без его использования.

**Требования:**
1. Создайте функцию \`MultiWriter(writers ...io.Writer) io.Writer\`
2. Верните writer, реализующий интерфейс io.Writer
3. Принимайте переменное количество writers (вариативный параметр)
4. При вызове Write() пишите во ВСЕ предоставленные writers
5. Вернитесь рано если writers не предоставлены (писать некуда)
6. Если любая запись не удалась, остановитесь и верните ошибку
7. Верните количество записанных байт и любую ошибку
8. Убедитесь что все writers получают ровно те же данные

**Пример:**
\`\`\`go
// Писать в файл и stdout одновременно
file, _ := os.Create("output.txt")
multi := MultiWriter(file, os.Stdout)

n, err := multi.Write([]byte("Hello World"))
// "Hello World" записан в файл И stdout

// Писать в несколько файлов логов
accessLog, _ := os.Create("access.log")
errorLog, _ := os.Create("error.log")
auditLog, _ := os.Create("audit.log")

logger := MultiWriter(accessLog, errorLog, auditLog)
logger.Write([]byte("User logged in"))
// Сообщение записано во ВСЕ три файла логов

// Резервное копирование в несколько мест назначения
primary := s3.NewWriter("primary/backup.tar")
secondary := s3.NewWriter("secondary/backup.tar")
local, _ := os.Create("local/backup.tar")

backup := MultiWriter(primary, secondary, local)
io.Copy(backup, dataSource) // Данные записаны во все 3 места назначения
\`\`\`

**Ограничения:**
- Не должен использовать io.MultiWriter
- Должен реализовать интерфейс io.Writer
- Должен корректно обрабатывать нулевое количество writers
- Должен писать в writers по порядку
- Остановиться на первой ошибке и вернуть её
- Все успешные записи должны быть одного и того же byte slice`,
			hint1: `Создайте кастомный struct, содержащий slice writers, затем реализуйте Write(), который итерирует по каждому writer и вызывает Write() на каждом.`,
			hint2: `В вашей реализации Write(), циклом пройдитесь по всем writers и запишите p в каждый. Проверяйте ошибки после каждой записи и возвращайтесь немедленно если любая ошибка произошла. Убедитесь что n == len(p) для каждой записи.`,
			whyItMatters: `MultiWriter критичен для избыточности данных, логирования и мониторинга в распределённых системах где вам нужно гарантировать что данные достигают нескольких мест назначения.

**Почему несколько мест назначения:**
- **Избыточность:** Писать в основное и резервное одновременно
- **Наблюдаемость:** Отправлять данные в хранилище и системы мониторинга
- **Соответствие:** Писать в несколько логов аудита для регуляторных требований
- **Производительность:** Распределять нагрузку по нескольким бэкендам хранилища
- **Надёжность:** Гарантировать что данные переживут отказы одной точки

**Production паттерны:**

\`\`\`go
// Избыточная система логирования
func SetupLogging() io.Writer {
    stdout := os.Stdout
    file, _ := os.OpenFile("app.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
    syslog, _ := syslog.New(syslog.LOG_INFO, "myapp")

    return MultiWriter(stdout, file, syslog) // Логи в 3 места назначения
}

// Резервное копирование БД в несколько мест
func BackupDatabase(db *sql.DB) error {
    s3Primary := s3.NewWriter("us-east-1", "backup.sql")
    s3Secondary := s3.NewWriter("eu-west-1", "backup.sql")
    localDisk, _ := os.Create("/backup/db.sql")

    backup := MultiWriter(s3Primary, s3Secondary, localDisk)

    rows, _ := db.Query("SELECT * FROM users")
    return StreamResults(rows, backup) // Записано во все 3 места
}

// Репликация данных в реальном времени
func ReplicateData(source io.Reader, replicas []string) error {
    writers := make([]io.Writer, len(replicas))
    for i, replica := range replicas {
        conn, err := net.Dial("tcp", replica)
        if err != nil {
            return err
        }
        defer conn.Close()
        writers[i] = conn
    }

    multi := MultiWriter(writers...)
    _, err := io.Copy(multi, source) // Данные отправлены во все реплики
    return err
}

// Комбинация метрик и логирования
func InstrumentedWriter(dst io.Writer, metrics MetricsCollector) io.Writer {
    metricWriter := &MetricWriter{collector: metrics}
    return MultiWriter(dst, metricWriter) // Записать данные И собрать метрики
}

// Загрузка файла с проверкой хеша
func UploadWithHash(src io.Reader, dst io.Writer) (string, error) {
    hash := sha256.New()
    multi := MultiWriter(dst, hash)

    _, err := io.Copy(multi, src)
    if err != nil {
        return "", err
    }

    return fmt.Sprintf("%x", hash.Sum(nil)), nil
}

// Кеширование на edge узлах CDN
func CacheToEdges(content io.Reader, edges []Edge) error {
    writers := make([]io.Writer, len(edges))
    for i, edge := range edges {
        writers[i] = edge.Writer()
    }

    multi := MultiWriter(writers...)
    _, err := io.Copy(multi, content) // Контент отправлен во все edge узлы
    return err
}

// Лог аудита с шифрованием
func SecureAuditLog(message []byte) error {
    plainLog, _ := os.OpenFile("audit.log", os.O_APPEND|os.O_WRONLY, 0600)
    encryptedLog := EncryptWriter(plainLog)
    remoteAudit := RemoteAuditService()

    multi := MultiWriter(plainLog, encryptedLog, remoteAudit)
    _, err := multi.Write(message) // Записано во все места аудита
    return err
}

// Разделение записей для тестирования
func TestMultiWriter(t *testing.T) {
    var buf1, buf2, buf3 bytes.Buffer
    multi := MultiWriter(&buf1, &buf2, &buf3)

    data := []byte("test data")
    n, err := multi.Write(data)

    assert.NoError(t, err)
    assert.Equal(t, len(data), n)
    assert.Equal(t, data, buf1.Bytes())
    assert.Equal(t, data, buf2.Bytes())
    assert.Equal(t, data, buf3.Bytes())
}
\`\`\`

**Реальные сценарии:**
- **Агрегация логов:** Запись логов на локальный диск, syslog и облако одновременно
- **Системы резервного копирования:** Запись резервных копий в несколько географических локаций
- **Стриминг платформы:** Распределение видео потоков на несколько CDN edge узлов
- **Репликация БД:** Запись в основную и реплику БД
- **Мониторинг:** Отправка метрик в несколько систем мониторинга
- **Соответствие:** Поддержка нескольких аудит логов для регулирования

**Распространённые ошибки:**
- Не проверять nil writers
- Продолжать запись после первой ошибки (должны немедленно остановиться)
- Не проверять что все байты были записаны (n == len(p))
- Возвращать неверное количество байт при ошибке
- Не обрабатывать пустой список writers

Без MultiWriter вам пришлось бы вручную циклиться по местам назначения и обрабатывать ошибки, что приводит к дублированию кода и потенциальным несоответствиям в обработке ошибок.`,
			solutionCode: `package interfaces

import (
	"io"
)

// multiWriter реализует io.Writer и пишет в несколько writers
type multiWriter struct {
	writers []io.Writer                                 // Slice целевых writers
}

// Write реализует интерфейс io.Writer
func (m *multiWriter) Write(p []byte) (n int, err error) {
	for _, w := range m.writers {                       // Итерация по всем writers
		n, err = w.Write(p)                             // Писать в текущий writer
		if err != nil {                                 // Проверить ошибки записи
			return                                      // Вернуться немедленно при ошибке
		}
		if n != len(p) {                                // Проверить все байты были записаны
			err = io.ErrShortWrite                      // Установить ошибку короткой записи
			return                                      // Вернуться с ошибкой
		}
	}
	return len(p), nil                                  // Все записи успешны
}

func MultiWriter(writers ...io.Writer) io.Writer {
	w := make([]io.Writer, 0, len(writers))             // Создать slice с capacity
	for _, writer := range writers {                    // Фильтровать nil writers
		if writer != nil {                              // Только включать не-nil writers
			w = append(w, writer)                       // Добавить в наш slice
		}
	}
	return &multiWriter{writers: w}                     // Вернуть кастомный multi writer
}`
		},
		uz: {
			title: 'MultiWriter: bir necha manzillarga yozish',
			description: `Barcha taqdim etilgan writerlar ga yozishni dublikatlaydigan writer yaratadigan **MultiWriter**ni amalga oshiring, io.MultiWriter ga o'xshash lekin uni ishlatmasdan.

**Talablar:**
1. \`MultiWriter(writers ...io.Writer) io.Writer\` funksiyasini yarating
2. io.Writer interfeysini amalga oshiradigan writer qaytaring
3. O'zgaruvchan sonli writerlarni qabul qiling (variadik parametr)
4. Write() chaqiruvida BARCHA taqdim etilgan writerlar ga yozing
5. Agar writerlar taqdim etilmagan bo'lsa erta qayting (yozish uchun hech narsa yo'q)
6. Agar biron yozish muvaffaqiyatsiz bo'lsa, to'xtang va xatoni qaytaring
7. Yozilgan baytlar sonini va har qanday xatoni qaytaring
8. Barcha writerlar aynan bir xil ma'lumotlarni olishiga ishonch hosil qiling

**Misol:**
\`\`\`go
// Fayl va stdout ga bir vaqtning o'zida yozish
file, _ := os.Create("output.txt")
multi := MultiWriter(file, os.Stdout)

n, err := multi.Write([]byte("Hello World"))
// "Hello World" fayl VA stdout ga yozilgan

// Bir nechta log fayllariga yozish
accessLog, _ := os.Create("access.log")
errorLog, _ := os.Create("error.log")
auditLog, _ := os.Create("audit.log")

logger := MultiWriter(accessLog, errorLog, auditLog)
logger.Write([]byte("User logged in"))
// Xabar BARCHA uchta log fayllariga yozilgan

// Bir nechta manzillarga zaxira nusxa
primary := s3.NewWriter("primary/backup.tar")
secondary := s3.NewWriter("secondary/backup.tar")
local, _ := os.Create("local/backup.tar")

backup := MultiWriter(primary, secondary, local)
io.Copy(backup, dataSource) // Ma'lumotlar barcha 3 ta manzilga yozilgan
\`\`\`

**Cheklovlar:**
- io.MultiWriter ishlatmasligi kerak
- io.Writer interfeysini amalga oshirishi kerak
- Nol writerlarni yaxshi ishlashi kerak
- Writerlar ga tartibda yozishi kerak
- Birinchi xatoda to'xtab uni qaytarish
- Barcha muvaffaqiyatli yozishlar bir xil byte slice bo'lishi kerak`,
			hint1: `Writerlar slice ni saqlaydigan maxsus struct yarating, keyin har bir writer bo'ylab takrorlanadigan va har birida Write() ni chaqiradigan Write() ni amalga oshiring.`,
			hint2: `Write() amalga oshirishda, barcha writerlar bo'ylab tsikl qiling va har biriga p ni yozing. Har bir yozishdan keyin xatolarni tekshiring va agar biron xato yuz bersa darhol qayting. Har bir yozish uchun n == len(p) ekanligiga ishonch hosil qiling.`,
			whyItMatters: `MultiWriter tarqatilgan tizimlarda ma'lumotlar ortiqchaligi, loglash va monitoring uchun muhim, bu yerda siz ma'lumotlar bir nechta manzillarga yetib borishini kafolatlashingiz kerak.

**Nega bir nechta manzillar:**
- **Ortiqcha:** Asosiy va zaxiraga bir vaqtning o'zida yozish
- **Kuzatuvchanlik:** Ma'lumotlarni saqlash va monitoring tizimlariga yuborish
- **Muvofiqlik:** Tartibga solish talablari uchun bir nechta audit loglarini yozish
- **Ish unumdorligi:** Yukni bir nechta saqlash backendlari bo'ylab taqsimlash
- **Ishonchlilik:** Ma'lumotlar yagona nuqta nosozliklaridan omon qolishini ta'minlash

**Production patternlari:**

\`\`\`go
// Ortiqcha loglash tizimi
func SetupLogging() io.Writer {
    stdout := os.Stdout
    file, _ := os.OpenFile("app.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
    syslog, _ := syslog.New(syslog.LOG_INFO, "myapp")

    return MultiWriter(stdout, file, syslog) // 3 ta manzilga loglar
}

// Ma'lumotlar bazasini bir nechta joyga zaxiralash
func BackupDatabase(db *sql.DB) error {
    s3Primary := s3.NewWriter("us-east-1", "backup.sql")
    s3Secondary := s3.NewWriter("eu-west-1", "backup.sql")
    localDisk, _ := os.Create("/backup/db.sql")

    backup := MultiWriter(s3Primary, s3Secondary, localDisk)

    rows, _ := db.Query("SELECT * FROM users")
    return StreamResults(rows, backup) // Barcha 3 ta joyga yozildi
}

// Real vaqt ma'lumotlar replikatsiyasi
func ReplicateData(source io.Reader, replicas []string) error {
    writers := make([]io.Writer, len(replicas))
    for i, replica := range replicas {
        conn, err := net.Dial("tcp", replica)
        if err != nil {
            return err
        }
        defer conn.Close()
        writers[i] = conn
    }

    multi := MultiWriter(writers...)
    _, err := io.Copy(multi, source) // Ma'lumotlar barcha replikalarga yuborildi
    return err
}

// Metrikalar va loglash birlashtirilgan
func InstrumentedWriter(dst io.Writer, metrics MetricsCollector) io.Writer {
    metricWriter := &MetricWriter{collector: metrics}
    return MultiWriter(dst, metricWriter) // Ma'lumot yozish VA metrika yig'ish
}

// Hash tekshiruv bilan fayl yuklash
func UploadWithHash(src io.Reader, dst io.Writer) (string, error) {
    hash := sha256.New()
    multi := MultiWriter(dst, hash)

    _, err := io.Copy(multi, src)
    if err != nil {
        return "", err
    }

    return fmt.Sprintf("%x", hash.Sum(nil)), nil
}

// CDN edge tugunlariga keshlash
func CacheToEdges(content io.Reader, edges []Edge) error {
    writers := make([]io.Writer, len(edges))
    for i, edge := range edges {
        writers[i] = edge.Writer()
    }

    multi := MultiWriter(writers...)
    _, err := io.Copy(multi, content) // Kontent barcha edge tugunlariga yuborildi
    return err
}

// Shifrlash bilan audit log
func SecureAuditLog(message []byte) error {
    plainLog, _ := os.OpenFile("audit.log", os.O_APPEND|os.O_WRONLY, 0600)
    encryptedLog := EncryptWriter(plainLog)
    remoteAudit := RemoteAuditService()

    multi := MultiWriter(plainLog, encryptedLog, remoteAudit)
    _, err := multi.Write(message) // Barcha audit joylariga yozildi
    return err
}

// Test uchun yozishlarni ajratish
func TestMultiWriter(t *testing.T) {
    var buf1, buf2, buf3 bytes.Buffer
    multi := MultiWriter(&buf1, &buf2, &buf3)

    data := []byte("test data")
    n, err := multi.Write(data)

    assert.NoError(t, err)
    assert.Equal(t, len(data), n)
    assert.Equal(t, data, buf1.Bytes())
    assert.Equal(t, data, buf2.Bytes())
    assert.Equal(t, data, buf3.Bytes())
}
\`\`\`

**Haqiqiy stsenariylar:**
- **Log agregatsiya:** Loglarni lokal disk, syslog va bulutga bir vaqtning o'zida yozish
- **Zaxira tizimlari:** Zaxira nusxalarni bir nechta geografik joylarga yozish
- **Streaming platformalar:** Video oqimlarni bir nechta CDN edge tugunlariga tarqatish
- **DB replikatsiya:** Asosiy va replika DBga yozish
- **Monitoring:** Metrikalarni bir nechta monitoring tizimlariga yuborish
- **Muvofiqlik:** Tartibga solish uchun bir nechta audit loglarni saqlash

**Keng tarqalgan xatolar:**
- Nil writerlarni tekshirmaslik
- Birinchi xatodan keyin yozishni davom ettirish (darhol to'xtatish kerak)
- Barcha baytlar yozilganligini tekshirmaslik (n == len(p))
- Xato bo'lganda noto'g'ri bayt sonini qaytarish
- Bo'sh writerlar ro'yxatini ishlamaslik

MultiWriter bo'lmasa, siz qo'lda manzillar bo'ylab tsikl qilishingiz va xatolarni ishlashingiz kerak bo'ladi, bu kod dublikatsiyasiga va xatolarni ishlashda potentsial nomuvofiqliklarga olib keladi.`,
			solutionCode: `package interfaces

import (
	"io"
)

// multiWriter io.Writer ni amalga oshiradi va bir nechta writerlar ga yozadi
type multiWriter struct {
	writers []io.Writer                                 // Maqsad writerlar slice
}

// Write io.Writer interfeysini amalga oshiradi
func (m *multiWriter) Write(p []byte) (n int, err error) {
	for _, w := range m.writers {                       // Barcha writerlar bo'ylab iteratsiya
		n, err = w.Write(p)                             // Joriy writer ga yozish
		if err != nil {                                 // Yozish xatolarini tekshirish
			return                                      // Xato bo'lsa darhol qaytish
		}
		if n != len(p) {                                // Barcha baytlar yozilganligini tekshirish
			err = io.ErrShortWrite                      // Qisqa yozish xatosini o'rnatish
			return                                      // Xato bilan qaytish
		}
	}
	return len(p), nil                                  // Barcha yozishlar muvaffaqiyatli
}

func MultiWriter(writers ...io.Writer) io.Writer {
	w := make([]io.Writer, 0, len(writers))             // Capacity bilan slice yaratish
	for _, writer := range writers {                    // Nil writerlarni filtrlash
		if writer != nil {                              // Faqat nil bo'lmagan writerlarni qo'shish
			w = append(w, writer)                       // Bizning slice ga qo'shish
		}
	}
	return &multiWriter{writers: w}                     // Maxsus multi writer ni qaytarish
}`
		}
	}
};

export default task;
