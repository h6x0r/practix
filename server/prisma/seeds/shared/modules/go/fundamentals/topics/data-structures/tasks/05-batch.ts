import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-fundamentals-batch',
	title: 'Split Slice into Chunks',
	difficulty: 'medium',	tags: ['go', 'data-structures', 'maps/slices/strings', 'generics'],
	estimatedTime: '15-20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Batch** that splits a slice into chunks of size n.

**Requirements:**
1. Create function \`Batch[T any](in []T, n int) [][]T\`
2. Split input slice into chunks of size n
3. Handle invalid batch size (n <= 0) by returning nil
4. Handle empty input slice by returning nil
5. Last chunk may be smaller if input length is not divisible by n
6. Use slice views (no copying of individual elements)
7. Pre-calculate capacity for efficiency
8. Return slice of slices with proper references

**Example:**
\`\`\`go
result := Batch([]int{1, 2, 3, 4, 5}, 2)
// result = [][]int{{1, 2}, {3, 4}, {5}}

result2 := Batch([]string{"a", "b", "c", "d", "e", "f"}, 3)
// result2 = [][]string{{"a", "b", "c"}, {"d", "e", "f"}}

result3 := Batch([]int{1, 2, 3}, 5)
// result3 = [][]int{{1, 2, 3}} (single chunk smaller than n)

result4 := Batch([]int{}, 2)
// result4 = nil
\`\`\`

**Constraints:**
- Must return nil for invalid batch size (n <= 0)
- Must return nil for empty input
- Must not copy individual elements (use slice views)
- Must properly handle partial last chunk
- Should pre-allocate result slice capacity`,
	initialCode: `package datastructures

// TODO: Implement Batch
func Batch[T any](in []T, n int) [][]T {
	// TODO: Implement
}`,
	solutionCode: `package datastructures

func Batch[T any](in []T, n int) [][]T {
	if n <= 0 || len(in) == 0 {                             // Check for invalid batch size or empty input
		return nil                                      // Return nil
	}
	chunks := make([][]T, 0, (len(in)+n-1)/n)              // Pre-allocate with ceiling division
	for start := 0; start < len(in); start += n {          // Iterate in steps of n
		end := start + n                                // Calculate end index
		if end > len(in) {                              // Check if end exceeds slice length
			end = len(in)                           // Adjust to actual end
		}
		chunks = append(chunks, in[start:end])          // Append slice view (no copy)
	}
	return chunks                                           // Return slice of slices
}`,
	testCode: `package datastructures

import (
	"reflect"
	"testing"
)

func Test1(t *testing.T) {
	// Basic batching
	result := Batch([]int{1, 2, 3, 4, 5}, 2)
	expected := [][]int{{1, 2}, {3, 4}, {5}}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test2(t *testing.T) {
	// Even division
	result := Batch([]int{1, 2, 3, 4, 5, 6}, 3)
	expected := [][]int{{1, 2, 3}, {4, 5, 6}}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test3(t *testing.T) {
	// Empty slice
	result := Batch([]int{}, 2)
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func Test4(t *testing.T) {
	// Batch size zero
	result := Batch([]int{1, 2, 3}, 0)
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func Test5(t *testing.T) {
	// Batch size negative
	result := Batch([]int{1, 2, 3}, -1)
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func Test6(t *testing.T) {
	// Batch size larger than slice
	result := Batch([]int{1, 2, 3}, 5)
	expected := [][]int{{1, 2, 3}}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test7(t *testing.T) {
	// Single element
	result := Batch([]int{42}, 2)
	expected := [][]int{{42}}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test8(t *testing.T) {
	// Batch size 1
	result := Batch([]int{1, 2, 3}, 1)
	expected := [][]int{{1}, {2}, {3}}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test9(t *testing.T) {
	// Strings batching
	result := Batch([]string{"a", "b", "c", "d", "e", "f"}, 3)
	expected := [][]string{{"a", "b", "c"}, {"d", "e", "f"}}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test10(t *testing.T) {
	// Batch size equals length
	result := Batch([]int{1, 2, 3}, 3)
	expected := [][]int{{1, 2, 3}}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}`,
	hint1: `Use ceiling division ((len+n-1)/n) to calculate the exact capacity needed for the result slice.`,
			hint2: `Use slice views in[start:end] to create chunks without copying elements - they reference the original data.`,
			whyItMatters: `Batch is essential for processing large datasets in manageable chunks, enabling parallel processing, streaming, pagination, and memory-efficient operations that would otherwise cause resource exhaustion.

**Why Batching:**
- **Resource Management:** Process limited amounts of data at a time
- **Parallel Processing:** Distribute chunks to goroutines for concurrent work
- **Memory Efficiency:** Avoid loading entire dataset into memory
- **Pagination:** Display results in manageable chunks
- **Streaming:** Process data without buffering everything upfront
- **Pipeline Stages:** Feed controlled amounts to next processing stage

**Production Pattern:**
\`\`\`go
// Database bulk insert with batching
func InsertUsersBatch(users []User, batchSize int) error {
    batches := Batch(users, batchSize)

    for _, batch := range batches {
        if err := database.InsertBatch(batch); err != nil {
            return err
        }
    }
    return nil
}

// Parallel processing with worker pool
func ProcessItemsParallel(items []Item, batchSize int) {
    batches := Batch(items, batchSize)
    workerCount := 4

    jobs := make(chan []Item, workerCount)
    go func() {
        for _, batch := range batches {
            jobs <- batch
        }
        close(jobs)
    }()

    for i := 0; i < workerCount; i++ {
        go func() {
            for batch := range jobs {
                processBatch(batch)
            }
        }()
    }
}

// API pagination with fixed page size
type APIResponse struct {
    Items      []interface{} \`json:"items"\`
    TotalPages int           \`json:"total_pages"\`
    PageNumber int           \`json:"page_number"\`
}

func GetPaginatedResults(allResults []interface{}, pageNumber, pageSize int) APIResponse {
    batches := Batch(allResults, pageSize)

    if pageNumber < 1 || pageNumber > len(batches) {
        return APIResponse{Items: []interface{}{}}
    }

    return APIResponse{
        Items:      batches[pageNumber-1],
        TotalPages: len(batches),
        PageNumber: pageNumber,
    }
}

// CSV export with chunking for memory efficiency
func ExportToCSVChunked(records []Record, chunkSize int) error {
    batches := Batch(records, chunkSize)

    for i, batch := range batches {
        file, _ := os.Create(fmt.Sprintf("export_%d.csv", i))
        defer file.Close()

        writer := csv.NewWriter(file)
        for _, record := range batch {
            writer.Write(record.ToCSVRow())
        }
        writer.Flush()
    }
    return nil
}

// Queue message batching for efficient network transport
type MessageBatcher struct {
    messages []Message
    batchSize int
}

func (mb *MessageBatcher) SendBatched() error {
    batches := Batch(mb.messages, mb.batchSize)

    for _, batch := range batches {
        if err := sendNetworkRequest(batch); err != nil {
            return err
        }
    }
    return nil
}

// Search results pagination
func SearchWithPagination(query string, pageSize int) []Page {
    allResults := database.Search(query)
    batches := Batch(allResults, pageSize)

    pages := make([]Page, len(batches))
    for i, batch := range batches {
        pages[i] = Page{
            Number:  i + 1,
            Results: batch,
            HasNext: i < len(batches)-1,
        }
    }
    return pages
}

// Stream processing with backpressure
type DataStream struct {
    data []DataPoint
}

func (ds *DataStream) StreamWithBackpressure(chunkSize int) {
    batches := Batch(ds.data, chunkSize)

    rateLimiter := time.NewTicker(100 * time.Millisecond)
    defer rateLimiter.Stop()

    for _, batch := range batches {
        <-rateLimiter.C
        processStream(batch)
    }
}

// Machine learning dataset batching
type Dataset struct {
    samples []Sample
}

func (ds *Dataset) GetBatches(batchSize int) []*Batch {
    batches := Batch(ds.samples, batchSize)

    result := make([]*Batch, len(batches))
    for i, batch := range batches {
        result[i] = &Batch{
            Samples: batch,
            Index:   i,
        }
    }
    return result
}

// Rate limiting with request batching
func RateLimitedAPICall(ids []string, maxPerRequest int) error {
    batches := Batch(ids, maxPerRequest)
    limiter := time.NewTicker(1 * time.Second)

    for _, batch := range batches {
        <-limiter.C
        if _, err := externalAPI.FetchData(batch); err != nil {
            return err
        }
    }
    return nil
}

// File reading with chunking
func ProcessLargeFile(filePath string, chunkSize int) error {
    lines, _ := readFileLines(filePath)
    batches := Batch(lines, chunkSize)

    for i, batch := range batches {
        if err := processLinesBatch(batch); err != nil {
            return fmt.Errorf("batch %d: %w", i, err)
        }
    }
    return nil
}

// Log aggregation with batching
type LogAggregator struct {
    logs []LogEntry
}

func (la *LogAggregator) AggregateAndSend(batchSize int) error {
    batches := Batch(la.logs, batchSize)

    for _, batch := range batches {
        aggregated := aggregateLogs(batch)
        if err := sendToAnalytics(aggregated); err != nil {
            return err
        }
    }
    return nil
}
\`\`\`

**Real-World Benefits:**
- **Large File Processing:** Process multi-gigabyte files in chunks without OOM
- **Database Operations:** Batch inserts/updates for optimal performance
- **API Rate Limiting:** Respect API rate limits with controlled request batching
- **Web Pagination:** Efficient pagination without loading all results
- **Data Pipeline:** Feed controlled amounts to next stage of processing
- **Parallel Processing:** Distribute work evenly to multiple workers

**Performance Advantages:**
- **Memory:** Only keep current chunk in active memory
- **I/O:** Batch I/O operations reduce syscall overhead
- **Network:** Fewer, larger packets instead of many small ones
- **Database:** Batch operations faster than individual inserts
- **Concurrency:** Better load distribution across workers

**Common Use Cases:**
- CSV/JSON file processing
- Database bulk operations
- API pagination
- Parallel worker processing
- Rate-limited external API calls
- Message queue batching

Without Batch, processing gigabyte-scale datasets would require loading everything into memory, causing OOM crashes and terrible performance.`,	order: 4,
	translations: {
		ru: {
			title: 'Разбиение слайса на чанки',
			description: `Реализуйте **Batch**, который разбивает слайс на чанки размера n.

**Требования:**
1. Создайте функцию \`Batch[T any](in []T, n int) [][]T\`
2. Разбейте входной слайс на чанки размера n
3. Обработайте неправильный размер batch (n <= 0) возвратив nil
4. Обработайте пустой входной слайс возвратив nil
5. Последний чанк может быть меньше если длина входа не делится на n нацело
6. Используйте slice views (без копирования отдельных элементов)
7. Предварительно вычислите capacity для эффективности
8. Верните слайс слайсов с правильными ссылками

**Пример:**
\`\`\`go
result := Batch([]int{1, 2, 3, 4, 5}, 2)
// result = [][]int{{1, 2}, {3, 4}, {5}}

result2 := Batch([]string{"a", "b", "c", "d", "e", "f"}, 3)
// result2 = [][]string{{"a", "b", "c"}, {"d", "e", "f"}}

result3 := Batch([]int{1, 2, 3}, 5)
// result3 = [][]int{{1, 2, 3}} (один чанк меньше чем n)

result4 := Batch([]int{}, 2)
// result4 = nil
\`\`\`

**Ограничения:**
- Должен возвращать nil для неправильного размера batch (n <= 0)
- Должен возвращать nil для пустого входа
- Не должен копировать отдельные элементы (используйте slice views)
- Должен правильно обработать неполный последний чанк
- Должен предварительно выделить capacity результирующего слайса`,
			hint1: `Используйте ceiling division ((len+n-1)/n) для вычисления точного capacity нужного для результирующего слайса.`,
			hint2: `Используйте slice views in[start:end] для создания чанков без копирования элементов - они ссылаются на оригинальные данные.`,
			whyItMatters: `Batch необходим для обработки больших наборов данных управляемыми чанками, включая параллельную обработку, streaming, pagination и memory-efficient операции которые иначе вызвали бы исчерпание ресурсов.

**Почему Batching:**
- **Управление ресурсами:** Обработка ограниченных объемов данных за раз
- **Параллельная обработка:** Распределение чанков на горутины для concurrent work
- **Эффективность памяти:** Избежание загрузки всего набора данных в память
- **Pagination:** Отображение результатов управляемыми чанками
- **Streaming:** Обработка данных без буферизации всего заранее
- **Этапы Pipeline:** Подача контролируемых объемов на следующий этап обработки

**Production Pattern:**
\`\`\`go
// Database bulk insert с batching
func InsertUsersBatch(users []User, batchSize int) error {
    batches := Batch(users, batchSize)

    for _, batch := range batches {
        if err := database.InsertBatch(batch); err != nil {
            return err
        }
    }
    return nil
}

// Параллельная обработка с worker pool
func ProcessItemsParallel(items []Item, batchSize int) {
    batches := Batch(items, batchSize)
    workerCount := 4

    jobs := make(chan []Item, workerCount)
    go func() {
        for _, batch := range batches {
            jobs <- batch
        }
        close(jobs)
    }()

    for i := 0; i < workerCount; i++ {
        go func() {
            for batch := range jobs {
                processBatch(batch)
            }
        }()
    }
}

// API pagination с фиксированным размером страницы
type APIResponse struct {
    Items      []interface{} \`json:"items"\`
    TotalPages int           \`json:"total_pages"\`
    PageNumber int           \`json:"page_number"\`
}

func GetPaginatedResults(allResults []interface{}, pageNumber, pageSize int) APIResponse {
    batches := Batch(allResults, pageSize)

    if pageNumber < 1 || pageNumber > len(batches) {
        return APIResponse{Items: []interface{}{}}
    }

    return APIResponse{
        Items:      batches[pageNumber-1],
        TotalPages: len(batches),
        PageNumber: pageNumber,
    }
}

// CSV экспорт с чанками для эффективности памяти
func ExportToCSVChunked(records []Record, chunkSize int) error {
    batches := Batch(records, chunkSize)

    for i, batch := range batches {
        file, _ := os.Create(fmt.Sprintf("export_%d.csv", i))
        defer file.Close()

        writer := csv.NewWriter(file)
        for _, record := range batch {
            writer.Write(record.ToCSVRow())
        }
        writer.Flush()
    }
    return nil
}

// Queue message batching для эффективного сетевого транспорта
type MessageBatcher struct {
    messages []Message
    batchSize int
}

func (mb *MessageBatcher) SendBatched() error {
    batches := Batch(mb.messages, mb.batchSize)

    for _, batch := range batches {
        if err := sendNetworkRequest(batch); err != nil {
            return err
        }
    }
    return nil
}

// Search results pagination
func SearchWithPagination(query string, pageSize int) []Page {
    allResults := database.Search(query)
    batches := Batch(allResults, pageSize)

    pages := make([]Page, len(batches))
    for i, batch := range batches {
        pages[i] = Page{
            Number:  i + 1,
            Results: batch,
            HasNext: i < len(batches)-1,
        }
    }
    return pages
}

// Stream обработка с backpressure
type DataStream struct {
    data []DataPoint
}

func (ds *DataStream) StreamWithBackpressure(chunkSize int) {
    batches := Batch(ds.data, chunkSize)

    rateLimiter := time.NewTicker(100 * time.Millisecond)
    defer rateLimiter.Stop()

    for _, batch := range batches {
        <-rateLimiter.C
        processStream(batch)
    }
}

// Machine learning dataset batching
type Dataset struct {
    samples []Sample
}

func (ds *Dataset) GetBatches(batchSize int) []*Batch {
    batches := Batch(ds.samples, batchSize)

    result := make([]*Batch, len(batches))
    for i, batch := range batches {
        result[i] = &Batch{
            Samples: batch,
            Index:   i,
        }
    }
    return result
}

// Rate limiting с request batching
func RateLimitedAPICall(ids []string, maxPerRequest int) error {
    batches := Batch(ids, maxPerRequest)
    limiter := time.NewTicker(1 * time.Second)

    for _, batch := range batches {
        <-limiter.C
        if _, err := externalAPI.FetchData(batch); err != nil {
            return err
        }
    }
    return nil
}

// File reading с chunking
func ProcessLargeFile(filePath string, chunkSize int) error {
    lines, _ := readFileLines(filePath)
    batches := Batch(lines, chunkSize)

    for i, batch := range batches {
        if err := processLinesBatch(batch); err != nil {
            return fmt.Errorf("batch %d: %w", i, err)
        }
    }
    return nil
}

// Log aggregation с batching
type LogAggregator struct {
    logs []LogEntry
}

func (la *LogAggregator) AggregateAndSend(batchSize int) error {
    batches := Batch(la.logs, batchSize)

    for _, batch := range batches {
        aggregated := aggregateLogs(batch)
        if err := sendToAnalytics(aggregated); err != nil {
            return err
        }
    }
    return nil
}
\`\`\`

**Практические преимущества:**
- **Обработка больших файлов:** Обработка многогигабайтных файлов чанками без OOM
- **Операции с БД:** Batch inserts/updates для оптимальной производительности
- **API Rate Limiting:** Соблюдение лимитов API с контролируемым request batching
- **Web Pagination:** Эффективная пагинация без загрузки всех результатов
- **Data Pipeline:** Подача контролируемых объемов на следующий этап обработки
- **Параллельная обработка:** Равномерное распределение работы на несколько воркеров

**Преимущества производительности:**
- **Память:** Только текущий чанк в активной памяти
- **I/O:** Batch I/O операции снижают overhead syscall
- **Сеть:** Меньше, но больших пакетов вместо множества маленьких
- **База данных:** Batch операции быстрее индивидуальных inserts
- **Concurrency:** Лучшее распределение нагрузки между воркерами

**Типичные сценарии использования:**
- Обработка CSV/JSON файлов
- Database bulk операции
- API pagination
- Параллельная обработка воркерами
- Rate-limited внешние API вызовы
- Message queue batching

Без Batch обработка гигабайтных наборов данных потребовала бы загрузки всего в память, вызывая OOM crashes и ужасную производительность.`,
			solutionCode: `package datastructures

func Batch[T any](in []T, n int) [][]T {
	if n <= 0 || len(in) == 0 {                             // Проверить неправильный размер batch или пустой ввод
		return nil                                      // Вернуть nil
	}
	chunks := make([][]T, 0, (len(in)+n-1)/n)              // Предварительно выделить с ceiling division
	for start := 0; start < len(in); start += n {          // Итерация с шагом n
		end := start + n                                // Вычислить конечный индекс
		if end > len(in) {                              // Проверить превышает ли конец длину слайса
			end = len(in)                           // Подогнать к фактическому концу
		}
		chunks = append(chunks, in[start:end])          // Добавить slice view (без копирования)
	}
	return chunks                                           // Вернуть слайс слайсов
}`
		},
		uz: {
			title: 'Slaysni bo\'laklarga bo\'lish',
			description: `Slaysni n o'lchami bilan bo'laklarga bo'lib beradigan **Batch** ni amalga oshiring.

**Talablar:**
1. \`Batch[T any](in []T, n int) [][]T\` funksiyasini yarating
2. Kiritish slaysni n o'lchamdagi bo'laklarga bo'ling
3. Noto'g'ri batch o'lchamini ishlang (n <= 0) nil qaytarish orqali
4. Bo'sh kiritish slaysni ishlang nil qaytarish orqali
5. Oxirgi bo'lak kiritish uzunligi n ga bo'linmaydigan bo'lsa kichikroq bo'lishi mumkin
6. Slice views dan foydalaning (alohida elementlarni nusxalashdan qochng)
7. Samaradorlik uchun capacity oldindan hisoblang
8. To'g'ri atsroflar bilan slayslar slaysini qaytaring

**Misol:**
\`\`\`go
result := Batch([]int{1, 2, 3, 4, 5}, 2)
// result = [][]int{{1, 2}, {3, 4}, {5}}

result2 := Batch([]string{"a", "b", "c", "d", "e", "f"}, 3)
// result2 = [][]string{{"a", "b", "c"}, {"d", "e", "f"}}

result3 := Batch([]int{1, 2, 3}, 5)
// result3 = [][]int{{1, 2, 3}} (n dan kichikroq bitta bo'lak)

result4 := Batch([]int{}, 2)
// result4 = nil
\`\`\`

**Cheklovlar:**
- Noto'g'ri batch o'lcham uchun nil qaytarishi kerak (n <= 0)
- Bo'sh kiritish uchun nil qaytarishi kerak
- Alohida elementlarni nusxalashmasligi kerak (slice views dan foydalaning)
- Noto'liq oxirgi bo'lakni to'g'ri ishlashi kerak
- Natija slaysning capacity ni oldindan ajratib qo'yishi kerak`,
			hint1: `Natija slaysis uchun zarur bo'lgan exact capacity ni hisoblash uchun ceiling division ((len+n-1)/n) dan foydalaning.`,
			hint2: `Elementlarni nusxalashsiz bo'laklar yaratish uchun slice views in[start:end] dan foydalaning - ular asl ma'lumotlarga atsrof qiladi.`,
			whyItMatters: `Batch katta ma'lumot to'plamlarini boshqariladigan bo'laklar bilan qayta ishlash uchun zarur bo'lib, parallel qayta ishlash, streaming, pagination va memory-efficient operatsiyalarni o'z ichiga oladi, aks holda resurslar tugashiga olib keladi.

**Nima uchun Batching:**
- **Resurslarni boshqarish:** Bir vaqtda cheklangan ma'lumot miqdorini qayta ishlash
- **Parallel qayta ishlash:** Bo'laklarni concurrent ish uchun goroutinalarga taqsimlash
- **Xotira samaradorligi:** Butun ma'lumot to'plamini xotiraga yuklashdan qochish
- **Pagination:** Natijalarni boshqariladigan bo'laklar bilan ko'rsatish
- **Streaming:** Ma'lumotlarni oldindan bufferlashdan qochib qayta ishlash
- **Pipeline Bosqichlari:** Keyingi bosqichga boshqariladigan miqdorlarni uzatish

**Production Pattern:**
\`\`\`go
// Database bulk insert batching bilan
func InsertUsersBatch(users []User, batchSize int) error {
    batches := Batch(users, batchSize)

    for _, batch := range batches {
        if err := database.InsertBatch(batch); err != nil {
            return err
        }
    }
    return nil
}

// Worker pool bilan parallel qayta ishlash
func ProcessItemsParallel(items []Item, batchSize int) {
    batches := Batch(items, batchSize)
    workerCount := 4

    jobs := make(chan []Item, workerCount)
    go func() {
        for _, batch := range batches {
            jobs <- batch
        }
        close(jobs)
    }()

    for i := 0; i < workerCount; i++ {
        go func() {
            for batch := range jobs {
                processBatch(batch)
            }
        }()
    }
}

// Belgilangan sahifa o'lchami bilan API pagination
type APIResponse struct {
    Items      []interface{} \`json:"items"\`
    TotalPages int           \`json:"total_pages"\`
    PageNumber int           \`json:"page_number"\`
}

func GetPaginatedResults(allResults []interface{}, pageNumber, pageSize int) APIResponse {
    batches := Batch(allResults, pageSize)

    if pageNumber < 1 || pageNumber > len(batches) {
        return APIResponse{Items: []interface{}{}}
    }

    return APIResponse{
        Items:      batches[pageNumber-1],
        TotalPages: len(batches),
        PageNumber: pageNumber,
    }
}

// Xotira samaradorligi uchun bo'laklarga bo'lib CSV eksport
func ExportToCSVChunked(records []Record, chunkSize int) error {
    batches := Batch(records, chunkSize)

    for i, batch := range batches {
        file, _ := os.Create(fmt.Sprintf("export_%d.csv", i))
        defer file.Close()

        writer := csv.NewWriter(file)
        for _, record := range batch {
            writer.Write(record.ToCSVRow())
        }
        writer.Flush()
    }
    return nil
}

// Samarali tarmoq transporti uchun queue xabarlari batching
type MessageBatcher struct {
    messages []Message
    batchSize int
}

func (mb *MessageBatcher) SendBatched() error {
    batches := Batch(mb.messages, mb.batchSize)

    for _, batch := range batches {
        if err := sendNetworkRequest(batch); err != nil {
            return err
        }
    }
    return nil
}

// Qidiruv natijalari pagination
func SearchWithPagination(query string, pageSize int) []Page {
    allResults := database.Search(query)
    batches := Batch(allResults, pageSize)

    pages := make([]Page, len(batches))
    for i, batch := range batches {
        pages[i] = Page{
            Number:  i + 1,
            Results: batch,
            HasNext: i < len(batches)-1,
        }
    }
    return pages
}

// Backpressure bilan stream qayta ishlash
type DataStream struct {
    data []DataPoint
}

func (ds *DataStream) StreamWithBackpressure(chunkSize int) {
    batches := Batch(ds.data, chunkSize)

    rateLimiter := time.NewTicker(100 * time.Millisecond)
    defer rateLimiter.Stop()

    for _, batch := range batches {
        <-rateLimiter.C
        processStream(batch)
    }
}

// Machine learning dataset batching
type Dataset struct {
    samples []Sample
}

func (ds *Dataset) GetBatches(batchSize int) []*Batch {
    batches := Batch(ds.samples, batchSize)

    result := make([]*Batch, len(batches))
    for i, batch := range batches {
        result[i] = &Batch{
            Samples: batch,
            Index:   i,
        }
    }
    return result
}

// So'rov batching bilan rate limiting
func RateLimitedAPICall(ids []string, maxPerRequest int) error {
    batches := Batch(ids, maxPerRequest)
    limiter := time.NewTicker(1 * time.Second)

    for _, batch := range batches {
        <-limiter.C
        if _, err := externalAPI.FetchData(batch); err != nil {
            return err
        }
    }
    return nil
}

// Chunking bilan fayl o'qish
func ProcessLargeFile(filePath string, chunkSize int) error {
    lines, _ := readFileLines(filePath)
    batches := Batch(lines, chunkSize)

    for i, batch := range batches {
        if err := processLinesBatch(batch); err != nil {
            return fmt.Errorf("batch %d: %w", i, err)
        }
    }
    return nil
}

// Batching bilan log yig'ish
type LogAggregator struct {
    logs []LogEntry
}

func (la *LogAggregator) AggregateAndSend(batchSize int) error {
    batches := Batch(la.logs, batchSize)

    for _, batch := range batches {
        aggregated := aggregateLogs(batch)
        if err := sendToAnalytics(aggregated); err != nil {
            return err
        }
    }
    return nil
}
\`\`\`

**Haqiqiy dunyo foydalari:**
- **Katta fayllarni qayta ishlash:** OOM siz ko'p gigabaytli fayllarni bo'laklar bilan qayta ishlash
- **Database operatsiyalari:** Optimal samaradorlik uchun batch inserts/updates
- **API Rate Limiting:** Boshqariladigan so'rov batching bilan API limitlariga rioya qilish
- **Web Pagination:** Barcha natijalarni yuklamasdan samarali pagination
- **Data Pipeline:** Keyingi bosqichga boshqariladigan miqdorlarni uzatish
- **Parallel qayta ishlash:** Ishni bir nechta workerlarga teng taqsimlash

**Samaradorlik afzalliklari:**
- **Xotira:** Faqat joriy bo'lak faol xotirada
- **I/O:** Batch I/O operatsiyalari syscall overheadni kamaytiradi
- **Tarmoq:** Ko'p kichik paketlar o'rniga kamroq, lekin katta paketlar
- **Database:** Batch operatsiyalari individual insertsdan tezroq
- **Concurrency:** Workerlar o'rtasida yukni yaxshiroq taqsimlash

**Umumiy foydalanish holatlari:**
- CSV/JSON fayllarni qayta ishlash
- Database bulk operatsiyalar
- API pagination
- Workerlar bilan parallel qayta ishlash
- Rate-limited tashqi API chaqiruvlari
- Message queue batching

Batch siz gigabayт hajmidagi ma'lumot to'plamlarini qayta ishlash hamma narsani xotiraga yuklashni talab qiladi, OOM crashlar va dahshatli samaradorlikka olib keladi.`,
			solutionCode: `package datastructures

func Batch[T any](in []T, n int) [][]T {
	if n <= 0 || len(in) == 0 {                             // Noto'g'ri batch o'lchami yoki bo'sh kirishni tekshirish
		return nil                                      // Nil qaytarish
	}
	chunks := make([][]T, 0, (len(in)+n-1)/n)              // Ceiling division bilan oldindan ajratish
	for start := 0; start < len(in); start += n {          // n qadam bilan iteratsiya
		end := start + n                                // Tugash indeksini hisoblash
		if end > len(in) {                              // Tugash slayz uzunligidan oshib ketishini tekshirish
			end = len(in)                           // Haqiqiy tugashga moslashtirish
		}
		chunks = append(chunks, in[start:end])          // Slice view qo'shish (nusxalashsiz)
	}
	return chunks                                           // Slayslar slaysini qaytarish
}`
		}
	}
};

export default task;
