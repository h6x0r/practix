import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-pipeline-square',
	title: 'Square',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'pipeline', 'fan-out', 'workers'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Square** that computes squares of input numbers using parallel workers (fan-out pattern).

**Requirements:**
1. Create function \`Square(in <-chan int, workers int) <-chan int\`
2. Handle workers <= 0 (default to 1 worker)
3. Create output channel (unbuffered)
4. Launch 'workers' goroutines that read from input
5. Each worker squares values and sends to output
6. Use sync.WaitGroup to track all workers
7. Close output channel when all workers finish
8. Return output channel immediately

**Example:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5)
out := Square(in, 3) // 3 parallel workers

for v := range out {
    fmt.Println(v)
}
// Output: 1 4 9 16 25 (order may vary due to parallelism)

in = Gen(2, 3, 4)
out = Square(in, 0) // Default to 1 worker
for v := range out {
    fmt.Println(v)
}
// Output: 4 9 16
\`\`\`

**Constraints:**
- Must use sync.WaitGroup for worker coordination
- Must handle workers <= 0 case
- All workers must share the same input channel
- Output order is not guaranteed (parallelism)`,
	initialCode: `package concurrency

import "sync"

// TODO: Implement Square
func Square(in <-chan int, workers int) <-chan int {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import "sync"

func Square(in <-chan int, workers int) <-chan int {
	if workers <= 0 {                                           // Validate worker count
		workers = 1                                         // Default to 1 worker
	}
	out := make(chan int)                                       // Create output channel
	var wg sync.WaitGroup                                       // WaitGroup for workers
	wg.Add(workers)                                             // Add worker count
	for i := 0; i < workers; i++ {                              // Launch workers
		go func() {                                         // Worker goroutine
			defer wg.Done()                             // Mark worker done
			for v := range in {                         // Read from input
				out <- v * v                        // Send squared value
			}
		}()
	}
	go func() {                                                 // Closer goroutine
		wg.Wait()                                           // Wait for all workers
		close(out)                                          // Close output channel
	}()
	return out                                                  // Return immediately
}`,
			hint1: `Use a sync.WaitGroup to track all workers. Add(workers) before launching, Done() in each worker, and Wait() before closing the output.`,
			hint2: `Launch all workers in a loop, then launch a separate goroutine that waits for all workers to finish (wg.Wait()) before closing the output channel.`,
			testCode: `package concurrency

import (
	"sort"
	"sync"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	in := make(chan int)
	close(in)
	out := Square(in, 1)
	count := 0
	for range out {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 values from closed input, got %d", count)
	}
}

func Test2(t *testing.T) {
	in := make(chan int, 1)
	in <- 5
	close(in)
	out := Square(in, 1)
	v := <-out
	if v != 25 {
		t.Errorf("expected 25, got %d", v)
	}
}

func Test3(t *testing.T) {
	in := make(chan int, 5)
	for _, v := range []int{1, 2, 3, 4, 5} {
		in <- v
	}
	close(in)
	out := Square(in, 3)
	results := make([]int, 0, 5)
	for v := range out {
		results = append(results, v)
	}
	sort.Ints(results)
	expected := []int{1, 4, 9, 16, 25}
	for i, v := range results {
		if v != expected[i] {
			t.Errorf("expected %v, got %v", expected, results)
			break
		}
	}
}

func Test4(t *testing.T) {
	in := make(chan int, 3)
	in <- 2
	in <- 3
	in <- 4
	close(in)
	out := Square(in, 0)
	count := 0
	for range out {
		count++
	}
	if count != 3 {
		t.Errorf("expected 3 values with 0 workers (default 1), got %d", count)
	}
}

func Test5(t *testing.T) {
	in := make(chan int, 3)
	in <- 2
	in <- 3
	in <- 4
	close(in)
	out := Square(in, -5)
	count := 0
	for range out {
		count++
	}
	if count != 3 {
		t.Errorf("expected 3 values with negative workers (default 1), got %d", count)
	}
}

func Test6(t *testing.T) {
	in := make(chan int, 10)
	for i := 1; i <= 10; i++ {
		in <- i
	}
	close(in)
	out := Square(in, 5)
	results := make([]int, 0, 10)
	for v := range out {
		results = append(results, v)
	}
	if len(results) != 10 {
		t.Errorf("expected 10 results, got %d", len(results))
	}
}

func Test7(t *testing.T) {
	in := make(chan int, 1)
	in <- 0
	close(in)
	out := Square(in, 1)
	v := <-out
	if v != 0 {
		t.Errorf("expected 0*0=0, got %d", v)
	}
}

func Test8(t *testing.T) {
	in := make(chan int, 1)
	in <- -5
	close(in)
	out := Square(in, 1)
	v := <-out
	if v != 25 {
		t.Errorf("expected (-5)*(-5)=25, got %d", v)
	}
}

func Test9(t *testing.T) {
	done := make(chan bool, 1)
	go func() {
		in := make(chan int)
		close(in)
		_ = Square(in, 3)
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Error("Square should return immediately")
	}
}

func Test10(t *testing.T) {
	in := make(chan int)
	out := Square(in, 3)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 1; i <= 5; i++ {
			in <- i
		}
		close(in)
	}()
	results := make([]int, 0, 5)
	for v := range out {
		results = append(results, v)
	}
	wg.Wait()
	if len(results) != 5 {
		t.Errorf("expected 5 results, got %d", len(results))
	}
}
`,
	whyItMatters: `Square demonstrates the fan-out pattern where multiple workers process from a single input channel, enabling parallel processing for CPU-bound operations.

**Why Fan-Out:**
- **Parallelism:** Multiple workers process concurrently
- **CPU Utilization:** Use multiple cores effectively
- **Throughput:** Process more items per second
- **Scalability:** Adjust worker count based on workload

**Production Pattern:**
\`\`\`go
// Image processing with parallel workers
func ProcessImages(images <-chan Image, workers int) <-chan ProcessedImage {
    if workers <= 0 {
        workers = runtime.NumCPU()
    }

    out := make(chan ProcessedImage)
    var wg sync.WaitGroup
    wg.Add(workers)

    for i := 0; i < workers; i++ {
        go func() {
            defer wg.Done()
            for img := range images {
                // CPU-intensive work
                processed := img.Resize().Compress().AddWatermark()
                out <- processed
            }
        }()
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

// Data validation with workers
func ValidateRecords(records <-chan Record, workers int) <-chan ValidationResult {
    if workers <= 0 {
        workers = 10
    }

    out := make(chan ValidationResult)
    var wg sync.WaitGroup
    wg.Add(workers)

    for i := 0; i < workers; i++ {
        go func() {
            defer wg.Done()
            for record := range records {
                result := ValidationResult{
                    ID:    record.ID,
                    Valid: validateEmail(record.Email) && validatePhone(record.Phone),
                }
                out <- result
            }
        }()
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

// Hash computation with parallelism
func ComputeHashes(files <-chan string, workers int) <-chan FileHash {
    if workers <= 0 {
        workers = 4
    }

    out := make(chan FileHash)
    var wg sync.WaitGroup
    wg.Add(workers)

    for i := 0; i < workers; i++ {
        go func() {
            defer wg.Done()
            for filename := range files {
                data, _ := os.ReadFile(filename)
                hash := sha256.Sum256(data)
                out <- FileHash{
                    File: filename,
                    Hash: hex.EncodeToString(hash[:]),
                }
            }
        }()
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

// Dynamic worker scaling
func ProcessWithScaling(in <-chan Task) <-chan Result {
    workers := runtime.NumCPU()
    out := make(chan Result, workers)
    var wg sync.WaitGroup

    // Scale up during high load
    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            for task := range in {
                result := processTask(task)
                out <- result
            }
        }(i)
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

// Worker pool with metrics
type WorkerPool struct {
    workers int
    wg      sync.WaitGroup
    metrics *Metrics
}

func (wp *WorkerPool) Process(in <-chan Work) <-chan Result {
    out := make(chan Result)
    wp.wg.Add(wp.workers)

    for i := 0; i < wp.workers; i++ {
        go func(workerID int) {
            defer wp.wg.Done()
            for work := range in {
                start := time.Now()
                result := work.Process()
                wp.metrics.RecordLatency(time.Since(start))
                wp.metrics.IncrementProcessed(workerID)
                out <- result
            }
        }(i)
    }

    go func() {
        wp.wg.Wait()
        close(out)
    }()
    return out
}
\`\`\`

**Real-World Benefits:**
- **Performance:** Process large datasets faster
- **Resource Control:** Limit concurrent operations
- **Predictable Load:** Control system resource usage
- **Error Isolation:** One worker failure doesn't stop others

**Worker Count Guidelines:**
- **CPU-Bound:** workers = runtime.NumCPU() or NumCPU() * 2
- **I/O-Bound:** workers = 10-100+ (higher concurrency)
- **Memory-Limited:** workers = available_memory / per_worker_memory
- **External API:** workers = rate_limit / items_per_second

**Common Use Cases:**
- **Image Processing:** Resize, compress, filter images
- **Data Validation:** Validate records in parallel
- **File Processing:** Hash, compress, encrypt files
- **API Requests:** Make concurrent HTTP requests
- **Database Operations:** Parallel inserts/updates

Without fan-out, a single worker processes items sequentially, wasting available CPU cores and taking much longer to complete.`,	order: 2,
	translations: {
		ru: {
			title: 'Преобразование значений в pipeline (квадрат)',
			description: `Реализуйте **Square**, который вычисляет квадраты входных чисел используя параллельных рабочих (паттерн fan-out).

**Требования:**
1. Создайте функцию \`Square(in <-chan int, workers int) <-chan int\`
2. Обработайте workers <= 0 (по умолчанию 1 рабочий)
3. Создайте выходной канал (небуферизованный)
4. Запустите 'workers' горутин которые читают из входа
5. Каждый рабочий возводит в квадрат значения и отправляет в выход
6. Используйте sync.WaitGroup для отслеживания всех рабочих
7. Закройте выходной канал когда все рабочие закончат
8. Верните выходной канал немедленно

**Пример:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5)
out := Square(in, 3) // 3 параллельных рабочих

for v := range out {
    fmt.Println(v)
}
// Вывод: 1 4 9 16 25 (порядок может меняться из-за параллелизма)
\`\`\`

**Ограничения:**
- Должен использовать sync.WaitGroup для координации рабочих
- Должен обрабатывать случай workers <= 0
- Все рабочие должны использовать один входной канал
- Порядок вывода не гарантирован (параллелизм)`,
			hint1: `Используйте sync.WaitGroup для отслеживания всех рабочих. Add(workers) перед запуском, Done() в каждом рабочем, и Wait() перед закрытием выхода.`,
			hint2: `Запустите всех рабочих в цикле, затем запустите отдельную горутину которая ждёт завершения всех рабочих (wg.Wait()) перед закрытием выходного канала.`,
			whyItMatters: `Square демонстрирует паттерн fan-out где несколько рабочих обрабатывают из одного входного канала, обеспечивая параллельную обработку для CPU-bound операций.

**Зачем Fan-Out:**
- **Параллелизм:** Несколько рабочих обрабатывают одновременно
- **Использование CPU:** Эффективное использование нескольких ядер
- **Пропускная способность:** Обработка большего количества элементов в секунду
- **Масштабируемость:** Настройка количества рабочих на основе нагрузки

**Продакшен паттерн:**
\`\`\`go
// Обработка изображений с параллельными рабочими
func ProcessImages(images <-chan Image, workers int) <-chan ProcessedImage {
    if workers <= 0 {
        workers = runtime.NumCPU()
    }

    out := make(chan ProcessedImage)
    var wg sync.WaitGroup
    wg.Add(workers)

    for i := 0; i < workers; i++ {
        go func() {
            defer wg.Done()
            for img := range images {
                // CPU-интенсивная работа
                processed := img.Resize().Compress().AddWatermark()
                out <- processed
            }
        }()
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

// Валидация данных с рабочими
func ValidateRecords(records <-chan Record, workers int) <-chan ValidationResult {
    if workers <= 0 {
        workers = 10
    }

    out := make(chan ValidationResult)
    var wg sync.WaitGroup
    wg.Add(workers)

    for i := 0; i < workers; i++ {
        go func() {
            defer wg.Done()
            for record := range records {
                result := ValidationResult{
                    ID:    record.ID,
                    Valid: validateEmail(record.Email) && validatePhone(record.Phone),
                }
                out <- result
            }
        }()
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

// Вычисление хешей с параллелизмом
func ComputeHashes(files <-chan string, workers int) <-chan FileHash {
    if workers <= 0 {
        workers = 4
    }

    out := make(chan FileHash)
    var wg sync.WaitGroup
    wg.Add(workers)

    for i := 0; i < workers; i++ {
        go func() {
            defer wg.Done()
            for filename := range files {
                data, _ := os.ReadFile(filename)
                hash := sha256.Sum256(data)
                out <- FileHash{
                    File: filename,
                    Hash: hex.EncodeToString(hash[:]),
                }
            }
        }()
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

// Динамическое масштабирование рабочих
func ProcessWithScaling(in <-chan Task) <-chan Result {
    workers := runtime.NumCPU()
    out := make(chan Result, workers)
    var wg sync.WaitGroup

    // Масштабирование при высокой нагрузке
    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            for task := range in {
                result := processTask(task)
                out <- result
            }
        }(i)
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

// Worker pool с метриками
type WorkerPool struct {
    workers int
    wg      sync.WaitGroup
    metrics *Metrics
}

func (wp *WorkerPool) Process(in <-chan Work) <-chan Result {
    out := make(chan Result)
    wp.wg.Add(wp.workers)

    for i := 0; i < wp.workers; i++ {
        go func(workerID int) {
            defer wp.wg.Done()
            for work := range in {
                start := time.Now()
                result := work.Process()
                wp.metrics.RecordLatency(time.Since(start))
                wp.metrics.IncrementProcessed(workerID)
                out <- result
            }
        }(i)
    }

    go func() {
        wp.wg.Wait()
        close(out)
    }()
    return out
}
\`\`\`

**Практические преимущества:**
- **Производительность:** Быстрая обработка больших наборов данных
- **Контроль ресурсов:** Ограничение одновременных операций
- **Предсказуемая нагрузка:** Контроль использования системных ресурсов
- **Изоляция ошибок:** Сбой одного рабочего не останавливает других

**Рекомендации по количеству workers:**
- **CPU-Bound:** workers = runtime.NumCPU() или NumCPU() * 2
- **I/O-Bound:** workers = 10-100+ (более высокая конкурентность)
- **Ограниченная память:** workers = available_memory / per_worker_memory
- **Внешний API:** workers = rate_limit / items_per_second

**Обычные сценарии использования:**
- **Обработка изображений:** Изменение размера, сжатие, фильтрация изображений
- **Валидация данных:** Параллельная валидация записей
- **Обработка файлов:** Хеширование, сжатие, шифрование файлов
- **API запросы:** Параллельные HTTP запросы
- **Операции с БД:** Параллельные вставки/обновления

Без fan-out один рабочий обрабатывает элементы последовательно, теряя доступные ядра CPU и занимая гораздо больше времени на завершение.`,
			solutionCode: `package concurrency

import "sync"

func Square(in <-chan int, workers int) <-chan int {
	if workers <= 0 {                                           // Проверяем количество рабочих
		workers = 1                                         // По умолчанию 1 рабочий
	}
	out := make(chan int)                                       // Создаём выходной канал
	var wg sync.WaitGroup                                       // WaitGroup для рабочих
	wg.Add(workers)                                             // Добавляем количество рабочих
	for i := 0; i < workers; i++ {                              // Запускаем рабочих
		go func() {                                         // Горутина рабочего
			defer wg.Done()                             // Отмечаем завершение рабочего
			for v := range in {                         // Читаем из входа
				out <- v * v                        // Отправляем квадрат значения
			}
		}()
	}
	go func() {                                                 // Горутина закрывателя
		wg.Wait()                                           // Ждём всех рабочих
		close(out)                                          // Закрываем выходной канал
	}()
	return out                                                  // Возвращаем немедленно
}`
		},
		uz: {
			title: 'Pipeline da qiymatlarni o\'zgartirish (kvadrat)',
			description: `Parallel workerlar yordamida kirish raqamlarining kvadratlarini hisoblaydigan **Square** ni amalga oshiring (fan-out pattern).

**Talablar:**
1. \`Square(in <-chan int, workers int) <-chan int\` funksiyasini yarating
2. workers <= 0 ni ishlang (standart 1 worker)
3. Chiqish kanalini yarating (buferlanmagan)
4. Kirishdan o'qiydigan 'workers' goroutinelarni ishga tushiring
5. Har bir worker qiymatlarni kvadratga ko'taradi va chiqishga yuboradi
6. Barcha workerlarni kuzatish uchun sync.WaitGroup dan foydalaning
7. Barcha workerlar tugaganda chiqish kanalini yoping
8. Chiqish kanalini darhol qaytaring

**Misol:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5)
out := Square(in, 3) // 3 parallel worker

for v := range out {
    fmt.Println(v)
}
// Natija: 1 4 9 16 25 (paralellik tufayli tartib o'zgarishi mumkin)
\`\`\`

**Cheklovlar:**
- Workerlarni muvofiqlashtirish uchun sync.WaitGroup dan foydalanishi kerak
- workers <= 0 holatini ishlashi kerak
- Barcha workerlar bir kirish kanalidan foydalanishi kerak
- Chiqish tartibi kafolatlanmagan (paralellik)`,
			hint1: `Barcha workerlarni kuzatish uchun sync.WaitGroup dan foydalaning. Ishga tushirishdan oldin Add(workers), har bir workerda Done(), va chiqishni yopishdan oldin Wait().`,
			hint2: `Barcha workerlarni siklda ishga tushiring, keyin chiqish kanalini yopishdan oldin barcha workerlar tugashini kutadigan (wg.Wait()) alohida goroutine ishga tushiring.`,
			whyItMatters: `Square bir kirish kanalidan bir nechta workerlar qayta ishlash uchun fan-out patternini namoyish etadi, CPU-bound operatsiyalar uchun parallel qayta ishlashni ta'minlaydi.

**Nega Fan-Out kerak:**
- **Paralellik:** Bir nechta workerlar bir vaqtning o'zida qayta ishlaydi
- **CPU foydalanish:** Bir nechta yadrolardan samarali foydalanish
- **O'tkazish qobiliyati:** Sekundiga ko'proq elementlarni qayta ishlash
- **Miqyoslilik:** Ish yukiga qarab workerlar sonini sozlash

**Ishlab chiqarish patterni:**
\`\`\`go
// Parallel workerlar bilan tasvirlarni qayta ishlash
func ProcessImages(images <-chan Image, workers int) <-chan ProcessedImage {
    if workers <= 0 {
        workers = runtime.NumCPU()
    }

    out := make(chan ProcessedImage)
    var wg sync.WaitGroup
    wg.Add(workers)

    for i := 0; i < workers; i++ {
        go func() {
            defer wg.Done()
            for img := range images {
                // CPU-intensiv ish
                processed := img.Resize().Compress().AddWatermark()
                out <- processed
            }
        }()
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

// Workerlar bilan ma'lumotlarni tekshirish
func ValidateRecords(records <-chan Record, workers int) <-chan ValidationResult {
    if workers <= 0 {
        workers = 10
    }

    out := make(chan ValidationResult)
    var wg sync.WaitGroup
    wg.Add(workers)

    for i := 0; i < workers; i++ {
        go func() {
            defer wg.Done()
            for record := range records {
                result := ValidationResult{
                    ID:    record.ID,
                    Valid: validateEmail(record.Email) && validatePhone(record.Phone),
                }
                out <- result
            }
        }()
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

// Paralellik bilan xeshlarni hisoblash
func ComputeHashes(files <-chan string, workers int) <-chan FileHash {
    if workers <= 0 {
        workers = 4
    }

    out := make(chan FileHash)
    var wg sync.WaitGroup
    wg.Add(workers)

    for i := 0; i < workers; i++ {
        go func() {
            defer wg.Done()
            for filename := range files {
                data, _ := os.ReadFile(filename)
                hash := sha256.Sum256(data)
                out <- FileHash{
                    File: filename,
                    Hash: hex.EncodeToString(hash[:]),
                }
            }
        }()
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

// Dinamik workerlarni miqyoslash
func ProcessWithScaling(in <-chan Task) <-chan Result {
    workers := runtime.NumCPU()
    out := make(chan Result, workers)
    var wg sync.WaitGroup

    // Yuqori yuklama paytida miqyoslash
    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            for task := range in {
                result := processTask(task)
                out <- result
            }
        }(i)
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

// Metrikalar bilan worker pool
type WorkerPool struct {
    workers int
    wg      sync.WaitGroup
    metrics *Metrics
}

func (wp *WorkerPool) Process(in <-chan Work) <-chan Result {
    out := make(chan Result)
    wp.wg.Add(wp.workers)

    for i := 0; i < wp.workers; i++ {
        go func(workerID int) {
            defer wp.wg.Done()
            for work := range in {
                start := time.Now()
                result := work.Process()
                wp.metrics.RecordLatency(time.Since(start))
                wp.metrics.IncrementProcessed(workerID)
                out <- result
            }
        }(i)
    }

    go func() {
        wp.wg.Wait()
        close(out)
    }()
    return out
}
\`\`\`

**Amaliy foydalari:**
- **Unumdorlik:** Katta ma'lumotlar to'plamlarini tezroq qayta ishlash
- **Resurs boshqaruvi:** Bir vaqtdagi operatsiyalarni cheklash
- **Bashorat qilinadigan yuk:** Tizim resurslaridan foydalanishni boshqarish
- **Xato ajratish:** Bitta workerning ishlamay qolishi boshqalarni to'xtatmaydi

**Workerlar soni bo'yicha tavsiyalar:**
- **CPU-Bound:** workers = runtime.NumCPU() yoki NumCPU() * 2
- **I/O-Bound:** workers = 10-100+ (yuqoriroq konkurentlik)
- **Cheklangan xotira:** workers = mavjud_xotira / worker_uchun_xotira
- **Tashqi API:** workers = tezlik_limiti / elementlar_sekundiga

**Umumiy foydalanish holatlari:**
- **Tasvirlarni qayta ishlash:** Tasvirlarni o'lchamini o'zgartirish, siqish, filtrlash
- **Ma'lumotlarni tekshirish:** Yozuvlarni parallel tekshirish
- **Fayllarni qayta ishlash:** Fayllarni xeshlash, siqish, shifrlash
- **API so'rovlari:** Parallel HTTP so'rovlar
- **Ma'lumotlar bazasi operatsiyalari:** Parallel insert/update

Fan-out bo'lmasa, bitta worker elementlarni ketma-ket qayta ishlaydi, mavjud CPU yadrolarini isrof qiladi va tugatish uchun ancha ko'p vaqt oladi.`,
			solutionCode: `package concurrency

import "sync"

func Square(in <-chan int, workers int) <-chan int {
	if workers <= 0 {                                           // Workerlar sonini tekshiramiz
		workers = 1                                         // Standart 1 worker
	}
	out := make(chan int)                                       // Chiqish kanalini yaratamiz
	var wg sync.WaitGroup                                       // Workerlar uchun WaitGroup
	wg.Add(workers)                                             // Workerlar sonini qo'shamiz
	for i := 0; i < workers; i++ {                              // Workerlarni ishga tushiramiz
		go func() {                                         // Worker goroutinesi
			defer wg.Done()                             // Worker tugaganini belgilaymiz
			for v := range in {                         // Kirishdan o'qiymiz
				out <- v * v                        // Kvadrat qiymatni yuboramiz
			}
		}()
	}
	go func() {                                                 // Yopuvchi goroutine
		wg.Wait()                                           // Barcha workerlarni kutamiz
		close(out)                                          // Chiqish kanalini yopamiz
	}()
	return out                                                  // Darhol qaytaramiz
}`
		}
	}
};

export default task;
