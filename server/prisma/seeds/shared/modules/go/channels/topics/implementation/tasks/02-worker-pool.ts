import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-channels-worker-pool',
	title: 'Worker Pool Pattern',
	difficulty: 'medium',	tags: ['go', 'channels', 'concurrency', 'worker-pool'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement a worker pool that processes channel items with concurrent workers and error handling.

**Requirements:**
1. **RunWorkerPool**: Create pool of N workers consuming from input channel
2. **Handler Function**: Each worker calls handler for items
3. **Error Handling**: Return first error encountered, stop all workers
4. **Context Awareness**: Stop on context cancellation

**Worker Pool Pattern:**
\`\`\`go
func RunWorkerPool[T any](
    ctx context.Context,
    in <-chan T,
    workers int,
    h Handler[T],
) error {
    // Launch N worker goroutines
    // Each reads from shared input channel
    // Call handler for each item
    // Return first error
}
\`\`\`

**Example Usage:**
\`\`\`go
// Process orders concurrently
func ProcessOrders(orders <-chan Order) error {
    ctx := context.Background()

    handler := func(ctx context.Context, order Order) error {
        return processOrder(order)
    }

    // 10 workers process orders concurrently
    return RunWorkerPool(ctx, orders, 10, handler)
}

// Image processing pipeline
func ProcessImages(images <-chan Image) error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
    defer cancel()

    handler := func(ctx context.Context, img Image) error {
        resized := resize(img)
        optimized := optimize(resized)
        return upload(optimized)
    }

    // 20 workers for CPU-intensive work
    return RunWorkerPool(ctx, images, 20, handler)
}
\`\`\`

**Constraints:**
- Must handle workers parameter (number of concurrent workers)
- Must stop all workers on first error
- Must respect context cancellation
- Must not leak goroutines`,
	initialCode: `package channelsx

import (
	"context"
	"sync"
)

type Handler[T any] func(context.Context, T) error

// TODO: Implement RunWorkerPool
// Launch 'workers' goroutines
// Each goroutine reads from 'in' channel
// Call handler for each item
// Return first error, cancel context to stop others
func RunWorkerPool[T any](ctx context.Context, in <-chan T, workers int, h Handler[T]) error {
	// TODO: Implement
}`,
	solutionCode: `package channelsx

import (
	"context"
	"sync"
)

type Handler[T any] func(context.Context, T) error

func RunWorkerPool[T any](ctx context.Context, in <-chan T, workers int, h Handler[T]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if h == nil || workers <= 0 || in == nil {
		return nil
	}

	var (
		wg       sync.WaitGroup
		once     sync.Once
		firstErr error
	)

	ctx, cancel := context.WithCancel(ctx)        // create cancelable context
	defer cancel()

	recordErr := func(err error) {
		if err != nil {
			once.Do(func() {                       // capture only first error
				firstErr = err
				cancel()                            // stop all workers
			})
		}
	}

	worker := func() {
		defer wg.Done()
		for {
			select {
			case <-ctx.Done():                      // context canceled
				return
			case v, ok := <-in:
				if !ok {                             // channel closed
					return
				}
				if err := h(ctx, v); err != nil {    // process item
					recordErr(err)                   // record error and stop
				}
			}
		}
	}

	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go worker()                                 // launch worker goroutines
	}
	wg.Wait()                                      // wait for all workers
	return firstErr                                // return first error or nil
}`,
	testCode: `package channelsx

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestRunWorkerPool_BasicProcessing(t *testing.T) {
	ctx := context.Background()
	in := make(chan int, 5)
	for i := 1; i <= 5; i++ {
		in <- i
	}
	close(in)

	var sum atomic.Int32
	handler := func(ctx context.Context, v int) error {
		sum.Add(int32(v))
		return nil
	}

	err := RunWorkerPool(ctx, in, 2, handler)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if sum.Load() != 15 {
		t.Errorf("expected sum 15, got %d", sum.Load())
	}
}

func TestRunWorkerPool_SingleWorker(t *testing.T) {
	ctx := context.Background()
	in := make(chan int, 3)
	in <- 1
	in <- 2
	in <- 3
	close(in)

	count := atomic.Int32{}
	handler := func(ctx context.Context, v int) error {
		count.Add(1)
		return nil
	}

	err := RunWorkerPool(ctx, in, 1, handler)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if count.Load() != 3 {
		t.Errorf("expected count 3, got %d", count.Load())
	}
}

func TestRunWorkerPool_MultipleWorkers(t *testing.T) {
	ctx := context.Background()
	in := make(chan int, 10)
	for i := 0; i < 10; i++ {
		in <- i
	}
	close(in)

	count := atomic.Int32{}
	handler := func(ctx context.Context, v int) error {
		count.Add(1)
		time.Sleep(10 * time.Millisecond)
		return nil
	}

	err := RunWorkerPool(ctx, in, 5, handler)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if count.Load() != 10 {
		t.Errorf("expected count 10, got %d", count.Load())
	}
}

func TestRunWorkerPool_HandlerError(t *testing.T) {
	ctx := context.Background()
	in := make(chan int, 5)
	for i := 1; i <= 5; i++ {
		in <- i
	}
	close(in)

	expectedErr := errors.New("handler error")
	handler := func(ctx context.Context, v int) error {
		if v == 3 {
			return expectedErr
		}
		return nil
	}

	err := RunWorkerPool(ctx, in, 2, handler)
	if err == nil {
		t.Error("expected error, got nil")
	}
	if err != expectedErr {
		t.Errorf("expected specific error, got %v", err)
	}
}

func TestRunWorkerPool_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	in := make(chan int, 100)
	for i := 0; i < 100; i++ {
		in <- i
	}
	close(in)

	count := atomic.Int32{}
	handler := func(ctx context.Context, v int) error {
		if count.Load() >= 5 {
			cancel()
		}
		count.Add(1)
		time.Sleep(10 * time.Millisecond)
		return nil
	}

	RunWorkerPool(ctx, in, 3, handler)
	if count.Load() >= 100 {
		t.Error("expected context cancellation to stop workers early")
	}
}

func TestRunWorkerPool_EmptyChannel(t *testing.T) {
	ctx := context.Background()
	in := make(chan int)
	close(in)

	handler := func(ctx context.Context, v int) error {
		return nil
	}

	err := RunWorkerPool(ctx, in, 2, handler)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestRunWorkerPool_FirstErrorOnly(t *testing.T) {
	ctx := context.Background()
	in := make(chan int, 10)
	for i := 1; i <= 10; i++ {
		in <- i
	}
	close(in)

	err1 := errors.New("error 1")
	err2 := errors.New("error 2")
	errorCount := atomic.Int32{}

	handler := func(ctx context.Context, v int) error {
		if v == 3 || v == 5 {
			errorCount.Add(1)
			if v == 3 {
				return err1
			}
			return err2
		}
		time.Sleep(10 * time.Millisecond)
		return nil
	}

	err := RunWorkerPool(ctx, in, 5, handler)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestRunWorkerPool_ManyWorkers(t *testing.T) {
	ctx := context.Background()
	in := make(chan int, 20)
	for i := 0; i < 20; i++ {
		in <- i
	}
	close(in)

	count := atomic.Int32{}
	handler := func(ctx context.Context, v int) error {
		count.Add(1)
		return nil
	}

	err := RunWorkerPool(ctx, in, 10, handler)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if count.Load() != 20 {
		t.Errorf("expected count 20, got %d", count.Load())
	}
}

func TestRunWorkerPool_SlowHandler(t *testing.T) {
	ctx := context.Background()
	in := make(chan int, 5)
	for i := 0; i < 5; i++ {
		in <- i
	}
	close(in)

	handler := func(ctx context.Context, v int) error {
		time.Sleep(50 * time.Millisecond)
		return nil
	}

	start := time.Now()
	err := RunWorkerPool(ctx, in, 5, handler)
	duration := time.Since(start)

	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if duration < 50*time.Millisecond {
		t.Error("expected concurrent processing to take at least 50ms")
	}
	if duration > 150*time.Millisecond {
		t.Error("expected concurrent processing with 5 workers to be fast")
	}
}
`,
			hint1: `Create a worker function that reads from the input channel in a select with ctx.Done(). Launch it workers times.`,
			hint2: `Use sync.Once to capture the first error and cancel the context to stop other workers.`,
			whyItMatters: `Worker pools are the backbone of scalable concurrent processing in Go production systems.

**Why This Matters:**

**1. Controlled Concurrency**
Without worker pool, uncontrolled goroutine creation can crash your app:
\`\`\`go
// BAD - Unbounded concurrency
func ProcessAll(items []Item) {
    for _, item := range items {
        go process(item)  // 1M items = 1M goroutines = OOM!
    }
}

// GOOD - Controlled with worker pool
func ProcessAll(items []Item) error {
    in := make(chan Item, 100)
    go func() {
        for _, item := range items {
            in <- item
        }
        close(in)
    }()

    // Only 10 goroutines, handles 1M items safely
    return RunWorkerPool(ctx, in, 10, processItem)
}
\`\`\`

**2. Real Production: Image Processing**
Photo sharing app resizing uploaded images:
\`\`\`go
// Before: Sequential processing
// 1000 images × 500ms each = 500 seconds

// After: Worker pool with 20 workers
// 1000 images ÷ 20 workers = 50 batches × 500ms = 25 seconds
// 20x faster!

handler := func(ctx context.Context, img Image) error {
    resized := resize(img, 800, 600)
    thumbnail := resize(img, 150, 150)
    return upload(resized, thumbnail)
}

return RunWorkerPool(ctx, images, 20, handler)
\`\`\`

**3. Database Batch Operations**
Inserting millions of records efficiently:
\`\`\`go
func BulkInsert(records <-chan Record) error {
    ctx := context.Background()

    handler := func(ctx context.Context, rec Record) error {
        return db.Insert(rec)
    }

    // 50 workers = 50 concurrent DB connections
    return RunWorkerPool(ctx, records, 50, handler)
}

// Sequential: 1M records in 2 hours
// Worker pool: 1M records in 4 minutes (30x faster!)
\`\`\`

**Real-World Impact:**
E-commerce order processing:
- Before: 5 minutes to process 1000 orders
- After: 15 seconds with 20-worker pool (20x faster!)
- Revenue impact: Can handle 20x more orders = $2M additional revenue/year`,	order: 1,
	translations: {
		ru: {
			title: 'Паттерн пула воркеров',
			solutionCode: `package channelsx

import (
	"context"
	"sync"
)

type Handler[T any] func(context.Context, T) error

func RunWorkerPool[T any](ctx context.Context, in <-chan T, workers int, h Handler[T]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if h == nil || workers <= 0 || in == nil {
		return nil
	}

	var (
		wg       sync.WaitGroup
		once     sync.Once
		firstErr error
	)

	ctx, cancel := context.WithCancel(ctx)        // создаём отменяемый контекст
	defer cancel()

	recordErr := func(err error) {
		if err != nil {
			once.Do(func() {                       // захватываем только первую ошибку
				firstErr = err
				cancel()                            // останавливаем всех workerов
			})
		}
	}

	worker := func() {
		defer wg.Done()
		for {
			select {
			case <-ctx.Done():                      // контекст отменён
				return
			case v, ok := <-in:
				if !ok {                             // канал закрыт
					return
				}
				if err := h(ctx, v); err != nil {    // обрабатываем элемент
					recordErr(err)                   // записываем ошибку и останавливаем
				}
			}
		}
	}

	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go worker()                                 // запускаем worker горутины
	}
	wg.Wait()                                      // ждём всех workerов
	return firstErr                                // возвращаем первую ошибку или nil
}`,
			description: `Реализуйте worker pool который обрабатывает элементы канала с конкурентными workerами и обработкой ошибок.

**Требования:**
1. **RunWorkerPool**: Создать pool из N workerов потребляющих из входного канала
2. **Handler Function**: Каждый worker вызывает handler для элементов
3. **Error Handling**: Вернуть первую встреченную ошибку, остановить всех workerов
4. **Context Awareness**: Остановка при отмене контекста

**Worker Pool паттерн:**
\`\`\`go
func RunWorkerPool[T any](
    ctx context.Context,
    in <-chan T,
    workers int,
    h Handler[T],
) error {
    // Запустить N worker горутин
    // Каждая читает из общего входного канала
    // Вызывает handler для каждого элемента
    // Возвращает первую ошибку
}
\`\`\`

**Пример использования:**
\`\`\`go
// Параллельная обработка заказов
func ProcessOrders(orders <-chan Order) error {
    ctx := context.Background()

    handler := func(ctx context.Context, order Order) error {
        return processOrder(order)
    }

    // 10 workerов обрабатывают заказы параллельно
    return RunWorkerPool(ctx, orders, 10, handler)
}

// Пайплайн обработки изображений
func ProcessImages(images <-chan Image) error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
    defer cancel()

    handler := func(ctx context.Context, img Image) error {
        resized := resize(img)
        optimized := optimize(resized)
        return upload(optimized)
    }

    // 20 workerов для CPU-интенсивной работы
    return RunWorkerPool(ctx, images, 20, handler)
}
\`\`\`

**Ограничения:**
- Должен обрабатывать параметр workers (количество параллельных workerов)
- Должен останавливать всех workerов при первой ошибке
- Должен уважать отмену контекста
- Не должен утекать горутины`,
			hint1: `Создайте worker функцию которая читает из входного канала в select с ctx.Done(). Запустите её workers раз.`,
			hint2: `Используйте sync.Once чтобы захватить первую ошибку и отменить контекст для остановки других workerов.`,
			whyItMatters: `Продакшен паттерн Worker pool — основа масштабируемой конкурентной обработки в production Go системах.

**Практические преимущества:**

**1. Контролируемая конкурентность**
Без worker pool неконтролируемое создание горутин может уронить приложение:
\`\`\`go
// ПЛОХО - Неограниченная конкурентность
func ProcessAll(items []Item) {
    for _, item := range items {
        go process(item)  // 1M элементов = 1M горутин = OOM!
    }
}

// ХОРОШО - Контроль через worker pool
func ProcessAll(items []Item) error {
    in := make(chan Item, 100)
    go func() {
        for _, item := range items {
            in <- item
        }
        close(in)
    }()

    // Только 10 горутин, безопасно обрабатывает 1M элементов
    return RunWorkerPool(ctx, in, 10, processItem)
}
\`\`\`

**2. Реальный Production сценарий: Обработка изображений**
Фото-приложение изменяет размер загруженных изображений:
\`\`\`go
// До: Последовательная обработка
// 1000 изображений × 500мс каждое = 500 секунд

// После: Worker pool с 20 workerами
// 1000 изображений ÷ 20 workerов = 50 батчей × 500мс = 25 секунд
// 20x быстрее!

handler := func(ctx context.Context, img Image) error {
    resized := resize(img, 800, 600)
    thumbnail := resize(img, 150, 150)
    return upload(resized, thumbnail)
}

return RunWorkerPool(ctx, images, 20, handler)
\`\`\`

**3. Batch операции с базой данных**
Эффективная вставка миллионов записей:
\`\`\`go
func BulkInsert(records <-chan Record) error {
    ctx := context.Background()

    handler := func(ctx context.Context, rec Record) error {
        return db.Insert(rec)
    }

    // 50 workerов = 50 параллельных соединений с БД
    return RunWorkerPool(ctx, records, 50, handler)
}

// Последовательно: 1M записей за 2 часа
// Worker pool: 1M записей за 4 минуты (30x быстрее!)
\`\`\`

**Реальное влияние:**
Обработка заказов в e-commerce:
- До: 5 минут на обработку 1000 заказов
- После: 15 секунд с 20-worker pool (20x быстрее!)
- Бизнес-эффект: Можно обработать 20x больше заказов = $2M дополнительной выручки/год`
		},
		uz: {
			title: `Worker Pool patterni`,
			solutionCode: `package channelsx

import (
	"context"
	"sync"
)

type Handler[T any] func(context.Context, T) error

func RunWorkerPool[T any](ctx context.Context, in <-chan T, workers int, h Handler[T]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if h == nil || workers <= 0 || in == nil {
		return nil
	}

	var (
		wg       sync.WaitGroup
		once     sync.Once
		firstErr error
	)

	ctx, cancel := context.WithCancel(ctx)        // bekor qilinadigan kontekst yaratamiz
	defer cancel()

	recordErr := func(err error) {
		if err != nil {
			once.Do(func() {                       // faqat birinchi xatoni qo'lga olamiz
				firstErr = err
				cancel()                            // barcha workerlarni to'xtatamiz
			})
		}
	}

	worker := func() {
		defer wg.Done()
		for {
			select {
			case <-ctx.Done():                      // kontekst bekor qilindi
				return
			case v, ok := <-in:
				if !ok {                             // kanal yopilgan
					return
				}
				if err := h(ctx, v); err != nil {    // elementni qayta ishlaymiz
					recordErr(err)                   // xatoni yozamiz va to'xtatamiz
				}
			}
		}
	}

	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go worker()                                 // worker gorutinlarni ishga tushiramiz
	}
	wg.Wait()                                      // barcha workerlarni kutamiz
	return firstErr                                // birinchi xato yoki nil qaytaramiz
}`,
			description: `Parallel workerlar va xatolarni qayta ishlash bilan kanal elementlarini qayta ishlaydigan worker pool ni amalga oshiring.

**Talablar:**
1. **RunWorkerPool**: Kirish kanalidan iste'mol qiladigan N ta workerdan iborat pool yaratish
2. **Handler Function**: Har bir worker elementlar uchun handler ni chaqiradi
3. **Error Handling**: Birinchi uchragan xatoni qaytarish, barcha workerlarni to'xtatish
4. **Context Awareness**: Kontekst bekor qilinganda to'xtatish

**Worker Pool pattern:**
\`\`\`go
func RunWorkerPool[T any](
    ctx context.Context,
    in <-chan T,
    workers int,
    h Handler[T],
) error {
    // N ta worker gorutinni ishga tushirish
    // Har biri umumiy kirish kanalidan o'qiydi
    // Har bir element uchun handler ni chaqiradi
    // Birinchi xatoni qaytaradi
}
\`\`\`

**Foydalanish misoli:**
\`\`\`go
// Buyurtmalarni parallel qayta ishlash
func ProcessOrders(orders <-chan Order) error {
    ctx := context.Background()

    handler := func(ctx context.Context, order Order) error {
        return processOrder(order)
    }

    // 10 ta worker buyurtmalarni parallel qayta ishlaydi
    return RunWorkerPool(ctx, orders, 10, handler)
}

// Rasmlarni qayta ishlash pipeline
func ProcessImages(images <-chan Image) error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
    defer cancel()

    handler := func(ctx context.Context, img Image) error {
        resized := resize(img)
        optimized := optimize(resized)
        return upload(optimized)
    }

    // CPU-intensiv ish uchun 20 ta worker
    return RunWorkerPool(ctx, images, 20, handler)
}
\`\`\`

**Cheklovlar:**
- workers parametrini (parallel workerlar soni) qayta ishlashi kerak
- Birinchi xatoda barcha workerlarni to'xtatishi kerak
- Kontekst bekor qilishni hurmat qilishi kerak
- Gorutinlarni sizib chiqarmasligi kerak`,
			hint1: `Kirish kanalidan ctx.Done() bilan select da o'qiydigan worker funksiyani yarating. Uni workers marta ishga tushiring.`,
			hint2: `Birinchi xatoni qo'lga olish uchun sync.Once dan foydalaning va boshqa workerlarni to'xtatish uchun kontekstni bekor qiling.`,
			whyItMatters: `Ishlab chiqarish patterni Worker pool — production Go tizimlarida kengaytiriladigan parallel qayta ishlashning asosini tashkil qiladi.

**Amaliy foydalari:**

**1. Nazorat qilinadigan parallellik**
Worker pool bo'lmasa, nazorat qilinmagan gorutin yaratish ilovani qulatishi mumkin:
\`\`\`go
// YOMON - Cheksiz parallellik
func ProcessAll(items []Item) {
    for _, item := range items {
        go process(item)  // 1M element = 1M gorutin = OOM!
    }
}

// YAXSHI - Worker pool bilan nazorat
func ProcessAll(items []Item) error {
    in := make(chan Item, 100)
    go func() {
        for _, item := range items {
            in <- item
        }
        close(in)
    }()

    // Faqat 10 ta gorutin, 1M elementni xavfsiz qayta ishlaydi
    return RunWorkerPool(ctx, in, 10, processItem)
}
\`\`\`

**2. Haqiqiy Production stsenariy: Rasmlarni qayta ishlash**
Foto-ilova yuklangan rasmlarning hajmini o'zgartiradi:
\`\`\`go
// Oldin: Ketma-ket qayta ishlash
// 1000 rasm × har biri 500ms = 500 soniya

// Keyin: 20 ta workerli Worker pool
// 1000 rasm ÷ 20 worker = 50 batch × 500ms = 25 soniya
// 20x tezroq!

handler := func(ctx context.Context, img Image) error {
    resized := resize(img, 800, 600)
    thumbnail := resize(img, 150, 150)
    return upload(resized, thumbnail)
}

return RunWorkerPool(ctx, images, 20, handler)
\`\`\`

**3. Ma'lumotlar bazasi bilan batch operatsiyalar**
Millionlab yozuvlarni samarali kiritish:
\`\`\`go
func BulkInsert(records <-chan Record) error {
    ctx := context.Background()

    handler := func(ctx context.Context, rec Record) error {
        return db.Insert(rec)
    }

    // 50 ta worker = 50 ta parallel DB ulanish
    return RunWorkerPool(ctx, records, 50, handler)
}

// Ketma-ket: 1M yozuv 2 soatda
// Worker pool: 1M yozuv 4 daqiqada (30x tezroq!)
\`\`\`

**Haqiqiy ta'sir:**
E-commerce da buyurtmalarni qayta ishlash:
- Oldin: 1000 buyurtmani qayta ishlash uchun 5 daqiqa
- Keyin: 20-worker pool bilan 15 soniya (20x tezroq!)
- Biznes ta'siri: 20x ko'p buyurtmalarni qayta ishlash = yiliga $2M qo'shimcha daromad`
		}
	}
};

export default task;
