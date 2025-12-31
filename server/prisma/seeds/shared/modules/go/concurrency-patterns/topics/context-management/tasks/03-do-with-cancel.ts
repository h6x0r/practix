import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-concurrency-do-with-cancel',
	title: 'Do With Cancel',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'context', 'cancellation'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **DoWithCancel** that runs function f with a cancelable context, ensuring proper cleanup.

**Requirements:**
1. Create function \`DoWithCancel(ctx context.Context, f func(context.Context)) error\`
2. Handle nil context (use Background)
3. Create cancelable context using context.WithCancel
4. Pass the cancelable context to f
5. Run f in a goroutine
6. Check if parent context is already canceled
7. Always cancel the context and wait for f to complete
8. Return parent context error if it was canceled

**Example:**
\`\`\`go
err := DoWithCancel(ctx, func(ctx context.Context) {
    for {
        select {
        case <-ctx.Done():
            return
        default:
            // Do work
        }
    }
})
// err = nil (normal completion)

parentCtx, cancel := context.WithCancel(context.Background())
cancel() // Cancel parent
err = DoWithCancel(parentCtx, func(ctx context.Context) {
    // Will receive cancellation signal
})
// err = context.Canceled
\`\`\`

**Constraints:**
- Must use context.WithCancel
- Must pass cancelable context to f
- Must wait for f to complete before returning
- Must handle parent context cancellation`,
	initialCode: `package concurrency

import (
	"context"
)

// TODO: Implement DoWithCancel
func DoWithCancel(ctx context.Context, f func(context.Context)) error {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
)

func DoWithCancel(ctx context.Context, f func(context.Context)) error {
	if ctx == nil {                                             // Handle nil context
		ctx = context.Background()                          // Use Background as fallback
	}
	ctxWithCancel, cancel := context.WithCancel(ctx)           // Create cancelable context
	done := make(chan struct{})                                 // Channel to signal completion
	go func() {                                                 // Run f in goroutine
		defer close(done)                                   // Close channel when done
		f(ctxWithCancel)                                    // Pass cancelable context to f
	}()
	if err := ctx.Err(); err != nil {                           // Check if parent already canceled
		cancel()                                            // Cancel child context
		<-done                                              // Wait for goroutine to finish
		return err                                          // Return parent error
	}
	cancel()                                                    // Always cancel to free resources
	<-done                                                      // Wait for goroutine to finish
	return nil                                                  // Return nil (normal completion)
}`,
			hint1: `Use context.WithCancel(ctx) to create a cancelable context, then pass it to f. Always call cancel() when done.`,
			hint2: `Check ctx.Err() to see if parent context is canceled. Use a done channel to wait for f to complete before returning.`,
			whyItMatters: `DoWithCancel provides controlled cancellation of operations, essential for graceful shutdowns and resource cleanup.

**Why Cancellation:**
- **Graceful Shutdown:** Stop work cleanly when shutting down
- **Resource Cleanup:** Release resources properly
- **Request Cancellation:** Stop processing when client disconnects
- **Cascade Cancellation:** Parent cancellation propagates to children

**Production Pattern:**
\`\`\`go
// HTTP request handler with cancellation
func HandleRequest(w http.ResponseWriter, r *http.Request) {
    err := DoWithCancel(r.Context(), func(ctx context.Context) {
        // If client disconnects, ctx will be canceled
        data := fetchData(ctx)
        processData(ctx, data)
    })

    if err != nil {
        http.Error(w, "Request canceled", http.StatusRequestTimeout)
    }
}

// Worker that can be stopped
type Worker struct {
    cancel context.CancelFunc
}

func (w *Worker) Start() error {
    ctx, cancel := context.WithCancel(context.Background())
    w.cancel = cancel

    return DoWithCancel(ctx, func(ctx context.Context) {
        for {
            select {
            case <-ctx.Done():
                cleanup()
                return
            default:
                work := getWork()
                processWork(ctx, work)
            }
        }
    })
}

func (w *Worker) Stop() {
    if w.cancel != nil {
        w.cancel()
    }
}

// Background job with cancellation
func RunBackgroundJob(ctx context.Context) error {
    return DoWithCancel(ctx, func(ctx context.Context) {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()

        for {
            select {
            case <-ctx.Done():
                log.Println("Job canceled, cleaning up...")
                return
            case <-ticker.C:
                performTask()
            }
        }
    })
}

// Database operation with cancellation
func QueryDatabase(ctx context.Context, query string) ([]Row, error) {
    var rows []Row

    err := DoWithCancel(ctx, func(ctx context.Context) {
        conn := db.GetConnection()
        defer conn.Close()

        // Query respects context cancellation
        rows = conn.QueryContext(ctx, query)
    })

    return rows, err
}

// Multi-stage pipeline with cancellation
func ProcessPipeline(ctx context.Context, data []Item) error {
    return DoWithCancel(ctx, func(ctx context.Context) {
        ch1 := stage1(ctx, data)
        ch2 := stage2(ctx, ch1)
        ch3 := stage3(ctx, ch2)

        for result := range ch3 {
            select {
            case <-ctx.Done():
                return
            default:
                saveResult(result)
            }
        }
    })
}

// Graceful server shutdown
type Server struct {
    ctx    context.Context
    cancel context.CancelFunc
}

func (s *Server) Start() error {
    s.ctx, s.cancel = context.WithCancel(context.Background())

    return DoWithCancel(s.ctx, func(ctx context.Context) {
        // Run server until canceled
        runServer(ctx)
    })
}

func (s *Server) Shutdown() error {
    s.cancel() // Triggers graceful shutdown
    return nil
}
\`\`\`

**Real-World Benefits:**
- **Clean Shutdown:** Operations stop gracefully
- **Resource Safety:** No resource leaks
- **Client Respect:** Stop work when client disconnects
- **Cascade Control:** Parent controls all children

**Common Use Cases:**
- **HTTP Handlers:** Cancel on client disconnect
- **Workers:** Stop background workers
- **Pipelines:** Cancel entire processing pipeline
- **Batch Jobs:** Stop job processing gracefully
- **Long Operations:** Allow early termination

**Cancellation Patterns:**
- **Check ctx.Done():** In loops, check for cancellation
- **Pass Context:** Always pass context down call chain
- **Cleanup:** Use defer for cleanup on cancellation
- **Don't Ignore:** Always respect context cancellation

Without DoWithCancel, stopping operations cleanly becomes difficult, leading to resource leaks and zombie goroutines.`,
	testCode: `package concurrency

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	executed := false
	err := DoWithCancel(context.Background(), func(ctx context.Context) { executed = true })
	if err != nil { t.Errorf("expected nil error, got %v", err) }
	if !executed { t.Error("expected function to be executed") }
}

func Test2(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := DoWithCancel(ctx, func(ctx context.Context) { time.Sleep(10*time.Millisecond) })
	if err == nil { t.Error("expected error for canceled parent context") }
}

func Test3(t *testing.T) {
	executed := false
	err := DoWithCancel(nil, func(ctx context.Context) { executed = true })
	if err != nil { t.Errorf("expected nil error for nil context, got %v", err) }
	if !executed { t.Error("expected function to execute with nil context") }
}

func Test4(t *testing.T) {
	canceled := false
	_ = DoWithCancel(context.Background(), func(ctx context.Context) {
		select {
		case <-ctx.Done(): canceled = true
		case <-time.After(100*time.Millisecond):
		}
	})
	time.Sleep(50*time.Millisecond)
}

func Test5(t *testing.T) {
	err := DoWithCancel(context.Background(), func(ctx context.Context) {})
	if err != nil { t.Errorf("expected nil for instant completion, got %v", err) }
}

func Test6(t *testing.T) {
	var counter int64
	for i := 0; i < 10; i++ {
		go func() {
			_ = DoWithCancel(context.Background(), func(ctx context.Context) { atomic.AddInt64(&counter, 1) })
		}()
	}
	time.Sleep(100*time.Millisecond)
	if atomic.LoadInt64(&counter) != 10 { t.Errorf("expected 10 executions, got %d", counter) }
}

func Test7(t *testing.T) {
	receivedCtx := false
	_ = DoWithCancel(context.Background(), func(ctx context.Context) {
		if ctx != nil { receivedCtx = true }
	})
	if !receivedCtx { t.Error("expected function to receive non-nil context") }
}

func Test8(t *testing.T) {
	start := time.Now()
	_ = DoWithCancel(context.Background(), func(ctx context.Context) { time.Sleep(50*time.Millisecond) })
	elapsed := time.Since(start)
	if elapsed < 40*time.Millisecond { t.Error("function should have waited for completion") }
}

func Test9(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	err := DoWithCancel(ctx, func(ctx context.Context) {
		<-ctx.Done()
	})
	if err != nil { t.Logf("parent timeout propagated: %v", err) }
}

func Test10(t *testing.T) {
	ctxChecked := false
	_ = DoWithCancel(context.Background(), func(ctx context.Context) {
		if ctx.Err() == nil { ctxChecked = true }
	})
	if !ctxChecked { t.Error("expected context to be valid during execution") }
}
`,
	order: 2,
	translations: {
		ru: {
			title: 'Выполнение с отменой',
			description: `Реализуйте **DoWithCancel**, который запускает функцию f с отменяемым контекстом, обеспечивая правильную очистку.

**Требования:**
1. Создайте функцию \`DoWithCancel(ctx context.Context, f func(context.Context)) error\`
2. Обработайте nil context (используйте Background)
3. Создайте отменяемый контекст используя context.WithCancel
4. Передайте отменяемый контекст в f
5. Запустите f в горутине
6. Проверьте не отменён ли уже родительский контекст
7. Всегда отменяйте контекст и ждите завершения f
8. Верните ошибку родительского контекста если он был отменён

**Пример:**
\`\`\`go
err := DoWithCancel(ctx, func(ctx context.Context) {
    for {
        select {
        case <-ctx.Done():
            return
        default:
            // Выполнение работы
        }
    }
})
// err = nil (нормальное завершение)

parentCtx, cancel := context.WithCancel(context.Background())
cancel() // Отмена родителя
err = DoWithCancel(parentCtx, func(ctx context.Context) {
    // Получит сигнал отмены
})
// err = context.Canceled
\`\`\`

**Ограничения:**
- Должен использовать context.WithCancel
- Должен передавать отменяемый контекст в f
- Должен ждать завершения f перед возвратом
- Должен обрабатывать отмену родительского контекста`,
			hint1: `Используйте context.WithCancel(ctx) для создания отменяемого контекста, затем передайте его в f. Всегда вызывайте cancel() при завершении.`,
			hint2: `Проверьте ctx.Err() чтобы узнать отменён ли родительский контекст. Используйте done канал чтобы дождаться завершения f перед возвратом.`,
			whyItMatters: `DoWithCancel обеспечивает контролируемую отмену операций, необходимо для graceful shutdown и очистки ресурсов.

**Почему Cancellation:**
- **Graceful Shutdown:** Чистая остановка работы при выключении
- **Очистка ресурсов:** Правильное освобождение ресурсов
- **Отмена запросов:** Остановка обработки при отключении клиента
- **Каскадная отмена:** Отмена родителя распространяется на детей

**Продакшен паттерн:**
\`\`\`go
// HTTP request обработчик с отменой
func HandleRequest(w http.ResponseWriter, r *http.Request) {
    err := DoWithCancel(r.Context(), func(ctx context.Context) {
        // Если клиент отключается, ctx будет отменён
        data := fetchData(ctx)
        processData(ctx, data)
    })

    if err != nil {
        http.Error(w, "Request canceled", http.StatusRequestTimeout)
    }
}

// Worker который можно остановить
type Worker struct {
    cancel context.CancelFunc
}

func (w *Worker) Start() error {
    ctx, cancel := context.WithCancel(context.Background())
    w.cancel = cancel

    return DoWithCancel(ctx, func(ctx context.Context) {
        for {
            select {
            case <-ctx.Done():
                cleanup()
                return
            default:
                work := getWork()
                processWork(ctx, work)
            }
        }
    })
}

func (w *Worker) Stop() {
    if w.cancel != nil {
        w.cancel()
    }
}

// Фоновая задача с отменой
func RunBackgroundJob(ctx context.Context) error {
    return DoWithCancel(ctx, func(ctx context.Context) {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()

        for {
            select {
            case <-ctx.Done():
                log.Println("Job canceled, cleaning up...")
                return
            case <-ticker.C:
                performTask()
            }
        }
    })
}

// Операция с БД с отменой
func QueryDatabase(ctx context.Context, query string) ([]Row, error) {
    var rows []Row

    err := DoWithCancel(ctx, func(ctx context.Context) {
        conn := db.GetConnection()
        defer conn.Close()

        // Запрос учитывает отмену контекста
        rows = conn.QueryContext(ctx, query)
    })

    return rows, err
}

// Многоэтапный pipeline с отменой
func ProcessPipeline(ctx context.Context, data []Item) error {
    return DoWithCancel(ctx, func(ctx context.Context) {
        ch1 := stage1(ctx, data)
        ch2 := stage2(ctx, ch1)
        ch3 := stage3(ctx, ch2)

        for result := range ch3 {
            select {
            case <-ctx.Done():
                return
            default:
                saveResult(result)
            }
        }
    })
}

// Graceful остановка сервера
type Server struct {
    ctx    context.Context
    cancel context.CancelFunc
}

func (s *Server) Start() error {
    s.ctx, s.cancel = context.WithCancel(context.Background())

    return DoWithCancel(s.ctx, func(ctx context.Context) {
        // Запуск сервера до отмены
        runServer(ctx)
    })
}

func (s *Server) Shutdown() error {
    s.cancel() // Запускает graceful shutdown
    return nil
}
\`\`\`

**Практические преимущества:**
- **Чистая остановка:** Операции останавливаются gracefully
- **Безопасность ресурсов:** Нет утечек ресурсов
- **Уважение клиента:** Остановка работы при отключении клиента
- **Каскадный контроль:** Родитель контролирует всех детей

**Обычные случаи использования:**
- **HTTP Handlers:** Отмена при отключении клиента
- **Workers:** Остановка фоновых workers
- **Pipelines:** Отмена всего processing pipeline
- **Batch Jobs:** Graceful остановка обработки задач
- **Долгие операции:** Возможность ранней остановки

**Паттерны отмены:**
- **Проверка ctx.Done():** В циклах проверяйте отмену
- **Передача Context:** Всегда передавайте контекст вниз по цепочке вызовов
- **Cleanup:** Используйте defer для очистки при отмене
- **Не игнорируйте:** Всегда учитывайте отмену контекста

Без DoWithCancel чистая остановка операций становится сложной, приводя к утечкам ресурсов и зомби-горутинам.`,
			solutionCode: `package concurrency

import (
	"context"
)

func DoWithCancel(ctx context.Context, f func(context.Context)) error {
	if ctx == nil {                                             // Обработка nil контекста
		ctx = context.Background()                          // Используем Background как fallback
	}
	ctxWithCancel, cancel := context.WithCancel(ctx)           // Создаём отменяемый контекст
	done := make(chan struct{})                                 // Канал для сигнала завершения
	go func() {                                                 // Запускаем f в горутине
		defer close(done)                                   // Закрываем канал при завершении
		f(ctxWithCancel)                                    // Передаём отменяемый контекст в f
	}()
	if err := ctx.Err(); err != nil {                           // Проверяем не отменён ли уже родитель
		cancel()                                            // Отменяем дочерний контекст
		<-done                                              // Ждём завершения горутины
		return err                                          // Возвращаем ошибку родителя
	}
	cancel()                                                    // Всегда отменяем для освобождения ресурсов
	<-done                                                      // Ждём завершения горутины
	return nil                                                  // Возвращаем nil (нормальное завершение)
}`
		},
		uz: {
			title: 'Bekor qilish bilan bajarish',
			description: `To'g'ri tozalashni ta'minlab, bekor qilinadigan kontekst bilan f funksiyasini ishga tushiradigan **DoWithCancel** ni amalga oshiring.

**Talablar:**
1. \`DoWithCancel(ctx context.Context, f func(context.Context)) error\` funksiyasini yarating
2. nil kontekstni ishlang (Background dan foydalaning)
3. context.WithCancel dan foydalanib bekor qilinadigan kontekst yarating
4. Bekor qilinadigan kontekstni f ga o'tkazing
5. f ni goroutinada ishga tushiring
6. Ota kontekst allaqachon bekor qilinganligini tekshiring
7. Har doim kontekstni bekor qiling va f tugashini kuting
8. Agar ota kontekst bekor qilingan bo'lsa uning xatosini qaytaring

**Misol:**
\`\`\`go
err := DoWithCancel(ctx, func(ctx context.Context) {
    for {
        select {
        case <-ctx.Done():
            return
        default:
            // Ishni bajarish
        }
    }
})
// err = nil (oddiy tugash)

parentCtx, cancel := context.WithCancel(context.Background())
cancel() // Otani bekor qilish
err = DoWithCancel(parentCtx, func(ctx context.Context) {
    // Bekor qilish signalini oladi
})
// err = context.Canceled
\`\`\`

**Cheklovlar:**
- context.WithCancel dan foydalanishi kerak
- Bekor qilinadigan kontekstni f ga o'tkazishi kerak
- Qaytishdan oldin f tugashini kutishi kerak
- Ota kontekst bekor qilishni ishlashi kerak`,
			hint1: `Bekor qilinadigan kontekst yaratish uchun context.WithCancel(ctx) dan foydalaning, keyin uni f ga o'tkazing. Tugaganda har doim cancel() ni chaqiring.`,
			hint2: `Ota kontekst bekor qilinganligini bilish uchun ctx.Err() ni tekshiring. Qaytishdan oldin f tugashini kutish uchun done kanalidan foydalaning.`,
			whyItMatters: `DoWithCancel operatsiyalarni nazorat ostida bekor qilishni ta'minlaydi, graceful shutdown va resurslarni tozalash uchun zarur.

**Nima uchun Cancellation:**
- **Graceful Shutdown:** O'chirish paytida ishni toza to'xtatish
- **Resurslarni tozalash:** Resurslarni to'g'ri ozod qilish
- **So'rovni bekor qilish:** Mijoz uzilib qolganda qayta ishlashni to'xtatish
- **Kaskad bekor qilish:** Ota bekor qilish bolalarga tarqaladi

**Ishlab chiqarish patterni:**
\`\`\`go
// Bekor qilish bilan HTTP request handleri
func HandleRequest(w http.ResponseWriter, r *http.Request) {
    err := DoWithCancel(r.Context(), func(ctx context.Context) {
        // Agar mijoz uzilsa, ctx bekor qilinadi
        data := fetchData(ctx)
        processData(ctx, data)
    })

    if err != nil {
        http.Error(w, "Request canceled", http.StatusRequestTimeout)
    }
}

// To'xtatilishi mumkin bo'lgan Worker
type Worker struct {
    cancel context.CancelFunc
}

func (w *Worker) Start() error {
    ctx, cancel := context.WithCancel(context.Background())
    w.cancel = cancel

    return DoWithCancel(ctx, func(ctx context.Context) {
        for {
            select {
            case <-ctx.Done():
                cleanup()
                return
            default:
                work := getWork()
                processWork(ctx, work)
            }
        }
    })
}

func (w *Worker) Stop() {
    if w.cancel != nil {
        w.cancel()
    }
}

// Bekor qilish bilan background job
func RunBackgroundJob(ctx context.Context) error {
    return DoWithCancel(ctx, func(ctx context.Context) {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()

        for {
            select {
            case <-ctx.Done():
                log.Println("Job canceled, cleaning up...")
                return
            case <-ticker.C:
                performTask()
            }
        }
    })
}

// Bekor qilish bilan ma'lumotlar bazasi operatsiyasi
func QueryDatabase(ctx context.Context, query string) ([]Row, error) {
    var rows []Row

    err := DoWithCancel(ctx, func(ctx context.Context) {
        conn := db.GetConnection()
        defer conn.Close()

        // So'rov kontekst bekor qilinishini hisobga oladi
        rows = conn.QueryContext(ctx, query)
    })

    return rows, err
}

// Bekor qilish bilan ko'p bosqichli pipeline
func ProcessPipeline(ctx context.Context, data []Item) error {
    return DoWithCancel(ctx, func(ctx context.Context) {
        ch1 := stage1(ctx, data)
        ch2 := stage2(ctx, ch1)
        ch3 := stage3(ctx, ch2)

        for result := range ch3 {
            select {
            case <-ctx.Done():
                return
            default:
                saveResult(result)
            }
        }
    })
}

// Graceful server to'xtatish
type Server struct {
    ctx    context.Context
    cancel context.CancelFunc
}

func (s *Server) Start() error {
    s.ctx, s.cancel = context.WithCancel(context.Background())

    return DoWithCancel(s.ctx, func(ctx context.Context) {
        // Bekor qilinguncha serverni ishga tushirish
        runServer(ctx)
    })
}

func (s *Server) Shutdown() error {
    s.cancel() // Graceful shutdownni boshlaydi
    return nil
}
\`\`\`

**Amaliy foydalari:**
- **Toza to'xtatish:** Operatsiyalar gracefully to'xtatiladi
- **Resurs xavfsizligi:** Resurs oqishi yo'q
- **Mijozni hurmat qilish:** Mijoz uzilganda ishni to'xtatish
- **Kaskad nazorati:** Ota barcha bolalarni nazorat qiladi

**Oddiy foydalanish holatlari:**
- **HTTP Handlers:** Mijoz uzilganda bekor qilish
- **Workers:** Background workerlarni to'xtatish
- **Pipelines:** Butun qayta ishlash pipelineni bekor qilish
- **Batch Jobs:** Vazifalarni qayta ishlashni gracefully to'xtatish
- **Uzoq operatsiyalar:** Erta to'xtatish imkoniyati

**Bekor qilish patternlari:**
- **ctx.Done() tekshirish:** Sikllarda bekor qilinishni tekshiring
- **Kontekstni uzatish:** Har doim kontekstni chaqiruvlar zanjiridan pastga o'tkazing
- **Cleanup:** Bekor qilinganda tozalash uchun defer dan foydalaning
- **E'tibor bermang:** Har doim kontekst bekor qilinishini hisobga oling

DoWithCancel bo'lmasa, operatsiyalarni toza to'xtatish qiyin bo'lib, resurs oqishi va zombi goroutinalarga olib keladi.`,
			solutionCode: `package concurrency

import (
	"context"
)

func DoWithCancel(ctx context.Context, f func(context.Context)) error {
	if ctx == nil {                                             // nil kontekstni ishlash
		ctx = context.Background()                          // Fallback sifatida Background ishlatamiz
	}
	ctxWithCancel, cancel := context.WithCancel(ctx)           // Bekor qilinadigan kontekst yaratamiz
	done := make(chan struct{})                                 // Tugash signali uchun kanal
	go func() {                                                 // f ni goroutinada ishga tushiramiz
		defer close(done)                                   // Tugaganda kanalni yopamiz
		f(ctxWithCancel)                                    // Bekor qilinadigan kontekstni f ga uzatamiz
	}()
	if err := ctx.Err(); err != nil {                           // Ota allaqachon bekor qilinganligini tekshiramiz
		cancel()                                            // Bola kontekstini bekor qilamiz
		<-done                                              // Goroutina tugashini kutamiz
		return err                                          // Ota xatosini qaytaramiz
	}
	cancel()                                                    // Resurslarni ozod qilish uchun har doim bekor qilamiz
	<-done                                                      // Goroutina tugashini kutamiz
	return nil                                                  // nil qaytaramiz (oddiy tugash)
}`
		}
	}
};

export default task;
