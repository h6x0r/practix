import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-concurrency-do-with-deadline',
	title: 'Do With Deadline',
	difficulty: 'easy',	tags: ['go', 'concurrency', 'context', 'deadline'],
	estimatedTime: '15m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **DoWithDeadline** that runs function f in a goroutine and waits for completion or absolute deadline.

**Requirements:**
1. Create function \`DoWithDeadline(ctx context.Context, deadline time.Time, f func()) error\`
2. Handle nil context (use Background)
3. Create context with deadline using context.WithDeadline
4. Run f in a goroutine
5. Wait for either completion or deadline
6. Return context error if deadline is reached
7. Return nil if function completes successfully before deadline

**Example:**
\`\`\`go
deadline := time.Now().Add(200 * time.Millisecond)
err := DoWithDeadline(ctx, deadline, func() {
    time.Sleep(100 * time.Millisecond)
    fmt.Println("Task completed")
})
// err = nil (completed before deadline)

deadline = time.Now().Add(100 * time.Millisecond)
err = DoWithDeadline(ctx, deadline, func() {
    time.Sleep(300 * time.Millisecond)
})
// err = context.DeadlineExceeded
\`\`\`

**Constraints:**
- Must use context.WithDeadline
- Must run f in separate goroutine
- Must wait for completion or deadline`,
	initialCode: `package concurrency

import (
	"context"
	"time"
)

// TODO: Implement DoWithDeadline
func DoWithDeadline(ctx context.Context, deadline time.Time, f func()) error {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
	"time"
)

func DoWithDeadline(ctx context.Context, deadline time.Time, f func()) error {
	if ctx == nil {                                                 // Handle nil context
		ctx = context.Background()                              // Use Background as fallback
	}
	ctxWithDeadline, cancel := context.WithDeadline(ctx, deadline) // Create deadline context
	defer cancel()                                                  // Always cancel to free resources
	done := make(chan struct{})                                     // Channel to signal completion
	go func() {                                                     // Run f in goroutine
		defer close(done)                                       // Close channel when done
		f()                                                     // Execute function
	}()
	select {
	case <-ctxWithDeadline.Done():                                  // Deadline reached
		return ctxWithDeadline.Err()                            // Return deadline error
	case <-done:                                                    // Function completed
		return ctxWithDeadline.Err()                            // Return nil (no deadline)
	}
}`,
			hint1: `Use context.WithDeadline(ctx, deadline) to create a context that automatically cancels at the specified time.`,
			hint2: `Create a done channel, run f() in a goroutine that closes the channel when complete, then select between ctx.Done() and done.`,
			whyItMatters: `DoWithDeadline enforces absolute time boundaries on operations, essential for scheduled tasks and time-sensitive operations.

**Why Deadlines:**
- **Absolute Time:** Set specific completion time (e.g., "must finish by 3 PM")
- **Coordinated Tasks:** Synchronize operations across distributed systems
- **SLA Enforcement:** Meet strict service-level agreement times
- **Batch Processing:** Complete jobs before next batch window

**Production Pattern:**
\`\`\`go
// Process batch job before cutoff time
func ProcessBatchJob(data []Record) error {
    // Must complete by end of business day
    deadline := time.Date(2024, 1, 1, 17, 0, 0, 0, time.Local)

    ctx := context.Background()
    return DoWithDeadline(ctx, deadline, func() {
        for _, record := range data {
            processRecord(record)
        }
    })
}

// Time-sensitive API request
func FetchMarketData(symbol string) (*Quote, error) {
    // Market closes at 4 PM
    marketClose := time.Date(2024, 1, 1, 16, 0, 0, 0, time.Local)

    ctx := context.Background()
    var quote *Quote

    err := DoWithDeadline(ctx, marketClose, func() {
        quote = fetchQuote(symbol)
    })

    return quote, err
}

// Scheduled maintenance window
func RunMaintenance() error {
    // Maintenance window ends at 2 AM
    maintenanceEnd := time.Date(2024, 1, 1, 2, 0, 0, 0, time.Local)

    return DoWithDeadline(context.Background(), maintenanceEnd, func() {
        cleanupOldData()
        rebuildIndexes()
        optimizeTables()
    })
}

// Report generation deadline
func GenerateMonthlyReport(month time.Month) error {
    // Report due at start of next month
    deadline := time.Date(2024, month+1, 1, 0, 0, 0, 0, time.UTC)

    return DoWithDeadline(context.Background(), deadline, func() {
        data := gatherMonthlyData(month)
        report := buildReport(data)
        submitReport(report)
    })
}

// Distributed task coordination
func CoordinatedTask(taskID string, globalDeadline time.Time) error {
    // All nodes must complete before global deadline
    ctx := context.Background()

    return DoWithDeadline(ctx, globalDeadline, func() {
        result := performWork(taskID)
        reportToCoordinator(taskID, result)
    })
}

// Cache refresh with deadline
func RefreshCacheBeforeExpiry(cacheKey string, expiresAt time.Time) error {
    // Refresh must complete before cache expires
    refreshDeadline := expiresAt.Add(-10 * time.Second)

    return DoWithDeadline(context.Background(), refreshDeadline, func() {
        data := fetchFreshData(cacheKey)
        updateCache(cacheKey, data)
    })
}
\`\`\`

**Real-World Benefits:**
- **Time Guarantees:** Operations respect absolute time boundaries
- **Coordination:** Multiple systems can share same deadline
- **Resource Planning:** Know exactly when resources will be freed
- **Compliance:** Meet regulatory deadlines

**Common Use Cases:**
- **Batch Jobs:** Complete before next batch starts
- **Market Operations:** Finish before market closes
- **Reporting:** Generate reports by deadline
- **Maintenance Windows:** Complete maintenance in time window
- **Cache Updates:** Refresh before expiry

**Deadline vs Timeout:**
- **Timeout:** Relative duration (run for 5 seconds)
- **Deadline:** Absolute time (finish by 3 PM)
- Use deadline when external time matters
- Use timeout when operation duration matters

Without DoWithDeadline, coordinating time-sensitive operations across systems becomes complex and error-prone.`,
	testCode: `package concurrency

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	executed := false
	deadline := time.Now().Add(100 * time.Millisecond)
	err := DoWithDeadline(context.Background(), deadline, func() { executed = true })
	if err != nil { t.Errorf("expected nil error, got %v", err) }
	if !executed { t.Error("expected function to be executed") }
}

func Test2(t *testing.T) {
	deadline := time.Now().Add(50 * time.Millisecond)
	err := DoWithDeadline(context.Background(), deadline, func() { time.Sleep(200*time.Millisecond) })
	if !errors.Is(err, context.DeadlineExceeded) { t.Errorf("expected DeadlineExceeded, got %v", err) }
}

func Test3(t *testing.T) {
	executed := false
	deadline := time.Now().Add(100 * time.Millisecond)
	err := DoWithDeadline(nil, deadline, func() { executed = true })
	if err != nil { t.Errorf("expected nil error for nil context, got %v", err) }
	if !executed { t.Error("expected function to execute with nil context") }
}

func Test4(t *testing.T) {
	deadline := time.Now().Add(-1 * time.Second)
	err := DoWithDeadline(context.Background(), deadline, func() { time.Sleep(10*time.Millisecond) })
	if !errors.Is(err, context.DeadlineExceeded) { t.Errorf("expected DeadlineExceeded for past deadline, got %v", err) }
}

func Test5(t *testing.T) {
	start := time.Now()
	deadline := time.Now().Add(50 * time.Millisecond)
	_ = DoWithDeadline(context.Background(), deadline, func() { time.Sleep(500*time.Millisecond) })
	elapsed := time.Since(start)
	if elapsed > 150*time.Millisecond { t.Error("deadline did not work, took too long") }
}

func Test6(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	deadline := time.Now().Add(100 * time.Millisecond)
	err := DoWithDeadline(ctx, deadline, func() { time.Sleep(10*time.Millisecond) })
	if err == nil { t.Error("expected error for canceled parent context") }
}

func Test7(t *testing.T) {
	var counter int64
	deadline := time.Now().Add(200 * time.Millisecond)
	for i := 0; i < 10; i++ {
		go func() {
			_ = DoWithDeadline(context.Background(), deadline, func() { atomic.AddInt64(&counter, 1) })
		}()
	}
	time.Sleep(100*time.Millisecond)
	if atomic.LoadInt64(&counter) != 10 { t.Errorf("expected 10 executions, got %d", counter) }
}

func Test8(t *testing.T) {
	deadline := time.Now().Add(1 * time.Second)
	err := DoWithDeadline(context.Background(), deadline, func() {})
	if err != nil { t.Errorf("expected nil for instant completion, got %v", err) }
}

func Test9(t *testing.T) {
	deadline := time.Now()
	err := DoWithDeadline(context.Background(), deadline, func() { time.Sleep(10*time.Millisecond) })
	if err == nil || !errors.Is(err, context.DeadlineExceeded) { t.Errorf("expected DeadlineExceeded for now deadline, got %v", err) }
}

func Test10(t *testing.T) {
	executed := false
	deadline := time.Now().Add(500 * time.Millisecond)
	err := DoWithDeadline(context.Background(), deadline, func() { time.Sleep(10*time.Millisecond); executed = true })
	if err != nil { t.Errorf("expected nil, got %v", err) }
	if !executed { t.Error("expected execution before deadline") }
}
`,
	order: 1,
	translations: {
		ru: {
			title: 'Выполнение с дедлайном',
			description: `Реализуйте **DoWithDeadline**, который запускает функцию f в горутине и ждёт завершения или абсолютного дедлайна.

**Требования:**
1. Создайте функцию \`DoWithDeadline(ctx context.Context, deadline time.Time, f func()) error\`
2. Обработайте nil context (используйте Background)
3. Создайте контекст с дедлайном используя context.WithDeadline
4. Запустите f в горутине
5. Ждите завершения или дедлайна
6. Верните ошибку контекста если дедлайн достигнут
7. Верните nil если функция завершилась успешно до дедлайна

**Пример:**
\`\`\`go
deadline := time.Now().Add(200 * time.Millisecond)
err := DoWithDeadline(ctx, deadline, func() {
    time.Sleep(100 * time.Millisecond)
    fmt.Println("Task completed")
})
// err = nil (завершилось до дедлайна)

deadline = time.Now().Add(100 * time.Millisecond)
err = DoWithDeadline(ctx, deadline, func() {
    time.Sleep(300 * time.Millisecond)
})
// err = context.DeadlineExceeded
\`\`\`

**Ограничения:**
- Должен использовать context.WithDeadline
- Должен запускать f в отдельной горутине
- Должен ждать завершения или дедлайна`,
			hint1: `Используйте context.WithDeadline(ctx, deadline) для создания контекста который автоматически отменяется в указанное время.`,
			hint2: `Создайте done канал, запустите f() в горутине которая закрывает канал при завершении, затем select между ctx.Done() и done.`,
			whyItMatters: `DoWithDeadline устанавливает абсолютные временные границы операций, необходимо для запланированных задач и критичных по времени операций.

**Почему Deadlines:**
- **Абсолютное время:** Установка конкретного времени завершения (например "должно завершиться к 15:00")
- **Координация задач:** Синхронизация операций в распределённых системах
- **SLA:** Соблюдение строгих временных рамок соглашений об уровне обслуживания
- **Batch обработка:** Завершение задач до следующего окна

**Продакшен паттерн:**
\`\`\`go
// Обработка batch задачи до времени окончания
func ProcessBatchJob(data []Record) error {
    // Должно завершиться до конца рабочего дня
    deadline := time.Date(2024, 1, 1, 17, 0, 0, 0, time.Local)

    ctx := context.Background()
    return DoWithDeadline(ctx, deadline, func() {
        for _, record := range data {
            processRecord(record)
        }
    })
}

// Time-sensitive API запрос
func FetchMarketData(symbol string) (*Quote, error) {
    // Рынок закрывается в 16:00
    marketClose := time.Date(2024, 1, 1, 16, 0, 0, 0, time.Local)

    ctx := context.Background()
    var quote *Quote

    err := DoWithDeadline(ctx, marketClose, func() {
        quote = fetchQuote(symbol)
    })

    return quote, err
}

// Окно обслуживания по расписанию
func RunMaintenance() error {
    // Окно обслуживания заканчивается в 2:00
    maintenanceEnd := time.Date(2024, 1, 1, 2, 0, 0, 0, time.Local)

    return DoWithDeadline(context.Background(), maintenanceEnd, func() {
        cleanupOldData()
        rebuildIndexes()
        optimizeTables()
    })
}

// Дедлайн генерации отчёта
func GenerateMonthlyReport(month time.Month) error {
    // Отчёт должен быть готов к началу следующего месяца
    deadline := time.Date(2024, month+1, 1, 0, 0, 0, 0, time.UTC)

    return DoWithDeadline(context.Background(), deadline, func() {
        data := gatherMonthlyData(month)
        report := buildReport(data)
        submitReport(report)
    })
}

// Координация распределённых задач
func CoordinatedTask(taskID string, globalDeadline time.Time) error {
    // Все узлы должны завершиться до глобального дедлайна
    ctx := context.Background()

    return DoWithDeadline(ctx, globalDeadline, func() {
        result := performWork(taskID)
        reportToCoordinator(taskID, result)
    })
}

// Обновление кеша с дедлайном
func RefreshCacheBeforeExpiry(cacheKey string, expiresAt time.Time) error {
    // Обновление должно завершиться до истечения кеша
    refreshDeadline := expiresAt.Add(-10 * time.Second)

    return DoWithDeadline(context.Background(), refreshDeadline, func() {
        data := fetchFreshData(cacheKey)
        updateCache(cacheKey, data)
    })
}
\`\`\`

**Практические преимущества:**
- **Гарантии времени:** Операции соблюдают абсолютные временные границы
- **Координация:** Множество систем могут делить один дедлайн
- **Планирование ресурсов:** Точно известно когда ресурсы будут освобождены
- **Соответствие:** Соблюдение регуляторных дедлайнов

**Обычные случаи использования:**
- **Batch задачи:** Завершение до начала следующего batch
- **Рыночные операции:** Завершение до закрытия рынка
- **Отчётность:** Генерация отчётов по дедлайну
- **Окна обслуживания:** Завершение обслуживания в временном окне
- **Обновление кеша:** Обновление до истечения

**Deadline vs Timeout:**
- **Timeout:** Относительная длительность (выполнять 5 секунд)
- **Deadline:** Абсолютное время (закончить к 15:00)
- Используйте deadline когда важно внешнее время
- Используйте timeout когда важна длительность операции

Без DoWithDeadline координация критичных по времени операций между системами становится сложной и подверженной ошибкам.`,
			solutionCode: `package concurrency

import (
	"context"
	"time"
)

func DoWithDeadline(ctx context.Context, deadline time.Time, f func()) error {
	if ctx == nil {                                                 // Обработка nil контекста
		ctx = context.Background()                              // Используем Background как fallback
	}
	ctxWithDeadline, cancel := context.WithDeadline(ctx, deadline) // Создаём контекст с дедлайном
	defer cancel()                                                  // Всегда отменяем для освобождения ресурсов
	done := make(chan struct{})                                     // Канал для сигнала завершения
	go func() {                                                     // Запускаем f в горутине
		defer close(done)                                       // Закрываем канал при завершении
		f()                                                     // Выполняем функцию
	}()
	select {
	case <-ctxWithDeadline.Done():                                  // Дедлайн достигнут
		return ctxWithDeadline.Err()                            // Возвращаем ошибку дедлайна
	case <-done:                                                    // Функция завершилась
		return ctxWithDeadline.Err()                            // Возвращаем nil (без дедлайна)
	}
}`
		},
		uz: {
			title: 'Deadline bilan bajarish',
			description: `Goroutinada f funksiyasini ishga tushiradigan va tugash yoki mutlaq deadlineni kutadigan **DoWithDeadline** ni amalga oshiring.

**Talablar:**
1. \`DoWithDeadline(ctx context.Context, deadline time.Time, f func()) error\` funksiyasini yarating
2. nil kontekstni ishlang (Background dan foydalaning)
3. context.WithDeadline dan foydalanib deadline bilan kontekst yarating
4. f ni goroutinada ishga tushiring
5. Tugash yoki deadlineni kuting
6. Agar deadline yetib kelsa kontekst xatosini qaytaring
7. Agar funksiya deadlinedan oldin muvaffaqiyatli tugasa nil qaytaring

**Misol:**
\`\`\`go
deadline := time.Now().Add(200 * time.Millisecond)
err := DoWithDeadline(ctx, deadline, func() {
    time.Sleep(100 * time.Millisecond)
    fmt.Println("Task completed")
})
// err = nil (deadlinedan oldin tugadi)

deadline = time.Now().Add(100 * time.Millisecond)
err = DoWithDeadline(ctx, deadline, func() {
    time.Sleep(300 * time.Millisecond)
})
// err = context.DeadlineExceeded
\`\`\`

**Cheklovlar:**
- context.WithDeadline dan foydalanishi kerak
- f ni alohida goroutinada ishga tushirishi kerak
- Tugash yoki deadlineni kutishi kerak`,
			hint1: `Belgilangan vaqtda avtomatik bekor qilinadigan kontekst yaratish uchun context.WithDeadline(ctx, deadline) dan foydalaning.`,
			hint2: `done kanali yarating, f() ni tugaganda kanalni yopadigan goroutinada ishga tushiring, keyin ctx.Done() va done o'rtasida select qiling.`,
			whyItMatters: `DoWithDeadline operatsiyalarga mutlaq vaqt chegaralarini o'rnatadi, rejalashtirilgan vazifalar va vaqt bo'yicha muhim operatsiyalar uchun zarur.

**Nima uchun Deadlines:**
- **Mutlaq vaqt:** Aniq tugash vaqtini belgilash (masalan "15:00 gacha tugashi kerak")
- **Koordinatsiya:** Taqsimlangan tizimlarda operatsiyalarni sinxronlashtirish
- **SLA:** Qat'iy xizmat darajasi kelishuvi vaqtlarini bajarish
- **Batch qayta ishlash:** Keyingi batch oynasidan oldin ishlarni tugatish

**Ishlab chiqarish patterni:**
\`\`\`go
// Tugash vaqtigacha batch vazifasini qayta ishlash
func ProcessBatchJob(data []Record) error {
    // Ish kuni oxirigacha tugashi kerak
    deadline := time.Date(2024, 1, 1, 17, 0, 0, 0, time.Local)

    ctx := context.Background()
    return DoWithDeadline(ctx, deadline, func() {
        for _, record := range data {
            processRecord(record)
        }
    })
}

// Vaqtga sezgir API so'rovi
func FetchMarketData(symbol string) (*Quote, error) {
    // Bozor 16:00 da yopiladi
    marketClose := time.Date(2024, 1, 1, 16, 0, 0, 0, time.Local)

    ctx := context.Background()
    var quote *Quote

    err := DoWithDeadline(ctx, marketClose, func() {
        quote = fetchQuote(symbol)
    })

    return quote, err
}

// Rejalashtiririlgan texnik xizmat oynasi
func RunMaintenance() error {
    // Texnik xizmat oynasi 2:00 da tugaydi
    maintenanceEnd := time.Date(2024, 1, 1, 2, 0, 0, 0, time.Local)

    return DoWithDeadline(context.Background(), maintenanceEnd, func() {
        cleanupOldData()
        rebuildIndexes()
        optimizeTables()
    })
}

// Hisobot yaratish deadlinei
func GenerateMonthlyReport(month time.Month) error {
    // Hisobot keyingi oy boshigacha tayyor bo'lishi kerak
    deadline := time.Date(2024, month+1, 1, 0, 0, 0, 0, time.UTC)

    return DoWithDeadline(context.Background(), deadline, func() {
        data := gatherMonthlyData(month)
        report := buildReport(data)
        submitReport(report)
    })
}

// Taqsimlangan vazifalarni koordinatsiyalash
func CoordinatedTask(taskID string, globalDeadline time.Time) error {
    // Barcha tugunlar global deadlinedan oldin tugashi kerak
    ctx := context.Background()

    return DoWithDeadline(ctx, globalDeadline, func() {
        result := performWork(taskID)
        reportToCoordinator(taskID, result)
    })
}

// Deadlineli keshni yangilash
func RefreshCacheBeforeExpiry(cacheKey string, expiresAt time.Time) error {
    // Yangilash kesh tugashidan oldin tugashi kerak
    refreshDeadline := expiresAt.Add(-10 * time.Second)

    return DoWithDeadline(context.Background(), refreshDeadline, func() {
        data := fetchFreshData(cacheKey)
        updateCache(cacheKey, data)
    })
}
\`\`\`

**Amaliy foydalari:**
- **Vaqt kafolatlari:** Operatsiyalar mutlaq vaqt chegaralariga rioya qiladi
- **Koordinatsiya:** Ko'p tizimlar bir deadlineni bo'lishishi mumkin
- **Resurslarni rejalashtirish:** Resurslar qachon ozod qilinishi aniq ma'lum
- **Muvofiqlik:** Tartibga soluvchi deadlinelarni bajarish

**Oddiy foydalanish holatlari:**
- **Batch vazifalar:** Keyingi batch boshlanishidan oldin tugatish
- **Bozor operatsiyalari:** Bozor yopilishidan oldin tugatish
- **Hisobotlar:** Deadlinedan hisobotlar yaratish
- **Texnik xizmat oynalari:** Vaqt oynasida texnik xizmatni tugatish
- **Keshni yangilash:** Muddati tugashidan oldin yangilash

**Deadline vs Timeout:**
- **Timeout:** Nisbiy davomiylik (5 soniya davomida bajarish)
- **Deadline:** Mutlaq vaqt (15:00 gacha tugatish)
- Tashqi vaqt muhim bo'lganda deadlinedan foydalaning
- Operatsiya davomiyligi muhim bo'lganda timeoutdan foydalaning

DoWithDeadline bo'lmasa, tizimlar o'rtasida vaqtga sezgir operatsiyalarni muvofiqlashtirish murakkab va xatolarga moyil bo'ladi.`,
			solutionCode: `package concurrency

import (
	"context"
	"time"
)

func DoWithDeadline(ctx context.Context, deadline time.Time, f func()) error {
	if ctx == nil {                                                 // nil kontekstni ishlash
		ctx = context.Background()                              // Fallback sifatida Background ishlatamiz
	}
	ctxWithDeadline, cancel := context.WithDeadline(ctx, deadline) // Deadline bilan kontekst yaratamiz
	defer cancel()                                                  // Resurslarni ozod qilish uchun har doim bekor qilamiz
	done := make(chan struct{})                                     // Tugash signali uchun kanal
	go func() {                                                     // f ni goroutinada ishga tushiramiz
		defer close(done)                                       // Tugaganda kanalni yopamiz
		f()                                                     // Funksiyani bajaramiz
	}()
	select {
	case <-ctxWithDeadline.Done():                                  // Deadline yetib keldi
		return ctxWithDeadline.Err()                            // Deadline xatosini qaytaramiz
	case <-done:                                                    // Funksiya tugadi
		return ctxWithDeadline.Err()                            // nil qaytaramiz (deadline yo'q)
	}
}`
		}
	}
};

export default task;
