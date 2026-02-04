import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-concurrency-do-with-timeout',
	title: 'Do With Timeout',
	difficulty: 'easy',	tags: ['go', 'concurrency', 'context', 'timeout'],
	estimatedTime: '15m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **DoWithTimeout** that runs function f in a goroutine and waits for completion or timeout d.

**Requirements:**
1. Create function \`DoWithTimeout(ctx context.Context, f func(), d time.Duration) error\`
2. Handle nil context (use Background)
3. Create context with timeout using context.WithTimeout
4. Run f in a goroutine
5. Wait for either completion or timeout
6. Return context error if timeout occurs
7. Return nil if function completes successfully before timeout

**Example:**
\`\`\`go
err := DoWithTimeout(ctx, func() {
    time.Sleep(100 * time.Millisecond)
    fmt.Println("Task completed")
}, 200 * time.Millisecond)
// err = nil (completed before timeout)

err = DoWithTimeout(ctx, func() {
    time.Sleep(300 * time.Millisecond)
}, 100 * time.Millisecond)
// err = context.DeadlineExceeded
\`\`\`

**Constraints:**
- Must use context.WithTimeout
- Must run f in separate goroutine
- Must wait for completion or timeout`,
	initialCode: `package concurrency

import (
	"context"
	"time"
)

// TODO: Implement DoWithTimeout
func DoWithTimeout(ctx context.Context, f func(), d time.Duration) error {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
	"time"
)

func DoWithTimeout(ctx context.Context, f func(), d time.Duration) error {
	if ctx == nil {                                                 // Handle nil context
		ctx = context.Background()                              // Use Background as fallback
	}
	ctxWithTimeout, cancel := context.WithTimeout(ctx, d)           // Create timeout context
	defer cancel()                                                  // Always cancel to free resources
	done := make(chan struct{})                                     // Channel to signal completion
	go func() {                                                     // Run f in goroutine
		defer close(done)                                       // Close channel when done
		f()                                                     // Execute function
	}()
	select {
	case <-ctxWithTimeout.Done():                                   // Timeout occurred
		return ctxWithTimeout.Err()                             // Return timeout error
	case <-done:                                                    // Function completed
		return ctxWithTimeout.Err()                             // Return nil (no timeout)
	}
}`,
			hint1: `Use context.WithTimeout(ctx, d) to create a context that automatically cancels after duration d.`,
			hint2: `Create a done channel, run f() in a goroutine that closes the channel when complete, then select between ctx.Done() and done.`,
			whyItMatters: `DoWithTimeout prevents functions from running indefinitely, essential for setting time bounds on operations that might hang or take too long.

**Why Timeouts:**
- **Resource Protection:** Prevent goroutines from hanging forever
- **Responsive Systems:** Fail fast instead of waiting indefinitely
- **SLA Compliance:** Enforce maximum operation durations
- **Debugging:** Identify slow operations easily

**Production Pattern:**
\`\`\`go
// API request with timeout
func FetchUserData(userID string) (*User, error) {
    ctx := context.Background()

    var user *User
    err := DoWithTimeout(ctx, func() {
        user = database.GetUser(userID)
    }, 2*time.Second)

    if err != nil {
        return nil, fmt.Errorf("fetch timeout: %w", err)
    }

    return user, nil
}

// External API call with timeout
func CallExternalAPI(url string) ([]byte, error) {
    ctx := context.Background()
    var result []byte

    err := DoWithTimeout(ctx, func() {
        resp, _ := http.Get(url)
        defer resp.Body.Close()
        result, _ = ioutil.ReadAll(resp.Body)
    }, 5*time.Second)

    return result, err
}

// Heavy computation with timeout
func ProcessLargeDataset(data []int) (int, error) {
    ctx := context.Background()
    var sum int

    err := DoWithTimeout(ctx, func() {
        for _, v := range data {
            sum += v * v
            time.Sleep(time.Millisecond) // Simulate heavy work
        }
    }, 10*time.Second)

    return sum, err
}

// Timeout with cleanup
func DoWorkWithCleanup(ctx context.Context, timeout time.Duration) error {
    done := make(chan struct{})
    errChan := make(chan error, 1)

    go func() {
        defer close(done)
        // Perform work...
        errChan <- nil
    }()

    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    select {
    case <-ctx.Done():
        // Cleanup on timeout
        cleanup()
        return ctx.Err()
    case err := <-errChan:
        return err
    }
}

// Different timeouts for different operations
const (
    FastOperationTimeout   = 100 * time.Millisecond
    NormalOperationTimeout = 1 * time.Second
    SlowOperationTimeout   = 10 * time.Second
)

func PerformOperation(opType string) error {
    timeout := NormalOperationTimeout

    switch opType {
    case "cache":
        timeout = FastOperationTimeout
    case "database":
        timeout = NormalOperationTimeout
    case "external_api":
        timeout = SlowOperationTimeout
    }

    return DoWithTimeout(context.Background(), func() {
        // Perform operation
    }, timeout)
}
\`\`\`

**Real-World Benefits:**
- **Prevents Hangs:** Operations can't block indefinitely
- **Resource Management:** Free resources from stalled operations
- **Better UX:** Users get fast failures instead of infinite loading
- **Debugging:** Easily identify which operations are slow

**Common Timeout Values:**
- **Cache Operations:** 50-100ms
- **Database Queries:** 1-5 seconds
- **HTTP Requests:** 5-30 seconds
- **Heavy Processing:** 30-300 seconds

Without DoWithTimeout, a slow or hanging function can block your application indefinitely, consuming resources and degrading user experience.`,
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
	err := DoWithTimeout(context.Background(), func() { executed = true }, 100*time.Millisecond)
	if err != nil { t.Errorf("expected nil error, got %v", err) }
	if !executed { t.Error("expected function to be executed") }
}

func Test2(t *testing.T) {
	err := DoWithTimeout(context.Background(), func() { time.Sleep(200*time.Millisecond) }, 50*time.Millisecond)
	if !errors.Is(err, context.DeadlineExceeded) { t.Errorf("expected DeadlineExceeded, got %v", err) }
}

func Test3(t *testing.T) {
	executed := false
	err := DoWithTimeout(nil, func() { executed = true }, 100*time.Millisecond)
	if err != nil { t.Errorf("expected nil error for nil context, got %v", err) }
	if !executed { t.Error("expected function to execute with nil context") }
}

func Test4(t *testing.T) {
	start := time.Now()
	_ = DoWithTimeout(context.Background(), func() { time.Sleep(500*time.Millisecond) }, 50*time.Millisecond)
	elapsed := time.Since(start)
	if elapsed > 150*time.Millisecond { t.Error("timeout did not work, took too long") }
}

func Test5(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := DoWithTimeout(ctx, func() { time.Sleep(10*time.Millisecond) }, 100*time.Millisecond)
	if err == nil { t.Error("expected error for canceled parent context") }
}

func Test6(t *testing.T) {
	err := DoWithTimeout(context.Background(), func() {}, 1*time.Second)
	if err != nil { t.Errorf("expected nil for instant completion, got %v", err) }
}

func Test7(t *testing.T) {
	var counter int64
	for i := 0; i < 10; i++ {
		go func() {
			_ = DoWithTimeout(context.Background(), func() { atomic.AddInt64(&counter, 1) }, 100*time.Millisecond)
		}()
	}
	time.Sleep(200*time.Millisecond)
	if atomic.LoadInt64(&counter) != 10 { t.Errorf("expected 10 executions, got %d", counter) }
}

func Test8(t *testing.T) {
	err := DoWithTimeout(context.Background(), func() {}, 0)
	if err != nil && !errors.Is(err, context.DeadlineExceeded) { t.Errorf("unexpected error for zero timeout: %v", err) }
}

func Test9(t *testing.T) {
	executed := false
	err := DoWithTimeout(context.Background(), func() { time.Sleep(10*time.Millisecond); executed = true }, 500*time.Millisecond)
	if err != nil { t.Errorf("expected nil, got %v", err) }
	if !executed { t.Error("expected execution with generous timeout") }
}

func Test10(t *testing.T) {
	panicked := false
	func() {
		defer func() { if r := recover(); r != nil { panicked = true } }()
		_ = DoWithTimeout(context.Background(), func() { panic("test") }, 100*time.Millisecond)
	}()
	if !panicked { t.Log("function panic propagates or is handled") }
}
`,
	order: 0,
	translations: {
		ru: {
			title: 'Выполнение операции с ограничением времени',
			description: `Реализуйте **DoWithTimeout**, который запускает функцию f в горутине и ждёт завершения или таймаута d.

**Требования:**
1. Создайте функцию \`DoWithTimeout(ctx context.Context, f func(), d time.Duration) error\`
2. Обработайте nil context (используйте Background)
3. Создайте контекст с таймаутом используя context.WithTimeout
4. Запустите f в горутине
5. Ждите завершения или таймаута
6. Верните ошибку контекста если произошёл таймаут
7. Верните nil если функция завершилась успешно до таймаута

**Пример:**
\`\`\`go
err := DoWithTimeout(ctx, func() {
    time.Sleep(100 * time.Millisecond)
    fmt.Println("Task completed")
}, 200 * time.Millisecond)
// err = nil (завершилось до таймаута)

err = DoWithTimeout(ctx, func() {
    time.Sleep(300 * time.Millisecond)
}, 100 * time.Millisecond)
// err = context.DeadlineExceeded
\`\`\`

**Ограничения:**
- Должен использовать context.WithTimeout
- Должен запускать f в отдельной горутине
- Должен ждать завершения или таймаута`,
			hint1: `Используйте context.WithTimeout(ctx, d) для создания контекста который автоматически отменяется после duration d.`,
			hint2: `Создайте done канал, запустите f() в горутине которая закрывает канал при завершении, затем select между ctx.Done() и done.`,
			whyItMatters: `DoWithTimeout предотвращает бесконечное выполнение функций, необходимо для установки временных границ операций которые могут зависнуть или занять слишком много времени.

**Почему Timeouts критичны:**
- **Защита ресурсов:** Предотвращение вечного зависания горутин
- **Отзывчивые системы:** Быстрый fail вместо бесконечного ожидания
- **SLA соблюдение:** Обеспечение максимальных длительностей операций
- **Отладка:** Лёгкое выявление медленных операций

**Продакшен паттерны:**
\`\`\`go
// API запрос с таймаутом
func FetchUserData(userID string) (*User, error) {
    ctx := context.Background()
    var user *User

    err := DoWithTimeout(ctx, func() {
        user = database.GetUser(userID)
    }, 2*time.Second)

    if err != nil {
        return nil, fmt.Errorf("fetch timeout: %w", err)
    }
    return user, nil
}

// Внешний API вызов с таймаутом
func CallExternalAPI(url string) ([]byte, error) {
    ctx := context.Background()
    var result []byte

    err := DoWithTimeout(ctx, func() {
        resp, _ := http.Get(url)
        defer resp.Body.Close()
        result, _ = ioutil.ReadAll(resp.Body)
    }, 5*time.Second)

    return result, err
}

// Тяжёлые вычисления с таймаутом
func ProcessLargeDataset(data []int) (int, error) {
    ctx := context.Background()
    var sum int

    err := DoWithTimeout(ctx, func() {
        for _, v := range data {
            sum += v * v
            time.Sleep(time.Millisecond)  // Симуляция тяжёлой работы
        }
    }, 10*time.Second)

    return sum, err
}
\`\`\`

**Разные таймауты для разных операций:**
\`\`\`go
const (
    FastOperationTimeout   = 100 * time.Millisecond  // Кеш, память
    NormalOperationTimeout = 1 * time.Second         // БД запросы
    SlowOperationTimeout   = 10 * time.Second        // Внешние API
)

func PerformOperation(opType string) error {
    timeout := NormalOperationTimeout
    switch opType {
    case "cache":
        timeout = FastOperationTimeout
    case "database":
        timeout = NormalOperationTimeout
    case "external_api":
        timeout = SlowOperationTimeout
    }

    return DoWithTimeout(context.Background(), func() {
        // Выполнение операции соответствующего типа
    }, timeout)
}
\`\`\`

**Реальные производственные преимущества:**
- **Предотвращение зависаний:** Операции не могут блокировать бесконечно
- **Управление ресурсами:** Освобождение ресурсов от застрявших операций
- **Лучший UX:** Пользователи получают быстрые ошибки вместо бесконечной загрузки
- **Отладка:** Легко идентифицировать какие операции медленные

**Рекомендуемые значения таймаутов в production:**
- **Кеш операции:** 50-100ms (быстрый доступ к памяти)
- **Запросы к БД:** 1-5 секунд (индексированные запросы)
- **HTTP запросы:** 5-30 секунд (зависит от API)
- **Тяжёлая обработка:** 30-300 секунд (аналитика, отчёты)

Без DoWithTimeout медленная или зависшая функция может блокировать ваше приложение бесконечно, потребляя ресурсы и ухудшая пользовательский опыт.`,
			solutionCode: `package concurrency

import (
	"context"
	"time"
)

func DoWithTimeout(ctx context.Context, f func(), d time.Duration) error {
	if ctx == nil {                                                 // Обработка nil контекста
		ctx = context.Background()                              // Используем Background как fallback
	}
	ctxWithTimeout, cancel := context.WithTimeout(ctx, d)           // Создаём контекст с таймаутом
	defer cancel()                                                  // Всегда отменяем для освобождения ресурсов
	done := make(chan struct{})                                     // Канал для сигнала завершения
	go func() {                                                     // Запускаем f в горутине
		defer close(done)                                       // Закрываем канал при завершении
		f()                                                     // Выполняем функцию
	}()
	select {
	case <-ctxWithTimeout.Done():                                   // Произошёл таймаут
		return ctxWithTimeout.Err()                             // Возвращаем ошибку таймаута
	case <-done:                                                    // Функция завершилась
		return ctxWithTimeout.Err()                             // Возвращаем nil (без таймаута)
	}
}`
		},
		uz: {
			title: 'Vaqt cheklovi bilan operatsiyani bajarish',
			description: `Goroutinada f funksiyasini ishga tushiradigan va tugash yoki d timeoutni kutadigan **DoWithTimeout** ni amalga oshiring.

**Talablar:**
1. \`DoWithTimeout(ctx context.Context, f func(), d time.Duration) error\` funksiyasini yarating
2. nil kontekstni ishlang (Background dan foydalaning)
3. context.WithTimeout dan foydalanib timeout bilan kontekst yarating
4. f ni goroutinada ishga tushiring
5. Tugash yoki timeoutni kuting
6. Agar timeout yuz bersa kontekst xatosini qaytaring
7. Agar funksiya timeoutdan oldin muvaffaqiyatli tugasa nil qaytaring

**Misol:**
\`\`\`go
err := DoWithTimeout(ctx, func() {
    time.Sleep(100 * time.Millisecond)
    fmt.Println("Task completed")
}, 200 * time.Millisecond)
// err = nil (timeoutdan oldin tugadi)

err = DoWithTimeout(ctx, func() {
    time.Sleep(300 * time.Millisecond)
}, 100 * time.Millisecond)
// err = context.DeadlineExceeded
\`\`\`

**Cheklovlar:**
- context.WithTimeout dan foydalanishi kerak
- f ni alohida goroutinada ishga tushirishi kerak
- Tugash yoki timeoutni kutishi kerak`,
			hint1: `d davomiyligidan keyin avtomatik bekor qilinadigan kontekst yaratish uchun context.WithTimeout(ctx, d) dan foydalaning.`,
			hint2: `done kanali yarating, f() ni tugaganda kanalni yopadigan goroutinada ishga tushiring, keyin ctx.Done() va done o'rtasida select qiling.`,
			whyItMatters: `DoWithTimeout funksiyalarning cheksiz bajarilishining oldini oladi, osilib qolishi yoki juda ko'p vaqt olishi mumkin bo'lgan operatsiyalar uchun vaqt chegaralarini o'rnatish uchun zarur.

**Nima uchun Timeouts muhim:**
- **Resurslarni himoya qilish:** Goroutinalarning abadiy osilib qolishining oldini olish
- **Tezkor tizimlar:** Cheksiz kutish o'rniga tez fail
- **SLA muvofiqlik:** Maksimal operatsiya davomiyligini ta'minlash
- **Debugging:** Sekin operatsiyalarni osongina aniqlash

**Ishlab chiqarish patternlari:**
\`\`\`go
// Timeout bilan API so'rovi
func FetchUserData(userID string) (*User, error) {
    ctx := context.Background()
    var user *User

    err := DoWithTimeout(ctx, func() {
        user = database.GetUser(userID)
    }, 2*time.Second)

    if err != nil {
        return nil, fmt.Errorf("fetch timeout: %w", err)
    }
    return user, nil
}

// Timeout bilan tashqi API chaqiruvi
func CallExternalAPI(url string) ([]byte, error) {
    ctx := context.Background()
    var result []byte

    err := DoWithTimeout(ctx, func() {
        resp, _ := http.Get(url)
        defer resp.Body.Close()
        result, _ = ioutil.ReadAll(resp.Body)
    }, 5*time.Second)

    return result, err
}

// Timeout bilan og'ir hisoblashlar
func ProcessLargeDataset(data []int) (int, error) {
    ctx := context.Background()
    var sum int

    err := DoWithTimeout(ctx, func() {
        for _, v := range data {
            sum += v * v
            time.Sleep(time.Millisecond)  // Og'ir ishni simulyatsiya qilish
        }
    }, 10*time.Second)

    return sum, err
}
\`\`\`

**Turli operatsiyalar uchun turli timeoutlar:**
\`\`\`go
const (
    FastOperationTimeout   = 100 * time.Millisecond  // Kesh, xotira
    NormalOperationTimeout = 1 * time.Second         // DB so'rovlari
    SlowOperationTimeout   = 10 * time.Second        // Tashqi APIlar
)

func PerformOperation(opType string) error {
    timeout := NormalOperationTimeout
    switch opType {
    case "cache":
        timeout = FastOperationTimeout
    case "database":
        timeout = NormalOperationTimeout
    case "external_api":
        timeout = SlowOperationTimeout
    }

    return DoWithTimeout(context.Background(), func() {
        // Tegishli turdagi operatsiyani bajarish
    }, timeout)
}
\`\`\`

**Haqiqiy ishlab chiqarish foydalari:**
- **Osilib qolishni oldini olish:** Operatsiyalar cheksiz bloklanmaydi
- **Resurs boshqaruvi:** Tiqilib qolgan operatsiyalardan resurslarni ozod qilish
- **Yaxshiroq UX:** Foydalanuvchilar cheksiz yuklanish o'rniga tez xatolarni olishadi
- **Debugging:** Qaysi operatsiyalar sekin ekanligini osongina aniqlash

**Productiondan tavsiya etilgan timeout qiymatlari:**
- **Kesh operatsiyalari:** 50-100ms (xotiraga tez kirish)
- **DB so'rovlari:** 1-5 soniya (indekslangan so'rovlar)
- **HTTP so'rovlari:** 5-30 soniya (APIga bog'liq)
- **Og'ir qayta ishlash:** 30-300 soniya (tahlil, hisobotlar)

DoWithTimeout bo'lmasa, sekin yoki osilib qolgan funksiya sizning ilovangizni cheksiz bloklashi mumkin, resurslarni iste'mol qiladi va foydalanuvchi tajribasini yomonlashtiradi.`,
			solutionCode: `package concurrency

import (
	"context"
	"time"
)

func DoWithTimeout(ctx context.Context, f func(), d time.Duration) error {
	if ctx == nil {                                                 // nil kontekstni ishlash
		ctx = context.Background()                              // Fallback sifatida Background ishlatamiz
	}
	ctxWithTimeout, cancel := context.WithTimeout(ctx, d)           // Timeout bilan kontekst yaratamiz
	defer cancel()                                                  // Resurslarni ozod qilish uchun har doim bekor qilamiz
	done := make(chan struct{})                                     // Tugash signali uchun kanal
	go func() {                                                     // f ni goroutinada ishga tushiramiz
		defer close(done)                                       // Tugaganda kanalni yopamiz
		f()                                                     // Funksiyani bajaramiz
	}()
	select {
	case <-ctxWithTimeout.Done():                                   // Timeout yuz berdi
		return ctxWithTimeout.Err()                             // Timeout xatosini qaytaramiz
	case <-done:                                                    // Funksiya tugadi
		return ctxWithTimeout.Err()                             // nil qaytaramiz (timeout yo'q)
	}
}`
		}
	}
};

export default task;
