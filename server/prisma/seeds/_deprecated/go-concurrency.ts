/**
 * Go Concurrency Module Seeds - v2
 *
 * Structure:
 * - User-friendly titles
 * - Proper descriptions without duplication
 * - Solution explanations with line-by-line comments
 * - Multi-language support (EN, RU, UZ)
 * - Logical topic groupings (3-5 tasks per topic)
 */

export const GO_CONCURRENCY_MODULES = [
  {
    title: 'Concurrency Basics',
    description: 'Master Go context package and channel fundamentals for building cancellable, timeout-aware concurrent code.',
    section: 'core',
    order: 1,
    translations: {
      ru: {
        title: 'Основы конкурентности',
        description: 'Освойте пакет context и основы каналов для создания отменяемого и timeout-aware конкурентного кода.'
      },
      uz: {
        title: 'Konkurrentlik asoslari',
        description: 'Go context paketi va kanallar asoslarini o\'rganing - bekor qilinadigan va timeout-aware kod yaratish uchun.'
      }
    },
    topics: [
      // Topic 1: Context Fundamentals
      {
        title: 'Context Fundamentals',
        description: 'Learn to create and use context with timeouts and deadlines.',
        difficulty: 'easy',
        estimatedTime: '45m',
        order: 1,
        translations: {
          ru: {
            title: 'Основы Context',
            description: 'Научитесь создавать и использовать context с таймаутами и дедлайнами.'
          },
          uz: {
            title: 'Context asoslari',
            description: 'Timeout va deadline bilan context yaratish va ishlatishni o\'rganing.'
          }
        },
        tasks: [
          {
            slug: 'go-ctx-timeout',
            title: 'Execute Function with Timeout',
            difficulty: 'easy',
            tags: ['go', 'context', 'concurrency', 'timeout'],
            estimatedTime: '15m',
            isPremium: false,
            description: `Run a function in a goroutine with a maximum execution time limit.

If the function doesn't complete within the specified duration, the operation should be cancelled and return a timeout error.

**Requirements:**
- Create a derived context using \`context.WithTimeout\`
- Execute the function in a separate goroutine
- Use \`select\` to wait for either completion or timeout
- Return the appropriate error if timeout occurs

**Example:**
\`\`\`go
err := DoWithTimeout(ctx, func() {
    time.Sleep(100 * time.Millisecond)
}, 50*time.Millisecond)
// err == context.DeadlineExceeded
\`\`\`

**Constraints:**
- Handle nil context by using context.Background()
- Always call cancel() to release resources`,
            initialCode: `package concurrency

import (
	"context"
	"time"
)

// DoWithTimeout runs f in a goroutine and waits for completion or timeout.
func DoWithTimeout(ctx context.Context, f func(), d time.Duration) error {
	// TODO: Implement timeout logic
	panic("not implemented")
}`,
            solutionCode: `package concurrency

import (
	"context"
	"time"
)

func DoWithTimeout(ctx context.Context, f func(), d time.Duration) error {
	if ctx == nil {                                         // Handle nil context safely
		ctx = context.Background()                      // Use Background as default
	}
	ctxWithTimeout, cancel := context.WithTimeout(ctx, d)   // Create context with timeout
	defer cancel()                                          // MUST call to release timer resources

	done := make(chan struct{})                             // Channel to signal completion
	go func() {                                             // Launch function in goroutine
		defer close(done)                               // Signal done by closing channel
		f()                                             // Execute the actual function
	}()

	select {                                                // Wait for first event
	case <-ctxWithTimeout.Done():                           // Timeout or parent cancelled
		return ctxWithTimeout.Err()                     // Return DeadlineExceeded or Canceled
	case <-done:                                            // Function completed
		return nil                                      // Success - no error
	}
}`,
            solutionExplanation: null,
            hint1: 'Use context.WithTimeout and a done channel.',
            hint2: 'Select between ctx.Done() and your done channel.',
            whyItMatters: `This pattern is fundamental in production Go services. Every HTTP handler, database query, and external API call should have a timeout to prevent resource exhaustion.

Without timeouts, a slow downstream service can cause cascading failures - your goroutines pile up waiting, memory grows, and eventually your service crashes. Companies like Uber and Netflix enforce strict timeout policies on all network calls.

You'll use this pattern when building: API clients, database connection pools, microservice communication, and any operation that touches external resources.`,
            order: 0,
            translations: {
              ru: {
                title: 'Выполнение функции с таймаутом',
                description: `Запустите функцию в горутине с максимальным временем выполнения.

Если функция не завершится в течение указанного времени, операция должна быть отменена с возвратом ошибки таймаута.

**Требования:**
- Создайте производный context с помощью \`context.WithTimeout\`
- Выполните функцию в отдельной горутине
- Используйте \`select\` для ожидания завершения или таймаута
- Верните соответствующую ошибку при таймауте

**Пример:**
\`\`\`go
err := DoWithTimeout(ctx, func() {
    time.Sleep(100 * time.Millisecond)
}, 50*time.Millisecond)
// err == context.DeadlineExceeded
\`\`\`

**Ограничения:**
- Обработайте nil context используя context.Background()
- Всегда вызывайте cancel() для освобождения ресурсов`,
                hint1: 'Используйте context.WithTimeout и done канал.',
                hint2: 'Select между ctx.Done() и вашим done каналом.',
                solutionCode: `package concurrency

import (
	"context"
	"time"
)

func DoWithTimeout(ctx context.Context, f func(), d time.Duration) error {
	if ctx == nil {                                         // Безопасная обработка nil контекста
		ctx = context.Background()                      // Используем Background как значение по умолчанию
	}
	ctxWithTimeout, cancel := context.WithTimeout(ctx, d)   // Создаём контекст с таймаутом
	defer cancel()                                          // ОБЯЗАТЕЛЬНО вызываем для освобождения таймера

	done := make(chan struct{})                             // Канал для сигнала о завершении
	go func() {                                             // Запускаем функцию в горутине
		defer close(done)                               // Сигнализируем закрытием канала
		f()                                             // Выполняем саму функцию
	}()

	select {                                                // Ждём первого события
	case <-ctxWithTimeout.Done():                           // Таймаут или родитель отменён
		return ctxWithTimeout.Err()                     // Возвращаем DeadlineExceeded или Canceled
	case <-done:                                            // Функция завершилась
		return nil                                      // Успех - без ошибки
	}
}`,
                whyItMatters: `Этот паттерн фундаментален для production Go сервисов. Каждый HTTP обработчик, запрос к БД и внешний API вызов должен иметь таймаут для предотвращения исчерпания ресурсов.

Без таймаутов медленный downstream сервис может вызвать каскадные сбои - горутины накапливаются в ожидании, память растет, и сервис падает. Компании как Uber и Netflix применяют строгие политики таймаутов на все сетевые вызовы.

Вы будете использовать этот паттерн при создании: API клиентов, пулов соединений с БД, микросервисной коммуникации.`
              },
              uz: {
                title: 'Funktsiyani timeout bilan bajarish',
                description: `Funktsiyani goroutine'da maksimal bajarilish vaqti bilan ishga tushiring.

Agar funktsiya belgilangan vaqt ichida tugamasa, operatsiya bekor qilinishi va timeout xatosi qaytarilishi kerak.

**Talablar:**
- \`context.WithTimeout\` yordamida hosila context yarating
- Funktsiyani alohida goroutine'da bajaring
- Tugash yoki timeout'ni kutish uchun \`select\` dan foydalaning
- Timeout bo'lganda tegishli xatoni qaytaring

**Misol:**
\`\`\`go
err := DoWithTimeout(ctx, func() {
    time.Sleep(100 * time.Millisecond)
}, 50*time.Millisecond)
// err == context.DeadlineExceeded
\`\`\`

**Cheklovlar:**
- nil context'ni context.Background() yordamida qayta ishlang
- Resurslarni bo'shatish uchun doimo cancel() ni chaqiring`,
                hint1: 'context.WithTimeout va done kanalidan foydalaning.',
                hint2: 'ctx.Done() va done kanalingiz orasida select qiling.',
                solutionCode: `package concurrency

import (
	"context"
	"time"
)

func DoWithTimeout(ctx context.Context, f func(), d time.Duration) error {
	if ctx == nil {                                         // nil contextni xavfsiz qayta ishlash
		ctx = context.Background()                      // Standart sifatida Background dan foydalaning
	}
	ctxWithTimeout, cancel := context.WithTimeout(ctx, d)   // Timeout bilan context yarating
	defer cancel()                                          // Timer resurslarini bo'shatish uchun ALBATTA chaqiring

	done := make(chan struct{})                             // Tugallash signali uchun kanal
	go func() {                                             // Funktsiyani goroutine'da ishga tushiring
		defer close(done)                               // Kanalni yopish orqali signal bering
		f()                                             // Haqiqiy funktsiyani bajaring
	}()

	select {                                                // Birinchi voqeani kuting
	case <-ctxWithTimeout.Done():                           // Timeout yoki ota-ona bekor qilindi
		return ctxWithTimeout.Err()                     // DeadlineExceeded yoki Canceled qaytaring
	case <-done:                                            // Funktsiya tugallandi
		return nil                                      // Muvaffaqiyat - xato yo'q
	}
}`,
                whyItMatters: `Bu pattern production Go servislari uchun fundamental. Har bir HTTP handler, DB so'rovi va tashqi API chaqiruvi resurslarning tugashini oldini olish uchun timeout'ga ega bo'lishi kerak.

Timeout'larsiz sekin downstream servis kaskad nosozliklarni keltirib chiqarishi mumkin - goroutine'lar kutishda to'planadi, xotira o'sadi va servis qulab tushadi. Uber va Netflix kabi kompaniyalar barcha tarmoq chaqiruvlarida qat'iy timeout siyosatlarini qo'llaydi.`
              }
            }
          },
          {
            slug: 'go-ctx-deadline',
            title: 'Execute Function with Deadline',
            difficulty: 'easy',
            tags: ['go', 'context', 'concurrency', 'deadline'],
            estimatedTime: '15m',
            isPremium: false,
            description: `Run a function that must complete by a specific point in time.

Unlike timeout (relative duration), deadline uses an absolute time. This is useful when you need operations to complete by a specific moment regardless of when they start.

**Requirements:**
- Create a derived context using \`context.WithDeadline\`
- Execute the function in a separate goroutine
- Return error if deadline is exceeded

**Example:**
\`\`\`go
deadline := time.Now().Add(100 * time.Millisecond)
err := DoWithDeadline(ctx, func() {
    time.Sleep(200 * time.Millisecond)
}, deadline)
// err == context.DeadlineExceeded
\`\`\`

**When to use Deadline vs Timeout:**
- Deadline: "Must complete by 3:00 PM"
- Timeout: "Must complete within 5 seconds"`,
            initialCode: `package concurrency

import (
	"context"
	"time"
)

// DoWithDeadline runs f and ensures it completes by the given deadline.
func DoWithDeadline(ctx context.Context, f func(), deadline time.Time) error {
	// TODO: Implement deadline logic
	panic("not implemented")
}`,
            solutionCode: `package concurrency

import (
	"context"
	"time"
)

func DoWithDeadline(ctx context.Context, f func(), deadline time.Time) error {
	if ctx == nil {                                              // nil context check for safety
		ctx = context.Background()                           // use Background as fallback
	}
	ctxWithDeadline, cancel := context.WithDeadline(ctx, deadline) // absolute time vs duration
	defer cancel()                                               // always release resources

	done := make(chan struct{})                                  // completion signal channel
	go func() {                                                  // launch work in goroutine
		defer close(done)                                    // signal done by closing
		f()                                                  // execute the function
	}()

	select {                                                     // wait for first event
	case <-ctxWithDeadline.Done():                               // deadline reached
		return ctxWithDeadline.Err()                         // DeadlineExceeded or Canceled
	case <-done:                                                 // function completed in time
		return nil                                           // success
	}
}`,
            solutionExplanation: null,
            hint1: 'context.WithDeadline takes absolute time.Time.',
            hint2: 'Pattern is same as timeout, different context creation.',
            whyItMatters: `Deadlines are preferred over timeouts in request-scoped operations. When an HTTP request comes in at 2:59:50 PM with a 15-second timeout, and it spawns 3 database calls, each call should know "I must finish by 3:00:05 PM" - not "I have 15 seconds."

This is how context propagation works in production: the deadline flows through the entire call chain. Google's gRPC framework uses deadlines extensively - every RPC carries an absolute deadline, not a relative timeout.

Use deadlines when coordinating multiple operations that share a common completion requirement, like distributed transactions or multi-service API calls.`,
            order: 1,
            translations: {
              ru: {
                title: 'Выполнение функции с дедлайном',
                description: `Запустите функцию, которая должна завершиться к определенному моменту времени.

В отличие от таймаута (относительная длительность), дедлайн использует абсолютное время.

**Требования:**
- Создайте производный context с помощью \`context.WithDeadline\`
- Выполните функцию в отдельной горутине
- Верните ошибку при превышении дедлайна

**Пример:**
\`\`\`go
deadline := time.Now().Add(100 * time.Millisecond)
err := DoWithDeadline(ctx, func() {
    time.Sleep(200 * time.Millisecond)
}, deadline)
// err == context.DeadlineExceeded
\`\`\`

**Когда использовать Deadline vs Timeout:**
- Deadline: "Должно завершиться к 15:00"
- Timeout: "Должно завершиться за 5 секунд"`,
                hint1: 'context.WithDeadline принимает абсолютное time.Time.',
                hint2: 'Паттерн такой же как timeout, другое создание context.',
                solutionCode: `package concurrency

import (
	"context"
	"time"
)

func DoWithDeadline(ctx context.Context, f func(), deadline time.Time) error {
	if ctx == nil {                                              // проверка nil контекста для безопасности
		ctx = context.Background()                           // используем Background как запасной вариант
	}
	ctxWithDeadline, cancel := context.WithDeadline(ctx, deadline) // абсолютное время вместо длительности
	defer cancel()                                               // всегда освобождаем ресурсы

	done := make(chan struct{})                                  // канал сигнала о завершении
	go func() {                                                  // запускаем работу в горутине
		defer close(done)                                    // сигнализируем закрытием
		f()                                                  // выполняем функцию
	}()

	select {                                                     // ждём первого события
	case <-ctxWithDeadline.Done():                               // дедлайн достигнут
		return ctxWithDeadline.Err()                         // DeadlineExceeded или Canceled
	case <-done:                                                 // функция завершилась вовремя
		return nil                                           // успех
	}
}`,
                whyItMatters: `Дедлайны предпочтительнее таймаутов в операциях с областью видимости запроса. Когда HTTP запрос приходит в 14:59:50 с таймаутом 15 секунд и порождает 3 вызова БД, каждый должен знать "я должен завершиться к 15:00:05" - а не "у меня есть 15 секунд."

Так работает распространение context в production: дедлайн проходит через всю цепочку вызовов. Google gRPC использует дедлайны - каждый RPC несет абсолютный дедлайн, а не относительный таймаут.

Используйте дедлайны при координации нескольких операций с общим требованием завершения.`
              },
              uz: {
                title: 'Funktsiyani deadline bilan bajarish',
                description: `Ma'lum vaqtgacha tugashi kerak bo'lgan funktsiyani ishga tushiring.

Timeout'dan (nisbiy davomiylik) farqli o'laroq, deadline absolyut vaqtni ishlatadi.

**Talablar:**
- \`context.WithDeadline\` yordamida hosila context yarating
- Funktsiyani alohida goroutine'da bajaring
- Deadline o'tib ketganda xato qaytaring

**Misol:**
\`\`\`go
deadline := time.Now().Add(100 * time.Millisecond)
err := DoWithDeadline(ctx, func() {
    time.Sleep(200 * time.Millisecond)
}, deadline)
// err == context.DeadlineExceeded
\`\`\`

**Deadline vs Timeout qachon ishlatish:**
- Deadline: "15:00 gacha tugashi kerak"
- Timeout: "5 soniya ichida tugashi kerak"`,
                hint1: 'context.WithDeadline absolyut time.Time qabul qiladi.',
                hint2: 'Pattern timeout bilan bir xil, faqat context yaratish farq qiladi.',
                solutionCode: `package concurrency

import (
	"context"
	"time"
)

func DoWithDeadline(ctx context.Context, f func(), deadline time.Time) error {
	if ctx == nil {                                              // xavfsizlik uchun nil context tekshiruvi
		ctx = context.Background()                           // zaxira sifatida Background ishlatamiz
	}
	ctxWithDeadline, cancel := context.WithDeadline(ctx, deadline) // davomiylik o'rniga absolyut vaqt
	defer cancel()                                               // doimo resurslarni bo'shatamiz

	done := make(chan struct{})                                  // tugallash signali kanali
	go func() {                                                  // ishni goroutine'da ishga tushiramiz
		defer close(done)                                    // yopish orqali signal beramiz
		f()                                                  // funktsiyani bajaramiz
	}()

	select {                                                     // birinchi voqeani kutamiz
	case <-ctxWithDeadline.Done():                               // deadline yetildi
		return ctxWithDeadline.Err()                         // DeadlineExceeded yoki Canceled
	case <-done:                                                 // funktsiya o'z vaqtida tugadi
		return nil                                           // muvaffaqiyat
	}
}`,
                whyItMatters: `Deadline'lar so'rov doirasidagi operatsiyalarda timeout'lardan afzalroq. HTTP so'rov 14:59:50 da 15 soniyalik timeout bilan kelganda va 3 ta DB chaqiruvini boshlasa, har biri "15:00:05 gacha tugashim kerak" deb bilishi kerak - "15 soniyam bor" emas.

Production'da context tarqalishi shunday ishlaydi: deadline butun chaqiruv zanjiri bo'ylab o'tadi. Google gRPC deadline'larni keng qo'llaydi - har bir RPC absolyut deadline tashiydi.

Umumiy tugash talabiga ega bir nechta operatsiyalarni muvofiqlashtirishda deadline'lardan foydalaning.`
              }
            }
          },
          {
            slug: 'go-ctx-cancel',
            title: 'Cancellable Function Execution',
            difficulty: 'easy',
            tags: ['go', 'context', 'concurrency', 'cancellation'],
            estimatedTime: '15m',
            isPremium: false,
            description: `Execute a function that can be cancelled externally.

The function receives a context and should respect cancellation signals. This pattern is essential for building responsive applications that can stop work gracefully.

**Requirements:**
- Create a cancellable context using \`context.WithCancel\`
- Pass the derived context to the function f
- The function should check ctx.Done() periodically
- Return context error if cancelled

**Example:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
go func() {
    time.Sleep(50 * time.Millisecond)
    cancel() // Cancel after 50ms
}()
err := DoWithCancel(ctx, func(ctx context.Context) {
    // Long operation that checks ctx.Done()
})
// err == context.Canceled
\`\`\``,
            initialCode: `package concurrency

import "context"

// DoWithCancel runs f with a cancellable context.
// The function f receives the context and should respect cancellation.
func DoWithCancel(ctx context.Context, f func(context.Context)) error {
	// TODO: Implement cancellable execution
	panic("not implemented")
}`,
            solutionCode: `package concurrency

import "context"

func DoWithCancel(ctx context.Context, f func(context.Context)) error {
	if ctx == nil {                                              // handle nil context safely
		ctx = context.Background()                           // use Background as default
	}
	ctxWithCancel, cancel := context.WithCancel(ctx)             // create cancellable child context
	defer cancel()                                               // cleanup: release resources on exit

	done := make(chan struct{})                                  // signal channel for completion
	go func() {                                                  // spawn work goroutine
		defer close(done)                                    // closing signals completion
		f(ctxWithCancel)                                     // pass child ctx so f can check Done()
	}()

	select {                                                     // block until event occurs
	case <-ctx.Done():                                           // parent cancelled
		return ctx.Err()                                     // propagate Canceled error
	case <-done:                                                 // work completed normally
		return nil                                           // success
	}
}`,
            solutionExplanation: null,
            hint1: 'Pass the derived context to f for reactivity.',
            hint2: 'Wait for done channel before returning.',
            whyItMatters: `Manual cancellation is essential for building responsive applications. When a user clicks "Cancel" or navigates away, you need to stop all in-flight operations immediately to free resources and provide good UX.

In microservices, cancellation cascades through the call chain. If a frontend request is cancelled, all downstream services should stop work. This prevents wasted computation and allows resources to serve other requests.

You'll use this pattern for: user-initiated cancellation, graceful shutdown handlers, and any operation that should stop when its parent context is cancelled.`,
            order: 2,
            translations: {
              ru: {
                title: 'Отменяемое выполнение функции',
                description: `Выполните функцию, которую можно отменить извне.

Функция получает context и должна реагировать на сигналы отмены.

**Требования:**
- Создайте отменяемый context с помощью \`context.WithCancel\`
- Передайте производный context в функцию f
- Функция должна периодически проверять ctx.Done()
- Верните ошибку context при отмене

**Пример:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
go func() {
    time.Sleep(50 * time.Millisecond)
    cancel() // Отмена через 50мс
}()
err := DoWithCancel(ctx, func(ctx context.Context) {
    // Долгая операция, проверяющая ctx.Done()
})
// err == context.Canceled
\`\`\``,
                hint1: 'Передайте производный context в f для реактивности.',
                hint2: 'Дождитесь done канала перед возвратом.',
                solutionCode: `package concurrency

import "context"

func DoWithCancel(ctx context.Context, f func(context.Context)) error {
	if ctx == nil {                                              // безопасная обработка nil контекста
		ctx = context.Background()                           // используем Background по умолчанию
	}
	ctxWithCancel, cancel := context.WithCancel(ctx)             // создаём отменяемый дочерний контекст
	defer cancel()                                               // очистка: освобождаем ресурсы при выходе

	done := make(chan struct{})                                  // канал сигнала завершения
	go func() {                                                  // запускаем рабочую горутину
		defer close(done)                                    // закрытие сигнализирует о завершении
		f(ctxWithCancel)                                     // передаём дочерний ctx чтобы f мог проверять Done()
	}()

	select {                                                     // блокируемся до события
	case <-ctx.Done():                                           // родитель отменён
		return ctx.Err()                                     // распространяем ошибку Canceled
	case <-done:                                                 // работа завершена нормально
		return nil                                           // успех
	}
}`,
                whyItMatters: `Ручная отмена необходима для создания отзывчивых приложений. Когда пользователь нажимает "Отмена" или уходит со страницы, нужно немедленно остановить все операции для освобождения ресурсов и хорошего UX.

В микросервисах отмена каскадируется через цепочку вызовов. Если фронтенд запрос отменен, все downstream сервисы должны прекратить работу. Это предотвращает напрасные вычисления.

Используйте этот паттерн для: отмены по инициативе пользователя, graceful shutdown и любых операций, которые должны остановиться при отмене родительского контекста.`
              },
              uz: {
                title: 'Bekor qilinadigan funktsiya bajarilishi',
                description: `Tashqaridan bekor qilinishi mumkin bo'lgan funktsiyani bajaring.

Funktsiya context qabul qiladi va bekor qilish signallariga javob berishi kerak.

**Talablar:**
- \`context.WithCancel\` yordamida bekor qilinadigan context yarating
- Hosila context'ni f funktsiyasiga uzating
- Funktsiya vaqti-vaqti bilan ctx.Done() ni tekshirishi kerak
- Bekor qilinganda context xatosini qaytaring

**Misol:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
go func() {
    time.Sleep(50 * time.Millisecond)
    cancel() // 50ms dan keyin bekor qilish
}()
err := DoWithCancel(ctx, func(ctx context.Context) {
    // ctx.Done() ni tekshiruvchi uzoq operatsiya
})
// err == context.Canceled
\`\`\``,
                hint1: 'Reaktivlik uchun hosila context\'ni f ga uzating.',
                hint2: 'Qaytishdan oldin done kanalini kuting.',
                solutionCode: `package concurrency

import "context"

func DoWithCancel(ctx context.Context, f func(context.Context)) error {
	if ctx == nil {                                              // nil contextni xavfsiz qayta ishlash
		ctx = context.Background()                           // standart sifatida Background ishlatamiz
	}
	ctxWithCancel, cancel := context.WithCancel(ctx)             // bekor qilinadigan bola context yaratamiz
	defer cancel()                                               // tozalash: chiqishda resurslarni bo'shatamiz

	done := make(chan struct{})                                  // tugallash signal kanali
	go func() {                                                  // ishchi goroutine ishga tushiramiz
		defer close(done)                                    // yopish tugallanganini bildiradi
		f(ctxWithCancel)                                     // bola ctx ni uzatamiz, f Done() ni tekshira olishi uchun
	}()

	select {                                                     // voqea sodir bo'lguncha bloklanamiz
	case <-ctx.Done():                                           // ota-ona bekor qilindi
		return ctx.Err()                                     // Canceled xatosini tarqatamiz
	case <-done:                                                 // ish normal tugadi
		return nil                                           // muvaffaqiyat
	}
}`,
                whyItMatters: `Qo'lda bekor qilish sezgir ilovalar yaratish uchun muhim. Foydalanuvchi "Bekor qilish" tugmasini bosganda yoki sahifadan ketganda, resurslarni bo'shatish va yaxshi UX ta'minlash uchun barcha operatsiyalarni darhol to'xtatish kerak.

Mikroservislarda bekor qilish chaqiruv zanjiri bo'ylab kaskadlanadi. Agar frontend so'rovi bekor qilinsa, barcha downstream servislar ishni to'xtatishi kerak.

Bu patternni qo'llang: foydalanuvchi tomonidan bekor qilish, graceful shutdown va ota-ona context bekor qilinganda to'xtashi kerak bo'lgan har qanday operatsiya uchun.`
              }
            }
          }
        ]
      },
      // Topic 2: Signal Handling
      {
        title: 'Signal & Channel Coordination',
        description: 'Coordinate multiple goroutines using channels as signals.',
        difficulty: 'medium',
        estimatedTime: '1h',
        order: 2,
        translations: {
          ru: {
            title: 'Сигналы и координация каналов',
            description: 'Координируйте несколько горутин используя каналы как сигналы.'
          },
          uz: {
            title: 'Signal va kanal koordinatsiyasi',
            description: 'Bir nechta goroutine\'larni kanallar yordamida muvofiqlashtiring.'
          }
        },
        tasks: [
          {
            slug: 'go-ctx-notify',
            title: 'Notify on Context Cancellation',
            difficulty: 'easy',
            tags: ['go', 'context', 'channels', 'signals'],
            estimatedTime: '10m',
            isPremium: false,
            description: `Create a notification channel that closes when the context is cancelled.

This pattern converts context cancellation into a channel signal, making it easier to integrate with other channel-based code.

**Requirements:**
- Return a channel that closes when ctx is cancelled
- Launch a goroutine to watch ctx.Done()
- The returned channel should be receive-only

**Example:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
notifyCh := NotifyCancel(ctx)
cancel()
<-notifyCh // This will unblock after cancel()
\`\`\``,
            initialCode: `package concurrency

import "context"

// NotifyCancel returns a channel that closes when ctx is cancelled.
func NotifyCancel(ctx context.Context) <-chan struct{} {
	// TODO: Implement notification channel
	panic("not implemented")
}`,
            solutionCode: `package concurrency

import "context"

func NotifyCancel(ctx context.Context) <-chan struct{} {
	ch := make(chan struct{})                                    // create notification channel
	if ctx == nil {                                              // nil context edge case
		close(ch)                                            // return already-closed channel
		return ch                                            // allows immediate receive
	}
	go func() {                                                  // spawn watcher goroutine
		<-ctx.Done()                                         // block until cancellation
		close(ch)                                            // signal by closing channel
	}()                                                          // goroutine ends after close
	return ch                                                    // return receive-only channel
}`,
            solutionExplanation: null,
            hint1: 'Create channel, close it when ctx.Done() fires.',
            hint2: 'Use goroutine to wait and close channel.',
            whyItMatters: `Converting context cancellation to a channel signal enables powerful composition. You can combine cancellation with other channel operations in a single select statement.

This pattern is common in event-driven architectures where different components communicate via channels. Rather than passing context to every component, you can pass a notification channel that integrates naturally with existing channel-based logic.

Use this when building: event buses, pub/sub systems, and any code that needs to mix cancellation awareness with other channel operations.`,
            order: 0,
            translations: {
              ru: {
                title: 'Уведомление об отмене контекста',
                description: `Создайте канал уведомлений, который закрывается при отмене контекста.

Этот паттерн преобразует отмену контекста в сигнал канала.

**Требования:**
- Верните канал, который закрывается при отмене ctx
- Запустите горутину для отслеживания ctx.Done()
- Возвращаемый канал должен быть только для чтения

**Пример:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
notifyCh := NotifyCancel(ctx)
cancel()
<-notifyCh // Разблокируется после cancel()
\`\`\``,
                hint1: 'Создайте канал, закройте его при срабатывании ctx.Done().',
                hint2: 'Используйте горутину для ожидания и закрытия канала.',
                solutionCode: `package concurrency

import "context"

func NotifyCancel(ctx context.Context) <-chan struct{} {
	ch := make(chan struct{})                                    // создаём канал уведомлений
	if ctx == nil {                                              // крайний случай nil контекста
		close(ch)                                            // возвращаем уже закрытый канал
		return ch                                            // позволяет немедленное получение
	}
	go func() {                                                  // запускаем горутину-наблюдатель
		<-ctx.Done()                                         // блокируемся до отмены
		close(ch)                                            // сигнализируем закрытием канала
	}()                                                          // горутина завершается после close
	return ch                                                    // возвращаем канал только для чтения
}`,
                whyItMatters: `Преобразование отмены контекста в канальный сигнал позволяет мощную композицию. Вы можете комбинировать отмену с другими канальными операциями в одном select.

Этот паттерн распространен в event-driven архитектурах где компоненты общаются через каналы. Вместо передачи context каждому компоненту, можно передать канал уведомлений который естественно интегрируется с существующей канальной логикой.

Используйте при создании: event bus, pub/sub систем и любого кода, который должен смешивать осведомленность об отмене с другими канальными операциями.`
              },
              uz: {
                title: 'Context bekor bo\'lishini xabar qilish',
                description: `Context bekor bo'lganda yopiladigan xabar kanalini yarating.

Bu pattern context bekor bo'lishini kanal signaliga aylantiradi.

**Talablar:**
- ctx bekor bo'lganda yopiladigan kanal qaytaring
- ctx.Done() ni kuzatish uchun goroutine ishga tushiring
- Qaytariladigan kanal faqat o'qish uchun bo'lishi kerak

**Misol:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
notifyCh := NotifyCancel(ctx)
cancel()
<-notifyCh // cancel() dan keyin blokdan chiqadi
\`\`\``,
                hint1: 'Kanal yarating, ctx.Done() ishlaganda yoping.',
                hint2: 'Kutish va kanalni yopish uchun goroutine\'dan foydalaning.',
                solutionCode: `package concurrency

import "context"

func NotifyCancel(ctx context.Context) <-chan struct{} {
	ch := make(chan struct{})                                    // xabar kanalini yaratamiz
	if ctx == nil {                                              // nil context chekka holati
		close(ch)                                            // allaqachon yopilgan kanalni qaytaramiz
		return ch                                            // darhol qabul qilishga imkon beradi
	}
	go func() {                                                  // kuzatuvchi goroutine ishga tushiramiz
		<-ctx.Done()                                         // bekor bo'lgunga qadar bloklanamiz
		close(ch)                                            // kanalni yopish orqali signal beramiz
	}()                                                          // goroutine close dan keyin tugaydi
	return ch                                                    // faqat o'qish uchun kanalni qaytaramiz
}`,
                whyItMatters: `Context bekor bo'lishini kanal signaliga aylantirish kuchli kompozitsiyani ta'minlaydi. Siz bitta select ifodasida bekor qilishni boshqa kanal operatsiyalari bilan birlashtira olasiz.

Bu pattern event-driven arxitekturalarda keng tarqalgan bo'lib, turli komponentlar kanallar orqali muloqot qiladi. Har bir komponentga context uzatish o'rniga, mavjud kanal-asosli mantiq bilan tabiiy integratsiya qiladigan xabar kanalini uzatishingiz mumkin.

Budan foydalaning: event bus, pub/sub tizimlari va bekor qilish bilan boshqa kanal operatsiyalarini aralashtirishi kerak bo'lgan har qanday kod uchun.`
              }
            }
          },
          {
            slug: 'go-ctx-wait-signal',
            title: 'Wait for External Signal',
            difficulty: 'easy',
            tags: ['go', 'context', 'channels', 'select'],
            estimatedTime: '10m',
            isPremium: false,
            description: `Wait for either context cancellation or an external signal channel.

This is a common pattern when you need to wait for some event but also respect cancellation.

**Requirements:**
- Use select to wait for either ctx.Done() or signal channel
- Return nil if signal received, error if context cancelled

**Example:**
\`\`\`go
signal := make(chan struct{})
go func() {
    time.Sleep(10 * time.Millisecond)
    close(signal)
}()
err := WaitForSignal(ctx, signal)
// err == nil (signal arrived)
\`\`\``,
            initialCode: `package concurrency

import "context"

// WaitForSignal waits for either ctx cancellation or signal.
func WaitForSignal(ctx context.Context, signal <-chan struct{}) error {
	// TODO: Implement signal waiting
	panic("not implemented")
}`,
            solutionCode: `package concurrency

import "context"

func WaitForSignal(ctx context.Context, signal <-chan struct{}) error {
	if ctx == nil {                                              // nil context means no cancellation
		<-signal                                             // just block on signal
		return nil                                           // signal received
	}
	select {                                                     // wait for first event
	case <-ctx.Done():                                           // context cancelled first
		return ctx.Err()                                     // Canceled or DeadlineExceeded
	case <-signal:                                               // signal received first
		return nil                                           // success - event occurred
	}
}`,
            solutionExplanation: null,
            hint1: 'Simple select with two cases is enough.',
            hint2: 'Return ctx.Err() on cancellation, nil on signal.',
            whyItMatters: `This is the fundamental pattern for cancellation-aware blocking operations. Every time you block waiting for something in production Go code, you should also listen for cancellation.

Without this pattern, goroutines can hang forever waiting for events that never come. This leads to goroutine leaks - one of the most common bugs in Go services. Tools like goleak help detect this in tests.

You'll use this pattern constantly: waiting for database connections, network responses, user input, or any asynchronous event while respecting shutdown signals.`,
            order: 1,
            translations: {
              ru: {
                title: 'Ожидание внешнего сигнала',
                description: `Ожидайте либо отмены контекста, либо внешнего сигнального канала.

Это распространенный паттерн когда нужно ждать события, но также учитывать отмену.

**Требования:**
- Используйте select для ожидания ctx.Done() или сигнального канала
- Верните nil при получении сигнала, ошибку при отмене контекста

**Пример:**
\`\`\`go
signal := make(chan struct{})
go func() {
    time.Sleep(10 * time.Millisecond)
    close(signal)
}()
err := WaitForSignal(ctx, signal)
// err == nil (сигнал получен)
\`\`\``,
                hint1: 'Простой select с двумя case достаточен.',
                hint2: 'Верните ctx.Err() при отмене, nil при сигнале.',
                solutionCode: `package concurrency

import "context"

func WaitForSignal(ctx context.Context, signal <-chan struct{}) error {
	if ctx == nil {                                              // nil контекст означает нет отмены
		<-signal                                             // просто блокируемся на сигнале
		return nil                                           // сигнал получен
	}
	select {                                                     // ждём первого события
	case <-ctx.Done():                                           // контекст отменён первым
		return ctx.Err()                                     // Canceled или DeadlineExceeded
	case <-signal:                                               // сигнал получен первым
		return nil                                           // успех - событие произошло
	}
}`,
                whyItMatters: `Это фундаментальный паттерн для блокирующих операций с учетом отмены. Каждый раз когда вы блокируетесь в ожидании чего-то в production Go коде, вы должны также слушать отмену.

Без этого паттерна горутины могут зависнуть навсегда в ожидании событий, которые никогда не произойдут. Это приводит к утечкам горутин - одному из самых частых багов в Go сервисах.

Вы будете использовать этот паттерн постоянно: ожидание соединений с БД, сетевых ответов, пользовательского ввода или любого асинхронного события с учетом сигналов shutdown.`
              },
              uz: {
                title: 'Tashqi signalni kutish',
                description: `Context bekor bo'lishi yoki tashqi signal kanalini kuting.

Bu keng tarqalgan pattern bo'lib, hodisani kutish kerak, lekin bekor qilishni ham hisobga olish kerak.

**Talablar:**
- ctx.Done() yoki signal kanalini kutish uchun select dan foydalaning
- Signal kelganda nil, context bekor bo'lganda xato qaytaring

**Misol:**
\`\`\`go
signal := make(chan struct{})
go func() {
    time.Sleep(10 * time.Millisecond)
    close(signal)
}()
err := WaitForSignal(ctx, signal)
// err == nil (signal keldi)
\`\`\``,
                hint1: 'Ikki case bilan oddiy select yetarli.',
                hint2: 'Bekor qilinganda ctx.Err(), signalda nil qaytaring.',
                solutionCode: `package concurrency

import "context"

func WaitForSignal(ctx context.Context, signal <-chan struct{}) error {
	if ctx == nil {                                              // nil context bekor qilish yo'q degani
		<-signal                                             // shunchaki signalda bloklanamiz
		return nil                                           // signal olindi
	}
	select {                                                     // birinchi voqeani kutamiz
	case <-ctx.Done():                                           // context birinchi bekor bo'ldi
		return ctx.Err()                                     // Canceled yoki DeadlineExceeded
	case <-signal:                                               // signal birinchi keldi
		return nil                                           // muvaffaqiyat - voqea sodir bo'ldi
	}
}`,
                whyItMatters: `Bu bekor qilishni hisobga oladigan blokirovka qiluvchi operatsiyalar uchun fundamental pattern. Production Go kodida biror narsani kutib bloklangan har safar, siz bekor qilishni ham tinglashingiz kerak.

Bu patternsiz goroutine'lar hech qachon kelmaydigan hodisalarni kutib abadiy osilib qolishi mumkin. Bu goroutine oqishlariga olib keladi - Go servislaridagi eng keng tarqalgan xatolardan biri.

Bu patternni doimiy ishlatasiz: DB ulanishlarini, tarmoq javoblarini, foydalanuvchi kiritishini yoki shutdown signallarini hisobga olgan holda har qanday asinxron hodisani kutish uchun.`
              }
            }
          },
          {
            slug: 'go-ctx-wait-all',
            title: 'Wait for All Signals',
            difficulty: 'medium',
            tags: ['go', 'context', 'channels', 'coordination'],
            estimatedTime: '20m',
            isPremium: false,
            description: `Wait for all provided signal channels to close, or until context is cancelled.

This pattern is useful for barrier synchronization - waiting until multiple operations complete.

**Requirements:**
- Wait for ALL channels in the slice to close
- Respect context cancellation at any point
- Return error if context cancelled before all signals received

**Example:**
\`\`\`go
signals := make([]<-chan struct{}, 3)
for i := range signals {
    ch := make(chan struct{})
    signals[i] = ch
    go func() { close(ch) }()
}
err := WaitAll(ctx, signals)
// err == nil when all channels closed
\`\`\``,
            initialCode: `package concurrency

import "context"

// WaitAll waits for all signal channels to close.
func WaitAll(ctx context.Context, signals []<-chan struct{}) error {
	// TODO: Implement wait for all
	panic("not implemented")
}`,
            solutionCode: `package concurrency

import "context"

func WaitAll(ctx context.Context, signals []<-chan struct{}) error {
	if ctx == nil {                                              // handle nil context
		ctx = context.Background()                           // use Background for nil safety
	}
	for _, sig := range signals {                                // iterate all signals sequentially
		if sig == nil {                                      // skip nil channels
			continue                                     // move to next signal
		}
		select {                                             // wait for this signal OR cancel
		case <-ctx.Done():                                   // cancelled before all received
			return ctx.Err()                             // propagate error upward
		case <-sig:                                          // this signal received
		}                                                    // continue to next signal
	}
	return nil                                                   // all signals received successfully
}`,
            solutionExplanation: null,
            hint1: 'Loop through signals, select on each.',
            hint2: 'Check ctx.Done() in each iteration.',
            whyItMatters: `Barrier synchronization is essential for coordinating parallel operations. You often need to wait for multiple workers to complete before proceeding - like map-reduce phases or batch processing stages.

In distributed systems, this pattern appears in consensus protocols, distributed transactions, and multi-region replication. You wait for acknowledgments from multiple nodes before considering an operation complete.

Use this for: parallel task completion, multi-stage pipelines, initialization sequences where multiple services must be ready, and any scenario requiring synchronization across multiple concurrent operations.`,
            order: 2,
            translations: {
              ru: {
                title: 'Ожидание всех сигналов',
                description: `Дождитесь закрытия всех предоставленных сигнальных каналов или отмены контекста.

Этот паттерн полезен для барьерной синхронизации.

**Требования:**
- Дождитесь закрытия ВСЕХ каналов в срезе
- Учитывайте отмену контекста в любой момент
- Верните ошибку если контекст отменен до получения всех сигналов

**Пример:**
\`\`\`go
signals := make([]<-chan struct{}, 3)
for i := range signals {
    ch := make(chan struct{})
    signals[i] = ch
    go func() { close(ch) }()
}
err := WaitAll(ctx, signals)
// err == nil когда все каналы закрыты
\`\`\``,
                hint1: 'Пройдите по сигналам, select на каждом.',
                hint2: 'Проверяйте ctx.Done() в каждой итерации.',
                solutionCode: `package concurrency

import "context"

func WaitAll(ctx context.Context, signals []<-chan struct{}) error {
	if ctx == nil {                                              // обработка nil контекста
		ctx = context.Background()                           // используем Background для безопасности
	}
	for _, sig := range signals {                                // итерируем все сигналы последовательно
		if sig == nil {                                      // пропускаем nil каналы
			continue                                     // переходим к следующему сигналу
		}
		select {                                             // ждём этот сигнал ИЛИ отмену
		case <-ctx.Done():                                   // отменено до получения всех
			return ctx.Err()                             // распространяем ошибку вверх
		case <-sig:                                          // этот сигнал получен
		}                                                    // продолжаем к следующему сигналу
	}
	return nil                                                   // все сигналы успешно получены
}`,
                whyItMatters: `Барьерная синхронизация необходима для координации параллельных операций. Часто нужно дождаться завершения нескольких воркеров перед продолжением - как фазы map-reduce или этапы пакетной обработки.

В распределенных системах этот паттерн появляется в протоколах консенсуса, распределенных транзакциях и репликации. Вы ждете подтверждений от нескольких узлов.

Используйте для: завершения параллельных задач, многоэтапных пайплайнов, последовательностей инициализации где несколько сервисов должны быть готовы.`
              },
              uz: {
                title: 'Barcha signallarni kutish',
                description: `Barcha signal kanallarining yopilishini yoki context bekor bo'lishini kuting.

Bu pattern to'siq sinxronizatsiyasi uchun foydali.

**Talablar:**
- Massivdagi BARCHA kanallarning yopilishini kuting
- Istalgan vaqtda context bekor bo'lishini hisobga oling
- Barcha signallar olinmasidan context bekor bo'lsa xato qaytaring

**Misol:**
\`\`\`go
signals := make([]<-chan struct{}, 3)
for i := range signals {
    ch := make(chan struct{})
    signals[i] = ch
    go func() { close(ch) }()
}
err := WaitAll(ctx, signals)
// err == nil barcha kanallar yopilganda
\`\`\``,
                hint1: 'Signallar bo\'ylab aylanib o\'ting, har birida select qiling.',
                hint2: 'Har bir iteratsiyada ctx.Done() ni tekshiring.',
                solutionCode: `package concurrency

import "context"

func WaitAll(ctx context.Context, signals []<-chan struct{}) error {
	if ctx == nil {                                              // nil contextni qayta ishlash
		ctx = context.Background()                           // xavfsizlik uchun Background ishlatamiz
	}
	for _, sig := range signals {                                // barcha signallarni ketma-ket iteratsiya qilamiz
		if sig == nil {                                      // nil kanallarni o'tkazib yuboramiz
			continue                                     // keyingi signalga o'tamiz
		}
		select {                                             // bu signal YOKI bekor qilishni kutamiz
		case <-ctx.Done():                                   // hammasi olinmasdan bekor qilindi
			return ctx.Err()                             // xatoni yuqoriga tarqatamiz
		case <-sig:                                          // bu signal olindi
		}                                                    // keyingi signalga davom etamiz
	}
	return nil                                                   // barcha signallar muvaffaqiyatli olindi
}`,
                whyItMatters: `To'siq sinxronizatsiyasi parallel operatsiyalarni muvofiqlashtirish uchun muhim. Ko'pincha davom etishdan oldin bir nechta worker'larning tugashini kutish kerak - map-reduce fazalari yoki paketli qayta ishlash bosqichlari kabi.

Tarqatilgan tizimlarda bu pattern konsensus protokollarida, tarqatilgan tranzaktsiyalarda va replikatsiyada paydo bo'ladi. Siz bir nechta tugunlardan tasdiqlashlarni kutasiz.

Budan foydalaning: parallel vazifalar tugashi, ko'p bosqichli pipeline'lar, bir nechta servislar tayyor bo'lishi kerak bo'lgan initsializatsiya ketma-ketliklari uchun.`
              }
            }
          },
          {
            slug: 'go-ctx-wait-any',
            title: 'Wait for Any Signal',
            difficulty: 'medium',
            tags: ['go', 'context', 'channels', 'select', 'reflect'],
            estimatedTime: '25m',
            isPremium: true,
            description: `Wait for any one of the provided signal channels to close, returning which one fired first.

This pattern is useful for racing multiple operations and acting on the first to complete.

**Requirements:**
- Wait for ANY channel to close (first one wins)
- Return the index of the channel that closed
- Return -1 and error if context cancelled first
- Handle empty slice and nil channels

**Example:**
\`\`\`go
signals := make([]<-chan struct{}, 3)
// Channel 1 closes first
idx, err := WaitAny(ctx, signals)
// idx == 1, err == nil
\`\`\`

**Hint:** Consider using \`reflect.Select\` for dynamic case count.`,
            initialCode: `package concurrency

import "context"

// WaitAny waits for any signal channel to close.
// Returns the index of the closed channel, or -1 with error if cancelled.
func WaitAny(ctx context.Context, signals []<-chan struct{}) (int, error) {
	// TODO: Implement wait for any
	panic("not implemented")
}`,
            solutionCode: `package concurrency

import (
	"context"
	"reflect"
)

func WaitAny(ctx context.Context, signals []<-chan struct{}) (int, error) {
	if len(signals) == 0 {                                       // edge case: no signals provided
		if ctx != nil {                                      // if context exists, wait for it
			<-ctx.Done()                                 // block until cancelled
			return -1, ctx.Err()                         // return cancellation error
		}
		select {}                                            // block forever if no ctx, no signals
	}

	cases := make([]reflect.SelectCase, 0, len(signals)+1)       // dynamic select cases
	indexMap := make([]int, 0, len(signals))                     // maps case index to signal index

	if ctx != nil {                                              // add context.Done() as case 0
		cases = append(cases, reflect.SelectCase{
			Dir:  reflect.SelectRecv,                    // receive direction
			Chan: reflect.ValueOf(ctx.Done()),           // context cancellation channel
		})
	}

	for i, sig := range signals {                                // add each signal as a case
		if sig == nil {                                      // skip nil channels
			continue
		}
		cases = append(cases, reflect.SelectCase{
			Dir:  reflect.SelectRecv,                    // receive from signal
			Chan: reflect.ValueOf(sig),                  // the signal channel
		})
		indexMap = append(indexMap, i)                       // remember original index
	}

	chosen, _, _ := reflect.Select(cases)                        // dynamic select - blocks until one ready

	if ctx != nil && chosen == 0 {                               // context case won
		return -1, ctx.Err()                                 // cancelled before any signal
	}

	offset := 0                                                  // calculate original signal index
	if ctx != nil {
		offset = 1                                           // skip context case in calculation
	}
	return indexMap[chosen-offset], nil                          // return which signal fired first
}`,
            solutionExplanation: null,
            hint1: 'reflect.Select handles dynamic number of cases.',
            hint2: 'Map chosen index back to original signal index.',
            whyItMatters: `Racing multiple operations is crucial for building responsive systems. When you send a request to multiple replicas, you often want the fastest response - whoever answers first wins.

This pattern appears in: hedged requests (send to multiple servers, use first response), failover systems (try primary, fall back to secondary on timeout), and service discovery (find any available instance).

reflect.Select is the only way to select over a dynamic number of channels in Go. While it has runtime overhead compared to compile-time select, it's essential for building flexible concurrent systems where the number of channels isn't known at compile time.`,
            order: 3,
            translations: {
              ru: {
                title: 'Ожидание любого сигнала',
                description: `Дождитесь закрытия любого из предоставленных сигнальных каналов, вернув индекс сработавшего.

Этот паттерн полезен для гонки нескольких операций.

**Требования:**
- Дождитесь закрытия ЛЮБОГО канала (первый выигрывает)
- Верните индекс закрытого канала
- Верните -1 и ошибку если контекст отменен первым

**Пример:**
\`\`\`go
signals := make([]<-chan struct{}, 3)
// Канал 1 закрывается первым
idx, err := WaitAny(ctx, signals)
// idx == 1, err == nil
\`\`\`

**Подсказка:** Рассмотрите использование \`reflect.Select\` для динамического количества case.`,
                hint1: 'reflect.Select обрабатывает динамическое количество case.',
                hint2: 'Отобразите выбранный индекс обратно на исходный индекс сигнала.',
                solutionCode: `package concurrency

import (
	"context"
	"reflect"
)

func WaitAny(ctx context.Context, signals []<-chan struct{}) (int, error) {
	if len(signals) == 0 {                                       // крайний случай: сигналов нет
		if ctx != nil {                                      // если контекст есть, ждём его
			<-ctx.Done()                                 // блокируемся до отмены
			return -1, ctx.Err()                         // возвращаем ошибку отмены
		}
		select {}                                            // блокируемся навсегда если нет ctx и сигналов
	}

	cases := make([]reflect.SelectCase, 0, len(signals)+1)       // динамические case для select
	indexMap := make([]int, 0, len(signals))                     // маппинг индекса case на индекс сигнала

	if ctx != nil {                                              // добавляем context.Done() как case 0
		cases = append(cases, reflect.SelectCase{
			Dir:  reflect.SelectRecv,                    // направление получения
			Chan: reflect.ValueOf(ctx.Done()),           // канал отмены контекста
		})
	}

	for i, sig := range signals {                                // добавляем каждый сигнал как case
		if sig == nil {                                      // пропускаем nil каналы
			continue
		}
		cases = append(cases, reflect.SelectCase{
			Dir:  reflect.SelectRecv,                    // получаем из сигнала
			Chan: reflect.ValueOf(sig),                  // сигнальный канал
		})
		indexMap = append(indexMap, i)                       // запоминаем исходный индекс
	}

	chosen, _, _ := reflect.Select(cases)                        // динамический select - блокируется пока один не готов

	if ctx != nil && chosen == 0 {                               // выиграл case контекста
		return -1, ctx.Err()                                 // отменено до любого сигнала
	}

	offset := 0                                                  // вычисляем исходный индекс сигнала
	if ctx != nil {
		offset = 1                                           // пропускаем case контекста в расчёте
	}
	return indexMap[chosen-offset], nil                          // возвращаем какой сигнал сработал первым
}`,
                whyItMatters: `Гонка нескольких операций критична для отзывчивых систем. Когда вы отправляете запрос нескольким репликам, часто нужен самый быстрый ответ - кто ответит первым, тот и победил.

Этот паттерн появляется в: hedged requests (отправка нескольким серверам, использование первого ответа), системах failover, и service discovery (найти любой доступный инстанс).

reflect.Select - единственный способ делать select над динамическим числом каналов в Go. Он необходим для гибких конкурентных систем где число каналов неизвестно во время компиляции.`
              },
              uz: {
                title: 'Ixtiyoriy signalni kutish',
                description: `Taqdim etilgan signal kanallaridan birortasining yopilishini kuting va qaysi biri birinchi bo'lganini qaytaring.

Bu pattern bir nechta operatsiyalarni poyga qildirish uchun foydali.

**Talablar:**
- IXTIYORIY kanalning yopilishini kuting (birinchi g'alaba qozonadi)
- Yopilgan kanal indeksini qaytaring
- Context birinchi bekor bo'lsa -1 va xato qaytaring

**Misol:**
\`\`\`go
signals := make([]<-chan struct{}, 3)
// 1-kanal birinchi yopiladi
idx, err := WaitAny(ctx, signals)
// idx == 1, err == nil
\`\`\`

**Maslahat:** Dinamik case soni uchun \`reflect.Select\` dan foydalaning.`,
                hint1: 'reflect.Select dinamik case sonini boshqaradi.',
                hint2: 'Tanlangan indeksni asl signal indeksiga qaytaring.',
                solutionCode: `package concurrency

import (
	"context"
	"reflect"
)

func WaitAny(ctx context.Context, signals []<-chan struct{}) (int, error) {
	if len(signals) == 0 {                                       // chekka holat: signallar yo'q
		if ctx != nil {                                      // agar context mavjud bo'lsa, uni kutamiz
			<-ctx.Done()                                 // bekor bo'lguncha bloklanamiz
			return -1, ctx.Err()                         // bekor qilish xatosini qaytaramiz
		}
		select {}                                            // ctx va signallar yo'q bo'lsa abadiy bloklanamiz
	}

	cases := make([]reflect.SelectCase, 0, len(signals)+1)       // select uchun dinamik case'lar
	indexMap := make([]int, 0, len(signals))                     // case indeksini signal indeksiga moslashtirish

	if ctx != nil {                                              // context.Done() ni case 0 sifatida qo'shamiz
		cases = append(cases, reflect.SelectCase{
			Dir:  reflect.SelectRecv,                    // qabul qilish yo'nalishi
			Chan: reflect.ValueOf(ctx.Done()),           // context bekor qilish kanali
		})
	}

	for i, sig := range signals {                                // har bir signalni case sifatida qo'shamiz
		if sig == nil {                                      // nil kanallarni o'tkazib yuboramiz
			continue
		}
		cases = append(cases, reflect.SelectCase{
			Dir:  reflect.SelectRecv,                    // signaldan qabul qilamiz
			Chan: reflect.ValueOf(sig),                  // signal kanali
		})
		indexMap = append(indexMap, i)                       // asl indeksni eslab qolamiz
	}

	chosen, _, _ := reflect.Select(cases)                        // dinamik select - biri tayyor bo'lguncha bloklanadi

	if ctx != nil && chosen == 0 {                               // context case g'alaba qildi
		return -1, ctx.Err()                                 // har qanday signaldan oldin bekor qilindi
	}

	offset := 0                                                  // asl signal indeksini hisoblaymiz
	if ctx != nil {
		offset = 1                                           // hisoblashda context case'ni o'tkazib yuboramiz
	}
	return indexMap[chosen-offset], nil                          // qaysi signal birinchi ishlaganini qaytaramiz
}`,
                whyItMatters: `Bir nechta operatsiyalarni poyga qildirish sezgir tizimlar yaratish uchun muhim. Bir nechta replikalarga so'rov yuborganingizda, ko'pincha eng tez javob kerak - kim birinchi javob bersa, u g'alaba qozonadi.

Bu pattern quyidagilarda paydo bo'ladi: hedged requests (bir nechta serverlarga yuborish, birinchi javobni ishlatish), failover tizimlari va service discovery (har qanday mavjud instance'ni topish).

reflect.Select - Go da dinamik kanal soni ustida select qilishning yagona usuli. U kompilyatsiya vaqtida kanal soni noma'lum bo'lgan moslashuvchan konkurrent tizimlar yaratish uchun zarur.`
              }
            }
          }
        ]
      },
      // Topic 3: Advanced Context Patterns
      {
        title: 'Advanced Context Patterns',
        description: 'Production-ready patterns for retries, heartbeats, and continuous operations.',
        difficulty: 'medium',
        estimatedTime: '1h',
        order: 3,
        translations: {
          ru: {
            title: 'Продвинутые паттерны Context',
            description: 'Production-ready паттерны для повторных попыток, heartbeat и непрерывных операций.'
          },
          uz: {
            title: 'Ilg\'or Context patternlari',
            description: 'Production-ready patternlar: retry, heartbeat va uzluksiz operatsiyalar.'
          }
        },
        tasks: [
          {
            slug: 'go-ctx-run-until',
            title: 'Run Until Signal Received',
            difficulty: 'medium',
            tags: ['go', 'context', 'loop', 'continuous'],
            estimatedTime: '15m',
            isPremium: false,
            description: `Continuously run a function until a stop signal is received or context is cancelled.

This pattern is common for background workers that need to process items until shutdown.

**Requirements:**
- Call f() repeatedly in a loop
- Stop when stop channel closes or context is cancelled
- Return error only if context was cancelled (not on clean stop)

**Example:**
\`\`\`go
stop := make(chan struct{})
counter := 0
go func() {
    time.Sleep(50 * time.Millisecond)
    close(stop)
}()
err := RunUntil(ctx, func() {
    counter++
}, stop)
// counter > 0, err == nil
\`\`\``,
            initialCode: `package concurrency

import "context"

// RunUntil runs f repeatedly until stop channel closes or ctx is cancelled.
func RunUntil(ctx context.Context, f func(), stop <-chan struct{}) error {
	// TODO: Implement continuous execution
	panic("not implemented")
}`,
            solutionCode: `package concurrency

import "context"

func RunUntil(ctx context.Context, f func(), stop <-chan struct{}) error {
	if ctx == nil {                                              // nil context safety
		ctx = context.Background()                           // use Background as default
	}
	for {                                                        // infinite loop
		select {                                             // non-blocking check
		case <-ctx.Done():                                   // context cancelled
			return ctx.Err()                             // return error (shutdown requested)
		case <-stop:                                         // clean stop signal received
			return nil                                   // no error on clean stop
		default:                                             // no signal received
			f()                                          // execute function and loop
		}
	}
}`,
            solutionExplanation: null,
            hint1: 'Use infinite for loop with select.',
            hint2: 'default case allows f() to run continuously.',
            whyItMatters: `This is the backbone of background workers and daemons. Most services have loops that run continuously: processing queue messages, polling for changes, serving requests, or maintaining connections.

The pattern separates two types of shutdown: clean stop (return nil) vs forced cancellation (return error). This distinction is important for logging, metrics, and deciding whether to restart the worker.

You'll use this for: message consumers, health check loops, cache refresh workers, connection heartbeats, and any long-running background task that needs graceful shutdown.`,
            order: 0,
            translations: {
              ru: {
                title: 'Запуск до получения сигнала',
                description: `Непрерывно выполняйте функцию пока не получен сигнал остановки или отмена контекста.

Этот паттерн распространен для фоновых воркеров.

**Требования:**
- Вызывайте f() повторно в цикле
- Остановитесь при закрытии stop канала или отмене контекста
- Верните ошибку только если контекст был отменен (не при чистой остановке)

**Пример:**
\`\`\`go
stop := make(chan struct{})
counter := 0
go func() {
    time.Sleep(50 * time.Millisecond)
    close(stop)
}()
err := RunUntil(ctx, func() {
    counter++
}, stop)
// counter > 0, err == nil
\`\`\``,
                hint1: 'Используйте бесконечный for цикл с select.',
                hint2: 'default case позволяет f() выполняться непрерывно.',
                solutionCode: `package concurrency

import "context"

func RunUntil(ctx context.Context, f func(), stop <-chan struct{}) error {
	if ctx == nil {                                              // безопасность nil контекста
		ctx = context.Background()                           // используем Background по умолчанию
	}
	for {                                                        // бесконечный цикл
		select {                                             // неблокирующая проверка
		case <-ctx.Done():                                   // контекст отменён
			return ctx.Err()                             // возвращаем ошибку (запрошено завершение)
		case <-stop:                                         // получен чистый сигнал остановки
			return nil                                   // нет ошибки при чистой остановке
		default:                                             // сигнал не получен
			f()                                          // выполняем функцию и продолжаем цикл
		}
	}
}`,
                whyItMatters: `Это основа фоновых воркеров и демонов. Большинство сервисов имеют циклы, работающие непрерывно: обработка сообщений очереди, polling изменений, обслуживание запросов, поддержание соединений.

Паттерн разделяет два типа завершения: чистая остановка (return nil) vs принудительная отмена (return error). Это различие важно для логирования и метрик.

Используйте для: потребителей сообщений, циклов health check, воркеров обновления кэша, heartbeat соединений и любых долгоживущих фоновых задач.`
              },
              uz: {
                title: 'Signal kelguncha ishlash',
                description: `To'xtatish signali kelmaguncha yoki context bekor bo'lmaguncha funktsiyani uzluksiz bajaring.

Bu pattern fon worker'lari uchun keng tarqalgan.

**Talablar:**
- f() ni siklda qayta-qayta chaqiring
- stop kanali yopilganda yoki context bekor bo'lganda to'xtating
- Faqat context bekor qilingan bo'lsa xato qaytaring (toza to'xtashda emas)

**Misol:**
\`\`\`go
stop := make(chan struct{})
counter := 0
go func() {
    time.Sleep(50 * time.Millisecond)
    close(stop)
}()
err := RunUntil(ctx, func() {
    counter++
}, stop)
// counter > 0, err == nil
\`\`\``,
                hint1: 'Select bilan cheksiz for sikldan foydalaning.',
                hint2: 'default case f() ni uzluksiz ishlashiga imkon beradi.',
                solutionCode: `package concurrency

import "context"

func RunUntil(ctx context.Context, f func(), stop <-chan struct{}) error {
	if ctx == nil {                                              // nil context xavfsizligi
		ctx = context.Background()                           // standart sifatida Background ishlatamiz
	}
	for {                                                        // cheksiz sikl
		select {                                             // bloklanmaydigan tekshiruv
		case <-ctx.Done():                                   // context bekor qilindi
			return ctx.Err()                             // xato qaytaramiz (to'xtatish so'raldi)
		case <-stop:                                         // toza to'xtatish signali olindi
			return nil                                   // toza to'xtashda xato yo'q
		default:                                             // signal olinmadi
			f()                                          // funktsiyani bajaramiz va siklni davom ettiramiz
		}
	}
}`,
                whyItMatters: `Bu fon worker'lari va daemon'larning asosi. Ko'pchilik servislar uzluksiz ishlaydigan sikllarga ega: navbat xabarlarini qayta ishlash, o'zgarishlarni polling qilish, so'rovlarga xizmat ko'rsatish, ulanishlarni saqlab turish.

Pattern ikkita to'xtash turini ajratadi: toza to'xtash (nil qaytarish) vs majburiy bekor qilish (xato qaytarish). Bu farq logging va metrikalar uchun muhim.

Budan foydalaning: xabar iste'molchilari, health check sikllari, kesh yangilash worker'lari, ulanish heartbeat'lari va graceful shutdown kerak bo'lgan har qanday uzoq muddatli fon vazifasi uchun.`
              }
            }
          },
          {
            slug: 'go-ctx-retry',
            title: 'Retry with Exponential Backoff',
            difficulty: 'medium',
            tags: ['go', 'context', 'retry', 'backoff', 'resilience'],
            estimatedTime: '25m',
            isPremium: true,
            description: `Retry a function with exponential backoff until it succeeds or context is cancelled.

Exponential backoff is essential for resilient systems - it prevents overwhelming a failing service and gives it time to recover.

**Requirements:**
- Retry f() up to maxRetries times
- Double the delay between each retry (exponential backoff)
- Start with initialDelay, cap at maxDelay
- Return nil on success, last error on max retries, ctx.Err() on cancellation

**Example:**
\`\`\`go
attempts := 0
err := RetryWithContext(ctx, func() error {
    attempts++
    if attempts < 3 {
        return errors.New("temporary error")
    }
    return nil
}, 3, 10*time.Millisecond, 100*time.Millisecond)
// err == nil, attempts == 3
\`\`\``,
            initialCode: `package concurrency

import (
	"context"
	"time"
)

// RetryWithContext retries f with exponential backoff.
func RetryWithContext(ctx context.Context, f func() error, maxRetries int, initialDelay, maxDelay time.Duration) error {
	// TODO: Implement retry with backoff
	panic("not implemented")
}`,
            solutionCode: `package concurrency

import (
	"context"
	"time"
)

func RetryWithContext(ctx context.Context, f func() error, maxRetries int, initialDelay, maxDelay time.Duration) error {
	if ctx == nil {                                              // nil context safety
		ctx = context.Background()                           // use Background as default
	}

	var lastErr error                                            // store last error for return
	delay := initialDelay                                        // current delay, will grow exponentially

	for i := 0; i < maxRetries; i++ {                            // retry loop
		select {                                             // check for cancellation first
		case <-ctx.Done():                                   // cancelled before attempt
			return ctx.Err()                             // don't retry if cancelled
		default:                                             // continue if not cancelled
		}

		if err := f(); err == nil {                          // try the function
			return nil                                   // success! stop retrying
		} else {
			lastErr = err                                // save error for later
		}

		if i < maxRetries-1 {                                // not last attempt - wait
			select {                                     // cancellable sleep
			case <-ctx.Done():                           // cancelled during sleep
				return ctx.Err()                     // abort immediately
			case <-time.After(delay):                    // sleep completed
			}
			delay *= 2                                   // exponential backoff: double delay
			if delay > maxDelay {                        // check against maximum
				delay = maxDelay                     // cap at maxDelay
			}
		}
	}
	return lastErr                                               // all retries exhausted
}`,
            solutionExplanation: null,
            hint1: 'Double delay after each failed attempt.',
            hint2: 'Check ctx.Done() before attempt and during sleep.',
            whyItMatters: `Exponential backoff is the standard approach for handling transient failures in distributed systems. Without it, clients hammer failing services, preventing recovery and causing cascading failures.

AWS, Google Cloud, and all major cloud providers recommend exponential backoff in their SDKs. The pattern is mandated by protocols like TCP congestion control and OAuth 2.0 error handling.

Key insight: the max delay cap prevents infinite wait times, while jitter (random variation) prevents thundering herd problems where all clients retry simultaneously. Production code often adds jitter to this pattern.`,
            order: 1,
            translations: {
              ru: {
                title: 'Повторная попытка с экспоненциальной задержкой',
                description: `Повторяйте функцию с экспоненциальной задержкой пока не успех или отмена контекста.

Экспоненциальная задержка важна для устойчивых систем.

**Требования:**
- Повторяйте f() до maxRetries раз
- Удваивайте задержку между каждой попыткой
- Начните с initialDelay, ограничьте maxDelay
- Верните nil при успехе, последнюю ошибку при исчерпании попыток

**Пример:**
\`\`\`go
attempts := 0
err := RetryWithContext(ctx, func() error {
    attempts++
    if attempts < 3 {
        return errors.New("временная ошибка")
    }
    return nil
}, 3, 10*time.Millisecond, 100*time.Millisecond)
// err == nil, attempts == 3
\`\`\``,
                hint1: 'Удваивайте delay после каждой неудачной попытки.',
                hint2: 'Проверяйте ctx.Done() перед попыткой и во время сна.',
                solutionCode: `package concurrency

import (
	"context"
	"time"
)

func RetryWithContext(ctx context.Context, f func() error, maxRetries int, initialDelay, maxDelay time.Duration) error {
	if ctx == nil {                                              // безопасность nil контекста
		ctx = context.Background()                           // используем Background по умолчанию
	}

	var lastErr error                                            // сохраняем последнюю ошибку для возврата
	delay := initialDelay                                        // текущая задержка, будет расти экспоненциально

	for i := 0; i < maxRetries; i++ {                            // цикл повторов
		select {                                             // сначала проверяем отмену
		case <-ctx.Done():                                   // отменено перед попыткой
			return ctx.Err()                             // не повторяем если отменено
		default:                                             // продолжаем если не отменено
		}

		if err := f(); err == nil {                          // пробуем функцию
			return nil                                   // успех! прекращаем повторы
		} else {
			lastErr = err                                // сохраняем ошибку
		}

		if i < maxRetries-1 {                                // не последняя попытка - ждём
			select {                                     // отменяемый сон
			case <-ctx.Done():                           // отменено во время сна
				return ctx.Err()                     // прерываем немедленно
			case <-time.After(delay):                    // сон завершён
			}
			delay *= 2                                   // экспоненциальная задержка: удваиваем
			if delay > maxDelay {                        // проверяем максимум
				delay = maxDelay                     // ограничиваем maxDelay
			}
		}
	}
	return lastErr                                               // все попытки исчерпаны
}`,
                whyItMatters: `Экспоненциальная задержка - стандартный подход для обработки временных сбоев в распределенных системах. Без неё клиенты забивают падающие сервисы, препятствуя восстановлению.

AWS, Google Cloud и все крупные облачные провайдеры рекомендуют exponential backoff в своих SDK. Паттерн требуется протоколами TCP и OAuth 2.0.

Важно: ограничение max delay предотвращает бесконечное ожидание, а jitter (случайная вариация) предотвращает thundering herd когда все клиенты делают retry одновременно.`
              },
              uz: {
                title: 'Eksponensial kechikish bilan qayta urinish',
                description: `Muvaffaqiyatga erishilmaguncha yoki context bekor bo'lmaguncha funktsiyani eksponensial kechikish bilan qayta urining.

Eksponensial kechikish barqaror tizimlar uchun muhim.

**Talablar:**
- f() ni maxRetries marta qayta urining
- Har bir urinish orasidagi kechikishni ikki baravar oshiring
- initialDelay dan boshlang, maxDelay bilan cheklang
- Muvaffaqiyatda nil, urinishlar tugaganda oxirgi xatoni qaytaring

**Misol:**
\`\`\`go
attempts := 0
err := RetryWithContext(ctx, func() error {
    attempts++
    if attempts < 3 {
        return errors.New("vaqtinchalik xato")
    }
    return nil
}, 3, 10*time.Millisecond, 100*time.Millisecond)
// err == nil, attempts == 3
\`\`\``,
                hint1: 'Har bir muvaffaqiyatsiz urinishdan keyin delay ni ikki baravar oshiring.',
                hint2: 'Urinishdan oldin va uyqu vaqtida ctx.Done() ni tekshiring.',
                solutionCode: `package concurrency

import (
	"context"
	"time"
)

func RetryWithContext(ctx context.Context, f func() error, maxRetries int, initialDelay, maxDelay time.Duration) error {
	if ctx == nil {                                              // nil context xavfsizligi
		ctx = context.Background()                           // standart sifatida Background ishlatamiz
	}

	var lastErr error                                            // qaytarish uchun oxirgi xatoni saqlaymiz
	delay := initialDelay                                        // joriy kechikish, eksponensial o'sadi

	for i := 0; i < maxRetries; i++ {                            // qayta urinish sikli
		select {                                             // avval bekor qilishni tekshiramiz
		case <-ctx.Done():                                   // urinishdan oldin bekor qilindi
			return ctx.Err()                             // bekor qilingan bo'lsa qayta urinmaymiz
		default:                                             // bekor qilinmagan bo'lsa davom etamiz
		}

		if err := f(); err == nil {                          // funktsiyani sinab ko'ramiz
			return nil                                   // muvaffaqiyat! qayta urinishni to'xtatamiz
		} else {
			lastErr = err                                // xatoni saqlaymiz
		}

		if i < maxRetries-1 {                                // oxirgi urinish emas - kutamiz
			select {                                     // bekor qilinadigan uyqu
			case <-ctx.Done():                           // uyqu vaqtida bekor qilindi
				return ctx.Err()                     // darhol to'xtatamiz
			case <-time.After(delay):                    // uyqu tugadi
			}
			delay *= 2                                   // eksponensial kechikish: ikki baravar oshiramiz
			if delay > maxDelay {                        // maksimumni tekshiramiz
				delay = maxDelay                     // maxDelay bilan cheklaymiz
			}
		}
	}
	return lastErr                                               // barcha urinishlar tugadi
}`,
                whyItMatters: `Eksponensial kechikish tarqatilgan tizimlarda vaqtinchalik nosozliklarni boshqarishning standart usuli. Usiz klientlar ishlamayotgan servislarni urib turishadi va tiklashga to'sqinlik qiladi.

AWS, Google Cloud va barcha yirik bulut provayderlari o'z SDK'larida exponential backoff'ni tavsiya qiladi. Bu pattern TCP va OAuth 2.0 protokollari tomonidan talab qilinadi.

Muhim nuqta: max delay cheklovi cheksiz kutishning oldini oladi, jitter (tasodifiy o'zgarish) esa barcha klientlar bir vaqtda qayta urinishda thundering herd muammosini oldini oladi.`
              }
            }
          },
          {
            slug: 'go-ctx-heartbeat',
            title: 'Heartbeat Monitor',
            difficulty: 'hard',
            tags: ['go', 'context', 'heartbeat', 'monitoring', 'health'],
            estimatedTime: '30m',
            isPremium: true,
            description: `Create a heartbeat that sends periodic signals and detects if a worker stops responding.

Heartbeats are crucial for distributed systems to detect unhealthy components.

**Requirements:**
- Return a heartbeat channel that receives periodic ticks
- Run doWork in a goroutine
- If doWork takes longer than interval, still send heartbeat
- Close heartbeat channel when doWork completes or context cancelled

**Example:**
\`\`\`go
heartbeat, results := Heartbeat(ctx, func(ctx context.Context) (string, error) {
    time.Sleep(50 * time.Millisecond)
    return "done", nil
}, 10*time.Millisecond)

count := 0
for range heartbeat {
    count++ // Receive heartbeats while work progresses
}
result := <-results
// count >= 4, result.Value == "done"
\`\`\``,
            initialCode: `package concurrency

import (
	"context"
	"time"
)

type HeartbeatResult[T any] struct {
	Value T
	Err   error
}

// Heartbeat sends periodic signals while doWork executes.
func Heartbeat[T any](ctx context.Context, doWork func(context.Context) (T, error), interval time.Duration) (<-chan struct{}, <-chan HeartbeatResult[T]) {
	// TODO: Implement heartbeat monitor
	panic("not implemented")
}`,
            solutionCode: `package concurrency

import (
	"context"
	"time"
)

type HeartbeatResult[T any] struct {
	Value T
	Err   error
}

func Heartbeat[T any](ctx context.Context, doWork func(context.Context) (T, error), interval time.Duration) (<-chan struct{}, <-chan HeartbeatResult[T]) {
	heartbeat := make(chan struct{}, 1)                          // buffered to prevent blocking
	results := make(chan HeartbeatResult[T], 1)                  // buffered for result delivery

	go func() {                                                  // coordinator goroutine
		defer close(heartbeat)                               // cleanup heartbeat channel
		defer close(results)                                 // cleanup results channel

		done := make(chan HeartbeatResult[T], 1)             // channel to receive work completion
		go func() {                                          // work goroutine
			v, err := doWork(ctx)                        // do the actual work
			done <- HeartbeatResult[T]{Value: v, Err: err} // send result
		}()

		ticker := time.NewTicker(interval)                   // periodic heartbeat ticker
		defer ticker.Stop()                                  // stop ticker on exit

		for {                                                // main loop
			select {                                     // multiplex three events
			case <-ctx.Done():                           // cancellation
				var zero T                           // zero value for error case
				results <- HeartbeatResult[T]{Value: zero, Err: ctx.Err()}
				return                               // exit coordinator
			case r := <-done:                            // work completed
				results <- r                         // forward result
				return                               // exit coordinator
			case <-ticker.C:                             // heartbeat interval elapsed
				select {                             // non-blocking send
				case heartbeat <- struct{}{}:        // send heartbeat tick
				default:                             // drop if no receiver ready
				}
			}
		}
	}()

	return heartbeat, results                                    // return both channels
}`,
            solutionExplanation: null,
            hint1: 'Use ticker for periodic heartbeats.',
            hint2: 'Buffer heartbeat channel, non-blocking send.',
            whyItMatters: `Heartbeats are the foundation of health monitoring in distributed systems. They allow you to detect when workers are stuck, connections are dead, or processes have hung without explicit failure.

This pattern is used in: load balancers (health checks), Kubernetes (liveness probes), database connection pools (keep-alive), and leader election (lease renewal).

The key insight is separating the work goroutine from the heartbeat goroutine. The heartbeat continues regardless of how long work takes, allowing observers to distinguish between "slow" and "dead" workers.`,
            order: 2,
            translations: {
              ru: {
                title: 'Монитор heartbeat',
                description: `Создайте heartbeat, отправляющий периодические сигналы и определяющий если воркер перестал отвечать.

Heartbeat критичны для распределенных систем.

**Требования:**
- Верните канал heartbeat с периодическими тиками
- Запустите doWork в горутине
- Если doWork занимает больше interval, все равно отправляйте heartbeat
- Закройте канал heartbeat при завершении doWork или отмене контекста

**Пример:**
\`\`\`go
heartbeat, results := Heartbeat(ctx, func(ctx context.Context) (string, error) {
    time.Sleep(50 * time.Millisecond)
    return "готово", nil
}, 10*time.Millisecond)

count := 0
for range heartbeat {
    count++ // Получаем heartbeat пока работа выполняется
}
result := <-results
// count >= 4, result.Value == "готово"
\`\`\``,
                hint1: 'Используйте ticker для периодических heartbeat.',
                hint2: 'Буферизуйте heartbeat канал, неблокирующая отправка.',
                solutionCode: `package concurrency

import (
	"context"
	"time"
)

type HeartbeatResult[T any] struct {
	Value T
	Err   error
}

func Heartbeat[T any](ctx context.Context, doWork func(context.Context) (T, error), interval time.Duration) (<-chan struct{}, <-chan HeartbeatResult[T]) {
	heartbeat := make(chan struct{}, 1)                          // буферизован для предотвращения блокировки
	results := make(chan HeartbeatResult[T], 1)                  // буферизован для доставки результата

	go func() {                                                  // горутина-координатор
		defer close(heartbeat)                               // очистка канала heartbeat
		defer close(results)                                 // очистка канала результатов

		done := make(chan HeartbeatResult[T], 1)             // канал для получения результата работы
		go func() {                                          // рабочая горутина
			v, err := doWork(ctx)                        // выполняем фактическую работу
			done <- HeartbeatResult[T]{Value: v, Err: err} // отправляем результат
		}()

		ticker := time.NewTicker(interval)                   // периодический тикер heartbeat
		defer ticker.Stop()                                  // останавливаем тикер при выходе

		for {                                                // основной цикл
			select {                                     // мультиплексируем три события
			case <-ctx.Done():                           // отмена
				var zero T                           // нулевое значение для случая ошибки
				results <- HeartbeatResult[T]{Value: zero, Err: ctx.Err()}
				return                               // выходим из координатора
			case r := <-done:                            // работа завершена
				results <- r                         // пересылаем результат
				return                               // выходим из координатора
			case <-ticker.C:                             // интервал heartbeat истёк
				select {                             // неблокирующая отправка
				case heartbeat <- struct{}{}:        // отправляем тик heartbeat
				default:                             // пропускаем если получатель не готов
				}
			}
		}
	}()

	return heartbeat, results                                    // возвращаем оба канала
}`,
                whyItMatters: `Heartbeat - основа мониторинга здоровья в распределенных системах. Они позволяют обнаружить когда воркеры зависли, соединения мертвы или процессы повисли без явного сбоя.

Этот паттерн используется в: балансировщиках нагрузки (health checks), Kubernetes (liveness probes), пулах соединений с БД (keep-alive), и выборе лидера (lease renewal).

Ключевая идея - разделение work goroutine и heartbeat goroutine. Heartbeat продолжается независимо от длительности работы, позволяя наблюдателям различать "медленных" и "мертвых" воркеров.`
              },
              uz: {
                title: 'Heartbeat monitoring',
                description: `Davriy signallar yuboradigan va worker javob berishni to'xtatganini aniqlashini yarating.

Heartbeat tarqatilgan tizimlar uchun muhim.

**Talablar:**
- Davriy tick'lar bilan heartbeat kanalini qaytaring
- doWork ni goroutine'da ishga tushiring
- doWork interval'dan ko'proq vaqt olsa ham, heartbeat yuborishda davom eting
- doWork tugaganda yoki context bekor bo'lganda heartbeat kanalini yoping

**Misol:**
\`\`\`go
heartbeat, results := Heartbeat(ctx, func(ctx context.Context) (string, error) {
    time.Sleep(50 * time.Millisecond)
    return "tayyor", nil
}, 10*time.Millisecond)

count := 0
for range heartbeat {
    count++ // Ish davom etayotganda heartbeat'larni qabul qilamiz
}
result := <-results
// count >= 4, result.Value == "tayyor"
\`\`\``,
                hint1: 'Davriy heartbeat uchun ticker\'dan foydalaning.',
                hint2: 'Heartbeat kanalini bufferlang, blokirovka qilmaydigan yuborish.',
                solutionCode: `package concurrency

import (
	"context"
	"time"
)

type HeartbeatResult[T any] struct {
	Value T
	Err   error
}

func Heartbeat[T any](ctx context.Context, doWork func(context.Context) (T, error), interval time.Duration) (<-chan struct{}, <-chan HeartbeatResult[T]) {
	heartbeat := make(chan struct{}, 1)                          // blokirovkani oldini olish uchun bufferli
	results := make(chan HeartbeatResult[T], 1)                  // natijani yetkazish uchun bufferli

	go func() {                                                  // koordinator goroutine
		defer close(heartbeat)                               // heartbeat kanalini tozalash
		defer close(results)                                 // natijalar kanalini tozalash

		done := make(chan HeartbeatResult[T], 1)             // ish natijasini qabul qilish kanali
		go func() {                                          // ishchi goroutine
			v, err := doWork(ctx)                        // haqiqiy ishni bajaramiz
			done <- HeartbeatResult[T]{Value: v, Err: err} // natijani yuboramiz
		}()

		ticker := time.NewTicker(interval)                   // davriy heartbeat ticker
		defer ticker.Stop()                                  // chiqishda ticker'ni to'xtatamiz

		for {                                                // asosiy sikl
			select {                                     // uchta voqeani multiplekslash
			case <-ctx.Done():                           // bekor qilish
				var zero T                           // xato holati uchun nol qiymat
				results <- HeartbeatResult[T]{Value: zero, Err: ctx.Err()}
				return                               // koordinatordan chiqamiz
			case r := <-done:                            // ish tugadi
				results <- r                         // natijani yo'naltiramiz
				return                               // koordinatordan chiqamiz
			case <-ticker.C:                             // heartbeat intervali o'tdi
				select {                             // blokirovka qilmaydigan yuborish
				case heartbeat <- struct{}{}:        // heartbeat tick'ni yuboramiz
				default:                             // qabul qiluvchi tayyor bo'lmasa o'tkazib yuboramiz
				}
			}
		}
	}()

	return heartbeat, results                                    // ikkala kanalni qaytaramiz
}`,
                whyItMatters: `Heartbeat tarqatilgan tizimlarda sog'liq monitoringining asosi. Ular worker'lar qotib qolganda, ulanishlar o'lganda yoki jarayonlar aniq nosozliksiz osilib qolganda aniqlash imkonini beradi.

Bu pattern quyidagilarda ishlatiladi: load balancer'lar (health check'lar), Kubernetes (liveness probe'lar), DB ulanish pool'lari (keep-alive) va lider saylovida (lease yangilash).

Asosiy g'oya - work goroutine va heartbeat goroutine'ni ajratish. Heartbeat ish qancha vaqt olishidan qat'i nazar davom etadi, bu kuzatuvchilarga "sekin" va "o'lik" worker'larni farqlash imkonini beradi.`
              }
            }
          }
        ]
      }
    ]
  },
  // Module 2: Concurrency Patterns
  {
    title: 'Concurrency Patterns',
    description: 'Master Go concurrency patterns: generators, pipelines, fan-in/fan-out, and worker pools.',
    section: 'core',
    order: 2,
    translations: {
      ru: {
        title: 'Паттерны конкурентности',
        description: 'Освойте паттерны конкурентности Go: генераторы, пайплайны, fan-in/fan-out и worker pools.'
      },
      uz: {
        title: 'Konkurrentlik patternlari',
        description: 'Go konkurrentlik patternlarini o\'rganing: generatorlar, pipeline\'lar, fan-in/fan-out va worker pool\'lar.'
      }
    },
    topics: [
      // Topic 1: Generator Pattern
      {
        title: 'Generator Pattern',
        description: 'Create data-producing channels that generate values on demand.',
        difficulty: 'easy',
        estimatedTime: '30m',
        order: 1,
        translations: {
          ru: {
            title: 'Паттерн генератор',
            description: 'Создавайте каналы, производящие данные по запросу.'
          },
          uz: {
            title: 'Generator pattern',
            description: 'Talab bo\'yicha qiymatlar ishlab chiqaradigan kanallarni yarating.'
          }
        },
        tasks: [
          {
            slug: 'go-pipeline-gen',
            title: 'Basic Generator',
            difficulty: 'easy',
            tags: ['go', 'channels', 'generator', 'pipeline'],
            estimatedTime: '10m',
            isPremium: false,
            description: `Create a generator function that converts a slice into a channel.

Generators are the first stage in Go pipelines - they produce data for downstream stages to process.

**Requirements:**
- Return a channel that yields all values from the input slice
- Close the channel after all values are sent
- Launch a goroutine to send values

**Example:**
\`\`\`go
ch := Gen(1, 2, 3, 4, 5)
for v := range ch {
    fmt.Println(v) // Prints 1, 2, 3, 4, 5
}
\`\`\``,
            initialCode: `package pipeline

// Gen creates a channel that yields all provided values.
func Gen[T any](values ...T) <-chan T {
	// TODO: Implement generator
	panic("not implemented")
}`,
            solutionCode: `package pipeline

func Gen[T any](values ...T) <-chan T {
	out := make(chan T)                                          // create output channel
	go func() {                                                  // spawn producer goroutine
		defer close(out)                                     // ALWAYS close when done - signals EOF
		for _, v := range values {                           // iterate all values
			out <- v                                     // send each value to channel
		}
	}()                                                          // goroutine starts immediately
	return out                                                   // return receive-only channel
}`,
            solutionExplanation: null,
            hint1: 'Create channel, goroutine sends values.',
            hint2: 'Always close channel with defer in goroutine.',
            whyItMatters: `Generators are the entry point of every pipeline. They convert static data (slices, files, API responses) into a stream that can flow through processing stages.

This pattern enables lazy evaluation - values are produced on demand, not all at once. A generator reading from a 10GB file doesn't load everything into memory; it streams one record at a time.

You'll use generators for: reading database cursors, parsing log files, transforming API paginated results, and any scenario where you want to process data incrementally rather than loading it all upfront.`,
            order: 0,
            translations: {
              ru: {
                title: 'Базовый генератор',
                description: `Создайте функцию-генератор, преобразующую срез в канал.

Генераторы - первая стадия в Go пайплайнах.

**Требования:**
- Верните канал, который выдает все значения из входного среза
- Закройте канал после отправки всех значений
- Запустите горутину для отправки значений

**Пример:**
\`\`\`go
ch := Gen(1, 2, 3, 4, 5)
for v := range ch {
    fmt.Println(v) // Выводит 1, 2, 3, 4, 5
}
\`\`\``,
                hint1: 'Создайте канал, горутина отправляет значения.',
                hint2: 'Всегда закрывайте канал с defer в горутине.',
                solutionCode: `package pipeline

func Gen[T any](values ...T) <-chan T {
	out := make(chan T)                                          // создаём выходной канал
	go func() {                                                  // запускаем горутину-производитель
		defer close(out)                                     // ВСЕГДА закрываем - сигнализирует EOF
		for _, v := range values {                           // итерируем все значения
			out <- v                                     // отправляем каждое значение в канал
		}
	}()                                                          // горутина запускается немедленно
	return out                                                   // возвращаем канал только для чтения
}`,
                whyItMatters: `Генераторы - точка входа каждого пайплайна. Они преобразуют статические данные (срезы, файлы, ответы API) в поток, который может проходить через стадии обработки.

Этот паттерн обеспечивает ленивую оценку - значения производятся по запросу, а не все сразу. Генератор, читающий 10GB файл, не загружает всё в память; он стримит по одной записи.

Используйте генераторы для: чтения курсоров БД, парсинга лог-файлов, преобразования пагинированных результатов API и обработки данных инкрементально.`
              },
              uz: {
                title: 'Asosiy generator',
                description: `Massivni kanalga aylantiruvchi generator funksiyasini yarating.

Generatorlar Go pipeline'larida birinchi bosqich.

**Talablar:**
- Kirish massividagi barcha qiymatlarni chiqaradigan kanal qaytaring
- Barcha qiymatlar yuborilgandan keyin kanalni yoping
- Qiymatlarni yuborish uchun goroutine ishga tushiring

**Misol:**
\`\`\`go
ch := Gen(1, 2, 3, 4, 5)
for v := range ch {
    fmt.Println(v) // 1, 2, 3, 4, 5 ni chiqaradi
}
\`\`\``,
                hint1: 'Kanal yarating, goroutine qiymatlarni yuboradi.',
                hint2: 'Har doim goroutine\'da defer bilan kanalni yoping.',
                solutionCode: `package pipeline

func Gen[T any](values ...T) <-chan T {
	out := make(chan T)                                          // chiqish kanalini yaratamiz
	go func() {                                                  // ishlab chiqaruvchi goroutine ishga tushiramiz
		defer close(out)                                     // DOIMO yopamiz - EOF ni bildiradi
		for _, v := range values {                           // barcha qiymatlarni iteratsiya qilamiz
			out <- v                                     // har bir qiymatni kanalga yuboramiz
		}
	}()                                                          // goroutine darhol boshlanadi
	return out                                                   // faqat o'qish uchun kanalni qaytaramiz
}`,
                whyItMatters: `Generatorlar har bir pipeline'ning kirish nuqtasi. Ular statik ma'lumotlarni (massivlar, fayllar, API javoblari) qayta ishlash bosqichlari orqali o'tadigan oqimga aylantiradi.

Bu pattern dangasa baholashni ta'minlaydi - qiymatlar talabga binoan ishlab chiqariladi, hammasi bir vaqtda emas. 10GB faylni o'qiydigan generator hammasini xotiraga yuklamaydi; bir vaqtda bitta yozuvni oqimda yuboradi.

Generatorlardan foydalaning: DB kursorlarini o'qish, log fayllarni tahlil qilish, API paginatsiya natijalarini o'zgartirish va ma'lumotlarni bosqichma-bosqich qayta ishlash uchun.`
              }
            }
          },
          {
            slug: 'go-pipeline-gen-ctx',
            title: 'Cancellable Generator',
            difficulty: 'easy',
            tags: ['go', 'channels', 'generator', 'context'],
            estimatedTime: '15m',
            isPremium: false,
            description: `Create a generator that respects context cancellation.

Production generators must stop when requested - otherwise they cause goroutine leaks.

**Requirements:**
- Stop sending values when context is cancelled
- Close the channel on cancellation or completion
- Use select to check for cancellation

**Example:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
ch := GenWithContext(ctx, 1, 2, 3, 4, 5)
<-ch // Receive 1
cancel()
// Channel will close, no more values
\`\`\``,
            initialCode: `package pipeline

import "context"

// GenWithContext creates a cancellable generator.
func GenWithContext[T any](ctx context.Context, values ...T) <-chan T {
	// TODO: Implement cancellable generator
	panic("not implemented")
}`,
            solutionCode: `package pipeline

import "context"

func GenWithContext[T any](ctx context.Context, values ...T) <-chan T {
	out := make(chan T)                                          // create output channel
	go func() {                                                  // spawn producer goroutine
		defer close(out)                                     // cleanup on exit
		for _, v := range values {                           // iterate all values
			select {                                     // cancellation-aware send
			case <-ctx.Done():                           // context cancelled
				return                               // stop immediately - prevent leak
			case out <- v:                               // successfully sent value
			}                                            // continue to next value
		}
	}()
	return out                                                   // return receive-only channel
}`,
            solutionExplanation: null,
            hint1: 'Wrap send in select with ctx.Done().',
            hint2: 'Return from goroutine on cancellation.',
            whyItMatters: `Without cancellation support, generators cause goroutine leaks. If a consumer stops reading (due to error, timeout, or finding enough data), the generator goroutine blocks forever on send.

This is one of the most common bugs in Go pipelines. Every production generator must check for cancellation on every send operation. Tools like goleak in tests help catch these issues.

Use this pattern universally: every channel send in a goroutine should be wrapped in a select with ctx.Done() to ensure clean shutdown. This applies to generators, pipeline stages, and worker pools.`,
            order: 1,
            translations: {
              ru: {
                title: 'Отменяемый генератор',
                description: `Создайте генератор, который реагирует на отмену контекста.

Production генераторы должны останавливаться по запросу.

**Требования:**
- Прекратите отправку значений при отмене контекста
- Закройте канал при отмене или завершении
- Используйте select для проверки отмены

**Пример:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
ch := GenWithContext(ctx, 1, 2, 3, 4, 5)
<-ch // Получаем 1
cancel()
// Канал закроется, больше значений не будет
\`\`\``,
                hint1: 'Оберните отправку в select с ctx.Done().',
                hint2: 'Выйдите из горутины при отмене.',
                solutionCode: `package pipeline

import "context"

func GenWithContext[T any](ctx context.Context, values ...T) <-chan T {
	out := make(chan T)                                          // создаём выходной канал
	go func() {                                                  // запускаем горутину-производитель
		defer close(out)                                     // очистка при выходе
		for _, v := range values {                           // итерируем все значения
			select {                                     // отправка с учётом отмены
			case <-ctx.Done():                           // контекст отменён
				return                               // останавливаемся - предотвращаем утечку
			case out <- v:                               // значение успешно отправлено
			}                                            // продолжаем к следующему значению
		}
	}()
	return out                                                   // возвращаем канал только для чтения
}`,
                whyItMatters: `Без поддержки отмены генераторы вызывают утечки горутин. Если потребитель перестает читать (из-за ошибки, таймаута или достаточного количества данных), горутина генератора блокируется навсегда.

Это одна из самых частых ошибок в Go пайплайнах. Каждый production генератор должен проверять отмену при каждой операции отправки.

Используйте этот паттерн везде: каждая отправка в канал в горутине должна быть обернута в select с ctx.Done() для обеспечения чистого завершения.`
              },
              uz: {
                title: 'Bekor qilinadigan generator',
                description: `Context bekor bo'lishiga hurmat qiladigan generator yarating.

Production generatorlar so'rov bo'yicha to'xtashi kerak.

**Talablar:**
- Context bekor bo'lganda qiymatlarni yuborishni to'xtating
- Bekor qilinganda yoki tugaganda kanalni yoping
- Bekor qilishni tekshirish uchun select dan foydalaning

**Misol:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
ch := GenWithContext(ctx, 1, 2, 3, 4, 5)
<-ch // 1 ni qabul qilamiz
cancel()
// Kanal yopiladi, boshqa qiymatlar yo'q
\`\`\``,
                hint1: 'Yuborishni ctx.Done() bilan select ichiga o\'rang.',
                hint2: 'Bekor qilinganda goroutine\'dan chiqing.',
                solutionCode: `package pipeline

import "context"

func GenWithContext[T any](ctx context.Context, values ...T) <-chan T {
	out := make(chan T)                                          // chiqish kanalini yaratamiz
	go func() {                                                  // ishlab chiqaruvchi goroutine ishga tushiramiz
		defer close(out)                                     // chiqishda tozalash
		for _, v := range values {                           // barcha qiymatlarni iteratsiya qilamiz
			select {                                     // bekor qilishni hisobga olgan yuborish
			case <-ctx.Done():                           // context bekor qilindi
				return                               // darhol to'xtaymiz - oqishni oldini olamiz
			case out <- v:                               // qiymat muvaffaqiyatli yuborildi
			}                                            // keyingi qiymatga davom etamiz
		}
	}()
	return out                                                   // faqat o'qish uchun kanalni qaytaramiz
}`,
                whyItMatters: `Bekor qilishni qo'llab-quvvatlamasdan generatorlar goroutine oqishlariga sabab bo'ladi. Agar iste'molchi o'qishni to'xtatsa (xato, timeout yoki yetarli ma'lumot tufayli), generator goroutine'i yuborishda abadiy blokirovkalanadi.

Bu Go pipeline'laridagi eng keng tarqalgan xatolardan biri. Har bir production generator har bir yuborish operatsiyasida bekor qilishni tekshirishi kerak.

Bu patternni hamma joyda ishlating: goroutine'dagi har bir kanal yuborishi toza to'xtashni ta'minlash uchun ctx.Done() bilan select ichiga o'ralishi kerak.`
              }
            }
          }
        ]
      },
      // Topic 2: Pipeline Stages
      {
        title: 'Pipeline Stages',
        description: 'Build transformation stages that process data flowing through channels.',
        difficulty: 'medium',
        estimatedTime: '1h',
        order: 2,
        translations: {
          ru: {
            title: 'Стадии пайплайна',
            description: 'Создавайте стадии трансформации для обработки данных, проходящих через каналы.'
          },
          uz: {
            title: 'Pipeline bosqichlari',
            description: 'Kanallar orqali o\'tayotgan ma\'lumotlarni qayta ishlaydigan transformatsiya bosqichlarini yarating.'
          }
        },
        tasks: [
          {
            slug: 'go-pipeline-square',
            title: 'Square Numbers Stage',
            difficulty: 'easy',
            tags: ['go', 'channels', 'pipeline', 'transform'],
            estimatedTime: '10m',
            isPremium: false,
            description: `Create a pipeline stage that squares each incoming number.

Pipeline stages transform data as it flows through - each stage reads from input, processes, and writes to output.

**Requirements:**
- Read integers from input channel
- Square each value and send to output
- Close output when input is exhausted

**Example:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5)
out := Square(in)
// out yields: 1, 4, 9, 16, 25
\`\`\``,
            initialCode: `package pipeline

// Square reads integers and outputs their squares.
func Square(in <-chan int) <-chan int {
	// TODO: Implement square stage
	panic("not implemented")
}`,
            solutionCode: `package pipeline

func Square(in <-chan int) <-chan int {
	out := make(chan int)                                        // create output channel
	go func() {                                                  // spawn transformer goroutine
		defer close(out)                                     // close output when input exhausted
		for n := range in {                                  // range stops when in closes
			out <- n * n                                 // square and forward
		}
	}()
	return out                                                   // return transformed stream
}`,
            solutionExplanation: null,
            hint1: 'Range over input channel until closed.',
            hint2: 'Send n*n to output channel.',
            whyItMatters: `This is the simplest form of a pipeline stage - a pure transformation. It demonstrates the composability of Go pipelines: the output of Square can feed into another stage, forming a processing chain.

Pipeline stages are the building blocks of stream processing systems. Each stage does one thing well, making the code testable, reusable, and easy to reason about.

You'll compose these into larger pipelines: Gen(data) → Filter(isValid) → Transform(process) → Reduce(aggregate). Each stage runs concurrently, enabling parallel processing of streaming data.`,
            order: 0,
            translations: {
              ru: {
                title: 'Стадия возведения в квадрат',
                description: `Создайте стадию пайплайна, которая возводит в квадрат каждое входящее число.

Стадии пайплайна трансформируют данные при прохождении.

**Требования:**
- Читайте целые числа из входного канала
- Возведите каждое значение в квадрат и отправьте на выход
- Закройте выход когда вход исчерпан

**Пример:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5)
out := Square(in)
// out выдаёт: 1, 4, 9, 16, 25
\`\`\``,
                hint1: 'Range по входному каналу до закрытия.',
                hint2: 'Отправьте n*n в выходной канал.',
                solutionCode: `package pipeline

func Square(in <-chan int) <-chan int {
	out := make(chan int)                                        // создаём выходной канал
	go func() {                                                  // запускаем горутину-трансформер
		defer close(out)                                     // закрываем выход когда вход исчерпан
		for n := range in {                                  // range останавливается когда in закрыт
			out <- n * n                                 // возводим в квадрат и пересылаем
		}
	}()
	return out                                                   // возвращаем трансформированный поток
}`,
                whyItMatters: `Это простейшая форма стадии пайплайна - чистая трансформация. Она демонстрирует компонуемость Go пайплайнов: выход Square может питать другую стадию, формируя цепочку обработки.

Стадии пайплайна - строительные блоки систем потоковой обработки. Каждая стадия делает одно дело хорошо, делая код тестируемым и легким для понимания.

Вы будете составлять их в большие пайплайны: Gen(data) → Filter(isValid) → Transform(process) → Reduce(aggregate).`
              },
              uz: {
                title: 'Kvadratga ko\'tarish bosqichi',
                description: `Har bir kelayotgan sonni kvadratga ko'taradigan pipeline bosqichini yarating.

Pipeline bosqichlari ma'lumotlarni o'tish jarayonida o'zgartiradi.

**Talablar:**
- Kirish kanalidan butun sonlarni o'qing
- Har bir qiymatni kvadratga ko'taring va chiqishga yuboring
- Kirish tugaganda chiqishni yoping

**Misol:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5)
out := Square(in)
// out chiqaradi: 1, 4, 9, 16, 25
\`\`\``,
                hint1: 'Kirish kanali yopilguncha range qiling.',
                hint2: 'n*n ni chiqish kanaliga yuboring.',
                solutionCode: `package pipeline

func Square(in <-chan int) <-chan int {
	out := make(chan int)                                        // chiqish kanalini yaratamiz
	go func() {                                                  // transformator goroutine ishga tushiramiz
		defer close(out)                                     // kirish tugaganda chiqishni yopamiz
		for n := range in {                                  // in yopilganda range to'xtaydi
			out <- n * n                                 // kvadratga ko'taramiz va yo'naltiramiz
		}
	}()
	return out                                                   // transformatsiya qilingan oqimni qaytaramiz
}`,
                whyItMatters: `Bu pipeline bosqichining eng oddiy shakli - sof transformatsiya. U Go pipeline'larining kompozitsiyalanishini ko'rsatadi: Square chiqishi boshqa bosqichni oziqlantirishi mumkin.

Pipeline bosqichlari oqim qayta ishlash tizimlarining qurilish bloklari. Har bir bosqich bitta ishni yaxshi bajaradi, kodni test qilinadigan va tushunarli qiladi.

Siz ularni kattaroq pipeline'larga birlashtirasiz: Gen(data) → Filter(isValid) → Transform(process) → Reduce(aggregate).`
              }
            }
          },
          {
            slug: 'go-pipeline-filter',
            title: 'Filter Stage',
            difficulty: 'easy',
            tags: ['go', 'channels', 'pipeline', 'filter'],
            estimatedTime: '15m',
            isPremium: false,
            description: `Create a filter stage that passes only values matching a predicate.

Filters are essential pipeline components - they remove unwanted data early in the pipeline.

**Requirements:**
- Apply predicate function to each value
- Only send values where predicate returns true
- Support context cancellation

**Example:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5, 6)
isEven := func(n int) bool { return n%2 == 0 }
out := FilterStage(ctx, in, isEven)
// out yields: 2, 4, 6
\`\`\``,
            initialCode: `package pipeline

import "context"

// FilterStage passes only values where predicate returns true.
func FilterStage[T any](ctx context.Context, in <-chan T, predicate func(T) bool) <-chan T {
	// TODO: Implement filter stage
	panic("not implemented")
}`,
            solutionCode: `package pipeline

import "context"

func FilterStage[T any](ctx context.Context, in <-chan T, predicate func(T) bool) <-chan T {
	out := make(chan T)                                          // output channel for filtered values
	go func() {                                                  // spawn filter goroutine
		defer close(out)                                     // cleanup on exit
		for v := range in {                                  // read all input values
			if !predicate(v) {                           // check predicate condition
				continue                             // skip non-matching values
			}
			select {                                     // cancellation-aware send
			case <-ctx.Done():                           // cancelled - stop filtering
				return
			case out <- v:                               // forward matching value
			}
		}
	}()
	return out                                                   // return filtered stream
}`,
            solutionExplanation: null,
            hint1: 'Skip values where predicate returns false.',
            hint2: 'Wrap output send in select for cancellation.',
            whyItMatters: `Filters reduce data volume early in the pipeline, improving performance. By eliminating unwanted data before expensive transformations, you avoid wasting CPU and memory.

This is the "fail fast" principle applied to streaming: if data doesn't match criteria, don't process it further. In data pipelines processing millions of records, early filtering can reduce processing time by orders of magnitude.

Use filters for: validation (skip invalid records), authorization (skip unauthorized items), deduplication, and any scenario where you want to reduce the data flowing downstream.`,
            order: 1,
            translations: {
              ru: {
                title: 'Стадия фильтрации',
                description: `Создайте стадию фильтра, пропускающую только значения, соответствующие предикату.

Фильтры удаляют ненужные данные на ранних стадиях пайплайна.

**Требования:**
- Применяйте функцию-предикат к каждому значению
- Отправляйте только значения где предикат возвращает true
- Поддержите отмену контекста

**Пример:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5, 6)
isEven := func(n int) bool { return n%2 == 0 }
out := FilterStage(ctx, in, isEven)
// out выдаёт: 2, 4, 6
\`\`\``,
                hint1: 'Пропускайте значения где predicate возвращает false.',
                hint2: 'Оберните отправку в select для отмены.',
                solutionCode: `package pipeline

import "context"

func FilterStage[T any](ctx context.Context, in <-chan T, predicate func(T) bool) <-chan T {
	out := make(chan T)                                          // выходной канал для отфильтрованных значений
	go func() {                                                  // запускаем горутину-фильтр
		defer close(out)                                     // очистка при выходе
		for v := range in {                                  // читаем все входные значения
			if !predicate(v) {                           // проверяем условие предиката
				continue                             // пропускаем несоответствующие значения
			}
			select {                                     // отправка с учётом отмены
			case <-ctx.Done():                           // отменено - прекращаем фильтрацию
				return
			case out <- v:                               // пересылаем подходящее значение
			}
		}
	}()
	return out                                                   // возвращаем отфильтрованный поток
}`,
                whyItMatters: `Фильтры уменьшают объем данных на ранних стадиях пайплайна, улучшая производительность. Устраняя ненужные данные до дорогих трансформаций, вы экономите CPU и память.

Это принцип "fail fast" применённый к стримингу: если данные не соответствуют критериям, не обрабатывайте их дальше. В пайплайнах с миллионами записей ранняя фильтрация может уменьшить время обработки на порядки.

Используйте фильтры для: валидации, авторизации, дедупликации и уменьшения данных идущих дальше.`
              },
              uz: {
                title: 'Filtrlash bosqichi',
                description: `Faqat predikatga mos keladigan qiymatlarni o'tkazadigan filtr bosqichini yarating.

Filtrlar pipeline'ning erta bosqichlarida keraksiz ma'lumotlarni olib tashlaydi.

**Talablar:**
- Predikat funksiyasini har bir qiymatga qo'llang
- Faqat predikat true qaytargan qiymatlarni yuboring
- Context bekor qilishni qo'llab-quvvatlang

**Misol:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5, 6)
isEven := func(n int) bool { return n%2 == 0 }
out := FilterStage(ctx, in, isEven)
// out natijasi: 2, 4, 6
\`\`\``,
                hint1: 'Predicate false qaytargan qiymatlarni o\'tkazib yuboring.',
                hint2: 'Yuborishni bekor qilish uchun select ichiga o\'rang.',
                solutionCode: `package pipeline

import "context"

func FilterStage[T any](ctx context.Context, in <-chan T, predicate func(T) bool) <-chan T {
	out := make(chan T)                                          // filtrlangan qiymatlar uchun chiqish kanali
	go func() {                                                  // filtr goroutine ishga tushiramiz
		defer close(out)                                     // chiqishda tozalash
		for v := range in {                                  // barcha kirish qiymatlarini o'qiymiz
			if !predicate(v) {                           // predikat shartini tekshiramiz
				continue                             // mos kelmaganlarni o'tkazib yuboramiz
			}
			select {                                     // bekor qilishni hisobga olgan yuborish
			case <-ctx.Done():                           // bekor qilindi - filtrlashni to'xtatamiz
				return
			case out <- v:                               // mos qiymatni uzatamiz
			}
		}
	}()
	return out                                                   // filtrlangan oqimni qaytaramiz
}`,
                whyItMatters: `Filtrlar pipeline'ning erta bosqichlarida ma'lumotlar hajmini kamaytiradi va unumdorlikni yaxshilaydi. Qimmat transformatsiyalardan oldin keraksiz ma'lumotlarni yo'q qilib, CPU va xotirani tejaysiz.

Bu oqimga qo'llaniladigan "tez muvaffaqiyatsizlik" tamoyili: agar ma'lumotlar mezonlarga mos kelmasa, ularni qayta ishlamang. Millionlab yozuvlarni qayta ishlaydigan pipeline'larda erta filtrlash qayta ishlash vaqtini darajalar bo'yicha kamaytirishi mumkin.

Filtrlardan foydalaning: validatsiya, avtorizatsiya, duplikatlarni olib tashlash va pastga oqayotgan ma'lumotlarni kamaytirish uchun.`
              }
            }
          },
          {
            slug: 'go-pipeline-take',
            title: 'Take N Elements Stage',
            difficulty: 'medium',
            tags: ['go', 'channels', 'pipeline', 'limit'],
            estimatedTime: '15m',
            isPremium: false,
            description: `Create a stage that takes only the first N elements from the input.

This is useful for limiting data flow - especially with infinite generators.

**Requirements:**
- Pass through only the first n values
- Close output after n values (or when input exhausted)
- Support context cancellation

**Example:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
out := TakeStage(ctx, in, 3)
// out yields: 1, 2, 3 (then closes)
\`\`\``,
            initialCode: `package pipeline

import "context"

// TakeStage passes only the first n values from input.
func TakeStage[T any](ctx context.Context, in <-chan T, n int) <-chan T {
	// TODO: Implement take stage
	panic("not implemented")
}`,
            solutionCode: `package pipeline

import "context"

func TakeStage[T any](ctx context.Context, in <-chan T, n int) <-chan T {
	out := make(chan T)                                          // output for limited elements
	go func() {                                                  // spawn limiter goroutine
		defer close(out)                                     // cleanup on exit
		count := 0                                           // track elements taken
		for v := range in {                                  // read from input
			if count >= n {                              // reached limit
				return                               // stop - we have enough
			}
			select {                                     // cancellation-aware send
			case <-ctx.Done():                           // cancelled
				return
			case out <- v:                               // forward value
				count++                              // increment counter
			}
		}
	}()
	return out                                                   // return limited stream
}`,
            solutionExplanation: null,
            hint1: 'Count elements sent, stop when reaching n.',
            hint2: 'Return early once count >= n.',
            whyItMatters: `Take is essential for working with infinite or large streams. It lets you sample data, implement pagination, or stop processing once you have enough results.

Combined with generators, Take enables efficient "find first N matching items" patterns. The pipeline stops as soon as N items are found, without processing the entire input.

Use Take for: pagination (take N items per page), sampling (take first N for preview), limiting results from infinite generators, and any "top N" or "first N" queries in streaming contexts.`,
            order: 2,
            translations: {
              ru: {
                title: 'Стадия взятия N элементов',
                description: `Создайте стадию, которая берет только первые N элементов из входа.

Полезно для ограничения потока данных - особенно с бесконечными генераторами.

**Требования:**
- Пропускайте только первые n значений
- Закройте выход после n значений
- Поддержите отмену контекста

**Пример:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
out := TakeStage(ctx, in, 3)
// out выдаёт: 1, 2, 3 (затем закрывается)
\`\`\``,
                hint1: 'Считайте отправленные элементы, остановитесь при достижении n.',
                hint2: 'Выйдите досрочно когда count >= n.',
                solutionCode: `package pipeline

import "context"

func TakeStage[T any](ctx context.Context, in <-chan T, n int) <-chan T {
	out := make(chan T)                                          // выход для ограниченных элементов
	go func() {                                                  // запускаем горутину-ограничитель
		defer close(out)                                     // очистка при выходе
		count := 0                                           // отслеживаем взятые элементы
		for v := range in {                                  // читаем из входа
			if count >= n {                              // достигнут лимит
				return                               // останавливаемся - достаточно
			}
			select {                                     // отправка с учётом отмены
			case <-ctx.Done():                           // отменено
				return
			case out <- v:                               // пересылаем значение
				count++                              // увеличиваем счётчик
			}
		}
	}()
	return out                                                   // возвращаем ограниченный поток
}`,
                whyItMatters: `Take необходим для работы с бесконечными или большими потоками. Он позволяет сэмплировать данные, реализовать пагинацию или остановить обработку когда достаточно результатов.

В сочетании с генераторами, Take обеспечивает эффективные паттерны "найти первые N подходящих элементов". Пайплайн останавливается как только N найдено.

Используйте Take для: пагинации, сэмплирования, ограничения результатов от бесконечных генераторов и любых "top N" запросов.`
              },
              uz: {
                title: 'N element olish bosqichi',
                description: `Kiruvchidan faqat birinchi N elementni oladigan bosqichni yarating.

Ma'lumotlar oqimini cheklash uchun foydali - ayniqsa cheksiz generatorlar bilan.

**Talablar:**
- Faqat birinchi n qiymatni o'tkazing
- n qiymatdan keyin chiqishni yoping
- Context bekor qilishni qo'llab-quvvatlang

**Misol:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
out := TakeStage(ctx, in, 3)
// out natijasi: 1, 2, 3 (keyin yopiladi)
\`\`\``,
                hint1: 'Yuborilgan elementlarni hisoblang, n ga yetganda to\'xtang.',
                hint2: 'count >= n bo\'lganda erta chiqing.',
                solutionCode: `package pipeline

import "context"

func TakeStage[T any](ctx context.Context, in <-chan T, n int) <-chan T {
	out := make(chan T)                                          // cheklangan elementlar uchun chiqish
	go func() {                                                  // cheklovchi goroutine ishga tushiramiz
		defer close(out)                                     // chiqishda tozalash
		count := 0                                           // olingan elementlarni kuzatamiz
		for v := range in {                                  // kiruvchidan o'qiymiz
			if count >= n {                              // chegaraga yetildi
				return                               // to'xtatamiz - yetarli
			}
			select {                                     // bekor qilishni hisobga olgan yuborish
			case <-ctx.Done():                           // bekor qilindi
				return
			case out <- v:                               // qiymatni uzatamiz
				count++                              // hisoblagichni oshiramiz
			}
		}
	}()
	return out                                                   // cheklangan oqimni qaytaramiz
}`,
                whyItMatters: `Take cheksiz yoki katta oqimlar bilan ishlash uchun muhim. U ma'lumotlarni namuna olish, paginatsiya amalga oshirish yoki yetarli natijaga ega bo'lganda qayta ishlashni to'xtatish imkonini beradi.

Generatorlar bilan birgalikda Take "birinchi N mos elementni topish" patternlarini samarali qiladi. Pipeline N element topilishi bilan to'xtaydi.

Take dan foydalaning: paginatsiya, namuna olish, cheksiz generatorlardan natijalarni cheklash va har qanday "top N" so'rovlari uchun.`
              }
            }
          }
        ]
      },
      // Topic 3: Fan-In Pattern
      {
        title: 'Fan-In Pattern',
        description: 'Merge multiple input channels into a single output channel.',
        difficulty: 'medium',
        estimatedTime: '45m',
        order: 3,
        translations: {
          ru: {
            title: 'Паттерн Fan-In',
            description: 'Объединяйте несколько входных каналов в один выходной канал.'
          },
          uz: {
            title: 'Fan-In pattern',
            description: 'Bir nechta kirish kanallarini bitta chiqish kanaliga birlashtiring.'
          }
        },
        tasks: [
          {
            slug: 'go-pipeline-fanin',
            title: 'Merge Multiple Channels',
            difficulty: 'medium',
            tags: ['go', 'channels', 'fan-in', 'sync'],
            estimatedTime: '25m',
            isPremium: false,
            description: `Implement the Fan-In pattern to merge multiple input channels into one output.

Fan-In is essential for parallelizing work - split data across workers, then merge results.

**Requirements:**
- Accept variadic number of input channels
- Output channel receives values from all inputs (order may vary)
- Close output only when ALL inputs are closed
- Support context cancellation
- No goroutine leaks

**Example:**
\`\`\`go
ch1 := Gen(1, 3, 5)
ch2 := Gen(2, 4, 6)
merged := FanIn(ctx, ch1, ch2)
// merged yields: some interleaving of 1,2,3,4,5,6
\`\`\``,
            initialCode: `package pipeline

import (
	"context"
	"sync"
)

// FanIn merges multiple channels into one.
func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
	// TODO: Implement fan-in
	panic("not implemented")
}`,
            solutionCode: `package pipeline

import (
	"context"
	"sync"
)

func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
	out := make(chan T)                                          // single merged output channel
	var wg sync.WaitGroup                                        // track active forwarder goroutines

	forward := func(in <-chan T) {                               // forwarder: one per input
		defer wg.Done()                                      // signal completion when done
		for v := range in {                                  // read until input closes
			select {                                     // cancellation-aware send
			case <-ctx.Done():                           // cancelled - stop forwarding
				return
			case out <- v:                               // forward value to output
			}
		}
	}

	for _, in := range ins {                                     // launch forwarders
		if in == nil {                                       // skip nil channels
			continue
		}
		wg.Add(1)                                            // track this forwarder
		go forward(in)                                       // start forwarding
	}

	go func() {                                                  // closer goroutine
		wg.Wait()                                            // wait for all forwarders
		close(out)                                           // safe to close now
	}()

	return out                                                   // return merged stream
}`,
            solutionExplanation: null,
            hint1: 'Use sync.WaitGroup to track forwarders.',
            hint2: 'One goroutine per input, close output when all done.',
            whyItMatters: `Fan-In is essential for parallelizing work. Split data across multiple workers, process in parallel, then merge results back into a single stream.

This pattern enables horizontal scaling: double the workers, roughly double the throughput. It's the basis for parallel map-reduce, concurrent API calls, and distributed processing.

Use Fan-In when: aggregating results from multiple sources (multi-region queries), merging parallel worker outputs, combining data from multiple API endpoints, or any scenario where you split work and need to rejoin results.`,
            order: 0,
            translations: {
              ru: {
                title: 'Объединение нескольких каналов',
                description: `Реализуйте паттерн Fan-In для объединения нескольких входных каналов в один выход.

Fan-In необходим для параллелизации работы.

**Требования:**
- Принимайте вариативное количество входных каналов
- Выходной канал получает значения от всех входов
- Закройте выход только когда ВСЕ входы закрыты
- Поддержите отмену контекста
- Без утечек горутин

**Пример:**
\`\`\`go
ch1 := Gen(1, 3, 5)
ch2 := Gen(2, 4, 6)
merged := FanIn(ctx, ch1, ch2)
// merged выдаёт: некоторое чередование 1,2,3,4,5,6
\`\`\``,
                hint1: 'Используйте sync.WaitGroup для отслеживания форвардеров.',
                hint2: 'Одна горутина на вход, закройте выход когда все завершены.',
                solutionCode: `package pipeline

import (
	"context"
	"sync"
)

func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
	out := make(chan T)                                          // единый объединённый выходной канал
	var wg sync.WaitGroup                                        // отслеживаем активные горутины-форвардеры

	forward := func(in <-chan T) {                               // форвардер: один на каждый вход
		defer wg.Done()                                      // сигнализируем завершение
		for v := range in {                                  // читаем пока вход не закроется
			select {                                     // отправка с учётом отмены
			case <-ctx.Done():                           // отменено - прекращаем форвардинг
				return
			case out <- v:                               // пересылаем значение на выход
			}
		}
	}

	for _, in := range ins {                                     // запускаем форвардеры
		if in == nil {                                       // пропускаем nil каналы
			continue
		}
		wg.Add(1)                                            // отслеживаем этот форвардер
		go forward(in)                                       // запускаем форвардинг
	}

	go func() {                                                  // горутина-закрыватель
		wg.Wait()                                            // ждём все форвардеры
		close(out)                                           // теперь безопасно закрыть
	}()

	return out                                                   // возвращаем объединённый поток
}`,
                whyItMatters: `Fan-In необходим для параллелизации работы. Разделите данные между воркерами, обработайте параллельно, затем объедините результаты в один поток.

Этот паттерн обеспечивает горизонтальное масштабирование: удвойте воркеров - примерно удвоите пропускную способность. Это основа parallel map-reduce и распределённой обработки.

Используйте Fan-In когда: агрегируете результаты из нескольких источников, объединяете выходы параллельных воркеров, комбинируете данные из нескольких API.`
              },
              uz: {
                title: 'Bir nechta kanallarni birlashtirish',
                description: `Bir nechta kirish kanallarini bitta chiqishga birlashtirish uchun Fan-In patternini amalga oshiring.

Fan-In ishni parallellashtirish uchun zarur.

**Talablar:**
- Variativ kirish kanallarini qabul qiling
- Chiqish kanali barcha kirishlardan qiymatlar oladi
- Chiqishni faqat BARCHA kirishlar yopilganda yoping
- Context bekor qilishni qo'llab-quvvatlang
- Goroutine oqishlarisiz

**Misol:**
\`\`\`go
ch1 := Gen(1, 3, 5)
ch2 := Gen(2, 4, 6)
merged := FanIn(ctx, ch1, ch2)
// merged natijasi: 1,2,3,4,5,6 ning qandaydir aralashmasi
\`\`\``,
                hint1: 'Forwarder\'larni kuzatish uchun sync.WaitGroup dan foydalaning.',
                hint2: 'Har bir kirish uchun bitta goroutine, hammasi tugaganda chiqishni yoping.',
                solutionCode: `package pipeline

import (
	"context"
	"sync"
)

func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
	out := make(chan T)                                          // yagona birlashtirilgan chiqish kanali
	var wg sync.WaitGroup                                        // faol forwarder goroutine'larni kuzatamiz

	forward := func(in <-chan T) {                               // forwarder: har bir kirish uchun bittadan
		defer wg.Done()                                      // tugallanganda signal beramiz
		for v := range in {                                  // kirish yopilguncha o'qiymiz
			select {                                     // bekor qilishni hisobga olgan yuborish
			case <-ctx.Done():                           // bekor qilindi - forwardingni to'xtatamiz
				return
			case out <- v:                               // qiymatni chiqishga uzatamiz
			}
		}
	}

	for _, in := range ins {                                     // forwarder'larni ishga tushiramiz
		if in == nil {                                       // nil kanallarni o'tkazib yuboramiz
			continue
		}
		wg.Add(1)                                            // bu forwarder'ni kuzatamiz
		go forward(in)                                       // forwardingni boshlaymiz
	}

	go func() {                                                  // yopuvchi goroutine
		wg.Wait()                                            // barcha forwarder'larni kutamiz
		close(out)                                           // endi yopish xavfsiz
	}()

	return out                                                   // birlashtirilgan oqimni qaytaramiz
}`,
                whyItMatters: `Fan-In ishni parallellashtirish uchun zarur. Ma'lumotlarni bir nechta worker'lar o'rtasida bo'ling, parallel qayta ishlang, keyin natijalarni bitta oqimga birlashtiring.

Bu pattern gorizontal masshtablashni ta'minlaydi: worker'larni ikki baravar oshiring - o'tkazuvchanlikni taxminan ikki baravar oshiring. Bu parallel map-reduce va taqsimlangan qayta ishlashning asosi.

Fan-In dan foydalaning: bir nechta manbalardan natijalarni birlashtirish, parallel worker chiqishlarini birlashtirish, bir nechta API'lardan ma'lumotlarni birlashtirish uchun.`
              }
            }
          },
          {
            slug: 'go-pipeline-sum',
            title: 'Sum Pipeline Values',
            difficulty: 'easy',
            tags: ['go', 'channels', 'aggregate', 'reduce'],
            estimatedTime: '10m',
            isPremium: false,
            description: `Create a function that sums all values from a channel.

This is a terminal/sink operation - it consumes the entire channel and returns a single result.

**Requirements:**
- Read all integers from the channel
- Return the sum when channel closes

**Example:**
\`\`\`go
ch := Gen(1, 2, 3, 4, 5)
total := Sum(ch)
// total == 15
\`\`\``,
            initialCode: `package pipeline

// Sum reads all integers from channel and returns their sum.
func Sum(in <-chan int) int {
	// TODO: Implement sum
	panic("not implemented")
}`,
            solutionCode: `package pipeline

func Sum(in <-chan int) int {
	sum := 0                                                     // accumulator variable
	for n := range in {                                          // read until channel closes
		sum += n                                             // accumulate each value
	}
	return sum                                                   // return final sum
}`,
            solutionExplanation: null,
            hint1: 'Simple loop accumulating values.',
            hint2: 'range over channel until closed.',
            whyItMatters: `This is a "sink" or "terminal" operation - it consumes the entire stream and produces a single result. Every pipeline needs a sink to actually execute the computation.

Sinks trigger the pipeline to run. Without a consumer like Sum, the generators and stages would block forever waiting for someone to read their output.

Common sinks include: aggregations (sum, count, average), collectors (to slice, to map), writers (to file, to database), and reporters (to metrics, to logs). They're the final stage that drives the entire pipeline.`,
            order: 1,
            translations: {
              ru: {
                title: 'Сумма значений пайплайна',
                description: `Создайте функцию, суммирующую все значения из канала.

Это терминальная операция - она потребляет весь канал и возвращает один результат.

**Требования:**
- Прочитайте все целые числа из канала
- Верните сумму когда канал закроется

**Пример:**
\`\`\`go
ch := Gen(1, 2, 3, 4, 5)
total := Sum(ch)
// total == 15
\`\`\``,
                hint1: 'Простой цикл, накапливающий значения.',
                hint2: 'range по каналу до закрытия.',
                solutionCode: `package pipeline

func Sum(in <-chan int) int {
	sum := 0                                                     // переменная-аккумулятор
	for n := range in {                                          // читаем пока канал не закроется
		sum += n                                             // накапливаем каждое значение
	}
	return sum                                                   // возвращаем итоговую сумму
}`,
                whyItMatters: `Это "sink" или терминальная операция - она потребляет весь поток и производит единственный результат. Каждому пайплайну нужен sink для фактического выполнения вычисления.

Sink'и запускают выполнение пайплайна. Без потребителя как Sum, генераторы и стадии блокировались бы навсегда в ожидании читателя.

Распространённые sink'и включают: агрегации (sum, count, average), коллекторы (в срез, в map), писатели (в файл, в БД) и репортеры (в метрики, в логи).`
              },
              uz: {
                title: 'Pipeline qiymatlarini yig\'ish',
                description: `Kanaldan barcha qiymatlarni yig'adigan funksiyani yarating.

Bu terminal operatsiya - u butun kanalni iste'mol qiladi va bitta natija qaytaradi.

**Talablar:**
- Kanaldan barcha butun sonlarni o'qing
- Kanal yopilganda yig'indini qaytaring

**Misol:**
\`\`\`go
ch := Gen(1, 2, 3, 4, 5)
total := Sum(ch)
// total == 15
\`\`\``,
                hint1: 'Qiymatlarni to\'playdigan oddiy sikl.',
                hint2: 'Kanal yopilguncha range qiling.',
                solutionCode: `package pipeline

func Sum(in <-chan int) int {
	sum := 0                                                     // akkumulyator o'zgaruvchi
	for n := range in {                                          // kanal yopilguncha o'qiymiz
		sum += n                                             // har bir qiymatni to'playmiz
	}
	return sum                                                   // yakuniy yig'indini qaytaramiz
}`,
                whyItMatters: `Bu "sink" yoki terminal operatsiya - u butun oqimni iste'mol qiladi va bitta natija ishlab chiqaradi. Har bir pipeline hisoblashni amalda bajarish uchun sink'ga muhtoj.

Sink'lar pipeline'ni ishga tushiradi. Sum kabi iste'molchisiz, generatorlar va bosqichlar chiqishini o'qiydigan kishini kutib abadiy blokirovkalanadi.

Keng tarqalgan sink'lar: agregatsiyalar (sum, count, average), kollektorlar (massivga, map'ga), yozuvchilar (faylga, bazaga) va hisobotchilar (metrikalarga, loglarga).`
              }
            }
          }
        ]
      },
      // Topic 4: Worker Pool
      {
        title: 'Worker Pool Pattern',
        description: 'Process tasks concurrently with a fixed number of workers.',
        difficulty: 'hard',
        estimatedTime: '1.5h',
        order: 4,
        translations: {
          ru: {
            title: 'Паттерн Worker Pool',
            description: 'Обрабатывайте задачи конкурентно с фиксированным количеством воркеров.'
          },
          uz: {
            title: 'Worker Pool pattern',
            description: 'Belgilangan miqdordagi worker\'lar bilan vazifalarni konkurrent qayta ishlang.'
          }
        },
        tasks: [
          {
            slug: 'go-workerpool-sequential',
            title: 'Sequential Processing Baseline',
            difficulty: 'easy',
            tags: ['go', 'concurrency', 'baseline'],
            estimatedTime: '10m',
            isPremium: false,
            description: `Process tasks sequentially as a baseline for comparison with concurrent versions.

Before optimizing with concurrency, establish a sequential baseline to measure improvements.

**Requirements:**
- Process each task in order using the handler function
- Return first error encountered (stop processing on error)
- Return nil if all tasks succeed

**Example:**
\`\`\`go
tasks := []int{1, 2, 3, 4, 5}
err := RunSequential(tasks, func(n int) error {
    fmt.Println(n)
    return nil
})
// Prints: 1, 2, 3, 4, 5 in order
\`\`\``,
            initialCode: `package workerpool

// RunSequential processes tasks one by one.
func RunSequential[T any](tasks []T, handler func(T) error) error {
	// TODO: Implement sequential processing
	panic("not implemented")
}`,
            solutionCode: `package workerpool

func RunSequential[T any](tasks []T, handler func(T) error) error {
	for _, task := range tasks {                                 // iterate through all tasks
		if err := handler(task); err != nil {                // process and check error
			return err                                   // fail fast on first error
		}
	}
	return nil                                                   // all tasks succeeded
}`,
            solutionExplanation: null,
            hint1: 'Simple for loop with error check.',
            hint2: 'Return immediately on first error.',
            whyItMatters: `Sequential processing is the baseline for comparison. Before parallelizing anything, you should measure how long it takes sequentially. Only then can you know if concurrency actually helps.

Some tasks are CPU-bound and won't benefit from goroutines on a single core. Others are I/O-bound and can see 10-100x speedups. Sequential baseline tells you which category you're in.

Use sequential when: tasks must execute in order, debugging concurrent code, establishing performance baselines, or when overhead of goroutines exceeds the benefit.`,
            order: 0,
            translations: {
              ru: {
                title: 'Последовательная обработка (базовая)',
                description: `Обрабатывайте задачи последовательно как базу для сравнения с конкурентными версиями.

Перед оптимизацией с конкурентностью установите последовательный базовый уровень.

**Требования:**
- Обработайте каждую задачу по порядку с помощью handler
- Верните первую встреченную ошибку
- Верните nil если все задачи успешны

**Пример:**
\`\`\`go
tasks := []int{1, 2, 3, 4, 5}
err := RunSequential(tasks, func(n int) error {
    fmt.Println(n)
    return nil
})
// Печатает: 1, 2, 3, 4, 5 по порядку
\`\`\``,
                hint1: 'Простой for цикл с проверкой ошибки.',
                hint2: 'Верните немедленно при первой ошибке.',
                solutionCode: `package workerpool

func RunSequential[T any](tasks []T, handler func(T) error) error {
	for _, task := range tasks {                                 // итерируем по всем задачам
		if err := handler(task); err != nil {                // обрабатываем и проверяем ошибку
			return err                                   // быстрый выход при первой ошибке
		}
	}
	return nil                                                   // все задачи успешно выполнены
}`,
                whyItMatters: `Последовательная обработка - это базовый уровень для сравнения. Перед параллелизацией чего-либо нужно измерить, сколько времени это занимает последовательно. Только тогда вы узнаете, помогает ли конкурентность.

Некоторые задачи CPU-bound и не выиграют от горутин на одном ядре. Другие I/O-bound и могут получить ускорение в 10-100 раз. Последовательный базовый уровень показывает, в какой категории вы находитесь.

Используйте последовательную обработку когда: задачи должны выполняться по порядку, при отладке конкурентного кода, установлении базовых показателей производительности.`
              },
              uz: {
                title: 'Ketma-ket qayta ishlash (asosiy)',
                description: `Konkurrent versiyalar bilan taqqoslash uchun vazifalarni ketma-ket qayta ishlang.

Konkurrentlik bilan optimizatsiyadan oldin ketma-ket asosiy darajani o'rnating.

**Talablar:**
- Har bir vazifani handler yordamida tartib bo'yicha qayta ishlang
- Birinchi uchragan xatoni qaytaring
- Barcha vazifalar muvaffaqiyatli bo'lsa nil qaytaring

**Misol:**
\`\`\`go
tasks := []int{1, 2, 3, 4, 5}
err := RunSequential(tasks, func(n int) error {
    fmt.Println(n)
    return nil
})
// Chiqaradi: 1, 2, 3, 4, 5 tartib bo'yicha
\`\`\``,
                hint1: 'Xato tekshiruvi bilan oddiy for sikli.',
                hint2: 'Birinchi xatoda darhol qaytaring.',
                solutionCode: `package workerpool

func RunSequential[T any](tasks []T, handler func(T) error) error {
	for _, task := range tasks {                                 // barcha vazifalar bo'ylab iteratsiya
		if err := handler(task); err != nil {                // qayta ishlaymiz va xatoni tekshiramiz
			return err                                   // birinchi xatoda tez chiqish
		}
	}
	return nil                                                   // barcha vazifalar muvaffaqiyatli
}`,
                whyItMatters: `Ketma-ket qayta ishlash taqqoslash uchun asosiy daraja. Biror narsani parallellashtirishdan oldin, ketma-ket qancha vaqt ketishini o'lchashingiz kerak. Faqat shundagina konkurrentlik yordam beradimi yoki yo'qligini bilasiz.

Ba'zi vazifalar CPU-bound va bitta yadrodagi goroutine'lardan foyda olmaydi. Boshqalari I/O-bound va 10-100 baravar tezlashishi mumkin. Ketma-ket asosiy daraja qaysi kategoriyada ekanligingizni ko'rsatadi.

Ketma-ket qayta ishlashni qo'llang: vazifalar tartib bo'yicha bajarilishi kerak bo'lganda, konkurrent kodni disk qilganda, ishlash asosiy ko'rsatkichlarini o'rnatganda.`
              }
            }
          },
          {
            slug: 'go-workerpool-parallel',
            title: 'Unlimited Parallel Processing',
            difficulty: 'medium',
            tags: ['go', 'concurrency', 'goroutines', 'waitgroup'],
            estimatedTime: '20m',
            isPremium: false,
            description: `Process all tasks in parallel using goroutines.

This is the opposite extreme - maximum parallelism. Good for I/O-bound tasks but may overwhelm resources.

**Requirements:**
- Launch one goroutine per task
- Wait for all to complete using sync.WaitGroup
- Collect all errors, return the first one encountered

**Example:**
\`\`\`go
tasks := []int{1, 2, 3, 4, 5}
err := RunParallel(tasks, func(n int) error {
    time.Sleep(time.Duration(n) * 10 * time.Millisecond)
    return nil
})
// All tasks run concurrently
\`\`\``,
            initialCode: `package workerpool

import "sync"

// RunParallel processes all tasks concurrently.
func RunParallel[T any](tasks []T, handler func(T) error) error {
	// TODO: Implement parallel processing
	panic("not implemented")
}`,
            solutionCode: `package workerpool

import "sync"

func RunParallel[T any](tasks []T, handler func(T) error) error {
	var wg sync.WaitGroup                                        // track goroutine completion
	var mu sync.Mutex                                            // protect shared firstErr
	var firstErr error                                           // store first error only

	for _, task := range tasks {                                 // iterate all tasks
		wg.Add(1)                                            // increment before goroutine
		go func(t T) {                                       // capture task as parameter
			defer wg.Done()                              // signal completion on exit
			if err := handler(t); err != nil {           // process task
				mu.Lock()                            // lock before write
				if firstErr == nil {                 // only keep first error
					firstErr = err               // store the error
				}
				mu.Unlock()                          // release lock
			}
		}(task)                                              // pass task to avoid closure bug
	}

	wg.Wait()                                                    // wait for all goroutines
	return firstErr                                              // return first error or nil
}`,
            solutionExplanation: null,
            hint1: 'Use sync.WaitGroup to wait for all goroutines.',
            hint2: 'Protect firstErr with mutex, capture only first.',
            whyItMatters: `Unlimited parallelism is the opposite extreme from sequential. It's perfect for I/O-bound tasks like HTTP requests where you're mostly waiting for responses.

The closure capture bug (passing task as parameter) is a classic Go gotcha. Without it, all goroutines would share the same loop variable and process the last task multiple times.

Use unlimited parallel when: tasks are I/O-bound, you have external rate limits (API quotas), or the task count is naturally limited. Avoid for CPU-bound work or when task count could be unbounded.`,
            order: 1,
            translations: {
              ru: {
                title: 'Неограниченная параллельная обработка',
                description: `Обрабатывайте все задачи параллельно используя горутины.

Это противоположная крайность - максимальный параллелизм.

**Требования:**
- Запустите одну горутину на задачу
- Дождитесь завершения всех с помощью sync.WaitGroup
- Соберите все ошибки, верните первую

**Пример:**
\`\`\`go
tasks := []int{1, 2, 3, 4, 5}
err := RunParallel(tasks, func(n int) error {
    time.Sleep(time.Duration(n) * 10 * time.Millisecond)
    return nil
})
// Все задачи выполняются конкурентно
\`\`\``,
                hint1: 'Используйте sync.WaitGroup для ожидания всех горутин.',
                hint2: 'Защитите firstErr мьютексом, сохраните только первую.',
                solutionCode: `package workerpool

import "sync"

func RunParallel[T any](tasks []T, handler func(T) error) error {
	var wg sync.WaitGroup                                        // отслеживаем завершение горутин
	var mu sync.Mutex                                            // защищаем общий firstErr
	var firstErr error                                           // сохраняем только первую ошибку

	for _, task := range tasks {                                 // итерируем все задачи
		wg.Add(1)                                            // увеличиваем перед горутиной
		go func(t T) {                                       // захватываем задачу как параметр
			defer wg.Done()                              // сигнализируем завершение при выходе
			if err := handler(t); err != nil {           // обрабатываем задачу
				mu.Lock()                            // блокируем перед записью
				if firstErr == nil {                 // сохраняем только первую ошибку
					firstErr = err               // запоминаем ошибку
				}
				mu.Unlock()                          // освобождаем блокировку
			}
		}(task)                                              // передаём task чтобы избежать бага замыкания
	}

	wg.Wait()                                                    // ждём все горутины
	return firstErr                                              // возвращаем первую ошибку или nil
}`,
                whyItMatters: `Неограниченный параллелизм - это противоположная крайность от последовательной обработки. Он идеален для I/O-bound задач как HTTP-запросы, где вы в основном ждёте ответов.

Баг захвата замыкания (передача task как параметра) - классическая ловушка Go. Без этого все горутины разделяли бы одну переменную цикла и обрабатывали бы последнюю задачу несколько раз.

Используйте неограниченный параллелизм когда: задачи I/O-bound, есть внешние лимиты (квоты API), или количество задач естественно ограничено.`
              },
              uz: {
                title: 'Cheksiz parallel qayta ishlash',
                description: `Barcha vazifalarni goroutine'lar yordamida parallel qayta ishlang.

Bu ketma-ketlikdan qarama-qarshi ekstrem - maksimal parallellik.

**Talablar:**
- Har bir vazifa uchun bitta goroutine ishga tushiring
- sync.WaitGroup yordamida barchasining tugashini kuting
- Barcha xatolarni to'plang, birinchisini qaytaring

**Misol:**
\`\`\`go
tasks := []int{1, 2, 3, 4, 5}
err := RunParallel(tasks, func(n int) error {
    time.Sleep(time.Duration(n) * 10 * time.Millisecond)
    return nil
})
// Barcha vazifalar konkurrent ishlaydi
\`\`\``,
                hint1: 'Barcha goroutine\'larni kutish uchun sync.WaitGroup\'dan foydalaning.',
                hint2: 'firstErr\'ni mutex bilan himoyalang, faqat birinchisini saqlang.',
                solutionCode: `package workerpool

import "sync"

func RunParallel[T any](tasks []T, handler func(T) error) error {
	var wg sync.WaitGroup                                        // goroutine tugallashini kuzatamiz
	var mu sync.Mutex                                            // umumiy firstErr'ni himoyalaymiz
	var firstErr error                                           // faqat birinchi xatoni saqlaymiz

	for _, task := range tasks {                                 // barcha vazifalar bo'ylab iteratsiya
		wg.Add(1)                                            // goroutine'dan oldin oshiramiz
		go func(t T) {                                       // vazifani parametr sifatida olamiz
			defer wg.Done()                              // chiqishda tugallanganini signal qilamiz
			if err := handler(t); err != nil {           // vazifani qayta ishlaymiz
				mu.Lock()                            // yozishdan oldin qulflaymiz
				if firstErr == nil {                 // faqat birinchi xatoni saqlaymiz
					firstErr = err               // xatoni eslab qolamiz
				}
				mu.Unlock()                          // qulfni bo'shatamiz
			}
		}(task)                                              // closure bug'dan qochish uchun task'ni uzatamiz
	}

	wg.Wait()                                                    // barcha goroutine'larni kutamiz
	return firstErr                                              // birinchi xatoni yoki nil qaytaramiz
}`,
                whyItMatters: `Cheksiz parallellik ketma-ket qayta ishlashdan qarama-qarshi ekstrem. Bu HTTP so'rovlar kabi asosan javoblarni kutadigan I/O-bound vazifalar uchun ideal.

Closure capture bug (task'ni parametr sifatida o'tkazish) - klassik Go tuzoq. Busiz barcha goroutine'lar bitta sikl o'zgaruvchisini bo'lishadi va oxirgi vazifani bir necha marta qayta ishlaydi.

Cheksiz parallellikni qo'llang: vazifalar I/O-bound bo'lganda, tashqi limitlar mavjud bo'lganda (API kvotalari), yoki vazifalar soni tabiiy ravishda cheklangan bo'lganda.`
              }
            }
          },
          {
            slug: 'go-workerpool-limited',
            title: 'Rate-Limited Parallel Processing',
            difficulty: 'medium',
            tags: ['go', 'concurrency', 'semaphore', 'ratelimit'],
            estimatedTime: '25m',
            isPremium: true,
            description: `Process tasks with limited concurrency using a semaphore pattern.

This balances sequential and parallel - limit concurrent operations to avoid overwhelming resources.

**Requirements:**
- Allow at most \`limit\` concurrent tasks
- Use a buffered channel as semaphore
- Return first error encountered

**Example:**
\`\`\`go
tasks := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
err := RunWithLimit(tasks, 3, func(n int) error {
    // At most 3 tasks run at any time
    return nil
})
\`\`\``,
            initialCode: `package workerpool

import "sync"

// RunWithLimit processes tasks with limited concurrency.
func RunWithLimit[T any](tasks []T, limit int, handler func(T) error) error {
	// TODO: Implement rate-limited processing
	panic("not implemented")
}`,
            solutionCode: `package workerpool

import "sync"

func RunWithLimit[T any](tasks []T, limit int, handler func(T) error) error {
	if limit <= 0 {                                              // validate limit
		limit = 1                                            // ensure at least 1 concurrent
	}

	sem := make(chan struct{}, limit)                            // buffered channel = semaphore
	var wg sync.WaitGroup                                        // track goroutine completion
	var mu sync.Mutex                                            // protect shared error
	var firstErr error                                           // store first error

	for _, task := range tasks {                                 // iterate all tasks
		sem <- struct{}{}                                    // acquire slot (blocks if full)
		wg.Add(1)                                            // track new goroutine
		go func(t T) {                                       // launch worker
			defer func() {                               // cleanup on exit
				<-sem                                // release semaphore slot
				wg.Done()                            // signal completion
			}()
			if err := handler(t); err != nil {           // process task
				mu.Lock()                            // lock for write
				if firstErr == nil {                 // capture first error only
					firstErr = err               // store error
				}
				mu.Unlock()                          // release lock
			}
		}(task)                                              // pass task as parameter
	}

	wg.Wait()                                                    // wait for all workers
	return firstErr                                              // return first error or nil
}`,
            solutionExplanation: null,
            hint1: 'Buffered channel as counting semaphore.',
            hint2: 'Send to acquire, receive to release slot.',
            whyItMatters: `Rate limiting is the most practical concurrency pattern. It balances throughput with resource protection - you process multiple tasks concurrently but avoid overwhelming databases, APIs, or memory.

The semaphore pattern (buffered channel) is elegant: channel capacity equals concurrent limit. Send blocks when full, receive releases a slot. No mutexes needed for the limiting logic itself.

Use rate-limited parallel when: calling external APIs with quotas, writing to databases with connection limits, processing files with memory constraints, or any scenario where "too much parallelism" causes problems.`,
            order: 2,
            translations: {
              ru: {
                title: 'Параллельная обработка с ограничением',
                description: `Обрабатывайте задачи с ограниченной конкурентностью используя паттерн семафора.

Это баланс между последовательным и параллельным.

**Требования:**
- Разрешите не более \`limit\` конкурентных задач
- Используйте буферизованный канал как семафор
- Верните первую ошибку

**Пример:**
\`\`\`go
tasks := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
err := RunWithLimit(tasks, 3, func(n int) error {
    // Максимум 3 задачи одновременно
    return nil
})
\`\`\``,
                hint1: 'Буферизованный канал как счетчик семафора.',
                hint2: 'Отправка для захвата, получение для освобождения слота.',
                solutionCode: `package workerpool

import "sync"

func RunWithLimit[T any](tasks []T, limit int, handler func(T) error) error {
	if limit <= 0 {                                              // валидируем лимит
		limit = 1                                            // минимум 1 конкурентно
	}

	sem := make(chan struct{}, limit)                            // буферизованный канал = семафор
	var wg sync.WaitGroup                                        // отслеживаем завершение горутин
	var mu sync.Mutex                                            // защищаем общую ошибку
	var firstErr error                                           // сохраняем первую ошибку

	for _, task := range tasks {                                 // итерируем все задачи
		sem <- struct{}{}                                    // захватываем слот (блокируется если полон)
		wg.Add(1)                                            // отслеживаем новую горутину
		go func(t T) {                                       // запускаем воркер
			defer func() {                               // очистка при выходе
				<-sem                                // освобождаем слот семафора
				wg.Done()                            // сигнализируем завершение
			}()
			if err := handler(t); err != nil {           // обрабатываем задачу
				mu.Lock()                            // блокируем для записи
				if firstErr == nil {                 // сохраняем только первую ошибку
					firstErr = err               // запоминаем ошибку
				}
				mu.Unlock()                          // освобождаем блокировку
			}
		}(task)                                              // передаём задачу как параметр
	}

	wg.Wait()                                                    // ждём всех воркеров
	return firstErr                                              // возвращаем первую ошибку или nil
}`,
                whyItMatters: `Ограничение скорости - самый практичный паттерн конкурентности. Он балансирует пропускную способность с защитой ресурсов - вы обрабатываете несколько задач конкурентно, но избегаете перегрузки БД, API или памяти.

Паттерн семафора (буферизованный канал) элегантен: ёмкость канала равна лимиту конкурентности. Отправка блокируется когда полон, получение освобождает слот. Мьютексы для логики ограничения не нужны.

Используйте когда: вызываете внешние API с квотами, пишете в БД с лимитами соединений, обрабатываете файлы с ограничениями памяти.`
              },
              uz: {
                title: 'Cheklangan parallel qayta ishlash',
                description: `Semafor pattern yordamida cheklangan konkurrentlik bilan vazifalarni qayta ishlang.

Bu ketma-ket va parallel o'rtasidagi muvozanat.

**Talablar:**
- Ko'pi bilan \`limit\` konkurrent vazifaga ruxsat bering
- Buferli kanalni semafor sifatida ishlating
- Birinchi xatoni qaytaring

**Misol:**
\`\`\`go
tasks := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
err := RunWithLimit(tasks, 3, func(n int) error {
    // Bir vaqtning o'zida ko'pi bilan 3 ta vazifa
    return nil
})
\`\`\``,
                hint1: 'Buferli kanal sanoq semafori sifatida.',
                hint2: 'Olish uchun yuboring, slotni bo\'shatish uchun oling.',
                solutionCode: `package workerpool

import "sync"

func RunWithLimit[T any](tasks []T, limit int, handler func(T) error) error {
	if limit <= 0 {                                              // limitni tekshiramiz
		limit = 1                                            // kamida 1 ta konkurrent
	}

	sem := make(chan struct{}, limit)                            // buferli kanal = semafor
	var wg sync.WaitGroup                                        // goroutine tugallashini kuzatamiz
	var mu sync.Mutex                                            // umumiy xatoni himoyalaymiz
	var firstErr error                                           // birinchi xatoni saqlaymiz

	for _, task := range tasks {                                 // barcha vazifalar bo'ylab iteratsiya
		sem <- struct{}{}                                    // slot olamiz (to'lganda bloklanadi)
		wg.Add(1)                                            // yangi goroutine'ni kuzatamiz
		go func(t T) {                                       // worker ishga tushiramiz
			defer func() {                               // chiqishda tozalash
				<-sem                                // semafor slotini bo'shatamiz
				wg.Done()                            // tugallanganini signal qilamiz
			}()
			if err := handler(t); err != nil {           // vazifani qayta ishlaymiz
				mu.Lock()                            // yozish uchun qulflaymiz
				if firstErr == nil {                 // faqat birinchi xatoni saqlaymiz
					firstErr = err               // xatoni eslab qolamiz
				}
				mu.Unlock()                          // qulfni bo'shatamiz
			}
		}(task)                                              // vazifani parametr sifatida uzatamiz
	}

	wg.Wait()                                                    // barcha worker'larni kutamiz
	return firstErr                                              // birinchi xatoni yoki nil qaytaramiz
}`,
                whyItMatters: `Tezlikni cheklash eng amaliy konkurrentlik patterni. U o'tkazuvchanlikni resurs himoyasi bilan muvozanatlaydi - siz bir nechta vazifalarni konkurrent qayta ishlaysiz, lekin bazalar, API'lar yoki xotirani ortiqcha yuklamaysiz.

Semafor patterni (buferli kanal) nafis: kanal sig'imi konkurrentlik limitiga teng. To'lganida yuborish bloklanadi, olish slotni bo'shatadi. Cheklash mantiqining o'zi uchun mutex kerak emas.

Qo'llang: kvotali tashqi API'larni chaqirganda, ulanish limitli bazalarga yozganda, xotira cheklovlari bilan fayllarni qayta ishlaganda.`
              }
            }
          },
          {
            slug: 'go-workerpool-pool',
            title: 'Worker Pool Implementation',
            difficulty: 'hard',
            tags: ['go', 'concurrency', 'worker-pool', 'channels'],
            estimatedTime: '30m',
            isPremium: true,
            description: `Implement a proper worker pool with fixed workers processing from a job channel.

This is the classic worker pool pattern - spawn fixed workers that consume from a shared job channel.

**Requirements:**
- Spawn exactly N worker goroutines
- Workers read from shared jobs channel
- Stop on first error (cancel context)
- Clean shutdown - all workers must exit

**Example:**
\`\`\`go
jobs := make(chan int)
go func() {
    for i := 0; i < 100; i++ { jobs <- i }
    close(jobs)
}()
err := RunWorkerPool(ctx, jobs, 5, func(ctx context.Context, n int) error {
    return processJob(n)
})
\`\`\``,
            initialCode: `package workerpool

import (
	"context"
	"sync"
)

// RunWorkerPool processes jobs using N workers.
func RunWorkerPool[T any](ctx context.Context, jobs <-chan T, workers int, handler func(context.Context, T) error) error {
	// TODO: Implement worker pool
	panic("not implemented")
}`,
            solutionCode: `package workerpool

import (
	"context"
	"sync"
)

func RunWorkerPool[T any](ctx context.Context, jobs <-chan T, workers int, handler func(context.Context, T) error) error {
	if workers <= 0 || jobs == nil || handler == nil {           // validate inputs
		return nil                                           // nothing to do
	}

	ctx, cancel := context.WithCancel(ctx)                       // create cancellable context
	defer cancel()                                               // cleanup on return

	var wg sync.WaitGroup                                        // track worker completion
	var once sync.Once                                           // ensure single error capture
	var firstErr error                                           // store first error

	worker := func() {                                           // worker function
		defer wg.Done()                                      // signal completion on exit
		for {                                                // loop until stopped
			select {                                     // wait for job or cancel
			case <-ctx.Done():                           // context cancelled
				return                               // exit worker
			case job, ok := <-jobs:                      // receive job
				if !ok {                             // channel closed
					return                       // no more jobs, exit
				}
				if err := handler(ctx, job); err != nil { // process job
					once.Do(func() {             // only first error
						firstErr = err       // store error
						cancel()             // stop all workers
					})
				}
			}
		}
	}

	wg.Add(workers)                                              // register all workers
	for i := 0; i < workers; i++ {                               // spawn exactly N workers
		go worker()                                          // launch worker goroutine
	}

	wg.Wait()                                                    // wait for all to finish
	return firstErr                                              // return first error or nil
}`,
            solutionExplanation: null,
            hint1: 'sync.Once ensures single error capture.',
            hint2: 'cancel() stops all workers on first error.',
            whyItMatters: `Worker pools are the standard pattern for processing streams of work. Unlike launching a goroutine per task, pools maintain a fixed number of workers that continuously pull from a job channel.

This pattern provides predictable resource usage and backpressure: if workers are busy, the job channel naturally buffers (or blocks the sender). It's how production systems process queues, handle HTTP requests, and manage background jobs.

Key insight: the select statement allows workers to respond to both jobs and cancellation. Without it, workers would block forever on a closed context. sync.Once ensures clean shutdown even when multiple workers hit errors simultaneously.`,
            order: 3,
            translations: {
              ru: {
                title: 'Реализация Worker Pool',
                description: `Реализуйте настоящий worker pool с фиксированными воркерами, обрабатывающими канал задач.

Это классический паттерн worker pool.

**Требования:**
- Запустите ровно N воркер-горутин
- Воркеры читают из общего канала задач
- Остановитесь при первой ошибке (отмените контекст)
- Чистое завершение - все воркеры должны выйти

**Пример:**
\`\`\`go
jobs := make(chan int)
go func() {
    for i := 0; i < 100; i++ { jobs <- i }
    close(jobs)
}()
err := RunWorkerPool(ctx, jobs, 5, func(ctx context.Context, n int) error {
    return processJob(n)
})
\`\`\``,
                hint1: 'sync.Once гарантирует однократный захват ошибки.',
                hint2: 'cancel() останавливает всех воркеров при первой ошибке.',
                solutionCode: `package workerpool

import (
	"context"
	"sync"
)

func RunWorkerPool[T any](ctx context.Context, jobs <-chan T, workers int, handler func(context.Context, T) error) error {
	if workers <= 0 || jobs == nil || handler == nil {           // валидируем входные данные
		return nil                                           // нечего делать
	}

	ctx, cancel := context.WithCancel(ctx)                       // создаём отменяемый контекст
	defer cancel()                                               // очистка при возврате

	var wg sync.WaitGroup                                        // отслеживаем завершение воркеров
	var once sync.Once                                           // гарантируем однократный захват ошибки
	var firstErr error                                           // сохраняем первую ошибку

	worker := func() {                                           // функция воркера
		defer wg.Done()                                      // сигнализируем завершение при выходе
		for {                                                // цикл до остановки
			select {                                     // ждём задачу или отмену
			case <-ctx.Done():                           // контекст отменён
				return                               // выходим из воркера
			case job, ok := <-jobs:                      // получаем задачу
				if !ok {                             // канал закрыт
					return                       // задач больше нет, выходим
				}
				if err := handler(ctx, job); err != nil { // обрабатываем задачу
					once.Do(func() {             // только первая ошибка
						firstErr = err       // сохраняем ошибку
						cancel()             // останавливаем всех воркеров
					})
				}
			}
		}
	}

	wg.Add(workers)                                              // регистрируем всех воркеров
	for i := 0; i < workers; i++ {                               // запускаем ровно N воркеров
		go worker()                                          // запускаем горутину воркера
	}

	wg.Wait()                                                    // ждём завершения всех
	return firstErr                                              // возвращаем первую ошибку или nil
}`,
                whyItMatters: `Worker pool'ы - стандартный паттерн для обработки потоков работы. В отличие от запуска горутины на задачу, пулы поддерживают фиксированное количество воркеров, непрерывно забирающих из канала задач.

Этот паттерн обеспечивает предсказуемое использование ресурсов и backpressure: если воркеры заняты, канал задач естественно буферизует (или блокирует отправителя). Так продакшн-системы обрабатывают очереди, HTTP-запросы и фоновые задачи.

Ключевой момент: select позволяет воркерам реагировать и на задачи, и на отмену. sync.Once гарантирует чистое завершение даже когда несколько воркеров сталкиваются с ошибками одновременно.`
              },
              uz: {
                title: 'Worker Pool implementatsiyasi',
                description: `Belgilangan worker'lar job kanalidan o'qiydigan haqiqiy worker pool'ni amalga oshiring.

Bu klassik worker pool pattern.

**Talablar:**
- Aynan N worker-goroutine ishga tushiring
- Worker'lar umumiy job kanalidan o'qiydi
- Birinchi xatoda to'xtang (contextni bekor qiling)
- Toza yakunlash - barcha worker'lar chiqishi kerak

**Misol:**
\`\`\`go
jobs := make(chan int)
go func() {
    for i := 0; i < 100; i++ { jobs <- i }
    close(jobs)
}()
err := RunWorkerPool(ctx, jobs, 5, func(ctx context.Context, n int) error {
    return processJob(n)
})
\`\`\``,
                hint1: 'sync.Once bir marta xato olishni ta\'minlaydi.',
                hint2: 'cancel() birinchi xatoda barcha worker\'larni to\'xtatadi.',
                solutionCode: `package workerpool

import (
	"context"
	"sync"
)

func RunWorkerPool[T any](ctx context.Context, jobs <-chan T, workers int, handler func(context.Context, T) error) error {
	if workers <= 0 || jobs == nil || handler == nil {           // kirish ma'lumotlarini tekshiramiz
		return nil                                           // qiladigan ish yo'q
	}

	ctx, cancel := context.WithCancel(ctx)                       // bekor qilinadigan context yaratamiz
	defer cancel()                                               // qaytishda tozalash

	var wg sync.WaitGroup                                        // worker tugallashini kuzatamiz
	var once sync.Once                                           // bir marta xato olishni ta'minlaymiz
	var firstErr error                                           // birinchi xatoni saqlaymiz

	worker := func() {                                           // worker funksiyasi
		defer wg.Done()                                      // chiqishda tugallanganini signal qilamiz
		for {                                                // to'xtatilguncha sikl
			select {                                     // job yoki bekor qilishni kutamiz
			case <-ctx.Done():                           // context bekor qilindi
				return                               // worker'dan chiqamiz
			case job, ok := <-jobs:                      // job olamiz
				if !ok {                             // kanal yopildi
					return                       // job yo'q, chiqamiz
				}
				if err := handler(ctx, job); err != nil { // job'ni qayta ishlaymiz
					once.Do(func() {             // faqat birinchi xato
						firstErr = err       // xatoni saqlaymiz
						cancel()             // barcha worker'larni to'xtatamiz
					})
				}
			}
		}
	}

	wg.Add(workers)                                              // barcha worker'larni ro'yxatga olamiz
	for i := 0; i < workers; i++ {                               // aynan N worker ishga tushiramiz
		go worker()                                          // worker goroutine ishga tushiramiz
	}

	wg.Wait()                                                    // barchasining tugashini kutamiz
	return firstErr                                              // birinchi xatoni yoki nil qaytaramiz
}`,
                whyItMatters: `Worker pool'lar ish oqimlarini qayta ishlashning standart patterni. Vazifa uchun goroutine ishga tushirishdan farqli, pool'lar job kanalidan doimiy ravishda oladigan belgilangan sonli worker'larni saqlaydi.

Bu pattern bashoratli resurs ishlatish va backpressure'ni ta'minlaydi: worker'lar band bo'lsa, job kanali tabiiy ravishda buferlanadi (yoki yuboruvchini bloklaydi). Production tizimlar navbatlarni, HTTP so'rovlarni va fon vazifalarni shunday qayta ishlaydi.

Asosiy tushuncha: select worker'larga ham job'larga, ham bekor qilishga javob berish imkonini beradi. sync.Once bir nechta worker'lar bir vaqtda xatolarga duch kelganda ham toza yakunlashni ta'minlaydi.`
              }
            }
          },
          {
            slug: 'go-workerpool-safe',
            title: 'Worker Pool with Panic Recovery',
            difficulty: 'hard',
            tags: ['go', 'concurrency', 'worker-pool', 'panic', 'recovery'],
            estimatedTime: '25m',
            isPremium: true,
            description: `Enhance the worker pool to recover from panics in handlers.

Production code must handle panics gracefully - one bad job shouldn't crash the entire pool.

**Requirements:**
- Recover from panics in handler functions
- Convert panic to error
- Continue processing other jobs
- Return the panic as an error

**Example:**
\`\`\`go
err := RunWorkerPoolSafe(ctx, jobs, 5, func(ctx context.Context, n int) error {
    if n == 42 {
        panic("bad number!")
    }
    return nil
})
// err contains panic message, other jobs processed
\`\`\``,
            initialCode: `package workerpool

import (
	"context"
	"fmt"
	"sync"
)

// RunWorkerPoolSafe processes jobs with panic recovery.
func RunWorkerPoolSafe[T any](ctx context.Context, jobs <-chan T, workers int, handler func(context.Context, T) error) error {
	// TODO: Implement worker pool with panic recovery
	panic("not implemented")
}`,
            solutionCode: `package workerpool

import (
	"context"
	"fmt"
	"sync"
)

func RunWorkerPoolSafe[T any](ctx context.Context, jobs <-chan T, workers int, handler func(context.Context, T) error) error {
	if workers <= 0 || jobs == nil || handler == nil {           // validate inputs
		return nil                                           // nothing to do
	}

	ctx, cancel := context.WithCancel(ctx)                       // cancellable context
	defer cancel()                                               // cleanup on return

	var wg sync.WaitGroup                                        // track worker completion
	var once sync.Once                                           // single error capture
	var firstErr error                                           // store first error

	recordErr := func(err error) {                               // helper to record errors
		if err != nil {                                      // only record non-nil
			once.Do(func() {                             // thread-safe, once only
				firstErr = err                       // store error
				cancel()                             // stop all workers
			})
		}
	}

	worker := func() {                                           // worker function
		defer wg.Done()                                      // signal completion
		for {                                                // loop until stopped
			select {                                     // wait for job or cancel
			case <-ctx.Done():                           // cancelled
				return                               // exit worker
			case job, ok := <-jobs:                      // receive job
				if !ok {                             // channel closed
					return                       // exit worker
				}
				func() {                             // wrapper for recovery
					defer func() {               // panic recovery
						if r := recover(); r != nil { // caught panic
							recordErr(fmt.Errorf("panic: %v", r)) // convert to error
						}
					}()
					if err := handler(ctx, job); err != nil { // process job
						recordErr(err)           // record handler error
					}
				}()                                  // execute immediately
			}
		}
	}

	wg.Add(workers)                                              // register workers
	for i := 0; i < workers; i++ {                               // spawn N workers
		go worker()                                          // launch goroutine
	}

	wg.Wait()                                                    // wait for completion
	return firstErr                                              // return first error
}`,
            solutionExplanation: null,
            hint1: 'Wrap handler call in defer/recover block.',
            hint2: 'Convert panic value to error with fmt.Errorf.',
            whyItMatters: `Production systems must be resilient. One bad input or edge case shouldn't crash your entire service. Panic recovery converts crashes into errors that can be logged, reported, and handled gracefully.

The key technique is wrapping the handler call in an anonymous function with defer/recover. This isolates the panic to that function's scope while letting the worker continue processing other jobs.

This pattern is used in HTTP servers (don't crash on one bad request), message queues (don't lose the whole batch), and any system where availability matters more than failing fast. Always log recovered panics - they indicate bugs that need fixing.`,
            order: 4,
            translations: {
              ru: {
                title: 'Worker Pool с восстановлением от паники',
                description: `Улучшите worker pool для восстановления от паник в обработчиках.

Production код должен обрабатывать паники gracefully.

**Требования:**
- Восстанавливайтесь от паник в функциях-обработчиках
- Конвертируйте панику в ошибку
- Продолжайте обработку других задач
- Верните панику как ошибку

**Пример:**
\`\`\`go
err := RunWorkerPoolSafe(ctx, jobs, 5, func(ctx context.Context, n int) error {
    if n == 42 {
        panic("bad number!")
    }
    return nil
})
// err содержит сообщение паники, остальные задачи обработаны
\`\`\``,
                hint1: 'Оберните вызов handler в defer/recover блок.',
                hint2: 'Конвертируйте значение паники в ошибку с fmt.Errorf.',
                solutionCode: `package workerpool

import (
	"context"
	"fmt"
	"sync"
)

func RunWorkerPoolSafe[T any](ctx context.Context, jobs <-chan T, workers int, handler func(context.Context, T) error) error {
	if workers <= 0 || jobs == nil || handler == nil {           // валидируем входные данные
		return nil                                           // нечего делать
	}

	ctx, cancel := context.WithCancel(ctx)                       // отменяемый контекст
	defer cancel()                                               // очистка при возврате

	var wg sync.WaitGroup                                        // отслеживаем завершение воркеров
	var once sync.Once                                           // однократный захват ошибки
	var firstErr error                                           // сохраняем первую ошибку

	recordErr := func(err error) {                               // хелпер для записи ошибок
		if err != nil {                                      // записываем только не-nil
			once.Do(func() {                             // потокобезопасно, только раз
				firstErr = err                       // сохраняем ошибку
				cancel()                             // останавливаем всех воркеров
			})
		}
	}

	worker := func() {                                           // функция воркера
		defer wg.Done()                                      // сигнализируем завершение
		for {                                                // цикл до остановки
			select {                                     // ждём задачу или отмену
			case <-ctx.Done():                           // отменено
				return                               // выходим из воркера
			case job, ok := <-jobs:                      // получаем задачу
				if !ok {                             // канал закрыт
					return                       // выходим из воркера
				}
				func() {                             // обёртка для recovery
					defer func() {               // восстановление от паники
						if r := recover(); r != nil { // поймали панику
							recordErr(fmt.Errorf("panic: %v", r)) // конвертируем в ошибку
						}
					}()
					if err := handler(ctx, job); err != nil { // обрабатываем задачу
						recordErr(err)           // записываем ошибку handler'а
					}
				}()                                  // выполняем немедленно
			}
		}
	}

	wg.Add(workers)                                              // регистрируем воркеров
	for i := 0; i < workers; i++ {                               // запускаем N воркеров
		go worker()                                          // запускаем горутину
	}

	wg.Wait()                                                    // ждём завершения
	return firstErr                                              // возвращаем первую ошибку
}`,
                whyItMatters: `Продакшн-системы должны быть устойчивыми. Один плохой ввод или крайний случай не должен ронять весь сервис. Восстановление от паники превращает краши в ошибки, которые можно логировать и обрабатывать gracefully.

Ключевая техника - обёртывание вызова handler в анонимную функцию с defer/recover. Это изолирует панику в область видимости этой функции, позволяя воркеру продолжить обработку других задач.

Этот паттерн используется в HTTP-серверах (не падать на одном плохом запросе), очередях сообщений (не терять весь batch), и любой системе где доступность важнее fail-fast. Всегда логируйте пойманные паники - они указывают на баги.`
              },
              uz: {
                title: 'Panic recovery bilan Worker Pool',
                description: `Handler'lardagi panic'lardan tiklanadigan worker pool'ni yaxshilang.

Production kod panic'larni gracefully qayta ishlashi kerak.

**Talablar:**
- Handler funksiyalardagi panic'lardan tikaning
- Panic'ni xatoga aylantiring
- Boshqa vazifalarni qayta ishlashni davom ettiring
- Panic'ni xato sifatida qaytaring

**Misol:**
\`\`\`go
err := RunWorkerPoolSafe(ctx, jobs, 5, func(ctx context.Context, n int) error {
    if n == 42 {
        panic("bad number!")
    }
    return nil
})
// err panic xabarini o'z ichiga oladi, boshqa vazifalar qayta ishlangan
\`\`\``,
                hint1: 'Handler chaqiruvini defer/recover blokiga o\'rang.',
                hint2: 'Panic qiymatini fmt.Errorf bilan xatoga aylantiring.',
                solutionCode: `package workerpool

import (
	"context"
	"fmt"
	"sync"
)

func RunWorkerPoolSafe[T any](ctx context.Context, jobs <-chan T, workers int, handler func(context.Context, T) error) error {
	if workers <= 0 || jobs == nil || handler == nil {           // kirish ma'lumotlarini tekshiramiz
		return nil                                           // qiladigan ish yo'q
	}

	ctx, cancel := context.WithCancel(ctx)                       // bekor qilinadigan context
	defer cancel()                                               // qaytishda tozalash

	var wg sync.WaitGroup                                        // worker tugallashini kuzatamiz
	var once sync.Once                                           // bir marta xato olish
	var firstErr error                                           // birinchi xatoni saqlaymiz

	recordErr := func(err error) {                               // xatolarni yozish uchun helper
		if err != nil {                                      // faqat nil bo'lmaganlarni yozamiz
			once.Do(func() {                             // thread-safe, faqat bir marta
				firstErr = err                       // xatoni saqlaymiz
				cancel()                             // barcha worker'larni to'xtatamiz
			})
		}
	}

	worker := func() {                                           // worker funksiyasi
		defer wg.Done()                                      // tugallanganini signal qilamiz
		for {                                                // to'xtatilguncha sikl
			select {                                     // job yoki bekor qilishni kutamiz
			case <-ctx.Done():                           // bekor qilindi
				return                               // worker'dan chiqamiz
			case job, ok := <-jobs:                      // job olamiz
				if !ok {                             // kanal yopildi
					return                       // worker'dan chiqamiz
				}
				func() {                             // recovery uchun wrapper
					defer func() {               // panic recovery
						if r := recover(); r != nil { // panic qo'lga olindi
							recordErr(fmt.Errorf("panic: %v", r)) // xatoga aylantiramiz
						}
					}()
					if err := handler(ctx, job); err != nil { // job'ni qayta ishlaymiz
						recordErr(err)           // handler xatosini yozamiz
					}
				}()                                  // darhol bajaramiz
			}
		}
	}

	wg.Add(workers)                                              // worker'larni ro'yxatga olamiz
	for i := 0; i < workers; i++ {                               // N worker ishga tushiramiz
		go worker()                                          // goroutine ishga tushiramiz
	}

	wg.Wait()                                                    // tugallashni kutamiz
	return firstErr                                              // birinchi xatoni qaytaramiz
}`,
                whyItMatters: `Production tizimlar bardoshli bo'lishi kerak. Bitta yomon kirish yoki chekka holat butun servisingizni qulatmasligi kerak. Panic recovery crashlarni log qilinadigan, hisobot beriladigan va gracefully qayta ishlanadigan xatolarga aylantiradi.

Asosiy texnika - handler chaqiruvini defer/recover bilan anonim funksiyaga o'rash. Bu panic'ni o'sha funksiya doirasiga izolyatsiya qiladi, worker boshqa vazifalarni qayta ishlashni davom ettirishga imkon beradi.

Bu pattern HTTP serverlarda (bitta yomon so'rovda qulamaslik), xabar navbatlarida (butun batch'ni yo'qotmaslik) va mavjudlik fail-fast'dan muhimroq bo'lgan har qanday tizimda ishlatiladi. Doimo qo'lga olingan panic'larni log qiling - ular tuzatilishi kerak bo'lgan buglarni ko'rsatadi.`
              }
            }
          }
        ]
      }
    ]
  }
];
