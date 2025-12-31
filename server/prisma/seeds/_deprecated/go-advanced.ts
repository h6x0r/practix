export const GO_ADVANCED_MODULES = [
	{
		title: 'Concurrency & Synchronization',
		description: 'Deep dive into Goroutines, Channels, and the sync package.',
		section: 'core',
		order: 1,
		topics: [
			{
				title: 'Channels Patterns',
				description: 'Advanced communication patterns using channels.',
				difficulty: 'hard',
				estimatedTime: '4h',
				order: 1,
				tasks: [
					{
						slug: 'go-fan-in',
						title: 'Fan-In Pattern',
						difficulty: 'medium',
						tags: ['go', 'concurrency', 'channels'],
						estimatedTime: '30m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement the **Fan-In** pattern to merge multiple input channels into a single output channel.

**Requirements:**
1. Accept a variadic number of read-only channels
2. The output channel must close **only** when all input channels are closed
3. It must be thread-safe
4. Ensure no goroutines leak after completion

**Example:**
\`\`\`go
ch1 := make(chan int, 2)
ch1 <- 1
ch1 <- 2
close(ch1)

ch2 := make(chan int, 2)
ch2 <- 3
ch2 <- 4
close(ch2)

out := FanIn(ctx, ch1, ch2)
// Output: [1, 3, 2, 4] (order may vary)
\`\`\`

**Constraints:**
- Handle nil context by using context.Background()
- Skip nil input channels`,
						initialCode: `package channelsx

import (
	"context"
)

// FanIn merges multiple input channels into a single output channel.
func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
	out := make(chan T)
	// TODO: Implement fan-in logic
	return out
}`,
						solutionCode: `package channelsx

import (
	"context"
	"sync"
)

func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
	if ctx == nil {                                             // Handle nil context
		ctx = context.Background()                          // Use Background as default
	}
	out := make(chan T)                                         // Create output channel
	var wg sync.WaitGroup                                       // Track active forwarders

	forward := func(in <-chan T) {                              // Forward from one input
		defer wg.Done()                                     // Signal completion
		for {
			select {
			case <-ctx.Done():                          // Context cancelled
				return                              // Exit goroutine
			case v, ok := <-in:                         // Receive from input
				if !ok {                            // Input closed
					return                      // Exit goroutine
				}
				select {
				case <-ctx.Done():                  // Check cancellation
					return                      // Exit if cancelled
				case out <- v:                      // Forward to output
				}
			}
		}
	}

	for _, in := range ins {                                    // Launch forwarders
		if in == nil {                                      // Skip nil channels
			continue
		}
		wg.Add(1)                                           // Register forwarder
		go forward(in)                                      // Launch goroutine
	}

	go func() {                                                 // Close coordinator
		wg.Wait()                                           // Wait for all forwarders
		close(out)                                          // Close output channel
	}()

	return out                                                  // Return output channel
}`,
						hint1: "Use sync.WaitGroup to track active input channels.",
						hint2: "Launch a goroutine per input to forward to out.",
						whyItMatters: `Fan-In is essential for aggregating results from multiple concurrent workers. You'll use this pattern when collecting data from parallel API calls, merging log streams from multiple services, or combining outputs from a distributed task queue.

In production, fan-in enables horizontal scaling - spawn N workers to process data in parallel, then merge results into a single stream for downstream processing. This pattern is used in Google's MapReduce, Kubernetes controllers watching multiple resources, and any system that needs to multiplex concurrent data sources.

The key challenge is coordinating closure - the output channel must close only after ALL input channels close, requiring sync.WaitGroup to track completion.`,
						translations: {
							ru: {
								title: 'Паттерн Fan-In',
								description: `Реализуйте паттерн **Fan-In** для объединения нескольких входных каналов в один выходной.

**Требования:**
1. Принимайте variadic количество read-only каналов
2. Выходной канал должен закрываться **только** когда все входные каналы закрыты
3. Должно быть потокобезопасным
4. Убедитесь, что горутины не утекают после завершения

**Пример:**
\`\`\`go
ch1 := make(chan int, 2)
ch1 <- 1
ch1 <- 2
close(ch1)

ch2 := make(chan int, 2)
ch2 <- 3
ch2 <- 4
close(ch2)

out := FanIn(ctx, ch1, ch2)
// Вывод: [1, 3, 2, 4] (порядок может варьироваться)
\`\`\`

**Ограничения:**
- Обработайте nil context используя context.Background()
- Пропускайте nil входные каналы`,
								hint1: 'Используйте sync.WaitGroup для отслеживания активных входных каналов.',
								hint2: 'Запустите горутину для каждого входа для пересылки в выходной канал.',
								whyItMatters: `Fan-In необходим для агрегации результатов от множества конкурентных воркеров. Этот паттерн используется при сборе данных от параллельных API вызовов, объединении потоков логов от нескольких сервисов или комбинировании выходов от распределённой очереди задач.

В production fan-in обеспечивает горизонтальное масштабирование - запускайте N воркеров для параллельной обработки данных, затем объединяйте результаты в один поток для downstream обработки. Этот паттерн используется в Google MapReduce, Kubernetes контроллерах и любой системе, которая мультиплексирует конкурентные источники данных.`
							},
							uz: {
								title: 'Fan-In pattern',
								description: `Ko'p kirish kanallarini bitta chiqish kanaliga birlashtiruvchi **Fan-In** patternni amalga oshiring.

**Talablar:**
1. Variadic miqdordagi faqat o'qish uchun kanallarni qabul qiling
2. Chiqish kanali **faqat** barcha kirish kanallari yopilgandan keyin yopilishi kerak
3. Thread-safe bo'lishi kerak
4. Goroutine'lar tugagandan keyin oqib ketmasligiga ishonch hosil qiling

**Misol:**
\`\`\`go
ch1 := make(chan int, 2)
ch1 <- 1
ch1 <- 2
close(ch1)

ch2 := make(chan int, 2)
ch2 <- 3
ch2 <- 4
close(ch2)

out := FanIn(ctx, ch1, ch2)
// Chiqish: [1, 3, 2, 4] (tartib o'zgarishi mumkin)
\`\`\`

**Cheklovlar:**
- nil context'ni context.Background() yordamida qayta ishlang
- nil kirish kanallarini o'tkazib yuboring`,
								hint1: 'Faol kirish kanallarini kuzatish uchun sync.WaitGroup dan foydalaning.',
								hint2: 'Har bir kirish uchun chiqishga yo\'naltirish uchun goroutine ishga tushiring.',
								whyItMatters: `Fan-In ko'p concurrent worker'lardan natijalarni yig'ish uchun zarur. Bu patterndan parallel API chaqiruvlardan ma'lumot yig'ishda, bir nechta servislardan log oqimlarini birlashtirishda yoki taqsimlangan vazifalar navbatidan chiqishlarni birlashtirishda foydalanasiz.

Production'da fan-in gorizontal miqyoslashni ta'minlaydi - ma'lumotlarni parallel qayta ishlash uchun N worker'lar ishga tushiring, keyin natijalarni downstream qayta ishlash uchun bitta oqimga birlashtiring.`
							}
						}
					},
					{
						slug: 'go-worker-pool',
						title: 'Worker Pool',
						difficulty: 'hard',
						tags: ['go', 'concurrency', 'patterns'],
						estimatedTime: '45m',
						isPremium: true,
						youtubeUrl: '',
						description: `Implement a **Worker Pool** to process jobs concurrently with a controlled number of workers.

**Requirements:**
1. \`RunWorkerPool\` accepts a context, an input channel, a worker count, and a handler function
2. Spawn exactly \`workers\` number of goroutines
3. If any handler returns an error, capture the first error and cancel remaining workers
4. Return the first error encountered

**Example:**
\`\`\`go
jobs := make(chan int, 10)
for i := 0; i < 10; i++ {
    jobs <- i
}
close(jobs)

err := RunWorkerPool(ctx, jobs, 5, func(ctx context.Context, n int) error {
    // Process job n
    return nil
})
\`\`\`

**Constraints:**
- Handle nil context, nil handler, or invalid worker count gracefully
- Workers should stop immediately on context cancellation`,
						initialCode: `package channelsx

import (
	"context"
)

type Handler[T any] func(context.Context, T) error

func RunWorkerPool[T any](ctx context.Context, in <-chan T, workers int, h Handler[T]) error {
	// TODO: Implement worker pool
	return nil
}`,
						solutionCode: `package channelsx

import (
	"context"
	"sync"
)

type Handler[T any] func(context.Context, T) error

func RunWorkerPool[T any](ctx context.Context, in <-chan T, workers int, h Handler[T]) error {
	if ctx == nil {                                             // Handle nil context
		ctx = context.Background()                          // Use Background as default
	}
	if h == nil || workers <= 0 || in == nil {                  // Validate inputs
		return nil                                          // Nothing to do
	}
	ctx, cancel := context.WithCancel(ctx)                      // Create cancellable context
	defer cancel()                                              // Cleanup on return

	var (
		wg       sync.WaitGroup                             // Track worker completion
		once     sync.Once                                  // Single error capture
		firstErr error                                      // Store first error
	)

	recordErr := func(err error) {                              // Error recording helper
		if err != nil {                                     // Only record non-nil
			once.Do(func() {                            // Thread-safe, once only
				firstErr = err                      // Store error
				cancel()                            // Stop all workers
			})
		}
	}

	worker := func() {                                          // Worker function
		defer wg.Done()                                     // Signal completion
		for {                                               // Loop until stopped
			select {
			case <-ctx.Done():                          // Context cancelled
				return                              // Exit worker
			case v, ok := <-in:                         // Receive job
				if !ok {                            // Channel closed
					return                      // Exit worker
				}
				if err := h(ctx, v); err != nil {   // Process job
					recordErr(err)              // Record error
				}
			}
		}
	}

	wg.Add(workers)                                             // Register workers
	for i := 0; i < workers; i++ {                              // Spawn N workers
		go worker()                                         // Launch goroutine
	}
	wg.Wait()                                                   // Wait for completion
	return firstErr                                             // Return first error
}`,
						hint1: "Use sync.Once to capture only the first error.",
						hint2: "Call cancel() in error handler to stop workers.",
						whyItMatters: `Worker pools control concurrency to prevent resource exhaustion. Without limiting workers, spawning 100,000 goroutines to process 100,000 tasks would consume excessive memory and context-switch overhead.

In production, worker pools are used for: HTTP server request handling (limit concurrent requests), database query processing (respect connection pool limits), file processing (limit open file descriptors), and API rate limiting (stay within quota).

The key pattern is fail-fast - when one worker encounters an error (bad connection, invalid data), cancel the entire batch immediately rather than wasting resources processing the rest. Companies like Stripe use this pattern in payment processing pipelines.`,
						translations: {
							ru: {
								title: 'Worker Pool',
								description: `Реализуйте **Worker Pool** для конкурентной обработки задач с контролируемым количеством воркеров.

**Требования:**
1. \`RunWorkerPool\` принимает context, входной канал, количество воркеров и функцию-обработчик
2. Создайте ровно \`workers\` горутин
3. Если любой обработчик вернёт ошибку, захватите первую ошибку и отмените оставшихся воркеров
4. Верните первую встреченную ошибку

**Пример:**
\`\`\`go
jobs := make(chan int, 10)
for i := 0; i < 10; i++ {
    jobs <- i
}
close(jobs)

err := RunWorkerPool(ctx, jobs, 5, func(ctx context.Context, n int) error {
    // Обработка задачи n
    return nil
})
\`\`\`

**Ограничения:**
- Обработайте nil context, nil handler или некорректное количество воркеров gracefully
- Воркеры должны остановиться немедленно при отмене context`,
								hint1: 'Используйте sync.Once для захвата только первой ошибки.',
								hint2: 'Вызовите cancel() в обработчике ошибок для остановки воркеров.',
								whyItMatters: `Worker pools контролируют конкурентность для предотвращения исчерпания ресурсов. Без ограничения воркеров, создание 100,000 горутин для обработки 100,000 задач потребляет чрезмерную память и накладные расходы на переключение контекста.

В production worker pools используются для: обработки HTTP запросов (ограничение конкурентных запросов), обработки запросов к БД (соблюдение лимитов connection pool), обработки файлов (ограничение открытых файловых дескрипторов) и API rate limiting.

Ключевой паттерн - fail-fast: когда один воркер сталкивается с ошибкой, отменить весь batch немедленно, а не тратить ресурсы на обработку остального.`
							},
							uz: {
								title: 'Worker Pool',
								description: `Vazifalarni boshqariladigan miqdordagi worker'lar bilan concurrent qayta ishlash uchun **Worker Pool** amalga oshiring.

**Talablar:**
1. \`RunWorkerPool\` context, kirish kanali, worker'lar soni va handler funksiyasini qabul qiladi
2. Aynan \`workers\` soni goroutine'lar yarating
3. Agar biron handler xato qaytarsa, birinchi xatoni ushlang va qolgan worker'larni bekor qiling
4. Duch kelgan birinchi xatoni qaytaring

**Misol:**
\`\`\`go
jobs := make(chan int, 10)
for i := 0; i < 10; i++ {
    jobs <- i
}
close(jobs)

err := RunWorkerPool(ctx, jobs, 5, func(ctx context.Context, n int) error {
    // n vazifasini qayta ishlash
    return nil
})
\`\`\`

**Cheklovlar:**
- nil context, nil handler yoki noto'g'ri worker sonini to'g'ri qayta ishlang
- Worker'lar context bekor qilinganda darhol to'xtashi kerak`,
								hint1: 'Faqat birinchi xatoni ushlash uchun sync.Once dan foydalaning.',
								hint2: 'Worker\'larni to\'xtatish uchun xato handler\'ida cancel() ni chaqiring.',
								whyItMatters: `Worker pool'lar resurslarning tugashini oldini olish uchun konkurrentlikni nazorat qiladi. Worker'larni cheklamasdan, 100,000 vazifani qayta ishlash uchun 100,000 goroutine yaratish haddan tashqari xotira va context-switch xarajatlarini talab qiladi.

Production'da worker pool'lar quyidagilar uchun ishlatiladi: HTTP so'rovlarini qayta ishlash (concurrent so'rovlarni cheklash), DB so'rovlarini qayta ishlash (connection pool limitlarini hurmat qilish), fayl qayta ishlash va API rate limiting.`
							}
						}
					}
				]
			}
		]
	},
	{
		title: 'Resilience & Reliability',
		description: 'Building robust systems that handle failures gracefully.',
		section: 'core',
		order: 2,
		topics: [
			{
				title: 'Caching Strategies',
				description: 'In-memory storage mechanisms and expiration policies.',
				difficulty: 'medium',
				estimatedTime: '2h',
				order: 1,
				tasks: [
					{
						slug: 'go-ttl-cache',
						title: 'TTL Cache',
						difficulty: 'medium',
						tags: ['go', 'cache', 'system-design'],
						estimatedTime: '45m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement a thread-safe **TTL (Time-To-Live) Cache** with lazy expiration.

**Requirements:**
1. \`NewTTLCache(ttl)\` initializes the cache with a default TTL
2. \`Set(key, value)\` adds or updates items with automatic expiration
3. \`Get(key)\` retrieves items, returning nil if expired
4. **Important**: Implement lazy expiration - remove expired items only when accessed via \`Get\`
5. Must be safe for concurrent use from multiple goroutines

**Example:**
\`\`\`go
cache := NewTTLCache(100 * time.Millisecond)
cache.Set("key", "value")
val, ok := cache.Get("key")  // val="value", ok=true

time.Sleep(150 * time.Millisecond)
val, ok = cache.Get("key")   // val=nil, ok=false (expired)
\`\`\`

**Constraints:**
- Use RWMutex for efficient concurrent reads
- Handle race conditions when upgrading from read to write lock`,
						initialCode: `package cache

import (
	"sync"
	"time"
)

type TTLCache struct {
	// TODO: Add fields
}

func NewTTLCache(ttl time.Duration) *TTLCache {
	return &TTLCache{}
}

func (c *TTLCache) Set(key string, v any) {
    // Implement
}

func (c *TTLCache) Get(key string) (any, bool) {
    // Implement
    return nil, false
}`,
						solutionCode: `package cache

import (
	"sync"
	"time"
)

type entry struct {                                                 // Cache entry with expiration
	v   any                                                     // Stored value
	exp time.Time                                               // Expiration timestamp
}

type TTLCache struct {
	mu  sync.RWMutex                                             // Read-write lock for concurrency
	m   map[string]entry                                         // Underlying storage map
	ttl time.Duration                                            // Default time-to-live
}

func NewTTLCache(ttl time.Duration) *TTLCache {
	return &TTLCache{m: make(map[string]entry), ttl: ttl}       // Initialize map and TTL
}

func (c *TTLCache) Set(key string, v any) {
	if c == nil {                                               // Nil receiver check
		return
	}
	c.mu.Lock()                                                 // Acquire write lock
	defer c.mu.Unlock()                                         // Release on return

	expire := time.Time{}                                       // Zero time = no expiration
	if c.ttl > 0 {                                              // If TTL configured
		expire = time.Now().Add(c.ttl)                      // Set expiration time
	}
	c.m[key] = entry{v: v, exp: expire}                         // Store entry
}

func (c *TTLCache) Get(key string) (any, bool) {
	if c == nil {                                               // Nil receiver check
		return nil, false
	}

	c.mu.RLock()                                                // Acquire read lock (cheap)
	ent, ok := c.m[key]                                         // Read entry
	c.mu.RUnlock()                                              // Release read lock

	if !ok {                                                    // Key not found
		return nil, false
	}

	if !ent.exp.IsZero() && time.Now().After(ent.exp) {        // Check expiration
		c.mu.Lock()                                         // Upgrade to write lock
		defer c.mu.Unlock()                                 // Release write lock
		if entCur, still := c.m[key]; still && entCur.exp == ent.exp { // Double-check pattern
			delete(c.m, key)                            // Remove expired entry
		}
		return nil, false                                   // Return not found
	}
	return ent.v, true                                          // Return valid entry
}`,
						hint1: "Use sync.RWMutex with RLock for reads.",
						hint2: "Re-check expiry after upgrading to Write lock.",
						whyItMatters: `TTL caches are critical for performance in distributed systems. They prevent expensive recomputations and reduce load on databases and external APIs. Without TTL, stale data lingers indefinitely, consuming memory and serving outdated information.

Lazy expiration (only remove on access) is more efficient than active expiration (background cleanup). Redis, Memcached, and most in-memory caches use this approach - expired entries are removed when accessed, not via periodic scans.

The double-check pattern after upgrading locks prevents race conditions: between releasing RLock and acquiring Lock, another goroutine might have already deleted the entry. This pattern is used in sync.Map and concurrent hash maps.

Real-world use cases: API response caching, session storage, rate limiting counters, DNS resolution caching.`,
						translations: {
							ru: {
								title: 'TTL Cache',
								description: `Реализуйте потокобезопасный **TTL (Time-To-Live) Cache** с ленивым истечением.

**Требования:**
1. \`NewTTLCache(ttl)\` инициализирует кеш с дефолтным TTL
2. \`Set(key, value)\` добавляет или обновляет элементы с автоматическим истечением
3. \`Get(key)\` возвращает элементы, возвращая nil если истёк срок
4. **Важно**: Реализуйте ленивое истечение - удаляйте истёкшие элементы только при доступе через \`Get\`
5. Должно быть безопасно для конкурентного использования из множества горутин

**Пример:**
\`\`\`go
cache := NewTTLCache(100 * time.Millisecond)
cache.Set("key", "value")
val, ok := cache.Get("key")  // val="value", ok=true

time.Sleep(150 * time.Millisecond)
val, ok = cache.Get("key")   // val=nil, ok=false (истёк)
\`\`\`

**Ограничения:**
- Используйте RWMutex для эффективных конкурентных чтений
- Обработайте race conditions при переходе с read на write lock`,
								hint1: 'Используйте sync.RWMutex с RLock для чтений.',
								hint2: 'Перепроверьте истечение после перехода на Write lock.',
								whyItMatters: `TTL кеши критичны для производительности в распределённых системах. Они предотвращают дорогие пересчёты и снижают нагрузку на БД и внешние API. Без TTL устаревшие данные остаются бесконечно, потребляя память.

Ленивое истечение (удаление только при доступе) эффективнее активного (фоновая очистка). Redis, Memcached и большинство in-memory кешей используют этот подход.

Паттерн double-check после перехода на locks предотвращает race conditions: между освобождением RLock и получением Lock другая горутина может удалить запись.

Реальные use cases: кеширование API ответов, хранение сессий, счётчики rate limiting, DNS кеширование.`
							},
							uz: {
								title: 'TTL Cache',
								description: `Thread-safe **TTL (Time-To-Live) Cache** ni lazy expiration bilan amalga oshiring.

**Talablar:**
1. \`NewTTLCache(ttl)\` default TTL bilan keshni boshlaydi
2. \`Set(key, value)\` avtomatik expiration bilan elementlarni qo'shadi yoki yangilaydi
3. \`Get(key)\` elementlarni qaytaradi, muddati o'tgan bo'lsa nil qaytaradi
4. **Muhim**: Lazy expiration amalga oshiring - muddati o'tgan elementlarni faqat \`Get\` orqali kirishda o'chiring
5. Ko'p goroutine'lardan concurrent foydalanish uchun xavfsiz bo'lishi kerak

**Misol:**
\`\`\`go
cache := NewTTLCache(100 * time.Millisecond)
cache.Set("key", "value")
val, ok := cache.Get("key")  // val="value", ok=true

time.Sleep(150 * time.Millisecond)
val, ok = cache.Get("key")   // val=nil, ok=false (muddati o'tdi)
\`\`\`

**Cheklovlar:**
- Samarali concurrent o'qish uchun RWMutex dan foydalaning
- Read'dan write lock'ga o'tganda race condition'larni qayta ishlang`,
								hint1: 'O\'qish uchun sync.RWMutex va RLock dan foydalaning.',
								hint2: 'Write lock\'ga o\'tgandan keyin expiration\'ni qayta tekshiring.',
								whyItMatters: `TTL keshlar taqsimlangan tizimlarda performance uchun kritik. Ular qimmat qayta hisoblashlarni oldini oladi va DB va tashqi API'larga yukni kamaytiradi.

Lazy expiration (faqat kirishda o'chirish) active expiration'dan (background cleanup) samaraliroq. Redis, Memcached va aksariyat in-memory keshlar bu yondashuvni qo'llaydi.

Real-world: API response caching, session storage, rate limiting counter'lar, DNS caching.`
							}
						}
					}
				]
			},
			{
				title: 'Fault Tolerance',
				description: 'Patterns like Circuit Breaker and Retries.',
				difficulty: 'hard',
				estimatedTime: '3h',
				order: 2,
				tasks: [
					{
						slug: 'go-circuit-breaker',
						title: 'Circuit Breaker',
						difficulty: 'hard',
						tags: ['go', 'resiliency'],
						estimatedTime: '1h',
						isPremium: true,
						youtubeUrl: '',
						description: `Implement a **Circuit Breaker** state machine to prevent cascading failures.

The breaker has 3 states with automatic transitions:

**Requirements:**
1. **Closed** state: Normal operation - requests pass through, errors count towards threshold
2. **Open** state: Fail-fast - requests fail immediately with \`ErrOpen\` without calling the function
3. **Half-Open** state: Recovery testing - allow limited requests to test if service recovered
4. Implement automatic state transitions based on success/failure patterns

**State Transitions:**
- Closed → Open: When failure count >= threshold
- Open → Half-Open: After \`openDur\` time passes
- Half-Open → Closed: After \`halfMax\` consecutive successes
- Half-Open → Open: If any request fails

**Example:**
\`\`\`go
breaker := New(3, 5*time.Second, 2)  // threshold=3, cooldown=5s, halfMax=2

// Closed state - 3 failures trip to Open
breaker.Do(ctx, failingFunc)  // err (1)
breaker.Do(ctx, failingFunc)  // err (2)
breaker.Do(ctx, failingFunc)  // err (3) -> trips to Open

breaker.Do(ctx, anyFunc)      // ErrOpen (fail-fast)

time.Sleep(5 * time.Second)   // Wait for cooldown -> Half-Open

breaker.Do(ctx, successFunc)  // success (1)
breaker.Do(ctx, successFunc)  // success (2) -> back to Closed
\`\`\`

**Constraints:**
- Thread-safe for concurrent use
- Check state before executing function to avoid unnecessary work when Open`,
						initialCode: `package circuitx

import (
	"context"
	"errors"
	"sync"
	"time"
)

var ErrOpen = errors.New("circuit open")

type State int

const (
	Closed State = iota
	Open
	HalfOpen
)

type Breaker struct {
	// TODO: Add fields
}

func New(threshold int, openDur time.Duration, halfMax int) *Breaker {
	return &Breaker{
        // Init
	}
}

func (b *Breaker) Do(ctx context.Context, f func(context.Context) error) error {
	// TODO: Implement state machine
    return nil
}`,
						solutionCode: `package circuitx

import (
	"context"
	"errors"
	"sync"
	"time"
)

var ErrOpen = errors.New("circuit open")

type State int

const (
	Closed State = iota
	Open
	HalfOpen
)

type Breaker struct {
	mu        sync.Mutex                                          // Protect state transitions
	state     State                                               // Current state
	errs      int                                                 // Error count in Closed
	threshold int                                                 // Errors before trip to Open
	openUntil time.Time                                           // When to transition from Open
	openDur   time.Duration                                       // How long to stay Open
	halfMax   int                                                 // Successes needed in Half-Open
	halfCount int                                                 // Success count in Half-Open
}

func New(threshold int, openDur time.Duration, halfMax int) *Breaker {
	return &Breaker{
		state:     Closed,                                      // Start in Closed state
		threshold: threshold,                                   // Store threshold
		openDur:   openDur,                                     // Store cooldown duration
		halfMax:   halfMax,                                     // Store half-open success target
	}
}

func (b *Breaker) Do(ctx context.Context, f func(context.Context) error) error {
	b.mu.Lock()                                                 // Lock for state check
	now := time.Now()
	switch b.state {
	case Open:
		if now.After(b.openUntil) {                         // Cooldown expired
			b.state = HalfOpen                          // Transition to Half-Open
			b.halfCount = 0                             // Reset success counter
		} else {
			b.mu.Unlock()                               // Unlock before return
			return ErrOpen                              // Fail-fast, don't call f
		}
	}
	b.mu.Unlock()                                               // Unlock before calling f

	err := f(ctx)                                               // Execute function

	b.mu.Lock()                                                 // Lock for state update
	defer b.mu.Unlock()                                         // Ensure unlock

	if err == nil {                                             // Success case
		switch b.state {
		case Closed:
			b.errs = 0                                  // Reset error counter
		case HalfOpen:
			b.halfCount++                               // Increment success count
			if b.halfCount >= b.halfMax {               // Enough successes
				b.state = Closed                    // Recover to Closed
				b.errs = 0                          // Reset counters
				b.halfCount = 0
			}
		}
		return nil                                          // Return success
	}

	// Failure case
	switch b.state {
	case Closed:
		b.errs++                                            // Count error
		if b.errs >= b.threshold {                          // Threshold exceeded
			b.tripToOpen()                              // Trip to Open
		}
	case HalfOpen:
		b.tripToOpen()                                      // Any failure trips back
	}
	return err                                                  // Return error
}

func (b *Breaker) tripToOpen() {
	b.state = Open                                              // Set state to Open
	b.openUntil = time.Now().Add(b.openDur)                     // Set cooldown expiry
	b.errs = 0                                                  // Reset error count
	b.halfCount = 0                                             // Reset half-open count
}`,
						hint1: "Use sync.Mutex to protect all state transitions.",
						hint2: "Check state before f(), update state after.",
						whyItMatters: `Circuit breakers prevent cascading failures in distributed systems. When a downstream service fails, continuing to send requests wastes resources and makes recovery harder. The circuit breaker "trips" to stop traffic, giving the failing service time to recover.

This pattern is critical in microservices architecture. Without circuit breakers, a failing database can cause your entire request queue to back up, consuming memory and threads until your service crashes. Companies like Netflix pioneered this pattern with Hystrix.

The three-state design is elegant: Closed (normal), Open (fail-fast during outage), Half-Open (cautious recovery testing). This prevents "thundering herd" - when a service recovers, gradually increase load instead of immediately flooding it.

Real-world implementations: AWS API Gateway, Istio service mesh, Envoy proxy, Spring Cloud Circuit Breaker. Used for: database connections, external API calls, microservice communication.`,
						translations: {
							ru: {
								title: 'Circuit Breaker',
								description: `Реализуйте машину состояний **Circuit Breaker** для предотвращения каскадных сбоев.

Breaker имеет 3 состояния с автоматическими переходами:

**Требования:**
1. Состояние **Closed**: Нормальная работа - запросы проходят, ошибки считаются к порогу
2. Состояние **Open**: Fail-fast - запросы сразу падают с \`ErrOpen\` без вызова функции
3. Состояние **Half-Open**: Тестирование восстановления - разрешить ограниченные запросы
4. Реализуйте автоматические переходы состояний на основе паттернов успеха/неудачи

**Переходы состояний:**
- Closed → Open: Когда количество ошибок >= threshold
- Open → Half-Open: После прохождения \`openDur\` времени
- Half-Open → Closed: После \`halfMax\` последовательных успехов
- Half-Open → Open: Если любой запрос падает

**Пример:**
\`\`\`go
breaker := New(3, 5*time.Second, 2)  // threshold=3, cooldown=5s, halfMax=2

// Closed - 3 ошибки триггерят Open
breaker.Do(ctx, failingFunc)  // err (1)
breaker.Do(ctx, failingFunc)  // err (2)
breaker.Do(ctx, failingFunc)  // err (3) -> переход в Open

breaker.Do(ctx, anyFunc)      // ErrOpen (fail-fast)

time.Sleep(5 * time.Second)   // Ждём cooldown -> Half-Open

breaker.Do(ctx, successFunc)  // success (1)
breaker.Do(ctx, successFunc)  // success (2) -> обратно в Closed
\`\`\`

**Ограничения:**
- Потокобезопасно для конкурентного использования
- Проверяйте состояние перед выполнением функции`,
								hint1: 'Используйте sync.Mutex для защиты всех переходов состояний.',
								hint2: 'Проверяйте состояние перед f(), обновляйте состояние после.',
								whyItMatters: `Circuit breakers предотвращают каскадные сбои в распределённых системах. Когда downstream сервис падает, продолжение отправки запросов тратит ресурсы и усложняет восстановление. Circuit breaker "срабатывает", останавливая трафик.

Этот паттерн критичен в микросервисной архитектуре. Без circuit breaker падающая БД может вызвать переполнение очереди запросов, потребляя память и потоки до краша сервиса. Netflix пионеры этого паттерна с Hystrix.

Трёхстадийный дизайн элегантен: Closed (норма), Open (fail-fast при аутедже), Half-Open (осторожное тестирование). Предотвращает "thundering herd".

Real-world: AWS API Gateway, Istio, Envoy, Spring Cloud Circuit Breaker. Используется для: DB соединений, внешних API, микросервисной коммуникации.`
							},
							uz: {
								title: 'Circuit Breaker',
								description: `Kaskad nosozliklarni oldini olish uchun **Circuit Breaker** state machine'ni amalga oshiring.

Breaker avtomatik o'tishlar bilan 3 holatga ega:

**Talablar:**
1. **Closed** holati: Oddiy operatsiya - so'rovlar o'tadi, xatolar threshold'ga hisoblanadi
2. **Open** holati: Fail-fast - so'rovlar darhol \`ErrOpen\` bilan muvaffaqiyatsiz bo'ladi
3. **Half-Open** holati: Tiklanishni test qilish - cheklangan so'rovlarga ruxsat bering
4. Muvaffaqiyat/muvaffaqiyatsizlik patternlariga asoslangan avtomatik o'tishlarni amalga oshiring

**Holat o'tishlari:**
- Closed → Open: Xatolar soni >= threshold bo'lganda
- Open → Half-Open: \`openDur\` vaqt o'tgandan keyin
- Half-Open → Closed: \`halfMax\` ketma-ket muvaffaqiyatlardan keyin
- Half-Open → Open: Agar biron so'rov muvaffaqiyatsiz bo'lsa

**Misol:**
\`\`\`go
breaker := New(3, 5*time.Second, 2)  // threshold=3, cooldown=5s, halfMax=2

breaker.Do(ctx, failingFunc)  // err (1)
breaker.Do(ctx, failingFunc)  // err (2)
breaker.Do(ctx, failingFunc)  // err (3) -> Open'ga o'tish

breaker.Do(ctx, anyFunc)      // ErrOpen (fail-fast)

time.Sleep(5 * time.Second)   // Cooldown kutish -> Half-Open

breaker.Do(ctx, successFunc)  // success (1)
breaker.Do(ctx, successFunc)  // success (2) -> Closed'ga qaytish
\`\`\`

**Cheklovlar:**
- Concurrent foydalanish uchun thread-safe
- Open bo'lganda keraksiz ishni oldini olish uchun holatni tekshiring`,
								hint1: 'Barcha holat o\'tishlarini himoya qilish uchun sync.Mutex dan foydalaning.',
								hint2: 'f() dan oldin holatni tekshiring, keyin holatni yangilang.',
								whyItMatters: `Circuit breaker'lar taqsimlangan tizimlarda kaskad nosozliklarni oldini oladi. Downstream servis ishlamay qolganda, so'rovlarni yuborishda davom etish resurslarni isrof qiladi va tiklanishni qiyinlashtiradi.

Bu pattern mikroservis arxitekturasida kritik. Circuit breaker'siz ishlamay qolgan DB butun so'rov navbatini to'ldirishga olib kelishi mumkin. Netflix Hystrix bilan bu patternning kashshofi.

Real-world: AWS API Gateway, Istio, Envoy, Spring Cloud. DB ulanishlar, tashqi API'lar, mikroservis aloqasi uchun ishlatiladi.`
							}
						}
					}
				]
			}
		]
	},
	{
		title: 'Functional Programming',
		description: 'Utilizing Go Generics for functional patterns.',
		section: 'frameworks',
		order: 3,
		topics: [
			{
				title: 'Generics',
				description: 'Map, Filter, and Reduce implementation.',
				difficulty: 'medium',
				estimatedTime: '1.5h',
				order: 1,
				tasks: [
					{
						slug: 'go-generic-map',
						title: 'Generic Map',
						difficulty: 'easy',
						tags: ['go', 'generics'],
						estimatedTime: '15m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement a generic \`Map\` function that transforms a slice of one type to another.

**Requirements:**
1. Accept a slice of type \`T\` and a transformation function \`func(T) R\`
2. Return a new slice of type \`R\` containing transformed values
3. Preserve the original slice order
4. Pre-allocate the result slice for efficiency

**Example:**
\`\`\`go
numbers := []int{1, 2, 3, 4, 5}
doubled := Map(numbers, func(n int) int {
    return n * 2
})
// doubled = [2, 4, 6, 8, 10]

strings := Map(numbers, func(n int) string {
    return fmt.Sprintf("num-%d", n)
})
// strings = ["num-1", "num-2", "num-3", "num-4", "num-5"]
\`\`\`

**Constraints:**
- Do not modify the input slice
- Handle empty slices correctly`,
						initialCode: `package genericsx

func Map[T any, R any](in []T, f func(T) R) []R {
	// Implement
    return nil
}`,
						solutionCode: `package genericsx

func Map[T any, R any](in []T, f func(T) R) []R {
	out := make([]R, len(in))                                   // Pre-allocate with exact size
	for i, v := range in {                                      // Iterate over input
		out[i] = f(v)                                       // Apply transform and store
	}
	return out                                                  // Return transformed slice
}`,
						hint1: "Pre-allocate result slice with make([]R, len(in)).",
						hint2: "Iterate with range and apply f to each element.",
						whyItMatters: `Map is a fundamental functional programming pattern that makes data transformations clean and composable. Without generics (pre-Go 1.18), you'd need to write separate map functions for each type or use reflection with interface{} (slow and type-unsafe).

Generics enable zero-cost abstractions - the compiler generates type-specific code at compile time, so Map[int, string] has the same performance as a hand-written loop. This is different from languages like Java where generics use type erasure.

Real-world use cases: transforming API response DTOs to domain models, converting database rows to business objects, formatting data for display, applying business logic transformations across collections.

This pattern appears in every modern codebase - understanding Map/Filter/Reduce helps you write more declarative, testable code. Companies like Google use this pattern extensively in internal Go libraries.`,
						translations: {
							ru: {
								title: 'Generic Map',
								description: `Реализуйте generic функцию \`Map\`, преобразующую срез одного типа в другой.

**Требования:**
1. Принимайте срез типа \`T\` и функцию трансформации \`func(T) R\`
2. Возвращайте новый срез типа \`R\` с преобразованными значениями
3. Сохраняйте порядок элементов оригинального среза
4. Предварительно выделяйте результирующий срез для эффективности

**Пример:**
\`\`\`go
numbers := []int{1, 2, 3, 4, 5}
doubled := Map(numbers, func(n int) int {
    return n * 2
})
// doubled = [2, 4, 6, 8, 10]

strings := Map(numbers, func(n int) string {
    return fmt.Sprintf("num-%d", n)
})
// strings = ["num-1", "num-2", "num-3", "num-4", "num-5"]
\`\`\`

**Ограничения:**
- Не изменяйте входной срез
- Корректно обрабатывайте пустые срезы`,
								hint1: 'Предварительно выделите результирующий срез с make([]R, len(in)).',
								hint2: 'Итерируйте с range и применяйте f к каждому элементу.',
								whyItMatters: `Map - фундаментальный паттерн функционального программирования, делающий трансформации данных чистыми и композируемыми. Без дженериков (до Go 1.18) нужно было писать отдельные map функции для каждого типа или использовать reflection с interface{} (медленно и небезопасно по типам).

Generics обеспечивают zero-cost abstractions - компилятор генерирует type-specific код во время компиляции, так что Map[int, string] имеет ту же производительность что и написанный вручную цикл.

Real-world: преобразование API DTO в domain модели, конвертация БД строк в бизнес-объекты, форматирование данных для отображения, применение бизнес-логики к коллекциям.`
							},
							uz: {
								title: 'Generic Map',
								description: `Bir turni boshqa turga o'zgartiradigan generic \`Map\` funksiyasini amalga oshiring.

**Talablar:**
1. \`T\` turidan slice va \`func(T) R\` transformatsiya funksiyasini qabul qiling
2. O'zgartirilgan qiymatlarni o'z ichiga olgan \`R\` turidan yangi slice qaytaring
3. Asl slice tartibini saqlang
4. Samaradorlik uchun natija slice'ini oldindan ajrating

**Misol:**
\`\`\`go
numbers := []int{1, 2, 3, 4, 5}
doubled := Map(numbers, func(n int) int {
    return n * 2
})
// doubled = [2, 4, 6, 8, 10]

strings := Map(numbers, func(n int) string {
    return fmt.Sprintf("num-%d", n)
})
// strings = ["num-1", "num-2", "num-3", "num-4", "num-5"]
\`\`\`

**Cheklovlar:**
- Kirish slice'ini o'zgartirmang
- Bo'sh slice'larni to'g'ri qayta ishlang`,
								hint1: 'Natija slice\'ini make([]R, len(in)) bilan oldindan ajrating.',
								hint2: 'Range bilan takrorlang va har bir elementga f ni qo\'llang.',
								whyItMatters: `Map - funktsional dasturlashning fundamental patterni bo'lib, ma'lumotlar transformatsiyalarini toza va kompozitsiyalanadigan qiladi. Genericsiz (Go 1.18 gacha) har bir tur uchun alohida map funksiyalarini yozish yoki reflection bilan interface{} ishlatish kerak edi (sekin va type-unsafe).

Generics zero-cost abstraction'larni ta'minlaydi - kompilyator compile vaqtida type-specific kod yaratadi, shuning uchun Map[int, string] qo'lda yozilgan tsikl bilan bir xil performancega ega.

Real-world: API DTO'larni domain modellariga o'zgartirish, DB qatorlarini biznes ob'ektlariga konvertatsiya qilish, ma'lumotlarni ko'rsatish uchun formatlash.`
							}
						}
					}
				]
			}
		]
	},
];
