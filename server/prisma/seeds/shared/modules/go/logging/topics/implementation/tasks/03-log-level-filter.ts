import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-logging-level-filter',
	title: 'Dynamic Log Level Filtering',
	difficulty: 'medium',
	tags: ['go', 'logging', 'level-filtering', 'production'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a production-grade logger with dynamic log level filtering to control verbosity at runtime.

**Requirements:**
1. **LogLevel**: Define levels (DEBUG=0, INFO=1, WARN=2, ERROR=3, FATAL=4)
2. **SetLevel**: Set minimum log level globally (thread-safe with sync/atomic)
3. **GetLevel**: Get current minimum log level
4. **Debug, Info, Warn, Error, Fatal**: Level-specific logging functions that respect minimum level

**Log Level Pattern:**
\`\`\`go
// Development: See everything
SetLevel(DEBUG)
Debug("Variable value: %v", x)  // Logged
Info("Request started")          // Logged
Warn("Deprecated API used")      // Logged

// Production: Only important messages
SetLevel(WARN)
Debug("Variable value: %v", x)  // SKIPPED (too verbose)
Info("Request started")          // SKIPPED (too verbose)
Warn("Deprecated API used")      // Logged
Error("Database connection lost") // Logged
\`\`\`

**Why Level Filtering:**
- **Performance**: Skip expensive debug logging in production
- **Signal-to-Noise**: Filter out verbose logs to see what matters
- **Dynamic Control**: Change level without restarting (via API/config reload)

**Key Concepts:**
- Thread-safe level storage (use \`atomic.Int32\` or mutex)
- Early return pattern (check level before expensive formatting)
- Standard level hierarchy: DEBUG < INFO < WARN < ERROR < FATAL
- Fatal should panic/exit after logging

**Example Production Usage:**
\`\`\`go
func ProcessOrder(ctx context.Context, order *Order) error {
    Debug("Processing order: %+v", order) // Detailed object dump

    if err := validateOrder(order); err != nil {
        Warn("Order validation warning: %v", err) // Non-critical issue
    }

    if err := chargePayment(order); err != nil {
        Error("Payment failed for order %s: %v", order.ID, err) // Critical error
        return err
    }

    Info("Order %s completed successfully", order.ID) // Important milestone
    return nil
}

// In development: SetLevel(DEBUG) - see order details
// In production: SetLevel(INFO) - only milestones and errors
// During incident: SetLevel(DEBUG) - enable detailed logging dynamically
\`\`\`

**Constraints:**
- Use \`atomic.Int32\` for thread-safe level storage (no mutex needed)
- Early return if message level < minimum level (avoid string formatting)
- Format: \`[LEVEL] message\` (e.g., \`[ERROR] Database timeout\`)
- Use \`log.Printf\` for output
- Fatal should call \`panic()\` after logging`,
	initialCode: `package loggingx

import (
	"log"
	"sync/atomic"
)

type LogLevel int32

const (
	DEBUG LogLevel = 0
	INFO  LogLevel = 1
	WARN  LogLevel = 2
	ERROR LogLevel = 3
	FATAL LogLevel = 4
)

var currentLevel atomic.Int32

// TODO: Implement SetLevel
// Store level using atomic.StoreInt32(&currentLevel, int32(level))
// This ensures thread-safe writes without mutex
func SetLevel(level LogLevel) {
	// TODO: Implement
}

// TODO: Implement GetLevel
// Read level using atomic.LoadInt32(&currentLevel)
// Cast back to LogLevel type
func GetLevel() LogLevel {
	// TODO: Implement
}

// TODO: Implement Debug
// Check if DEBUG >= GetLevel() before logging
// Format: [DEBUG] <message>
func Debug(format string, args ...any) {
	// TODO: Implement
}

// TODO: Implement Info
// Check if INFO >= GetLevel() before logging
// Format: [INFO] <message>
func Info(format string, args ...any) {
	// TODO: Implement
}

// TODO: Implement Warn
// Check if WARN >= GetLevel() before logging
// Format: [WARN] <message>
func Warn(format string, args ...any) {
	// TODO: Implement
}

// TODO: Implement Error
// Check if ERROR >= GetLevel() before logging
// Format: [ERROR] <message>
func Error(format string, args ...any) {
	// TODO: Implement
}

// TODO: Implement Fatal
// Always log (FATAL is highest level)
// Format: [FATAL] <message>
// Call panic() after logging
func Fatal(format string, args ...any) {
	// TODO: Implement
}`,
	solutionCode: `package loggingx

import (
	"log"
	"sync/atomic"
)

type LogLevel int32

const (
	DEBUG LogLevel = 0
	INFO  LogLevel = 1
	WARN  LogLevel = 2
	ERROR LogLevel = 3
	FATAL LogLevel = 4
)

var currentLevel atomic.Int32

func SetLevel(level LogLevel) {
	atomic.StoreInt32(&currentLevel, int32(level)) // atomic write prevents data races when multiple goroutines change level
}

func GetLevel() LogLevel {
	return LogLevel(atomic.LoadInt32(&currentLevel)) // atomic read ensures consistent view across goroutines
}

func Debug(format string, args ...any) {
	if DEBUG < GetLevel() { // early exit avoids expensive string formatting when level disabled
		return
	}
	log.Printf("[DEBUG] "+format, args...) // prefix message with level for grep filtering
}

func Info(format string, args ...any) {
	if INFO < GetLevel() { // skip processing when current level filters out info messages
		return
	}
	log.Printf("[INFO] "+format, args...) // log informational milestone events
}

func Warn(format string, args ...any) {
	if WARN < GetLevel() { // only log warnings when level permits
		return
	}
	log.Printf("[WARN] "+format, args...) // signal non-critical issues requiring attention
}

func Error(format string, args ...any) {
	if ERROR < GetLevel() { // respect level even for errors (fatal might be only enabled level)
		return
	}
	log.Printf("[ERROR] "+format, args...) // record critical failures
}

func Fatal(format string, args ...any) {
	log.Printf("[FATAL] "+format, args...) // always log fatal messages regardless of level
	panic("fatal error occurred")          // terminate execution after logging unrecoverable error
}`,
	testCode: `package loggingx

import (
	"testing"
)

func Test1(t *testing.T) {
	SetLevel(DEBUG)
	level := GetLevel()
	if level != DEBUG {
		t.Errorf("expected DEBUG, got %d", level)
	}
}

func Test2(t *testing.T) {
	SetLevel(INFO)
	level := GetLevel()
	if level != INFO {
		t.Errorf("expected INFO, got %d", level)
	}
}

func Test3(t *testing.T) {
	SetLevel(WARN)
	level := GetLevel()
	if level != WARN {
		t.Errorf("expected WARN, got %d", level)
	}
}

func Test4(t *testing.T) {
	SetLevel(ERROR)
	level := GetLevel()
	if level != ERROR {
		t.Errorf("expected ERROR, got %d", level)
	}
}

func Test5(t *testing.T) {
	SetLevel(DEBUG)
	Debug("debug message %d", 1)
	Info("info message %d", 2)
	Warn("warn message %d", 3)
	Error("error message %d", 4)
}

func Test6(t *testing.T) {
	SetLevel(WARN)
	Debug("should not print %d", 1)
	Info("should not print %d", 2)
	Warn("should print %d", 3)
	Error("should print %d", 4)
}

func Test7(t *testing.T) {
	SetLevel(ERROR)
	Debug("filtered debug")
	Info("filtered info")
	Warn("filtered warn")
	Error("visible error")
}

func Test8(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Fatal should panic")
		}
	}()
	Fatal("fatal test")
}

func Test9(t *testing.T) {
	SetLevel(FATAL)
	Debug("filtered")
	Info("filtered")
	Warn("filtered")
	Error("filtered")
}

func Test10(t *testing.T) {
	SetLevel(DEBUG)
	for i := 0; i < 10; i++ {
		Debug("concurrent debug %d", i)
	}
	level := GetLevel()
	if level != DEBUG {
		t.Errorf("expected DEBUG, got %d", level)
	}
}
`,
	hint1: `Use atomic.StoreInt32(&currentLevel, int32(level)) in SetLevel and atomic.LoadInt32(&currentLevel) in GetLevel.`,
	hint2: `In each logging function, check if level < GetLevel() and return early to skip logging. Use log.Printf("[LEVEL] "+format, args...).`,
	whyItMatters: `Log level filtering is essential for production systems to control verbosity and performance without code changes.

**Why This Matters:**

**1. Performance Impact**
Debug logging can significantly slow down your application:
\`\`\`go
// Expensive operation
Debug("User data: %+v, Orders: %+v, History: %+v",
    user,           // 50 fields
    orders,         // 100 items array
    history)        // 1000 records

// Without level filtering:
// - Go formats the entire string (expensive)
// - Even if you don't need this log!

// With level filtering (production level = WARN):
if DEBUG < GetLevel() {
    return // Exit before formatting - costs ~2 nanoseconds
}
// String formatting skipped - save milliseconds per request
\`\`\`

**Real numbers from production:**
- 10,000 requests/second
- 50 debug logs per request
- Each debug format takes 100μs
- **Without filtering**: 50 seconds of CPU wasted per second (impossible!)
- **With filtering**: 1μs overhead (early returns)
- **Savings**: 99.998% CPU reduction

**2. Signal-to-Noise Ratio**
In production, too many logs bury important information:
\`\`\`
Development (DEBUG level):
[DEBUG] Parsing request headers
[DEBUG] Validating user token
[DEBUG] Querying user table
[DEBUG] User found: {...}
[DEBUG] Checking permissions
[DEBUG] Permission granted
[INFO] Request completed

Production (INFO level):
[INFO] Request completed

Production during incident (WARN level):
[WARN] Slow query: 2.5s
[ERROR] Database connection failed
[FATAL] Service unavailable
\`\`\`

See the difference? In production, you only see what matters!

**3. Dynamic Level Control**
Change log level without restarting:
\`\`\`go
// HTTP endpoint to change level
func HandleLogLevel(w http.ResponseWriter, r *http.Request) {
    level := r.URL.Query().Get("level")
    switch level {
    case "debug":
        SetLevel(DEBUG)
    case "info":
        SetLevel(INFO)
    case "warn":
        SetLevel(WARN)
    case "error":
        SetLevel(ERROR)
    }

    fmt.Fprintf(w, "Log level set to %s", level)
}

// During production incident:
// 1. Customer reports bug
// 2. curl http://api/admin/log-level?level=debug
// 3. Detailed logs start appearing immediately
// 4. Find root cause
// 5. curl http://api/admin/log-level?level=info
// 6. Back to normal operation

// No restart needed! Service never went down!
\`\`\`

**4. Environment-Specific Configuration**
\`\`\`go
func init() {
    env := os.Getenv("ENVIRONMENT")
    switch env {
    case "development":
        SetLevel(DEBUG) // See everything during development
    case "staging":
        SetLevel(INFO)  // Important events in staging
    case "production":
        SetLevel(WARN)  // Only warnings and errors in prod
    }
}
\`\`\`

**5. Why Thread-Safe Atomic Operations?**
\`\`\`go
// WRONG: Data race!
var level LogLevel
func SetLevel(l LogLevel) {
    level = l // Multiple goroutines writing = undefined behavior
}

// RIGHT: Atomic operations
var currentLevel atomic.Int32
func SetLevel(level LogLevel) {
    atomic.StoreInt32(&currentLevel, int32(level)) // Safe from any goroutine
}

// Why it matters:
// Goroutine 1: SetLevel(DEBUG) while handling admin request
// Goroutine 2: GetLevel() while processing customer request
// Goroutine 3: GetLevel() while running background job
// All three happening simultaneously - atomic ensures no corruption
\`\`\`

**6. Real-World Example**

**Scenario:** E-commerce site during Black Friday

**Morning (normal traffic):**
\`\`\`go
SetLevel(INFO)
// Logs:
[INFO] Order 1001 created
[INFO] Payment processed
[INFO] Order 1002 created
// 1000 log lines per minute
\`\`\`

**Noon (traffic spike - site slowing down):**
\`\`\`go
SetLevel(WARN)
// Logs:
[WARN] Database query slow: 1.2s
[WARN] Redis connection pool exhausted
[ERROR] Payment gateway timeout
// 50 log lines per minute - only problems!
\`\`\`

**Investigation (need details):**
\`\`\`go
SetLevel(DEBUG)
// Logs:
[DEBUG] Query: SELECT * FROM orders WHERE user_id = ?
[DEBUG] Query took 1.5s - TABLE SCAN detected
[DEBUG] Redis key: cart:user:12345 - MISS
[DEBUG] Redis reconnecting - pool size: 0/100
// Found it! Redis pool configuration too small!
\`\`\`

**After Fix:**
\`\`\`go
SetLevel(INFO)
// Back to normal
\`\`\`

**Production Impact:**
- **Debugging time**: 3 hours → 15 minutes (-95%)
- **Service downtime**: Avoided (changed level without restart)
- **CPU overhead**: Reduced from 80% → 5% by filtering debug logs
- **Log storage**: $10K/month → $1K/month (90% reduction)

**Without level filtering:**
- Would need to deploy new code to add/remove logs
- Each deploy = 5 minutes downtime
- Risk of breaking changes during critical period
- Can't get detailed logs during incident

**The Bottom Line:**
Log level filtering turns logging from a performance problem into a powerful debugging tool that you can control in real-time without risk.`,
	order: 2,
	translations: {
		ru: {
			title: 'Фильтр уровня логов',
			solutionCode: `package loggingx

import (
	"log"
	"sync/atomic"
)

type LogLevel int32

const (
	DEBUG LogLevel = 0
	INFO  LogLevel = 1
	WARN  LogLevel = 2
	ERROR LogLevel = 3
	FATAL LogLevel = 4
)

var currentLevel atomic.Int32

func SetLevel(level LogLevel) {
	atomic.StoreInt32(&currentLevel, int32(level)) // атомарная запись предотвращает data races когда несколько goroutines меняют уровень
}

func GetLevel() LogLevel {
	return LogLevel(atomic.LoadInt32(&currentLevel)) // атомарное чтение обеспечивает консистентное представление для всех goroutines
}

func Debug(format string, args ...any) {
	if DEBUG < GetLevel() { // ранний выход избегает дорогого форматирования строк когда уровень отключен
		return
	}
	log.Printf("[DEBUG] "+format, args...) // префиксим сообщение уровнем для grep фильтрации
}

func Info(format string, args ...any) {
	if INFO < GetLevel() { // пропускаем обработку когда текущий уровень фильтрует info сообщения
		return
	}
	log.Printf("[INFO] "+format, args...) // логируем информационные milestone события
}

func Warn(format string, args ...any) {
	if WARN < GetLevel() { // логируем warnings только когда уровень разрешает
		return
	}
	log.Printf("[WARN] "+format, args...) // сигнализируем некритичные проблемы требующие внимания
}

func Error(format string, args ...any) {
	if ERROR < GetLevel() { // уважаем уровень даже для ошибок (может быть включен только fatal)
		return
	}
	log.Printf("[ERROR] "+format, args...) // записываем критичные сбои
}

func Fatal(format string, args ...any) {
	log.Printf("[FATAL] "+format, args...) // всегда логируем fatal сообщения независимо от уровня
	panic("fatal error occurred")          // завершаем выполнение после логирования невосстановимой ошибки
}`,
			description: `Реализуйте production-grade logger с динамической фильтрацией по уровню для контроля verbosity в runtime.

**Требования:**
1. **LogLevel**: Определить уровни (DEBUG=0, INFO=1, WARN=2, ERROR=3, FATAL=4)
2. **SetLevel**: Установить минимальный уровень логирования (thread-safe через atomic)
3. **GetLevel**: Получить текущий минимальный уровень
4. **Debug, Info, Warn, Error, Fatal**: Функции логирования учитывающие минимальный уровень

**Паттерн уровней:**
\`\`\`go
// Development: видеть всё
SetLevel(DEBUG)
Debug("Variable value: %v", x)  // Logged
Info("Request started")          // Logged

// Production: только важное
SetLevel(WARN)
Debug("Variable value: %v", x)  // ПРОПУЩЕНО
Info("Request started")          // ПРОПУЩЕНО
Warn("Deprecated API used")      // Logged
\`\`\`

**Почему фильтрация уровней:**
- **Производительность**: Пропустить дорогой debug logging в production
- **Signal-to-Noise**: Фильтровать verbose логи чтобы видеть важное
- **Динамический контроль**: Менять уровень без перезапуска

**Ограничения:**
- Использовать \`atomic.Int32\` для thread-safe хранения уровня
- Early return если уровень сообщения < минимального
- Формат: \`[LEVEL] message\`
- Использовать \`log.Printf\`
- Fatal должен вызывать \`panic()\` после логирования`,
			hint1: `Используйте atomic.StoreInt32(&currentLevel, int32(level)) в SetLevel и atomic.LoadInt32(&currentLevel) в GetLevel.`,
			hint2: `В каждой logging функции проверьте if level < GetLevel() и делайте ранний return. Используйте log.Printf("[LEVEL] "+format, args...).`,
			whyItMatters: `Фильтрация по уровню критична для production систем - контроль verbosity и производительности без изменения кода.

**Почему важно:**

**1. Влияние на производительность**
Debug logging может значительно замедлить ваше приложение:
\`\`\`go
// Дорогая операция
Debug("User data: %+v, Orders: %+v, History: %+v",
    user,           // 50 полей
    orders,         // массив из 100 элементов
    history)        // 1000 записей

// Без фильтрации уровня:
// - Go форматирует всю строку (дорого)
// - Даже если вам этот лог не нужен!

// С фильтрацией (production level = WARN):
if DEBUG < GetLevel() {
    return // Выход перед форматированием - стоит ~2 наносекунды
}
// Форматирование строки пропущено - экономия миллисекунд на запрос
\`\`\`

**Real числа из production:**
- 10,000 запросов/секунду
- 50 debug логов на запрос
- Каждый debug format занимает 100μs
- **Без фильтрации**: 50 секунд CPU тратится в секунду (невозможно!)
- **С фильтрацией**: 1μs overhead (ранние returns)
- **Экономия**: 99.998% снижение CPU

**2. Signal-to-Noise Ratio**
В production слишком много логов хоронят важную информацию:
\`\`\`
Development (DEBUG level):
[DEBUG] Parsing request headers
[DEBUG] Validating user token
[DEBUG] Querying user table
[DEBUG] User found: {...}
[DEBUG] Checking permissions
[DEBUG] Permission granted
[INFO] Request completed

Production (INFO level):
[INFO] Request completed

Production во время incident (WARN level):
[WARN] Slow query: 2.5s
[ERROR] Database connection failed
[FATAL] Service unavailable
\`\`\`

Видите разницу? В production вы видите только то, что важно!

**3. Динамический контроль уровня**
Меняйте уровень логов без перезапуска:
\`\`\`go
// HTTP endpoint для изменения уровня
func HandleLogLevel(w http.ResponseWriter, r *http.Request) {
    level := r.URL.Query().Get("level")
    switch level {
    case "debug":
        SetLevel(DEBUG)
    case "info":
        SetLevel(INFO)
    case "warn":
        SetLevel(WARN)
    case "error":
        SetLevel(ERROR)
    }

    fmt.Fprintf(w, "Log level set to %s", level)
}

// Во время production incident:
// 1. Клиент сообщает о баге
// 2. curl http://api/admin/log-level?level=debug
// 3. Детальные логи начинают появляться немедленно
// 4. Находите root cause
// 5. curl http://api/admin/log-level?level=info
// 6. Возврат к нормальной работе

// Перезапуск не нужен! Сервис никогда не падал!
\`\`\`

**4. Конфигурация в зависимости от окружения**
\`\`\`go
func init() {
    env := os.Getenv("ENVIRONMENT")
    switch env {
    case "development":
        SetLevel(DEBUG) // Видеть все во время разработки
    case "staging":
        SetLevel(INFO)  // Важные события в staging
    case "production":
        SetLevel(WARN)  // Только warnings и errors в prod
    }
}
\`\`\`

**5. Почему Thread-Safe Atomic операции?**
\`\`\`go
// НЕПРАВИЛЬНО: Data race!
var level LogLevel
func SetLevel(l LogLevel) {
    level = l // Несколько goroutines пишут = undefined behavior
}

// ПРАВИЛЬНО: Atomic операции
var currentLevel atomic.Int32
func SetLevel(level LogLevel) {
    atomic.StoreInt32(&currentLevel, int32(level)) // Безопасно из любой goroutine
}

// Почему это важно:
// Goroutine 1: SetLevel(DEBUG) при обработке admin запроса
// Goroutine 2: GetLevel() при обработке customer запроса
// Goroutine 3: GetLevel() при выполнении background job
// Все три одновременно - atomic обеспечивает отсутствие corruption
\`\`\`

**6. Real-World пример**

**Сценарий:** E-commerce сайт во время Black Friday

**Утро (нормальный трафик):**
\`\`\`go
SetLevel(INFO)
// Логи:
[INFO] Order 1001 created
[INFO] Payment processed
[INFO] Order 1002 created
// 1000 строк логов в минуту
\`\`\`

**Полдень (спайк трафика - сайт замедляется):**
\`\`\`go
SetLevel(WARN)
// Логи:
[WARN] Database query slow: 1.2s
[WARN] Redis connection pool exhausted
[ERROR] Payment gateway timeout
// 50 строк логов в минуту - только проблемы!
\`\`\`

**Расследование (нужны детали):**
\`\`\`go
SetLevel(DEBUG)
// Логи:
[DEBUG] Query: SELECT * FROM orders WHERE user_id = ?
[DEBUG] Query took 1.5s - TABLE SCAN detected
[DEBUG] Redis key: cart:user:12345 - MISS
[DEBUG] Redis reconnecting - pool size: 0/100
// Нашли! Redis pool конфигурация слишком маленькая!
\`\`\`

**После исправления:**
\`\`\`go
SetLevel(INFO)
// Обратно к норме
\`\`\`

**Production Impact:**
- **Время debugging**: 3 часа → 15 минут (-95%)
- **Downtime сервиса**: Избежали (изменили уровень без перезапуска)
- **CPU overhead**: Снижен с 80% → 5% фильтрацией debug логов
- **Log storage**: $10K/месяц → $1K/месяц (90% снижение)

**Без фильтрации уровней:**
- Нужно было бы деплоить новый код для добавления/удаления логов
- Каждый деплой = 5 минут downtime
- Риск breaking changes во время критического периода
- Невозможно получить детальные логи во время incident

**Итог:**
Фильтрация по уровню превращает логирование из проблемы производительности в мощный инструмент отладки, который вы можете контролировать в реальном времени без риска.`
		},
		uz: {
			title: `Log daraja filtri`,
			solutionCode: `package loggingx

import (
	"log"
	"sync/atomic"
)

type LogLevel int32

const (
	DEBUG LogLevel = 0
	INFO  LogLevel = 1
	WARN  LogLevel = 2
	ERROR LogLevel = 3
	FATAL LogLevel = 4
)

var currentLevel atomic.Int32

func SetLevel(level LogLevel) {
	atomic.StoreInt32(&currentLevel, int32(level)) // atomik yozish bir nechta goroutinelar daraja o'zgartirganda data racelarni oldini oladi
}

func GetLevel() LogLevel {
	return LogLevel(atomic.LoadInt32(&currentLevel)) // atomik o'qish barcha goroutinelar uchun izchil ko'rinishni ta'minlaydi
}

func Debug(format string, args ...any) {
	if DEBUG < GetLevel() { // erta chiqish daraja o'chirilganda qimmat satr formatlashdan qochadi
		return
	}
	log.Printf("[DEBUG] "+format, args...) // grep filtrlash uchun xabarni daraja bilan prefikslaymiz
}

func Info(format string, args ...any) {
	if INFO < GetLevel() { // joriy daraja info xabarlarini filtrlasa qayta ishlashni o'tkazib yuboramiz
		return
	}
	log.Printf("[INFO] "+format, args...) // ma'lumot milestone hodisalarini log qilamiz
}

func Warn(format string, args ...any) {
	if WARN < GetLevel() { // daraja ruxsat bergandagina warninglarni log qilamiz
		return
	}
	log.Printf("[WARN] "+format, args...) // e'tibor talab qiladigan muhim bo'lmagan muammolarni signallaymiz
}

func Error(format string, args ...any) {
	if ERROR < GetLevel() { // xatolar uchun ham darajaga rioya qilamiz (faqat fatal yoqilgan bo'lishi mumkin)
		return
	}
	log.Printf("[ERROR] "+format, args...) // kritik nosozliklarni qayd qilamiz
}

func Fatal(format string, args ...any) {
	log.Printf("[FATAL] "+format, args...) // darajadan qat'i nazar har doim fatal xabarlarni log qilamiz
	panic("fatal error occurred")          // tiklab bo'lmaydigan xatoni log qilgandan keyin bajarishni to'xtatamiz
}`,
			description: `Runtime da verbosity ni boshqarish uchun dinamik daraja filtrlash bilan production-grade loggerni amalga oshiring.

**Talablar:**
1. **LogLevel**: Darajalarni aniqlash (DEBUG=0, INFO=1, WARN=2, ERROR=3, FATAL=4)
2. **SetLevel**: Minimal log darajasini o'rnatish (atomic orqali thread-safe)
3. **GetLevel**: Joriy minimal darajani olish
4. **Debug, Info, Warn, Error, Fatal**: Minimal darajaga rioya qiladigan logging funksiyalari

**Daraja Pattern:**
\`\`\`go
// Development: hammasini ko'rish
SetLevel(DEBUG)
Debug("Variable value: %v", x)  // Logged
Info("Request started")          // Logged

// Production: faqat muhimi
SetLevel(WARN)
Debug("Variable value: %v", x)  // O'TKAZIB YUBORILDI
Info("Request started")          // O'TKAZIB YUBORILDI
Warn("Deprecated API used")      // Logged
\`\`\`

**Nima uchun daraja filtrlash:**
- **Samaradorlik**: Production da qimmat debug logging ni o'tkazib yuboring
- **Signal-to-Noise**: Muhimni ko'rish uchun verbose loglarni filtrlang
- **Dinamik boshqaruv**: Qayta ishga tushirmasdan darajani o'zgartiring

**Cheklovlar:**
- Darajani thread-safe saqlash uchun \`atomic.Int32\` dan foydalaning
- Agar xabar darajasi < minimal daraja bo'lsa erta qaytish
- Format: \`[LEVEL] message\`
- \`log.Printf\` dan foydalaning
- Fatal logging dan keyin \`panic()\` ni chaqirishi kerak`,
			hint1: `SetLevel da atomic.StoreInt32(&currentLevel, int32(level)) va GetLevel da atomic.LoadInt32(&currentLevel) dan foydalaning.`,
			hint2: `Har bir logging funksiyasida if level < GetLevel() ni tekshiring va erta return qiling. log.Printf("[LEVEL] "+format, args...) dan foydalaning.`,
			whyItMatters: `Daraja filtrlash production tizimlar uchun muhim - kod o'zgarishisiz verbosity va samaradorlikni boshqarish.

**Nima uchun bu muhim:**

**1. Samaradorlikka ta'siri**
Debug logging ilovangizni sezilarli darajada sekinlashtirishi mumkin:
\`\`\`go
// Qimmat operatsiya
Debug("User data: %+v, Orders: %+v, History: %+v",
    user,           // 50 ta maydon
    orders,         // 100 ta element massiv
    history)        // 1000 ta yozuv

// Daraja filtrlashsiz:
// - Go butun satrni formatlaydi (qimmat)
// - Sizga bu log kerak bo'lmasa ham!

// Daraja filtrlash bilan (production darajasi = WARN):
if DEBUG < GetLevel() {
    return // Formatlashdan oldin chiqish - ~2 nanosekund
}
// Satr formatlash o'tkazib yuborildi - har bir so'rov uchun millisekund tejaldi
\`\`\`

**Production dan haqiqiy raqamlar:**
- 10,000 so'rov/soniya
- So'rov uchun 50 debug log
- Har bir debug format 100μs oladi
- **Filtrlashsiz**: sekundiga 50 soniya CPU isrof qilinadi (imkonsiz!)
- **Filtrlash bilan**: 1μs overhead (erta qaytishlar)
- **Tejash**: 99.998% CPU kamayish

**2. Signal-to-Noise nisbati**
Production da juda ko'p loglar muhim ma'lumotni ko'madi:
\`\`\`
Development (DEBUG darajasi):
[DEBUG] Parsing request headers
[DEBUG] Validating user token
[DEBUG] Querying user table
[DEBUG] User found: {...}
[DEBUG] Checking permissions
[DEBUG] Permission granted
[INFO] Request completed

Production (INFO darajasi):
[INFO] Request completed

Production hodisa paytida (WARN darajasi):
[WARN] Slow query: 2.5s
[ERROR] Database connection failed
[FATAL] Service unavailable
\`\`\`

Farqni ko'rasizmi? Production da faqat muhimini ko'rasiz!

**3. Dinamik daraja boshqaruvi**
Qayta ishga tushirmasdan log darajasini o'zgartiring:
\`\`\`go
// Darajani o'zgartirish uchun HTTP endpoint
func HandleLogLevel(w http.ResponseWriter, r *http.Request) {
    level := r.URL.Query().Get("level")
    switch level {
    case "debug":
        SetLevel(DEBUG)
    case "info":
        SetLevel(INFO)
    case "warn":
        SetLevel(WARN)
    case "error":
        SetLevel(ERROR)
    }

    fmt.Fprintf(w, "Log darajasi %s ga o'rnatildi", level)
}

// Production hodisasi paytida:
// 1. Mijoz bug haqida xabar beradi
// 2. curl http://api/admin/log-level?level=debug
// 3. Batafsil loglar darhol paydo bo'la boshlaydi
// 4. Asosiy sababni toping
// 5. curl http://api/admin/log-level?level=info
// 6. Oddiy ishlashga qaytish

// Qayta ishga tushirish kerak emas! Xizmat hech qachon to'xtamadi!
\`\`\`

**4. Muhitga xos konfiguratsiya**
\`\`\`go
func init() {
    env := os.Getenv("ENVIRONMENT")
    switch env {
    case "development":
        SetLevel(DEBUG) // Development paytida hamma narsani ko'rish
    case "staging":
        SetLevel(INFO)  // Staging da muhim hodisalar
    case "production":
        SetLevel(WARN)  // Prod da faqat ogohlantirishlar va xatolar
    }
}
\`\`\`

**5. Nima uchun Thread-Safe Atomic operatsiyalar?**
\`\`\`go
// NOTO'G'RI: Data race!
var level LogLevel
func SetLevel(l LogLevel) {
    level = l // Bir nechta goroutinlar yozadi = aniqlanmagan xatti-harakat
}

// TO'G'RI: Atomic operatsiyalar
var currentLevel atomic.Int32
func SetLevel(level LogLevel) {
    atomic.StoreInt32(&currentLevel, int32(level)) // Har qanday goroutinedan xavfsiz
}

// Nima uchun bu muhim:
// Goroutine 1: admin so'rovini qayta ishlayotganda SetLevel(DEBUG)
// Goroutine 2: mijoz so'rovini qayta ishlayotganda GetLevel()
// Goroutine 3: background ishni bajarayotganda GetLevel()
// Uchalasi bir vaqtning o'zida - atomic buzilmaslikni ta'minlaydi
\`\`\`

**6. Real-World misol**

**Stsenariy:** Black Friday paytida E-commerce sayti

**Ertalab (oddiy trafik):**
\`\`\`go
SetLevel(INFO)
// Loglar:
[INFO] Order 1001 created
[INFO] Payment processed
[INFO] Order 1002 created
// Daqiqasiga 1000 log qatori
\`\`\`

**Tushlik (trafik portlashi - sayt sekinlashmoqda):**
\`\`\`go
SetLevel(WARN)
// Loglar:
[WARN] Database query slow: 1.2s
[WARN] Redis connection pool exhausted
[ERROR] Payment gateway timeout
// Daqiqasiga 50 log qatori - faqat muammolar!
\`\`\`

**Tekshiruv (tafsilotlar kerak):**
\`\`\`go
SetLevel(DEBUG)
// Loglar:
[DEBUG] Query: SELECT * FROM orders WHERE user_id = ?
[DEBUG] Query took 1.5s - TABLE SCAN detected
[DEBUG] Redis key: cart:user:12345 - MISS
[DEBUG] Redis reconnecting - pool size: 0/100
// Topildi! Redis pool konfiguratsiyasi juda kichik!
\`\`\`

**Tuzatishdan keyin:**
\`\`\`go
SetLevel(INFO)
// Oddiy holatga qaytish
\`\`\`

**Production ta'siri:**
- **Debugging vaqti**: 3 soat → 15 daqiqa (-95%)
- **Xizmat to'xtashi**: Oldini olindi (restart qilmasdan darajani o'zgartirdi)
- **CPU overhead**: 80% → 5% (debug loglarni filtrlash orqali)
- **Log storage**: oyiga $10K → $1K (90% kamayish)

**Daraja filtrlashsiz:**
- Loglarni qo'shish/olib tashlash uchun yangi kodni deploy qilish kerak bo'lardi
- Har bir deploy = 5 daqiqa to'xtash
- Kritik davr paytida o'zgarishlarni buzish xavfi
- Hodisa paytida batafsil loglarni ololmaysiz

**Xulosa:**
Daraja filtrlash logging ni samaradorlik muammosidan real vaqtda xavfsiz boshqarish mumkin bo'lgan kuchli debugging vositasiga aylantiradi.`
		}
	}
};

export default task;
