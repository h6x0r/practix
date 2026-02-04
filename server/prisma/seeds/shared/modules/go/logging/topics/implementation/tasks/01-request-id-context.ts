import { Task } from "../../../../types";

export const task: Task = {
  slug: "go-logging-request-id",
  title: "Context-Aware Logging with Request ID",
  difficulty: "easy",
  tags: ["go", "logging", "context", "observability"],
  estimatedTime: "20m",
  isPremium: false,
  youtubeUrl: "",
  description: `Implement context-aware logging functions that track request IDs across your application.

**Requirements:**
1. **WithRequestID**: Store request ID in context using \`context.WithValue\`
2. **RequestID**: Extract request ID from context with type assertion
3. **Logf**: Log messages with automatic request ID prefix when present

**Context Pattern:**
\`\`\`go
// Store request ID
ctx := WithRequestID(context.Background(), "req-123")

// Log with automatic prefixing
Logf(ctx, "Processing order")
// Output: [rid=req-123] Processing order

// Extract request ID
rid := RequestID(ctx)  // Returns "req-123"
\`\`\`

**Key Concepts:**
- Use custom type for context keys to avoid collisions
- Handle nil context gracefully (fallback to Background)
- Type assertion with comma-ok idiom for safe extraction

**Example Usage:**
\`\`\`go
func HandleRequest(w http.ResponseWriter, r *http.Request) {
    rid := r.Header.Get("X-Request-ID")
    if rid == "" {
        rid = generateRequestID()
    }

    ctx := WithRequestID(r.Context(), rid)

    Logf(ctx, "Request started: %s %s", r.Method, r.URL.Path)
    // Output: [rid=abc-123] Request started: GET /api/users

    processRequest(ctx, r)

    Logf(ctx, "Request completed")
    // Output: [rid=abc-123] Request completed
}
\`\`\`

**Constraints:**
- Context key must be custom type (\`type CtxKey string\`)
- Handle nil context by returning \`context.Background()\`
- Empty request ID should not add prefix
- Use \`log.Printf\` for output`,
  initialCode: `package loggingx

import (
	"context"
	"log"
)

type CtxKey string

const KeyRequestID CtxKey = "rid"

// TODO: Implement WithRequestID
// Store request ID in context
// Handle nil context by creating Background
// Return context unchanged if rid is empty
func WithRequestID(ctx context.Context, rid string) context.Context {
	// TODO: Implement
}

// TODO: Implement RequestID
// Extract request ID from context
// Return empty string for nil context or missing value
// Use type assertion with comma-ok idiom
func RequestID(ctx context.Context) string {
	return "" // TODO: Implement
}

// TODO: Implement Logf
// Log message with request ID prefix if present
// Format: [rid=<id>] <message>
// Use log.Printf for output
func Logf(ctx context.Context, format string, args ...interface{}) {
	// TODO: Implement
}`,
  solutionCode: `package loggingx

import (
	"context"
	"log"
)

type CtxKey string

const KeyRequestID CtxKey = "rid"

func WithRequestID(ctx context.Context, rid string) context.Context {
	if ctx == nil { // fall back to background context when nil supplied
		ctx = context.Background()
	}
	if rid == "" { // skip storing empty request id
		return ctx
	}
	return context.WithValue(ctx, KeyRequestID, rid) // store request id inside context for downstream readers
}

func RequestID(ctx context.Context) string {
	if ctx == nil { // nil context never carries values
		return ""
	}
	val, _ := ctx.Value(KeyRequestID).(string) // attempt to extract stored request id
	return val                                 // return identifier or empty string when missing
}

func Logf(ctx context.Context, format string, args ...interface{}) {
	rid := RequestID(ctx) // fetch request identifier if present
	if rid != "" {        // include request id prefix when available
		log.Printf("[rid=%s] "+format, append([]interface{}{rid}, args...)...) // prefix message with rid before formatting
		return
	}
	log.Printf(format, args...) // log without prefix when request id absent
}`,
  testCode: `package loggingx

import (
	"context"
	"testing"
)

func Test1(t *testing.T) {
	ctx := WithRequestID(context.Background(), "req-123")
	rid := RequestID(ctx)
	if rid != "req-123" {
		t.Errorf("expected req-123, got %s", rid)
	}
}

func Test2(t *testing.T) {
	ctx := WithRequestID(nil, "req-456")
	rid := RequestID(ctx)
	if rid != "req-456" {
		t.Errorf("expected req-456, got %s", rid)
	}
}

func Test3(t *testing.T) {
	rid := RequestID(nil)
	if rid != "" {
		t.Errorf("expected empty string for nil context, got %s", rid)
	}
}

func Test4(t *testing.T) {
	ctx := WithRequestID(context.Background(), "")
	rid := RequestID(ctx)
	if rid != "" {
		t.Errorf("expected empty string for empty rid, got %s", rid)
	}
}

func Test5(t *testing.T) {
	ctx := context.Background()
	rid := RequestID(ctx)
	if rid != "" {
		t.Errorf("expected empty string for context without rid, got %s", rid)
	}
}

func Test6(t *testing.T) {
	ctx := WithRequestID(context.Background(), "first")
	ctx = WithRequestID(ctx, "second")
	rid := RequestID(ctx)
	if rid != "second" {
		t.Errorf("expected second, got %s", rid)
	}
}

func Test7(t *testing.T) {
	Logf(context.Background(), "test message")
}

func Test8(t *testing.T) {
	ctx := WithRequestID(context.Background(), "log-test")
	Logf(ctx, "test message with %s", "format")
}

func Test9(t *testing.T) {
	Logf(nil, "test nil context")
}

func Test10(t *testing.T) {
	ctx := WithRequestID(context.Background(), "test-rid")
	Logf(ctx, "message %d %s", 42, "args")
	rid := RequestID(ctx)
	if rid != "test-rid" {
		t.Errorf("expected test-rid, got %s", rid)
	}
}
`,
  hint1: `Use context.WithValue(ctx, KeyRequestID, rid) to store the request ID.`,
  hint2: `In RequestID, use ctx.Value(KeyRequestID).(string) with comma-ok for safe extraction.`,
  whyItMatters: `Request ID tracking is essential for distributed system observability and debugging production issues.

**Why This Matters:**

**1. Request Tracing**
In microservices, a single user request flows through multiple services. Request IDs let you trace the entire flow:
\`\`\`go
// API Gateway
ctx := WithRequestID(ctx, "req-abc-123")

// Service A
Logf(ctx, "Fetching user data")           // [rid=req-abc-123] Fetching user data

// Service B
Logf(ctx, "Calculating recommendations")   // [rid=req-abc-123] Calculating recommendations

// Service C
Logf(ctx, "Generating response")          // [rid=req-abc-123] Generating response
\`\`\`

All logs share the same request ID - you can grep logs across services to see the full story!

**2. Production Debugging**
When users report bugs, they often include request IDs from error pages:
\`\`\`bash
# User reports: "Error with request req-abc-123"

# Find ALL logs for that request across ALL services
grep "rid=req-abc-123" *.log

# You now see:
[rid=req-abc-123] User authentication succeeded
[rid=req-abc-123] Database query took 2.3s (SLOW!)
[rid=req-abc-123] External API timeout
[rid=req-abc-123] Error: payment processing failed
\`\`\`

Without request IDs, you'd have thousands of logs with no way to connect them.

**3. Performance Analysis**
\`\`\`go
Logf(ctx, "Query started")
result := db.Query(ctx, "SELECT...")
Logf(ctx, "Query completed in %v", time.Since(start))

// Logs:
// [rid=req-1] Query started
// [rid=req-1] Query completed in 1.2s
// [rid=req-2] Query started
// [rid=req-2] Query completed in 45ms
\`\`\`

Easy to identify which specific requests are slow!

**4. Context Propagation Pattern**
\`\`\`go
func HandleAPI(w http.ResponseWriter, r *http.Request) {
    ctx := WithRequestID(r.Context(), extractRequestID(r))

    // Pass context to all downstream functions
    user, err := fetchUser(ctx, userID)      // Logs with request ID
    orders := getOrders(ctx, user.ID)        // Logs with request ID
    sendEmail(ctx, user.Email, orders)       // Logs with request ID
}

// Every function gets automatic request ID logging
func fetchUser(ctx context.Context, id string) (*User, error) {
    Logf(ctx, "Fetching user %s", id)  // Automatic [rid=...] prefix
    // ...
}
\`\`\`

**5. Why Custom Context Key Type?**
\`\`\`go
// BAD - string keys can collide
ctx = context.WithValue(ctx, "rid", "123")
ctx = context.WithValue(ctx, "rid", "456")  // Different package overwrites!

// GOOD - custom type prevents collisions
type CtxKey string
const KeyRequestID CtxKey = "rid"  // Unique type = unique key
\`\`\`

**Real-World Example:**
A payment processing company tracks requests from mobile app → API → payment service → bank API:
- User makes purchase (app generates request ID)
- Request ID flows through all services
- Bank declines card
- Support can trace entire flow in seconds using request ID
- Found issue: slow database query in user service was timing out
- Fixed one slow query, all payments working

**Production Impact:**
- Debugging time: Hours → Minutes
- Mean time to resolution: -75%
- Customer satisfaction: +40%

Without request IDs, they had to correlate logs manually across 12 services using timestamps - often impossible for high-traffic periods.`,
  order: 0,
  translations: {
    ru: {
      title: "Логирование с Request ID",
      solutionCode: `package loggingx

import (
	"context"
	"log"
)

type CtxKey string

const KeyRequestID CtxKey = "rid"

func WithRequestID(ctx context.Context, rid string) context.Context {
	if ctx == nil { // fallback к background context когда передан nil
		ctx = context.Background()
	}
	if rid == "" { // пропускаем сохранение пустого request id
		return ctx
	}
	return context.WithValue(ctx, KeyRequestID, rid) // сохраняем request id в контексте для читателей ниже по цепочке
}

func RequestID(ctx context.Context) string {
	if ctx == nil { // nil контекст никогда не несёт значений
		return ""
	}
	val, _ := ctx.Value(KeyRequestID).(string) // пытаемся извлечь сохранённый request id
	return val                                 // возвращаем идентификатор или пустую строку если отсутствует
}

func Logf(ctx context.Context, format string, args ...interface{}) {
	rid := RequestID(ctx) // получаем идентификатор запроса если есть
	if rid != "" {        // включаем префикс request id когда доступен
		log.Printf("[rid=%s] "+format, append([]interface{}{rid}, args...)...) // префиксим сообщение rid перед форматированием
		return
	}
	log.Printf(format, args...) // логируем без префикса когда request id отсутствует
}`,
      description: `Реализуйте функции контекстного логирования, отслеживающие request ID по всему приложению.

**Требования:**
1. **WithRequestID**: Сохранить request ID в контексте через \`context.WithValue\`
2. **RequestID**: Извлечь request ID из контекста с type assertion
3. **Logf**: Логировать сообщения с автоматическим префиксом request ID

**Паттерн Context:**
\`\`\`go
// Сохранить request ID
ctx := WithRequestID(context.Background(), "req-123")

// Логировать с автоматическим префиксом
Logf(ctx, "Processing order")
// Вывод: [rid=req-123] Processing order

// Извлечь request ID
rid := RequestID(ctx)  // Возвращает "req-123"
\`\`\`

**Ключевые концепции:**
- Используйте custom type для ключей контекста (избежание коллизий)
- Обрабатывайте nil context gracefully (fallback to Background)
- Type assertion с comma-ok для безопасного извлечения

**Ограничения:**
- Ключ контекста должен быть custom type (\`type CtxKey string\`)
- Обрабатывайте nil context через \`context.Background()\`
- Пустой request ID не должен добавлять префикс
- Используйте \`log.Printf\` для вывода`,
      hint1: `Используйте context.WithValue(ctx, KeyRequestID, rid) для сохранения request ID.`,
      hint2: `В RequestID используйте ctx.Value(KeyRequestID).(string) с comma-ok для безопасности.`,
      whyItMatters: `Request ID tracking критичен для observability распределенных систем и debugging production проблем.

**Почему важно:**

**1. Трассировка запросов**
В микросервисах один запрос пользователя проходит через множество сервисов. Request ID позволяет отследить весь путь:
\`\`\`go
// API Gateway
ctx := WithRequestID(ctx, "req-abc-123")

// Service A
Logf(ctx, "Fetching user data")           // [rid=req-abc-123] Fetching user data

// Service B
Logf(ctx, "Calculating recommendations")   // [rid=req-abc-123] Calculating recommendations

// Service C
Logf(ctx, "Generating response")          // [rid=req-abc-123] Generating response
\`\`\`

Все логи имеют одинаковый request ID - вы можете использовать grep по логам всех сервисов чтобы увидеть полную историю!

**2. Production Debugging**
Когда пользователи сообщают о багах, они часто включают request ID со страницы ошибки:
\`\`\`bash
# Пользователь сообщает: "Error with request req-abc-123"

# Найти ВСЕ логи для этого запроса через ВСЕ сервисы
grep "rid=req-abc-123" *.log

# Теперь вы видите:
[rid=req-abc-123] User authentication succeeded
[rid=req-abc-123] Database query took 2.3s (SLOW!)
[rid=req-abc-123] External API timeout
[rid=req-abc-123] Error: payment processing failed
\`\`\`

Без request ID у вас были бы тысячи логов без способа связать их.

**3. Анализ производительности**
\`\`\`go
Logf(ctx, "Query started")
result := db.Query(ctx, "SELECT...")
Logf(ctx, "Query completed in %v", time.Since(start))

// Логи:
// [rid=req-1] Query started
// [rid=req-1] Query completed in 1.2s
// [rid=req-2] Query started
// [rid=req-2] Query completed in 45ms
\`\`\`

Легко идентифицировать, какие конкретные запросы медленные!

**4. Паттерн Context Propagation**
\`\`\`go
func HandleAPI(w http.ResponseWriter, r *http.Request) {
    ctx := WithRequestID(r.Context(), extractRequestID(r))

    // Передать context всем downstream функциям
    user, err := fetchUser(ctx, userID)      // Логирует с request ID
    orders := getOrders(ctx, user.ID)        // Логирует с request ID
    sendEmail(ctx, user.Email, orders)       // Логирует с request ID
}

// Каждая функция получает автоматическое логирование request ID
func fetchUser(ctx context.Context, id string) (*User, error) {
    Logf(ctx, "Fetching user %s", id)  // Автоматический префикс [rid=...]
    // ...
}
\`\`\`

**5. Почему Custom Context Key Type?**
\`\`\`go
// ПЛОХО - строковые ключи могут конфликтовать
ctx = context.WithValue(ctx, "rid", "123")
ctx = context.WithValue(ctx, "rid", "456")  // Другой пакет перезаписывает!

// ХОРОШО - custom type предотвращает коллизии
type CtxKey string
const KeyRequestID CtxKey = "rid"  // Уникальный тип = уникальный ключ
\`\`\`

**Продакшен пример:**
Компания по обработке платежей отслеживает запросы от мобильного приложения → API → сервис платежей → bank API:
- Пользователь совершает покупку (приложение генерирует request ID)
- Request ID проходит через все сервисы
- Банк отклоняет карту
- Поддержка может отследить весь путь за секунды используя request ID
- Найдена проблема: медленный database query в user service вызывал timeout
- Исправлен один медленный запрос, все платежи работают

**Production Impact:**
- Время debugging: Часы → Минуты
- Mean time to resolution: -75%
- Customer satisfaction: +40%

Без request ID им пришлось бы вручную сопоставлять логи по 12 сервисам используя timestamp - часто невозможно для периодов высокого трафика.`,
    },
    uz: {
      title: `Request ID bilan loglash`,
      solutionCode: `package loggingx

import (
	"context"
	"log"
)

type CtxKey string

const KeyRequestID CtxKey = "rid"

func WithRequestID(ctx context.Context, rid string) context.Context {
	if ctx == nil { // nil berilganda background kontekstga fallback
		ctx = context.Background()
	}
	if rid == "" { // bo'sh request id ni saqlashni o'tkazib yuboramiz
		return ctx
	}
	return context.WithValue(ctx, KeyRequestID, rid) // pastdagi o'quvchilar uchun kontekstda request id ni saqlaymiz
}

func RequestID(ctx context.Context) string {
	if ctx == nil { // nil kontekst hech qachon qiymat tashimaydi
		return ""
	}
	val, _ := ctx.Value(KeyRequestID).(string) // saqlangan request id ni chiqarishga urinamiz
	return val                                 // identifikator yoki mavjud bo'lmaganda bo'sh satr qaytaramiz
}

func Logf(ctx context.Context, format string, args ...interface{}) {
	rid := RequestID(ctx) // mavjud bo'lsa so'rov identifikatorini olamiz
	if rid != "" {        // mavjud bo'lganda request id prefiksini qo'shamiz
		log.Printf("[rid=%s] "+format, append([]interface{}{rid}, args...)...) // formatlashdan oldin xabarni rid bilan prefikslaymiz
		return
	}
	log.Printf(format, args...) // request id mavjud bo'lmaganda prefikssiz log qilamiz
}`,
      description: `Ilova bo'ylab request ID larni kuzatuvchi kontekstga oid logging funksiyalarini amalga oshiring.

**Talablar:**
1. **WithRequestID**: Request ID ni kontekstda \`context.WithValue\` orqali saqlash
2. **RequestID**: Request ID ni kontekstdan type assertion bilan chiqarish
3. **Logf**: Mavjud bo'lganda avtomatik request ID prefiksi bilan xabarlarni log qilish

**Context Pattern:**
\`\`\`go
// Request ID ni saqlash
ctx := WithRequestID(context.Background(), "req-123")

// Avtomatik prefiks bilan log qilish
Logf(ctx, "Processing order")
// Chiqish: [rid=req-123] Processing order

// Request ID ni chiqarish
rid := RequestID(ctx)  // "req-123" qaytaradi
\`\`\`

**Asosiy tushunchalar:**
- Kontekst kalitlari uchun custom type dan foydalaning (to'qnashuvlarni oldini olish)
- nil kontekstni gracefully qayta ishlang (Background ga fallback)
- Xavfsiz chiqarish uchun comma-ok idiomasi bilan type assertion

**Cheklovlar:**
- Kontekst kaliti custom type bo'lishi kerak (\`type CtxKey string\`)
- nil kontekstni \`context.Background()\` orqali qayta ishlang
- Bo'sh request ID prefiks qo'shmasligi kerak
- Chiqish uchun \`log.Printf\` dan foydalaning`,
      hint1: `Request ID ni saqlash uchun context.WithValue(ctx, KeyRequestID, rid) dan foydalaning.`,
      hint2: `RequestID da xavfsizlik uchun ctx.Value(KeyRequestID).(string) ni comma-ok bilan ishlating.`,
      whyItMatters: `Request ID kuzatish taqsimlangan tizimlar observability va production muammolarni debugging uchun muhimdir.

**Nima uchun bu muhim:**

**1. So'rovlarni kuzatish**
Mikroxizmatlarda bitta foydalanuvchi so'rovi bir nechta xizmatlar orqali o'tadi. Request ID butun yo'lni kuzatishga imkon beradi:
\`\`\`go
// API Gateway
ctx := WithRequestID(ctx, "req-abc-123")

// Service A
Logf(ctx, "Fetching user data")           // [rid=req-abc-123] Fetching user data

// Service B
Logf(ctx, "Calculating recommendations")   // [rid=req-abc-123] Calculating recommendations

// Service C
Logf(ctx, "Generating response")          // [rid=req-abc-123] Generating response
\`\`\`

Barcha loglar bir xil request ID ga ega - to'liq tarixni ko'rish uchun barcha xizmatlarning loglari bo'ylab grep qilishingiz mumkin!

**2. Production Debugging**
Foydalanuvchilar buglar haqida xabar berishganda, ular ko'pincha xato sahifasidan request ID ni qo'shadi:
\`\`\`bash
# Foydalanuvchi xabar beradi: "Error with request req-abc-123"

# BARCHA xizmatlar bo'ylab ushbu so'rov uchun BARCHA loglarni toping
grep "rid=req-abc-123" *.log

# Endi siz ko'rasiz:
[rid=req-abc-123] User authentication succeeded
[rid=req-abc-123] Database query took 2.3s (SLOW!)
[rid=req-abc-123] External API timeout
[rid=req-abc-123] Error: payment processing failed
\`\`\`

Request ID siz minglab loglar bo'lardi ularni bog'lash usuli yo'q.

**3. Ishlash tahlili**
\`\`\`go
Logf(ctx, "Query started")
result := db.Query(ctx, "SELECT...")
Logf(ctx, "Query completed in %v", time.Since(start))

// Loglar:
// [rid=req-1] Query started
// [rid=req-1] Query completed in 1.2s
// [rid=req-2] Query started
// [rid=req-2] Query completed in 45ms
\`\`\`

Qaysi aniq so'rovlar sekin ekanligini aniqlash oson!

**4. Context Propagation Pattern**
\`\`\`go
func HandleAPI(w http.ResponseWriter, r *http.Request) {
    ctx := WithRequestID(r.Context(), extractRequestID(r))

    // Barcha downstream funksiyalarga kontekstni o'tkazing
    user, err := fetchUser(ctx, userID)      // Request ID bilan log qiladi
    orders := getOrders(ctx, user.ID)        // Request ID bilan log qiladi
    sendEmail(ctx, user.Email, orders)       // Request ID bilan log qiladi
}

// Har bir funksiya avtomatik request ID logging oladi
func fetchUser(ctx context.Context, id string) (*User, error) {
    Logf(ctx, "Fetching user %s", id)  // Avtomatik [rid=...] prefiksi
    // ...
}
\`\`\`

**5. Nima uchun Custom Context Key Type?**
\`\`\`go
// YOMON - satr kalitlari to'qnashishi mumkin
ctx = context.WithValue(ctx, "rid", "123")
ctx = context.WithValue(ctx, "rid", "456")  // Boshqa paket ustiga yozadi!

// YAXSHI - custom type to'qnashuvlarni oldini oladi
type CtxKey string
const KeyRequestID CtxKey = "rid"  // Noyob tur = noyob kalit
\`\`\`

**Ishlab chiqarish misoli:**
To'lov qayta ishlash kompaniyasi so'rovlarni mobil ilova → API → to'lov xizmati → bank API orqali kuzatadi:
- Foydalanuvchi xarid qiladi (ilova request ID yaratadi)
- Request ID barcha xizmatlar orqali o'tadi
- Bank kartani rad etadi
- Support request ID yordamida butun oqimni soniyalarda kuzatishi mumkin
- Muammo topildi: user service da sekin database query timeout ga sabab bo'ldi
- Bitta sekin so'rov tuzatildi, barcha to'lovlar ishlaydi

**Production ta'siri:**
- Debugging vaqti: Soatlar → Daqiqalar
- O'rtacha hal qilish vaqti: -75%
- Mijoz qoniqishi: +40%

Request ID siz ularga timestamp yordamida 12 xizmat bo'ylab loglarni qo'lda bog'lashga to'g'ri kelardi - yuqori trafik davrlari uchun ko'pincha imkonsiz.`,
    },
  },
};

export default task;
