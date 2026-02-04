import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-logging-structured-fields',
	title: 'Structured Logging with Key-Value Fields',
	difficulty: 'medium',	tags: ['go', 'logging', 'structured-logging', 'observability'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement structured logging with key-value fields that accumulate through context for machine-parseable logs.

**Requirements:**
1. **WithFields**: Add structured fields to context, merging with existing fields
2. **LogKV**: Output logs in \`key=value\` format with sorted keys for deterministic output

**Structured Logging Pattern:**
\`\`\`go
ctx := context.Background()
ctx = WithRequestID(ctx, "req-123")
ctx = WithFields(ctx, map[string]string{
    "user_id": "42",
    "action":  "purchase",
})

ctx = WithFields(ctx, map[string]string{
    "product": "laptop",
    "price":   "1299",
})

LogKV(ctx, "Order created", map[string]string{
    "order_id": "ord-456",
})

// Output (sorted keys for deterministic output):
// Order created action=purchase order_id=ord-456 price=1299 product=laptop rid=req-123 user_id=42
\`\`\`

**Why Key-Value Format:**
- **Machine Parseable**: Log aggregators can index each field
- **Searchable**: Query logs like \`user_id=42 AND action=purchase\`
- **Structured**: No regex parsing needed

**Key Implementation Details:**
- Fields accumulate through context chain
- Later fields override earlier ones (same key)
- Sort keys alphabetically for deterministic output
- Request ID automatically added as \`rid=<value>\` field

**Example Production Usage:**
\`\`\`go
func ProcessPayment(ctx context.Context, payment Payment) error {
    ctx = WithFields(ctx, map[string]string{
        "user_id":    payment.UserID,
        "amount":     payment.Amount,
        "currency":   payment.Currency,
    })

    LogKV(ctx, "Payment started", nil)

    if err := validateCard(ctx, payment.Card); err != nil {
        LogKV(ctx, "Card validation failed", map[string]string{
            "error": err.Error(),
        })
        return err
    }

    result, err := chargeCard(ctx, payment)
    LogKV(ctx, "Payment completed", map[string]string{
        "transaction_id": result.ID,
        "status":        result.Status,
    })

    return err
}
\`\`\`

**Constraints:**
- Merge existing context fields with new fields
- Later fields override earlier fields for same key
- Include request ID as \`rid\` field if present
- Sort keys alphabetically before output
- Use \`strings.Builder\` for efficient string concatenation`,
	initialCode: `package loggingx

import (
	"context"
	"log"
	"sort"
	"strings"
)

const keyFields CtxKey = "fields"

// TODO: Implement WithFields
// 1. Create new map to hold merged fields
// 2. Copy existing fields from context (if interface{})
// 3. Copy new fields, overriding existing keys
// 4. Store merged map in context under keyFields key
func WithFields(ctx context.Context, fields map[string]string) context.Context {
	// TODO: Implement
}

// TODO: Implement LogKV
// 1. Create combined map
// 2. Copy fields from context
// 3. Copy extra fields (overrides)
// 4. Add request ID as "rid" field if present
// 5. Sort keys alphabetically
// 6. Build output string: "msg key1=val1 key2=val2 ..."
// 7. Use log.Print() to output
func LogKV(ctx context.Context, msg string, extra map[string]string) {
	// TODO: Implement
}`,
	solutionCode: `package loggingx

import (
	"context"
	"log"
	"sort"
	"strings"
)

const keyFields CtxKey = "fields"

func WithFields(ctx context.Context, fields map[string]string) context.Context {
	if ctx == nil { // default to background context when nil provided
		ctx = context.Background()
	}
	merged := make(map[string]string)                                 // allocate new map to avoid mutating caller state
	if existing, ok := ctx.Value(keyFields).(map[string]string); ok { // copy any fields already present
		for k, v := range existing {
			merged[k] = v // duplicate key-value pairs from original map
		}
	}
	for k, v := range fields { // apply overrides from provided map
		merged[k] = v // store latest value for each key
	}
	return context.WithValue(ctx, keyFields, merged) // return new context holding merged fields
}

func LogKV(ctx context.Context, msg string, extra map[string]string) {
	combined := make(map[string]string)                             // hold merged field set for logging
	if fields, ok := ctx.Value(keyFields).(map[string]string); ok { // incorporate context fields when available
		for k, v := range fields {
			combined[k] = v // copy stored field values
		}
	}
	for k, v := range extra { // merge explicit extra fields overriding context
		combined[k] = v
	}
	if rid := RequestID(ctx); rid != "" { // expose request id as ordinary key-value pair
		combined["rid"] = rid
	}
	keys := make([]string, 0, len(combined)) // collect keys to sort for deterministic output
	for k := range combined {
		keys = append(keys, k)
	}
	sort.Strings(keys)           // ensure deterministic ordering for log line
	builder := strings.Builder{} // accumulate final log message
	builder.WriteString(msg)     // start with descriptive message portion
	for _, k := range keys {     // append each key-value pair in sorted order
		builder.WriteString(" ")         // separate message and kv entries with space
		builder.WriteString(k)           // write key name
		builder.WriteString("=")         // add equals delimiter
		builder.WriteString(combined[k]) // write corresponding value
	}
	log.Print(builder.String()) // emit assembled log entry via standard logger
}`,
	testCode: `package loggingx

import (
	"context"
	"testing"
)

func Test1(t *testing.T) {
	ctx := WithFields(context.Background(), map[string]string{"key": "value"})
	LogKV(ctx, "test message", nil)
}

func Test2(t *testing.T) {
	ctx := WithFields(nil, map[string]string{"key": "value"})
	LogKV(ctx, "test nil context", nil)
}

func Test3(t *testing.T) {
	ctx := WithFields(context.Background(), nil)
	LogKV(ctx, "test nil fields", nil)
}

func Test4(t *testing.T) {
	ctx := WithFields(context.Background(), map[string]string{"a": "1", "b": "2"})
	ctx = WithFields(ctx, map[string]string{"c": "3"})
	LogKV(ctx, "accumulated fields", nil)
}

func Test5(t *testing.T) {
	ctx := WithFields(context.Background(), map[string]string{"key": "first"})
	ctx = WithFields(ctx, map[string]string{"key": "second"})
	LogKV(ctx, "override test", nil)
}

func Test6(t *testing.T) {
	ctx := context.Background()
	LogKV(ctx, "no fields", map[string]string{"extra": "value"})
}

func Test7(t *testing.T) {
	ctx := WithRequestID(context.Background(), "req-123")
	ctx = WithFields(ctx, map[string]string{"user": "42"})
	LogKV(ctx, "with rid", nil)
}

func Test8(t *testing.T) {
	LogKV(nil, "nil context", nil)
}

func Test9(t *testing.T) {
	ctx := WithFields(context.Background(), map[string]string{})
	LogKV(ctx, "empty fields", map[string]string{})
}

func Test10(t *testing.T) {
	ctx := WithFields(context.Background(), map[string]string{"z": "last", "a": "first", "m": "mid"})
	LogKV(ctx, "sorted keys", nil)
}
`,
			hint1: `In WithFields, create a new map, copy existing fields from context, then copy new fields.`,
			hint2: `In LogKV, collect all keys, sort them with sort.Strings(), then build output using strings.Builder.`,
			whyItMatters: `Structured logging is the foundation of modern observability platforms and enables powerful log analysis at scale.

**Why Structured Logging Matters:**

**1. Machine-Readable Logs**
\`\`\`go
// OLD: Unstructured logging
log.Print("User 42 purchased laptop for $1299")

// Problem: Need regex to extract user_id, product, price
// Fragile, slow, breaks if format changes

// NEW: Structured logging
LogKV(ctx, "Purchase completed", map[string]string{
    "user_id": "42",
    "product": "laptop",
    "price":   "1299",
})
// Output: Purchase completed price=1299 product=laptop rid=req-123 user_id=42

// Solution: Log aggregators automatically index each field
// Query: user_id=42 AND product=laptop AND price>1000
\`\`\`

**2. Powerful Queries in Production**
With log aggregators (Elasticsearch, Splunk, CloudWatch):
\`\`\`
# Find all failed payments for user 42
user_id=42 AND status=failed

# Find slow database queries (>1s)
component=database AND duration>1000

# Find all errors in last hour for specific request
rid=req-abc-123 AND level=error AND timestamp>now-1h
\`\`\`

Without structured logging, you'd need complex regex patterns that break easily.

**3. Context Accumulation Pattern**
\`\`\`go
func HandleRequest(ctx context.Context, r *http.Request) {
    // Add request-level fields
    ctx = WithFields(ctx, map[string]string{
        "method": r.Method,
        "path":   r.URL.Path,
    })

    user := authenticate(ctx, r)
    // Add user-level fields
    ctx = WithFields(ctx, map[string]string{
        "user_id":   user.ID,
        "user_role": user.Role,
    })

    processOrder(ctx, user)
}

func processOrder(ctx context.Context, user *User) {
    // Add order-level fields
    ctx = WithFields(ctx, map[string]string{
        "order_id": generateOrderID(),
    })

    // ALL logs now have method, path, user_id, user_role, order_id
    LogKV(ctx, "Order created", map[string]string{
        "total": "1299.00",
    })
    // Output: Order created method=POST order_id=ord-456 path=/api/orders
    //         rid=req-123 total=1299.00 user_id=42 user_role=premium
}
\`\`\`

Fields accumulate as context flows through your application!

**4. Real-World Production Example**

**Before Structured Logging:**
\`\`\`
ERROR: Payment failed
ERROR: Card declined
ERROR: Database timeout
\`\`\`
No context! Which user? Which payment? Which card?

**After Structured Logging:**
\`\`\`
Payment processing failed amount=1299 card_last4=4242 currency=USD
merchant_id=m-123 payment_id=pay-456 reason=insufficient_funds
rid=req-789 user_id=42

Card validation error card_last4=4242 card_type=visa error=expired
merchant_id=m-123 rid=req-790 user_id=43

Database query timeout query=SELECT component=payment-service
duration=5000 operation=charge_card rid=req-791 table=transactions
\`\`\`

Now you can:
- Find all payment failures: \`error=*\`
- Find user-specific issues: \`user_id=42\`
- Find slow queries: \`duration>3000\`
- Trace request flow: \`rid=req-789\`

**5. Why Sort Keys?**
\`\`\`go
// Unsorted output (random map iteration):
// msg user_id=42 amount=1299 product=laptop
// msg product=laptop user_id=42 amount=1299
// msg amount=1299 product=laptop user_id=42

// Sorted output (deterministic):
// msg amount=1299 product=laptop user_id=42
// msg amount=1299 product=laptop user_id=42
// msg amount=1299 product=laptop user_id=42
\`\`\`

Benefits:
- Consistent format for log parsing
- Easier to grep/search
- Better for testing (stable output)
- Humans can read logs more easily

**6. Production Impact at Scale**

E-commerce company with 10M daily requests:

**Before Structured Logging:**
- Debugging time: 4-6 hours per incident
- Manual log parsing with regex
- 50% of time spent finding relevant logs
- Frequent regex bugs breaking log parsing

**After Structured Logging:**
- Debugging time: 15-30 minutes per incident (-88%)
- Query logs like database: \`user_id=X AND error=Y\`
- Instant log filtering in Elasticsearch
- Zero regex maintenance

Cost savings: \$500K/year in engineering time

**7. Integration with Observability Tools**

Structured logs feed directly into:
- **Metrics**: Count errors by \`error_type\` field
- **Alerts**: Alert when \`status=failed AND payment_method=card\` > 5%
- **Dashboards**: Graph \`duration\` field over time per \`service\` field
- **Tracing**: Link logs by \`rid\` to distributed traces

All automatic - just add fields to logs!`,	order: 1,
	translations: {
		ru: {
			title: 'Структурированные поля',
			solutionCode: `package loggingx

import (
	"context"
	"log"
	"sort"
	"strings"
)

const keyFields CtxKey = "fields"

func WithFields(ctx context.Context, fields map[string]string) context.Context {
	if ctx == nil { // по умолчанию background context когда передан nil
		ctx = context.Background()
	}
	merged := make(map[string]string)                                 // выделяем новую map чтобы не мутировать состояние вызывающего
	if existing, ok := ctx.Value(keyFields).(map[string]string); ok { // копируем уже присутствующие поля
		for k, v := range existing {
			merged[k] = v // дублируем key-value пары из оригинальной map
		}
	}
	for k, v := range fields { // применяем overrides из переданной map
		merged[k] = v // сохраняем последнее значение для каждого ключа
	}
	return context.WithValue(ctx, keyFields, merged) // возвращаем новый контекст с merged полями
}

func LogKV(ctx context.Context, msg string, extra map[string]string) {
	combined := make(map[string]string)                             // держим merged набор полей для логирования
	if fields, ok := ctx.Value(keyFields).(map[string]string); ok { // включаем поля контекста когда доступны
		for k, v := range fields {
			combined[k] = v // копируем сохранённые значения полей
		}
	}
	for k, v := range extra { // merge explicit extra fields переопределяя контекст
		combined[k] = v
	}
	if rid := RequestID(ctx); rid != "" { // экспонируем request id как обычную key-value пару
		combined["rid"] = rid
	}
	keys := make([]string, 0, len(combined)) // собираем ключи для сортировки для детерминированного вывода
	for k := range combined {
		keys = append(keys, k)
	}
	sort.Strings(keys)           // обеспечиваем детерминированный порядок для строки лога
	builder := strings.Builder{} // накапливаем финальное сообщение лога
	builder.WriteString(msg)     // начинаем с описательной части сообщения
	for _, k := range keys {     // добавляем каждую key-value пару в отсортированном порядке
		builder.WriteString(" ")         // разделяем сообщение и kv записи пробелом
		builder.WriteString(k)           // пишем имя ключа
		builder.WriteString("=")         // добавляем разделитель equals
		builder.WriteString(combined[k]) // пишем соответствующее значение
	}
	log.Print(builder.String()) // выводим собранную запись лога через стандартный logger
}`,
			description: `Реализуйте structured logging с key-value полями, накапливающимися через context для machine-parseable логов.

**Требования:**
1. **WithFields**: Добавить structured fields в context с merge существующих
2. **LogKV**: Вывод логов в формате \`key=value\` с отсортированными ключами

**Паттерн Structured Logging:**
\`\`\`go
ctx := context.Background()
ctx = WithRequestID(ctx, "req-123")
ctx = WithFields(ctx, map[string]string{
    "user_id": "42",
    "action":  "purchase",
})

LogKV(ctx, "Order created", map[string]string{
    "order_id": "ord-456",
})

// Вывод (отсортированные ключи):
// Order created action=purchase order_id=ord-456 price=1299 rid=req-123 user_id=42
\`\`\`

**Почему Key-Value формат:**
- **Machine Parseable**: Лог-агрегаторы индексируют каждое поле
- **Searchable**: Запросы как \`user_id=42 AND action=purchase\`
- **Structured**: Не нужен regex парсинг

**Ограничения:**
- Merge существующих context fields с новыми
- Поздние fields override ранние для одного ключа
- Включать request ID как \`rid\` field если присутствует
- Сортировать ключи алфавитно перед выводом
- Использовать \`strings.Builder\` для эффективной конкатенации`,
			hint1: `В WithFields создайте новую map, скопируйте существующие поля из context, затем новые поля.`,
			hint2: `В LogKV соберите все ключи, отсортируйте через sort.Strings(), постройте вывод через strings.Builder.`,
			whyItMatters: `Structured logging - основа современных observability платформ и позволяет мощный анализ логов at scale.

**Почему важно:**

**1. Machine-Readable логи**
\`\`\`go
// СТАРЫЙ: Unstructured logging
log.Print("User 42 purchased laptop for $1299")

// Проблема: Нужен regex для извлечения user_id, product, price
// Хрупкий, медленный, ломается при изменении формата

// НОВЫЙ: Structured logging
LogKV(ctx, "Purchase completed", map[string]string{
    "user_id": "42",
    "product": "laptop",
    "price":   "1299",
})
// Вывод: Purchase completed price=1299 product=laptop rid=req-123 user_id=42

// Решение: Log aggregators автоматически индексируют каждое поле
// Запрос: user_id=42 AND product=laptop AND price>1000
\`\`\`

**2. Мощные запросы в Production**
С log aggregators (Elasticsearch, Splunk, CloudWatch):
\`\`\`
# Найти все неудавшиеся платежи для user 42
user_id=42 AND status=failed

# Найти медленные database queries (>1s)
component=database AND duration>1000

# Найти все ошибки за последний час для конкретного запроса
rid=req-abc-123 AND level=error AND timestamp>now-1h
\`\`\`

Без structured logging нужны сложные regex паттерны которые легко ломаются.

**3. Context Accumulation Pattern**
\`\`\`go
func HandleRequest(ctx context.Context, r *http.Request) {
    // Добавить поля уровня запроса
    ctx = WithFields(ctx, map[string]string{
        "method": r.Method,
        "path":   r.URL.Path,
    })

    user := authenticate(ctx, r)
    // Добавить поля уровня пользователя
    ctx = WithFields(ctx, map[string]string{
        "user_id":   user.ID,
        "user_role": user.Role,
    })

    processOrder(ctx, user)
}

func processOrder(ctx context.Context, user *User) {
    // Добавить поля уровня заказа
    ctx = WithFields(ctx, map[string]string{
        "order_id": generateOrderID(),
    })

    // ВСЕ логи теперь имеют method, path, user_id, user_role, order_id
    LogKV(ctx, "Order created", map[string]string{
        "total": "1299.00",
    })
    // Вывод: Order created method=POST order_id=ord-456 path=/api/orders
    //         rid=req-123 total=1299.00 user_id=42 user_role=premium
}
\`\`\`

Поля накапливаются по мере прохождения context через приложение!

**4. Продакшен пример**

**До Structured Logging:**
\`\`\`
ERROR: Payment failed
ERROR: Card declined
ERROR: Database timeout
\`\`\`
Нет контекста! Какой пользователь? Какой платеж? Какая карта?

**После Structured Logging:**
\`\`\`
Payment processing failed amount=1299 card_last4=4242 currency=USD
merchant_id=m-123 payment_id=pay-456 reason=insufficient_funds
rid=req-789 user_id=42

Card validation error card_last4=4242 card_type=visa error=expired
merchant_id=m-123 rid=req-790 user_id=43

Database query timeout query=SELECT component=payment-service
duration=5000 operation=charge_card rid=req-791 table=transactions
\`\`\`

Теперь можно:
- Найти все payment failures: \`error=*\`
- Найти проблемы конкретного пользователя: \`user_id=42\`
- Найти медленные запросы: \`duration>3000\`
- Отследить flow запроса: \`rid=req-789\`

**5. Почему сортировать ключи?**
\`\`\`go
// Несортированный вывод (случайная итерация map):
// msg user_id=42 amount=1299 product=laptop
// msg product=laptop user_id=42 amount=1299
// msg amount=1299 product=laptop user_id=42

// Сортированный вывод (детерминированный):
// msg amount=1299 product=laptop user_id=42
// msg amount=1299 product=laptop user_id=42
// msg amount=1299 product=laptop user_id=42
\`\`\`

Преимущества:
- Согласованный формат для парсинга логов
- Легче grep/search
- Лучше для тестирования (стабильный вывод)
- Люди могут читать логи легче

**6. Production Impact at Scale**

E-commerce компания с 10M ежедневных запросов:

**До Structured Logging:**
- Время debugging: 4-6 часов на инцидент
- Ручной парсинг логов с regex
- 50% времени потрачено на поиск релевантных логов
- Частые regex баги ломающие парсинг логов

**После Structured Logging:**
- Время debugging: 15-30 минут на инцидент (-88%)
- Запросы к логам как к БД: \`user_id=X AND error=Y\`
- Мгновенная фильтрация логов в Elasticsearch
- Нулевое обслуживание regex

Экономия: $500K/год в инженерном времени

**7. Интеграция с Observability Tools**

Structured logs напрямую питают:
- **Metrics**: Подсчет ошибок по полю \`error_type\`
- **Alerts**: Алерт когда \`status=failed AND payment_method=card\` > 5%
- **Dashboards**: График поля \`duration\` по времени для каждого \`service\`
- **Tracing**: Связь логов по \`rid\` с distributed traces

Все автоматически - просто добавляйте поля в логи!`
		},
		uz: {
			title: `Strukturaviy maydonlar`,
			solutionCode: `package loggingx

import (
	"context"
	"log"
	"sort"
	"strings"
)

const keyFields CtxKey = "fields"

func WithFields(ctx context.Context, fields map[string]string) context.Context {
	if ctx == nil { // nil berilganda standart background kontekst
		ctx = context.Background()
	}
	merged := make(map[string]string)                                 // chaqiruvchi holatini o'zgartirmaslik uchun yangi map ajratamiz
	if existing, ok := ctx.Value(keyFields).(map[string]string); ok { // allaqachon mavjud maydonlarni nusxalaymiz
		for k, v := range existing {
			merged[k] = v // asl map dan key-value juftlarini dublikatsiya qilamiz
		}
	}
	for k, v := range fields { // berilgan map dan overridelarni qo'llaymiz
		merged[k] = v // har bir kalit uchun eng so'nggi qiymatni saqlaymiz
	}
	return context.WithValue(ctx, keyFields, merged) // birlashtirilgan maydonlarni ushlab turuvchi yangi kontekst qaytaramiz
}

func LogKV(ctx context.Context, msg string, extra map[string]string) {
	combined := make(map[string]string)                             // logging uchun birlashtirilgan maydonlar to'plamini ushlaymiz
	if fields, ok := ctx.Value(keyFields).(map[string]string); ok { // mavjud bo'lganda kontekst maydonlarini qo'shamiz
		for k, v := range fields {
			combined[k] = v // saqlangan maydon qiymatlarini nusxalaymiz
		}
	}
	for k, v := range extra { // kontekstni override qilib explicit extra maydonlarni birlashtiramiz
		combined[k] = v
	}
	if rid := RequestID(ctx); rid != "" { // request id ni oddiy key-value juftligi sifatida ko'rsatamiz
		combined["rid"] = rid
	}
	keys := make([]string, 0, len(combined)) // deterministik chiqish uchun kalitlarni tartiblash uchun to'playmiz
	for k := range combined {
		keys = append(keys, k)
	}
	sort.Strings(keys)           // log qatori uchun deterministik tartibni ta'minlaymiz
	builder := strings.Builder{} // yakuniy log xabarini to'playmiz
	builder.WriteString(msg)     // tavsiflovchi xabar qismi bilan boshlaymiz
	for _, k := range keys {     // har bir key-value juftligini tartiblangan tartibda qo'shamiz
		builder.WriteString(" ")         // xabar va kv yozuvlarini bo'shliq bilan ajratamiz
		builder.WriteString(k)           // kalit nomini yozamiz
		builder.WriteString("=")         // equals ajratgichini qo'shamiz
		builder.WriteString(combined[k]) // mos qiymatni yozamiz
	}
	log.Print(builder.String()) // yig'ilgan log yozuvini standart logger orqali chiqaramiz
}`,
			description: `Mashina tahlil qila oladigan loglar uchun kontekst orqali to'planadigan key-value maydonlari bilan structured logging ni amalga oshiring.

**Talablar:**
1. **WithFields**: Mavjud maydonlar bilan birlashtirib, kontekstga structured maydonlar qo'shish
2. **LogKV**: Deterministik chiqish uchun tartiblangan kalitlar bilan \`key=value\` formatida loglarni chiqarish

**Structured Logging Pattern:**
\`\`\`go
ctx := context.Background()
ctx = WithRequestID(ctx, "req-123")
ctx = WithFields(ctx, map[string]string{
    "user_id": "42",
    "action":  "purchase",
})

LogKV(ctx, "Order created", map[string]string{
    "order_id": "ord-456",
})

// Chiqish (tartiblangan kalitlar):
// Order created action=purchase order_id=ord-456 price=1299 rid=req-123 user_id=42
\`\`\`

**Nima uchun Key-Value formati:**
- **Mashina tahlil qila oladi**: Log aggregatorlar har bir maydonni indekslaydi
- **Qidiruv mumkin**: \`user_id=42 AND action=purchase\` kabi so'rovlar
- **Strukturalangan**: Regex tahlil kerak emas

**Cheklovlar:**
- Mavjud kontekst maydonlarini yangilari bilan birlashtiring
- Keyingi maydonlar bir xil kalit uchun oldingilarni override qiladi
- Mavjud bo'lsa request ID ni \`rid\` maydoni sifatida qo'shing
- Chiqishdan oldin kalitlarni alifbo tartibida tartiblang
- Samarali birlashtirish uchun \`strings.Builder\` dan foydalaning`,
			hint1: `WithFields da yangi map yarating, mavjud maydonlarni kontekstdan nusxalang, keyin yangi maydonlarni.`,
			hint2: `LogKV da barcha kalitlarni to'plang, sort.Strings() bilan tartiblang, strings.Builder yordamida chiqish quring.`,
			whyItMatters: `Structured logging zamonaviy observability platformalarining asosi va katta hajmda kuchli log tahlilini ta'minlaydi.

**Nima uchun bu muhim:**

**1. Mashina o'qiy oladigan loglar**
\`\`\`go
// ESKI: Strukturalanmagan logging
log.Print("User 42 purchased laptop for $1299")

// Muammo: user_id, product, price ni chiqarish uchun regex kerak
// Mo'rt, sekin, format o'zgarganda buziladi

// YANGI: Structured logging
LogKV(ctx, "Purchase completed", map[string]string{
    "user_id": "42",
    "product": "laptop",
    "price":   "1299",
})
// Chiqish: Purchase completed price=1299 product=laptop rid=req-123 user_id=42

// Yechim: Log aggregatorlar har bir maydonni avtomatik indekslaydi
// So'rov: user_id=42 AND product=laptop AND price>1000
\`\`\`

**2. Production da kuchli so'rovlar**
Log aggregatorlar (Elasticsearch, Splunk, CloudWatch) bilan:
\`\`\`
# User 42 uchun barcha muvaffaqiyatsiz to'lovlarni toping
user_id=42 AND status=failed

# Sekin database so'rovlarini toping (>1s)
component=database AND duration>1000

# Oxirgi soatda aniq so'rov uchun barcha xatolarni toping
rid=req-abc-123 AND level=error AND timestamp>now-1h
\`\`\`

Structured logging siz murakkab regex patternlari kerak bo'ladi va ular oson buziladi.

**3. Context Accumulation Pattern**
\`\`\`go
func HandleRequest(ctx context.Context, r *http.Request) {
    // So'rov darajasidagi maydonlarni qo'shish
    ctx = WithFields(ctx, map[string]string{
        "method": r.Method,
        "path":   r.URL.Path,
    })

    user := authenticate(ctx, r)
    // Foydalanuvchi darajasidagi maydonlarni qo'shish
    ctx = WithFields(ctx, map[string]string{
        "user_id":   user.ID,
        "user_role": user.Role,
    })

    processOrder(ctx, user)
}

func processOrder(ctx context.Context, user *User) {
    // Buyurtma darajasidagi maydonlarni qo'shish
    ctx = WithFields(ctx, map[string]string{
        "order_id": generateOrderID(),
    })

    // BARCHA loglar endi method, path, user_id, user_role, order_id ga ega
    LogKV(ctx, "Order created", map[string]string{
        "total": "1299.00",
    })
    // Chiqish: Order created method=POST order_id=ord-456 path=/api/orders
    //         rid=req-123 total=1299.00 user_id=42 user_role=premium
}
\`\`\`

Maydonlar kontekst ilova orqali o'tganda to'planadi!

**4. Real-World Production misoli**

**Structured Logging dan oldin:**
\`\`\`
ERROR: Payment failed
ERROR: Card declined
ERROR: Database timeout
\`\`\`
Kontekst yo'q! Qaysi foydalanuvchi? Qaysi to'lov? Qaysi karta?

**Structured Logging dan keyin:**
\`\`\`
Payment processing failed amount=1299 card_last4=4242 currency=USD
merchant_id=m-123 payment_id=pay-456 reason=insufficient_funds
rid=req-789 user_id=42

Card validation error card_last4=4242 card_type=visa error=expired
merchant_id=m-123 rid=req-790 user_id=43

Database query timeout query=SELECT component=payment-service
duration=5000 operation=charge_card rid=req-791 table=transactions
\`\`\`

Endi:
- Barcha to'lov muvaffaqiyatsizliklarini topish mumkin: \`error=*\`
- Foydalanuvchi muammolarini topish: \`user_id=42\`
- Sekin so'rovlarni topish: \`duration>3000\`
- So'rov oqimini kuzatish: \`rid=req-789\`

**5. Nima uchun kalitlarni tartiblash?**
\`\`\`go
// Tartiblanmagan chiqish (tasodifiy map iteratsiyasi):
// msg user_id=42 amount=1299 product=laptop
// msg product=laptop user_id=42 amount=1299
// msg amount=1299 product=laptop user_id=42

// Tartiblangan chiqish (deterministik):
// msg amount=1299 product=laptop user_id=42
// msg amount=1299 product=laptop user_id=42
// msg amount=1299 product=laptop user_id=42
\`\`\`

Afzalliklari:
- Log tahlil uchun izchil format
- grep/search uchun osonroq
- Testlash uchun yaxshiroq (barqaror chiqish)
- Odamlar loglarni osonroq o'qishlari mumkin

**6. Katta hajmda Production ta'siri**

Kuniga 10M so'rovli E-commerce kompaniyasi:

**Structured Logging dan oldin:**
- Debugging vaqti: hodisa uchun 4-6 soat
- Regex bilan qo'lda log tahlil
- Vaqtning 50% tegishli loglarni topishga ketadi
- Tez-tez regex xatolari log tahlilni buzadi

**Structured Logging dan keyin:**
- Debugging vaqti: hodisa uchun 15-30 daqiqa (-88%)
- Loglarni ma'lumotlar bazasi kabi so'rov qilish: \`user_id=X AND error=Y\`
- Elasticsearch da bir zumda log filtrlash
- Nol regex xizmati

Tejash: yiliga muhandislik vaqtida $500K

**7. Observability vositalari bilan integratsiya**

Structured loglar to'g'ridan-to'g'ri quyidagilarga kiradi:
- **Metrics**: \`error_type\` maydoni bo'yicha xatolarni hisoblash
- **Alerts**: \`status=failed AND payment_method=card\` > 5% bo'lganda ogohlantirish
- **Dashboards**: Har bir \`service\` maydoni uchun vaqt bo'yicha \`duration\` maydonini grafik qilish
- **Tracing**: \`rid\` bo'yicha loglarni distributed traces bilan bog'lash

Hammasi avtomatik - faqat loglarga maydonlar qo'shing!`
		}
	}
};

export default task;
