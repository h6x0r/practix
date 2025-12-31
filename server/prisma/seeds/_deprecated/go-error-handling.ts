// Error Handling & Best Practices - Production Go error patterns

export const GO_ERROR_HANDLING_MODULES = [
	{
		title: 'Error Handling Fundamentals',
		description: 'Master production-ready error handling patterns in Go.',
		section: 'best-practices',
		order: 5,
		topics: [
			{
				title: 'Sentinel Errors & Custom Types',
				description: 'Idiomatic error patterns for production Go code.',
				difficulty: 'medium',
				estimatedTime: '2.5h',
				order: 1,
				tasks: [
					{
						slug: 'go-sentinel-errors',
						title: 'Sentinel Errors',
						difficulty: 'easy',
						tags: ['go', 'errors', 'best-practices'],
						estimatedTime: '20m',
						isPremium: false,
						youtubeUrl: '',
						description: `Declare **sentinel error** variables for common error conditions.

**Requirements:**
1. Create package-level error variables \`ErrNotFound\` and \`ErrUnauthorized\`
2. Use \`errors.New()\` with descriptive messages
3. Export variables so they can be checked with \`errors.Is()\`
4. Follow Go naming convention (Err prefix)

**Example:**
\`\`\`go
func GetUser(id string) (*User, error) {
    user := db.Find(id)
    if user == nil {
        return nil, ErrNotFound
    }
    if !hasPermission() {
        return nil, ErrUnauthorized
    }
    return user, nil
}

// Caller can check:
if errors.Is(err, ErrNotFound) {
    return http.StatusNotFound
}
\`\`\`

**Constraints:**
- Must be package-level variables (not constants)
- Use clear, consistent error messages`,
						initialCode: `package errorsx

import "errors"

// TODO: Declare ErrNotFound and ErrUnauthorized sentinel errors
var (
	ErrNotFound     error
	ErrUnauthorized error
)`,
						solutionCode: `package errorsx

import "errors"

var (
	ErrNotFound     = errors.New("not found")     // Resource not found error
	ErrUnauthorized = errors.New("unauthorized")  // Authorization failure error
)`,
						hint1: 'Use errors.New() to create sentinel errors with descriptive messages.',
						hint2: 'Export variables (uppercase) so callers can use errors.Is() to check them.',
						whyItMatters: `Sentinel errors are Go's idiomatic way to represent well-known error conditions. They enable callers to make programmatic decisions without string parsing.

**Why This Matters:**
- **Type-safe checking:** Use \`errors.Is()\` instead of string comparison
- **API stability:** Error messages can change without breaking clients
- **Clear contracts:** Exported sentinels document expected error conditions

**Real-World Examples:**
- Standard library: \`io.EOF\`, \`sql.ErrNoRows\`, \`os.ErrNotExist\`
- Database drivers: \`ErrConnDone\`, \`ErrTxDone\`
- HTTP clients: \`context.Canceled\`, \`context.DeadlineExceeded\`

**Production Pattern:**
\`\`\`go
// Service layer
if errors.Is(err, repo.ErrNotFound) {
    return nil, ErrUserNotFound  // Wrap for domain
}

// HTTP handler
if errors.Is(err, service.ErrUserNotFound) {
    return 404, "User not found"
}
\`\`\`

Without sentinels, you'd resort to fragile string matching that breaks when messages are localized.`,
						translations: {
							ru: {
								title: 'Sentinel Errors',
								description: `Объявите переменные **sentinel error** для распространённых ошибок.

**Требования:**
1. Создайте package-level переменные \`ErrNotFound\` и \`ErrUnauthorized\`
2. Используйте \`errors.New()\` с описательными сообщениями
3. Экспортируйте переменные для проверки через \`errors.Is()\`
4. Следуйте соглашению Go (префикс Err)

**Пример:**
\`\`\`go
func GetUser(id string) (*User, error) {
    user := db.Find(id)
    if user == nil {
        return nil, ErrNotFound
    }
    return user, nil
}

if errors.Is(err, ErrNotFound) {
    return http.StatusNotFound
}
\`\`\`

**Ограничения:**
- Должны быть package-level переменными
- Используйте чёткие сообщения об ошибках`,
								hint1: 'Используйте errors.New() для создания sentinel errors.',
								hint2: 'Экспортируйте переменные для использования errors.Is().',
								whyItMatters: `Sentinel errors - идиоматичный способ Go представлять известные ошибки. Они позволяют принимать программные решения без парсинга строк.

**Почему важно:**
- **Type-safe проверка:** \`errors.Is()\` вместо сравнения строк
- **Стабильность API:** Сообщения могут меняться без нарушения клиентов
- **Чёткие контракты:** Экспортированные sentinels документируют ошибки

**Real-world:** \`io.EOF\`, \`sql.ErrNoRows\`, \`os.ErrNotExist\``
							},
							uz: {
								title: 'Sentinel Errors',
								description: `Umumiy xato holatlari uchun **sentinel error** o'zgaruvchilarini e'lon qiling.

**Talablar:**
1. \`ErrNotFound\` va \`ErrUnauthorized\` package-level o'zgaruvchilarini yarating
2. \`errors.New()\` dan tavsiflovchi xabarlar bilan foydalaning
3. \`errors.Is()\` bilan tekshirish uchun o'zgaruvchilarni eksport qiling
4. Go nomlash konventsiyasiga amal qiling (Err prefiksi)

**Misol:**
\`\`\`go
func GetUser(id string) (*User, error) {
    user := db.Find(id)
    if user == nil {
        return nil, ErrNotFound
    }
    return user, nil
}

if errors.Is(err, ErrNotFound) {
    return http.StatusNotFound
}
\`\`\`

**Cheklovlar:**
- Package-level o'zgaruvchilar bo'lishi kerak
- Aniq xato xabarlaridan foydalaning`,
								hint1: 'Sentinel errorlar yaratish uchun errors.New() dan foydalaning.',
								hint2: 'errors.Is() uchun o\'zgaruvchilarni eksport qiling.',
								whyItMatters: `Sentinel errorlar - Go ning ma'lum xatolarni ifodalashning idiomatik usuli. String parsing siz qarorlar qabul qilish imkonini beradi.

**Nima uchun muhim:**
- **Type-safe tekshirish:** String taqqoslash o'rniga \`errors.Is()\`
- **API barqarorligi:** Xato xabarlari o'zgarishi mumkin
- **Aniq shartnomalar:** Eksport qilingan sentinellar xatolarni hujjatlashtiradi

**Real-world:** \`io.EOF\`, \`sql.ErrNoRows\`, \`os.ErrNotExist\``
							}
						}
					},
					{
						slug: 'go-custom-error-type',
						title: 'Custom Error Type',
						difficulty: 'easy',
						tags: ['go', 'errors', 'types'],
						estimatedTime: '20m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement **AppError** custom error type with contextual information.

**Requirements:**
1. Define \`AppError\` struct with fields: \`Code\`, \`Op\` (operation), \`Err\` (underlying error)
2. Implement \`Error() string\` method to satisfy error interface
3. Format output as: \`"op=<operation>, code=<code>, err=<error>"\`
4. Handle nil receiver gracefully

**Example:**
\`\`\`go
err := &AppError{
    Code: "DB_ERROR",
    Op:   "users.Create",
    Err:  sql.ErrNoRows,
}

fmt.Println(err)
// Output: op=users.Create, code=DB_ERROR, err=sql: no rows in result set
\`\`\`

**Constraints:**
- Error() must never panic (check nil receiver)
- Include all fields in error message`,
						initialCode: `package errorsx

import "fmt"

// TODO: Define AppError struct with Code, Op, Err fields
type AppError struct {
	Code string
	Op   string
	Err  error
}

// TODO: Implement Error() method
func (e *AppError) Error() string {
	panic("TODO")
}`,
						solutionCode: `package errorsx

import "fmt"

type AppError struct {
	Code string  // Error code (e.g., "NOT_FOUND", "UNAUTHORIZED")
	Op   string  // Operation where error occurred
	Err  error   // Underlying error
}

func (e *AppError) Error() string {
	if e == nil {                                            // Nil receiver check
		return "<nil>"                                   // Prevent panic
	}
	return fmt.Sprintf("op=%s, code=%s, err=%v", e.Op, e.Code, e.Err)  // Format error message
}`,
						hint1: 'Check if receiver is nil before accessing fields to prevent panic.',
						hint2: 'Use fmt.Sprintf to format all fields into a readable error message.',
						whyItMatters: `Custom error types add structured context to errors, making debugging and monitoring easier in production.

**Why Custom Errors:**
- **Context:** Know WHERE (operation) and WHAT (code) failed, not just WHY (message)
- **Structured Logging:** Log fields separately for queryable error tracking
- **Error Classification:** Group errors by code for metrics and alerting

**Production Benefits:**
\`\`\`go
// Error without context
return errors.New("database error")

// Error with context
return &AppError{
    Code: "DB_UNAVAILABLE",
    Op:   "orders.Create",
    Err:  originalErr,
}
\`\`\`

In monitoring dashboards, you can:
- Query all DB_UNAVAILABLE errors
- See which operations are failing most
- Trace error chains through microservices

**Real-World Usage:**
- Google's upspin.io/errors package
- Uber's go.uber.org/multierr
- HashiCorp's go-multierror

The Op field enables error aggregation: if 100 requests fail in orders.Create with DB_UNAVAILABLE, you know exactly where to look.`,
						translations: {
							ru: {
								title: 'Пользовательский тип ошибки',
								description: `Реализуйте пользовательский тип ошибки **AppError** с контекстной информацией.

**Требования:**
1. Определите структуру \`AppError\` с полями: \`Code\`, \`Op\`, \`Err\`
2. Реализуйте метод \`Error() string\`
3. Форматируйте вывод: \`"op=<operation>, code=<code>, err=<error>"\`
4. Корректно обрабатывайте nil receiver

**Пример:**
\`\`\`go
err := &AppError{
    Code: "DB_ERROR",
    Op:   "users.Create",
    Err:  sql.ErrNoRows,
}

fmt.Println(err)
// Output: op=users.Create, code=DB_ERROR, err=sql: no rows
\`\`\`

**Ограничения:**
- Error() не должен паниковать
- Включите все поля в сообщение`,
								hint1: 'Проверьте nil receiver перед доступом к полям.',
								hint2: 'Используйте fmt.Sprintf для форматирования полей.',
								whyItMatters: `Пользовательские типы ошибок добавляют структурированный контекст, упрощая отладку в production.

**Почему важно:**
- **Контекст:** Знаете ГДЕ и ЧТО упало, не только ПОЧЕМУ
- **Структурированное логирование:** Поля логируются отдельно
- **Классификация:** Группировка ошибок по коду для метрик

**Production преимущества:** В мониторинге можно запросить все DB_UNAVAILABLE ошибки, увидеть какие операции падают чаще.`
							},
							uz: {
								title: 'Custom Error turi',
								description: `Kontekstli ma'lumotlar bilan **AppError** custom error turini amalga oshiring.

**Talablar:**
1. \`AppError\` strukturasini \`Code\`, \`Op\`, \`Err\` fieldlar bilan aniqlang
2. \`Error() string\` metodini amalga oshiring
3. Chiqishni formatlang: \`"op=<operation>, code=<code>, err=<error>"\`
4. nil receiver ni to'g'ri ishlang

**Misol:**
\`\`\`go
err := &AppError{
    Code: "DB_ERROR",
    Op:   "users.Create",
    Err:  sql.ErrNoRows,
}

fmt.Println(err)
// Natija: op=users.Create, code=DB_ERROR, err=sql: no rows
\`\`\`

**Cheklovlar:**
- Error() panic bo'lmasligi kerak
- Barcha fieldlarni xabar ichiga kiriting`,
								hint1: 'Fieldlarga kirishdan oldin nil receiver ni tekshiring.',
								hint2: 'Fieldlarni formatlash uchun fmt.Sprintf dan foydalaning.',
								whyItMatters: `Custom error turlari strukturali kontekst qo'shadi, production'da debugni osonlashtiradi.

**Nima uchun muhim:**
- **Kontekst:** QAYERDA va NIMA ishlamay qoldi, faqat NIMA UCHUN emas
- **Strukturali logging:** Fieldlar alohida loglanadi
- **Klassifikatsiya:** Kod bo'yicha xatolarni guruhlash

**Production foydalari:** Monitoring'da barcha DB_UNAVAILABLE xatolarini so'rash, qaysi operatsiyalar ko'proq ishlamay qolishini ko'rish mumkin.`
							}
						}
					},
					{
						slug: 'go-error-unwrap',
						title: 'Error Unwrapping',
						difficulty: 'medium',
						tags: ['go', 'errors', 'unwrap'],
						estimatedTime: '20m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement **Unwrap()** method for AppError to support error chain inspection.

**Requirements:**
1. Add \`Unwrap() error\` method to AppError
2. Return the underlying \`Err\` field
3. Handle nil receiver gracefully
4. Enable \`errors.Is()\` and \`errors.As()\` to traverse error chain

**Example:**
\`\`\`go
dbErr := sql.ErrNoRows
appErr := &AppError{
    Code: "NOT_FOUND",
    Op:   "users.GetByID",
    Err:  dbErr,
}

// Unwrap enables errors.Is to check underlying error
if errors.Is(appErr, sql.ErrNoRows) {
    fmt.Println("Database returned no rows")
}
\`\`\`

**Constraints:**
- Must return nil for nil receiver
- Must return the Err field`,
						initialCode: `package errorsx

type AppError struct {
	Code string
	Op   string
	Err  error
}

// TODO: Implement Unwrap() method to return underlying error
func (e *AppError) Unwrap() error {
	panic("TODO")
}`,
						solutionCode: `package errorsx

type AppError struct {
	Code string
	Op   string
	Err  error
}

func (e *AppError) Unwrap() error {
	if e == nil {        // Nil receiver check
		return nil   // Safe handling
	}
	return e.Err         // Return underlying error for chain inspection
}`,
						hint1: 'Check if receiver is nil before accessing fields.',
						hint2: 'Simply return the Err field to expose the underlying error.',
						whyItMatters: `Unwrap() enables Go's standard error inspection functions to traverse error chains.

**Why Unwrap:**
- **errors.Is():** Check if any error in chain matches a target
- **errors.As():** Extract specific error type from chain
- **Error Context:** Preserve original error while adding context layers

**Production Pattern:**
\`\`\`go
// Layer 1: Database error
dbErr := sql.ErrNoRows

// Layer 2: Repository adds context
repoErr := &AppError{Op: "repo.GetUser", Err: dbErr}

// Layer 3: Service adds business context
svcErr := &AppError{Code: "USER_NOT_FOUND", Op: "service.GetUser", Err: repoErr}

// Check original error through all layers
if errors.Is(svcErr, sql.ErrNoRows) {
    // True! Unwrap() lets errors.Is traverse the chain
}
\`\`\`

**Real-World Benefits:**
- **Microservices:** Pass errors between services without losing root cause
- **Observability:** Log error chains for full context
- **Error Handling:** Make decisions based on root cause, not wrapper

Without Unwrap(), you'd lose access to the original error and couldn't use errors.Is/As.`,
						translations: {
							ru: {
								title: 'Развертывание ошибок',
								description: `Реализуйте метод **Unwrap()** для AppError для поддержки цепочки ошибок.

**Требования:**
1. Добавьте метод \`Unwrap() error\` к AppError
2. Возвращайте вложенное поле \`Err\`
3. Обрабатывайте nil receiver корректно
4. Включите поддержку \`errors.Is()\` и \`errors.As()\`

**Пример:**
\`\`\`go
dbErr := sql.ErrNoRows
appErr := &AppError{
    Code: "NOT_FOUND",
    Op:   "users.GetByID",
    Err:  dbErr,
}

if errors.Is(appErr, sql.ErrNoRows) {
    fmt.Println("База данных не вернула строк")
}
\`\`\`

**Ограничения:**
- Должен возвращать nil для nil receiver
- Должен возвращать поле Err`,
								hint1: 'Проверьте nil receiver перед доступом к полям.',
								hint2: 'Просто верните поле Err для доступа к вложенной ошибке.',
								whyItMatters: `Unwrap() позволяет стандартным функциям Go проверять цепочки ошибок.

**Почему важно:**
- **errors.Is():** Проверка ошибки в цепочке
- **errors.As():** Извлечение конкретного типа ошибки
- **Контекст ошибок:** Сохранение оригинальной ошибки с добавлением контекста

**Production:** Без Unwrap() вы потеряете доступ к оригинальной ошибке.`
							},
							uz: {
								title: 'Error Unwrapping',
								description: `AppError uchun error zanjirini tekshirishni qo llab-quvvatlash uchun **Unwrap()** metodini amalga oshiring.

**Talablar:**
1. AppError ga \`Unwrap() error\` metodini qo shing
2. Ichki \`Err\` fieldni qaytaring
3. nil receiver ni to g ri ishlang
4. \`errors.Is()\` va \`errors.As()\` uchun support qo shing

**Misol:**
\`\`\`go
dbErr := sql.ErrNoRows
appErr := &AppError{
    Code: "NOT_FOUND",
    Op:   "users.GetByID",
    Err:  dbErr,
}

if errors.Is(appErr, sql.ErrNoRows) {
    fmt.Println("Database qatorlar qaytarmadi")
}
\`\`\`

**Cheklovlar:**
- nil receiver uchun nil qaytarishi kerak
- Err fieldni qaytarishi kerak`,
								hint1: 'Fieldlarga kirishdan oldin nil receiver ni tekshiring.',
								hint2: 'Ichki xatoga kirish uchun Err fieldni qaytaring.',
								whyItMatters: `Unwrap() Go ning standart funksiyalariga error zanjirlarini tekshirish imkonini beradi.

**Nima uchun muhim:**
- **errors.Is():** Zanjirdagi xatoni tekshirish
- **errors.As():** Muayyan xato turini ajratib olish
- **Error konteksti:** Asl xatoni kontekst qo shish bilan saqlash

**Production:** Unwrap() siz asl xatoga kirishni yo qotasiz.`
							}
						}
					},
					{
						slug: 'go-error-wrap',
						title: 'Error Wrapping Function',
						difficulty: 'medium',
						tags: ['go', 'errors', 'wrapping'],
						estimatedTime: '25m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement **Wrap()** function to add operation context to errors.

**Requirements:**
1. Create function \`Wrap(op string, err error) error\`
2. Return nil if input error is nil
3. Wrap non-nil errors in AppError with operation context
4. Don't set Code field (only Op and Err)

**Example:**
\`\`\`go
func GetUser(id string) (*User, error) {
    user, err := db.Query(id)
    if err != nil {
        return nil, Wrap("users.GetUser", err)  // Add operation context
    }
    return user, nil
}

// Error output: op=users.GetUser, code=, err=sql: connection refused
\`\`\`

**Constraints:**
- Must return nil when err is nil (preserve nil errors)
- Must create new AppError instance`,
						initialCode: `package errorsx

type AppError struct {
	Code string
	Op   string
	Err  error
}

// TODO: Implement Wrap function
func Wrap(op string, err error) error {
	panic("TODO")
}`,
						solutionCode: `package errorsx

type AppError struct {
	Code string
	Op   string
	Err  error
}

func Wrap(op string, err error) error {
	if err == nil {              // Preserve nil errors
		return nil           // Don't create AppError for nil
	}
	return &AppError{            // Wrap with operation context
		Op:  op,             // Operation where error occurred
		Err: err,            // Original error
	}
}`,
						hint1: 'Always check if err is nil first and return nil to preserve error absence.',
						hint2: 'Create and return a new AppError with Op and Err fields only.',
						whyItMatters: `Wrap() adds execution context to errors without changing their meaning, enabling call stack reconstruction.

**Why Wrap Errors:**
- **Call Stack:** Track error propagation through layers
- **Debugging:** Know exactly where error was handled
- **Observability:** Build error context trees for monitoring

**Production Pattern:**
\`\`\`go
// Repository layer
func (r *Repo) GetUser(id string) (*User, error) {
    user, err := r.db.Query(id)
    if err != nil {
        return nil, Wrap("repo.GetUser", err)
    }
    return user, nil
}

// Service layer
func (s *Service) GetUser(id string) (*User, error) {
    user, err := s.repo.GetUser(id)
    if err != nil {
        return nil, Wrap("service.GetUser", err)
    }
    return user, nil
}

// Handler layer
func (h *Handler) GetUser(w http.ResponseWriter, r *http.Request) {
    user, err := h.service.GetUser(id)
    if err != nil {
        // Error chain: handler.GetUser -> service.GetUser -> repo.GetUser -> sql: connection refused
        log.Error(Wrap("handler.GetUser", err))
    }
}
\`\`\`

**Real-World Benefits:**
- **Error Traces:** See full execution path without stack traces
- **Structured Logs:** Each layer adds operation name for filtering
- **Debugging:** Jump directly to failing operation in logs

**nil Preservation:** Returning nil for nil input is crucial - it prevents creating error objects when no error exists, which would turn success cases into failures.`,
						translations: {
							ru: {
								title: 'Функция обертывания ошибок',
								description: `Реализуйте функцию **Wrap()** для добавления контекста операции к ошибкам.

**Требования:**
1. Создайте функцию \`Wrap(op string, err error) error\`
2. Возвращайте nil если входная ошибка nil
3. Оборачивайте не-nil ошибки в AppError с контекстом операции
4. Не устанавливайте поле Code (только Op и Err)

**Пример:**
\`\`\`go
func GetUser(id string) (*User, error) {
    user, err := db.Query(id)
    if err != nil {
        return nil, Wrap("users.GetUser", err)
    }
    return user, nil
}
\`\`\`

**Ограничения:**
- Должна возвращать nil когда err nil
- Должна создавать новый экземпляр AppError`,
								hint1: 'Всегда проверяйте err на nil первым и возвращайте nil.',
								hint2: 'Создайте и верните новый AppError с полями Op и Err.',
								whyItMatters: `Wrap() добавляет контекст выполнения к ошибкам, позволяя восстановить стек вызовов.

**Почему важно:**
- **Стек вызовов:** Отслеживание распространения ошибки
- **Отладка:** Точное знание где ошибка обработана
- **Observability:** Построение деревьев контекста ошибок

**nil Preservation:** Возврат nil для nil входа критичен - это предотвращает создание объектов ошибок когда ошибки нет.`
							},
							uz: {
								title: 'Error o rash funksiyasi',
								description: `Xatolarga operatsiya kontekstini qo shish uchun **Wrap()** funksiyasini amalga oshiring.

**Talablar:**
1. \`Wrap(op string, err error) error\` funksiyasini yarating
2. Agar kirish xatosi nil bo lsa nil qaytaring
3. nil bo lmagan xatolarni operatsiya konteksti bilan AppError ga o rang
4. Code fieldni o rnatmang (faqat Op va Err)

**Misol:**
\`\`\`go
func GetUser(id string) (*User, error) {
    user, err := db.Query(id)
    if err != nil {
        return nil, Wrap("users.GetUser", err)
    }
    return user, nil
}
\`\`\`

**Cheklovlar:**
- err nil bo lganda nil qaytarishi kerak
- Yangi AppError instance yaratishi kerak`,
								hint1: 'Avval err ni nil ga tekshiring va nil qaytaring.',
								hint2: 'Faqat Op va Err fieldlari bilan yangi AppError yarating.',
								whyItMatters: `Wrap() xatolarga ijro kontekstini qo shadi, chaqiruvlar stekini qayta qurishga imkon beradi.

**Nima uchun muhim:**
- **Chaqiruvlar steki:** Xatoning tarqalishini kuzatish
- **Debugging:** Xato qaerda ishlangani aniq bilish
- **Observability:** Monitoring uchun xato kontekst daraxtlarini qurish

**nil Preservation:** nil kirish uchun nil qaytarish muhim - bu xato yo q bo lganda xato obyektlarini yaratishni oldini oladi.`
							}
						}
					},
					{
						slug: 'go-error-e',
						title: 'Error Constructor with Code',
						difficulty: 'medium',
						tags: ['go', 'errors', 'constructor'],
						estimatedTime: '25m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement **E()** constructor function to create AppError with error code.

**Requirements:**
1. Create function \`E(code, op string, err error) error\`
2. Return nil if input error is nil
3. Create AppError with all three fields: Code, Op, Err
4. Use short, descriptive name (E is idiomatic)

**Example:**
\`\`\`go
func GetUser(id string) (*User, error) {
    user, err := db.Query(id)
    if err != nil {
        return nil, E("DB_ERROR", "users.GetUser", err)
    }

    if user.DeletedAt != nil {
        return nil, E("USER_DELETED", "users.GetUser", ErrNotFound)
    }

    return user, nil
}

// Error output: op=users.GetUser, code=DB_ERROR, err=connection timeout
\`\`\`

**Constraints:**
- Must return nil when err is nil
- Must set all three AppError fields`,
						initialCode: `package errorsx

type AppError struct {
	Code string
	Op   string
	Err  error
}

// TODO: Implement E constructor function
func E(code, op string, err error) error {
	panic("TODO")
}`,
						solutionCode: `package errorsx

type AppError struct {
	Code string
	Op   string
	Err  error
}

func E(code, op string, err error) error {
	if err == nil {                  // Preserve nil errors
		return nil               // Don't create error object
	}
	return &AppError{                // Create fully populated AppError
		Code: code,              // Business error code
		Op:   op,                // Operation context
		Err:  err,               // Underlying error
	}
}`,
						hint1: 'Check if err is nil first and return nil to preserve success cases.',
						hint2: 'Create AppError with all three fields populated from function parameters.',
						whyItMatters: `E() provides a concise way to create fully contextualized errors with business codes.

**Why Error Codes:**
- **Error Categorization:** Group errors by type (AUTH_ERROR, DB_ERROR, VALIDATION_ERROR)
- **Client Communication:** Return consistent error codes in APIs
- **Monitoring:** Track error rates by category in dashboards
- **Alerting:** Trigger alerts based on error codes

**Production Pattern:**
\`\`\`go
// Define error codes as constants
const (
    ErrCodeAuth       = "AUTH_ERROR"
    ErrCodeDB         = "DB_ERROR"
    ErrCodeValidation = "VALIDATION_ERROR"
    ErrCodeNotFound   = "NOT_FOUND"
)

// Service layer
func (s *Service) CreateOrder(order *Order) error {
    if err := order.Validate(); err != nil {
        return E(ErrCodeValidation, "service.CreateOrder", err)
    }

    if !s.auth.HasPermission(order.UserID) {
        return E(ErrCodeAuth, "service.CreateOrder", ErrUnauthorized)
    }

    if err := s.repo.Save(order); err != nil {
        return E(ErrCodeDB, "service.CreateOrder", err)
    }

    return nil
}

// HTTP handler
func (h *Handler) CreateOrder(w http.ResponseWriter, r *http.Request) {
    err := h.service.CreateOrder(order)
    if err != nil {
        var appErr *AppError
        if errors.As(err, &appErr) {
            // Map error codes to HTTP status
            switch appErr.Code {
            case ErrCodeAuth:
                w.WriteHeader(401)
            case ErrCodeValidation:
                w.WriteHeader(400)
            case ErrCodeDB:
                w.WriteHeader(500)
            }
        }
    }
}
\`\`\`

**Real-World Benefits:**
- **Metrics:** \`errors{code="DB_ERROR",op="orders.Create"}\` in Prometheus
- **API Consistency:** Clients can handle errors programmatically
- **Debugging:** Filter logs by error code to find related failures
- **SLA Monitoring:** Track error budgets by error category

**Short Name:** E() is intentionally brief - it's used frequently in error handling, so a short name reduces noise in code.`,
						translations: {
							ru: {
								title: 'Конструктор ошибки с кодом',
								description: `Реализуйте функцию-конструктор **E()** для создания AppError с кодом ошибки.

**Требования:**
1. Создайте функцию \`E(code, op string, err error) error\`
2. Возвращайте nil если входная ошибка nil
3. Создайте AppError со всеми тремя полями: Code, Op, Err
4. Используйте короткое имя (E идиоматично)

**Пример:**
\`\`\`go
func GetUser(id string) (*User, error) {
    user, err := db.Query(id)
    if err != nil {
        return nil, E("DB_ERROR", "users.GetUser", err)
    }
    return user, nil
}
\`\`\`

**Ограничения:**
- Должна возвращать nil когда err nil
- Должна заполнять все три поля AppError`,
								hint1: 'Проверьте err на nil первым и верните nil.',
								hint2: 'Создайте AppError со всеми тремя полями из параметров.',
								whyItMatters: `E() предоставляет лаконичный способ создания полностью контекстуализированных ошибок с бизнес-кодами.

**Почему коды ошибок:**
- **Категоризация:** Группировка ошибок по типу
- **API:** Возврат консистентных кодов ошибок
- **Мониторинг:** Отслеживание частоты ошибок по категориям
- **Алерты:** Триггеры алертов по кодам ошибок

**Короткое имя:** E() намеренно краткое - оно используется часто.`
							},
							uz: {
								title: 'Kod bilan error konstruktori',
								description: `Xato kodi bilan AppError yaratish uchun **E()** konstruktor funksiyasini amalga oshiring.

**Talablar:**
1. \`E(code, op string, err error) error\` funksiyasini yarating
2. Agar kirish xatosi nil bo lsa nil qaytaring
3. Uchta field bilan AppError yarating: Code, Op, Err
4. Qisqa, tavsiflovchi nom ishlating (E idiomatik)

**Misol:**
\`\`\`go
func GetUser(id string) (*User, error) {
    user, err := db.Query(id)
    if err != nil {
        return nil, E("DB_ERROR", "users.GetUser", err)
    }
    return user, nil
}
\`\`\`

**Cheklovlar:**
- err nil bo lganda nil qaytarishi kerak
- AppError ning uchta fieldini to ldirishi kerak`,
								hint1: 'Avval err ni nil ga tekshiring va nil qaytaring.',
								hint2: 'Parametrlardan uchta field bilan AppError yarating.',
								whyItMatters: `E() biznes kodlari bilan to liq kontekstlashtirilgan xatolarni yaratishning qisqa usulini taqdim etadi.

**Nima uchun xato kodlari:**
- **Kategoriyalash:** Xatolarni tur bo yicha guruhlash
- **API:** API larda doimiy xato kodlarini qaytarish
- **Monitoring:** Dashboard larda kategoriya bo yicha xato darajalarini kuzatish
- **Alerting:** Xato kodlari asosida alert larni ishga tushirish

**Qisqa nom:** E() ataylab qisqa - u tez-tez ishlatiladi.`
							}
						}
					},
					{
						slug: 'go-is-not-found',
						title: 'Error Inspection Helper',
						difficulty: 'medium',
						tags: ['go', 'errors', 'inspection'],
						estimatedTime: '20m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement **IsNotFound()** helper to check if error chain contains ErrNotFound.

**Requirements:**
1. Create function \`IsNotFound(err error) bool\`
2. Use \`errors.Is()\` to check error chain
3. Return true if ErrNotFound is anywhere in the chain
4. Work with wrapped errors

**Example:**
\`\`\`go
var ErrNotFound = errors.New("not found")

func GetUser(id string) (*User, error) {
    user, err := db.Query(id)
    if err == sql.ErrNoRows {
        return nil, E("NOT_FOUND", "users.GetUser", ErrNotFound)
    }
    return user, err
}

// Caller code
user, err := GetUser("123")
if IsNotFound(err) {
    return http.StatusNotFound, "User not found"
}
\`\`\`

**Constraints:**
- Must use errors.Is() for chain inspection
- Must return false for nil errors`,
						initialCode: `package errorsx

import "errors"

var ErrNotFound = errors.New("not found")

// TODO: Implement IsNotFound helper
func IsNotFound(err error) bool {
	panic("TODO")
}`,
						solutionCode: `package errorsx

import "errors"

var ErrNotFound = errors.New("not found")

func IsNotFound(err error) bool {
	return errors.Is(err, ErrNotFound)  // Check entire error chain
}`,
						hint1: 'Use errors.Is() which automatically traverses the error chain via Unwrap().',
						hint2: 'One line of code - just return the result of errors.Is().',
						whyItMatters: `Error inspection helpers provide type-safe, semantic error checking that decouples caller code from implementation details.

**Why Inspection Helpers:**
- **Semantic Clarity:** \`IsNotFound(err)\` is clearer than \`errors.Is(err, ErrNotFound)\`
- **Encapsulation:** Callers don't need to know about ErrNotFound variable
- **Refactoring Safety:** Change error implementation without breaking callers
- **API Stability:** Public IsNotFound() can remain stable even if internal errors change

**Production Pattern:**
\`\`\`go
// Package errorsx provides error helpers
package errorsx

var (
    ErrNotFound     = errors.New("not found")
    ErrUnauthorized = errors.New("unauthorized")
    ErrConflict     = errors.New("conflict")
)

func IsNotFound(err error) bool     { return errors.Is(err, ErrNotFound) }
func IsUnauthorized(err error) bool { return errors.Is(err, ErrUnauthorized) }
func IsConflict(err error) bool     { return errors.Is(err, ErrConflict) }

// Service layer - clean error handling
func (s *Service) GetUser(id string) (*User, error) {
    user, err := s.repo.GetUser(id)
    if IsNotFound(err) {
        return nil, E("USER_NOT_FOUND", "service.GetUser", err)
    }
    return user, err
}

// HTTP handler - map to status codes
func (h *Handler) GetUser(w http.ResponseWriter, r *http.Request) {
    user, err := h.service.GetUser(id)
    if err != nil {
        switch {
        case IsNotFound(err):
            w.WriteHeader(404)
        case IsUnauthorized(err):
            w.WriteHeader(401)
        case IsConflict(err):
            w.WriteHeader(409)
        default:
            w.WriteHeader(500)
        }
        return
    }
    json.NewEncoder(w).Encode(user)
}
\`\`\`

**Real-World Benefits:**
- **HTTP Status Codes:** Map errors to status codes easily
- **Retry Logic:** Decide whether to retry based on error type
- **Circuit Breakers:** Open circuits for specific error types
- **Metrics:** Count errors by semantic type

**Standard Library Examples:**
- \`os.IsNotExist(err)\` - Check for file not found
- \`os.IsPermission(err)\` - Check for permission errors
- \`net.ErrClosed\` with helpers in network code

**Why errors.Is():** It's better than string comparison or type assertion because it traverses the entire error chain, finding the sentinel even when wrapped multiple times.`,
						translations: {
							ru: {
								title: 'Хелпер проверки ошибок',
								description: `Реализуйте хелпер **IsNotFound()** для проверки наличия ErrNotFound в цепочке ошибок.

**Требования:**
1. Создайте функцию \`IsNotFound(err error) bool\`
2. Используйте \`errors.Is()\` для проверки цепочки
3. Возвращайте true если ErrNotFound есть в цепочке
4. Работайте с обернутыми ошибками

**Пример:**
\`\`\`go
var ErrNotFound = errors.New("not found")

user, err := GetUser("123")
if IsNotFound(err) {
    return http.StatusNotFound, "User not found"
}
\`\`\`

**Ограничения:**
- Должна использовать errors.Is()
- Должна возвращать false для nil`,
								hint1: 'Используйте errors.Is() который автоматически обходит цепочку.',
								hint2: 'Одна строка кода - просто верните результат errors.Is().',
								whyItMatters: `Хелперы проверки ошибок обеспечивают type-safe семантическую проверку.

**Почему важно:**
- **Семантическая ясность:** \`IsNotFound(err)\` яснее
- **Инкапсуляция:** Вызывающий код не знает о ErrNotFound
- **Рефакторинг:** Изменение реализации не ломает вызывающий код
- **Стабильность API:** Публичный IsNotFound() стабилен

**Примеры stdlib:** \`os.IsNotExist(err)\`, \`os.IsPermission(err)\``
							},
							uz: {
								title: 'Error tekshirish yordamchisi',
								description: `Error zanjirida ErrNotFound borligini tekshirish uchun **IsNotFound()** yordamchisini amalga oshiring.

**Talablar:**
1. \`IsNotFound(err error) bool\` funksiyasini yarating
2. Zanjirni tekshirish uchun \`errors.Is()\` dan foydalaning
3. Agar ErrNotFound zanjirda bo lsa true qaytaring
4. O ralgan xatolar bilan ishlang

**Misol:**
\`\`\`go
var ErrNotFound = errors.New("not found")

user, err := GetUser("123")
if IsNotFound(err) {
    return http.StatusNotFound, "User not found"
}
\`\`\`

**Cheklovlar:**
- errors.Is() dan foydalanishi kerak
- nil xatolar uchun false qaytarishi kerak`,
								hint1: 'Avtomatik ravishda zanjirni aylanadigan errors.Is() dan foydalaning.',
								hint2: 'Bir qator kod - faqat errors.Is() natijasini qaytaring.',
								whyItMatters: `Error tekshirish yordamchilari type-safe semantik xato tekshiruvini ta minlaydi.

**Nima uchun muhim:**
- **Semantik aniqlik:** \`IsNotFound(err)\` aniqroq
- **Inkapsulatsiya:** Chaqiruvchi kod ErrNotFound haqida bilmaydi
- **Refaktoring xavfsizligi:** Implementatsiyani o zgartirish chaqiruvchi kodni buzmaydi
- **API barqarorligi:** Ommaviy IsNotFound() barqaror qoladi

**Stdlib misollari:** \`os.IsNotExist(err)\`, \`os.IsPermission(err)\``
							}
						}
					},
					{
						slug: 'go-format-not-found',
						title: 'Domain-Specific Error Formatting',
						difficulty: 'medium',
						tags: ['go', 'errors', 'formatting'],
						estimatedTime: '25m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement **FormatNotFound()** to create domain-specific NotFound errors.

**Requirements:**
1. Create function \`FormatNotFound(id string) error\`
2. Use \`fmt.Errorf()\` with \`%w\` verb to wrap ErrNotFound
3. Include entity identifier in error message
4. Format as: \`"entity <id>: %w"\`

**Example:**
\`\`\`go
var ErrNotFound = errors.New("not found")

func GetUser(id string) (*User, error) {
    user, err := db.Query(id)
    if err == sql.ErrNoRows {
        return nil, FormatNotFound(id)
    }
    return user, err
}

// Error message: "entity user-123: not found"
// errors.Is(err, ErrNotFound) returns true
\`\`\`

**Constraints:**
- Must use %w verb (not %v) to preserve error chain
- Must include id parameter in message`,
						initialCode: `package errorsx

import (
	"errors"
	"fmt"
)

var ErrNotFound = errors.New("not found")

// TODO: Implement FormatNotFound
func FormatNotFound(id string) error {
	panic("TODO")
}`,
						solutionCode: `package errorsx

import (
	"errors"
	"fmt"
)

var ErrNotFound = errors.New("not found")

func FormatNotFound(id string) error {
	return fmt.Errorf("entity %s: %w", id, ErrNotFound)  // %w wraps error, preserving chain
}`,
						hint1: 'Use fmt.Errorf() to create a formatted error message.',
						hint2: 'Use %w verb instead of %v to wrap ErrNotFound and preserve error chain.',
						whyItMatters: `Domain-specific error formatting adds context while preserving error identity for inspection.

**Why Format Errors:**
- **User-Friendly Messages:** "User user-123 not found" vs "not found"
- **Debugging Context:** Know which entity failed without full stack trace
- **Error Identity:** \`errors.Is()\` still works through formatting
- **Logging:** Structured log fields from formatted messages

**Production Pattern:**
\`\`\`go
// Error formatting functions
func FormatNotFound(entityType, id string) error {
    return fmt.Errorf("%s %s: %w", entityType, id, ErrNotFound)
}

func FormatUnauthorized(action, resource string) error {
    return fmt.Errorf("unauthorized %s on %s: %w", action, resource, ErrUnauthorized)
}

func FormatValidation(field, reason string) error {
    return fmt.Errorf("validation failed on %s: %s: %w", field, reason, ErrValidation)
}

// Repository layer
func (r *UserRepo) GetByID(id string) (*User, error) {
    user, err := r.db.QueryRow("SELECT * FROM users WHERE id = $1", id).Scan(&user)
    if err == sql.ErrNoRows {
        return nil, FormatNotFound("user", id)  // Context: which user
    }
    return user, err
}

// Service layer
func (s *Service) DeleteUser(userID, actorID string) error {
    if !s.auth.CanDelete(actorID, userID) {
        return FormatUnauthorized("delete", fmt.Sprintf("user %s", userID))
    }

    err := s.repo.Delete(userID)
    if IsNotFound(err) {
        // Error already has context from repo
        return err
    }
    return err
}

// HTTP handler
func (h *Handler) GetUser(w http.ResponseWriter, r *http.Request) {
    id := r.URL.Query().Get("id")
    user, err := h.service.GetUser(id)

    if err != nil {
        if IsNotFound(err) {
            // Error message: "user user-123: not found"
            http.Error(w, err.Error(), 404)
            return
        }
        http.Error(w, err.Error(), 500)
        return
    }

    json.NewEncoder(w).Encode(user)
}
\`\`\`

**Real-World Benefits:**
- **Client Errors:** Return formatted message directly to clients
- **Logs:** "entity order-456: not found" is more actionable than "not found"
- **Metrics:** Extract entity types from error messages for dashboards
- **Debugging:** Immediately know which resource failed

**%w vs %v:**
- \`%w\` preserves error chain - \`errors.Is()\` and \`errors.As()\` still work
- \`%v\` converts to string - loses error identity
- Always use \`%w\` when wrapping errors

**Standard Library:**
- \`os.PathError\` wraps system errors with file path context
- \`net.OpError\` wraps network errors with operation context
- \`json.UnmarshalTypeError\` wraps errors with type context`,
						translations: {
							ru: {
								title: 'Доменно-специфичное форматирование ошибок',
								description: `Реализуйте **FormatNotFound()** для создания доменно-специфичных ошибок NotFound.

**Требования:**
1. Создайте функцию \`FormatNotFound(id string) error\`
2. Используйте \`fmt.Errorf()\` с глаголом \`%w\` для обертывания ErrNotFound
3. Включите идентификатор сущности в сообщение
4. Формат: \`"entity <id>: %w"\`

**Пример:**
\`\`\`go
var ErrNotFound = errors.New("not found")

func GetUser(id string) (*User, error) {
    user, err := db.Query(id)
    if err == sql.ErrNoRows {
        return nil, FormatNotFound(id)
    }
    return user, err
}

// Сообщение: "entity user-123: not found"
\`\`\`

**Ограничения:**
- Должна использовать %w (не %v)
- Должна включать параметр id`,
								hint1: 'Используйте fmt.Errorf() для создания форматированного сообщения.',
								hint2: 'Используйте глагол %w вместо %v для сохранения цепочки ошибок.',
								whyItMatters: `Доменно-специфичное форматирование добавляет контекст, сохраняя идентичность ошибки.

**Почему важно:**
- **Понятные сообщения:** "User user-123 not found" вместо "not found"
- **Контекст отладки:** Знаете какая сущность упала
- **Идентичность ошибки:** \`errors.Is()\` все еще работает
- **Логирование:** Структурированные поля из сообщений

**%w vs %v:**
- \`%w\` сохраняет цепочку - \`errors.Is()\` работает
- \`%v\` преобразует в строку - теряет идентичность
- Всегда используйте \`%w\` при обертывании`
							},
							uz: {
								title: 'Domen-spetsifik error formatlash',
								description: `Domen-spetsifik NotFound xatolarini yaratish uchun **FormatNotFound()** ni amalga oshiring.

**Talablar:**
1. \`FormatNotFound(id string) error\` funksiyasini yarating
2. ErrNotFound ni o rash uchun \`%w\` bilan \`fmt.Errorf()\` dan foydalaning
3. Xabar ichiga entity identifikatorini qo shing
4. Format: \`"entity <id>: %w"\`

**Misol:**
\`\`\`go
var ErrNotFound = errors.New("not found")

func GetUser(id string) (*User, error) {
    user, err := db.Query(id)
    if err == sql.ErrNoRows {
        return nil, FormatNotFound(id)
    }
    return user, err
}

// Xabar: "entity user-123: not found"
\`\`\`

**Cheklovlar:**
- %w (%v emas) dan foydalanishi kerak
- id parametrini qo shishi kerak`,
								hint1: 'Formatlangan xabar yaratish uchun fmt.Errorf() dan foydalaning.',
								hint2: 'Error zanjirini saqlash uchun %v o rniga %w dan foydalaning.',
								whyItMatters: `Domen-spetsifik formatlash kontekst qo shadi, xato identifikatsiyasini saqlaydi.

**Nima uchun muhim:**
- **Tushunarli xabarlar:** "User user-123 not found" vs "not found"
- **Debug konteksti:** Qaysi entity ishlamay qolganini bilasiz
- **Error identifikatsiyasi:** \`errors.Is()\` hali ham ishlaydi
- **Logging:** Xabarlardan strukturali fieldlar

**%w vs %v:**
- \`%w\` zanjirni saqlaydi - \`errors.Is()\` ishlaydi
- \`%v\` stringga o zgartiradi - identifikatsiyani yo qotadi
- Xatolarni o rashda doim %w dan foydalaning`
							}
						}
					}
				]
			}
		]
	}
];
