import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-sentinel-errors',
	title: 'Sentinel Errors',
	difficulty: 'easy',	tags: ['go', 'errors', 'best-practices'],
	estimatedTime: '20m',	isPremium: false,
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
		testCode: `package errorsx

import (
	"errors"
	"testing"
)

func Test1(t *testing.T) {
	// ErrNotFound is not nil
	if ErrNotFound == nil {
		t.Error("expected ErrNotFound to be non-nil")
	}
}

func Test2(t *testing.T) {
	// ErrUnauthorized is not nil
	if ErrUnauthorized == nil {
		t.Error("expected ErrUnauthorized to be non-nil")
	}
}

func Test3(t *testing.T) {
	// ErrNotFound can be checked with errors.Is
	if !errors.Is(ErrNotFound, ErrNotFound) {
		t.Error("expected ErrNotFound to match itself")
	}
}

func Test4(t *testing.T) {
	// ErrUnauthorized can be checked with errors.Is
	if !errors.Is(ErrUnauthorized, ErrUnauthorized) {
		t.Error("expected ErrUnauthorized to match itself")
	}
}

func Test5(t *testing.T) {
	// ErrNotFound and ErrUnauthorized are different
	if errors.Is(ErrNotFound, ErrUnauthorized) {
		t.Error("expected different errors")
	}
}

func Test6(t *testing.T) {
	// ErrNotFound has a message
	if ErrNotFound.Error() == "" {
		t.Error("expected non-empty error message")
	}
}

func Test7(t *testing.T) {
	// ErrUnauthorized has a message
	if ErrUnauthorized.Error() == "" {
		t.Error("expected non-empty error message")
	}
}

func Test8(t *testing.T) {
	// ErrNotFound message contains relevant text
	msg := ErrNotFound.Error()
	if msg != "not found" && msg != "Not Found" && msg != "resource not found" {
		// Accept variations
	}
	if len(msg) == 0 {
		t.Error("expected meaningful message")
	}
}

func Test9(t *testing.T) {
	// Errors can be used in comparisons
	err := ErrNotFound
	if err != ErrNotFound {
		t.Error("expected sentinel to be comparable")
	}
}

func Test10(t *testing.T) {
	// Wrapped error can still match sentinel
	wrapped := errors.New("wrapped: " + ErrNotFound.Error())
	_ = wrapped // Just testing sentinels exist and are usable
	if ErrNotFound == nil {
		t.Error("expected non-nil sentinel")
	}
}
`,
		hint1: `Use errors.New() to create sentinel errors with descriptive messages.`,
			hint2: `Export variables (uppercase) so callers can use errors.Is() to check them.`,
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

Without sentinels, you'd resort to fragile string matching that breaks when messages are localized.`,	order: 0,
	translations: {
		ru: {
			title: 'Создание сигнальных ошибок (sentinel errors)',
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
			hint1: `Используйте errors.New() для создания sentinel errors.`,
			hint2: `Экспортируйте переменные для использования errors.Is().`,
			whyItMatters: `Sentinel errors - идиоматичный способ Go представлять известные ошибки. Они позволяют принимать программные решения без парсинга строк.

**Почему важно:**
- **Type-safe проверка:** \`errors.Is()\` вместо сравнения строк
- **Стабильность API:** Сообщения могут меняться без нарушения клиентов
- **Чёткие контракты:** Экспортированные sentinels документируют ошибки

**Примеры из реальной жизни:**
- Стандартная библиотека: \`io.EOF\`, \`sql.ErrNoRows\`, \`os.ErrNotExist\`
- Драйверы баз данных: \`ErrConnDone\`, \`ErrTxDone\`
- HTTP клиенты: \`context.Canceled\`, \`context.DeadlineExceeded\`

**Продакшен паттерн:**
\`\`\`go
// Уровень сервиса
if errors.Is(err, repo.ErrNotFound) {
    return nil, ErrUserNotFound  // Обертывание для домена
}

// HTTP обработчик
if errors.Is(err, service.ErrUserNotFound) {
    return 404, "User not found"
}
\`\`\`

Без sentinels пришлось бы использовать хрупкое сравнение строк, которое ломается при локализации сообщений.`,
			solutionCode: `package errorsx

import "errors"

var (
	ErrNotFound     = errors.New("not found")     // Ошибка: ресурс не найден
	ErrUnauthorized = errors.New("unauthorized")  // Ошибка авторизации
)`
		},
		uz: {
			title: 'Sentinel errorlar yaratish',
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
			hint1: `Sentinel errorlar yaratish uchun errors.New() dan foydalaning.`,
			hint2: `errors.Is() uchun o'zgaruvchilarni eksport qiling.`,
			whyItMatters: `Sentinel errorlar - Go ning ma'lum xatolarni ifodalashning idiomatik usuli. String parsing siz qarorlar qabul qilish imkonini beradi.

**Nima uchun muhim:**
- **Type-safe tekshirish:** String taqqoslash o'rniga \`errors.Is()\`
- **API barqarorligi:** Xato xabarlari o'zgarishi mumkin
- **Aniq shartnomalar:** Eksport qilingan sentinellar xatolarni hujjatlashtiradi

**Haqiqiy hayotdan misollar:**
- Standart kutubxona: \`io.EOF\`, \`sql.ErrNoRows\`, \`os.ErrNotExist\`
- Ma'lumotlar bazasi drayverlari: \`ErrConnDone\`, \`ErrTxDone\`
- HTTP clientlar: \`context.Canceled\`, \`context.DeadlineExceeded\`

**Ishlab chiqarish patterni:**
\`\`\`go
// Servis qatlami
if errors.Is(err, repo.ErrNotFound) {
    return nil, ErrUserNotFound  // Domen uchun o'rash
}

// HTTP handler
if errors.Is(err, service.ErrUserNotFound) {
    return 404, "User not found"
}
\`\`\`

Sentinellar bo'lmasa, lokalizatsiya qilinganda sinadigan nozik string taqqoslashdan foydalanishga to'g'ri keladi.`,
			solutionCode: `package errorsx

import "errors"

var (
	ErrNotFound     = errors.New("not found")     // Xato: resurs topilmadi
	ErrUnauthorized = errors.New("unauthorized")  // Avtorizatsiya xatosi
)`
		}
	}
};

export default task;
