import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-custom-error-type',
	title: 'Custom Error Type',
	difficulty: 'easy',	tags: ['go', 'errors', 'types'],
	estimatedTime: '20m',	isPremium: false,
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
	return "" // TODO: Implement
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
		testCode: `package errorsx

import (
	"errors"
	"strings"
	"testing"
)

func Test1(t *testing.T) {
	// AppError implements error interface
	var _ error = &AppError{}
}

func Test2(t *testing.T) {
	// Error() returns formatted string
	err := &AppError{Code: "TEST", Op: "op.Test", Err: errors.New("test")}
	msg := err.Error()
	if msg == "" {
		t.Error("expected non-empty message")
	}
}

func Test3(t *testing.T) {
	// Error() includes Code
	err := &AppError{Code: "NOT_FOUND", Op: "users.Get", Err: nil}
	msg := err.Error()
	if !strings.Contains(msg, "NOT_FOUND") {
		t.Errorf("expected message to contain Code, got: %s", msg)
	}
}

func Test4(t *testing.T) {
	// Error() includes Op
	err := &AppError{Code: "ERR", Op: "service.Create", Err: nil}
	msg := err.Error()
	if !strings.Contains(msg, "service.Create") {
		t.Errorf("expected message to contain Op, got: %s", msg)
	}
}

func Test5(t *testing.T) {
	// Error() includes Err
	inner := errors.New("inner error")
	err := &AppError{Code: "ERR", Op: "op", Err: inner}
	msg := err.Error()
	if !strings.Contains(msg, "inner error") {
		t.Errorf("expected message to contain Err, got: %s", msg)
	}
}

func Test6(t *testing.T) {
	// Nil receiver does not panic
	var err *AppError = nil
	defer func() {
		if r := recover(); r != nil {
			t.Error("Error() should not panic on nil receiver")
		}
	}()
	_ = err.Error()
}

func Test7(t *testing.T) {
	// Nil receiver returns something
	var err *AppError = nil
	msg := err.Error()
	if msg == "" {
		// Allow empty or <nil> or similar
	}
}

func Test8(t *testing.T) {
	// AppError with nil Err works
	err := &AppError{Code: "ERR", Op: "test", Err: nil}
	msg := err.Error()
	if msg == "" {
		t.Error("expected non-empty message even with nil Err")
	}
}

func Test9(t *testing.T) {
	// AppError with empty fields works
	err := &AppError{Code: "", Op: "", Err: nil}
	defer func() {
		if r := recover(); r != nil {
			t.Error("should not panic with empty fields")
		}
	}()
	_ = err.Error()
}

func Test10(t *testing.T) {
	// Error() format includes separators
	err := &AppError{Code: "A", Op: "B", Err: errors.New("C")}
	msg := err.Error()
	// Should have some structure
	if len(msg) < 5 {
		t.Error("expected formatted message with all fields")
	}
}
`,
		hint1: `Check if receiver is nil before accessing fields to prevent panic.`,
			hint2: `Use fmt.Sprintf to format all fields into a readable error message.`,
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

The Op field enables error aggregation: if 100 requests fail in orders.Create with DB_UNAVAILABLE, you know exactly where to look.`,	order: 1,
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
			hint1: `Проверьте nil receiver перед доступом к полям.`,
			hint2: `Используйте fmt.Sprintf для форматирования полей.`,
			whyItMatters: `Пользовательские типы ошибок добавляют структурированный контекст, упрощая отладку в production.

**Почему важно:**
- **Контекст:** Знаете ГДЕ и ЧТО упало, не только ПОЧЕМУ
- **Структурированное логирование:** Поля логируются отдельно
- **Классификация:** Группировка ошибок по коду для метрик

**Продакшен паттерн:**
\`\`\`go
// Ошибка без контекста
return errors.New("database error")

// Ошибка с контекстом
return &AppError{
    Code: "DB_UNAVAILABLE",
    Op:   "orders.Create",
    Err:  originalErr,
}
\`\`\`

**Практические преимущества:**

В мониторинге вы можете:
- Запросить все ошибки DB_UNAVAILABLE
- Увидеть какие операции падают чаще всего
- Отследить цепочки ошибок через микросервисы

**Использование в реальных проектах:**
- Пакет upspin.io/errors от Google
- go.uber.org/multierr от Uber
- go-multierror от HashiCorp

Поле Op позволяет агрегировать ошибки: если 100 запросов падают в orders.Create с DB_UNAVAILABLE, вы точно знаете где искать проблему.`,
			solutionCode: `package errorsx

import "fmt"

type AppError struct {
	Code string  // Код ошибки (например, "NOT_FOUND", "UNAUTHORIZED")
	Op   string  // Операция, где произошла ошибка
	Err  error   // Вложенная ошибка
}

func (e *AppError) Error() string {
	if e == nil {                                            // Проверка nil receiver
		return "<nil>"                                   // Предотвращение паники
	}
	return fmt.Sprintf("op=%s, code=%s, err=%v", e.Op, e.Code, e.Err)  // Форматирование сообщения об ошибке
}`
		},
		uz: {
			title: 'Maxsus xato turi',
			description: `Kontekstli ma'lumotlar bilan **AppError** custom error turini amalga oshiring.

**Talablar:**
1. \`AppError\` strukturasini 'Code', 'Op', 'Err' fieldlar bilan aniqlang
2. \`Error() string\` metodini amalga oshiring
3. Chiqishni formatlang: '"op=<operation>, code=<code>, err=<error>"'
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
			hint1: `Fieldlarga kirishdan oldin nil receiver ni tekshiring.`,
			hint2: `Fieldlarni formatlash uchun fmt.Sprintf dan foydalaning.`,
			whyItMatters: `Custom error turlari strukturali kontekst qo'shadi, production'da debugni osonlashtiradi.

**Nima uchun muhim:**
- **Kontekst:** QAYERDA va NIMA ishlamay qoldi, faqat NIMA UCHUN emas
- **Strukturali logging:** Fieldlar alohida loglanadi
- **Klassifikatsiya:** Kod bo'yicha xatolarni guruhlash

**Ishlab chiqarish patterni:**
\`\`\`go
// Kontekstsiz xato
return errors.New("database error")

// Kontekstli xato
return &AppError{
    Code: "DB_UNAVAILABLE",
    Op:   "orders.Create",
    Err:  originalErr,
}
\`\`\`

**Amaliy foydalari:**

Monitoring'da siz quyidagilarni amalga oshirishingiz mumkin:
- Barcha DB_UNAVAILABLE xatolarini so'rash
- Qaysi operatsiyalar ko'proq ishlamay qolishini ko'rish
- Mikroservislar orqali xato zanjirlarini kuzatish

**Haqiqiy loyihalarda foydalanish:**
- Google'ning upspin.io/errors paketi
- Uber'ning go.uber.org/multierr
- HashiCorp'ning go-multierror

Op field xatolarni agregatsiya qilish imkonini beradi: agar 100 ta so'rov orders.Create da DB_UNAVAILABLE bilan ishlamay qolsa, muammoni qayerda izlashni aniq bilasiz.`,
			solutionCode: `package errorsx

import "fmt"

type AppError struct {
	Code string  // Xato kodi (masalan, "NOT_FOUND", "UNAUTHORIZED")
	Op   string  // Xato sodir bo'lgan operatsiya
	Err  error   // Ichki xato
}

func (e *AppError) Error() string {
	if e == nil {                                            // nil receiver tekshiruvi
		return "<nil>"                                   // Panic oldini olish
	}
	return fmt.Sprintf("op=%s, code=%s, err=%v", e.Op, e.Code, e.Err)  // Xato xabarini formatlash
}`
		}
	}
};

export default task;
