import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-generics-generic-result-type',
    title: 'Generic Result Type Pattern',
    difficulty: 'hard',
    tags: ['go', 'generics', 'error-handling', 'patterns', 'functional'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a generic Result type for explicit error handling inspired by Rust and functional programming.

The Result pattern provides an alternative to Go's traditional error handling by wrapping success and error cases in a single type. This makes error handling more explicit and composable.

**Task:** Create a generic \`Result[T, E]\` type and implement methods for working with results:
1. \`Ok[T, E](value T) Result[T, E]\` - creates a successful result
2. \`Err[T, E](err E) Result[T, E]\` - creates an error result
3. \`IsOk() bool\` - checks if result is successful
4. \`IsErr() bool\` - checks if result is an error
5. \`Unwrap() (T, E)\` - extracts the value or error
6. \`UnwrapOr(defaultValue T) T\` - returns value or default if error

**Requirements:**
- Use a struct with boolean flag and two fields for value and error
- Implement all methods as shown above
- Both T and E can be any type

**Example Usage:**
\`\`\`go
// Successful result
r1 := Ok[int, string](42)
if r1.IsOk() {
    val, _ := r1.Unwrap()
    fmt.Println(val) // 42
}

// Error result
r2 := Err[int, string]("something went wrong")
if r2.IsErr() {
    _, err := r2.Unwrap()
    fmt.Println(err) // something went wrong
}

// Using UnwrapOr
val := r2.UnwrapOr(0)
fmt.Println(val) // 0
\`\`\``,
    initialCode: `package generics

// TODO: Define a generic Result type that can hold either a value or an error
type Result[T any, E any] struct {
    // Add fields here
}

// TODO: Implement Ok function to create a successful result
func Ok[T any, E any](value T) Result[T, E] {
    panic("TODO: implement Ok")
}

// TODO: Implement Err function to create an error result
func Err[T any, E any](err E) Result[T, E] {
    panic("TODO: implement Err")
}

// TODO: Implement IsOk method to check if result is successful
func (r Result[T, E]) IsOk() bool {
    panic("TODO: implement IsOk")
}

// TODO: Implement IsErr method to check if result is an error
func (r Result[T, E]) IsErr() bool {
    panic("TODO: implement IsErr")
}

// TODO: Implement Unwrap method to get the value or error
func (r Result[T, E]) Unwrap() (T, E) {
    panic("TODO: implement Unwrap")
}

// TODO: Implement UnwrapOr method to get the value or a default
func (r Result[T, E]) UnwrapOr(defaultValue T) T {
    panic("TODO: implement UnwrapOr")
}`,
    solutionCode: `package generics

// Result represents either a successful value or an error
type Result[T any, E any] struct {
    value T
    err   E
    isOk  bool
}

// Ok creates a successful Result containing a value
func Ok[T any, E any](value T) Result[T, E] {
    return Result[T, E]{
        value: value,
        isOk:  true,
    }
}

// Err creates an error Result containing an error
func Err[T any, E any](err E) Result[T, E] {
    return Result[T, E]{
        err:  err,
        isOk: false,
    }
}

// IsOk returns true if the Result contains a value
func (r Result[T, E]) IsOk() bool {
    return r.isOk
}

// IsErr returns true if the Result contains an error
func (r Result[T, E]) IsErr() bool {
    return !r.isOk
}

// Unwrap returns the value and error from the Result
func (r Result[T, E]) Unwrap() (T, E) {
    return r.value, r.err
}

// UnwrapOr returns the value if Ok, otherwise returns the default value
func (r Result[T, E]) UnwrapOr(defaultValue T) T {
    if r.isOk {
        return r.value
    }
    return defaultValue
}`,
    testCode: `package generics

import (
	"testing"
)

func Test1(t *testing.T) {
	// Test Ok result with integer
	r := Ok[int, string](42)
	if !r.IsOk() {
		t.Errorf("expected IsOk to be true")
	}
	if r.IsErr() {
		t.Errorf("expected IsErr to be false")
	}
}

func Test2(t *testing.T) {
	// Test Err result with string
	r := Err[int, string]("error message")
	if r.IsOk() {
		t.Errorf("expected IsOk to be false")
	}
	if !r.IsErr() {
		t.Errorf("expected IsErr to be true")
	}
}

func Test3(t *testing.T) {
	// Test Unwrap on Ok result
	r := Ok[int, string](100)
	val, err := r.Unwrap()
	if val != 100 {
		t.Errorf("expected value 100, got %v", val)
	}
	if err != "" {
		t.Errorf("expected empty error, got %v", err)
	}
}

func Test4(t *testing.T) {
	// Test Unwrap on Err result
	r := Err[int, string]("failed")
	val, err := r.Unwrap()
	if val != 0 {
		t.Errorf("expected zero value, got %v", val)
	}
	if err != "failed" {
		t.Errorf("expected error 'failed', got %v", err)
	}
}

func Test5(t *testing.T) {
	// Test UnwrapOr on Ok result
	r := Ok[int, string](50)
	val := r.UnwrapOr(99)
	if val != 50 {
		t.Errorf("expected 50, got %v", val)
	}
}

func Test6(t *testing.T) {
	// Test UnwrapOr on Err result
	r := Err[int, string]("error")
	val := r.UnwrapOr(99)
	if val != 99 {
		t.Errorf("expected default value 99, got %v", val)
	}
}

func Test7(t *testing.T) {
	// Test with float values
	r := Ok[float64, error](3.14)
	val, _ := r.Unwrap()
	if val != 3.14 {
		t.Errorf("expected 3.14, got %v", val)
	}
}

func Test8(t *testing.T) {
	// Test with boolean values
	r := Ok[bool, string](true)
	if !r.IsOk() {
		t.Errorf("expected IsOk to be true")
	}
	val, _ := r.Unwrap()
	if !val {
		t.Errorf("expected true, got false")
	}
}

func Test9(t *testing.T) {
	// Test with custom error type
	r := Err[string, int](404)
	_, err := r.Unwrap()
	if err != 404 {
		t.Errorf("expected error code 404, got %v", err)
	}
}

func Test10(t *testing.T) {
	// Test UnwrapOr with zero value default
	r := Err[int, string]("failed")
	val := r.UnwrapOr(0)
	if val != 0 {
		t.Errorf("expected 0, got %v", val)
	}
}`,
    hint1: `Use a struct with three fields: value of type T, err of type E, and a boolean flag isOk to track whether it's a success or error.`,
    hint2: `In Ok(), set isOk to true and store the value. In Err(), set isOk to false and store the error. Use the isOk flag in all checking methods.`,
    whyItMatters: `The Result pattern makes error handling more explicit and composable. It's particularly useful in functional-style code, chaining operations, and when you want to avoid Go's traditional multiple return values for errors. This pattern is common in modern languages like Rust and is gaining popularity in Go.

**Production Pattern:**

\`\`\`go
// Traditional Go style
func ParseInt(s string) (int, error) {
    return strconv.Atoi(s)
}

// Result style
func ParseIntResult(s string) Result[int, error] {
    val, err := strconv.Atoi(s)
    if err != nil {
        return Err[int, error](err)
    }
    return Ok[int, error](val)
}

// Usage
r := ParseIntResult("42")
if r.IsOk() {
    val, _ := r.Unwrap()
    fmt.Println(val)  // 42
}

// With default value
val := r.UnwrapOr(0)
\`\`\`

**Practical Benefits:**

1. **Explicit handling**: Cannot ignore errors
2. **Functional style**: Convenient for operation chains
3. **Universality**: Works with any value and error types
4. **Readability**: Clear success/error visibility`,
    order: 7,
    translations: {
        ru: {
            title: 'Паттерн обобщённого Result типа',
            solutionCode: `package generics

// Result представляет либо успешное значение, либо ошибку
type Result[T any, E any] struct {
    value T
    err   E
    isOk  bool
}

// Ok создает успешный Result, содержащий значение
func Ok[T any, E any](value T) Result[T, E] {
    return Result[T, E]{
        value: value,
        isOk:  true,
    }
}

// Err создает Result с ошибкой
func Err[T any, E any](err E) Result[T, E] {
    return Result[T, E]{
        err:  err,
        isOk: false,
    }
}

// IsOk возвращает true, если Result содержит значение
func (r Result[T, E]) IsOk() bool {
    return r.isOk
}

// IsErr возвращает true, если Result содержит ошибку
func (r Result[T, E]) IsErr() bool {
    return !r.isOk
}

// Unwrap возвращает значение и ошибку из Result
func (r Result[T, E]) Unwrap() (T, E) {
    return r.value, r.err
}

// UnwrapOr возвращает значение, если Ok, иначе возвращает значение по умолчанию
func (r Result[T, E]) UnwrapOr(defaultValue T) T {
    if r.isOk {
        return r.value
    }
    return defaultValue
}`,
            description: `Реализуйте обобщенный тип Result для явной обработки ошибок, вдохновленный Rust и функциональным программированием.

Паттерн Result предоставляет альтернативу традиционной обработке ошибок в Go, оборачивая случаи успеха и ошибки в единый тип. Это делает обработку ошибок более явной и композируемой.

**Задача:** Создайте обобщенный тип \`Result[T, E]\` и реализуйте методы для работы с результатами:
1. \`Ok[T, E](value T) Result[T, E]\` - создает успешный результат
2. \`Err[T, E](err E) Result[T, E]\` - создает результат с ошибкой
3. \`IsOk() bool\` - проверяет, успешен ли результат
4. \`IsErr() bool\` - проверяет, является ли результат ошибкой
5. \`Unwrap() (T, E)\` - извлекает значение или ошибку
6. \`UnwrapOr(defaultValue T) T\` - возвращает значение или значение по умолчанию при ошибке

**Требования:**
- Используйте структуру с булевым флагом и двумя полями для значения и ошибки
- Реализуйте все методы, как показано выше
- Оба типа T и E могут быть любыми

**Пример использования:**
\`\`\`go
// Успешный результат
r1 := Ok[int, string](42)
if r1.IsOk() {
    val, _ := r1.Unwrap()
    fmt.Println(val) // 42
}

// Результат с ошибкой
r2 := Err[int, string]("something went wrong")
if r2.IsErr() {
    _, err := r2.Unwrap()
    fmt.Println(err) // something went wrong
}

// Использование UnwrapOr
val := r2.UnwrapOr(0)
fmt.Println(val) // 0
\`\`\``,
            hint1: `Используйте структуру с тремя полями: value типа T, err типа E и булевым флагом isOk для отслеживания, является ли это успехом или ошибкой.`,
            hint2: `В Ok() установите isOk в true и сохраните значение. В Err() установите isOk в false и сохраните ошибку. Используйте флаг isOk во всех методах проверки.`,
            whyItMatters: `Паттерн Result делает обработку ошибок более явной и композируемой. Он особенно полезен в коде функционального стиля, цепочках операций и когда вы хотите избежать традиционных множественных возвращаемых значений Go для ошибок. Этот паттерн распространен в современных языках, таких как Rust, и набирает популярность в Go.

**Продакшен паттерн:**

\`\`\`go
// Традиционный Go стиль
func ParseInt(s string) (int, error) {
    return strconv.Atoi(s)
}

// Result стиль
func ParseIntResult(s string) Result[int, error] {
    val, err := strconv.Atoi(s)
    if err != nil {
        return Err[int, error](err)
    }
    return Ok[int, error](val)
}

// Использование
r := ParseIntResult("42")
if r.IsOk() {
    val, _ := r.Unwrap()
    fmt.Println(val)  // 42
}

// С default значением
val := r.UnwrapOr(0)
\`\`\`

**Практические преимущества:**

1. **Явная обработка**: Невозможно игнорировать ошибку
2. **Функциональный стиль**: Удобно для цепочек операций
3. **Универсальность**: Работает с любыми типами значений и ошибок
4. **Читаемость**: Ясно видно успех/ошибку`
        },
        uz: {
            title: 'Generic Result tipi patterni',
            solutionCode: `package generics

// Result muvaffaqiyatli qiymat yoki xatoni ifodalaydi
type Result[T any, E any] struct {
    value T
    err   E
    isOk  bool
}

// Ok qiymat o'z ichiga olgan muvaffaqiyatli Result yaratadi
func Ok[T any, E any](value T) Result[T, E] {
    return Result[T, E]{
        value: value,
        isOk:  true,
    }
}

// Err xato o'z ichiga olgan Result yaratadi
func Err[T any, E any](err E) Result[T, E] {
    return Result[T, E]{
        err:  err,
        isOk: false,
    }
}

// IsOk Result qiymat o'z ichiga olsa true qaytaradi
func (r Result[T, E]) IsOk() bool {
    return r.isOk
}

// IsErr Result xato o'z ichiga olsa true qaytaradi
func (r Result[T, E]) IsErr() bool {
    return !r.isOk
}

// Unwrap Result dan qiymat va xatoni qaytaradi
func (r Result[T, E]) Unwrap() (T, E) {
    return r.value, r.err
}

// UnwrapOr Ok bo'lsa qiymatni, aks holda standart qiymatni qaytaradi
func (r Result[T, E]) UnwrapOr(defaultValue T) T {
    if r.isOk {
        return r.value
    }
    return defaultValue
}`,
            description: `Rust va funksional dasturlashdan ilhomlangan aniq xatolarni qayta ishlash uchun umumiy Result tipini amalga oshiring.

Result naqshi muvaffaqiyat va xato holatlarini bitta tipda o'rab, Go-ning an'anaviy xatolarni qayta ishlashga alternativa taqdim etadi. Bu xatolarni qayta ishlashni yanada aniq va kompozitsion qiladi.

**Vazifa:** Umumiy \`Result[T, E]\` tipini yarating va natijalar bilan ishlash uchun metodlarni amalga oshiring:
1. \`Ok[T, E](value T) Result[T, E]\` - muvaffaqiyatli natija yaratadi
2. \`Err[T, E](err E) Result[T, E]\` - xato natijasini yaratadi
3. \`IsOk() bool\` - natija muvaffaqiyatli yoki yo'qligini tekshiradi
4. \`IsErr() bool\` - natija xato yoki yo'qligini tekshiradi
5. \`Unwrap() (T, E)\` - qiymat yoki xatoni chiqaradi
6. \`UnwrapOr(defaultValue T) T\` - qiymatni yoki xato bo'lsa standart qiymatni qaytaradi

**Talablar:**
- Qiymat va xato uchun ikkita maydon va boolean flag bilan strukturadan foydalaning
- Yuqorida ko'rsatilganidek barcha metodlarni amalga oshiring
- T va E har ikkalasi ham har qanday tip bo'lishi mumkin

**Foydalanish misoli:**
\`\`\`go
// Muvaffaqiyatli natija
r1 := Ok[int, string](42)
if r1.IsOk() {
    val, _ := r1.Unwrap()
    fmt.Println(val) // 42
}

// Xato natijasi
r2 := Err[int, string]("something went wrong")
if r2.IsErr() {
    _, err := r2.Unwrap()
    fmt.Println(err) // something went wrong
}

// UnwrapOr dan foydalanish
val := r2.UnwrapOr(0)
fmt.Println(val) // 0
\`\`\``,
            hint1: `Uchta maydonli strukturadan foydalaning: T tipidagi value, E tipidagi err va bu muvaffaqiyat yoki xato ekanligini kuzatish uchun isOk boolean flag.`,
            hint2: `Ok() da isOk ni true qilib qo'ying va qiymatni saqlang. Err() da isOk ni false qilib qo'ying va xatoni saqlang. Barcha tekshirish metodlarida isOk flagidan foydalaning.`,
            whyItMatters: `Result naqshi xatolarni qayta ishlashni yanada aniq va kompozitsion qiladi. U ayniqsa funksional uslubdagi kodda, operatsiyalar zanjirida va xatolar uchun Go-ning an'anaviy bir nechta qaytariladigan qiymatlaridan qochmoqchi bo'lganingizda foydalidir. Bu naqsh Rust kabi zamonaviy tillarda keng tarqalgan va Go-da ommalashib bormoqda.

**Ishlab chiqarish patterni:**

\`\`\`go
// An'anaviy Go uslubi
func ParseInt(s string) (int, error) {
    return strconv.Atoi(s)
}

// Result uslubi
func ParseIntResult(s string) Result[int, error] {
    val, err := strconv.Atoi(s)
    if err != nil {
        return Err[int, error](err)
    }
    return Ok[int, error](val)
}

// Foydalanish
r := ParseIntResult("42")
if r.IsOk() {
    val, _ := r.Unwrap()
    fmt.Println(val)  // 42
}

// Default qiymat bilan
val := r.UnwrapOr(0)
\`\`\`

**Amaliy foydalari:**

1. **Aniq qayta ishlash**: Xatoni e'tiborsiz qoldirish mumkin emas
2. **Funksional uslub**: Operatsiyalar zanjirlari uchun qulay
3. **Universallik**: Har qanday qiymat va xato turlari bilan ishlaydi
4. **O'qilishi**: Muvaffaqiyat/xato aniq ko'rinadi`
        }
    }
};

export default task;
