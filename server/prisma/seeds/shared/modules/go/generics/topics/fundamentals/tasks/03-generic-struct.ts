import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-generics-generic-struct',
    title: 'Generic Data Structures',
    difficulty: 'medium',
    tags: ['go', 'generics', 'struct', 'stack', 'data-structures'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Build type-safe data structures using generic types.

Generic structs allow you to create reusable data structures that work with any type. This is perfect for implementing common data structures like stacks, queues, linked lists, etc.

**Task:** Implement a generic \`Stack[T]\` data structure with \`Push\`, \`Pop\`, and \`IsEmpty\` methods.

**Requirements:**
- Define a \`Stack[T any]\` struct with a slice field to store elements
- Implement \`Push(item T)\` to add an element to the top
- Implement \`Pop() (T, bool)\` to remove and return the top element (returns false if empty)
- Implement \`IsEmpty() bool\` to check if the stack is empty
- Use pointer receivers for methods that modify the stack

**Example Usage:**
\`\`\`go
stack := &Stack[int]{}
stack.Push(1)
stack.Push(2)
stack.Push(3)

val, ok := stack.Pop()  // val = 3, ok = true
val, ok = stack.Pop()   // val = 2, ok = true
\`\`\``,
    initialCode: `package generics

// TODO: Define a generic Stack struct
type Stack[T any] struct {
    // Add a field to store items
}

// TODO: Implement Push method to add an item to the stack
func (s *Stack[T]) Push(item T) {
    panic("TODO: implement Push")
}

// TODO: Implement Pop method to remove and return the top item
// Returns the item and true if successful, or zero value and false if empty
func (s *Stack[T]) Pop() (T, bool) {
    panic("TODO: implement Pop")
}

// TODO: Implement IsEmpty method to check if stack is empty
func (s *Stack[T]) IsEmpty() bool {
    panic("TODO: implement IsEmpty")
}`,
    solutionCode: `package generics

// Stack is a generic LIFO data structure
type Stack[T any] struct {
    items []T
}

// Push adds an item to the top of the stack
func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

// Pop removes and returns the top item from the stack
func (s *Stack[T]) Pop() (T, bool) {
    if s.IsEmpty() {
        var zero T
        return zero, false
    }

    index := len(s.items) - 1
    item := s.items[index]
    s.items = s.items[:index]

    return item, true
}

// IsEmpty returns true if the stack has no items
func (s *Stack[T]) IsEmpty() bool {
    return len(s.items) == 0
}`,
    testCode: `package generics

import (
	"testing"
)

func Test1(t *testing.T) {
	// Test push and pop with integers
	stack := &Stack[int]{}
	stack.Push(1)
	stack.Push(2)
	stack.Push(3)

	val, ok := stack.Pop()
	if !ok || val != 3 {
		t.Errorf("expected 3, got %v (ok=%v)", val, ok)
	}
}

func Test2(t *testing.T) {
	// Test IsEmpty on new stack
	stack := &Stack[int]{}
	if !stack.IsEmpty() {
		t.Errorf("expected stack to be empty")
	}
}

func Test3(t *testing.T) {
	// Test IsEmpty after push
	stack := &Stack[string]{}
	stack.Push("test")
	if stack.IsEmpty() {
		t.Errorf("expected stack to not be empty")
	}
}

func Test4(t *testing.T) {
	// Test pop on empty stack
	stack := &Stack[int]{}
	val, ok := stack.Pop()
	if ok {
		t.Errorf("expected ok=false on empty stack, got ok=true with val=%v", val)
	}
}

func Test5(t *testing.T) {
	// Test LIFO order
	stack := &Stack[int]{}
	stack.Push(1)
	stack.Push(2)
	stack.Push(3)

	val1, _ := stack.Pop()
	val2, _ := stack.Pop()
	val3, _ := stack.Pop()

	if val1 != 3 || val2 != 2 || val3 != 1 {
		t.Errorf("expected LIFO order 3,2,1, got %v,%v,%v", val1, val2, val3)
	}
}

func Test6(t *testing.T) {
	// Test with strings
	stack := &Stack[string]{}
	stack.Push("first")
	stack.Push("second")

	val, ok := stack.Pop()
	if !ok || val != "second" {
		t.Errorf("expected 'second', got %v", val)
	}
}

func Test7(t *testing.T) {
	// Test IsEmpty after pop all items
	stack := &Stack[int]{}
	stack.Push(1)
	stack.Push(2)
	stack.Pop()
	stack.Pop()

	if !stack.IsEmpty() {
		t.Errorf("expected stack to be empty after popping all items")
	}
}

func Test8(t *testing.T) {
	// Test single item
	stack := &Stack[float64]{}
	stack.Push(3.14)

	val, ok := stack.Pop()
	if !ok || val != 3.14 {
		t.Errorf("expected 3.14, got %v", val)
	}
}

func Test9(t *testing.T) {
	// Test multiple pushes and pops
	stack := &Stack[int]{}
	stack.Push(1)
	stack.Pop()
	stack.Push(2)
	stack.Push(3)

	val, ok := stack.Pop()
	if !ok || val != 3 {
		t.Errorf("expected 3, got %v", val)
	}
}

func Test10(t *testing.T) {
	// Test zero value on empty pop
	stack := &Stack[int]{}
	val, ok := stack.Pop()
	if ok || val != 0 {
		t.Errorf("expected zero value (0) and ok=false, got val=%v, ok=%v", val, ok)
	}
}`,
    hint1: `Use a slice \`[]T\` to store the stack items. Use \`append\` for Push.`,
    hint2: `For Pop, check if the stack is empty first. Use slicing \`s.items[:len(s.items)-1]\` to remove the last element.`,
    whyItMatters: `Generic data structures eliminate the need for type-specific implementations or unsafe interface{} usage. You get compile-time type safety with zero runtime overhead.

**Production Pattern:**

\`\`\`go
// WITHOUT generics: Separate implementations for each type
type IntStack struct {
    items []int
}

type StringStack struct {
    items []string
}

type UserStack struct {
    items []User
}
// ... need copies for each type!

// WITH generics: One implementation for all types
type Stack[T any] struct {
    items []T
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, true
}

// Usage
intStack := &Stack[int]{}
stringStack := &Stack[string]{}
userStack := &Stack[User]{}
\`\`\`

**Practical Benefits:**

1. **Less duplication**: One implementation instead of many copies
2. **Type safety**: Compiler guarantees type correctness
3. **Reusability**: Works with any type without changes
4. **Performance**: No runtime overhead like with interface{}`,
    order: 2,
    translations: {
        ru: {
            title: 'Обобщённые структуры данных',
            solutionCode: `package generics

// Stack - обобщенная структура данных LIFO
type Stack[T any] struct {
    items []T
}

// Push добавляет элемент на верх стека
func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

// Pop удаляет и возвращает верхний элемент стека
func (s *Stack[T]) Pop() (T, bool) {
    if s.IsEmpty() {
        var zero T
        return zero, false
    }

    index := len(s.items) - 1
    item := s.items[index]
    s.items = s.items[:index]

    return item, true
}

// IsEmpty возвращает true, если стек пуст
func (s *Stack[T]) IsEmpty() bool {
    return len(s.items) == 0
}`,
            description: `Создавайте типобезопасные структуры данных, используя обобщенные типы.

Обобщенные структуры позволяют создавать переиспользуемые структуры данных, которые работают с любым типом. Это идеально подходит для реализации общих структур данных, таких как стеки, очереди, связанные списки и т.д.

**Задача:** Реализуйте обобщенную структуру данных \`Stack[T]\` с методами \`Push\`, \`Pop\` и \`IsEmpty\`.

**Требования:**
- Определите структуру \`Stack[T any]\` с полем среза для хранения элементов
- Реализуйте \`Push(item T)\` для добавления элемента на верх
- Реализуйте \`Pop() (T, bool)\` для удаления и возврата верхнего элемента (возвращает false, если пусто)
- Реализуйте \`IsEmpty() bool\` для проверки, пуст ли стек
- Используйте указательные приемники для методов, изменяющих стек

**Пример использования:**
\`\`\`go
stack := &Stack[int]{}
stack.Push(1)
stack.Push(2)
stack.Push(3)

val, ok := stack.Pop()  // val = 3, ok = true
val, ok = stack.Pop()   // val = 2, ok = true
\`\`\``,
            hint1: `Используйте срез \`[]T\` для хранения элементов стека. Используйте \`append\` для Push.`,
            hint2: `Для Pop сначала проверьте, пуст ли стек. Используйте срезы \`s.items[:len(s.items)-1]\` для удаления последнего элемента.`,
            whyItMatters: `Обобщенные структуры данных устраняют необходимость в типоспецифичных реализациях или небезопасном использовании interface{}. Вы получаете безопасность типов на этапе компиляции без накладных расходов во время выполнения.

**Продакшен паттерн:**

\`\`\`go
// БЕЗ generics: Отдельные реализации для каждого типа
type IntStack struct {
    items []int
}

type StringStack struct {
    items []string
}

type UserStack struct {
    items []User
}
// ... нужны копии для каждого типа!

// С generics: Одна реализация для всех типов
type Stack[T any] struct {
    items []T
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, true
}

// Использование
intStack := &Stack[int]{}
stringStack := &Stack[string]{}
userStack := &Stack[User]{}
\`\`\`

**Практические преимущества:**

1. **Меньше дублирования**: Одна имплементация вместо множества копий
2. **Типобезопасность**: Компилятор гарантирует корректность типов
3. **Переиспользуемость**: Работает с любым типом без изменений
4. **Производительность**: Нет runtime overhead, как с interface{}`
        },
        uz: {
            title: 'Generic ma\'lumot tuzilmalari',
            solutionCode: `package generics

// Stack - umumiy LIFO ma'lumot tuzilmasi
type Stack[T any] struct {
    items []T
}

// Push element stackning tepasiga qo'shadi
func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

// Pop stackning tepasidagi elementni o'chiradi va qaytaradi
func (s *Stack[T]) Pop() (T, bool) {
    if s.IsEmpty() {
        var zero T
        return zero, false
    }

    index := len(s.items) - 1
    item := s.items[index]
    s.items = s.items[:index]

    return item, true
}

// IsEmpty stack bo'sh bo'lsa true qaytaradi
func (s *Stack[T]) IsEmpty() bool {
    return len(s.items) == 0
}`,
            description: `Umumiy tiplardan foydalanib tip-xavfsiz ma'lumot tuzilmalarini yarating.

Umumiy strukturalar har qanday tip bilan ishlaydigan qayta foydalaniladigan ma'lumot tuzilmalarini yaratish imkonini beradi. Bu stacklar, navbatlar, bog'langan ro'yxatlar va boshqalar kabi umumiy ma'lumot tuzilmalarini amalga oshirish uchun juda mos keladi.

**Vazifa:** \`Push\`, \`Pop\` va \`IsEmpty\` metodlari bilan umumiy \`Stack[T]\` ma'lumot tuzilmasini yarating.

**Talablar:**
- Elementlarni saqlash uchun srez maydoni bilan \`Stack[T any]\` strukturasini aniqlang
- Tepaga element qo'shish uchun \`Push(item T)\` ni amalga oshiring
- Tepasidagi elementni o'chirish va qaytarish uchun \`Pop() (T, bool)\` ni amalga oshiring (bo'sh bo'lsa false qaytaradi)
- Stack bo'sh yoki yo'qligini tekshirish uchun \`IsEmpty() bool\` ni amalga oshiring
- Stackni o'zgartiruvchi metodlar uchun ko'rsatgich qabul qiluvchilardan foydalaning

**Foydalanish misoli:**
\`\`\`go
stack := &Stack[int]{}
stack.Push(1)
stack.Push(2)
stack.Push(3)

val, ok := stack.Pop()  // val = 3, ok = true
val, ok = stack.Pop()   // val = 2, ok = true
\`\`\``,
            hint1: `Stack elementlarini saqlash uchun \`[]T\` srezdan foydalaning. Push uchun \`append\` dan foydalaning.`,
            hint2: `Pop uchun avval stackning bo'sh yoki yo'qligini tekshiring. Oxirgi elementni o'chirish uchun \`s.items[:len(s.items)-1]\` srezlashdan foydalaning.`,
            whyItMatters: `Umumiy ma'lumot tuzilmalari tipga xos implementatsiyalar yoki xavfli interface{} foydalanish zaruriyatini yo'q qiladi. Siz kompilyatsiya vaqtida tip xavfsizligini olasiz va runtime qo'shimcha xarajatlari yo'q.

**Ishlab chiqarish patterni:**

\`\`\`go
// Genericlarsiz: Har bir tip uchun alohida implementatsiyalar
type IntStack struct {
    items []int
}

type StringStack struct {
    items []string
}

type UserStack struct {
    items []User
}
// ... har bir tip uchun nusxalar kerak!

// Genericlar bilan: Barcha tiplar uchun bitta implementatsiya
type Stack[T any] struct {
    items []T
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, true
}

// Foydalanish
intStack := &Stack[int]{}
stringStack := &Stack[string]{}
userStack := &Stack[User]{}
\`\`\`

**Amaliy foydalari:**

1. **Kamroq dublikatsiya**: Ko'p nusxalar o'rniga bitta implementatsiya
2. **Tip xavfsizligi**: Kompilyator tiplarning to'g'riligini kafolatlaydi
3. **Qayta foydalanish**: O'zgarishsiz har qanday tip bilan ishlaydi
4. **Ishlash samaradorligi**: interface{} kabi runtime overhead yo'q`
        }
    }
};

export default task;
