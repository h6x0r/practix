import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-datastructsx-generic-stack',
	title: 'Generic Stack Implementation with Type Safety',
	difficulty: 'easy',
	tags: ['go', 'generics', 'stack', 'data-structures', 'lifo'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Build a production-ready generic Stack data structure implementing the LIFO (Last-In-First-Out) principle with full type safety.

**You will implement:**

**Level 1 (Easy) - Core Stack Operations:**
1. **NewStack[T any]() *Stack[T]** - Create new empty stack
2. **Push(value T)** - Add element to top of stack
3. **Pop() (T, bool)** - Remove and return top element (value, ok)
4. **Peek() (T, bool)** - View top element without removing (value, ok)
5. **IsEmpty() bool** - Check if stack has no elements
6. **Size() int** - Return number of elements in stack

**Key Concepts:**
- **LIFO Principle**: Last element added is first one removed
- **Generic Stack**: Works with any type (int, string, struct, interface)
- **Safe Operations**: Return ok flag to prevent panic on empty stack
- **Slice-based**: Uses Go slice as underlying storage
- **O(1) Operations**: Push, Pop, Peek are constant time

**Example Usage:**

\`\`\`go
// Integer stack
intStack := NewStack[int]()
intStack.Push(1)
intStack.Push(2)
intStack.Push(3)

val, ok := intStack.Pop()
// val == 3, ok == true

top, ok := intStack.Peek()
// top == 2, ok == true (not removed)

size := intStack.Size()
// size == 2

isEmpty := intStack.IsEmpty()
// isEmpty == false

// String stack - same interface
strStack := NewStack[string]()
strStack.Push("hello")
strStack.Push("world")

word, ok := strStack.Pop()
// word == "world", ok == true

// Custom type stack
type Task struct {
    ID   int
    Name string
}

taskStack := NewStack[Task]()
taskStack.Push(Task{1, "Write code"})
taskStack.Push(Task{2, "Test code"})

task, ok := taskStack.Pop()
// task == Task{2, "Test code"}, ok == true

// Safe empty stack operations
emptyStack := NewStack[int]()
val, ok := emptyStack.Pop()
// val == 0 (zero value), ok == false

val, ok = emptyStack.Peek()
// val == 0 (zero value), ok == false
\`\`\`

**Constraints:**
- Use slice as underlying storage
- Pop/Peek on empty stack return zero value and false
- Push should handle growing slice automatically
- All operations must be O(1) amortized time
- Stack must be safe for any type T`,
	initialCode: `package datastructsx

// Stack represents a LIFO (Last-In-First-Out) data structure
type Stack[T any] struct {
	// TODO: Add fields for storing elements
}

// TODO: Implement NewStack
// Create and return a new empty stack
func NewStack[T any]() *Stack[T] {
	// TODO: Implement
}

// TODO: Implement Push
// Add element to top of stack
func (s *Stack[T]) Push(value T) {
	// TODO: Implement
}

// TODO: Implement Pop
// Remove and return top element
// Return (zero value, false) if stack is empty
func (s *Stack[T]) Pop() (T, bool) {
	// TODO: Implement
}

// TODO: Implement Peek
// Return top element without removing it
// Return (zero value, false) if stack is empty
func (s *Stack[T]) Peek() (T, bool) {
	// TODO: Implement
}

// TODO: Implement IsEmpty
// Return true if stack has no elements
func (s *Stack[T]) IsEmpty() bool {
	// TODO: Implement
}

// TODO: Implement Size
// Return number of elements in stack
func (s *Stack[T]) Size() int {
	// TODO: Implement
}`,
	solutionCode: `package datastructsx

// Stack represents a LIFO (Last-In-First-Out) data structure
type Stack[T any] struct {
	elements []T
}

func NewStack[T any]() *Stack[T] {
	return &Stack[T]{
		elements: make([]T, 0),
	}
}

func (s *Stack[T]) Push(value T) {
	s.elements = append(s.elements, value)
}

func (s *Stack[T]) Pop() (T, bool) {
	if s.IsEmpty() {
		var zero T
		return zero, false
	}

	index := len(s.elements) - 1
	value := s.elements[index]
	s.elements = s.elements[:index]

	return value, true
}

func (s *Stack[T]) Peek() (T, bool) {
	if s.IsEmpty() {
		var zero T
		return zero, false
	}

	return s.elements[len(s.elements)-1], true
}

func (s *Stack[T]) IsEmpty() bool {
	return len(s.elements) == 0
}

func (s *Stack[T]) Size() int {
	return len(s.elements)
}`,
	testCode: `package datastructsx

import "testing"

func Test1(t *testing.T) {
	// Basic Push and Pop
	s := NewStack[int]()
	s.Push(1)
	s.Push(2)
	val, ok := s.Pop()
	if !ok || val != 2 {
		t.Errorf("expected 2, got %d", val)
	}
}

func Test2(t *testing.T) {
	// Pop returns last pushed
	s := NewStack[int]()
	s.Push(1)
	s.Push(2)
	s.Push(3)
	val, _ := s.Pop()
	if val != 3 {
		t.Errorf("expected 3, got %d", val)
	}
}

func Test3(t *testing.T) {
	// Pop on empty stack
	s := NewStack[int]()
	val, ok := s.Pop()
	if ok || val != 0 {
		t.Errorf("expected 0 and false, got %d %v", val, ok)
	}
}

func Test4(t *testing.T) {
	// Peek returns top without removing
	s := NewStack[int]()
	s.Push(5)
	val, ok := s.Peek()
	if !ok || val != 5 || s.Size() != 1 {
		t.Errorf("expected 5 and size 1, got %d %d", val, s.Size())
	}
}

func Test5(t *testing.T) {
	// IsEmpty
	s := NewStack[int]()
	if !s.IsEmpty() {
		t.Error("expected empty")
	}
	s.Push(1)
	if s.IsEmpty() {
		t.Error("expected not empty")
	}
}

func Test6(t *testing.T) {
	// Size
	s := NewStack[int]()
	if s.Size() != 0 {
		t.Errorf("expected 0, got %d", s.Size())
	}
	s.Push(1)
	s.Push(2)
	if s.Size() != 2 {
		t.Errorf("expected 2, got %d", s.Size())
	}
}

func Test7(t *testing.T) {
	// String stack
	s := NewStack[string]()
	s.Push("hello")
	s.Push("world")
	val, _ := s.Pop()
	if val != "world" {
		t.Errorf("expected world, got %s", val)
	}
}

func Test8(t *testing.T) {
	// Multiple pops
	s := NewStack[int]()
	s.Push(1)
	s.Push(2)
	s.Push(3)
	s.Pop()
	s.Pop()
	val, _ := s.Pop()
	if val != 1 {
		t.Errorf("expected 1, got %d", val)
	}
}

func Test9(t *testing.T) {
	// Peek on empty
	s := NewStack[int]()
	val, ok := s.Peek()
	if ok || val != 0 {
		t.Errorf("expected 0 and false, got %d %v", val, ok)
	}
}

func Test10(t *testing.T) {
	// Push after pop
	s := NewStack[int]()
	s.Push(1)
	s.Pop()
	s.Push(2)
	val, _ := s.Pop()
	if val != 2 {
		t.Errorf("expected 2, got %d", val)
	}
}`,
	hint1: `Define Stack[T any] struct with a slice field: elements []T. NewStack returns &Stack[T]{elements: make([]T, 0)}. Push appends to slice.`,
	hint2: `Pop: Check IsEmpty first, if empty return (var zero T, false). Otherwise get last element at index len-1, slice to [:len-1], return (element, true). Peek is similar but doesn't modify slice.`,
	whyItMatters: `Stacks are fundamental data structures used throughout computer science and production systems for managing execution flow, parsing, and algorithms.

**Why Stacks Matter:**

**1. Function Call Stack - How Programs Run**

Every program uses a stack internally to manage function calls:

\`\`\`go
func main() {
    fmt.Println("Start")
    processOrder()
    fmt.Println("End")
}

func processOrder() {
    validateOrder()
}

func validateOrder() {
    checkInventory()
}

// Call stack during execution:
// [main] <- bottom
// [main, processOrder]
// [main, processOrder, validateOrder]
// [main, processOrder, validateOrder, checkInventory] <- top
// Then unwinds: Pop checkInventory, Pop validateOrder, Pop processOrder
\`\`\`

**Real Incident**: A recursive function without base case caused stack overflow. Understanding the call stack helped identify the infinite recursion.

**2. Undo/Redo Functionality**

Every text editor, IDE, and design tool uses stacks:

\`\`\`go
type Editor struct {
    undoStack *Stack[Command]
    redoStack *Stack[Command]
}

func (e *Editor) Execute(cmd Command) {
    cmd.Execute()
    e.undoStack.Push(cmd)
    // Clear redo stack on new action
    e.redoStack = NewStack[Command]()
}

func (e *Editor) Undo() {
    if cmd, ok := e.undoStack.Pop(); ok {
        cmd.Undo()
        e.redoStack.Push(cmd)
    }
}

func (e *Editor) Redo() {
    if cmd, ok := e.redoStack.Pop(); ok {
        cmd.Execute()
        e.undoStack.Push(cmd)
    }
}
\`\`\`

**Production Use**: Google Docs, VS Code, Photoshop all use dual-stack pattern for unlimited undo/redo.

**3. Expression Evaluation and Parsing**

Compilers and calculators use stacks to evaluate expressions:

\`\`\`go
// Evaluate postfix expression: "3 4 + 5 *"
func EvaluatePostfix(expression string) int {
    stack := NewStack[int]()

    for _, token := range strings.Split(expression, " ") {
        if isOperator(token) {
            b, _ := stack.Pop()
            a, _ := stack.Pop()
            result := apply(token, a, b)
            stack.Push(result)
        } else {
            num, _ := strconv.Atoi(token)
            stack.Push(num)
        }
    }

    result, _ := stack.Pop()
    return result
}

// "3 4 + 5 *" = (3 + 4) * 5 = 35
// Stack evolution:
// [3]
// [3, 4]
// [7]        // Pop 4, pop 3, push 3+4
// [7, 5]
// [35]       // Pop 5, pop 7, push 7*5
\`\`\`

**Real Use**: Every programming language compiler uses stacks for parsing syntax and evaluating expressions.

**4. Browser Navigation History**

Back/Forward buttons use dual-stack pattern:

\`\`\`go
type Browser struct {
    backStack    *Stack[string]  // History of visited pages
    forwardStack *Stack[string]  // Pages to go forward to
    currentPage  string
}

func (b *Browser) Visit(url string) {
    if b.currentPage != "" {
        b.backStack.Push(b.currentPage)
    }
    b.currentPage = url
    b.forwardStack = NewStack[string]()  // Clear forward history
}

func (b *Browser) Back() {
    if page, ok := b.backStack.Pop(); ok {
        b.forwardStack.Push(b.currentPage)
        b.currentPage = page
    }
}

func (b *Browser) Forward() {
    if page, ok := b.forwardStack.Pop(); ok {
        b.backStack.Push(b.currentPage)
        b.currentPage = page
    }
}
\`\`\`

**5. Balanced Parentheses Validation**

Critical for code editors, JSON parsers, HTML validators:

\`\`\`go
func IsBalanced(s string) bool {
    stack := NewStack[rune]()
    pairs := map[rune]rune{')': '(', '}': '{', ']': '['}

    for _, char := range s {
        switch char {
        case '(', '{', '[':
            stack.Push(char)
        case ')', '}', ']':
            if opening, ok := stack.Pop(); !ok || opening != pairs[char] {
                return false
            }
        }
    }

    return stack.IsEmpty()
}

// IsBalanced("({[]})") == true
// IsBalanced("({[})") == false
\`\`\`

**Real Incident**: A JSON parser crashed on malformed input. Adding balanced bracket validation with stack prevented the crash and provided clear error messages.

**6. Depth-First Search (DFS)**

Graph and tree traversal algorithms:

\`\`\`go
func DFS(root *Node) []int {
    result := []int{}
    stack := NewStack[*Node]()
    stack.Push(root)

    for !stack.IsEmpty() {
        node, _ := stack.Pop()
        if node == nil {
            continue
        }

        result = append(result, node.Value)

        // Push children (right first for left-to-right traversal)
        if node.Right != nil {
            stack.Push(node.Right)
        }
        if node.Left != nil {
            stack.Push(node.Left)
        }
    }

    return result
}
\`\`\`

**7. Stack vs Queue vs Array**

| Operation | Stack (LIFO) | Queue (FIFO) | Array (Index) |
|-----------|--------------|--------------|---------------|
| Add       | O(1) Push    | O(1) Enqueue | O(1) Append   |
| Remove    | O(1) Pop     | O(1) Dequeue | O(n) Remove   |
| Access    | O(1) Peek    | O(1) Peek    | O(1) Index    |
| Use Case  | Undo/Redo    | Task Queue   | Random Access |

**8. Generic Benefits**

Before generics (Go 1.17 and earlier):

\`\`\`go
// Needed separate stack for each type
type IntStack struct {
    elements []int
}

type StringStack struct {
    elements []string
}

// Or used interface{} (not type-safe)
type Stack struct {
    elements []interface{}
}

stack.Push(42)
stack.Push("hello")  // Runtime error waiting to happen!
\`\`\`

With generics (Go 1.18+):

\`\`\`go
// One implementation, full type safety
intStack := NewStack[int]()
intStack.Push(42)
intStack.Push("hello")  // Compile-time error! Type safety!

strStack := NewStack[string]()
strStack.Push("hello")  // Works perfectly
\`\`\`

**Key Takeaways:**
- Stacks implement LIFO (Last-In-First-Out) principle
- Used for undo/redo, parsing, DFS, function calls
- Generic implementation works with any type
- O(1) operations make them highly efficient
- Safe operations prevent panics on empty stack
- Fundamental building block for complex algorithms`,
	order: 1,
	translations: {
		ru: {
			title: 'Реализация Generic Stack с типобезопасностью',
			description: `Постройте production-ready generic Stack структуру данных, реализующую принцип LIFO (Last-In-First-Out) с полной типобезопасностью.

**Вы реализуете:**

**Уровень 1 (Лёгкий) — Основные операции Stack:**
1. **NewStack[T any]() *Stack[T]** — Создать новый пустой стек
2. **Push(value T)** — Добавить элемент на вершину стека
3. **Pop() (T, bool)** — Удалить и вернуть верхний элемент (value, ok)
4. **Peek() (T, bool)** — Просмотреть верхний элемент без удаления (value, ok)
5. **IsEmpty() bool** — Проверить, нет ли элементов в стеке
6. **Size() int** — Вернуть количество элементов в стеке

**Ключевые концепции:**
- **Принцип LIFO**: Последний добавленный элемент удаляется первым
- **Generic Stack**: Работает с любым типом (int, string, struct, interface)
- **Безопасные операции**: Возврат флага ok для предотвращения паники на пустом стеке
- **На основе Slice**: Использует Go slice как базовое хранилище
- **O(1) операции**: Push, Pop, Peek выполняются за константное время

**Пример использования:**

\`\`\`go
// Стек целых чисел
intStack := NewStack[int]()
intStack.Push(1)
intStack.Push(2)
intStack.Push(3)

val, ok := intStack.Pop()
// val == 3, ok == true

top, ok := intStack.Peek()
// top == 2, ok == true (не удалён)

size := intStack.Size()
// size == 2

isEmpty := intStack.IsEmpty()
// isEmpty == false

// Стек строк — тот же интерфейс
strStack := NewStack[string]()
strStack.Push("hello")
strStack.Push("world")

word, ok := strStack.Pop()
// word == "world", ok == true

// Стек пользовательского типа
type Task struct {
    ID   int
    Name string
}

taskStack := NewStack[Task]()
taskStack.Push(Task{1, "Write code"})
taskStack.Push(Task{2, "Test code"})

task, ok := taskStack.Pop()
// task == Task{2, "Test code"}, ok == true

// Безопасные операции на пустом стеке
emptyStack := NewStack[int]()
val, ok := emptyStack.Pop()
// val == 0 (нулевое значение), ok == false

val, ok = emptyStack.Peek()
// val == 0 (нулевое значение), ok == false
\`\`\`

**Ограничения:**
- Использовать slice как базовое хранилище
- Pop/Peek на пустом стеке возвращают нулевое значение и false
- Push должен автоматически обрабатывать рост slice
- Все операции должны быть O(1) амортизированное время
- Stack должен быть безопасным для любого типа T`,
			hint1: `Определите Stack[T any] структуру с полем slice: elements []T. NewStack возвращает &Stack[T]{elements: make([]T, 0)}. Push добавляет в slice.`,
			hint2: `Pop: Сначала проверьте IsEmpty, если пуст вернуть (var zero T, false). Иначе получить последний элемент по индексу len-1, slice до [:len-1], вернуть (element, true). Peek аналогичен, но не модифицирует slice.`,
			whyItMatters: `Стеки — фундаментальные структуры данных, используемые повсеместно в computer science и production системах для управления потоком выполнения, парсинга и алгоритмов.

**Почему стеки важны:**

**1. Function Call Stack — Как работают программы**

Каждая программа использует стек внутренне для управления вызовами функций:

\`\`\`go
func main() {
    fmt.Println("Start")
    processOrder()
    fmt.Println("End")
}

func processOrder() {
    validateOrder()
}

func validateOrder() {
    checkInventory()
}

// Стек вызовов во время выполнения:
// [main] <- низ
// [main, processOrder]
// [main, processOrder, validateOrder]
// [main, processOrder, validateOrder, checkInventory] <- верх
// Затем раскручивается: Pop checkInventory, Pop validateOrder, Pop processOrder
\`\`\`

**Реальный инцидент**: Рекурсивная функция без базового случая вызвала переполнение стека. Понимание call stack помогло идентифицировать бесконечную рекурсию.

**2. Функциональность Undo/Redo**

Каждый текстовый редактор, IDE и инструмент дизайна использует стеки:

\`\`\`go
type Editor struct {
    undoStack *Stack[Command]
    redoStack *Stack[Command]
}

func (e *Editor) Execute(cmd Command) {
    cmd.Execute()
    e.undoStack.Push(cmd)
    // Очистить redo стек при новом действии
    e.redoStack = NewStack[Command]()
}

func (e *Editor) Undo() {
    if cmd, ok := e.undoStack.Pop(); ok {
        cmd.Undo()
        e.redoStack.Push(cmd)
    }
}

func (e *Editor) Redo() {
    if cmd, ok := e.redoStack.Pop(); ok {
        cmd.Execute()
        e.undoStack.Push(cmd)
    }
}
\`\`\`

**Продакшен использование**: Google Docs, VS Code, Photoshop все используют паттерн двух стеков для неограниченного undo/redo.

**3. Вычисление выражений и парсинг**

Компиляторы и калькуляторы используют стеки для вычисления выражений:

\`\`\`go
// Вычислить постфиксное выражение: "3 4 + 5 *"
func EvaluatePostfix(expression string) int {
    stack := NewStack[int]()

    for _, token := range strings.Split(expression, " ") {
        if isOperator(token) {
            b, _ := stack.Pop()
            a, _ := stack.Pop()
            result := apply(token, a, b)
            stack.Push(result)
        } else {
            num, _ := strconv.Atoi(token)
            stack.Push(num)
        }
    }

    result, _ := stack.Pop()
    return result
}

// "3 4 + 5 *" = (3 + 4) * 5 = 35
// Эволюция стека:
// [3]
// [3, 4]
// [7]        // Pop 4, pop 3, push 3+4
// [7, 5]
// [35]       // Pop 5, pop 7, push 7*5
\`\`\`

**Реальное использование**: Каждый компилятор языка программирования использует стеки для парсинга синтаксиса и вычисления выражений.

**4. История навигации браузера**

Кнопки Назад/Вперёд используют паттерн двух стеков:

\`\`\`go
type Browser struct {
    backStack    *Stack[string]  // История посещённых страниц
    forwardStack *Stack[string]  // Страницы для перехода вперёд
    currentPage  string
}

func (b *Browser) Visit(url string) {
    if b.currentPage != "" {
        b.backStack.Push(b.currentPage)
    }
    b.currentPage = url
    b.forwardStack = NewStack[string]()  // Очистить историю вперёд
}

func (b *Browser) Back() {
    if page, ok := b.backStack.Pop(); ok {
        b.forwardStack.Push(b.currentPage)
        b.currentPage = page
    }
}

func (b *Browser) Forward() {
    if page, ok := b.forwardStack.Pop(); ok {
        b.backStack.Push(b.currentPage)
        b.currentPage = page
    }
}
\`\`\`

**5. Валидация сбалансированных скобок**

Критично для редакторов кода, JSON-парсеров, HTML-валидаторов:

\`\`\`go
func IsBalanced(s string) bool {
    stack := NewStack[rune]()
    pairs := map[rune]rune{')': '(', '}': '{', ']': '['}

    for _, char := range s {
        switch char {
        case '(', '{', '[':
            stack.Push(char)
        case ')', '}', ']':
            if opening, ok := stack.Pop(); !ok || opening != pairs[char] {
                return false
            }
        }
    }

    return stack.IsEmpty()
}

// IsBalanced("({[]})") == true
// IsBalanced("({[})") == false
\`\`\`

**Реальный инцидент**: JSON-парсер падал на некорректном вводе. Добавление валидации сбалансированных скобок со стеком предотвратило падения и предоставило чёткие сообщения об ошибках.

**Ключевые выводы:**
- Стеки реализуют принцип LIFO (Last-In-First-Out)
- Используются для undo/redo, парсинга, DFS, вызовов функций
- Generic реализация работает с любым типом
- O(1) операции делают их высокоэффективными
- Безопасные операции предотвращают панику на пустом стеке
- Фундаментальный строительный блок для сложных алгоритмов`,
			solutionCode: `package datastructsx

// Stack представляет структуру данных LIFO (Last-In-First-Out)
type Stack[T any] struct {
	elements []T
}

func NewStack[T any]() *Stack[T] {
	return &Stack[T]{
		elements: make([]T, 0),
	}
}

func (s *Stack[T]) Push(value T) {
	s.elements = append(s.elements, value)
}

func (s *Stack[T]) Pop() (T, bool) {
	if s.IsEmpty() {
		var zero T
		return zero, false
	}

	index := len(s.elements) - 1
	value := s.elements[index]
	s.elements = s.elements[:index]

	return value, true
}

func (s *Stack[T]) Peek() (T, bool) {
	if s.IsEmpty() {
		var zero T
		return zero, false
	}

	return s.elements[len(s.elements)-1], true
}

func (s *Stack[T]) IsEmpty() bool {
	return len(s.elements) == 0
}

func (s *Stack[T]) Size() int {
	return len(s.elements)
}`
		},
		uz: {
			title: `Tipxavfsizlik bilan Generic Stack implementatsiyasi`,
			description: `LIFO (Last-In-First-Out) printsipini to'liq tipxavfsizlik bilan amalga oshiradigan production-ready generic Stack ma'lumotlar strukturasini yarating.

**Siz amalga oshirasiz:**

**1-Daraja (Oson) — Asosiy Stack operatsiyalari:**
1. **NewStack[T any]() *Stack[T]** — Yangi bo'sh stack yaratish
2. **Push(value T)** — Stack tepasiga element qo'shish
3. **Pop() (T, bool)** — Yuqori elementni o'chirib qaytarish (value, ok)
4. **Peek() (T, bool)** — Yuqori elementni o'chirmasdan ko'rish (value, ok)
5. **IsEmpty() bool** — Stack da elementlar yo'qligini tekshirish
6. **Size() int** — Stack dagi elementlar sonini qaytarish

**Asosiy tushunchalar:**
- **LIFO Prinsipi**: Oxirgi qo'shilgan element birinchi o'chiriladi
- **Generic Stack**: Har qanday tur bilan ishlaydi (int, string, struct, interface)
- **Xavfsiz operatsiyalar**: Bo'sh stack da panikadan qochish uchun ok bayroqni qaytarish
- **Slice-asosli**: Go slice ni asosiy saqlash sifatida ishlatadi
- **O(1) operatsiyalar**: Push, Pop, Peek doimiy vaqtda bajariladi

**Foydalanish misoli:**

\`\`\`go
// Butun sonlar stack i
intStack := NewStack[int]()
intStack.Push(1)
intStack.Push(2)
intStack.Push(3)

val, ok := intStack.Pop()
// val == 3, ok == true

top, ok := intStack.Peek()
// top == 2, ok == true (o'chirilmagan)

size := intStack.Size()
// size == 2

isEmpty := intStack.IsEmpty()
// isEmpty == false

// Satr stack i — bir xil interfeys
strStack := NewStack[string]()
strStack.Push("hello")
strStack.Push("world")

word, ok := strStack.Pop()
// word == "world", ok == true

// Maxsus tip stack i
type Task struct {
    ID   int
    Name string
}

taskStack := NewStack[Task]()
taskStack.Push(Task{1, "Write code"})
taskStack.Push(Task{2, "Test code"})

task, ok := taskStack.Pop()
// task == Task{2, "Test code"}, ok == true

// Bo'sh stack da xavfsiz operatsiyalar
emptyStack := NewStack[int]()
val, ok := emptyStack.Pop()
// val == 0 (nol qiymat), ok == false

val, ok = emptyStack.Peek()
// val == 0 (nol qiymat), ok == false
\`\`\`

**Cheklovlar:**
- Slice ni asosiy saqlash sifatida ishlatish
- Bo'sh stack da Pop/Peek nol qiymat va false qaytaradi
- Push avtomatik ravishda slice o'sishini boshqarishi kerak
- Barcha operatsiyalar O(1) amortizatsiyalangan vaqt bo'lishi kerak
- Stack har qanday T turi uchun xavfsiz bo'lishi kerak`,
			hint1: `Stack[T any] strukturasini slice maydoni bilan aniqlang: elements []T. NewStack &Stack[T]{elements: make([]T, 0)} qaytaradi. Push slice ga qo'shadi.`,
			hint2: `Pop: Avval IsEmpty ni tekshiring, agar bo'sh bo'lsa (var zero T, false) qaytaring. Aks holda len-1 indeksida oxirgi elementni oling, [:len-1] ga slice qiling, (element, true) qaytaring. Peek o'xshash, lekin slice ni o'zgartirmaydi.`,
			whyItMatters: `Stack lar computer science va production tizimlarida ijro oqimini boshqarish, parsing va algoritmlar uchun qo'llaniladigan fundamental ma'lumotlar strukturalaridir.

**Nega stack lar muhim:**

**1. Function Call Stack — Dasturlar qanday ishlaydi**

Har bir dastur funktsiya chaqiruvlarini boshqarish uchun ichki stack dan foydalanadi:

\`\`\`go
func main() {
    fmt.Println("Start")
    processOrder()
    fmt.Println("End")
}

func processOrder() {
    validateOrder()
}

func validateOrder() {
    checkInventory()
}

// Bajarish vaqtida call stack:
// [main] <- past
// [main, processOrder]
// [main, processOrder, validateOrder]
// [main, processOrder, validateOrder, checkInventory] <- yuqori
// So'ngra ochiladi: Pop checkInventory, Pop validateOrder, Pop processOrder
\`\`\`

**Haqiqiy hodisa**: Asosiy holatsiz rekursiv funktsiya stack overflow ni keltirib chiqardi. Call stack ni tushunish cheksiz rekursiyani aniqlashga yordam berdi.

**2. Undo/Redo funksionalligi**

Har bir matn muharriri, IDE va dizayn vositasi stack lardan foydalanadi:

\`\`\`go
type Editor struct {
    undoStack *Stack[Command]
    redoStack *Stack[Command]
}

func (e *Editor) Execute(cmd Command) {
    cmd.Execute()
    e.undoStack.Push(cmd)
    // Yangi amalda redo stack ni tozalash
    e.redoStack = NewStack[Command]()
}

func (e *Editor) Undo() {
    if cmd, ok := e.undoStack.Pop(); ok {
        cmd.Undo()
        e.redoStack.Push(cmd)
    }
}

func (e *Editor) Redo() {
    if cmd, ok := e.redoStack.Pop(); ok {
        cmd.Execute()
        e.undoStack.Push(cmd)
    }
}
\`\`\`

**Production foydalanish**: Google Docs, VS Code, Photoshop hammasi cheksiz undo/redo uchun ikki stack patternidan foydalanadi.

**3. Ifodalarni hisoblash va parsing**

Kompilyatorlar va kalkulyatorlar ifodalarni hisoblash uchun stack lardan foydalanadi:

\`\`\`go
// Postfix ifodani hisoblash: "3 4 + 5 *"
func EvaluatePostfix(expression string) int {
    stack := NewStack[int]()

    for _, token := range strings.Split(expression, " ") {
        if isOperator(token) {
            b, _ := stack.Pop()
            a, _ := stack.Pop()
            result := apply(token, a, b)
            stack.Push(result)
        } else {
            num, _ := strconv.Atoi(token)
            stack.Push(num)
        }
    }

    result, _ := stack.Pop()
    return result
}

// "3 4 + 5 *" = (3 + 4) * 5 = 35
// Stack evolyutsiyasi:
// [3]
// [3, 4]
// [7]        // Pop 4, pop 3, push 3+4
// [7, 5]
// [35]       // Pop 5, pop 7, push 7*5
\`\`\`

**Haqiqiy foydalanish**: Har bir dasturlash tili kompilyatori sintaksisni parsing va ifodalarni hisoblash uchun stack lardan foydalanadi.

**4. Brauzer navigatsiya tarixi**

Orqaga/Oldinga tugmalari ikki stack patternidan foydalanadi:

\`\`\`go
type Browser struct {
    backStack    *Stack[string]  // Tashrif buyurilgan sahifalar tarixi
    forwardStack *Stack[string]  // Oldinga o'tiladigan sahifalar
    currentPage  string
}

func (b *Browser) Visit(url string) {
    if b.currentPage != "" {
        b.backStack.Push(b.currentPage)
    }
    b.currentPage = url
    b.forwardStack = NewStack[string]()  // Oldinga tarixni tozalash
}

func (b *Browser) Back() {
    if page, ok := b.backStack.Pop(); ok {
        b.forwardStack.Push(b.currentPage)
        b.currentPage = page
    }
}

func (b *Browser) Forward() {
    if page, ok := b.forwardStack.Pop(); ok {
        b.backStack.Push(b.currentPage)
        b.currentPage = page
    }
}
\`\`\`

**5. Muvozanatlashgan qavslar validatsiyasi**

Kod muharrirlari, JSON parserlar, HTML validatorlar uchun muhim:

\`\`\`go
func IsBalanced(s string) bool {
    stack := NewStack[rune]()
    pairs := map[rune]rune{')': '(', '}': '{', ']': '['}

    for _, char := range s {
        switch char {
        case '(', '{', '[':
            stack.Push(char)
        case ')', '}', ']':
            if opening, ok := stack.Pop(); !ok || opening != pairs[char] {
                return false
            }
        }
    }

    return stack.IsEmpty()
}

// IsBalanced("({[]})") == true
// IsBalanced("({[})") == false
\`\`\`

**Haqiqiy hodisa**: JSON parser noto'g'ri kiritishda ishdan chiqdi. Stack bilan muvozanatlashgan qavslar validatsiyasini qo'shish ishdan chiqishni oldini oldi va aniq xato xabarlarini berdi.

**6. Depth-First Search (DFS)**

Graf va daraxt traversal algoritmlari:

\`\`\`go
func DFS(root *Node) []int {
    result := []int{}
    stack := NewStack[*Node]()
    stack.Push(root)

    for !stack.IsEmpty() {
        node, _ := stack.Pop()
        if node == nil {
            continue
        }

        result = append(result, node.Value)

        // Bolalarni push qilish (chapdan o'ngga traversal uchun o'ngdan boshlash)
        if node.Right != nil {
            stack.Push(node.Right)
        }
        if node.Left != nil {
            stack.Push(node.Left)
        }
    }

    return result
}
\`\`\`

**7. Stack vs Queue vs Array**

| Operatsiya | Stack (LIFO) | Queue (FIFO) | Array (Index) |
|------------|--------------|--------------|---------------|
| Qo'shish   | O(1) Push    | O(1) Enqueue | O(1) Append   |
| O'chirish  | O(1) Pop     | O(1) Dequeue | O(n) Remove   |
| Kirish     | O(1) Peek    | O(1) Peek    | O(1) Index    |
| Foydalanish | Undo/Redo   | Task Queue   | Random Access |

**8. Generic afzalliklari**

Genericlardan oldin (Go 1.17 va oldingi):

\`\`\`go
// Har bir tur uchun alohida stack kerak edi
type IntStack struct {
    elements []int
}

type StringStack struct {
    elements []string
}

// Yoki interface{} ishlatilgan (tipxavfsiz emas)
type Stack struct {
    elements []interface{}
}

stack.Push(42)
stack.Push("hello")  // Runtime xatosi kutilmoqda!
\`\`\`

Genericlar bilan (Go 1.18+):

\`\`\`go
// Bitta implementatsiya, to'liq tipxavfsizlik
intStack := NewStack[int]()
intStack.Push(42)
intStack.Push("hello")  // Kompilatsiya vaqtida xato! Tipxavfsizlik!

strStack := NewStack[string]()
strStack.Push("hello")  // Mukammal ishlaydi
\`\`\`

**Asosiy xulosalar:**
- Stack lar LIFO (Last-In-First-Out) printsipini amalga oshiradi
- Undo/redo, parsing, DFS, funktsiya chaqiruvlari uchun ishlatiladi
- Generic implementatsiya har qanday tur bilan ishlaydi
- O(1) operatsiyalar ularni yuqori samarali qiladi
- Xavfsiz operatsiyalar bo'sh stack da panikadan himoya qiladi
- Murakkab algoritmlar uchun fundamental qurilish bloki`,
			solutionCode: `package datastructsx

// Stack LIFO (Last-In-First-Out) ma'lumotlar strukturasini ifodalaydi
type Stack[T any] struct {
	elements []T
}

func NewStack[T any]() *Stack[T] {
	return &Stack[T]{
		elements: make([]T, 0),
	}
}

func (s *Stack[T]) Push(value T) {
	s.elements = append(s.elements, value)
}

func (s *Stack[T]) Pop() (T, bool) {
	if s.IsEmpty() {
		var zero T
		return zero, false
	}

	index := len(s.elements) - 1
	value := s.elements[index]
	s.elements = s.elements[:index]

	return value, true
}

func (s *Stack[T]) Peek() (T, bool) {
	if s.IsEmpty() {
		var zero T
		return zero, false
	}

	return s.elements[len(s.elements)-1], true
}

func (s *Stack[T]) IsEmpty() bool {
	return len(s.elements) == 0
}

func (s *Stack[T]) Size() int {
	return len(s.elements)
}`
		}
	}
};

export default task;
