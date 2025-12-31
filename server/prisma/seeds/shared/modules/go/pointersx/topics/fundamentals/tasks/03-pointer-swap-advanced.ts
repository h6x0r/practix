import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-pointersx-swap-advanced',
	title: 'Advanced Pointer Swapping Techniques',
	difficulty: 'medium',
	tags: ['go', 'pointers', 'algorithms', 'memory-manipulation'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Master advanced pointer manipulation by implementing various swap operations that modify values through pointers without using temporary variables or with special constraints.

**You will implement:**

1. **SwapInts(a, b *int)** - Swap two integers using XOR trick (no temp variable)
2. **SwapPointers(a, b **int)** - Swap the pointers themselves (double pointers)
3. **RotateThree(a, b, c *int)** - Rotate three values: a→b, b→c, c→a
4. **SwapIfGreater(a, b *int)** - Conditional swap: swap only if *a > *b
5. **SwapStructFields(s *Pair)** - Swap two fields within a struct
6. **ReversePointerArray(arr []*int)** - Reverse array of pointers in-place
7. **SwapAndSum(a, b *int) int** - Swap values and return their sum

**Key Concepts:**
- **XOR Swap**: Bitwise swap without temporary variable
- **Double Pointers**: Swapping pointer addresses themselves
- **In-Place Operations**: Modifying data without extra memory
- **Conditional Mutations**: Only modify when conditions are met
- **Multi-Value Rotation**: Circular value shifting
- **Struct Field Manipulation**: Accessing and modifying struct internals

**Example Usage:**
\`\`\`go
// XOR swap without temporary variable
a, b := 10, 20
SwapInts(&a, &b)
// a == 20, b == 10

// Swap pointers themselves (not values)
x, y := 100, 200
px, py := &x, &y
SwapPointers(&px, &py)
// px points to y (200), py points to x (100)
// x == 100, y == 200 (original values unchanged)

// Rotate three values in circular fashion
a, b, c := 1, 2, 3
RotateThree(&a, &b, &c)
// a == 3, b == 1, c == 2

// Conditional swap
a, b := 50, 30
SwapIfGreater(&a, &b)
// a == 30, b == 50 (swapped because 50 > 30)

a, b = 10, 20
SwapIfGreater(&a, &b)
// a == 10, b == 20 (not swapped because 10 < 20)

// Swap struct fields
pair := &Pair{First: 100, Second: 200}
SwapStructFields(pair)
// pair.First == 200, pair.Second == 100

// Reverse pointer array
nums := []int{1, 2, 3, 4, 5}
ptrs := []*int{&nums[0], &nums[1], &nums[2], &nums[3], &nums[4]}
ReversePointerArray(ptrs)
// ptrs[0] points to 5, ptrs[1] points to 4, etc.

// Swap and calculate
a, b := 7, 3
sum := SwapAndSum(&a, &b)
// a == 3, b == 7, sum == 10
\`\`\`

**Constraints:**
- SwapInts must use XOR trick (no temp variable)
- All functions must handle nil pointers safely
- In-place operations (no auxiliary arrays)
- Preserve original pointer validity`,
	initialCode: `package pointersx

// Pair represents a struct with two integer fields
type Pair struct {
	First  int
	Second int
}

// TODO: Implement SwapInts using XOR swap
// Swap two integers without using temporary variable
// Hint: Use XOR trick: a^=b, b^=a, a^=b
func SwapInts(a, b *int) {
	// TODO: Implement
}

// TODO: Implement SwapPointers
// Swap the pointer addresses themselves (not values)
// Hint: Use double pointers to modify what a and b point to
func SwapPointers(a, b **int) {
	// TODO: Implement
}

// TODO: Implement RotateThree
// Rotate three values: a gets c's value, b gets a's value, c gets b's value
// Hint: Save one value, then cascade: a→temp, c→a, b→c, temp→b
func RotateThree(a, b, c *int) {
	// TODO: Implement
}

// TODO: Implement SwapIfGreater
// Swap only if *a > *b (ascending order)
// Hint: Check both pointers for nil, compare *a > *b, then swap
func SwapIfGreater(a, b *int) {
	// TODO: Implement
}

// TODO: Implement SwapStructFields
// Swap First and Second fields in the Pair struct
// Hint: Check for nil, then swap s.First and s.Second
func SwapStructFields(s *Pair) {
	// TODO: Implement
}

// TODO: Implement ReversePointerArray
// Reverse array of pointers in-place
// Hint: Two-pointer technique: swap arr[i] with arr[len-1-i]
func ReversePointerArray(arr []*int) {
	// TODO: Implement
}

// TODO: Implement SwapAndSum
// Swap values at a and b, then return their sum
// Hint: Check nil, swap values, return *a + *b
func SwapAndSum(a, b *int) int {
	return 0 // TODO: Implement
}`,
	solutionCode: `package pointersx

type Pair struct {
	First  int // first element in the pair
	Second int // second element in the pair
}

func SwapInts(a, b *int) {
	if a == nil || b == nil { // prevent nil dereference in XOR operations
		return
	}
	*a ^= *b // a becomes a XOR b
	*b ^= *a // b becomes (a XOR b) XOR original_b = original_a
	*a ^= *b // a becomes (a XOR b) XOR original_a = original_b
}

func SwapPointers(a, b **int) {
	if a == nil || b == nil { // guard against nil double pointers
		return
	}
	*a, *b = *b, *a // swap the addresses stored in the pointer variables
}

func RotateThree(a, b, c *int) {
	if a == nil || b == nil || c == nil { // all three pointers must be valid
		return
	}
	temp := *a     // preserve original value of a
	*a = *c        // move c's value into a
	*c = *b        // move b's value into c
	*b = temp      // move saved a's value into b
}

func SwapIfGreater(a, b *int) {
	if a == nil || b == nil { // cannot compare nil pointers
		return
	}
	if *a > *b { // check if values are out of order
		*a, *b = *b, *a // perform swap to enforce ascending order
	}
}

func SwapStructFields(s *Pair) {
	if s == nil { // nil struct pointer cannot have fields swapped
		return
	}
	s.First, s.Second = s.Second, s.First // exchange the two field values
}

func ReversePointerArray(arr []*int) {
	left, right := 0, len(arr)-1 // initialize two-pointer indices
	for left < right {           // continue until pointers meet in middle
		arr[left], arr[right] = arr[right], arr[left] // swap pointers at symmetric positions
		left++  // move left pointer toward center
		right-- // move right pointer toward center
	}
}

func SwapAndSum(a, b *int) int {
	if a == nil || b == nil { // cannot swap or sum with nil pointers
		return 0 // return neutral value for invalid input
	}
	*a, *b = *b, *a       // exchange the values
	return *a + *b        // compute sum of swapped values
}`,
	testCode: `package pointersx

import "testing"

func Test1(t *testing.T) {
	// SwapInts XOR
	a, b := 10, 20
	SwapInts(&a, &b)
	if a != 20 || b != 10 {
		t.Errorf("expected 20,10, got %d,%d", a, b)
	}
}

func Test2(t *testing.T) {
	// SwapInts nil safe
	SwapInts(nil, nil) // should not panic
}

func Test3(t *testing.T) {
	// SwapPointers
	x, y := 100, 200
	px, py := &x, &y
	SwapPointers(&px, &py)
	if *px != 200 || *py != 100 {
		t.Errorf("expected pointers swapped, got %d,%d", *px, *py)
	}
}

func Test4(t *testing.T) {
	// RotateThree
	a, b, c := 1, 2, 3
	RotateThree(&a, &b, &c)
	if a != 3 || b != 1 || c != 2 {
		t.Errorf("expected 3,1,2, got %d,%d,%d", a, b, c)
	}
}

func Test5(t *testing.T) {
	// SwapIfGreater - should swap
	a, b := 50, 30
	SwapIfGreater(&a, &b)
	if a != 30 || b != 50 {
		t.Errorf("expected 30,50, got %d,%d", a, b)
	}
}

func Test6(t *testing.T) {
	// SwapIfGreater - should not swap
	a, b := 10, 20
	SwapIfGreater(&a, &b)
	if a != 10 || b != 20 {
		t.Errorf("expected 10,20 unchanged, got %d,%d", a, b)
	}
}

func Test7(t *testing.T) {
	// SwapStructFields
	pair := &Pair{First: 100, Second: 200}
	SwapStructFields(pair)
	if pair.First != 200 || pair.Second != 100 {
		t.Errorf("expected 200,100, got %d,%d", pair.First, pair.Second)
	}
}

func Test8(t *testing.T) {
	// ReversePointerArray
	nums := []int{1, 2, 3, 4, 5}
	ptrs := []*int{&nums[0], &nums[1], &nums[2], &nums[3], &nums[4]}
	ReversePointerArray(ptrs)
	if *ptrs[0] != 5 || *ptrs[4] != 1 {
		t.Errorf("expected reversed pointers")
	}
}

func Test9(t *testing.T) {
	// SwapAndSum
	a, b := 7, 3
	sum := SwapAndSum(&a, &b)
	if a != 3 || b != 7 || sum != 10 {
		t.Errorf("expected 3,7,10, got %d,%d,%d", a, b, sum)
	}
}

func Test10(t *testing.T) {
	// SwapInts same values
	a, b := 42, 42
	SwapInts(&a, &b)
	if a != 42 || b != 42 {
		t.Errorf("expected 42,42, got %d,%d", a, b)
	}
}`,
	hint1: `For SwapInts: XOR has property that a^b^b = a. Pattern: *a ^= *b (a=a^b), *b ^= *a (b=b^(a^b)=original_a), *a ^= *b (a=(a^b)^original_a=original_b). For SwapPointers: *a and *b are the actual pointers, swap them with *a, *b = *b, *a.`,
	hint2: `For RotateThree: Save *a in temp, then move in chain: *a = *c, *c = *b, *b = temp. For ReversePointerArray: Use two indices (left=0, right=len-1), swap arr[left] with arr[right], increment left, decrement right until they meet.`,
	whyItMatters: `Advanced pointer manipulation techniques are fundamental for systems programming, algorithm optimization, and understanding how low-level data structures work.

**Why Advanced Pointer Swapping Matters:**

**1. XOR Swap: Memory-Efficient Technique**

The XOR swap trick is a classic algorithm that swaps two values without allocating temporary memory:

\`\`\`go
// Traditional swap (uses stack space for temp)
func SwapTraditional(a, b *int) {
    temp := *a  // Allocates memory
    *a = *b
    *b = temp
}

// XOR swap (no extra memory)
func SwapXOR(a, b *int) {
    *a ^= *b  // a = a ⊕ b
    *b ^= *a  // b = b ⊕ (a ⊕ b) = original_a
    *a ^= *b  // a = (a ⊕ b) ⊕ original_a = original_b
}
\`\`\`

**Why it works:**
- XOR property: x ⊕ x = 0, x ⊕ 0 = x
- Reversible: (a ⊕ b) ⊕ b = a

**Real-world usage:**
- Embedded systems (limited memory)
- Cryptography (obfuscation)
- Low-level bit manipulation
- Interview favorite!

**Performance note:** Modern compilers optimize both equally, but XOR swap is still valuable for understanding bit operations.

**2. Double Pointers: Modifying Pointer Addresses**

Double pointers (**int) let you change what a pointer points to:

\`\`\`go
// Swap what the pointers point to (not the values)
func SwapPointers(a, b **int) {
    *a, *b = *b, *a
}

// Use case: Linked list head modification
type Node struct {
    Value int
    Next  *Node
}

func RemoveHead(head **Node) {
    if head == nil || *head == nil {
        return
    }
    *head = (*head).Next  // Modify caller's head pointer
}

// Usage
var list *Node = &Node{Value: 1, Next: &Node{Value: 2}}
RemoveHead(&list)  // list now points to second node
\`\`\`

**Production pattern:** Used in C-style APIs that Go wraps (cgo), data structure implementations, and resource pool management.

**3. In-Place Array Reversal**

Reversing pointer arrays in-place is a common interview question and practical technique:

\`\`\`go
// Two-pointer technique
func ReversePointerArray(arr []*int) {
    left, right := 0, len(arr)-1
    for left < right {
        arr[left], arr[right] = arr[right], arr[left]
        left++
        right--
    }
}
\`\`\`

**Time complexity:** O(n)
**Space complexity:** O(1) - truly in-place

**Real-world usage:**
- Image processing (flip pixels)
- String reversal algorithms
- Undo/redo stacks
- Circular buffer manipulation

**4. Conditional Swap: Sorting Building Block**

SwapIfGreater is the core operation of bubble sort and selection sort:

\`\`\`go
// Bubble sort using conditional swap
func BubbleSort(arr []int) {
    n := len(arr)
    for i := 0; i < n; i++ {
        for j := 0; j < n-i-1; j++ {
            SwapIfGreater(&arr[j], &arr[j+1])
        }
    }
}

// Partition in quicksort
func partition(arr []int, low, high int) int {
    pivot := arr[high]
    i := low - 1
    for j := low; j < high; j++ {
        if arr[j] < pivot {
            i++
            SwapIfGreater(&arr[i], &arr[j])
        }
    }
    i++
    arr[i], arr[high] = arr[high], arr[i]
    return i
}
\`\`\`

**5. Three-Way Rotation: Circular Shifts**

Rotating three values is useful in algorithms that maintain sorted invariants:

\`\`\`go
// Rotate right: a→b, b→c, c→a
func RotateThree(a, b, c *int) {
    temp := *a
    *a = *c
    *c = *b
    *b = temp
}

// Use case: Median-of-three pivot selection
func medianOfThree(arr []int, low, mid, high int) {
    if arr[low] > arr[mid] {
        swap(&arr[low], &arr[mid])
    }
    if arr[mid] > arr[high] {
        swap(&arr[mid], &arr[high])
    }
    if arr[low] > arr[mid] {
        swap(&arr[low], &arr[mid])
    }
    // Now arr[low] <= arr[mid] <= arr[high]
}
\`\`\`

**Used in:**
- QuickSort pivot selection
- Graph algorithms (cycle detection)
- State machine transitions
- Ring buffer management

**6. Production Example: Lock-Free Swap**

Advanced pointer swapping is crucial in concurrent programming:

\`\`\`go
import "sync/atomic"

type Node struct {
    Value int
    Next  *Node
}

// Lock-free stack push using atomic compare-and-swap
func (s *Stack) Push(value int) {
    newNode := &Node{Value: value}
    for {
        old := atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&s.head)))
        newNode.Next = (*Node)(old)
        if atomic.CompareAndSwapPointer(
            (*unsafe.Pointer)(unsafe.Pointer(&s.head)),
            old,
            unsafe.Pointer(newNode),
        ) {
            return
        }
        // CAS failed, retry
    }
}
\`\`\`

This pattern is used in:
- Go's sync.Pool
- Lock-free data structures
- Memory allocators
- High-performance queues

**7. Struct Field Swapping: In-Place Transformations**

Swapping struct fields is common in computational geometry and graphics:

\`\`\`go
type Point struct {
    X, Y float64
}

// Rotate point 90 degrees clockwise
func Rotate90CW(p *Point) {
    p.X, p.Y = p.Y, -p.X
}

// Mirror point across y-axis
func MirrorY(p *Point) {
    p.X = -p.X
}

// Transpose matrix (swap rows/columns)
type Matrix struct {
    A, B, C, D float64
}

func Transpose(m *Matrix) {
    m.B, m.C = m.C, m.B
}
\`\`\`

**8. Performance Characteristics**

\`\`\`
Operation           Time    Space   Cache-Friendly
Traditional Swap    O(1)    O(1)    Yes
XOR Swap           O(1)    O(0)    Yes
Pointer Swap       O(1)    O(1)    Yes
Array Reverse      O(n)    O(1)    Sequential
Rotate Three       O(1)    O(1)    Yes
\`\`\`

**9. Common Mistakes**

**Mistake 1: XOR with same pointer**
\`\`\`go
// WRONG - results in zero if a and b point to same location
SwapXOR(&x, &x)  // x becomes 0!
\`\`\`

**Mistake 2: Forgetting nil checks**
\`\`\`go
// WRONG - panics on nil
func SwapInts(a, b *int) {
    *a ^= *b  // PANIC if either is nil
}

// RIGHT
func SwapInts(a, b *int) {
    if a == nil || b == nil {
        return
    }
    *a ^= *b
    *b ^= *a
    *a ^= *b
}
\`\`\`

**Mistake 3: Double pointer confusion**
\`\`\`go
// WRONG - swaps local copies
func SwapPointers(a, b *int) {
    a, b = b, a  // Only swaps function parameters!
}

// RIGHT - modifies caller's pointers
func SwapPointers(a, b **int) {
    *a, *b = *b, *a
}
\`\`\`

**10. Interview Questions Using These Patterns**

1. **Reverse linked list in-place** - Uses pointer swapping
2. **Rotate array k positions** - Uses multi-value rotation
3. **Sort array without extra space** - Uses conditional swap
4. **Detect cycle in linked list** - Uses two-pointer with swap
5. **Dutch National Flag problem** - Uses in-place swapping

**Real-World Impact:**

- Linux kernel uses XOR swap in some low-level code
- Redis uses pointer swapping for memory efficiency
- Game engines use in-place rotations for transformations
- Databases use conditional swaps in B-tree maintenance

**Key Takeaways:**
- XOR swap eliminates temporary variables (memory efficiency)
- Double pointers enable modifying pointer addresses
- In-place operations save memory (O(1) space)
- Conditional swaps are building blocks of sorting
- Multi-value rotation enables circular shifts
- These patterns appear in technical interviews
- Understanding pointers deeply is crucial for systems programming`,
	order: 2,
	translations: {
		ru: {
			title: 'Продвинутые техники обмена через указатели',
			description: `Освойте продвинутую манипуляцию указателями, реализовав различные операции обмена, которые модифицируют значения через указатели без использования временных переменных или со специальными ограничениями.

**Вы реализуете:**

1. **SwapInts(a, b *int)** - Обмен двух целых чисел используя XOR трюк (без временной переменной)
2. **SwapPointers(a, b **int)** - Обмен самих указателей (двойные указатели)
3. **RotateThree(a, b, c *int)** - Ротация трёх значений: a→b, b→c, c→a
4. **SwapIfGreater(a, b *int)** - Условный обмен: менять только если *a > *b
5. **SwapStructFields(s *Pair)** - Обмен двух полей внутри структуры
6. **ReversePointerArray(arr []*int)** - Развернуть массив указателей на месте
7. **SwapAndSum(a, b *int) int** - Поменять значения и вернуть их сумму

**Ключевые концепции:**
- **XOR Swap**: Побитовый обмен без временной переменной
- **Double Pointers**: Обмен самих адресов указателей
- **In-Place Operations**: Модификация данных без дополнительной памяти
- **Conditional Mutations**: Модифицировать только при выполнении условий
- **Multi-Value Rotation**: Круговое смещение значений
- **Struct Field Manipulation**: Доступ и модификация внутренностей структуры

**Пример использования:**
\`\`\`go
// XOR обмен без временной переменной
a, b := 10, 20
SwapInts(&a, &b)
// a == 20, b == 10

// Обмен самих указателей (не значений)
x, y := 100, 200
px, py := &x, &y
SwapPointers(&px, &py)
// px указывает на y (200), py указывает на x (100)
// x == 100, y == 200 (оригинальные значения не изменены)

// Ротация трёх значений по кругу
a, b, c := 1, 2, 3
RotateThree(&a, &b, &c)
// a == 3, b == 1, c == 2

// Условный обмен
a, b := 50, 30
SwapIfGreater(&a, &b)
// a == 30, b == 50 (обменялись потому что 50 > 30)

a, b = 10, 20
SwapIfGreater(&a, &b)
// a == 10, b == 20 (не обменялись потому что 10 < 20)

// Обмен полей структуры
pair := &Pair{First: 100, Second: 200}
SwapStructFields(pair)
// pair.First == 200, pair.Second == 100

// Развернуть массив указателей
nums := []int{1, 2, 3, 4, 5}
ptrs := []*int{&nums[0], &nums[1], &nums[2], &nums[3], &nums[4]}
ReversePointerArray(ptrs)
// ptrs[0] указывает на 5, ptrs[1] указывает на 4, и т.д.

// Обменять и вычислить
a, b := 7, 3
sum := SwapAndSum(&a, &b)
// a == 3, b == 7, sum == 10
\`\`\`

**Ограничения:**
- SwapInts должен использовать XOR трюк (без временной переменной)
- Все функции должны безопасно обрабатывать nil указатели
- Операции на месте (без вспомогательных массивов)
- Сохранять валидность оригинальных указателей`,
			hint1: `Для SwapInts: XOR имеет свойство что a^b^b = a. Паттерн: *a ^= *b (a=a^b), *b ^= *a (b=b^(a^b)=original_a), *a ^= *b (a=(a^b)^original_a=original_b). Для SwapPointers: *a и *b это фактические указатели, обменяйте их через *a, *b = *b, *a.`,
			hint2: `Для RotateThree: Сохраните *a в temp, затем двигайте по цепочке: *a = *c, *c = *b, *b = temp. Для ReversePointerArray: Используйте два индекса (left=0, right=len-1), меняйте arr[left] с arr[right], инкрементируйте left, декрементируйте right пока они не встретятся.`,
			whyItMatters: `Техники продвинутой манипуляции указателями фундаментальны для системного программирования, оптимизации алгоритмов и понимания того, как работают низкоуровневые структуры данных.

**Почему продвинутый обмен через указатели важен:**

**1. XOR Swap: Эффективная по памяти техника**

XOR swap трюк — классический алгоритм обмена двух значений без выделения временной памяти:

\`\`\`go
// Традиционный обмен (использует stack память для temp)
func SwapTraditional(a, b *int) {
    temp := *a  // Выделяет память
    *a = *b
    *b = temp
}

// XOR обмен (без дополнительной памяти)
func SwapXOR(a, b *int) {
    *a ^= *b  // a = a ⊕ b
    *b ^= *a  // b = b ⊕ (a ⊕ b) = original_a
    *a ^= *b  // a = (a ⊕ b) ⊕ original_a = original_b
}
\`\`\`

**Почему это работает:**
- XOR свойство: x ⊕ x = 0, x ⊕ 0 = x
- Обратимо: (a ⊕ b) ⊕ b = a

**Использование в реальном мире:**
- Встраиваемые системы (ограниченная память)
- Криптография (обфускация)
- Низкоуровневая побитовая манипуляция
- Любимый вопрос на собеседованиях!

**Примечание о производительности:** Современные компиляторы оптимизируют оба варианта одинаково, но XOR swap всё ещё ценен для понимания битовых операций.

**2. Double Pointers: Модификация адресов указателей**

Двойные указатели (**int) позволяют изменять на что указывает указатель:

\`\`\`go
// Обмен того, на что указывают указатели (не значений)
func SwapPointers(a, b **int) {
    *a, *b = *b, *a
}

// Случай использования: Модификация head связного списка
type Node struct {
    Value int
    Next  *Node
}

func RemoveHead(head **Node) {
    if head == nil || *head == nil {
        return
    }
    *head = (*head).Next  // Модифицируем указатель head вызывающего кода
}

// Использование
var list *Node = &Node{Value: 1, Next: &Node{Value: 2}}
RemoveHead(&list)  // list теперь указывает на второй узел
\`\`\`

**Продакшен паттерн:** Используется в C-style API которые оборачивает Go (cgo), реализациях структур данных и управлении пулами ресурсов.

**3. In-Place разворот массива**

Разворот массивов указателей на месте — распространённый вопрос на собеседованиях и практическая техника:

\`\`\`go
// Техника двух указателей
func ReversePointerArray(arr []*int) {
    left, right := 0, len(arr)-1
    for left < right {
        arr[left], arr[right] = arr[right], arr[left]
        left++
        right--
    }
}
\`\`\`

**Временная сложность:** O(n)
**Пространственная сложность:** O(1) - действительно на месте

**Использование в реальном мире:**
- Обработка изображений (переворот пикселей)
- Алгоритмы разворота строк
- Стеки Undo/Redo
- Манипуляция кольцевым буфером

**4. Условный обмен: Строительный блок сортировки**

SwapIfGreater — это основная операция пузырьковой и выборочной сортировки:

\`\`\`go
// Пузырьковая сортировка с условным обменом
func BubbleSort(arr []int) {
    n := len(arr)
    for i := 0; i < n; i++ {
        for j := 0; j < n-i-1; j++ {
            SwapIfGreater(&arr[j], &arr[j+1])
        }
    }
}

// Разбиение в quicksort
func partition(arr []int, low, high int) int {
    pivot := arr[high]
    i := low - 1
    for j := low; j < high; j++ {
        if arr[j] < pivot {
            i++
            SwapIfGreater(&arr[i], &arr[j])
        }
    }
    i++
    arr[i], arr[high] = arr[high], arr[i]
    return i
}
\`\`\`

**5. Трёхсторонняя ротация: Круговые сдвиги**

Ротация трёх значений полезна в алгоритмах которые поддерживают отсортированные инварианты:

\`\`\`go
// Ротация вправо: a→b, b→c, c→a
func RotateThree(a, b, c *int) {
    temp := *a
    *a = *c
    *c = *b
    *b = temp
}

// Случай использования: Выбор медианы из трёх для pivot
func medianOfThree(arr []int, low, mid, high int) {
    if arr[low] > arr[mid] {
        swap(&arr[low], &arr[mid])
    }
    if arr[mid] > arr[high] {
        swap(&arr[mid], &arr[high])
    }
    if arr[low] > arr[mid] {
        swap(&arr[low], &arr[mid])
    }
    // Теперь arr[low] <= arr[mid] <= arr[high]
}
\`\`\`

**Используется в:**
- QuickSort выбор pivot
- Алгоритмы графов (обнаружение циклов)
- Переходы конечных автоматов
- Управление кольцевым буфером

**6. Продакшен пример: Lock-Free обмен**

Продвинутый обмен указателей критичен в параллельном программировании:

\`\`\`go
import "sync/atomic"

type Node struct {
    Value int
    Next  *Node
}

// Lock-free push в стек используя atomic compare-and-swap
func (s *Stack) Push(value int) {
    newNode := &Node{Value: value}
    for {
        old := atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&s.head)))
        newNode.Next = (*Node)(old)
        if atomic.CompareAndSwapPointer(
            (*unsafe.Pointer)(unsafe.Pointer(&s.head)),
            old,
            unsafe.Pointer(newNode),
        ) {
            return
        }
        // CAS не удался, повторить
    }
}
\`\`\`

Этот паттерн используется в:
- sync.Pool в Go
- Lock-free структуры данных
- Аллокаторы памяти
- Высокопроизводительные очереди

**7. Обмен полей структур: In-Place трансформации**

Обмен полей структур распространён в вычислительной геометрии и графике:

\`\`\`go
type Point struct {
    X, Y float64
}

// Поворот точки на 90 градусов по часовой стрелке
func Rotate90CW(p *Point) {
    p.X, p.Y = p.Y, -p.X
}

// Зеркальное отражение точки относительно оси Y
func MirrorY(p *Point) {
    p.X = -p.X
}

// Транспонирование матрицы (обмен строк/столбцов)
type Matrix struct {
    A, B, C, D float64
}

func Transpose(m *Matrix) {
    m.B, m.C = m.C, m.B
}
\`\`\`

**8. Характеристики производительности**

\`\`\`
Операция           Время    Память   Cache-Friendly
Традиц. Swap       O(1)    O(1)    Да
XOR Swap           O(1)    O(0)    Да
Pointer Swap       O(1)    O(1)    Да
Array Reverse      O(n)    O(1)    Последовательно
Rotate Three       O(1)    O(1)    Да
\`\`\`

**9. Распространённые ошибки**

**Ошибка 1: XOR с тем же указателем**
\`\`\`go
// НЕПРАВИЛЬНО - результат ноль если a и b указывают на одно место
SwapXOR(&x, &x)  // x становится 0!
\`\`\`

**Ошибка 2: Забывание nil проверок**
\`\`\`go
// НЕПРАВИЛЬНО - паника на nil
func SwapInts(a, b *int) {
    *a ^= *b  // ПАНИКА если любой nil
}

// ПРАВИЛЬНО
func SwapInts(a, b *int) {
    if a == nil || b == nil {
        return
    }
    *a ^= *b
    *b ^= *a
    *a ^= *b
}
\`\`\`

**Ошибка 3: Путаница с двойными указателями**
\`\`\`go
// НЕПРАВИЛЬНО - меняет локальные копии
func SwapPointers(a, b *int) {
    a, b = b, a  // Только меняет параметры функции!
}

// ПРАВИЛЬНО - модифицирует указатели вызывающего кода
func SwapPointers(a, b **int) {
    *a, *b = *b, *a
}
\`\`\`

**10. Вопросы на собеседовании использующие эти паттерны**

1. **Разворот связного списка на месте** - Использует обмен указателей
2. **Ротация массива на k позиций** - Использует многозначную ротацию
3. **Сортировка массива без дополнительной памяти** - Использует условный обмен
4. **Обнаружение цикла в связном списке** - Использует два указателя с обменом
5. **Проблема Голландского флага** - Использует обмен на месте

**Влияние в реальном мире:**

- Ядро Linux использует XOR swap в некотором низкоуровневом коде
- Redis использует обмен указателей для эффективности памяти
- Игровые движки используют ротации на месте для трансформаций
- Базы данных используют условные обмены в обслуживании B-деревьев

**Ключевые выводы:**
- XOR swap устраняет временные переменные (эффективность памяти)
- Double pointers позволяют модифицировать адреса указателей
- In-place операции экономят память (O(1) пространство)
- Условный обмен — строительные блоки сортировки
- Многозначная ротация позволяет круговые сдвиги
- Эти паттерны появляются на технических собеседованиях
- Глубокое понимание указателей критично для системного программирования`,
			solutionCode: `package pointersx

type Pair struct {
	First  int // первый элемент в паре
	Second int // второй элемент в паре
}

func SwapInts(a, b *int) {
	if a == nil || b == nil { // предотвращаем nil разыменование в XOR операциях
		return
	}
	*a ^= *b // a становится a XOR b
	*b ^= *a // b становится (a XOR b) XOR original_b = original_a
	*a ^= *b // a становится (a XOR b) XOR original_a = original_b
}

func SwapPointers(a, b **int) {
	if a == nil || b == nil { // защита от nil двойных указателей
		return
	}
	*a, *b = *b, *a // меняем адреса хранимые в переменных указателей
}

func RotateThree(a, b, c *int) {
	if a == nil || b == nil || c == nil { // все три указателя должны быть валидны
		return
	}
	temp := *a     // сохраняем оригинальное значение a
	*a = *c        // перемещаем значение c в a
	*c = *b        // перемещаем значение b в c
	*b = temp      // перемещаем сохранённое значение a в b
}

func SwapIfGreater(a, b *int) {
	if a == nil || b == nil { // невозможно сравнить nil указатели
		return
	}
	if *a > *b { // проверяем не нарушен ли порядок значений
		*a, *b = *b, *a // выполняем обмен для принудительного порядка возрастания
	}
}

func SwapStructFields(s *Pair) {
	if s == nil { // nil указатель на структуру не может иметь обменянные поля
		return
	}
	s.First, s.Second = s.Second, s.First // обмениваем два значения полей
}

func ReversePointerArray(arr []*int) {
	left, right := 0, len(arr)-1 // инициализируем индексы двух указателей
	for left < right {           // продолжаем пока указатели не встретятся в середине
		arr[left], arr[right] = arr[right], arr[left] // меняем указатели в симметричных позициях
		left++  // двигаем левый указатель к центру
		right-- // двигаем правый указатель к центру
	}
}

func SwapAndSum(a, b *int) int {
	if a == nil || b == nil { // невозможно обменять или суммировать с nil указателями
		return 0 // возвращаем нейтральное значение для невалидного ввода
	}
	*a, *b = *b, *a       // обмениваем значения
	return *a + *b        // вычисляем сумму обменённых значений
}`
		},
		uz: {
			title: `Ilg'or ko'rsatkich almashtirish texnikasi`,
			description: `Vaqtinchalik o'zgaruvchilardan foydalanmasdan yoki maxsus cheklovlar bilan ko'rsatkichlar orqali qiymatlarni o'zgartiradigan turli almashtirish operatsiyalarini amalga oshirish orqali ilg'or ko'rsatkich manipulyatsiyasini o'zlashtiring.

**Siz amalga oshirasiz:**

1. **SwapInts(a, b *int)** - XOR hiylasi yordamida ikkita butun sonni almashtirish (vaqtinchalik o'zgaruvchisiz)
2. **SwapPointers(a, b **int)** - Ko'rsatkichlarning o'zini almashtirish (ikki darajali ko'rsatkichlar)
3. **RotateThree(a, b, c *int)** - Uch qiymatni aylantirish: a→b, b→c, c→a
4. **SwapIfGreater(a, b *int)** - Shartli almashtirish: faqat *a > *b bo'lsa almashtirish
5. **SwapStructFields(s *Pair)** - Struktura ichida ikki maydonni almashtirish
6. **ReversePointerArray(arr []*int)** - Ko'rsatkichlar massivini o'z joyida teskari aylantirish
7. **SwapAndSum(a, b *int) int** - Qiymatlarni almashtirish va ularning yig'indisini qaytarish

**Asosiy tushunchalar:**
- **XOR Swap**: Vaqtinchalik o'zgaruvchisiz bitli almashtirish
- **Double Pointers**: Ko'rsatkich manzillarning o'zini almashtirish
- **In-Place Operations**: Ma'lumotlarni qo'shimcha xotirasiz o'zgartirish
- **Conditional Mutations**: Faqat shartlar bajarilganda o'zgartirish
- **Multi-Value Rotation**: Aylanma qiymat siljishi
- **Struct Field Manipulation**: Struktura ichki qismlariga kirish va o'zgartirish

**Foydalanish misoli:**
\`\`\`go
// Vaqtinchalik o'zgaruvchisiz XOR almashtirish
a, b := 10, 20
SwapInts(&a, &b)
// a == 20, b == 10

// Ko'rsatkichlarning o'zini almashtirish (qiymatlarni emas)
x, y := 100, 200
px, py := &x, &y
SwapPointers(&px, &py)
// px y ga ishora qiladi (200), py x ga ishora qiladi (100)
// x == 100, y == 200 (asl qiymatlar o'zgartirilmagan)

// Uch qiymatni aylanma tarzda aylantirish
a, b, c := 1, 2, 3
RotateThree(&a, &b, &c)
// a == 3, b == 1, c == 2

// Shartli almashtirish
a, b := 50, 30
SwapIfGreater(&a, &b)
// a == 30, b == 50 (almashingan chunki 50 > 30)

a, b = 10, 20
SwapIfGreater(&a, &b)
// a == 10, b == 20 (almashtirilmagan chunki 10 < 20)

// Struktura maydonlarini almashtirish
pair := &Pair{First: 100, Second: 200}
SwapStructFields(pair)
// pair.First == 200, pair.Second == 100

// Ko'rsatkichlar massivini teskari aylantirish
nums := []int{1, 2, 3, 4, 5}
ptrs := []*int{&nums[0], &nums[1], &nums[2], &nums[3], &nums[4]}
ReversePointerArray(ptrs)
// ptrs[0] 5 ga ishora qiladi, ptrs[1] 4 ga ishora qiladi, va hokazo

// Almashtirish va hisoblash
a, b := 7, 3
sum := SwapAndSum(&a, &b)
// a == 3, b == 7, sum == 10
\`\`\`

**Cheklovlar:**
- SwapInts XOR hiylasidan foydalanishi kerak (vaqtinchalik o'zgaruvchisiz)
- Barcha funksiyalar nil ko'rsatkichlarni xavfsiz qayta ishlashi kerak
- O'z joyida operatsiyalar (yordamchi massivlarsiz)
- Asl ko'rsatkich yaroqliligini saqlash`,
			hint1: `SwapInts uchun: XOR xossasi a^b^b = a. Pattern: *a ^= *b (a=a^b), *b ^= *a (b=b^(a^b)=original_a), *a ^= *b (a=(a^b)^original_a=original_b). SwapPointers uchun: *a va *b haqiqiy ko'rsatkichlar, ularni *a, *b = *b, *a bilan almashtiring.`,
			hint2: `RotateThree uchun: *a ni temp ga saqlang, keyin zanjir bo'ylab siljiting: *a = *c, *c = *b, *b = temp. ReversePointerArray uchun: Ikki indeksdan foydalaning (left=0, right=len-1), arr[left] ni arr[right] bilan almashtiring, left ni oshiring, right ni kamaytiring uchrashguncha.`,
			whyItMatters: `Ilg'or ko'rsatkich manipulyatsiya texnikasi tizim dasturlash, algoritm optimizatsiyasi va past darajali ma'lumotlar strukturalari qanday ishlashini tushunish uchun asosiy hisoblanadi.

**Nima uchun ilg'or ko'rsatkich almashtirish muhim:**

**1. XOR Swap: Xotira-samarali texnika**

XOR swap hiylasi vaqtinchalik xotira ajratmasdan ikki qiymatni almashtiruvchi klassik algoritm:

\`\`\`go
// An'anaviy almashtirish (temp uchun stack xotiradan foydalanadi)
func SwapTraditional(a, b *int) {
    temp := *a  // Xotira ajratadi
    *a = *b
    *b = temp
}

// XOR almashtirish (qo'shimcha xotirasiz)
func SwapXOR(a, b *int) {
    *a ^= *b  // a = a ⊕ b
    *b ^= *a  // b = b ⊕ (a ⊕ b) = original_a
    *a ^= *b  // a = (a ⊕ b) ⊕ original_a = original_b
}
\`\`\`

**Nima uchun bu ishlaydi:**
- XOR xossasi: x ⊕ x = 0, x ⊕ 0 = x
- Qaytariladigan: (a ⊕ b) ⊕ b = a

**Haqiqiy dunyoda foydalanish:**
- Ichki tizimlar (cheklangan xotira)
- Kriptografiya (obfuskatsiya)
- Past darajali bit manipulyatsiyasi
- Intervyularda sevimli savol!

**Ish faoliyati eslatmasi:** Zamonaviy kompilyatorlar ikkisini ham bir xil darajada optimizatsiya qiladi, lekin XOR swap hali ham bit operatsiyalarini tushunish uchun qimmatlidir.

**2. Double Pointers: Ko'rsatkich manzillarini o'zgartirish**

Ikki darajali ko'rsatkichlar (**int) ko'rsatkichning nimaga ishora qilishini o'zgartirishga imkon beradi:

\`\`\`go
// Ko'rsatkichlar nimaga ishora qilishini almashtirish (qiymatlarni emas)
func SwapPointers(a, b **int) {
    *a, *b = *b, *a
}

// Foydalanish holati: Bog'langan ro'yxat boshini o'zgartirish
type Node struct {
    Value int
    Next  *Node
}

func RemoveHead(head **Node) {
    if head == nil || *head == nil {
        return
    }
    *head = (*head).Next  // Chaqiruvchining head ko'rsatkichini o'zgartirish
}

// Foydalanish
var list *Node = &Node{Value: 1, Next: &Node{Value: 2}}
RemoveHead(&list)  // list endi ikkinchi nodega ishora qiladi
\`\`\`

**Production patterni:** Go wrap qiladigan C-style APIlarda (cgo), ma'lumotlar strukturalarini amalga oshirishda va resurs puli boshqaruvida ishlatiladi.

**3. In-Place massiv teskarilashи**

Ko'rsatkichlar massivini o'z joyida teskari aylantirish intervyudagi keng tarqalgan savol va amaliy texnika:

\`\`\`go
// Ikki ko'rsatkichli texnika
func ReversePointerArray(arr []*int) {
    left, right := 0, len(arr)-1
    for left < right {
        arr[left], arr[right] = arr[right], arr[left]
        left++
        right--
    }
}
\`\`\`

**Vaqt murakkabligi:** O(n)
**Bo'sh joy murakkabligi:** O(1) - chinakam o'z joyida

**Haqiqiy dunyoda foydalanish:**
- Tasvirni qayta ishlash (piksellarni aylantirish)
- Satr teskari aylantirish algoritmlari
- Undo/Redo stacklar
- Aylanma bufer manipulyatsiyasi

**4. Shartli Almashtirish: Saralash qurilish bloki**

SwapIfGreater pufakcha va tanlov saralashning asosiy operatsiyasi:

\`\`\`go
// Shartli almashtirish bilan pufakcha saralash
func BubbleSort(arr []int) {
    n := len(arr)
    for i := 0; i < n; i++ {
        for j := 0; j < n-i-1; j++ {
            SwapIfGreater(&arr[j], &arr[j+1])
        }
    }
}

// Quicksort da bo'lim
func partition(arr []int, low, high int) int {
    pivot := arr[high]
    i := low - 1
    for j := low; j < high; j++ {
        if arr[j] < pivot {
            i++
            SwapIfGreater(&arr[i], &arr[j])
        }
    }
    i++
    arr[i], arr[high] = arr[high], arr[i]
    return i
}
\`\`\`

**5. Uch tomonlama aylanish: Aylanma siljishlar**

Uch qiymatni aylantirish saralangan invariantlarni saqlagan algoritmlarda foydalidir:

\`\`\`go
// O'ngga aylantirish: a→b, b→c, c→a
func RotateThree(a, b, c *int) {
    temp := *a
    *a = *c
    *c = *b
    *b = temp
}

// Foydalanish holati: Uchdan median pivot tanlash
func medianOfThree(arr []int, low, mid, high int) {
    if arr[low] > arr[mid] {
        swap(&arr[low], &arr[mid])
    }
    if arr[mid] > arr[high] {
        swap(&arr[mid], &arr[high])
    }
    if arr[low] > arr[mid] {
        swap(&arr[low], &arr[mid])
    }
    // Endi arr[low] <= arr[mid] <= arr[high]
}
\`\`\`

**Qayerda ishlatiladi:**
- QuickSort pivot tanlash
- Graf algoritmlari (tsikl aniqlash)
- Holat mashinasi o'tishlar
- Halqa bufer boshqaruvi

**6. Production misoli: Lock-Free almashtirish**

Ilg'or ko'rsatkich almashtirish parallel dasturlashda muhim:

\`\`\`go
import "sync/atomic"

type Node struct {
    Value int
    Next  *Node
}

// Atomic compare-and-swap yordamida lock-free stack push
func (s *Stack) Push(value int) {
    newNode := &Node{Value: value}
    for {
        old := atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&s.head)))
        newNode.Next = (*Node)(old)
        if atomic.CompareAndSwapPointer(
            (*unsafe.Pointer)(unsafe.Pointer(&s.head)),
            old,
            unsafe.Pointer(newNode),
        ) {
            return
        }
        // CAS muvaffaqiyatsiz, qayta urinish
    }
}
\`\`\`

Bu pattern quyidagilarda ishlatiladi:
- Go ning sync.Pool
- Lock-free ma'lumotlar strukturalari
- Xotira ajratuvchilar
- Yuqori unumli navbatlar

**7. Struktura maydonlarini almashtirish: In-Place transformatsiyalar**

Struktura maydonlarini almashtirish hisoblash geometriyasi va grafikada keng tarqalgan:

\`\`\`go
type Point struct {
    X, Y float64
}

// Nuqtani soat yo'nalishi bo'yicha 90 darajaga aylantirish
func Rotate90CW(p *Point) {
    p.X, p.Y = p.Y, -p.X
}

// Nuqtani Y o'qi bo'ylab aks ettirish
func MirrorY(p *Point) {
    p.X = -p.X
}

// Matritsani transpoz qilish (qatorlar/ustunlarni almashtirish)
type Matrix struct {
    A, B, C, D float64
}

func Transpose(m *Matrix) {
    m.B, m.C = m.C, m.B
}
\`\`\`

**8. Ish faoliyati xususiyatlari**

\`\`\`
Operatsiya           Vaqt    Bo'shliq   Cache-Friendly
An'anaviy Swap       O(1)    O(1)       Ha
XOR Swap             O(1)    O(0)       Ha
Pointer Swap         O(1)    O(1)       Ha
Array Reverse        O(n)    O(1)       Ketma-ket
Rotate Three         O(1)    O(1)       Ha
\`\`\`

**9. Keng tarqalgan xatolar**

**Xato 1: Bir xil ko'rsatkich bilan XOR**
\`\`\`go
// NOTO'G'RI - a va b bir joyga ishora qilsa natija nolga teng
SwapXOR(&x, &x)  // x 0 ga aylanadi!
\`\`\`

**Xato 2: Nil tekshiruvlarni unutish**
\`\`\`go
// NOTO'G'RI - nil da panika
func SwapInts(a, b *int) {
    *a ^= *b  // Agar biri nil bo'lsa PANIKA
}

// TO'G'RI
func SwapInts(a, b *int) {
    if a == nil || b == nil {
        return
    }
    *a ^= *b
    *b ^= *a
    *a ^= *b
}
\`\`\`

**Xato 3: Ikki darajali ko'rsatkichda chalkashlik**
\`\`\`go
// NOTO'G'RI - lokal nusxalarni almashtiradi
func SwapPointers(a, b *int) {
    a, b = b, a  // Faqat funksiya parametrlarini almashtiradi!
}

// TO'G'RI - chaqiruvchining ko'rsatkichlarini o'zgartiradi
func SwapPointers(a, b **int) {
    *a, *b = *b, *a
}
\`\`\`

**10. Bu patternlardan foydalanadigan intervyu savollari**

1. **Bog'langan ro'yxatni o'z joyida teskari aylantirish** - Ko'rsatkich almashtirish ishlatadi
2. **Massivni k pozitsiyaga aylantirish** - Ko'p qiymatli aylanishdan foydalanadi
3. **Massivni qo'shimcha bo'shliqsiz saralash** - Shartli almashtirish ishlatadi
4. **Bog'langan ro'yxatda tsiklni aniqlash** - Ikki ko'rsatkichli almashtirish ishlatadi
5. **Gollandiya bayrog'i muammosi** - O'z joyida almashtirish ishlatadi

**Haqiqiy dunyoda ta'sir:**

- Linux yadrosi ba'zi past darajali kodda XOR swap ishlatadi
- Redis xotira samaradorligi uchun ko'rsatkich almashtirish ishlatadi
- O'yin dvigatellari transformatsiyalar uchun o'z joyida aylanishlardan foydalanadi
- Ma'lumotlar bazalari B-daraxt saqlashda shartli almashtirishdan foydalanadi

**Asosiy xulosalar:**
- XOR swap vaqtinchalik o'zgaruvchilarni yo'q qiladi (xotira samaradorligi)
- Double pointers ko'rsatkich manzillarini o'zgartirishga imkon beradi
- In-place operatsiyalar xotirani tejaydi (O(1) bo'shliq)
- Shartli almashtirish saralashning qurilish bloklari
- Ko'p qiymatli aylanish aylanma siljishlarga imkon beradi
- Bu patternlar texnik intervyularda paydo bo'ladi
- Ko'rsatkichlarni chuqur tushunish tizim dasturlash uchun muhim`,
			solutionCode: `package pointersx

type Pair struct {
	First  int // juftlikdagi birinchi element
	Second int // juftlikdagi ikkinchi element
}

func SwapInts(a, b *int) {
	if a == nil || b == nil { // XOR operatsiyalarida nil dereferenceni oldini olish
		return
	}
	*a ^= *b // a a XOR b ga aylanadi
	*b ^= *a // b (a XOR b) XOR original_b = original_a ga aylanadi
	*a ^= *b // a (a XOR b) XOR original_a = original_b ga aylanadi
}

func SwapPointers(a, b **int) {
	if a == nil || b == nil { // nil ikki darajali ko'rsatkichlardan himoyalanish
		return
	}
	*a, *b = *b, *a // ko'rsatkich o'zgaruvchilarida saqlangan manzillarni almashtirish
}

func RotateThree(a, b, c *int) {
	if a == nil || b == nil || c == nil { // barcha uchta ko'rsatkich yaroqli bo'lishi kerak
		return
	}
	temp := *a     // a ning asl qiymatini saqlash
	*a = *c        // c ning qiymatini a ga ko'chirish
	*c = *b        // b ning qiymatini c ga ko'chirish
	*b = temp      // saqlangan a ning qiymatini b ga ko'chirish
}

func SwapIfGreater(a, b *int) {
	if a == nil || b == nil { // nil ko'rsatkichlarni solishtirish mumkin emas
		return
	}
	if *a > *b { // qiymatlar tartibsiz ekanligini tekshirish
		*a, *b = *b, *a // o'sish tartibini ta'minlash uchun almashtirish
	}
}

func SwapStructFields(s *Pair) {
	if s == nil { // nil struktura ko'rsatkichi maydonlar almashtirilishi mumkin emas
		return
	}
	s.First, s.Second = s.Second, s.First // ikki maydon qiymatlarini almashtirish
}

func ReversePointerArray(arr []*int) {
	left, right := 0, len(arr)-1 // ikki ko'rsatkichli indekslarni initsializatsiya qilish
	for left < right {           // ko'rsatkichlar o'rtada uchrashguncha davom etish
		arr[left], arr[right] = arr[right], arr[left] // simmetrik pozitsiylardagi ko'rsatkichlarni almashtirish
		left++  // chap ko'rsatkichni markazga siljitish
		right-- // o'ng ko'rsatkichni markazga siljitish
	}
}

func SwapAndSum(a, b *int) int {
	if a == nil || b == nil { // nil ko'rsatkichlar bilan almashtirib yoki yig'ib bo'lmaydi
		return 0 // noto'g'ri kirish uchun neytral qiymat qaytarish
	}
	*a, *b = *b, *a       // qiymatlarni almashtirish
	return *a + *b        // almashtirilgan qiymatlar yig'indisini hisoblash
}`
		}
	}
};

export default task;
