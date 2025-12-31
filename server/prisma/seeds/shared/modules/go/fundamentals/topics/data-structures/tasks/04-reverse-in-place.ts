import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-fundamentals-reverse-in-place',
	title: 'In-Place Slice Reversal',
	difficulty: 'medium',	tags: ['go', 'data-structures', 'maps/slices/strings', 'generics'],
	estimatedTime: '15-20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **ReverseInPlace** that reverses a slice without allocating additional memory.

**Requirements:**
1. Create function \`ReverseInPlace[T any](in []T)\`
2. Modify the slice in-place (no new allocation)
3. Use two-pointer technique from both ends
4. Swap elements moving towards the center
5. Stop when pointers meet in the middle
6. Work with any type via generics
7. Handle edge cases (empty, single element)

**Example:**
\`\`\`go
nums := []int{1, 2, 3, 4, 5}
ReverseInPlace(nums)
// nums = []int{5, 4, 3, 2, 1} (modified in-place)

strs := []string{"a", "b", "c"}
ReverseInPlace(strs)
// strs = []string{"c", "b", "a"}

single := []float64{42.0}
ReverseInPlace(single)
// single = []float64{42.0} (unchanged)
\`\`\`

**Constraints:**
- Must use in-place reversal (no extra allocation)
- Must use two-pointer technique
- Must work with any type
- Must handle empty and single-element slices
- Must not create intermediate slices`,
	initialCode: `package datastructures

// TODO: Implement ReverseInPlace
func ReverseInPlace[T any](in []T) {
	// TODO: Implement
}`,
	solutionCode: `package datastructures

func ReverseInPlace[T any](in []T) {
	for left, right := 0, len(in)-1; left < right; left, right = left+1, right-1 {
		in[left], in[right] = in[right], in[left]                  // Swap elements from left and right
	}                                                               // Continue until pointers meet
}`,
	testCode: `package datastructures

import (
	"reflect"
	"testing"
)

func Test1(t *testing.T) {
	// Basic reversal
	nums := []int{1, 2, 3, 4, 5}
	ReverseInPlace(nums)
	expected := []int{5, 4, 3, 2, 1}
	if !reflect.DeepEqual(nums, expected) {
		t.Errorf("expected %v, got %v", expected, nums)
	}
}

func Test2(t *testing.T) {
	// String reversal
	strs := []string{"a", "b", "c"}
	ReverseInPlace(strs)
	expected := []string{"c", "b", "a"}
	if !reflect.DeepEqual(strs, expected) {
		t.Errorf("expected %v, got %v", expected, strs)
	}
}

func Test3(t *testing.T) {
	// Empty slice
	empty := []int{}
	ReverseInPlace(empty)
	if len(empty) != 0 {
		t.Errorf("expected empty slice, got %v", empty)
	}
}

func Test4(t *testing.T) {
	// Single element
	single := []int{42}
	ReverseInPlace(single)
	expected := []int{42}
	if !reflect.DeepEqual(single, expected) {
		t.Errorf("expected %v, got %v", expected, single)
	}
}

func Test5(t *testing.T) {
	// Two elements
	pair := []int{1, 2}
	ReverseInPlace(pair)
	expected := []int{2, 1}
	if !reflect.DeepEqual(pair, expected) {
		t.Errorf("expected %v, got %v", expected, pair)
	}
}

func Test6(t *testing.T) {
	// Even length
	even := []int{1, 2, 3, 4}
	ReverseInPlace(even)
	expected := []int{4, 3, 2, 1}
	if !reflect.DeepEqual(even, expected) {
		t.Errorf("expected %v, got %v", expected, even)
	}
}

func Test7(t *testing.T) {
	// Odd length
	odd := []int{1, 2, 3, 4, 5, 6, 7}
	ReverseInPlace(odd)
	expected := []int{7, 6, 5, 4, 3, 2, 1}
	if !reflect.DeepEqual(odd, expected) {
		t.Errorf("expected %v, got %v", expected, odd)
	}
}

func Test8(t *testing.T) {
	// Float reversal
	floats := []float64{1.1, 2.2, 3.3}
	ReverseInPlace(floats)
	expected := []float64{3.3, 2.2, 1.1}
	if !reflect.DeepEqual(floats, expected) {
		t.Errorf("expected %v, got %v", expected, floats)
	}
}

func Test9(t *testing.T) {
	// Negative numbers
	neg := []int{-1, -2, -3}
	ReverseInPlace(neg)
	expected := []int{-3, -2, -1}
	if !reflect.DeepEqual(neg, expected) {
		t.Errorf("expected %v, got %v", expected, neg)
	}
}

func Test10(t *testing.T) {
	// Double reversal returns original
	orig := []int{1, 2, 3}
	ReverseInPlace(orig)
	ReverseInPlace(orig)
	expected := []int{1, 2, 3}
	if !reflect.DeepEqual(orig, expected) {
		t.Errorf("expected %v, got %v", expected, orig)
	}
}`,
	hint1: `Use two pointers starting from both ends, swap elements, and move pointers toward the center until they meet.`,
			hint2: `The loop condition should be left < right. When they meet or cross, the reversal is complete.`,
			whyItMatters: `ReverseInPlace demonstrates efficient memory usage and the two-pointer pattern, crucial for performance-critical operations where memory allocation overhead can impact system performance.

**Why In-Place Reversal:**
- **Memory Efficiency:** No extra allocation for large datasets
- **Cache Performance:** Modifying existing array maintains cache locality
- **Garbage Collection:** Fewer allocations mean less GC pressure
- **Performance:** Essential for processing billions of records
- **Resource Constraints:** Critical in embedded systems or memory-limited environments

**Production Pattern:**
\`\`\`go
// Reverse log entries for display (newest first)
func DisplayLogsSortedByTime(logs []LogEntry) {
    ReverseInPlace(logs)
    for _, log := range logs {
        fmt.Println(log.Timestamp, log.Message)
    }
}

// Undo stack - reverse operation history
type UndoStack struct {
    operations []Operation
}

func (us *UndoStack) ReverseForRedo() {
    ReverseInPlace(us.operations)
}

// Parse command arguments in reverse
func ProcessArgumentsBackwards(args []string) []string {
    ReverseInPlace(args)
    return args
}

// Reverse TCP packet sequence for debugging
type PacketCapture struct {
    packets []Packet
}

func (pc *PacketCapture) ReversePacketOrder() {
    ReverseInPlace(pc.packets)
}

// In-memory log rotation - reverse and reorder logs
func RotateLogs(logs []LogLine, maxAge time.Duration) {
    ReverseInPlace(logs)
    cutoff := time.Now().Add(-maxAge)

    for i, log := range logs {
        if log.Time.Before(cutoff) {
            logs = logs[:i]
            break
        }
    }
}

// Stack LIFO order - process requests in reverse arrival
func ProcessRequestStack(requests []Request) {
    ReverseInPlace(requests)
    for _, req := range requests {
        handleRequest(req)
    }
}

// Path traversal in reverse (backtrack from leaf to root)
func TraversePath(path []Node) {
    ReverseInPlace(path)
    for _, node := range path {
        processNode(node)
    }
}

// Memory-efficient sorting preparation
func PrepareForSortingByReversal(items []Item) {
    ReverseInPlace(items)
}

// Data pipeline - reverse intermediate results
type DataPipeline struct {
    buffer []DataPoint
}

func (dp *DataPipeline) ReverseBufferForAnalysis() {
    ReverseInPlace(dp.buffer)
}

// Test scenario generation - reverse execution flow
func GenerateReverseScenarios(scenarios []TestScenario) {
    ReverseInPlace(scenarios)
}

// Bitwise operations - reverse bit representation
func ReverseBitArray(bits []bool) {
    ReverseInPlace(bits)
}

// Video playback - reverse frame order for backwards play
type VideoBuffer struct {
    frames []Frame
}

func (vb *VideoBuffer) PlayBackwards() {
    ReverseInPlace(vb.frames)
    for _, frame := range vb.frames {
        display(frame)
    }
}

// Transaction rollback - process in reverse order
type TransactionLog struct {
    entries []Transaction
}

func (tl *TransactionLog) RollbackInReverseOrder() {
    ReverseInPlace(tl.entries)
    for _, entry := range tl.entries {
        entry.Rollback()
    }
}
\`\`\`

**Real-World Benefits:**
- **Large Data Processing:** Reverse multi-gigabyte datasets without doubling memory
- **Streaming:** Reverse chunk buffers in data pipelines without extra allocation
- **Gaming:** Reverse animation frames, particles, or sprite buffers efficiently
- **Time Series:** Reverse chronological data for analysis (newest first)
- **Networking:** Reverse packet sequences for debugging and analysis

**Performance Comparison:**
- **In-place:** O(n/2) swaps, no allocation
- **Creating new slice:** n copy operations + allocation overhead
- **Recursive reversal:** n stack frames overhead

**Common Use Cases:**
- Undo/redo stacks
- Displaying newest items first
- Processing historical data backwards
- Reversing operation order for cleanup
- Bidirectional iteration support

Without ReverseInPlace, processing multi-gigabyte datasets would require double the memory allocation, causing OOM errors and massive GC pauses in production systems.`,	order: 3,
	translations: {
		ru: {
			title: 'Разворот массива без выделения памяти',
			description: `Реализуйте **ReverseInPlace**, который разворачивает слайс без выделения дополнительной памяти.

**Требования:**
1. Создайте функцию \`ReverseInPlace[T any](in []T)\`
2. Модифицируйте слайс in-place (без новых выделений)
3. Используйте two-pointer технику с обоих концов
4. Меняйте элементы двигаясь к центру
5. Остановитесь когда указатели встретятся в центре
6. Работайте с любым типом через generics
7. Обработайте граничные случаи (пусто, один элемент)

**Пример:**
\`\`\`go
nums := []int{1, 2, 3, 4, 5}
ReverseInPlace(nums)
// nums = []int{5, 4, 3, 2, 1} (модифицирован in-place)

strs := []string{"a", "b", "c"}
ReverseInPlace(strs)
// strs = []string{"c", "b", "a"}

single := []float64{42.0}
ReverseInPlace(single)
// single = []float64{42.0} (не изменён)
\`\`\`

**Ограничения:**
- Должен использовать in-place reversal (без доп выделений)
- Должен использовать two-pointer технику
- Должен работать с любым типом
- Должен обработать пустые и single-element слайсы
- Не должен создавать промежуточные слайсы`,
			hint1: `Используйте два указателя начиная с обоих концов, меняйте элементы и двигайте указатели к центру пока они не встретятся.`,
			hint2: `Условие цикла должно быть left < right. Когда они встретятся или пересекутся, reversal завершён.`,
			whyItMatters: `ReverseInPlace демонстрирует эффективное использование памяти и two-pointer паттерн, критические для performance-critical операций где overhead выделения памяти может повлиять на системную производительность.

**Почему In-Place Reversal:**
- **Эффективность памяти:** Нет доп выделения для больших наборов данных
- **Cache производительность:** Модификация существующего массива поддерживает cache locality
- **Garbage Collection:** Меньше выделений означает меньше GC pressure
- **Производительность:** Необходимо для обработки миллиардов записей
- **Resource constraints:** Критично в embedded системах или memory-limited окружениях

**Production Pattern:**
\`\`\`go
// Разворот log entries для отображения (newest first)
func DisplayLogsSortedByTime(logs []LogEntry) {
    ReverseInPlace(logs)
    for _, log := range logs {
        fmt.Println(log.Timestamp, log.Message)
    }
}

// Undo stack - разворот истории операций
type UndoStack struct {
    operations []Operation
}

func (us *UndoStack) ReverseForRedo() {
    ReverseInPlace(us.operations)
}

// Обработка аргументов команд в обратном порядке
func ProcessArgumentsBackwards(args []string) []string {
    ReverseInPlace(args)
    return args
}

// Разворот последовательности TCP пакетов для отладки
type PacketCapture struct {
    packets []Packet
}

func (pc *PacketCapture) ReversePacketOrder() {
    ReverseInPlace(pc.packets)
}

// Ротация логов в памяти - разворот и переупорядочивание
func RotateLogs(logs []LogLine, maxAge time.Duration) {
    ReverseInPlace(logs)
    cutoff := time.Now().Add(-maxAge)

    for i, log := range logs {
        if log.Time.Before(cutoff) {
            logs = logs[:i]
            break
        }
    }
}

// Stack LIFO порядок - обработка запросов в обратном порядке поступления
func ProcessRequestStack(requests []Request) {
    ReverseInPlace(requests)
    for _, req := range requests {
        handleRequest(req)
    }
}

// Обход пути в обратном направлении (откат от листа к корню)
func TraversePath(path []Node) {
    ReverseInPlace(path)
    for _, node := range path {
        processNode(node)
    }
}

// Подготовка к сортировке с эффективной памятью
func PrepareForSortingByReversal(items []Item) {
    ReverseInPlace(items)
}

// Data pipeline - разворот промежуточных результатов
type DataPipeline struct {
    buffer []DataPoint
}

func (dp *DataPipeline) ReverseBufferForAnalysis() {
    ReverseInPlace(dp.buffer)
}

// Генерация тестовых сценариев - разворот потока выполнения
func GenerateReverseScenarios(scenarios []TestScenario) {
    ReverseInPlace(scenarios)
}

// Побитовые операции - разворот битового представления
func ReverseBitArray(bits []bool) {
    ReverseInPlace(bits)
}

// Воспроизведение видео - разворот порядка кадров для обратного воспроизведения
type VideoBuffer struct {
    frames []Frame
}

func (vb *VideoBuffer) PlayBackwards() {
    ReverseInPlace(vb.frames)
    for _, frame := range vb.frames {
        display(frame)
    }
}

// Откат транзакций - обработка в обратном порядке
type TransactionLog struct {
    entries []Transaction
}

func (tl *TransactionLog) RollbackInReverseOrder() {
    ReverseInPlace(tl.entries)
    for _, entry := range tl.entries {
        entry.Rollback()
    }
}
\`\`\`

**Практические преимущества:**
- **Обработка больших данных:** Разворот многогигабайтных наборов данных без удвоения памяти
- **Потоковая обработка:** Разворот chunk буферов в data pipelines без дополнительного выделения
- **Игры:** Эффективный разворот кадров анимации, частиц или sprite буферов
- **Временные ряды:** Разворот хронологических данных для анализа (newest first)
- **Сеть:** Разворот последовательностей пакетов для отладки и анализа

**Сравнение производительности:**
- **In-place:** O(n/2) обменов, без выделения
- **Создание нового слайса:** n операций копирования + overhead выделения
- **Рекурсивный разворот:** overhead n stack frames

**Частые случаи использования:**
- Undo/redo стеки
- Отображение новейших элементов первыми
- Обработка исторических данных в обратном направлении
- Разворот порядка операций для cleanup
- Поддержка двунаправленной итерации

Без ReverseInPlace обработка многогигабайтных наборов данных потребовала бы удвоения выделения памяти, вызывая OOM ошибки и массивные паузы GC в production системах.`,
			solutionCode: `package datastructures

func ReverseInPlace[T any](in []T) {
	for left, right := 0, len(in)-1; left < right; left, right = left+1, right-1 {
		in[left], in[right] = in[right], in[left]                  // Обменять элементы слева и справа
	}                                                               // Продолжать пока указатели не встретятся
}`
		},
		uz: {
			title: 'Massivni joyida teskari aylantirish',
			description: `Qo'shimcha xotira ajratilmasdan slaysni teskari buraydiagan **ReverseInPlace** ni amalga oshiring.

**Talablar:**
1. \`ReverseInPlace[T any](in []T)\` funksiyasini yarating
2. Slaysni in-place o'zgartiring (yangi ajratilmagan)
3. Ikkala uchidan two-pointer teknikasidan foydalaning
4. Elementlarni markazga siljiyotganda almashtiring
5. Pointerlar markazda uchrashganda to'xtang
6. Generics orqali har qanday tipi bilan ishlang
7. Edge caslarni ishlang (bo'sh, bitta element)

**Misol:**
\`\`\`go
nums := []int{1, 2, 3, 4, 5}
ReverseInPlace(nums)
// nums = []int{5, 4, 3, 2, 1} (in-place o'zgartirilgan)

strs := []string{"a", "b", "c"}
ReverseInPlace(strs)
// strs = []string{"c", "b", "a"}

single := []float64{42.0}
ReverseInPlace(single)
// single = []float64{42.0} (o'zgartirilmagan)
\`\`\`

**Cheklovlar:**
- In-place reversal foydalanishi kerak (qo'shimcha ajratilmasdan)
- Two-pointer teknikadan foydalanishi kerak
- Har qanday tipi bilan ishlashi kerak
- Bo'sh va single-element slayslarni ishlashi kerak
- Oraliq slayslarni yaratmasligi kerak`,
			hint1: `Ikkala uchidan boshlab ikkita pointerdan foydalanib, elementlarni almashtiring va pointerlarni markazga siljiyting toki ular uchrashmaguncha.`,
			hint2: `Loop sharti left < right bo'lishi kerak. Ular uchrashganda yoki kesishganda, reversal tugallangan.`,
			whyItMatters: `ReverseInPlace samarali xotira foydalanishni va two-pointer patternni ko'rsatadi, performance-critical operatsiyalar uchun critical bo'lib memory ajratilishi sistemaning samaradorligiga ta'sir qilishi mumkin.

**Nima uchun In-Place Reversal:**
- **Xotira samaradorligi:** Katta ma'lumot to'plamlar uchun qo'shimcha ajratilmagan
- **Cache samaradorligi:** Mavjud massivni o'zgartirish cache locality ni saqlaydi
- **Garbage Collection:** Kamroq ajratilish GC pressurni kamaytirib beradi
- **Samaradorlik:** Milyardlab yozuvlarni qayta ishlash uchun kerak
- **Resource constraints:** Embedded sistemalar yoki memory-limited muhitlarda kritik

**Production Pattern:**
\`\`\`go
// Ko'rsatish uchun log entries ni teskari burayish (newest first)
func DisplayLogsSortedByTime(logs []LogEntry) {
    ReverseInPlace(logs)
    for _, log := range logs {
        fmt.Println(log.Timestamp, log.Message)
    }
}

// Undo stack - operatsiyalar tarixini teskari burayish
type UndoStack struct {
    operations []Operation
}

func (us *UndoStack) ReverseForRedo() {
    ReverseInPlace(us.operations)
}

// Buyruq argumentlarni teskari tartibda qayta ishlash
func ProcessArgumentsBackwards(args []string) []string {
    ReverseInPlace(args)
    return args
}

// Debug uchun TCP paket ketma-ketligini teskari burayish
type PacketCapture struct {
    packets []Packet
}

func (pc *PacketCapture) ReversePacketOrder() {
    ReverseInPlace(pc.packets)
}

// Xotiradagi log rotatsiyasi - teskari burayish va qayta tartiblash
func RotateLogs(logs []LogLine, maxAge time.Duration) {
    ReverseInPlace(logs)
    cutoff := time.Now().Add(-maxAge)

    for i, log := range logs {
        if log.Time.Before(cutoff) {
            logs = logs[:i]
            break
        }
    }
}

// Stack LIFO tartib - so'rovlarni teskari kelish tartibida qayta ishlash
func ProcessRequestStack(requests []Request) {
    ReverseInPlace(requests)
    for _, req := range requests {
        handleRequest(req)
    }
}

// Yo'lni teskari aylanish (bargdan ildizga qaytish)
func TraversePath(path []Node) {
    ReverseInPlace(path)
    for _, node := range path {
        processNode(node)
    }
}

// Xotira samarali saralash tayyorgarlik
func PrepareForSortingByReversal(items []Item) {
    ReverseInPlace(items)
}

// Data pipeline - oraliq natijalarni teskari burayish
type DataPipeline struct {
    buffer []DataPoint
}

func (dp *DataPipeline) ReverseBufferForAnalysis() {
    ReverseInPlace(dp.buffer)
}

// Test stsenariylarini yaratish - bajarilish oqimini teskari burayish
func GenerateReverseScenarios(scenarios []TestScenario) {
    ReverseInPlace(scenarios)
}

// Bitwise operatsiyalar - bit tasviri teskari burayish
func ReverseBitArray(bits []bool) {
    ReverseInPlace(bits)
}

// Video qayta ishlatish - orqaga ijro uchun freym tartibini teskari burayish
type VideoBuffer struct {
    frames []Frame
}

func (vb *VideoBuffer) PlayBackwards() {
    ReverseInPlace(vb.frames)
    for _, frame := range vb.frames {
        display(frame)
    }
}

// Tranzaksiya rollback - teskari tartibda qayta ishlash
type TransactionLog struct {
    entries []Transaction
}

func (tl *TransactionLog) RollbackInReverseOrder() {
    ReverseInPlace(tl.entries)
    for _, entry := range tl.entries {
        entry.Rollback()
    }
}
\`\`\`

**Amaliy afzalliklar:**
- **Katta ma'lumot qayta ishlash:** Ko'p gigabaytli ma'lumot to'plamlarini xotira ikki baravar oshirmasdan teskari burayish
- **Streaming:** Data pipeline larda chunk buferlarini qo'shimcha ajratmasdan teskari burayish
- **O'yinlar:** Animatsiya freymlar, zarrachalar yoki sprite buferlarini samarali teskari burayish
- **Vaqt qatorlari:** Tahlil uchun xronologik ma'lumotlarni teskari burayish (newest first)
- **Tarmoq:** Debug va tahlil uchun paket ketma-ketliklarini teskari burayish

**Samaradorlik taqqoslash:**
- **In-place:** O(n/2) ta almashtirish, ajratilish yo'q
- **Yangi slayz yaratish:** n ta nusxalash operatsiyasi + ajratilish overhead
- **Rekursiv teskari burayish:** n ta stack frame overhead

**Umumiy foydalanish holatlari:**
- Undo/redo stacklar
- Yangi elementlarni birinchi bo'lib ko'rsatish
- Tarixiy ma'lumotlarni teskari yo'nalishda qayta ishlash
- Cleanup uchun operatsiyalar tartibini teskari burayish
- Ikki tomonlama iteratsiya qo'llab-quvvatlash

ReverseInPlace siz, ko'p gigabaytli ma'lumot to'plamlarini qayta ishlash xotira ajratilishini ikki baravar oshirishni talab qilgan bo'lar edi, production tizimlarda OOM xatolar va katta GC pauzalarga olib keladi.`,
			solutionCode: `package datastructures

func ReverseInPlace[T any](in []T) {
	for left, right := 0, len(in)-1; left < right; left, right = left+1, right-1 {
		in[left], in[right] = in[right], in[left]                  // Chap va o'ng elementlarni almashtirish
	}                                                               // Pointerlar uchrashgunga qadar davom etish
}`
		}
	}
};

export default task;
