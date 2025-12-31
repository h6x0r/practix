import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-ap-golden-hammer-advanced',
	title: 'Golden Hammer Anti-pattern - Advanced',
	difficulty: 'medium',
	tags: ['go', 'anti-patterns', 'golden-hammer', 'refactoring'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Choose appropriate data structures based on usage patterns instead of using maps for everything.

**The Problem:**

Using map[string]interface{} for everything when specialized data structures would be better.

**You will implement:**
1. **Stack** - LIFO operations (use slice)
2. **Queue** - FIFO operations (use slice)
3. **Set** - Unique elements (use map[string]struct{})

**Your Task:**

Implement proper data structures instead of forcing maps everywhere.`,
	initialCode: `package antipatterns

type Stack struct {
	items []string
}

func NewStack() *Stack {
}

func (s *Stack) Push(item string) {
}

func (s *Stack) Pop() string {
}

type Queue struct {
	items []string
}

func NewQueue() *Queue {
}

func (q *Queue) Enqueue(item string) {
}

func (q *Queue) Dequeue() string {
}

type Set struct {
	items map[string]struct{}
}

func NewSet() *Set {
}

func (s *Set) Add(item string) {
}

func (s *Set) Contains(item string) bool {
}

func (s *Set) Size() int {
}`,
	solutionCode: `package antipatterns

// Stack - LIFO using slice (right tool for the job)
type Stack struct {
	items []string	// slice is perfect for stack operations
}

func NewStack() *Stack {
	return &Stack{items: []string{}}
}

func (s *Stack) Push(item string) {
	s.items = append(s.items, item)	// append to end
}

func (s *Stack) Pop() string {
	if len(s.items) == 0 {
		return ""	// empty stack
	}
	item := s.items[len(s.items)-1]	// get last item
	s.items = s.items[:len(s.items)-1]	// remove last item
	return item
}

// Queue - FIFO using slice (right tool for the job)
type Queue struct {
	items []string	// slice works for queue
}

func NewQueue() *Queue {
	return &Queue{items: []string{}}
}

func (q *Queue) Enqueue(item string) {
	q.items = append(q.items, item)	// append to end
}

func (q *Queue) Dequeue() string {
	if len(q.items) == 0 {
		return ""	// empty queue
	}
	item := q.items[0]		// get first item
	q.items = q.items[1:]	// remove first item
	return item
}

// Set - Unique elements using map[string]struct{} (right tool)
// struct{} uses zero memory - more efficient than map[string]bool
type Set struct {
	items map[string]struct{}	// map is perfect for O(1) lookup
}

func NewSet() *Set {
	return &Set{items: make(map[string]struct{})}
}

func (s *Set) Add(item string) {
	s.items[item] = struct{}{}	// struct{} literal, zero memory
}

func (s *Set) Contains(item string) bool {
	_, exists := s.items[item]	// O(1) lookup
	return exists
}

func (s *Set) Size() int {
	return len(s.items)	// number of keys
}`,
	hint1: `Stack: Push appends to slice, Pop gets last element and truncates slice. Queue: Enqueue appends, Dequeue gets first and reslices. Set: use map[string]struct{} for O(1) operations.`,
	hint2: `Check len before Pop/Dequeue to avoid panic. Set.Add: s.items[item] = struct{}{}. Set.Contains: use _, exists := s.items[item]. Set.Size: return len(s.items).`,
	whyItMatters: `Choosing the right data structure for your use case dramatically improves performance and code clarity.

**Performance Comparison:**

\`\`\`go
// BAD: Using map for stack
type BadStack struct {
	items map[int]string  // map for sequential access!
	top   int
}

func (s *BadStack) Push(item string) {
	s.items[s.top] = item  // O(1) but wasteful
	s.top++
}
// Slow, uses more memory, harder to understand

// GOOD: Using slice for stack
type GoodStack struct {
	items []string  // slice is perfect for sequential data
}

func (s *GoodStack) Push(item string) {
	s.items = append(s.items, item)  // O(1) amortized, clear intent
}
// Fast, memory efficient, idiomatic Go
\`\`\`

**Real-World Impact:**

Using the right data structure can make the difference between milliseconds and seconds in production!`,
	order: 5,
	testCode: `package antipatterns

import (
	"testing"
)

// Test1: Stack Push and Pop LIFO order
func Test1(t *testing.T) {
	s := NewStack()
	s.Push("a")
	s.Push("b")
	if s.Pop() != "b" {
		t.Error("Expected 'b' (LIFO)")
	}
}

// Test2: Stack Pop on empty returns empty string
func Test2(t *testing.T) {
	s := NewStack()
	if s.Pop() != "" {
		t.Error("Expected empty string from empty stack")
	}
}

// Test3: Queue Enqueue and Dequeue FIFO order
func Test3(t *testing.T) {
	q := NewQueue()
	q.Enqueue("a")
	q.Enqueue("b")
	if q.Dequeue() != "a" {
		t.Error("Expected 'a' (FIFO)")
	}
}

// Test4: Queue Dequeue on empty returns empty string
func Test4(t *testing.T) {
	q := NewQueue()
	if q.Dequeue() != "" {
		t.Error("Expected empty string from empty queue")
	}
}

// Test5: Set Add and Contains
func Test5(t *testing.T) {
	s := NewSet()
	s.Add("apple")
	if !s.Contains("apple") {
		t.Error("Set should contain 'apple'")
	}
}

// Test6: Set Contains returns false for missing item
func Test6(t *testing.T) {
	s := NewSet()
	if s.Contains("banana") {
		t.Error("Set should not contain 'banana'")
	}
}

// Test7: Set Size counts unique elements
func Test7(t *testing.T) {
	s := NewSet()
	s.Add("a")
	s.Add("b")
	s.Add("a") // duplicate
	if s.Size() != 2 {
		t.Errorf("Expected size 2, got %d", s.Size())
	}
}

// Test8: NewStack returns non-nil
func Test8(t *testing.T) {
	s := NewStack()
	if s == nil {
		t.Error("NewStack should return non-nil")
	}
}

// Test9: NewQueue returns non-nil
func Test9(t *testing.T) {
	q := NewQueue()
	if q == nil {
		t.Error("NewQueue should return non-nil")
	}
}

// Test10: NewSet returns non-nil with Size 0
func Test10(t *testing.T) {
	s := NewSet()
	if s == nil || s.Size() != 0 {
		t.Error("NewSet should return non-nil with size 0")
	}
}
`,
	translations: {
		ru: {
			title: 'Антипаттерн Golden Hammer - Продвинутый',
			description: `Выбирайте подходящие структуры данных на основе паттернов использования вместо использования map для всего.`,
			hint1: `Stack: Push добавляет в slice, Pop получает последний элемент и обрезает slice. Queue: Enqueue добавляет, Dequeue получает первый и переделывает slice. Set: используйте map[string]struct{} для O(1) операций.`,
			hint2: `Проверяйте len перед Pop/Dequeue чтобы избежать panic. Set.Add: s.items[item] = struct{}{}. Set.Contains: используйте _, exists := s.items[item]. Set.Size: верните len(s.items).`,
			whyItMatters: `Выбор правильной структуры данных для вашего случая значительно улучшает производительность и ясность кода.`,
			solutionCode: `package antipatterns

type Stack struct {
	items []string
}

func NewStack() *Stack {
	return &Stack{items: []string{}}
}

func (s *Stack) Push(item string) {
	s.items = append(s.items, item)
}

func (s *Stack) Pop() string {
	if len(s.items) == 0 {
		return ""
	}
	item := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return item
}

type Queue struct {
	items []string
}

func NewQueue() *Queue {
	return &Queue{items: []string{}}
}

func (q *Queue) Enqueue(item string) {
	q.items = append(q.items, item)
}

func (q *Queue) Dequeue() string {
	if len(q.items) == 0 {
		return ""
	}
	item := q.items[0]
	q.items = q.items[1:]
	return item
}

type Set struct {
	items map[string]struct{}
}

func NewSet() *Set {
	return &Set{items: make(map[string]struct{})}
}

func (s *Set) Add(item string) {
	s.items[item] = struct{}{}
}

func (s *Set) Contains(item string) bool {
	_, exists := s.items[item]
	return exists
}

func (s *Set) Size() int {
	return len(s.items)
}`
		},
		uz: {
			title: 'Golden Hammer Anti-pattern - Ilg\'or',
			description: `Hamma narsa uchun map ishlatish o'rniga foydalanish patternlariga asoslangan mos ma'lumot strukturalarini tanlang.`,
			hint1: `Stack: Push slice ga qo'shadi, Pop oxirgi elementni oladi va slice ni qisqartiradi. Queue: Enqueue qo'shadi, Dequeue birinchisini oladi va slice ni qayta kesadi. Set: O(1) operatsiyalar uchun map[string]struct{} ishlating.`,
			hint2: `Panic dan qochish uchun Pop/Dequeue dan oldin len ni tekshiring. Set.Add: s.items[item] = struct{}{}. Set.Contains: _, exists := s.items[item] ishlating. Set.Size: len(s.items) ni qaytaring.`,
			whyItMatters: `Sizning holatlaringiz uchun to'g'ri ma'lumot strukturasini tanlash ishlash va kod ravshanligini sezilarli darajada yaxshilaydi.`,
			solutionCode: `package antipatterns

type Stack struct {
	items []string
}

func NewStack() *Stack {
	return &Stack{items: []string{}}
}

func (s *Stack) Push(item string) {
	s.items = append(s.items, item)
}

func (s *Stack) Pop() string {
	if len(s.items) == 0 {
		return ""
	}
	item := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return item
}

type Queue struct {
	items []string
}

func NewQueue() *Queue {
	return &Queue{items: []string{}}
}

func (q *Queue) Enqueue(item string) {
	q.items = append(q.items, item)
}

func (q *Queue) Dequeue() string {
	if len(q.items) == 0 {
		return ""
	}
	item := q.items[0]
	q.items = q.items[1:]
	return item
}

type Set struct {
	items map[string]struct{}
}

func NewSet() *Set {
	return &Set{items: make(map[string]struct{})}
}

func (s *Set) Add(item string) {
	s.items[item] = struct{}{}
}

func (s *Set) Contains(item string) bool {
	_, exists := s.items[item]
	return exists
}

func (s *Set) Size() int {
	return len(s.items)
}`
		}
	}
};

export default task;
