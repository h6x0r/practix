import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-ap-premature-optimization-basic',
	title: 'Premature Optimization - Basic',
	difficulty: 'easy',
	tags: ['go', 'anti-patterns', 'premature-optimization', 'refactoring'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Write simple, clear code first. Optimize only when you have measured performance problems.

**The Problem:**

Premature optimization is optimizing code before you know if it's actually a bottleneck. This makes code complex without proven benefit.

> "Premature optimization is the root of all evil" - Donald Knuth

**You will implement:**

Simple, readable implementations instead of "clever" optimized code.

**Implement:**
1. **Sum** - Simple sum of numbers (no optimization tricks)
2. **Contains** - Simple linear search (no complex data structures)
3. **Reverse** - Simple reverse (no bit tricks)

**Your Task:**

Write the simplest, most readable solution. Performance can be improved later if needed.`,
	initialCode: `package antipatterns

func Sum(numbers []int) int {
}

func Contains(numbers []int, target int) bool {
}

func Reverse(numbers []int) {
}`,
	solutionCode: `package antipatterns

// Sum uses simple, readable loop
// KISS: Keep It Simple, Stupid
// If this becomes a bottleneck (profile first!), then optimize
func Sum(numbers []int) int {
	sum := 0					// start at zero
	for _, n := range numbers {	// iterate through all numbers
		sum += n				// add each number
	}
	return sum					// return total
}
// Clear, correct, maintainable. Optimize only if profiling shows it's slow.

// Contains uses simple linear search
// O(n) is fine for small slices
// If you need O(1), use a map - but only if you measure the need!
func Contains(numbers []int, target int) bool {
	for _, n := range numbers {	// check each element
		if n == target {		// found it
			return true
		}
	}
	return false				// not found
}
// Readable > Clever. Optimize when you have data showing it's slow.

// Reverse uses simple two-pointer swap
// Clear algorithm, easy to understand and verify
func Reverse(numbers []int) {
	for i, j := 0, len(numbers)-1; i < j; i, j = i+1, j-1 {
		numbers[i], numbers[j] = numbers[j], numbers[i]	// swap
	}
}
// Simple is better. Don't use bit tricks unless profiling shows a problem.`,
	hint1: `Sum: initialize sum to 0, loop through numbers adding each to sum. Contains: loop through numbers, return true if found, false after loop. Reverse: two pointers i and j, swap and move toward center.`,
	hint2: `All three functions should use straightforward algorithms. No optimization tricks, no complex data structures. Focus on readability and correctness.`,
	whyItMatters: `Premature optimization wastes time, creates bugs, and makes code hard to maintain - all without proven benefit.

**Why Premature Optimization is Bad:**

**1. Complexity Without Benefit**

\`\`\`go
// BAD: Premature optimization - bit manipulation for sum
func Sum(numbers []int) int {
	// "Optimized" using bit tricks and loop unrolling
	sum := 0
	i := 0
	// Process 4 at a time (loop unrolling)
	for ; i+3 < len(numbers); i += 4 {
		sum += numbers[i] + numbers[i+1] + numbers[i+2] + numbers[i+3]
	}
	// Handle remainder
	for ; i < len(numbers); i++ {
		sum += numbers[i]
	}
	return sum
}
// Complex, hard to read, and the compiler already does this!

// GOOD: Simple and clear
func Sum(numbers []int) int {
	sum := 0
	for _, n := range numbers {
		sum += n
	}
	return sum
}
// Compiler optimizes this just as well! And it's readable!
\`\`\`

**2. Real-World Waste of Time**

\`\`\`go
// Spent 3 hours optimizing this function
func ProcessUser(user User) {
	// Complex caching, pooling, bit packing...
	// 200 lines of "optimized" code
}

// After profiling: this function takes 0.01% of total runtime!
// The real bottleneck was database queries (99% of time)
// Wasted 3 hours optimizing the wrong thing!
\`\`\`

**The Right Approach: Profile First**

\`\`\`go
// Step 1: Write simple code
func ProcessOrders(orders []Order) {
	for _, order := range orders {
		validateOrder(order)
		calculateTotal(order)
		saveToDatabase(order)
		sendEmail(order)
	}
}

// Step 2: Profile to find bottlenecks
import _ "net/http/pprof"

// Run with: go tool pprof http://localhost:6060/debug/pprof/profile
// Result: saveToDatabase takes 95% of time!

// Step 3: Optimize the ACTUAL bottleneck
func ProcessOrders(orders []Order) {
	// Batch database writes (the real bottleneck)
	validated := validateAll(orders)  // fast
	totals := calculateTotals(orders) // fast
	saveBatch(orders)                 // THIS was slow, now optimized
	sendEmailsBatch(orders)           // fast
}
// Result: 10x speedup by optimizing the right thing!
\`\`\`

**Donald Knuth's Full Quote:**

> "We should forget about small efficiencies, say about 97% of the time:
> premature optimization is the root of all evil. Yet we should not pass
> up our opportunities in that critical 3%."

**The 3-Step Rule:**

1. **Make it work** - Write simple, correct code
2. **Make it right** - Refactor for clarity
3. **Make it fast** - Optimize ONLY the measured bottlenecks

**When to Optimize:**

\`\`\`go
// DON'T optimize:
// - Without profiling data
// - For code that runs once
// - At the expense of readability
// - Based on assumptions

// DO optimize:
// - After profiling shows a bottleneck
// - When you have benchmarks
// - When performance requirements aren't met
// - After maintaining readability where possible
\`\`\`

**Real Example - String Concatenation:**

\`\`\`go
// Simple code (GOOD for small N)
func Join(words []string) string {
	result := ""
	for _, word := range words {
		result += word  // simple and clear
	}
	return result
}

// Measure first!
func BenchmarkJoin(b *testing.B) {
	words := []string{"hello", "world", "foo", "bar"}
	for i := 0; i < b.N; i++ {
		Join(words)
	}
}
// Result: 4 words = 200ns. Fast enough? Then done!

// If profile shows it's slow, THEN optimize:
func Join(words []string) string {
	return strings.Join(words, "")  // optimized
}
// But only if you measured the need!
\`\`\`

**Optimization Tips:**

1. **Profile before optimizing**: Use pprof, benchmarks
2. **Optimize algorithms first**: O(n²) → O(n log n) is better than micro-optimizations
3. **Measure the impact**: Benchmark before and after
4. **Keep it readable**: Comments explaining "why" the optimization
5. **Know when to stop**: 80% improvement is often good enough`,
	order: 10,
	testCode: `package antipatterns

import (
	"testing"
)

// Test1: Sum of positive numbers
func Test1(t *testing.T) {
	if Sum([]int{1, 2, 3, 4, 5}) != 15 {
		t.Error("Expected 15")
	}
}

// Test2: Sum of empty slice
func Test2(t *testing.T) {
	if Sum([]int{}) != 0 {
		t.Error("Expected 0 for empty slice")
	}
}

// Test3: Contains finds existing element
func Test3(t *testing.T) {
	if !Contains([]int{1, 2, 3}, 2) {
		t.Error("Should find 2")
	}
}

// Test4: Contains returns false for missing element
func Test4(t *testing.T) {
	if Contains([]int{1, 2, 3}, 5) {
		t.Error("Should not find 5")
	}
}

// Test5: Reverse simple slice
func Test5(t *testing.T) {
	nums := []int{1, 2, 3}
	Reverse(nums)
	if nums[0] != 3 || nums[2] != 1 {
		t.Errorf("Expected [3,2,1], got %v", nums)
	}
}

// Test6: Reverse empty slice
func Test6(t *testing.T) {
	nums := []int{}
	Reverse(nums) // should not panic
}

// Test7: Reverse single element
func Test7(t *testing.T) {
	nums := []int{42}
	Reverse(nums)
	if nums[0] != 42 {
		t.Error("Single element should remain 42")
	}
}

// Test8: Sum with negative numbers
func Test8(t *testing.T) {
	if Sum([]int{-1, -2, 3}) != 0 {
		t.Error("Expected 0")
	}
}

// Test9: Contains with empty slice
func Test9(t *testing.T) {
	if Contains([]int{}, 1) {
		t.Error("Empty slice should not contain any element")
	}
}

// Test10: Reverse even length slice
func Test10(t *testing.T) {
	nums := []int{1, 2, 3, 4}
	Reverse(nums)
	if nums[0] != 4 || nums[3] != 1 {
		t.Errorf("Expected [4,3,2,1], got %v", nums)
	}
}
`,
	translations: {
		ru: {
			title: 'Преждевременная оптимизация - Базовый',
			description: `Сначала пишите простой, понятный код. Оптимизируйте только когда измерили проблемы с производительностью.

**Проблема:**

Преждевременная оптимизация — это оптимизация кода до того, как вы узнали, что он действительно узкое место. Это делает код сложным без доказанной пользы.

> "Преждевременная оптимизация — корень всех зол" - Дональд Кнут`,
			hint1: `Sum: инициализируйте sum на 0, циклом пройдите numbers добавляя каждое к sum. Contains: циклом пройдите numbers, верните true если найдено, false после цикла. Reverse: два указателя i и j, меняйте местами и двигайтесь к центру.`,
			hint2: `Все три функции должны использовать простые алгоритмы. Никаких трюков оптимизации, никаких сложных структур данных. Фокус на читаемости и корректности.`,
			whyItMatters: `Преждевременная оптимизация тратит время, создаёт баги и делает код трудным для поддержки - всё без доказанной пользы.`,
			solutionCode: `package antipatterns

func Sum(numbers []int) int {
	sum := 0
	for _, n := range numbers {
		sum += n
	}
	return sum
}

func Contains(numbers []int, target int) bool {
	for _, n := range numbers {
		if n == target {
			return true
		}
	}
	return false
}

func Reverse(numbers []int) {
	for i, j := 0, len(numbers)-1; i < j; i, j = i+1, j-1 {
		numbers[i], numbers[j] = numbers[j], numbers[i]
	}
}`
		},
		uz: {
			title: 'Erta Optimizatsiya - Asosiy',
			description: `Avval oddiy, tushunarli kod yozing. Faqat ishlash muammolarini o'lchagandan keyin optimallashtiring.

**Muammo:**

Erta optimizatsiya - bu kod haqiqatda tor bo'g'in ekanligini bilishdan oldin uni optimallashtirish. Bu isbotlangan foydasiz kodni murakkablashtiradi.

> "Erta optimizatsiya barcha yomonliklarning ildizi" - Donald Knuth`,
			hint1: `Sum: sum ni 0 ga initsializatsiya qiling, numbers orqali tsikl qilib har birini sum ga qo'shing. Contains: numbers orqali tsikl qiling, topilsa true qaytaring, tsikldan keyin false. Reverse: ikki ko'rsatgich i va j, almashtiring va markazga yuring.`,
			hint2: `Barcha uchta funksiya to'g'ri algoritmlardan foydalanishi kerak. Optimizatsiya hiylalari yo'q, murakkab ma'lumot strukturalari yo'q. O'qilishi va to'g'riligiga e'tibor bering.`,
			whyItMatters: `Erta optimizatsiya vaqtni isrof qiladi, xatolar yaratadi va kodni qo'llab-quvvatlashni qiyinlashtiradi - barchasi isbotlangan foydasiz.`,
			solutionCode: `package antipatterns

func Sum(numbers []int) int {
	sum := 0
	for _, n := range numbers {
		sum += n
	}
	return sum
}

func Contains(numbers []int, target int) bool {
	for _, n := range numbers {
		if n == target {
			return true
		}
	}
	return false
}

func Reverse(numbers []int) {
	for i, j := 0, len(numbers)-1; i < j; i, j = i+1, j-1 {
		numbers[i], numbers[j] = numbers[j], numbers[i]
	}
}`
		}
	}
};

export default task;
