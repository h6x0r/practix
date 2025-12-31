import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-ap-copy-paste-advanced',
	title: 'Copy-Paste Programming - Advanced',
	difficulty: 'medium',
	tags: ['go', 'anti-patterns', 'copy-paste', 'dry', 'refactoring'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Refactor duplicated business logic using higher-order functions.

**The Problem:**

Similar functions with slightly different logic are duplicated instead of parameterized.

**You will refactor:**

Duplicate filtering functions into a generic Filter function that accepts a predicate.

**Implement:**
1. **Filter** - Generic filter with predicate function
2. **FilterEven** - Filter even numbers using Filter
3. **FilterPositive** - Filter positive numbers using Filter

**Your Task:**

Use function parameters to eliminate duplication.`,
	initialCode: `package antipatterns

func Filter(numbers []int, predicate func(int) bool) []int {
}

func FilterEven(numbers []int) []int {
}

func FilterPositive(numbers []int) []int {
}`,
	solutionCode: `package antipatterns

// Filter is a generic higher-order function
// Takes a slice and a predicate function, returns filtered slice
// Single implementation serves many use cases - DRY principle
func Filter(numbers []int, predicate func(int) bool) []int {
	result := []int{}	// start with empty slice

	for _, n := range numbers {
		if predicate(n) {	// apply predicate to each element
			result = append(result, n)	// include if predicate returns true
		}
	}

	return result
}

// FilterEven uses Filter with even-checking predicate
// No duplication - reuses generic Filter logic
func FilterEven(numbers []int) []int {
	return Filter(numbers, func(n int) bool {
		return n%2 == 0	// predicate: is even?
	})
}

// FilterPositive uses Filter with positive-checking predicate
// No duplication - reuses generic Filter logic
func FilterPositive(numbers []int) []int {
	return Filter(numbers, func(n int) bool {
		return n > 0	// predicate: is positive?
	})
}`,
	hint1: `Filter iterates through numbers, applies predicate to each element, and appends to result if predicate returns true. Return the result slice.`,
	hint2: `FilterEven calls Filter with a func(n int) bool that returns n%2 == 0. FilterPositive calls Filter with func(n int) bool that returns n > 0.`,
	whyItMatters: `Higher-order functions eliminate duplication in business logic while maintaining flexibility.

**Power of Parameterized Functions:**

\`\`\`go
// BAD: Duplicated filtering logic
func FilterAdults(users []User) []User {
	result := []User{}
	for _, u := range users {
		if u.Age >= 18 {
			result = append(result, u)
		}
	}
	return result
}

func FilterActive(users []User) []User {
	result := []User{}  // DUPLICATE loop structure
	for _, u := range users {
		if u.IsActive {
			result = append(result, u)
		}
	}
	return result
}

func FilterPremium(users []User) []User {
	result := []User{}  // DUPLICATE loop structure
	for _, u := range users {
		if u.IsPremium {
			result = append(result, u)
		}
	}
	return result
}
// 30+ lines of duplicate loop logic!

// GOOD: Generic filter function
func FilterUsers(users []User, predicate func(User) bool) []User {
	result := []User{}
	for _, u := range users {
		if predicate(u) {
			result = append(result, u)
		}
	}
	return result
}

// Now all filters are one-liners!
func FilterAdults(users []User) []User {
	return FilterUsers(users, func(u User) bool { return u.Age >= 18 })
}

func FilterActive(users []User) []User {
	return FilterUsers(users, func(u User) bool { return u.IsActive })
}

func FilterPremium(users []User) []User {
	return FilterUsers(users, func(u User) bool { return u.IsPremium })
}
// 10 lines total! And you can create new filters instantly!
\`\`\`

**DRY with Higher-Order Functions:**
- Write loop logic once
- Parameterize the varying part
- Reuse for all similar operations`,
	order: 7,
	testCode: `package antipatterns

import (
	"testing"
)

// Test1: Filter with even predicate
func Test1(t *testing.T) {
	nums := []int{1, 2, 3, 4, 5}
	result := Filter(nums, func(n int) bool { return n%2 == 0 })
	if len(result) != 2 || result[0] != 2 || result[1] != 4 {
		t.Errorf("Expected [2, 4], got %v", result)
	}
}

// Test2: Filter with no matches returns empty
func Test2(t *testing.T) {
	nums := []int{1, 3, 5}
	result := Filter(nums, func(n int) bool { return n%2 == 0 })
	if len(result) != 0 {
		t.Errorf("Expected empty, got %v", result)
	}
}

// Test3: FilterEven basic
func Test3(t *testing.T) {
	result := FilterEven([]int{1, 2, 3, 4, 5, 6})
	if len(result) != 3 {
		t.Errorf("Expected 3 even numbers, got %d", len(result))
	}
}

// Test4: FilterPositive basic
func Test4(t *testing.T) {
	result := FilterPositive([]int{-2, -1, 0, 1, 2})
	if len(result) != 2 || result[0] != 1 || result[1] != 2 {
		t.Errorf("Expected [1, 2], got %v", result)
	}
}

// Test5: FilterEven empty input
func Test5(t *testing.T) {
	result := FilterEven([]int{})
	if len(result) != 0 {
		t.Error("Expected empty slice")
	}
}

// Test6: FilterPositive all negative
func Test6(t *testing.T) {
	result := FilterPositive([]int{-5, -3, -1})
	if len(result) != 0 {
		t.Error("Expected empty slice")
	}
}

// Test7: Filter all match
func Test7(t *testing.T) {
	nums := []int{2, 4, 6}
	result := Filter(nums, func(n int) bool { return n%2 == 0 })
	if len(result) != 3 {
		t.Error("Expected all 3 elements")
	}
}

// Test8: FilterEven with zero (0 is even)
func Test8(t *testing.T) {
	result := FilterEven([]int{0, 1, 2})
	if len(result) != 2 || result[0] != 0 {
		t.Errorf("0 should be included as even, got %v", result)
	}
}

// Test9: FilterPositive excludes zero
func Test9(t *testing.T) {
	result := FilterPositive([]int{0, 1})
	if len(result) != 1 || result[0] != 1 {
		t.Errorf("0 should not be positive, got %v", result)
	}
}

// Test10: Filter does not modify original
func Test10(t *testing.T) {
	nums := []int{1, 2, 3}
	Filter(nums, func(n int) bool { return n > 1 })
	if len(nums) != 3 {
		t.Error("Original slice should not be modified")
	}
}
`,
	translations: {
		ru: {
			title: 'Copy-Paste Programming - Продвинутый',
			description: `Рефакторьте дублированную бизнес-логику используя функции высшего порядка.`,
			hint1: `Filter итерирует через numbers, применяет predicate к каждому элементу и добавляет в result если predicate возвращает true. Верните slice result.`,
			hint2: `FilterEven вызывает Filter с func(n int) bool которая возвращает n%2 == 0. FilterPositive вызывает Filter с func(n int) bool которая возвращает n > 0.`,
			whyItMatters: `Функции высшего порядка устраняют дублирование в бизнес-логике сохраняя гибкость.`,
			solutionCode: `package antipatterns

func Filter(numbers []int, predicate func(int) bool) []int {
	result := []int{}

	for _, n := range numbers {
		if predicate(n) {
			result = append(result, n)
		}
	}

	return result
}

func FilterEven(numbers []int) []int {
	return Filter(numbers, func(n int) bool {
		return n%2 == 0
	})
}

func FilterPositive(numbers []int) []int {
	return Filter(numbers, func(n int) bool {
		return n > 0
	})
}`
		},
		uz: {
			title: 'Copy-Paste Programming - Ilg\'or',
			description: `Yuqori tartibli funksiyalardan foydalanib, dublikatlangan biznes mantiqni refaktoring qiling.`,
			hint1: `Filter numbers orqali iteratsiya qiladi, har bir elementga predicate ni qo'llaydi va predicate true qaytarsa result ga qo'shadi. result slice ni qaytaring.`,
			hint2: `FilterEven n%2 == 0 ni qaytaradigan func(n int) bool bilan Filter ni chaqiradi. FilterPositive n > 0 ni qaytaradigan func(n int) bool bilan Filter ni chaqiradi.`,
			whyItMatters: `Yuqori tartibli funksiyalar moslashuvchanlikni saqlab, biznes mantiqidagi dublikatsiyani yo'q qiladi.`,
			solutionCode: `package antipatterns

func Filter(numbers []int, predicate func(int) bool) []int {
	result := []int{}

	for _, n := range numbers {
		if predicate(n) {
			result = append(result, n)
		}
	}

	return result
}

func FilterEven(numbers []int) []int {
	return Filter(numbers, func(n int) bool {
		return n%2 == 0
	})
}

func FilterPositive(numbers []int) []int {
	return Filter(numbers, func(n int) bool {
		return n > 0
	})
}`
		}
	}
};

export default task;
