import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-generics-generic-slice-operations',
    title: 'Generic Slice Operations',
    difficulty: 'medium',
    tags: ['go', 'generics', 'slices', 'functional-programming', 'map-filter'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Build functional programming utilities for slices using generics.

Generic slice operations allow you to write reusable transformation functions that work with any type. This is a powerful pattern borrowed from functional programming languages.

**Task:** Implement three generic slice utility functions:
1. \`Filter[T]\` - filters elements based on a predicate function
2. \`Map[T, R]\` - transforms elements from type T to type R
3. \`Reduce[T, R]\` - reduces a slice to a single value

**Requirements:**
- \`Filter[T any](slice []T, predicate func(T) bool) []T\`
- \`Map[T any, R any](slice []T, transform func(T) R) []R\`
- \`Reduce[T any, R any](slice []T, initial R, accumulator func(R, T) R) R\`
- All functions should work with any type

**Example Usage:**
\`\`\`go
numbers := []int{1, 2, 3, 4, 5, 6}

// Filter even numbers
evens := Filter(numbers, func(n int) bool { return n%2 == 0 })
// Result: [2, 4, 6]

// Map to squares
squares := Map(numbers, func(n int) int { return n * n })
// Result: [1, 4, 9, 16, 25, 36]

// Reduce to sum
sum := Reduce(numbers, 0, func(acc, n int) int { return acc + n })
// Result: 21
\`\`\``,
    initialCode: `package generics

// TODO: Implement Filter function that returns elements matching the predicate
func Filter[T any](slice []T, predicate func(T) bool) []T {
    panic("TODO: implement Filter")
}

// TODO: Implement Map function that transforms elements from type T to type R
func Map[T any, R any](slice []T, transform func(T) R) []R {
    panic("TODO: implement Map")
}

// TODO: Implement Reduce function that accumulates elements into a single value
func Reduce[T any, R any](slice []T, initial R, accumulator func(R, T) R) R {
    panic("TODO: implement Reduce")
}`,
    solutionCode: `package generics

// Filter returns a new slice containing only elements that match the predicate
func Filter[T any](slice []T, predicate func(T) bool) []T {
    result := make([]T, 0)
    for _, item := range slice {
        if predicate(item) {
            result = append(result, item)
        }
    }
    return result
}

// Map transforms each element in the slice using the transform function
func Map[T any, R any](slice []T, transform func(T) R) []R {
    result := make([]R, len(slice))
    for i, item := range slice {
        result[i] = transform(item)
    }
    return result
}

// Reduce accumulates all elements into a single value using the accumulator function
func Reduce[T any, R any](slice []T, initial R, accumulator func(R, T) R) R {
    result := initial
    for _, item := range slice {
        result = accumulator(result, item)
    }
    return result
}`,
    testCode: `package generics

import (
	"testing"
)

func Test1(t *testing.T) {
	// Test Filter with even numbers
	numbers := []int{1, 2, 3, 4, 5, 6}
	result := Filter(numbers, func(n int) bool { return n%2 == 0 })
	expected := []int{2, 4, 6}
	if len(result) != len(expected) {
		t.Errorf("expected length %d, got %d", len(expected), len(result))
	}
	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("at index %d: expected %v, got %v", i, expected[i], result[i])
		}
	}
}

func Test2(t *testing.T) {
	// Test Filter with no matches
	numbers := []int{1, 3, 5}
	result := Filter(numbers, func(n int) bool { return n%2 == 0 })
	if len(result) != 0 {
		t.Errorf("expected empty slice, got %v", result)
	}
}

func Test3(t *testing.T) {
	// Test Map to squares
	numbers := []int{1, 2, 3, 4}
	result := Map(numbers, func(n int) int { return n * n })
	expected := []int{1, 4, 9, 16}
	if len(result) != len(expected) {
		t.Errorf("expected length %d, got %d", len(expected), len(result))
	}
	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("at index %d: expected %v, got %v", i, expected[i], result[i])
		}
	}
}

func Test4(t *testing.T) {
	// Test Map with type transformation
	numbers := []int{1, 2, 3}
	result := Map(numbers, func(n int) string {
		if n == 1 { return "one" }
		if n == 2 { return "two" }
		return "three"
	})
	expected := []string{"one", "two", "three"}
	if len(result) != len(expected) {
		t.Errorf("expected length %d, got %d", len(expected), len(result))
	}
	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("at index %d: expected %v, got %v", i, expected[i], result[i])
		}
	}
}

func Test5(t *testing.T) {
	// Test Reduce to sum
	numbers := []int{1, 2, 3, 4, 5}
	result := Reduce(numbers, 0, func(acc, n int) int { return acc + n })
	expected := 15
	if result != expected {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test6(t *testing.T) {
	// Test Reduce to product
	numbers := []int{2, 3, 4}
	result := Reduce(numbers, 1, func(acc, n int) int { return acc * n })
	expected := 24
	if result != expected {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test7(t *testing.T) {
	// Test Filter with strings
	words := []string{"apple", "banana", "apricot", "cherry"}
	result := Filter(words, func(s string) bool { return s[0] == 'a' })
	expected := []string{"apple", "apricot"}
	if len(result) != len(expected) {
		t.Errorf("expected length %d, got %d", len(expected), len(result))
	}
	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("at index %d: expected %v, got %v", i, expected[i], result[i])
		}
	}
}

func Test8(t *testing.T) {
	// Test Map on empty slice
	numbers := []int{}
	result := Map(numbers, func(n int) int { return n * 2 })
	if len(result) != 0 {
		t.Errorf("expected empty slice, got %v", result)
	}
}

func Test9(t *testing.T) {
	// Test Reduce on empty slice
	numbers := []int{}
	result := Reduce(numbers, 10, func(acc, n int) int { return acc + n })
	expected := 10
	if result != expected {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test10(t *testing.T) {
	// Test Reduce concatenating strings
	words := []string{"Hello", " ", "World"}
	result := Reduce(words, "", func(acc, s string) string { return acc + s })
	expected := "Hello World"
	if result != expected {
		t.Errorf("expected %v, got %v", expected, result)
	}
}`,
    hint1: `For Filter, create a new empty slice and append only items that match the predicate.`,
    hint2: `For Map, pre-allocate the result slice with the same length. For Reduce, start with the initial value and iterate through the slice, updating the accumulator.`,
    whyItMatters: `Generic slice operations enable functional programming patterns in Go. They eliminate repetitive looping code and make data transformations more declarative and composable. These utilities are essential for modern Go development.

**Production Pattern:**

\`\`\`go
numbers := []int{1, 2, 3, 4, 5, 6}

// Filter: Keep only even numbers
evens := Filter(numbers, func(n int) bool {
    return n%2 == 0
})
// [2, 4, 6]

// Map: Transform to squares
squares := Map(numbers, func(n int) int {
    return n * n
})
// [1, 4, 9, 16, 25, 36]

// Reduce: Sum all elements
sum := Reduce(numbers, 0, func(acc, n int) int {
    return acc + n
})
// 21
\`\`\`

**Practical Benefits:**

1. **Declarative**: Say WHAT to do, not HOW
2. **Composability**: Operations can be chained together
3. **Type safety**: Works with any types
4. **Fewer errors**: No manual loops and indices`,
    order: 5,
    translations: {
        ru: {
            title: 'Обобщённые операции со слайсами',
            solutionCode: `package generics

// Filter возвращает новый срез, содержащий только элементы, соответствующие предикату
func Filter[T any](slice []T, predicate func(T) bool) []T {
    result := make([]T, 0)
    for _, item := range slice {
        if predicate(item) {
            result = append(result, item)
        }
    }
    return result
}

// Map преобразует каждый элемент среза с помощью функции преобразования
func Map[T any, R any](slice []T, transform func(T) R) []R {
    result := make([]R, len(slice))
    for i, item := range slice {
        result[i] = transform(item)
    }
    return result
}

// Reduce накапливает все элементы в одно значение с помощью функции-аккумулятора
func Reduce[T any, R any](slice []T, initial R, accumulator func(R, T) R) R {
    result := initial
    for _, item := range slice {
        result = accumulator(result, item)
    }
    return result
}`,
            description: `Создайте утилиты функционального программирования для срезов с использованием обобщений.

Обобщенные операции со срезами позволяют писать переиспользуемые функции преобразования, которые работают с любым типом. Это мощный паттерн, заимствованный из функциональных языков программирования.

**Задача:** Реализуйте три обобщенные утилитарные функции для срезов:
1. \`Filter[T]\` - фильтрует элементы на основе функции-предиката
2. \`Map[T, R]\` - преобразует элементы из типа T в тип R
3. \`Reduce[T, R]\` - сводит срез к одному значению

**Требования:**
- \`Filter[T any](slice []T, predicate func(T) bool) []T\`
- \`Map[T any, R any](slice []T, transform func(T) R) []R\`
- \`Reduce[T any, R any](slice []T, initial R, accumulator func(R, T) R) R\`
- Все функции должны работать с любым типом

**Пример использования:**
\`\`\`go
numbers := []int{1, 2, 3, 4, 5, 6}

// Фильтровать четные числа
evens := Filter(numbers, func(n int) bool { return n%2 == 0 })
// Результат: [2, 4, 6]

// Преобразовать в квадраты
squares := Map(numbers, func(n int) int { return n * n })
// Результат: [1, 4, 9, 16, 25, 36]

// Свести к сумме
sum := Reduce(numbers, 0, func(acc, n int) int { return acc + n })
// Результат: 21
\`\`\``,
            hint1: `Для Filter создайте новый пустой срез и добавляйте только элементы, соответствующие предикату.`,
            hint2: `Для Map предварительно выделите срез результата той же длины. Для Reduce начните с начального значения и пройдите по срезу, обновляя аккумулятор.`,
            whyItMatters: `Обобщенные операции со срезами позволяют использовать паттерны функционального программирования в Go. Они устраняют повторяющийся код циклов и делают преобразования данных более декларативными и композируемыми. Эти утилиты необходимы для современной разработки на Go.

**Продакшен паттерн:**

\`\`\`go
numbers := []int{1, 2, 3, 4, 5, 6}

// Filter: Оставить только чётные
evens := Filter(numbers, func(n int) bool {
    return n%2 == 0
})
// [2, 4, 6]

// Map: Преобразовать в квадраты
squares := Map(numbers, func(n int) int {
    return n * n
})
// [1, 4, 9, 16, 25, 36]

// Reduce: Суммировать
sum := Reduce(numbers, 0, func(acc, n int) int {
    return acc + n
})
// 21
\`\`\`

**Практические преимущества:**

1. **Декларативность**: Говорим ЧТО делать, а не КАК
2. **Композируемость**: Можно соединять операции в цепочки
3. **Типобезопасность**: Работает с любыми типами
4. **Меньше ошибок**: Нет ручных циклов и индексов`
        },
        uz: {
            title: 'Generic slayz operatsiyalari',
            solutionCode: `package generics

// Filter predikatga mos keladigan elementlarni o'z ichiga olgan yangi srez qaytaradi
func Filter[T any](slice []T, predicate func(T) bool) []T {
    result := make([]T, 0)
    for _, item := range slice {
        if predicate(item) {
            result = append(result, item)
        }
    }
    return result
}

// Map srezdagi har bir elementni transformatsiya funksiyasi yordamida o'zgartiradi
func Map[T any, R any](slice []T, transform func(T) R) []R {
    result := make([]R, len(slice))
    for i, item := range slice {
        result[i] = transform(item)
    }
    return result
}

// Reduce akkumulyator funksiyasi yordamida barcha elementlarni bitta qiymatga to'playdi
func Reduce[T any, R any](slice []T, initial R, accumulator func(R, T) R) R {
    result := initial
    for _, item := range slice {
        result = accumulator(result, item)
    }
    return result
}`,
            description: `Umumiy tiplardan foydalanib srezlar uchun funksional dasturlash utilitalarini yarating.

Umumiy srez operatsiyalari har qanday tip bilan ishlaydigan qayta foydalaniladigan transformatsiya funksiyalarini yozish imkonini beradi. Bu funksional dasturlash tillaridan olingan kuchli naqsh.

**Vazifa:** Uchta umumiy srez utility funksiyalarini amalga oshiring:
1. \`Filter[T]\` - predikat funksiyasiga asoslanib elementlarni filtrlaydi
2. \`Map[T, R]\` - elementlarni T tipidan R tipiga o'zgartiradi
3. \`Reduce[T, R]\` - srezni bitta qiymatga kamaytiradi

**Talablar:**
- \`Filter[T any](slice []T, predicate func(T) bool) []T\`
- \`Map[T any, R any](slice []T, transform func(T) R) []R\`
- \`Reduce[T any, R any](slice []T, initial R, accumulator func(R, T) R) R\`
- Barcha funksiyalar har qanday tip bilan ishlashi kerak

**Foydalanish misoli:**
\`\`\`go
numbers := []int{1, 2, 3, 4, 5, 6}

// Juft sonlarni filtrlash
evens := Filter(numbers, func(n int) bool { return n%2 == 0 })
// Natija: [2, 4, 6]

// Kvadratlarga o'zgartirish
squares := Map(numbers, func(n int) int { return n * n })
// Natija: [1, 4, 9, 16, 25, 36]

// Yig'indiga kamaytirish
sum := Reduce(numbers, 0, func(acc, n int) int { return acc + n })
// Natija: 21
\`\`\``,
            hint1: `Filter uchun yangi bo'sh srez yarating va faqat predikatga mos keladigan elementlarni qo'shing.`,
            hint2: `Map uchun natija srezini bir xil uzunlikda oldindan ajrating. Reduce uchun boshlang'ich qiymatdan boshlang va srez bo'ylab aylanib, akkumulyatorni yangilang.`,
            whyItMatters: `Umumiy srez operatsiyalari Go-da funksional dasturlash naqshlarini yoqadi. Ular takrorlanuvchi tsikl kodini yo'q qiladi va ma'lumotlar transformatsiyalarini yanada deklarativ va kompozitsion qiladi. Bu utilitalar zamonaviy Go ishlab chiqish uchun zarurdir.

**Ishlab chiqarish patterni:**

\`\`\`go
numbers := []int{1, 2, 3, 4, 5, 6}

// Filter: Faqat juftlarni qoldirish
evens := Filter(numbers, func(n int) bool {
    return n%2 == 0
})
// [2, 4, 6]

// Map: Kvadratlarga o'zgartirish
squares := Map(numbers, func(n int) int {
    return n * n
})
// [1, 4, 9, 16, 25, 36]

// Reduce: Yig'indilash
sum := Reduce(numbers, 0, func(acc, n int) int {
    return acc + n
})
// 21
\`\`\`

**Amaliy foydalari:**

1. **Deklarativlik**: NIMA qilishni aytamiz, QANDAY emas
2. **Kompozitsiya**: Operatsiyalarni zanjirga ulash mumkin
3. **Tip xavfsizligi**: Har qanday tiplar bilan ishlaydi
4. **Kamroq xatolar**: Qo'lda tsikllar va indekslar yo'q`
        }
    }
};

export default task;
