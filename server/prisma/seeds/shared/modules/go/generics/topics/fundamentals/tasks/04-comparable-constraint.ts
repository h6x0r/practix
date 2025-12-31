import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-generics-comparable-constraint',
    title: 'Comparable Constraint',
    difficulty: 'easy',
    tags: ['go', 'generics', 'comparable', 'constraints'],
    estimatedTime: '15m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn to use the \`comparable\` constraint for types that support equality comparison.

The \`comparable\` constraint is a built-in constraint that allows only types that can be compared using \`==\` and \`!=\` operators. This is essential for operations like finding elements, removing duplicates, or using types as map keys.

**Task:** Implement a generic \`Contains\` function that checks if a slice contains a specific value.

**Requirements:**
- Use type parameter \`T\` with \`comparable\` constraint
- Loop through the slice and compare each element with the target value
- Return \`true\` if the value is found, \`false\` otherwise
- The function should work with any comparable type (int, string, bool, etc.)

**Example Usage:**
\`\`\`go
fmt.Println(Contains([]int{1, 2, 3, 4, 5}, 3))        // Output: true
fmt.Println(Contains([]string{"a", "b", "c"}, "d"))   // Output: false
\`\`\``,
    initialCode: `package generics

// TODO: Implement a generic Contains function that checks if a value exists in a slice
// Use the comparable constraint to allow equality comparison
func Contains[T comparable](slice []T, value T) bool {
    panic("TODO: implement Contains function")
}`,
    solutionCode: `package generics

// Contains checks if a value exists in a slice
func Contains[T comparable](slice []T, value T) bool {
    for _, item := range slice {
        if item == value {
            return true
        }
    }
    return false
}`,
    testCode: `package generics

import (
	"testing"
)

func Test1(t *testing.T) {
	// Test finding integer in slice
	result := Contains([]int{1, 2, 3, 4, 5}, 3)
	if !result {
		t.Errorf("expected true, got false")
	}
}

func Test2(t *testing.T) {
	// Test integer not in slice
	result := Contains([]int{1, 2, 3, 4, 5}, 10)
	if result {
		t.Errorf("expected false, got true")
	}
}

func Test3(t *testing.T) {
	// Test string in slice
	result := Contains([]string{"apple", "banana", "cherry"}, "banana")
	if !result {
		t.Errorf("expected true, got false")
	}
}

func Test4(t *testing.T) {
	// Test string not in slice
	result := Contains([]string{"a", "b", "c"}, "d")
	if result {
		t.Errorf("expected false, got true")
	}
}

func Test5(t *testing.T) {
	// Test empty slice
	result := Contains([]int{}, 5)
	if result {
		t.Errorf("expected false for empty slice, got true")
	}
}

func Test6(t *testing.T) {
	// Test with boolean
	result := Contains([]bool{true, false, true}, false)
	if !result {
		t.Errorf("expected true, got false")
	}
}

func Test7(t *testing.T) {
	// Test first element
	result := Contains([]int{10, 20, 30}, 10)
	if !result {
		t.Errorf("expected true for first element, got false")
	}
}

func Test8(t *testing.T) {
	// Test last element
	result := Contains([]int{10, 20, 30}, 30)
	if !result {
		t.Errorf("expected true for last element, got false")
	}
}

func Test9(t *testing.T) {
	// Test single element slice - found
	result := Contains([]string{"only"}, "only")
	if !result {
		t.Errorf("expected true, got false")
	}
}

func Test10(t *testing.T) {
	// Test single element slice - not found
	result := Contains([]string{"only"}, "other")
	if result {
		t.Errorf("expected false, got true")
	}
}`,
    hint1: `Use a \`for range\` loop to iterate through the slice.`,
    hint2: `Compare each item with the target value using \`==\`. Return \`true\` immediately if a match is found.`,
    whyItMatters: `The comparable constraint is fundamental for search operations, set implementations, and using custom types as map keys. It ensures type safety while enabling equality-based algorithms.

**Production Pattern:**

\`\`\`go
// WITHOUT comparable: Repetitive code for each type
func ContainsInt(slice []int, value int) bool {
    for _, item := range slice {
        if item == value { return true }
    }
    return false
}

func ContainsString(slice []string, value string) bool {
    for _, item := range slice {
        if item == value { return true }
    }
    return false
}

// WITH comparable: One function for all comparable types
func Contains[T comparable](slice []T, value T) bool {
    for _, item := range slice {
        if item == value { return true }
    }
    return false
}

// Usage
Contains([]int{1, 2, 3}, 2)        // true
Contains([]string{"a", "b"}, "c")  // false
Contains([]bool{true, false}, true) // true
\`\`\`

**Practical Benefits:**

1. **Universality**: Works with int, string, bool, custom types
2. **Type safety**: Compiler checks comparability
3. **Less code**: One implementation instead of many
4. **Ready for map keys**: Types with comparable can be map keys`,
    order: 3,
    translations: {
        ru: {
            title: 'Ограничение Comparable',
            solutionCode: `package generics

// Contains проверяет, существует ли значение в срезе
func Contains[T comparable](slice []T, value T) bool {
    for _, item := range slice {
        if item == value {
            return true
        }
    }
    return false
}`,
            description: `Научитесь использовать ограничение \`comparable\` для типов, поддерживающих сравнение на равенство.

Ограничение \`comparable\` - это встроенное ограничение, которое разрешает только типы, которые можно сравнивать с помощью операторов \`==\` и \`!=\`. Это необходимо для операций, таких как поиск элементов, удаление дубликатов или использование типов в качестве ключей карты.

**Задача:** Реализуйте обобщенную функцию \`Contains\`, которая проверяет, содержит ли срез определенное значение.

**Требования:**
- Используйте параметр типа \`T\` с ограничением \`comparable\`
- Пройдитесь по срезу и сравните каждый элемент с целевым значением
- Верните \`true\`, если значение найдено, иначе \`false\`
- Функция должна работать с любым сравнимым типом (int, string, bool и т.д.)

**Пример использования:**
\`\`\`go
fmt.Println(Contains([]int{1, 2, 3, 4, 5}, 3))        // Вывод: true
fmt.Println(Contains([]string{"a", "b", "c"}, "d"))   // Вывод: false
\`\`\``,
            hint1: `Используйте цикл \`for range\` для итерации по срезу.`,
            hint2: `Сравните каждый элемент с целевым значением, используя \`==\`. Немедленно верните \`true\`, если найдено совпадение.`,
            whyItMatters: `Ограничение comparable является фундаментальным для операций поиска, реализации множеств и использования пользовательских типов в качестве ключей карты. Оно обеспечивает безопасность типов, позволяя алгоритмы на основе равенства.

**Продакшен паттерн:**

\`\`\`go
// БЕЗ comparable: Повторяющийся код для каждого типа
func ContainsInt(slice []int, value int) bool {
    for _, item := range slice {
        if item == value { return true }
    }
    return false
}

func ContainsString(slice []string, value string) bool {
    for _, item := range slice {
        if item == value { return true }
    }
    return false
}

// С comparable: Одна функция для всех сравнимых типов
func Contains[T comparable](slice []T, value T) bool {
    for _, item := range slice {
        if item == value { return true }
    }
    return false
}

// Использование
Contains([]int{1, 2, 3}, 2)        // true
Contains([]string{"a", "b"}, "c")  // false
Contains([]bool{true, false}, true) // true
\`\`\`

**Практические преимущества:**

1. **Универсальность**: Работает с int, string, bool, custom types
2. **Типобезопасность**: Компилятор проверяет возможность сравнения
3. **Меньше кода**: Одна реализация вместо множества
4. **Готовность для map keys**: Типы с comparable могут быть ключами map`
        },
        uz: {
            title: 'Comparable cheklovi',
            solutionCode: `package generics

// Contains qiymat srezda mavjudligini tekshiradi
func Contains[T comparable](slice []T, value T) bool {
    for _, item := range slice {
        if item == value {
            return true
        }
    }
    return false
}`,
            description: `Tenglik taqqoslashni qo'llab-quvvatlaydigan tiplar uchun \`comparable\` cheklovidan foydalanishni o'rganing.

\`comparable\` cheklovi - bu faqat \`==\` va \`!=\` operatorlari yordamida taqqoslash mumkin bo'lgan tiplarni qabul qiladigan o'rnatilgan cheklov. Bu elementlarni topish, dublikatlarni o'chirish yoki tiplarni map kalitlari sifatida ishlatish kabi operatsiyalar uchun zarur.

**Vazifa:** Srez ma'lum bir qiymatni o'z ichiga olgan yoki yo'qligini tekshiradigan umumiy \`Contains\` funksiyasini yarating.

**Talablar:**
- \`T\` tip parametrini \`comparable\` cheklovi bilan ishlating
- Srez bo'ylab aylanib har bir elementni maqsad qiymat bilan taqqoslang
- Qiymat topilsa \`true\`, aks holda \`false\` qaytaring
- Funksiya har qanday taqqoslanadigan tip bilan ishlashi kerak (int, string, bool va boshqalar)

**Foydalanish misoli:**
\`\`\`go
fmt.Println(Contains([]int{1, 2, 3, 4, 5}, 3))        // Natija: true
fmt.Println(Contains([]string{"a", "b", "c"}, "d"))   // Natija: false
\`\`\``,
            hint1: `Srez bo'ylab takrorlash uchun \`for range\` tsiklidan foydalaning.`,
            hint2: `Har bir elementni \`==\` yordamida maqsad qiymat bilan taqqoslang. Mos kelish topilsa darhol \`true\` qaytaring.`,
            whyItMatters: `Comparable cheklovi qidiruv operatsiyalari, to'plam implementatsiyalari va maxsus tiplarni map kalitlari sifatida ishlatish uchun asosiy hisoblanadi. U tenglikka asoslangan algoritmlarni yoqish bilan birga tip xavfsizligini ta'minlaydi.

**Ishlab chiqarish patterni:**

\`\`\`go
// Comparablesiz: Har bir tip uchun takrorlanuvchi kod
func ContainsInt(slice []int, value int) bool {
    for _, item := range slice {
        if item == value { return true }
    }
    return false
}

func ContainsString(slice []string, value string) bool {
    for _, item := range slice {
        if item == value { return true }
    }
    return false
}

// Comparable bilan: Barcha taqqoslanadigan tiplar uchun bitta funksiya
func Contains[T comparable](slice []T, value T) bool {
    for _, item := range slice {
        if item == value { return true }
    }
    return false
}

// Foydalanish
Contains([]int{1, 2, 3}, 2)        // true
Contains([]string{"a", "b"}, "c")  // false
Contains([]bool{true, false}, true) // true
\`\`\`

**Amaliy foydalari:**

1. **Universallik**: int, string, bool, maxsus turlar bilan ishlaydi
2. **Tip xavfsizligi**: Kompilyator taqqoslash imkoniyatini tekshiradi
3. **Kamroq kod**: Ko'p implementatsiyalar o'rniga bitta
4. **Map kalitlari uchun tayyor**: Comparable tiplar map kalitlari bo'lishi mumkin`
        }
    }
};

export default task;
