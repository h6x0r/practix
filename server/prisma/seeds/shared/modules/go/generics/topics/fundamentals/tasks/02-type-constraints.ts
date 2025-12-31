import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-generics-type-constraints',
    title: 'Type Constraints',
    difficulty: 'easy',
    tags: ['go', 'generics', 'constraints', 'type-parameters'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master type constraints to control which types can be used with your generic functions.

Type constraints limit what types can be passed as type arguments. Go provides built-in constraints like \`any\`, \`comparable\`, and the \`constraints\` package offers more specialized ones like \`Ordered\`.

**Task:** Implement a generic \`Sum\` function that calculates the sum of all elements in a slice. The function should work with any numeric type (int, float64, etc.).

**Requirements:**
- Use type parameter \`T\` with a constraint that allows addition
- Define a custom constraint interface that includes all numeric types
- Iterate through the slice and accumulate the sum
- Return the total sum

**Example Usage:**
\`\`\`go
fmt.Println(Sum([]int{1, 2, 3, 4, 5}))           // Output: 15
fmt.Println(Sum([]float64{1.5, 2.5, 3.0}))       // Output: 7.0
\`\`\`

**Hint:** You can define a constraint like:
\`\`\`go
type Number interface {
    int | int64 | float64 | float32
}
\`\`\``,
    initialCode: `package generics

// TODO: Define a Number constraint that includes numeric types
type Number interface {
    // Add numeric types here using | operator
}

// TODO: Implement a generic Sum function that adds all elements in a slice
func Sum[T Number](numbers []T) T {
    panic("TODO: implement Sum function")
}`,
    solutionCode: `package generics

// Number constraint allows any numeric type
type Number interface {
    int | int8 | int16 | int32 | int64 |
    uint | uint8 | uint16 | uint32 | uint64 |
    float32 | float64
}

// Sum returns the sum of all elements in a slice
func Sum[T Number](numbers []T) T {
    var sum T
    for _, num := range numbers {
        sum += num
    }
    return sum
}`,
    testCode: `package generics

import (
	"testing"
)

func Test1(t *testing.T) {
	// Test sum of positive integers
	result := Sum([]int{1, 2, 3, 4, 5})
	expected := 15
	if result != expected {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test2(t *testing.T) {
	// Test sum of floats
	result := Sum([]float64{1.5, 2.5, 3.0})
	expected := 7.0
	if result != expected {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test3(t *testing.T) {
	// Test empty slice
	result := Sum([]int{})
	expected := 0
	if result != expected {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test4(t *testing.T) {
	// Test single element
	result := Sum([]int{42})
	expected := 42
	if result != expected {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test5(t *testing.T) {
	// Test negative integers
	result := Sum([]int{-1, -2, -3})
	expected := -6
	if result != expected {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test6(t *testing.T) {
	// Test mixed positive and negative
	result := Sum([]int{10, -5, 3, -2})
	expected := 6
	if result != expected {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test7(t *testing.T) {
	// Test with float32
	result := Sum([]float32{1.1, 2.2, 3.3})
	expected := float32(6.6)
	if result < expected-0.01 || result > expected+0.01 {
		t.Errorf("expected approximately %v, got %v", expected, result)
	}
}

func Test8(t *testing.T) {
	// Test with zeros
	result := Sum([]int{0, 0, 0, 0})
	expected := 0
	if result != expected {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test9(t *testing.T) {
	// Test large numbers
	result := Sum([]int{1000000, 2000000, 3000000})
	expected := 6000000
	if result != expected {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test10(t *testing.T) {
	// Test with uint
	result := Sum([]uint{1, 2, 3, 4})
	expected := uint(10)
	if result != expected {
		t.Errorf("expected %v, got %v", expected, result)
	}
}`,
    hint1: `Define a Number constraint interface using the \`|\` operator to list all numeric types.`,
    hint2: `Initialize a zero-value variable of type \`T\`, then loop through the slice adding each element to it.`,
    whyItMatters: `Type constraints ensure type safety while maintaining flexibility. They allow you to write generic code that only works with types that support the required operations, catching errors at compile time.

**Production Pattern:**

\`\`\`go
// WITHOUT constraints: Unsafe interface{} usage
func SumUnsafe(numbers []interface{}) float64 {
    var sum float64
    for _, num := range numbers {
        // Type assertion can panic at runtime!
        switch v := num.(type) {
        case int:
            sum += float64(v)
        case float64:
            sum += v
        default:
            panic("unsupported type")
        }
    }
    return sum
}

// WITH constraints: Type-safe generic
type Number interface {
    int | int64 | float32 | float64
}

func Sum[T Number](numbers []T) T {
    var sum T
    for _, num := range numbers {
        sum += num  // Compiler checks + support
    }
    return sum
}
\`\`\`

**Practical Benefits:**

1. **Compile-time checking**: Type errors are caught before running the program
2. **Performance**: No overhead from type assertions or reflection
3. **Clean code**: No need for switch statements to handle different types
4. **Safety**: Cannot pass incompatible types`,
    order: 1,
    translations: {
        ru: {
            title: 'Ограничения типов',
            solutionCode: `package generics

// Number ограничение позволяет любой числовой тип
type Number interface {
    int | int8 | int16 | int32 | int64 |
    uint | uint8 | uint16 | uint32 | uint64 |
    float32 | float64
}

// Sum возвращает сумму всех элементов в срезе
func Sum[T Number](numbers []T) T {
    var sum T
    for _, num := range numbers {
        sum += num
    }
    return sum
}`,
            description: `Освойте ограничения типов для контроля того, какие типы могут использоваться с вашими обобщенными функциями.

Ограничения типов ограничивают, какие типы могут быть переданы в качестве аргументов типа. Go предоставляет встроенные ограничения, такие как \`any\`, \`comparable\`, а пакет \`constraints\` предлагает более специализированные, такие как \`Ordered\`.

**Задача:** Реализуйте обобщенную функцию \`Sum\`, которая вычисляет сумму всех элементов в срезе. Функция должна работать с любым числовым типом (int, float64 и т.д.).

**Требования:**
- Используйте параметр типа \`T\` с ограничением, которое позволяет сложение
- Определите пользовательский интерфейс ограничения, включающий все числовые типы
- Пройдитесь по срезу и накопите сумму
- Верните общую сумму

**Пример использования:**
\`\`\`go
fmt.Println(Sum([]int{1, 2, 3, 4, 5}))           // Вывод: 15
fmt.Println(Sum([]float64{1.5, 2.5, 3.0}))       // Вывод: 7.0
\`\`\`

**Подсказка:** Вы можете определить ограничение так:
\`\`\`go
type Number interface {
    int | int64 | float64 | float32
}
\`\`\``,
            hint1: `Определите интерфейс ограничения Number, используя оператор \`|\` для перечисления всех числовых типов.`,
            hint2: `Инициализируйте переменную с нулевым значением типа \`T\`, затем пройдитесь по срезу, добавляя каждый элемент к ней.`,
            whyItMatters: `Ограничения типов обеспечивают безопасность типов при сохранении гибкости. Они позволяют писать обобщенный код, который работает только с типами, поддерживающими требуемые операции, обнаруживая ошибки на этапе компиляции.

**Продакшен паттерн:**

\`\`\`go
// БЕЗ constraints: Небезопасное использование interface{}
func SumUnsafe(numbers []interface{}) float64 {
    var sum float64
    for _, num := range numbers {
        // Type assertion может panic во время выполнения!
        switch v := num.(type) {
        case int:
            sum += float64(v)
        case float64:
            sum += v
        default:
            panic("unsupported type")
        }
    }
    return sum
}

// С constraints: Типобезопасный generic
type Number interface {
    int | int64 | float32 | float64
}

func Sum[T Number](numbers []T) T {
    var sum T
    for _, num := range numbers {
        sum += num  // Компилятор проверяет поддержку +
    }
    return sum
}
\`\`\`

**Практические преимущества:**

1. **Компиляционная проверка**: Ошибки типов обнаруживаются до запуска программы
2. **Производительность**: Нет overhead от type assertions или reflection
3. **Чистый код**: Не нужны switch statements для обработки разных типов
4. **Безопасность**: Невозможно передать несовместимый тип`
        },
        uz: {
            title: 'Tip cheklovlari',
            solutionCode: `package generics

// Number cheklovi har qanday raqamli tipga ruxsat beradi
type Number interface {
    int | int8 | int16 | int32 | int64 |
    uint | uint8 | uint16 | uint32 | uint64 |
    float32 | float64
}

// Sum srezadagi barcha elementlar yig'indisini qaytaradi
func Sum[T Number](numbers []T) T {
    var sum T
    for _, num := range numbers {
        sum += num
    }
    return sum
}`,
            description: `Umumiy funksiyalaringizda qaysi tiplardan foydalanish mumkinligini nazorat qilish uchun tip cheklovlarini o'zlashtirasiz.

Tip cheklovlari qaysi tiplarni tip argumentlari sifatida uzatish mumkinligini cheklaydi. Go \`any\`, \`comparable\` kabi o'rnatilgan cheklovlarni taqdim etadi va \`constraints\` paketi \`Ordered\` kabi maxsus cheklovlarni taklif qiladi.

**Vazifa:** Srezdagi barcha elementlar yig'indisini hisoblaydigan umumiy \`Sum\` funksiyasini yarating. Funksiya har qanday raqamli tip (int, float64 va boshqalar) bilan ishlashi kerak.

**Talablar:**
- Qo'shishga ruxsat beradigan cheklov bilan \`T\` tip parametridan foydalaning
- Barcha raqamli tiplarni o'z ichiga olgan maxsus cheklov interfeysini aniqlang
- Srez bo'ylab takrorlang va yig'indini to'plang
- Umumiy yig'indini qaytaring

**Foydalanish misoli:**
\`\`\`go
fmt.Println(Sum([]int{1, 2, 3, 4, 5}))           // Natija: 15
fmt.Println(Sum([]float64{1.5, 2.5, 3.0}))       // Natija: 7.0
\`\`\`

**Maslahat:** Cheklovni quyidagicha aniqlashingiz mumkin:
\`\`\`go
type Number interface {
    int | int64 | float64 | float32
}
\`\`\``,
            hint1: `Barcha raqamli tiplarni sanab o'tish uchun \`|\` operatoridan foydalanib Number cheklov interfeysini aniqlang.`,
            hint2: `\`T\` tipidagi nol-qiymatli o'zgaruvchini ishga tushiring, keyin srez bo'ylab aylanib har bir elementni qo'shing.`,
            whyItMatters: `Tip cheklovlari moslashuvchanlikni saqlab qolgan holda tip xavfsizligini ta'minlaydi. Ular faqat kerakli operatsiyalarni qo'llab-quvvatlaydigan tiplar bilan ishlaydigan umumiy kod yozish imkonini beradi va xatolarni kompilyatsiya vaqtida aniqlaydi.

**Ishlab chiqarish patterni:**

\`\`\`go
// Cheklovlarsiz: Xavfsiz bo'lmagan interface{} dan foydalanish
func SumUnsafe(numbers []interface{}) float64 {
    var sum float64
    for _, num := range numbers {
        // Type assertion runtime paytida panic qilishi mumkin!
        switch v := num.(type) {
        case int:
            sum += float64(v)
        case float64:
            sum += v
        default:
            panic("unsupported type")
        }
    }
    return sum
}

// Cheklovlar bilan: Tip-xavfsiz generic
type Number interface {
    int | int64 | float32 | float64
}

func Sum[T Number](numbers []T) T {
    var sum T
    for _, num := range numbers {
        sum += num  // Kompilyator + qo'llab-quvvatlashni tekshiradi
    }
    return sum
}
\`\`\`

**Amaliy foydalari:**

1. **Kompilyatsiya tekshiruvi**: Tip xatolari dastur ishga tushishidan oldin aniqlanadi
2. **Ishlash samaradorligi**: Type assertions yoki reflection dan overhead yo'q
3. **Toza kod**: Turli tiplarni qayta ishlash uchun switch statementlar kerak emas
4. **Xavfsizlik**: Mos kelmaydigan tipni uzatish mumkin emas`
        }
    }
};

export default task;
