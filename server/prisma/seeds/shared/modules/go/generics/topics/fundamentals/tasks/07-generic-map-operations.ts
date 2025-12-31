import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-generics-generic-map-operations',
    title: 'Generic Map Operations',
    difficulty: 'medium',
    tags: ['go', 'generics', 'maps', 'collections', 'utilities'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Create reusable utility functions for working with maps using generics.

Maps are fundamental data structures in Go, but working with them often requires repetitive code. Generic map operations provide type-safe utilities for common map tasks.

**Task:** Implement three generic map utility functions:
1. \`Keys[K, V]\` - extracts all keys from a map
2. \`Values[K, V]\` - extracts all values from a map
3. \`Merge[K, V]\` - merges two maps (second map overwrites first on key conflicts)

**Requirements:**
- \`Keys[K comparable, V any](m map[K]V) []K\` - returns slice of all keys
- \`Values[K comparable, V any](m map[K]V) []V\` - returns slice of all values
- \`Merge[K comparable, V any](m1, m2 map[K]V) map[K]V\` - returns new merged map
- Keys must be comparable (required for map keys in Go)

**Example Usage:**
\`\`\`go
ages := map[string]int{"Alice": 30, "Bob": 25, "Charlie": 35}

keys := Keys(ages)
// Result: ["Alice", "Bob", "Charlie"] (order may vary)

values := Values(ages)
// Result: [30, 25, 35] (order may vary)

m1 := map[string]int{"a": 1, "b": 2}
m2 := map[string]int{"b": 3, "c": 4}
merged := Merge(m1, m2)
// Result: map[a:1 b:3 c:4]
\`\`\``,
    initialCode: `package generics

// TODO: Implement Keys function that returns all keys from a map
func Keys[K comparable, V any](m map[K]V) []K {
    panic("TODO: implement Keys")
}

// TODO: Implement Values function that returns all values from a map
func Values[K comparable, V any](m map[K]V) []V {
    panic("TODO: implement Values")
}

// TODO: Implement Merge function that combines two maps
// If a key exists in both maps, the value from m2 should be used
func Merge[K comparable, V any](m1, m2 map[K]V) map[K]V {
    panic("TODO: implement Merge")
}`,
    solutionCode: `package generics

// Keys returns a slice of all keys in the map
func Keys[K comparable, V any](m map[K]V) []K {
    keys := make([]K, 0, len(m))
    for k := range m {
        keys = append(keys, k)
    }
    return keys
}

// Values returns a slice of all values in the map
func Values[K comparable, V any](m map[K]V) []V {
    values := make([]V, 0, len(m))
    for _, v := range m {
        values = append(values, v)
    }
    return values
}

// Merge combines two maps into a new map
// If a key exists in both maps, the value from m2 takes precedence
func Merge[K comparable, V any](m1, m2 map[K]V) map[K]V {
    result := make(map[K]V, len(m1)+len(m2))

    // Copy all entries from m1
    for k, v := range m1 {
        result[k] = v
    }

    // Copy all entries from m2 (overwrites m1 values if keys match)
    for k, v := range m2 {
        result[k] = v
    }

    return result
}`,
    testCode: `package generics

import (
	"testing"
)

func Test1(t *testing.T) {
	// Test Keys with string keys
	m := map[string]int{"a": 1, "b": 2, "c": 3}
	result := Keys(m)
	if len(result) != 3 {
		t.Errorf("expected 3 keys, got %d", len(result))
	}
	// Check all keys exist
	keyMap := make(map[string]bool)
	for _, k := range result {
		keyMap[k] = true
	}
	if !keyMap["a"] || !keyMap["b"] || !keyMap["c"] {
		t.Errorf("missing expected keys in result")
	}
}

func Test2(t *testing.T) {
	// Test Keys with empty map
	m := map[string]int{}
	result := Keys(m)
	if len(result) != 0 {
		t.Errorf("expected empty slice, got %v", result)
	}
}

func Test3(t *testing.T) {
	// Test Values with integers
	m := map[string]int{"a": 1, "b": 2, "c": 3}
	result := Values(m)
	if len(result) != 3 {
		t.Errorf("expected 3 values, got %d", len(result))
	}
	// Check all values exist
	sum := 0
	for _, v := range result {
		sum += v
	}
	if sum != 6 {
		t.Errorf("expected sum of 6, got %d", sum)
	}
}

func Test4(t *testing.T) {
	// Test Values with empty map
	m := map[int]string{}
	result := Values(m)
	if len(result) != 0 {
		t.Errorf("expected empty slice, got %v", result)
	}
}

func Test5(t *testing.T) {
	// Test Merge with non-overlapping keys
	m1 := map[string]int{"a": 1, "b": 2}
	m2 := map[string]int{"c": 3, "d": 4}
	result := Merge(m1, m2)
	if len(result) != 4 {
		t.Errorf("expected 4 entries, got %d", len(result))
	}
	if result["a"] != 1 || result["b"] != 2 || result["c"] != 3 || result["d"] != 4 {
		t.Errorf("unexpected values in merged map")
	}
}

func Test6(t *testing.T) {
	// Test Merge with overlapping keys (m2 overwrites m1)
	m1 := map[string]int{"a": 1, "b": 2}
	m2 := map[string]int{"b": 3, "c": 4}
	result := Merge(m1, m2)
	if len(result) != 3 {
		t.Errorf("expected 3 entries, got %d", len(result))
	}
	if result["a"] != 1 || result["b"] != 3 || result["c"] != 4 {
		t.Errorf("expected map[a:1 b:3 c:4], got %v", result)
	}
}

func Test7(t *testing.T) {
	// Test Merge with empty first map
	m1 := map[string]int{}
	m2 := map[string]int{"a": 1, "b": 2}
	result := Merge(m1, m2)
	if len(result) != 2 {
		t.Errorf("expected 2 entries, got %d", len(result))
	}
}

func Test8(t *testing.T) {
	// Test Merge with empty second map
	m1 := map[string]int{"a": 1, "b": 2}
	m2 := map[string]int{}
	result := Merge(m1, m2)
	if len(result) != 2 {
		t.Errorf("expected 2 entries, got %d", len(result))
	}
}

func Test9(t *testing.T) {
	// Test Keys with integer keys
	m := map[int]string{1: "one", 2: "two", 3: "three"}
	result := Keys(m)
	if len(result) != 3 {
		t.Errorf("expected 3 keys, got %d", len(result))
	}
}

func Test10(t *testing.T) {
	// Test Merge doesn't modify original maps
	m1 := map[string]int{"a": 1}
	m2 := map[string]int{"b": 2}
	result := Merge(m1, m2)
	result["c"] = 3
	if len(m1) != 1 || len(m2) != 1 {
		t.Errorf("original maps were modified")
	}
}`,
    hint1: `For Keys and Values, pre-allocate the slice with capacity \`len(m)\` for better performance.`,
    hint2: `For Merge, create a new map and copy entries from both input maps. Entries from m2 will naturally overwrite m1 entries with the same key.`,
    whyItMatters: `Generic map utilities reduce boilerplate code and improve code reusability across your codebase. These operations are commonly needed in data processing, API responses, and configuration management.

**Production Pattern:**

\`\`\`go
ages := map[string]int{
    "Alice": 30,
    "Bob": 25
}

// Extract all keys
keys := Keys(ages)  // ["Alice", "Bob"]

// Extract all values
values := Values(ages)  // [30, 25]

// Merge maps
defaults := map[string]int{"Bob": 20, "Charlie": 35}
merged := Merge(defaults, ages)
// map[Alice:30 Bob:25 Charlie:35]
// Bob overwritten with value from ages
\`\`\`

**Practical Benefits:**

1. **Universality**: Works with any key/value types
2. **Safe merging**: Creates new map without modifying originals
3. **Convenience**: Common operations in one line
4. **Type safety**: Compiler guarantees correctness`,
    order: 6,
    translations: {
        ru: {
            title: 'Обобщённые операции с map',
            solutionCode: `package generics

// Keys возвращает срез всех ключей в карте
func Keys[K comparable, V any](m map[K]V) []K {
    keys := make([]K, 0, len(m))
    for k := range m {
        keys = append(keys, k)
    }
    return keys
}

// Values возвращает срез всех значений в карте
func Values[K comparable, V any](m map[K]V) []V {
    values := make([]V, 0, len(m))
    for _, v := range m {
        values = append(values, v)
    }
    return values
}

// Merge объединяет две карты в новую карту
// Если ключ существует в обеих картах, значение из m2 имеет приоритет
func Merge[K comparable, V any](m1, m2 map[K]V) map[K]V {
    result := make(map[K]V, len(m1)+len(m2))

    // Копировать все записи из m1
    for k, v := range m1 {
        result[k] = v
    }

    // Копировать все записи из m2 (перезаписывает значения m1 при совпадении ключей)
    for k, v := range m2 {
        result[k] = v
    }

    return result
}`,
            description: `Создайте переиспользуемые утилитарные функции для работы с картами, используя обобщения.

Карты являются фундаментальными структурами данных в Go, но работа с ними часто требует повторяющегося кода. Обобщенные операции с картами предоставляют типобезопасные утилиты для общих задач с картами.

**Задача:** Реализуйте три обобщенные утилитарные функции для карт:
1. \`Keys[K, V]\` - извлекает все ключи из карты
2. \`Values[K, V]\` - извлекает все значения из карты
3. \`Merge[K, V]\` - объединяет две карты (вторая карта перезаписывает первую при конфликтах ключей)

**Требования:**
- \`Keys[K comparable, V any](m map[K]V) []K\` - возвращает срез всех ключей
- \`Values[K comparable, V any](m map[K]V) []V\` - возвращает срез всех значений
- \`Merge[K comparable, V any](m1, m2 map[K]V) map[K]V\` - возвращает новую объединенную карту
- Ключи должны быть сравнимыми (требуется для ключей карт в Go)

**Пример использования:**
\`\`\`go
ages := map[string]int{"Alice": 30, "Bob": 25, "Charlie": 35}

keys := Keys(ages)
// Результат: ["Alice", "Bob", "Charlie"] (порядок может отличаться)

values := Values(ages)
// Результат: [30, 25, 35] (порядок может отличаться)

m1 := map[string]int{"a": 1, "b": 2}
m2 := map[string]int{"b": 3, "c": 4}
merged := Merge(m1, m2)
// Результат: map[a:1 b:3 c:4]
\`\`\``,
            hint1: `Для Keys и Values предварительно выделите срез с емкостью \`len(m)\` для лучшей производительности.`,
            hint2: `Для Merge создайте новую карту и скопируйте записи из обеих входных карт. Записи из m2 естественным образом перезапишут записи m1 с одинаковыми ключами.`,
            whyItMatters: `Обобщенные утилиты для карт уменьшают шаблонный код и улучшают переиспользуемость кода в вашей кодовой базе. Эти операции часто требуются при обработке данных, ответах API и управлении конфигурацией.

**Продакшен паттерн:**

\`\`\`go
ages := map[string]int{
    "Alice": 30,
    "Bob": 25
}

// Извлечь все ключи
keys := Keys(ages)  // ["Alice", "Bob"]

// Извлечь все значения
values := Values(ages)  // [30, 25]

// Объединить карты
defaults := map[string]int{"Bob": 20, "Charlie": 35}
merged := Merge(defaults, ages)
// map[Alice:30 Bob:25 Charlie:35]
// Bob перезаписан значением из ages
\`\`\`

**Практические преимущества:**

1. **Универсальность**: Работает с любыми типами ключей/значений
2. **Безопасное слияние**: Создаёт новую map, не изменяя исходные
3. **Удобство**: Часто нужные операции в одной строке
4. **Типобезопасность**: Компилятор гарантирует корректность`
        },
        uz: {
            title: 'Generic map operatsiyalari',
            solutionCode: `package generics

// Keys mapdagi barcha kalitlar srezini qaytaradi
func Keys[K comparable, V any](m map[K]V) []K {
    keys := make([]K, 0, len(m))
    for k := range m {
        keys = append(keys, k)
    }
    return keys
}

// Values mapdagi barcha qiymatlar srezini qaytaradi
func Values[K comparable, V any](m map[K]V) []V {
    values := make([]V, 0, len(m))
    for _, v := range m {
        values = append(values, v)
    }
    return values
}

// Merge ikkita mapni yangi mapga birlashtiradi
// Agar kalit ikkala mapda mavjud bo'lsa, m2 dagi qiymat ustunlik qiladi
func Merge[K comparable, V any](m1, m2 map[K]V) map[K]V {
    result := make(map[K]V, len(m1)+len(m2))

    // m1 dan barcha yozuvlarni nusxalash
    for k, v := range m1 {
        result[k] = v
    }

    // m2 dan barcha yozuvlarni nusxalash (kalitlar mos kelsa m1 qiymatlarini qayta yozadi)
    for k, v := range m2 {
        result[k] = v
    }

    return result
}`,
            description: `Umumiy tiplardan foydalanib maplar bilan ishlash uchun qayta foydalaniladigan utility funksiyalarini yarating.

Maplar Go-da asosiy ma'lumot tuzilmalari hisoblanadi, lekin ular bilan ishlash ko'pincha takrorlanuvchi kodni talab qiladi. Umumiy map operatsiyalari umumiy map vazifalari uchun tip-xavfsiz utilitalarni taqdim etadi.

**Vazifa:** Uchta umumiy map utility funksiyasini amalga oshiring:
1. \`Keys[K, V]\` - mapdan barcha kalitlarni chiqaradi
2. \`Values[K, V]\` - mapdan barcha qiymatlarni chiqaradi
3. \`Merge[K, V]\` - ikkita mapni birlashtiradi (kalit to'qnashuvlarida ikkinchi map birinchisini qayta yozadi)

**Talablar:**
- \`Keys[K comparable, V any](m map[K]V) []K\` - barcha kalitlar srezini qaytaradi
- \`Values[K comparable, V any](m map[K]V) []V\` - barcha qiymatlar srezini qaytaradi
- \`Merge[K comparable, V any](m1, m2 map[K]V) map[K]V\` - yangi birlashtirilgan mapni qaytaradi
- Kalitlar taqqoslanishi kerak (Go-da map kalitlari uchun talab)

**Foydalanish misoli:**
\`\`\`go
ages := map[string]int{"Alice": 30, "Bob": 25, "Charlie": 35}

keys := Keys(ages)
// Natija: ["Alice", "Bob", "Charlie"] (tartib farq qilishi mumkin)

values := Values(ages)
// Natija: [30, 25, 35] (tartib farq qilishi mumkin)

m1 := map[string]int{"a": 1, "b": 2}
m2 := map[string]int{"b": 3, "c": 4}
merged := Merge(m1, m2)
// Natija: map[a:1 b:3 c:4]
\`\`\``,
            hint1: `Keys va Values uchun yaxshi ishlash uchun srezni \`len(m)\` sig'imi bilan oldindan ajrating.`,
            hint2: `Merge uchun yangi map yarating va ikkala kirish mapdan yozuvlarni nusxalang. m2 dagi yozuvlar bir xil kalitga ega m1 yozuvlarini tabiiy ravishda qayta yozadi.`,
            whyItMatters: `Umumiy map utitalari shablonli kodni kamaytiradi va kod bazangizda kodning qayta foydalanilishini yaxshilaydi. Bu operatsiyalar ma'lumotlarni qayta ishlash, API javoblari va konfiguratsiyani boshqarishda tez-tez kerak bo'ladi.

**Ishlab chiqarish patterni:**

\`\`\`go
ages := map[string]int{
    "Alice": 30,
    "Bob": 25
}

// Barcha kalitlarni chiqarish
keys := Keys(ages)  // ["Alice", "Bob"]

// Barcha qiymatlarni chiqarish
values := Values(ages)  // [30, 25]

// Maplarni birlashtirish
defaults := map[string]int{"Bob": 20, "Charlie": 35}
merged := Merge(defaults, ages)
// map[Alice:30 Bob:25 Charlie:35]
// Bob ages dan qiymat bilan qayta yozildi
\`\`\`

**Amaliy foydalari:**

1. **Universallik**: Har qanday kalit/qiymat turlari bilan ishlaydi
2. **Xavfsiz birlashtirish**: Asllarini o'zgartirmasdan yangi map yaratadi
3. **Qulaylik**: Tez-tez kerak operatsiyalar bir qatorda
4. **Tip xavfsizligi**: Kompilyator to'g'rilikni kafolatlaydi`
        }
    }
};

export default task;
