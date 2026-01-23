import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-unique-elements',
	title: 'Remove Duplicates',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'lists', 'sets'],
	estimatedTime: '10m',
	isPremium: false,
	order: 2,

	description: `# Remove Duplicates

Learn to use sets for removing duplicate elements.

## Task

Implement the function \`unique_elements(items)\` that returns a list with duplicates removed.

## Requirements

- Preserve the original order of first occurrences
- Return an empty list for empty input

## Examples

\`\`\`python
>>> unique_elements([1, 2, 2, 3, 3, 3])
[1, 2, 3]

>>> unique_elements(["a", "b", "a", "c"])
["a", "b", "c"]

>>> unique_elements([])
[]
\`\`\``,

	initialCode: `def unique_elements(items: list) -> list:
    """Return a list with duplicate elements removed, preserving order.

    Args:
        items: Input list (may contain duplicates)

    Returns:
        List with unique elements in original order
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def unique_elements(items: list) -> list:
    """Return a list with duplicate elements removed, preserving order.

    Args:
        items: Input list (may contain duplicates)

    Returns:
        List with unique elements in original order
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Remove number duplicates"""
        self.assertEqual(unique_elements([1, 2, 2, 3, 3, 3]), [1, 2, 3])

    def test_2(self):
        """Remove string duplicates"""
        self.assertEqual(unique_elements(["a", "b", "a", "c"]), ["a", "b", "c"])

    def test_3(self):
        """Empty list"""
        self.assertEqual(unique_elements([]), [])

    def test_4(self):
        """No duplicates"""
        self.assertEqual(unique_elements([1, 2, 3]), [1, 2, 3])

    def test_5(self):
        """All same"""
        self.assertEqual(unique_elements([5, 5, 5, 5]), [5])

    def test_6(self):
        """Order preserved"""
        self.assertEqual(unique_elements([3, 1, 2, 1, 3]), [3, 1, 2])

    def test_7(self):
        """Single element"""
        self.assertEqual(unique_elements([42]), [42])

    def test_8(self):
        """Boolean values"""
        self.assertEqual(unique_elements([True, False, True, False]), [True, False])

    def test_9(self):
        """Mixed duplicates"""
        self.assertEqual(unique_elements([1, "1", 1, "1"]), [1, "1"])

    def test_10(self):
        """Long list"""
        result = unique_elements([1, 2] * 100)
        self.assertEqual(result, [1, 2])

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use a set to track which elements you have already seen.',
	hint2: 'Loop through items, add to result only if not in the "seen" set.',

	whyItMatters: `Removing duplicates while preserving order is a common data cleaning task.

**Production Pattern:**

\`\`\`python
def deduplicate_by_key(items: list[dict], key: str) -> list[dict]:
    """Remove duplicate objects by a specific key."""
    seen = set()
    result = []
    for item in items:
        value = item.get(key)
        if value not in seen:
            seen.add(value)
            result.append(item)
    return result

# Usage: deduplicate_by_key(users, "email")
\`\`\`

**Practical Benefits:**
- Set lookup is O(1) for fast duplicate detection
- Order preservation is important for user-facing data
- Essential for data cleaning pipelines`,

	translations: {
		ru: {
			title: 'Удаление дубликатов',
			description: `# Удаление дубликатов

Научитесь использовать множества для удаления дубликатов.

## Задача

Реализуйте функцию \`unique_elements(items)\`, которая возвращает список без дубликатов.

## Требования

- Сохраните порядок первых вхождений
- Для пустого ввода верните пустой список

## Примеры

\`\`\`python
>>> unique_elements([1, 2, 2, 3, 3, 3])
[1, 2, 3]

>>> unique_elements(["a", "b", "a", "c"])
["a", "b", "c"]

>>> unique_elements([])
[]
\`\`\``,
			hint1: 'Используйте множество для отслеживания уже встреченных элементов.',
			hint2: 'Пройдитесь по items, добавляйте в result только если нет в "seen".',
			whyItMatters: `Удаление дубликатов с сохранением порядка — частая задача очистки данных.

**Продакшен паттерн:**

\`\`\`python
def deduplicate_by_key(items: list[dict], key: str) -> list[dict]:
    """Удаление дубликатов объектов по ключу."""
    seen = set()
    result = []
    for item in items:
        value = item.get(key)
        if value not in seen:
            seen.add(value)
            result.append(item)
    return result
\`\`\`

**Практические преимущества:**
- Поиск в множестве O(1) для быстрого обнаружения дубликатов
- Сохранение порядка важно для данных пользователя`,
		},
		uz: {
			title: 'Takrorlarni olib tashlash',
			description: `# Takrorlarni olib tashlash

Takrorlarni olib tashlash uchun to'plamlardan foydalanishni o'rganing.

## Vazifa

Takrorlarsiz ro'yxat qaytaruvchi \`unique_elements(items)\` funksiyasini amalga oshiring.

## Talablar

- Birinchi uchrashuvlar tartibini saqlang
- Bo'sh kiritish uchun bo'sh ro'yxat qaytaring

## Misollar

\`\`\`python
>>> unique_elements([1, 2, 2, 3, 3, 3])
[1, 2, 3]

>>> unique_elements(["a", "b", "a", "c"])
["a", "b", "c"]

>>> unique_elements([])
[]
\`\`\``,
			hint1: "Allaqachon ko'rilgan elementlarni kuzatish uchun to'plamdan foydalaning.",
			hint2: "items bo'ylab yuring, faqat \"seen\" da bo'lmasa result ga qo'shing.",
			whyItMatters: `Tartibni saqlab takrorlarni olib tashlash ma'lumotlarni tozalashning keng tarqalgan vazifasi.

**Ishlab chiqarish patterni:**

\`\`\`python
def deduplicate_by_key(items: list[dict], key: str) -> list[dict]:
    """Ob'ektlarni kalit bo'yicha takrorlarini olib tashlash."""
    seen = set()
    result = []
    for item in items:
        value = item.get(key)
        if value not in seen:
            seen.add(value)
            result.append(item)
    return result
\`\`\`

**Amaliy foydalari:**
- To'plamda qidirish takrorlarni tez aniqlash uchun O(1)
- Tartibni saqlash foydalanuvchi ma'lumotlari uchun muhim`,
		},
	},
};

export default task;
