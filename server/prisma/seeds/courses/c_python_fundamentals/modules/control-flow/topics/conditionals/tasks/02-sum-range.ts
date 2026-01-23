import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-sum-range',
	title: 'Sum of Range',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'loops', 'for'],
	estimatedTime: '10m',
	isPremium: false,
	order: 2,

	description: `# Sum of Range

Practice using for loops with the \`range()\` function.

## Task

Implement the function \`sum_range(start, end)\` that returns the sum of all integers from start to end (inclusive).

## Requirements

- Include both start and end in the sum
- If start > end, return 0

## Examples

\`\`\`python
>>> sum_range(1, 5)
15  # 1 + 2 + 3 + 4 + 5

>>> sum_range(3, 3)
3

>>> sum_range(5, 1)
0
\`\`\``,

	initialCode: `def sum_range(start: int, end: int) -> int:
    """Calculate the sum of integers from start to end (inclusive).

    Args:
        start: Starting number
        end: Ending number (inclusive)

    Returns:
        Sum of all integers in the range
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def sum_range(start: int, end: int) -> int:
    """Calculate the sum of integers from start to end (inclusive).

    Args:
        start: Starting number
        end: Ending number (inclusive)

    Returns:
        Sum of all integers in the range
    """
    if start > end:
        return 0

    total = 0
    for i in range(start, end + 1):
        total += i
    return total`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic range"""
        self.assertEqual(sum_range(1, 5), 15)

    def test_2(self):
        """Same start and end"""
        self.assertEqual(sum_range(3, 3), 3)

    def test_3(self):
        """Invalid range"""
        self.assertEqual(sum_range(5, 1), 0)

    def test_4(self):
        """Range from 0"""
        self.assertEqual(sum_range(0, 3), 6)

    def test_5(self):
        """Negative range"""
        self.assertEqual(sum_range(-2, 2), 0)

    def test_6(self):
        """Large range"""
        self.assertEqual(sum_range(1, 100), 5050)

    def test_7(self):
        """Single zero"""
        self.assertEqual(sum_range(0, 0), 0)

    def test_8(self):
        """Negative numbers"""
        self.assertEqual(sum_range(-5, -1), -15)

    def test_9(self):
        """Range of two"""
        self.assertEqual(sum_range(10, 11), 21)

    def test_10(self):
        """Starting from 1"""
        self.assertEqual(sum_range(1, 10), 55)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use range(start, end + 1) to include the end value in the loop.',
	hint2: 'Initialize a total variable to 0, then add each number in the loop.',

	whyItMatters: `Looping through ranges is fundamental to many algorithms. Understanding range() is essential for Python.

**Production Pattern:**

\`\`\`python
def process_batch(items: list, batch_size: int = 100):
    """Process items in batches."""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield batch

def generate_report_pages(total_items: int, page_size: int = 10):
    """Calculate page numbers for pagination."""
    total_pages = (total_items + page_size - 1) // page_size
    return list(range(1, total_pages + 1))
\`\`\`

**Practical Benefits:**
- range() is memory-efficient (lazy evaluation)
- Batch processing prevents memory overflow
- Pagination requires range calculations`,

	translations: {
		ru: {
			title: 'Сумма диапазона',
			description: `# Сумма диапазона

Практика использования циклов for с функцией \`range()\`.

## Задача

Реализуйте функцию \`sum_range(start, end)\`, которая возвращает сумму всех целых чисел от start до end (включительно).

## Требования

- Включите start и end в сумму
- Если start > end, верните 0

## Примеры

\`\`\`python
>>> sum_range(1, 5)
15  # 1 + 2 + 3 + 4 + 5

>>> sum_range(3, 3)
3

>>> sum_range(5, 1)
0
\`\`\``,
			hint1: 'Используйте range(start, end + 1), чтобы включить end в цикл.',
			hint2: 'Инициализируйте переменную total = 0, затем добавляйте каждое число.',
			whyItMatters: `Циклы по диапазонам — основа многих алгоритмов.

**Продакшен паттерн:**

\`\`\`python
def process_batch(items: list, batch_size: int = 100):
    """Обработка элементов пакетами."""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield batch
\`\`\`

**Практические преимущества:**
- range() эффективен по памяти (ленивые вычисления)
- Пакетная обработка предотвращает переполнение памяти`,
		},
		uz: {
			title: 'Diapazon yig\'indisi',
			description: `# Diapazon yig'indisi

\`range()\` funksiyasi bilan for sikllaridan foydalanish mashqi.

## Vazifa

start dan end gacha (shu jumladan) barcha butun sonlar yig'indisini qaytaruvchi \`sum_range(start, end)\` funksiyasini amalga oshiring.

## Talablar

- start va end ni yig'indiga qo'shing
- Agar start > end bo'lsa, 0 qaytaring

## Misollar

\`\`\`python
>>> sum_range(1, 5)
15  # 1 + 2 + 3 + 4 + 5

>>> sum_range(3, 3)
3

>>> sum_range(5, 1)
0
\`\`\``,
			hint1: "end ni siklga qo'shish uchun range(start, end + 1) dan foydalaning.",
			hint2: "total o'zgaruvchisini 0 ga tenglang, keyin har bir sonni qo'shing.",
			whyItMatters: `Diapazonlar bo'yicha sikllash ko'p algoritmlarning asosidir.

**Ishlab chiqarish patterni:**

\`\`\`python
def process_batch(items: list, batch_size: int = 100):
    """Elementlarni paketlarda qayta ishlash."""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield batch
\`\`\`

**Amaliy foydalari:**
- range() xotira bo'yicha samarali
- Paketli qayta ishlash xotira to'lib ketishini oldini oladi`,
		},
	},
};

export default task;
