import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-list-comprehension',
	title: 'List Comprehension',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'lists', 'comprehension'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,

	description: `# List Comprehension

Master Python's powerful list comprehension syntax.

## Task

Implement the function \`squares_of_even(numbers)\` that returns squares of all even numbers.

## Requirements

- Use list comprehension syntax
- Return squares of even numbers only
- Preserve order

## Examples

\`\`\`python
>>> squares_of_even([1, 2, 3, 4, 5, 6])
[4, 16, 36]  # 2²=4, 4²=16, 6²=36

>>> squares_of_even([1, 3, 5])
[]

>>> squares_of_even([2, 4])
[4, 16]
\`\`\``,

	initialCode: `def squares_of_even(numbers: list[int]) -> list[int]:
    """Return squares of all even numbers using list comprehension.

    Args:
        numbers: List of integers

    Returns:
        List of squares of even numbers
    """
    # TODO: Use list comprehension
    pass`,

	solutionCode: `def squares_of_even(numbers: list[int]) -> list[int]:
    """Return squares of all even numbers using list comprehension.

    Args:
        numbers: List of integers

    Returns:
        List of squares of even numbers
    """
    return [n ** 2 for n in numbers if n % 2 == 0]`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Mixed numbers"""
        self.assertEqual(squares_of_even([1, 2, 3, 4, 5, 6]), [4, 16, 36])

    def test_2(self):
        """No even numbers"""
        self.assertEqual(squares_of_even([1, 3, 5]), [])

    def test_3(self):
        """All even"""
        self.assertEqual(squares_of_even([2, 4]), [4, 16])

    def test_4(self):
        """Empty list"""
        self.assertEqual(squares_of_even([]), [])

    def test_5(self):
        """Zero included"""
        self.assertEqual(squares_of_even([0, 1, 2]), [0, 4])

    def test_6(self):
        """Negative even"""
        self.assertEqual(squares_of_even([-2, -1, 0]), [4, 0])

    def test_7(self):
        """Single even"""
        self.assertEqual(squares_of_even([4]), [16])

    def test_8(self):
        """Large numbers"""
        self.assertEqual(squares_of_even([10, 11, 12]), [100, 144])

    def test_9(self):
        """All same even"""
        self.assertEqual(squares_of_even([2, 2, 2]), [4, 4, 4])

    def test_10(self):
        """Consecutive"""
        self.assertEqual(squares_of_even(list(range(1, 11))), [4, 16, 36, 64, 100])

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'List comprehension syntax: [expression for item in list if condition]',
	hint2: '[n ** 2 for n in numbers if n % 2 == 0]',

	whyItMatters: `List comprehensions are more Pythonic, readable, and often faster than loops.

**Production Pattern:**

\`\`\`python
# Transform and filter data
valid_emails = [u["email"].lower() for u in users if "@" in u.get("email", "")]

# Extract specific fields
names = [item["name"] for item in response["data"] if item.get("active")]

# Nested comprehension for flattening
all_items = [item for sublist in nested_lists for item in sublist]

# Dictionary comprehension
word_lengths = {word: len(word) for word in words}
\`\`\`

**Practical Benefits:**
- More concise than explicit loops
- Often faster due to internal optimization
- Clearly expresses intent: transform + filter`,

	translations: {
		ru: {
			title: 'List Comprehension',
			description: `# List Comprehension

Освойте мощный синтаксис list comprehension в Python.

## Задача

Реализуйте функцию \`squares_of_even(numbers)\`, которая возвращает квадраты всех чётных чисел.

## Требования

- Используйте синтаксис list comprehension
- Верните квадраты только чётных чисел
- Сохраните порядок

## Примеры

\`\`\`python
>>> squares_of_even([1, 2, 3, 4, 5, 6])
[4, 16, 36]  # 2²=4, 4²=16, 6²=36

>>> squares_of_even([1, 3, 5])
[]

>>> squares_of_even([2, 4])
[4, 16]
\`\`\``,
			hint1: 'Синтаксис: [выражение for элемент in список if условие]',
			hint2: '[n ** 2 for n in numbers if n % 2 == 0]',
			whyItMatters: `List comprehensions более питоничны, читаемы и часто быстрее циклов.

**Продакшен паттерн:**

\`\`\`python
# Трансформация и фильтрация данных
valid_emails = [u["email"].lower() for u in users if "@" in u.get("email", "")]

# Извлечение полей
names = [item["name"] for item in response["data"] if item.get("active")]
\`\`\`

**Практические преимущества:**
- Более компактны чем циклы
- Часто быстрее благодаря оптимизации
- Ясно выражают намерение: трансформация + фильтр`,
		},
		uz: {
			title: 'List Comprehension',
			description: `# List Comprehension

Python ning kuchli list comprehension sintaksisini o'rganing.

## Vazifa

Barcha juft sonlarning kvadratlarini qaytaruvchi \`squares_of_even(numbers)\` funksiyasini amalga oshiring.

## Talablar

- List comprehension sintaksisidan foydalaning
- Faqat juft sonlarning kvadratlarini qaytaring
- Tartibni saqlang

## Misollar

\`\`\`python
>>> squares_of_even([1, 2, 3, 4, 5, 6])
[4, 16, 36]  # 2²=4, 4²=16, 6²=36

>>> squares_of_even([1, 3, 5])
[]

>>> squares_of_even([2, 4])
[4, 16]
\`\`\``,
			hint1: "Sintaksis: [ifoda for element in ro'yxat if shart]",
			hint2: "[n ** 2 for n in numbers if n % 2 == 0]",
			whyItMatters: `List comprehensions ko'proq Pythonik, o'qilishi oson va ko'pincha sikllardan tezroq.

**Ishlab chiqarish patterni:**

\`\`\`python
# Ma'lumotlarni o'zgartirish va filtrlash
valid_emails = [u["email"].lower() for u in users if "@" in u.get("email", "")]

# Maydonlarni ajratib olish
names = [item["name"] for item in response["data"] if item.get("active")]
\`\`\`

**Amaliy foydalari:**
- Sikllardan ko'ra ixchamroq
- Optimizatsiya tufayli ko'pincha tezroq
- Maqsadni aniq ifodalaydi: o'zgartirish + filtr`,
		},
	},
};

export default task;
