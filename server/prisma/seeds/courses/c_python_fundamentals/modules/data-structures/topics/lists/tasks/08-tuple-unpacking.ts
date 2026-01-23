import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-tuple-unpacking',
	title: 'Tuple Unpacking',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'tuples', 'unpacking'],
	estimatedTime: '10m',
	isPremium: false,
	order: 8,

	description: `# Tuple Unpacking

Tuples are immutable sequences often used for returning multiple values.

## Task

Implement the function \`swap_and_analyze(a, b)\` that returns a tuple with swapped values and their analysis.

## Requirements

Return a tuple with 5 elements:
1. Swapped values as a tuple: \`(b, a)\`
2. Sum of the values
3. Product of the values
4. Minimum value
5. Maximum value

## Examples

\`\`\`python
>>> swap_and_analyze(3, 7)
((7, 3), 10, 21, 3, 7)

>>> swap_and_analyze(5, 5)
((5, 5), 10, 25, 5, 5)

>>> swap_and_analyze(-2, 4)
((4, -2), 2, -8, -2, 4)
\`\`\``,

	initialCode: `def swap_and_analyze(a: int, b: int) -> tuple:
    """Swap values and return analysis tuple.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Tuple containing:
        - (b, a): swapped values
        - sum of a and b
        - product of a and b
        - minimum of a and b
        - maximum of a and b
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def swap_and_analyze(a: int, b: int) -> tuple:
    """Swap values and return analysis tuple.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Tuple containing:
        - (b, a): swapped values
        - sum of a and b
        - product of a and b
        - minimum of a and b
        - maximum of a and b
    """
    # Create swapped tuple using tuple packing
    swapped = (b, a)

    # Calculate sum and product
    total = a + b
    product = a * b

    # Find min and max
    minimum = min(a, b)
    maximum = max(a, b)

    # Return tuple with all values
    return (swapped, total, product, minimum, maximum)`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic positive numbers"""
        result = swap_and_analyze(3, 7)
        self.assertEqual(result[0], (7, 3))
        self.assertEqual(result[1], 10)
        self.assertEqual(result[2], 21)
        self.assertEqual(result[3], 3)
        self.assertEqual(result[4], 7)

    def test_2(self):
        """Equal values"""
        result = swap_and_analyze(5, 5)
        self.assertEqual(result[0], (5, 5))
        self.assertEqual(result[1], 10)
        self.assertEqual(result[2], 25)

    def test_3(self):
        """Negative and positive"""
        result = swap_and_analyze(-2, 4)
        self.assertEqual(result[0], (4, -2))
        self.assertEqual(result[1], 2)
        self.assertEqual(result[2], -8)
        self.assertEqual(result[3], -2)
        self.assertEqual(result[4], 4)

    def test_4(self):
        """Both negative"""
        result = swap_and_analyze(-5, -3)
        self.assertEqual(result[0], (-3, -5))
        self.assertEqual(result[2], 15)
        self.assertEqual(result[3], -5)

    def test_5(self):
        """With zero"""
        result = swap_and_analyze(0, 10)
        self.assertEqual(result[1], 10)
        self.assertEqual(result[2], 0)

    def test_6(self):
        """Both zeros"""
        result = swap_and_analyze(0, 0)
        self.assertEqual(result, ((0, 0), 0, 0, 0, 0))

    def test_7(self):
        """Large numbers"""
        result = swap_and_analyze(100, 200)
        self.assertEqual(result[1], 300)
        self.assertEqual(result[2], 20000)

    def test_8(self):
        """One and negative one"""
        result = swap_and_analyze(1, -1)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], -1)

    def test_9(self):
        """Reversed order input"""
        result = swap_and_analyze(10, 2)
        self.assertEqual(result[0], (2, 10))

    def test_10(self):
        """Tuple structure check"""
        result = swap_and_analyze(1, 2)
        self.assertEqual(len(result), 5)
        self.assertIsInstance(result[0], tuple)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Create a tuple of swapped values: swapped = (b, a). Then calculate sum, product, min, max.',
	hint2: 'Return a tuple containing all five results: (swapped, sum, product, min, max).',

	whyItMatters: `Tuple unpacking is a powerful Python feature for returning and handling multiple values.

**Production Pattern:**

\`\`\`python
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float

class Rectangle(NamedTuple):
    top_left: Point
    width: float
    height: float

def parse_coordinates(text: str) -> tuple[float, float]:
    """Parse 'lat,lng' string into coordinate tuple."""
    parts = text.split(",")
    return (float(parts[0].strip()), float(parts[1].strip()))

def get_stats(numbers: list[int]) -> tuple[int, int, float, int]:
    """Return (min, max, average, count) for a list of numbers."""
    if not numbers:
        return (0, 0, 0.0, 0)

    return (
        min(numbers),
        max(numbers),
        sum(numbers) / len(numbers),
        len(numbers),
    )

# Usage with unpacking:
minimum, maximum, avg, count = get_stats([1, 2, 3, 4, 5])

# Python swap idiom - elegant tuple unpacking:
a, b = b, a  # Swap values in one line
\`\`\`

**Practical Benefits:**
- Functions can return multiple values cleanly
- Named tuples provide readable, typed data structures
- Unpacking enables elegant variable assignment`,

	translations: {
		ru: {
			title: 'Распаковка кортежей',
			description: `# Распаковка кортежей

Кортежи — неизменяемые последовательности, часто используемые для возврата нескольких значений.

## Задача

Реализуйте функцию \`swap_and_analyze(a, b)\`, которая возвращает кортеж с обменянными значениями и их анализом.

## Требования

Верните кортеж из 5 элементов:
1. Обменянные значения как кортеж: \`(b, a)\`
2. Сумма значений
3. Произведение значений
4. Минимум
5. Максимум

## Примеры

\`\`\`python
>>> swap_and_analyze(3, 7)
((7, 3), 10, 21, 3, 7)

>>> swap_and_analyze(5, 5)
((5, 5), 10, 25, 5, 5)

>>> swap_and_analyze(-2, 4)
((4, -2), 2, -8, -2, 4)
\`\`\``,
			hint1: 'Создайте кортеж обменянных значений: swapped = (b, a). Затем вычислите сумму, произведение, min, max.',
			hint2: 'Верните кортеж со всеми пятью результатами: (swapped, sum, product, min, max).',
			whyItMatters: `Распаковка кортежей — мощная возможность Python для возврата множества значений.

**Продакшен паттерн:**

\`\`\`python
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float

def get_stats(numbers: list[int]) -> tuple[int, int, float, int]:
    """Вернуть (min, max, среднее, количество)."""
    if not numbers:
        return (0, 0, 0.0, 0)
    return (min(numbers), max(numbers), sum(numbers) / len(numbers), len(numbers))

# Использование с распаковкой:
minimum, maximum, avg, count = get_stats([1, 2, 3, 4, 5])

# Идиома обмена в Python:
a, b = b, a  # Обмен значений в одну строку
\`\`\`

**Практические преимущества:**
- Функции могут возвращать несколько значений
- Named tuples обеспечивают читаемые структуры данных`,
		},
		uz: {
			title: "Kortejlarni ochish",
			description: `# Kortejlarni ochish

Kortejlar — ko'p qiymatlar qaytarish uchun ishlatiladigan o'zgarmas ketma-ketliklar.

## Vazifa

Almashtirilgan qiymatlar va ularning tahlilini qaytaruvchi \`swap_and_analyze(a, b)\` funksiyasini amalga oshiring.

## Talablar

5 elementdan iborat kortej qaytaring:
1. Almashtirilgan qiymatlar korteji: \`(b, a)\`
2. Qiymatlar yig'indisi
3. Qiymatlar ko'paytmasi
4. Minimal qiymat
5. Maksimal qiymat

## Misollar

\`\`\`python
>>> swap_and_analyze(3, 7)
((7, 3), 10, 21, 3, 7)

>>> swap_and_analyze(5, 5)
((5, 5), 10, 25, 5, 5)

>>> swap_and_analyze(-2, 4)
((4, -2), 2, -8, -2, 4)
\`\`\``,
			hint1: "Almashtirilgan qiymatlar korteji yarating: swapped = (b, a). So'ng yig'indi, ko'paytma, min, max hisoblang.",
			hint2: "Barcha beshta natijani kortejda qaytaring: (swapped, sum, product, min, max).",
			whyItMatters: `Kortejlarni ochish — ko'p qiymatlarni qaytarish uchun Python ning kuchli imkoniyati.

**Ishlab chiqarish patterni:**

\`\`\`python
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float

def get_stats(numbers: list[int]) -> tuple[int, int, float, int]:
    """(min, max, o'rtacha, soni) qaytarish."""
    if not numbers:
        return (0, 0, 0.0, 0)
    return (min(numbers), max(numbers), sum(numbers) / len(numbers), len(numbers))

# Ochish bilan foydalanish:
minimum, maximum, avg, count = get_stats([1, 2, 3, 4, 5])

# Python almashtirish idiomasi:
a, b = b, a  # Qiymatlarni bir qatorda almashtirish
\`\`\`

**Amaliy foydalari:**
- Funksiyalar bir nechta qiymatlarni qaytarishi mumkin
- Named tuples o'qilishi oson ma'lumot tuzilmalarini ta'minlaydi`,
		},
	},
};

export default task;
