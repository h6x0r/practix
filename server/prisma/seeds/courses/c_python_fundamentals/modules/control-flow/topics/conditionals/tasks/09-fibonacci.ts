import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-fibonacci',
	title: 'Fibonacci Sequence',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'loops', 'math', 'sequences'],
	estimatedTime: '15m',
	isPremium: false,
	order: 9,

	description: `# Fibonacci Sequence

The Fibonacci sequence is a series where each number is the sum of the two preceding ones.

## Task

Implement the function \`fibonacci(n)\` that returns the first n numbers in the Fibonacci sequence.

## Requirements

- Start with 0 and 1: [0, 1, 1, 2, 3, 5, 8, ...]
- If n is 0, return an empty list
- If n is 1, return [0]
- If n is 2, return [0, 1]

## Examples

\`\`\`python
>>> fibonacci(7)
[0, 1, 1, 2, 3, 5, 8]

>>> fibonacci(1)
[0]

>>> fibonacci(0)
[]

>>> fibonacci(10)
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
\`\`\``,

	initialCode: `def fibonacci(n: int) -> list[int]:
    """Generate the first n Fibonacci numbers.

    The Fibonacci sequence starts with 0, 1, and each
    subsequent number is the sum of the previous two.

    Args:
        n: Number of Fibonacci numbers to generate

    Returns:
        List of the first n Fibonacci numbers
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def fibonacci(n: int) -> list[int]:
    """Generate the first n Fibonacci numbers.

    The Fibonacci sequence starts with 0, 1, and each
    subsequent number is the sum of the previous two.

    Args:
        n: Number of Fibonacci numbers to generate

    Returns:
        List of the first n Fibonacci numbers
    """
    # Handle edge cases
    if n <= 0:
        return []
    if n == 1:
        return [0]

    # Start with the first two Fibonacci numbers
    result = [0, 1]

    # Generate remaining numbers
    # Each new number is the sum of the previous two
    while len(result) < n:
        next_num = result[-1] + result[-2]  # Sum of last two
        result.append(next_num)

    return result`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """First 7 Fibonacci numbers"""
        self.assertEqual(fibonacci(7), [0, 1, 1, 2, 3, 5, 8])

    def test_2(self):
        """Single element"""
        self.assertEqual(fibonacci(1), [0])

    def test_3(self):
        """Empty for zero"""
        self.assertEqual(fibonacci(0), [])

    def test_4(self):
        """First 10 numbers"""
        self.assertEqual(fibonacci(10), [0, 1, 1, 2, 3, 5, 8, 13, 21, 34])

    def test_5(self):
        """Two elements"""
        self.assertEqual(fibonacci(2), [0, 1])

    def test_6(self):
        """Three elements"""
        self.assertEqual(fibonacci(3), [0, 1, 1])

    def test_7(self):
        """Negative input returns empty"""
        self.assertEqual(fibonacci(-5), [])

    def test_8(self):
        """Five elements"""
        self.assertEqual(fibonacci(5), [0, 1, 1, 2, 3])

    def test_9(self):
        """First 15 numbers"""
        result = fibonacci(15)
        self.assertEqual(len(result), 15)
        self.assertEqual(result[-1], 377)

    def test_10(self):
        """Verify sequence property"""
        result = fibonacci(20)
        for i in range(2, len(result)):
            self.assertEqual(result[i], result[i-1] + result[i-2])

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Handle edge cases first: n <= 0 returns [], n == 1 returns [0]. Start with [0, 1] for n >= 2.',
	hint2: 'Use result[-1] and result[-2] to get the last two elements, then append their sum.',

	whyItMatters: `The Fibonacci sequence appears in mathematics, nature, and computer science algorithms.

**Production Pattern:**

\`\`\`python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_recursive(n: int) -> int:
    """Memoized recursive Fibonacci - O(n) time."""
    if n < 2:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

def fib_generator(limit: int):
    """Memory-efficient Fibonacci generator."""
    a, b = 0, 1
    count = 0
    while count < limit:
        yield a
        a, b = b, a + b
        count += 1

def matrix_power_fib(n: int) -> int:
    """O(log n) Fibonacci using matrix exponentiation."""
    def multiply(a, b):
        return [
            [a[0][0]*b[0][0] + a[0][1]*b[1][0],
             a[0][0]*b[0][1] + a[0][1]*b[1][1]],
            [a[1][0]*b[0][0] + a[1][1]*b[1][0],
             a[1][0]*b[0][1] + a[1][1]*b[1][1]]
        ]

    def matrix_pow(m, p):
        if p == 1:
            return m
        if p % 2 == 0:
            half = matrix_pow(m, p // 2)
            return multiply(half, half)
        return multiply(m, matrix_pow(m, p - 1))

    if n == 0:
        return 0
    result = matrix_pow([[1, 1], [1, 0]], n)
    return result[0][1]
\`\`\`

**Practical Benefits:**
- Understanding sequences helps with dynamic programming
- Generators are memory-efficient for large sequences
- Matrix exponentiation demonstrates algorithmic optimization`,

	translations: {
		ru: {
			title: 'Последовательность Фибоначчи',
			description: `# Последовательность Фибоначчи

Последовательность Фибоначчи — это ряд, где каждое число равно сумме двух предыдущих.

## Задача

Реализуйте функцию \`fibonacci(n)\`, которая возвращает первые n чисел Фибоначчи.

## Требования

- Начните с 0 и 1: [0, 1, 1, 2, 3, 5, 8, ...]
- Если n равно 0, верните пустой список
- Если n равно 1, верните [0]
- Если n равно 2, верните [0, 1]

## Примеры

\`\`\`python
>>> fibonacci(7)
[0, 1, 1, 2, 3, 5, 8]

>>> fibonacci(1)
[0]

>>> fibonacci(0)
[]

>>> fibonacci(10)
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
\`\`\``,
			hint1: 'Сначала обработайте краевые случаи: n <= 0 возвращает [], n == 1 возвращает [0].',
			hint2: 'Используйте result[-1] и result[-2] для двух последних элементов, добавьте их сумму.',
			whyItMatters: `Последовательность Фибоначчи встречается в математике, природе и алгоритмах.

**Продакшен паттерн:**

\`\`\`python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_recursive(n: int) -> int:
    """Мемоизированный рекурсивный Фибоначчи — O(n)."""
    if n < 2:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

def fib_generator(limit: int):
    """Экономичный по памяти генератор Фибоначчи."""
    a, b = 0, 1
    count = 0
    while count < limit:
        yield a
        a, b = b, a + b
        count += 1
\`\`\`

**Практические преимущества:**
- Понимание последовательностей помогает с динамическим программированием
- Генераторы экономят память для больших последовательностей`,
		},
		uz: {
			title: 'Fibonachchi ketma-ketligi',
			description: `# Fibonachchi ketma-ketligi

Fibonachchi ketma-ketligi — har bir son oldingi ikkitasining yig'indisiga teng bo'lgan qator.

## Vazifa

Birinchi n ta Fibonachchi sonlarini qaytaruvchi \`fibonacci(n)\` funksiyasini amalga oshiring.

## Talablar

- 0 va 1 dan boshlang: [0, 1, 1, 2, 3, 5, 8, ...]
- n 0 bo'lsa, bo'sh ro'yxat qaytaring
- n 1 bo'lsa, [0] qaytaring
- n 2 bo'lsa, [0, 1] qaytaring

## Misollar

\`\`\`python
>>> fibonacci(7)
[0, 1, 1, 2, 3, 5, 8]

>>> fibonacci(1)
[0]

>>> fibonacci(0)
[]

>>> fibonacci(10)
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
\`\`\``,
			hint1: "Avval chegaraviy holatlarni ko'rib chiqing: n <= 0 [] qaytaradi, n == 1 [0] qaytaradi.",
			hint2: "Oxirgi ikki element uchun result[-1] va result[-2] ishlating, ularning yig'indisini qo'shing.",
			whyItMatters: `Fibonachchi ketma-ketligi matematika, tabiat va algoritmlarda uchraydi.

**Ishlab chiqarish patterni:**

\`\`\`python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_recursive(n: int) -> int:
    """Memoizatsiyalangan rekursiv Fibonachchi — O(n)."""
    if n < 2:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

def fib_generator(limit: int):
    """Xotirani tejaydigan Fibonachchi generatori."""
    a, b = 0, 1
    count = 0
    while count < limit:
        yield a
        a, b = b, a + b
        count += 1
\`\`\`

**Amaliy foydalari:**
- Ketma-ketliklarni tushunish dinamik dasturlashga yordam beradi
- Generatorlar katta ketma-ketliklar uchun xotirani tejaydi`,
		},
	},
};

export default task;
