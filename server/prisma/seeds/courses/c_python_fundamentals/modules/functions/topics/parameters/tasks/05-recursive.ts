import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-recursive',
	title: 'Recursive Functions',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'functions', 'recursion'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,

	description: `# Recursive Functions

Learn the basics of recursion with a classic example.

## Task

Implement \`factorial(n)\` using recursion.

## Requirements

- factorial(0) = 1
- factorial(n) = n * factorial(n-1)
- Return None for negative numbers

## Examples

\`\`\`python
>>> factorial(5)
120  # 5 * 4 * 3 * 2 * 1

>>> factorial(0)
1

>>> factorial(-1)
None
\`\`\``,

	initialCode: `def factorial(n: int) -> int | None:
    """Calculate factorial recursively.

    Args:
        n: Non-negative integer

    Returns:
        n! or None if n is negative
    """
    # TODO: Implement recursively
    pass`,

	solutionCode: `def factorial(n: int) -> int | None:
    """Calculate factorial recursively.

    Args:
        n: Non-negative integer

    Returns:
        n! or None if n is negative
    """
    if n < 0:
        return None
    if n == 0:
        return 1
    return n * factorial(n - 1)`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Factorial of 5"""
        self.assertEqual(factorial(5), 120)

    def test_2(self):
        """Factorial of 0"""
        self.assertEqual(factorial(0), 1)

    def test_3(self):
        """Negative number"""
        self.assertIsNone(factorial(-1))

    def test_4(self):
        """Factorial of 1"""
        self.assertEqual(factorial(1), 1)

    def test_5(self):
        """Factorial of 3"""
        self.assertEqual(factorial(3), 6)

    def test_6(self):
        """Factorial of 10"""
        self.assertEqual(factorial(10), 3628800)

    def test_7(self):
        """Factorial of 7"""
        self.assertEqual(factorial(7), 5040)

    def test_8(self):
        """Large negative"""
        self.assertIsNone(factorial(-100))

    def test_9(self):
        """Factorial of 2"""
        self.assertEqual(factorial(2), 2)

    def test_10(self):
        """Factorial of 4"""
        self.assertEqual(factorial(4), 24)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Base case: if n == 0, return 1',
	hint2: 'Recursive case: return n * factorial(n - 1)',

	whyItMatters: `Recursion is essential for tree traversal, divide-and-conquer algorithms, and many data structures.

**Production Pattern:**

\`\`\`python
def flatten_nested(data):
    """Recursively flatten nested lists/dicts."""
    result = []
    if isinstance(data, list):
        for item in data:
            result.extend(flatten_nested(item))
    elif isinstance(data, dict):
        for value in data.values():
            result.extend(flatten_nested(value))
    else:
        result.append(data)
    return result

def find_in_tree(node, target):
    """Search a tree structure recursively."""
    if node is None:
        return None
    if node.value == target:
        return node
    for child in node.children:
        result = find_in_tree(child, target)
        if result:
            return result
    return None
\`\`\`

**Practical Benefits:**
- Natural solution for hierarchical data
- Foundation for tree/graph algorithms
- Essential for parsing and traversal`,

	translations: {
		ru: {
			title: 'Рекурсивные функции',
			description: `# Рекурсивные функции

Изучите основы рекурсии на классическом примере.

## Задача

Реализуйте \`factorial(n)\` с помощью рекурсии.

## Требования

- factorial(0) = 1
- factorial(n) = n * factorial(n-1)
- Для отрицательных чисел верните None

## Примеры

\`\`\`python
>>> factorial(5)
120  # 5 * 4 * 3 * 2 * 1

>>> factorial(0)
1

>>> factorial(-1)
None
\`\`\``,
			hint1: 'Базовый случай: if n == 0, return 1',
			hint2: 'Рекурсивный случай: return n * factorial(n - 1)',
			whyItMatters: `Рекурсия необходима для обхода деревьев и алгоритмов "разделяй и властвуй".

**Продакшен паттерн:**

\`\`\`python
def flatten_nested(data):
    """Рекурсивное выравнивание вложенных структур."""
    result = []
    if isinstance(data, list):
        for item in data:
            result.extend(flatten_nested(item))
    elif isinstance(data, dict):
        for value in data.values():
            result.extend(flatten_nested(value))
    else:
        result.append(data)
    return result
\`\`\`

**Практические преимущества:**
- Естественное решение для иерархических данных
- Основа для алгоритмов на деревьях/графах`,
		},
		uz: {
			title: 'Rekursiv funksiyalar',
			description: `# Rekursiv funksiyalar

Klassik misol bilan rekursiya asoslarini o'rganing.

## Vazifa

Rekursiya yordamida \`factorial(n)\` ni amalga oshiring.

## Talablar

- factorial(0) = 1
- factorial(n) = n * factorial(n-1)
- Manfiy sonlar uchun None qaytaring

## Misollar

\`\`\`python
>>> factorial(5)
120  # 5 * 4 * 3 * 2 * 1

>>> factorial(0)
1

>>> factorial(-1)
None
\`\`\``,
			hint1: "Asosiy holat: if n == 0, return 1",
			hint2: "Rekursiv holat: return n * factorial(n - 1)",
			whyItMatters: `Rekursiya daraxtlarni aylanish va "bo'l va hukmronlik qil" algoritmlari uchun zarur.

**Ishlab chiqarish patterni:**

\`\`\`python
def flatten_nested(data):
    """Ichma-ich tuzilmalarni rekursiv tekislash."""
    result = []
    if isinstance(data, list):
        for item in data:
            result.extend(flatten_nested(item))
    elif isinstance(data, dict):
        for value in data.values():
            result.extend(flatten_nested(value))
    else:
        result.append(data)
    return result
\`\`\`

**Amaliy foydalari:**
- Ierarxik ma'lumotlar uchun tabiiy yechim
- Daraxt/graf algoritmlarining asosi`,
		},
	},
};

export default task;
