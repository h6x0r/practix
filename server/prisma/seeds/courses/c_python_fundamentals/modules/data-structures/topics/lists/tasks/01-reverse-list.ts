import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-reverse-list',
	title: 'Reverse a List',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'lists', 'slicing'],
	estimatedTime: '10m',
	isPremium: false,
	order: 1,

	description: `# Reverse a List

Learn different ways to reverse a list in Python.

## Task

Implement the function \`reverse_list(items)\` that returns a new list with elements in reverse order.

## Requirements

- Return a NEW list (don't modify the original)
- Handle empty lists

## Examples

\`\`\`python
>>> reverse_list([1, 2, 3, 4, 5])
[5, 4, 3, 2, 1]

>>> reverse_list(["a", "b", "c"])
["c", "b", "a"]

>>> reverse_list([])
[]
\`\`\``,

	initialCode: `def reverse_list(items: list) -> list:
    """Return a new list with elements in reverse order.

    Args:
        items: Input list

    Returns:
        New list with reversed elements
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def reverse_list(items: list) -> list:
    """Return a new list with elements in reverse order.

    Args:
        items: Input list

    Returns:
        New list with reversed elements
    """
    # Using slice notation - creates a new list
    return items[::-1]`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Reverse numbers"""
        self.assertEqual(reverse_list([1, 2, 3, 4, 5]), [5, 4, 3, 2, 1])

    def test_2(self):
        """Reverse strings"""
        self.assertEqual(reverse_list(["a", "b", "c"]), ["c", "b", "a"])

    def test_3(self):
        """Empty list"""
        self.assertEqual(reverse_list([]), [])

    def test_4(self):
        """Single element"""
        self.assertEqual(reverse_list([1]), [1])

    def test_5(self):
        """Two elements"""
        self.assertEqual(reverse_list([1, 2]), [2, 1])

    def test_6(self):
        """Mixed types"""
        self.assertEqual(reverse_list([1, "a", 2, "b"]), ["b", 2, "a", 1])

    def test_7(self):
        """Original not modified"""
        original = [1, 2, 3]
        reverse_list(original)
        self.assertEqual(original, [1, 2, 3])

    def test_8(self):
        """Nested lists"""
        self.assertEqual(reverse_list([[1], [2], [3]]), [[3], [2], [1]])

    def test_9(self):
        """Boolean values"""
        self.assertEqual(reverse_list([True, False, True]), [True, False, True])

    def test_10(self):
        """Large list"""
        self.assertEqual(reverse_list(list(range(100)))[0], 99)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Python has a shortcut for reversing: items[::-1]',
	hint2: 'Alternatively, use list(reversed(items)) or items.copy() then .reverse()',

	whyItMatters: `List reversal is common in algorithms and data processing.

**Production Pattern:**

\`\`\`python
def get_recent_logs(logs: list[str], count: int = 10) -> list[str]:
    """Get most recent logs (newest first)."""
    return logs[::-1][:count]

def undo_stack(actions: list) -> list:
    """Reverse action order for undo functionality."""
    return actions[::-1]
\`\`\`

**Practical Benefits:**
- Slice [::-1] is the most Pythonic way
- Creates a new list (immutable approach)
- Essential for stack-like operations`,

	translations: {
		ru: {
			title: 'Разворот списка',
			description: `# Разворот списка

Изучите разные способы развернуть список в Python.

## Задача

Реализуйте функцию \`reverse_list(items)\`, которая возвращает новый список с элементами в обратном порядке.

## Требования

- Верните НОВЫЙ список (не изменяйте оригинал)
- Обработайте пустые списки

## Примеры

\`\`\`python
>>> reverse_list([1, 2, 3, 4, 5])
[5, 4, 3, 2, 1]

>>> reverse_list(["a", "b", "c"])
["c", "b", "a"]

>>> reverse_list([])
[]
\`\`\``,
			hint1: 'Python имеет сокращение для разворота: items[::-1]',
			hint2: 'Альтернатива: list(reversed(items)) или items.copy().reverse()',
			whyItMatters: `Разворот списка часто используется в алгоритмах.

**Продакшен паттерн:**

\`\`\`python
def get_recent_logs(logs: list[str], count: int = 10) -> list[str]:
    """Получить последние логи (новые первыми)."""
    return logs[::-1][:count]
\`\`\`

**Практические преимущества:**
- Срез [::-1] — самый питонический способ
- Создаёт новый список (иммутабельный подход)`,
		},
		uz: {
			title: "Ro'yxatni teskari aylantirish",
			description: `# Ro'yxatni teskari aylantirish

Python da ro'yxatni teskari aylantirishning turli usullarini o'rganing.

## Vazifa

Elementlari teskari tartibda bo'lgan yangi ro'yxat qaytaruvchi \`reverse_list(items)\` funksiyasini amalga oshiring.

## Talablar

- YANGI ro'yxat qaytaring (aslini o'zgartirmang)
- Bo'sh ro'yxatlarni ham ishlov bering

## Misollar

\`\`\`python
>>> reverse_list([1, 2, 3, 4, 5])
[5, 4, 3, 2, 1]

>>> reverse_list(["a", "b", "c"])
["c", "b", "a"]

>>> reverse_list([])
[]
\`\`\``,
			hint1: "Python da teskari aylantirish uchun qisqartma bor: items[::-1]",
			hint2: "Alternativa: list(reversed(items)) yoki items.copy().reverse()",
			whyItMatters: `Ro'yxatni teskari aylantirish algoritmlarda tez-tez ishlatiladi.

**Ishlab chiqarish patterni:**

\`\`\`python
def get_recent_logs(logs: list[str], count: int = 10) -> list[str]:
    """Oxirgi loglarni olish (yangilari birinchi)."""
    return logs[::-1][:count]
\`\`\`

**Amaliy foydalari:**
- [::-1] kesimi eng Pythonik usul
- Yangi ro'yxat yaratadi (o'zgarmas yondashuv)`,
		},
	},
};

export default task;
