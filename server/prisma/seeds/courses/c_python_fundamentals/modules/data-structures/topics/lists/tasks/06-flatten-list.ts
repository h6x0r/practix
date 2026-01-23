import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-flatten-list',
	title: 'Flatten Nested List',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'lists', 'recursion'],
	estimatedTime: '15m',
	isPremium: false,
	order: 6,

	description: `# Flatten Nested List

Transform a nested list into a flat, one-dimensional list.

## Task

Implement the function \`flatten(nested)\` that converts a nested list into a single flat list.

## Requirements

- Handle lists nested to any depth
- Preserve the order of elements
- Non-list elements should be added directly

## Examples

\`\`\`python
>>> flatten([1, [2, 3], [4, [5, 6]]])
[1, 2, 3, 4, 5, 6]

>>> flatten([[1, 2], [3, 4]])
[1, 2, 3, 4]

>>> flatten([1, 2, 3])
[1, 2, 3]

>>> flatten([])
[]
\`\`\``,

	initialCode: `def flatten(nested: list) -> list:
    """Flatten a nested list into a single-level list.

    Args:
        nested: A list that may contain other lists at any depth

    Returns:
        A flat list with all elements in order
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def flatten(nested: list) -> list:
    """Flatten a nested list into a single-level list.

    Args:
        nested: A list that may contain other lists at any depth

    Returns:
        A flat list with all elements in order
    """
    result = []

    for item in nested:
        # If item is a list, recursively flatten it
        if isinstance(item, list):
            # Extend result with flattened sublist
            result.extend(flatten(item))
        else:
            # Non-list items are added directly
            result.append(item)

    return result`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Mixed nesting levels"""
        self.assertEqual(flatten([1, [2, 3], [4, [5, 6]]]), [1, 2, 3, 4, 5, 6])

    def test_2(self):
        """Two-level nesting"""
        self.assertEqual(flatten([[1, 2], [3, 4]]), [1, 2, 3, 4])

    def test_3(self):
        """Already flat"""
        self.assertEqual(flatten([1, 2, 3]), [1, 2, 3])

    def test_4(self):
        """Empty list"""
        self.assertEqual(flatten([]), [])

    def test_5(self):
        """Deeply nested"""
        self.assertEqual(flatten([[[1]], [[2]], [[3]]]), [1, 2, 3])

    def test_6(self):
        """Empty sublists"""
        self.assertEqual(flatten([[], [1], [], [2], []]), [1, 2])

    def test_7(self):
        """Single element nested"""
        self.assertEqual(flatten([[[[[1]]]]]), [1])

    def test_8(self):
        """Mixed types"""
        self.assertEqual(flatten([1, ["a", "b"], [2, ["c"]]]), [1, "a", "b", 2, "c"])

    def test_9(self):
        """Numbers and empty lists"""
        self.assertEqual(flatten([1, [], 2, [], 3]), [1, 2, 3])

    def test_10(self):
        """Three levels"""
        self.assertEqual(flatten([1, [2, [3, [4]]]]), [1, 2, 3, 4])

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Check if an item is a list using isinstance(item, list). If it is, you need to process it further.',
	hint2: 'Use recursion: if the item is a list, call flatten() on it and extend your result with the returned list.',

	whyItMatters: `Flattening nested structures is common when processing hierarchical data.

**Production Pattern:**

\`\`\`python
from typing import Any
from collections.abc import Iterable

def flatten_any(data: Any, depth: int = -1) -> list:
    """Flatten any iterable to specified depth (-1 = unlimited)."""
    result = []

    def _flatten(item, current_depth):
        if current_depth == 0:
            result.append(item)
        elif isinstance(item, str):
            # Don't iterate over string characters
            result.append(item)
        elif isinstance(item, Iterable):
            for sub_item in item:
                _flatten(sub_item, current_depth - 1 if current_depth > 0 else -1)
        else:
            result.append(item)

    _flatten(data, depth)
    return result

def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dictionary with dot notation keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Example:
# flatten_dict({"a": {"b": 1, "c": 2}}) → {"a.b": 1, "a.c": 2}
\`\`\`

**Practical Benefits:**
- JSON/API response processing often needs flattening
- Database denormalization uses similar patterns
- Configuration parsing frequently requires flattening`,

	translations: {
		ru: {
			title: 'Сглаживание вложенного списка',
			description: `# Сглаживание вложенного списка

Преобразуйте вложенный список в плоский одномерный список.

## Задача

Реализуйте функцию \`flatten(nested)\`, которая преобразует вложенный список в единый плоский список.

## Требования

- Обработайте списки с любой глубиной вложенности
- Сохраните порядок элементов
- Элементы, не являющиеся списками, добавляются напрямую

## Примеры

\`\`\`python
>>> flatten([1, [2, 3], [4, [5, 6]]])
[1, 2, 3, 4, 5, 6]

>>> flatten([[1, 2], [3, 4]])
[1, 2, 3, 4]

>>> flatten([1, 2, 3])
[1, 2, 3]

>>> flatten([])
[]
\`\`\``,
			hint1: 'Проверьте, является ли элемент списком с помощью isinstance(item, list).',
			hint2: 'Используйте рекурсию: если элемент — список, вызовите flatten() для него.',
			whyItMatters: `Сглаживание вложенных структур часто используется при обработке иерархических данных.

**Продакшен паттерн:**

\`\`\`python
def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Сглаживание словаря с точечной нотацией ключей."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Пример:
# flatten_dict({"a": {"b": 1, "c": 2}}) → {"a.b": 1, "a.c": 2}
\`\`\`

**Практические преимущества:**
- Обработка JSON/API ответов часто требует сглаживания
- Денормализация БД использует похожие паттерны`,
		},
		uz: {
			title: "Ichma-ich ro'yxatni tekislash",
			description: `# Ichma-ich ro'yxatni tekislash

Ichma-ich ro'yxatni tekis, bir o'lchamli ro'yxatga aylantiring.

## Vazifa

Ichma-ich ro'yxatni yagona tekis ro'yxatga aylantiradigan \`flatten(nested)\` funksiyasini amalga oshiring.

## Talablar

- Har qanday chuqurlikdagi ro'yxatlarni ishlang
- Elementlar tartibini saqlang
- Ro'yxat bo'lmagan elementlar to'g'ridan-to'g'ri qo'shiladi

## Misollar

\`\`\`python
>>> flatten([1, [2, 3], [4, [5, 6]]])
[1, 2, 3, 4, 5, 6]

>>> flatten([[1, 2], [3, 4]])
[1, 2, 3, 4]

>>> flatten([1, 2, 3])
[1, 2, 3]

>>> flatten([])
[]
\`\`\``,
			hint1: "isinstance(item, list) yordamida element ro'yxat ekanligini tekshiring.",
			hint2: "Rekursiyadan foydalaning: agar element ro'yxat bo'lsa, unga flatten() chaqiring.",
			whyItMatters: `Ichma-ich tuzilmalarni tekislash ierarxik ma'lumotlarni qayta ishlashda keng tarqalgan.

**Ishlab chiqarish patterni:**

\`\`\`python
def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Lug'atni nuqtali belgi kalitlari bilan tekislash."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Misol:
# flatten_dict({"a": {"b": 1, "c": 2}}) → {"a.b": 1, "a.c": 2}
\`\`\`

**Amaliy foydalari:**
- JSON/API javoblarini qayta ishlash ko'pincha tekislashni talab qiladi
- Ma'lumotlar bazasi denormalizatsiyasi shunga o'xshash patternlardan foydalanadi`,
		},
	},
};

export default task;
