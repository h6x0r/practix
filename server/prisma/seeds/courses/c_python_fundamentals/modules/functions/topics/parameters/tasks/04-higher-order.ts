import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-higher-order',
	title: 'Higher-Order Functions',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'functions', 'filter'],
	estimatedTime: '15m',
	isPremium: false,
	order: 4,

	description: `# Higher-Order Functions

Learn to use functions as arguments (filter).

## Task

Implement \`filter_by(items, predicate)\` that filters items using a predicate function.

## Requirements

- Return items where predicate(item) is True
- Preserve original order

## Examples

\`\`\`python
>>> filter_by([1, 2, 3, 4, 5], lambda x: x > 3)
[4, 5]

>>> filter_by(["a", "bb", "ccc"], lambda s: len(s) > 1)
["bb", "ccc"]
\`\`\``,

	initialCode: `def filter_by(items: list, predicate) -> list:
    """Filter items using a predicate function.

    Args:
        items: List to filter
        predicate: Function returning True for items to keep

    Returns:
        Filtered list
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def filter_by(items: list, predicate) -> list:
    """Filter items using a predicate function.

    Args:
        items: List to filter
        predicate: Function returning True for items to keep

    Returns:
        Filtered list
    """
    return [item for item in items if predicate(item)]`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Filter greater than"""
        self.assertEqual(filter_by([1, 2, 3, 4, 5], lambda x: x > 3), [4, 5])

    def test_2(self):
        """Filter by length"""
        self.assertEqual(filter_by(["a", "bb", "ccc"], lambda s: len(s) > 1), ["bb", "ccc"])

    def test_3(self):
        """Empty result"""
        self.assertEqual(filter_by([1, 2, 3], lambda x: x > 10), [])

    def test_4(self):
        """All pass"""
        self.assertEqual(filter_by([1, 2, 3], lambda x: x > 0), [1, 2, 3])

    def test_5(self):
        """Filter even"""
        self.assertEqual(filter_by([1, 2, 3, 4], lambda x: x % 2 == 0), [2, 4])

    def test_6(self):
        """Empty input"""
        self.assertEqual(filter_by([], lambda x: True), [])

    def test_7(self):
        """Filter by type"""
        self.assertEqual(filter_by([1, "a", 2, "b"], lambda x: isinstance(x, int)), [1, 2])

    def test_8(self):
        """Filter truthy"""
        self.assertEqual(filter_by([0, 1, "", "a", None], lambda x: x), [1, "a"])

    def test_9(self):
        """Filter negative"""
        self.assertEqual(filter_by([-2, -1, 0, 1, 2], lambda x: x < 0), [-2, -1])

    def test_10(self):
        """Complex predicate"""
        self.assertEqual(filter_by([1, 2, 3, 4, 5, 6], lambda x: x % 2 == 0 and x > 2), [4, 6])

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use list comprehension with if: [item for item in items if predicate(item)]',
	hint2: 'Or use the built-in filter(): list(filter(predicate, items))',

	whyItMatters: `Higher-order functions are the foundation of functional programming.

**Production Pattern:**

\`\`\`python
def create_validator(min_val: int, max_val: int):
    """Return a validator function."""
    return lambda x: min_val <= x <= max_val

def process_with_filters(data: list, *filters):
    """Apply multiple filter functions."""
    result = data
    for f in filters:
        result = [x for x in result if f(x)]
    return result

# Usage
is_valid_age = create_validator(18, 120)
adults = filter_by(users, lambda u: is_valid_age(u["age"]))
\`\`\`

**Practical Benefits:**
- Enables flexible, reusable filtering logic
- Foundation for data processing pipelines
- Key concept in modern Python programming`,

	translations: {
		ru: {
			title: 'Функции высшего порядка',
			description: `# Функции высшего порядка

Научитесь использовать функции как аргументы (filter).

## Задача

Реализуйте \`filter_by(items, predicate)\`, которая фильтрует элементы с помощью функции-предиката.

## Требования

- Верните элементы, для которых predicate(item) равен True
- Сохраните порядок

## Примеры

\`\`\`python
>>> filter_by([1, 2, 3, 4, 5], lambda x: x > 3)
[4, 5]

>>> filter_by(["a", "bb", "ccc"], lambda s: len(s) > 1)
["bb", "ccc"]
\`\`\``,
			hint1: 'Используйте list comprehension с if',
			hint2: 'Или встроенный filter(): list(filter(predicate, items))',
			whyItMatters: `Функции высшего порядка — основа функционального программирования.

**Продакшен паттерн:**

\`\`\`python
def create_validator(min_val: int, max_val: int):
    """Возвращает функцию-валидатор."""
    return lambda x: min_val <= x <= max_val

is_valid_age = create_validator(18, 120)
adults = filter_by(users, lambda u: is_valid_age(u["age"]))
\`\`\`

**Практические преимущества:**
- Гибкая, переиспользуемая логика фильтрации
- Основа для пайплайнов обработки данных`,
		},
		uz: {
			title: "Yuqori tartibli funksiyalar",
			description: `# Yuqori tartibli funksiyalar

Funksiyalarni argument sifatida ishlatishni (filter) o'rganing.

## Vazifa

Elementlarni predikat funksiyasi yordamida filtrlovchi \`filter_by(items, predicate)\` ni amalga oshiring.

## Talablar

- predicate(item) True bo'lgan elementlarni qaytaring
- Tartibni saqlang

## Misollar

\`\`\`python
>>> filter_by([1, 2, 3, 4, 5], lambda x: x > 3)
[4, 5]

>>> filter_by(["a", "bb", "ccc"], lambda s: len(s) > 1)
["bb", "ccc"]
\`\`\``,
			hint1: "if bilan list comprehension dan foydalaning",
			hint2: "Yoki o'rnatilgan filter(): list(filter(predicate, items))",
			whyItMatters: `Yuqori tartibli funksiyalar funksional dasturlashning asosidir.

**Ishlab chiqarish patterni:**

\`\`\`python
def create_validator(min_val: int, max_val: int):
    """Validator funksiyasini qaytaradi."""
    return lambda x: min_val <= x <= max_val

is_valid_age = create_validator(18, 120)
adults = filter_by(users, lambda u: is_valid_age(u["age"]))
\`\`\`

**Amaliy foydalari:**
- Moslashuvchan, qayta ishlatiladigan filtrlash mantig'i
- Ma'lumotlarni qayta ishlash konveyerlarining asosi`,
		},
	},
};

export default task;
