import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-map-filter',
	title: 'Map and Filter',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'functions', 'functional'],
	estimatedTime: '15m',
	isPremium: false,
	order: 9,

	description: `# Map and Filter

Use functional programming to transform and filter data.

## Task

Implement the function \`process_numbers(numbers, transform, condition)\` that:
1. Filters numbers that satisfy the condition
2. Transforms the filtered numbers using the transform function
3. Returns the result as a list

## Requirements

- \`transform\` is a function that takes a number and returns a transformed number
- \`condition\` is a function that takes a number and returns True/False
- First filter, then transform
- Return a list (not a filter/map object)

## Examples

\`\`\`python
>>> process_numbers([1, 2, 3, 4, 5], lambda x: x * 2, lambda x: x > 2)
[6, 8, 10]  # Filter: [3, 4, 5], then double each

>>> process_numbers([1, 2, 3, 4], lambda x: x ** 2, lambda x: x % 2 == 0)
[4, 16]  # Filter: [2, 4], then square each

>>> process_numbers([], lambda x: x, lambda x: True)
[]
\`\`\``,

	initialCode: `def process_numbers(numbers: list, transform, condition) -> list:
    """Filter numbers by condition, then transform them.

    Args:
        numbers: List of numbers to process
        transform: Function to apply to each number
        condition: Function that returns True for numbers to keep

    Returns:
        List of transformed numbers that passed the condition
    """
    # TODO: Implement using filter and map
    pass`,

	solutionCode: `def process_numbers(numbers: list, transform, condition) -> list:
    """Filter numbers by condition, then transform them.

    Args:
        numbers: List of numbers to process
        transform: Function to apply to each number
        condition: Function that returns True for numbers to keep

    Returns:
        List of transformed numbers that passed the condition
    """
    # Step 1: Filter - keep only numbers where condition returns True
    filtered = filter(condition, numbers)

    # Step 2: Transform - apply transform function to each filtered number
    transformed = map(transform, filtered)

    # Step 3: Convert to list and return
    return list(transformed)`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Filter > 2, then double"""
        result = process_numbers([1, 2, 3, 4, 5], lambda x: x * 2, lambda x: x > 2)
        self.assertEqual(result, [6, 8, 10])

    def test_2(self):
        """Filter even, then square"""
        result = process_numbers([1, 2, 3, 4], lambda x: x ** 2, lambda x: x % 2 == 0)
        self.assertEqual(result, [4, 16])

    def test_3(self):
        """Empty list"""
        result = process_numbers([], lambda x: x, lambda x: True)
        self.assertEqual(result, [])

    def test_4(self):
        """No elements pass filter"""
        result = process_numbers([1, 2, 3], lambda x: x, lambda x: x > 10)
        self.assertEqual(result, [])

    def test_5(self):
        """All elements pass filter"""
        result = process_numbers([1, 2, 3], lambda x: x + 1, lambda x: True)
        self.assertEqual(result, [2, 3, 4])

    def test_6(self):
        """Filter negative, then abs"""
        result = process_numbers([-3, -1, 0, 1, 3], lambda x: abs(x), lambda x: x < 0)
        self.assertEqual(result, [3, 1])

    def test_7(self):
        """Identity transform"""
        result = process_numbers([1, 2, 3, 4, 5], lambda x: x, lambda x: x % 2 == 1)
        self.assertEqual(result, [1, 3, 5])

    def test_8(self):
        """String conversion"""
        result = process_numbers([1, 2, 3], lambda x: str(x), lambda x: x > 1)
        self.assertEqual(result, ["2", "3"])

    def test_9(self):
        """Float operations"""
        result = process_numbers([1, 2, 3, 4], lambda x: x / 2, lambda x: x > 2)
        self.assertEqual(result, [1.5, 2.0])

    def test_10(self):
        """Returns list type"""
        result = process_numbers([1, 2, 3], lambda x: x, lambda x: True)
        self.assertIsInstance(result, list)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use filter(condition, numbers) to keep only matching numbers, then map(transform, filtered) to transform them.',
	hint2: 'filter() and map() return iterators. Wrap the final result in list() to get a list.',

	whyItMatters: `Functional programming with map/filter leads to cleaner, more declarative code that's easier to understand and parallelize.

**Production Pattern:**

\`\`\`python
from functools import reduce
from typing import Callable, TypeVar, Iterable

T = TypeVar('T')
R = TypeVar('R')

def pipeline(*functions):
    """Compose multiple functions into a pipeline."""
    def apply(data):
        result = data
        for func in functions:
            result = func(result)
        return result
    return apply

# Example usage
process_users = pipeline(
    lambda users: filter(lambda u: u["active"], users),
    lambda users: map(lambda u: u["email"], users),
    list
)

def transform_data(data: list[dict], transformations: list[Callable]) -> list:
    """Apply multiple transformations to data."""
    result = data
    for transform in transformations:
        result = list(map(transform, result))
    return result

# Using reduce for aggregation
def sum_if(numbers: list, condition: Callable[[int], bool]) -> int:
    """Sum numbers that satisfy condition."""
    return reduce(
        lambda acc, x: acc + x if condition(x) else acc,
        numbers,
        0
    )

# Modern Python: list comprehensions often preferred
# Equivalent to map + filter:
result = [x * 2 for x in numbers if x > 2]
\`\`\`

**Practical Benefits:**
- Data transformation pipelines
- ETL (Extract, Transform, Load) operations
- Parallel processing (map is inherently parallelizable)`,

	translations: {
		ru: {
			title: 'Map и Filter',
			description: `# Map и Filter

Используйте функциональное программирование для трансформации и фильтрации данных.

## Задача

Реализуйте функцию \`process_numbers(numbers, transform, condition)\`, которая:
1. Фильтрует числа, удовлетворяющие условию
2. Трансформирует отфильтрованные числа
3. Возвращает результат как список

## Требования

- \`transform\` — функция, которая принимает число и возвращает трансформированное
- \`condition\` — функция, которая принимает число и возвращает True/False
- Сначала фильтрация, потом трансформация
- Верните список (не объект filter/map)

## Примеры

\`\`\`python
>>> process_numbers([1, 2, 3, 4, 5], lambda x: x * 2, lambda x: x > 2)
[6, 8, 10]  # Фильтр: [3, 4, 5], затем удвоение

>>> process_numbers([1, 2, 3, 4], lambda x: x ** 2, lambda x: x % 2 == 0)
[4, 16]  # Фильтр: [2, 4], затем возведение в квадрат

>>> process_numbers([], lambda x: x, lambda x: True)
[]
\`\`\``,
			hint1: 'Используйте filter(condition, numbers) для отбора, затем map(transform, filtered) для трансформации.',
			hint2: 'filter() и map() возвращают итераторы. Оберните результат в list().',
			whyItMatters: `Функциональное программирование с map/filter делает код чище и декларативнее.

**Продакшен паттерн:**

\`\`\`python
from functools import reduce

def pipeline(*functions):
    """Композиция нескольких функций в конвейер."""
    def apply(data):
        result = data
        for func in functions:
            result = func(result)
        return result
    return apply

process_users = pipeline(
    lambda users: filter(lambda u: u["active"], users),
    lambda users: map(lambda u: u["email"], users),
    list
)

# Современный Python: list comprehensions часто предпочтительнее
result = [x * 2 for x in numbers if x > 2]
\`\`\`

**Практические преимущества:**
- Конвейеры трансформации данных
- ETL операции
- Параллельная обработка`,
		},
		uz: {
			title: 'Map va Filter',
			description: `# Map va Filter

Ma'lumotlarni o'zgartirish va filtrlash uchun funksional dasturlashdan foydalaning.

## Vazifa

Quyidagilarni bajaruvchi \`process_numbers(numbers, transform, condition)\` funksiyasini amalga oshiring:
1. Shartga mos keladigan sonlarni filtrlash
2. Filtrlangan sonlarni transform funksiyasi bilan o'zgartirish
3. Natijani ro'yxat sifatida qaytarish

## Talablar

- \`transform\` — sonni qabul qilib o'zgartirilgan sonni qaytaruvchi funksiya
- \`condition\` — sonni qabul qilib True/False qaytaruvchi funksiya
- Avval filtrlang, keyin o'zgartiring
- Ro'yxat qaytaring (filter/map ob'ekti emas)

## Misollar

\`\`\`python
>>> process_numbers([1, 2, 3, 4, 5], lambda x: x * 2, lambda x: x > 2)
[6, 8, 10]  # Filtr: [3, 4, 5], keyin ikki baravar

>>> process_numbers([1, 2, 3, 4], lambda x: x ** 2, lambda x: x % 2 == 0)
[4, 16]  # Filtr: [2, 4], keyin kvadrat

>>> process_numbers([], lambda x: x, lambda x: True)
[]
\`\`\``,
			hint1: "Mos sonlarni saqlash uchun filter(condition, numbers), keyin o'zgartirish uchun map(transform, filtered) ishlating.",
			hint2: "filter() va map() iteratorlar qaytaradi. Yakuniy natijani list() ga o'rang.",
			whyItMatters: `map/filter bilan funksional dasturlash kodni toza va deklarativ qiladi.

**Ishlab chiqarish patterni:**

\`\`\`python
from functools import reduce

def pipeline(*functions):
    """Bir nechta funksiyalarni konveyerga birlashtirish."""
    def apply(data):
        result = data
        for func in functions:
            result = func(result)
        return result
    return apply

process_users = pipeline(
    lambda users: filter(lambda u: u["active"], users),
    lambda users: map(lambda u: u["email"], users),
    list
)

# Zamonaviy Python: list comprehensions ko'pincha afzal
result = [x * 2 for x in numbers if x > 2]
\`\`\`

**Amaliy foydalari:**
- Ma'lumotlarni o'zgartirish konveyerlari
- ETL operatsiyalari
- Parallel qayta ishlash`,
		},
	},
};

export default task;
