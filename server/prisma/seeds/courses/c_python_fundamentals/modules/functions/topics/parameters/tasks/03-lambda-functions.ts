import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-lambda-functions',
	title: 'Lambda Functions',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'functions', 'lambda'],
	estimatedTime: '10m',
	isPremium: false,
	order: 3,

	description: `# Lambda Functions

Learn to use anonymous (lambda) functions for simple operations.

## Task

Implement the function \`apply_operation(numbers, operation)\` that applies a lambda function to each number.

## Requirements

- Use \`map()\` with the provided operation
- Return a list of results
- Operation is a function (can be lambda)

## Examples

\`\`\`python
>>> apply_operation([1, 2, 3], lambda x: x * 2)
[2, 4, 6]

>>> apply_operation([1, 4, 9], lambda x: x ** 0.5)
[1.0, 2.0, 3.0]
\`\`\``,

	initialCode: `def apply_operation(numbers: list, operation) -> list:
    """Apply an operation to each number in the list.

    Args:
        numbers: List of numbers
        operation: Function to apply to each number

    Returns:
        List of results after applying operation
    """
    # TODO: Use map() to apply operation
    pass`,

	solutionCode: `def apply_operation(numbers: list, operation) -> list:
    """Apply an operation to each number in the list.

    Args:
        numbers: List of numbers
        operation: Function to apply to each number

    Returns:
        List of results after applying operation
    """
    return list(map(operation, numbers))`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Double each number"""
        self.assertEqual(apply_operation([1, 2, 3], lambda x: x * 2), [2, 4, 6])

    def test_2(self):
        """Square root"""
        self.assertEqual(apply_operation([1, 4, 9], lambda x: x ** 0.5), [1.0, 2.0, 3.0])

    def test_3(self):
        """Empty list"""
        self.assertEqual(apply_operation([], lambda x: x), [])

    def test_4(self):
        """Identity function"""
        self.assertEqual(apply_operation([5, 10], lambda x: x), [5, 10])

    def test_5(self):
        """Square"""
        self.assertEqual(apply_operation([1, 2, 3], lambda x: x ** 2), [1, 4, 9])

    def test_6(self):
        """Add constant"""
        self.assertEqual(apply_operation([1, 2, 3], lambda x: x + 10), [11, 12, 13])

    def test_7(self):
        """Negative"""
        self.assertEqual(apply_operation([1, -2, 3], lambda x: -x), [-1, 2, -3])

    def test_8(self):
        """String conversion"""
        self.assertEqual(apply_operation([1, 2, 3], lambda x: str(x)), ["1", "2", "3"])

    def test_9(self):
        """Boolean conversion"""
        self.assertEqual(apply_operation([0, 1, 2], lambda x: bool(x)), [False, True, True])

    def test_10(self):
        """Complex operation"""
        self.assertEqual(apply_operation([1, 2, 3], lambda x: x * x + 1), [2, 5, 10])

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'map(function, iterable) applies function to each element.',
	hint2: 'Wrap map() result in list() to get a list back.',

	whyItMatters: `Lambda functions are perfect for short, one-time operations in functional programming.

**Production Pattern:**

\`\`\`python
# Sorting with custom key
users = [{"name": "Bob", "age": 25}, {"name": "Alice", "age": 30}]
sorted_users = sorted(users, key=lambda u: u["age"])

# Filtering with condition
active_users = list(filter(lambda u: u.get("active", False), users))

# Data transformation pipeline
pipeline = [
    lambda x: x.strip(),
    lambda x: x.lower(),
    lambda x: x.replace(" ", "_"),
]
result = text
for transform in pipeline:
    result = transform(result)
\`\`\`

**Practical Benefits:**
- Concise for simple transformations
- Works well with map(), filter(), sorted()
- No need to define named functions for one-time use`,

	translations: {
		ru: {
			title: 'Lambda-функции',
			description: `# Lambda-функции

Научитесь использовать анонимные (lambda) функции для простых операций.

## Задача

Реализуйте функцию \`apply_operation(numbers, operation)\`, которая применяет lambda к каждому числу.

## Требования

- Используйте \`map()\` с переданной операцией
- Верните список результатов

## Примеры

\`\`\`python
>>> apply_operation([1, 2, 3], lambda x: x * 2)
[2, 4, 6]

>>> apply_operation([1, 4, 9], lambda x: x ** 0.5)
[1.0, 2.0, 3.0]
\`\`\``,
			hint1: 'map(function, iterable) применяет функцию к каждому элементу.',
			hint2: 'Оберните результат map() в list().',
			whyItMatters: `Lambda идеальны для коротких одноразовых операций.

**Продакшен паттерн:**

\`\`\`python
# Сортировка с пользовательским ключом
users = [{"name": "Bob", "age": 25}, {"name": "Alice", "age": 30}]
sorted_users = sorted(users, key=lambda u: u["age"])
\`\`\`

**Практические преимущества:**
- Компактны для простых трансформаций
- Хорошо работают с map(), filter(), sorted()`,
		},
		uz: {
			title: 'Lambda funksiyalari',
			description: `# Lambda funksiyalari

Oddiy operatsiyalar uchun anonim (lambda) funksiyalardan foydalanishni o'rganing.

## Vazifa

Har bir raqamga lambda ni qo'llovchi \`apply_operation(numbers, operation)\` funksiyasini amalga oshiring.

## Talablar

- Berilgan operatsiya bilan \`map()\` dan foydalaning
- Natijalar ro'yxatini qaytaring

## Misollar

\`\`\`python
>>> apply_operation([1, 2, 3], lambda x: x * 2)
[2, 4, 6]

>>> apply_operation([1, 4, 9], lambda x: x ** 0.5)
[1.0, 2.0, 3.0]
\`\`\``,
			hint1: "map(function, iterable) funksiyani har bir elementga qo'llaydi.",
			hint2: "map() natijasini list() ga o'rang.",
			whyItMatters: `Lambda qisqa bir martalik operatsiyalar uchun ideal.

**Ishlab chiqarish patterni:**

\`\`\`python
# Maxsus kalit bilan saralash
users = [{"name": "Bob", "age": 25}, {"name": "Alice", "age": 30}]
sorted_users = sorted(users, key=lambda u: u["age"])
\`\`\`

**Amaliy foydalari:**
- Oddiy o'zgartirishlar uchun ixcham
- map(), filter(), sorted() bilan yaxshi ishlaydi`,
		},
	},
};

export default task;
