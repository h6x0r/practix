import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-generators',
	title: 'Generator Functions',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'generators', 'iterators'],
	estimatedTime: '15m',
	isPremium: false,
	order: 8,

	description: `# Generator Functions

Generators produce values one at a time using \`yield\`, saving memory.

## Task

Implement the generator function \`range_with_step(start, stop, step)\` that yields numbers like Python's range, but also works with floats.

## Requirements

- Yield numbers from \`start\` (inclusive) up to \`stop\` (exclusive)
- Increment by \`step\` each iteration
- Handle both positive and negative steps
- If step is 0, raise ValueError

## Examples

\`\`\`python
>>> list(range_with_step(0, 5, 1))
[0, 1, 2, 3, 4]

>>> list(range_with_step(0, 1, 0.2))
[0, 0.2, 0.4, 0.6, 0.8]

>>> list(range_with_step(5, 0, -1))
[5, 4, 3, 2, 1]
\`\`\``,

	initialCode: `def range_with_step(start, stop, step):
    """Generator that yields numbers from start to stop by step.

    Unlike built-in range(), this works with floats.

    Args:
        start: Starting value (inclusive)
        stop: Ending value (exclusive)
        step: Increment/decrement amount

    Yields:
        Numbers from start to stop

    Raises:
        ValueError: If step is 0
    """
    # TODO: Implement generator
    pass`,

	solutionCode: `def range_with_step(start, stop, step):
    """Generator that yields numbers from start to stop by step.

    Unlike built-in range(), this works with floats.

    Args:
        start: Starting value (inclusive)
        stop: Ending value (exclusive)
        step: Increment/decrement amount

    Yields:
        Numbers from start to stop

    Raises:
        ValueError: If step is 0
    """
    # Validate step
    if step == 0:
        raise ValueError("step cannot be zero")

    # Current position
    current = start

    # Handle positive step (going up)
    if step > 0:
        while current < stop:
            yield current
            current += step

    # Handle negative step (going down)
    else:
        while current > stop:
            yield current
            current += step  # step is negative, so this decrements`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic integer range"""
        self.assertEqual(list(range_with_step(0, 5, 1)), [0, 1, 2, 3, 4])

    def test_2(self):
        """Float step"""
        result = list(range_with_step(0, 1, 0.2))
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0], 0, places=5)
        self.assertAlmostEqual(result[1], 0.2, places=5)

    def test_3(self):
        """Negative step"""
        self.assertEqual(list(range_with_step(5, 0, -1)), [5, 4, 3, 2, 1])

    def test_4(self):
        """Zero step raises ValueError"""
        with self.assertRaises(ValueError):
            list(range_with_step(0, 5, 0))

    def test_5(self):
        """Empty range (start >= stop with positive step)"""
        self.assertEqual(list(range_with_step(5, 0, 1)), [])

    def test_6(self):
        """Start equals stop"""
        self.assertEqual(list(range_with_step(5, 5, 1)), [])

    def test_7(self):
        """Large step"""
        self.assertEqual(list(range_with_step(0, 10, 3)), [0, 3, 6, 9])

    def test_8(self):
        """Negative range with negative step"""
        self.assertEqual(list(range_with_step(-1, -5, -1)), [-1, -2, -3, -4])

    def test_9(self):
        """Float start and stop"""
        result = list(range_with_step(0.5, 2.5, 0.5))
        self.assertEqual(len(result), 4)

    def test_10(self):
        """Is a generator"""
        gen = range_with_step(0, 3, 1)
        self.assertEqual(next(gen), 0)
        self.assertEqual(next(gen), 1)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use "yield" instead of "return" to produce values one at a time. Use a while loop to control iteration.',
	hint2: 'Check if step is positive or negative to determine the loop condition (current < stop or current > stop).',

	whyItMatters: `Generators are memory-efficient for processing large datasets and implementing custom iteration.

**Production Pattern:**

\`\`\`python
def read_large_file(file_path: str, chunk_size: int = 1024):
    """Memory-efficient file reading generator."""
    with open(file_path, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def paginate_api(url: str, page_size: int = 100):
    """Generator for paginated API responses."""
    import requests
    page = 1
    while True:
        response = requests.get(f"{url}?page={page}&size={page_size}")
        data = response.json()
        if not data["items"]:
            break
        yield from data["items"]
        page += 1

def infinite_sequence():
    """Infinite generator - use with islice or break."""
    num = 0
    while True:
        yield num
        num += 1

# Generator expression (like list comprehension but lazy)
squares = (x**2 for x in range(1000000))  # No memory used yet
first_ten = [next(squares) for _ in range(10)]

# Using itertools for advanced iteration
from itertools import islice, chain, cycle
first_five = list(islice(infinite_sequence(), 5))
\`\`\`

**Practical Benefits:**
- Process files larger than available memory
- Infinite sequences (like event streams)
- Pipeline data processing with generator chaining`,

	translations: {
		ru: {
			title: 'Функции-генераторы',
			description: `# Функции-генераторы

Генераторы производят значения по одному с помощью \`yield\`, экономя память.

## Задача

Реализуйте генератор \`range_with_step(start, stop, step)\`, который выдаёт числа как range в Python, но работает и с float.

## Требования

- Выдавайте числа от \`start\` (включительно) до \`stop\` (исключительно)
- Увеличивайте на \`step\` каждую итерацию
- Обработайте положительные и отрицательные шаги
- Если step равен 0, вызовите ValueError

## Примеры

\`\`\`python
>>> list(range_with_step(0, 5, 1))
[0, 1, 2, 3, 4]

>>> list(range_with_step(0, 1, 0.2))
[0, 0.2, 0.4, 0.6, 0.8]

>>> list(range_with_step(5, 0, -1))
[5, 4, 3, 2, 1]
\`\`\``,
			hint1: 'Используйте "yield" вместо "return" для выдачи значений по одному. Используйте цикл while.',
			hint2: 'Проверьте, положительный или отрицательный step для определения условия цикла.',
			whyItMatters: `Генераторы эффективны по памяти для обработки больших данных.

**Продакшен паттерн:**

\`\`\`python
def read_large_file(file_path: str, chunk_size: int = 1024):
    """Эффективное по памяти чтение файла."""
    with open(file_path, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def paginate_api(url: str, page_size: int = 100):
    """Генератор для пагинированных API ответов."""
    import requests
    page = 1
    while True:
        response = requests.get(f"{url}?page={page}&size={page_size}")
        data = response.json()
        if not data["items"]:
            break
        yield from data["items"]
        page += 1
\`\`\`

**Практические преимущества:**
- Обработка файлов больше доступной памяти
- Бесконечные последовательности (потоки событий)
- Конвейерная обработка данных`,
		},
		uz: {
			title: 'Generator funksiyalar',
			description: `# Generator funksiyalar

Generatorlar \`yield\` yordamida qiymatlarni bittadan ishlab chiqaradi va xotirani tejaydi.

## Vazifa

Python range kabi, lekin float bilan ham ishlaydigan \`range_with_step(start, stop, step)\` generatorini amalga oshiring.

## Talablar

- \`start\` (kiruvchi) dan \`stop\` (chiqaruvchi) gacha sonlarni bering
- Har bir iteratsiyada \`step\` ga oshiring
- Musbat va manfiy qadamlarni ishlang
- Agar step 0 bo'lsa, ValueError ko'taring

## Misollar

\`\`\`python
>>> list(range_with_step(0, 5, 1))
[0, 1, 2, 3, 4]

>>> list(range_with_step(0, 1, 0.2))
[0, 0.2, 0.4, 0.6, 0.8]

>>> list(range_with_step(5, 0, -1))
[5, 4, 3, 2, 1]
\`\`\``,
			hint1: 'Qiymatlarni bittadan berish uchun "return" o\'rniga "yield" ishlating. while tsiklidan foydalaning.',
			hint2: "Tsikl shartini aniqlash uchun step musbat yoki manfiy ekanligini tekshiring.",
			whyItMatters: `Generatorlar katta ma'lumotlarni qayta ishlash uchun xotira bo'yicha samarali.

**Ishlab chiqarish patterni:**

\`\`\`python
def read_large_file(file_path: str, chunk_size: int = 1024):
    """Xotirani tejaydigan fayl o'qish."""
    with open(file_path, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def paginate_api(url: str, page_size: int = 100):
    """Sahifalangan API javoblari uchun generator."""
    import requests
    page = 1
    while True:
        response = requests.get(f"{url}?page={page}&size={page_size}")
        data = response.json()
        if not data["items"]:
            break
        yield from data["items"]
        page += 1
\`\`\`

**Amaliy foydalari:**
- Mavjud xotiradan kattaroq fayllarni qayta ishlash
- Cheksiz ketma-ketliklar (voqea oqimlari)
- Konveyer ma'lumotlarni qayta ishlash`,
		},
	},
};

export default task;
