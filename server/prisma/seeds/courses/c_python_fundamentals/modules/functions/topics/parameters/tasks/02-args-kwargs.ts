import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-args-kwargs',
	title: 'Variable Arguments',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'functions', 'args', 'kwargs'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,

	description: `# Variable Arguments (*args)

Learn to create functions that accept any number of arguments.

## Task

Implement the function \`average(*numbers)\` that calculates the average of any number of arguments.

## Requirements

- Accept any number of numeric arguments
- Return 0.0 if no arguments are provided
- Return the average as a float

## Examples

\`\`\`python
>>> average(1, 2, 3, 4, 5)
3.0

>>> average(10, 20)
15.0

>>> average()
0.0
\`\`\``,

	initialCode: `def average(*numbers) -> float:
    """Calculate the average of any number of arguments.

    Args:
        *numbers: Variable number of numeric arguments

    Returns:
        Average of all numbers, or 0.0 if none provided
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def average(*numbers) -> float:
    """Calculate the average of any number of arguments.

    Args:
        *numbers: Variable number of numeric arguments

    Returns:
        Average of all numbers, or 0.0 if none provided
    """
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Multiple numbers"""
        self.assertEqual(average(1, 2, 3, 4, 5), 3.0)

    def test_2(self):
        """Two numbers"""
        self.assertEqual(average(10, 20), 15.0)

    def test_3(self):
        """No arguments"""
        self.assertEqual(average(), 0.0)

    def test_4(self):
        """Single number"""
        self.assertEqual(average(42), 42.0)

    def test_5(self):
        """Floats"""
        self.assertAlmostEqual(average(1.5, 2.5, 3.0), 2.333, places=2)

    def test_6(self):
        """Negative numbers"""
        self.assertEqual(average(-10, 10), 0.0)

    def test_7(self):
        """All same"""
        self.assertEqual(average(5, 5, 5, 5), 5.0)

    def test_8(self):
        """Large numbers"""
        self.assertEqual(average(1000000, 2000000), 1500000.0)

    def test_9(self):
        """Mixed int and float"""
        self.assertEqual(average(1, 2.0, 3), 2.0)

    def test_10(self):
        """Many arguments"""
        self.assertEqual(average(*range(1, 11)), 5.5)

if __name__ == '__main__':
    unittest.main()`,

	hint1: '*args collects all positional arguments into a tuple called "numbers".',
	hint2: 'Check if numbers is empty before dividing. Use sum() and len().',

	whyItMatters: `Variable arguments enable flexible APIs that can handle different numbers of inputs.

**Production Pattern:**

\`\`\`python
def log(*messages, level: str = "INFO", separator: str = " "):
    """Flexible logging with multiple messages."""
    text = separator.join(str(m) for m in messages)
    print(f"[{level}] {text}")

def merge_results(*responses: dict) -> dict:
    """Merge any number of API responses."""
    result = {}
    for response in responses:
        result.update(response)
    return result

def validate(*validators, data: dict) -> list[str]:
    """Run multiple validators on data."""
    errors = []
    for validator in validators:
        error = validator(data)
        if error:
            errors.append(error)
    return errors
\`\`\`

**Practical Benefits:**
- Makes functions adaptable to varying inputs
- Cleaner than passing lists explicitly
- Essential for decorator functions`,

	translations: {
		ru: {
			title: 'Переменные аргументы',
			description: `# Переменные аргументы (*args)

Научитесь создавать функции, принимающие любое количество аргументов.

## Задача

Реализуйте функцию \`average(*numbers)\`, которая вычисляет среднее любого количества аргументов.

## Требования

- Принимайте любое количество числовых аргументов
- Возвращайте 0.0, если аргументов нет
- Верните среднее как float

## Примеры

\`\`\`python
>>> average(1, 2, 3, 4, 5)
3.0

>>> average(10, 20)
15.0

>>> average()
0.0
\`\`\``,
			hint1: '*args собирает все позиционные аргументы в кортеж "numbers".',
			hint2: 'Проверьте, пуст ли numbers перед делением. Используйте sum() и len().',
			whyItMatters: `Переменные аргументы позволяют создавать гибкие API.

**Продакшен паттерн:**

\`\`\`python
def log(*messages, level: str = "INFO", separator: str = " "):
    """Гибкое логирование с несколькими сообщениями."""
    text = separator.join(str(m) for m in messages)
    print(f"[{level}] {text}")
\`\`\`

**Практические преимущества:**
- Делает функции адаптируемыми к разному количеству входных данных
- Чище чем передача списков явно
- Необходимы для декораторов`,
		},
		uz: {
			title: "O'zgaruvchan argumentlar",
			description: `# O'zgaruvchan argumentlar (*args)

Istalgan sonli argumentlarni qabul qiluvchi funksiyalar yaratishni o'rganing.

## Vazifa

Istalgan sonli argumentlarning o'rtachasini hisoblovchi \`average(*numbers)\` funksiyasini amalga oshiring.

## Talablar

- Istalgan sonli raqamli argumentlarni qabul qiling
- Agar argument berilmasa, 0.0 qaytaring
- O'rtachani float sifatida qaytaring

## Misollar

\`\`\`python
>>> average(1, 2, 3, 4, 5)
3.0

>>> average(10, 20)
15.0

>>> average()
0.0
\`\`\``,
			hint1: '*args barcha pozitsion argumentlarni "numbers" korteji ga yig\'adi.',
			hint2: "Bo'lishdan oldin numbers bo'sh ekanligini tekshiring. sum() va len() dan foydalaning.",
			whyItMatters: `O'zgaruvchan argumentlar moslashuvchan API larni yaratish imkonini beradi.

**Ishlab chiqarish patterni:**

\`\`\`python
def log(*messages, level: str = "INFO", separator: str = " "):
    """Bir nechta xabarlar bilan moslashuvchan loglash."""
    text = separator.join(str(m) for m in messages)
    print(f"[{level}] {text}")
\`\`\`

**Amaliy foydalari:**
- Funksiyalarni turli kirish sonlariga moslashuvchan qiladi
- Ro'yxatlarni aniq uzatishdan toza
- Dekoratorlar uchun zarur`,
		},
	},
};

export default task;
