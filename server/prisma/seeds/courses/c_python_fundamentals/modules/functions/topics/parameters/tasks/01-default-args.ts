import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-default-args',
	title: 'Default Arguments',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'functions', 'arguments'],
	estimatedTime: '10m',
	isPremium: false,
	order: 1,

	description: `# Default Arguments

Learn to use default parameter values in functions.

## Task

Implement the function \`power(base, exponent=2)\` that raises a number to a power.

## Requirements

- If exponent is not provided, default to 2 (square)
- Return the result as a float

## Examples

\`\`\`python
>>> power(3)
9.0  # 3² = 9

>>> power(2, 3)
8.0  # 2³ = 8

>>> power(5, 0)
1.0  # 5⁰ = 1
\`\`\``,

	initialCode: `def power(base: float, exponent: int = 2) -> float:
    """Raise base to the power of exponent.

    Args:
        base: The base number
        exponent: The power to raise to (default: 2)

    Returns:
        base raised to exponent as float
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def power(base: float, exponent: int = 2) -> float:
    """Raise base to the power of exponent.

    Args:
        base: The base number
        exponent: The power to raise to (default: 2)

    Returns:
        base raised to exponent as float
    """
    return float(base ** exponent)`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Default exponent (square)"""
        self.assertEqual(power(3), 9.0)

    def test_2(self):
        """Custom exponent"""
        self.assertEqual(power(2, 3), 8.0)

    def test_3(self):
        """Zero exponent"""
        self.assertEqual(power(5, 0), 1.0)

    def test_4(self):
        """Negative base"""
        self.assertEqual(power(-2, 3), -8.0)

    def test_5(self):
        """Exponent 1"""
        self.assertEqual(power(7, 1), 7.0)

    def test_6(self):
        """Float base"""
        self.assertAlmostEqual(power(2.5, 2), 6.25)

    def test_7(self):
        """Large exponent"""
        self.assertEqual(power(2, 10), 1024.0)

    def test_8(self):
        """Zero base"""
        self.assertEqual(power(0, 5), 0.0)

    def test_9(self):
        """Base 1"""
        self.assertEqual(power(1, 100), 1.0)

    def test_10(self):
        """Negative exponent"""
        self.assertEqual(power(2, -1), 0.5)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use the ** operator for exponentiation: base ** exponent',
	hint2: 'Wrap result in float() to ensure float return type.',

	whyItMatters: `Default arguments make functions more flexible and easier to use.

**Production Pattern:**

\`\`\`python
def fetch_data(url: str, timeout: int = 30, retries: int = 3, verify_ssl: bool = True) -> dict:
    """Fetch data with sensible defaults."""
    for attempt in range(retries):
        try:
            return requests.get(url, timeout=timeout, verify=verify_ssl).json()
        except RequestException:
            if attempt == retries - 1:
                raise

def format_number(n: float, decimals: int = 2, thousands_sep: str = ",") -> str:
    """Format number with customizable options."""
    return f"{n:,.{decimals}f}".replace(",", thousands_sep)
\`\`\`

**Practical Benefits:**
- Reduces boilerplate in calling code
- Provides sensible defaults for common cases
- Makes API backwards-compatible when adding parameters`,

	translations: {
		ru: {
			title: 'Аргументы по умолчанию',
			description: `# Аргументы по умолчанию

Научитесь использовать значения параметров по умолчанию.

## Задача

Реализуйте функцию \`power(base, exponent=2)\`, которая возводит число в степень.

## Требования

- Если exponent не указан, по умолчанию 2 (квадрат)
- Верните результат как float

## Примеры

\`\`\`python
>>> power(3)
9.0  # 3² = 9

>>> power(2, 3)
8.0  # 2³ = 8

>>> power(5, 0)
1.0  # 5⁰ = 1
\`\`\``,
			hint1: 'Используйте оператор ** для возведения в степень',
			hint2: 'Оберните результат в float() для гарантии типа.',
			whyItMatters: `Аргументы по умолчанию делают функции гибче и удобнее.

**Продакшен паттерн:**

\`\`\`python
def fetch_data(url: str, timeout: int = 30, retries: int = 3) -> dict:
    """Получение данных с разумными дефолтами."""
    for attempt in range(retries):
        try:
            return requests.get(url, timeout=timeout).json()
        except RequestException:
            if attempt == retries - 1:
                raise
\`\`\`

**Практические преимущества:**
- Уменьшает шаблонный код при вызовах
- Предоставляет разумные дефолты
- Обеспечивает обратную совместимость API`,
		},
		uz: {
			title: 'Standart argumentlar',
			description: `# Standart argumentlar

Funksiyalarda standart parametr qiymatlaridan foydalanishni o'rganing.

## Vazifa

Sonni darajaga ko'taruvchi \`power(base, exponent=2)\` funksiyasini amalga oshiring.

## Talablar

- Agar exponent berilmasa, standart 2 (kvadrat)
- Natijani float sifatida qaytaring

## Misollar

\`\`\`python
>>> power(3)
9.0  # 3² = 9

>>> power(2, 3)
8.0  # 2³ = 8

>>> power(5, 0)
1.0  # 5⁰ = 1
\`\`\``,
			hint1: "Darajaga ko'tarish uchun ** operatoridan foydalaning",
			hint2: "Float turini kafolatlash uchun natijani float() ga o'rang.",
			whyItMatters: `Standart argumentlar funksiyalarni moslashuvchan va qulay qiladi.

**Ishlab chiqarish patterni:**

\`\`\`python
def fetch_data(url: str, timeout: int = 30, retries: int = 3) -> dict:
    """Oqilona standartlar bilan ma'lumotlarni olish."""
    for attempt in range(retries):
        try:
            return requests.get(url, timeout=timeout).json()
        except RequestException:
            if attempt == retries - 1:
                raise
\`\`\`

**Amaliy foydalari:**
- Chaqirish kodidagi shablonni kamaytiradi
- Umumiy holatlar uchun oqilona standartlarni taqdim etadi
- Parametrlar qo'shilganda API orqaga mosligini ta'minlaydi`,
		},
	},
};

export default task;
