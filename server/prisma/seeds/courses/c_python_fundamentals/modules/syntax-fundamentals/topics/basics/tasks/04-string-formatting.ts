import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-string-formatting',
	title: 'String Formatting',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'strings', 'f-strings'],
	estimatedTime: '10m',
	isPremium: false,
	order: 4,

	description: `# String Formatting

Python offers several ways to format strings. The modern approach uses f-strings (formatted string literals).

## Task

Implement the function \`format_person(name, age, city)\` that returns a formatted description.

## Requirements

- Return: \`"{name} is {age} years old and lives in {city}."\`
- Age should be displayed as an integer (no decimals)

## Examples

\`\`\`python
>>> format_person("Alice", 25, "New York")
"Alice is 25 years old and lives in New York."

>>> format_person("Bob", 30.5, "London")
"Bob is 30 years old and lives in London."
\`\`\``,

	initialCode: `def format_person(name: str, age: float, city: str) -> str:
    """Format a person's information into a readable sentence.

    Args:
        name: The person's name
        age: The person's age (may be float)
        city: The city where they live

    Returns:
        A formatted description string
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def format_person(name: str, age: float, city: str) -> str:
    """Format a person's information into a readable sentence.

    Args:
        name: The person's name
        age: The person's age (may be float)
        city: The city where they live

    Returns:
        A formatted description string
    """
    return f"{name} is {int(age)} years old and lives in {city}."`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic formatting"""
        self.assertEqual(format_person("Alice", 25, "New York"), "Alice is 25 years old and lives in New York.")

    def test_2(self):
        """Float age should be truncated"""
        self.assertEqual(format_person("Bob", 30.9, "London"), "Bob is 30 years old and lives in London.")

    def test_3(self):
        """Zero age"""
        self.assertEqual(format_person("Baby", 0, "Tokyo"), "Baby is 0 years old and lives in Tokyo.")

    def test_4(self):
        """Large age"""
        self.assertEqual(format_person("Elder", 100, "Paris"), "Elder is 100 years old and lives in Paris.")

    def test_5(self):
        """City with spaces"""
        self.assertEqual(format_person("Jane", 35, "San Francisco"), "Jane is 35 years old and lives in San Francisco.")

    def test_6(self):
        """Name with spaces"""
        self.assertEqual(format_person("John Doe", 40, "Berlin"), "John Doe is 40 years old and lives in Berlin.")

    def test_7(self):
        """Unicode name"""
        self.assertEqual(format_person("Мария", 28, "Moscow"), "Мария is 28 years old and lives in Moscow.")

    def test_8(self):
        """Negative age (edge case)"""
        self.assertEqual(format_person("Test", -5, "Test City"), "Test is -5 years old and lives in Test City.")

    def test_9(self):
        """Single character name"""
        self.assertEqual(format_person("A", 1, "B"), "A is 1 years old and lives in B.")

    def test_10(self):
        """Float age with .0"""
        self.assertEqual(format_person("Test", 25.0, "City"), "Test is 25 years old and lives in City.")

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use f-strings: f"text {variable} more text"',
	hint2: 'To convert float to int, use int(age) inside the f-string.',

	whyItMatters: `String formatting is used everywhere: log messages, user interfaces, reports, and API responses.

**Production Pattern:**

\`\`\`python
def format_log_entry(level: str, timestamp: str, message: str) -> str:
    """Standard log format for production systems."""
    return f"[{level.upper():8}] {timestamp} | {message}"

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format money for display."""
    return f"{currency} {amount:,.2f}"

# Examples:
# "[INFO    ] 2024-01-15 10:30:00 | User logged in"
# "USD 1,234.56"
\`\`\`

**Practical Benefits:**
- F-strings are fastest and most readable
- Format specifiers control number precision and alignment
- Consistent formatting improves debugging and monitoring`,

	translations: {
		ru: {
			title: 'Форматирование строк',
			description: `# Форматирование строк

Python предлагает несколько способов форматирования строк. Современный подход использует f-строки.

## Задача

Реализуйте функцию \`format_person(name, age, city)\`, которая возвращает отформатированное описание.

## Требования

- Верните: \`"{name} is {age} years old and lives in {city}."\`
- Возраст должен отображаться как целое число (без десятичных)

## Примеры

\`\`\`python
>>> format_person("Alice", 25, "New York")
"Alice is 25 years old and lives in New York."

>>> format_person("Bob", 30.5, "London")
"Bob is 30 years old and lives in London."
\`\`\``,
			hint1: 'Используйте f-строки: f"текст {переменная} ещё текст"',
			hint2: 'Для преобразования float в int используйте int(age) внутри f-строки.',
			whyItMatters: `Форматирование строк используется повсеместно: логи, интерфейсы, отчёты, API.

**Продакшен паттерн:**

\`\`\`python
def format_log_entry(level: str, timestamp: str, message: str) -> str:
    """Стандартный формат логов для продакшен систем."""
    return f"[{level.upper():8}] {timestamp} | {message}"

def format_currency(amount: float, currency: str = "USD") -> str:
    """Форматирование денег для отображения."""
    return f"{currency} {amount:,.2f}"
\`\`\`

**Практические преимущества:**
- F-строки самые быстрые и читаемые
- Спецификаторы формата контролируют точность и выравнивание
- Единообразное форматирование улучшает отладку`,
		},
		uz: {
			title: 'Satrlarni formatlash',
			description: `# Satrlarni formatlash

Python satrlarni formatlashning bir necha usullarini taklif qiladi. Zamonaviy yondashuv f-satrlardan foydalanadi.

## Vazifa

Formatlangan tavsifni qaytaruvchi \`format_person(name, age, city)\` funksiyasini amalga oshiring.

## Talablar

- Qaytaring: \`"{name} is {age} years old and lives in {city}."\`
- Yosh butun son sifatida ko'rsatilishi kerak

## Misollar

\`\`\`python
>>> format_person("Alice", 25, "New York")
"Alice is 25 years old and lives in New York."

>>> format_person("Bob", 30.5, "London")
"Bob is 30 years old and lives in London."
\`\`\``,
			hint1: 'F-satrlardan foydalaning: f"matn {ozgaruvchi} yana matn"',
			hint2: "Float ni int ga aylantirish uchun f-satr ichida int(age) dan foydalaning.",
			whyItMatters: `Satrlarni formatlash hamma joyda ishlatiladi: loglar, interfeyslar, hisobotlar, API.

**Ishlab chiqarish patterni:**

\`\`\`python
def format_log_entry(level: str, timestamp: str, message: str) -> str:
    """Ishlab chiqarish tizimlari uchun standart log formati."""
    return f"[{level.upper():8}] {timestamp} | {message}"

def format_currency(amount: float, currency: str = "USD") -> str:
    """Pulni ko'rsatish uchun formatlash."""
    return f"{currency} {amount:,.2f}"
\`\`\`

**Amaliy foydalari:**
- F-satrlar eng tez va o'qilishi oson
- Format spetsifikatorlari aniqlik va tekislashni boshqaradi
- Yagona formatlash disk raskadrovkani yaxshilaydi`,
		},
	},
};

export default task;
