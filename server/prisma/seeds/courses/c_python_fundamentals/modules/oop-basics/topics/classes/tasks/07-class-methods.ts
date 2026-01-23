import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-class-methods',
	title: 'Class and Static Methods',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'oop', 'classmethod'],
	estimatedTime: '15m',
	isPremium: false,
	order: 7,

	description: `# Class and Static Methods

Learn the difference between instance methods, class methods, and static methods.

## Task

Create a class \`Date\` with different types of methods.

## Requirements

- \`__init__(self, year, month, day)\`: Initialize date
- \`from_string(cls, date_str)\`: Class method to create Date from "YYYY-MM-DD" string
- \`is_valid_date(year, month, day)\`: Static method to validate date components
- \`format(self)\`: Instance method returning "YYYY-MM-DD"

## Examples

\`\`\`python
>>> d = Date(2024, 1, 15)
>>> d.format()
"2024-01-15"

>>> d2 = Date.from_string("2024-12-25")
>>> d2.month
12

>>> Date.is_valid_date(2024, 13, 1)
False
>>> Date.is_valid_date(2024, 6, 15)
True
\`\`\``,

	initialCode: `class Date:
    """A simple Date class demonstrating different method types."""

    def __init__(self, year: int, month: int, day: int):
        # TODO: Initialize
        pass

    @classmethod
    def from_string(cls, date_str: str):
        # TODO: Parse "YYYY-MM-DD" and create Date instance
        pass

    @staticmethod
    def is_valid_date(year: int, month: int, day: int) -> bool:
        # TODO: Basic validation (1-12 months, 1-31 days)
        pass

    def format(self) -> str:
        # TODO: Return formatted string
        pass`,

	solutionCode: `class Date:
    """A simple Date class demonstrating different method types."""

    def __init__(self, year: int, month: int, day: int):
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def from_string(cls, date_str: str):
        # Parse string and use cls to create new instance
        parts = date_str.split("-")
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        return cls(year, month, day)

    @staticmethod
    def is_valid_date(year: int, month: int, day: int) -> bool:
        # Basic validation (simplified)
        if month < 1 or month > 12:
            return False
        if day < 1 or day > 31:
            return False
        return True

    def format(self) -> str:
        # Zero-pad month and day
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        d = Date(2024, 1, 15)
        self.assertEqual(d.format(), "2024-01-15")

    def test_2(self):
        d = Date.from_string("2024-12-25")
        self.assertEqual(d.year, 2024)

    def test_3(self):
        d = Date.from_string("2024-12-25")
        self.assertEqual(d.month, 12)

    def test_4(self):
        self.assertFalse(Date.is_valid_date(2024, 13, 1))

    def test_5(self):
        self.assertTrue(Date.is_valid_date(2024, 6, 15))

    def test_6(self):
        self.assertFalse(Date.is_valid_date(2024, 0, 15))

    def test_7(self):
        self.assertFalse(Date.is_valid_date(2024, 1, 32))

    def test_8(self):
        d = Date(2024, 7, 4)
        self.assertEqual(d.day, 4)

    def test_9(self):
        d = Date.from_string("2000-01-01")
        self.assertEqual(d.format(), "2000-01-01")

    def test_10(self):
        d = Date(1, 1, 1)
        self.assertEqual(d.format(), "0001-01-01")

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use @classmethod for from_string - it receives cls and can create new instances using cls().',
	hint2: 'Use @staticmethod for is_valid_date - it doesn\'t need self or cls, just validates data.',

	whyItMatters: `Class methods enable alternative constructors, static methods provide utility functions.`,

	translations: {
		ru: {
			title: 'Методы класса и статические методы',
			description: `# Методы класса и статические методы

Изучите разницу между методами экземпляра, класса и статическими методами.

## Задача

Создайте класс \`Date\` с разными типами методов.`,
			hint1: 'Используйте @classmethod для from_string - он получает cls и создаёт экземпляры через cls().',
			hint2: 'Используйте @staticmethod для is_valid_date - не нужен self или cls, только валидация.',
			whyItMatters: `Методы класса позволяют создавать альтернативные конструкторы.`,
		},
		uz: {
			title: 'Klass va statik metodlar',
			description: `# Klass va statik metodlar

Ekzemplyar metodlari, klass metodlari va statik metodlar orasidagi farqni o'rganing.

## Vazifa

Turli metod turlariga ega \`Date\` klassini yarating.`,
			hint1: "from_string uchun @classmethod ishlating - u cls qabul qiladi va cls() orqali yangi nusxalar yaratadi.",
			hint2: "is_valid_date uchun @staticmethod ishlating - self yoki cls kerak emas, faqat tekshirish.",
			whyItMatters: `Klass metodlari muqobil konstruktorlarni yaratishga imkon beradi.`,
		},
	},
};

export default task;
