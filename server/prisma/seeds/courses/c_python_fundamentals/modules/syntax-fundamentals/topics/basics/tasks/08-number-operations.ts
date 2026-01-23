import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-number-operations',
	title: 'Number Operations',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'numbers', 'math'],
	estimatedTime: '10m',
	isPremium: false,
	order: 8,

	description: `# Number Operations

Practice Python's arithmetic operators: \`+\`, \`-\`, \`*\`, \`/\`, \`//\`, \`%\`, \`**\`.

## Task

Implement the function \`analyze_number(n)\` that returns a dictionary with various properties of a number.

## Requirements

Return a dictionary with these keys:
- \`"is_positive"\`: True if n > 0
- \`"is_even"\`: True if n is even (divisible by 2)
- \`"abs_value"\`: Absolute value of n
- \`"squared"\`: n squared (n^2)
- \`"last_digit"\`: Last digit of the absolute value

## Examples

\`\`\`python
>>> analyze_number(42)
{"is_positive": True, "is_even": True, "abs_value": 42, "squared": 1764, "last_digit": 2}

>>> analyze_number(-15)
{"is_positive": False, "is_even": False, "abs_value": 15, "squared": 225, "last_digit": 5}

>>> analyze_number(0)
{"is_positive": False, "is_even": True, "abs_value": 0, "squared": 0, "last_digit": 0}
\`\`\``,

	initialCode: `def analyze_number(n: int) -> dict:
    """Analyze various properties of a number.

    Args:
        n: Any integer (positive, negative, or zero)

    Returns:
        Dictionary with keys:
        - is_positive: True if n > 0
        - is_even: True if n is divisible by 2
        - abs_value: Absolute value of n
        - squared: n squared
        - last_digit: Last digit of |n|
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def analyze_number(n: int) -> dict:
    """Analyze various properties of a number.

    Args:
        n: Any integer (positive, negative, or zero)

    Returns:
        Dictionary with keys:
        - is_positive: True if n > 0
        - is_even: True if n is divisible by 2
        - abs_value: Absolute value of n
        - squared: n squared
        - last_digit: Last digit of |n|
    """
    # Get absolute value - works for both positive and negative
    abs_val = abs(n)

    # Build result dictionary with all properties
    return {
        # Positive means greater than zero (0 is not positive)
        "is_positive": n > 0,

        # Even numbers have remainder 0 when divided by 2
        # Note: 0 is considered even
        "is_even": n % 2 == 0,

        # Absolute value removes the sign
        "abs_value": abs_val,

        # Square using exponentiation operator
        "squared": n ** 2,

        # Last digit: remainder when divided by 10
        # Use abs_val to handle negative numbers
        "last_digit": abs_val % 10,
    }`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Positive even number"""
        result = analyze_number(42)
        self.assertEqual(result["is_positive"], True)
        self.assertEqual(result["is_even"], True)
        self.assertEqual(result["abs_value"], 42)
        self.assertEqual(result["squared"], 1764)
        self.assertEqual(result["last_digit"], 2)

    def test_2(self):
        """Negative odd number"""
        result = analyze_number(-15)
        self.assertEqual(result["is_positive"], False)
        self.assertEqual(result["is_even"], False)
        self.assertEqual(result["abs_value"], 15)
        self.assertEqual(result["squared"], 225)
        self.assertEqual(result["last_digit"], 5)

    def test_3(self):
        """Zero"""
        result = analyze_number(0)
        self.assertEqual(result["is_positive"], False)
        self.assertEqual(result["is_even"], True)
        self.assertEqual(result["abs_value"], 0)
        self.assertEqual(result["squared"], 0)
        self.assertEqual(result["last_digit"], 0)

    def test_4(self):
        """Positive one"""
        result = analyze_number(1)
        self.assertEqual(result["is_positive"], True)
        self.assertEqual(result["is_even"], False)
        self.assertEqual(result["last_digit"], 1)

    def test_5(self):
        """Negative even"""
        result = analyze_number(-100)
        self.assertEqual(result["is_positive"], False)
        self.assertEqual(result["is_even"], True)
        self.assertEqual(result["abs_value"], 100)
        self.assertEqual(result["last_digit"], 0)

    def test_6(self):
        """Large number"""
        result = analyze_number(999)
        self.assertEqual(result["squared"], 998001)
        self.assertEqual(result["last_digit"], 9)

    def test_7(self):
        """Small negative"""
        result = analyze_number(-1)
        self.assertEqual(result["is_positive"], False)
        self.assertEqual(result["abs_value"], 1)

    def test_8(self):
        """Two-digit number"""
        result = analyze_number(73)
        self.assertEqual(result["last_digit"], 3)
        self.assertEqual(result["is_even"], False)

    def test_9(self):
        """Perfect square input"""
        result = analyze_number(10)
        self.assertEqual(result["squared"], 100)
        self.assertEqual(result["last_digit"], 0)

    def test_10(self):
        """Negative two-digit"""
        result = analyze_number(-88)
        self.assertEqual(result["abs_value"], 88)
        self.assertEqual(result["is_even"], True)
        self.assertEqual(result["last_digit"], 8)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use % (modulo) operator: n % 2 == 0 checks if even, n % 10 gets the last digit.',
	hint2: 'Use abs() function for absolute value and ** operator for exponentiation (squaring).',

	whyItMatters: `Number operations are fundamental to all programming - from simple calculations to complex algorithms.

**Production Pattern:**

\`\`\`python
def calculate_pagination(total_items: int, page_size: int) -> dict:
    """Calculate pagination metadata."""
    # Integer division for total pages
    total_pages = (total_items + page_size - 1) // page_size

    return {
        "total_items": total_items,
        "page_size": page_size,
        "total_pages": total_pages,
    }

def format_file_size(bytes: int) -> str:
    """Convert bytes to human-readable size."""
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0

    size = float(bytes)
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.1f} {units[unit_index]}"

def validate_credit_card(number: str) -> bool:
    """Luhn algorithm for credit card validation."""
    digits = [int(d) for d in number if d.isdigit()]
    # Double every second digit from right, sum digits if > 9
    for i in range(len(digits) - 2, -1, -2):
        digits[i] *= 2
        if digits[i] > 9:
            digits[i] -= 9
    return sum(digits) % 10 == 0
\`\`\`

**Practical Benefits:**
- Pagination calculations use integer division
- File size formatting needs power operations
- Validation algorithms rely on modulo operations`,

	translations: {
		ru: {
			title: 'Операции с числами',
			description: `# Операции с числами

Практика арифметических операторов Python: \`+\`, \`-\`, \`*\`, \`/\`, \`//\`, \`%\`, \`**\`.

## Задача

Реализуйте функцию \`analyze_number(n)\`, которая возвращает словарь с различными свойствами числа.

## Требования

Верните словарь со следующими ключами:
- \`"is_positive"\`: True если n > 0
- \`"is_even"\`: True если n чётное (делится на 2)
- \`"abs_value"\`: Абсолютное значение n
- \`"squared"\`: n в квадрате (n^2)
- \`"last_digit"\`: Последняя цифра абсолютного значения

## Примеры

\`\`\`python
>>> analyze_number(42)
{"is_positive": True, "is_even": True, "abs_value": 42, "squared": 1764, "last_digit": 2}

>>> analyze_number(-15)
{"is_positive": False, "is_even": False, "abs_value": 15, "squared": 225, "last_digit": 5}

>>> analyze_number(0)
{"is_positive": False, "is_even": True, "abs_value": 0, "squared": 0, "last_digit": 0}
\`\`\``,
			hint1: 'Используйте % (остаток от деления): n % 2 == 0 проверяет чётность, n % 10 даёт последнюю цифру.',
			hint2: 'Используйте abs() для абсолютного значения и ** для возведения в степень.',
			whyItMatters: `Операции с числами — основа всего программирования.

**Продакшен паттерн:**

\`\`\`python
def calculate_pagination(total_items: int, page_size: int) -> dict:
    """Расчёт метаданных пагинации."""
    total_pages = (total_items + page_size - 1) // page_size
    return {
        "total_items": total_items,
        "page_size": page_size,
        "total_pages": total_pages,
    }

def format_file_size(bytes: int) -> str:
    """Конвертация байт в читаемый формат."""
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(bytes)
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.1f} {units[unit_index]}"
\`\`\`

**Практические преимущества:**
- Пагинация использует целочисленное деление
- Форматирование размеров требует операций степени
- Алгоритмы валидации опираются на операции с остатком`,
		},
		uz: {
			title: 'Sonlar bilan amallar',
			description: `# Sonlar bilan amallar

Python arifmetik operatorlarini mashq qiling: \`+\`, \`-\`, \`*\`, \`/\`, \`//\`, \`%\`, \`**\`.

## Vazifa

Sonning turli xususiyatlarini qaytaruvchi \`analyze_number(n)\` funksiyasini amalga oshiring.

## Talablar

Quyidagi kalitlarga ega lug'at qaytaring:
- \`"is_positive"\`: n > 0 bo'lsa True
- \`"is_even"\`: n juft (2 ga bo'linadi) bo'lsa True
- \`"abs_value"\`: n ning absolyut qiymati
- \`"squared"\`: n ning kvadrati (n^2)
- \`"last_digit"\`: |n| ning oxirgi raqami

## Misollar

\`\`\`python
>>> analyze_number(42)
{"is_positive": True, "is_even": True, "abs_value": 42, "squared": 1764, "last_digit": 2}

>>> analyze_number(-15)
{"is_positive": False, "is_even": False, "abs_value": 15, "squared": 225, "last_digit": 5}

>>> analyze_number(0)
{"is_positive": False, "is_even": True, "abs_value": 0, "squared": 0, "last_digit": 0}
\`\`\``,
			hint1: "% (qoldiq) operatoridan foydalaning: n % 2 == 0 juftlikni tekshiradi, n % 10 oxirgi raqamni beradi.",
			hint2: 'abs() absolyut qiymat uchun va ** darajaga ko\'tarish uchun foydalaning.',
			whyItMatters: `Sonlar bilan amallar barcha dasturlashning asosidir.

**Ishlab chiqarish patterni:**

\`\`\`python
def calculate_pagination(total_items: int, page_size: int) -> dict:
    """Sahifalash metama'lumotlarini hisoblash."""
    total_pages = (total_items + page_size - 1) // page_size
    return {
        "total_items": total_items,
        "page_size": page_size,
        "total_pages": total_pages,
    }

def format_file_size(bytes: int) -> str:
    """Baytlarni o'qilishi oson formatga o'tkazish."""
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(bytes)
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.1f} {units[unit_index]}"
\`\`\`

**Amaliy foydalari:**
- Sahifalash butun sonli bo'lishni ishlatadi
- Hajm formatlash daraja amallarini talab qiladi
- Tasdiqlash algoritmlari qoldiq amallariga tayanadi`,
		},
	},
};

export default task;
