import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-digit-sum',
	title: 'Sum of Digits',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'loops', 'math'],
	estimatedTime: '10m',
	isPremium: false,
	order: 7,

	description: `# Sum of Digits

Practice extracting digits from a number using loops and modulo.

## Task

Implement the function \`digit_sum(n)\` that calculates the sum of all digits in a number.

## Requirements

- Handle both positive and negative numbers (use absolute value)
- Return the sum of all individual digits
- Handle zero correctly

## Examples

\`\`\`python
>>> digit_sum(123)
6  # 1 + 2 + 3

>>> digit_sum(9999)
36  # 9 + 9 + 9 + 9

>>> digit_sum(-456)
15  # 4 + 5 + 6

>>> digit_sum(0)
0
\`\`\``,

	initialCode: `def digit_sum(n: int) -> int:
    """Calculate the sum of all digits in a number.

    Args:
        n: Any integer (positive, negative, or zero)

    Returns:
        Sum of all individual digits
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def digit_sum(n: int) -> int:
    """Calculate the sum of all digits in a number.

    Args:
        n: Any integer (positive, negative, or zero)

    Returns:
        Sum of all individual digits
    """
    # Use absolute value to handle negative numbers
    n = abs(n)

    # Initialize sum
    total = 0

    # Extract digits one by one from right to left
    while n > 0:
        # Get the rightmost digit using modulo
        digit = n % 10
        total += digit

        # Remove the rightmost digit using integer division
        n = n // 10

    return total`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Three-digit positive number"""
        self.assertEqual(digit_sum(123), 6)

    def test_2(self):
        """All same digits"""
        self.assertEqual(digit_sum(9999), 36)

    def test_3(self):
        """Negative number"""
        self.assertEqual(digit_sum(-456), 15)

    def test_4(self):
        """Zero"""
        self.assertEqual(digit_sum(0), 0)

    def test_5(self):
        """Single digit"""
        self.assertEqual(digit_sum(7), 7)

    def test_6(self):
        """Number with zeros"""
        self.assertEqual(digit_sum(1001), 2)

    def test_7(self):
        """Large number"""
        self.assertEqual(digit_sum(123456789), 45)

    def test_8(self):
        """Two-digit number"""
        self.assertEqual(digit_sum(99), 18)

    def test_9(self):
        """Power of ten"""
        self.assertEqual(digit_sum(1000), 1)

    def test_10(self):
        """Negative single digit"""
        self.assertEqual(digit_sum(-5), 5)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use abs(n) to handle negative numbers. The sign doesn\'t affect the digits themselves.',
	hint2: 'Use n % 10 to get the last digit, and n // 10 to remove it. Loop while n > 0.',

	whyItMatters: `Digit manipulation is common in data validation, checksums, and numerical algorithms.

**Production Pattern:**

\`\`\`python
def calculate_check_digit(number: str) -> int:
    """Calculate check digit for validation (simplified)."""
    digits = [int(d) for d in number if d.isdigit()]
    weighted_sum = sum(d * (i + 1) for i, d in enumerate(digits))
    return weighted_sum % 10

def digital_root(n: int) -> int:
    """Calculate digital root (repeated digit sum until single digit)."""
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n

def is_armstrong_number(n: int) -> bool:
    """Check if number equals sum of digits raised to power of digit count."""
    digits = [int(d) for d in str(n)]
    power = len(digits)
    return n == sum(d ** power for d in digits)

# Example: 153 = 1^3 + 5^3 + 3^3 = 1 + 125 + 27 = 153
\`\`\`

**Practical Benefits:**
- Credit card validation uses digit algorithms (Luhn)
- ISBN validation requires digit sums
- Data integrity checks use checksum calculations`,

	translations: {
		ru: {
			title: 'Сумма цифр',
			description: `# Сумма цифр

Практика извлечения цифр из числа с помощью циклов и остатка от деления.

## Задача

Реализуйте функцию \`digit_sum(n)\`, которая вычисляет сумму всех цифр числа.

## Требования

- Обработайте и положительные, и отрицательные числа (используйте модуль)
- Верните сумму всех отдельных цифр
- Корректно обработайте ноль

## Примеры

\`\`\`python
>>> digit_sum(123)
6  # 1 + 2 + 3

>>> digit_sum(9999)
36  # 9 + 9 + 9 + 9

>>> digit_sum(-456)
15  # 4 + 5 + 6

>>> digit_sum(0)
0
\`\`\``,
			hint1: 'Используйте abs(n) для отрицательных чисел. Знак не влияет на сами цифры.',
			hint2: 'Используйте n % 10 для получения последней цифры и n // 10 для её удаления.',
			whyItMatters: `Манипуляции с цифрами часто встречаются в валидации данных и контрольных суммах.

**Продакшен паттерн:**

\`\`\`python
def calculate_check_digit(number: str) -> int:
    """Расчёт контрольной цифры для валидации."""
    digits = [int(d) for d in number if d.isdigit()]
    weighted_sum = sum(d * (i + 1) for i, d in enumerate(digits))
    return weighted_sum % 10

def digital_root(n: int) -> int:
    """Цифровой корень (повторная сумма до одной цифры)."""
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n
\`\`\`

**Практические преимущества:**
- Валидация кредитных карт использует алгоритмы с цифрами (Luhn)
- Валидация ISBN требует сумм цифр`,
		},
		uz: {
			title: "Raqamlar yig'indisi",
			description: `# Raqamlar yig'indisi

Tsikllar va qoldiq yordamida sondan raqamlarni ajratib olishni mashq qiling.

## Vazifa

Sondagi barcha raqamlar yig'indisini hisoblovchi \`digit_sum(n)\` funksiyasini amalga oshiring.

## Talablar

- Musbat va manfiy sonlarni ham ko'rib chiqing (absolyut qiymat ishlating)
- Barcha alohida raqamlar yig'indisini qaytaring
- Nolni to'g'ri ishlang

## Misollar

\`\`\`python
>>> digit_sum(123)
6  # 1 + 2 + 3

>>> digit_sum(9999)
36  # 9 + 9 + 9 + 9

>>> digit_sum(-456)
15  # 4 + 5 + 6

>>> digit_sum(0)
0
\`\`\``,
			hint1: "Manfiy sonlar uchun abs(n) ishlating. Belgi raqamlarning o'ziga ta'sir qilmaydi.",
			hint2: "Oxirgi raqamni olish uchun n % 10, uni olib tashlash uchun n // 10 ishlating.",
			whyItMatters: `Raqamlar bilan ishlash ma'lumotlarni tekshirish va nazorat summalarida keng tarqalgan.

**Ishlab chiqarish patterni:**

\`\`\`python
def calculate_check_digit(number: str) -> int:
    """Tekshirish uchun nazorat raqamini hisoblash."""
    digits = [int(d) for d in number if d.isdigit()]
    weighted_sum = sum(d * (i + 1) for i, d in enumerate(digits))
    return weighted_sum % 10

def digital_root(n: int) -> int:
    """Raqamli ildiz (bitta raqamgacha takroriy yig'indi)."""
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n
\`\`\`

**Amaliy foydalari:**
- Kredit karta tekshiruvi raqam algoritimlaridan foydalanadi (Luhn)
- ISBN tekshiruvi raqamlar yig'indisini talab qiladi`,
		},
	},
};

export default task;
