import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-fizzbuzz',
	title: 'FizzBuzz',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'conditionals', 'modulo'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,

	description: `# FizzBuzz

The classic programming challenge! Test your understanding of conditionals and the modulo operator.

## Task

Implement the function \`fizzbuzz(n)\` that returns a string based on the number.

## Requirements

- If n is divisible by both 3 and 5, return \`"FizzBuzz"\`
- If n is divisible by 3 only, return \`"Fizz"\`
- If n is divisible by 5 only, return \`"Buzz"\`
- Otherwise, return the number as a string

## Examples

\`\`\`python
>>> fizzbuzz(3)
"Fizz"

>>> fizzbuzz(5)
"Buzz"

>>> fizzbuzz(15)
"FizzBuzz"

>>> fizzbuzz(7)
"7"
\`\`\``,

	initialCode: `def fizzbuzz(n: int) -> str:
    """Return FizzBuzz result for a number.

    Args:
        n: A positive integer

    Returns:
        "Fizz", "Buzz", "FizzBuzz", or the number as string
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def fizzbuzz(n: int) -> str:
    """Return FizzBuzz result for a number.

    Args:
        n: A positive integer

    Returns:
        "Fizz", "Buzz", "FizzBuzz", or the number as string
    """
    # Check divisible by both first (most specific condition)
    if n % 3 == 0 and n % 5 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    else:
        return str(n)`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Divisible by 3"""
        self.assertEqual(fizzbuzz(3), "Fizz")

    def test_2(self):
        """Divisible by 5"""
        self.assertEqual(fizzbuzz(5), "Buzz")

    def test_3(self):
        """Divisible by both"""
        self.assertEqual(fizzbuzz(15), "FizzBuzz")

    def test_4(self):
        """Not divisible by 3 or 5"""
        self.assertEqual(fizzbuzz(7), "7")

    def test_5(self):
        """Another FizzBuzz"""
        self.assertEqual(fizzbuzz(30), "FizzBuzz")

    def test_6(self):
        """Fizz for 9"""
        self.assertEqual(fizzbuzz(9), "Fizz")

    def test_7(self):
        """Buzz for 10"""
        self.assertEqual(fizzbuzz(10), "Buzz")

    def test_8(self):
        """Number 1"""
        self.assertEqual(fizzbuzz(1), "1")

    def test_9(self):
        """Number 2"""
        self.assertEqual(fizzbuzz(2), "2")

    def test_10(self):
        """Large FizzBuzz"""
        self.assertEqual(fizzbuzz(45), "FizzBuzz")

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use the modulo operator (%) to check divisibility. n % 3 == 0 means n is divisible by 3.',
	hint2: 'Check the "divisible by both" condition FIRST, before checking individual conditions.',

	whyItMatters: `FizzBuzz teaches the importance of condition ordering. The most specific condition must come first.

**Production Pattern:**

\`\`\`python
def categorize_order(total: float, items: int) -> str:
    """Categorize order for shipping logic."""
    if total >= 100 and items >= 5:
        return "bulk_priority"
    elif total >= 100:
        return "high_value"
    elif items >= 5:
        return "bulk"
    else:
        return "standard"
\`\`\`

**Practical Benefits:**
- Condition ordering prevents logic bugs
- Modulo is used everywhere: pagination, cycling, validation
- Clear conditionals make business logic readable`,

	translations: {
		ru: {
			title: 'FizzBuzz',
			description: `# FizzBuzz

Классическая задача программирования! Проверьте понимание условий и оператора деления по модулю.

## Задача

Реализуйте функцию \`fizzbuzz(n)\`, которая возвращает строку на основе числа.

## Требования

- Если n делится на 3 и на 5, верните \`"FizzBuzz"\`
- Если n делится только на 3, верните \`"Fizz"\`
- Если n делится только на 5, верните \`"Buzz"\`
- Иначе верните число как строку

## Примеры

\`\`\`python
>>> fizzbuzz(3)
"Fizz"

>>> fizzbuzz(15)
"FizzBuzz"

>>> fizzbuzz(7)
"7"
\`\`\``,
			hint1: 'Используйте оператор модуля (%) для проверки делимости.',
			hint2: 'Проверяйте условие "делится на оба" ПЕРВЫМ, до отдельных условий.',
			whyItMatters: `FizzBuzz учит важности порядка условий.

**Продакшен паттерн:**

\`\`\`python
def categorize_order(total: float, items: int) -> str:
    """Категоризация заказа для логики доставки."""
    if total >= 100 and items >= 5:
        return "bulk_priority"
    elif total >= 100:
        return "high_value"
    elif items >= 5:
        return "bulk"
    else:
        return "standard"
\`\`\`

**Практические преимущества:**
- Порядок условий предотвращает логические ошибки
- Модуло используется везде: пагинация, циклы, валидация`,
		},
		uz: {
			title: 'FizzBuzz',
			description: `# FizzBuzz

Klassik dasturlash vazifasi! Shartlar va modul operatorini tushunishingizni tekshiring.

## Vazifa

Songa asoslangan satrni qaytaruvchi \`fizzbuzz(n)\` funksiyasini amalga oshiring.

## Talablar

- Agar n 3 ga ham 5 ga ham bo'linsa, \`"FizzBuzz"\` qaytaring
- Agar n faqat 3 ga bo'linsa, \`"Fizz"\` qaytaring
- Agar n faqat 5 ga bo'linsa, \`"Buzz"\` qaytaring
- Aks holda sonni satr sifatida qaytaring

## Misollar

\`\`\`python
>>> fizzbuzz(3)
"Fizz"

>>> fizzbuzz(15)
"FizzBuzz"

>>> fizzbuzz(7)
"7"
\`\`\``,
			hint1: "Bo'linishni tekshirish uchun modul operatoridan (%) foydalaning.",
			hint2: '"Ikkisiga ham bo\'linadi" shartini BIRINCHI tekshiring.',
			whyItMatters: `FizzBuzz shartlar tartibining muhimligini o'rgatadi.

**Ishlab chiqarish patterni:**

\`\`\`python
def categorize_order(total: float, items: int) -> str:
    """Yetkazib berish mantig'i uchun buyurtmani tasniflash."""
    if total >= 100 and items >= 5:
        return "bulk_priority"
    elif total >= 100:
        return "high_value"
    elif items >= 5:
        return "bulk"
    else:
        return "standard"
\`\`\`

**Amaliy foydalari:**
- Shartlar tartibi mantiqiy xatolarni oldini oladi
- Modul hamma joyda ishlatiladi: sahifalash, sikllash, tekshirish`,
		},
	},
};

export default task;
