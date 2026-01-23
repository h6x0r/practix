import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-basic-math',
	title: 'Basic Math Operations',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'math', 'operators'],
	estimatedTime: '10m',
	isPremium: false,
	order: 2,

	description: `# Basic Math Operations

Python supports all standard mathematical operations. Let's practice them!

## Task

Implement the function \`calculate(a, b, operation)\` that performs basic math operations.

## Requirements

- Support operations: \`"add"\`, \`"subtract"\`, \`"multiply"\`, \`"divide"\`
- For division by zero, return \`None\`
- Return \`None\` for unknown operations

## Examples

\`\`\`python
>>> calculate(10, 5, "add")
15

>>> calculate(10, 5, "subtract")
5

>>> calculate(10, 5, "multiply")
50

>>> calculate(10, 5, "divide")
2.0

>>> calculate(10, 0, "divide")
None
\`\`\``,

	initialCode: `def calculate(a: float, b: float, operation: str) -> float | None:
    """Perform a basic math operation on two numbers.

    Args:
        a: First number
        b: Second number
        operation: One of "add", "subtract", "multiply", "divide"

    Returns:
        Result of the operation, or None if invalid
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def calculate(a: float, b: float, operation: str) -> float | None:
    """Perform a basic math operation on two numbers.

    Args:
        a: First number
        b: Second number
        operation: One of "add", "subtract", "multiply", "divide"

    Returns:
        Result of the operation, or None if invalid
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            return None
        return a / b
    else:
        return None`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Addition"""
        self.assertEqual(calculate(10, 5, "add"), 15)

    def test_2(self):
        """Subtraction"""
        self.assertEqual(calculate(10, 5, "subtract"), 5)

    def test_3(self):
        """Multiplication"""
        self.assertEqual(calculate(10, 5, "multiply"), 50)

    def test_4(self):
        """Division"""
        self.assertEqual(calculate(10, 5, "divide"), 2.0)

    def test_5(self):
        """Division by zero"""
        self.assertIsNone(calculate(10, 0, "divide"))

    def test_6(self):
        """Unknown operation"""
        self.assertIsNone(calculate(10, 5, "power"))

    def test_7(self):
        """Negative numbers addition"""
        self.assertEqual(calculate(-5, -3, "add"), -8)

    def test_8(self):
        """Float division"""
        self.assertEqual(calculate(7, 2, "divide"), 3.5)

    def test_9(self):
        """Zero multiplication"""
        self.assertEqual(calculate(100, 0, "multiply"), 0)

    def test_10(self):
        """Large numbers"""
        self.assertEqual(calculate(1000000, 1000000, "add"), 2000000)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use if/elif/else to check the operation string and perform the corresponding calculation.',
	hint2: 'For division, always check if b == 0 before dividing to avoid ZeroDivisionError.',

	whyItMatters: `Understanding basic operations is fundamental to all programming. This pattern of dispatching based on a string command is extremely common.

**Production Pattern:**

\`\`\`python
OPERATIONS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b if b != 0 else None,
}

def calculate(a, b, op):
    return OPERATIONS.get(op, lambda x, y: None)(a, b)
\`\`\`

**Practical Benefits:**
- Dictionary dispatch is cleaner than long if/elif chains
- Lambda functions enable concise operation definitions
- Safe division prevents runtime crashes`,

	translations: {
		ru: {
			title: 'Базовые математические операции',
			description: `# Базовые математические операции

Python поддерживает все стандартные математические операции. Давайте попрактикуемся!

## Задача

Реализуйте функцию \`calculate(a, b, operation)\`, которая выполняет базовые математические операции.

## Требования

- Поддержите операции: \`"add"\`, \`"subtract"\`, \`"multiply"\`, \`"divide"\`
- При делении на ноль верните \`None\`
- Для неизвестных операций верните \`None\`

## Примеры

\`\`\`python
>>> calculate(10, 5, "add")
15

>>> calculate(10, 5, "divide")
2.0

>>> calculate(10, 0, "divide")
None
\`\`\``,
			hint1: 'Используйте if/elif/else для проверки строки операции.',
			hint2: 'Для деления всегда проверяйте b == 0 перед делением.',
			whyItMatters: `Понимание базовых операций — основа всего программирования.

**Продакшен паттерн:**

\`\`\`python
OPERATIONS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b if b != 0 else None,
}

def calculate(a, b, op):
    return OPERATIONS.get(op, lambda x, y: None)(a, b)
\`\`\`

**Практические преимущества:**
- Словарь диспетчеризации чище длинных цепочек if/elif
- Lambda функции позволяют компактно определять операции
- Безопасное деление предотвращает падения`,
		},
		uz: {
			title: 'Asosiy matematik amallar',
			description: `# Asosiy matematik amallar

Python barcha standart matematik amallarni qo'llab-quvvatlaydi. Mashq qilaylik!

## Vazifa

Asosiy matematik amallarni bajaradigan \`calculate(a, b, operation)\` funksiyasini amalga oshiring.

## Talablar

- Amallarni qo'llab-quvvatlang: \`"add"\`, \`"subtract"\`, \`"multiply"\`, \`"divide"\`
- Nolga bo'lganda \`None\` qaytaring
- Noma'lum amallar uchun \`None\` qaytaring

## Misollar

\`\`\`python
>>> calculate(10, 5, "add")
15

>>> calculate(10, 5, "divide")
2.0

>>> calculate(10, 0, "divide")
None
\`\`\``,
			hint1: "Amal satrini tekshirish uchun if/elif/else dan foydalaning.",
			hint2: "Bo'lishdan oldin har doim b == 0 ekanligini tekshiring.",
			whyItMatters: `Asosiy amallarni tushunish barcha dasturlashning asosidir.

**Ishlab chiqarish patterni:**

\`\`\`python
OPERATIONS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b if b != 0 else None,
}

def calculate(a, b, op):
    return OPERATIONS.get(op, lambda x, y: None)(a, b)
\`\`\`

**Amaliy foydalari:**
- Lug'at dispatchi uzun if/elif zanjirlaridan toza
- Lambda funksiyalari amallarni ixcham aniqlash imkonini beradi
- Xavfsiz bo'lish dastur qulab tushishini oldini oladi`,
		},
	},
};

export default task;
