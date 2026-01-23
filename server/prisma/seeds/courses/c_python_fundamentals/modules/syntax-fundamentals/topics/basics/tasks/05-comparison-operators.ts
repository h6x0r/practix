import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-comparison-operators',
	title: 'Comparison Operators',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'operators', 'comparison'],
	estimatedTime: '10m',
	isPremium: false,
	order: 5,

	description: `# Comparison Operators

Python has several comparison operators: \`==\`, \`!=\`, \`<\`, \`>\`, \`<=\`, \`>=\`.

## Task

Implement the function \`compare(a, b)\` that returns a string describing the relationship between two numbers.

## Requirements

Return one of these strings:
- \`"equal"\` if a == b
- \`"greater"\` if a > b
- \`"less"\` if a < b

## Examples

\`\`\`python
>>> compare(5, 3)
"greater"

>>> compare(2, 7)
"less"

>>> compare(4, 4)
"equal"
\`\`\``,

	initialCode: `def compare(a: float, b: float) -> str:
    """Compare two numbers and return their relationship.

    Args:
        a: First number
        b: Second number

    Returns:
        "equal", "greater", or "less"
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def compare(a: float, b: float) -> str:
    """Compare two numbers and return their relationship.

    Args:
        a: First number
        b: Second number

    Returns:
        "equal", "greater", or "less"
    """
    if a == b:
        return "equal"
    elif a > b:
        return "greater"
    else:
        return "less"`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """a is greater"""
        self.assertEqual(compare(5, 3), "greater")

    def test_2(self):
        """a is less"""
        self.assertEqual(compare(2, 7), "less")

    def test_3(self):
        """Equal values"""
        self.assertEqual(compare(4, 4), "equal")

    def test_4(self):
        """Negative numbers, greater"""
        self.assertEqual(compare(-1, -5), "greater")

    def test_5(self):
        """Negative numbers, less"""
        self.assertEqual(compare(-10, -2), "less")

    def test_6(self):
        """Zero comparison"""
        self.assertEqual(compare(0, 0), "equal")

    def test_7(self):
        """Float comparison"""
        self.assertEqual(compare(3.14, 2.71), "greater")

    def test_8(self):
        """Very close floats"""
        self.assertEqual(compare(1.0, 1.0), "equal")

    def test_9(self):
        """Large numbers"""
        self.assertEqual(compare(1000000, 999999), "greater")

    def test_10(self):
        """Mixed positive and negative"""
        self.assertEqual(compare(-5, 5), "less")

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use == to check for equality, > for greater than, and < for less than.',
	hint2: 'With if/elif/else, you only need to check two conditions - the third is implied.',

	whyItMatters: `Comparison operators are the foundation of all decision-making in code. Every conditional, filter, and sort uses comparisons.

**Production Pattern:**

\`\`\`python
def validate_age(age: int) -> tuple[bool, str]:
    """Validate user age with clear feedback."""
    if age < 0:
        return False, "Age cannot be negative"
    elif age < 13:
        return False, "Must be 13 or older to register"
    elif age > 120:
        return False, "Please enter a valid age"
    return True, "Age is valid"

def sort_by_priority(tasks: list[dict]) -> list[dict]:
    """Sort tasks by priority (high first)."""
    priority_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(tasks, key=lambda t: priority_order.get(t["priority"], 99))
\`\`\`

**Practical Benefits:**
- Comparisons enable input validation
- Sorting and filtering depend on comparisons
- Clear comparison logic makes code maintainable`,

	translations: {
		ru: {
			title: 'Операторы сравнения',
			description: `# Операторы сравнения

Python имеет несколько операторов сравнения: \`==\`, \`!=\`, \`<\`, \`>\`, \`<=\`, \`>=\`.

## Задача

Реализуйте функцию \`compare(a, b)\`, которая возвращает строку, описывающую отношение между двумя числами.

## Требования

Верните одну из строк:
- \`"equal"\` если a == b
- \`"greater"\` если a > b
- \`"less"\` если a < b

## Примеры

\`\`\`python
>>> compare(5, 3)
"greater"

>>> compare(2, 7)
"less"

>>> compare(4, 4)
"equal"
\`\`\``,
			hint1: 'Используйте == для проверки равенства, > для больше, < для меньше.',
			hint2: 'С if/elif/else достаточно проверить два условия — третье подразумевается.',
			whyItMatters: `Операторы сравнения — основа принятия решений в коде.

**Продакшен паттерн:**

\`\`\`python
def validate_age(age: int) -> tuple[bool, str]:
    """Валидация возраста с понятной обратной связью."""
    if age < 0:
        return False, "Возраст не может быть отрицательным"
    elif age < 13:
        return False, "Для регистрации нужно быть старше 13"
    elif age > 120:
        return False, "Введите корректный возраст"
    return True, "Возраст корректен"
\`\`\`

**Практические преимущества:**
- Сравнения обеспечивают валидацию ввода
- Сортировка и фильтрация зависят от сравнений
- Понятная логика сравнений делает код поддерживаемым`,
		},
		uz: {
			title: 'Taqqoslash operatorlari',
			description: `# Taqqoslash operatorlari

Python da bir nechta taqqoslash operatorlari bor: \`==\`, \`!=\`, \`<\`, \`>\`, \`<=\`, \`>=\`.

## Vazifa

Ikki son orasidagi munosabatni tavsiflovchi satrni qaytaruvchi \`compare(a, b)\` funksiyasini amalga oshiring.

## Talablar

Quyidagi satrlardan birini qaytaring:
- \`"equal"\` agar a == b
- \`"greater"\` agar a > b
- \`"less"\` agar a < b

## Misollar

\`\`\`python
>>> compare(5, 3)
"greater"

>>> compare(2, 7)
"less"

>>> compare(4, 4)
"equal"
\`\`\``,
			hint1: "Tenglikni tekshirish uchun == dan, katta uchun > dan, kichik uchun < dan foydalaning.",
			hint2: "if/elif/else bilan faqat ikkita shartni tekshirish kifoya - uchinchisi nazarda tutilgan.",
			whyItMatters: `Taqqoslash operatorlari koddagi barcha qarorlarning asosidir.

**Ishlab chiqarish patterni:**

\`\`\`python
def validate_age(age: int) -> tuple[bool, str]:
    """Aniq qayta aloqa bilan yoshni tekshirish."""
    if age < 0:
        return False, "Yosh manfiy bo'lishi mumkin emas"
    elif age < 13:
        return False, "Ro'yxatdan o'tish uchun 13 yoshdan katta bo'lish kerak"
    elif age > 120:
        return False, "To'g'ri yoshni kiriting"
    return True, "Yosh to'g'ri"
\`\`\`

**Amaliy foydalari:**
- Taqqoslashlar kiritishni tekshirish imkonini beradi
- Saralash va filtrlash taqqoslashga bog'liq
- Aniq taqqoslash mantig'i kodni qo'llab-quvvatlanadigan qiladi`,
		},
	},
};

export default task;
