import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-find-first',
	title: 'Find First Match',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'loops', 'break'],
	estimatedTime: '10m',
	isPremium: false,
	order: 4,

	description: `# Find First Match

Learn to use the \`break\` statement to exit a loop early.

## Task

Implement the function \`find_first_even(numbers)\` that returns the first even number in a list.

## Requirements

- Return the first even number found
- Return \`None\` if no even number exists
- Stop searching as soon as you find one (use break for efficiency)

## Examples

\`\`\`python
>>> find_first_even([1, 3, 4, 6, 8])
4

>>> find_first_even([1, 3, 5, 7])
None

>>> find_first_even([2, 4, 6])
2
\`\`\``,

	initialCode: `def find_first_even(numbers: list[int]) -> int | None:
    """Find the first even number in a list.

    Args:
        numbers: List of integers

    Returns:
        First even number, or None if not found
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def find_first_even(numbers: list[int]) -> int | None:
    """Find the first even number in a list.

    Args:
        numbers: List of integers

    Returns:
        First even number, or None if not found
    """
    for num in numbers:
        if num % 2 == 0:
            return num  # Return exits immediately
    return None`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """First even in middle"""
        self.assertEqual(find_first_even([1, 3, 4, 6, 8]), 4)

    def test_2(self):
        """No even numbers"""
        self.assertIsNone(find_first_even([1, 3, 5, 7]))

    def test_3(self):
        """First element is even"""
        self.assertEqual(find_first_even([2, 4, 6]), 2)

    def test_4(self):
        """Empty list"""
        self.assertIsNone(find_first_even([]))

    def test_5(self):
        """Single even"""
        self.assertEqual(find_first_even([4]), 4)

    def test_6(self):
        """Single odd"""
        self.assertIsNone(find_first_even([3]))

    def test_7(self):
        """Zero is even"""
        self.assertEqual(find_first_even([1, 0, 2]), 0)

    def test_8(self):
        """Negative even"""
        self.assertEqual(find_first_even([1, -2, 4]), -2)

    def test_9(self):
        """Last element is even"""
        self.assertEqual(find_first_even([1, 3, 5, 6]), 6)

    def test_10(self):
        """Large list"""
        self.assertEqual(find_first_even(list(range(1, 100, 2)) + [100]), 100)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use num % 2 == 0 to check if a number is even.',
	hint2: 'You can use return inside a loop to exit immediately when found.',

	whyItMatters: `Early exit from loops is crucial for performance when you only need the first match.

**Production Pattern:**

\`\`\`python
def find_user_by_email(users: list[dict], email: str) -> dict | None:
    """Find first user matching email (case-insensitive)."""
    email_lower = email.lower()
    for user in users:
        if user.get("email", "").lower() == email_lower:
            return user
    return None

def first_available_slot(schedule: list[dict]) -> dict | None:
    """Find first available time slot."""
    for slot in schedule:
        if not slot.get("booked", False):
            return slot
    return None
\`\`\`

**Practical Benefits:**
- Early return prevents unnecessary iterations
- Finding first match is common in search operations
- Database-like operations often need "first or none" logic`,

	translations: {
		ru: {
			title: 'Найти первое совпадение',
			description: `# Найти первое совпадение

Научитесь использовать оператор \`break\` для раннего выхода из цикла.

## Задача

Реализуйте функцию \`find_first_even(numbers)\`, которая возвращает первое чётное число в списке.

## Требования

- Верните первое найденное чётное число
- Верните \`None\`, если чётных чисел нет
- Прекратите поиск сразу после нахождения

## Примеры

\`\`\`python
>>> find_first_even([1, 3, 4, 6, 8])
4

>>> find_first_even([1, 3, 5, 7])
None

>>> find_first_even([2, 4, 6])
2
\`\`\``,
			hint1: 'Используйте num % 2 == 0 для проверки чётности.',
			hint2: 'Можно использовать return внутри цикла для немедленного выхода.',
			whyItMatters: `Ранний выход из цикла критичен для производительности.

**Продакшен паттерн:**

\`\`\`python
def find_user_by_email(users: list[dict], email: str) -> dict | None:
    """Найти пользователя по email."""
    email_lower = email.lower()
    for user in users:
        if user.get("email", "").lower() == email_lower:
            return user
    return None
\`\`\`

**Практические преимущества:**
- Ранний return предотвращает лишние итерации
- Поиск первого совпадения — частая операция`,
		},
		uz: {
			title: 'Birinchi moslikni topish',
			description: `# Birinchi moslikni topish

Sikldan erta chiqish uchun \`break\` operatoridan foydalanishni o'rganing.

## Vazifa

Ro'yxatdagi birinchi juft sonni qaytaruvchi \`find_first_even(numbers)\` funksiyasini amalga oshiring.

## Talablar

- Topilgan birinchi juft sonni qaytaring
- Agar juft son bo'lmasa, \`None\` qaytaring
- Topilgandan so'ng qidirishni to'xtating

## Misollar

\`\`\`python
>>> find_first_even([1, 3, 4, 6, 8])
4

>>> find_first_even([1, 3, 5, 7])
None

>>> find_first_even([2, 4, 6])
2
\`\`\``,
			hint1: "Juftlikni tekshirish uchun num % 2 == 0 dan foydalaning.",
			hint2: "Topilganda darhol chiqish uchun sikl ichida return dan foydalanishingiz mumkin.",
			whyItMatters: `Sikldan erta chiqish ishlash uchun muhim.

**Ishlab chiqarish patterni:**

\`\`\`python
def find_user_by_email(users: list[dict], email: str) -> dict | None:
    """Email bo'yicha foydalanuvchini topish."""
    email_lower = email.lower()
    for user in users:
        if user.get("email", "").lower() == email_lower:
            return user
    return None
\`\`\`

**Amaliy foydalari:**
- Erta return keraksiz iteratsiyalarni oldini oladi
- Birinchi moslikni qidirish tez-tez ishlatiladigan operatsiya`,
		},
	},
};

export default task;
