import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-count-vowels',
	title: 'Count Vowels',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'loops', 'strings'],
	estimatedTime: '10m',
	isPremium: false,
	order: 3,

	description: `# Count Vowels

Practice iterating over strings and using the \`in\` operator.

## Task

Implement the function \`count_vowels(text)\` that counts the number of vowels in a string.

## Requirements

- Count both uppercase and lowercase vowels (a, e, i, o, u)
- Return 0 for empty strings

## Examples

\`\`\`python
>>> count_vowels("hello")
2  # e, o

>>> count_vowels("HELLO")
2  # E, O

>>> count_vowels("rhythm")
0
\`\`\``,

	initialCode: `def count_vowels(text: str) -> int:
    """Count the number of vowels in a string.

    Args:
        text: Input string

    Returns:
        Number of vowels (a, e, i, o, u)
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def count_vowels(text: str) -> int:
    """Count the number of vowels in a string.

    Args:
        text: Input string

    Returns:
        Number of vowels (a, e, i, o, u)
    """
    vowels = "aeiouAEIOU"
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic lowercase"""
        self.assertEqual(count_vowels("hello"), 2)

    def test_2(self):
        """Basic uppercase"""
        self.assertEqual(count_vowels("HELLO"), 2)

    def test_3(self):
        """No vowels"""
        self.assertEqual(count_vowels("rhythm"), 0)

    def test_4(self):
        """Empty string"""
        self.assertEqual(count_vowels(""), 0)

    def test_5(self):
        """All vowels"""
        self.assertEqual(count_vowels("aeiou"), 5)

    def test_6(self):
        """Mixed case"""
        self.assertEqual(count_vowels("HeLLo WoRLd"), 3)

    def test_7(self):
        """Numbers and special chars"""
        self.assertEqual(count_vowels("h3ll0 w0rld!"), 0)

    def test_8(self):
        """Long text"""
        self.assertEqual(count_vowels("The quick brown fox"), 5)

    def test_9(self):
        """Only vowels mixed case"""
        self.assertEqual(count_vowels("AeIoU"), 5)

    def test_10(self):
        """Spaces only"""
        self.assertEqual(count_vowels("   "), 0)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Create a string of vowels "aeiouAEIOU" and use the `in` operator to check membership.',
	hint2: 'Loop through each character in the text: for char in text:',

	whyItMatters: `String iteration and membership testing are used constantly in text processing.

**Production Pattern:**

\`\`\`python
def sanitize_input(text: str, allowed_chars: str = "abcdefghijklmnopqrstuvwxyz0123456789") -> str:
    """Remove characters not in allowed set."""
    return "".join(char for char in text.lower() if char in allowed_chars)

def count_char_types(text: str) -> dict:
    """Analyze character composition."""
    return {
        "letters": sum(1 for c in text if c.isalpha()),
        "digits": sum(1 for c in text if c.isdigit()),
        "spaces": sum(1 for c in text if c.isspace()),
        "other": sum(1 for c in text if not c.isalnum() and not c.isspace()),
    }
\`\`\`

**Practical Benefits:**
- Text validation requires character checking
- Search and filtering need membership tests
- Analytics depend on character counting`,

	translations: {
		ru: {
			title: 'Подсчёт гласных',
			description: `# Подсчёт гласных

Практика итерации по строкам и использования оператора \`in\`.

## Задача

Реализуйте функцию \`count_vowels(text)\`, которая подсчитывает количество гласных в строке.

## Требования

- Считайте гласные в обоих регистрах (a, e, i, o, u)
- Для пустых строк верните 0

## Примеры

\`\`\`python
>>> count_vowels("hello")
2  # e, o

>>> count_vowels("HELLO")
2  # E, O

>>> count_vowels("rhythm")
0
\`\`\``,
			hint1: 'Создайте строку гласных "aeiouAEIOU" и используйте оператор `in`.',
			hint2: 'Пройдитесь по каждому символу: for char in text:',
			whyItMatters: `Итерация по строкам используется постоянно в обработке текста.

**Продакшен паттерн:**

\`\`\`python
def sanitize_input(text: str, allowed_chars: str = "abcdefghijklmnopqrstuvwxyz0123456789") -> str:
    """Удаление недопустимых символов."""
    return "".join(char for char in text.lower() if char in allowed_chars)
\`\`\`

**Практические преимущества:**
- Валидация текста требует проверки символов
- Поиск и фильтрация нуждаются в проверке принадлежности`,
		},
		uz: {
			title: 'Unlillarni hisoblash',
			description: `# Unlillarni hisoblash

Satrlar bo'yicha iteratsiya va \`in\` operatoridan foydalanish mashqi.

## Vazifa

Satrdagi unlillar sonini hisoblovchi \`count_vowels(text)\` funksiyasini amalga oshiring.

## Talablar

- Katta va kichik harfli unlillarni hisoblang (a, e, i, o, u)
- Bo'sh satrlar uchun 0 qaytaring

## Misollar

\`\`\`python
>>> count_vowels("hello")
2  # e, o

>>> count_vowels("HELLO")
2  # E, O

>>> count_vowels("rhythm")
0
\`\`\``,
			hint1: 'Unlillar satrini yarating "aeiouAEIOU" va `in` operatoridan foydalaning.',
			hint2: "Har bir belgi bo'ylab yuring: for char in text:",
			whyItMatters: `Satrlar bo'yicha iteratsiya matn qayta ishlashda doimiy ishlatiladi.

**Ishlab chiqarish patterni:**

\`\`\`python
def sanitize_input(text: str, allowed_chars: str = "abcdefghijklmnopqrstuvwxyz0123456789") -> str:
    """Ruxsat etilmagan belgilarni olib tashlash."""
    return "".join(char for char in text.lower() if char in allowed_chars)
\`\`\`

**Amaliy foydalari:**
- Matn tekshiruvi belgilarni tekshirishni talab qiladi
- Qidirish va filtrlash tegishlilik tekshiruvini talab qiladi`,
		},
	},
};

export default task;
