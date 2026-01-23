import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-string-methods',
	title: 'String Methods',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'strings', 'methods'],
	estimatedTime: '15m',
	isPremium: false,
	order: 7,

	description: `# String Methods

Learn essential string methods: \`split()\`, \`join()\`, \`strip()\`, \`replace()\`.

## Task

Implement the function \`clean_and_format(text)\` that cleans and formats a messy string.

## Requirements

1. Remove leading/trailing whitespace
2. Replace multiple spaces with single space
3. Capitalize each word (title case)
4. Return the cleaned string

## Examples

\`\`\`python
>>> clean_and_format("  hello   world  ")
"Hello World"

>>> clean_and_format("python  is   awesome")
"Python Is Awesome"

>>> clean_and_format("   ")
""

>>> clean_and_format("HELLO")
"Hello"
\`\`\``,

	initialCode: `def clean_and_format(text: str) -> str:
    """Clean and format a messy string.

    Operations performed:
    1. Remove leading/trailing whitespace
    2. Replace multiple spaces with single space
    3. Capitalize each word (title case)

    Args:
        text: Input string that may have extra whitespace

    Returns:
        Cleaned and formatted string
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def clean_and_format(text: str) -> str:
    """Clean and format a messy string.

    Operations performed:
    1. Remove leading/trailing whitespace
    2. Replace multiple spaces with single space
    3. Capitalize each word (title case)

    Args:
        text: Input string that may have extra whitespace

    Returns:
        Cleaned and formatted string
    """
    # Step 1: Remove leading/trailing whitespace
    text = text.strip()

    # Handle empty string after stripping
    if not text:
        return ""

    # Step 2: Split by whitespace (handles multiple spaces)
    # split() without args splits on any whitespace and removes empty strings
    words = text.split()

    # Step 3: Capitalize each word and join with single space
    cleaned = " ".join(word.capitalize() for word in words)

    return cleaned`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic case with extra spaces"""
        self.assertEqual(clean_and_format("  hello   world  "), "Hello World")

    def test_2(self):
        """Multiple words with extra spaces"""
        self.assertEqual(clean_and_format("python  is   awesome"), "Python Is Awesome")

    def test_3(self):
        """Only whitespace returns empty"""
        self.assertEqual(clean_and_format("   "), "")

    def test_4(self):
        """All uppercase input"""
        self.assertEqual(clean_and_format("HELLO"), "Hello")

    def test_5(self):
        """Empty string"""
        self.assertEqual(clean_and_format(""), "")

    def test_6(self):
        """Single word with spaces"""
        self.assertEqual(clean_and_format("   python   "), "Python")

    def test_7(self):
        """Tabs and newlines"""
        self.assertEqual(clean_and_format("hello\\tworld\\n"), "Hello World")

    def test_8(self):
        """Mixed case input"""
        self.assertEqual(clean_and_format("hElLo WoRlD"), "Hello World")

    def test_9(self):
        """Numbers in text"""
        self.assertEqual(clean_and_format("  python  3  "), "Python 3")

    def test_10(self):
        """Special characters preserved"""
        self.assertEqual(clean_and_format("  hello-world  "), "Hello-world")

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use strip() to remove leading/trailing whitespace, then split() to break into words.',
	hint2: 'split() without arguments splits on any whitespace and ignores multiple spaces.',

	whyItMatters: `String manipulation is one of the most common programming tasks.

**Production Pattern:**

\`\`\`python
def normalize_name(raw_name: str) -> str:
    """Normalize user input for consistent storage."""
    # Remove extra whitespace
    name = " ".join(raw_name.split())

    # Handle special cases
    if not name:
        return ""

    # Title case, but preserve certain patterns
    parts = []
    for word in name.split():
        if word.lower() in ("van", "de", "la"):
            parts.append(word.lower())
        elif word.upper() == word and len(word) <= 3:
            parts.append(word)  # Keep acronyms
        else:
            parts.append(word.capitalize())

    return " ".join(parts)

def create_slug(title: str) -> str:
    """Create URL-safe slug from title."""
    # Normalize whitespace
    slug = " ".join(title.lower().split())
    # Replace spaces with hyphens
    slug = slug.replace(" ", "-")
    # Remove non-alphanumeric except hyphens
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    return slug
\`\`\`

**Practical Benefits:**
- Data cleaning is essential for consistent storage
- User input always needs normalization
- URL slugs require specific formatting rules`,

	translations: {
		ru: {
			title: 'Методы строк',
			description: `# Методы строк

Изучите основные методы строк: \`split()\`, \`join()\`, \`strip()\`, \`replace()\`.

## Задача

Реализуйте функцию \`clean_and_format(text)\`, которая очищает и форматирует строку.

## Требования

1. Удалите пробелы в начале и конце
2. Замените множественные пробелы одинарными
3. Сделайте каждое слово с заглавной буквы (Title Case)
4. Верните очищенную строку

## Примеры

\`\`\`python
>>> clean_and_format("  hello   world  ")
"Hello World"

>>> clean_and_format("python  is   awesome")
"Python Is Awesome"

>>> clean_and_format("   ")
""

>>> clean_and_format("HELLO")
"Hello"
\`\`\``,
			hint1: 'Используйте strip() для удаления пробелов по краям, затем split() для разбиения.',
			hint2: 'split() без аргументов разбивает по любым пробелам и игнорирует множественные.',
			whyItMatters: `Работа со строками — одна из самых частых задач в программировании.

**Продакшен паттерн:**

\`\`\`python
def normalize_name(raw_name: str) -> str:
    """Нормализация пользовательского ввода."""
    name = " ".join(raw_name.split())
    if not name:
        return ""
    return " ".join(word.capitalize() for word in name.split())

def create_slug(title: str) -> str:
    """Создание URL-безопасного слага."""
    slug = " ".join(title.lower().split())
    slug = slug.replace(" ", "-")
    return "".join(c for c in slug if c.isalnum() or c == "-")
\`\`\`

**Практические преимущества:**
- Очистка данных необходима для консистентного хранения
- Пользовательский ввод всегда нужно нормализовать`,
		},
		uz: {
			title: 'Satr metodlari',
			description: `# Satr metodlari

Asosiy satr metodlarini o'rganing: \`split()\`, \`join()\`, \`strip()\`, \`replace()\`.

## Vazifa

Satrni tozalovchi va formatlovchi \`clean_and_format(text)\` funksiyasini amalga oshiring.

## Talablar

1. Bosh va oxiridagi bo'sh joylarni olib tashlang
2. Bir nechta bo'shliqlarni bitta bilan almashtiring
3. Har bir so'zni bosh harf bilan yozing (Title Case)
4. Tozalangan satrni qaytaring

## Misollar

\`\`\`python
>>> clean_and_format("  hello   world  ")
"Hello World"

>>> clean_and_format("python  is   awesome")
"Python Is Awesome"

>>> clean_and_format("   ")
""

>>> clean_and_format("HELLO")
"Hello"
\`\`\``,
			hint1: "Chetdagi bo'shliqlarni olib tashlash uchun strip(), so'ng split() dan foydalaning.",
			hint2: "Argumentsiz split() har qanday bo'shliqda ajratadi va bir nechtasini e'tiborsiz qoldiradi.",
			whyItMatters: `Satrlar bilan ishlash dasturlashdagi eng keng tarqalgan vazifalardan biri.

**Ishlab chiqarish patterni:**

\`\`\`python
def normalize_name(raw_name: str) -> str:
    """Foydalanuvchi kiritishini normallashtirish."""
    name = " ".join(raw_name.split())
    if not name:
        return ""
    return " ".join(word.capitalize() for word in name.split())

def create_slug(title: str) -> str:
    """URL uchun xavfsiz slug yaratish."""
    slug = " ".join(title.lower().split())
    slug = slug.replace(" ", "-")
    return "".join(c for c in slug if c.isalnum() or c == "-")
\`\`\`

**Amaliy foydalari:**
- Ma'lumotlarni tozalash izchil saqlash uchun zarur
- Foydalanuvchi kiritishi doimo normallashtirilishi kerak`,
		},
	},
};

export default task;
