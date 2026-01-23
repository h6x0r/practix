import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-palindrome',
	title: 'Palindrome Check',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'strings', 'loops'],
	estimatedTime: '10m',
	isPremium: false,
	order: 8,

	description: `# Palindrome Check

A palindrome reads the same forwards and backwards.

## Task

Implement the function \`is_palindrome(text)\` that checks if a string is a palindrome.

## Requirements

- Ignore case (treat "A" and "a" as equal)
- Ignore spaces and punctuation
- Consider only alphanumeric characters
- Empty strings and single characters are palindromes

## Examples

\`\`\`python
>>> is_palindrome("racecar")
True

>>> is_palindrome("A man a plan a canal Panama")
True  # Ignoring spaces and case

>>> is_palindrome("hello")
False

>>> is_palindrome("Was it a car or a cat I saw?")
True  # Ignoring punctuation
\`\`\``,

	initialCode: `def is_palindrome(text: str) -> bool:
    """Check if a string is a palindrome.

    Args:
        text: Input string (may contain spaces and punctuation)

    Returns:
        True if the string is a palindrome (ignoring case,
        spaces, and non-alphanumeric characters)
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def is_palindrome(text: str) -> bool:
    """Check if a string is a palindrome.

    Args:
        text: Input string (may contain spaces and punctuation)

    Returns:
        True if the string is a palindrome (ignoring case,
        spaces, and non-alphanumeric characters)
    """
    # Step 1: Clean the string - keep only alphanumeric, lowercase
    cleaned = ""
    for char in text.lower():
        if char.isalnum():
            cleaned += char

    # Step 2: Compare string with its reverse
    # Slicing with [::-1] reverses the string
    return cleaned == cleaned[::-1]`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Simple palindrome"""
        self.assertTrue(is_palindrome("racecar"))

    def test_2(self):
        """Famous phrase with spaces"""
        self.assertTrue(is_palindrome("A man a plan a canal Panama"))

    def test_3(self):
        """Not a palindrome"""
        self.assertFalse(is_palindrome("hello"))

    def test_4(self):
        """With punctuation"""
        self.assertTrue(is_palindrome("Was it a car or a cat I saw?"))

    def test_5(self):
        """Empty string is palindrome"""
        self.assertTrue(is_palindrome(""))

    def test_6(self):
        """Single character"""
        self.assertTrue(is_palindrome("a"))

    def test_7(self):
        """Two same characters"""
        self.assertTrue(is_palindrome("aa"))

    def test_8(self):
        """Two different characters"""
        self.assertFalse(is_palindrome("ab"))

    def test_9(self):
        """Numbers as palindrome"""
        self.assertTrue(is_palindrome("12321"))

    def test_10(self):
        """Mixed alphanumeric"""
        self.assertTrue(is_palindrome("A1b2B1a"))

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'First clean the string: convert to lowercase and keep only letters and numbers using isalnum().',
	hint2: 'Python slicing [::-1] reverses a string. Compare cleaned string with its reverse.',

	whyItMatters: `String manipulation and comparison are fundamental to text processing and validation.

**Production Pattern:**

\`\`\`python
def normalize_for_comparison(text: str) -> str:
    """Normalize text for consistent comparison."""
    # Remove non-alphanumeric, lowercase
    return "".join(c.lower() for c in text if c.isalnum())

def find_longest_palindrome_substring(s: str) -> str:
    """Find the longest palindromic substring."""
    if len(s) < 2:
        return s

    def expand_around_center(left: int, right: int) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    longest = ""
    for i in range(len(s)):
        # Odd length palindromes
        odd = expand_around_center(i, i)
        if len(odd) > len(longest):
            longest = odd

        # Even length palindromes
        even = expand_around_center(i, i + 1)
        if len(even) > len(longest):
            longest = even

    return longest
\`\`\`

**Practical Benefits:**
- Text processing often requires normalization
- Understanding string reversal helps with algorithms
- Pattern matching is essential for search functionality`,

	translations: {
		ru: {
			title: 'Проверка палиндрома',
			description: `# Проверка палиндрома

Палиндром читается одинаково слева направо и справа налево.

## Задача

Реализуйте функцию \`is_palindrome(text)\`, которая проверяет, является ли строка палиндромом.

## Требования

- Игнорируйте регистр ("A" и "a" равны)
- Игнорируйте пробелы и пунктуацию
- Учитывайте только буквенно-цифровые символы
- Пустые строки и одиночные символы — палиндромы

## Примеры

\`\`\`python
>>> is_palindrome("racecar")
True

>>> is_palindrome("A man a plan a canal Panama")
True  # Игнорируя пробелы и регистр

>>> is_palindrome("hello")
False

>>> is_palindrome("Was it a car or a cat I saw?")
True  # Игнорируя пунктуацию
\`\`\``,
			hint1: 'Сначала очистите строку: переведите в нижний регистр, оставьте только буквы и цифры (isalnum()).',
			hint2: 'Срез [::-1] в Python разворачивает строку. Сравните очищенную строку с её реверсом.',
			whyItMatters: `Манипуляции и сравнение строк — основа обработки текста и валидации.

**Продакшен паттерн:**

\`\`\`python
def normalize_for_comparison(text: str) -> str:
    """Нормализация текста для сравнения."""
    return "".join(c.lower() for c in text if c.isalnum())

def find_longest_palindrome_substring(s: str) -> str:
    """Поиск самой длинной палиндромной подстроки."""
    if len(s) < 2:
        return s
    # ... алгоритм расширения от центра
\`\`\`

**Практические преимущества:**
- Обработка текста часто требует нормализации
- Понимание разворота строк помогает с алгоритмами`,
		},
		uz: {
			title: 'Palindrom tekshiruvi',
			description: `# Palindrom tekshiruvi

Palindrom chapdan o'ngga va o'ngdan chapga bir xil o'qiladi.

## Vazifa

Satrning palindrom ekanligini tekshiruvchi \`is_palindrome(text)\` funksiyasini amalga oshiring.

## Talablar

- Registrni e'tiborsiz qoldiring ("A" va "a" teng)
- Bo'shliqlar va tinish belgilarini e'tiborsiz qoldiring
- Faqat harf-raqam belgilarini hisobga oling
- Bo'sh satrlar va bitta belgi palindrom hisoblanadi

## Misollar

\`\`\`python
>>> is_palindrome("racecar")
True

>>> is_palindrome("A man a plan a canal Panama")
True  # Bo'shliq va registrni hisobga olmasdan

>>> is_palindrome("hello")
False

>>> is_palindrome("Was it a car or a cat I saw?")
True  # Tinish belgilarini hisobga olmasdan
\`\`\``,
			hint1: "Avval satrni tozalang: kichik harfga o'tkazing, faqat harflar va raqamlarni qoldiring (isalnum()).",
			hint2: "Python kesimi [::-1] satrni teskari aylantiradi. Tozalangan satrni teskarisiga taqqoslang.",
			whyItMatters: `Satrlar bilan ishlash va taqqoslash matn qayta ishlash va tekshiruvning asosidir.

**Ishlab chiqarish patterni:**

\`\`\`python
def normalize_for_comparison(text: str) -> str:
    """Taqqoslash uchun matnni normallashtirish."""
    return "".join(c.lower() for c in text if c.isalnum())

def find_longest_palindrome_substring(s: str) -> str:
    """Eng uzun palindrom pastki satrni topish."""
    if len(s) < 2:
        return s
    # ... markazdan kengayish algoritmi
\`\`\`

**Amaliy foydalari:**
- Matn qayta ishlash ko'pincha normallashtirishni talab qiladi
- Satr teskarilashni tushunish algoritmlarga yordam beradi`,
		},
	},
};

export default task;
