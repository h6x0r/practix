import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-reverse-words',
	title: 'Reverse Words in String',
	difficulty: 'medium',
	tags: ['python', 'strings', 'two-pointers', 'in-place'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Reverse the order of words in a string.

**Problem:**

Given a string \`s\`, reverse the order of words. A word is defined as a sequence of non-space characters. Words are separated by at least one space.

Return a string with words in reverse order, concatenated by single spaces.

**Examples:**

\`\`\`
Input: s = "the sky is blue"
Output: "blue is sky the"

Input: s = "  hello world  "
Output: "world hello"
Explanation: Leading/trailing spaces removed, multiple spaces reduced to one

Input: s = "a good   example"
Output: "example good a"
\`\`\`

**Approach:**

1. Split string into words (handling multiple spaces)
2. Reverse the word array
3. Join with single space

Alternative (in-place):
1. Reverse entire string
2. Reverse each word

**Time Complexity:** O(n)
**Space Complexity:** O(n) for result`,
	initialCode: `def reverse_words(s: str) -> str:
    # TODO: Reverse the order of words in a string

    return ""`,
	solutionCode: `def reverse_words(s: str) -> str:
    """
    Reverse the order of words in a string.

    Args:
        s: Input string

    Returns:
        String with words in reverse order
    """
    # Split by whitespace (handles multiple spaces, leading/trailing)
    words = s.split()

    # Reverse and join with single space
    return " ".join(reversed(words))


# Alternative: Manual implementation for learning
def reverse_words_manual(s: str) -> str:
    """Manual implementation without built-in split/join."""
    # Split into words manually
    words = []
    start = -1

    for i in range(len(s) + 1):
        if i == len(s) or s[i] == ' ':
            if start != -1:
                words.append(s[start:i])
                start = -1
        elif start == -1:
            start = i

    # Reverse using two pointers
    left, right = 0, len(words) - 1
    while left < right:
        words[left], words[right] = words[right], words[left]
        left += 1
        right -= 1

    # Join with space
    return " ".join(words)`,
	testCode: `import pytest
from solution import reverse_words

class TestReverseWords:
    def test_basic(self):
        """Test basic sentence reversal"""
        assert reverse_words("the sky is blue") == "blue is sky the"

    def test_leading_trailing(self):
        """Test with leading and trailing spaces"""
        assert reverse_words("  hello world  ") == "world hello"

    def test_multiple_spaces(self):
        """Test with multiple spaces between words"""
        assert reverse_words("a good   example") == "example good a"

    def test_single_word(self):
        """Test single word"""
        assert reverse_words("hello") == "hello"

    def test_empty(self):
        """Test empty string"""
        assert reverse_words("") == ""

    def test_only_spaces(self):
        """Test string with only spaces"""
        assert reverse_words("   ") == ""

    def test_two_words(self):
        """Test two words"""
        assert reverse_words("hello world") == "world hello"

    def test_single_char_words(self):
        """Test single character words"""
        assert reverse_words("a b c") == "c b a"

    def test_mixed_spaces_and_words(self):
        """Test mixed spaces"""
        assert reverse_words("  a   b  ") == "b a"

    def test_long_sentence(self):
        """Test longer sentence"""
        assert reverse_words("the quick brown fox") == "fox brown quick the"`,
	hint1: `Python's split() method without arguments splits on whitespace and removes empty strings. Use it to get a clean list of words from any string.`,
	hint2: `Use slicing [::-1] or reversed() to reverse the list. Then join with " ".join(). The one-liner: " ".join(s.split()[::-1])`,
	whyItMatters: `String tokenization and word manipulation are common operations.

**Why This Matters:**

**1. Text Processing**

This is fundamental for:
- Text editors (word operations)
- Search engines (query parsing)
- Natural language processing
- Command line argument parsing

**2. In-Place String Manipulation**

The in-place approach teaches a valuable technique:
\`\`\`python
# "hello world" -> "dlrow olleh" -> "world hello"
# Step 1: Reverse entire string
# Step 2: Reverse each word
\`\`\`

This pattern works for many rotation problems.

**3. Space Handling Edge Cases**

Real-world strings are messy:
\`\`\`python
"  multiple   spaces  "  # Leading, trailing, multiple
"singleword"             # No spaces
""                       # Empty
"   "                    # Only spaces
\`\`\`

Learning to handle these makes you a better programmer.

**4. Python Built-in Power**

\`\`\`python
# One-liner using Python built-ins
def reverse_words(s):
    return " ".join(s.split()[::-1])

# s.split() handles all whitespace
"  hello   world  ".split()  # ["hello", "world"]

# Reversed with slice or reversed()
words[::-1]           # Slice notation
list(reversed(words)) # Built-in function
\`\`\``,
	order: 4,
	translations: {
		ru: {
			title: 'Перевернуть слова в строке',
			description: `Переверните порядок слов в строке.

**Задача:**

Дана строка \`s\`, переверните порядок слов. Слово определяется как последовательность непробельных символов. Слова разделены хотя бы одним пробелом.

Верните строку со словами в обратном порядке, соединёнными одиночными пробелами.

**Примеры:**

\`\`\`
Вход: s = "the sky is blue"
Выход: "blue is sky the"

Вход: s = "  hello world  "
Выход: "world hello"
Объяснение: Ведущие/завершающие пробелы удалены, множественные пробелы сокращены до одного

Вход: s = "a good   example"
Выход: "example good a"
\`\`\`

**Подход:**

1. Разбить строку на слова (обрабатывая множественные пробелы)
2. Перевернуть массив слов
3. Соединить одиночным пробелом

**Временная сложность:** O(n)
**Пространственная сложность:** O(n) для результата`,
			hint1: `Создайте функцию splitWords, которая итерирует по строке, отслеживая начальные позиции слов. Когда встречаете пробел (или конец строки), если вы были в слове, извлеките его.`,
			hint2: `Используйте два указателя для переворота среза слов на месте: поменяйте words[left] с words[right], затем двигайте указатели друг к другу.`,
			whyItMatters: `Токенизация строк и манипуляция словами - распространённые операции.

**Почему это важно:**

**1. Обработка текста**

Это фундаментально для:
- Текстовых редакторов
- Поисковых систем
- Обработки естественного языка

**2. Обработка граничных случаев с пробелами**

Реальные строки беспорядочны:
- Ведущие/завершающие/множественные пробелы
- Одно слово или пустая строка`,
			solutionCode: `def reverse_words(s: str) -> str:
    """
    Переворачивает порядок слов в строке.

    Args:
        s: Входная строка

    Returns:
        Строка со словами в обратном порядке
    """
    # Разбиваем по пробелам (обрабатывает множественные пробелы)
    words = s.split()

    # Переворачиваем и соединяем одиночным пробелом
    return " ".join(reversed(words))`
		},
		uz: {
			title: 'Satrdagi so\'zlarni teskari aylantirish',
			description: `Satrdagi so'zlar tartibini teskari aylantiring.

**Masala:**

Satr \`s\` berilgan, so'zlar tartibini teskari aylantiring. So'z bo'sh bo'lmagan belgilar ketma-ketligi sifatida belgilanadi. So'zlar kamida bitta bo'sh joy bilan ajratilgan.

So'zlar teskari tartibda, yagona bo'sh joylar bilan birlashtirilgan satrni qaytaring.

**Misollar:**

\`\`\`
Kirish: s = "the sky is blue"
Chiqish: "blue is sky the"

Kirish: s = "  hello world  "
Chiqish: "world hello"
Tushuntirish: Boshidagi/oxiridagi bo'sh joylar olib tashlandi
\`\`\`

**Yondashuv:**

1. Satrni so'zlarga bo'ling (ko'p bo'sh joylarni qayta ishlang)
2. So'zlar massivini teskari aylantiring
3. Yagona bo'sh joy bilan birlashtiring

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(n) natija uchun`,
			hint1: `So'z boshi pozitsiyalarini kuzatib, satr bo'ylab takrorlanadigan splitWords funktsiyasini yarating. Bo'sh joyga (yoki satr oxiriga) duch kelganingizda, agar so'z ichida bo'lsangiz, uni ajratib oling.`,
			hint2: `So'zlar slaysini joyida teskari aylantirish uchun ikki ko'rsatkichdan foydalaning: words[left] ni words[right] bilan almashtiring, keyin ko'rsatkichlarni bir-biriga qarab siljiting.`,
			whyItMatters: `Satr tokenizatsiyasi va so'z manipulyatsiyasi keng tarqalgan operatsiyalar.

**Bu nima uchun muhim:**

**1. Matn qayta ishlash**

Bu quyidagilar uchun asosiy:
- Matn muharrirlari
- Qidiruv tizimlari
- Tabiiy tilni qayta ishlash`,
			solutionCode: `def reverse_words(s: str) -> str:
    """
    Satrdagi so'zlar tartibini teskari aylantiradi.

    Args:
        s: Kirish satri

    Returns:
        So'zlar teskari tartibda joylashgan satr
    """
    # Bo'sh joylar bo'yicha bo'lamiz (ko'p bo'sh joylarni qayta ishlaydi)
    words = s.split()

    # Teskari aylantiramiz va yagona bo'sh joy bilan birlashtiramiz
    return " ".join(reversed(words))`
		}
	}
};

export default task;
