import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-first-unique-char',
	title: 'First Unique Character',
	difficulty: 'easy',
	tags: ['python', 'strings', 'hash-map', 'counting'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the first non-repeating character in a string.

**Problem:**

Given a string \`s\`, find the first non-repeating character and return its index. If no unique character exists, return -1.

**Examples:**

\`\`\`
Input: s = "leetcode"
Output: 0
Explanation: 'l' is the first unique character

Input: s = "loveleetcode"
Output: 2
Explanation: 'v' is the first unique character (at index 2)

Input: s = "aabb"
Output: -1
Explanation: No unique characters exist
\`\`\`

**Algorithm:**

1. Count frequency of each character (first pass)
2. Find first character with count = 1 (second pass)

**Time Complexity:** O(n)
**Space Complexity:** O(1) - only 26 letters`,
	initialCode: `def first_uniq_char(s: str) -> int:
    # TODO: Return index of first non-repeating character (-1 if none)

    return -1`,
	solutionCode: `from collections import Counter

def first_uniq_char(s: str) -> int:
    """
    Return index of first non-repeating character.
    Returns -1 if no unique character exists.

    Args:
        s: Input string

    Returns:
        Index of first unique character or -1
    """
    # Count character frequencies
    count = Counter(s)

    # Find first unique character
    for i, char in enumerate(s):
        if count[char] == 1:
            return i

    return -1


# Alternative without Counter
def first_uniq_char_manual(s: str) -> int:
    """Manual implementation using dictionary."""
    # First pass: count frequencies
    count = {}
    for char in s:
        count[char] = count.get(char, 0) + 1

    # Second pass: find first unique
    for i, char in enumerate(s):
        if count[char] == 1:
            return i

    return -1`,
	testCode: `import pytest
from solution import first_uniq_char

class TestFirstUniqChar:
    def test_basic(self):
        """Test basic case - first char is unique"""
        assert first_uniq_char("leetcode") == 0

    def test_middle(self):
        """Test unique char in middle"""
        assert first_uniq_char("loveleetcode") == 2

    def test_none(self):
        """Test no unique characters"""
        assert first_uniq_char("aabb") == -1

    def test_empty(self):
        """Test empty string"""
        assert first_uniq_char("") == -1

    def test_single(self):
        """Test single character"""
        assert first_uniq_char("a") == 0

    def test_all_same(self):
        """Test all same characters"""
        assert first_uniq_char("aaaa") == -1

    def test_last(self):
        """Test unique char at end"""
        assert first_uniq_char("aabbccd") == 6

    def test_first_unique_at_end(self):
        """Test first unique at end with multiple uniques"""
        assert first_uniq_char("aabbccde") == 6

    def test_multiple_uniques(self):
        """Test multiple unique characters"""
        assert first_uniq_char("abcabc") == -1

    def test_repeating_pattern(self):
        """Test repeating pattern"""
        assert first_uniq_char("aabbccdd") == -1`,
	hint1: `Use collections.Counter(s) to count all character frequencies in one line. Or use a dictionary with count[c] = count.get(c, 0) + 1.`,
	hint2: `In the second pass, use enumerate(s) to get both index and character. Return the index of the first character where count[char] equals 1.`,
	whyItMatters: `Two-pass algorithms with frequency counting are very common.

**Why This Matters:**

**1. Two-Pass Pattern**

Many problems require two passes:
\`\`\`python
from collections import Counter

# Pass 1: Gather information
count = Counter(s)

# Pass 2: Use information
for i, c in enumerate(s):
    if count[c] == 1:
        return i
\`\`\`

**2. Python Counter Class**

\`\`\`python
from collections import Counter

# Counter is a powerful tool
count = Counter("aabbcc")  # {'a': 2, 'b': 2, 'c': 2}
count.most_common(1)       # [('a', 2)]
count['x']                 # 0 (missing keys return 0)
\`\`\`

**3. Variations**

This pattern solves many similar problems:
- First repeating character
- All unique characters
- Most frequent character
- Characters that appear exactly K times

**4. Real-World Applications**

- Stream processing (first unique in data stream)
- Database queries (finding unique records)
- Log analysis (first occurrence of error)
- Text analysis (rare words detection)`,
	order: 6,
	translations: {
		ru: {
			title: 'Первый уникальный символ',
			description: `Найдите первый неповторяющийся символ в строке.

**Задача:**

Дана строка \`s\`, найдите первый неповторяющийся символ и верните его индекс. Если уникального символа нет, верните -1.

**Примеры:**

\`\`\`
Вход: s = "leetcode"
Выход: 0
Объяснение: 'l' - первый уникальный символ

Вход: s = "loveleetcode"
Выход: 2
Объяснение: 'v' - первый уникальный символ (индекс 2)

Вход: s = "aabb"
Выход: -1
Объяснение: Уникальных символов нет
\`\`\`

**Алгоритм:**

1. Подсчитать частоту каждого символа (первый проход)
2. Найти первый символ с count = 1 (второй проход)

**Временная сложность:** O(n)
**Пространственная сложность:** O(1) - только 26 букв`,
			hint1: `Используйте collections.Counter(s) для подсчёта всех частот символов в одну строку. Или используйте словарь с count[c] = count.get(c, 0) + 1.`,
			hint2: `Во втором проходе используйте enumerate(s), чтобы получить и индекс, и символ. Верните индекс первого символа, где count[char] равен 1.`,
			whyItMatters: `Двухпроходные алгоритмы с подсчётом частот очень распространены.

**Почему это важно:**

**1. Паттерн двух проходов**

Многие задачи требуют двух проходов:
- Проход 1: Собрать информацию
- Проход 2: Использовать информацию

**2. Вариации**

Этот паттерн решает много похожих задач:
- Первый повторяющийся символ
- Все уникальные символы
- Наиболее частый символ`,
			solutionCode: `from collections import Counter

def first_uniq_char(s: str) -> int:
    """
    Возвращает индекс первого неповторяющегося символа.
    Возвращает -1 если уникального символа нет.

    Args:
        s: Входная строка

    Returns:
        Индекс первого уникального символа или -1
    """
    # Подсчитываем частоты символов
    count = Counter(s)

    # Находим первый уникальный символ
    for i, char in enumerate(s):
        if count[char] == 1:
            return i

    return -1`
		},
		uz: {
			title: 'Birinchi noyob belgi',
			description: `Satrdagi birinchi takrorlanmaydigan belgini toping.

**Masala:**

Satr \`s\` berilgan, birinchi takrorlanmaydigan belgini toping va uning indeksini qaytaring. Agar noyob belgi mavjud bo'lmasa, -1 qaytaring.

**Misollar:**

\`\`\`
Kirish: s = "leetcode"
Chiqish: 0
Tushuntirish: 'l' birinchi noyob belgi

Kirish: s = "loveleetcode"
Chiqish: 2
Tushuntirish: 'v' birinchi noyob belgi (indeks 2)

Kirish: s = "aabb"
Chiqish: -1
Tushuntirish: Noyob belgilar mavjud emas
\`\`\`

**Algoritm:**

1. Har bir belgining chastotasini hisoblang (birinchi o'tish)
2. count = 1 bo'lgan birinchi belgini toping (ikkinchi o'tish)

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1) - faqat 26 harf`,
			hint1: `Barcha belgi chastotalarini bir qatorda hisoblash uchun collections.Counter(s) dan foydalaning. Yoki lug'atdan count[c] = count.get(c, 0) + 1 bilan foydalaning.`,
			hint2: `Ikkinchi o'tishda indeks va belgini olish uchun enumerate(s) dan foydalaning. count[char] 1 ga teng bo'lgan birinchi belgining indeksini qaytaring.`,
			whyItMatters: `Chastota hisoblash bilan ikki o'tishli algoritmlar juda keng tarqalgan.

**Bu nima uchun muhim:**

**1. Ikki o'tish patterni**

Ko'p masalalar ikki o'tishni talab qiladi:
- 1-o'tish: Ma'lumot to'plash
- 2-o'tish: Ma'lumotdan foydalanish

**2. Variatsiyalar**

Bu pattern ko'p o'xshash masalalarni yechadi:
- Birinchi takrorlanuvchi belgi
- Barcha noyob belgilar
- Eng ko'p uchraydigan belgi`,
			solutionCode: `from collections import Counter

def first_uniq_char(s: str) -> int:
    """
    Birinchi takrorlanmaydigan belgining indeksini qaytaradi.
    Agar noyob belgi mavjud bo'lmasa -1 qaytaradi.

    Args:
        s: Kirish satri

    Returns:
        Birinchi noyob belgining indeksi yoki -1
    """
    # Belgi chastotalarini hisoblaymiz
    count = Counter(s)

    # Birinchi noyob belgini topamiz
    for i, char in enumerate(s):
        if count[char] == 1:
            return i

    return -1`
		}
	}
};

export default task;
