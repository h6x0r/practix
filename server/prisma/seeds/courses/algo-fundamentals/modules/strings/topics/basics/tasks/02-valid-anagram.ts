import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-valid-anagram',
	title: 'Valid Anagram',
	difficulty: 'easy',
	tags: ['python', 'strings', 'hash-map', 'counting'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Determine if two strings are anagrams of each other.

**Problem:**

Given two strings \`s\` and \`t\`, return \`True\` if \`t\` is an anagram of \`s\`, and \`False\` otherwise.

An anagram uses exactly the same characters with the same frequencies.

**Examples:**

\`\`\`
Input: s = "anagram", t = "nagaram"
Output: True

Input: s = "rat", t = "car"
Output: False

Input: s = "listen", t = "silent"
Output: True
\`\`\`

**Counting Approach:**

1. If lengths differ, return False
2. Count character frequencies in first string
3. Decrement counts for second string
4. All counts should be zero

\`\`\`python
from collections import Counter
return Counter(s) == Counter(t)

# Or manually:
count = {}
for c in s:
    count[c] = count.get(c, 0) + 1
for c in t:
    count[c] = count.get(c, 0) - 1
return all(v == 0 for v in count.values())
\`\`\`

**Time Complexity:** O(n)
**Space Complexity:** O(1) - only 26 letters`,
	initialCode: `def is_anagram(s: str, t: str) -> bool:
    # TODO: Check if t is an anagram of s

    return False`,
	solutionCode: `def is_anagram(s: str, t: str) -> bool:
    """
    Check if t is an anagram of s.

    Args:
        s: First string
        t: Second string

    Returns:
        True if t is an anagram of s, False otherwise
    """
    # Quick length check
    if len(s) != len(t):
        return False

    # Count character frequencies
    count = {}

    # Count characters in s, decrement for t
    for i in range(len(s)):
        count[s[i]] = count.get(s[i], 0) + 1
        count[t[i]] = count.get(t[i], 0) - 1

    # Check all counts are zero
    return all(c == 0 for c in count.values())`,
	testCode: `import pytest
from solution import is_anagram

class TestIsAnagram:
    def test_basic_true(self):
        """Test basic anagram"""
        assert is_anagram("anagram", "nagaram") == True

    def test_basic_false(self):
        """Test non-anagram"""
        assert is_anagram("rat", "car") == False

    def test_classic_anagram(self):
        """Test classic listen/silent"""
        assert is_anagram("listen", "silent") == True

    def test_different_length(self):
        """Test strings with different lengths"""
        assert is_anagram("ab", "abc") == False

    def test_empty_strings(self):
        """Test empty strings"""
        assert is_anagram("", "") == True

    def test_single_char_same(self):
        """Test single same character"""
        assert is_anagram("a", "a") == True

    def test_single_char_diff(self):
        """Test single different character"""
        assert is_anagram("a", "b") == False

    def test_repeated_chars(self):
        """Test with repeated characters"""
        assert is_anagram("aabb", "bbaa") == True

    def test_same_letters_diff_count(self):
        """Test same letters but different counts"""
        assert is_anagram("aab", "abb") == False

    def test_unicode_anagram(self):
        """Test with unicode characters"""
        assert is_anagram("café", "éfac") == True`,
	hint1: `First check if lengths are equal. Then use a dictionary to count character frequencies. Use dict.get(key, 0) to handle missing keys.`,
	hint2: `You can increment counts for the first string and decrement for the second in a single loop. After the loop, check if all counts are zero using all().`,
	whyItMatters: `Character counting is a fundamental technique for many string problems.

**Why This Matters:**

**1. Frequency Counting Pattern**

This pattern solves many problems:
- Group Anagrams
- Find All Anagrams in String
- Minimum Window Substring
- Permutation in String

\`\`\`python
# Generic pattern
from collections import Counter
count = Counter(s)

# Or manually
count = {}
for c in s:
    count[c] = count.get(c, 0) + 1
\`\`\`

**2. Python Counter Class**

\`\`\`python
from collections import Counter

# One-liner anagram check
def is_anagram(s, t):
    return Counter(s) == Counter(t)

# Counter operations
c = Counter("aabbcc")  # {'a': 2, 'b': 2, 'c': 2}
c.most_common(2)       # [('a', 2), ('b', 2)]
c.subtract("abc")      # {'a': 1, 'b': 1, 'c': 1}
\`\`\`

**3. Single Pass Optimization**

\`\`\`python
# Two passes
for c in s: count[c] = count.get(c, 0) + 1
for c in t: count[c] = count.get(c, 0) - 1

# Single pass (when same length)
for i in range(len(s)):
    count[s[i]] = count.get(s[i], 0) + 1
    count[t[i]] = count.get(t[i], 0) - 1
\`\`\`

**4. Real-World Applications**

- Spell checkers (word suggestions)
- Plagiarism detection
- DNA sequence analysis
- Cryptography (frequency analysis)`,
	order: 2,
	translations: {
		ru: {
			title: 'Проверка анаграммы',
			description: `Определите, являются ли две строки анаграммами друг друга.

**Задача:**

Даны две строки \`s\` и \`t\`, верните \`True\`, если \`t\` является анаграммой \`s\`, и \`False\` в противном случае.

Анаграмма использует точно те же символы с теми же частотами.

**Примеры:**

\`\`\`
Вход: s = "anagram", t = "nagaram"
Выход: True

Вход: s = "rat", t = "car"
Выход: False

Вход: s = "listen", t = "silent"
Выход: True
\`\`\`

**Подход подсчёта:**

1. Если длины различаются, вернуть False
2. Подсчитать частоты символов в первой строке
3. Уменьшить счётчики для второй строки
4. Все счётчики должны быть нулевыми

\`\`\`python
from collections import Counter
return Counter(s) == Counter(t)

# Или вручную:
count = {}
for c in s:
    count[c] = count.get(c, 0) + 1
for c in t:
    count[c] = count.get(c, 0) - 1
return all(v == 0 for v in count.values())
\`\`\`

**Временная сложность:** O(n)
**Пространственная сложность:** O(1) - только 26 букв`,
			hint1: `Сначала проверьте равенство длин. Затем используйте словарь для подсчёта частот символов. Используйте dict.get(key, 0) для обработки отсутствующих ключей.`,
			hint2: `Вы можете увеличивать счётчики для первой строки и уменьшать для второй в одном цикле. После цикла проверьте, что все счётчики равны нулю, используя all().`,
			whyItMatters: `Подсчёт символов - фундаментальная техника для многих задач со строками.

**Почему это важно:**

**1. Паттерн подсчёта частот**

Этот паттерн решает много задач:
- Group Anagrams
- Find All Anagrams in String
- Minimum Window Substring
- Permutation in String

\`\`\`python
# Общий паттерн
from collections import Counter
count = Counter(s)

# Или вручную
count = {}
for c in s:
    count[c] = count.get(c, 0) + 1
\`\`\`

**2. Класс Counter в Python**

\`\`\`python
from collections import Counter

# Однострочная проверка анаграммы
def is_anagram(s, t):
    return Counter(s) == Counter(t)

# Операции Counter
c = Counter("aabbcc")  # {'a': 2, 'b': 2, 'c': 2}
c.most_common(2)       # [('a', 2), ('b', 2)]
\`\`\`

**3. Применения в реальном мире**

- Проверка орфографии (предложения слов)
- Обнаружение плагиата
- Анализ ДНК последовательностей
- Криптография (частотный анализ)`,
			solutionCode: `def is_anagram(s: str, t: str) -> bool:
    """
    Проверяет, является ли t анаграммой s.

    Args:
        s: Первая строка
        t: Вторая строка

    Returns:
        True если t - анаграмма s, иначе False
    """
    # Быстрая проверка длины
    if len(s) != len(t):
        return False

    # Подсчитываем частоты символов
    count = {}

    # Подсчитываем символы в s, уменьшаем для t
    for i in range(len(s)):
        count[s[i]] = count.get(s[i], 0) + 1
        count[t[i]] = count.get(t[i], 0) - 1

    # Проверяем, что все счётчики равны нулю
    return all(c == 0 for c in count.values())`
		},
		uz: {
			title: 'Anagrammani tekshirish',
			description: `Ikki satr bir-birining anagrammasi ekanligini aniqlang.

**Masala:**

Ikki satr \`s\` va \`t\` berilgan, agar \`t\` \`s\` ning anagrammasi bo'lsa \`True\`, aks holda \`False\` qaytaring.

Anagramma aynan bir xil belgilarni bir xil chastotalar bilan ishlatadi.

**Misollar:**

\`\`\`
Kirish: s = "anagram", t = "nagaram"
Chiqish: True

Kirish: s = "rat", t = "car"
Chiqish: False

Kirish: s = "listen", t = "silent"
Chiqish: True
\`\`\`

**Hisoblash yondashuvi:**

1. Agar uzunliklar farq qilsa, False qaytaring
2. Birinchi satrdagi belgilar chastotasini hisoblang
3. Ikkinchi satr uchun hisoblagichlarni kamaytiring
4. Barcha hisoblagichlar nol bo'lishi kerak

\`\`\`python
from collections import Counter
return Counter(s) == Counter(t)

# Yoki qo'lda:
count = {}
for c in s:
    count[c] = count.get(c, 0) + 1
for c in t:
    count[c] = count.get(c, 0) - 1
return all(v == 0 for v in count.values())
\`\`\`

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1) - faqat 26 harf`,
			hint1: `Avval uzunliklar tengligini tekshiring. Keyin belgilar chastotasini hisoblash uchun lug'atdan foydalaning. Yo'q kalitlarni qayta ishlash uchun dict.get(key, 0) dan foydalaning.`,
			hint2: `Birinchi satr uchun hisoblagichlarni oshirish va ikkinchisi uchun kamaytirishni bitta tsiklda qilishingiz mumkin. Tsikldan keyin all() yordamida barcha hisoblagichlar nol ekanligini tekshiring.`,
			whyItMatters: `Belgilarni hisoblash ko'plab satr masalalari uchun asosiy texnika.

**Bu nima uchun muhim:**

**1. Chastota hisoblash patterni**

Bu pattern ko'p masalalarni yechadi:
- Group Anagrams
- Find All Anagrams in String
- Minimum Window Substring
- Permutation in String

\`\`\`python
# Umumiy pattern
from collections import Counter
count = Counter(s)

# Yoki qo'lda
count = {}
for c in s:
    count[c] = count.get(c, 0) + 1
\`\`\`

**2. Python Counter klassi**

\`\`\`python
from collections import Counter

# Bir qatorli anagramma tekshiruvi
def is_anagram(s, t):
    return Counter(s) == Counter(t)

# Counter operatsiyalari
c = Counter("aabbcc")  # {'a': 2, 'b': 2, 'c': 2}
c.most_common(2)       # [('a', 2), ('b', 2)]
\`\`\`

**3. Haqiqiy dunyo qo'llanilishi**

- Imlo tekshirgichlar (so'z takliflari)
- Plagiat aniqlash
- DNK ketma-ketlik tahlili
- Kriptografiya (chastota tahlili)`,
			solutionCode: `def is_anagram(s: str, t: str) -> bool:
    """
    t s ning anagrammasi ekanligini tekshiradi.

    Args:
        s: Birinchi satr
        t: Ikkinchi satr

    Returns:
        Agar t s ning anagrammasi bo'lsa True, aks holda False
    """
    # Tez uzunlik tekshiruvi
    if len(s) != len(t):
        return False

    # Belgilar chastotasini hisoblaymiz
    count = {}

    # s dagi belgilarni hisoblaymiz, t uchun kamaytiramiz
    for i in range(len(s)):
        count[s[i]] = count.get(s[i], 0) + 1
        count[t[i]] = count.get(t[i], 0) - 1

    # Barcha hisoblagichlar nol ekanligini tekshiramiz
    return all(c == 0 for c in count.values())`
		}
	}
};

export default task;
