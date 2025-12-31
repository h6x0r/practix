import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'dp-edit-distance',
	title: 'Edit Distance',
	difficulty: 'medium',
	tags: ['python', 'dynamic-programming', '2d-dp', 'string', 'levenshtein'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the minimum number of operations to convert one string to another.

**Problem:**

Given two strings \`word1\` and \`word2\`, return the minimum number of operations required to convert \`word1\` to \`word2\`.

You have three operations:
- **Insert** a character
- **Delete** a character
- **Replace** a character

This is also known as the **Levenshtein Distance**.

**Examples:**

\`\`\`
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation:
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation:
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')

Input: word1 = "", word2 = "abc"
Output: 3 (insert 'a', 'b', 'c')

Input: word1 = "abc", word2 = "abc"
Output: 0 (strings are equal)
\`\`\`

**DP Recurrence:**

\`\`\`
If word1[i-1] == word2[j-1]:
    dp[i][j] = dp[i-1][j-1]  (no operation needed)
Else:
    dp[i][j] = 1 + min(
        dp[i-1][j],    (delete from word1)
        dp[i][j-1],    (insert into word1)
        dp[i-1][j-1]   (replace in word1)
    )
\`\`\`

**Constraints:**
- 0 <= word1.length, word2.length <= 500
- word1 and word2 consist of lowercase English letters

**Time Complexity:** O(m × n)
**Space Complexity:** O(m × n), can be optimized to O(min(m, n))`,
	initialCode: `def min_distance(word1: str, word2: str) -> int:
    # TODO: Find minimum edit distance (insert, delete, replace operations)

    return 0`,
	solutionCode: `def min_distance(word1: str, word2: str) -> int:
    """
    Find minimum edit distance between two strings.

    Args:
        word1: Source string
        word2: Target string

    Returns:
        Minimum number of operations to convert word1 to word2
    """
    m, n = len(word1), len(word2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete i characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert j characters

    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                # Characters match - no operation needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Take minimum of three operations
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )

    return dp[m][n]


# Space-optimized version
def min_distance_optimized(word1: str, word2: str) -> int:
    """Space-optimized edit distance using O(n) space."""
    m, n = len(word1), len(word2)

    # Make word2 the shorter string
    if m < n:
        word1, word2 = word2, word1
        m, n = n, m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


# Version with operation reconstruction
def min_distance_with_operations(word1: str, word2: str) -> tuple:
    """Returns distance and list of operations."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack to find operations
    operations = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and word1[i - 1] == word2[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            operations.append(f"Replace '{word1[i-1]}' with '{word2[j-1]}'")
            i, j = i - 1, j - 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            operations.append(f"Insert '{word2[j-1]}'")
            j -= 1
        else:
            operations.append(f"Delete '{word1[i-1]}'")
            i -= 1

    return dp[m][n], list(reversed(operations))`,
	testCode: `import pytest
from solution import min_distance


class TestEditDistance:
    def test_horse_to_ros(self):
        """Test horse -> ros = 3"""
        assert min_distance("horse", "ros") == 3

    def test_intention_to_execution(self):
        """Test intention -> execution = 5"""
        assert min_distance("intention", "execution") == 5

    def test_empty_to_string(self):
        """Test empty to non-empty"""
        assert min_distance("", "abc") == 3

    def test_string_to_empty(self):
        """Test non-empty to empty"""
        assert min_distance("abc", "") == 3

    def test_both_empty(self):
        """Test both empty strings"""
        assert min_distance("", "") == 0

    def test_identical_strings(self):
        """Test identical strings"""
        assert min_distance("abc", "abc") == 0

    def test_single_char_different(self):
        """Test single character difference"""
        assert min_distance("a", "b") == 1

    def test_single_char_same(self):
        """Test single character same"""
        assert min_distance("a", "a") == 0

    def test_prefix(self):
        """Test when one is prefix of other"""
        assert min_distance("abc", "abcdef") == 3

    def test_completely_different(self):
        """Test completely different strings"""
        assert min_distance("abc", "xyz") == 3`,
	hint1: `Create a 2D table where dp[i][j] is the minimum operations to convert word1[0:i] to word2[0:j]. Initialize dp[0][j] = j and dp[i][0] = i.`,
	hint2: `If characters match, no operation needed: dp[i][j] = dp[i-1][j-1]. Otherwise, take min of delete (dp[i-1][j]+1), insert (dp[i][j-1]+1), or replace (dp[i-1][j-1]+1).`,
	whyItMatters: `Edit Distance (Levenshtein Distance) is fundamental to spell checkers, DNA analysis, and diff tools. It's a classic FAANG interview problem.

**Why This Matters:**

**1. Three Operations Pattern**

Understanding which operation leads where:

\`\`\`python
# Visual representation:
#           word2[j-1]
#              ↓
# word1[i-1] → dp[i-1][j-1]  dp[i-1][j]   (delete)
#              dp[i][j-1]    dp[i][j]
#              (insert)

# Delete: we used word1[i-1], move to dp[i-1][j]
# Insert: we add word2[j-1], move to dp[i][j-1]
# Replace: change word1[i-1] to word2[j-1], move to dp[i-1][j-1]
\`\`\`

**2. Real-World Applications**

\`\`\`python
# Spell checker
def suggest_corrections(word, dictionary, max_distance=2):
    suggestions = []
    for dict_word in dictionary:
        if min_distance(word, dict_word) <= max_distance:
            suggestions.append(dict_word)
    return suggestions

# DNA sequence alignment
# Insertions = gaps, deletions = gaps, replacements = mutations

# Fuzzy string matching
def fuzzy_search(query, text, threshold=0.8):
    distance = min_distance(query, text)
    similarity = 1 - distance / max(len(query), len(text))
    return similarity >= threshold
\`\`\`

**3. Related FAANG Problems**

\`\`\`python
# One Edit Distance (LeetCode 161)
# Check if strings are exactly one edit apart
def is_one_edit(s, t):
    if abs(len(s) - len(t)) > 1:
        return False
    return min_distance(s, t) == 1

# Delete Operation for Two Strings (LeetCode 583)
# Only delete operations allowed
# Answer: m + n - 2 * LCS(word1, word2)

# Minimum ASCII Delete Sum (LeetCode 712)
# Minimize sum of ASCII values of deleted chars
\`\`\`

**4. Variations**

\`\`\`python
# Weighted edit distance (different costs)
def weighted_edit_distance(word1, word2, insert_cost, delete_cost, replace_cost):
    # Same structure, but multiply by costs instead of +1

# Damerau-Levenshtein (allows transpositions)
# "ab" -> "ba" is one operation, not two
\`\`\`

**5. Complexity Analysis**

\`\`\`python
# Time: O(m * n) - fill entire table
# Space: O(m * n) - full table, or O(min(m,n)) optimized

# Why can we optimize space?
# Each row only depends on previous row
# Keep two rows and alternate
\`\`\``,
	order: 7,
	translations: {
		ru: {
			title: 'Редакционное расстояние',
			description: `Найдите минимальное количество операций для преобразования одной строки в другую.

**Задача:**

Даны две строки \`word1\` и \`word2\`, верните минимальное количество операций для преобразования \`word1\` в \`word2\`.

Доступны три операции:
- **Вставить** символ
- **Удалить** символ
- **Заменить** символ

Это также известно как **расстояние Левенштейна**.

**Примеры:**

\`\`\`
Вход: word1 = "horse", word2 = "ros"
Выход: 3
Объяснение:
horse -> rorse (заменить 'h' на 'r')
rorse -> rose (удалить 'r')
rose -> ros (удалить 'e')

Вход: word1 = "intention", word2 = "execution"
Выход: 5
\`\`\`

**Рекуррентное соотношение:**

\`\`\`
Если word1[i-1] == word2[j-1]:
    dp[i][j] = dp[i-1][j-1]  (операция не нужна)
Иначе:
    dp[i][j] = 1 + min(
        dp[i-1][j],    (удаление из word1)
        dp[i][j-1],    (вставка в word1)
        dp[i-1][j-1]   (замена в word1)
    )
\`\`\`

**Ограничения:**
- 0 <= word1.length, word2.length <= 500

**Временная сложность:** O(m × n)
**Пространственная сложность:** O(min(m, n)) с оптимизацией`,
			hint1: `Создайте 2D таблицу, где dp[i][j] - минимум операций для преобразования word1[0:i] в word2[0:j]. Инициализируйте dp[0][j] = j и dp[i][0] = i.`,
			hint2: `Если символы совпадают, операция не нужна: dp[i][j] = dp[i-1][j-1]. Иначе берём минимум из удаления, вставки или замены.`,
			whyItMatters: `Редакционное расстояние (расстояние Левенштейна) - основа проверки орфографии, анализа ДНК и инструментов diff.

**Почему это важно:**

**1. Паттерн трёх операций**

Понимание, какая операция куда ведёт.

**2. Применения в реальном мире**

- Проверка орфографии
- Выравнивание последовательностей ДНК
- Нечёткий поиск строк

**3. Связанные задачи FAANG**

- One Edit Distance
- Delete Operation for Two Strings
- Minimum ASCII Delete Sum

**4. Оптимизация памяти**

Каждая строка зависит только от предыдущей - можно хранить только две строки.`,
			solutionCode: `def min_distance(word1: str, word2: str) -> int:
    """
    Находит минимальное редакционное расстояние между строками.

    Args:
        word1: Исходная строка
        word2: Целевая строка

    Returns:
        Минимальное количество операций для преобразования word1 в word2
    """
    m, n = len(word1), len(word2)

    # Создаём DP таблицу
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Базовые случаи
    for i in range(m + 1):
        dp[i][0] = i  # Удалить i символов
    for j in range(n + 1):
        dp[0][j] = j  # Вставить j символов

    # Заполняем таблицу
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                # Символы совпадают - операция не нужна
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Берём минимум из трёх операций
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Удаление
                    dp[i][j - 1],      # Вставка
                    dp[i - 1][j - 1]   # Замена
                )

    return dp[m][n]`
		},
		uz: {
			title: 'Tahrir masofasi',
			description: `Bitta satrni boshqasiga aylantirish uchun kerakli minimal operatsiyalar sonini toping.

**Masala:**

Ikki satr \`word1\` va \`word2\` berilgan, \`word1\` ni \`word2\` ga aylantirish uchun kerakli minimal operatsiyalar sonini qaytaring.

Sizda uchta operatsiya bor:
- Belgi **qo'shish**
- Belgi **o'chirish**
- Belgi **almashtirish**

Bu **Levenshtein masofasi** deb ham ataladi.

**Misollar:**

\`\`\`
Kirish: word1 = "horse", word2 = "ros"
Chiqish: 3
Izoh:
horse -> rorse ('h' ni 'r' ga almashtirish)
rorse -> rose ('r' ni o'chirish)
rose -> ros ('e' ni o'chirish)

Kirish: word1 = "intention", word2 = "execution"
Chiqish: 5
\`\`\`

**DP rekurrensiyasi:**

\`\`\`
Agar word1[i-1] == word2[j-1]:
    dp[i][j] = dp[i-1][j-1]  (operatsiya kerak emas)
Aks holda:
    dp[i][j] = 1 + min(
        dp[i-1][j],    (word1 dan o'chirish)
        dp[i][j-1],    (word1 ga qo'shish)
        dp[i-1][j-1]   (word1 da almashtirish)
    )
\`\`\`

**Cheklovlar:**
- 0 <= word1.length, word2.length <= 500

**Vaqt murakkabligi:** O(m × n)
**Xotira murakkabligi:** O(min(m, n)) optimallashtirish bilan`,
			hint1: `2D jadval yarating, bu yerda dp[i][j] word1[0:i] ni word2[0:j] ga aylantirish uchun minimal operatsiyalar. dp[0][j] = j va dp[i][0] = i bilan boshlang.`,
			hint2: `Agar belgilar mos kelsa, operatsiya kerak emas: dp[i][j] = dp[i-1][j-1]. Aks holda o'chirish, qo'shish yoki almashtirishdan minimumni oling.`,
			whyItMatters: `Tahrir masofasi (Levenshtein masofasi) imlo tekshirgichlar, DNK tahlili va diff vositalarining asosi.

**Bu nima uchun muhim:**

**1. Uchta operatsiya patterni**

Qaysi operatsiya qayerga olib borishini tushunish.

**2. Haqiqiy dunyo qo'llanishlari**

- Imlo tekshirish
- DNK ketma-ketliklarini tekislash
- Noaniq satr qidirish

**3. Bog'liq FAANG masalalari**

- One Edit Distance
- Delete Operation for Two Strings
- Minimum ASCII Delete Sum

**4. Xotira optimallashtirish**

Har bir qator faqat oldingi qatorga bog'liq - faqat ikki qator saqlash mumkin.`,
			solutionCode: `def min_distance(word1: str, word2: str) -> int:
    """
    Ikki satr orasidagi minimal tahrir masofasini topadi.

    Args:
        word1: Manba satr
        word2: Maqsad satr

    Returns:
        word1 ni word2 ga aylantirish uchun minimal operatsiyalar soni
    """
    m, n = len(word1), len(word2)

    # DP jadval yaratamiz
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Asosiy holatlar
    for i in range(m + 1):
        dp[i][0] = i  # i ta belgi o'chirish
    for j in range(n + 1):
        dp[0][j] = j  # j ta belgi qo'shish

    # Jadvalni to'ldiramiz
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                # Belgilar mos keladi - operatsiya kerak emas
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Uchta operatsiyadan minimumni olamiz
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # O'chirish
                    dp[i][j - 1],      # Qo'shish
                    dp[i - 1][j - 1]   # Almashtirish
                )

    return dp[m][n]`
		}
	}
};

export default task;
