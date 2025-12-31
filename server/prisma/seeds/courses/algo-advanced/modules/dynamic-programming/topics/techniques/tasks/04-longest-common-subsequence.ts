import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'dp-longest-common-subsequence',
	title: 'Longest Common Subsequence',
	difficulty: 'medium',
	tags: ['python', 'dynamic-programming', '2d-dp', 'string', 'subsequence'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the length of the longest common subsequence of two strings.

**Problem:**

Given two strings \`text1\` and \`text2\`, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A **subsequence** of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

**Examples:**

\`\`\`
Input: text1 = "abcde", text2 = "ace"
Output: 3
Explanation: The longest common subsequence is "ace"

Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc"

Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: No common subsequence exists

Input: text1 = "AGGTAB", text2 = "GXTXAYB"
Output: 4
Explanation: The longest common subsequence is "GTAB"
\`\`\`

**Key Insight:**

Compare characters from both strings:
- If \`text1[i-1] == text2[j-1]\`: both characters are part of LCS
  - \`dp[i][j] = dp[i-1][j-1] + 1\`
- If different: take the best from excluding either character
  - \`dp[i][j] = max(dp[i-1][j], dp[i][j-1])\`

**DP Table Visualization:**

\`\`\`
      ""  a  c  e
  ""   0  0  0  0
  a    0  1  1  1
  b    0  1  1  1
  c    0  1  2  2
  d    0  1  2  2
  e    0  1  2  3
\`\`\`

**Constraints:**
- 1 <= text1.length, text2.length <= 1000
- text1 and text2 consist of only lowercase English characters

**Time Complexity:** O(m × n)
**Space Complexity:** O(m × n), can be optimized to O(min(m, n))`,
	initialCode: `def longest_common_subsequence(text1: str, text2: str) -> int:
    # TODO: Find length of longest common subsequence

    return 0`,
	solutionCode: `def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Find length of longest common subsequence.

    Args:
        text1: First string
        text2: Second string

    Returns:
        Length of the longest common subsequence
    """
    m, n = len(text1), len(text2)

    # Create 2D DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                # Characters match - extend LCS
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # Characters don't match - take best from either
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


# Space-optimized version: O(min(m, n)) space
def longest_common_subsequence_optimized(text1: str, text2: str) -> int:
    """Space-optimized LCS using only two rows."""
    # Make text2 the shorter string for space optimization
    if len(text1) < len(text2):
        text1, text2 = text2, text1

    m, n = len(text1), len(text2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


# Version that also reconstructs the LCS
def lcs_with_string(text1: str, text2: str) -> tuple:
    """Returns both length and the actual LCS string."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find actual LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return dp[m][n], ''.join(reversed(lcs))`,
	testCode: `import pytest
from solution import longest_common_subsequence


class TestLongestCommonSubsequence:
    def test_basic_example(self):
        """Test text1='abcde', text2='ace' -> 3"""
        assert longest_common_subsequence("abcde", "ace") == 3

    def test_identical_strings(self):
        """Test identical strings"""
        assert longest_common_subsequence("abc", "abc") == 3

    def test_no_common(self):
        """Test no common subsequence"""
        assert longest_common_subsequence("abc", "def") == 0

    def test_one_empty(self):
        """Test with empty string"""
        assert longest_common_subsequence("", "abc") == 0
        assert longest_common_subsequence("abc", "") == 0

    def test_single_char_match(self):
        """Test single character match"""
        assert longest_common_subsequence("a", "a") == 1

    def test_single_char_no_match(self):
        """Test single character no match"""
        assert longest_common_subsequence("a", "b") == 0

    def test_longer_example(self):
        """Test AGGTAB vs GXTXAYB -> 4 (GTAB)"""
        assert longest_common_subsequence("AGGTAB", "GXTXAYB") == 4

    def test_subsequence_not_substring(self):
        """Test that it finds subsequence, not substring"""
        # LCS is "aec" = 3, not "ae" = 2
        assert longest_common_subsequence("abecfd", "aexcy") == 3

    def test_repeated_characters(self):
        """Test with repeated characters"""
        assert longest_common_subsequence("aaa", "aa") == 2
        assert longest_common_subsequence("aaaa", "aaaa") == 4

    def test_reversed_strings(self):
        """Test reversed strings"""
        assert longest_common_subsequence("abc", "cba") == 1`,
	hint1: `Create a 2D table where dp[i][j] represents the LCS length of text1[0:i] and text2[0:j]. Initialize first row and column with zeros.`,
	hint2: `For each position, if characters match (text1[i-1] == text2[j-1]), add 1 to the diagonal value. Otherwise, take the maximum of the cell above or to the left.`,
	whyItMatters: `LCS is the foundation for diff algorithms, DNA sequence alignment, and file comparison tools. It's a must-know 2D DP problem.

**Why This Matters:**

**1. 2D DP Foundation**

This pattern extends to many other problems:

\`\`\`python
# Grid-based problems
dp[i][j] depends on dp[i-1][j-1], dp[i-1][j], dp[i][j-1]

# Common pattern:
for i in range(1, m + 1):
    for j in range(1, n + 1):
        # Make decision based on neighboring cells
\`\`\`

**2. Real-World Applications**

\`\`\`python
# Git diff algorithm (simplified concept)
def diff(old_lines, new_lines):
    lcs = find_lcs(old_lines, new_lines)
    # Lines in old but not in LCS = deleted
    # Lines in new but not in LCS = added
    # Lines in LCS = unchanged

# DNA sequence alignment
def align_sequences(seq1, seq2):
    # LCS helps find conserved regions
    # Mutations appear where sequences differ
\`\`\`

**3. Related FAANG Problems**

\`\`\`python
# Longest Common Substring (contiguous)
if text1[i-1] == text2[j-1]:
    dp[i][j] = dp[i-1][j-1] + 1
else:
    dp[i][j] = 0  # Reset! (must be contiguous)

# Edit Distance (Levenshtein)
if text1[i-1] == text2[j-1]:
    dp[i][j] = dp[i-1][j-1]  # No operation needed
else:
    dp[i][j] = 1 + min(
        dp[i-1][j],     # Delete
        dp[i][j-1],     # Insert
        dp[i-1][j-1]    # Replace
    )

# Shortest Common Supersequence
# Length = m + n - LCS(text1, text2)
\`\`\`

**4. Space Optimization Pattern**

\`\`\`python
# 2D to 1D: O(mn) -> O(min(m,n))
# Only need previous row
prev = [0] * (n + 1)
curr = [0] * (n + 1)

for i in range(1, m + 1):
    for j in range(1, n + 1):
        if text1[i-1] == text2[j-1]:
            curr[j] = prev[j-1] + 1
        else:
            curr[j] = max(prev[j], curr[j-1])
    prev, curr = curr, prev
\`\`\`

**5. Reconstructing the Solution**

\`\`\`python
# Backtrack from dp[m][n] to build actual LCS
result = []
i, j = m, n
while i > 0 and j > 0:
    if text1[i-1] == text2[j-1]:
        result.append(text1[i-1])
        i, j = i-1, j-1
    elif dp[i-1][j] > dp[i][j-1]:
        i -= 1
    else:
        j -= 1
return ''.join(reversed(result))
\`\`\``,
	order: 4,
	translations: {
		ru: {
			title: 'Наибольшая общая подпоследовательность',
			description: `Найдите длину наибольшей общей подпоследовательности двух строк.

**Задача:**

Даны две строки \`text1\` и \`text2\`, верните длину их наибольшей общей подпоследовательности. Если общей подпоследовательности нет, верните 0.

**Подпоследовательность** строки - это новая строка, сгенерированная из исходной строки с удалением некоторых символов (возможно, ни одного) без изменения относительного порядка оставшихся символов.

**Примеры:**

\`\`\`
Вход: text1 = "abcde", text2 = "ace"
Выход: 3
Объяснение: Наибольшая общая подпоследовательность - "ace"

Вход: text1 = "abc", text2 = "def"
Выход: 0
Объяснение: Общей подпоследовательности не существует
\`\`\`

**Ключевая идея:**

Сравниваем символы обеих строк:
- Если \`text1[i-1] == text2[j-1]\`: оба символа часть LCS
  - \`dp[i][j] = dp[i-1][j-1] + 1\`
- Если разные: берём лучшее из исключения любого символа
  - \`dp[i][j] = max(dp[i-1][j], dp[i][j-1])\`

**Ограничения:**
- 1 <= text1.length, text2.length <= 1000

**Временная сложность:** O(m × n)
**Пространственная сложность:** O(m × n), можно оптимизировать до O(min(m, n))`,
			hint1: `Создайте 2D таблицу, где dp[i][j] - длина LCS для text1[0:i] и text2[0:j]. Инициализируйте первую строку и столбец нулями.`,
			hint2: `Для каждой позиции, если символы совпадают (text1[i-1] == text2[j-1]), добавьте 1 к диагональному значению. Иначе возьмите максимум из ячейки сверху или слева.`,
			whyItMatters: `LCS - основа алгоритмов diff, выравнивания ДНК последовательностей и инструментов сравнения файлов. Это обязательная к знанию задача 2D DP.

**Почему это важно:**

**1. Основа 2D DP**

Этот паттерн распространяется на многие другие задачи.

**2. Применения в реальном мире**

- Алгоритм Git diff
- Выравнивание ДНК последовательностей
- Сравнение файлов

**3. Связанные задачи FAANG**

- Longest Common Substring (непрерывная)
- Edit Distance (расстояние Левенштейна)
- Shortest Common Supersequence

**4. Паттерн оптимизации памяти**

2D -> 1D: O(mn) -> O(min(m,n))`,
			solutionCode: `def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Находит длину наибольшей общей подпоследовательности.

    Args:
        text1: Первая строка
        text2: Вторая строка

    Returns:
        Длина наибольшей общей подпоследовательности
    """
    m, n = len(text1), len(text2)

    # Создаём 2D DP таблицу
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Заполняем таблицу
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                # Символы совпадают - расширяем LCS
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # Символы не совпадают - берём лучшее
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]`
		},
		uz: {
			title: 'Eng uzun umumiy qism ketma-ketlik',
			description: `Ikki satrning eng uzun umumiy qism ketma-ketligi uzunligini toping.

**Masala:**

Ikki satr \`text1\` va \`text2\` berilgan, ularning eng uzun umumiy qism ketma-ketligi uzunligini qaytaring. Agar umumiy qism ketma-ketlik bo'lmasa, 0 qaytaring.

Satrning **qism ketma-ketligi** - bu asl satrdan ba'zi belgilarni (hech biri ham bo'lishi mumkin) qolgan belgilarning nisbiy tartibini o'zgartirmasdan o'chirish orqali hosil qilingan yangi satr.

**Misollar:**

\`\`\`
Kirish: text1 = "abcde", text2 = "ace"
Chiqish: 3
Izoh: Eng uzun umumiy qism ketma-ketlik - "ace"

Kirish: text1 = "abc", text2 = "def"
Chiqish: 0
Izoh: Umumiy qism ketma-ketlik mavjud emas
\`\`\`

**Asosiy tushuncha:**

Ikkala satrdan belgilarni solishtiramiz:
- Agar \`text1[i-1] == text2[j-1]\`: ikkala belgi LCS qismi
  - \`dp[i][j] = dp[i-1][j-1] + 1\`
- Agar farqli: har qanday belgini chiqarib tashlashdan eng yaxshisini olamiz
  - \`dp[i][j] = max(dp[i-1][j], dp[i][j-1])\`

**Cheklovlar:**
- 1 <= text1.length, text2.length <= 1000

**Vaqt murakkabligi:** O(m × n)
**Xotira murakkabligi:** O(m × n), O(min(m, n)) ga optimallashtirilishi mumkin`,
			hint1: `2D jadval yarating, bu yerda dp[i][j] text1[0:i] va text2[0:j] uchun LCS uzunligi. Birinchi qator va ustunni nollar bilan boshlang.`,
			hint2: `Har bir pozitsiya uchun, agar belgilar mos kelsa (text1[i-1] == text2[j-1]), diagonal qiymatga 1 qo'shing. Aks holda yuqoridagi yoki chapdagi katakdan maksimumni oling.`,
			whyItMatters: `LCS - diff algoritmlari, DNK ketma-ketliklarini tekislash va fayl taqqoslash vositalarining asosi. Bu bilish kerak bo'lgan 2D DP masalasi.

**Bu nima uchun muhim:**

**1. 2D DP asosi**

Bu pattern boshqa ko'plab masalalarga kengayadi.

**2. Haqiqiy dunyo qo'llanishlari**

- Git diff algoritmi
- DNK ketma-ketliklarini tekislash
- Fayllarni taqqoslash

**3. Bog'liq FAANG masalalari**

- Longest Common Substring (uzluksiz)
- Edit Distance (Levenshtein masofasi)
- Shortest Common Supersequence

**4. Xotira optimallashtirish patterni**

2D -> 1D: O(mn) -> O(min(m,n))`,
			solutionCode: `def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Eng uzun umumiy qism ketma-ketlik uzunligini topadi.

    Args:
        text1: Birinchi satr
        text2: Ikkinchi satr

    Returns:
        Eng uzun umumiy qism ketma-ketlik uzunligi
    """
    m, n = len(text1), len(text2)

    # 2D DP jadval yaratamiz
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Jadvalni to'ldiramiz
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                # Belgilar mos keladi - LCS ni kengaytiramiz
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # Belgilar mos kelmaydi - eng yaxshisini olamiz
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]`
		}
	}
};

export default task;
