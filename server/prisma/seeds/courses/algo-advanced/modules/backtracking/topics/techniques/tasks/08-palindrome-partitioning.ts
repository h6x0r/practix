import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'backtracking-palindrome-partitioning',
	title: 'Palindrome Partitioning',
	difficulty: 'medium',
	tags: ['python', 'backtracking', 'recursion', 'string', 'palindrome', 'dp'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Partition a string so that every substring is a palindrome.

**Problem:**

Given a string \`s\`, partition it such that every substring of the partition is a palindrome. Return all possible palindrome partitions.

**Examples:**

\`\`\`
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]

Explanation:
- ["a","a","b"]: "a", "a", "b" are all palindromes
- ["aa","b"]: "aa" and "b" are both palindromes

Input: s = "a"
Output: [["a"]]

Input: s = "aba"
Output: [["a","b","a"],["aba"]]
\`\`\`

**Visualization:**

\`\`\`
s = "aab"

                  "aab"
          /         |        \\
        "a"       "aa"      "aab" (not palindrome, skip)
       /   \\        |
    "a"    "ab"    "b"
     |     (not)
    "b"

Valid partitions:
1. ["a", "a", "b"]
2. ["aa", "b"]
\`\`\`

**Key Insight:**

For each position, try all possible prefixes that are palindromes. If the prefix is a palindrome, recurse on the remaining substring.

**Constraints:**
- 1 <= s.length <= 16
- s contains only lowercase English letters

**Time Complexity:** O(n × 2^n) - potentially 2^n partitions
**Space Complexity:** O(n) for recursion stack`,
	initialCode: `from typing import List

def partition(s: str) -> List[List[str]]:
    # TODO: Partition string so every substring is a palindrome

    return []`,
	solutionCode: `from typing import List

def partition(s: str) -> List[List[str]]:
    """
    Find all palindrome partitions of the string.
    """
    result = []
    n = len(s)

    def is_palindrome(start: int, end: int) -> bool:
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True

    def backtrack(start: int, current: List[str]) -> None:
        if start == n:
            result.append(current.copy())
            return

        for end in range(start, n):
            if is_palindrome(start, end):
                current.append(s[start:end + 1])
                backtrack(end + 1, current)
                current.pop()

    backtrack(0, [])
    return result


# Optimized with DP precomputation
def partition_optimized(s: str) -> List[List[str]]:
    """Optimized with precomputed palindrome table."""
    n = len(s)
    result = []

    # dp[i][j] = True if s[i:j+1] is palindrome
    dp = [[False] * n for _ in range(n)]

    # Every single character is a palindrome
    for i in range(n):
        dp[i][i] = True

    # Fill the DP table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length == 2:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i + 1][j - 1]

    def backtrack(start: int, current: List[str]) -> None:
        if start == n:
            result.append(current.copy())
            return

        for end in range(start, n):
            if dp[start][end]:
                current.append(s[start:end + 1])
                backtrack(end + 1, current)
                current.pop()

    backtrack(0, [])
    return result


# Minimum cuts for palindrome partitioning
def min_cut(s: str) -> int:
    """Find minimum cuts needed for palindrome partitioning."""
    n = len(s)

    # dp_pal[i][j] = True if s[i:j+1] is palindrome
    dp_pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i <= 2 or dp_pal[i + 1][j - 1]):
                dp_pal[i][j] = True

    # cuts[i] = minimum cuts for s[0:i+1]
    cuts = [0] * n

    for i in range(n):
        if dp_pal[0][i]:
            cuts[i] = 0
        else:
            cuts[i] = float('inf')
            for j in range(i):
                if dp_pal[j + 1][i]:
                    cuts[i] = min(cuts[i], cuts[j] + 1)

    return cuts[n - 1]`,
	testCode: `import pytest
from solution import partition


class TestPalindromePartitioning:
    def test_aab(self):
        """Test 'aab'"""
        result = partition("aab")
        expected = [["a","a","b"],["aa","b"]]
        assert sorted([sorted(p) for p in result]) == sorted([sorted(p) for p in expected])

    def test_single_char(self):
        """Test single character"""
        assert partition("a") == [["a"]]

    def test_aba(self):
        """Test 'aba'"""
        result = partition("aba")
        assert ["a","b","a"] in result
        assert ["aba"] in result
        assert len(result) == 2

    def test_all_same(self):
        """Test string with all same characters"""
        result = partition("aaa")
        assert ["a","a","a"] in result
        assert ["aa","a"] in result
        assert ["a","aa"] in result
        assert ["aaa"] in result

    def test_no_palindrome_substring(self):
        """Test string like 'abc' (only single chars)"""
        result = partition("abc")
        assert result == [["a","b","c"]]

    def test_valid_partitions(self):
        """Test all partitions contain only palindromes"""
        def is_palindrome(s):
            return s == s[::-1]

        result = partition("abba")
        for parts in result:
            for part in parts:
                assert is_palindrome(part)

    def test_concatenation(self):
        """Test parts concatenate to original string"""
        s = "aabb"
        result = partition(s)
        for parts in result:
            assert ''.join(parts) == s

    def test_longer_string(self):
        """Test longer string"""
        result = partition("abbab")
        # Should have multiple valid partitions
        assert len(result) > 1

    def test_full_palindrome(self):
        """Test when entire string is palindrome"""
        result = partition("abcba")
        assert ["abcba"] in result

    def test_no_duplicates(self):
        """Test no duplicate partitions"""
        result = partition("aabb")
        tuples = [tuple(p) for p in result]
        assert len(tuples) == len(set(tuples))`,
	hint1: `Try all possible prefixes at each position. If a prefix is a palindrome, include it and recurse on the remaining string.`,
	hint2: `Use DP to precompute which substrings are palindromes for O(1) lookup. dp[i][j] = True if s[i:j+1] is palindrome. Base: single chars. Transition: s[i]==s[j] and dp[i+1][j-1].`,
	whyItMatters: `Palindrome Partitioning combines backtracking with palindrome detection. It teaches how to optimize with DP precomputation and has applications in string processing and NLP.

**Why This Matters:**

**1. Two-Phase Optimization**

\`\`\`python
# Phase 1: Precompute all palindrome substrings with DP
# O(n²) time, O(n²) space
dp = [[False] * n for _ in range(n)]

for i in range(n):
    dp[i][i] = True  # Single chars

for length in range(2, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        if s[i] == s[j]:
            dp[i][j] = length == 2 or dp[i + 1][j - 1]

# Phase 2: Backtrack with O(1) palindrome checks
def backtrack(start, current):
    if start == n:
        result.append(current.copy())
        return
    for end in range(start, n):
        if dp[start][end]:  # O(1) check!
            current.append(s[start:end + 1])
            backtrack(end + 1, current)
            current.pop()
\`\`\`

**2. Palindrome DP Pattern**

\`\`\`python
# Expand around center (O(n²) time, O(1) space)
def count_palindromes(s):
    count = 0
    n = len(s)

    for center in range(2 * n - 1):
        left = center // 2
        right = left + center % 2

        while left >= 0 and right < n and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1

    return count

# Manacher's algorithm: O(n) for longest palindrome
\`\`\`

**3. Minimum Cuts Variant**

\`\`\`python
def min_cut(s):
    n = len(s)
    # cuts[i] = min cuts for s[0:i+1]
    cuts = list(range(n))  # worst case: n-1 cuts

    for i in range(n):
        # Try all palindromes ending at i
        for j in range(i + 1):
            if is_palindrome(s[j:i+1]):
                cuts[i] = 0 if j == 0 else min(cuts[i], cuts[j-1] + 1)

    return cuts[n - 1]
\`\`\`

**4. Applications**

\`\`\`python
# Text segmentation in NLP
# DNA sequence analysis
# Spell checking (finding palindromic patterns)
# Game design (word puzzles)
\`\`\``,
	order: 8,
	translations: {
		ru: {
			title: 'Палиндромное разбиение',
			description: `Разбейте строку так, чтобы каждая подстрока была палиндромом.

**Задача:**

Дана строка \`s\`. Разбейте её так, чтобы каждая подстрока была палиндромом. Верните все возможные разбиения.

**Примеры:**

\`\`\`
Вход: s = "aab"
Выход: [["a","a","b"],["aa","b"]]

Объяснение:
- ["a","a","b"]: "a", "a", "b" - все палиндромы
- ["aa","b"]: "aa" и "b" - оба палиндромы
\`\`\`

**Ключевая идея:**

Для каждой позиции пробуем все возможные префиксы, являющиеся палиндромами. Если префикс - палиндром, рекурсивно обрабатываем оставшуюся часть.

**Ограничения:**
- 1 <= s.length <= 16

**Временная сложность:** O(n × 2^n)
**Пространственная сложность:** O(n)`,
			hint1: `Пробуйте все возможные префиксы на каждой позиции. Если префикс - палиндром, включите его и рекурсивно обработайте остаток.`,
			hint2: `Используйте DP для предварительного вычисления палиндромов для O(1) проверки. dp[i][j] = True если s[i:j+1] палиндром.`,
			whyItMatters: `Palindrome Partitioning сочетает бэктрекинг с проверкой палиндромов. Учит оптимизации с DP.

**Почему это важно:**

**1. Двухфазная оптимизация**

Фаза 1: предвычисление палиндромов через DP. Фаза 2: бэктрекинг с O(1) проверками.

**2. Паттерн DP для палиндромов**

Расширение от центра или таблица DP.

**3. Вариант минимальных разрезов**

Найти минимальное количество разрезов для палиндромного разбиения.`,
			solutionCode: `from typing import List

def partition(s: str) -> List[List[str]]:
    """Находит все палиндромные разбиения строки."""
    result = []
    n = len(s)

    def is_palindrome(start: int, end: int) -> bool:
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True

    def backtrack(start: int, current: List[str]) -> None:
        if start == n:
            result.append(current.copy())
            return

        for end in range(start, n):
            if is_palindrome(start, end):
                current.append(s[start:end + 1])
                backtrack(end + 1, current)
                current.pop()

    backtrack(0, [])
    return result`
		},
		uz: {
			title: 'Palindromli bo\'lish',
			description: `Satrni har bir qism palindrom bo'ladigan qilib bo'ling.

**Masala:**

\`s\` satri berilgan. Uni har bir qism palindrom bo'ladigan qilib bo'ling. Barcha mumkin bo'lgan bo'linishlarni qaytaring.

**Misollar:**

\`\`\`
Kirish: s = "aab"
Chiqish: [["a","a","b"],["aa","b"]]

Izoh:
- ["a","a","b"]: "a", "a", "b" barchasi palindrom
- ["aa","b"]: "aa" va "b" ikkalasi palindrom
\`\`\`

**Asosiy tushuncha:**

Har bir pozitsiyada palindrom bo'lgan barcha prefikslarni sinab ko'ring. Agar prefiks palindrom bo'lsa, qolgan qismni rekursiv qayta ishlang.

**Cheklovlar:**
- 1 <= s.length <= 16

**Vaqt murakkabligi:** O(n × 2^n)
**Xotira murakkabligi:** O(n)`,
			hint1: `Har bir pozitsiyada barcha mumkin bo'lgan prefikslarni sinab ko'ring. Agar prefiks palindrom bo'lsa, uni qo'shing va qoldiqni rekursiv qayta ishlang.`,
			hint2: `O(1) tekshirish uchun palindromlarni oldindan hisoblash uchun DP ishlating. dp[i][j] = True agar s[i:j+1] palindrom bo'lsa.`,
			whyItMatters: `Palindrome Partitioning backtracking ni palindrom tekshiruvi bilan birlashtiradi. DP bilan optimallashtirish o'rgatadi.

**Bu nima uchun muhim:**

**1. Ikki bosqichli optimallashtirish**

1-bosqich: DP orqali palindromlarni oldindan hisoblash. 2-bosqich: O(1) tekshiruvlar bilan backtracking.

**2. Palindromlar uchun DP patterni**

Markazdan kengaytirish yoki DP jadvali.

**3. Minimal kesishlar varianti**

Palindromli bo'linish uchun minimal kesishlar sonini topish.`,
			solutionCode: `from typing import List

def partition(s: str) -> List[List[str]]:
    """Satrning barcha palindromli bo'linishlarini topadi."""
    result = []
    n = len(s)

    def is_palindrome(start: int, end: int) -> bool:
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True

    def backtrack(start: int, current: List[str]) -> None:
        if start == n:
            result.append(current.copy())
            return

        for end in range(start, n):
            if is_palindrome(start, end):
                current.append(s[start:end + 1])
                backtrack(end + 1, current)
                current.pop()

    backtrack(0, [])
    return result`
		}
	}
};

export default task;
