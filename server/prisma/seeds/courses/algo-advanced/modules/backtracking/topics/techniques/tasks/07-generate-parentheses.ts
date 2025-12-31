import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'backtracking-generate-parentheses',
	title: 'Generate Parentheses',
	difficulty: 'medium',
	tags: ['python', 'backtracking', 'recursion', 'string', 'catalan'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Generate all valid combinations of n pairs of parentheses.

**Problem:**

Given \`n\` pairs of parentheses, generate all combinations of well-formed parentheses.

A string is well-formed if:
1. Every '(' has a matching ')'
2. At any point, the number of ')' doesn't exceed '('

**Examples:**

\`\`\`
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

Input: n = 1
Output: ["()"]

Input: n = 2
Output: ["(())","()()"]
\`\`\`

**Visualization:**

\`\`\`
n = 2

              ""
           /      \\
          (        ✗ ) invalid (would need '(' first)
        /   \\
      ((     ()
      |     /  \\
     (()   ()(   ✗ ()) invalid
      |     |
    (())  ()()

Valid: ["(())", "()()"]
\`\`\`

**Key Insight:**

At each step, we can:
1. Add '(' if open < n
2. Add ')' if close < open

**Constraints:**
- 1 <= n <= 8

**Time Complexity:** O(4^n / √n) - Catalan number
**Space Complexity:** O(n) for recursion stack`,
	initialCode: `from typing import List

def generate_parenthesis(n: int) -> List[str]:
    # TODO: Generate all valid combinations of n pairs of parentheses

    return []`,
	solutionCode: `from typing import List

def generate_parenthesis(n: int) -> List[str]:
    """
    Generate all valid parentheses combinations.
    """
    result = []

    def backtrack(current: str, open_count: int, close_count: int) -> None:
        if len(current) == 2 * n:
            result.append(current)
            return

        # Can add '(' if we haven't used all
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)

        # Can add ')' if it won't exceed '(' count
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)

    backtrack('', 0, 0)
    return result


# Using list for efficiency (avoiding string concatenation)
def generate_parenthesis_optimized(n: int) -> List[str]:
    """Optimized with list instead of string concatenation."""
    result = []

    def backtrack(current: List[str], open_count: int, close_count: int) -> None:
        if len(current) == 2 * n:
            result.append(''.join(current))
            return

        if open_count < n:
            current.append('(')
            backtrack(current, open_count + 1, close_count)
            current.pop()

        if close_count < open_count:
            current.append(')')
            backtrack(current, open_count, close_count + 1)
            current.pop()

    backtrack([], 0, 0)
    return result


# Iterative approach (BFS-like)
def generate_parenthesis_iterative(n: int) -> List[str]:
    """Generate parentheses iteratively."""
    if n == 0:
        return []

    result = []
    # (current_string, open_count, close_count)
    stack = [('', 0, 0)]

    while stack:
        current, open_count, close_count = stack.pop()

        if len(current) == 2 * n:
            result.append(current)
            continue

        if close_count < open_count:
            stack.append((current + ')', open_count, close_count + 1))

        if open_count < n:
            stack.append((current + '(', open_count + 1, close_count))

    return result


# Count valid parentheses (Catalan number)
def count_parentheses(n: int) -> int:
    """
    Count valid parentheses = Catalan number.
    C(n) = C(2n, n) / (n + 1) = (2n)! / ((n+1)! * n!)
    """
    from math import factorial
    return factorial(2 * n) // (factorial(n + 1) * factorial(n))`,
	testCode: `import pytest
from solution import generate_parenthesis


class TestGenerateParentheses:
    def test_n3(self):
        """Test n=3"""
        result = generate_parenthesis(3)
        expected = ["((()))","(()())","(())()","()(())","()()()"]
        assert sorted(result) == sorted(expected)

    def test_n1(self):
        """Test n=1"""
        assert generate_parenthesis(1) == ["()"]

    def test_n2(self):
        """Test n=2"""
        result = generate_parenthesis(2)
        expected = ["(())", "()()"]
        assert sorted(result) == sorted(expected)

    def test_correct_count(self):
        """Test correct number of combinations (Catalan numbers)"""
        # Catalan: 1, 1, 2, 5, 14, 42, 132, 429
        expected_counts = [1, 1, 2, 5, 14, 42]
        for n in range(1, 6):
            result = generate_parenthesis(n)
            assert len(result) == expected_counts[n]

    def test_valid_length(self):
        """Test each string has correct length"""
        result = generate_parenthesis(4)
        for s in result:
            assert len(s) == 8  # 2 * n

    def test_valid_parentheses(self):
        """Test all generated strings are valid"""
        def is_valid(s):
            count = 0
            for c in s:
                count += 1 if c == '(' else -1
                if count < 0:
                    return False
            return count == 0

        result = generate_parenthesis(4)
        for s in result:
            assert is_valid(s)

    def test_no_duplicates(self):
        """Test no duplicate strings"""
        result = generate_parenthesis(4)
        assert len(result) == len(set(result))

    def test_balanced_counts(self):
        """Test equal number of ( and )"""
        result = generate_parenthesis(3)
        for s in result:
            assert s.count('(') == s.count(')') == 3

    def test_only_parentheses(self):
        """Test strings contain only ( and )"""
        result = generate_parenthesis(3)
        for s in result:
            assert all(c in '()' for c in s)

    def test_n4_catalan_count(self):
        """Test n=4 produces 14 combinations (4th Catalan)"""
        result = generate_parenthesis(4)
        assert len(result) == 14`,
	hint1: `Track open and close count. You can add '(' if open < n. You can add ')' only if close < open (to ensure validity).`,
	hint2: `Base case: when string length is 2n. Don't need to validate at the end - the rules ensure all generated strings are valid.`,
	whyItMatters: `Generate Parentheses demonstrates constrained backtracking where we prune invalid paths early. The count of valid combinations follows the Catalan number sequence.

**Why This Matters:**

**1. Catalan Numbers**

\`\`\`python
# Number of valid parentheses = Catalan number
# C(n) = (2n)! / ((n+1)! * n!)

# Catalan sequence: 1, 1, 2, 5, 14, 42, 132, 429, 1430, ...

# Catalan numbers appear in:
# - Valid parentheses combinations
# - Binary search trees with n nodes
# - Ways to triangulate a polygon
# - Paths in a grid that don't cross diagonal
\`\`\`

**2. Constrained Backtracking**

\`\`\`python
# Key insight: we don't generate all 2^(2n) combinations
# and then filter. We only generate valid ones.

def backtrack(current, open, close):
    # Only proceed if constraints are satisfied
    if open < n:
        backtrack(current + '(', open + 1, close)
    if close < open:  # Constraint: never more ) than (
        backtrack(current + ')', open, close + 1)
\`\`\`

**3. Applications**

\`\`\`python
# Generate valid expressions
# Generate balanced brackets [] {} ()
# Generate valid XML/HTML tags
# Dyck paths (combinatorics)
\`\`\`

**4. Extended Problem: Multiple Bracket Types**

\`\`\`python
def generate_brackets(pairs):
    # pairs = {'(': ')', '[': ']', '{': '}'}
    # Generate all valid combinations
    result = []

    def backtrack(current, stack, remaining):
        if not remaining and not stack:
            result.append(current)
            return

        for open_bracket, close_bracket in pairs.items():
            if remaining.get(open_bracket, 0) > 0:
                remaining[open_bracket] -= 1
                stack.append(close_bracket)
                backtrack(current + open_bracket, stack, remaining)
                stack.pop()
                remaining[open_bracket] += 1

        if stack:
            close = stack.pop()
            backtrack(current + close, stack, remaining)
            stack.append(close)

    # ...
\`\`\``,
	order: 7,
	translations: {
		ru: {
			title: 'Генерация скобок',
			description: `Сгенерируйте все допустимые комбинации из n пар скобок.

**Задача:**

Дано \`n\` пар скобок. Сгенерируйте все комбинации правильных скобочных последовательностей.

Строка правильная если:
1. Каждой '(' соответствует ')'
2. В любой точке количество ')' не превышает '('

**Примеры:**

\`\`\`
Вход: n = 3
Выход: ["((()))","(()())","(())()","()(())","()()()"]

Вход: n = 2
Выход: ["(())","()()"]
\`\`\`

**Ключевая идея:**

На каждом шаге можно:
1. Добавить '(' если open < n
2. Добавить ')' если close < open

**Ограничения:**
- 1 <= n <= 8

**Временная сложность:** O(4^n / √n) - число Каталана
**Пространственная сложность:** O(n)`,
			hint1: `Отслеживайте open и close. Можно добавить '(' если open < n. Можно добавить ')' только если close < open.`,
			hint2: `Базовый случай: длина строки 2n. Не нужно проверять в конце - правила гарантируют валидность.`,
			whyItMatters: `Generate Parentheses демонстрирует бэктрекинг с ограничениями. Количество комбинаций - числа Каталана.

**Почему это важно:**

**1. Числа Каталана**

Количество правильных скобочных последовательностей = число Каталана.

**2. Бэктрекинг с ограничениями**

Генерируем только валидные комбинации, не все 2^(2n).

**3. Применения**

Генерация выражений, сбалансированных скобок, XML-тегов.`,
			solutionCode: `from typing import List

def generate_parenthesis(n: int) -> List[str]:
    """Генерирует все допустимые скобочные комбинации."""
    result = []

    def backtrack(current: str, open_count: int, close_count: int) -> None:
        if len(current) == 2 * n:
            result.append(current)
            return

        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)

        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)

    backtrack('', 0, 0)
    return result`
		},
		uz: {
			title: 'Qavslarni yaratish',
			description: `n juft qavsning barcha to'g'ri kombinatsiyalarini yarating.

**Masala:**

\`n\` juft qavs berilgan. To'g'ri shakllangan qavslarning barcha kombinatsiyalarini yarating.

Satr to'g'ri shakllangan agar:
1. Har bir '(' ga mos ')' bor
2. Istalgan nuqtada ')' soni '(' dan oshmaydi

**Misollar:**

\`\`\`
Kirish: n = 3
Chiqish: ["((()))","(()())","(())()","()(())","()()()"]

Kirish: n = 2
Chiqish: ["(())","()()"]
\`\`\`

**Asosiy tushuncha:**

Har bir qadamda:
1. open < n bo'lsa '(' qo'shish mumkin
2. close < open bo'lsa ')' qo'shish mumkin

**Cheklovlar:**
- 1 <= n <= 8

**Vaqt murakkabligi:** O(4^n / √n) - Katalan soni
**Xotira murakkabligi:** O(n)`,
			hint1: `open va close ni kuzating. open < n bo'lsa '(' qo'shish mumkin. close < open bo'lsagina ')' qo'shish mumkin.`,
			hint2: `Asosiy holat: satr uzunligi 2n. Oxirida tekshirish shart emas - qoidalar to'g'rilikni kafolatlaydi.`,
			whyItMatters: `Generate Parentheses cheklovli backtracking ni ko'rsatadi. Kombinatsiyalar soni Katalan sonlari.

**Bu nima uchun muhim:**

**1. Katalan sonlari**

To'g'ri qavsli ketma-ketliklar soni = Katalan soni.

**2. Cheklovli backtracking**

Faqat to'g'ri kombinatsiyalarni yaratamiz, barchasini emas.

**3. Qo'llanishlar**

Ifodalar, muvozanatlangan qavslar, XML teglarini yaratish.`,
			solutionCode: `from typing import List

def generate_parenthesis(n: int) -> List[str]:
    """Barcha to'g'ri qavsli kombinatsiyalarni yaratadi."""
    result = []

    def backtrack(current: str, open_count: int, close_count: int) -> None:
        if len(current) == 2 * n:
            result.append(current)
            return

        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)

        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)

    backtrack('', 0, 0)
    return result`
		}
	}
};

export default task;
