import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'backtracking-combinations',
	title: 'Combinations',
	difficulty: 'medium',
	tags: ['python', 'backtracking', 'recursion', 'combinatorics'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Generate all combinations of k numbers from 1 to n.

**Problem:**

Given two integers \`n\` and \`k\`, return all possible combinations of \`k\` numbers chosen from the range \`[1, n]\`.

You may return the answer in **any order**.

**Examples:**

\`\`\`
Input: n = 4, k = 2
Output: [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]

Input: n = 1, k = 1
Output: [[1]]

Input: n = 5, k = 3
Output: [[1,2,3], [1,2,4], [1,2,5], [1,3,4], [1,3,5], [1,4,5],
         [2,3,4], [2,3,5], [2,4,5], [3,4,5]]
\`\`\`

**Key Insight:**

Combinations differ from permutations:
- **Permutations**: Order matters - [1,2] ≠ [2,1]
- **Combinations**: Order doesn't matter - [1,2] = [2,1]

To avoid duplicates, always pick numbers in increasing order (use a start index).

**Constraints:**
- 1 <= n <= 20
- 1 <= k <= n

**Time Complexity:** O(k × C(n,k)) where C(n,k) = n!/(k!(n-k)!)
**Space Complexity:** O(k) for recursion stack`,
	initialCode: `from typing import List

def combine(n: int, k: int) -> List[List[int]]:
    # TODO: Generate all combinations of k numbers from 1 to n

    return []`,
	solutionCode: `from typing import List

def combine(n: int, k: int) -> List[List[int]]:
    """
    Generate all combinations of k numbers from 1 to n.
    """
    result = []

    def backtrack(start: int, current: List[int]) -> None:
        if len(current) == k:
            result.append(current.copy())
            return

        # Optimization: need at least (k - len(current)) more numbers
        # So we can only go up to n - (k - len(current)) + 1
        need = k - len(current)
        for i in range(start, n - need + 2):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()

    backtrack(1, [])
    return result


# Without optimization (cleaner but slower)
def combine_simple(n: int, k: int) -> List[List[int]]:
    """Simple version without pruning."""
    result = []

    def backtrack(start: int, current: List[int]) -> None:
        if len(current) == k:
            result.append(current.copy())
            return

        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()

    backtrack(1, [])
    return result


# Iterative approach using bit manipulation
def combine_iterative(n: int, k: int) -> List[List[int]]:
    """Generate combinations using bitmasks."""
    result = []

    for mask in range(1 << n):
        if bin(mask).count('1') == k:
            combination = [i + 1 for i in range(n) if mask & (1 << i)]
            result.append(combination)

    return result


# Using itertools (for reference)
from itertools import combinations

def combine_itertools(n: int, k: int) -> List[List[int]]:
    """Using Python's itertools."""
    return [list(c) for c in combinations(range(1, n + 1), k)]`,
	testCode: `import pytest
from solution import combine


class TestCombinations:
    def test_n4_k2(self):
        """Test n=4, k=2"""
        result = combine(4, 2)
        assert len(result) == 6  # C(4,2) = 6
        expected = [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
        for comb in expected:
            assert comb in result

    def test_n1_k1(self):
        """Test n=1, k=1"""
        result = combine(1, 1)
        assert result == [[1]]

    def test_n5_k3(self):
        """Test n=5, k=3"""
        result = combine(5, 3)
        assert len(result) == 10  # C(5,3) = 10

    def test_n_equals_k(self):
        """Test when n equals k (only one combination)"""
        result = combine(3, 3)
        assert result == [[1, 2, 3]]

    def test_k_equals_1(self):
        """Test when k equals 1 (n combinations)"""
        result = combine(5, 1)
        assert len(result) == 5
        for i in range(1, 6):
            assert [i] in result

    def test_correct_count(self):
        """Test correct number of combinations C(n,k)"""
        from math import comb
        for n in range(1, 8):
            for k in range(1, n + 1):
                result = combine(n, k)
                assert len(result) == comb(n, k)

    def test_no_duplicates(self):
        """Test no duplicate combinations"""
        result = combine(5, 3)
        tuples = [tuple(c) for c in result]
        assert len(tuples) == len(set(tuples))

    def test_combination_size(self):
        """Test each combination has exactly k elements"""
        result = combine(6, 4)
        for comb in result:
            assert len(comb) == 4

    def test_elements_in_range(self):
        """Test all elements are in range [1, n]"""
        result = combine(5, 3)
        for comb in result:
            for num in comb:
                assert 1 <= num <= 5

    def test_sorted_order(self):
        """Test elements within each combination are sorted"""
        result = combine(5, 3)
        for comb in result:
            assert comb == sorted(comb)`,
	hint1: `Use backtracking with a start index. Always pick numbers >= start to ensure increasing order and avoid duplicates like [1,2] and [2,1].`,
	hint2: `Optimize by pruning: if remaining numbers (n - start + 1) < needed numbers (k - len(current)), stop early. Base case: when current has k elements.`,
	whyItMatters: `Combinations are fundamental in combinatorics. They appear in probability, statistics, and countless algorithmic problems involving selection without regard to order.

**Why This Matters:**

**1. Combinations vs Permutations vs Subsets**

\`\`\`python
# From {1, 2, 3}, choose 2:

# Combinations (order doesn't matter):
# {1,2}, {1,3}, {2,3}  → C(3,2) = 3

# Permutations (order matters):
# (1,2), (2,1), (1,3), (3,1), (2,3), (3,2)  → P(3,2) = 6

# Subsets (any size):
# {}, {1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}  → 2^3 = 8
\`\`\`

**2. Pruning Optimization**

\`\`\`python
def combine_optimized(n, k):
    result = []

    def backtrack(start, current):
        if len(current) == k:
            result.append(current.copy())
            return

        # Pruning: need (k - len(current)) more numbers
        # Available: n - start + 1
        # If not enough, skip this branch
        need = k - len(current)
        available = n - start + 1
        if available < need:
            return

        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()

    backtrack(1, [])
    return result
\`\`\`

**3. Combination Sum Variants**

\`\`\`python
# Combination Sum: find combinations that sum to target
def combination_sum(candidates, target):
    result = []

    def backtrack(start, current, remaining):
        if remaining == 0:
            result.append(current.copy())
            return
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            current.append(candidates[i])
            # i (not i+1) allows reuse of same element
            backtrack(i, current, remaining - candidates[i])
            current.pop()

    backtrack(0, [], target)
    return result
\`\`\`

**4. Pascal's Triangle Connection**

\`\`\`
C(n,k) = C(n-1,k-1) + C(n-1,k)

     1           C(0,0)
    1 1          C(1,0) C(1,1)
   1 2 1         C(2,0) C(2,1) C(2,2)
  1 3 3 1        C(3,0) C(3,1) C(3,2) C(3,3)
 1 4 6 4 1       ...
\`\`\`

**5. Applications**

\`\`\`python
# Lottery numbers
# Team selection
# Feature selection in ML
# Graph edge selection
# Password combinations
\`\`\``,
	order: 3,
	translations: {
		ru: {
			title: 'Комбинации',
			description: `Сгенерируйте все комбинации из k чисел от 1 до n.

**Задача:**

Даны два целых числа \`n\` и \`k\`. Верните все возможные комбинации из \`k\` чисел из диапазона \`[1, n]\`.

**Примеры:**

\`\`\`
Вход: n = 4, k = 2
Выход: [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]

Вход: n = 5, k = 3
Выход: [[1,2,3], [1,2,4], [1,2,5], [1,3,4], [1,3,5], [1,4,5],
        [2,3,4], [2,3,5], [2,4,5], [3,4,5]]
\`\`\`

**Ключевая идея:**

- **Перестановки**: Порядок важен - [1,2] ≠ [2,1]
- **Комбинации**: Порядок не важен - [1,2] = [2,1]

Чтобы избежать дубликатов, выбираем числа в возрастающем порядке.

**Ограничения:**
- 1 <= n <= 20
- 1 <= k <= n

**Временная сложность:** O(k × C(n,k))
**Пространственная сложность:** O(k)`,
			hint1: `Используйте бэктрекинг с начальным индексом. Выбирайте числа >= start для возрастающего порядка и избежания дубликатов.`,
			hint2: `Оптимизируйте отсечением: если оставшихся чисел < нужных, остановитесь. Базовый случай: когда current содержит k элементов.`,
			whyItMatters: `Комбинации фундаментальны в комбинаторике. Они появляются в вероятности, статистике и задачах выбора без учёта порядка.

**Почему это важно:**

**1. Комбинации vs Перестановки vs Подмножества**

Понимание различий критично.

**2. Оптимизация отсечением**

Пропуск ветвей когда недостаточно оставшихся элементов.

**3. Связь с треугольником Паскаля**

C(n,k) = C(n-1,k-1) + C(n-1,k)`,
			solutionCode: `from typing import List

def combine(n: int, k: int) -> List[List[int]]:
    """Генерирует все комбинации из k чисел от 1 до n."""
    result = []

    def backtrack(start: int, current: List[int]) -> None:
        if len(current) == k:
            result.append(current.copy())
            return

        need = k - len(current)
        for i in range(start, n - need + 2):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()

    backtrack(1, [])
    return result`
		},
		uz: {
			title: 'Kombinatsiyalar',
			description: `1 dan n gacha bo'lgan k ta sonning barcha kombinatsiyalarini yarating.

**Masala:**

Ikkita butun son \`n\` va \`k\` berilgan. \`[1, n]\` diapazonidan \`k\` ta sonning barcha mumkin bo'lgan kombinatsiyalarini qaytaring.

**Misollar:**

\`\`\`
Kirish: n = 4, k = 2
Chiqish: [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]

Kirish: n = 5, k = 3
Chiqish: [[1,2,3], [1,2,4], [1,2,5], [1,3,4], [1,3,5], [1,4,5],
         [2,3,4], [2,3,5], [2,4,5], [3,4,5]]
\`\`\`

**Asosiy tushuncha:**

- **Permutatsiyalar**: Tartib muhim - [1,2] ≠ [2,1]
- **Kombinatsiyalar**: Tartib muhim emas - [1,2] = [2,1]

Dublikatlardan qochish uchun sonlarni o'sish tartibida tanlaymiz.

**Cheklovlar:**
- 1 <= n <= 20
- 1 <= k <= n

**Vaqt murakkabligi:** O(k × C(n,k))
**Xotira murakkabligi:** O(k)`,
			hint1: `Boshlang'ich indeks bilan backtracking ishlating. O'sish tartibini ta'minlash va dublikatlardan qochish uchun >= start sonlarni tanlang.`,
			hint2: `Kesish bilan optimallashtiring: agar qolgan sonlar < kerakli sonlar bo'lsa, to'xtating. Asosiy holat: current k ta elementga ega bo'lganda.`,
			whyItMatters: `Kombinatsiyalar kombinatorikada asosiy hisoblanadi. Ular ehtimollik, statistika va tartibni hisobga olmagan tanlash masalalarida uchraydi.

**Bu nima uchun muhim:**

**1. Kombinatsiyalar vs Permutatsiyalar vs Kichik to'plamlar**

Farqlarni tushunish muhim.

**2. Kesish bilan optimallashtirish**

Qolgan elementlar yetarli bo'lmaganda tarmoqlarni o'tkazib yuborish.

**3. Paskal uchburchagi bilan bog'liqlik**

C(n,k) = C(n-1,k-1) + C(n-1,k)`,
			solutionCode: `from typing import List

def combine(n: int, k: int) -> List[List[int]]:
    """1 dan n gacha k ta sonning barcha kombinatsiyalarini yaratadi."""
    result = []

    def backtrack(start: int, current: List[int]) -> None:
        if len(current) == k:
            result.append(current.copy())
            return

        need = k - len(current)
        for i in range(start, n - need + 2):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()

    backtrack(1, [])
    return result`
		}
	}
};

export default task;
