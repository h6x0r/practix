import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'backtracking-combination-sum',
	title: 'Combination Sum',
	difficulty: 'medium',
	tags: ['python', 'backtracking', 'recursion', 'array'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find all unique combinations of candidates that sum to target.

**Problem:**

Given an array of **distinct** integers \`candidates\` and a target integer \`target\`, return all **unique combinations** of candidates where the chosen numbers sum to target.

The **same number may be chosen unlimited times**. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

Return the combinations in **any order**.

**Examples:**

\`\`\`
Input: candidates = [2, 3, 6, 7], target = 7
Output: [[2, 2, 3], [7]]

Explanation:
- 2 + 2 + 3 = 7
- 7 = 7

Input: candidates = [2, 3, 5], target = 8
Output: [[2, 2, 2, 2], [2, 3, 3], [3, 5]]

Input: candidates = [2], target = 1
Output: []
\`\`\`

**Key Insight:**

Unlike regular combinations, the same element can be used multiple times. So when recursing, use the **same start index** (not start + 1).

**Constraints:**
- 1 <= candidates.length <= 30
- 2 <= candidates[i] <= 40
- All elements of candidates are **distinct**
- 1 <= target <= 40

**Time Complexity:** O(n^(target/min)) in worst case
**Space Complexity:** O(target/min) for recursion depth`,
	initialCode: `from typing import List

def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    # TODO: Find all unique combinations that sum to target

    return []`,
	solutionCode: `from typing import List

def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    Find all combinations summing to target.
    """
    result = []
    candidates.sort()  # Sort for early termination

    def backtrack(start: int, current: List[int], remaining: int) -> None:
        if remaining == 0:
            result.append(current.copy())
            return

        for i in range(start, len(candidates)):
            # Optimization: break early if current candidate > remaining
            if candidates[i] > remaining:
                break

            current.append(candidates[i])
            # Use same index (i) to allow reuse of same element
            backtrack(i, current, remaining - candidates[i])
            current.pop()

    backtrack(0, [], target)
    return result


# Combination Sum II: Each number used at most once
def combination_sum2(candidates: List[int], target: int) -> List[List[int]]:
    """Find combinations where each number used at most once."""
    result = []
    candidates.sort()

    def backtrack(start: int, current: List[int], remaining: int) -> None:
        if remaining == 0:
            result.append(current.copy())
            return

        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break

            # Skip duplicates at same level
            if i > start and candidates[i] == candidates[i - 1]:
                continue

            current.append(candidates[i])
            backtrack(i + 1, current, remaining - candidates[i])  # i+1 for no reuse
            current.pop()

    backtrack(0, [], target)
    return result


# Combination Sum III: Find k numbers that sum to n (1-9 only)
def combination_sum3(k: int, n: int) -> List[List[int]]:
    """Find k numbers from 1-9 that sum to n."""
    result = []

    def backtrack(start: int, current: List[int], remaining: int) -> None:
        if len(current) == k:
            if remaining == 0:
                result.append(current.copy())
            return

        for i in range(start, 10):
            if i > remaining:
                break
            current.append(i)
            backtrack(i + 1, current, remaining - i)
            current.pop()

    backtrack(1, [], n)
    return result


# Combination Sum IV: Count combinations (DP approach)
def combination_sum4(nums: List[int], target: int) -> int:
    """Count number of combinations (order matters - permutations)."""
    dp = [0] * (target + 1)
    dp[0] = 1

    for i in range(1, target + 1):
        for num in nums:
            if i >= num:
                dp[i] += dp[i - num]

    return dp[target]`,
	testCode: `import pytest
from solution import combination_sum


class TestCombinationSum:
    def test_basic(self):
        """Test basic case"""
        result = combination_sum([2, 3, 6, 7], 7)
        assert len(result) == 2
        assert sorted([2, 2, 3]) in [sorted(r) for r in result]
        assert [7] in result

    def test_multiple_ways(self):
        """Test target with multiple combinations"""
        result = combination_sum([2, 3, 5], 8)
        assert len(result) == 3

    def test_no_solution(self):
        """Test when no solution exists"""
        result = combination_sum([2], 1)
        assert result == []

    def test_single_element_used_multiple_times(self):
        """Test reusing same element"""
        result = combination_sum([2], 4)
        assert result == [[2, 2]]

    def test_target_equals_candidate(self):
        """Test when target equals a candidate"""
        result = combination_sum([3, 5, 7], 7)
        assert [7] in result

    def test_sum_correctness(self):
        """Test that all combinations sum to target"""
        target = 10
        result = combination_sum([2, 3, 5], target)
        for combo in result:
            assert sum(combo) == target

    def test_no_duplicates(self):
        """Test no duplicate combinations"""
        result = combination_sum([2, 3, 5], 8)
        tuples = [tuple(sorted(r)) for r in result]
        assert len(tuples) == len(set(tuples))

    def test_sorted_order(self):
        """Test elements in each combination are in order"""
        result = combination_sum([5, 2, 3], 8)
        for combo in result:
            assert combo == sorted(combo)

    def test_larger_target(self):
        """Test with larger target"""
        result = combination_sum([2, 3], 12)
        for combo in result:
            assert sum(combo) == 12

    def test_single_candidate(self):
        """Test with single candidate"""
        result = combination_sum([3], 9)
        assert result == [[3, 3, 3]]`,
	hint1: `Sort candidates for optimization. Use backtracking with (start, current, remaining). When remaining is 0, found a combination.`,
	hint2: `Key difference: use same index (i) when recursing to allow reusing elements. Break early when candidate > remaining (since sorted).`,
	whyItMatters: `Combination Sum is a classic backtracking problem that teaches handling unlimited element reuse. It has applications in coin change, resource allocation, and optimization.

**Why This Matters:**

**1. Combination Sum Variants**

\`\`\`python
# Original (unlimited reuse):
backtrack(i, ...)  # Same index

# Combination Sum II (no reuse, with duplicates):
backtrack(i + 1, ...)  # Next index
if i > start and nums[i] == nums[i-1]: continue  # Skip dups

# Combination Sum III (1-9, exactly k numbers):
if len(current) == k and remaining == 0: ...

# Combination Sum IV (count, order matters):
# Use DP, not backtracking
\`\`\`

**2. Optimization Techniques**

\`\`\`python
def combination_sum_optimized(candidates, target):
    candidates.sort()  # Sort for early termination
    result = []

    def backtrack(start, current, remaining):
        if remaining == 0:
            result.append(current.copy())
            return

        for i in range(start, len(candidates)):
            # Early termination: if current > remaining, all subsequent too
            if candidates[i] > remaining:
                break

            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])
            current.pop()

    backtrack(0, [], target)
    return result
\`\`\`

**3. Coin Change Connection**

\`\`\`python
# Combination Sum finds all combinations
# Coin Change finds minimum count

def coin_change(coins, amount):
    # DP approach for minimum coins
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1

    return dp[amount] if dp[amount] != float('inf') else -1
\`\`\`

**4. Applications**

\`\`\`python
# Shopping cart totals
# Resource allocation
# Scheduling with durations
# Game level design (point combinations)
\`\`\``,
	order: 5,
	translations: {
		ru: {
			title: 'Сумма комбинаций',
			description: `Найдите все уникальные комбинации кандидатов, сумма которых равна цели.

**Задача:**

Дан массив **различных** целых чисел \`candidates\` и целевое число \`target\`. Верните все уникальные комбинации, сумма которых равна target.

**Одно число может использоваться неограниченное количество раз.**

**Примеры:**

\`\`\`
Вход: candidates = [2, 3, 6, 7], target = 7
Выход: [[2, 2, 3], [7]]

Объяснение:
- 2 + 2 + 3 = 7
- 7 = 7
\`\`\`

**Ключевая идея:**

В отличие от обычных комбинаций, один элемент можно использовать многократно. Поэтому при рекурсии используем **тот же индекс** (не start + 1).

**Ограничения:**
- 1 <= candidates.length <= 30
- 1 <= target <= 40

**Временная сложность:** O(n^(target/min))
**Пространственная сложность:** O(target/min)`,
			hint1: `Отсортируйте кандидатов для оптимизации. Бэктрекинг с (start, current, remaining). Когда remaining == 0, нашли комбинацию.`,
			hint2: `Ключевое отличие: используйте тот же индекс (i) при рекурсии для повторного использования. Прервите когда candidate > remaining.`,
			whyItMatters: `Combination Sum учит обработке неограниченного повторного использования элементов. Применяется в размене монет и распределении ресурсов.

**Почему это важно:**

**1. Варианты Combination Sum**

Оригинальный (неограниченное повторение), II (без повторения), III (k чисел из 1-9), IV (подсчёт).

**2. Техники оптимизации**

Сортировка для раннего прерывания.

**3. Связь с разменом монет**

Combination Sum находит все комбинации, Coin Change - минимальное количество.`,
			solutionCode: `from typing import List

def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """Находит все комбинации с суммой равной target."""
    result = []
    candidates.sort()

    def backtrack(start: int, current: List[int], remaining: int) -> None:
        if remaining == 0:
            result.append(current.copy())
            return

        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break

            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])
            current.pop()

    backtrack(0, [], target)
    return result`
		},
		uz: {
			title: 'Kombinatsiya yig\'indisi',
			description: `Nomzodlarning yig'indisi maqsadga teng bo'lgan barcha noyob kombinatsiyalarni toping.

**Masala:**

**Turli** butun sonlar massivi \`candidates\` va maqsad \`target\` berilgan. Yig'indisi target ga teng bo'lgan barcha noyob kombinatsiyalarni qaytaring.

**Bir xil son cheksiz marta ishlatilishi mumkin.**

**Misollar:**

\`\`\`
Kirish: candidates = [2, 3, 6, 7], target = 7
Chiqish: [[2, 2, 3], [7]]

Izoh:
- 2 + 2 + 3 = 7
- 7 = 7
\`\`\`

**Asosiy tushuncha:**

Oddiy kombinatsiyalardan farqli, bitta element ko'p marta ishlatilishi mumkin. Shuning uchun rekursiyada **bir xil indeks** (start + 1 emas) ishlating.

**Cheklovlar:**
- 1 <= candidates.length <= 30
- 1 <= target <= 40

**Vaqt murakkabligi:** O(n^(target/min))
**Xotira murakkabligi:** O(target/min)`,
			hint1: `Optimallashtirish uchun nomzodlarni saralang. (start, current, remaining) bilan backtracking. remaining == 0 bo'lganda kombinatsiya topildi.`,
			hint2: `Asosiy farq: qayta ishlatish uchun rekursiyada bir xil indeks (i) ishlating. candidate > remaining bo'lganda to'xtating.`,
			whyItMatters: `Combination Sum elementlarni cheksiz qayta ishlatishni o'rgatadi. Tanga almashtirish va resurslarni taqsimlashda qo'llaniladi.

**Bu nima uchun muhim:**

**1. Combination Sum variantlari**

Asl (cheksiz takrorlash), II (takrorlanmasdan), III (1-9 dan k ta son), IV (hisoblash).

**2. Optimallashtirish texnikalari**

Erta to'xtatish uchun saralash.

**3. Tanga almashtirish bilan bog'liqlik**

Combination Sum barcha kombinatsiyalarni topadi, Coin Change - minimal miqdorni.`,
			solutionCode: `from typing import List

def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """Yig'indisi target ga teng barcha kombinatsiyalarni topadi."""
    result = []
    candidates.sort()

    def backtrack(start: int, current: List[int], remaining: int) -> None:
        if remaining == 0:
            result.append(current.copy())
            return

        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break

            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])
            current.pop()

    backtrack(0, [], target)
    return result`
		}
	}
};

export default task;
