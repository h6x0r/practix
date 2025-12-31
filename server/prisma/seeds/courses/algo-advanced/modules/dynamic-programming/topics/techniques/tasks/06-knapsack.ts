import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'dp-01-knapsack',
	title: '0/1 Knapsack',
	difficulty: 'medium',
	tags: ['python', 'dynamic-programming', 'knapsack', 'optimization'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Solve the classic 0/1 Knapsack problem using Dynamic Programming.

**Problem:**

Given \`n\` items, each with a weight and a value, determine the maximum value you can carry in a knapsack of capacity \`W\`.

Each item can be included at most once (0/1 choice - either take it or leave it).

**Examples:**

\`\`\`
Input: weights = [1, 2, 3], values = [6, 10, 12], capacity = 5
Output: 22
Explanation: Take items with weights 2 and 3 (values 10 + 12 = 22)

Input: weights = [1, 3, 4, 5], values = [1, 4, 5, 7], capacity = 7
Output: 9
Explanation: Take items with weights 3 and 4 (values 4 + 5 = 9)

Input: weights = [10, 20, 30], values = [60, 100, 120], capacity = 50
Output: 220
Explanation: Take items with weights 20 and 30 (values 100 + 120 = 220)
\`\`\`

**Key Insight:**

For each item, we have two choices:
1. **Don't take it:** Value = dp[i-1][w]
2. **Take it (if fits):** Value = dp[i-1][w-weight[i]] + value[i]

Choose the maximum of both options.

**DP Table:**

\`\`\`
weights = [1, 2, 3], values = [6, 10, 12], W = 5

        capacity →
item ↓    0    1    2    3    4    5
  0       0    0    0    0    0    0
  1       0    6    6    6    6    6
  2       0    6   10   16   16   16
  3       0    6   10   16   18   22
\`\`\`

**Constraints:**
- 1 <= n <= 100
- 1 <= weights[i], values[i] <= 1000
- 1 <= capacity <= 1000

**Time Complexity:** O(n × capacity)
**Space Complexity:** O(n × capacity), can be optimized to O(capacity)`,
	initialCode: `from typing import List

def knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    # TODO: Find maximum value that can fit in knapsack (0/1 knapsack)

    return 0`,
	solutionCode: `from typing import List


def knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Find maximum value that can fit in knapsack.

    Args:
        weights: Weight of each item
        values: Value of each item
        capacity: Maximum weight capacity

    Returns:
        Maximum value achievable
    """
    n = len(weights)

    # Create 2D DP table
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Fill the table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Option 1: Don't take item i
            dp[i][w] = dp[i - 1][w]

            # Option 2: Take item i (if it fits)
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i][w],
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]
                )

    return dp[n][capacity]


# Space-optimized version: O(capacity) space
def knapsack_optimized(weights: List[int], values: List[int], capacity: int) -> int:
    """Space-optimized 0/1 knapsack using 1D array."""
    n = len(weights)
    dp = [0] * (capacity + 1)

    for i in range(n):
        # Iterate backwards to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]


# Version that also returns selected items
def knapsack_with_items(weights: List[int], values: List[int], capacity: int) -> tuple:
    """Returns maximum value and list of selected item indices."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

    # Backtrack to find selected items
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)  # Item index
            w -= weights[i - 1]

    return dp[n][capacity], list(reversed(selected))`,
	testCode: `import pytest
from solution import knapsack


class TestKnapsack:
    def test_basic_example(self):
        """Test weights=[1,2,3], values=[6,10,12], W=5 -> 22"""
        assert knapsack([1, 2, 3], [6, 10, 12], 5) == 22

    def test_example_2(self):
        """Test weights=[1,3,4,5], values=[1,4,5,7], W=7 -> 9"""
        assert knapsack([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9

    def test_example_3(self):
        """Test weights=[10,20,30], values=[60,100,120], W=50 -> 220"""
        assert knapsack([10, 20, 30], [60, 100, 120], 50) == 220

    def test_zero_capacity(self):
        """Test zero capacity"""
        assert knapsack([1, 2, 3], [10, 20, 30], 0) == 0

    def test_no_items(self):
        """Test empty items"""
        assert knapsack([], [], 10) == 0

    def test_single_item_fits(self):
        """Test single item that fits"""
        assert knapsack([5], [10], 10) == 10

    def test_single_item_doesnt_fit(self):
        """Test single item that doesn't fit"""
        assert knapsack([15], [10], 10) == 0

    def test_all_items_fit(self):
        """Test when all items fit"""
        assert knapsack([1, 2, 3], [10, 20, 30], 10) == 60

    def test_exact_capacity(self):
        """Test when items exactly fill capacity"""
        assert knapsack([2, 3], [10, 15], 5) == 25

    def test_greedy_fails(self):
        """Test case where greedy (best value/weight) fails"""
        # Greedy would pick item 2 (v/w = 7/3 = 2.33), then item 1
        # Optimal is items 1 and 3
        assert knapsack([3, 4, 5], [4, 5, 7], 8) == 11`,
	hint1: `Create a 2D table dp[i][w] representing maximum value using first i items with capacity w. Base cases: dp[0][w] = 0 and dp[i][0] = 0.`,
	hint2: `For each item, you have two choices: skip it (dp[i][w] = dp[i-1][w]) or take it if it fits (dp[i][w] = dp[i-1][w-weight] + value). Take the maximum.`,
	whyItMatters: `The 0/1 Knapsack is one of the most important DP problems. It's the foundation for many optimization problems and appears frequently in FAANG interviews.

**Why This Matters:**

**1. The Two Choices Pattern**

\`\`\`python
# For each item: take it or leave it
# This pattern appears in MANY problems:

# Knapsack
dp[i][w] = max(
    dp[i-1][w],                              # Don't take
    dp[i-1][w-weight[i]] + value[i]          # Take
)

# Subset Sum
dp[i][s] = dp[i-1][s] or dp[i-1][s-nums[i]]  # Exists or not

# Partition Equal Subset Sum
# Same as subset sum with target = total/2
\`\`\`

**2. Space Optimization**

\`\`\`python
# 2D to 1D: iterate backwards!
dp = [0] * (capacity + 1)

for i in range(n):
    # MUST go backwards to avoid using updated values
    for w in range(capacity, weights[i] - 1, -1):
        dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

# Why backwards?
# dp[w] needs dp[w - weight] from PREVIOUS row
# Going backwards ensures we don't overwrite needed values
\`\`\`

**3. Knapsack Variants**

\`\`\`python
# Unbounded Knapsack (items can be used multiple times)
for w in range(capacity + 1):  # Forward iteration!
    for i in range(n):
        if weights[i] <= w:
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

# Bounded Knapsack (limited quantity of each item)
# Use binary representation or treat as multiple items

# Fractional Knapsack (can take partial items)
# Greedy works! Sort by value/weight ratio
\`\`\`

**4. Related FAANG Problems**

\`\`\`python
# Target Sum (LeetCode 494)
# Find ways to assign +/- to reach target

# Coin Change (covered earlier)
# Minimum coins for amount

# Partition Equal Subset Sum (LeetCode 416)
def can_partition(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    return dp[target]
\`\`\`

**5. Real-World Applications**

- Resource allocation
- Budget optimization
- Cargo loading
- Investment portfolio selection
- Memory/storage management`,
	order: 6,
	translations: {
		ru: {
			title: 'Задача о рюкзаке 0/1',
			description: `Решите классическую задачу о рюкзаке 0/1 с помощью динамического программирования.

**Задача:**

Дано \`n\` предметов, каждый с весом и ценностью. Определите максимальную ценность, которую можно унести в рюкзаке вместимостью \`W\`.

Каждый предмет можно взять не более одного раза (выбор 0/1 - либо берём, либо нет).

**Примеры:**

\`\`\`
Вход: weights = [1, 2, 3], values = [6, 10, 12], capacity = 5
Выход: 22
Объяснение: Берём предметы с весами 2 и 3 (ценности 10 + 12 = 22)

Вход: weights = [10, 20, 30], values = [60, 100, 120], capacity = 50
Выход: 220
\`\`\`

**Ключевая идея:**

Для каждого предмета есть два выбора:
1. **Не брать:** Ценность = dp[i-1][w]
2. **Взять (если помещается):** Ценность = dp[i-1][w-weight[i]] + value[i]

Выбираем максимум из обоих вариантов.

**Ограничения:**
- 1 <= n <= 100
- 1 <= weights[i], values[i] <= 1000
- 1 <= capacity <= 1000

**Временная сложность:** O(n × capacity)
**Пространственная сложность:** O(capacity) с оптимизацией`,
			hint1: `Создайте 2D таблицу dp[i][w] - максимальная ценность первых i предметов при вместимости w. Базовые случаи: dp[0][w] = 0 и dp[i][0] = 0.`,
			hint2: `Для каждого предмета два выбора: пропустить (dp[i][w] = dp[i-1][w]) или взять если помещается (dp[i][w] = dp[i-1][w-weight] + value). Берём максимум.`,
			whyItMatters: `Задача о рюкзаке 0/1 - одна из важнейших задач DP. Это основа многих оптимизационных задач, часто встречается на FAANG-интервью.

**Почему это важно:**

**1. Паттерн двух выборов**

Для каждого предмета: взять или оставить. Этот паттерн встречается во многих задачах.

**2. Оптимизация памяти**

2D -> 1D: итерация в обратном порядке!

**3. Варианты рюкзака**

- Неограниченный рюкзак
- Ограниченный рюкзак
- Дробный рюкзак (жадный алгоритм работает)

**4. Применения в реальном мире**

Распределение ресурсов, оптимизация бюджета, загрузка грузов.`,
			solutionCode: `from typing import List


def knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Находит максимальную ценность, помещающуюся в рюкзак.

    Args:
        weights: Вес каждого предмета
        values: Ценность каждого предмета
        capacity: Максимальная вместимость

    Returns:
        Максимально достижимая ценность
    """
    n = len(weights)

    # Создаём 2D DP таблицу
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Заполняем таблицу
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Вариант 1: Не берём предмет i
            dp[i][w] = dp[i - 1][w]

            # Вариант 2: Берём предмет i (если помещается)
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i][w],
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]
                )

    return dp[n][capacity]`
		},
		uz: {
			title: '0/1 Ryukzak masalasi',
			description: `Dinamik dasturlash yordamida klassik 0/1 Ryukzak masalasini yeching.

**Masala:**

\`n\` ta buyum berilgan, har birining og'irligi va qiymati bor. \`W\` sig'imli ryukzakda ko'tara oladigan maksimal qiymatni aniqlang.

Har bir buyumni ko'pi bilan bir marta olish mumkin (0/1 tanlov - yoki olasiz yoki yo'q).

**Misollar:**

\`\`\`
Kirish: weights = [1, 2, 3], values = [6, 10, 12], capacity = 5
Chiqish: 22
Izoh: 2 va 3 og'irlikdagi buyumlarni olamiz (qiymatlar 10 + 12 = 22)

Kirish: weights = [10, 20, 30], values = [60, 100, 120], capacity = 50
Chiqish: 220
\`\`\`

**Asosiy tushuncha:**

Har bir buyum uchun ikkita tanlov bor:
1. **Olmaslik:** Qiymat = dp[i-1][w]
2. **Olish (agar sig'sa):** Qiymat = dp[i-1][w-weight[i]] + value[i]

Ikkala variantdan maksimumni tanlaymiz.

**Cheklovlar:**
- 1 <= n <= 100
- 1 <= weights[i], values[i] <= 1000
- 1 <= capacity <= 1000

**Vaqt murakkabligi:** O(n × capacity)
**Xotira murakkabligi:** O(capacity) optimallashtirish bilan`,
			hint1: `dp[i][w] 2D jadval yarating - w sig'imida birinchi i buyumning maksimal qiymati. Asosiy holatlar: dp[0][w] = 0 va dp[i][0] = 0.`,
			hint2: `Har bir buyum uchun ikkita tanlov: o'tkazib yuborish (dp[i][w] = dp[i-1][w]) yoki sig'sa olish (dp[i][w] = dp[i-1][w-weight] + value). Maksimumni olamiz.`,
			whyItMatters: `0/1 Ryukzak masalasi - eng muhim DP masalalaridan biri. Bu ko'plab optimallashtirish masalalari uchun asos bo'lib, FAANG intervyularida tez-tez uchraydi.

**Bu nima uchun muhim:**

**1. Ikki tanlov patterni**

Har bir buyum uchun: olish yoki qoldirish. Bu pattern ko'p masalalarda uchraydi.

**2. Xotira optimallashtirish**

2D -> 1D: teskari tartibda iteratsiya!

**3. Ryukzak variantlari**

- Cheklanmagan ryukzak
- Chegaralangan ryukzak
- Kasr ryukzak (ochko'z algoritm ishlaydi)

**4. Haqiqiy dunyo qo'llanishlari**

Resurs taqsimlash, byudjet optimallashtirish, yuk yuklash.`,
			solutionCode: `from typing import List


def knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Ryukzakka sig'adigan maksimal qiymatni topadi.

    Args:
        weights: Har bir buyumning og'irligi
        values: Har bir buyumning qiymati
        capacity: Maksimal sig'im

    Returns:
        Erishish mumkin bo'lgan maksimal qiymat
    """
    n = len(weights)

    # 2D DP jadval yaratamiz
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Jadvalni to'ldiramiz
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Variant 1: i buyumni olmaymiz
            dp[i][w] = dp[i - 1][w]

            # Variant 2: i buyumni olamiz (agar sig'sa)
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i][w],
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]
                )

    return dp[n][capacity]`
		}
	}
};

export default task;
