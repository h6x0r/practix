import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'dp-climbing-stairs',
	title: 'Climbing Stairs',
	difficulty: 'easy',
	tags: ['python', 'dynamic-programming', 'memoization', 'tabulation'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Count the number of distinct ways to climb to the top of a staircase.

**Problem:**

You are climbing a staircase. It takes \`n\` steps to reach the top. Each time you can either climb 1 or 2 steps.

In how many distinct ways can you climb to the top?

**Examples:**

\`\`\`
Input: n = 2
Output: 2
Explanation:
1. 1 step + 1 step
2. 2 steps

Input: n = 3
Output: 3
Explanation:
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step

Input: n = 4
Output: 5
Explanation:
1. 1+1+1+1
2. 1+1+2
3. 1+2+1
4. 2+1+1
5. 2+2
\`\`\`

**Key Insight:**

To reach step \`n\`, you must come from either:
- Step \`n-1\` (taking 1 step), or
- Step \`n-2\` (taking 2 steps)

So: \`ways(n) = ways(n-1) + ways(n-2)\`

This is exactly the Fibonacci sequence!

**Constraints:**
- 1 <= n <= 45

**Time Complexity:** O(n)
**Space Complexity:** O(1) with optimization`,
	initialCode: `def climb_stairs(n: int) -> int:
    # TODO: Count distinct ways to climb n stairs (1 or 2 steps at a time)

    return 0`,
	solutionCode: `def climb_stairs(n: int) -> int:
    """
    Count distinct ways to climb n stairs.

    Args:
        n: Number of stairs (1 <= n <= 45)

    Returns:
        Number of distinct ways to climb to the top
    """
    if n == 1:
        return 1
    if n == 2:
        return 2

    # Space-optimized approach
    prev2, prev1 = 1, 2

    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    return prev1


# Alternative: Tabulation approach
def climb_stairs_tab(n: int) -> int:
    """Bottom-up with full array."""
    if n <= 2:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2

    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


# Alternative: Memoization approach
def climb_stairs_memo(n: int, memo: dict = None) -> int:
    """Top-down with memoization."""
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    if n <= 2:
        return n

    memo[n] = climb_stairs_memo(n - 1, memo) + climb_stairs_memo(n - 2, memo)
    return memo[n]`,
	testCode: `import pytest
from solution import climb_stairs


class TestClimbingStairs:
    def test_one_stair(self):
        """Test n=1: only one way"""
        assert climb_stairs(1) == 1

    def test_two_stairs(self):
        """Test n=2: two ways (1+1 or 2)"""
        assert climb_stairs(2) == 2

    def test_three_stairs(self):
        """Test n=3: three ways"""
        assert climb_stairs(3) == 3

    def test_four_stairs(self):
        """Test n=4: five ways"""
        assert climb_stairs(4) == 5

    def test_five_stairs(self):
        """Test n=5: eight ways"""
        assert climb_stairs(5) == 8

    def test_ten_stairs(self):
        """Test n=10"""
        assert climb_stairs(10) == 89

    def test_twenty_stairs(self):
        """Test n=20"""
        assert climb_stairs(20) == 10946

    def test_fibonacci_relation(self):
        """Test that climb_stairs(n) = climb_stairs(n-1) + climb_stairs(n-2)"""
        for n in range(3, 15):
            assert climb_stairs(n) == climb_stairs(n - 1) + climb_stairs(n - 2)

    def test_large_input(self):
        """Test n=45 (maximum constraint)"""
        assert climb_stairs(45) == 1836311903

    def test_sequence(self):
        """Test first 10 values match expected sequence"""
        expected = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        for i, exp in enumerate(expected, start=1):
            assert climb_stairs(i) == exp`,
	hint1: `Think about how you can reach step n. You can only come from step n-1 (taking 1 step) or step n-2 (taking 2 steps).`,
	hint2: `The recurrence relation is: ways(n) = ways(n-1) + ways(n-2). This is like Fibonacci! Use space optimization with just two variables.`,
	whyItMatters: `Climbing Stairs is one of the most popular FAANG interview questions and teaches how to recognize Fibonacci-like patterns.

**Why This Matters:**

**1. Pattern Recognition**

Many DP problems are disguised Fibonacci sequences:

\`\`\`python
# Climbing Stairs: ways(n) = ways(n-1) + ways(n-2)
# House Robber: dp[i] = max(dp[i-1], dp[i-2] + nums[i])
# Decode Ways: similar pattern with conditions
\`\`\`

**2. State Definition**

The key is defining what dp[i] represents:

\`\`\`python
# dp[i] = number of distinct ways to reach step i
# To find dp[n], we need:
# - dp[n-1]: ways to reach step n-1, then take 1 step
# - dp[n-2]: ways to reach step n-2, then take 2 steps
\`\`\`

**3. Generalization: K Steps**

What if you can take 1, 2, or 3 steps?

\`\`\`python
def climb_k_stairs(n: int, k: int) -> int:
    """Climb with up to k steps at a time."""
    if n == 0:
        return 1

    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        for j in range(1, min(k, i) + 1):
            dp[i] += dp[i - j]

    return dp[n]
\`\`\`

**4. Common Variations (FAANG)**

\`\`\`python
# Min cost climbing stairs
def min_cost(cost):
    n = len(cost)
    dp = [0] * (n + 1)
    for i in range(2, n + 1):
        dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
    return dp[n]

# With obstacles (some steps blocked)
def climb_with_obstacles(n, obstacles):
    if obstacles[0] or obstacles[n]:
        return 0
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        if not obstacles[i]:
            dp[i] = dp[i-1] + (dp[i-2] if i >= 2 else 0)
    return dp[n]
\`\`\`

**5. Real-World Applications**

- Path counting in grids
- Decision tree analysis
- Game move combinations
- Combinatorics problems`,
	order: 2,
	translations: {
		ru: {
			title: 'Подъём по лестнице',
			description: `Подсчитайте количество различных способов подняться на вершину лестницы.

**Задача:**

Вы поднимаетесь по лестнице. Нужно \`n\` ступеней, чтобы достичь вершины. Каждый раз вы можете подняться на 1 или 2 ступени.

Сколько различных способов подняться на вершину?

**Примеры:**

\`\`\`
Вход: n = 2
Выход: 2
Объяснение:
1. 1 ступень + 1 ступень
2. 2 ступени

Вход: n = 3
Выход: 3
Объяснение:
1. 1 + 1 + 1
2. 1 + 2
3. 2 + 1

Вход: n = 4
Выход: 5
\`\`\`

**Ключевая идея:**

Чтобы достичь ступени \`n\`, вы должны прийти либо с:
- Ступени \`n-1\` (сделав 1 шаг), либо
- Ступени \`n-2\` (сделав 2 шага)

Итак: \`ways(n) = ways(n-1) + ways(n-2)\`

Это в точности последовательность Фибоначчи!

**Ограничения:**
- 1 <= n <= 45

**Временная сложность:** O(n)
**Пространственная сложность:** O(1) с оптимизацией`,
			hint1: `Подумайте, как вы можете достичь ступени n. Вы можете прийти только со ступени n-1 (сделав 1 шаг) или со ступени n-2 (сделав 2 шага).`,
			hint2: `Рекуррентное соотношение: ways(n) = ways(n-1) + ways(n-2). Это как Фибоначчи! Используйте оптимизацию памяти с двумя переменными.`,
			whyItMatters: `Climbing Stairs - один из самых популярных вопросов на FAANG-интервью. Он учит распознавать паттерны, похожие на Фибоначчи.

**Почему это важно:**

**1. Распознавание паттернов**

Многие DP задачи - это замаскированные последовательности Фибоначчи.

**2. Определение состояния**

Ключ - определить, что означает dp[i]:
- dp[i] = количество различных способов достичь ступени i

**3. Обобщение: K шагов**

Что если можно делать 1, 2 или 3 шага? Паттерн расширяется.

**4. Распространённые вариации (FAANG)**

- Минимальная стоимость подъёма
- С препятствиями (некоторые ступени заблокированы)

**5. Применения в реальном мире**

Подсчёт путей в сетках, анализ деревьев решений, комбинаторика.`,
			solutionCode: `def climb_stairs(n: int) -> int:
    """
    Подсчитывает различные способы подняться на n ступеней.

    Args:
        n: Количество ступеней (1 <= n <= 45)

    Returns:
        Количество различных способов подняться наверх
    """
    if n == 1:
        return 1
    if n == 2:
        return 2

    # Оптимизированный по памяти подход
    prev2, prev1 = 1, 2

    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    return prev1`
		},
		uz: {
			title: 'Zinadan chiqish',
			description: `Zinaning tepasiga chiqishning turli usullarini hisoblang.

**Masala:**

Siz zinadan chiqyapsiz. Tepaga yetish uchun \`n\` ta qadam kerak. Har safar 1 yoki 2 qadam ko'tarilishingiz mumkin.

Tepaga nechta turli usulda chiqish mumkin?

**Misollar:**

\`\`\`
Kirish: n = 2
Chiqish: 2
Izoh:
1. 1 qadam + 1 qadam
2. 2 qadam

Kirish: n = 3
Chiqish: 3
Izoh:
1. 1 + 1 + 1
2. 1 + 2
3. 2 + 1

Kirish: n = 4
Chiqish: 5
\`\`\`

**Asosiy tushuncha:**

\`n\`-chi qadamga yetish uchun siz quyidagilardan biridan kelishingiz kerak:
- \`n-1\`-chi qadam (1 qadam qilib), yoki
- \`n-2\`-chi qadam (2 qadam qilib)

Shunday qilib: \`ways(n) = ways(n-1) + ways(n-2)\`

Bu aynan Fibonachchi ketma-ketligi!

**Cheklovlar:**
- 1 <= n <= 45

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1) optimallashtirish bilan`,
			hint1: `n-chi qadamga qanday yetishingiz mumkinligini o'ylang. Faqat n-1 qadamdan (1 qadam qilib) yoki n-2 qadamdan (2 qadam qilib) kelishingiz mumkin.`,
			hint2: `Rekurrent munosabat: ways(n) = ways(n-1) + ways(n-2). Bu Fibonachchi kabi! Faqat ikki o'zgaruvchi bilan xotira optimallashtirish qo'llang.`,
			whyItMatters: `Climbing Stairs - FAANG intervyularida eng mashhur savollardan biri. U Fibonachchi-ga o'xshash patternlarni tanishni o'rgatadi.

**Bu nima uchun muhim:**

**1. Pattern tanish**

Ko'plab DP masalalari niqoblangan Fibonachchi ketma-ketliklari.

**2. Holat ta'rifi**

Kalit - dp[i] nimani anglatishini aniqlash:
- dp[i] = i-chi qadamga yetishning turli usullari soni

**3. Umumlashtirish: K qadam**

1, 2 yoki 3 qadam qilish mumkin bo'lsa-chi? Pattern kengayadi.

**4. Keng tarqalgan variatsiyalar (FAANG)**

- Minimal xarajat bilan ko'tarilish
- To'siqlar bilan (ba'zi qadamlar bloklangan)

**5. Haqiqiy dunyo qo'llanishlari**

To'rlarda yo'llarni hisoblash, qaror daraxtlari tahlili, kombinatorika.`,
			solutionCode: `def climb_stairs(n: int) -> int:
    """
    n ta zinaga chiqishning turli usullarini hisoblaydi.

    Args:
        n: Zinalar soni (1 <= n <= 45)

    Returns:
        Tepaga chiqishning turli usullari soni
    """
    if n == 1:
        return 1
    if n == 2:
        return 2

    # Xotira bo'yicha optimallashtirilgan yondashuv
    prev2, prev1 = 1, 2

    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    return prev1`
		}
	}
};

export default task;
