import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'dp-coin-change',
	title: 'Coin Change',
	difficulty: 'medium',
	tags: ['python', 'dynamic-programming', 'unbounded-knapsack', 'optimization'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the minimum number of coins needed to make up a given amount.

**Problem:**

You are given an integer array \`coins\` representing coins of different denominations and an integer \`amount\` representing a total amount of money.

Return the fewest number of coins needed to make up that amount. If that amount cannot be made up by any combination of coins, return -1.

You may assume you have an infinite number of each kind of coin.

**Examples:**

\`\`\`
Input: coins = [1, 2, 5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1

Input: coins = [2], amount = 3
Output: -1
Explanation: Cannot make 3 with only 2-cent coins

Input: coins = [1], amount = 0
Output: 0
Explanation: No coins needed for amount 0

Input: coins = [1, 2, 5], amount = 100
Output: 20
Explanation: 100 = 5 * 20
\`\`\`

**Key Insight:**

For each amount \`i\`, we try using each coin \`c\`:
- If we use coin \`c\`, we need 1 + dp[i - c] coins
- dp[i] = min(dp[i], 1 + dp[i - c]) for all valid coins

**Constraints:**
- 1 <= coins.length <= 12
- 1 <= coins[i] <= 2^31 - 1
- 0 <= amount <= 10^4

**Time Complexity:** O(amount × len(coins))
**Space Complexity:** O(amount)`,
	initialCode: `from typing import List

def coin_change(coins: List[int], amount: int) -> int:
    # TODO: Find minimum number of coins to make up the amount (-1 if impossible)

    return -1`,
	solutionCode: `from typing import List


def coin_change(coins: List[int], amount: int) -> int:
    """
    Find minimum number of coins to make up the amount.

    Args:
        coins: List of coin denominations
        amount: Target amount

    Returns:
        Minimum number of coins, or -1 if impossible
    """
    # dp[i] = minimum coins needed to make amount i
    # Initialize with amount + 1 (impossible value)
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0  # Base case: 0 coins for amount 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] <= amount else -1


# Alternative: BFS approach (finds minimum steps)
from collections import deque

def coin_change_bfs(coins: List[int], amount: int) -> int:
    """BFS approach - finds shortest path."""
    if amount == 0:
        return 0

    visited = set([0])
    queue = deque([(0, 0)])  # (current_amount, num_coins)

    while queue:
        curr_amount, num_coins = queue.popleft()

        for coin in coins:
            new_amount = curr_amount + coin
            if new_amount == amount:
                return num_coins + 1
            if new_amount < amount and new_amount not in visited:
                visited.add(new_amount)
                queue.append((new_amount, num_coins + 1))

    return -1


# Alternative: Top-down with memoization
def coin_change_memo(coins: List[int], amount: int) -> int:
    """Top-down memoization approach."""
    memo = {}

    def dp(remaining):
        if remaining == 0:
            return 0
        if remaining < 0:
            return float('inf')
        if remaining in memo:
            return memo[remaining]

        min_coins = float('inf')
        for coin in coins:
            result = dp(remaining - coin)
            if result != float('inf'):
                min_coins = min(min_coins, result + 1)

        memo[remaining] = min_coins
        return min_coins

    result = dp(amount)
    return result if result != float('inf') else -1`,
	testCode: `import pytest
from solution import coin_change


class TestCoinChange:
    def test_basic_example(self):
        """Test coins=[1,2,5], amount=11 -> 3"""
        assert coin_change([1, 2, 5], 11) == 3

    def test_impossible(self):
        """Test impossible case: coins=[2], amount=3"""
        assert coin_change([2], 3) == -1

    def test_zero_amount(self):
        """Test amount=0 -> 0 coins needed"""
        assert coin_change([1], 0) == 0
        assert coin_change([1, 2, 5], 0) == 0

    def test_single_coin_exact(self):
        """Test when single coin equals amount"""
        assert coin_change([5], 5) == 1
        assert coin_change([1, 5, 10], 10) == 1

    def test_single_coin_type(self):
        """Test with only one coin denomination"""
        assert coin_change([1], 5) == 5
        assert coin_change([3], 9) == 3

    def test_large_amount(self):
        """Test coins=[1,2,5], amount=100"""
        assert coin_change([1, 2, 5], 100) == 20

    def test_greedy_fails(self):
        """Test case where greedy approach fails"""
        # Greedy would pick 6, then need 1+1+1 = 4 coins
        # Optimal is 3+3+3 = 3 coins
        assert coin_change([1, 3, 4], 6) == 2  # 3+3

    def test_multiple_solutions_same_count(self):
        """Test that returns correct minimum"""
        # 7 = 2+2+3 or 4+3, both 3 coins but 4+3 is 2 coins
        assert coin_change([2, 3, 4], 7) == 2

    def test_large_coins(self):
        """Test with large coin denominations"""
        assert coin_change([186, 419, 83, 408], 6249) == 20

    def test_empty_impossible(self):
        """Test impossible with various coins"""
        assert coin_change([2, 4, 6], 7) == -1
        assert coin_change([5, 10], 3) == -1`,
	hint1: `Define dp[i] as the minimum number of coins needed to make amount i. Initialize with a large value (amount + 1) to represent "impossible".`,
	hint2: `For each amount i, try using each coin c. If c <= i, then dp[i] = min(dp[i], dp[i-c] + 1). The +1 represents using one coin of denomination c.`,
	whyItMatters: `Coin Change is a classic "Unbounded Knapsack" problem that appears frequently in FAANG interviews. It teaches optimization DP patterns.

**Why This Matters:**

**1. Optimization DP Pattern**

Unlike counting problems, here we minimize:

\`\`\`python
# Counting ways (accumulate)
dp[i] += dp[i - coin]

# Minimum coins (optimize)
dp[i] = min(dp[i], dp[i - coin] + 1)

# Maximum value (optimize)
dp[i] = max(dp[i], dp[i - coin] + value)
\`\`\`

**2. Unbounded Knapsack**

Coins can be used unlimited times:

\`\`\`python
# Unbounded: same coin can be used multiple times
for i in range(1, amount + 1):
    for coin in coins:
        if coin <= i:
            dp[i] = min(dp[i], dp[i - coin] + 1)

# 0/1 Knapsack: each item used at most once
# (would iterate coins in outer loop, amounts in reverse)
\`\`\`

**3. Why Greedy Fails**

\`\`\`python
# Greedy: always pick largest coin first
coins = [1, 3, 4], amount = 6
# Greedy: 4 + 1 + 1 = 3 coins
# Optimal: 3 + 3 = 2 coins

# DP considers ALL possibilities
\`\`\`

**4. Related FAANG Problems**

\`\`\`python
# Coin Change II: Count number of combinations
def coin_change_2(coins, amount):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:  # coin first to avoid duplicates
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    return dp[amount]

# Perfect Squares: min squares summing to n
def perfect_squares(n):
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j*j] + 1)
            j += 1
    return dp[n]
\`\`\`

**5. Real-World Applications**

- Vending machine change algorithms
- Currency exchange optimization
- Resource allocation problems
- Network packet optimization`,
	order: 3,
	translations: {
		ru: {
			title: 'Размен монет',
			description: `Найдите минимальное количество монет для набора заданной суммы.

**Задача:**

Дан массив целых чисел \`coins\`, представляющий монеты разных номиналов, и целое число \`amount\` - сумма денег.

Верните наименьшее количество монет, необходимое для набора этой суммы. Если сумму невозможно набрать никакой комбинацией монет, верните -1.

Предполагается, что у вас бесконечное количество монет каждого номинала.

**Примеры:**

\`\`\`
Вход: coins = [1, 2, 5], amount = 11
Выход: 3
Объяснение: 11 = 5 + 5 + 1

Вход: coins = [2], amount = 3
Выход: -1
Объяснение: Невозможно набрать 3 монетами по 2

Вход: coins = [1], amount = 0
Выход: 0
\`\`\`

**Ключевая идея:**

Для каждой суммы \`i\` пробуем использовать каждую монету \`c\`:
- Если используем монету \`c\`, нужно 1 + dp[i - c] монет
- dp[i] = min(dp[i], 1 + dp[i - c]) для всех подходящих монет

**Ограничения:**
- 1 <= coins.length <= 12
- 0 <= amount <= 10^4

**Временная сложность:** O(amount × len(coins))
**Пространственная сложность:** O(amount)`,
			hint1: `Определите dp[i] как минимальное количество монет для суммы i. Инициализируйте большим значением (amount + 1) для обозначения "невозможно".`,
			hint2: `Для каждой суммы i пробуйте каждую монету c. Если c <= i, то dp[i] = min(dp[i], dp[i-c] + 1). +1 означает использование одной монеты номинала c.`,
			whyItMatters: `Coin Change - классическая задача "Неограниченного рюкзака", которая часто встречается на FAANG-интервью. Она учит паттернам оптимизационного DP.

**Почему это важно:**

**1. Паттерн оптимизационного DP**

В отличие от задач подсчёта, здесь мы минимизируем.

**2. Неограниченный рюкзак**

Монеты можно использовать неограниченное количество раз.

**3. Почему жадный подход не работает**

coins = [1, 3, 4], amount = 6
Жадный: 4 + 1 + 1 = 3 монеты
Оптимальный: 3 + 3 = 2 монеты

**4. Связанные задачи FAANG**

- Coin Change II: подсчёт количества комбинаций
- Perfect Squares: минимум квадратов

**5. Применения в реальном мире**

Алгоритмы сдачи в торговых автоматах, оптимизация обмена валют.`,
			solutionCode: `from typing import List


def coin_change(coins: List[int], amount: int) -> int:
    """
    Находит минимальное количество монет для набора суммы.

    Args:
        coins: Список номиналов монет
        amount: Целевая сумма

    Returns:
        Минимальное количество монет, или -1 если невозможно
    """
    # dp[i] = минимум монет для суммы i
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0  # Базовый случай: 0 монет для суммы 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] <= amount else -1`
		},
		uz: {
			title: 'Tangalar almashtirish',
			description: `Berilgan summani to'ldirish uchun kerakli minimal tangalar sonini toping.

**Masala:**

Sizga turli nominaldagi tangalarni ifodalovchi \`coins\` butun sonlar massivi va pul miqdorini ifodalovchi \`amount\` butun soni berilgan.

O'sha miqdorni to'ldirish uchun kerakli eng kam tangalar sonini qaytaring. Agar miqdorni tangalar kombinatsiyasi bilan to'ldirib bo'lmasa, -1 qaytaring.

Har bir tanga turidan cheksiz miqdorda bor deb hisoblanadi.

**Misollar:**

\`\`\`
Kirish: coins = [1, 2, 5], amount = 11
Chiqish: 3
Izoh: 11 = 5 + 5 + 1

Kirish: coins = [2], amount = 3
Chiqish: -1
Izoh: Faqat 2 so'mlik tangalar bilan 3 ni yig'ib bo'lmaydi

Kirish: coins = [1], amount = 0
Chiqish: 0
\`\`\`

**Asosiy tushuncha:**

Har bir \`i\` summasi uchun har bir \`c\` tangasini sinab ko'ramiz:
- Agar \`c\` tangasini ishlatsak, 1 + dp[i - c] ta tanga kerak
- dp[i] = min(dp[i], 1 + dp[i - c]) barcha yaroqli tangalar uchun

**Cheklovlar:**
- 1 <= coins.length <= 12
- 0 <= amount <= 10^4

**Vaqt murakkabligi:** O(amount × len(coins))
**Xotira murakkabligi:** O(amount)`,
			hint1: `dp[i] ni i summasi uchun minimal tangalar soni sifatida aniqlang. "Imkonsiz" ni bildirish uchun katta qiymat (amount + 1) bilan boshlang.`,
			hint2: `Har bir i summasi uchun har bir c tangasini sinab ko'ring. Agar c <= i bo'lsa, dp[i] = min(dp[i], dp[i-c] + 1). +1 c nominalli bitta tangani ishlatishni bildiradi.`,
			whyItMatters: `Coin Change - FAANG intervyularida tez-tez uchraydigan klassik "Cheklanmagan ryukzak" masalasi. U optimallashtirish DP patternlarini o'rgatadi.

**Bu nima uchun muhim:**

**1. Optimallashtirish DP patterni**

Hisoblash masalalaridan farqli, bu yerda minimallashtiramiz.

**2. Cheklanmagan ryukzak**

Tangalarni cheksiz marta ishlatish mumkin.

**3. Nima uchun ochko'z yondashuv ishlamaydi**

coins = [1, 3, 4], amount = 6
Ochko'z: 4 + 1 + 1 = 3 ta tanga
Optimal: 3 + 3 = 2 ta tanga

**4. Bog'liq FAANG masalalari**

- Coin Change II: kombinatsiyalar sonini hisoblash
- Perfect Squares: minimal kvadratlar

**5. Haqiqiy dunyo qo'llanishlari**

Savdo avtomatlarida qaytim algoritmlari, valyuta almashtirishni optimallashtirish.`,
			solutionCode: `from typing import List


def coin_change(coins: List[int], amount: int) -> int:
    """
    Summani to'ldirish uchun minimal tangalar sonini topadi.

    Args:
        coins: Tanga nominallari ro'yxati
        amount: Maqsad summasi

    Returns:
        Minimal tangalar soni, yoki imkonsiz bo'lsa -1
    """
    # dp[i] = i summasi uchun minimal tangalar
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0  # Asosiy holat: 0 summasi uchun 0 ta tanga

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] <= amount else -1`
		}
	}
};

export default task;
