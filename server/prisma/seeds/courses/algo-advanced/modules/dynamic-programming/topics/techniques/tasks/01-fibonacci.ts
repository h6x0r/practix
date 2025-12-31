import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'dp-fibonacci',
	title: 'Fibonacci Number',
	difficulty: 'easy',
	tags: ['python', 'dynamic-programming', 'memoization', 'tabulation'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Calculate the n-th Fibonacci number using Dynamic Programming.

**Problem:**

The Fibonacci sequence is defined as:
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) for n > 1

Given an integer \`n\`, return the n-th Fibonacci number.

**Why Dynamic Programming?**

The naive recursive approach has exponential time complexity O(2^n) because it recalculates the same subproblems many times:

\`\`\`
                    fib(5)
                 /         \\
            fib(4)           fib(3)
           /     \\          /     \\
       fib(3)   fib(2)   fib(2)   fib(1)
       /   \\    /   \\    /   \\
   fib(2) fib(1) ...  ...  ...
\`\`\`

DP solves this by storing computed values (memoization) or building up from base cases (tabulation).

**Examples:**

\`\`\`
Input: n = 0
Output: 0

Input: n = 1
Output: 1

Input: n = 10
Output: 55

Sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, ...
\`\`\`

**Constraints:**
- 0 <= n <= 45

**Approaches:**
1. **Top-down (Memoization):** Recursive with cache
2. **Bottom-up (Tabulation):** Iterative with array
3. **Space-optimized:** Only keep last two values

**Time Complexity:** O(n)
**Space Complexity:** O(n) for memoization/tabulation, O(1) for space-optimized`,
	initialCode: `def fibonacci(n: int) -> int:
    # TODO: Calculate the n-th Fibonacci number using DP

    return 0`,
	solutionCode: `def fibonacci(n: int) -> int:
    """
    Calculate the n-th Fibonacci number.

    Args:
        n: Non-negative integer (0 <= n <= 45)

    Returns:
        The n-th Fibonacci number
    """
    # Space-optimized bottom-up approach
    if n == 0:
        return 0
    if n == 1:
        return 1

    prev2, prev1 = 0, 1

    for _ in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    return prev1


# Alternative: Memoization approach
def fibonacci_memo(n: int, memo: dict = None) -> int:
    """Top-down approach with memoization."""
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


# Alternative: Tabulation approach
def fibonacci_tab(n: int) -> int:
    """Bottom-up approach with tabulation."""
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]`,
	testCode: `import pytest
from solution import fibonacci


class TestFibonacci:
    def test_base_case_zero(self):
        """Test F(0) = 0"""
        assert fibonacci(0) == 0

    def test_base_case_one(self):
        """Test F(1) = 1"""
        assert fibonacci(1) == 1

    def test_small_number(self):
        """Test F(5) = 5"""
        assert fibonacci(5) == 5

    def test_fibonacci_10(self):
        """Test F(10) = 55"""
        assert fibonacci(10) == 55

    def test_fibonacci_20(self):
        """Test F(20) = 6765"""
        assert fibonacci(20) == 6765

    def test_fibonacci_sequence(self):
        """Test first 10 Fibonacci numbers"""
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        for i, exp in enumerate(expected):
            assert fibonacci(i) == exp

    def test_large_number(self):
        """Test F(30) = 832040"""
        assert fibonacci(30) == 832040

    def test_fibonacci_45(self):
        """Test maximum constraint F(45)"""
        assert fibonacci(45) == 1134903170

    def test_consecutive_property(self):
        """Test F(n) = F(n-1) + F(n-2) for n=15"""
        n = 15
        assert fibonacci(n) == fibonacci(n - 1) + fibonacci(n - 2)

    def test_multiple_calls_same_result(self):
        """Test consistency - multiple calls return same result"""
        result1 = fibonacci(25)
        result2 = fibonacci(25)
        assert result1 == result2 == 75025`,
	hint1: `Start with the base cases: F(0) = 0 and F(1) = 1. For n > 1, you need F(n-1) and F(n-2).`,
	hint2: `For space optimization, you only need to keep track of the last two Fibonacci numbers. Use two variables and update them in each iteration.`,
	whyItMatters: `Fibonacci is the "Hello World" of Dynamic Programming - it teaches the core concepts that apply to hundreds of DP problems.

**Why This Matters:**

**1. Understanding Overlapping Subproblems**

The key insight of DP: we solve the same subproblems repeatedly.

\`\`\`python
# Naive recursive - O(2^n) time
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)  # Exponential!

# With memoization - O(n) time
def fib_memo(n, cache={}):
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = fib_memo(n-1) + fib_memo(n-2)
    return cache[n]
\`\`\`

**2. Top-Down vs Bottom-Up**

\`\`\`python
# Top-down: Start from n, work down to base cases
# (Recursive with memoization)

# Bottom-up: Start from base cases, build up to n
# (Iterative with tabulation)
dp = [0, 1]
for i in range(2, n+1):
    dp.append(dp[i-1] + dp[i-2])
\`\`\`

**3. Space Optimization Pattern**

\`\`\`python
# O(n) space - store all values
dp = [0] * (n + 1)

# O(1) space - only keep what you need
prev2, prev1 = 0, 1
for _ in range(2, n+1):
    prev2, prev1 = prev1, prev1 + prev2
\`\`\`

**4. FAANG Interview Pattern**

This pattern appears in:
- Climbing Stairs (LeetCode 70)
- House Robber (LeetCode 198)
- Decode Ways (LeetCode 91)
- Minimum Cost Climbing Stairs (LeetCode 746)

**5. Real-World Applications**

- Financial modeling (compound interest)
- Algorithm analysis (divide-and-conquer recurrences)
- Nature patterns (golden ratio)
- Computer graphics (spiral generation)`,
	order: 1,
	translations: {
		ru: {
			title: 'Числа Фибоначчи',
			description: `Вычислите n-е число Фибоначчи с помощью динамического программирования.

**Задача:**

Последовательность Фибоначчи определяется как:
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) для n > 1

Дано целое число \`n\`, верните n-е число Фибоначчи.

**Почему динамическое программирование?**

Наивный рекурсивный подход имеет экспоненциальную сложность O(2^n), потому что многократно пересчитывает одни и те же подзадачи.

DP решает это сохранением вычисленных значений (мемоизация) или построением от базовых случаев (табуляция).

**Примеры:**

\`\`\`
Вход: n = 0
Выход: 0

Вход: n = 10
Выход: 55

Последовательность: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, ...
\`\`\`

**Ограничения:**
- 0 <= n <= 45

**Временная сложность:** O(n)
**Пространственная сложность:** O(1) при оптимизации`,
			hint1: `Начните с базовых случаев: F(0) = 0 и F(1) = 1. Для n > 1 вам нужны F(n-1) и F(n-2).`,
			hint2: `Для оптимизации памяти достаточно хранить только два последних числа Фибоначчи. Используйте две переменные и обновляйте их на каждой итерации.`,
			whyItMatters: `Фибоначчи - это "Hello World" динамического программирования. Он учит основным концепциям, которые применяются к сотням задач DP.

**Почему это важно:**

**1. Понимание перекрывающихся подзадач**

Ключевая идея DP: мы решаем одни и те же подзадачи многократно.

**2. Сверху вниз vs Снизу вверх**

- Сверху вниз: начинаем с n, спускаемся к базовым случаям (рекурсия с мемоизацией)
- Снизу вверх: начинаем с базовых случаев, строим до n (итерация с табуляцией)

**3. Паттерн оптимизации памяти**

Вместо хранения всех значений - храним только необходимые.

**4. Паттерн FAANG-интервью**

Этот паттерн встречается в: Climbing Stairs, House Robber, Decode Ways.`,
			solutionCode: `def fibonacci(n: int) -> int:
    """
    Вычисляет n-е число Фибоначчи.

    Args:
        n: Неотрицательное целое число (0 <= n <= 45)

    Returns:
        n-е число Фибоначчи
    """
    # Оптимизированный по памяти подход снизу вверх
    if n == 0:
        return 0
    if n == 1:
        return 1

    prev2, prev1 = 0, 1

    for _ in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    return prev1`
		},
		uz: {
			title: 'Fibonachchi soni',
			description: `Dinamik dasturlash yordamida n-chi Fibonachchi sonini hisoblang.

**Masala:**

Fibonachchi ketma-ketligi quyidagicha aniqlanadi:
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) n > 1 uchun

Butun son \`n\` berilgan, n-chi Fibonachchi sonini qaytaring.

**Nima uchun dinamik dasturlash?**

Oddiy rekursiv yondashuv O(2^n) eksponensial murakkablikka ega, chunki bir xil kichik masalalarni ko'p marta qayta hisoblaydi.

DP buni hisoblangan qiymatlarni saqlash (memoizatsiya) yoki asosiy holatlardan qurish (tabulyatsiya) orqali hal qiladi.

**Misollar:**

\`\`\`
Kirish: n = 0
Chiqish: 0

Kirish: n = 10
Chiqish: 55

Ketma-ketlik: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, ...
\`\`\`

**Cheklovlar:**
- 0 <= n <= 45

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1) optimallashtirish bilan`,
			hint1: `Asosiy holatlardan boshlang: F(0) = 0 va F(1) = 1. n > 1 uchun F(n-1) va F(n-2) kerak.`,
			hint2: `Xotirani optimallashtirish uchun faqat oxirgi ikkita Fibonachchi sonini saqlash kifoya. Ikkita o'zgaruvchi ishlating va har bir iteratsiyada ularni yangilang.`,
			whyItMatters: `Fibonachchi - dinamik dasturlashning "Hello World" i. U yuzlab DP masalalariga qo'llaniladigan asosiy tushunchalarni o'rgatadi.

**Bu nima uchun muhim:**

**1. Ustma-ust tushadigan kichik masalalarni tushunish**

DP ning asosiy g'oyasi: biz bir xil kichik masalalarni takroran hal qilamiz.

**2. Yuqoridan pastga vs Pastdan yuqoriga**

- Yuqoridan pastga: n dan boshlab asosiy holatlarga tushamiz (memoizatsiya bilan rekursiya)
- Pastdan yuqoriga: asosiy holatlardan boshlab n gacha quramiz (tabulyatsiya bilan iteratsiya)

**3. Xotira optimallashtirish patterni**

Barcha qiymatlarni saqlash o'rniga faqat keraklilarini saqlaymiz.

**4. FAANG intervyu patterni**

Bu pattern quyidagilarda uchraydi: Climbing Stairs, House Robber, Decode Ways.`,
			solutionCode: `def fibonacci(n: int) -> int:
    """
    n-chi Fibonachchi sonini hisoblaydi.

    Args:
        n: Manfiy bo'lmagan butun son (0 <= n <= 45)

    Returns:
        n-chi Fibonachchi soni
    """
    # Xotira bo'yicha optimallashtirilgan pastdan yuqoriga yondashuv
    if n == 0:
        return 0
    if n == 1:
        return 1

    prev2, prev1 = 0, 1

    for _ in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    return prev1`
		}
	}
};

export default task;
