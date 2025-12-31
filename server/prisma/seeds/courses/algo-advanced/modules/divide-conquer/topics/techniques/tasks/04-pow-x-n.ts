import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'divide-conquer-pow-x-n',
	title: 'Pow(x, n)',
	difficulty: 'medium',
	tags: ['python', 'divide-conquer', 'math', 'recursion', 'binary-exponentiation'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement pow(x, n) which calculates x raised to the power n.

**Problem:**

Implement \`pow(x, n)\`, which calculates \`x\` raised to the power \`n\` (i.e., \`x^n\`).

Use the **divide and conquer** (binary exponentiation) approach for O(log n) time complexity.

**Examples:**

\`\`\`
Input: x = 2.00000, n = 10
Output: 1024.00000

Input: x = 2.10000, n = 3
Output: 9.26100

Input: x = 2.00000, n = -2
Output: 0.25000

Explanation: 2^(-2) = 1/(2^2) = 1/4 = 0.25
\`\`\`

**Key Insight:**

\`\`\`
x^n = (x^(n/2))^2           if n is even
x^n = x * (x^((n-1)/2))^2   if n is odd

Example: 2^10 = (2^5)^2 = (2 * (2^2)^2)^2
\`\`\`

This reduces the problem size by half at each step, giving O(log n) time.

**Constraints:**
- -100.0 < x < 100.0
- -2^31 <= n <= 2^31 - 1
- n is an integer
- Either x != 0 or n > 0
- -10^4 <= x^n <= 10^4

**Time Complexity:** O(log n)
**Space Complexity:** O(log n) recursive, O(1) iterative`,
	initialCode: `def my_pow(x: float, n: int) -> float:
    # TODO: Calculate x raised to power n using binary exponentiation

    return 0.0`,
	solutionCode: `def my_pow(x: float, n: int) -> float:
    """
    Calculate x raised to power n using binary exponentiation.
    """
    def helper(x: float, n: int) -> float:
        if n == 0:
            return 1

        half = helper(x, n // 2)

        if n % 2 == 0:
            return half * half
        else:
            return half * half * x

    if n < 0:
        x = 1 / x
        n = -n

    return helper(x, n)


# Iterative version (O(1) space)
def my_pow_iterative(x: float, n: int) -> float:
    """Iterative binary exponentiation."""
    if n < 0:
        x = 1 / x
        n = -n

    result = 1

    while n > 0:
        if n % 2 == 1:
            result *= x
        x *= x
        n //= 2

    return result


# With modular arithmetic (for large numbers)
def mod_pow(x: int, n: int, mod: int) -> int:
    """Calculate (x^n) % mod efficiently."""
    result = 1
    x = x % mod

    while n > 0:
        if n % 2 == 1:
            result = (result * x) % mod
        x = (x * x) % mod
        n //= 2

    return result


# Matrix exponentiation (for Fibonacci)
def matrix_pow(matrix: list, n: int) -> list:
    """Raise 2x2 matrix to power n."""
    def multiply(A, B):
        return [
            [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
        ]

    result = [[1, 0], [0, 1]]  # Identity matrix

    while n > 0:
        if n % 2 == 1:
            result = multiply(result, matrix)
        matrix = multiply(matrix, matrix)
        n //= 2

    return result


def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number using matrix exponentiation."""
    if n <= 1:
        return n

    matrix = [[1, 1], [1, 0]]
    result = matrix_pow(matrix, n - 1)
    return result[0][0]`,
	testCode: `import pytest
from solution import my_pow


class TestPow:
    def test_positive_exponent(self):
        """Test positive exponent"""
        assert abs(my_pow(2.0, 10) - 1024.0) < 1e-5

    def test_small_positive(self):
        """Test small positive exponent"""
        assert abs(my_pow(2.1, 3) - 9.261) < 1e-3

    def test_negative_exponent(self):
        """Test negative exponent"""
        assert abs(my_pow(2.0, -2) - 0.25) < 1e-5

    def test_zero_exponent(self):
        """Test zero exponent"""
        assert my_pow(5.0, 0) == 1.0

    def test_one_exponent(self):
        """Test exponent of 1"""
        assert my_pow(5.0, 1) == 5.0

    def test_base_one(self):
        """Test base of 1"""
        assert my_pow(1.0, 100) == 1.0

    def test_base_zero(self):
        """Test base of 0"""
        assert my_pow(0.0, 5) == 0.0

    def test_negative_base_even(self):
        """Test negative base with even exponent"""
        assert my_pow(-2.0, 2) == 4.0

    def test_negative_base_odd(self):
        """Test negative base with odd exponent"""
        assert my_pow(-2.0, 3) == -8.0

    def test_fractional_base(self):
        """Test fractional base"""
        assert abs(my_pow(0.5, 2) - 0.25) < 1e-5

    def test_large_exponent(self):
        """Test larger exponent"""
        result = my_pow(2.0, 20)
        assert abs(result - 1048576.0) < 1e-5`,
	hint1: `Use the property: x^n = (x^(n/2))^2 for even n, and x^n = x * (x^((n-1)/2))^2 for odd n. This halves the problem at each step.`,
	hint2: `Handle negative exponents by converting: x^(-n) = 1/x^n. Be careful with edge cases like n = 0 and n = -2^31.`,
	whyItMatters: `Binary exponentiation is a fundamental technique that reduces O(n) operations to O(log n). It's used extensively in cryptography, competitive programming, and matrix operations.

**Why This Matters:**

**1. Binary Exponentiation Concept**

\`\`\`python
# Instead of n multiplications:
# x * x * x * ... (n times) = O(n)

# We use:
# x^10 = (x^5)^2
# x^5 = x * (x^2)^2
# x^2 = x * x

# Only log(n) multiplications = O(log n)
\`\`\`

**2. Iterative vs Recursive**

\`\`\`python
# Recursive (natural but uses O(log n) stack space):
def pow_recursive(x, n):
    if n == 0: return 1
    half = pow_recursive(x, n // 2)
    return half * half * (x if n % 2 else 1)

# Iterative (O(1) space):
def pow_iterative(x, n):
    result = 1
    while n > 0:
        if n % 2 == 1:
            result *= x
        x *= x
        n //= 2
    return result
\`\`\`

**3. Modular Exponentiation**

\`\`\`python
# (x^n) mod m - crucial for cryptography
def mod_pow(x, n, mod):
    result = 1
    x %= mod
    while n > 0:
        if n % 2 == 1:
            result = (result * x) % mod
        x = (x * x) % mod
        n //= 2
    return result

# Used in: RSA, Diffie-Hellman, primality testing
\`\`\`

**4. Matrix Exponentiation**

\`\`\`python
# For linear recurrences like Fibonacci:
# F(n) = [[1,1],[1,0]]^n * [[1],[0]]

# O(log n) Fibonacci!
def fib(n):
    matrix = [[1, 1], [1, 0]]
    result = matrix_pow(matrix, n - 1)
    return result[0][0]
\`\`\`

**5. Applications**

\`\`\`python
# Cryptography (RSA, key exchange)
# Competitive programming
# Fibonacci in O(log n)
# Counting paths in graph
# Linear recurrence solutions
\`\`\``,
	order: 4,
	translations: {
		ru: {
			title: 'Pow(x, n)',
			description: `Реализуйте pow(x, n) - возведение x в степень n.

**Задача:**

Реализуйте \`pow(x, n)\` - вычисление \`x^n\`.

Используйте метод **"разделяй и властвуй"** (бинарное возведение в степень) для O(log n) сложности.

**Примеры:**

\`\`\`
Вход: x = 2.00000, n = 10
Выход: 1024.00000

Вход: x = 2.00000, n = -2
Выход: 0.25000

Объяснение: 2^(-2) = 1/(2^2) = 1/4 = 0.25
\`\`\`

**Ключевая идея:**

\`\`\`
x^n = (x^(n/2))^2           если n чётное
x^n = x * (x^((n-1)/2))^2   если n нечётное
\`\`\`

Это уменьшает размер задачи вдвое на каждом шаге.

**Ограничения:**
- -100.0 < x < 100.0
- -2^31 <= n <= 2^31 - 1

**Временная сложность:** O(log n)
**Пространственная сложность:** O(log n) рекурсивно, O(1) итеративно`,
			hint1: `Используйте свойство: x^n = (x^(n/2))^2 для чётного n, и x^n = x * (x^((n-1)/2))^2 для нечётного.`,
			hint2: `Отрицательные степени: x^(-n) = 1/x^n. Осторожно с краевыми случаями n = 0 и n = -2^31.`,
			whyItMatters: `Бинарное возведение в степень - фундаментальная техника, сокращающая O(n) операций до O(log n). Используется в криптографии.

**Почему это важно:**

**1. Концепция бинарного возведения**

Вместо n умножений - только log(n).

**2. Модульная арифметика**

(x^n) mod m - критично для криптографии.

**3. Матричное возведение**

Для линейных рекуррентностей как Фибоначчи за O(log n).`,
			solutionCode: `def my_pow(x: float, n: int) -> float:
    """Вычисляет x в степени n методом бинарного возведения."""
    def helper(x: float, n: int) -> float:
        if n == 0:
            return 1

        half = helper(x, n // 2)

        if n % 2 == 0:
            return half * half
        else:
            return half * half * x

    if n < 0:
        x = 1 / x
        n = -n

    return helper(x, n)`
		},
		uz: {
			title: 'Pow(x, n)',
			description: `pow(x, n) ni amalga oshiring - x ni n darajaga ko'tarish.

**Masala:**

\`pow(x, n)\` ni amalga oshiring - \`x^n\` ni hisoblash.

O(log n) murakkablik uchun **"bo'l va hukmronlik qil"** (binar darajaga ko'tarish) usulini ishlating.

**Misollar:**

\`\`\`
Kirish: x = 2.00000, n = 10
Chiqish: 1024.00000

Kirish: x = 2.00000, n = -2
Chiqish: 0.25000

Izoh: 2^(-2) = 1/(2^2) = 1/4 = 0.25
\`\`\`

**Asosiy tushuncha:**

\`\`\`
x^n = (x^(n/2))^2           agar n juft
x^n = x * (x^((n-1)/2))^2   agar n toq
\`\`\`

Bu har bir qadamda masala hajmini ikki baravar kamaytiradi.

**Cheklovlar:**
- -100.0 < x < 100.0
- -2^31 <= n <= 2^31 - 1

**Vaqt murakkabligi:** O(log n)
**Xotira murakkabligi:** O(log n) rekursiv, O(1) iterativ`,
			hint1: `x^n = (x^(n/2))^2 juft n uchun, va x^n = x * (x^((n-1)/2))^2 toq n uchun xususiyatdan foydalaning.`,
			hint2: `Manfiy darajalar: x^(-n) = 1/x^n. n = 0 va n = -2^31 chegaraviy holatlar bilan ehtiyot bo'ling.`,
			whyItMatters: `Binar darajaga ko'tarish - O(n) amallarni O(log n) ga kamaytiradigan asosiy texnika. Kriptografiyada ishlatiladi.

**Bu nima uchun muhim:**

**1. Binar darajaga ko'tarish konsepti**

n ta ko'paytirish o'rniga - faqat log(n).

**2. Modulli arifmetika**

(x^n) mod m - kriptografiya uchun muhim.

**3. Matritsali darajaga ko'tarish**

Fibonachchi kabi chiziqli takrorlanishlar uchun O(log n) da.`,
			solutionCode: `def my_pow(x: float, n: int) -> float:
    """x ni n darajaga binar ko'tarish usuli bilan hisoblaydi."""
    def helper(x: float, n: int) -> float:
        if n == 0:
            return 1

        half = helper(x, n // 2)

        if n % 2 == 0:
            return half * half
        else:
            return half * half * x

    if n < 0:
        x = 1 / x
        n = -n

    return helper(x, n)`
		}
	}
};

export default task;
