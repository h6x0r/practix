import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'bit-manipulation-sum-two-integers',
	title: 'Sum of Two Integers',
	difficulty: 'medium',
	tags: ['python', 'bit-manipulation', 'math'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Calculate the sum of two integers without using + or - operators.

**Problem:**

Given two integers \`a\` and \`b\`, return the sum of the two integers without using the \`+\` and \`-\` operators.

**Examples:**

\`\`\`
Input: a = 1, b = 2
Output: 3

Input: a = 2, b = 3
Output: 5

Input: a = -1, b = 1
Output: 0
\`\`\`

**Key Insight:**

Binary addition can be broken into two parts:
1. **XOR** gives the sum without carry: \`a ^ b\`
2. **AND** then left shift gives the carry: \`(a & b) << 1\`

Repeat until there's no carry!

**Visualization:**

\`\`\`
a = 5 (101), b = 3 (011)

Step 1:
  sum without carry = 5 ^ 3 = 110 (6)
  carry = (5 & 3) << 1 = (001) << 1 = 010 (2)

Step 2:
  sum without carry = 6 ^ 2 = 100 (4)
  carry = (6 & 2) << 1 = (010) << 1 = 100 (4)

Step 3:
  sum without carry = 4 ^ 4 = 000 (0)
  carry = (4 & 4) << 1 = 1000 (8)

Step 4:
  sum without carry = 0 ^ 8 = 1000 (8)
  carry = 0

Result: 8 ✓
\`\`\`

**Constraints:**
- -1000 <= a, b <= 1000

**Note:** In Python, integers have arbitrary precision, so we need to handle the 32-bit masking carefully for negative numbers.

**Time Complexity:** O(1) - maximum 32 iterations
**Space Complexity:** O(1)`,
	initialCode: `def get_sum(a: int, b: int) -> int:
    # TODO: Calculate sum of two integers without + or - operators

    return 0`,
	solutionCode: `def get_sum(a: int, b: int) -> int:
    """
    Calculate sum using bit manipulation with 32-bit handling.
    """
    # Mask to get 32-bit representation
    MASK = 0xFFFFFFFF
    MAX_INT = 0x7FFFFFFF  # 2^31 - 1

    while b != 0:
        # XOR gives sum without carry
        # AND << 1 gives carry
        carry = ((a & b) << 1) & MASK
        a = (a ^ b) & MASK
        b = carry

    # If a is negative in 32-bit representation, convert to Python negative
    if a > MAX_INT:
        return ~(a ^ MASK)
    return a


# Alternative: Using recursion
def get_sum_recursive(a: int, b: int) -> int:
    """Recursive implementation."""
    MASK = 0xFFFFFFFF
    MAX_INT = 0x7FFFFFFF

    # Mask inputs
    a &= MASK
    b &= MASK

    if b == 0:
        return a if a <= MAX_INT else ~(a ^ MASK)

    return get_sum_recursive(a ^ b, (a & b) << 1)


# Subtraction without - operator
def get_subtract(a: int, b: int) -> int:
    """
    Subtract: a - b = a + (-b)
    -b = ~b + 1 (two's complement)
    """
    MASK = 0xFFFFFFFF
    MAX_INT = 0x7FFFFFFF

    # Two's complement of b
    neg_b = (get_sum(~b, 1)) & MASK

    return get_sum(a, neg_b)


# Multiplication using bit manipulation
def get_multiply(a: int, b: int) -> int:
    """
    Multiply using shift and add.
    """
    negative = (a < 0) ^ (b < 0)
    a, b = abs(a), abs(b)

    result = 0
    while b > 0:
        if b & 1:
            result = get_sum(result, a)
        a <<= 1
        b >>= 1

    return -result if negative else result


# Division using bit manipulation
def get_divide(dividend: int, divisor: int) -> int:
    """
    Divide using binary search with subtraction.
    """
    if divisor == 0:
        raise ValueError("Division by zero")

    negative = (dividend < 0) ^ (divisor < 0)
    dividend, divisor = abs(dividend), abs(divisor)

    quotient = 0
    for i in range(31, -1, -1):
        if (dividend >> i) >= divisor:
            quotient |= (1 << i)
            dividend = get_subtract(dividend, divisor << i)

    return -quotient if negative else quotient


# Negate without -
def negate(n: int) -> int:
    """Negate using two's complement: -n = ~n + 1."""
    return get_sum(~n, 1)


# Absolute value without conditionals
def abs_bit(n: int) -> int:
    """Absolute value using bit manipulation."""
    # For 32-bit signed integers
    mask = n >> 31  # All 1s if negative, all 0s if positive
    return (n ^ mask) - mask`,
	testCode: `import pytest
from solution import get_sum


class TestSumTwoIntegers:
    def test_positive_numbers(self):
        """Test two positive numbers"""
        assert get_sum(1, 2) == 3
        assert get_sum(2, 3) == 5

    def test_negative_and_positive(self):
        """Test negative and positive"""
        assert get_sum(-1, 1) == 0
        assert get_sum(-2, 3) == 1

    def test_two_negatives(self):
        """Test two negative numbers"""
        assert get_sum(-1, -2) == -3
        assert get_sum(-5, -7) == -12

    def test_zero(self):
        """Test with zero"""
        assert get_sum(0, 0) == 0
        assert get_sum(0, 5) == 5
        assert get_sum(5, 0) == 5

    def test_same_number(self):
        """Test adding same number"""
        assert get_sum(5, 5) == 10
        assert get_sum(-5, -5) == -10

    def test_boundary_values(self):
        """Test boundary values"""
        assert get_sum(1000, 0) == 1000
        assert get_sum(-1000, 0) == -1000

    def test_carry_propagation(self):
        """Test cases with multiple carries"""
        assert get_sum(7, 1) == 8   # 111 + 001 = 1000
        assert get_sum(15, 1) == 16  # 1111 + 1 = 10000

    def test_large_sum(self):
        """Test larger sums"""
        assert get_sum(100, 200) == 300
        assert get_sum(500, 500) == 1000

    def test_negative_result(self):
        """Test cases with negative result"""
        assert get_sum(5, -10) == -5
        assert get_sum(-10, 3) == -7

    def test_opposite_numbers(self):
        """Test opposite numbers"""
        assert get_sum(100, -100) == 0
        assert get_sum(-50, 50) == 0`,
	hint1: `Binary addition: XOR gives sum without carry (like half-adder). AND gives positions where both bits are 1 (needs carry). Left shift the AND result to get the carry.`,
	hint2: `Keep adding (XORing) and carrying until there's no more carry. In Python, use masking (& 0xFFFFFFFF) for 32-bit arithmetic and handle negative number conversion.`,
	whyItMatters: `Sum of Two Integers reveals how hardware actually performs addition at the gate level. Understanding this connects high-level code to CPU architecture.

**Why This Matters:**

**1. How Computers Actually Add**

\`\`\`python
# At the hardware level, addition uses:
# - XOR gates for sum without carry
# - AND gates for carry detection
# - Shift for carry propagation

# This is how a "ripple-carry adder" works!
# More advanced: carry-lookahead adders
\`\`\`

**2. Half-Adder and Full-Adder**

\`\`\`python
# Half-adder (2 inputs):
#   sum = a ^ b
#   carry = a & b

# Full-adder (3 inputs including carry-in):
#   sum = a ^ b ^ c_in
#   carry = (a & b) | (c_in & (a ^ b))

# Chain 32 full-adders = 32-bit adder
\`\`\`

**3. Two's Complement for Negatives**

\`\`\`python
# -n = ~n + 1 (invert all bits, add 1)
# This makes subtraction = addition of negative

# Python quirk: integers have arbitrary precision
# Need explicit masking for 32-bit behavior
\`\`\`

**4. Building Arithmetic from Bits**

\`\`\`python
# Multiplication = shift and add
# Division = shift and subtract
# All arithmetic operations can be built from
# AND, OR, XOR, NOT, and shifts!

# This is how ALUs (Arithmetic Logic Units) work
\`\`\``,
	order: 8,
	translations: {
		ru: {
			title: 'Сумма двух целых чисел',
			description: `Вычислите сумму двух чисел без использования операторов + и -.

**Задача:**

Даны два целых числа \`a\` и \`b\`, верните их сумму без использования операторов \`+\` и \`-\`.

**Примеры:**

\`\`\`
Вход: a = 1, b = 2
Выход: 3

Вход: a = 2, b = 3
Выход: 5

Вход: a = -1, b = 1
Выход: 0
\`\`\`

**Ключевая идея:**

Двоичное сложение можно разбить на две части:
1. **XOR** даёт сумму без переноса: \`a ^ b\`
2. **AND** со сдвигом даёт перенос: \`(a & b) << 1\`

Повторяйте пока перенос не станет нулём!

**Визуализация:**

\`\`\`
a = 5 (101), b = 3 (011)

Шаг 1:
  сумма без переноса = 5 ^ 3 = 110 (6)
  перенос = (5 & 3) << 1 = (001) << 1 = 010 (2)

Шаг 2:
  сумма без переноса = 6 ^ 2 = 100 (4)
  перенос = (6 & 2) << 1 = 100 (4)

Шаг 3:
  сумма без переноса = 4 ^ 4 = 0
  перенос = 8

Шаг 4:
  сумма = 8, перенос = 0

Результат: 8 ✓
\`\`\`

**Ограничения:**
- -1000 <= a, b <= 1000

**Временная сложность:** O(1) - максимум 32 итерации
**Пространственная сложность:** O(1)`,
			hint1: `XOR даёт сумму без переноса. AND даёт позиции где оба бита = 1 (нужен перенос). Сдвиг AND влево даёт перенос.`,
			hint2: `Продолжайте XOR и перенос пока перенос не станет 0. В Python используйте маску (& 0xFFFFFFFF) для 32-битной арифметики.`,
			whyItMatters: `Сумма двух чисел показывает как аппаратно выполняется сложение на уровне логических вентилей. Это связывает код с архитектурой CPU.

**Почему это важно:**

**1. Как компьютеры складывают**

На аппаратном уровне: XOR для суммы, AND для переноса, сдвиг для распространения. Это "ripple-carry adder"!

**2. Полусумматор и полный сумматор**

Half-adder: sum = a ^ b, carry = a & b. Цепочка из 32 сумматоров = 32-битный сумматор.

**3. Дополнительный код для отрицательных**

-n = ~n + 1. Вычитание = сложение с отрицательным.

**4. Построение арифметики из битов**

Умножение = сдвиг и сложение. Деление = сдвиг и вычитание. Так работают ALU!`,
			solutionCode: `def get_sum(a: int, b: int) -> int:
    """Вычисляет сумму используя битовые операции с 32-битной обработкой."""
    MASK = 0xFFFFFFFF
    MAX_INT = 0x7FFFFFFF

    while b != 0:
        carry = ((a & b) << 1) & MASK
        a = (a ^ b) & MASK
        b = carry

    if a > MAX_INT:
        return ~(a ^ MASK)
    return a`
		},
		uz: {
			title: 'Ikki butun sonning yigindisi',
			description: `Ikki sonning yig'indisini + va - operatorlarisiz hisoblang.

**Masala:**

Ikki butun son \`a\` va \`b\` berilgan, ularning yig'indisini \`+\` va \`-\` operatorlarisiz qaytaring.

**Misollar:**

\`\`\`
Kirish: a = 1, b = 2
Chiqish: 3

Kirish: a = 2, b = 3
Chiqish: 5

Kirish: a = -1, b = 1
Chiqish: 0
\`\`\`

**Asosiy tushuncha:**

Ikkilik qo'shishni ikkiga bo'lish mumkin:
1. **XOR** ko'chirishsiz yig'indi beradi: \`a ^ b\`
2. **AND** siljitish bilan ko'chirish beradi: \`(a & b) << 1\`

Ko'chirish nolga aylanguncha takrorlang!

**Vizualizatsiya:**

\`\`\`
a = 5 (101), b = 3 (011)

1-qadam:
  ko'chirishsiz yig'indi = 5 ^ 3 = 110 (6)
  ko'chirish = (5 & 3) << 1 = 010 (2)

2-qadam:
  ko'chirishsiz yig'indi = 6 ^ 2 = 100 (4)
  ko'chirish = (6 & 2) << 1 = 100 (4)

3-qadam:
  ko'chirishsiz yig'indi = 4 ^ 4 = 0
  ko'chirish = 8

4-qadam:
  yig'indi = 8, ko'chirish = 0

Natija: 8 ✓
\`\`\`

**Cheklovlar:**
- -1000 <= a, b <= 1000

**Vaqt murakkabligi:** O(1) - maksimum 32 iteratsiya
**Xotira murakkabligi:** O(1)`,
			hint1: `XOR ko'chirishsiz yig'indi beradi. AND ikkala bit = 1 bo'lgan pozitsiyalarni beradi (ko'chirish kerak). AND ni chapga siljitish ko'chirishni beradi.`,
			hint2: `Ko'chirish 0 bo'lguncha XOR va ko'chirishni davom ettiring. Python'da 32-bitlik arifmetika uchun maska (& 0xFFFFFFFF) ishlating.`,
			whyItMatters: `Ikki sonning yig'indisi apparatda qo'shish mantiqiy eshiklar darajasida qanday amalga oshirilishini ko'rsatadi. Bu kodni CPU arxitekturasiga bog'laydi.

**Bu nima uchun muhim:**

**1. Kompyuterlar qanday qo'shadi**

Apparat darajasida: XOR yig'indi uchun, AND ko'chirish uchun, siljitish tarqatish uchun. Bu "ripple-carry adder"!

**2. Yarim va to'liq summator**

Half-adder: sum = a ^ b, carry = a & b. 32 ta summator zanjiri = 32-bitli summator.

**3. Manfiy sonlar uchun ikkilik to'ldiruvchi**

-n = ~n + 1. Ayirish = manfiy bilan qo'shish.

**4. Bitlardan arifmetika qurish**

Ko'paytirish = siljitish va qo'shish. Bo'lish = siljitish va ayirish. ALU shunday ishlaydi!`,
			solutionCode: `def get_sum(a: int, b: int) -> int:
    """32-bitlik ishlov berish bilan bit operatsiyalari yordamida yig'indini hisoblaydi."""
    MASK = 0xFFFFFFFF
    MAX_INT = 0x7FFFFFFF

    while b != 0:
        carry = ((a & b) << 1) & MASK
        a = (a ^ b) & MASK
        b = carry

    if a > MAX_INT:
        return ~(a ^ MASK)
    return a`
		}
	}
};

export default task;
