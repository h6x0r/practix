import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'bit-manipulation-hamming-weight',
	title: 'Number of 1 Bits (Hamming Weight)',
	difficulty: 'easy',
	tags: ['python', 'bit-manipulation', 'counting'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Count the number of set bits (1s) in a number's binary representation.

**Problem:**

Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the **Hamming weight**).

**Examples:**

\`\`\`
Input: n = 11 (binary: 00000000000000000000000000001011)
Output: 3

Explanation: The input has three '1' bits.

Input: n = 128 (binary: 00000000000000000000000010000000)
Output: 1

Input: n = 2147483645 (binary: 01111111111111111111111111111101)
Output: 30
\`\`\`

**Key Insight:**

Use \`n & (n - 1)\` to clear the lowest set bit. Count how many times you can do this until n becomes 0.

**Visualization:**

\`\`\`
n = 11 (1011)

Step 1: n & (n-1) = 1011 & 1010 = 1010, count = 1
Step 2: n & (n-1) = 1010 & 1001 = 1000, count = 2
Step 3: n & (n-1) = 1000 & 0111 = 0000, count = 3

Answer: 3
\`\`\`

**Constraints:**
- The input is an unsigned 32-bit integer

**Time Complexity:** O(k) where k = number of 1 bits
**Space Complexity:** O(1)`,
	initialCode: `def hamming_weight(n: int) -> int:
    # TODO: Count number of 1 bits in the binary representation

    return 0`,
	solutionCode: `def hamming_weight(n: int) -> int:
    """
    Count number of 1 bits using Brian Kernighan's algorithm.
    """
    count = 0
    while n:
        n = n & (n - 1)  # Clear lowest set bit
        count += 1
    return count


# Approach 2: Check each bit
def hamming_weight_shift(n: int) -> int:
    """Count by checking each bit."""
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


# Approach 3: Built-in
def hamming_weight_builtin(n: int) -> int:
    """Using Python's bin() and count()."""
    return bin(n).count('1')


# Approach 4: Lookup table
def hamming_weight_lookup(n: int) -> int:
    """Using precomputed lookup table for bytes."""
    lookup = [bin(i).count('1') for i in range(256)]

    count = 0
    while n:
        count += lookup[n & 0xFF]
        n >>= 8

    return count


# Hamming distance between two numbers
def hamming_distance(x: int, y: int) -> int:
    """Count differing bits between x and y."""
    return hamming_weight(x ^ y)


# Count total set bits from 0 to n
def count_total_bits(n: int) -> int:
    """Count total 1-bits in all numbers from 0 to n."""
    if n < 0:
        return 0

    # Using pattern: for numbers 0 to 2^k - 1,
    # total 1s at each position = 2^(k-1)
    # Total = k * 2^(k-1)

    total = 0
    power = 1
    while power <= n:
        # Count complete groups
        total_pairs = (n + 1) // (power * 2)
        total += total_pairs * power

        # Count remainder
        remainder = (n + 1) % (power * 2)
        total += max(0, remainder - power)

        power *= 2

    return total`,
	testCode: `import pytest
from solution import hamming_weight


class TestHammingWeight:
    def test_basic_case(self):
        """Test basic case"""
        assert hamming_weight(11) == 3  # 1011

    def test_power_of_two(self):
        """Test power of 2"""
        assert hamming_weight(128) == 1  # 10000000

    def test_large_number(self):
        """Test large number"""
        assert hamming_weight(2147483645) == 30

    def test_zero(self):
        """Test zero"""
        assert hamming_weight(0) == 0

    def test_one(self):
        """Test one"""
        assert hamming_weight(1) == 1

    def test_all_ones_byte(self):
        """Test 255 (all 1s in byte)"""
        assert hamming_weight(255) == 8

    def test_alternating(self):
        """Test alternating bits"""
        assert hamming_weight(0b10101010) == 4
        assert hamming_weight(0b01010101) == 4

    def test_max_int(self):
        """Test maximum 32-bit value"""
        assert hamming_weight(0xFFFFFFFF) == 32

    def test_single_high_bit(self):
        """Test single high bit"""
        assert hamming_weight(0x80000000) == 1

    def test_consecutive_ones(self):
        """Test consecutive 1s"""
        assert hamming_weight(0b111) == 3
        assert hamming_weight(0b11111) == 5`,
	hint1: `The trick n & (n-1) clears the lowest set bit. For example: 1100 & 1011 = 1000. Count how many times you can do this.`,
	hint2: `Alternative: check each bit with n & 1 and right shift. This always takes 32 iterations, while the n & (n-1) approach only iterates as many times as there are 1 bits.`,
	whyItMatters: `Hamming weight is a fundamental bit operation used in error detection/correction, cryptography, and numerous algorithms. Brian Kernighan's trick is an elegant optimization.

**Why This Matters:**

**1. Brian Kernighan's Algorithm**

\`\`\`python
# n & (n-1) clears the lowest set bit
# Example: n = 1100
# n - 1 = 1011 (flip bits up to and including lowest 1)
# n & (n-1) = 1000 (lowest 1 is cleared)

# This is optimal: O(k) where k = number of 1 bits
# vs O(32) for checking each bit
\`\`\`

**2. Common Bit Tricks**

\`\`\`python
# Isolate lowest set bit:
lowest_bit = n & (-n)

# Clear lowest set bit:
n & (n - 1)

# Check if power of 2:
is_power_of_2 = n > 0 and (n & (n - 1)) == 0

# Get lowest k bits:
lowest_k = n & ((1 << k) - 1)
\`\`\`

**3. Applications**

\`\`\`python
# Hamming distance: hamming_weight(x ^ y)
# Error detection in transmission
# Population count in bitmap indexes
# Sparse bit vector operations
\`\`\`

**4. Hardware Support**

\`\`\`python
# Most CPUs have POPCNT instruction
# Python's bin(n).count('1') may use this

# For performance-critical code:
# Use built-in when available
# Use lookup table for byte-by-byte counting
\`\`\``,
	order: 2,
	translations: {
		ru: {
			title: 'Количество единичных битов',
			description: `Подсчитайте количество установленных битов (1) в двоичном представлении числа.

**Задача:**

Напишите функцию, которая принимает беззнаковое целое число и возвращает количество '1' битов (также известно как **вес Хэмминга**).

**Примеры:**

\`\`\`
Вход: n = 11 (бинарно: 1011)
Выход: 3

Вход: n = 128 (бинарно: 10000000)
Выход: 1
\`\`\`

**Ключевая идея:**

Используйте \`n & (n - 1)\` для очистки младшего установленного бита. Считайте сколько раз можно это сделать до n = 0.

**Ограничения:**
- Вход - беззнаковое 32-битное целое

**Временная сложность:** O(k), где k = количество единичных битов
**Пространственная сложность:** O(1)`,
			hint1: `Трюк n & (n-1) очищает младший установленный бит. Например: 1100 & 1011 = 1000. Считайте сколько раз это можно сделать.`,
			hint2: `Альтернатива: проверять каждый бит через n & 1 и сдвиг вправо. Это всегда 32 итерации, а n & (n-1) итерирует только по количеству единиц.`,
			whyItMatters: `Вес Хэмминга - фундаментальная битовая операция для обнаружения ошибок и криптографии. Алгоритм Кернигана - элегантная оптимизация.

**Почему это важно:**

**1. Алгоритм Кернигана**

n & (n-1) очищает младший установленный бит. Оптимален: O(k).

**2. Распространённые битовые трюки**

Выделение младшего бита, проверка степени двойки.

**3. Применения**

Расстояние Хэмминга, обнаружение ошибок, bitmap индексы.`,
			solutionCode: `def hamming_weight(n: int) -> int:
    """Подсчитывает количество единичных битов алгоритмом Кернигана."""
    count = 0
    while n:
        n = n & (n - 1)
        count += 1
    return count`
		},
		uz: {
			title: '1-bitlar soni',
			description: `Sonning ikkilik ko'rinishidagi o'rnatilgan bitlar (1) sonini hisoblang.

**Masala:**

Ishorasiz butun sonni qabul qilib, '1' bitlar sonini qaytaradigan funksiya yozing (**Hemming og'irligi**).

**Misollar:**

\`\`\`
Kirish: n = 11 (ikkilik: 1011)
Chiqish: 3

Kirish: n = 128 (ikkilik: 10000000)
Chiqish: 1
\`\`\`

**Asosiy tushuncha:**

Eng kichik o'rnatilgan bitni tozalash uchun \`n & (n - 1)\` ishlating. n = 0 bo'lguncha necha marta qilish mumkinligini hisoblang.

**Cheklovlar:**
- Kirish ishorasiz 32-bitli butun son

**Vaqt murakkabligi:** O(k), bu yerda k = 1-bitlar soni
**Xotira murakkabligi:** O(1)`,
			hint1: `n & (n-1) hiylasi eng kichik o'rnatilgan bitni tozalaydi. Masalan: 1100 & 1011 = 1000. Buni necha marta qilish mumkinligini hisoblang.`,
			hint2: `Alternativ: har bir bitni n & 1 va o'ngga siljitish bilan tekshiring. Bu har doim 32 iteratsiya, n & (n-1) esa faqat 1-lar soniga qarab iteratsiya qiladi.`,
			whyItMatters: `Hemming og'irligi - xatolarni aniqlash va kriptografiya uchun asosiy bit operatsiyasi. Kernigan algoritmi - nafis optimallashtirish.

**Bu nima uchun muhim:**

**1. Kernigan algoritmi**

n & (n-1) eng kichik o'rnatilgan bitni tozalaydi. Optimal: O(k).

**2. Keng tarqalgan bit hiylalari**

Eng kichik bitni ajratish, 2 darajasi tekshirish.

**3. Qo'llanishlar**

Hemming masofasi, xatolarni aniqlash, bitmap indekslar.`,
			solutionCode: `def hamming_weight(n: int) -> int:
    """Kernigan algoritmi bilan 1-bitlar sonini hisoblaydi."""
    count = 0
    while n:
        n = n & (n - 1)
        count += 1
    return count`
		}
	}
};

export default task;
