import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'bit-manipulation-power-of-two',
	title: 'Power of Two',
	difficulty: 'easy',
	tags: ['python', 'bit-manipulation', 'math'],
	estimatedTime: '10m',
	isPremium: false,
	youtubeUrl: '',
	description: `Determine if a given integer is a power of two.

**Problem:**

Given an integer \`n\`, return \`true\` if it is a power of two. Otherwise, return \`false\`.

An integer \`n\` is a power of two if there exists an integer \`x\` such that \`n == 2^x\`.

**Examples:**

\`\`\`
Input: n = 1
Output: true
Explanation: 2^0 = 1

Input: n = 16
Output: true
Explanation: 2^4 = 16

Input: n = 3
Output: false

Input: n = 0
Output: false

Input: n = -1
Output: false
\`\`\`

**Key Insight:**

A power of two has exactly one bit set in its binary representation:
- 1 = 0001
- 2 = 0010
- 4 = 0100
- 8 = 1000

The trick: \`n & (n - 1)\` clears the lowest set bit. For powers of two, this results in 0.

**Formula:** \`n > 0 and (n & (n - 1)) == 0\`

**Constraints:**
- -2^31 <= n <= 2^31 - 1

**Time Complexity:** O(1)
**Space Complexity:** O(1)`,
	initialCode: `def is_power_of_two(n: int) -> bool:
    # TODO: Check if n is a power of two

    return False`,
	solutionCode: `def is_power_of_two(n: int) -> bool:
    """
    Check if n is a power of two using bit manipulation.
    """
    return n > 0 and (n & (n - 1)) == 0


# Approach 2: Using lowest bit isolation
def is_power_of_two_lowest_bit(n: int) -> bool:
    """
    n & -n isolates the lowest set bit.
    For powers of two, this equals n itself.
    """
    return n > 0 and (n & -n) == n


# Approach 3: Count bits
def is_power_of_two_count(n: int) -> bool:
    """Check if exactly one bit is set."""
    if n <= 0:
        return False
    return bin(n).count('1') == 1


# Approach 4: Using log (math approach)
import math

def is_power_of_two_log(n: int) -> bool:
    """Using logarithm (prone to floating point errors)."""
    if n <= 0:
        return False
    log_val = math.log2(n)
    return log_val == int(log_val)


# Approach 5: Division method
def is_power_of_two_division(n: int) -> bool:
    """Repeatedly divide by 2."""
    if n <= 0:
        return False
    while n > 1:
        if n % 2 != 0:
            return False
        n //= 2
    return True


# Check power of three
def is_power_of_three(n: int) -> bool:
    """
    Power of 3 check.
    3^19 = 1162261467 is the largest power of 3 fitting in 32-bit int.
    """
    return n > 0 and 1162261467 % n == 0


# Check power of four
def is_power_of_four(n: int) -> bool:
    """
    Power of 4: must be power of 2 AND have the 1 bit
    at an even position (0, 2, 4, ...).
    Mask 0x55555555 = 01010101... checks even positions.
    """
    return n > 0 and (n & (n - 1)) == 0 and (n & 0x55555555) != 0


# Find next power of two
def next_power_of_two(n: int) -> int:
    """Find smallest power of 2 >= n."""
    if n <= 0:
        return 1
    if is_power_of_two(n):
        return n
    # Fill all bits to the right of the highest bit
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


# Count trailing zeros (log2 for powers of 2)
def count_trailing_zeros(n: int) -> int:
    """Count trailing zeros = log2 for powers of 2."""
    if n == 0:
        return 32
    count = 0
    while (n & 1) == 0:
        count += 1
        n >>= 1
    return count`,
	testCode: `import pytest
from solution import is_power_of_two


class TestPowerOfTwo:
    def test_one(self):
        """Test 2^0 = 1"""
        assert is_power_of_two(1) == True

    def test_two(self):
        """Test 2^1 = 2"""
        assert is_power_of_two(2) == True

    def test_sixteen(self):
        """Test 2^4 = 16"""
        assert is_power_of_two(16) == True

    def test_three(self):
        """Test 3 (not power of 2)"""
        assert is_power_of_two(3) == False

    def test_zero(self):
        """Test 0"""
        assert is_power_of_two(0) == False

    def test_negative(self):
        """Test negative number"""
        assert is_power_of_two(-1) == False
        assert is_power_of_two(-16) == False

    def test_large_power_of_two(self):
        """Test large power of 2"""
        assert is_power_of_two(1024) == True  # 2^10
        assert is_power_of_two(2**20) == True

    def test_near_power_of_two(self):
        """Test numbers near powers of 2"""
        assert is_power_of_two(15) == False
        assert is_power_of_two(17) == False
        assert is_power_of_two(31) == False
        assert is_power_of_two(33) == False

    def test_max_int_power_of_two(self):
        """Test largest 32-bit power of 2"""
        assert is_power_of_two(2**30) == True
        assert is_power_of_two(2**31) == True

    def test_various_non_powers(self):
        """Test various non-powers of 2"""
        assert is_power_of_two(6) == False
        assert is_power_of_two(12) == False
        assert is_power_of_two(100) == False`,
	hint1: `Powers of two have exactly one bit set (1, 10, 100, 1000 in binary). How can you check for exactly one bit?`,
	hint2: `Use \`n & (n - 1)\` which clears the lowest set bit. For powers of two, this makes the result 0. Don't forget to check \`n > 0\`.`,
	whyItMatters: `Power of Two is a foundational bit manipulation problem. Understanding why \`n & (n - 1)\` works unlocks many related problems and optimizations.

**Why This Matters:**

**1. Binary Representation of Powers of 2**

\`\`\`python
# Powers of 2 have exactly one bit set:
# 1  = 0001
# 2  = 0010
# 4  = 0100
# 8  = 1000
# 16 = 10000

# n - 1 flips all bits from the rightmost 1 to the end:
# 8     = 1000
# 8 - 1 = 0111
# 8 & 7 = 0000
\`\`\`

**2. The n & (n - 1) Trick**

\`\`\`python
# This clears the lowest set bit:
# n     = ...xyz1000
# n - 1 = ...xyz0111
# n & (n-1) = ...xyz0000

# For powers of 2: only one bit, so result is 0
# For others: some bits remain
\`\`\`

**3. Related Tricks**

\`\`\`python
# Isolate lowest bit: n & (-n)
# Check power of 4: power of 2 AND bit at even position
# Next power of 2: round up using bit filling
# Count set bits: count iterations of n & (n-1)
\`\`\`

**4. Applications**

\`\`\`python
# Memory alignment (must be power of 2)
# Hash table sizes (often powers of 2 for fast modulo)
# Buffer sizes in systems programming
# Binary search tree balancing checks
\`\`\``,
	order: 6,
	translations: {
		ru: {
			title: 'Степень двойки',
			description: `Определите, является ли число степенью двойки.

**Задача:**

Дано целое число \`n\`, верните \`true\` если это степень двойки, иначе \`false\`.

Число \`n\` является степенью двойки, если существует \`x\` такой что \`n == 2^x\`.

**Примеры:**

\`\`\`
Вход: n = 1
Выход: true (2^0 = 1)

Вход: n = 16
Выход: true (2^4 = 16)

Вход: n = 3
Выход: false

Вход: n = 0
Выход: false
\`\`\`

**Ключевая идея:**

Степень двойки имеет ровно один установленный бит. Трюк: \`n & (n - 1)\` очищает младший бит. Для степеней двойки результат = 0.

**Формула:** \`n > 0 and (n & (n - 1)) == 0\`

**Ограничения:**
- -2^31 <= n <= 2^31 - 1

**Временная сложность:** O(1)
**Пространственная сложность:** O(1)`,
			hint1: `Степени двойки имеют ровно один установленный бит (1, 10, 100, 1000 в двоичном). Как проверить наличие ровно одного бита?`,
			hint2: `Используйте \`n & (n - 1)\` - очищает младший бит. Для степеней двойки результат = 0. Не забудьте проверить \`n > 0\`.`,
			whyItMatters: `Power of Two - базовая задача на битовые манипуляции. Понимание почему работает \`n & (n - 1)\` открывает множество связанных задач.

**Почему это важно:**

**1. Двоичное представление степеней 2**

Степени 2 имеют ровно один бит: 1, 10, 100, 1000. n - 1 инвертирует биты от младшей единицы до конца.

**2. Трюк n & (n - 1)**

Очищает младший установленный бит. Для степеней 2 результат = 0.

**3. Связанные трюки**

Изоляция младшего бита, проверка степени 4, следующая степень 2.

**4. Применения**

Выравнивание памяти, размеры хеш-таблиц, буферы в системном программировании.`,
			solutionCode: `def is_power_of_two(n: int) -> bool:
    """Проверяет является ли n степенью двойки используя битовые операции."""
    return n > 0 and (n & (n - 1)) == 0`
		},
		uz: {
			title: 'Ikkining darajasi',
			description: `Berilgan son ikkining darajasi ekanligini aniqlang.

**Masala:**

Butun son \`n\` berilgan, agar u ikkining darajasi bo'lsa \`true\`, aks holda \`false\` qaytaring.

\`n\` soni ikkining darajasi, agar \`n == 2^x\` bo'ladigan \`x\` mavjud bo'lsa.

**Misollar:**

\`\`\`
Kirish: n = 1
Chiqish: true (2^0 = 1)

Kirish: n = 16
Chiqish: true (2^4 = 16)

Kirish: n = 3
Chiqish: false

Kirish: n = 0
Chiqish: false
\`\`\`

**Asosiy tushuncha:**

Ikkining darajasi aynan bitta o'rnatilgan bitga ega. Hiyla: \`n & (n - 1)\` eng kichik bitni tozalaydi. Ikkining darajalari uchun natija = 0.

**Formula:** \`n > 0 and (n & (n - 1)) == 0\`

**Cheklovlar:**
- -2^31 <= n <= 2^31 - 1

**Vaqt murakkabligi:** O(1)
**Xotira murakkabligi:** O(1)`,
			hint1: `Ikkining darajalari aynan bitta o'rnatilgan bitga ega (ikkilikda 1, 10, 100, 1000). Aynan bitta bitni qanday tekshirish mumkin?`,
			hint2: `\`n & (n - 1)\` ishlating - eng kichik bitni tozalaydi. Ikkining darajalari uchun natija = 0. \`n > 0\` ni tekshirishni unutmang.`,
			whyItMatters: `Power of Two - asosiy bit manipulyatsiya masalasi. \`n & (n - 1)\` nima uchun ishlashini tushunish ko'plab bog'liq masalalarni ochadi.

**Bu nima uchun muhim:**

**1. Ikkining darajalarining ikkilik ko'rinishi**

Ikkining darajalari aynan bitta bitga ega: 1, 10, 100, 1000. n - 1 eng kichik 1 dan oxirigacha bitlarni o'zgartiradi.

**2. n & (n - 1) hiylasi**

Eng kichik o'rnatilgan bitni tozalaydi. Ikkining darajalari uchun natija = 0.

**3. Bog'liq hiylalar**

Eng kichik bitni ajratish, 4 ning darajasini tekshirish, keyingi ikkining darajasi.

**4. Qo'llanishlar**

Xotira tekislash, xesh-jadval o'lchamlari, tizim dasturlashda buferlar.`,
			solutionCode: `def is_power_of_two(n: int) -> bool:
    """Bit operatsiyalari yordamida n ikkining darajasi ekanligini tekshiradi."""
    return n > 0 and (n & (n - 1)) == 0`
		}
	}
};

export default task;
