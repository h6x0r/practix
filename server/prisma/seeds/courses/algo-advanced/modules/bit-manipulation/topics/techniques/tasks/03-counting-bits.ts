import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'bit-manipulation-counting-bits',
	title: 'Counting Bits',
	difficulty: 'easy',
	tags: ['python', 'bit-manipulation', 'dynamic-programming'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Given an integer n, return an array of the number of 1's in the binary representation of every number from 0 to n.

**Problem:**

Given an integer \`n\`, return an array \`ans\` of length \`n + 1\` such that for each \`i\` (0 <= i <= n), \`ans[i]\` is the **number of 1's** in the binary representation of \`i\`.

**Examples:**

\`\`\`
Input: n = 2
Output: [0, 1, 1]
Explanation:
0 --> 0 (0 ones)
1 --> 1 (1 one)
2 --> 10 (1 one)

Input: n = 5
Output: [0, 1, 1, 2, 1, 2]
Explanation:
0 --> 0    (0 ones)
1 --> 1    (1 one)
2 --> 10   (1 one)
3 --> 11   (2 ones)
4 --> 100  (1 one)
5 --> 101  (2 ones)
\`\`\`

**Key Insight:**

Use dynamic programming! For any number \`i\`:
- \`countBits(i) = countBits(i >> 1) + (i & 1)\`
- Or: \`countBits(i) = countBits(i & (i-1)) + 1\`

The first approach: right-shifting removes the last bit, we add 1 if the last bit was 1.

**Constraints:**
- 0 <= n <= 10^5

**Follow up:**
- Can you do it in O(n) time and O(n) space?
- Can you do it without using any built-in function?

**Time Complexity:** O(n)
**Space Complexity:** O(n) for the output array`,
	initialCode: `from typing import List

def count_bits(n: int) -> List[int]:
    # TODO: Return array where ans[i] = number of 1 bits in i, for 0 to n

    return []`,
	solutionCode: `from typing import List

def count_bits(n: int) -> List[int]:
    """
    Count bits using DP with right shift approach.
    """
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        # ans[i >> 1] is already computed
        # (i & 1) adds 1 if i is odd
        ans[i] = ans[i >> 1] + (i & 1)
    return ans


# Approach 2: DP with Brian Kernighan's trick
def count_bits_kernighan(n: int) -> List[int]:
    """Using i & (i-1) to clear lowest bit."""
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        # i & (i-1) clears the lowest set bit
        # ans[i & (i-1)] is already computed (smaller than i)
        ans[i] = ans[i & (i - 1)] + 1
    return ans


# Approach 3: Using least significant bit pattern
def count_bits_lsb(n: int) -> List[int]:
    """
    Pattern observation:
    - Even numbers have same bits as n/2
    - Odd numbers have one more bit than n/2
    """
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        if i % 2 == 0:
            ans[i] = ans[i // 2]
        else:
            ans[i] = ans[i // 2] + 1
    return ans


# Approach 4: Most significant bit pattern
def count_bits_msb(n: int) -> List[int]:
    """Using most significant bit offset."""
    ans = [0] * (n + 1)
    offset = 1  # Highest power of 2 <= i

    for i in range(1, n + 1):
        if offset * 2 == i:
            offset = i
        ans[i] = 1 + ans[i - offset]

    return ans


# Approach 5: Brute force (for comparison)
def count_bits_brute(n: int) -> List[int]:
    """Brute force: count bits for each number."""
    def popcount(x):
        count = 0
        while x:
            x &= x - 1
            count += 1
        return count

    return [popcount(i) for i in range(n + 1)]


# Approach 6: Built-in
def count_bits_builtin(n: int) -> List[int]:
    """Using Python's bin().count()."""
    return [bin(i).count('1') for i in range(n + 1)]`,
	testCode: `import pytest
from solution import count_bits


class TestCountingBits:
    def test_zero(self):
        """Test n = 0"""
        assert count_bits(0) == [0]

    def test_one(self):
        """Test n = 1"""
        assert count_bits(1) == [0, 1]

    def test_two(self):
        """Test n = 2"""
        assert count_bits(2) == [0, 1, 1]

    def test_five(self):
        """Test n = 5"""
        assert count_bits(5) == [0, 1, 1, 2, 1, 2]

    def test_power_of_two(self):
        """Test power of 2"""
        result = count_bits(8)
        assert result[8] == 1  # 1000 in binary
        assert result[7] == 3  # 111 in binary

    def test_fifteen(self):
        """Test n = 15"""
        result = count_bits(15)
        assert result[15] == 4  # 1111 in binary
        assert len(result) == 16

    def test_pattern_even_odd(self):
        """Test even/odd pattern"""
        result = count_bits(10)
        # Even numbers: 0, 2, 4, 6, 8, 10 -> [0, 1, 1, 2, 1, 2]
        # Odd numbers: 1, 3, 5, 7, 9 -> [1, 2, 2, 3, 2]
        assert result[4] == 1   # 100
        assert result[5] == 2   # 101 (one more than 4)

    def test_all_ones(self):
        """Test number with all 1s"""
        result = count_bits(31)
        assert result[31] == 5  # 11111 in binary

    def test_length(self):
        """Test output array length"""
        assert len(count_bits(10)) == 11
        assert len(count_bits(100)) == 101

    def test_monotonic_for_powers(self):
        """Test that powers of 2 always have 1 bit"""
        result = count_bits(128)
        for i in [1, 2, 4, 8, 16, 32, 64, 128]:
            assert result[i] == 1`,
	hint1: `Think about how the number of 1 bits in \`i\` relates to \`i >> 1\` (i divided by 2). The only difference is whether the last bit is 1 or 0.`,
	hint2: `Use DP: \`ans[i] = ans[i >> 1] + (i & 1)\`. Since \`i >> 1 < i\`, we've already computed it. The \`(i & 1)\` adds 1 for odd numbers.`,
	whyItMatters: `Counting Bits combines bit manipulation with dynamic programming, showing how mathematical patterns in binary representation enable O(1) computation per number.

**Why This Matters:**

**1. DP Recurrence Relations**

\`\`\`python
# Two equivalent approaches:

# Approach 1: Right shift
# countBits(i) = countBits(i >> 1) + (i & 1)
# Example: 5 (101) -> 2 (10) has 1 bit, plus 1 for odd = 2

# Approach 2: Clear lowest bit
# countBits(i) = countBits(i & (i-1)) + 1
# Example: 5 (101) -> 4 (100) has 1 bit, plus 1 = 2
\`\`\`

**2. Binary Number Patterns**

\`\`\`python
# Interesting patterns:
# - Powers of 2: always have exactly 1 bit
# - 2^n - 1: always have n bits (all 1s)
# - Even numbers: same bits as n/2
# - Odd numbers: one more bit than n/2

# The sequence has fractal-like structure:
# [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, ...]
\`\`\`

**3. Applications**

\`\`\`python
# Gray code generation
# Hamming distance calculations
# Population count in bitmap indexes
# Efficient set cardinality in bitsets
# Cryptographic weight analysis
\`\`\`

**4. Hardware Perspective**

\`\`\`python
# CPUs have POPCNT instruction for single numbers
# But for ranges, DP is more efficient than n POPCNTs
# This shows software can beat hardware for specific patterns
\`\`\``,
	order: 3,
	translations: {
		ru: {
			title: 'Подсчёт битов',
			description: `Для данного целого числа n верните массив с количеством единичных битов для каждого числа от 0 до n.

**Задача:**

Дано целое число \`n\`, верните массив \`ans\` длины \`n + 1\`, где \`ans[i]\` - **количество единиц** в двоичном представлении числа \`i\`.

**Примеры:**

\`\`\`
Вход: n = 2
Выход: [0, 1, 1]
Объяснение:
0 --> 0 (0 единиц)
1 --> 1 (1 единица)
2 --> 10 (1 единица)

Вход: n = 5
Выход: [0, 1, 1, 2, 1, 2]
\`\`\`

**Ключевая идея:**

Используйте динамическое программирование:
- \`countBits(i) = countBits(i >> 1) + (i & 1)\`
- Сдвиг вправо убирает последний бит, добавляем 1 если он был единицей.

**Ограничения:**
- 0 <= n <= 10^5

**Временная сложность:** O(n)
**Пространственная сложность:** O(n)`,
			hint1: `Подумайте как количество единиц в \`i\` связано с \`i >> 1\`. Разница только в последнем бите.`,
			hint2: `Используйте ДП: \`ans[i] = ans[i >> 1] + (i & 1)\`. Так как \`i >> 1 < i\`, мы уже вычислили это значение.`,
			whyItMatters: `Подсчёт битов объединяет битовые манипуляции с ДП, показывая как паттерны в двоичном представлении позволяют O(1) вычисление для каждого числа.

**Почему это важно:**

**1. Рекуррентные соотношения ДП**

countBits(i) = countBits(i >> 1) + (i & 1)
countBits(i) = countBits(i & (i-1)) + 1

**2. Паттерны двоичных чисел**

Степени 2 имеют 1 бит, 2^n - 1 имеют n битов, чётные числа имеют столько же битов как n/2.

**3. Применения**

Код Грея, расстояние Хэмминга, bitmap индексы, мощность множеств в битсетах.`,
			solutionCode: `from typing import List

def count_bits(n: int) -> List[int]:
    """Подсчитывает биты используя ДП со сдвигом вправо."""
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        ans[i] = ans[i >> 1] + (i & 1)
    return ans`
		},
		uz: {
			title: 'Bitlarni sanash',
			description: `Berilgan butun son n uchun 0 dan n gacha har bir son uchun 1-bitlar sonini qaytaring.

**Masala:**

Butun son \`n\` berilgan, \`n + 1\` uzunlikdagi \`ans\` massivini qaytaring, bu yerda \`ans[i]\` - \`i\` sonining ikkilik ko'rinishidagi **birlar soni**.

**Misollar:**

\`\`\`
Kirish: n = 2
Chiqish: [0, 1, 1]
Tushuntirish:
0 --> 0 (0 ta bir)
1 --> 1 (1 ta bir)
2 --> 10 (1 ta bir)

Kirish: n = 5
Chiqish: [0, 1, 1, 2, 1, 2]
\`\`\`

**Asosiy tushuncha:**

Dinamik dasturlash ishlating:
- \`countBits(i) = countBits(i >> 1) + (i & 1)\`
- O'ngga siljitish oxirgi bitni olib tashlaydi, agar u 1 bo'lsa 1 qo'shamiz.

**Cheklovlar:**
- 0 <= n <= 10^5

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(n)`,
			hint1: `\`i\` dagi birlar soni \`i >> 1\` bilan qanday bog'liqligini o'ylang. Farq faqat oxirgi bitda.`,
			hint2: `DP ishlating: \`ans[i] = ans[i >> 1] + (i & 1)\`. \`i >> 1 < i\` bo'lgani uchun buni allaqachon hisoblaganmiz.`,
			whyItMatters: `Bitlarni sanash bit manipulyatsiyasini DP bilan birlashtiradi, ikkilik ko'rinishdagi patternlar har bir son uchun O(1) hisoblashga imkon berishini ko'rsatadi.

**Bu nima uchun muhim:**

**1. DP rekurrent munosabatlari**

countBits(i) = countBits(i >> 1) + (i & 1)
countBits(i) = countBits(i & (i-1)) + 1

**2. Ikkilik sonlar patternlari**

2 darajalari 1 bitga ega, 2^n - 1 n bitga ega, juft sonlar n/2 bilan bir xil bitlarga ega.

**3. Qo'llanishlar**

Grey kodi, Hemming masofasi, bitmap indekslar, bitsetlarda to'plam quvvati.`,
			solutionCode: `from typing import List

def count_bits(n: int) -> List[int]:
    """O'ngga siljitish bilan DP yordamida bitlarni sanaydi."""
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        ans[i] = ans[i >> 1] + (i & 1)
    return ans`
		}
	}
};

export default task;
