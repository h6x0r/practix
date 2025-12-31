import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'bit-manipulation-bitwise-and-range',
	title: 'Bitwise AND of Numbers Range',
	difficulty: 'medium',
	tags: ['python', 'bit-manipulation', 'math'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the bitwise AND of all numbers in a given range [left, right].

**Problem:**

Given two integers \`left\` and \`right\` that represent the range \`[left, right]\`, return the bitwise AND of all numbers in this range, inclusive.

**Examples:**

\`\`\`
Input: left = 5, right = 7
Output: 4
Explanation:
5 = 101
6 = 110
7 = 111
AND = 100 = 4

Input: left = 0, right = 0
Output: 0

Input: left = 1, right = 2147483647
Output: 0
\`\`\`

**Key Insight:**

When you AND all numbers in a range, only the **common prefix** of their binary representations survives. Any bit that changes within the range becomes 0.

**Approach:** Find the common prefix of \`left\` and \`right\` by right-shifting both until they're equal, then shift back.

**Visualization:**

\`\`\`
left  = 5:  101
right = 7:  111
            ↑ common prefix: 1

After shifting right until equal:
5 >> 1 = 2 (10)    7 >> 1 = 3 (11) - not equal
5 >> 2 = 1 (1)     7 >> 2 = 1 (1)  - equal!

Shift count = 2
Result = 1 << 2 = 4 (100)
\`\`\`

**Constraints:**
- 0 <= left <= right <= 2^31 - 1

**Time Complexity:** O(log n) where n is the range
**Space Complexity:** O(1)`,
	initialCode: `def range_bitwise_and(left: int, right: int) -> int:
    # TODO: Find bitwise AND of all numbers in range [left, right]

    return 0`,
	solutionCode: `def range_bitwise_and(left: int, right: int) -> int:
    """
    Find common prefix by shifting until equal.
    """
    shift = 0
    while left != right:
        left >>= 1
        right >>= 1
        shift += 1
    return left << shift


# Approach 2: Brian Kernighan's approach
def range_bitwise_and_kernighan(left: int, right: int) -> int:
    """
    Keep clearing rightmost bit of right until right <= left.
    The result is the common prefix.
    """
    while right > left:
        # Clear the rightmost bit
        right &= (right - 1)
    return right


# Approach 3: Find common prefix using XOR
def range_bitwise_and_xor(left: int, right: int) -> int:
    """
    XOR finds differing bits, then create mask for common prefix.
    """
    if left == right:
        return left

    # Find XOR of left and right
    xor = left ^ right

    # Find the position of highest differing bit
    # All bits from this position to the right will be 0
    diff_bits = 0
    while xor:
        xor >>= 1
        diff_bits += 1

    # Create mask for common prefix
    mask = ~((1 << diff_bits) - 1)
    return left & mask


# Approach 4: Using bit length
def range_bitwise_and_bit_length(left: int, right: int) -> int:
    """Using bit length to find common prefix."""
    if left == 0:
        return 0

    # If bit lengths differ, result is 0
    left_bits = left.bit_length()
    right_bits = right.bit_length()

    if left_bits != right_bits:
        return 0

    # Find common prefix
    shift = 0
    while left != right:
        left >>= 1
        right >>= 1
        shift += 1

    return left << shift


# Brute force (for verification, not for submission)
def range_bitwise_and_brute(left: int, right: int) -> int:
    """Brute force: AND all numbers. O(n) - too slow!"""
    result = left
    for num in range(left + 1, right + 1):
        result &= num
        if result == 0:  # Early termination
            break
    return result


# Related: OR of range
def range_bitwise_or(left: int, right: int) -> int:
    """
    Bitwise OR of range.
    Different approach: fill bits from highest differing bit.
    """
    if left == right:
        return left

    # Find highest differing bit position
    diff = left ^ right
    highest_bit = 0
    while diff:
        diff >>= 1
        highest_bit += 1

    # All bits below highest differing bit become 1
    # Keep common prefix, set rest to 1
    mask = (1 << highest_bit) - 1
    return right | mask`,
	testCode: `import pytest
from solution import range_bitwise_and


class TestBitwiseAndRange:
    def test_example_one(self):
        """Test range [5, 7]"""
        assert range_bitwise_and(5, 7) == 4

    def test_zero_range(self):
        """Test [0, 0]"""
        assert range_bitwise_and(0, 0) == 0

    def test_large_range(self):
        """Test [1, 2147483647]"""
        assert range_bitwise_and(1, 2147483647) == 0

    def test_same_number(self):
        """Test when left == right"""
        assert range_bitwise_and(5, 5) == 5
        assert range_bitwise_and(1000, 1000) == 1000

    def test_consecutive_numbers(self):
        """Test consecutive numbers"""
        # Last bit always differs
        assert range_bitwise_and(2, 3) == 2
        assert range_bitwise_and(4, 5) == 4

    def test_power_of_two_range(self):
        """Test range crossing power of 2"""
        assert range_bitwise_and(3, 4) == 0  # 011 & 100 = 0
        assert range_bitwise_and(7, 8) == 0  # 0111 & 1000 = 0

    def test_within_same_prefix(self):
        """Test range within same prefix"""
        assert range_bitwise_and(12, 15) == 12  # 1100, 1101, 1110, 1111 -> 1100

    def test_zero_start(self):
        """Test range starting with 0"""
        assert range_bitwise_and(0, 5) == 0
        assert range_bitwise_and(0, 1) == 0

    def test_small_range(self):
        """Test small ranges"""
        assert range_bitwise_and(10, 12) == 8  # 1010, 1011, 1100

    def test_binary_boundary(self):
        """Test at binary boundaries"""
        assert range_bitwise_and(15, 16) == 0  # 01111 & 10000
        assert range_bitwise_and(16, 17) == 16  # 10000 & 10001`,
	hint1: `Think about what happens to each bit position. If any two numbers in the range differ at a bit position, that bit becomes 0 in the result.`,
	hint2: `Find the common prefix of left and right. Right-shift both until they're equal, count the shifts, then shift back. The common prefix is the only part that survives.`,
	whyItMatters: `Bitwise AND of Range reveals how bit operations behave across number sequences. Understanding common prefixes in binary is crucial for many advanced algorithms.

**Why This Matters:**

**1. Common Prefix Insight**

\`\`\`python
# When ANDing a range, only the common prefix survives:
# 5 = 101
# 6 = 110
# 7 = 111
# Common prefix: 1 (first bit only)

# Why? Between any consecutive numbers, at least
# the last bit changes. Larger ranges have more changes.
\`\`\`

**2. Why Brute Force Fails**

\`\`\`python
# Range [1, 2^31 - 1] has billions of numbers
# But answer is 0 (no common prefix)
# Our O(log n) solution handles this instantly
\`\`\`

**3. The Shifting Technique**

\`\`\`python
# Right-shift until left == right
# This finds the common prefix
# Number of shifts = length of differing suffix

# This pattern applies to many "range" problems
\`\`\`

**4. Applications**

\`\`\`python
# IP address subnetting (common prefix = network)
# Finding common ancestors in binary trees
# Range queries in segment trees
# Trie-based data structures
\`\`\``,
	order: 7,
	translations: {
		ru: {
			title: 'Побитовое И диапазона чисел',
			description: `Найдите побитовое И всех чисел в диапазоне [left, right].

**Задача:**

Даны два числа \`left\` и \`right\`, верните побитовое И всех чисел в диапазоне \`[left, right]\` включительно.

**Примеры:**

\`\`\`
Вход: left = 5, right = 7
Выход: 4
Объяснение:
5 = 101
6 = 110
7 = 111
AND = 100 = 4

Вход: left = 0, right = 0
Выход: 0

Вход: left = 1, right = 2147483647
Выход: 0
\`\`\`

**Ключевая идея:**

При AND всех чисел диапазона выживает только **общий префикс** их двоичных представлений. Любой бит, меняющийся в диапазоне, становится 0.

**Подход:** Найдите общий префикс \`left\` и \`right\` сдвигом вправо пока не станут равны, затем сдвиньте обратно.

**Ограничения:**
- 0 <= left <= right <= 2^31 - 1

**Временная сложность:** O(log n)
**Пространственная сложность:** O(1)`,
			hint1: `Подумайте что происходит с каждой битовой позицией. Если любые два числа в диапазоне различаются в позиции, этот бит станет 0.`,
			hint2: `Найдите общий префикс left и right. Сдвигайте оба вправо пока не станут равны, считая сдвиги, затем сдвиньте обратно.`,
			whyItMatters: `Побитовое И диапазона показывает поведение битовых операций над последовательностями чисел. Понимание общих префиксов в двоичном коде важно для многих алгоритмов.

**Почему это важно:**

**1. Идея общего префикса**

При AND диапазона выживает только общий префикс. Между соседними числами минимум последний бит меняется.

**2. Почему перебор не работает**

Диапазон [1, 2^31 - 1] имеет миллиарды чисел, но ответ = 0. Наше O(log n) решение мгновенно.

**3. Техника сдвига**

Сдвиг вправо пока left == right находит общий префикс.

**4. Применения**

IP подсети (общий префикс = сеть), общие предки в деревьях, диапазонные запросы.`,
			solutionCode: `def range_bitwise_and(left: int, right: int) -> int:
    """Находит общий префикс сдвигом пока числа не станут равны."""
    shift = 0
    while left != right:
        left >>= 1
        right >>= 1
        shift += 1
    return left << shift`
		},
		uz: {
			title: "Diapazon sonlarining bit bo'yicha AND",
			description: `[left, right] diapazonidagi barcha sonlarning bit bo'yicha AND ni toping.

**Masala:**

\`[left, right]\` diapazonini ifodalovchi ikkita butun son \`left\` va \`right\` berilgan, ushbu diapazondagi barcha sonlarning bit bo'yicha AND ini qaytaring.

**Misollar:**

\`\`\`
Kirish: left = 5, right = 7
Chiqish: 4
Tushuntirish:
5 = 101
6 = 110
7 = 111
AND = 100 = 4

Kirish: left = 0, right = 0
Chiqish: 0

Kirish: left = 1, right = 2147483647
Chiqish: 0
\`\`\`

**Asosiy tushuncha:**

Diapazondagi barcha sonlarni AND qilganda, faqat ularning ikkilik ko'rinishlarining **umumiy prefiksi** saqlanadi. Diapazon ichida o'zgaradigan har qanday bit 0 ga aylanadi.

**Yondashuv:** \`left\` va \`right\` ning umumiy prefiksini topish uchun teng bo'lguncha o'ngga siljiting, keyin orqaga siljiting.

**Cheklovlar:**
- 0 <= left <= right <= 2^31 - 1

**Vaqt murakkabligi:** O(log n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Har bir bit pozitsiyasiga nima bo'lishini o'ylang. Agar diapazondagi har qanday ikkita son pozitsiyada farq qilsa, bu bit natijada 0 bo'ladi.`,
			hint2: `left va right ning umumiy prefiksini toping. Teng bo'lguncha ikkalasini o'ngga siljiting, siljishlarni sanaib, keyin orqaga siljiting.`,
			whyItMatters: `Diapazon sonlarining bit bo'yicha AND bit operatsiyalari sonlar ketma-ketligida qanday ishlashini ko'rsatadi. Ikkilik koddagi umumiy prefikslarni tushunish ko'p algoritmlar uchun muhim.

**Bu nima uchun muhim:**

**1. Umumiy prefiks tushunchasi**

Diapazonni AND qilganda faqat umumiy prefiks saqlanadi. Qo'shni sonlar orasida kamida oxirgi bit o'zgaradi.

**2. Nima uchun brute force ishlamaydi**

[1, 2^31 - 1] diapazoni milliardlab sonlarga ega, lekin javob = 0. Bizning O(log n) yechimimiz bir zumda ishlaydi.

**3. Siljitish texnikasi**

left == right bo'lguncha o'ngga siljitish umumiy prefiksni topadi.

**4. Qo'llanishlar**

IP tarmoqlari (umumiy prefiks = tarmoq), daraxtlardagi umumiy ajdodlar, diapazon so'rovlari.`,
			solutionCode: `def range_bitwise_and(left: int, right: int) -> int:
    """Sonlar teng bo'lguncha siljitish orqali umumiy prefiksni topadi."""
    shift = 0
    while left != right:
        left >>= 1
        right >>= 1
        shift += 1
    return left << shift`
		}
	}
};

export default task;
